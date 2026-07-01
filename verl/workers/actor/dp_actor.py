# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_self_distillation_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, slice_input_tensor, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TrustRegionTeacher(nn.Module):
    def __init__(self, ref_module: nn.Module, student_module: nn.Module, mix_coef: float) -> None:
        super().__init__()
        self.ref_module = ref_module
        self.student_module = student_module
        self.mix_coef = float(mix_coef)

    def forward(self, *args, **kwargs):
        ref_out = self.ref_module(*args, **kwargs)
        student_out = self.student_module(*args, **kwargs)
        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out[0]
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out[0]
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.teacher_module: Optional[nn.Module] = None
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _update_teacher(self) -> None:
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if not self_distillation_cfg or loss_mode != "sdpo":
            return
        teacher_regularization = getattr(self_distillation_cfg, "teacher_regularization", "ema")
        if teacher_regularization != "ema":
            return
        update_rate = getattr(self_distillation_cfg, "teacher_update_rate", 0.0)
        if update_rate == 0.0:
            return
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise ValueError("EMA teacher requires a separate teacher_module in the actor worker.")
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_module.parameters(),
                self.actor_module.parameters(),
            ):
                student_data = student_param.data.to(device=teacher_param.device)
                teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)

    @staticmethod
    def _has_non_empty_multi_modal_inputs(multi_modal_inputs) -> bool:
        if multi_modal_inputs is None:
            return False
        for inputs in multi_modal_inputs:
            if inputs is None:
                continue
            inputs = getattr(inputs, "data", inputs)
            if isinstance(inputs, dict):
                if not inputs:
                    continue
                for value in inputs.values():
                    if value is None:
                        continue
                    if isinstance(value, torch.Tensor) and value.numel() == 0:
                        continue
                    return True
            else:
                return True
        return False

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_all_logps: bool = False,
        distill_topk: Optional[int] = None,
        topk_indices: Optional[torch.Tensor] = None,
        module: Optional[nn.Module] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
                if distill_topk or topk_indices is set:
                    topk_logps: (bs, response_len, k)
                    topk_indices: (bs, response_len, k)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        use_topk = distill_topk is not None or topk_indices is not None
        compute_all_logps = return_all_logps and not use_topk
        return_topk_indices = use_topk and topk_indices is None
        if (return_all_logps or use_topk) and self.use_fused_kernels:
            raise ValueError("Logit distillation requires disabling fused kernels.")

        model = module or self.actor_module

        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
                and not return_all_logps
                and not use_topk
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=model,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(model, "module", model).config,
                        "vision_config",
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    all_logps_rmpad = torch.log_softmax(logits_rmpad, dim=-1) if compute_all_logps else None

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits_rmpad.shape[-1])
                            topk_logits_rmpad, topk_indices_rmpad = torch.topk(logits_rmpad, topk, dim=-1)
                        else:
                            topk = topk_indices.size(-1)
                            full_topk_indices = torch.zeros(
                                batch_size,
                                seqlen,
                                topk,
                                device=topk_indices.device,
                                dtype=topk_indices.dtype,
                            )
                            full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"), indices
                            )
                            if self.use_ulysses_sp:
                                topk_indices_rmpad = slice_input_tensor(
                                    topk_indices_rmpad.unsqueeze(0), dim=1, padding=True
                                ).squeeze(0)
                            topk_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_indices_rmpad)
                        logsumexp_rmpad = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                        topk_logps_rmpad = topk_logits_rmpad - logsumexp_rmpad

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if use_topk:
                        topk_logps_rmpad = gather_outputs_and_unpad(
                            topk_logps_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        if return_topk_indices:
                            topk_indices_rmpad = gather_outputs_and_unpad(
                                topk_indices_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if compute_all_logps:
                        all_logps_rmpad = all_logps_rmpad[:0]
                    if use_topk:
                        topk_logps_rmpad = topk_logps_rmpad[:0]
                        if return_topk_indices:
                            topk_indices_rmpad = topk_indices_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if compute_all_logps:
                    full_all_logps = pad_input(
                        hidden_states=all_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if use_topk:
                    full_topk_logps = pad_input(
                        hidden_states=topk_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if return_topk_indices:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if compute_all_logps:
                    all_logps = full_all_logps[:, -response_length - 1 : -1, :]
                if use_topk:
                    topk_logps = full_topk_logps[:, -response_length - 1 : -1, :]
                    if return_topk_indices:
                        topk_indices = full_topk_indices[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if compute_all_logps:
                        all_logps = torch.log_softmax(logits, dim=-1)
                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits.size(-1))
                            topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)
                        else:
                            topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
                        logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
                        topk_logps = topk_logits - logsumexp
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if compute_all_logps:
                outputs["all_logps"] = all_logps
            if use_topk:
                outputs["topk_logps"] = topk_logps
                if return_topk_indices:
                    outputs["topk_indices"] = topk_indices
            return outputs

    def _apply_branch_loss_mode(
        self,
        *,
        response_mask: torch.Tensor,
        branch_token_mask: Optional[torch.Tensor],
        branch_loss_mode: str,
        is_branching_fallback: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Apply the teacher-guided branching ablation to ``response_mask``.

        Four modes:
          ``all``    : default. mask is ignored; every response token contributes.
          ``mask``   : zero-out branch tokens before aggregation
                       (response_mask AND NOT branch_token_mask).
          ``only``   : keep ONLY branch tokens (response_mask AND branch_token_mask).
          ``suffix`` : keep only tokens AFTER the last branch token in each row.
                       This isolates each leaf's unique student continuation,
                       eliminating gradient cancellation on shared prefixes
                       (where GRPO advantage sums to zero across siblings).

        For rows where ``is_branching_fallback`` is True (BranchingAgentLoop's
        owner pipeline raised, the row is a plain student rollout), the
        configured mode is OVERRIDDEN to ``all`` for that row only — otherwise
        ``btm=only``/``suffix`` would silently zero out every fallback row's
        gradient (mask is all-zero on fallback leaves).

        Returns ``(effective_response_mask, metrics)``. Diagnostic metrics
        are emitted REGARDLESS of mode so btm=all runs are still observable
        in SwanLab — without this, a run that silently degraded to 100%
        fallback would be indistinguishable from healthy branching.
        """
        # Fast path: mask absent (non-branching rollout) — no telemetry needed.
        if branch_token_mask is None:
            return response_mask, {}
        btm = branch_token_mask.to(response_mask.device).to(response_mask.dtype)

        # Per-row mode application: fallback rows always run as "all".
        if branch_loss_mode == "all":
            effective = response_mask
        elif branch_loss_mode in ("mask", "only", "suffix"):
            if branch_loss_mode == "mask":
                masked = response_mask * (1 - btm)
            elif branch_loss_mode == "only":
                masked = response_mask * btm
            else:  # "suffix"
                # Build a per-row suffix mask: 1 for positions strictly AFTER
                # the last branch token, 0 elsewhere. This keeps only the
                # leaf's unique continuation segment (on-policy, non-shared).
                T = btm.shape[1]
                positions = torch.arange(T, device=btm.device).unsqueeze(0)  # [1, T]
                # Weighted positions: btm * pos → max gives rightmost branch token idx.
                # For rows with NO branch tokens, max of all-zeros = 0; we fix below.
                has_branch = btm.any(dim=1)  # [B]
                weighted = btm * positions  # [B, T]
                last_branch_pos = weighted.max(dim=1).values.long()  # [B]
                # Rows without branch tokens → set last_branch_pos = -1 so
                # suffix_mask covers the full response (all positions > -1).
                last_branch_pos = torch.where(
                    has_branch, last_branch_pos,
                    torch.tensor(-1, device=btm.device, dtype=last_branch_pos.dtype),
                )
                suffix_mask = (positions > last_branch_pos.unsqueeze(1)).to(response_mask.dtype)
                masked = response_mask * suffix_mask

            if is_branching_fallback is not None and is_branching_fallback.numel() > 0:
                # is_branching_fallback shape: [B]; broadcast to [B, T] selector.
                ifb = is_branching_fallback.to(response_mask.device).to(response_mask.dtype)
                ifb_row = ifb.view(-1, *([1] * (response_mask.dim() - 1)))
                # Where ifb=1, use response_mask (all-tokens); where 0, use masked.
                effective = ifb_row * response_mask + (1.0 - ifb_row) * masked
            else:
                effective = masked
        else:
            raise ValueError(
                f"branch_token_loss_mode must be one of {{all, mask, only, suffix}}, "
                f"got {branch_loss_mode!r}"
            )

        # Always-on diagnostics so btm=all and btm=mask runs are
        # distinguishable, and silent-fallback runs are immediately visible.
        with torch.no_grad():
            base_active = response_mask.sum().clamp_min(1.0)
            eff_active = effective.sum()
            mode_id_map = {"all": 0, "mask": 1, "only": 2, "suffix": 3}
            metrics = {
                "branching/loss_mode_id": float(mode_id_map[branch_loss_mode]),
                "branching/active_token_ratio": (eff_active / base_active).item(),
                "branching/branch_token_count": btm.sum().item(),
            }
            if is_branching_fallback is not None and is_branching_fallback.numel() > 0:
                ifb = is_branching_fallback.to(torch.float32)
                metrics["branching/fallback_row_fraction"] = ifb.mean().item()
                metrics["branching/fallback_row_count"] = int(ifb.sum().item())
        return effective, metrics

    def _compute_on_policy_dpo_loss(
        self,
        log_prob: torch.Tensor,
        response_mask: torch.Tensor,
        branch_token_mask: torch.Tensor,
        is_two_stage_stage1: torch.Tensor,
        dpo_beta: float,
        ref_log_prob: torch.Tensor | None = None,
        teacher_branch_logprob: torch.Tensor | None = None,
        teacher_sibling_logprob: torch.Tensor | None = None,
        teacher_guided_beta: bool = False,
        teacher_beta_alpha: float = 1.0,
        teacher_beta_min: float = 0.1,
        teacher_beta_max: float = 3.0,
        dpo_stage1_pair: bool = False,
        stage1_pair_weight: float = 1.0,
        sample_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """On-policy DPO loss for Stage 2 pos/neg branch pairs.

        For each pair of samples from the same branch point, computes:
            L_DPO = -log σ(β × (avg_logp_chosen - avg_logp_rejected))

        where avg_logp is the mean logprob over suffix tokens (after branch point).

        When dpo_stage1_pair is True, also constructs 3-way ranking pairs using
        Stage 1 student chains as the "middle" version:
            L_S1 = -log σ(β × (avg_logp_argmax - avg_logp_stage1))
                 + -log σ(β × (avg_logp_stage1 - avg_logp_argmin))

        Stage 2 samples are naturally ordered: [tree0_pos, tree0_neg, tree1_pos, ...]
        so we pair them by consecutive positions.

        Args:
            log_prob: [B, T] current policy logprobs
            response_mask: [B, T] valid token mask
            branch_token_mask: [B, T] marks branch positions
            is_two_stage_stage1: [B] 1 for Stage 1, 0 for Stage 2
            dpo_beta: temperature parameter β
            ref_log_prob: [B, T] reference model logprobs (for with-ref DPO)
            teacher_branch_logprob: [B] teacher logprob of this sample's branch token
            teacher_sibling_logprob: [B] teacher logprob of the sibling's branch token
            teacher_guided_beta: whether to use teacher margin for adaptive β
            teacher_beta_alpha: scaling factor for margin → β mapping
            teacher_beta_min: minimum β multiplier (clamp lower bound)
            teacher_beta_max: maximum β multiplier (clamp upper bound)
            dpo_stage1_pair: whether to construct Stage 1 3-way pairs
            stage1_pair_weight: weight for Stage 1 pair loss
            sample_index: [B] prompt index for grouping Stage 1/Stage 2 samples

        Returns:
            (loss, metrics) tuple
        """
        T = log_prob.shape[1]
        device = log_prob.device

        # Identify Stage 2 samples (where DPO applies)
        is_stage2 = (is_two_stage_stage1 == 0)
        stage2_indices = torch.where(is_stage2)[0]

        if len(stage2_indices) < 2:
            # Not enough Stage 2 samples for pairing
            return torch.tensor(0.0, device=device, requires_grad=True), {"actor/dpo_loss": 0.0, "actor/dpo_n_pairs": 0.0}

        # Build suffix mask for each sample (tokens after last branch point)
        # Suffix mask: 1 for positions > last_branch_pos, 0 otherwise
        positions = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        has_branch = branch_token_mask.any(dim=1)  # [B]
        weighted = branch_token_mask * positions  # [B, T]
        last_branch_pos = weighted.max(dim=1).values.long()  # [B]
        last_branch_pos = torch.where(
            has_branch, last_branch_pos,
            torch.tensor(-1, device=device, dtype=last_branch_pos.dtype),
        )
        suffix_mask = (positions > last_branch_pos.unsqueeze(1)).to(log_prob.dtype)  # [B, T]

        # Combine suffix mask with response mask
        suffix_mask = suffix_mask * response_mask

        # Compute average logprob over suffix for each Stage 2 sample
        # avg_logp = sum(log_prob * suffix_mask) / sum(suffix_mask)
        logp_sum = (log_prob * suffix_mask).sum(dim=1)  # [B]
        suffix_len = suffix_mask.sum(dim=1).clamp_min(1.0)  # [B]
        avg_logp = logp_sum / suffix_len  # [B]

        # Extract Stage 2 avg logprobs
        stage2_avg_logp = avg_logp[stage2_indices]  # [N_stage2]

        # Pair by consecutive positions: [0,1] = pair0, [2,3] = pair1, etc.
        n_pairs = len(stage2_indices) // 2
        if n_pairs == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), {"actor/dpo_loss": 0.0, "actor/dpo_n_pairs": 0.0}

        paired_logp = stage2_avg_logp[:2 * n_pairs].reshape(n_pairs, 2)
        logp_chosen = paired_logp[:, 0]  # [n_pairs] — pos branch (teacher's preferred)
        logp_rejected = paired_logp[:, 1]  # [n_pairs] — neg branch

        # With-ref DPO: subtract frozen reference model logprobs for KL constraint
        if ref_log_prob is not None:
            ref_logp_sum = (ref_log_prob * suffix_mask).sum(dim=1)
            ref_avg_logp = ref_logp_sum / suffix_len
            ref_stage2_avg_logp = ref_avg_logp[stage2_indices]
            ref_paired = ref_stage2_avg_logp[:2 * n_pairs].reshape(n_pairs, 2)
            ref_chosen = ref_paired[:, 0]
            ref_rejected = ref_paired[:, 1]
            logp_chosen = logp_chosen - ref_chosen
            logp_rejected = logp_rejected - ref_rejected

        # Teacher-Guided β: compute margin at branch point for each pair
        # β_i = β_base · clamp(α · margin_i, β_min, β_max)
        # where margin_i = teacher_logp(chosen_token) - teacher_logp(rejected_token)
        effective_beta = dpo_beta  # default: scalar β (fixed mode)
        teacher_margin = None
        if teacher_guided_beta and teacher_branch_logprob is not None and teacher_sibling_logprob is not None:
            # Extract per-sample teacher logprobs for Stage 2
            t_branch = teacher_branch_logprob[stage2_indices]  # [N_stage2]
            t_sibling = teacher_sibling_logprob[stage2_indices]  # [N_stage2]

            # Pair by consecutive positions: [0,1] = pair0, [2,3] = pair1, etc.
            # Stage 2 ordering: [tree0_pos, tree0_neg, tree1_pos, tree1_neg, ...]
            # teacher_branch_logprob = teacher's logprob for THIS sample's branch token
            # teacher_sibling_logprob = teacher's logprob for the SIBLING's branch token
            t_branch_paired = t_branch[:2 * n_pairs].reshape(n_pairs, 2)
            t_sibling_paired = t_sibling[:2 * n_pairs].reshape(n_pairs, 2)

            # margin_i = teacher_logp(chosen_token) - teacher_logp(rejected_token)
            # For chosen (index 0): its branch token is the chosen token,
            # so margin = t_branch[chosen] - t_sibling[chosen]
            # But t_sibling[chosen] = teacher logprob for the REJECTED token
            # (since sibling of chosen is rejected)
            teacher_margin = (t_branch_paired[:, 0] - t_sibling_paired[:, 0]).detach()  # [n_pairs]

            # Dynamic β: β_base · clamp(α · margin, β_min, β_max)
            effective_beta = dpo_beta * torch.clamp(
                teacher_beta_alpha * teacher_margin,
                min=teacher_beta_min,
                max=teacher_beta_max,
            )  # [n_pairs]

        # DPO loss: -log σ(β * (logp_chosen - logp_rejected))
        # effective_beta is either scalar (fixed mode) or [n_pairs] tensor (adaptive mode)
        logits = effective_beta * (logp_chosen - logp_rejected)
        dpo_loss = -torch.nn.functional.logsigmoid(logits).mean()

        # Stage 1 middle-version DPO pairing: construct 3-way ranking pairs
        # using Stage 1 student chains as the "middle" version between
        # argmax (best) and argmin (worst).
        stage1_pair_loss = None
        n_s1_pairs = 0
        if dpo_stage1_pair and sample_index is not None:
            # Compute avg_logp for ALL samples (including Stage 1)
            all_avg_logp = avg_logp  # already computed above for all B samples

            # With-ref: compute ref avg_logp for all samples
            all_ref_avg_logp = None
            if ref_log_prob is not None:
                ref_logp_sum_all = (ref_log_prob * suffix_mask).sum(dim=1)
                all_ref_avg_logp = ref_logp_sum_all / suffix_len

            # Group samples by prompt index
            from collections import defaultdict
            prompt_groups = defaultdict(lambda: {"stage1": [], "stage2": []})
            for i in range(len(sample_index)):
                idx = int(sample_index[i].item())
                if is_two_stage_stage1[i].item() == 1:
                    prompt_groups[idx]["stage1"].append(i)
                else:
                    prompt_groups[idx]["stage2"].append(i)

            # Construct Stage 1 pairs for each prompt group
            s1_logp_best_list = []  # "best > middle" pairs
            s1_logp_worst_list = []  # "middle > worst" pairs

            for pid, group in prompt_groups.items():
                s1_indices = group["stage1"]
                s2_indices = group["stage2"]

                if not s1_indices or len(s2_indices) < 2:
                    continue

                # Use the first Stage 1 output as "middle"
                s1_idx = s1_indices[0]
                s1_logp = all_avg_logp[s1_idx]
                s1_ref_logp = all_ref_avg_logp[s1_idx] if all_ref_avg_logp is not None else None

                # Stage 2 pairs: consecutive [argmax, argmin]
                for p in range(len(s2_indices) // 2):
                    argmax_idx = s2_indices[2 * p]
                    argmin_idx = s2_indices[2 * p + 1]
                    argmax_logp = all_avg_logp[argmax_idx]
                    argmin_logp = all_avg_logp[argmin_idx]

                    # With-ref adjustment
                    if ref_log_prob is not None:
                        argmax_ref = all_ref_avg_logp[argmax_idx]
                        argmin_ref = all_ref_avg_logp[argmin_idx]
                        argmax_logp = argmax_logp - argmax_ref
                        argmin_logp = argmin_logp - argmin_ref
                        s1_logp_adj = s1_logp - s1_ref_logp
                    else:
                        s1_logp_adj = s1_logp

                    # 3-way pairs: (argmax > stage1) and (stage1 > argmin)
                    s1_logp_best_list.append(argmax_logp - s1_logp_adj)
                    s1_logp_worst_list.append(s1_logp_adj - argmin_logp)

            if s1_logp_best_list:
                n_s1_pairs = len(s1_logp_best_list)
                s1_logp_best = torch.stack(s1_logp_best_list)
                s1_logp_worst = torch.stack(s1_logp_worst_list)

                # Stage 1 pairs always use fixed β (no teacher margin available)
                s1_logits_best = dpo_beta * s1_logp_best
                s1_logits_worst = dpo_beta * s1_logp_worst

                s1_loss_best = -torch.nn.functional.logsigmoid(s1_logits_best).mean()
                s1_loss_worst = -torch.nn.functional.logsigmoid(s1_logits_worst).mean()
                stage1_pair_loss = s1_loss_best + s1_loss_worst

        # Total loss
        total_loss = dpo_loss
        if stage1_pair_loss is not None:
            total_loss = dpo_loss + stage1_pair_weight * stage1_pair_loss

        metrics = {
            "actor/dpo_loss": dpo_loss.detach().item(),
            "actor/dpo_n_pairs": float(n_pairs),
            "actor/dpo_logp_chosen": logp_chosen.mean().detach().item(),
            "actor/dpo_logp_rejected": logp_rejected.mean().detach().item(),
            "actor/dpo_margin": (logp_chosen - logp_rejected).mean().detach().item(),
        }
        if ref_log_prob is not None:
            metrics["actor/dpo_ref_chosen"] = ref_chosen.mean().detach().item()
            metrics["actor/dpo_ref_rejected"] = ref_rejected.mean().detach().item()

        # Stage 1 pairing metrics
        if stage1_pair_loss is not None:
            metrics["actor/dpo_stage1_pairs"] = float(n_s1_pairs)
            metrics["actor/dpo_stage1_loss"] = stage1_pair_loss.detach().item()
            metrics["actor/dpo_total_loss"] = total_loss.detach().item()

        # Teacher-Guided β metrics
        if teacher_guided_beta and teacher_margin is not None:
            metrics["actor/dpo_teacher_margin_mean"] = teacher_margin.mean().item()
            metrics["actor/dpo_teacher_margin_std"] = teacher_margin.std().item()
            metrics["actor/dpo_beta_effective_mean"] = effective_beta.mean().item()
            metrics["actor/dpo_beta_effective_min"] = effective_beta.min().item()
            metrics["actor/dpo_beta_effective_max"] = effective_beta.max().item()
        else:
            metrics["actor/dpo_beta_effective_mean"] = float(dpo_beta)

        return total_loss, metrics

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
            return grad_norm

        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

        self_distillation_enabled = loss_mode == "sdpo"
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        if self_distillation_enabled:
            self_distillation_required_keys = {
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_position_ids",
                "self_distillation_mask",
            }
            assert self_distillation_required_keys.issubset(set(data.batch.keys())), f"Missing required keys: {self_distillation_required_keys - set(data.batch.keys())}"

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        elif (float(self.config.policy_loss.get("dpo_coefficient", 0.0)) > 0
              and self.config.policy_loss.get("dpo_use_ref", False)
              and "ref_log_prob" in data.batch.keys()):
            select_keys.append("ref_log_prob")
        if self_distillation_enabled:
            select_keys.extend(list(self_distillation_required_keys))
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        # Include branch_token_mask for the teacher-guided branching rollout's
        # GRPO loss-mode ablations (lands in Phase 2).
        if "branch_token_mask" in data.batch.keys():
            select_keys.append("branch_token_mask")
        # is_branching_fallback travels alongside branch_token_mask: tells
        # _apply_branch_loss_mode which rows came from a fallback path so it
        # can override mode='all' for those rows (else btm=only silently zeros
        # the gradient on the fallback rows).
        if "is_branching_fallback" in data.batch.keys():
            select_keys.append("is_branching_fallback")
        # DPO: is_two_stage_stage1 identifies Stage 2 samples for DPO loss
        if "is_two_stage_stage1" in data.batch.keys():
            select_keys.append("is_two_stage_stage1")

        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )
        non_tensor_select_keys = []
        # DPO Stage 1 pairing: pass sample index for prompt grouping
        if self.config.policy_loss.get("dpo_stage1_pair", False) and "index" in data.non_tensor_batch:
            non_tensor_select_keys.append("index")
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        did_update = False
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)
                    self_distillation_mask = model_inputs.get("self_distillation_mask") if self_distillation_enabled else None
                    if self_distillation_enabled:
                        assert not has_multi_modal_inputs, "Multi-modal inputs are not supported for distillation"

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    teacher_regularization = self_distillation_cfg.get("teacher_regularization", "ema")
                    if teacher_regularization == "trust-region" and self.use_fused_kernels:
                        raise ValueError("trust-region teacher requires disabling fused kernels to access logits.")
                    # all return: (bsz, response_length)
                    return_all_logps = self_distillation_cfg.full_logit_distillation and not self_distillation_cfg.distillation_topk
                    distill_topk = self_distillation_cfg.distillation_topk if self_distillation_cfg.full_logit_distillation else None
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_all_logps=return_all_logps,
                        distill_topk=distill_topk,
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None
                    student_all_logps = outputs.get("all_logps") if return_all_logps else None
                    student_topk_logps = outputs.get("topk_logps") if distill_topk else None
                    student_topk_indices = outputs.get("topk_indices") if distill_topk else None

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # Teacher-guided branching: derive an effective response_mask
                    # from the teacher-injected branch_token_mask, applied to BOTH
                    # the SDPO distill loss and the GRPO/PPO PG loss path so the
                    # ablation modes (all/mask/only) compose consistently with
                    # whichever loss is active. Fallback rows (where the owner
                    # pipeline raised) are forced to mode='all' on a per-row
                    # basis so btm=only doesn't silently zero them out.
                    effective_response_mask, branch_metrics = self._apply_branch_loss_mode(
                        response_mask=response_mask,
                        branch_token_mask=model_inputs.get("branch_token_mask", None),
                        branch_loss_mode=self.config.policy_loss.get("branch_token_loss_mode", "all"),
                        is_branching_fallback=model_inputs.get("is_branching_fallback", None),
                    )
                    if branch_metrics:
                        micro_batch_metrics.update(branch_metrics)

                    if self_distillation_enabled:
                        teacher_inputs = {
                            "responses": model_inputs["responses"],
                            "input_ids": model_inputs["teacher_input_ids"],
                            "attention_mask": model_inputs["teacher_attention_mask"],
                            "position_ids": model_inputs["teacher_position_ids"],
                        }
                        teacher_model = self.teacher_module or self.actor_module
                        if teacher_regularization == "trust-region" and (
                            self.teacher_module is None or self.teacher_module is self.actor_module
                        ):
                            raise ValueError("trust-region teacher requires a separate teacher_module in the actor worker.")
                        with torch.no_grad():
                            teacher_outputs = self._forward_micro_batch(
                                teacher_inputs,
                                temperature=temperature,
                                calculate_entropy=False,
                                return_all_logps=return_all_logps,
                                distill_topk=distill_topk,
                                topk_indices=student_topk_indices,
                                module=teacher_model,
                            )
                        teacher_log_prob = teacher_outputs["log_probs"]
                        teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
                        teacher_topk_logps = teacher_outputs.get("topk_logps") if distill_topk else None
                        pg_loss, pg_metrics = compute_self_distillation_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_prob,
                            response_mask=effective_response_mask,
                            self_distillation_config=self_distillation_cfg,
                            old_log_probs=old_log_prob,
                            student_all_log_probs=student_all_logps,
                            teacher_all_log_probs=teacher_all_logps,
                            student_topk_log_probs=student_topk_logps,
                            teacher_topk_log_probs=teacher_topk_logps,
                            self_distillation_mask=self_distillation_mask,
                            loss_agg_mode=loss_agg_mode,
                            rollout_is_weights=rollout_is_weights,
                        )

                        pg_metrics["self_distillation/empty_target_batch"] = self_distillation_mask.sum().item() == 0
                        micro_batch_metrics.update(pg_metrics)
                    else:
                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)

                        # Compute policy loss (any function is expected to return 2 values)
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=effective_response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss

                    # Pure on-policy DPO: when dpo_coefficient > 0 and this is
                    # a two-stage batch, replace pg_loss with DPO loss on
                    # Stage 2 pos/neg pairs. Stage 1 samples contribute zero
                    # gradient (exploration only).
                    dpo_coefficient = float(self.config.policy_loss.get("dpo_coefficient", 0.0))
                    is_two_stage_stage1 = model_inputs.get("is_two_stage_stage1", None)
                    if dpo_coefficient > 0 and is_two_stage_stage1 is not None and "branch_token_mask" in model_inputs:
                        # Count Stage 2 samples for pairing
                        is_stage2_mask = (is_two_stage_stage1 == 0).float()  # [B]
                        n_stage2 = is_stage2_mask.sum().item()

                        if n_stage2 >= 2:
                            # Replace pg_loss entirely with DPO loss
                            dpo_use_ref = self.config.policy_loss.get("dpo_use_ref", False)
                            dpo_ref = model_inputs.get("ref_log_prob") if dpo_use_ref else None

                            # Teacher-Guided β: extract teacher logprobs from batch
                            teacher_guided_beta = self.config.policy_loss.get("dpo_teacher_guided_beta", False)
                            teacher_branch_lp = model_inputs.get("teacher_branch_logprob") if teacher_guided_beta else None
                            teacher_sibling_lp = model_inputs.get("teacher_sibling_logprob") if teacher_guided_beta else None

                            dpo_loss, dpo_metrics = self._compute_on_policy_dpo_loss(
                                log_prob=log_prob,
                                response_mask=response_mask,
                                branch_token_mask=model_inputs["branch_token_mask"],
                                is_two_stage_stage1=is_two_stage_stage1,
                                dpo_beta=dpo_coefficient,
                                ref_log_prob=dpo_ref,
                                teacher_branch_logprob=teacher_branch_lp,
                                teacher_sibling_logprob=teacher_sibling_lp,
                                teacher_guided_beta=teacher_guided_beta,
                                teacher_beta_alpha=self.config.policy_loss.get("dpo_teacher_beta_alpha", 1.0),
                                teacher_beta_min=self.config.policy_loss.get("dpo_teacher_beta_min", 0.1),
                                teacher_beta_max=self.config.policy_loss.get("dpo_teacher_beta_max", 3.0),
                                dpo_stage1_pair=self.config.policy_loss.get("dpo_stage1_pair", False),
                                stage1_pair_weight=self.config.policy_loss.get("dpo_stage1_pair_weight", 1.0),
                                sample_index=model_inputs.get("index", None),
                            )
                            policy_loss = dpo_loss
                            micro_batch_metrics.update(dpo_metrics)
                        else:
                            # Not enough Stage 2 samples — fall back to pg_loss
                            # but only for Stage 1 (no Stage 2 pairs to train)
                            micro_batch_metrics["actor/dpo_n_pairs"] = 0.0

                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                if torch.isfinite(grad_norm).item():
                    did_update = True
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        if did_update:
            self._update_teacher()
        return metrics
