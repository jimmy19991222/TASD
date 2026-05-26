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
from verl.trainer.ppo.core_algos import (
    agg_loss,
    compute_geodesic_manifold_weight,
    compute_self_distillation_loss,
    compute_teacher_qv_advantage,
    compute_teacher_qv_full_logit_loss,
    get_policy_loss_fn,
    kl_penalty,
)
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


def _qv_add_tail(log_probs: torch.Tensor) -> torch.Tensor:
    """Append a synthetic 'everything else' bucket so a topk log-prob slab
    normalizes to 1 over the last dim. Same trick used in SDPO's
    compute_self_distillation_loss.

    log_probs: (..., K) -> (..., K+1) with last column = log(1 - sum exp(log_probs)).

    Sanitization: at masked positions the forward returns garbage logits, which
    can drive log_s past 0 and trigger NaN through expm1/log. We additionally
    clamp log_s below as well, and scrub any residual NaN/inf to a safe constant
    so downstream V_t computation stays finite.
    """
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    # Clamp BOTH sides: above to avoid log(<=0), below to avoid log(0).
    log_s = torch.clamp(log_s, min=-30.0, max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_s))
    tail_log = torch.nan_to_num(tail_log, nan=-30.0, posinf=-30.0, neginf=-30.0)
    return torch.cat([log_probs, tail_log], dim=-1)

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
        if not self_distillation_cfg or loss_mode not in ("sdpo", "teacher_qv"):
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
        teacher_qv_enabled = loss_mode == "teacher_qv"
        # VCAC: GRPO PG loss with per-token advantage shaped by CDM dual-teacher
        # δ_t = log π_T^+(y_t) − log π_T^-(y_t). Requires teacher_context_mode='cdm'
        # so the dual teacher batch (teacher_input_ids_{pos,neg}) is built.
        vcac_enabled = loss_mode == "vcac"
        # Bayes-DR: A_t = R − σ(Σ_{s<t} δ_s + logit ḡ_GRPO).
        # Cumulative δ_t is the Bayes-implied success log-odds shift; the per-token
        # baseline b_t depends only on y_<t so subtraction is an unbiased control
        # variate over the PG estimator. Requires the CDM dual-teacher forward.
        bayes_dr_enabled = loss_mode == "bayes_dr"
        teacher_forward_required = self_distillation_enabled or teacher_qv_enabled or vcac_enabled or bayes_dr_enabled
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        teacher_qv_cfg = self.config.policy_loss.get("teacher_qv", None) if teacher_qv_enabled else None
        verdict_loss_active = (
            self_distillation_enabled
            and self_distillation_cfg is not None
            and self_distillation_cfg.get("loss_method", "sdpo") in ("vc_opsd_sign", "opd_bayes", "vec")
        )
        cdm_loss_active = (
            self_distillation_enabled
            and self_distillation_cfg is not None
            and self_distillation_cfg.get("loss_method", "sdpo") == "cdm"
        )
        # Both families need the second (negative) teacher forward.
        dual_teacher_active = verdict_loss_active or cdm_loss_active or vcac_enabled or bayes_dr_enabled
        if vcac_enabled:
            if self_distillation_cfg is None:
                raise ValueError("loss_mode='vcac' requires self_distillation config to be present.")
            if self_distillation_cfg.get("teacher_context_mode", "ref") != "cdm":
                raise ValueError(
                    "loss_mode='vcac' requires self_distillation.teacher_context_mode='cdm' "
                    "so the dual (positive/negative verdict) teacher inputs are built."
                )
        if bayes_dr_enabled:
            if self_distillation_cfg is None:
                raise ValueError("loss_mode='bayes_dr' requires self_distillation config to be present.")
            if self_distillation_cfg.get("teacher_context_mode", "ref") != "cdm":
                raise ValueError(
                    "loss_mode='bayes_dr' requires self_distillation.teacher_context_mode='cdm' "
                    "so the dual (positive/negative verdict) teacher inputs are built."
                )
        if teacher_forward_required:
            self_distillation_required_keys = {
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_position_ids",
                "self_distillation_mask",
            }
            if dual_teacher_active:
                self_distillation_required_keys |= {
                    "teacher_input_ids_neg",
                    "teacher_attention_mask_neg",
                    "teacher_position_ids_neg",
                }
            if verdict_loss_active:
                # Verdict family also needs the sequence-level R label.
                self_distillation_required_keys |= {"verdict_R"}
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
        if teacher_forward_required:
            select_keys.extend(list(self_distillation_required_keys))
        if bayes_dr_enabled:
            # ray_trainer writes seq_R (per-sequence R) and grpo_group_mean
            # (per-prompt mean R, used as the prior in logit space).
            for _bdr_key in ("seq_R", "grpo_group_mean"):
                if _bdr_key in data.batch.keys() and _bdr_key not in select_keys:
                    select_keys.append(_bdr_key)
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        teacher_qv_needs_uid = (
            teacher_qv_enabled
            and teacher_qv_cfg is not None
            and teacher_qv_cfg.get("baseline_type", "student") in ("group_hier",)
        )
        if (
            (self.use_prefix_grouper or teacher_qv_needs_uid)
            and "uid" in data.non_tensor_batch.keys()
            and "uid" not in non_tensor_select_keys
        ):
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
                    self_distillation_mask = model_inputs.get("self_distillation_mask") if teacher_forward_required else None
                    if teacher_forward_required:
                        assert not has_multi_modal_inputs, "Multi-modal inputs are not supported for distillation"

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    if teacher_forward_required:
                        teacher_regularization = self_distillation_cfg.get("teacher_regularization", "ema")
                        if teacher_regularization == "trust-region" and self.use_fused_kernels:
                            raise ValueError("trust-region teacher requires disabling fused kernels to access logits.")
                        # teacher_qv with baseline_type='ce' needs full or topk-aligned vocab
                        # log-probs on both sides to compute V_t = E_{y~p_s}[log p_t(y)].
                        # Other baselines only need the sampled-token log-prob.
                        qv_needs_full_logits = (
                            teacher_qv_enabled
                            and teacher_qv_cfg is not None
                            and (
                                teacher_qv_cfg.get("baseline_type", "student") == "ce"
                                or teacher_qv_cfg.get("gradient_mode", "sampled") == "full_logit"
                            )
                        )
                        if teacher_qv_enabled and not qv_needs_full_logits:
                            return_all_logps = False
                            distill_topk = None
                        else:
                            return_all_logps = self_distillation_cfg.full_logit_distillation and not self_distillation_cfg.distillation_topk
                            distill_topk = self_distillation_cfg.distillation_topk if self_distillation_cfg.full_logit_distillation else None
                            if qv_needs_full_logits and not (return_all_logps or distill_topk):
                                raise ValueError(
                                    "teacher_qv baseline_type='ce' requires self_distillation.full_logit_distillation=True "
                                    "(optionally with distillation_topk to limit memory)."
                                )
                    else:
                        teacher_regularization = None
                        return_all_logps = False
                        distill_topk = None
                        qv_needs_full_logits = False
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

                    if teacher_forward_required:
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

                        teacher_log_prob_neg = None
                        teacher_all_logps_neg = None
                        teacher_topk_logps_neg = None
                        verdict_R = None
                        if dual_teacher_active:
                            teacher_inputs_neg = {
                                "responses": model_inputs["responses"],
                                "input_ids": model_inputs["teacher_input_ids_neg"],
                                "attention_mask": model_inputs["teacher_attention_mask_neg"],
                                "position_ids": model_inputs["teacher_position_ids_neg"],
                            }
                            with torch.no_grad():
                                teacher_outputs_neg = self._forward_micro_batch(
                                    teacher_inputs_neg,
                                    temperature=temperature,
                                    calculate_entropy=False,
                                    return_all_logps=return_all_logps,
                                    distill_topk=distill_topk,
                                    topk_indices=student_topk_indices,
                                    module=teacher_model,
                                )
                            teacher_log_prob_neg = teacher_outputs_neg["log_probs"]
                            teacher_all_logps_neg = teacher_outputs_neg.get("all_logps") if return_all_logps else None
                            teacher_topk_logps_neg = teacher_outputs_neg.get("topk_logps") if distill_topk else None
                            if verdict_loss_active:
                                verdict_R = model_inputs["verdict_R"]

                    if self_distillation_enabled:
                        pg_loss, pg_metrics = compute_self_distillation_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_prob,
                            response_mask=response_mask,
                            self_distillation_config=self_distillation_cfg,
                            old_log_probs=old_log_prob,
                            student_all_log_probs=student_all_logps,
                            teacher_all_log_probs=teacher_all_logps,
                            student_topk_log_probs=student_topk_logps,
                            teacher_topk_log_probs=teacher_topk_logps,
                            student_topk_indices=student_topk_indices,
                            self_distillation_mask=self_distillation_mask,
                            loss_agg_mode=loss_agg_mode,
                            rollout_is_weights=rollout_is_weights,
                            teacher_log_probs_neg=teacher_log_prob_neg if dual_teacher_active else None,
                            teacher_all_log_probs_neg=teacher_all_logps_neg if dual_teacher_active else None,
                            teacher_topk_log_probs_neg=teacher_topk_logps_neg if dual_teacher_active else None,
                            verdict_R=verdict_R if verdict_loss_active else None,
                            response_ids=model_inputs.get("responses"),
                        )

                        pg_metrics["self_distillation/empty_target_batch"] = self_distillation_mask.sum().item() == 0
                        micro_batch_metrics.update(pg_metrics)
                    elif teacher_qv_enabled:
                        # PG with token-level A_t = log p_teacher - V_t.
                        # V_t is selected by ``policy_loss.teacher_qv.baseline_type``.
                        loss_mask = response_mask
                        if self_distillation_mask is not None:
                            loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)
                        qv_index = model_inputs.get("uid", None)
                        qv_baseline_type = teacher_qv_cfg.get("baseline_type", "student")
                        qv_gradient_mode = teacher_qv_cfg.get("gradient_mode", "sampled")

                        # Full vocab log probs needed when:
                        #   - baseline_type=ce (V_t expectation)
                        #   - gradient_mode=full_logit (vocab-summed PG)
                        qv_student_full = None
                        qv_teacher_full = None
                        if qv_baseline_type == "ce" or qv_gradient_mode == "full_logit":
                            if student_all_logps is not None and teacher_all_logps is not None:
                                qv_student_full = student_all_logps
                                qv_teacher_full = teacher_all_logps
                            elif student_topk_logps is not None and teacher_topk_logps is not None:
                                qv_student_full = _qv_add_tail(student_topk_logps)
                                qv_teacher_full = _qv_add_tail(teacher_topk_logps)
                            else:
                                raise ValueError(
                                    f"teacher_qv (baseline={qv_baseline_type}, gradient_mode={qv_gradient_mode}) "
                                    "requires self_distillation.full_logit_distillation=True."
                                )

                        if qv_gradient_mode == "full_logit":
                            # Vocab-summed PG: loss = -Σ_v p_s(v)·A(v)·log p_s(v).
                            # Matches old SDPO alpha=1 full_logit for baseline=student;
                            # forward CE for baseline in {ce, group_mean, group_hier}.
                            assert qv_student_full is not None
                            pg_loss, pg_metrics = compute_teacher_qv_full_logit_loss(
                                student_full_log_probs=qv_student_full,
                                teacher_full_log_probs=qv_teacher_full,
                                response_mask=loss_mask,
                                baseline_type=qv_baseline_type,
                                loss_agg_mode=loss_agg_mode,
                                rollout_is_weights=rollout_is_weights,
                                **self.config.global_batch_info,
                            )
                        else:
                            # Sampled-token PG: A_t = log p_t(y_t) - V_t, loss = -A · log p_s(y_t).
                            qv_advantages, qv_metrics = compute_teacher_qv_advantage(
                                student_log_probs=log_prob,
                                teacher_log_probs=teacher_log_prob,
                                response_mask=loss_mask,
                                baseline_type=qv_baseline_type,
                                index=qv_index,
                                norm_by_std=teacher_qv_cfg.get("norm_by_std", False),
                                clip_value=teacher_qv_cfg.get("clip_value", None),
                                std_floor=teacher_qv_cfg.get("std_floor", 1e-3),
                                student_all_log_probs=qv_student_full,
                                teacher_all_log_probs=qv_teacher_full,
                            )
                            # Geodesic Fisher manifold weight (orthogonal to baseline_type).
                            # Reuses self_distillation.use_geodesic / geodesic_trust_region so SDPO
                            # and teacher_qv share the same Geodesic switch.
                            if self_distillation_cfg.get("use_geodesic", False):
                                trust_region = float(
                                    self_distillation_cfg.get("geodesic_trust_region", 5.0)
                                )
                                geodesic_full = qv_student_full  # populated when ce mode runs
                                manifold_weight, geodesic_metrics = compute_geodesic_manifold_weight(
                                    student_log_probs=log_prob,
                                    student_all_log_probs=geodesic_full,
                                    trust_region_scale=trust_region,
                                    response_mask=loss_mask,
                                )
                                qv_advantages = qv_advantages * manifold_weight
                                qv_advantages = torch.where(
                                    loss_mask.bool() & torch.isfinite(qv_advantages),
                                    qv_advantages,
                                    torch.zeros_like(qv_advantages),
                                )
                                qv_metrics.update(geodesic_metrics)
                            from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
                            pg_loss, pg_metrics = compute_policy_loss_vanilla(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=qv_advantages,
                                response_mask=loss_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                            )
                            pg_metrics.update(qv_metrics)
                        # ---- SDPO-style alpha mixing (forward-KL anti-collapse) ----
                        # When alpha < 1, mirror SDPO's JSD form:
                        #     loss = alpha · PG_loss + (1 - alpha) · KL(p_teacher || p_student)
                        # The forward-KL component is mass-covering and gives the student
                        # distribution a floor on tokens the teacher still assigns mass to,
                        # counteracting the reverse-KL mode-seeking collapse. Reuses the
                        # same ``self_distillation.alpha`` knob that SDPO already exposes —
                        # no teacher_qv-specific hyperparameter is introduced. alpha == 1
                        # (default) preserves the original behaviour exactly.
                        qv_alpha = float(self_distillation_cfg.get("alpha", 1.0))
                        if qv_alpha < 1.0:
                            if qv_student_full is None or qv_teacher_full is None:
                                raise ValueError(
                                    "teacher_qv with self_distillation.alpha < 1 requires "
                                    "full-vocab log probs (gradient_mode='full_logit' or "
                                    "baseline_type='ce', with full_logit_distillation=True)."
                                )
                            with torch.no_grad():
                                p_teacher = qv_teacher_full.exp()
                            fkl_per_token = (
                                p_teacher * (qv_teacher_full - qv_student_full)
                            ).sum(dim=-1)
                            fkl_per_token = torch.where(
                                loss_mask.bool() & torch.isfinite(fkl_per_token),
                                fkl_per_token,
                                torch.zeros_like(fkl_per_token),
                            )
                            fkl_loss = agg_loss(
                                loss_mat=fkl_per_token,
                                loss_mask=loss_mask,
                                loss_agg_mode=loss_agg_mode,
                            )
                            pg_loss = qv_alpha * pg_loss + (1.0 - qv_alpha) * fkl_loss
                            pg_metrics["actor/teacher_qv_fkl_loss"] = fkl_loss.detach().item()
                            pg_metrics["actor/teacher_qv_alpha"] = qv_alpha
                        # ---- shared post-dispatch ----
                        if self_distillation_mask is not None:
                            pg_metrics["teacher_qv/empty_target_batch"] = (
                                self_distillation_mask.sum().item() == 0
                            )
                        micro_batch_metrics.update(pg_metrics)
                    elif vcac_enabled:
                        # VCAC: GRPO PG loss with per-token credit shaping.
                        #   δ_t = log π_T^+(y_t|x,y_<t) − log π_T^-(y_t|x,y_<t)
                        # is the Bayes-clean verdict log-odds shift from the CDM
                        # dual-teacher forward (sampled-token log-probs). Mixed
                        # into the GRPO advantage and fed into the standard
                        # vanilla PG loss with PPO clip.
                        assert teacher_log_prob_neg is not None, (
                            "VCAC requires the negative-teacher forward (dual_teacher_active)."
                        )
                        from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

                        delta_t = (teacher_log_prob - teacher_log_prob_neg).detach()
                        # δ_t is only defined on response positions; zero elsewhere.
                        delta_t = delta_t * response_mask
                        vcac_lambda = float(self.config.policy_loss.get("vcac_lambda", 0.3))
                        vcac_clip_val = self.config.policy_loss.get("vcac_clip", None)
                        vcac_normalize = bool(self.config.policy_loss.get("vcac_normalize", False))
                        vcac_use_grpo = bool(
                            self.config.policy_loss.get("vcac_use_grpo_advantage", True)
                        )

                        # All VCAC δ-metrics are initialized up-front with NaN so every
                        # micro-batch contributes the same key set. Otherwise conditional
                        # appends (R0/R1 partition, normalize-only, empty-target) make
                        # reduce_metrics see inhomogeneous lists across ranks and crash on
                        # np.mean(val).
                        delta_metrics = {
                            "vcac/delta_norm_mean_pre": float("nan"),
                            "vcac/delta_norm_std_pre": float("nan"),
                            "vcac/delta_mean_R0": float("nan"),
                            "vcac/delta_abs_mean_R0": float("nan"),
                            "vcac/delta_mean_R1": float("nan"),
                            "vcac/delta_abs_mean_R1": float("nan"),
                        }
                        mask_bool = response_mask.bool()
                        valid_count = mask_bool.sum().clamp_min(1)

                        if vcac_normalize:
                            delta_sum = delta_t.sum()
                            delta_mean = (delta_sum / valid_count).detach()
                            delta_centered = (delta_t - delta_mean) * response_mask
                            delta_var = ((delta_centered ** 2).sum() / valid_count).detach()
                            delta_std = (delta_var.clamp_min(1e-12).sqrt() + 1e-6)
                            delta_t = delta_centered / delta_std
                            delta_metrics["vcac/delta_norm_mean_pre"] = delta_mean.item()
                            delta_metrics["vcac/delta_norm_std_pre"] = delta_std.item()

                        if vcac_clip_val is not None and float(vcac_clip_val) > 0:
                            cv = float(vcac_clip_val)
                            delta_t = torch.clamp(delta_t, min=-cv, max=cv)

                        # δ-distribution diagnostics on valid response tokens.
                        with torch.no_grad():
                            d_valid = delta_t[mask_bool]
                            if d_valid.numel() > 0:
                                delta_metrics["vcac/delta_mean"] = d_valid.mean().item()
                                delta_metrics["vcac/delta_std"] = d_valid.std().item() if d_valid.numel() > 1 else 0.0
                                delta_metrics["vcac/delta_abs_mean"] = d_valid.abs().mean().item()
                                delta_metrics["vcac/delta_pos_frac"] = (d_valid > 0).float().mean().item()
                                delta_metrics["vcac/delta_neg_frac"] = (d_valid < 0).float().mean().item()
                            else:
                                delta_metrics["vcac/delta_mean"] = 0.0
                                delta_metrics["vcac/delta_std"] = 0.0

                            # Sign-check on R=0 subset (see feedback_belief_pg_sign_check):
                            # when GRPO advantage is non-positive on these rows, δ adding
                            # mass back to teacher-pos tokens should not contradict R.
                            a_seq = advantages.sum(dim=-1) if advantages.dim() == 2 else advantages
                            # Use sample-level R proxy: GRPO advantage sum sign per sequence.
                            r0_mask_seq = (a_seq < 0)
                            r1_mask_seq = (a_seq > 0)
                            if r0_mask_seq.any():
                                d_r0 = delta_t[r0_mask_seq][response_mask[r0_mask_seq].bool()]
                                if d_r0.numel() > 0:
                                    delta_metrics["vcac/delta_mean_R0"] = d_r0.mean().item()
                                    delta_metrics["vcac/delta_abs_mean_R0"] = d_r0.abs().mean().item()
                            if r1_mask_seq.any():
                                d_r1 = delta_t[r1_mask_seq][response_mask[r1_mask_seq].bool()]
                                if d_r1.numel() > 0:
                                    delta_metrics["vcac/delta_mean_R1"] = d_r1.mean().item()
                                    delta_metrics["vcac/delta_abs_mean_R1"] = d_r1.abs().mean().item()

                        if vcac_use_grpo:
                            advantages_vcac = advantages + vcac_lambda * delta_t
                        else:
                            # Pure-δ ablation: drop the GRPO baseline term.
                            advantages_vcac = (vcac_lambda * delta_t) * response_mask
                        # Guard against NaN/inf from teacher forward.
                        advantages_vcac = torch.where(
                            torch.isfinite(advantages_vcac),
                            advantages_vcac,
                            advantages,
                        )

                        pg_loss, pg_metrics = compute_policy_loss_vanilla(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages_vcac,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        pg_metrics.update(delta_metrics)
                        pg_metrics["vcac/lambda"] = vcac_lambda
                        pg_metrics["vcac/use_grpo_advantage"] = float(vcac_use_grpo)
                        # Always emit so reduce_metrics sees a uniform key set across
                        # ranks / micro-batches (avoids inhomogeneous-list np.mean crash).
                        pg_metrics["vcac/empty_target_batch"] = float(
                            self_distillation_mask.sum().item() == 0
                            if self_distillation_mask is not None
                            else 0.0
                        )
                        micro_batch_metrics.update(pg_metrics)
                    elif bayes_dr_enabled:
                        # Bayes-DR: A_t = R − σ(Σ_{s<t} δ_s + logit ḡ_GRPO).
                        # The cumulative-δ baseline b_t depends only on y_<t, so
                        # subtracting it from R is an unbiased Rao-Blackwell
                        # control variate. With well-calibrated δ, b_t tracks the
                        # current model's posterior success probability and A_t
                        # concentrates credit on the verdict-shifting turn-tokens.
                        assert teacher_log_prob_neg is not None, (
                            "Bayes-DR requires the negative-teacher forward (dual_teacher_active)."
                        )
                        seq_R = model_inputs.get("seq_R", None)
                        grpo_group_mean = model_inputs.get("grpo_group_mean", None)
                        if seq_R is None or grpo_group_mean is None:
                            raise ValueError(
                                "loss_mode='bayes_dr' requires seq_R and grpo_group_mean in batch "
                                "(written by ray_trainer GRPO advantage path)."
                            )
                        from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla

                        bdr_cfg = self.config.policy_loss
                        bdr_delta_clip = bdr_cfg.get("bayes_dr_delta_clip", None)
                        bdr_prior_floor = float(bdr_cfg.get("bayes_dr_prior_floor", 0.05))
                        bdr_use_seq_anchor = bool(bdr_cfg.get("bayes_dr_use_seq_anchor", True))
                        bdr_normalize_adv = bool(bdr_cfg.get("bayes_dr_normalize_adv", False))
                        bdr_baseline_detach = bool(bdr_cfg.get("bayes_dr_baseline_detach", True))

                        delta_t = (teacher_log_prob - teacher_log_prob_neg).detach()
                        delta_t = delta_t * response_mask
                        if bdr_delta_clip is not None and float(bdr_delta_clip) > 0:
                            cv = float(bdr_delta_clip)
                            delta_t = torch.clamp(delta_t, min=-cv, max=cv)

                        # Cumulative δ shifted by one position: b_t depends only on y_<t.
                        cum_delta = torch.cumsum(delta_t, dim=-1)
                        cum_delta_prev = torch.cat(
                            [torch.zeros_like(cum_delta[:, :1]), cum_delta[:, :-1]], dim=-1
                        )

                        # Prior in logit space, clamped away from {0, 1}.
                        prior = grpo_group_mean.clamp(bdr_prior_floor, 1.0 - bdr_prior_floor)
                        prior_logit = torch.log(prior / (1.0 - prior))
                        # [B] -> [B, 1] broadcast over T.
                        b_t = torch.sigmoid(cum_delta_prev + prior_logit[:, None])

                        if bdr_use_seq_anchor:
                            # A_t = R − b_t (R is per-sequence outcome reward).
                            anchor = seq_R[:, None].to(b_t.dtype)
                        else:
                            # Ablation: use sign(per-sequence GRPO advantage) as anchor.
                            a_seq = advantages.sum(dim=-1) if advantages.dim() == 2 else advantages
                            anchor = torch.sign(a_seq)[:, None].to(b_t.dtype)
                        advantages_bdr = (anchor - b_t) * response_mask
                        if bdr_baseline_detach:
                            advantages_bdr = advantages_bdr.detach()

                        # Guard against NaN/inf from teacher forward.
                        advantages_bdr = torch.where(
                            torch.isfinite(advantages_bdr),
                            advantages_bdr,
                            advantages,
                        )

                        if bdr_normalize_adv:
                            mask_bool = response_mask.bool()
                            valid_count = mask_bool.sum().clamp_min(1)
                            a_mean = (advantages_bdr.sum() / valid_count).detach()
                            a_var = (((advantages_bdr - a_mean) ** 2 * response_mask).sum() / valid_count).detach()
                            a_std = (a_var.clamp_min(1e-12).sqrt() + 1e-6)
                            advantages_bdr = ((advantages_bdr - a_mean) / a_std) * response_mask

                        # Diagnostics initialized as NaN up-front so reduce_metrics
                        # sees a uniform key set across micro-batches/ranks.
                        bdr_metrics = {
                            "bayes_dr/cum_delta_T_mean": float("nan"),
                            "bayes_dr/b_t_mean": float("nan"),
                            "bayes_dr/b_t_std": float("nan"),
                            "bayes_dr/A_t_mean": float("nan"),
                            "bayes_dr/A_t_std": float("nan"),
                            "bayes_dr/A_t_abs_mean": float("nan"),
                            "bayes_dr/corr_cum_delta_R": float("nan"),
                            "bayes_dr/delta_abs_mean": float("nan"),
                        }
                        with torch.no_grad():
                            mask_b = response_mask.bool()
                            if mask_b.any():
                                d_valid = delta_t[mask_b]
                                bdr_metrics["bayes_dr/delta_abs_mean"] = d_valid.abs().mean().item()
                                b_valid = b_t[mask_b]
                                bdr_metrics["bayes_dr/b_t_mean"] = b_valid.mean().item()
                                bdr_metrics["bayes_dr/b_t_std"] = (
                                    b_valid.std().item() if b_valid.numel() > 1 else 0.0
                                )
                                a_valid = advantages_bdr[mask_b]
                                bdr_metrics["bayes_dr/A_t_mean"] = a_valid.mean().item()
                                bdr_metrics["bayes_dr/A_t_std"] = (
                                    a_valid.std().item() if a_valid.numel() > 1 else 0.0
                                )
                                bdr_metrics["bayes_dr/A_t_abs_mean"] = a_valid.abs().mean().item()

                                # Per-sequence cumulative δ at the last response token
                                # vs sequence-level R; high correlation ⇒ δ is
                                # well-calibrated as a success-log-odds proxy.
                                last_idx = response_mask.long().sum(dim=-1).clamp_min(1) - 1
                                arange = torch.arange(cum_delta.shape[0], device=cum_delta.device)
                                cum_delta_T = cum_delta[arange, last_idx]
                                bdr_metrics["bayes_dr/cum_delta_T_mean"] = cum_delta_T.mean().item()
                                if cum_delta_T.numel() > 1 and seq_R.numel() > 1:
                                    a = cum_delta_T.float()
                                    b = seq_R.float()
                                    a_c = a - a.mean()
                                    b_c = b - b.mean()
                                    denom = (a_c.norm() * b_c.norm()).clamp_min(1e-8)
                                    bdr_metrics["bayes_dr/corr_cum_delta_R"] = (
                                        (a_c * b_c).sum() / denom
                                    ).item()

                        pg_loss, pg_metrics = compute_policy_loss_vanilla(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages_bdr,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        pg_metrics.update(bdr_metrics)
                        pg_metrics["bayes_dr/prior_floor"] = bdr_prior_floor
                        pg_metrics["bayes_dr/use_seq_anchor"] = float(bdr_use_seq_anchor)
                        pg_metrics["bayes_dr/normalize_adv"] = float(bdr_normalize_adv)
                        pg_metrics["bayes_dr/empty_target_batch"] = float(
                            self_distillation_mask.sum().item() == 0
                            if self_distillation_mask is not None
                            else 0.0
                        )
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
                            response_mask=response_mask,
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
