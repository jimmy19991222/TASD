# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
TCCA V2: Iterative Teacher-Guided Chain Rollout with Divergence-Point Credit.

设计文档: research/tcca_v2_design.md (commit cf18d20 之后版本)

核心流程:
  Step 0: y_0 = standard student rollout (per prompt, n=1)
  for i in 1..chain_length-1:
    Step A: 构造 OPSD teacher context (含 reference answer)
    Step B+C: teacher forward → logp_T_opsd; student logp from rollout_log_probs
    Step D: t_i = argmax |logp_T - logp_S|, exclude tail 8
    Step E: teacher's argmax token at t_i under OPSD context (1 token)
    Step F+G: student async continuation from (prompt + y_<t_i + teacher_token)
    Step H: 构造 y_i = y_<t_i + teacher_token + student_continuation
            response_mask[:, :t_i] = 0 (Layer 2: shared prefix off)
    Step I: reward_fn(y_i) → R(y_i); ΔR_i = R(y_i) - R(y_{i-1})

  Final: augmented = concat([y_0, y_1, ..., y_{n-1}])
         同 prompt 的 n samples 同 uid → GRPO group-relative 自然处理
         divergence_credit (B*n, T) 写入 batch 供 advantage 计算用

Advantage 公式 (intervention_credit estimator):
  A_t = (A_seq + λ_div · c_t) · response_mask · length_scale     (additive)
  其中 A_seq = R_i - mean_group(R) (GRPO group-relative)
       c_t = +ΔR_i 在 y_i 的 t_i 位置, -ΔR_i 在 y_{i-1} 的同位置, 0 elsewhere
"""

from __future__ import annotations

import asyncio
from typing import Optional
from uuid import uuid4

import numpy as np
import torch

from verl.protocol import DataProto


def compute_position_id_with_mask(attention_mask):
    """Compute position ids from attention mask (left-padded sequences)."""
    return torch.clip(attention_mask.cumsum(dim=-1) - 1, min=0, max=None)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def tcca_v2_chain_rollout(
    prompt_batch: DataProto,
    actor_rollout_wg,
    async_rollout_manager,
    reward_fn,
    config,
    tokenizer,
) -> DataProto:
    """Main entry. 替换标准 rollout 时使用。

    Args:
        prompt_batch: 含 non_tensor_batch[raw_prompt, reward_model, ...] 的 DataProto
                      尚未 repeat (1 个 prompt 对应 1 行)
        actor_rollout_wg: 用于 teacher FSDP forward + 1-token decode
        async_rollout_manager: AgentLoopManager, .server_manager.generate 用于 student 续写
        reward_fn: AbstractRewardManager
        config: AlgoConfig
        tokenizer: HF tokenizer

    Returns:
        DataProto with B*chain_length rows, 含:
          - prompts, responses, response_mask (shared prefix 已 mask), input_ids, attention_mask, position_ids
          - token_level_scores, token_level_rewards (reward 已重打分)
          - non_tensor_batch[uid] (同 prompt 的 chain samples 同 uid)
          - batch['divergence_credit'] (B*chain_length, T): per-token c_t for Layer 3
    """
    ic_cfg = config.algorithm.get("intervention_credit", {}) or {}
    chain_length: int = int(ic_cfg.get("chain_length", 8))
    enable_intervention: bool = bool(ic_cfg.get("enable_intervention", False))

    # 退化路径: 标准 GRPO rollout
    if not enable_intervention:
        return _standard_rollout(prompt_batch, n=chain_length, async_rollout_manager=async_rollout_manager)

    # 构造 OPSD ctx (一次, 所有 iteration 共用)
    opsd_ctx = _build_opsd_teacher_context(prompt_batch, tokenizer, config)

    # Step 0: y_0 = standard rollout n=1 per prompt
    y_0_batch = _standard_rollout(prompt_batch, n=1, async_rollout_manager=async_rollout_manager)
    chain_batches = [y_0_batch]
    delta_r_history = []   # ΔR_1, ..., ΔR_{n-1}
    t_history = [None]     # None, t_1, ..., t_{n-1}

    B = y_0_batch.batch.batch_size[0]
    T = y_0_batch.batch["responses"].shape[1]

    R_prev = y_0_batch.batch["token_level_rewards"].sum(dim=-1) if "token_level_rewards" in y_0_batch.batch else None
    if R_prev is None:
        # y_0 还没打分, 现在打
        y_0_batch = _rescore_reward(y_0_batch, reward_fn)
        chain_batches[0] = y_0_batch
        R_prev = y_0_batch.batch["token_level_rewards"].sum(dim=-1)

    for i in range(1, chain_length):
        y_prev = chain_batches[-1]

        # Step B+C: divergence
        divergence = _compute_divergence_opsd(y_prev, opsd_ctx, actor_rollout_wg, config, tokenizer)
        # Step D: t_i
        exclude_tail = int(ic_cfg.get("exclude_tail_tokens", 8))
        t_i = _select_t_i(divergence, y_prev.batch["response_mask"], exclude_tail=exclude_tail)
        t_history.append(t_i)

        # Step E: teacher 写 1 token
        teacher_tokens = _teacher_write_one_token(
            y_prev, opsd_ctx, t_i, actor_rollout_wg, config, tokenizer
        )

        # Step F+G: student 续写
        continuations = _student_continue_async(
            y_prev, t_i, teacher_tokens, async_rollout_manager, config, tokenizer
        )

        # Step H: 构造 y_i
        y_i = _build_y_i_batch(y_prev, t_i, teacher_tokens, continuations, tokenizer)

        # Step I: reward
        y_i = _rescore_reward(y_i, reward_fn)
        R_i = y_i.batch["token_level_rewards"].sum(dim=-1)
        delta_r_i = R_i - R_prev   # (B,)
        delta_r_history.append(delta_r_i)
        R_prev = R_i

        chain_batches.append(y_i)

    # === Concat all chain samples ===
    augmented = DataProto.concat(chain_batches)

    # === Build divergence_credit ===
    divergence_credit = _build_divergence_credit(
        chain_batches, t_history, delta_r_history, B, chain_length
    )
    augmented.batch["divergence_credit"] = divergence_credit

    return augmented


# ──────────────────────────────────────────────────────────────────────
# Step 0 & fallback: standard rollout
# ──────────────────────────────────────────────────────────────────────


def _standard_rollout(prompt_batch: DataProto, n: int, async_rollout_manager) -> DataProto:
    """调 async_rollout_manager.generate_sequences 做 n 次 i.i.d. rollout per prompt."""
    rep = prompt_batch.repeat(repeat_times=n, interleave=True)
    rep.meta_info.setdefault("global_steps", prompt_batch.meta_info.get("global_steps", 0))
    return async_rollout_manager.generate_sequences(rep)


# ──────────────────────────────────────────────────────────────────────
# Step A: OPSD teacher context (含 reference answer)
# ──────────────────────────────────────────────────────────────────────


def _build_opsd_teacher_context(prompt_batch: DataProto, tokenizer, config) -> dict:
    """加 'The correct answer is {r}.\n' 到 user prompt 前缀, apply chat template."""
    ic_cfg = config.algorithm.get("intervention_credit", {}) or {}
    ref_template: str = str(ic_cfg.get("opsd_ref_template", "The correct answer is {r}.\n"))

    B = prompt_batch.batch.batch_size[0]
    messages_with_ref = []
    for i in range(B):
        rm = prompt_batch.non_tensor_batch.get("reward_model", [None])[i]
        if isinstance(rm, dict):
            ref = str(rm.get("ground_truth", ""))
        else:
            ref = str(rm) if rm is not None else ""
        raw_msgs = prompt_batch.non_tensor_batch["raw_prompt"][i]
        ref_text = ref_template.format(r=ref)
        new_msgs = [dict(m) for m in raw_msgs]
        # 加 ref 到最后一个 user message 前缀
        for j in range(len(new_msgs) - 1, -1, -1):
            if new_msgs[j].get("role") == "user":
                new_msgs[j]["content"] = ref_text + new_msgs[j]["content"]
                break
        messages_with_ref.append(new_msgs)

    sd_cfg = config.actor_rollout_ref.actor.get("self_distillation", {}) or {}
    max_len = int(sd_cfg.get("max_reprompt_len", 10240))
    try:
        enable_thinking = config.data.apply_chat_template_kwargs.get("enable_thinking", True) \
                          if config.data.apply_chat_template_kwargs else True
    except Exception:
        enable_thinking = True

    opsd = tokenizer.apply_chat_template(
        messages_with_ref,
        tokenize=True, return_tensors="pt", return_dict=True,
        add_generation_prompt=True, padding=True, truncation=True, max_length=max_len,
        enable_thinking=enable_thinking,
    )
    device = prompt_batch.batch["input_ids"].device
    opsd_input_ids = opsd["input_ids"].to(device)
    opsd_attn = opsd["attention_mask"].to(device)
    opsd_pos = compute_position_id_with_mask(opsd_attn)
    return {"input_ids": opsd_input_ids, "attention_mask": opsd_attn, "position_ids": opsd_pos}


# ──────────────────────────────────────────────────────────────────────
# Step B+C: divergence via OPSD teacher forward
# ──────────────────────────────────────────────────────────────────────


def _compute_divergence_opsd(y_prev_batch, opsd_ctx, actor_rollout_wg, config, tokenizer) -> torch.Tensor:
    """logp_T (OPSD ctx) - logp_S 绝对差, 已 mask response_mask 外."""
    teacher_input_ids = torch.cat([opsd_ctx["input_ids"], y_prev_batch.batch["responses"]], dim=1)
    teacher_attn = torch.cat([opsd_ctx["attention_mask"], y_prev_batch.batch["response_mask"]], dim=1)
    teacher_pos = compute_position_id_with_mask(teacher_attn)

    teacher_fwd_batch = DataProto.from_dict(tensors={
        "teacher_input_ids": teacher_input_ids,
        "teacher_attention_mask": teacher_attn,
        "teacher_position_ids": teacher_pos,
        "responses": y_prev_batch.batch["responses"],
        "input_ids": y_prev_batch.batch["input_ids"],
        "attention_mask": y_prev_batch.batch["attention_mask"],
        "position_ids": y_prev_batch.batch["position_ids"],
    })
    teacher_fwd_batch.meta_info = {
        "temperature": float(config.actor_rollout_ref.rollout.temperature),
        "micro_batch_size": int(config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu),
        "pad_token_id": tokenizer.pad_token_id,
        "distill_topk": None,
        "compute_prior_shift_surprise": False,
    }
    teacher_result = actor_rollout_wg.compute_teacher_log_probs(teacher_fwd_batch)
    logp_T = teacher_result.batch["teacher_log_probs_on_response"]  # (B, T)

    if "rollout_log_probs" in y_prev_batch.batch:
        logp_S = y_prev_batch.batch["rollout_log_probs"]
    elif "old_log_probs" in y_prev_batch.batch:
        logp_S = y_prev_batch.batch["old_log_probs"]
    else:
        # 降级: 没 student logp, 用 0 (退化为纯 teacher logp magnitude as proxy)
        logp_S = torch.zeros_like(logp_T)

    divergence = (logp_T.float() - logp_S.float()).abs()
    divergence = divergence * y_prev_batch.batch["response_mask"].float()
    return divergence


# ──────────────────────────────────────────────────────────────────────
# Step D: 选 t_i
# ──────────────────────────────────────────────────────────────────────


def _select_t_i(divergence: torch.Tensor, response_mask: torch.Tensor, exclude_tail: int = 8) -> torch.Tensor:
    """argmax divergence, exclude last `exclude_tail` tokens (防选 EOS)."""
    B, T = divergence.shape
    L = response_mask.sum(dim=-1).long()
    idx = torch.arange(T, device=divergence.device).unsqueeze(0).expand(B, T)
    keep = idx < (L - exclude_tail).unsqueeze(-1).clamp(min=1)
    divergence = divergence * keep.float()
    return divergence.argmax(dim=-1)  # (B,)


# ──────────────────────────────────────────────────────────────────────
# Step E: teacher 写 1 token (OPSD ctx 下 argmax)
# ──────────────────────────────────────────────────────────────────────


def _teacher_write_one_token(y_prev_batch, opsd_ctx, t_i, actor_rollout_wg, config, tokenizer) -> torch.Tensor:
    """复用 actor_rollout_wg.teacher_generate_at_positions, k=1."""
    B = len(t_i)
    P_opsd = opsd_ctx["input_ids"].shape[1]
    t_star_abs = (P_opsd + t_i).long()

    teacher_input_ids = torch.cat([opsd_ctx["input_ids"], y_prev_batch.batch["responses"]], dim=1)
    teacher_attn = torch.cat([opsd_ctx["attention_mask"], y_prev_batch.batch["response_mask"]], dim=1)
    teacher_pos = compute_position_id_with_mask(teacher_attn)

    # FSDP chunk divisibility padding
    try:
        world_size = int(config.trainer.n_gpus_per_node)
    except Exception:
        world_size = 1
    world_size = max(1, world_size)
    pad_n = (world_size - B % world_size) % world_size

    if pad_n > 0:
        teacher_input_ids_p = torch.cat([teacher_input_ids, teacher_input_ids[:pad_n]], dim=0)
        teacher_attn_p = torch.cat([teacher_attn, teacher_attn[:pad_n]], dim=0)
        teacher_pos_p = torch.cat([teacher_pos, teacher_pos[:pad_n]], dim=0)
        t_star_abs_p = torch.cat([t_star_abs, t_star_abs[:pad_n]], dim=0)
    else:
        teacher_input_ids_p = teacher_input_ids
        teacher_attn_p = teacher_attn
        teacher_pos_p = teacher_pos
        t_star_abs_p = t_star_abs

    ic_cfg = config.algorithm.get("intervention_credit", {}) or {}
    teacher_temp = float(ic_cfg.get("teacher_decode_temperature", 0.0))

    sub_proto = DataProto.from_dict(tensors={
        "teacher_input_ids": teacher_input_ids_p,
        "teacher_attention_mask": teacher_attn_p,
        "teacher_position_ids": teacher_pos_p,
        "t_star_abs": t_star_abs_p,
    }, meta_info={"k": 1, "temperature": teacher_temp})

    out = actor_rollout_wg.teacher_generate_at_positions(sub_proto)
    teacher_tokens = out.batch["intervention_tokens"][:B, 0].to(t_i.device)  # (B,) drop padding & k=1
    return teacher_tokens


# ──────────────────────────────────────────────────────────────────────
# Step F+G: student 续写 (async via server_manager)
# ──────────────────────────────────────────────────────────────────────


async def _student_continue_one(prefix_ids: list, max_tokens: int, server_manager, sampling: dict) -> list:
    """单 sample 调 server_manager.generate, 返回 continuation token_ids list."""
    request_id = uuid4().hex
    out = await server_manager.generate(
        request_id=request_id,
        prompt_ids=prefix_ids,
        sampling_params={**sampling, "max_tokens": max(1, max_tokens)},
    )
    # out is TokenOutput(token_ids=list[int], ...) per replica.py:33-41
    if hasattr(out, "token_ids"):
        return list(out.token_ids)
    elif isinstance(out, dict) and "token_ids" in out:
        return list(out["token_ids"])
    else:
        return list(out)


def _student_continue_async(y_prev_batch, t_i, teacher_tokens, async_rollout_manager, config, tokenizer) -> list:
    """asyncio.gather over B samples, each with different prefix."""
    B = len(t_i)
    T = y_prev_batch.batch["responses"].shape[1]
    P_orig = y_prev_batch.batch["input_ids"].shape[1] - T
    rollout_cfg = config.actor_rollout_ref.rollout
    sampling = {
        "temperature": float(rollout_cfg.temperature),
        "top_p": float(getattr(rollout_cfg, "top_p", 1.0)),
    }
    server_manager = async_rollout_manager.server_manager

    # Build per-sample prefixes (strip left padding from orig prompt + append y_<t_i + teacher_token)
    prefixes = []
    budgets = []
    for j in range(B):
        full_prompt = y_prev_batch.batch["input_ids"][j, :P_orig]
        prompt_mask = y_prev_batch.batch["attention_mask"][j, :P_orig].bool()
        prompt_ids = full_prompt[prompt_mask].tolist()
        t_j = int(t_i[j].item())
        y_prefix = y_prev_batch.batch["responses"][j, :t_j].tolist()
        prefix = prompt_ids + y_prefix + [int(teacher_tokens[j].item())]
        prefixes.append(prefix)
        budgets.append(T - t_j - 1)

    # asyncio.gather
    coros = [
        _student_continue_one(p, b, server_manager, sampling)
        for p, b in zip(prefixes, budgets)
    ]
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in async context; create a new loop in a fresh thread
            import concurrent.futures
            def _run():
                inner_loop = asyncio.new_event_loop()
                try:
                    return inner_loop.run_until_complete(asyncio.gather(*coros))
                finally:
                    inner_loop.close()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                continuations = ex.submit(_run).result()
        else:
            continuations = loop.run_until_complete(asyncio.gather(*coros))
    except RuntimeError:
        continuations = asyncio.run(asyncio.gather(*coros))
    return continuations


# ──────────────────────────────────────────────────────────────────────
# Step H: build y_i_batch
# ──────────────────────────────────────────────────────────────────────


def _build_y_i_batch(y_prev_batch, t_i, teacher_tokens, continuations, tokenizer) -> DataProto:
    """构造 y_i = y_prev[:t_i] + teacher_token + continuation, mask shared prefix."""
    B = len(t_i)
    T = y_prev_batch.batch["responses"].shape[1]
    P_orig = y_prev_batch.batch["input_ids"].shape[1] - T
    device = y_prev_batch.batch["responses"].device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    new_responses = torch.full((B, T), pad_id, dtype=torch.long, device=device)
    new_response_mask = torch.zeros((B, T), dtype=torch.long, device=device)

    for j in range(B):
        t_j = int(t_i[j].item())
        # copy y_prev[:t_j]
        if t_j > 0:
            new_responses[j, :t_j] = y_prev_batch.batch["responses"][j, :t_j]
        # teacher token at t_j
        if t_j < T:
            new_responses[j, t_j] = int(teacher_tokens[j].item())
        # student continuation from t_j+1
        cont_ids = continuations[j] if j < len(continuations) else []
        cont_start = t_j + 1
        if cont_start < T and len(cont_ids) > 0:
            cont_end = min(T, cont_start + len(cont_ids))
            cont_take = cont_end - cont_start
            new_responses[j, cont_start:cont_end] = torch.tensor(
                cont_ids[:cont_take], dtype=torch.long, device=device
            )
            new_response_mask[j, t_j:cont_end] = 1
        elif t_j < T:
            # 只有 teacher token, 没 continuation (空续写)
            new_response_mask[j, t_j:t_j+1] = 1
        # else: t_j >= T, mask 0 (degenerate)

    # rebuild input_ids, attention_mask, position_ids
    prompts = y_prev_batch.batch["prompts"]  # (B, P_orig)
    prompt_attn = y_prev_batch.batch["attention_mask"][:, :P_orig]
    new_input_ids = torch.cat([prompts, new_responses], dim=1)
    new_attention = torch.cat([prompt_attn, new_response_mask], dim=1)
    new_position = compute_position_id_with_mask(new_attention)

    new_non_tensor = {}
    for k, v in y_prev_batch.non_tensor_batch.items():
        if isinstance(v, np.ndarray):
            new_non_tensor[k] = v.copy()
        else:
            new_non_tensor[k] = v

    return DataProto.from_dict(
        tensors={
            "prompts": prompts.clone(),
            "responses": new_responses,
            "response_mask": new_response_mask,
            "input_ids": new_input_ids,
            "attention_mask": new_attention,
            "position_ids": new_position,
        },
        non_tensor_batch=new_non_tensor,
        meta_info=dict(y_prev_batch.meta_info or {}),
    )


# ──────────────────────────────────────────────────────────────────────
# Step I: reward rescore
# ──────────────────────────────────────────────────────────────────────


def _rescore_reward(y_i_batch: DataProto, reward_fn) -> DataProto:
    """删旧 reward 字段, 调 reward_fn 重打分."""
    for k in ["rm_scores", "token_level_scores", "token_level_rewards", "advantages", "returns"]:
        if k in y_i_batch.batch.keys():
            y_i_batch.batch.pop(k)
    try:
        r = reward_fn(y_i_batch, return_dict=True)
        reward_tensor = r["reward_tensor"]
    except TypeError:
        reward_tensor = reward_fn(y_i_batch)
    device = y_i_batch.batch["responses"].device
    reward_tensor = reward_tensor.to(device)
    y_i_batch.batch["token_level_scores"] = reward_tensor
    y_i_batch.batch["token_level_rewards"] = reward_tensor
    return y_i_batch


# ──────────────────────────────────────────────────────────────────────
# Build divergence_credit (Layer 3 input to intervention_credit estimator)
# ──────────────────────────────────────────────────────────────────────


def _build_divergence_credit(chain_batches, t_history, delta_r_history, B: int, chain_length: int) -> torch.Tensor:
    """
    Build divergence_credit (B * chain_length, T) for the augmented batch.

    Convention (Layer 3 选项 C):
      for i in 1..chain_length-1:
        ΔR_i = delta_r_history[i-1]  # R(y_i) - R(y_{i-1})
        t_i  = t_history[i]
        c_t[y_i,    t_i] += +ΔR_i      # teacher's choice at t_i (positive credit if helped)
        c_t[y_{i-1}, t_i] -= +ΔR_i      # student's choice at same position (mirror)

    端点:
      y_0 只 contributes at t_1 with -ΔR_1
      y_{n-1} only contributes at t_{n-1} with +ΔR_{n-1}
    """
    T = chain_batches[0].batch["responses"].shape[1]
    device = chain_batches[0].batch["responses"].device

    chain_credits = [torch.zeros(B, T, dtype=torch.float32, device=device) for _ in range(chain_length)]

    for i in range(1, chain_length):
        delta_r_i = delta_r_history[i - 1].float().to(device)  # (B,)
        t_i = t_history[i].long().to(device)  # (B,)
        for j in range(B):
            t_j = int(t_i[j].item())
            if 0 <= t_j < T:
                chain_credits[i][j, t_j] += delta_r_i[j].item()
                chain_credits[i - 1][j, t_j] -= delta_r_i[j].item()

    # Concat along batch dim, matching DataProto.concat(chain_batches) order
    return torch.cat(chain_credits, dim=0)  # (B * chain_length, T)
