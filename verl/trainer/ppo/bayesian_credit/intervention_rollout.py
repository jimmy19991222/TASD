# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Teacher-Guided Dynamic Intervention rollout helper (Tier 3, ours).

设计文档: /Users/awesome_jimmy/lazada/SDPO/Teacher-Guided_Dynamic_Intervention_Rollout.md

工作流 (Mode B append):

    Round 1: y_s ~ P_S(·|x)                                      [n_initial 个]
    Step 2:  teacher forward → logp_T, g_t                       [复用 prior_shift 通道]
    Step 3:  failed = R(y_s) < threshold                         [失败样本检测]
    Step 4:  t* = divergence_position(logp_T, logp_S, g_t)       [3 种策略消融]
    Step 5:  y_int ~ P_T(·|x, y_<t*) for k tokens                [teacher 接管]
    Step 6:  y_tail ~ P_S(·|x, y_<t*, y_int)                     [student 续写]
    Step 7:  y' = (y_<t*, y_int, y_tail), append to batch
    Step 8:  ΔR = R(y') - R(y_s)
    Step 9:  写入 batch.batch[{intervention_delta_reward, intervention_used}]

实现状态：
    - Phase 1 (本提交): 完整 plumbing + divergence detection + metrics
                       enable_intervention=False 时 batch 不变，ΔR=0；
                       enable_intervention=True 时 raise NotImplementedError(
                           "real teacher generation needs follow-up commit"
                       )
    - Phase 2 (follow-up): 实现 step 5-7 真实 intervention 调用

之所以分阶段：teacher generation 需要在 dp_actor 中加 `teacher_generate_at_positions`
新接口（~30 行），然后 student tail 需要 per-sample async_rollout_manager.generate
的复杂调用。Phase 1 先把外围 (yaml / estimator / nebula scripts) 全部跑通，
Phase 2 只动 worker 端 + 本文件的 _do_real_intervention 函数。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────
# 公开 API
# ──────────────────────────────────────────────────────────────────────


@dataclass
class InterventionResult:
    """run_teacher_intervention_rollout 的返回结构"""
    batch: object  # DataProto (potentially augmented with composite samples)
    metrics: dict = field(default_factory=dict)


def run_teacher_intervention_rollout(
    batch,                  # DataProto: 当前 step 的 batch (含 reward, teacher logp, g_t)
    actor_rollout_wg,       # actor worker group (用于 teacher_log_probs / generate)
    async_rollout_manager,  # AsyncLLMServerManager (用于 student tail rollout)
    reward_fn,              # 用于对 composite rollout 重新打分
    config,                 # AlgoConfig dict-like
    tokenizer,              # 用于 prompt / response 编解码
) -> InterventionResult:
    """主入口：检测失败样本 → 找 t* → teacher 接管 k 个 token → student 续 tail → 加权 ΔR。

    Phase 1 行为:
      - enable_intervention=False  : 写零 ΔR/used，返回 batch 原样
      - enable_intervention=True   : 触发 NotImplementedError (Phase 2 接通)
    """
    ic_cfg = config.algorithm.get("intervention_credit", {}) or {}
    enable_intervention: bool = bool(ic_cfg.get("enable_intervention", False))

    B = batch.batch.batch_size[0]
    device = batch.batch["responses"].device

    # ── 默认写 ΔR=0 / used=False（intervention_credit estimator 的兜底）──
    batch.batch["intervention_delta_reward"] = torch.zeros(B, device=device, dtype=torch.float32)
    batch.batch["intervention_used"] = torch.zeros(B, device=device, dtype=torch.float32)

    metrics = {
        "intervention/applied_rate": 0.0,
        "intervention/failed_sample_rate": 0.0,
        "intervention/delta_reward_mean": 0.0,
        "intervention/divergence_position_mean": 0.0,
    }

    if not enable_intervention:
        # Phase 1 默认路径：仅做失败样本检测和 divergence 统计，不修改 batch
        diag = _diagnose_only(batch, ic_cfg)
        metrics.update(diag)
        return InterventionResult(batch=batch, metrics=metrics)

    # Phase 2：真实 intervention（待实现）
    return _do_real_intervention(
        batch=batch,
        actor_rollout_wg=actor_rollout_wg,
        async_rollout_manager=async_rollout_manager,
        reward_fn=reward_fn,
        config=config,
        tokenizer=tokenizer,
        ic_cfg=ic_cfg,
    )


# ──────────────────────────────────────────────────────────────────────
# Phase 1: 诊断 / 度量（不改 batch）
# ──────────────────────────────────────────────────────────────────────


def _diagnose_only(batch, ic_cfg: dict) -> dict:
    """仅做失败样本统计 + divergence position 分布度量，不做真实 intervention。

    意义：让 SwanLab 上能直接看到 v3 关心的指标（失败率、divergence 分布），
    便于在 Phase 2 接通真实 intervention 前先 baseline 这些数字。

    指标分组:
        intervention/failed_sample_rate     — 触发 intervention 的样本比例
        intervention/seq_reward_{mean,std}  — 全 batch reward 分布
        intervention/seq_reward_failed_mean — 失败样本 reward 均值（应 < threshold）
        intervention/divergence_position_{mean,std,p25,p50,p75} — t* 绝对位置
        intervention/divergence_position_normalized_mean — t*/L (0..1)
        intervention/divergence_in_first_quarter_rate     — t* < L/4 比例（早期错误）
        intervention/divergence_in_last_quarter_rate      — t* > 3L/4 比例（尾部错误，排查 EOS-bias）
    """
    failed_threshold: float = float(ic_cfg.get("failed_threshold", 0.5))

    seq_reward = batch.batch["token_level_rewards"].sum(dim=-1)  # (B,)
    failed_mask = seq_reward < failed_threshold                  # (B,) bool

    metrics = {
        "intervention/failed_sample_rate": failed_mask.float().mean().item(),
        "intervention/applied_rate": 0.0,
        "intervention/delta_reward_mean": 0.0,
        "intervention/seq_reward_mean": seq_reward.float().mean().item(),
        "intervention/seq_reward_std": seq_reward.float().std().item() if seq_reward.numel() > 1 else 0.0,
    }
    if failed_mask.any():
        metrics["intervention/seq_reward_failed_mean"] = seq_reward[failed_mask].float().mean().item()
    else:
        metrics["intervention/seq_reward_failed_mean"] = 0.0

    # divergence position 度量（如果 teacher logp / g_t 可用）
    t_star = compute_divergence_position(batch, ic_cfg)
    response_mask = batch.batch.get("response_mask")
    if t_star is not None and response_mask is not None:
        L = response_mask.float().sum(dim=-1).clamp(min=1.0)        # (B,)
        if failed_mask.any():
            idx = failed_mask
            t_failed = t_star[idx].float()
            L_failed = L[idx]
            t_norm = (t_failed / L_failed).clamp(0.0, 1.0)          # (B_f,) ∈ [0,1]

            metrics["intervention/divergence_position_mean"] = t_failed.mean().item()
            metrics["intervention/divergence_position_std"] = (
                t_failed.std().item() if t_failed.numel() > 1 else 0.0
            )
            # 分位数（torch.quantile 需要 float）
            if t_failed.numel() >= 4:
                qs = torch.quantile(t_failed, torch.tensor([0.25, 0.5, 0.75], device=t_failed.device))
                metrics["intervention/divergence_position_p25"] = qs[0].item()
                metrics["intervention/divergence_position_p50"] = qs[1].item()
                metrics["intervention/divergence_position_p75"] = qs[2].item()
            metrics["intervention/divergence_position_normalized_mean"] = t_norm.mean().item()
            metrics["intervention/divergence_in_first_quarter_rate"] = (t_norm < 0.25).float().mean().item()
            metrics["intervention/divergence_in_last_quarter_rate"] = (t_norm > 0.75).float().mean().item()
        else:
            metrics["intervention/divergence_position_mean"] = 0.0
            metrics["intervention/divergence_position_std"] = 0.0
            metrics["intervention/divergence_position_normalized_mean"] = 0.0
            metrics["intervention/divergence_in_first_quarter_rate"] = 0.0
            metrics["intervention/divergence_in_last_quarter_rate"] = 0.0

    return metrics


# ──────────────────────────────────────────────────────────────────────
# divergence position 选择（3 种策略，消融维度）
# ──────────────────────────────────────────────────────────────────────


def compute_divergence_position(batch, ic_cfg: dict) -> Optional[torch.Tensor]:
    """根据配置选择 t* (divergence position)。

    Returns:
        (B,) long tensor，每行是该样本的 t*（在 response_mask 内）；
        若所需输入不在 batch 中则返回 None。
    """
    metric: str = str(ic_cfg.get("divergence_metric", "argmax_excl_eos"))
    exclude_tail: int = int(ic_cfg.get("exclude_tail_tokens", 8))

    response_mask = batch.batch.get("response_mask")
    if response_mask is None:
        return None

    if metric == "g_t_argmax":
        g = batch.batch.get("bc_teacher_prior_shift_surprise")
        if g is None:
            return None
        score = g  # (B, T)
    else:
        # logp diff 路径
        logp_T = batch.batch.get("bc_teacher_log_probs")
        logp_S = batch.batch.get("old_log_probs")
        if logp_T is None or logp_S is None:
            return None
        score = (logp_T.float() - logp_S.float()).abs()  # (B, T)

    score = score * response_mask.to(score.dtype)

    if metric == "argmax_excl_eos":
        # 排除尾部 N 个 token（防选 EOS / punctuation 散度）
        B, T = score.shape
        L = response_mask.sum(dim=-1).long()  # (B,)
        # 构造一个掩码: 位置 t < L - exclude_tail 时为 1，否则为 0
        idx = torch.arange(T, device=score.device).unsqueeze(0).expand(B, T)  # (B, T)
        keep = idx < (L - exclude_tail).unsqueeze(-1).clamp(min=1)            # (B, T)
        score = score * keep.to(score.dtype)

    t_star = score.argmax(dim=-1)  # (B,)
    return t_star


# ──────────────────────────────────────────────────────────────────────
# Phase 2: 真实 intervention（占位）
# ──────────────────────────────────────────────────────────────────────


def _do_real_intervention(
    batch,
    actor_rollout_wg,
    async_rollout_manager,
    reward_fn,
    config,
    tokenizer,
    ic_cfg: dict,
) -> InterventionResult:
    """Phase 2 V1: 真实 teacher 介入 + Mode B append.

    流程:
      1. failed = R < threshold；t* = compute_divergence_position(batch)
      2. 调用 actor_rollout_wg.teacher_generate_at_positions 在 [t*..t*+k-1] 生成 k 个 token
      3. 构造 composite responses = y_<t* + intervention + y_{t*+k:T}（保留学生原 tail）
      4. reward_fn(composite_batch) → R(y')；ΔR = R(y') - R(y_s)
      5. composite 复制原样本的 old_log_probs / ref_log_prob / bc_teacher_*（V1 简化；
         k 个 teacher 位置上 PPO ratio 略不准，但占比 k/T ≈ 0.5%，clip 兜底）
      6. DataProto.concat([batch, composite_batch]) 同 uid 加权
      7. 写入 intervention_delta_reward / intervention_used 到 augmented batch

    V1 限制 (留 Phase 2.1 follow-up):
      - 不真做 student tail 续写（保留学生原 tail 接 teacher intervention）
      - 不重算 old_log_prob 在 composite 上（复制原样本，仅 k 位置不准）
      - 不重算 bc_teacher_fwd 在 composite 上（uniform_fallback 兜底）
    """
    from verl.protocol import DataProto
    import numpy as np
    import torch

    failed_threshold: float = float(ic_cfg.get("failed_threshold", 0.5))
    k: int = int(ic_cfg.get("intervention_length_k", 2))
    teacher_temp: float = float(ic_cfg.get("teacher_decode_temperature", 0.0))
    max_per_group: int = int(ic_cfg.get("max_intervention_per_group", 7))

    B = batch.batch.batch_size[0]
    device = batch.batch["responses"].device

    # ── Step 1: 失败样本检测 ──────────────────────────────────────────
    seq_reward = batch.batch["token_level_rewards"].sum(dim=-1)  # (B,)
    failed_mask = seq_reward < failed_threshold                  # (B,) bool
    n_failed_total = int(failed_mask.sum().item())

    metrics = {
        "intervention/failed_sample_rate": failed_mask.float().mean().item(),
        "intervention/seq_reward_mean": seq_reward.float().mean().item(),
        "intervention/seq_reward_std": (
            seq_reward.float().std().item() if seq_reward.numel() > 1 else 0.0
        ),
    }

    if n_failed_total == 0:
        # 没有失败样本 → 不 intervention，写零通道
        batch.batch["intervention_delta_reward"] = torch.zeros(B, device=device, dtype=torch.float32)
        batch.batch["intervention_used"] = torch.zeros(B, device=device, dtype=torch.float32)
        metrics.update({
            "intervention/applied_rate": 0.0,
            "intervention/delta_reward_mean": 0.0,
        })
        return InterventionResult(batch=batch, metrics=metrics)

    # 按 uid 限制每组最多 max_per_group 个 intervention（避免组膨胀）
    uid = batch.non_tensor_batch["uid"]
    failed_indices_all = failed_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    selected_indices = []
    per_group_count: dict = {}
    for idx in failed_indices_all:
        u = uid[idx]
        if per_group_count.get(u, 0) < max_per_group:
            selected_indices.append(int(idx))
            per_group_count[u] = per_group_count.get(u, 0) + 1
    failed_idx = torch.tensor(selected_indices, device=device, dtype=torch.long)
    n_failed = len(failed_idx)

    if n_failed == 0:
        batch.batch["intervention_delta_reward"] = torch.zeros(B, device=device, dtype=torch.float32)
        batch.batch["intervention_used"] = torch.zeros(B, device=device, dtype=torch.float32)
        return InterventionResult(batch=batch, metrics=metrics)

    # ── Step 2: 计算 t_star (失败样本) ─────────────────────────────────
    t_star_all = compute_divergence_position(batch, ic_cfg)  # (B,) long
    if t_star_all is None:
        # 没有 teacher logp / g_t 通道 → 退化
        batch.batch["intervention_delta_reward"] = torch.zeros(B, device=device, dtype=torch.float32)
        batch.batch["intervention_used"] = torch.zeros(B, device=device, dtype=torch.float32)
        metrics["intervention/applied_rate"] = 0.0
        return InterventionResult(batch=batch, metrics=metrics)

    t_star_failed = t_star_all[failed_idx]  # (n_failed,) — relative to response

    # 推导 prompt_length: input_ids = (B, P+T), responses = (B, T) → P = (P+T) - T
    T = batch.batch["responses"].shape[1]
    L = batch.batch["input_ids"].shape[1]
    P = L - T  # 假定 batch 内固定 prompt_length（左 pad 到 max_prompt_len）

    t_star_abs_failed = (P + t_star_failed).long()  # 在 input_ids / teacher_input_ids 坐标系

    # ── Step 3: 调用 worker 让 teacher 生成 k 个 token ──────────────
    # 仅传 failed 子集
    teacher_input_ids_f = batch.batch["teacher_input_ids"][failed_idx]
    teacher_attn_f = batch.batch["teacher_attention_mask"][failed_idx]
    teacher_pos_f = batch.batch["teacher_position_ids"][failed_idx]

    sub_proto = DataProto.from_dict(
        tensors={
            "teacher_input_ids": teacher_input_ids_f,
            "teacher_attention_mask": teacher_attn_f,
            "teacher_position_ids": teacher_pos_f,
            "t_star_abs": t_star_abs_failed,
        },
        meta_info={
            "k": k,
            "temperature": teacher_temp,
        },
    )
    teacher_gen_out = actor_rollout_wg.teacher_generate_at_positions(sub_proto)
    intervention_tokens = teacher_gen_out.batch["intervention_tokens"].to(device)  # (n_failed, k)

    # ── Step 4: 构造 composite responses (替换 [t*, t*+k]) ─────────────
    # 复制 failed 子集的 batch 字段，原地替换 token
    composite_responses = batch.batch["responses"][failed_idx].clone()       # (n_failed, T)
    composite_input_ids = batch.batch["input_ids"][failed_idx].clone()       # (n_failed, L)
    composite_attention = batch.batch["attention_mask"][failed_idx].clone()  # (n_failed, L)
    composite_position = batch.batch["position_ids"][failed_idx].clone()     # (n_failed, L)
    composite_response_mask = batch.batch["response_mask"][failed_idx].clone()  # (n_failed, T)

    bidx = torch.arange(n_failed, device=device)
    for step in range(k):
        pos_rel = (t_star_failed + step).long()  # (n_failed,)
        pos_abs = (t_star_abs_failed + step).long()
        valid_rel = (pos_rel < T)
        valid_abs = (pos_abs < L)
        if valid_rel.any():
            v_idx = bidx[valid_rel]
            composite_responses[v_idx, pos_rel[valid_rel]] = intervention_tokens[v_idx, step].long()
            composite_response_mask[v_idx, pos_rel[valid_rel]] = 1
        if valid_abs.any():
            v_idx = bidx[valid_abs]
            composite_input_ids[v_idx, pos_abs[valid_abs]] = intervention_tokens[v_idx, step].long()
            composite_attention[v_idx, pos_abs[valid_abs]] = 1

    # ── Step 5: 构造 composite DataProto，re-score reward ─────────────
    # 复制全部 batch / non_tensor_batch / meta_info 字段，仅替换被改动的 tensor
    composite_tensors = {}
    for key, val in batch.batch.items():
        if not torch.is_tensor(val):
            continue
        if val.shape[0] != B:
            # 不是 batch 维度的 tensor，跳过（不应发生但保险）
            continue
        composite_tensors[key] = val[failed_idx].clone()
    # 替换被改动的几项
    composite_tensors["responses"] = composite_responses
    composite_tensors["input_ids"] = composite_input_ids
    composite_tensors["attention_mask"] = composite_attention
    composite_tensors["position_ids"] = composite_position
    composite_tensors["response_mask"] = composite_response_mask

    # non_tensor: copy 所有 numpy array 字段对应的子集
    composite_non_tensor = {}
    for key, arr in batch.non_tensor_batch.items():
        if isinstance(arr, np.ndarray) and len(arr) == B:
            composite_non_tensor[key] = arr[failed_idx.cpu().numpy()].copy()

    composite_batch = DataProto(
        batch=DataProto.from_dict(tensors=composite_tensors).batch,
        non_tensor_batch=composite_non_tensor,
        meta_info=dict(batch.meta_info or {}),
    )

    # 删除会让 reward_fn 短路的字段（rm_scores 若存在会跳过重算）
    if "rm_scores" in composite_batch.batch.keys():
        composite_batch.batch.pop("rm_scores")
    # 同样删除 token_level_scores / token_level_rewards 让 reward_fn 重新计算
    for k_pop in ["token_level_scores", "token_level_rewards", "advantages", "returns"]:
        if k_pop in composite_batch.batch.keys():
            composite_batch.batch.pop(k_pop)

    # 调用 reward_fn 重新打分
    try:
        rew_result = reward_fn(composite_batch, return_dict=True)
        new_reward_tensor = rew_result["reward_tensor"].to(device)  # (n_failed, T)
    except TypeError:
        # 兼容只返回 tensor 的 manager
        new_reward_tensor = reward_fn(composite_batch).to(device)

    # ── Step 6: ΔR 计算 ──────────────────────────────────────────────
    composite_seq_reward = new_reward_tensor.sum(dim=-1)             # (n_failed,)
    original_seq_reward = seq_reward[failed_idx]                     # (n_failed,)
    delta_r = composite_seq_reward - original_seq_reward             # (n_failed,)

    # 在 composite_batch 上写入新 reward 字段（让后续 advantage 计算能用）
    composite_batch.batch["token_level_scores"] = new_reward_tensor
    composite_batch.batch["token_level_rewards"] = new_reward_tensor

    # composite 上的 intervention_delta_reward / intervention_used
    composite_batch.batch["intervention_delta_reward"] = delta_r.float()
    composite_batch.batch["intervention_used"] = torch.ones(n_failed, device=device, dtype=torch.float32)

    # ── Step 7: 原 batch 上写入零通道 ──────────────────────────────────
    batch.batch["intervention_delta_reward"] = torch.zeros(B, device=device, dtype=torch.float32)
    batch.batch["intervention_used"] = torch.zeros(B, device=device, dtype=torch.float32)

    # ── Step 8: concat ────────────────────────────────────────────────
    augmented_batch = DataProto.concat([batch, composite_batch])

    # ── Phase 2 metrics（7 个新指标）─────────────────────────────────
    composite_resp_mask_sum = composite_response_mask.float().sum(dim=-1)  # (n_failed,)
    metrics.update({
        "intervention/applied_rate": n_failed / max(B, 1),
        "intervention/delta_reward_mean": float(delta_r.mean().item()),
        "intervention/delta_reward_std": float(delta_r.std().item()) if delta_r.numel() > 1 else 0.0,
        "intervention/delta_reward_min": float(delta_r.min().item()),
        "intervention/delta_reward_max": float(delta_r.max().item()),
        "intervention/delta_reward_pos_rate": float((delta_r > 0).float().mean().item()),
        "intervention/composite_response_length_mean": float(composite_resp_mask_sum.mean().item()),
        "intervention/n_appended_per_group_mean": (
            float(np.mean(list(per_group_count.values()))) if per_group_count else 0.0
        ),
        "intervention/group_size_post_append_mean": float(B + n_failed) / float(len(set(uid))),
        "intervention/group_size_post_append_max": (
            float(max(per_group_count.values()) + max_per_group) if per_group_count else float(max_per_group)
        ),
        "intervention/n_failed_total": float(n_failed_total),
        "intervention/n_intervention_applied": float(n_failed),
    })

    # 还要补上 divergence position 度量（与 _diagnose_only 同款，保证两条路径指标一致）
    diag_metrics = _diagnose_only(batch, ic_cfg)
    # diag_metrics 里有些已被覆盖，仅补 divergence_position_*
    for key in [
        "intervention/divergence_position_mean",
        "intervention/divergence_position_std",
        "intervention/divergence_position_p25",
        "intervention/divergence_position_p50",
        "intervention/divergence_position_p75",
        "intervention/divergence_position_normalized_mean",
        "intervention/divergence_in_first_quarter_rate",
        "intervention/divergence_in_last_quarter_rate",
        "intervention/seq_reward_failed_mean",
    ]:
        if key in diag_metrics:
            metrics[key] = diag_metrics[key]

    return InterventionResult(batch=augmented_batch, metrics=metrics)


# ──────────────────────────────────────────────────────────────────────
# 工具：失败样本检测（公开给 ray_trainer 用于度量）
# ──────────────────────────────────────────────────────────────────────


def detect_failed_samples(token_level_rewards: torch.Tensor, failed_threshold: float) -> torch.Tensor:
    """(B, T) → (B,) bool, True for samples with seq-sum reward below threshold."""
    seq_reward = token_level_rewards.sum(dim=-1)
    return seq_reward < failed_threshold
