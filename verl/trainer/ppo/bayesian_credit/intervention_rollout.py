# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Teacher-Guided Dynamic Intervention rollout helper (Tier 3, ours).

设计文档: research/paper_idea.md (TCCA 当前方法) + research/design_history.md (TGDI→TCCA 演化)

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
        # 但需要写 token_causal_credit=0 兜底（让 advantage estimator 一致取得到字段）
        T = batch.batch["responses"].shape[1]
        batch.batch["token_causal_credit"] = torch.zeros(B, T, device=device, dtype=torch.float32)
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


def compute_topk_divergence_positions(
    batch, ic_cfg: dict, top_k: int = 3
) -> Optional[torch.Tensor]:
    """**TCCA 新增**：返回每个 sample 的 top-K divergence 位置。

    Returns:
        (B, K) long tensor: positions[i, k] = i 样本第 k 高 divergence 位置 (response 坐标);
        若所需输入不在 batch 中则返回 None。
        K 个位置之间至少相隔 max(intervention_length_k, 4) 个 token (避免重叠).
    """
    metric: str = str(ic_cfg.get("divergence_metric", "argmax_excl_eos"))
    exclude_tail: int = int(ic_cfg.get("exclude_tail_tokens", 8))
    intervention_length: int = int(ic_cfg.get("intervention_length_k", 2))
    min_gap: int = max(intervention_length, 4)

    response_mask = batch.batch.get("response_mask")
    if response_mask is None:
        return None

    if metric == "g_t_argmax":
        g = batch.batch.get("bc_teacher_prior_shift_surprise")
        if g is None:
            return None
        score = g.float()  # (B, T)
    else:
        logp_T = batch.batch.get("bc_teacher_log_probs")
        logp_S = batch.batch.get("old_log_probs")
        if logp_T is None or logp_S is None:
            return None
        score = (logp_T.float() - logp_S.float()).abs()

    score = score * response_mask.float()

    if metric == "argmax_excl_eos":
        B, T = score.shape
        L = response_mask.sum(dim=-1).long()
        idx = torch.arange(T, device=score.device).unsqueeze(0).expand(B, T)
        keep = idx < (L - exclude_tail).unsqueeze(-1).clamp(min=1)
        score = score * keep.float()

    # ── greedy top-K with min_gap suppression ─────────────────────────
    # 不能直接 topk(K)，因为相邻位置 score 可能都高且会 overlap。
    # 用 greedy: 选最大 → 屏蔽 [t-min_gap, t+min_gap] → 选次大 → ...
    B, T = score.shape
    device = score.device
    work_score = score.clone()
    topk_positions = torch.zeros(B, top_k, dtype=torch.long, device=device)

    for k in range(top_k):
        t_k = work_score.argmax(dim=-1)  # (B,)
        topk_positions[:, k] = t_k
        # 屏蔽该位置周围 min_gap 区间
        idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        mask_around = (idx >= (t_k - min_gap).unsqueeze(-1)) & (idx <= (t_k + min_gap).unsqueeze(-1))
        work_score = work_score.masked_fill(mask_around, float("-inf"))

    # 若 sample 的所有 score 都 0（degenerate），所有 t_k 会指向 0；
    # 调用方 (TCCA) 会按 failed_mask 过滤，degenerate 样本通常不在 failed_mask 内
    return topk_positions


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
    """**TCCA-Lite** — single-step counterfactual + real student continuation.

    详见 research/tcca_v2_design.md 的 TCCA-Lite 路径 (2026-05-16 pivot from chain).

    流程:
      1. failed = R(y_s) < threshold
      2. 构造 OPSD teacher context (含 reference answer) on failed subset
      3. teacher_fwd_opsd → divergence per token on failed subset
      4. t* = argmax divergence (single position per failed)
      5. teacher 写 1 token at t* (under OPSD ctx)
      6. student 续写到 EOS (async per-sample via server_manager)
      7. composite y' = y_<t* + teacher_token + student_continuation
      8. reward_fn(composite) → R(y'); ΔR = R(y') - R(y_s)
      9. Mode B append composite to batch
      10. divergence_credit:
          - 原 y at t*: -ΔR (negative credit at student's wrong choice)
          - composite y' at t*: +ΔR (positive credit at teacher's choice)
          - composite y'.response_mask[:, :t*] = 0 (Layer 2: shared prefix off)

    所有 helper 复用自 tcca_chain.py (OPSD ctx, async continuation, composite build, rescore).
    """
    # Lazy imports (avoid circular: intervention_credit's __init__ pulls tcca_chain)
    from verl.protocol import DataProto
    from verl.trainer.ppo.bayesian_credit.tcca_chain import (
        _build_opsd_teacher_context,
        _compute_divergence_opsd,
        _select_t_i,
        _teacher_write_one_token,
        _student_continue_async,
        _build_y_i_batch,
        _rescore_reward,
    )

    failed_threshold: float = float(ic_cfg.get("failed_threshold", 0.5))
    max_per_prompt: int = int(ic_cfg.get("max_intervention_per_prompt",
                                          ic_cfg.get("max_intervention_per_group", 4)))
    exclude_tail: int = int(ic_cfg.get("exclude_tail_tokens", 8))

    B = batch.batch.batch_size[0]
    device = batch.batch["responses"].device
    T = batch.batch["responses"].shape[1]

    # world_size for FSDP chunk divisibility
    try:
        world_size = int(config.trainer.n_gpus_per_node)
    except Exception:
        world_size = 1
    world_size = max(1, world_size)

    # ── Step 1: 失败样本检测 ──────────────────────────────────────────
    seq_reward = batch.batch["token_level_rewards"].sum(dim=-1)
    failed_mask = seq_reward < failed_threshold
    n_failed_total = int(failed_mask.sum().item())

    metrics = {
        "intervention/failed_sample_rate": float(failed_mask.float().mean().item()),
        "intervention/seq_reward_mean": float(seq_reward.float().mean().item()),
        "intervention/seq_reward_std": float(seq_reward.float().std().item()) if seq_reward.numel() > 1 else 0.0,
    }

    def _write_zeros_and_return(b, msg_metrics):
        b.batch["divergence_credit"] = torch.zeros(B, T, device=device, dtype=torch.float32)
        b.batch["intervention_delta_reward"] = torch.zeros(B, device=device, dtype=torch.float32)
        b.batch["intervention_used"] = torch.zeros(B, device=device, dtype=torch.float32)
        return InterventionResult(batch=b, metrics={**msg_metrics, "intervention/applied_rate": 0.0})

    if n_failed_total == 0:
        return _write_zeros_and_return(batch, metrics)

    # ── Step 2: cap by max_intervention_per_prompt + FSDP divisibility 截断 ──
    uid = batch.non_tensor_batch["uid"]
    failed_indices_all = failed_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    selected_indices = []
    per_uid_count: dict = {}
    for idx in failed_indices_all:
        u = uid[idx]
        if per_uid_count.get(u, 0) < max_per_prompt:
            selected_indices.append(int(idx))
            per_uid_count[u] = per_uid_count.get(u, 0) + 1

    # 让 augmented batch (B + n_failed) % world_size == 0
    n_failed = len(selected_indices)
    target_total = B + n_failed
    rem = target_total % world_size
    while rem != 0 and n_failed > 0:
        n_failed -= 1
        target_total = B + n_failed
        rem = target_total % world_size
    selected_indices = selected_indices[:n_failed]
    if n_failed == 0:
        return _write_zeros_and_return(batch, metrics)

    failed_idx = torch.tensor(selected_indices, device=device, dtype=torch.long)

    # ── Step 3: build failed_subset DataProto (subset of original batch) ──
    sub_tensors = {}
    for k, v in batch.batch.items():
        if not torch.is_tensor(v) or v.shape[0] != B:
            continue
        sub_tensors[k] = v[failed_idx].clone()
    sub_non_tensor = {}
    for k, v in batch.non_tensor_batch.items():
        if isinstance(v, np.ndarray) and len(v) == B:
            sub_non_tensor[k] = v[failed_idx.cpu().numpy()].copy()
    failed_subset = DataProto.from_dict(
        tensors=sub_tensors,
        non_tensor_batch=sub_non_tensor,
        meta_info=dict(batch.meta_info or {}),
    )

    # ── Step 4: OPSD ctx ──────────────────────────────────────────────
    try:
        opsd_ctx = _build_opsd_teacher_context(failed_subset, tokenizer, config)
    except Exception as e:
        # 退化: 缺 raw_prompt 或 ref answer
        return _write_zeros_and_return(batch, {**metrics, "intervention/opsd_build_failed": 1.0})

    # ── Step 5: divergence + t* ─────────────────────────────────────
    divergence = _compute_divergence_opsd(failed_subset, opsd_ctx, actor_rollout_wg, config, tokenizer)
    t_star = _select_t_i(divergence, failed_subset.batch["response_mask"], exclude_tail=exclude_tail)

    # ── Step 6: teacher 写 1 token at t* (under OPSD ctx) ────────────
    teacher_tokens = _teacher_write_one_token(failed_subset, opsd_ctx, t_star, actor_rollout_wg, config, tokenizer)

    # ── Step 7: student 真续写 ───────────────────────────────────────
    continuations = _student_continue_async(failed_subset, t_star, teacher_tokens, async_rollout_manager, config, tokenizer)

    # ── Step 8: 构造 composite y' ────────────────────────────────────
    composite_batch = _build_y_i_batch(failed_subset, t_star, teacher_tokens, continuations, tokenizer)

    # ── Step 9: re-score reward on composite ─────────────────────────
    composite_batch = _rescore_reward(composite_batch, reward_fn)

    # ── Step 10: ΔR ──────────────────────────────────────────────────
    composite_R = composite_batch.batch["token_level_rewards"].sum(dim=-1)
    original_R = seq_reward[failed_idx]
    delta_r = (composite_R - original_R).float()  # (n_failed,)

    composite_batch.batch["intervention_delta_reward"] = delta_r.clone()
    composite_batch.batch["intervention_used"] = torch.ones(n_failed, device=device, dtype=torch.float32)

    # ── Step 11: 构造 divergence_credit ──────────────────────────────
    # 原 batch: c_t[failed_idx, t*] = -ΔR (negative credit at student's wrong choice)
    # composite: c_t[i, t*] = +ΔR (positive credit at teacher's choice)
    original_dc = torch.zeros(B, T, dtype=torch.float32, device=device)
    composite_dc = torch.zeros(n_failed, T, dtype=torch.float32, device=device)
    for j_local in range(n_failed):
        t_j = int(t_star[j_local].item())
        if 0 <= t_j < T:
            target_b = int(failed_idx[j_local].item())
            original_dc[target_b, t_j] = -delta_r[j_local]
            composite_dc[j_local, t_j] = +delta_r[j_local]

    batch.batch["divergence_credit"] = original_dc
    batch.batch["intervention_delta_reward"] = torch.zeros(B, device=device, dtype=torch.float32)
    batch.batch["intervention_used"] = torch.zeros(B, device=device, dtype=torch.float32)

    composite_batch.batch["divergence_credit"] = composite_dc

    # ── Step 12: Mode B append ──────────────────────────────────────
    augmented = DataProto.concat([batch, composite_batch])

    # ── Metrics ─────────────────────────────────────────────────────
    composite_response_mask_sum = composite_batch.batch["response_mask"].float().sum(dim=-1)
    metrics.update({
        "intervention/applied_rate": float(n_failed) / max(B, 1),
        "intervention/n_failed_total": float(n_failed_total),
        "intervention/n_failed_selected": float(n_failed),
        "intervention/delta_reward_mean": float(delta_r.mean().item()),
        "intervention/delta_reward_std": float(delta_r.std().item()) if delta_r.numel() > 1 else 0.0,
        "intervention/delta_reward_min": float(delta_r.min().item()),
        "intervention/delta_reward_max": float(delta_r.max().item()),
        "intervention/delta_reward_pos_rate": float((delta_r > 0).float().mean().item()),
        "intervention/composite_response_length_mean": float(composite_response_mask_sum.mean().item()),
        "intervention/t_star_mean": float(t_star.float().mean().item()),
        "intervention/divergence_credit_abs_mean": float(
            (original_dc.abs().sum() + composite_dc.abs().sum()).item()
            / max(1.0, float(original_dc.numel() + composite_dc.numel()))
        ),
    })

    return InterventionResult(batch=augmented, metrics=metrics)


def detect_failed_samples(token_level_rewards: torch.Tensor, failed_threshold: float) -> torch.Tensor:
    """(B, T) → (B,) bool, True for samples with seq-sum reward below threshold."""
    seq_reward = token_level_rewards.sum(dim=-1)
    return seq_reward < failed_threshold
