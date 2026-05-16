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
    """**TCCA (Token-level Causal Credit Assignment)** — 真实 teacher 介入 + per-token causal credit.

    对每个失败 sample 在 top-K positions 分别做 intervention，得到 K 个独立 ΔR_{t_k} 作为
    **per-token causal credit**。这是 token-level credit assignment 的真创新（不是数据增强）。

    流程:
      1. failed = R < threshold；top_k_positions = compute_topk_divergence_positions
      2. for k in 1..K:
         a. teacher 在 t_k 改写 intervention_length 个 token → composite y'_{t_k}
         b. R(y'_{t_k}) → ΔR_{t_k} = R(y'_{t_k}) - R(y)
         c. 构造 composite_k 的 token_causal_credit: c_t[k] = +ΔR_{t_k} at [t_k, t_k+intervention_length)
      3. 构造原 batch 的 token_causal_credit: c_t[k] = -ΔR_{t_k} at 相同位置 (惩罚错误 token)
      4. DataProto.concat([batch, composite_1, ..., composite_K])
      5. 写入 13 个 intervention/* 指标 (含每 K 的 ΔR 分布)

    V1 限制 (留 Phase 2.2 follow-up):
      - 不真做 student tail 续写（保留学生原 tail）
      - composite 复制原样本 old_log_probs (k 位置 PPO ratio 略不准, clip 兜底)
      - 不重算 bc_teacher_* 在 composite (uniform_fallback 兜底)
    """
    from verl.protocol import DataProto
    import numpy as np
    import torch

    failed_threshold: float = float(ic_cfg.get("failed_threshold", 0.5))
    intervention_length: int = int(ic_cfg.get("intervention_length_k", 2))
    top_k: int = int(ic_cfg.get("top_k_positions", 3))   # TCCA: K 个 intervention 位置
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

    def _write_zeros_and_return(b):
        b.batch["intervention_delta_reward"] = torch.zeros(B, device=device, dtype=torch.float32)
        b.batch["intervention_used"] = torch.zeros(B, device=device, dtype=torch.float32)
        b.batch["token_causal_credit"] = torch.zeros(
            B, b.batch["responses"].shape[1], device=device, dtype=torch.float32
        )
        return InterventionResult(batch=b, metrics={**metrics, "intervention/applied_rate": 0.0})

    if n_failed_total == 0:
        return _write_zeros_and_return(batch)

    # 按 uid 限制每组最多 max_per_group 个 intervention sample（不是 K，是 uid 内 intervention 样本数）
    # 由于每个失败 sample 会产出 K 个 composite，max_per_group 应理解为"每个 prompt 最多挑出几个失败 sample 做 TCCA"
    # 防止单个 prompt 占满整个 batch 的 intervention 配额
    max_failed_per_group: int = max(1, max_per_group // max(top_k, 1))
    uid = batch.non_tensor_batch["uid"]
    failed_indices_all = failed_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    selected_indices = []
    per_group_count: dict = {}
    for idx in failed_indices_all:
        u = uid[idx]
        if per_group_count.get(u, 0) < max_failed_per_group:
            selected_indices.append(int(idx))
            per_group_count[u] = per_group_count.get(u, 0) + 1
    failed_idx = torch.tensor(selected_indices, device=device, dtype=torch.long)
    n_failed = len(failed_idx)

    if n_failed == 0:
        return _write_zeros_and_return(batch)

    # ── Step 2: 计算 top-K t_star (失败样本) ───────────────────────────
    topk_all = compute_topk_divergence_positions(batch, ic_cfg, top_k=top_k)  # (B, K)
    if topk_all is None:
        return _write_zeros_and_return(batch)

    topk_failed = topk_all[failed_idx]  # (n_failed, K) — 在 response 坐标

    # 推导 prompt_length
    T = batch.batch["responses"].shape[1]
    L = batch.batch["input_ids"].shape[1]
    P = L - T

    topk_abs_failed = (P + topk_failed).long()  # (n_failed, K) 在 teacher_input_ids 坐标

    # ── Step 3: 对每个 k 做一次 intervention rollout ──────────────────
    # 每个失败 sample 会产生 K 个 composite samples (Mode B append × K)

    all_composite_batches = []
    all_delta_r_per_k = []          # K-list of (n_failed,) tensors
    all_composite_response_masks = []  # K-list of (n_failed, T) tensors（用于度量）

    for k_step in range(top_k):
        t_star_k = topk_failed[:, k_step]                  # (n_failed,) response 坐标
        t_star_abs_k = topk_abs_failed[:, k_step]          # (n_failed,) input_ids 坐标

        # 子 batch
        sub_proto = DataProto.from_dict(
            tensors={
                "teacher_input_ids": batch.batch["teacher_input_ids"][failed_idx],
                "teacher_attention_mask": batch.batch["teacher_attention_mask"][failed_idx],
                "teacher_position_ids": batch.batch["teacher_position_ids"][failed_idx],
                "t_star_abs": t_star_abs_k,
            },
            meta_info={"k": intervention_length, "temperature": teacher_temp},
        )
        teacher_gen_out = actor_rollout_wg.teacher_generate_at_positions(sub_proto)
        intervention_tokens = teacher_gen_out.batch["intervention_tokens"].to(device)  # (n_failed, intervention_length)

        # 构造 composite_k：复制 failed 子集，替换 [t_star_k, t_star_k+intervention_length) 位置
        comp_responses = batch.batch["responses"][failed_idx].clone()
        comp_input_ids = batch.batch["input_ids"][failed_idx].clone()
        comp_attention = batch.batch["attention_mask"][failed_idx].clone()
        comp_position = batch.batch["position_ids"][failed_idx].clone()
        comp_response_mask = batch.batch["response_mask"][failed_idx].clone()

        bidx = torch.arange(n_failed, device=device)
        for d in range(intervention_length):
            pos_rel = (t_star_k + d).long()
            pos_abs = (t_star_abs_k + d).long()
            valid_rel = (pos_rel >= 0) & (pos_rel < T)
            valid_abs = (pos_abs >= 0) & (pos_abs < L)
            if valid_rel.any():
                v = bidx[valid_rel]
                comp_responses[v, pos_rel[valid_rel]] = intervention_tokens[v, d].long()
                comp_response_mask[v, pos_rel[valid_rel]] = 1
            if valid_abs.any():
                v = bidx[valid_abs]
                comp_input_ids[v, pos_abs[valid_abs]] = intervention_tokens[v, d].long()
                comp_attention[v, pos_abs[valid_abs]] = 1

        # 构造 composite_k DataProto
        comp_tensors = {}
        for key, val in batch.batch.items():
            if not torch.is_tensor(val) or val.shape[0] != B:
                continue
            comp_tensors[key] = val[failed_idx].clone()
        comp_tensors["responses"] = comp_responses
        comp_tensors["input_ids"] = comp_input_ids
        comp_tensors["attention_mask"] = comp_attention
        comp_tensors["position_ids"] = comp_position
        comp_tensors["response_mask"] = comp_response_mask

        comp_non_tensor = {}
        for key, arr in batch.non_tensor_batch.items():
            if isinstance(arr, np.ndarray) and len(arr) == B:
                comp_non_tensor[key] = arr[failed_idx.cpu().numpy()].copy()

        composite_k = DataProto(
            batch=DataProto.from_dict(tensors=comp_tensors).batch,
            non_tensor_batch=comp_non_tensor,
            meta_info=dict(batch.meta_info or {}),
        )

        # 清掉会让 reward_fn 短路的字段
        for k_pop in ["rm_scores", "token_level_scores", "token_level_rewards", "advantages", "returns"]:
            if k_pop in composite_k.batch.keys():
                composite_k.batch.pop(k_pop)

        # 重打分
        try:
            rew_result = reward_fn(composite_k, return_dict=True)
            new_reward_tensor_k = rew_result["reward_tensor"].to(device)
        except TypeError:
            new_reward_tensor_k = reward_fn(composite_k).to(device)

        # ΔR_k
        composite_seq_reward_k = new_reward_tensor_k.sum(dim=-1)             # (n_failed,)
        original_seq_reward = seq_reward[failed_idx]                          # (n_failed,)
        delta_r_k = composite_seq_reward_k - original_seq_reward              # (n_failed,)

        composite_k.batch["token_level_scores"] = new_reward_tensor_k
        composite_k.batch["token_level_rewards"] = new_reward_tensor_k
        composite_k.batch["intervention_delta_reward"] = delta_r_k.float()
        composite_k.batch["intervention_used"] = torch.ones(n_failed, device=device, dtype=torch.float32)

        # ── TCCA 核心：构造 composite_k 的 token_causal_credit ────
        # composite_k 在 [t_star_k, t_star_k+intervention_length) 这些 token 是 teacher 选的,
        # 它们造成了 R(y'_k) (相对 R(y) 有 ΔR_k 的因果效应)。
        # → 这些 token 的 c_t = +ΔR_k (teacher token 应被奖励/惩罚, 由 ΔR_k 符号决定)
        cc_k = torch.zeros(n_failed, T, device=device, dtype=torch.float32)
        for d in range(intervention_length):
            pos_rel = (t_star_k + d).long()
            valid_rel = (pos_rel >= 0) & (pos_rel < T)
            if valid_rel.any():
                cc_k[bidx[valid_rel], pos_rel[valid_rel]] = delta_r_k[valid_rel].float()
        composite_k.batch["token_causal_credit"] = cc_k

        all_composite_batches.append(composite_k)
        all_delta_r_per_k.append(delta_r_k)
        all_composite_response_masks.append(comp_response_mask)

    # ── Step 4: 构造原 batch 的 token_causal_credit ──────────────────
    # 对原 sample, 同样位置 c_t = -ΔR_k (惩罚 student 在这些位置的错误选择)
    # 多个 k 落到同一位置时取 sum (罕见, 我们已用 min_gap 避免大量重叠)
    original_cc = torch.zeros(B, T, device=device, dtype=torch.float32)
    for k_step in range(top_k):
        t_star_k = topk_failed[:, k_step]
        delta_r_k = all_delta_r_per_k[k_step]
        for d in range(intervention_length):
            pos_rel = (t_star_k + d).long()
            valid_rel = (pos_rel >= 0) & (pos_rel < T)
            if valid_rel.any():
                bidx_valid = torch.arange(n_failed, device=device)[valid_rel]
                target_b = failed_idx[bidx_valid]
                # c_t accumulate (subtraction = penalty for wrong tokens)
                original_cc[target_b, pos_rel[valid_rel]] -= delta_r_k[valid_rel].float()

    batch.batch["token_causal_credit"] = original_cc
    batch.batch["intervention_delta_reward"] = torch.zeros(B, device=device, dtype=torch.float32)
    batch.batch["intervention_used"] = torch.zeros(B, device=device, dtype=torch.float32)

    # ── Step 5: concat 全部 K 个 composite + 原 batch ────────────────
    augmented_batch = DataProto.concat([batch] + all_composite_batches)

    # ── TCCA metrics ─────────────────────────────────────────────────
    # 汇总所有 K 个 ΔR
    all_delta_r_flat = torch.cat(all_delta_r_per_k, dim=0)  # (K * n_failed,)
    composite_resp_mean = torch.cat(
        [m.float().sum(dim=-1) for m in all_composite_response_masks], dim=0
    ).mean()

    metrics.update({
        "intervention/applied_rate": n_failed / max(B, 1),
        "intervention/top_k_positions": float(top_k),
        "intervention/delta_reward_mean": float(all_delta_r_flat.mean().item()),
        "intervention/delta_reward_std": float(all_delta_r_flat.std().item()) if all_delta_r_flat.numel() > 1 else 0.0,
        "intervention/delta_reward_min": float(all_delta_r_flat.min().item()),
        "intervention/delta_reward_max": float(all_delta_r_flat.max().item()),
        "intervention/delta_reward_pos_rate": float((all_delta_r_flat > 0).float().mean().item()),
        "intervention/composite_response_length_mean": float(composite_resp_mean.item()),
        "intervention/n_failed_total": float(n_failed_total),
        "intervention/n_failed_selected": float(n_failed),
        "intervention/n_composites_total": float(n_failed * top_k),
        "intervention/group_size_post_append_mean": float(B + n_failed * top_k) / float(len(set(uid))),
        # token causal credit 度量
        "intervention/token_causal_credit_abs_mean": float(
            (original_cc.abs().sum() + sum(c.batch["token_causal_credit"].abs().sum() for c in all_composite_batches))
            / max(1.0, (original_cc.numel() + sum(c.batch["token_causal_credit"].numel() for c in all_composite_batches)))
        ),
    })

    # divergence position 度量（用第一个位置 t_star_k=0 当代表）
    diag_metrics = _diagnose_only(batch, ic_cfg)
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
