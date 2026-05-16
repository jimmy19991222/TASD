# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Teacher-Guided Dynamic Intervention advantage (Bayesian Credit Assignment Tier 3, ours).

Causal counterfactual credit on top of Prior-Shift's correlational g_t reweight.

Formula (per sample i, after intervention rollout has appended composite samples to batch):

    A_seq[i] = (R_i - mean_{j ∈ group(i)} R_j)         # GRPO baseline
             + λ · ΔR_i · 𝟙[i is intervention sample]   # causal credit
    g_t      = KL( P_T(·|x, y_≤t) ‖ P_T(·|x, y_<t) )   # teacher forward Bayes surprise
    ĝ_t      = clip(g_t / mean_t(g_t), max_ratio)       # per-sequence normalize + clip
    A_t      = A_seq · ĝ_t · length_scale(L)            # final per-token advantage

Mode B (append) 关键性质:
    - 失败样本 y_s (R_s 低)         → A_seq < 0, 推低对应 token
    - 复合样本 y' (R' > R_s)        → A_seq > 0 + λ·ΔR > 0, 推高 teacher 推荐 token
    天然形成 contrastive pair on the same prefix。

实现层面:
    - ΔR 通道由 ray_trainer 的 intervention_rollout block 写入 batch.batch["intervention_delta_reward"]
    - g_t 通道复用 prior_shift 的 batch.batch["bc_teacher_prior_shift_surprise"]
    - 当 enable_intervention=False (或 ΔR/used 通道缺失) 时退化为 prior_shift
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch

from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("intervention_credit")
def compute_intervention_credit_advantage(
    token_level_rewards: torch.Tensor,                   # (B, T)
    response_mask: torch.Tensor,                         # (B, T)
    index,                                               # uid array, len B
    teacher_prior_shift_surprise: Optional[torch.Tensor] = None,  # (B, T) g_t ≥ 0
    intervention_delta_reward: Optional[torch.Tensor] = None,     # (B,) ΔR
    intervention_used: Optional[torch.Tensor] = None,             # (B,) bool
    config: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Intervention-Credit advantage.

    Args:
        token_level_rewards: (B, T) rule-based reward (only seq-sum used → R_i).
        response_mask: (B, T) bool/0-1 mask.
        index: list[str|int]，长度 B，相同 uid = 同一 prompt 的 group。
        teacher_prior_shift_surprise: (B, T) g_t = KL(D_t‖D_{t-1})。
            若为 None，则 g_t reweight 被关闭，token-level advantage 走平摊。
        intervention_delta_reward: (B,) ΔR = R(y') - R(y_s)。
            非 intervention 样本应为 0；若为 None 则整体 ΔR 通道关闭（退化为 prior_shift）。
        intervention_used: (B,) bool/0-1，标识哪些样本是 composite rollout。
        config: AlgoConfig dict-like. 读取 algorithm.intervention_credit.{...}。

    Returns:
        advantages: (B, T) per-token advantage
        returns:    (B, T) 与 advantages 相同（无 critic）
    """
    ic_cfg: dict = {}
    if config is not None:
        ic_cfg = config.get("intervention_credit", {}) or {}
    g_t_cfg: dict = ic_cfg.get("g_t", {}) or {}

    eps_norm: float = float(g_t_cfg.get("eps_norm", 1e-6))
    max_ratio: float = float(g_t_cfg.get("max_ratio", 3.0))
    uniform_fallback: bool = bool(g_t_cfg.get("uniform_fallback", True))
    renormalize_after_clip: bool = bool(g_t_cfg.get("renormalize_after_clip", True))

    lambda_delta_r: float = float(ic_cfg.get("lambda_delta_r", 1.0))
    norm_adv_by_std: bool = bool(ic_cfg.get("norm_adv_by_std_in_grpo", False))

    min_response_length: int = int(ic_cfg.get("min_response_length", 50))
    length_penalty_type: str = str(ic_cfg.get("length_penalty_type", "linear"))

    B, T = token_level_rewards.shape
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype

    with torch.no_grad():
        mask = response_mask.to(dtype)

        # ── Step 1: GRPO baseline seq-level advantage ─────────────────────
        # 注意: append 后 group_size 可变，uid_to_indices 天然支持
        seq_reward = token_level_rewards.sum(dim=-1)  # (B,)
        uid_to_indices: dict = defaultdict(list)
        for i in range(B):
            uid_to_indices[index[i]].append(i)

        seq_advantage = torch.zeros(B, device=device, dtype=dtype)
        for uid, idxs in uid_to_indices.items():
            group_r = seq_reward[idxs]
            mu = group_r.mean()
            if norm_adv_by_std and group_r.numel() > 1:
                sigma = group_r.std(unbiased=False).clamp(min=1e-8)
                seq_advantage[idxs] = (group_r - mu) / sigma
            else:
                seq_advantage[idxs] = group_r - mu

        # ── Step 2: ΔR causal credit injection (仅 intervention 样本) ──────
        if intervention_delta_reward is not None:
            delta_r = intervention_delta_reward.to(device=device, dtype=dtype)  # (B,)
            if intervention_used is not None:
                used = intervention_used.to(device=device, dtype=dtype)         # (B,)
            else:
                # 没传 used → 用 delta_r != 0 兜底
                used = (delta_r.abs() > 1e-12).to(dtype)
            seq_advantage = seq_advantage + lambda_delta_r * delta_r * used

        A_seq = seq_advantage.unsqueeze(1).expand_as(token_level_rewards)  # (B, T)

        # ── Step 3: per-sequence normalize surprise → ĝ_t ──────────────────
        if teacher_prior_shift_surprise is not None:
            g = teacher_prior_shift_surprise.to(dtype) * mask                  # (B, T)
            L_eff = mask.sum(dim=-1).clamp(min=1.0)                            # (B,)
            g_sum = g.sum(dim=-1)
            g_mean = g_sum / L_eff
            degenerate = g_mean < eps_norm

            denom = g_mean.clamp(min=eps_norm).unsqueeze(-1)                   # (B, 1)
            g_hat = g / denom                                                  # (B, T)
            g_hat = g_hat.clamp(max=max_ratio)

            if renormalize_after_clip:
                g_hat_mean = (g_hat * mask).sum(dim=-1, keepdim=True) / L_eff.unsqueeze(-1).clamp(min=1.0)
                g_hat = g_hat / g_hat_mean.clamp(min=eps_norm)
                g_hat = g_hat * mask

            if uniform_fallback and degenerate.any():
                uniform = mask.clone()
                g_hat = torch.where(degenerate.unsqueeze(-1), uniform, g_hat)

            g_hat = g_hat * mask
        else:
            # 没有 g_t (例如 intervention_credit 第一次跑还没接通 teacher fwd)
            # 退化为均匀分配
            g_hat = mask.clone()
            L_eff = mask.sum(dim=-1).clamp(min=1.0)

        # ── Step 4: 合成 token-level advantage ─────────────────────────────
        advantages = A_seq * g_hat * mask

        # ── Step 5: length floor penalty (复用 prior_shift v2 防护) ─────────
        if min_response_length > 0:
            length_ratio = (L_eff / float(min_response_length)).clamp(max=1.0)
            if length_penalty_type == "zero":
                length_scale = (L_eff >= min_response_length).to(dtype)
            else:  # linear
                length_scale = length_ratio.to(dtype)
            advantages = advantages * length_scale.unsqueeze(-1)

    return advantages, advantages
