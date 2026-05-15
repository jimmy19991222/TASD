# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Prior-Shift advantage (Bayesian Credit Assignment, ours).

Formula:

    A_seq[i] = R_i  -  mean_{j ∈ group(i)} R_j        # GRPO outcome (seq-level)
    g_t      = KL( P_T(·|x, y_≤t) ‖ P_T(·|x, y_<t) )  # teacher forward-Bayes surprise
    ĝ_t      = g_t / mean_t(g_t)                       # per-sequence normalize on response_mask
    A_t      = A_seq · ĝ_t                             # preserve total: mean_t A_t = A_seq

vs RLSD:
    - RLSD   : A_t = A_seq · clip(exp(sign(A_seq)·(logp_T - logp_S)), 1±ε_w)  → 基于 teacher-student log-ratio
    - Ours   : A_t = A_seq · ĝ_t                                              → 基于 teacher 自己的 belief shift
    - 解释：RLSD 是 "模仿 teacher"；Prior-Shift 是 "找 teacher 认为信息含量大的位置"

实现层面，g_t（即 prior-shift surprise）由 ray_trainer.py 的 lightweight teacher
forward 路径写入 batch.batch["bc_teacher_prior_shift_surprise"]，由 dp_actor 的
compute_teacher_log_probs 在 micro-batch 内计算（避免外传 (B,T,V) logits）。
本函数只做 advantage 数值计算。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch

from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("prior_shift")
def compute_prior_shift_advantage(
    token_level_rewards: torch.Tensor,           # (B, T)
    response_mask: torch.Tensor,                 # (B, T)
    index,                                       # uid array, len B
    teacher_prior_shift_surprise: torch.Tensor,  # (B, T) g_t ≥ 0
    config: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prior-Shift: GRPO outcome advantage redistributed by teacher Bayesian surprise.

    Args:
        token_level_rewards: (B, T) rule-based reward (only seq-sum used → R_i).
        response_mask: (B, T) bool/0-1 mask.
        index: list[str|int]，长度 B，相同 uid = 同一 prompt 的 group。
        teacher_prior_shift_surprise: (B, T) g_t = KL(D_t‖D_{t-1})，由 teacher forward 内部算好。
            约定 g_t ≥ 0，且 g_0 = 0（首 token 无 prior 可比）。masked 位置应已置 0。
        config: AlgoConfig dict-like. 读取 algorithm.prior_shift.{eps_norm, max_ratio,
            uniform_fallback, norm_adv_by_std_in_grpo}。

    Returns:
        advantages: (B, T) per-token advantage
        returns:    (B, T) 与 advantages 相同（无 critic）
    """
    if teacher_prior_shift_surprise is None:
        raise ValueError(
            "Prior-Shift advantage requires teacher_prior_shift_surprise in batch. "
            "Make sure adv_estimator=prior_shift triggers compute_prior_shift_surprise=True "
            "in the lightweight teacher forward (ray_trainer.py)."
        )

    ps_cfg: dict = {}
    if config is not None:
        ps_cfg = config.get("prior_shift", {}) or {}
    eps_norm: float = float(ps_cfg.get("eps_norm", 1e-6))           # mean(g_t) 的下限，避免除 0
    max_ratio: float = float(ps_cfg.get("max_ratio", 3.0))          # ĝ_t 的上限 (v2: 10→3 防 length collapse)
    uniform_fallback: bool = bool(ps_cfg.get("uniform_fallback", True))  # mean(g_t) ≈ 0 时退化为 GRPO 平摊
    norm_adv_by_std_in_grpo: bool = bool(ps_cfg.get("norm_adv_by_std_in_grpo", False))
    # v2 新增：clip 后重新归一化（保证 mean(ĝ)=1，梯度量级不随 clip 比例漂移）
    renormalize_after_clip: bool = bool(ps_cfg.get("renormalize_after_clip", True))
    # v2 新增：response 长度下限惩罚（抵抗 length collapse）
    min_response_length: int = int(ps_cfg.get("min_response_length", 50))
    length_penalty_type: str = str(ps_cfg.get("length_penalty_type", "linear"))  # linear / zero

    B, T = token_level_rewards.shape
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype

    with torch.no_grad():
        mask = response_mask.to(dtype)

        # ── Step 1: GRPO outcome advantage (seq-level scalar) ──────────────
        seq_reward = token_level_rewards.sum(dim=-1)  # (B,)
        uid_to_indices: dict = defaultdict(list)
        for i in range(B):
            uid_to_indices[index[i]].append(i)

        seq_advantage = torch.zeros(B, device=device, dtype=dtype)
        for uid, idxs in uid_to_indices.items():
            group_r = seq_reward[idxs]
            mu = group_r.mean()
            if norm_adv_by_std_in_grpo and group_r.numel() > 1:
                sigma = group_r.std(unbiased=False).clamp(min=1e-8)
                seq_advantage[idxs] = (group_r - mu) / sigma
            else:
                seq_advantage[idxs] = group_r - mu

        A_seq = seq_advantage.unsqueeze(1).expand_as(token_level_rewards)  # (B, T)

        # ── Step 2: per-sequence normalize surprise → ĝ_t ───────────────────
        # 仅在 response_mask=1 的位置参与统计；其余位置贡献 0。
        g = teacher_prior_shift_surprise.to(dtype) * mask                  # (B, T)
        L = mask.sum(dim=-1).clamp(min=1.0)                                # (B,) effective length
        g_sum = g.sum(dim=-1)                                              # (B,)
        g_mean = g_sum / L                                                 # (B,) avg surprise
        # 退化判定：mean(g_t) 太小（teacher 没在任何位置感到惊讶）→ 平摊
        degenerate = g_mean < eps_norm                                     # (B,)

        denom = g_mean.clamp(min=eps_norm).unsqueeze(-1)                   # (B, 1)
        g_hat = g / denom                                                  # (B, T) ĝ_t = g_t / mean_t(g_t)

        # 上限保护：超大 surprise 不让 advantage 爆掉
        g_hat = g_hat.clamp(max=max_ratio)

        # v2: clip 后重新归一化，保证 mean_t(ĝ_t)=1 → total magnitude = A_seq
        if renormalize_after_clip:
            g_hat_mean = (g_hat * mask).sum(dim=-1, keepdim=True) / L.unsqueeze(-1).clamp(min=1.0)
            g_hat = g_hat / g_hat_mean.clamp(min=eps_norm)
            g_hat = g_hat * mask  # 再 mask 一遍防 leak

        # 退化 fallback：整序列 g≈0 → ĝ_t = 1（与 GRPO 平摊一致）
        if uniform_fallback and degenerate.any():
            uniform = mask.clone()                                         # (B, T) =1 on response, else 0
            g_hat = torch.where(degenerate.unsqueeze(-1), uniform, g_hat)

        # 关闭 mask 之外的位置（防御）
        g_hat = g_hat * mask

        # ── Step 3: 合成 token-level advantage ──────────────────────────────
        advantages = A_seq * g_hat * mask

        # v2: length floor penalty — 抵抗 length collapse
        if min_response_length > 0:
            length_ratio = (L / float(min_response_length)).clamp(max=1.0)  # (B,) ∈ (0,1] 短序列<1
            if length_penalty_type == "zero":
                # 极端：短于阈值直接清零 advantage（不学这些 stub 回答）
                length_scale = (L >= min_response_length).float()
            else:
                # linear：L/min_len 线性衰减，越短惩罚越重
                length_scale = length_ratio
            advantages = advantages * length_scale.unsqueeze(-1)

    return advantages, advantages
