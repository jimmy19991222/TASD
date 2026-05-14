# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
RLSD baseline (Reinforcement Learning with Self-Distillation).

Reference: arXiv:2604.03128v2.

Public formula (per-token advantage, weighted seq advantage):

    A_seq[i]   = R_i - mean_{j∈group(i)} R_j        # GRPO outcome (seq-level)
    delta_t    = log P_T(y_t | x, y_<t) - log P_S(y_t | x, y_<t)
    weight_t   = clip( exp( sign(A_seq) · delta_t ), 1 - eps_w, 1 + eps_w )
    A_t        = A_seq · weight_t

teacher 来源（由 self_distillation.teacher_regularization 切换）：
    - "ema"        : 慢衰 EMA (teacher_update_rate)
    - "hard_sync"  : 每 N step 把 teacher 拷贝为当前 student (TODO: 后续接入)

实现层面，teacher_log_probs 已由 ray_trainer.py 的 teacher forward 链路写入
batch.batch["rlsd_teacher_log_probs"]，student_log_probs 直接复用 old_log_probs。
本函数只做 advantage 数值计算。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch

from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("rlsd")
def compute_rlsd_advantage(
    token_level_rewards: torch.Tensor,        # (B, T)
    response_mask: torch.Tensor,              # (B, T)
    index,                                    # uid array, len B
    teacher_log_probs: torch.Tensor,          # (B, T)  log P_T(y_t)
    student_log_probs: torch.Tensor,          # (B, T)  log P_S(y_t)，等于 old_log_probs
    config: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RLSD: GRPO outcome advantage reweighted by teacher-student log-ratio.

    Args:
        token_level_rewards: (B, T) rule-based reward。RLSD 只用其 seq sum 作为 R_i。
        response_mask: (B, T) bool/0-1 mask
        index: list[str|int]，长度 B，相同 uid = 同一 prompt 的 group
        teacher_log_probs: (B, T) log P_T(y_t)，由 frozen/EMA teacher forward 算出
        student_log_probs: (B, T) log P_S(y_t)，复用 old_log_probs
        config: AlgoConfig dict-like。读取 algorithm.rlsd.{eps_w, norm_adv_by_std_in_grpo}

    Returns:
        advantages: (B, T) per-token advantage
        returns:    (B, T) 与 advantages 相同（无 critic）
    """
    if teacher_log_probs is None:
        raise ValueError(
            "RLSD advantage requires teacher_log_probs in batch. "
            "Make sure teacher forward is enabled (loss_mode/adv_estimator triggers self-distillation)."
        )
    if student_log_probs is None:
        raise ValueError("RLSD advantage requires student_log_probs (old_log_probs) in batch.")

    rlsd_cfg = {}
    if config is not None:
        rlsd_cfg = config.get("rlsd", {}) or {}
    eps_w: float = float(rlsd_cfg.get("eps_w", 0.2))
    norm_adv_by_std_in_grpo: bool = bool(rlsd_cfg.get("norm_adv_by_std_in_grpo", False))

    B, T = token_level_rewards.shape
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype

    with torch.no_grad():
        # ── Step 1: GRPO outcome advantage (seq-level scalar) ──────────────
        seq_reward = token_level_rewards.sum(dim=-1)   # (B,)
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

        # ── Step 2: teacher-student log-ratio token reweighting ─────────────
        # delta_t = log P_T(y_t) - log P_S(y_t)
        delta = (teacher_log_probs - student_log_probs) * response_mask
        sign_A = torch.sign(A_seq)
        weight = torch.exp(sign_A * delta)
        weight = weight.clamp(min=1.0 - eps_w, max=1.0 + eps_w)

        advantages = A_seq * weight * response_mask

    return advantages, advantages
