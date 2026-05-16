# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
TCCA (Token-level Causal Credit Assignment) advantage estimator.

公式 (additive, V2 chain rollout 配套):

    A_seq[i] = R_i - mean_{j ∈ group}(R_j)               # GRPO group-relative
    A_t[i, t] = (A_seq[i] + λ_div · c_t[i, t]) · response_mask[i, t] · length_scale(L_i)
                └─────────────┘   └──────────┘
                  base credit       divergence-point credit
                  (uniform)         (additive bonus at t_i where teacher intervened)

其中:
    c_t[i, t] = divergence_credit field, 由 tcca_chain.py 写入 batch:
        for chain (y_0, y_1, ..., y_{n-1}) of each prompt:
          for i in 1..n-1:
            ΔR_i = R(y_i) - R(y_{i-1})
            c_t[y_i,   t_i] += +ΔR_i      # teacher's choice at t_i
            c_t[y_{i-1}, t_i] -= +ΔR_i     # student's choice at same position (mirror)
        端点 y_0, y_{n-1} 自然只有单向 credit

设计要点:
    - additive (而非 multiplicative): c_t 保持符号语义 (正=好 token, 负=坏 token)
    - λ_div=0 → 退化为纯 GRPO + masked prefix (论文 baseline ablation = 选项 D)
    - λ_div>0 → TCCA divergence-point modulation (论文 main = 选项 C)
    - response_mask 在 chain rollout 时已对 shared prefix 置 0 (Layer 2)

设计文档: research/tcca_v2_design.md
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch

from verl.trainer.ppo.core_algos import register_adv_est


@register_adv_est("intervention_credit")
def compute_intervention_credit_advantage(
    token_level_rewards: torch.Tensor,                       # (B, T)
    response_mask: torch.Tensor,                             # (B, T), shared prefix 已 mask
    index,                                                   # uid array (chain samples 同 uid)
    divergence_credit: Optional[torch.Tensor] = None,        # (B, T) c_t from chain rollout
    config: Optional[dict] = None,
    # legacy kwargs (向后兼容, 不再使用):
    teacher_log_probs: Optional[torch.Tensor] = None,
    student_log_probs: Optional[torch.Tensor] = None,
    teacher_prior_shift_surprise: Optional[torch.Tensor] = None,
    intervention_delta_reward: Optional[torch.Tensor] = None,
    intervention_used: Optional[torch.Tensor] = None,
    token_causal_credit: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TCCA advantage (additive, chain rollout 配套).

    Args:
        token_level_rewards: (B, T) outcome reward (sum over T = R_i per sample).
        response_mask: (B, T) shared prefix 已置 0 (Layer 2).
        index: uid array, chain samples of same prompt 共享 uid.
        divergence_credit: (B, T) c_t per-token bonus, 由 tcca_chain 写入.
        config: AlgoConfig.

    Returns:
        advantages: (B, T)
        returns: (B, T) (= advantages, no critic)

    Legacy kwargs (token_causal_credit, intervention_delta_reward, etc.) 接受但忽略,
    保持向后兼容 commit 7dd41d8 之前的 ray_trainer dispatch.
    """
    ic_cfg: dict = {}
    if config is not None:
        ic_cfg = config.get("intervention_credit", {}) or {}

    lambda_div: float = float(ic_cfg.get("lambda_div_credit", 1.0))
    divergence_credit_clip: float = float(ic_cfg.get("divergence_credit_clip", 1.0))
    norm_adv_by_std: bool = bool(ic_cfg.get("norm_adv_by_std_in_grpo", False))
    min_response_length: int = int(ic_cfg.get("min_response_length", 50))
    length_penalty_type: str = str(ic_cfg.get("length_penalty_type", "linear"))

    # 向后兼容: 如果只有 token_causal_credit (旧字段名), 当作 divergence_credit 用
    if divergence_credit is None and token_causal_credit is not None:
        divergence_credit = token_causal_credit

    B, T = token_level_rewards.shape
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype

    with torch.no_grad():
        mask = response_mask.to(dtype)

        # ── Step 1: GRPO baseline seq advantage ──────────────────────
        # group_size 可变 (chain rollout 端点失败 fallback 可能让 group < n)
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

        A_seq = seq_advantage.unsqueeze(1).expand_as(token_level_rewards)  # (B, T) broadcast

        # ── Step 2: divergence-point credit modulation (additive) ────
        # A_t = (A_seq + λ_div · c_t) · response_mask · length_scale
        if divergence_credit is not None and lambda_div > 0.0:
            c_t = divergence_credit.to(device=device, dtype=dtype)
            c_t = c_t.clamp(min=-divergence_credit_clip, max=divergence_credit_clip)
            advantages = (A_seq + lambda_div * c_t) * mask
        else:
            # Layer 3 D (baseline): pure GRPO + masked prefix
            advantages = A_seq * mask

        # ── Step 3: length floor penalty (复用 prior_shift v2 防护) ──
        if min_response_length > 0:
            L_eff = mask.sum(dim=-1).clamp(min=1.0)
            length_ratio = (L_eff / float(min_response_length)).clamp(max=1.0)
            if length_penalty_type == "zero":
                length_scale = (L_eff >= min_response_length).to(dtype)
            else:  # linear
                length_scale = length_ratio.to(dtype)
            advantages = advantages * length_scale.unsqueeze(-1)

    return advantages, advantages
