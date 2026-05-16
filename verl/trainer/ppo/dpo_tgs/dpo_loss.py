# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""DPO-TGS: linearized DPO advantage estimator.

We avoid surgery on the actor by encoding the DPO gradient direction as a
per-token advantage that flows through the standard PPO clipped surrogate.

For each (chosen, rejected) pair with token-summed (log π_old − log π_ref):
    s_c = β · Σ_t mask_c[t] · (log π_old_c[t] − log π_ref_c[t])
    s_r = β · Σ_t mask_r[t] · (log π_old_r[t] − log π_ref_r[t])
    margin = s_c − s_r
    g      = β · σ(−margin)          # = β·(1 − σ(margin))

Per-token advantages (length-normalized so longer sequences don't dominate):
    A_t[chosen]   = +g / L_chosen     for t in response (mask=1)
    A_t[rejected] = −g / L_rejected   for t in response (mask=1)

Non-pair samples fall back to GRPO group-relative advantage (mixed via α).

This is the *first-order* DPO update — exact at the start of each PPO mini-epoch
when π_θ = π_old, drifts as π_θ moves away from π_old. The standard PPO ratio
clip keeps that drift bounded.

References:
    - OAIF (Guo et al. 2024) validates on-policy generation > offline pre-collected.
      Our chain rollout is on-policy by construction.
    - OFS-DPO (Qi et al. 2024) Proposition 3.3.1 warns about DPO gradient vanishing
      as π_θ drifts from π_ref. The α·GRPO fallback keeps signal alive in that regime;
      an EMA-teacher chase term is reserved for v2.

Design doc: research/dpo_teacher_guided_sampling.md
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch

from verl.trainer.ppo.core_algos import register_adv_est


def _length_per_row(mask: torch.Tensor) -> torch.Tensor:
    """(B, T) mask → (B,) float length, clamped at 1."""
    return mask.float().sum(dim=-1).clamp(min=1.0)


def _grpo_group_advantage(
    rewards: torch.Tensor,        # (B,) sequence-summed reward
    index,                        # uid array length B
    norm_by_std: bool = False,
) -> torch.Tensor:
    """GRPO group-relative sequence advantage (B,)."""
    device = rewards.device
    dtype = rewards.dtype
    B = rewards.shape[0]
    uid_to_idx = defaultdict(list)
    for i in range(B):
        uid_to_idx[index[i]].append(i)
    adv = torch.zeros(B, device=device, dtype=dtype)
    for _, idxs in uid_to_idx.items():
        group = rewards[idxs]
        mu = group.mean()
        if norm_by_std and group.numel() > 1:
            sigma = group.std(unbiased=False).clamp(min=1e-8)
            adv[idxs] = (group - mu) / sigma
        else:
            adv[idxs] = group - mu
    return adv


def _vectorized_pair_advantage(
    *,
    seq_logp_diff: torch.Tensor,   # (B,) Σ_t mask·(logπ_old − logπ_ref)
    pair_id: torch.Tensor,         # (B,) long, -1 means no pair
    pair_role: torch.Tensor,       # (B,) long, +1/-1/0
    mask: torch.Tensor,            # (B, T)
    length: torch.Tensor,          # (B,) float, clamped ≥ 1
    beta: float,
) -> tuple[torch.Tensor, dict]:
    """Compute per-token DPO advantage for paired rows. Vectorized.

    Returns A_dpo (B, T) and a stats dict {num_pairs, margin_mean, margin_pos_rate}.
    """
    B, T = mask.shape
    device = seq_logp_diff.device
    dtype = mask.dtype
    A_dpo = torch.zeros(B, T, device=device, dtype=dtype)

    pair_mask = pair_id >= 0
    if not pair_mask.any():
        return A_dpo, {"num_pairs": 0, "margin_mean": 0.0, "margin_pos_rate": 0.0}

    # max pair id determines tensor size for scatter
    max_pid = int(pair_id.max().item()) + 1
    s_per_row = beta * seq_logp_diff                # (B,)

    # Build per-pair (chosen, rejected) sums via scatter_add
    # chosen contributes s_c at pair_id, rejected contributes s_r at pair_id (negated below)
    chosen_mask = (pair_role == +1)
    rejected_mask = (pair_role == -1)

    s_chosen = torch.zeros(max_pid, device=device, dtype=s_per_row.dtype)
    s_rejected = torch.zeros(max_pid, device=device, dtype=s_per_row.dtype)
    s_chosen.scatter_add_(0, pair_id.clamp(min=0)[chosen_mask], s_per_row[chosen_mask])
    s_rejected.scatter_add_(0, pair_id.clamp(min=0)[rejected_mask], s_per_row[rejected_mask])

    margin = s_chosen - s_rejected                  # (max_pid,)
    g = beta * torch.sigmoid(-margin)               # gradient magnitude per pair

    # Distribute g back to each row by sign(role)
    role_sign = pair_role.to(dtype)
    g_per_row = torch.zeros(B, device=device, dtype=dtype)
    paired_idx = pair_mask.nonzero(as_tuple=False).squeeze(-1)
    pid_lookup = pair_id[paired_idx].clamp(min=0)
    g_per_row[paired_idx] = role_sign[paired_idx] * g.to(dtype)[pid_lookup]

    # Per-token advantage = sign * g / L, broadcast across mask
    A_dpo = (g_per_row / length).unsqueeze(-1) * mask

    # Stats over realized pairs only
    realized_pairs = (s_chosen.abs() + s_rejected.abs() > 0).sum().item()  # rough count
    margin_for_pairs = margin[(s_chosen.abs() + s_rejected.abs()) > 0]
    if margin_for_pairs.numel() > 0:
        margin_mean = float(margin_for_pairs.mean().item())
        margin_pos_rate = float((margin_for_pairs > 0).float().mean().item())
    else:
        margin_mean = 0.0
        margin_pos_rate = 0.0

    return A_dpo, {
        "num_pairs": int(realized_pairs),
        "margin_mean": margin_mean,
        "margin_pos_rate": margin_pos_rate,
    }


@register_adv_est("dpo_teacher_guided")
def compute_dpo_tgs_advantage(
    token_level_rewards: torch.Tensor,       # (B, T)
    response_mask: torch.Tensor,             # (B, T) chain shared prefix already masked
    index,                                   # uid (B,) chain samples of one prompt share uid
    rollout_log_probs: Optional[torch.Tensor] = None,   # (B, T) π_old
    ref_log_prob: Optional[torch.Tensor] = None,        # (B, T) π_ref
    dpo_pair_id: Optional[torch.Tensor] = None,         # (B,)   long, -1 = no pair
    dpo_pair_role: Optional[torch.Tensor] = None,       # (B,)   +1/-1/0
    config: Optional[dict] = None,
    **_unused,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearized DPO advantage with GRPO fallback.

    Returns (advantages, returns); returns == advantages (no critic).

    Note: per-step diagnostic metrics (num_pairs, margin distribution, lengths) are
    computed in `pair_collector.compute_dpo_metrics` and logged from ray_trainer —
    keep this function pure (no I/O / no metrics side-channel).
    """
    dpo_cfg: dict = {}
    if config is not None:
        dpo_cfg = config.get("dpo", {}) or {}

    beta: float = float(dpo_cfg.get("beta", 0.1))
    alpha: float = float(dpo_cfg.get("alpha", 1.0))   # 1.0 = pure DPO, 0.0 = pure GRPO
    grpo_norm_std: bool = bool(dpo_cfg.get("norm_adv_by_std_in_grpo", False))
    min_response_length: int = int(dpo_cfg.get("min_response_length", 0))
    length_penalty_type: str = str(dpo_cfg.get("length_penalty_type", "linear"))

    B, T = token_level_rewards.shape
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype
    mask = response_mask.to(dtype)

    with torch.no_grad():
        # ── Step 1: GRPO baseline (always computed, used as fallback / mix) ──
        seq_reward = token_level_rewards.sum(dim=-1)
        A_grpo_seq = _grpo_group_advantage(seq_reward, index, norm_by_std=grpo_norm_std)
        A_grpo = A_grpo_seq.unsqueeze(1).expand_as(token_level_rewards) * mask

        # ── Step 2: linearized DPO advantage (vectorized over pairs) ─────────
        has_pair_info = (
            dpo_pair_id is not None
            and dpo_pair_role is not None
            and rollout_log_probs is not None
            and ref_log_prob is not None
        )

        if has_pair_info:
            pair_id = dpo_pair_id.to(device).long()
            pair_role = dpo_pair_role.to(device).long()
            logp_diff = (rollout_log_probs.float() - ref_log_prob.float()).to(device)
            seq_logp_diff = (logp_diff * mask).sum(dim=-1)  # (B,)
            L = _length_per_row(mask)
            A_dpo, _ = _vectorized_pair_advantage(
                seq_logp_diff=seq_logp_diff,
                pair_id=pair_id,
                pair_role=pair_role,
                mask=mask,
                length=L,
                beta=beta,
            )

            # ── Step 3: mix DPO + GRPO ──────────────────────────────────────
            in_pair = (pair_id >= 0).to(dtype).unsqueeze(1)   # (B, 1)
            advantages = in_pair * (alpha * A_dpo + (1.0 - alpha) * A_grpo) \
                       + (1.0 - in_pair) * A_grpo
        else:
            advantages = A_grpo

        # ── Step 4: length floor (rescues over-short responses from full credit) ──
        if min_response_length > 0:
            L_eff = mask.sum(dim=-1).clamp(min=1.0)
            if length_penalty_type == "zero":
                length_scale = (L_eff >= min_response_length).to(dtype)
            else:
                length_scale = (L_eff / float(min_response_length)).clamp(max=1.0).to(dtype)
            advantages = advantages * length_scale.unsqueeze(-1)

    return advantages, advantages
