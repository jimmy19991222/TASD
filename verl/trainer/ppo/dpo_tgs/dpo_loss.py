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


def _delta_r_weight(delta_r: torch.Tensor, mode: str, tau: float = 1.0) -> torch.Tensor:
    """③ ΔR-Weighted DPO: per-pair weight from reward gap magnitude.

    delta_r: (n_pairs,) float tensor of R(chosen) - R(rejected), assumed ≥ 0 for valid pairs.
    Returns: (n_pairs,) float weight tensor; weight=1.0 when mode='none'.
    """
    if mode == "none" or mode == "":
        return torch.ones_like(delta_r)
    abs_dr = delta_r.clamp(min=0.0)
    if mode == "linear":
        return abs_dr
    if mode == "sqrt":
        return abs_dr.sqrt()
    if mode == "squared":
        return abs_dr * abs_dr
    if mode == "sigmoid":
        return torch.sigmoid(abs_dr / max(tau, 1e-6))
    return torch.ones_like(delta_r)


def _vectorized_pair_advantage(
    *,
    seq_logp_diff: torch.Tensor,   # (B,) Σ_t mask·(logπ_old − logπ_ref) [or − logπ_T_OPSD if anchored]
    pair_id: torch.Tensor,         # (B,) long, -1 means no pair
    pair_role: torch.Tensor,       # (B,) long, +1/-1/0
    mask: torch.Tensor,            # (B, T) effective mask (can be mask_full / mask_token / mask_cont)
    length: torch.Tensor,          # (B,) float, clamped ≥ 1
    beta: float,
    pair_weight: Optional[torch.Tensor] = None,   # (max_pid,) per-pair scalar weight (③ ΔR-Weighted)
    margin_extra_per_pair: Optional[torch.Tensor] = None,  # (max_pid,) extra margin added inside σ (① combine multiple components)
) -> tuple[torch.Tensor, dict]:
    """Compute per-token DPO advantage for paired rows. Vectorized.

    Returns A_dpo (B, T) and a stats dict.
    """
    B, T = mask.shape
    device = seq_logp_diff.device
    # Force float32 for the advantage tensor regardless of mask dtype: response_mask is
    # often torch.long (set in tcca_chain._build_y_i_batch) and integer division would
    # silently truncate the per-token advantage to 0.
    dtype = torch.float32
    A_dpo = torch.zeros(B, T, device=device, dtype=dtype)

    pair_mask_present = pair_id >= 0
    if not pair_mask_present.any():
        return A_dpo, {"num_pairs": 0, "margin_mean": 0.0, "margin_pos_rate": 0.0}

    max_pid = int(pair_id.max().item()) + 1
    s_per_row = beta * seq_logp_diff                # (B,)

    chosen_mask = (pair_role == +1)
    rejected_mask = (pair_role == -1)

    s_chosen = torch.zeros(max_pid, device=device, dtype=s_per_row.dtype)
    s_rejected = torch.zeros(max_pid, device=device, dtype=s_per_row.dtype)
    s_chosen.scatter_add_(0, pair_id.clamp(min=0)[chosen_mask], s_per_row[chosen_mask])
    s_rejected.scatter_add_(0, pair_id.clamp(min=0)[rejected_mask], s_per_row[rejected_mask])

    margin = s_chosen - s_rejected
    if margin_extra_per_pair is not None:
        margin = margin + margin_extra_per_pair.to(margin.dtype)
    g = beta * torch.sigmoid(-margin)

    if pair_weight is not None:
        g = g * pair_weight.to(g.dtype)

    role_sign = pair_role.to(dtype)
    g_per_row = torch.zeros(B, device=device, dtype=dtype)
    paired_idx = pair_mask_present.nonzero(as_tuple=False).squeeze(-1)
    pid_lookup = pair_id[paired_idx].clamp(min=0)
    g_per_row[paired_idx] = role_sign[paired_idx] * g.to(dtype)[pid_lookup]

    # Per-token advantage = sign * g / L, broadcast across the supplied mask
    A_dpo = (g_per_row / length).unsqueeze(-1) * mask

    realized_pairs = (s_chosen.abs() + s_rejected.abs() > 0).sum().item()
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


def _build_token_localize_masks(
    *,
    response_mask: torch.Tensor,   # (B, T) base mask
    pair_id: torch.Tensor,         # (B,) long
    pair_role: torch.Tensor,       # (B,) long
    t_star: torch.Tensor,          # (B,) long, -1 if no t_star (init samples)
) -> tuple[torch.Tensor, torch.Tensor]:
    """① Causal-Localized: split base mask into (mask_token, mask_continuation).

    For paired rows, mask_token has 1 only at t_star and mask_cont has 1 from t_star+1 to end
    (intersected with response_mask). For pair rows where t_star is missing (== -1) or for
    non-paired rows, mask_token = 0 and mask_cont = response_mask (they don't contribute to
    causal-localized advantage; non-paired rows fall to GRPO).

    Important: for chain pairs we use the t_star OF THE CHOSEN sample (which is the t_i that
    produced it from its parent rejected). The same t_star applies to both chosen and rejected
    because they share prefix exactly until t_star.
    """
    B, T = response_mask.shape
    device = response_mask.device
    dtype = response_mask.dtype

    # Resolve per-pair t_star: gather chosen rows' t_star and propagate to rejected partner
    pair_to_tstar = {}
    if (pair_id >= 0).any():
        pair_id_np = pair_id.detach().cpu().numpy()
        pair_role_np = pair_role.detach().cpu().numpy()
        t_star_np = t_star.detach().cpu().numpy()
        for j in range(B):
            if pair_role_np[j] == +1 and pair_id_np[j] >= 0 and t_star_np[j] >= 0:
                pair_to_tstar[int(pair_id_np[j])] = int(t_star_np[j])

    mask_token = torch.zeros(B, T, dtype=dtype, device=device)
    mask_cont = response_mask.clone()
    if not pair_to_tstar:
        return mask_token, mask_cont

    # For each paired row (chosen or rejected), set mask_token at the resolved t_star.
    pair_id_np = pair_id.detach().cpu().numpy()
    for j in range(B):
        pid = int(pair_id_np[j])
        if pid < 0 or pid not in pair_to_tstar:
            continue
        ts = pair_to_tstar[pid]
        if 0 <= ts < T and response_mask[j, ts] > 0:
            mask_token[j, ts] = 1.0
            mask_cont[j, ts] = 0.0  # remove t* from continuation portion

    return mask_token, mask_cont


@register_adv_est("dpo_teacher_guided")
def compute_dpo_tgs_advantage(
    token_level_rewards: torch.Tensor,       # (B, T)
    response_mask: torch.Tensor,             # (B, T) chain shared prefix already masked
    index,                                   # uid (B,) chain samples of one prompt share uid
    rollout_log_probs: Optional[torch.Tensor] = None,   # (B, T) π_old
    ref_log_prob: Optional[torch.Tensor] = None,        # (B, T) π_ref
    dpo_pair_id: Optional[torch.Tensor] = None,         # (B,)   long, -1 = no pair
    dpo_pair_role: Optional[torch.Tensor] = None,       # (B,)   +1/-1/0
    teacher_log_prob_opsd: Optional[torch.Tensor] = None,  # (B, T) ② Teacher-Anchored ref (NaN where missing)
    dpo_t_star: Optional[torch.Tensor] = None,          # (B,) long ① Causal-Localized t* per sample
    config: Optional[dict] = None,
    **_unused,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linearized DPO advantage with GRPO fallback + 3 teacher-guided innovations.

    Innovations (each independently togglable via dpo_cfg):
      ① Causal-Localized DPO  (causal_localize=True, beta_token, beta_continuation)
         Splits margin into token-level (at t*) + continuation-level (t*+1..end), each with
         its own β. Lets us study where the DPO signal actually lives.
      ② Teacher-Anchored ref (use_teacher_anchored_ref=True)
         Replaces π_ref with π_T^OPSD (privileged-context teacher) in the margin computation.
         Per Samplers-DPO ICLR-25 Theorem 4: reward-aware ref ⇒ quadratic convergence.
         Falls back to π_ref per-row when teacher_log_prob_opsd is NaN.
      ③ ΔR-Weighted (delta_r_weight_mode)
         Per-pair scalar weight from |R(chosen) − R(rejected)| magnitude. Emphasizes
         high-Δ pairs (the "decisive" pairs).

    Returns (advantages, returns); returns == advantages (no critic).
    """
    dpo_cfg: dict = {}
    if config is not None:
        dpo_cfg = config.get("dpo", {}) or {}

    beta: float = float(dpo_cfg.get("beta", 0.1))
    alpha: float = float(dpo_cfg.get("alpha", 1.0))
    grpo_norm_std: bool = bool(dpo_cfg.get("norm_adv_by_std_in_grpo", False))
    min_response_length: int = int(dpo_cfg.get("min_response_length", 0))
    length_penalty_type: str = str(dpo_cfg.get("length_penalty_type", "linear"))

    # Innovation knobs (all default OFF for v1 backwards compat)
    use_teacher_anchored_ref: bool = bool(dpo_cfg.get("use_teacher_anchored_ref", False))
    causal_localize: bool = bool(dpo_cfg.get("causal_localize", False))
    beta_token_cfg = dpo_cfg.get("beta_token", None)
    beta_token: float = float(beta_token_cfg) if beta_token_cfg is not None else beta
    beta_cont_cfg = dpo_cfg.get("beta_continuation", None)
    beta_cont: float = float(beta_cont_cfg) if beta_cont_cfg is not None else (beta * 0.5)
    delta_r_weight_mode: str = str(dpo_cfg.get("delta_r_weight_mode", "none"))
    delta_r_weight_tau: float = float(dpo_cfg.get("delta_r_weight_tau", 1.0))

    B, T = token_level_rewards.shape
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype
    mask = response_mask.to(dtype)

    with torch.no_grad():
        # ── Step 1: GRPO baseline ────────────────────────────────────────────
        seq_reward = token_level_rewards.sum(dim=-1)
        A_grpo_seq = _grpo_group_advantage(seq_reward, index, norm_by_std=grpo_norm_std)
        A_grpo = A_grpo_seq.unsqueeze(1).expand_as(token_level_rewards) * mask

        has_pair_info = (
            dpo_pair_id is not None
            and dpo_pair_role is not None
            and rollout_log_probs is not None
            and ref_log_prob is not None
        )

        if not has_pair_info:
            advantages = A_grpo
        else:
            pair_id = dpo_pair_id.to(device).long()
            pair_role = dpo_pair_role.to(device).long()

            # ② Teacher-Anchored ref: replace π_ref with π_T^OPSD when available;
            #    NaN rows fall back to π_ref (per-token blend via NaN-safe substitute).
            if use_teacher_anchored_ref and teacher_log_prob_opsd is not None:
                tlp = teacher_log_prob_opsd.to(device).float()
                ref_full = ref_log_prob.to(device).float()
                # Where tlp is NaN, fall back to ref_log_prob; preserves non-paired rows too.
                anchor_logp = torch.where(torch.isnan(tlp), ref_full, tlp)
            else:
                anchor_logp = ref_log_prob.to(device).float()

            logp_diff_full = (rollout_log_probs.float().to(device) - anchor_logp)

            # ③ ΔR weight: per-pair scalar from R(chosen) − R(rejected).
            pair_weight: Optional[torch.Tensor] = None
            if delta_r_weight_mode != "none":
                if (pair_id >= 0).any():
                    max_pid = int(pair_id.max().item()) + 1
                    chosen_mask = pair_role == +1
                    rejected_mask = pair_role == -1
                    r_chosen = torch.zeros(max_pid, device=device, dtype=torch.float32)
                    r_rejected = torch.zeros(max_pid, device=device, dtype=torch.float32)
                    r_chosen.scatter_add_(0, pair_id.clamp(min=0)[chosen_mask], seq_reward[chosen_mask].float())
                    r_rejected.scatter_add_(0, pair_id.clamp(min=0)[rejected_mask], seq_reward[rejected_mask].float())
                    delta_r = r_chosen - r_rejected
                    pair_weight = _delta_r_weight(delta_r, mode=delta_r_weight_mode, tau=delta_r_weight_tau)

            if not causal_localize:
                # ── Standard linearized DPO (with optional ② or ③ on top) ─────
                seq_logp_diff = (logp_diff_full * mask).sum(dim=-1)
                L = _length_per_row(mask)
                A_dpo, _ = _vectorized_pair_advantage(
                    seq_logp_diff=seq_logp_diff,
                    pair_id=pair_id,
                    pair_role=pair_role,
                    mask=mask,
                    length=L,
                    beta=beta,
                    pair_weight=pair_weight,
                )
            else:
                # ── ① Causal-Localized: split margin into token-level + continuation ──
                if dpo_t_star is None:
                    # Can't localize without t_star → fall back to standard.
                    seq_logp_diff = (logp_diff_full * mask).sum(dim=-1)
                    L = _length_per_row(mask)
                    A_dpo, _ = _vectorized_pair_advantage(
                        seq_logp_diff=seq_logp_diff, pair_id=pair_id, pair_role=pair_role,
                        mask=mask, length=L, beta=beta, pair_weight=pair_weight,
                    )
                else:
                    t_star = dpo_t_star.to(device).long()
                    mask_token, mask_cont = _build_token_localize_masks(
                        response_mask=mask, pair_id=pair_id, pair_role=pair_role, t_star=t_star,
                    )

                    # Compute per-pair token-level margin extra (β_tok-scaled), used inside σ.
                    seq_logp_diff_tok = (logp_diff_full * mask_token).sum(dim=-1)
                    chosen_mask = pair_role == +1
                    rejected_mask = pair_role == -1
                    pair_mask_present = pair_id >= 0
                    if pair_mask_present.any():
                        max_pid = int(pair_id.max().item()) + 1
                        s_tok_c = torch.zeros(max_pid, device=device, dtype=torch.float32)
                        s_tok_r = torch.zeros(max_pid, device=device, dtype=torch.float32)
                        s_tok_c.scatter_add_(0, pair_id.clamp(min=0)[chosen_mask], beta_token * seq_logp_diff_tok[chosen_mask].float())
                        s_tok_r.scatter_add_(0, pair_id.clamp(min=0)[rejected_mask], beta_token * seq_logp_diff_tok[rejected_mask].float())
                        margin_token_per_pair = s_tok_c - s_tok_r
                    else:
                        margin_token_per_pair = None

                    # Continuation advantage (uses β_cont) carries the σ-gating; the token
                    # margin is added as `margin_extra_per_pair` so σ sees the combined margin.
                    seq_logp_diff_cont = (logp_diff_full * mask_cont).sum(dim=-1)
                    L_cont = _length_per_row(mask_cont)
                    A_cont, _ = _vectorized_pair_advantage(
                        seq_logp_diff=seq_logp_diff_cont, pair_id=pair_id, pair_role=pair_role,
                        mask=mask_cont, length=L_cont, beta=beta_cont, pair_weight=pair_weight,
                        margin_extra_per_pair=margin_token_per_pair,
                    )

                    # Token advantage (β_token, length=1) — same σ-gate (margin includes both
                    # components inside σ for self-consistency of gradient direction).
                    L_tok = _length_per_row(mask_token).clamp(min=1.0)
                    # For token component, set seq_logp_diff to 0 (margin already accounted for via extra)
                    A_tok, _ = _vectorized_pair_advantage(
                        seq_logp_diff=torch.zeros(B, device=device, dtype=torch.float32),
                        pair_id=pair_id, pair_role=pair_role,
                        mask=mask_token, length=L_tok, beta=beta_token, pair_weight=pair_weight,
                        margin_extra_per_pair=margin_token_per_pair,
                    )
                    A_dpo = A_tok + A_cont

            # ── Step 3: mix DPO + GRPO ──────────────────────────────────────
            in_pair = (pair_id >= 0).to(dtype).unsqueeze(1)
            advantages = in_pair * (alpha * A_dpo + (1.0 - alpha) * A_grpo) \
                       + (1.0 - in_pair) * A_grpo

        # ── Step 4: length floor ─────────────────────────────────────────────
        if min_response_length > 0:
            L_eff = mask.sum(dim=-1).clamp(min=1.0)
            if length_penalty_type == "zero":
                length_scale = (L_eff >= min_response_length).to(dtype)
            else:
                length_scale = (L_eff / float(min_response_length)).clamp(max=1.0).to(dtype)
            advantages = advantages * length_scale.unsqueeze(-1)

    return advantages, advantages
