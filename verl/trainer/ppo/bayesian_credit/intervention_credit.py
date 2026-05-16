# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Teacher-Guided Dynamic Intervention advantage (ours, base-agnostic causal layer).

ΔR causal credit can be applied **on top of any base advantage estimator**:

    A_seq[i]_base = base_estimator_seq_advantage(R_i, group)            # 不同 base 不同算法
    A_seq[i]      = A_seq[i]_base + λ·ΔR_i · 𝟙[i is intervention sample]
    A_t           = A_seq[i] · base_token_reweight(t, ...) · length_scale

支持的 base estimator:
  - "grpo"        : A_t = A_seq · response_mask  (uniform, no token reweight)
  - "rlsd"        : A_t = A_seq · clip(exp(sign(A_seq) · (logp_T - logp_S)), 1±ε_w)
                    (RLSD baseline, arXiv:2604.03128v2)
  - "prior_shift" : A_t = A_seq · ĝ_t  where g_t = KL(P_T(·|y_≤t) ‖ P_T(·|y_<t))
                    (Tier 1 自家方法，已知在 sciknoweval 上 v2 best 0.33 < RLSD 0.585)

论文叙事:
    ΔR = R(y_intervened) - R(y_original) 是 base-agnostic 因果证据 layer。
    通过在 GRPO / RLSD 等已知 baseline 上加 ΔR 验证："causal credit 是否能在最强相关性
    baseline 上仍能涨 val_acc"——这是论文真正的 claim。

Mode B (append) 关键性质:
    - 失败样本 y_s (R_s 低)         → A_seq < 0
    - 复合样本 y' (R' > R_s)        → A_seq > 0 + λ·ΔR > 0
    天然形成 contrastive pair on the same prefix。

实现层面:
    - ΔR 通道: ray_trainer 的 intervention_rollout block 写入 batch.batch["intervention_delta_reward"]
    - prior_shift 通道: batch.batch["bc_teacher_prior_shift_surprise"]
    - rlsd 通道:        batch.batch["bc_teacher_log_probs"] + batch.batch["old_log_probs"]
    - 当 enable_intervention=False 时, ΔR 始终为 0, 估计器退化为 pure base estimator
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import torch

from verl.trainer.ppo.core_algos import register_adv_est


def _grpo_baseline_seq_advantage(
    token_level_rewards: torch.Tensor,
    index,
    norm_adv_by_std: bool,
) -> torch.Tensor:
    """GRPO group-relative seq advantage: (R_i - mean_group(R)) / [std]."""
    B = token_level_rewards.shape[0]
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype

    seq_reward = token_level_rewards.sum(dim=-1)
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
    return seq_advantage


def _rlsd_token_reweight(
    A_seq: torch.Tensor,                # (B, T) seq adv broadcast
    teacher_log_probs: torch.Tensor,    # (B, T)
    student_log_probs: torch.Tensor,    # (B, T)
    response_mask: torch.Tensor,        # (B, T)
    eps_w: float,
) -> torch.Tensor:
    """RLSD reweight: clip(exp(sign(A) · (logp_T - logp_S)), 1±ε_w)"""
    delta = (teacher_log_probs - student_log_probs) * response_mask
    sign_A = torch.sign(A_seq)
    weight = torch.exp(sign_A * delta)
    weight = weight.clamp(min=1.0 - eps_w, max=1.0 + eps_w)
    return weight


def _prior_shift_token_reweight(
    teacher_prior_shift_surprise: torch.Tensor,  # (B, T) g_t
    response_mask: torch.Tensor,                 # (B, T)
    eps_norm: float,
    max_ratio: float,
    uniform_fallback: bool,
    renormalize_after_clip: bool,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Prior-Shift reweight: ĝ_t = g_t / mean_t(g_t), clipped."""
    mask = response_mask.to(dtype)
    g = teacher_prior_shift_surprise.to(dtype) * mask
    L = mask.sum(dim=-1).clamp(min=1.0)
    g_mean = g.sum(dim=-1) / L
    degenerate = g_mean < eps_norm

    denom = g_mean.clamp(min=eps_norm).unsqueeze(-1)
    g_hat = g / denom
    g_hat = g_hat.clamp(max=max_ratio)

    if renormalize_after_clip:
        g_hat_mean = (g_hat * mask).sum(dim=-1, keepdim=True) / L.unsqueeze(-1).clamp(min=1.0)
        g_hat = g_hat / g_hat_mean.clamp(min=eps_norm)
        g_hat = g_hat * mask

    if uniform_fallback and degenerate.any():
        uniform = mask.clone()
        g_hat = torch.where(degenerate.unsqueeze(-1), uniform, g_hat)

    return g_hat * mask


@register_adv_est("intervention_credit")
def compute_intervention_credit_advantage(
    token_level_rewards: torch.Tensor,                            # (B, T)
    response_mask: torch.Tensor,                                  # (B, T)
    index,                                                        # uid array, len B
    teacher_log_probs: Optional[torch.Tensor] = None,             # (B, T)  for base=rlsd
    student_log_probs: Optional[torch.Tensor] = None,             # (B, T)  for base=rlsd (= old_log_probs)
    teacher_prior_shift_surprise: Optional[torch.Tensor] = None,  # (B, T)  for base=prior_shift
    intervention_delta_reward: Optional[torch.Tensor] = None,     # (B,)    ΔR (legacy seq-level)
    intervention_used: Optional[torch.Tensor] = None,             # (B,)    bool/float
    token_causal_credit: Optional[torch.Tensor] = None,           # (B, T)  TCCA per-token causal credit c_t
    config: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """**TCCA (Token-level Causal Credit Assignment)** — base-agnostic causal token-level credit.

    流程:
      1. base seq advantage A_seq_base = grpo_seq_advantage(R, group)
      2. ΔR seq 级注入 (legacy 兼容, 默认 λ_seq=0): A_seq = A_seq_base + λ_seq · ΔR · 𝟙[intervention]
      3. base token reweight: weight_t (依 base_estimator)
      4. **TCCA: per-token causal credit modulation**: weight_t' = weight_t · (1 + λ_token · c_t_norm)
      5. 最终: A_t = A_seq · weight_t' · length_scale(L)

    c_t (token_causal_credit) 由 intervention_rollout 写入:
      - For original samples y: c_t = -ΔR_{t_k} at intervention positions (penalty for wrong tokens)
      - For composite samples y'_{t_k}: c_t = +ΔR_{t_k} at teacher-replaced positions (reward correct)
      - Else: c_t = 0

    若 c_t 缺失或全 0: TCCA 退化为旧 TGDI (Mode B append + λ_seq·ΔR)

    Args:
        token_level_rewards: (B, T) rule-based reward。
        response_mask: (B, T) bool/0-1 mask。
        index: list[str|int]，长度 B，uid 标识同 prompt 组。
        teacher_log_probs: (B, T) base=rlsd 必需（teacher forward 算出）。
        student_log_probs: (B, T) base=rlsd 必需（= old_log_probs）。
        teacher_prior_shift_surprise: (B, T) base=prior_shift 必需。
        intervention_delta_reward: (B,) ΔR；非 intervention 样本应为 0。None 等价无因果注入。
        intervention_used: (B,) bool/float，标识 composite rollout 样本。
        config: AlgoConfig dict-like，读取 algorithm.intervention_credit.{...}。

    Returns:
        advantages: (B, T) per-token advantage
        returns:    (B, T) 与 advantages 相同（无 critic）
    """
    ic_cfg: dict = {}
    if config is not None:
        ic_cfg = config.get("intervention_credit", {}) or {}

    base_estimator: str = str(ic_cfg.get("base_estimator", "grpo")).lower()
    # legacy seq-level ΔR injection (TCCA 升级后默认关掉，避免 double counting，
    # 因为 c_t 已经把因果信号注入 token 级了)
    lambda_delta_r: float = float(ic_cfg.get("lambda_delta_r", 0.0))
    # TCCA 核心: per-token causal credit modulation strength
    lambda_token_credit: float = float(ic_cfg.get("lambda_token_credit", 1.0))
    # c_t 归一化的 magnitude clip (防止极端 ΔR 单点爆炸)
    token_credit_clip: float = float(ic_cfg.get("token_credit_clip", 2.0))
    norm_adv_by_std: bool = bool(ic_cfg.get("norm_adv_by_std_in_grpo", False))

    min_response_length: int = int(ic_cfg.get("min_response_length", 50))
    length_penalty_type: str = str(ic_cfg.get("length_penalty_type", "linear"))

    # base-specific configs
    rlsd_eps_w: float = float(ic_cfg.get("rlsd_eps_w", 0.2))
    g_t_cfg: dict = ic_cfg.get("g_t", {}) or {}
    eps_norm: float = float(g_t_cfg.get("eps_norm", 1e-6))
    max_ratio: float = float(g_t_cfg.get("max_ratio", 3.0))
    uniform_fallback: bool = bool(g_t_cfg.get("uniform_fallback", True))
    renormalize_after_clip: bool = bool(g_t_cfg.get("renormalize_after_clip", True))

    B, T = token_level_rewards.shape
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype

    with torch.no_grad():
        mask = response_mask.to(dtype)

        # ── Step 1: base seq-level advantage ─────────────────────────────
        seq_advantage = _grpo_baseline_seq_advantage(token_level_rewards, index, norm_adv_by_std)

        # ── Step 2: ΔR causal injection ──────────────────────────────────
        if intervention_delta_reward is not None:
            delta_r = intervention_delta_reward.to(device=device, dtype=dtype)  # (B,)
            if intervention_used is not None:
                used = intervention_used.to(device=device, dtype=dtype)
            else:
                used = (delta_r.abs() > 1e-12).to(dtype)
            seq_advantage = seq_advantage + lambda_delta_r * delta_r * used

        A_seq = seq_advantage.unsqueeze(1).expand_as(token_level_rewards)  # (B, T)

        # ── Step 3: base token reweight ──────────────────────────────────
        if base_estimator == "grpo":
            # uniform: no extra reweight, just response_mask
            weight = mask
        elif base_estimator == "rlsd":
            if teacher_log_probs is None or student_log_probs is None:
                raise ValueError(
                    "intervention_credit with base_estimator=rlsd requires teacher_log_probs "
                    "and student_log_probs in batch (set bc_teacher_log_probs and old_log_probs)."
                )
            weight = _rlsd_token_reweight(
                A_seq=A_seq,
                teacher_log_probs=teacher_log_probs.to(dtype),
                student_log_probs=student_log_probs.to(dtype),
                response_mask=mask,
                eps_w=rlsd_eps_w,
            )
            weight = weight * mask
        elif base_estimator == "prior_shift":
            if teacher_prior_shift_surprise is None:
                # 退化：当 g_t 缺失时，behave like grpo
                weight = mask
            else:
                weight = _prior_shift_token_reweight(
                    teacher_prior_shift_surprise=teacher_prior_shift_surprise,
                    response_mask=response_mask,
                    eps_norm=eps_norm,
                    max_ratio=max_ratio,
                    uniform_fallback=uniform_fallback,
                    renormalize_after_clip=renormalize_after_clip,
                    dtype=dtype,
                )
        else:
            raise ValueError(f"Unknown base_estimator: {base_estimator}. Choose grpo|rlsd|prior_shift.")

        # ── Step 3.5 (TCCA core): per-token causal credit modulation ──
        # weight_t' = weight_t · (1 + λ_token · clip(c_t, ±token_credit_clip))
        # c_t > 0 (composite at teacher position 且 ΔR>0): boost weight (奖励正确选择)
        # c_t < 0 (original sample 同位置): reduce/flip weight (惩罚错误选择)
        if token_causal_credit is not None and lambda_token_credit > 0.0:
            c_t = token_causal_credit.to(device=device, dtype=dtype)  # (B, T)
            c_t = c_t.clamp(min=-token_credit_clip, max=token_credit_clip)
            tcca_factor = 1.0 + lambda_token_credit * c_t            # (B, T)
            # 保证 weight non-negative (避免符号翻转干扰 PPO)
            tcca_factor = tcca_factor.clamp(min=0.0)
            weight = weight * tcca_factor * mask                     # 再 mask 一遍防 leak

        # ── Step 4: 合成 token-level advantage ────────────────────────────
        advantages = A_seq * weight * mask

        # ── Step 5: length floor penalty (复用 prior_shift v2 防护) ─────────
        if min_response_length > 0:
            L_eff = mask.sum(dim=-1).clamp(min=1.0)
            length_ratio = (L_eff / float(min_response_length)).clamp(max=1.0)
            if length_penalty_type == "zero":
                length_scale = (L_eff >= min_response_length).to(dtype)
            else:  # linear
                length_scale = length_ratio.to(dtype)
            advantages = advantages * length_scale.unsqueeze(-1)

    return advantages, advantages
