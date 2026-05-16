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
    """完整 v3-full 路径（Phase 2 实现）。

    需要的工程改动:
      1. dp_actor 增加 `teacher_generate_at_positions(prefixes, k)` 接口
         - 在 teacher_module 上做 k 步 forward + argmax/采样（k=2 串行可接受）
         - prefixes 是 per-sample 的 prompt + y_<t* 拼接
      2. 调用 async_rollout_manager.generate_sequences(prefix_batch) 续 tail
         - prefix_batch 是 DataProto，input_ids = prompt + y_<t* + intervention
      3. 拼接 composite responses，pad 到原 max_response_length
      4. 构造与 batch 同 schema 的 augmented_batch (含 uid 复用，data_source/extra_info 等 copy)
      5. self.reward_fn(augmented_batch) 得 R(y')
      6. ΔR = R(y') - R(y_s)，写入 augmented_batch.batch["intervention_delta_reward"]
      7. DataProto.concat([batch, augmented_batch])
      8. 标记 intervention_used: 原样本=False，新样本=True

    工程量预估: 单文件 +250 行 + dp_actor +30 行 + worker dispatch +20 行
    """
    raise NotImplementedError(
        "Real intervention rollout is not yet implemented (Phase 2 follow-up). "
        "Set algorithm.intervention_credit.enable_intervention=False to run "
        "in plumbing-validation mode (degenerates to prior_shift-equivalent)."
    )


# ──────────────────────────────────────────────────────────────────────
# 工具：失败样本检测（公开给 ray_trainer 用于度量）
# ──────────────────────────────────────────────────────────────────────


def detect_failed_samples(token_level_rewards: torch.Tensor, failed_threshold: float) -> torch.Tensor:
    """(B, T) → (B,) bool, True for samples with seq-sum reward below threshold."""
    seq_reward = token_level_rewards.sum(dim=-1)
    return seq_reward < failed_threshold
