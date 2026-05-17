# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""DPO-TGS pair collection.

Inputs: an augmented batch produced by `tcca_v2_chain_rollout` containing
B*chain_length rows, with same-uid groups of chain samples (y_0, y_1, ..., y_{n-1}).

Output: per-sample tensors written to `batch.batch`:
    dpo_pair_id      (B*n,) long  pair index (>=0) or -1 if this sample is not in any pair
    dpo_pair_role    (B*n,) long  +1 if chosen, -1 if rejected, 0 if not in any pair

A "pair" is (chosen=y_i, rejected=y_{i-1}) wherever R(y_i) - R(y_{i-1}) > margin.
Each sample may appear in at most one pair (the first chain-edge that promotes it).
The role tensor lets the advantage estimator look up its partner via dpo_pair_id.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import torch


def _extract_rewards_uids_lineage(batch):
    """Common extractor: (rewards_np, uids, lineage_ids, attempt_idx) from a batch.

    Returns dpo_lineage_id / dpo_attempt_idx as None when adaptive_rollout was not
    used (e.g. legacy tcca_v2_chain path). Caller must then fall back to uid-order
    grouping.
    """
    rewards = batch.batch["token_level_rewards"].sum(dim=-1)
    if isinstance(rewards, torch.Tensor):
        rewards_np = rewards.detach().float().cpu().numpy()
    else:
        rewards_np = np.asarray(rewards, dtype=np.float64)

    uids = batch.non_tensor_batch["uid"]
    if not isinstance(uids, np.ndarray):
        uids = np.array(uids, dtype=object)

    lineage = batch.non_tensor_batch.get("dpo_lineage_id")
    attempt = batch.non_tensor_batch.get("dpo_attempt_idx")
    if lineage is not None and not isinstance(lineage, np.ndarray):
        lineage = np.array(lineage, dtype=np.int64)
    if attempt is not None and not isinstance(attempt, np.ndarray):
        attempt = np.array(attempt, dtype=np.int64)
    return rewards_np, uids, lineage, attempt


def collect_chain_consecutive_pairs(
    batch,
    *,
    margin: float = 0.0,
) -> dict:
    """Build chain-consecutive pair indices from a chain-rollout augmented batch.

    If `dpo_lineage_id` / `dpo_attempt_idx` are present (set by adaptive_rollout.py),
    group samples by (uid, lineage_id) and sort by attempt_idx — this correctly
    handles the new pipeline where one prompt has n_init originals + per-failed chain
    derivatives. Otherwise fall back to uid-order (legacy tcca_v2_chain_rollout).

    Returns:
        dict with pair_id, pair_role, num_pairs, win_rate.
    """
    rewards_np, uids, lineage, attempt = _extract_rewards_uids_lineage(batch)
    N = rewards_np.shape[0]
    pair_id = np.full(N, -1, dtype=np.int64)
    pair_role = np.zeros(N, dtype=np.int64)
    used = np.zeros(N, dtype=bool)
    next_pair = 0
    chain_edges_total = 0

    # Group: legacy (uid only) or lineage-aware (uid, lineage_id)
    chains: dict = defaultdict(list)
    if lineage is None or attempt is None:
        for i in range(N):
            chains[uids[i]].append((i, i))   # second entry is implicit order
    else:
        for i in range(N):
            chains[(uids[i], int(lineage[i]))].append((i, int(attempt[i])))

    for _, items in chains.items():
        # Sort by second tuple element (attempt_idx or insertion order)
        items_sorted = sorted(items, key=lambda t: t[1])
        if len(items_sorted) < 2:
            continue
        for i in range(1, len(items_sorted)):
            chain_edges_total += 1
            j_chosen = items_sorted[i][0]
            j_rejected = items_sorted[i - 1][0]
            if used[j_chosen] or used[j_rejected]:
                continue
            gap = rewards_np[j_chosen] - rewards_np[j_rejected]
            if gap > margin:
                pair_id[j_chosen] = next_pair
                pair_id[j_rejected] = next_pair
                pair_role[j_chosen] = +1
                pair_role[j_rejected] = -1
                used[j_chosen] = True
                used[j_rejected] = True
                next_pair += 1

    win_rate = (next_pair / chain_edges_total) if chain_edges_total > 0 else 0.0
    return {
        "pair_id": pair_id,
        "pair_role": pair_role,
        "num_pairs": int(next_pair),
        "win_rate": float(win_rate),
    }


def collect_hybrid_init_chain_pairs(
    batch,
    *,
    margin: float = 0.0,
    correct_threshold: float = 1.0,
) -> dict:
    """Combine init-pool best-vs-worst pairs with chain_consecutive pairs.

    Init-pool pair (per prompt): pair the highest-reward init sample (correct) with
    the lowest-reward init sample (failed) iff their R-gap exceeds margin AND the
    correct one has R >= correct_threshold. Skips all-correct / all-failed prompts.
    Inspired by Meta's "Bridging Offline-Online" recipe + GRPO baseline.

    Chain pair: existing chain_consecutive on chain attempts (lineage-aware).

    A sample can participate in at most one pair (chain takes precedence if both
    would include the same sample).
    """
    rewards_np, uids, lineage, attempt = _extract_rewards_uids_lineage(batch)
    N = rewards_np.shape[0]
    pair_id = np.full(N, -1, dtype=np.int64)
    pair_role = np.zeros(N, dtype=np.int64)
    used = np.zeros(N, dtype=bool)
    next_pair = 0
    chain_edges_total = 0
    init_edges_total = 0

    # ── Pass 1: chain_consecutive pairs (lineage-aware) ─────────────────
    if lineage is not None and attempt is not None:
        chains: dict = defaultdict(list)
        for i in range(N):
            chains[(uids[i], int(lineage[i]))].append((i, int(attempt[i])))
        for _, items in chains.items():
            items_sorted = sorted(items, key=lambda t: t[1])
            if len(items_sorted) < 2:
                continue
            for i in range(1, len(items_sorted)):
                chain_edges_total += 1
                j_chosen = items_sorted[i][0]
                j_rejected = items_sorted[i - 1][0]
                if used[j_chosen] or used[j_rejected]:
                    continue
                if rewards_np[j_chosen] - rewards_np[j_rejected] > margin:
                    pair_id[j_chosen] = next_pair
                    pair_id[j_rejected] = next_pair
                    pair_role[j_chosen] = +1
                    pair_role[j_rejected] = -1
                    used[j_chosen] = used[j_rejected] = True
                    next_pair += 1

    # ── Pass 2: init-pool best-vs-worst pair (per prompt) ──────────────
    # Restrict to attempt_idx == 0 (init samples) iff lineage present.
    init_by_uid: dict = defaultdict(list)
    for i in range(N):
        if attempt is None or int(attempt[i]) == 0:
            init_by_uid[uids[i]].append(i)

    for _, idxs in init_by_uid.items():
        if len(idxs) < 2:
            continue
        init_edges_total += 1
        # best (highest R) and worst (lowest R)
        idxs_sorted = sorted(idxs, key=lambda j: rewards_np[j])
        j_rejected = idxs_sorted[0]
        j_chosen = idxs_sorted[-1]
        if used[j_chosen] or used[j_rejected]:
            continue
        if rewards_np[j_chosen] < correct_threshold:
            continue  # no correct sample in init → skip (Meta-style)
        if rewards_np[j_chosen] - rewards_np[j_rejected] <= margin:
            continue
        pair_id[j_chosen] = next_pair
        pair_id[j_rejected] = next_pair
        pair_role[j_chosen] = +1
        pair_role[j_rejected] = -1
        used[j_chosen] = used[j_rejected] = True
        next_pair += 1

    total_edges = chain_edges_total + init_edges_total
    win_rate = (next_pair / total_edges) if total_edges > 0 else 0.0
    return {
        "pair_id": pair_id,
        "pair_role": pair_role,
        "num_pairs": int(next_pair),
        "win_rate": float(win_rate),
        "chain_edges_total": int(chain_edges_total),
        "init_edges_total": int(init_edges_total),
    }


def write_pair_info_to_batch(batch, pair_info: dict) -> None:
    """Attach pair_id / pair_role tensors to batch.batch in-place."""
    device = batch.batch["token_level_rewards"].device
    batch.batch["dpo_pair_id"] = torch.from_numpy(pair_info["pair_id"]).to(device)
    batch.batch["dpo_pair_role"] = torch.from_numpy(pair_info["pair_role"]).to(device)


def collect_dpo_pairs(
    batch,
    *,
    strategy: str = "chain_consecutive",
    margin: float = 0.0,
    correct_threshold: float = 1.0,
) -> dict:
    """Public entry. Dispatches by strategy and writes pair info to batch in-place."""
    if strategy == "chain_consecutive":
        info = collect_chain_consecutive_pairs(batch, margin=margin)
    elif strategy == "hybrid_init_chain":
        info = collect_hybrid_init_chain_pairs(
            batch, margin=margin, correct_threshold=correct_threshold
        )
    else:
        raise NotImplementedError(f"Pair strategy {strategy!r} not implemented (yet).")
    write_pair_info_to_batch(batch, info)
    return info


# Compatibility shim for the existing dpo_tgs/__init__.py stub that imports DPOPairCollector
class DPOPairCollector:
    """Thin wrapper kept for API compatibility with the original design doc."""

    def __init__(self, strategy: str = "chain_consecutive", margin: float = 0.0):
        self.strategy = strategy
        self.margin = margin

    def __call__(self, batch) -> dict:
        return collect_dpo_pairs(batch, strategy=self.strategy, margin=self.margin)


def compute_dpo_metrics(
    batch,
    pair_info: dict,
    beta: float,
    correct_threshold: float = 1.0,
) -> dict:
    """Compute diagnostic metrics for DPO-TGS after pair collection + ref log_prob.

    Expected on batch.batch:
        token_level_rewards   (B*n, T)
        response_mask         (B*n, T)
        rollout_log_probs     (B*n, T)   π_old
        ref_log_prob          (B*n, T)   π_ref
        dpo_pair_id           (B*n,)
        dpo_pair_role         (B*n,)
    Optional on batch.non_tensor_batch:
        dpo_lineage_id, dpo_attempt_idx   (for chain rollout health metrics)
        uid                               (for prompt-level metrics)

    Metric categories:
      pair quality (P0):  pairs_total, pair_win_rate, reward_gap_mean, length_ratio
      DPO loss (P0):      margin_mean, margin_pos_rate, sigma_neg_margin_mean,
                          implicit_reward_accuracy, kl_to_ref_chosen_mean
      chain health (P1):  prompts_with_no_correct_pct, correct_per_prompt_init_mean,
                          chain_attempt_success_rate_at_k, chain_vs_init_pair_ratio
    """
    out: dict = {
        "dpo/pairs_total": int(pair_info.get("num_pairs", 0)),
        "dpo/pair_win_rate": float(pair_info.get("win_rate", 0.0)),
    }
    # hybrid pair source split (if available)
    if "chain_edges_total" in pair_info or "init_edges_total" in pair_info:
        ch_edges = int(pair_info.get("chain_edges_total", 0))
        init_edges = int(pair_info.get("init_edges_total", 0))
        out["dpo/chain_edges_total"] = ch_edges
        out["dpo/init_edges_total"] = init_edges
        # Pair count split is not directly returned by collector; approximate via edges ratio
        if ch_edges + init_edges > 0:
            out["dpo/chain_vs_init_pair_ratio"] = float(ch_edges / max(1, init_edges))

    if "dpo_pair_id" not in batch.batch or "dpo_pair_role" not in batch.batch:
        return out

    pair_id = batch.batch["dpo_pair_id"]
    pair_role = batch.batch["dpo_pair_role"]
    mask = batch.batch["response_mask"].float()
    rewards = batch.batch["token_level_rewards"].sum(dim=-1).float()
    L = mask.sum(dim=-1).clamp(min=1.0)

    chosen_mask = pair_role == +1
    rejected_mask = pair_role == -1

    if chosen_mask.any():
        avg_chosen_len = float(L[chosen_mask].mean().item())
        out["dpo/avg_chosen_length"] = avg_chosen_len
        out["dpo/avg_chosen_reward"] = float(rewards[chosen_mask].mean().item())
    if rejected_mask.any():
        avg_rejected_len = float(L[rejected_mask].mean().item())
        out["dpo/avg_rejected_length"] = avg_rejected_len
        out["dpo/avg_rejected_reward"] = float(rewards[rejected_mask].mean().item())
    if chosen_mask.any() and rejected_mask.any():
        out["dpo/reward_gap_mean"] = float(
            rewards[chosen_mask].mean().item() - rewards[rejected_mask].mean().item()
        )
        # P0: length_ratio (OAIF length bias direct quantification)
        denom = avg_rejected_len if avg_rejected_len > 0 else 1.0
        out["dpo/length_ratio_chosen_over_rejected"] = avg_chosen_len / denom

    # ── DPO loss-side diagnostics (require ref_log_prob) ─────────────────
    has_logp = "rollout_log_probs" in batch.batch and "ref_log_prob" in batch.batch
    if has_logp:
        logp_old = batch.batch["rollout_log_probs"].float()
        logp_ref = batch.batch["ref_log_prob"].float()
        per_token_diff = (logp_old - logp_ref) * mask
        seq_diff = per_token_diff.sum(dim=-1)        # (B,) Σ_t mask·(logπ_old − logπ_ref)
        s_per_row = beta * seq_diff

        # P0: kl_to_ref_chosen_mean (Meta Bridging Offline-Online warning)
        # kl is approximated by Σ_t (logπ_old - logπ_ref); on-policy proxy of KL(π_old || π_ref)
        if chosen_mask.any():
            out["dpo/kl_to_ref_chosen_mean"] = float(seq_diff[chosen_mask].mean().item())
        if rejected_mask.any():
            out["dpo/kl_to_ref_rejected_mean"] = float(seq_diff[rejected_mask].mean().item())

        # Per-pair: margin, sigma(-margin), implicit reward correctness
        import numpy as _np
        margins = []
        for pid in pair_id.unique().tolist():
            if pid < 0:
                continue
            rows = (pair_id == pid).nonzero(as_tuple=False).squeeze(-1)
            roles = pair_role[rows]
            ch = rows[roles == +1]
            rj = rows[roles == -1]
            if ch.numel() == 1 and rj.numel() == 1:
                margins.append(float(s_per_row[ch.item()].item() - s_per_row[rj.item()].item()))

        if margins:
            arr = _np.array(margins, dtype=_np.float64)
            out["dpo/margin_mean"] = float(arr.mean())
            out["dpo/margin_std"] = float(arr.std())
            out["dpo/margin_pos_rate"] = float((arr > 0).mean())
            # P0: implicit reward accuracy (DPO field standard)
            # = fraction where DPO's implicit reward r̂_chosen > r̂_rejected (margin > 0).
            # Same as margin_pos_rate but kept under a more standard DPO name.
            out["dpo/implicit_reward_accuracy"] = float((arr > 0).mean())
            # P0: sigma(-margin) = per-pair gradient magnitude (OFS-DPO vanishing warning)
            sigma_neg = 1.0 / (1.0 + _np.exp(arr))   # σ(-margin) numerically stable
            out["dpo/sigma_neg_margin_mean"] = float(sigma_neg.mean())
            out["dpo/sigma_neg_margin_min"] = float(sigma_neg.min())

    out["dpo/grpo_fallback_rate"] = float((pair_id < 0).float().mean().item())

    # ── Innovation diagnostic metrics ────────────────────────────────────
    # ② Teacher-Anchored: how many samples got valid OPSD teacher logp
    if "teacher_log_prob_opsd" in batch.batch:
        tlp = batch.batch["teacher_log_prob_opsd"]
        valid = (~torch.isnan(tlp)).any(dim=-1)
        out["dpo/teacher_anchored_coverage"] = float(valid.float().mean().item())
        # Mean of (logπ_old - logπ_T_opsd) on chosen samples = "how much further teacher
        # would push than ref" (informally: the divergence the teacher-anchored loss closes)
        if has_logp and chosen_mask.any() and valid.any():
            teacher_diff = (logp_actor.float() - tlp.float()) * mask
            row_diff = teacher_diff.sum(dim=-1)
            ch_valid = chosen_mask & valid
            if ch_valid.any():
                out["dpo/teacher_anchored_kl_chosen_mean"] = float(row_diff[ch_valid].mean().item())

    # ③ ΔR-Weighted: per-pair ΔR distribution (independent of weight mode being on)
    if "dpo_pair_id" in batch.batch and chosen_mask.any() and rejected_mask.any():
        # Already exposed as reward_gap_mean above; expose also the std + max for shape
        # of the gap distribution (relevant when picking weight mode).
        rew_chosen_mean = rewards[chosen_mask]
        rew_rejected_mean = rewards[rejected_mask]
        if rew_chosen_mean.numel() > 0 and rew_rejected_mean.numel() > 0:
            # Pair-level ΔR using pair_id matching
            import numpy as _np
            pair_id_np = pair_id.detach().cpu().numpy()
            rewards_np = rewards.detach().cpu().numpy()
            role_np = pair_role.detach().cpu().numpy()
            dr_list = []
            seen_pids = set()
            for j in range(len(pair_id_np)):
                pid = int(pair_id_np[j])
                if pid < 0 or pid in seen_pids:
                    continue
                ch_idx = [i for i in range(len(pair_id_np))
                          if pair_id_np[i] == pid and role_np[i] == +1]
                rj_idx = [i for i in range(len(pair_id_np))
                          if pair_id_np[i] == pid and role_np[i] == -1]
                if ch_idx and rj_idx:
                    dr_list.append(float(rewards_np[ch_idx[0]] - rewards_np[rj_idx[0]]))
                seen_pids.add(pid)
            if dr_list:
                arr = _np.array(dr_list, dtype=_np.float64)
                out["dpo/delta_r_max"] = float(arr.max())
                out["dpo/delta_r_std"] = float(arr.std())

    # ① Causal-Localized: t* coverage + position distribution
    t_star_arr = batch.non_tensor_batch.get("dpo_t_star")
    if t_star_arr is not None:
        if not isinstance(t_star_arr, np.ndarray):
            t_star_arr = np.array(t_star_arr, dtype=np.int64)
        valid_ts = t_star_arr[t_star_arr >= 0]
        if valid_ts.size > 0:
            out["dpo/t_star_mean_position"] = float(valid_ts.mean())
            out["dpo/t_star_std_position"] = float(valid_ts.std())

    # ── P1: chain rollout health (DPO-TGS V2 only) ────────────────────────
    lineage = batch.non_tensor_batch.get("dpo_lineage_id")
    attempt = batch.non_tensor_batch.get("dpo_attempt_idx")
    uids = batch.non_tensor_batch.get("uid")
    if lineage is not None and attempt is not None and uids is not None:
        if not isinstance(lineage, np.ndarray):
            lineage = np.array(lineage, dtype=np.int64)
        if not isinstance(attempt, np.ndarray):
            attempt = np.array(attempt, dtype=np.int64)
        if not isinstance(uids, np.ndarray):
            uids = np.array(uids, dtype=object)

        rewards_np = rewards.detach().cpu().numpy()

        # Init samples = attempt_idx == 0
        is_init = attempt == 0
        init_correct = is_init & (rewards_np >= correct_threshold)

        # prompts_with_no_correct_pct: fraction of unique uids that had ZERO correct in y_init
        uid_has_correct: dict = {}
        for j in range(len(uids)):
            u = uids[j]
            if is_init[j]:
                uid_has_correct[u] = uid_has_correct.get(u, False) or bool(init_correct[j])
        if uid_has_correct:
            no_correct_pct = 1.0 - (sum(uid_has_correct.values()) / len(uid_has_correct))
            out["dpo/prompts_with_no_correct_pct"] = float(no_correct_pct)

        # correct_per_prompt_init_mean: average # correct init samples per prompt
        uid_init_count: dict = {}
        uid_init_correct: dict = {}
        for j in range(len(uids)):
            if is_init[j]:
                u = uids[j]
                uid_init_count[u] = uid_init_count.get(u, 0) + 1
                if rewards_np[j] >= correct_threshold:
                    uid_init_correct[u] = uid_init_correct.get(u, 0) + 1
        if uid_init_count:
            per_prompt_correct = [uid_init_correct.get(u, 0) for u in uid_init_count.keys()]
            out["dpo/correct_per_prompt_init_mean"] = float(np.mean(per_prompt_correct))

        # chain_attempt_success_rate@k: per k in {1..max_attempt}, fraction of (lineage, k) where
        # R(chain[k]) > R(chain[k-1]). Grouped by (uid, lineage_id).
        # First, build per-chain reward lookup
        chain_rewards: dict = {}
        for j in range(len(uids)):
            key = (uids[j], int(lineage[j]))
            chain_rewards.setdefault(key, {})[int(attempt[j])] = float(rewards_np[j])

        # For each attempt k, count successes vs total opportunities
        max_attempt = int(attempt.max()) if attempt.size > 0 else 0
        for k in range(1, max_attempt + 1):
            succ = 0
            tot = 0
            for key, ar_map in chain_rewards.items():
                if k in ar_map and (k - 1) in ar_map:
                    tot += 1
                    if ar_map[k] > ar_map[k - 1]:
                        succ += 1
            if tot > 0:
                out[f"dpo/chain_attempt_success_rate@{k}"] = float(succ / tot)

    return out
