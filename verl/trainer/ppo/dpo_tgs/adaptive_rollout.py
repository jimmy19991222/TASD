# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""DPO-TGS V2 adaptive rollout.

Different from `tcca_v2_chain_rollout` (which forces chain_length attempts on
EVERY sample): this function does selective intervention on only the failed
samples within a standard GRPO baseline, using SDPO-style sibling-correct
rollouts as teacher reference context (no dataset ground-truth needed).

Pipeline:
  Phase 1: standard rollout n_init per prompt (n=2 or 4, smaller than GRPO baseline)
           → y_init with B*n_init samples
  Phase 2: per-prompt SDPO teacher context:
           - find correct rollouts in y_init (R >= correct_threshold)
           - decode one of them as reference answer
           - build ctx = "Refer to this correct answer: {decoded}\n" + original prompt
           - if no correct sibling: skip prompt (or fallback to GT, per ablation)
  Phase 3: iterative intervention on failed samples (R < correct_threshold):
           For each failed sample:
             For attempt in 1..n_attempts:
               a. teacher fwd under SDPO ctx → divergence
               b. select t* (argmax divergence, exclude tail + used positions)
               c. teacher decode 1 token at t*; enforce z_T != student_token via reselect_t
               d. student continue from prefix
               e. y_attempt = y_prev[:t*] + z_T + continuation
               f. rescore reward; append to chain
  Phase 4: concat y_init + all chain attempts; tag each sample with:
           - dpo_lineage_id: which y_init sample this chain derives from (-1 for non-derived)
           - dpo_attempt_idx: 0 for y_init, 1..n_attempts for chain derivatives
           pair_collector uses (uid, lineage_id, attempt_idx) to form chain_consecutive pairs.

Design doc: research/dpo_teacher_guided_sampling.md
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from verl.protocol import DataProto

# Reuse tcca_chain helpers
from verl.trainer.ppo.bayesian_credit.tcca_chain import (
    _standard_rollout,
    _compute_divergence_opsd,
    _select_t_i,
    _teacher_write_one_token,
    _student_continue_async,
    _build_y_i_batch,
    _rescore_reward,
    _build_opsd_teacher_context,  # fallback when sdpo_ctx_source='gt' or no correct sibling
    compute_position_id_with_mask,
)


def dpo_tgs_adaptive_rollout(
    prompt_batch: DataProto,
    actor_rollout_wg,
    async_rollout_manager,
    reward_fn,
    config,
    tokenizer,
) -> DataProto:
    """Main entry. Replaces tcca_v2_chain_rollout for adv_estimator=dpo_teacher_guided."""
    dpo_cfg = config.algorithm.get("dpo", {}) or {}
    n_init: int = int(dpo_cfg.get("n_init", 2))
    n_attempts: int = int(dpo_cfg.get("n_attempts", 2))
    correct_threshold: float = float(dpo_cfg.get("correct_threshold", 1.0))
    sdpo_ctx_source: str = str(dpo_cfg.get("sdpo_ctx_source", "sibling_correct"))
    all_failed_strategy: str = str(dpo_cfg.get("all_failed_strategy", "skip"))
    sdpo_ref_template: str = str(dpo_cfg.get(
        "sdpo_ref_template", "Refer to this correct answer: {r}\n"
    ))
    exclude_tail: int = int(dpo_cfg.get("exclude_tail_tokens", 8))
    max_reselect: int = int(dpo_cfg.get("max_reselect_attempts", 3))

    # ── Phase 1: standard rollout ─────────────────────────────────────────
    y_init = _standard_rollout(prompt_batch, n=n_init, async_rollout_manager=async_rollout_manager)
    if "token_level_rewards" not in y_init.batch:
        y_init = _rescore_reward(y_init, reward_fn)

    B_init = y_init.batch.batch_size[0]
    T = y_init.batch["responses"].shape[1]
    device = y_init.batch["responses"].device

    # Lineage tagging on y_init
    lineage_id_init = np.arange(B_init, dtype=np.int64)
    attempt_idx_init = np.zeros(B_init, dtype=np.int64)
    y_init.non_tensor_batch["dpo_lineage_id"] = lineage_id_init
    y_init.non_tensor_batch["dpo_attempt_idx"] = attempt_idx_init
    # ① Causal-Localized: t_star tag (-1 = init sample, no intervention point)
    y_init.non_tensor_batch["dpo_t_star"] = np.full(B_init, -1, dtype=np.int64)

    # ── Phase 2: per-prompt SDPO contexts ─────────────────────────────────
    R_init = y_init.batch["token_level_rewards"].sum(dim=-1).cpu().numpy()
    uids_init = y_init.non_tensor_batch["uid"]
    uid_to_correct_idx, prompts_with_no_correct = _group_correct_by_uid(
        uids_init, R_init, correct_threshold
    )

    # Build per-uid SDPO ctx (either sibling-correct or GT fallback)
    sdpo_ctx_by_uid: dict = _build_sdpo_ctx_by_uid(
        y_init=y_init,
        uid_to_correct_idx=uid_to_correct_idx,
        prompts_with_no_correct=prompts_with_no_correct,
        all_failed_strategy=all_failed_strategy,
        sdpo_ctx_source=sdpo_ctx_source,
        sdpo_ref_template=sdpo_ref_template,
        tokenizer=tokenizer,
        config=config,
        device=device,
    )

    # ── Phase 3: select failed samples that have a valid ctx ─────────────
    failed_indices = []
    for j in range(B_init):
        uid = uids_init[j]
        if R_init[j] < correct_threshold and uid in sdpo_ctx_by_uid:
            failed_indices.append(j)

    if not failed_indices:
        # No intervention possible; return y_init as-is.
        return y_init

    failed_indices_arr = np.array(failed_indices, dtype=np.int64)
    failed_batch = y_init[failed_indices_arr]
    # Carry lineage tags forward
    failed_batch.non_tensor_batch["dpo_lineage_id"] = lineage_id_init[failed_indices_arr].copy()
    failed_batch.non_tensor_batch["dpo_attempt_idx"] = np.zeros(len(failed_indices_arr), dtype=np.int64)

    chain_batches: list[DataProto] = []
    cur = failed_batch
    used_positions: list[set] = [set() for _ in range(len(failed_indices_arr))]

    for attempt_k in range(1, n_attempts + 1):
        # SDPO ctx for current sub-batch (gather per uid)
        sdpo_ctx = _gather_sdpo_ctx_for_sub_batch(cur, sdpo_ctx_by_uid)

        # Step B+C: divergence under SDPO ctx
        div = _compute_divergence_opsd(cur, sdpo_ctx, actor_rollout_wg, config, tokenizer)

        # Step D + E with mismatch reselect: t_i and z_T jointly
        t_i, z_T = _select_t_and_token_with_mismatch(
            y_prev=cur,
            sdpo_ctx=sdpo_ctx,
            divergence=div,
            actor_rollout_wg=actor_rollout_wg,
            config=config,
            tokenizer=tokenizer,
            used_positions=used_positions,
            exclude_tail=exclude_tail,
            max_reselect=max_reselect,
        )

        # Step F+G: student continue
        cont = _student_continue_async(cur, t_i, z_T, async_rollout_manager, config, tokenizer)

        # Step H: build y_attempt
        y_attempt = _build_y_i_batch(cur, t_i, z_T, cont, tokenizer)

        # Step I: rescore
        y_attempt = _rescore_reward(y_attempt, reward_fn)

        # Tag lineage (preserve uid, lineage_id; bump attempt_idx)
        y_attempt.non_tensor_batch["uid"] = cur.non_tensor_batch["uid"].copy()
        y_attempt.non_tensor_batch["dpo_lineage_id"] = cur.non_tensor_batch["dpo_lineage_id"].copy()
        y_attempt.non_tensor_batch["dpo_attempt_idx"] = np.full(
            len(failed_indices_arr), attempt_k, dtype=np.int64
        )
        # ① Causal-Localized: persist the divergence position used to produce this attempt
        y_attempt.non_tensor_batch["dpo_t_star"] = t_i.detach().cpu().numpy().astype(np.int64)

        chain_batches.append(y_attempt)

        # Update used_positions for each sample
        t_i_np = t_i.detach().cpu().numpy()
        for j in range(len(failed_indices_arr)):
            used_positions[j].add(int(t_i_np[j]))

        cur = y_attempt  # next attempt builds on this

    # ── Phase 4: concat y_init + chain attempts ──────────────────────────
    augmented = DataProto.concat([y_init] + chain_batches)

    # ── Phase 4b (optional): post-hoc OPSD teacher fwd for Teacher-Anchored DPO ──
    # ② Teacher-Anchored: compute logp under OPSD ctx for every sample in augmented batch.
    # Adds 1 teacher fwd per chain rollout; only runs when use_teacher_anchored_ref=True.
    if bool(dpo_cfg.get("use_teacher_anchored_ref", False)):
        try:
            tlp = _post_hoc_opsd_teacher_logp(
                augmented=augmented,
                sdpo_ctx_by_uid=sdpo_ctx_by_uid,
                actor_rollout_wg=actor_rollout_wg,
                config=config,
                tokenizer=tokenizer,
            )
            # tlp: (B_aug, T) float, NaN where no SDPO ctx (uid not in sdpo_ctx_by_uid).
            augmented.batch["teacher_log_prob_opsd"] = tlp
        except Exception as e:
            print(f"[DPO-TGS] post-hoc OPSD teacher fwd failed: {e}; "
                  "Teacher-Anchored DPO will gracefully fall back to π_ref.")

    return augmented


def _post_hoc_opsd_teacher_logp(
    *,
    augmented: DataProto,
    sdpo_ctx_by_uid: dict,
    actor_rollout_wg,
    config,
    tokenizer,
) -> torch.Tensor:
    """Compute logp_T under OPSD ctx for every sample in `augmented`.

    Samples whose uid is not in sdpo_ctx_by_uid (e.g. all-failed prompts where SDPO
    ctx couldn't be built) get NaN — the dpo_loss layer will treat NaN as missing
    and fall back to π_ref for those rows.
    """
    B = augmented.batch.batch_size[0]
    T = augmented.batch["responses"].shape[1]
    device = augmented.batch["responses"].device
    out = torch.full((B, T), float("nan"), dtype=torch.float32, device=device)

    # Filter to rows that have a valid SDPO ctx
    uids = augmented.non_tensor_batch["uid"]
    valid_idx = [j for j in range(B) if uids[j] in sdpo_ctx_by_uid]
    if not valid_idx:
        return out

    valid_arr = np.array(valid_idx, dtype=np.int64)
    sub = augmented[valid_arr]
    # Gather per-row OPSD ctx
    sub_ctx = _gather_sdpo_ctx_for_sub_batch(sub, sdpo_ctx_by_uid)

    # One teacher fwd over the whole sub (re-uses same machinery as divergence step)
    teacher_input_ids = torch.cat([sub_ctx["input_ids"], sub.batch["responses"]], dim=1)
    teacher_attn = torch.cat([sub_ctx["attention_mask"], sub.batch["response_mask"]], dim=1)
    teacher_pos = compute_position_id_with_mask(teacher_attn)
    teacher_fwd_batch = DataProto.from_dict(tensors={
        "teacher_input_ids": teacher_input_ids,
        "teacher_attention_mask": teacher_attn,
        "teacher_position_ids": teacher_pos,
        "responses": sub.batch["responses"],
        "input_ids": sub.batch["input_ids"],
        "attention_mask": sub.batch["attention_mask"],
        "position_ids": sub.batch["position_ids"],
    })
    teacher_fwd_batch.meta_info = {
        "temperature": float(config.actor_rollout_ref.rollout.temperature),
        "micro_batch_size": int(config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu),
        "pad_token_id": tokenizer.pad_token_id,
        "distill_topk": None,
        "compute_prior_shift_surprise": False,
    }
    teacher_result = actor_rollout_wg.compute_teacher_log_probs(teacher_fwd_batch)
    sub_logp_T = teacher_result.batch["teacher_log_probs_on_response"].float()  # (n_valid, T)

    out[valid_arr] = sub_logp_T.to(device)
    return out


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _group_correct_by_uid(uids, R, correct_threshold):
    """Return (uid → list of correct-indices, set of uids with no correct)."""
    uid_to_correct: dict = {}
    uid_to_any: dict = {}
    for j, uid in enumerate(uids):
        uid_to_any.setdefault(uid, []).append(j)
        if R[j] >= correct_threshold:
            uid_to_correct.setdefault(uid, []).append(j)
    prompts_with_no_correct = set(uid_to_any.keys()) - set(uid_to_correct.keys())
    return uid_to_correct, prompts_with_no_correct


def _build_sdpo_ctx_by_uid(
    *,
    y_init: DataProto,
    uid_to_correct_idx: dict,
    prompts_with_no_correct: set,
    all_failed_strategy: str,
    sdpo_ctx_source: str,
    sdpo_ref_template: str,
    tokenizer,
    config,
    device,
) -> dict:
    """For each prompt (uid), build the teacher context as a dict {input_ids, attention_mask, position_ids}.

    Source priority (controlled by sdpo_ctx_source + all_failed_strategy):
      - 'sibling_correct': use one decoded correct rollout from the same prompt
      - 'gt': always use dataset GT (legacy OPSD)
    All-failed prompts (no sibling correct):
      - 'skip': uid not added to the dict → caller skips intervention on these
      - 'gt_fallback': uid uses GT-based ctx
    """
    sdpo_ctx_by_uid: dict = {}

    # Group raw_prompt / reward_model by uid (any one sample per uid works since same prompt)
    uids = y_init.non_tensor_batch["uid"]
    raw_prompt = y_init.non_tensor_batch["raw_prompt"]
    reward_model = y_init.non_tensor_batch.get("reward_model", None)

    # First-seen index per uid (use to pull raw_prompt + reward_model)
    seen_uid_idx: dict = {}
    for j, uid in enumerate(uids):
        if uid not in seen_uid_idx:
            seen_uid_idx[uid] = j

    sd_cfg = config.actor_rollout_ref.actor.get("self_distillation", {}) or {}
    max_len = int(sd_cfg.get("max_reprompt_len", 10240))
    try:
        enable_thinking = (
            config.data.apply_chat_template_kwargs.get("enable_thinking", True)
            if config.data.apply_chat_template_kwargs else True
        )
    except Exception:
        enable_thinking = True

    # Build per-uid ref text
    uid_to_ref_text: dict = {}
    for uid in seen_uid_idx.keys():
        ref_text = None
        if sdpo_ctx_source == "sibling_correct":
            corrects = uid_to_correct_idx.get(uid)
            if corrects:
                # Pick first correct sibling
                sibling_idx = corrects[0]
                resp_ids = y_init.batch["responses"][sibling_idx]
                resp_mask = y_init.batch["response_mask"][sibling_idx].bool()
                resp_ids_valid = resp_ids[resp_mask].cpu().tolist()
                decoded = tokenizer.decode(resp_ids_valid, skip_special_tokens=True).strip()
                if decoded:
                    ref_text = sdpo_ref_template.format(r=decoded)
        # Fallback to GT?
        if ref_text is None:
            if uid in prompts_with_no_correct and all_failed_strategy == "skip":
                continue  # skip this uid
            if sdpo_ctx_source == "gt" or all_failed_strategy == "gt_fallback":
                rm_idx = seen_uid_idx[uid]
                rm = reward_model[rm_idx] if reward_model is not None else None
                gt = ""
                if isinstance(rm, dict):
                    gt = str(rm.get("ground_truth", ""))
                elif rm is not None:
                    gt = str(rm)
                if gt:
                    ref_text = sdpo_ref_template.format(r=gt)
            if ref_text is None:
                continue
        uid_to_ref_text[uid] = ref_text

    if not uid_to_ref_text:
        return {}

    # Build chat-template inputs for all uids with valid ref text
    uid_list = list(uid_to_ref_text.keys())
    messages_with_ref = []
    for uid in uid_list:
        idx = seen_uid_idx[uid]
        ref_text = uid_to_ref_text[uid]
        new_msgs = [dict(m) for m in raw_prompt[idx]]
        for k in range(len(new_msgs) - 1, -1, -1):
            if new_msgs[k].get("role") == "user":
                new_msgs[k]["content"] = ref_text + new_msgs[k]["content"]
                break
        messages_with_ref.append(new_msgs)

    opsd = tokenizer.apply_chat_template(
        messages_with_ref,
        tokenize=True, return_tensors="pt", return_dict=True,
        add_generation_prompt=True, padding=True, truncation=True, max_length=max_len,
        enable_thinking=enable_thinking,
    )
    opsd_input_ids = opsd["input_ids"].to(device)
    opsd_attn = opsd["attention_mask"].to(device)
    opsd_pos = compute_position_id_with_mask(opsd_attn)

    for k, uid in enumerate(uid_list):
        sdpo_ctx_by_uid[uid] = {
            "input_ids": opsd_input_ids[k:k + 1],
            "attention_mask": opsd_attn[k:k + 1],
            "position_ids": opsd_pos[k:k + 1],
        }
    return sdpo_ctx_by_uid


def _gather_sdpo_ctx_for_sub_batch(sub_batch: DataProto, sdpo_ctx_by_uid: dict) -> dict:
    """Stack per-uid ctx rows into a (B_sub, P_opsd) batched ctx aligned with sub_batch order.

    Pads ctx rows to the max sequence length across the sub-batch.
    """
    uids = sub_batch.non_tensor_batch["uid"]
    rows_input = []
    rows_attn = []
    rows_pos = []
    max_P = 0
    for uid in uids:
        ctx = sdpo_ctx_by_uid[uid]
        rows_input.append(ctx["input_ids"])
        rows_attn.append(ctx["attention_mask"])
        rows_pos.append(ctx["position_ids"])
        max_P = max(max_P, ctx["input_ids"].shape[1])

    device = rows_input[0].device
    pad_input = []
    pad_attn = []
    pad_pos = []
    for r_input, r_attn, r_pos in zip(rows_input, rows_attn, rows_pos):
        P = r_input.shape[1]
        if P < max_P:
            # Left-pad with zeros (chat tokenizer uses left-padding by default)
            pad_n = max_P - P
            zeros_input = torch.zeros((1, pad_n), dtype=r_input.dtype, device=device)
            zeros_attn = torch.zeros((1, pad_n), dtype=r_attn.dtype, device=device)
            zeros_pos = torch.zeros((1, pad_n), dtype=r_pos.dtype, device=device)
            r_input = torch.cat([zeros_input, r_input], dim=1)
            r_attn = torch.cat([zeros_attn, r_attn], dim=1)
            r_pos = torch.cat([zeros_pos, r_pos], dim=1)
        pad_input.append(r_input)
        pad_attn.append(r_attn)
        pad_pos.append(r_pos)

    return {
        "input_ids": torch.cat(pad_input, dim=0),
        "attention_mask": torch.cat(pad_attn, dim=0),
        "position_ids": compute_position_id_with_mask(torch.cat(pad_attn, dim=0)),
    }


def _select_t_and_token_with_mismatch(
    *,
    y_prev: DataProto,
    sdpo_ctx: dict,
    divergence: torch.Tensor,
    actor_rollout_wg,
    config,
    tokenizer,
    used_positions: list,
    exclude_tail: int,
    max_reselect: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select t* and decode teacher token, enforcing z_T != y_S[t*] via reselect_t.

    If reselect exhausts after `max_reselect` rounds, use the last attempt regardless
    (rare; only happens when divergence is concentrated on positions where teacher
    agrees with student).
    """
    B, T = divergence.shape
    device = divergence.device
    response_mask = y_prev.batch["response_mask"]
    student_resp = y_prev.batch["responses"]   # (B, T)

    cur_div = divergence.clone()
    # Pre-mask already-used positions per sample
    for j in range(B):
        for pos in used_positions[j]:
            if 0 <= pos < T:
                cur_div[j, pos] = 0.0

    final_t = torch.zeros(B, dtype=torch.long, device=device)
    final_z = torch.zeros(B, dtype=torch.long, device=device)
    pending = torch.ones(B, dtype=torch.bool, device=device)

    for _ in range(max_reselect + 1):
        if not bool(pending.any()):
            break
        t_i = _select_t_i(cur_div, response_mask, exclude_tail=exclude_tail)
        z_T = _teacher_write_one_token(y_prev, sdpo_ctx, t_i, actor_rollout_wg, config, tokenizer)
        student_tokens = student_resp.gather(1, t_i.unsqueeze(1)).squeeze(1)
        mismatch = z_T != student_tokens

        commit = pending & mismatch
        final_t = torch.where(commit, t_i, final_t)
        final_z = torch.where(commit, z_T, final_z)
        pending = pending & ~mismatch
        # For still-pending rows, zero-out the just-tried t*
        if bool(pending.any()):
            cur_div.scatter_(1, t_i.unsqueeze(1),
                             torch.zeros((B, 1), device=device, dtype=cur_div.dtype))

    # Fallback: still-pending rows keep last t_i/z_T (no-op intervention, will not produce a pair
    # because R won't change, and pair_collector filters by R_i > R_{i-1} + margin)
    if bool(pending.any()):
        final_t = torch.where(pending, t_i, final_t)
        final_z = torch.where(pending, z_T, final_z)

    return final_t, final_z
