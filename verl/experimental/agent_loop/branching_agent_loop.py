# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Teacher-guided branching rollout — agent loop.

For each prompt we generate a single student trajectory while tracking per-token
entropy. At positions where the entropy spikes (rolling z-score > σ, with σ
adaptively relaxed to hit ``n_splits`` targets), we ask the teacher (privileged-
context same-engine call with ``logprobs=K``) to score student's top-K and pick
its argmax / argmin. Two continuations branch off, each independently finds its
next decision token, etc. After ``n_splits`` binary splits we have 2^n_splits
leaves with shared prefixes.

Architectural note — coordination across the 8 sibling rows:
The verl agent-loop dispatcher creates **one BranchingAgentLoop instance per
sibling row**. Without coordination each row would redundantly generate the
shared prefix. We therefore use a module-level asyncio cache keyed by a
deterministic id derived from the prompt; the first sibling row to start
performs the full branching computation and stores 2^n_splits leaves; all
subsequent rows simply await the cache and pluck their assigned leaf.

Phase 1c.1 shipped the scaffold + coordination layer + a stub pipeline.
Phase 1c.2 (this file post-edit) replaces the stub with the real
generate → spike-detect → teacher-query → branch algorithm.

See research/teacher_branching_rollout.md.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.teacher_prompt import build_marker_text, build_ref_context_messages, build_ref_gt_messages
from verl.workers.rollout.vllm_rollout.branching_utils import (
    BranchNode,
    collect_leaves,
    entropy_from_topk_logprobs,
    find_decision_positions_with_sigma_relaxation,
    pick_teacher_branches,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Module-level coordination across sibling agent-loop instances.
# ---------------------------------------------------------------------------
# Key   : deterministic id derived from (prompt_ids, generation_step, tree_idx).
# Value : asyncio.Future resolving to a list of 2^n_splits AgentLoopOutput
#         objects (one tree's worth of leaves). The first sibling that maps
#         into a given (prompt, tree_idx) creates the future and launches
#         _run_branching_pipeline; remaining siblings of the same tree await
#         and pluck their per-tree leaf by branch_index. Different trees of
#         the same prompt run INDEPENDENT student rollouts (different traj_id
#         + sampling stochasticity), so each (prompt, tree_idx) gets its own
#         owner / Future.
#
# IMPORTANT: This cache lives in the AgentLoopWorker process (one Ray actor).
# A single PPO step's batch is dispatched through one AgentLoopWorker instance
# at a time, so all rollout.n siblings of a uid land in the same dict. Across
# PPO steps the cache grows unboundedly — _branching_cache_clear() is provided
# for the manager to call between steps.

_BRANCHING_CACHE: dict[str, asyncio.Future] = {}
_BRANCHING_CACHE_LOCK = asyncio.Lock()


def _branching_cache_clear() -> None:
    """Clear the module-level coordination cache. Call between PPO steps."""
    _BRANCHING_CACHE.clear()


def _branching_cache_key(prompt_ids: list[int], generation_step: int) -> str:
    """Deterministic BASE cache key shared by all sibling rows of a prompt
    within a step. Tree-level coordination uses ``f"{base}#t{tree_idx}"``.

    The key must collide for sibling rows of the same prompt within a step and
    diverge across steps (so that teacher-prompt drift across PPO updates is
    respected).
    """
    h = hashlib.blake2b(digest_size=16)
    # blake2b doesn't take a list of ints directly; pack to bytes.
    h.update(len(prompt_ids).to_bytes(4, "little"))
    for tok in prompt_ids:
        h.update(int(tok).to_bytes(4, "little", signed=False))
    h.update(int(generation_step).to_bytes(4, "little"))
    return h.hexdigest()


def _tree_cache_key(base_key: str, tree_idx: int) -> str:
    """Per-tree cache key. Each (prompt, tree_idx) pair gets its own owner."""
    return f"{base_key}#t{int(tree_idx)}"


# ---------------------------------------------------------------------------
# Branch-index assignment.
# ---------------------------------------------------------------------------
# Each sibling row needs to know which (tree_idx, leaf_idx_in_tree) it should
# return. We support two assignment modes:
#   - kwargs["branch_index"]: caller-provided GLOBAL slot in
#     [0, n_trees * 2^n_splits). Decomposes to tree_idx = slot // n_per_tree
#     and leaf_idx_in_tree = slot % n_per_tree.
#   - per-uid round-robin counter: when not provided, we derive the global
#     slot from a module-level monotonic counter keyed by the BASE cache key.
#     The first ``n_total`` sibling rows for a given prompt are assigned
#     slots 0..n_total-1 in arrival order.

_BRANCHING_INDEX_COUNTERS: dict[str, int] = {}
_BRANCHING_INDEX_COUNTERS_LOCK = asyncio.Lock()


async def _claim_branch_index(cache_key: str, n_total: int) -> int:
    async with _BRANCHING_INDEX_COUNTERS_LOCK:
        idx = _BRANCHING_INDEX_COUNTERS.get(cache_key, 0)
        if idx >= n_total:
            # Sibling pool exceeded the configured total leaf count — happens
            # if the user configured rollout.n > n_trees * 2^n_splits.
            # Wrap-around assigns extras round-robin; the trainer will still
            # see rollout.n rows.
            idx = idx % n_total
        _BRANCHING_INDEX_COUNTERS[cache_key] = _BRANCHING_INDEX_COUNTERS.get(cache_key, 0) + 1
        return idx


def _branching_index_counters_clear() -> None:
    _BRANCHING_INDEX_COUNTERS.clear()


# ---------------------------------------------------------------------------
# BranchingAgentLoop
# ---------------------------------------------------------------------------


@register("branching_agent")
class BranchingAgentLoop(AgentLoopBase):
    """Teacher-guided branching rollout. See module docstring."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.branching_cfg = self.config.actor_rollout_ref.rollout.get("branching", None)
        if self.branching_cfg is None:
            raise RuntimeError(
                "BranchingAgentLoop requires actor_rollout_ref.rollout.branching to be configured. "
                "Set rollout.branching.enabled=True and rollout.agent.default_agent_loop=branching_agent."
            )

        tool_config_path = self.config.data.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]

    @property
    def n_leaves(self) -> int:
        """Number of leaves PER TREE (= 2^n_splits). Used by per-tree pipeline
        / fallback paths. The total leaf count across all sibling rows of a
        prompt is ``self.n_leaves_total``.
        """
        return 2 ** int(self.branching_cfg.n_splits)

    @property
    def n_trees(self) -> int:
        n = int(self.branching_cfg.get("n_trees", 1) or 1)
        n = max(1, n)
        # In two-stage mode, n_trees must equal stage1_n so each tree gets
        # its own stage1 rollout as privileged context. Warn + auto-correct.
        if self._two_stage and n != self._stage1_n:
            logger.warning(
                "n_trees=%d != stage1_n=%d in two-stage mode. "
                "Forcing n_trees = stage1_n = %d to avoid wasted rollouts.",
                n, self._stage1_n, self._stage1_n,
            )
            n = self._stage1_n
        return n

    @property
    def n_leaves_total(self) -> int:
        """Total leaves across all trees of one prompt (= n_trees * n_leaves).
        When ``two_stage=True``, includes the Stage 1 normal rollouts:
        stage1_n + n_trees * n_leaves.
        Equals ``rollout.n`` under the validated parametric config.
        """
        branching_leaves = self.n_trees * self.n_leaves
        if self._two_stage:
            return self._stage1_n + branching_leaves
        return branching_leaves

    @property
    def _two_stage(self) -> bool:
        return bool(self.branching_cfg.get("two_stage", False))

    @property
    def _stage1_n(self) -> int:
        return int(self.branching_cfg.get("stage1_n", 4) or 4)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # Validate-time fast path: skip the entire branching pipeline so val
        # rollouts match a vanilla single-turn agent (no teacher peek, no GT
        # leakage via priv ctx). Without this guard, val_before_train and the
        # periodic test_freq=N evals also run the teacher-guided pipeline,
        # which under teacher_context_mode in {gt_marker, ref_gt} feeds the
        # teacher the ground-truth answer at scoring time. That inflates val
        # reward by ~0.04 vs vanilla GRPO baseline at step 0 and makes any
        # post-deploy comparison (no teacher, no GT) misleading.
        if bool(kwargs.get("validate", False)):
            return await self._run_validate_single_rollout(
                sampling_params=sampling_params, **kwargs
            )

        messages = list(kwargs["raw_prompt"])

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        prompt_ids = await self.apply_chat_template(
            messages, tools=self.tool_schemas, images=images, videos=videos,
        )
        # Stash the root prompt length so _query_teacher_topk can stitch
        # priv_ctx + student_response_so_far without recomputing the boundary.
        self._root_prompt_len = len(prompt_ids)

        gen_step = int(kwargs.get("generation_step", 0))
        base_cache_key = _branching_cache_key(prompt_ids, gen_step)
        n_total = self.n_leaves_total

        # --- Two-stage mode: single owner per prompt (not per-tree) ---
        if self._two_stage:
            return await self._run_two_stage_coordination(
                base_cache_key=base_cache_key,
                n_total=n_total,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                images=images,
                videos=videos,
                multi_modal_data=multi_modal_data,
                kwargs=kwargs,
            )

        # --- Original per-tree mode ---
        # Cache key shared across the sibling rows of this prompt. We claim a
        # GLOBAL slot in [0, n_leaves_total) and decompose it into
        # (tree_idx, leaf_idx_in_tree); each tree has its OWN cache key /
        # owner / Future so trees run independent student rollouts.
        n_per_tree = self.n_leaves

        # Each sibling row claims a global slot. If rollout.n > n_total the
        # extras wrap around (duplicate slots); typical config aligns the two.
        # If branch_index is explicit (callers pre-assign), we don't bump the
        # counter — bumping with no consumer was redundant and risked drift
        # against claimed indices.
        explicit_index = int(kwargs.get("branch_index", -1))
        if explicit_index >= 0:
            global_slot = explicit_index
            counter_bumped = False
        else:
            global_slot = await _claim_branch_index(base_cache_key, n_total)
            counter_bumped = True

        tree_idx = (global_slot // n_per_tree) % self.n_trees
        leaf_idx_in_tree = global_slot % n_per_tree
        branch_index = leaf_idx_in_tree
        cache_key = _tree_cache_key(base_cache_key, tree_idx)

        metrics: dict[str, Any] = {}
        # Coordination: at most one sibling actually runs the pipeline; others await.
        async with _BRANCHING_CACHE_LOCK:
            future = _BRANCHING_CACHE.get(cache_key)
            if future is None:
                # Use get_running_loop in async context to avoid Python 3.12
                # implicit-loop creation deprecation.
                future = asyncio.get_running_loop().create_future()
                _BRANCHING_CACHE[cache_key] = future
                am_owner = True
            else:
                am_owner = False

        if am_owner:
            # Wrap the pipeline in try/except so a single bad prompt cannot
            # blast-radius into all sibling rows + the entire batch chunk.
            # On failure the owner emits a fallback set of identical leaves and
            # surfaces the error via diagnostics rather than re-raising.
            try:
                with simple_timer("branching_pipeline", metrics):
                    leaves = await self._run_branching_pipeline(
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        images=images,
                        videos=videos,
                        multi_modal_data=multi_modal_data,
                        kwargs=kwargs,
                        tree_idx=tree_idx,
                    )
                future.set_result(leaves)
            except BaseException as e:  # noqa: BLE001
                logger.exception(
                    "BranchingAgentLoop pipeline failed (cache_key=%s); "
                    "attempting plain student rollout fallback.", cache_key,
                )
                # Best-effort fallback. CRITICAL: emit ``n_leaves`` INDEPENDENT
                # student rollouts (different vLLM seeds → different responses)
                # rather than 1 rollout × n copies. If we copied a single
                # rollout, GRPO's group baseline computes ``id_std=0`` over
                # identical rewards and ``adv_std_floor`` clamps the divisor
                # — every advantage becomes ``(R - R)/floor = 0`` and the run
                # is dead (grad_norm=0, pg_loss=0). Verified on the first TGB
                # pilot: vLLM rejected logprobs=50, every owner pipeline took
                # the fallback, identical leaves killed GRPO.
                fallback_reason = f"pipeline_exception:{type(e).__name__}"
                try:
                    plain_sp = dict(sampling_params)
                    plain_sp.pop("logprobs", None)
                    plain_sp["logprobs"] = False
                    fallback_leaves = await self._fallback_n_independent(
                        prompt_ids=prompt_ids,
                        sampling_params=plain_sp,
                        multi_modal_data=multi_modal_data,
                        images=images,
                        videos=videos,
                        reason=f"{fallback_reason}:plain_rollout_ok",
                        n_leaves=n_per_tree,
                        tree_idx=tree_idx,
                    )
                    future.set_result(fallback_leaves)
                except BaseException as plain_err:  # noqa: BLE001
                    # Both branching AND plain rollout failed for this prompt.
                    # Previously we substituted EOS-only leaves here, but that
                    # produced n_leaves IDENTICAL responses → group_std=0 →
                    # GRPO advantage=0 → grad_norm=0 — the exact dead-batch
                    # collapse the outer fallback was added to prevent.
                    # Re-raise so the trainer sees a real failure and the
                    # async_rollout_manager can surface it (the alternative
                    # — silently degenerating the PPO group baseline — was
                    # found by the Phase 1c.2 audit to be worse).
                    logger.exception(
                        "Plain-rollout fallback also failed for cache_key=%s; "
                        "propagating to siblings.", cache_key,
                    )
                    future.set_exception(plain_err)
                    raise

        try:
            leaves = await future
        except BaseException:
            # Owner pipeline genuinely failed and even the fallback didn't fire
            # (or our row is non-owner and the owner's fallback also raised).
            # Roll back this sibling's counter claim so the next PPO step starts
            # clean even if clear_branching_cache is somehow skipped. The
            # counter lives on the BASE prompt key (shared across trees).
            if counter_bumped:
                async with _BRANCHING_INDEX_COUNTERS_LOCK:
                    if base_cache_key in _BRANCHING_INDEX_COUNTERS:
                        _BRANCHING_INDEX_COUNTERS[base_cache_key] = max(
                            0, _BRANCHING_INDEX_COUNTERS[base_cache_key] - 1
                        )
            raise

        if not (0 <= branch_index < len(leaves)):
            raise RuntimeError(
                f"BranchingAgentLoop: branch_index={branch_index} out of range "
                f"(have {len(leaves)} leaves)"
            )
        leaf = leaves[branch_index]
        # Surface metrics from the owner's run on the owner's row only; non-owners
        # inherit empty metrics so we don't double-count timing.
        leaf.metrics = AgentLoopMetrics(**metrics) if am_owner else AgentLoopMetrics()
        return leaf

    # ------------------------------------------------------------------
    # Validate-time fast path — bypass branching pipeline entirely.
    # ------------------------------------------------------------------

    async def _run_validate_single_rollout(
        self, *, sampling_params: dict[str, Any], **kwargs
    ) -> AgentLoopOutput:
        """Plain single-turn student rollout for val.

        Equivalent in semantics to ``SingleTurnAgentLoop.run``: apply chat
        template, call the rollout server once, wrap the result in an
        ``AgentLoopOutput``. We deliberately do NOT touch the module-level
        cache or branch-index counter so the train-time owner/sibling
        coordination is unaffected when val and train rollouts interleave.

        We also strip ``logprobs`` from the sampling params: the branching
        pipeline asks vLLM for top-K logprobs, but plain val rollouts don't
        need them and some vLLM configs reject ``logprobs=K`` paired with
        the simple chat path.
        """
        messages = list(kwargs["raw_prompt"])

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        prompt_ids = await self.apply_chat_template(
            messages, tools=self.tool_schemas, images=images, videos=videos,
        )

        plain_sp = dict(sampling_params)
        plain_sp.pop("logprobs", None)
        plain_sp["logprobs"] = False

        metrics: dict[str, Any] = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=plain_sp,
                image_data=images,
                video_data=videos,
            )
        response_mask = [1] * len(output.token_ids)
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=(
                output.log_probs[: self.response_length] if output.log_probs else None
            ),
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=AgentLoopMetrics(**metrics),
        )

    # ------------------------------------------------------------------
    # Two-stage coordination: Stage 1 normal + Stage 2 branching.
    # ------------------------------------------------------------------

    async def _run_two_stage_coordination(
        self,
        *,
        base_cache_key: str,
        n_total: int,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        images: Optional[list[Any]],
        videos: Optional[list[Any]],
        multi_modal_data: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> AgentLoopOutput:
        """Two-stage coordination: a single owner per prompt runs the entire
        two-stage pipeline (Stage 1 normal rollouts + Stage 2 branching).
        All sibling rows await the single cache entry and pick their slot.
        """
        # In two-stage mode we use the BASE cache key (not per-tree) since
        # one owner produces ALL outputs for the prompt.
        cache_key = f"{base_cache_key}#2stage"

        explicit_index = int(kwargs.get("branch_index", -1))
        if explicit_index >= 0:
            global_slot = explicit_index
            counter_bumped = False
        else:
            global_slot = await _claim_branch_index(base_cache_key, n_total)
            counter_bumped = True

        metrics: dict[str, Any] = {}
        async with _BRANCHING_CACHE_LOCK:
            future = _BRANCHING_CACHE.get(cache_key)
            if future is None:
                future = asyncio.get_running_loop().create_future()
                _BRANCHING_CACHE[cache_key] = future
                am_owner = True
            else:
                am_owner = False

        if am_owner:
            try:
                with simple_timer("two_stage_pipeline", metrics):
                    all_outputs = await self._run_two_stage_pipeline(
                        prompt_ids=prompt_ids,
                        sampling_params=sampling_params,
                        images=images,
                        videos=videos,
                        multi_modal_data=multi_modal_data,
                        kwargs=kwargs,
                    )
                future.set_result(all_outputs)
            except BaseException as e:  # noqa: BLE001
                logger.exception(
                    "Two-stage pipeline failed (cache_key=%s); "
                    "attempting plain student rollout fallback.", cache_key,
                )
                try:
                    plain_sp = dict(sampling_params)
                    plain_sp.pop("logprobs", None)
                    plain_sp["logprobs"] = False
                    fallback_leaves = await self._fallback_n_independent(
                        prompt_ids=prompt_ids,
                        sampling_params=plain_sp,
                        multi_modal_data=multi_modal_data,
                        images=images,
                        videos=videos,
                        reason=f"two_stage_exception:{type(e).__name__}",
                        n_leaves=n_total,
                    )
                    future.set_result(fallback_leaves)
                except BaseException as plain_err:  # noqa: BLE001
                    logger.exception(
                        "Fallback also failed for two-stage cache_key=%s.", cache_key,
                    )
                    future.set_exception(plain_err)
                    raise

        try:
            all_outputs = await future
        except BaseException:
            if counter_bumped:
                async with _BRANCHING_INDEX_COUNTERS_LOCK:
                    if base_cache_key in _BRANCHING_INDEX_COUNTERS:
                        _BRANCHING_INDEX_COUNTERS[base_cache_key] = max(
                            0, _BRANCHING_INDEX_COUNTERS[base_cache_key] - 1
                        )
            raise

        # Wrap-around for safety
        slot = global_slot % len(all_outputs) if all_outputs else 0
        if not (0 <= slot < len(all_outputs)):
            raise RuntimeError(
                f"Two-stage: slot={slot} out of range (have {len(all_outputs)} outputs)"
            )
        leaf = all_outputs[slot]
        leaf.metrics = AgentLoopMetrics(**metrics) if am_owner else AgentLoopMetrics()
        return leaf

    async def _run_two_stage_pipeline(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        images: Optional[list[Any]],
        videos: Optional[list[Any]],
        multi_modal_data: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> list[AgentLoopOutput]:
        """Execute the full two-stage pipeline:
        Stage 1: generate stage1_n independent student rollouts
        Stage 2: build teacher context (ref or marker) → branching
        Returns stage1_n + n_trees*n_leaves outputs.

        Bug-fix (2026-07-03): Each tree[i] now uses stage1[i] as its own
        privileged context (1:1 mapping) instead of all trees sharing the
        first successful stage1. If stage1[i] fails (below threshold), the
        tree falls back to any successful stage1, or static marker.

        Streaming optimisation (2026-07-02): Stage 1 rollouts are processed as
        they complete (``asyncio.as_completed``). Each rollout is scored
        immediately upon arrival; the FIRST rollout that meets the success
        threshold triggers Stage 2 branching — remaining Stage 1 rollouts
        continue generating in the background and are collected before
        returning. This overlaps Stage 2 start with tail Stage 1 generation,
        saving ~50-100 s per training step on long-CoT tasks (math, GSM8K).
        """
        cfg = self.branching_cfg
        stage1_n = self._stage1_n
        n_trees = self.n_trees  # == stage1_n after the property auto-correction
        two_stage_teacher_mode = str(cfg.get("two_stage_teacher_mode", "ref_or_marker"))

        # --- Stage 1: Normal student rollouts (streaming) ---
        # Launch all Stage 1 rollouts as independent tasks. Use as_completed
        # to process each rollout the moment it finishes (not wait-for-all).
        stage1_coros = [
            self._generate_one_stage1_rollout(
                idx=i,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                images=images,
                videos=videos,
                multi_modal_data=multi_modal_data,
            )
            for i in range(stage1_n)
        ]
        stage1_tasks = [asyncio.ensure_future(c) for c in stage1_coros]

        score_fn = kwargs.get("score_fn")
        threshold = float(cfg.get("success_reward_threshold", 1.0))
        # Per-stage1 scoring: track which indices succeeded
        stage1_outputs: list[AgentLoopOutput] = [None] * stage1_n  # type: ignore[list-item]
        stage1_scores: list[Optional[float]] = [None] * stage1_n
        stage1_response_texts: list[Optional[str]] = [None] * stage1_n
        first_success_event = asyncio.Event()
        first_success_idx: Optional[int] = None

        # Score each rollout as it completes. Track per-index results so we
        # can map tree[i] -> stage1[i] later.
        for done in asyncio.as_completed(stage1_tasks):
            try:
                out = await done
            except Exception:  # noqa: BLE001
                continue
            idx = int(out.extra_fields.get("leaf_id", 0))
            stage1_outputs[idx] = out
            if (
                two_stage_teacher_mode == "ref_or_marker"
                and score_fn is not None
            ):
                s = await score_fn(
                    prompt_ids=out.prompt_ids,
                    response_ids=out.response_ids,
                    raw_prompt=kwargs.get("raw_prompt", []),
                )
                stage1_scores[idx] = s
                if s >= threshold:
                    stage1_response_texts[idx] = self.tokenizer.decode(
                        out.response_ids, skip_special_tokens=True,
                    )
                    if first_success_idx is None:
                        first_success_idx = idx
                        first_success_event.set()
                        # Don't break — continue scoring remaining to get
                        # per-tree privileged context, but start Stage 2 early
                        # via the event.
                        break

        # Collect remaining Stage 1 outputs that were still running when we
        # broke out of the as_completed loop.
        remaining = [t for t in stage1_tasks if not t.done()]
        if remaining:
            remaining_results = await asyncio.gather(*remaining, return_exceptions=True)
            for r in remaining_results:
                if isinstance(r, BaseException):
                    continue
                idx = int(r.extra_fields.get("leaf_id", 0))
                stage1_outputs[idx] = r
                # Score remaining if we need per-tree contexts
                if (
                    two_stage_teacher_mode == "ref_or_marker"
                    and score_fn is not None
                    and stage1_scores[idx] is None
                ):
                    s = await score_fn(
                        prompt_ids=r.prompt_ids,
                        response_ids=r.response_ids,
                        raw_prompt=kwargs.get("raw_prompt", []),
                    )
                    stage1_scores[idx] = s
                    if s >= threshold:
                        stage1_response_texts[idx] = self.tokenizer.decode(
                            r.response_ids, skip_special_tokens=True,
                        )
                        if first_success_idx is None:
                            first_success_idx = idx

        # --- Build per-tree privileged contexts ---
        # Each tree[i] uses stage1[i] if it succeeded, else falls back to any
        # successful stage1 (preferring first_success_idx), else static marker.
        fallback_success_text: Optional[str] = None
        if first_success_idx is not None:
            fallback_success_text = stage1_response_texts[first_success_idx]

        async def _build_priv_ctx_for_tree(tree_i: int):
            """Build privileged context for tree_i using stage1[tree_i] if successful."""
            text = stage1_response_texts[tree_i] if tree_i < len(stage1_response_texts) else None
            if text is None:
                text = fallback_success_text  # fallback to any success
            if text is not None:
                return await self._build_ref_privileged_context(
                    prompt_ids=prompt_ids,
                    successful_response_text=text,
                    kwargs=kwargs,
                    images=images,
                    videos=videos,
                )
            else:
                # No successful stage1 at all — fallback to static marker
                return await self._build_privileged_context(
                    prompt_ids=prompt_ids, kwargs=kwargs, images=images, videos=videos,
                )

        # --- Stage 2: Branching with per-tree privileged context ---
        async def _stage2_for_tree(tree_i: int) -> list[AgentLoopOutput]:
            priv_ctx_ids, priv_ctx_meta = await _build_priv_ctx_for_tree(tree_i)
            leaves = await self._run_branching_with_priv_ctx(
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                images=images,
                videos=videos,
                multi_modal_data=multi_modal_data,
                kwargs=kwargs,
                tree_idx=tree_i,
                priv_ctx_ids=priv_ctx_ids,
                priv_ctx_meta=priv_ctx_meta,
            )
            return leaves

        stage2_coros = [_stage2_for_tree(ti) for ti in range(n_trees)]
        stage2_task = asyncio.ensure_future(
            asyncio.gather(*stage2_coros, return_exceptions=True)
        )

        # Await Stage 2 branching results.
        branched_trees = await stage2_task
        branched_outputs: list[AgentLoopOutput] = []
        for tree_idx, tree_leaves in enumerate(branched_trees):
            if isinstance(tree_leaves, BaseException):
                continue
            branched_outputs.extend(tree_leaves)

        # Combine: [stage1_outputs..., branched_outputs...]
        # Filter out None entries (failed stage1 rollouts).
        valid_stage1 = [o for o in stage1_outputs if o is not None]
        all_outputs = valid_stage1 + branched_outputs

        # --- DPO pairing metadata -------------------------------------------
        # The on-policy DPO loss needs each teacher-branched (chosen, rejected)
        # sibling pair identified by a STABLE key that survives batch reordering
        # (balance_batch) and size-1 micro-batching in dp_actor. Without this,
        # the loss fell back to the fragile "consecutive order" assumption and,
        # combined with ppo_micro_batch_size_per_gpu=1, never found >=2 Stage 2
        # samples in a single forward — so DPO never engaged (dpo_n_pairs=0).
        #
        # We tag here, at rollout time, where the leaf ordering is still clean:
        #   dpo_pair_id : per-prompt-local pair index shared by the two siblings
        #                 (-1 for Stage 1 / fallback / unpaired leaves).
        #   dpo_role    : 0 = chosen (teacher-preferred), 1 = rejected, -1 else.
        # (uid, dpo_pair_id) is globally unique across the training batch.
        #
        # A leaf is a valid pair member iff it carries teacher branch logprobs
        # (set only by a real split in _flatten_tree_to_leaves) and is not a
        # padding/fallback rollout. Role is derived from the leaf's own teacher
        # logprobs (branch vs sibling), so it is independent of list order.
        for o in all_outputs:
            o.extra_fields.setdefault("dpo_pair_id", -1)
            o.extra_fields.setdefault("dpo_role", -1)
        valid_stage2 = [
            o for o in branched_outputs
            if o.extra_fields.get("teacher_branch_logprob") is not None
            and int(o.extra_fields.get("is_branching_fallback", 0)) == 0
            and int(o.extra_fields.get("is_two_stage_stage1", 0)) == 0
        ]
        # Group by tree_idx before pairing to avoid cross-tree mismatch when
        # n_trees > 1. Without this, flat iteration pairs tree_0 leaves with
        # tree_1 leaves — completely wrong chosen/rejected labels.
        from collections import defaultdict as _defaultdict
        by_tree: dict[int, list] = _defaultdict(list)
        for o in valid_stage2:
            tree_idx_val = int(o.extra_fields.get("tree_idx", -1))
            if tree_idx_val < 0:
                logger.warning(f"Missing tree_idx for leaf {o.extra_fields.get('leaf_id')}; skipping DPO pairing")
                continue
            by_tree[tree_idx_val].append(o)

        global_pair_id = 0
        for _ti in sorted(by_tree.keys()):
            leaves = by_tree[_ti]
            if len(leaves) % 2 != 0:
                import logging
                logging.getLogger(__name__).warning(
                    f"tree_idx={_ti}: odd number of valid stage2 leaves ({len(leaves)}), "
                    f"last leaf will be unpaired."
                )
            for pair_idx in range(len(leaves) // 2):
                a = leaves[2 * pair_idx]
                b = leaves[2 * pair_idx + 1]
                for o in (a, b):
                    tb = o.extra_fields.get("teacher_branch_logprob")
                    ts = o.extra_fields.get("teacher_sibling_logprob")
                    o.extra_fields["dpo_pair_id"] = global_pair_id
                    o.extra_fields["dpo_role"] = 0 if (ts is None or tb >= ts) else 1
                # Guard: if teacher logprobs made both siblings the same role
                # (degenerate/tied margin), force the second to the opposite role so
                # the pair still has exactly one chosen + one rejected.
                if a.extra_fields["dpo_role"] == b.extra_fields["dpo_role"]:
                    b.extra_fields["dpo_role"] = 1 - a.extra_fields["dpo_role"]
                global_pair_id += 1
        return all_outputs

    async def _generate_one_stage1_rollout(
        self,
        *,
        idx: int,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        images: Optional[list[Any]],
        videos: Optional[list[Any]],
        multi_modal_data: dict[str, Any],
    ) -> AgentLoopOutput:
        """Generate a single Stage 1 rollout with entropy-based branch-point
        detection. Extracted from ``_generate_stage1_rollouts`` to support the
        streaming two-stage pipeline where each rollout is scored individually
        as it completes.
        """
        cfg = self.branching_cfg
        sp = self._student_sampling_params(sampling_params, max_tokens=None)
        pw_override = cfg.get("entropy_protect_window", None)
        pw = int(pw_override) if pw_override is not None else int(cfg.entropy_window)

        out = await self.server_manager.generate(
            request_id=uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=dict(sp),
            image_data=images,
            video_data=videos,
        )

        tokens = list(out.token_ids)[: self.response_length]
        if not tokens:
            eos_id = (
                getattr(self.tokenizer, "eos_token_id", None)
                or getattr(self.tokenizer, "pad_token_id", None)
                or 0
            )
            tokens = [int(eos_id)]
        lps = list((out.log_probs or [0.0] * len(tokens))[: self.response_length])
        if len(lps) < len(tokens):
            lps = lps + [0.0] * (len(tokens) - len(lps))

        branch_mask = [0] * len(tokens)
        top_logprobs = list(out.top_logprobs or [])[: self.response_length]
        if top_logprobs and len(top_logprobs) > pw + 1:
            entropies = [entropy_from_topk_logprobs(d) for d in top_logprobs]
            positions, _, _ = find_decision_positions_with_sigma_relaxation(
                entropies,
                target_count=1,
                window_size=int(cfg.entropy_window),
                protect_window=pw,
                sigma_start=float(cfg.entropy_sigma_start),
                sigma_step=float(cfg.entropy_sigma_step),
                sigma_floor=float(cfg.entropy_sigma_floor),
            )
            if positions:
                branch_pos = positions[0]
                if branch_pos < len(branch_mask):
                    branch_mask[branch_pos] = 1

        return AgentLoopOutput(
            prompt_ids=list(prompt_ids),
            response_ids=tokens,
            response_mask=[1] * len(tokens),
            response_logprobs=lps,
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=AgentLoopMetrics(),
            extra_fields={
                "branch_token_mask": branch_mask,
                "is_branching_fallback": 0,
                "is_two_stage_stage1": 1,
                "leaf_id": idx,
                "leaf_depth": 0,
                "tree_idx": idx,  # stage1[i] maps to tree[i]
                "branching_diag": {},
                "priv_ctx_meta": {},
                "branching_fallback": "",
            },
        )

    async def _generate_stage1_rollouts(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        images: Optional[list[Any]],
        videos: Optional[list[Any]],
        multi_modal_data: dict[str, Any],
    ) -> list[AgentLoopOutput]:
        """Generate stage1_n independent student rollouts in parallel.

        Each rollout is generated with ``logprobs=top_k`` so we can detect
        the entropy spike position (same detection as Stage 2 branching).
        The detected branch point is marked in ``branch_token_mask`` so the
        existing suffix loss mode zeros out the prefix and only trains
        tokens after the branch point — keeping Stage 1 consistent with
        Stage 2 branching.
        """
        cfg = self.branching_cfg
        K = int(cfg.top_k)
        stage1_n = self._stage1_n

        # Use student sampling params with logprobs=top_k (same as Stage 2).
        sp = self._student_sampling_params(sampling_params, max_tokens=None)

        # Entropy detection params for branch-point identification.
        pw_override = cfg.get("entropy_protect_window", None)
        pw = int(pw_override) if pw_override is not None else int(cfg.entropy_window)

        async def _gen_one(idx: int):
            return await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=dict(sp),
                image_data=images,
                video_data=videos,
            )

        outs = await asyncio.gather(*(_gen_one(i) for i in range(stage1_n)))

        leaves: list[AgentLoopOutput] = []
        for idx, out in enumerate(outs):
            tokens = list(out.token_ids)[: self.response_length]
            if not tokens:
                eos_id = (
                    getattr(self.tokenizer, "eos_token_id", None)
                    or getattr(self.tokenizer, "pad_token_id", None)
                    or 0
                )
                tokens = [int(eos_id)]
            lps = list((out.log_probs or [0.0] * len(tokens))[: self.response_length])
            if len(lps) < len(tokens):
                lps = lps + [0.0] * (len(tokens) - len(lps))

            # Detect entropy spike to find the branch point (same as Stage 2).
            branch_mask = [0] * len(tokens)
            top_logprobs = list(out.top_logprobs or [])[: self.response_length]
            if top_logprobs and len(top_logprobs) > pw + 1:
                entropies = [entropy_from_topk_logprobs(d) for d in top_logprobs]
                positions, _, _ = find_decision_positions_with_sigma_relaxation(
                    entropies,
                    target_count=1,
                    window_size=int(cfg.entropy_window),
                    protect_window=pw,
                    sigma_start=float(cfg.entropy_sigma_start),
                    sigma_step=float(cfg.entropy_sigma_step),
                    sigma_floor=float(cfg.entropy_sigma_floor),
                )
                if positions:
                    branch_pos = positions[0]
                    if branch_pos < len(branch_mask):
                        branch_mask[branch_pos] = 1

            leaves.append(AgentLoopOutput(
                prompt_ids=list(prompt_ids),
                response_ids=tokens,
                response_mask=[1] * len(tokens),
                response_logprobs=lps,
                multi_modal_data=multi_modal_data,
                num_turns=2,
                metrics=AgentLoopMetrics(),
                extra_fields={
                    "branch_token_mask": branch_mask,
                    "is_branching_fallback": 0,
                    "is_two_stage_stage1": 1,
                    "leaf_id": idx,
                    "leaf_depth": 0,
                    "tree_idx": idx,  # stage1[i] maps to tree[i]
                    "branching_diag": {},
                    "priv_ctx_meta": {},
                    "branching_fallback": "",
                },
            ))
        return leaves

    async def _build_ref_privileged_context(
        self,
        *,
        prompt_ids: list[int],
        successful_response_text: str,
        kwargs: dict[str, Any],
        images: Optional[list[Any]],
        videos: Optional[list[Any]],
    ) -> tuple[list[int], dict[str, Any]]:
        """Build privileged context using a successful Stage 1 response,
        formatted identically to SDPO training-side ref mode (reprompt with
        solution section).
        """
        cfg = self.branching_cfg
        raw_prompt = list(kwargs.get("raw_prompt", []))

        messages = build_ref_context_messages(
            raw_prompt=raw_prompt,
            successful_response_text=successful_response_text,
            self_distillation_cfg=cfg,
        )
        if not messages:
            # Fallback to plain prompt
            return list(prompt_ids), {
                "mode": "ref", "ref_available": False, "marker_len": 0,
            }

        ref_ids = await self.apply_chat_template(
            messages, tools=self.tool_schemas, images=images, videos=videos,
        )
        cap = int(cfg.get("max_reprompt_len", 10240) or 10240)
        if len(ref_ids) > cap:
            ref_ids = ref_ids[-cap:]
        return list(ref_ids), {
            "mode": "ref",
            "ref_available": True,
            "ref_ids_len": len(ref_ids),
            "marker_len": 0,
        }

    async def _run_branching_with_priv_ctx(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        images: Optional[list[Any]],
        videos: Optional[list[Any]],
        multi_modal_data: dict[str, Any],
        kwargs: dict[str, Any],
        tree_idx: int,
        priv_ctx_ids: list[int],
        priv_ctx_meta: dict[str, Any],
    ) -> list[AgentLoopOutput]:
        """Run the branching pipeline for a single tree using pre-built
        privileged context (from two-stage). This is a slimmed version of
        _run_branching_pipeline that skips _build_privileged_context.
        """
        cfg = self.branching_cfg
        K = int(cfg.top_k)
        n_splits = int(cfg.n_splits)
        _max_depth_override = cfg.get("max_branch_depth", None)
        max_depth = int(_max_depth_override) if _max_depth_override is not None else int(n_splits)
        split_trigger = str(cfg.get("split_trigger", "entropy") or "entropy")
        traj_id = uuid4().hex

        student_sp = self._student_sampling_params(
            sampling_params, max_tokens=None, tree_idx=int(tree_idx),
        )

        # Student initial chain
        init_out = await self.server_manager.generate(
            request_id=traj_id,
            prompt_ids=prompt_ids,
            sampling_params=student_sp,
            image_data=images,
            video_data=videos,
        )
        init_tokens = list(init_out.token_ids)
        init_top = list(init_out.top_logprobs or [])
        if not init_top:
            return self._fallback_n_copies(
                prompt_ids=prompt_ids, response_tokens=init_tokens,
                response_logprobs=init_out.log_probs, multi_modal_data=multi_modal_data,
                reason="two_stage_no_top_logprobs",
            )
        init_entropies = [entropy_from_topk_logprobs(d) for d in init_top]

        # Recursive split
        diag = {
            "sigma_relaxations": 0,
            "teacher_intersect_misses": 0,
            "split_attempts": 0,
            "splits_succeeded": 0,
            "no_disagreement_count": 0,
            "tree_idx": int(tree_idx),
            "two_stage": 1,
        }
        root = await self._split_recursive(
            prefix_ids=list(prompt_ids),
            segment_tokens=init_tokens,
            segment_entropies=init_entropies,
            segment_top_logprobs=init_top,
            segment_realized_logprobs=list(init_out.log_probs or [0.0] * len(init_tokens)),
            depth=0,
            target=n_splits,
            traj_id=traj_id,
            priv_ctx_ids=priv_ctx_ids,
            student_sp=student_sp,
            images=images,
            videos=videos,
            cfg=cfg,
            max_depth=max_depth,
            diag=diag,
            split_trigger=split_trigger,
        )

        # Flatten + pad
        leaves = self._flatten_tree_to_leaves(
            root=root, multi_modal_data=multi_modal_data, diag=diag, priv_ctx_meta=priv_ctx_meta,
        )
        leaves = await self._pad_leaves_to_n(
            leaves, self.n_leaves,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            multi_modal_data=multi_modal_data,
            images=images,
            videos=videos,
            tree_idx=int(tree_idx),
        )
        # Tag tree_idx on each leaf so trainer can match stage1[i] <-> tree[i]
        for leaf in leaves:
            leaf.extra_fields["tree_idx"] = int(tree_idx)
        return leaves

    # ------------------------------------------------------------------
    # Pipeline entry — owner row only (siblings await the cache).
    # ------------------------------------------------------------------

    async def _run_branching_pipeline(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        images: Optional[list[Any]],
        videos: Optional[list[Any]],
        multi_modal_data: dict[str, Any],
        kwargs: dict[str, Any],
        tree_idx: int = 0,
    ) -> list[AgentLoopOutput]:
        """Generate 2^n_splits shared-prefix leaves via teacher-guided branching.

        ``tree_idx`` selects which independent tree this owner is running.
        Different trees of the same prompt run with different ``traj_id`` and
        a tree-specific seed offset injected into ``sampling_params``, so the
        student initial rollouts diverge across trees even though they share
        the same prompt prefix.

        Algorithm (see research/teacher_branching_rollout.md §4-§7):

        1. Build privileged-context prompt ids once (mode dispatch in
           ``_build_privileged_context``).
        2. Run the student initial chain with ``logprobs=top_k`` so we get
           per-position top-K dicts, used both for (truncated) entropy
           estimation and for teacher branch-point scoring.
        3. Recurse: at each segment find decision positions via σ-relaxation,
           teacher-pick (argmax, argmin) within student top-K, branch into
           two children, generate continuation per child, recurse. Fresh
           detector per segment with adaptive protect_window.
        4. Collect leaves; pad/truncate to exactly ``n_leaves`` so the per-
           sibling claim is well-defined.

        All vLLM calls share a single ``traj_id`` so AsyncLLMServerManager's
        sticky LRU keeps them on the same replica → prefix cache hits.
        """
        cfg = self.branching_cfg
        K = int(cfg.top_k)
        n_splits = int(cfg.n_splits)
        # Resolve effective values inline — `cfg` here is an OmegaConf DictConfig
        # (the rollout.branching block read via .get(...)) and does NOT carry
        # the BranchingConfig dataclass `@property` methods. Reading
        # `cfg.effective_max_branch_depth` raises ConfigAttributeError.
        _max_depth_override = cfg.get("max_branch_depth", None)
        max_depth = int(_max_depth_override) if _max_depth_override is not None else int(n_splits)
        split_trigger = str(cfg.get("split_trigger", "entropy") or "entropy")
        traj_id = uuid4().hex

        priv_ctx_ids, priv_ctx_meta = await self._build_privileged_context(
            prompt_ids=prompt_ids, kwargs=kwargs, images=images, videos=videos,
        )

        # Inject a tree-specific seed offset so independent trees produce
        # divergent student initial rollouts even when run from the same
        # base sampling_params + same prompt. Without this each owner relies
        # solely on vLLM's internal randomness; with the explicit offset the
        # divergence is deterministic and reproducible across runs.
        student_sp = self._student_sampling_params(
            sampling_params, max_tokens=None, tree_idx=int(tree_idx),
        )

        # 1. Student initial chain.
        init_out = await self.server_manager.generate(
            request_id=traj_id,
            prompt_ids=prompt_ids,
            sampling_params=student_sp,
            image_data=images,
            video_data=videos,
        )
        init_tokens = list(init_out.token_ids)
        init_top = list(init_out.top_logprobs or [])
        if not init_top:
            return self._fallback_n_copies(
                prompt_ids=prompt_ids, response_tokens=init_tokens,
                response_logprobs=init_out.log_probs, multi_modal_data=multi_modal_data,
                reason="no_top_logprobs",
            )
        init_entropies = [entropy_from_topk_logprobs(d) for d in init_top]

        # 2. Recursive split.
        diag = {
            "sigma_relaxations": 0,
            "teacher_intersect_misses": 0,
            "split_attempts": 0,
            "splits_succeeded": 0,
            "no_disagreement_count": 0,
            "tree_idx": int(tree_idx),
        }
        root = await self._split_recursive(
            prefix_ids=list(prompt_ids),
            segment_tokens=init_tokens,
            segment_entropies=init_entropies,
            segment_top_logprobs=init_top,
            segment_realized_logprobs=list(init_out.log_probs or [0.0] * len(init_tokens)),
            depth=0,
            target=n_splits,
            traj_id=traj_id,
            priv_ctx_ids=priv_ctx_ids,
            student_sp=student_sp,
            images=images,
            videos=videos,
            cfg=cfg,
            max_depth=max_depth,
            diag=diag,
            split_trigger=split_trigger,
        )

        # 3. Flatten + pad. Each tree owns ``n_leaves`` leaves; pad up to that
        # per-tree count when the tree was shallower than expected.
        leaves = self._flatten_tree_to_leaves(
            root=root, multi_modal_data=multi_modal_data, diag=diag, priv_ctx_meta=priv_ctx_meta,
        )
        leaves = await self._pad_leaves_to_n(
            leaves, self.n_leaves,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            multi_modal_data=multi_modal_data,
            images=images,
            videos=videos,
            tree_idx=int(tree_idx),
        )
        return leaves

    # ------------------------------------------------------------------
    # Privileged context builder (mode dispatch).
    # ------------------------------------------------------------------

    async def _build_privileged_context(
        self,
        *,
        prompt_ids: list[int],
        kwargs: dict[str, Any],
        images: Optional[list[Any]],
        videos: Optional[list[Any]],
    ) -> tuple[list[int], dict[str, Any]]:
        """Construct privileged-context prompt ids per ``teacher_context_mode``.

        Returns (priv_ctx_ids, meta) where meta carries diagnostics (mode used,
        gt_available, marker_len, etc).
        """
        cfg = self.branching_cfg
        mode = str(cfg.teacher_context_mode)
        rm = kwargs.get("reward_model") or {}
        gt = rm.get("ground_truth") if isinstance(rm, dict) else None
        if isinstance(gt, str) and not gt:
            gt = None

        if mode == "ref_gt":
            messages = build_ref_gt_messages(
                raw_prompt=list(kwargs.get("raw_prompt", [])),
                ground_truth=gt,
                self_distillation_cfg=cfg,
            )
            if not messages:
                return list(prompt_ids), {"mode": mode, "gt_available": gt is not None, "marker_len": 0}
            ref_ids = await self.apply_chat_template(
                messages, tools=self.tool_schemas, images=images, videos=videos,
            )
            cap = int(cfg.max_reprompt_len)
            if len(ref_ids) > cap:
                # Trim from the LEFT of the user message — keeps the role-start /
                # assistant boundary at the tail intact.
                ref_ids = ref_ids[-cap:]
            return list(ref_ids), {
                "mode": mode,
                "gt_available": gt is not None,
                "marker_len": 0,
                "ref_ids_len": len(ref_ids),
            }

        # marker / gt_marker
        marker_text = build_marker_text(
            mode=mode if mode in ("marker", "gt_marker") else "marker",
            ground_truth=gt,
            self_distillation_cfg=cfg,
        )
        marker_ids = self.tokenizer.encode(marker_text, add_special_tokens=False)
        return list(prompt_ids) + list(marker_ids), {
            "mode": mode,
            "gt_available": gt is not None,
            "marker_len": len(marker_ids),
        }

    def _student_sampling_params(
        self,
        sampling_params: dict[str, Any],
        *,
        max_tokens: Optional[int],
        tree_idx: int = 0,
    ) -> dict[str, Any]:
        """Coerce sampling_params to the student-call shape: logprobs=K (int)
        and (optionally) override max_tokens. Caller passes max_tokens=None to
        let vllm_async_server compute the default from response_length.

        When ``tree_idx > 0`` (multi-tree topology), we inject a deterministic
        per-tree seed offset so independent trees of the same prompt produce
        divergent student initial rollouts in a reproducible way. The offset
        is added to any caller-provided ``seed`` (so an outer reproducibility
        seed still composes); when no seed was provided we synthesise one from
        the tree index alone.
        """
        sp = dict(sampling_params)
        sp["logprobs"] = int(self.branching_cfg.top_k)
        if max_tokens is not None:
            sp["max_tokens"] = max_tokens
        if int(tree_idx) > 0:
            base_seed = sp.get("seed")
            try:
                base_seed_int = int(base_seed) if base_seed is not None else 0
            except (TypeError, ValueError):
                base_seed_int = 0
            # Large prime stride keeps tree seeds well-separated even for
            # n_trees up to a few hundred. Hash with prompt-independent
            # constant so the stride is purely a function of tree_idx.
            sp["seed"] = (base_seed_int + int(tree_idx) * 1_000_003) & 0x7FFFFFFF
        return sp

    def _teacher_sampling_params(self) -> dict[str, Any]:
        """Sampling params for the teacher branch-point query: max_tokens=1,
        logprobs=teacher_top_k. Temperature 0.0 — we read a distribution,
        not sample.

        We use ``teacher_top_k`` (default 50, independent of student ``top_k``
        default 10) so the teacher returns enough logprobs to cover all of
        student's top-K candidates with high probability. Without this
        asymmetry, the intersection between student and teacher top-K is
        small and many of student's candidates have UNKNOWN teacher logprob
        — they get silently dropped from argmax/argmin selection.
        """
        teacher_k = int(self.branching_cfg.get("teacher_top_k", None) or self.branching_cfg.top_k)
        return {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "logprobs": teacher_k,
            "max_tokens": int(self.branching_cfg.teacher_branch_query_max_tokens),
            "repetition_penalty": 1.0,
        }

    # ------------------------------------------------------------------
    # Recursive split.
    # ------------------------------------------------------------------

    async def _split_recursive(
        self,
        *,
        prefix_ids: list[int],
        segment_tokens: list[int],
        segment_entropies: list[float],
        segment_top_logprobs: list[dict[int, float]],
        segment_realized_logprobs: list[float],
        depth: int,
        target: int,
        traj_id: str,
        priv_ctx_ids: list[int],
        student_sp: dict[str, Any],
        images: Optional[list[Any]],
        videos: Optional[list[Any]],
        cfg,
        max_depth: int,
        diag: dict[str, int],
        split_trigger: str = "entropy",
    ) -> BranchNode:
        n = len(segment_tokens)
        # Adaptive protect window: don't waste >25% of a short segment.
        # Same caveat as effective_max_branch_depth above — `cfg` is a
        # DictConfig without dataclass properties; resolve inline.
        _pw_override = cfg.get("entropy_protect_window", None)
        pw_default = int(_pw_override) if _pw_override is not None else int(cfg.entropy_window)
        pw = min(pw_default, max(2, n // 4))

        node = BranchNode(
            prefix_tokens=list(prefix_ids),
            segment_tokens=list(segment_tokens),
            segment_entropies=list(segment_entropies),
            segment_logprobs=list(segment_realized_logprobs),
            segment_branch_mask=[0] * n,
            depth=depth,
            is_leaf=True,
        )

        if depth >= max_depth or n < pw + 2:
            return node

        positions, sigma_used, relaxations = find_decision_positions_with_sigma_relaxation(
            segment_entropies,
            target_count=target,
            window_size=int(cfg.entropy_window),
            protect_window=pw,
            sigma_start=float(cfg.entropy_sigma_start),
            sigma_step=float(cfg.entropy_sigma_step),
            sigma_floor=float(cfg.entropy_sigma_floor),
        )
        diag["sigma_relaxations"] += int(relaxations)
        if not positions:
            # Argmax fallback over the testable region.
            if n > pw + 1:
                tail = segment_entropies[pw:]
                positions = [pw + max(range(len(tail)), key=lambda i: tail[i])]
            else:
                return node
        if not positions:
            return node

        diag["split_attempts"] += 1

        # ----- Candidate-position validation ------------------------------
        # In ``entropy`` mode we only try the earliest spike (legacy behaviour).
        # In ``entropy_disagreement`` mode we walk all spikes in chronological
        # order and pick the FIRST position where the teacher's argmax within
        # student's pick-k differs from the student's actually sampled token
        # (i.e. teacher would have chosen differently). This filters out
        # low-signal split points where teacher already agrees with student.
        _pick_k_override = cfg.get("teacher_pick_top_k", None)
        pick_k = int(_pick_k_override) if _pick_k_override is not None else int(cfg.top_k)
        require_distinct = bool(cfg.require_distinct_branches)
        fallback_to_student = bool(cfg.fallback_to_student_topk)

        if split_trigger == "entropy_disagreement":
            candidate_positions = list(positions)
        else:
            candidate_positions = positions[:1]

        chosen_p: Optional[int] = None
        chosen_pairs: list[tuple[int, float]] = []
        chosen_teacher_top: Optional[dict[int, float]] = None
        chosen_pos_tok: Optional[int] = None
        chosen_neg_tok: Optional[int] = None

        for cand_p in candidate_positions:
            cand_student_top = segment_top_logprobs[cand_p] if cand_p < len(segment_top_logprobs) else {}
            if not cand_student_top:
                continue
            cand_teacher_top = await self._query_teacher_topk(
                priv_ctx_ids=priv_ctx_ids,
                prefix_ids=prefix_ids,
                segment_tokens=segment_tokens,
                branch_pos=cand_p,
                traj_id=traj_id,
                images=images,
                videos=videos,
            )
            if cand_teacher_top is None:
                continue

            # Sort student candidates by descending logprob — teacher picks
            # within the top ``teacher_pick_top_k`` (default = top_k, i.e. all
            # candidates). Slicing here lets us request a deep student top-K
            # for entropy estimation while still presenting only the high-
            # probability core to the teacher for branch selection (avoids
            # OOD branches).
            student_topk_pairs = sorted(cand_student_top.items(), key=lambda kv: -kv[1])
            if pick_k > 0:
                student_topk_pairs = student_topk_pairs[:pick_k]

            # Coverage diagnostic: how many of student's pick_k candidates
            # appear in teacher's top-K? If teacher_top_k is too small the
            # intersection is thin and many student candidates are silently
            # dropped from the argmax/argmin selection. Aggregate across all
            # split decisions so coverage_ratio surfaces in SwanLab.
            intersect_count = sum(1 for tok, _ in student_topk_pairs if tok in cand_teacher_top)
            diag["teacher_coverage_intersect_total"] = (
                diag.get("teacher_coverage_intersect_total", 0) + intersect_count
            )
            diag["teacher_coverage_pick_total"] = (
                diag.get("teacher_coverage_pick_total", 0) + len(student_topk_pairs)
            )

            # Disagreement check: skip positions where teacher's argmax (within
            # student top-K) equals student's actually sampled token. No
            # disagreement ⇒ splitting here would just duplicate effort.
            if split_trigger == "entropy_disagreement":
                scored = [
                    (tok, cand_teacher_top[tok])
                    for tok, _ in student_topk_pairs
                    if tok in cand_teacher_top
                ]
                if not scored:
                    # No teacher coverage of student's top-K; can't validate
                    # disagreement reliably — skip this position.
                    continue
                teacher_argmax_token = max(scored, key=lambda kv: kv[1])[0]
                if cand_p < len(segment_tokens):
                    student_actual_token = int(segment_tokens[cand_p])
                    if int(teacher_argmax_token) == student_actual_token:
                        # Teacher agrees with student — no value in branching.
                        continue

            choice = pick_teacher_branches(
                student_topk_pairs, cand_teacher_top,
                fallback_to_student=fallback_to_student,
            )
            if choice == (None, None):
                diag["teacher_intersect_misses"] += 1
                continue
            (cand_pos_tok, _), (cand_neg_tok, _) = choice
            if require_distinct and cand_pos_tok == cand_neg_tok:
                continue

            chosen_p = cand_p
            chosen_pairs = student_topk_pairs
            chosen_teacher_top = cand_teacher_top
            chosen_pos_tok = int(cand_pos_tok)
            chosen_neg_tok = int(cand_neg_tok)
            break

        if chosen_p is None:
            if split_trigger == "entropy_disagreement":
                diag["no_disagreement_count"] = diag.get("no_disagreement_count", 0) + 1
            return node

        p = chosen_p
        pos_tok = chosen_pos_tok
        neg_tok = chosen_neg_tok
        diag["splits_succeeded"] += 1

        # Pre-flight budget check: if any child would have zero room to fit even
        # the branch token + at least one continuation token, abort the split
        # entirely and return this node as a leaf. The earlier remaining==0
        # branch produced a zero-segment child that silently dropped the branch
        # token from the response (it lived only in prefix_tokens, which the
        # leaf flattener never emits) — degrading the split to two byte-
        # identical siblings with depth-1 ones in the mask. Cleanest fix is to
        # not split at all when the budget is exhausted.
        already_used_at_branch = (
            (len(prefix_ids) - self._root_prompt_len) + len(segment_tokens[:p]) + 1
        )
        if already_used_at_branch >= self.response_length:
            return node

        node.is_leaf = False
        node.branch_position = p
        # Generate the two child continuations.
        children: list[BranchNode] = []
        # DPO: teacher logprobs for both branch tokens (pos and neg).
        _pos_lp = float(chosen_teacher_top.get(int(pos_tok), float("-inf")))
        _neg_lp = float(chosen_teacher_top.get(int(neg_tok), float("-inf")))
        for chosen_token in (pos_tok, neg_tok):
            child_prefix = list(prefix_ids) + list(segment_tokens[:p]) + [int(chosen_token)]
            # Response budget remaining: total response_length minus all
            # response tokens consumed by every ancestor + this segment + the
            # branch token. ``prefix_ids - root_prompt_len`` counts ancestors;
            # ``len(segment_tokens[:p]) + 1`` counts this segment's preamble +
            # the branch token we're about to insert.
            already_used = (
                (len(prefix_ids) - self._root_prompt_len)
                + len(segment_tokens[:p])
                + 1
            )
            remaining = max(0, self.response_length - already_used)
            if remaining == 0:
                # Edge: pre-flight passed but a sibling consumed the last token.
                # The branch token still needs to land in the response, so emit
                # a 1-token leaf segment carrying just the chosen token.
                child_node = BranchNode(
                    prefix_tokens=list(prefix_ids) + list(segment_tokens[:p]),
                    segment_tokens=[int(chosen_token)],
                    segment_entropies=[0.0],
                    segment_logprobs=[0.0],
                    segment_branch_mask=[1],
                    depth=depth + 1,
                    is_leaf=True,
                )
                child_node.teacher_branch_logprob = _pos_lp if chosen_token == pos_tok else _neg_lp
                child_node.teacher_sibling_logprob = _neg_lp if chosen_token == pos_tok else _pos_lp
                children.append(child_node)
                continue
            child_sp = self._student_sampling_params(student_sp, max_tokens=remaining)
            child_out = await self.server_manager.generate(
                request_id=traj_id,
                prompt_ids=child_prefix,
                sampling_params=child_sp,
                image_data=images,
                video_data=videos,
            )
            child_tokens = list(child_out.token_ids)
            child_top = list(child_out.top_logprobs or [])
            child_lps = list(child_out.log_probs or [0.0] * len(child_tokens))
            child_entropies = (
                [entropy_from_topk_logprobs(d) for d in child_top]
                if child_top else [0.0] * len(child_tokens)
            )
            # The branch token itself is the first entry of the child's segment.
            seg_tokens = [int(chosen_token)] + child_tokens
            seg_entropies = [0.0] + child_entropies  # branch token's own entropy is undefined
            seg_lps = [0.0] + child_lps
            seg_top = [{}] + child_top  # branch token has no top-K from student's POV
            child_node = await self._split_recursive(
                prefix_ids=child_prefix[: -1],  # don't include the branch token in prefix
                segment_tokens=seg_tokens,
                segment_entropies=seg_entropies,
                segment_top_logprobs=seg_top,
                segment_realized_logprobs=seg_lps,
                depth=depth + 1,
                target=1,
                traj_id=traj_id,
                priv_ctx_ids=priv_ctx_ids,
                student_sp=student_sp,
                images=images,
                videos=videos,
                cfg=cfg,
                max_depth=max_depth,
                diag=diag,
                split_trigger=split_trigger,
            )
            # Mark the branch token position in the child's mask.
            if child_node.segment_branch_mask:
                child_node.segment_branch_mask[0] = 1
            # DPO: store teacher logprobs for reward shaping.
            child_node.teacher_branch_logprob = _pos_lp if chosen_token == pos_tok else _neg_lp
            child_node.teacher_sibling_logprob = _neg_lp if chosen_token == pos_tok else _pos_lp
            children.append(child_node)

        node.children = children
        # Truncate node's segment to before branch position so the leaf
        # walker doesn't double-emit tokens past the split.
        node.segment_tokens = node.segment_tokens[:p]
        node.segment_entropies = node.segment_entropies[:p]
        node.segment_logprobs = node.segment_logprobs[:p]
        node.segment_branch_mask = node.segment_branch_mask[:p]
        return node

    async def _query_teacher_topk(
        self,
        *,
        priv_ctx_ids: list[int],
        prefix_ids: list[int],
        segment_tokens: list[int],
        branch_pos: int,
        traj_id: str,
        images: Optional[list[Any]],
        videos: Optional[list[Any]],
    ) -> Optional[dict[int, float]]:
        """One vLLM call with the privileged-context prefix; read top-K of
        the next-token distribution. Returns the {token_id: logprob} dict."""
        # priv_ctx_ids already encodes (prompt + assistant_role_start + marker).
        # The student response so far = (the response that PREDATES this
        # segment's prefix_ids tail) + segment_tokens up to branch_pos.
        # prefix_ids = original_prompt_ids + earlier_segment_tokens; the
        # student response so far in TOKEN form is therefore:
        #     prefix_ids[len(original_prompt_ids):] + segment_tokens[:branch_pos]
        # But we don't carry "len(original_prompt_ids)" through — the priv_ctx
        # already contains the original prompt portion. To stitch correctly we
        # need (priv_ctx + student_response_so_far). Compute student_response_so_far
        # from what we know: the parent called this segment with prefix_ids that
        # equal original_prompt + previously-chosen branch tokens + earlier
        # segments. The simplest correct construction is to take the suffix of
        # prefix_ids beyond the original prompt, but we never tracked the
        # original prompt length here. Instead, compute by length:
        # priv_ctx_ids was built as prompt_ids (root) + marker. So
        # student_response_so_far = prefix_ids[len(prompt_ids_root):] +
        # segment_tokens[:branch_pos]. We don't have prompt_ids_root in scope —
        # store it on self at the start of run() for this purpose.
        student_so_far = (
            prefix_ids[self._root_prompt_len :]
            + list(segment_tokens[:branch_pos])
        )
        teacher_prompt = list(priv_ctx_ids) + list(student_so_far)
        try:
            out = await self.server_manager.generate(
                request_id=traj_id,
                prompt_ids=teacher_prompt,
                sampling_params=self._teacher_sampling_params(),
                image_data=images,
                video_data=videos,
            )
        except Exception:  # noqa: BLE001
            return None
        if not out.top_logprobs:
            return None
        return out.top_logprobs[0]

    # ------------------------------------------------------------------
    # Tree → AgentLoopOutput leaves.
    # ------------------------------------------------------------------

    def _flatten_tree_to_leaves(
        self,
        *,
        root: BranchNode,
        multi_modal_data: dict[str, Any],
        diag: dict[str, int],
        priv_ctx_meta: dict[str, Any],
    ) -> list[AgentLoopOutput]:
        outputs: list[AgentLoopOutput] = []
        path_tokens: list[int] = []
        path_entropies: list[float] = []
        path_logprobs: list[float] = []
        path_branch_mask: list[int] = []
        leaf_counter = [0]
        # DPO: track teacher logprobs from the split-node parent to leaf outputs.
        # Each leaf inherits from its immediate parent child-of-split.
        path_teacher_branch_logprob: list[Optional[float]] = []
        path_teacher_sibling_logprob: list[Optional[float]] = []

        def _visit(node: BranchNode) -> None:
            saved = (
                len(path_tokens), len(path_entropies),
                len(path_logprobs), len(path_branch_mask),
            )
            path_tokens.extend(node.segment_tokens)
            path_entropies.extend(node.segment_entropies)
            path_logprobs.extend(node.segment_logprobs)
            path_branch_mask.extend(node.segment_branch_mask)
            # Push this node's teacher logprobs onto the path stack.
            path_teacher_branch_logprob.append(node.teacher_branch_logprob)
            path_teacher_sibling_logprob.append(node.teacher_sibling_logprob)

            if node.is_leaf:
                node.leaf_id = leaf_counter[0]
                leaf_counter[0] += 1
                truncate = self.response_length
                leaf_tokens = list(path_tokens[:truncate])
                leaf_lps = list(path_logprobs[:truncate])
                leaf_mask = list(path_branch_mask[:truncate])
                # Defense: tokenizer.pad on an empty list returns a Python list
                # (not a tensor), which crashes downstream
                # _agent_loop_postprocess (response_output["input_ids"].dim()).
                # Force at least one token (EOS / pad) so the leaf is always a
                # well-formed sequence even if the student emitted nothing.
                if not leaf_tokens:
                    eos_id = (
                        getattr(self.tokenizer, "eos_token_id", None)
                        or getattr(self.tokenizer, "pad_token_id", None)
                        or 0
                    )
                    leaf_tokens = [int(eos_id)]
                    leaf_lps = [0.0]
                    leaf_mask = [0]
                # DPO: the teacher logprobs for the branch token are the last
                # non-None values pushed by ancestor nodes.
                t_logprob: Optional[float] = None
                for v in reversed(path_teacher_branch_logprob):
                    if v is not None:
                        t_logprob = v
                        break
                t_sibling_logprob: Optional[float] = None
                for v in reversed(path_teacher_sibling_logprob):
                    if v is not None:
                        t_sibling_logprob = v
                        break
                outputs.append(
                    AgentLoopOutput(
                        prompt_ids=root.prefix_tokens,
                        response_ids=leaf_tokens,
                        response_mask=[1] * len(leaf_tokens),
                        response_logprobs=leaf_lps,
                        multi_modal_data=multi_modal_data,
                        num_turns=2,
                        metrics=AgentLoopMetrics(),
                        extra_fields={
                            "branch_token_mask": leaf_mask,
                            "is_branching_fallback": 0,
                            "leaf_id": node.leaf_id,
                            "leaf_depth": node.depth,
                            "teacher_branch_logprob": t_logprob,
                            "teacher_sibling_logprob": t_sibling_logprob,
                            "branching_diag": dict(diag),
                            "priv_ctx_meta": dict(priv_ctx_meta),
                            "branching_fallback": "",
                        },
                    )
                )
            else:
                for child in node.children:
                    _visit(child)

            (lt, le, lp, lb) = saved
            del path_tokens[lt:]
            del path_entropies[le:]
            del path_logprobs[lp:]
            del path_branch_mask[lb:]
            path_teacher_branch_logprob.pop()
            path_teacher_sibling_logprob.pop()

        _visit(root)
        # Sanity check via the structural collector.
        leaves = collect_leaves(root)
        assert len(outputs) == len(leaves), (
            f"Tree flatten produced {len(outputs)} outputs but tree has {len(leaves)} leaves"
        )
        return outputs

    async def _pad_leaves_to_n(
        self,
        leaves: list[AgentLoopOutput],
        n: int,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        multi_modal_data: dict[str, Any],
        images: Optional[list[Any]] = None,
        videos: Optional[list[Any]] = None,
        tree_idx: int = 0,
    ) -> list[AgentLoopOutput]:
        """Ensure exactly ``n`` leaves.

        When the tree is shallower than expected (fewer leaves than n_leaves),
        fill missing slots with INDEPENDENT student rollouts instead of cloning
        the last leaf. This preserves GRPO group diversity — cloned duplicates
        have identical rewards and collapse the group variance to zero, starving
        the policy gradient of learning signal.

        Each independent rollout is tagged ``is_branching_fallback=1`` so
        dp_actor treats it with ``branch_token_loss_mode='all'`` (its
        branch_token_mask is all-zeros anyway).
        """
        if len(leaves) >= n:
            return leaves[:n]
        if not leaves:
            raise RuntimeError("BranchingAgentLoop._pad_leaves_to_n: no leaves to pad from")

        n_missing = n - len(leaves)
        # Generate independent student rollouts to fill the gap.
        fill_leaves = await self._fill_independent_rollouts(
            count=n_missing,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            multi_modal_data=multi_modal_data,
            images=images,
            videos=videos,
            start_leaf_id=len(leaves),
            reason=f"tree_shallow_{len(leaves)}_of_{n}",
            tree_idx=int(tree_idx),
        )
        leaves.extend(fill_leaves)
        return leaves[:n]

    async def _fill_independent_rollouts(
        self,
        *,
        count: int,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        multi_modal_data: dict[str, Any],
        images: Optional[list[Any]],
        videos: Optional[list[Any]],
        start_leaf_id: int,
        reason: str,
        tree_idx: int = 0,
    ) -> list[AgentLoopOutput]:
        """Generate ``count`` independent student rollouts to fill tree gaps.

        Unlike _fallback_n_independent (which always produces n_leaves), this
        generates exactly ``count`` rollouts. Each is a plain student generation
        tagged as is_branching_fallback=1 so dp_actor uses mode='all' for these
        rows (their branch_token_mask is all-zeros). When ``tree_idx > 0`` we
        offset the seed so different trees of the same prompt fill with
        divergent rollouts.
        """
        sp = dict(sampling_params)
        if sp.get("logprobs") is False or sp.get("logprobs") is None:
            sp["logprobs"] = 0
        if int(tree_idx) > 0:
            base_seed = sp.get("seed")
            try:
                base_seed_int = int(base_seed) if base_seed is not None else 0
            except (TypeError, ValueError):
                base_seed_int = 0
            sp["seed"] = (base_seed_int + int(tree_idx) * 1_000_003) & 0x7FFFFFFF

        async def _one_call(_idx: int):
            return await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=dict(sp),
                image_data=images,
                video_data=videos,
            )

        outs = await asyncio.gather(*(_one_call(i) for i in range(count)))

        leaves: list[AgentLoopOutput] = []
        for idx, out in enumerate(outs):
            tokens = list(out.token_ids)[: self.response_length]
            if not tokens:
                eos_id = (
                    getattr(self.tokenizer, "eos_token_id", None)
                    or getattr(self.tokenizer, "pad_token_id", None)
                    or 0
                )
                tokens = [int(eos_id)]
            lps = list((out.log_probs or [0.0] * len(tokens))[: self.response_length])
            if len(lps) < len(tokens):
                lps = lps + [0.0] * (len(tokens) - len(lps))
            leaves.append(AgentLoopOutput(
                prompt_ids=list(prompt_ids),
                response_ids=tokens,
                response_mask=[1] * len(tokens),
                response_logprobs=lps,
                multi_modal_data=multi_modal_data,
                num_turns=2,
                metrics=AgentLoopMetrics(),
                extra_fields={
                    "branch_token_mask": [0] * len(tokens),
                    "is_branching_fallback": 1,
                    "leaf_id": start_leaf_id + idx,
                    "leaf_depth": 0,
                    "tree_idx": int(tree_idx),
                    "branching_diag": {},
                    "priv_ctx_meta": {},
                    "branching_fallback": reason,
                },
            ))
        return leaves

    @staticmethod
    def _clone_leaf(
        source: AgentLoopOutput,
        *,
        leaf_id_override: Optional[int] = None,
        pad_index: int = 0,
    ) -> AgentLoopOutput:
        """Build a fresh AgentLoopOutput with its OWN extra_fields/list buffers,
        deep-copying mutable per-row state so post-processing can safely pop /
        write into one without affecting siblings."""
        src_extra = source.extra_fields or {}
        # Lists are stored by-reference inside the Pydantic model — we copy them
        # explicitly so per-row mutations stay local.
        src_metrics = getattr(source, "metrics", None)
        return AgentLoopOutput(
            prompt_ids=list(source.prompt_ids),
            response_ids=list(source.response_ids),
            response_mask=list(source.response_mask),
            response_logprobs=(
                list(source.response_logprobs) if source.response_logprobs is not None else None
            ),
            routed_experts=getattr(source, "routed_experts", None),
            multi_modal_data=source.multi_modal_data,
            num_turns=source.num_turns,
            metrics=AgentLoopMetrics(**src_metrics.model_dump()) if src_metrics is not None else AgentLoopMetrics(),
            extra_fields={
                **{k: (list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v))
                   for k, v in src_extra.items()},
                "leaf_id": leaf_id_override if leaf_id_override is not None else src_extra.get("leaf_id", 0),
                "is_padded_duplicate": True,
                "pad_index": pad_index,
            },
        )

    async def _fallback_n_independent(
        self,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        multi_modal_data: dict[str, Any],
        images: Optional[list[Any]],
        videos: Optional[list[Any]],
        reason: str,
        n_leaves: Optional[int] = None,
        tree_idx: int = 0,
    ) -> list[AgentLoopOutput]:
        """Fire ``n_leaves`` INDEPENDENT student rollouts in PARALLEL and wrap
        each as a leaf. Used when the branching pipeline raises but a plain
        student rollout still works.

        ``n_leaves`` defaults to the per-tree leaf count (``self.n_leaves``);
        the run() owner path always passes the explicit per-tree count so the
        fallback's leaf shape stays consistent with the per-tree owner /
        sibling-row contract regardless of how many trees the prompt uses.

        Three properties matter:
          (a) Different vLLM request_ids → different rollouts → GRPO group
              baseline retains variance (no zero-gradient collapse).
          (b) ``asyncio.gather`` parallelises the n_leaves calls so a fallback
              uid does NOT pay 8× wall-clock vs the normal branching path.
          (c) Every leaf is tagged ``is_branching_fallback=True``; dp_actor
              treats those rows as ``branch_token_loss_mode='all'`` regardless
              of the configured mode, so the mode×fallback interaction does
              not silently zero out PG (B1 of the 1c.2 audit).
        """
        # logprobs=0 = vLLM's "realized-token only" mode (vs False which yields
        # None). Keeping logprobs in the request preserves response_logprobs
        # downstream so the importance ratio doesn't degenerate.
        sp = dict(sampling_params)
        if sp.get("logprobs") is False or sp.get("logprobs") is None:
            sp["logprobs"] = 0
        if int(tree_idx) > 0:
            base_seed = sp.get("seed")
            try:
                base_seed_int = int(base_seed) if base_seed is not None else 0
            except (TypeError, ValueError):
                base_seed_int = 0
            sp["seed"] = (base_seed_int + int(tree_idx) * 1_000_003) & 0x7FFFFFFF

        target_n = int(n_leaves) if n_leaves is not None else int(self.n_leaves)

        # Parallel fan-out, one call per leaf, each with its own request_id.
        # AsyncLLMServerManager's sticky LRU keys on request_id, so distinct
        # uuid4 per leaf gives different replicas / different seeds.
        async def _one_call(_idx: int):
            return await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=dict(sp),
                image_data=images,
                video_data=videos,
            )

        outs = await asyncio.gather(*(_one_call(i) for i in range(target_n)))

        leaves: list[AgentLoopOutput] = []
        for idx, out in enumerate(outs):
            tokens = list(out.token_ids)[: self.response_length]
            if not tokens:
                eos_id = (
                    getattr(self.tokenizer, "eos_token_id", None)
                    or getattr(self.tokenizer, "pad_token_id", None)
                    or 0
                )
                tokens = [int(eos_id)]
            lps = list((out.log_probs or [0.0] * len(tokens))[: self.response_length])
            if len(lps) < len(tokens):
                lps = lps + [0.0] * (len(tokens) - len(lps))
            leaves.append(AgentLoopOutput(
                prompt_ids=list(prompt_ids),
                response_ids=tokens,
                response_mask=[1] * len(tokens),
                response_logprobs=lps,
                multi_modal_data=multi_modal_data,
                num_turns=2,
                metrics=AgentLoopMetrics(),
                extra_fields={
                    "branch_token_mask": [0] * len(tokens),
                    "is_branching_fallback": 1,
                    "leaf_id": idx,
                    "leaf_depth": 0,
                    "tree_idx": int(tree_idx),
                    "branching_diag": {},
                    "priv_ctx_meta": {},
                    "branching_fallback": reason,
                },
            ))
        return leaves

    def _fallback_n_copies(
        self,
        *,
        prompt_ids: list[int],
        response_tokens: list[int],
        response_logprobs: Optional[list[float]],
        multi_modal_data: dict[str, Any],
        reason: str,
    ) -> list[AgentLoopOutput]:
        """Build ``n_leaves`` identical leaves from a single chain. Used when
        we cannot branch (no top_logprobs surfaced, chain too short, etc.).

        Each leaf gets its OWN extra_fields dict / list buffers so per-row
        mutation in ``_agent_loop_postprocess`` cannot leak across siblings.

        IMPORTANT: ``response_tokens=[]`` would cause downstream
        ``tokenizer.pad`` to return a Python list (not a tensor) — the
        postprocess ``response_output["input_ids"].dim()`` call then raises
        AttributeError and brings down the entire validation/rollout step.
        We force at least 1 token (EOS or pad) so the response is always a
        well-formed sequence.
        """
        truncate = self.response_length
        tokens = list(response_tokens[:truncate])
        if not tokens:
            # Pad with a single end-of-sequence (or pad) token so postprocess
            # produces a 1×response_length tensor, not a degenerate list.
            eos_id = (
                getattr(self.tokenizer, "eos_token_id", None)
                or getattr(self.tokenizer, "pad_token_id", None)
                or 0
            )
            tokens = [int(eos_id)]
        lps_seed = response_logprobs if response_logprobs else [0.0] * len(tokens)
        lps = list(lps_seed[:truncate])
        if len(lps) < len(tokens):
            lps = lps + [0.0] * (len(tokens) - len(lps))

        def _build(leaf_id: int) -> AgentLoopOutput:
            return AgentLoopOutput(
                prompt_ids=list(prompt_ids),
                response_ids=list(tokens),
                response_mask=[1] * len(tokens),
                response_logprobs=list(lps),
                multi_modal_data=multi_modal_data,
                num_turns=2,
                metrics=AgentLoopMetrics(),
                extra_fields={
                    "branch_token_mask": [0] * len(tokens),
                    "is_branching_fallback": 1,
                    "leaf_id": leaf_id,
                    "leaf_depth": 0,
                    "branching_diag": {},
                    "priv_ctx_meta": {},
                    "branching_fallback": reason,
                },
            )

        return [_build(i) for i in range(self.n_leaves)]
