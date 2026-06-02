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
from verl.utils.teacher_prompt import build_marker_text, build_ref_gt_messages
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
# Key   : deterministic id derived from (prompt_ids, generation_step).
# Value : asyncio.Future resolving to a list of 2^n_splits AgentLoopOutput
#         objects. The first sibling to enter run() creates the future and
#         launches _run_branching_pipeline; the rest await the same future
#         and pick their assigned leaf by branch_index.
#
# IMPORTANT: This cache lives in the AgentLoopWorker process (one Ray actor).
# A single PPO step's batch is dispatched through one AgentLoopWorker instance
# at a time, so all 8 siblings of a uid land in the same dict. Across PPO
# steps the cache grows unboundedly — _branching_cache_clear() is provided for
# the manager to call between steps.

_BRANCHING_CACHE: dict[str, asyncio.Future] = {}
_BRANCHING_CACHE_LOCK = asyncio.Lock()


def _branching_cache_clear() -> None:
    """Clear the module-level coordination cache. Call between PPO steps."""
    _BRANCHING_CACHE.clear()


def _branching_cache_key(prompt_ids: list[int], generation_step: int) -> str:
    """Deterministic cache key for sibling coordination.

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


# ---------------------------------------------------------------------------
# Branch-index assignment.
# ---------------------------------------------------------------------------
# Each sibling row needs to know which leaf (0..2^n_splits - 1) it should
# return. We support two assignment modes:
#   - kwargs["branch_index"]: caller-provided (e.g. via dataset metadata).
#   - per-uid round-robin counter: when not provided, we derive it from a
#     module-level monotonic counter keyed by the cache key. The first 8
#     sibling rows for a given key are assigned indices 0..7 in arrival order.

_BRANCHING_INDEX_COUNTERS: dict[str, int] = {}
_BRANCHING_INDEX_COUNTERS_LOCK = asyncio.Lock()


async def _claim_branch_index(cache_key: str, n_leaves: int) -> int:
    async with _BRANCHING_INDEX_COUNTERS_LOCK:
        idx = _BRANCHING_INDEX_COUNTERS.get(cache_key, 0)
        if idx >= n_leaves:
            # Sibling pool exceeded the tree leaf count — happens if the user
            # configured rollout.n > 2^n_splits. Wrap-around assigns extras
            # round-robin; the trainer will still see rollout.n rows.
            idx = idx % n_leaves
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
        return 2 ** int(self.branching_cfg.n_splits)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
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

        # Cache key shared across the 2^n_splits sibling rows of this prompt.
        gen_step = int(kwargs.get("generation_step", 0))
        cache_key = _branching_cache_key(prompt_ids, gen_step)

        # Each sibling row claims a leaf index. If rollout.n > 2^n_splits the
        # extras wrap around (duplicate leaves); typical config aligns the two.
        # If branch_index is explicit (callers pre-assign), we don't bump the
        # counter — bumping with no consumer was redundant and risked drift
        # against claimed indices.
        explicit_index = int(kwargs.get("branch_index", -1))
        if explicit_index >= 0:
            branch_index = explicit_index
            counter_bumped = False
        else:
            branch_index = await _claim_branch_index(cache_key, self.n_leaves)
            counter_bumped = True

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
            # blast-radius into all 8 sibling rows + the entire batch chunk.
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
            # clean even if clear_branching_cache is somehow skipped.
            if counter_bumped:
                async with _BRANCHING_INDEX_COUNTERS_LOCK:
                    if cache_key in _BRANCHING_INDEX_COUNTERS:
                        _BRANCHING_INDEX_COUNTERS[cache_key] = max(
                            0, _BRANCHING_INDEX_COUNTERS[cache_key] - 1
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
    ) -> list[AgentLoopOutput]:
        """Generate 2^n_splits shared-prefix leaves via teacher-guided branching.

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
        traj_id = uuid4().hex

        priv_ctx_ids, priv_ctx_meta = await self._build_privileged_context(
            prompt_ids=prompt_ids, kwargs=kwargs, images=images, videos=videos,
        )

        student_sp = self._student_sampling_params(sampling_params, max_tokens=None)

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
        )

        # 3. Flatten + pad.
        leaves = self._flatten_tree_to_leaves(
            root=root, multi_modal_data=multi_modal_data, diag=diag, priv_ctx_meta=priv_ctx_meta,
        )
        leaves = self._pad_leaves_to_n(leaves, self.n_leaves)
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
    ) -> dict[str, Any]:
        """Coerce sampling_params to the student-call shape: logprobs=K (int)
        and (optionally) override max_tokens. Caller passes max_tokens=None to
        let vllm_async_server compute the default from response_length."""
        sp = dict(sampling_params)
        sp["logprobs"] = int(self.branching_cfg.top_k)
        if max_tokens is not None:
            sp["max_tokens"] = max_tokens
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
        p = positions[0]  # earliest by chronological sort

        # Teacher branch-point query.
        student_top = segment_top_logprobs[p] if p < len(segment_top_logprobs) else {}
        teacher_top = await self._query_teacher_topk(
            priv_ctx_ids=priv_ctx_ids,
            prefix_ids=prefix_ids,
            segment_tokens=segment_tokens,
            branch_pos=p,
            traj_id=traj_id,
            images=images,
            videos=videos,
        )
        if not student_top or teacher_top is None:
            return node  # no signal to branch on

        # Sort student candidates by descending logprob — teacher picks within
        # the top ``teacher_pick_top_k`` (default = top_k, i.e. all candidates).
        # Slicing here lets us request a deep student top-K for entropy
        # estimation while still presenting only the high-probability core
        # to the teacher for branch selection (avoids OOD branches).
        student_topk_pairs = sorted(student_top.items(), key=lambda kv: -kv[1])
        _pick_k_override = cfg.get("teacher_pick_top_k", None)
        pick_k = int(_pick_k_override) if _pick_k_override is not None else int(cfg.top_k)
        if pick_k > 0:
            student_topk_pairs = student_topk_pairs[:pick_k]

        # Coverage diagnostic: how many of student's pick_k candidates appear
        # in teacher's top-K? If teacher_top_k is too small the intersection
        # is thin and many student candidates are silently dropped from the
        # argmax/argmin selection. We aggregate (intersect_total, pick_total)
        # across all split decisions and surface coverage_ratio in SwanLab.
        intersect_count = sum(1 for tok, _ in student_topk_pairs if tok in teacher_top)
        diag["teacher_coverage_intersect_total"] = (
            diag.get("teacher_coverage_intersect_total", 0) + intersect_count
        )
        diag["teacher_coverage_pick_total"] = (
            diag.get("teacher_coverage_pick_total", 0) + len(student_topk_pairs)
        )

        choice = pick_teacher_branches(
            student_topk_pairs, teacher_top,
            fallback_to_student=bool(cfg.fallback_to_student_topk),
        )
        if choice == (None, None):
            diag["teacher_intersect_misses"] += 1
            return node
        (pos_tok, _), (neg_tok, _) = choice
        if bool(cfg.require_distinct_branches) and pos_tok == neg_tok:
            return node
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
            )
            # Mark the branch token position in the child's mask.
            if child_node.segment_branch_mask:
                child_node.segment_branch_mask[0] = 1
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

        def _visit(node: BranchNode) -> None:
            saved = (
                len(path_tokens), len(path_entropies),
                len(path_logprobs), len(path_branch_mask),
            )
            path_tokens.extend(node.segment_tokens)
            path_entropies.extend(node.segment_entropies)
            path_logprobs.extend(node.segment_logprobs)
            path_branch_mask.extend(node.segment_branch_mask)

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

        _visit(root)
        # Sanity check via the structural collector.
        leaves = collect_leaves(root)
        assert len(outputs) == len(leaves), (
            f"Tree flatten produced {len(outputs)} outputs but tree has {len(leaves)} leaves"
        )
        return outputs

    def _pad_leaves_to_n(self, leaves: list[AgentLoopOutput], n: int) -> list[AgentLoopOutput]:
        """Ensure exactly ``n`` leaves by duplicating the last produced leaf
        when the tree was shallower than expected. Never produce empty leaves
        — torch.cat in _postprocess would explode on shape mismatch.

        IMPORTANT: each duplicate must own its own ``extra_fields`` dict
        (Pydantic ``model_copy(deep=False)`` shares dict references). Without
        this, downstream mutation in ``_agent_loop_postprocess`` (which
        ``pop()``s ``branch_token_mask`` and writes ``raw_prompt``) would
        clobber siblings and crash ``torch.cat`` in the aggregator.
        """
        if len(leaves) >= n:
            return leaves[:n]
        if not leaves:
            raise RuntimeError("BranchingAgentLoop._pad_leaves_to_n: no leaves to pad from")
        last = leaves[-1]
        pad_idx = 0
        while len(leaves) < n:
            pad_idx += 1
            leaves.append(self._clone_leaf(last, leaf_id_override=last.extra_fields.get("leaf_id", 0), pad_index=pad_idx))
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
    ) -> list[AgentLoopOutput]:
        """Fire ``n_leaves`` INDEPENDENT student rollouts in PARALLEL and wrap
        each as a leaf. Used when the branching pipeline raises but a plain
        student rollout still works.

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

        outs = await asyncio.gather(*(_one_call(i) for i in range(self.n_leaves)))

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
