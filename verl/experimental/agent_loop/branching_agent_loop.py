# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Teacher-guided branching rollout — agent loop scaffold.

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

Phase 1c.1 (this commit): scaffold + coordination layer + a STUB pipeline
that mirrors SingleTurnAgentLoop behaviour when the cache is hot. The real
generate→detect→branch logic ships in Phase 1c.2.

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
from verl.workers.rollout.vllm_rollout.branching_utils import (
    BranchNode,
    collect_leaves,
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

        # Cache key shared across the 2^n_splits sibling rows of this prompt.
        gen_step = int(kwargs.get("generation_step", 0))
        cache_key = _branching_cache_key(prompt_ids, gen_step)

        # Each sibling row claims a leaf index. If rollout.n > 2^n_splits the
        # extras wrap around (duplicate leaves); typical config aligns the two.
        branch_index = int(kwargs.get("branch_index", -1))
        if branch_index < 0:
            branch_index = await _claim_branch_index(cache_key, self.n_leaves)
        else:
            # Even with explicit indices we still bump the counter so that
            # _branching_index_counters_clear keeps a consistent picture.
            await _claim_branch_index(cache_key, self.n_leaves)

        metrics: dict[str, Any] = {}
        # Coordination: at most one sibling actually runs the pipeline; others await.
        async with _BRANCHING_CACHE_LOCK:
            future = _BRANCHING_CACHE.get(cache_key)
            if future is None:
                future = asyncio.get_event_loop().create_future()
                _BRANCHING_CACHE[cache_key] = future
                am_owner = True
            else:
                am_owner = False

        if am_owner:
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
                future.set_exception(e)
                raise

        leaves = await future

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
        """Generate 2^n_splits leaves with shared prefixes. STUB in 1c.1.

        Phase 1c.1 fallback: until the real branching pipeline lands in 1c.2,
        produce ``n_leaves`` independent rollouts (no branching, no shared
        prefix). This keeps the pipeline shape compatible with downstream code
        and lets us land 1c.1 incrementally. With branching.enabled=False at
        the YAML level the regular SingleTurnAgentLoop is used instead, so
        this fallback only fires when the user explicitly opted in.
        """
        # Inject logprobs=top_k so when 1c.2 lands we already have the data the
        # real pipeline needs.
        sp = dict(sampling_params)
        sp.setdefault("logprobs", int(self.branching_cfg.top_k))

        leaves: list[AgentLoopOutput] = []
        for _ in range(self.n_leaves):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sp,
                image_data=images,
                video_data=videos,
            )
            response_mask = [1] * len(output.token_ids)
            leaves.append(
                AgentLoopOutput(
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
                    metrics=AgentLoopMetrics(),
                    extra_fields={
                        "branching_phase": "1c.1_stub",
                        "branching_top_k": int(self.branching_cfg.top_k),
                    },
                )
            )
        return leaves

    # ------------------------------------------------------------------
    # Helpers (used by 1c.2 — kept here so tests can target them).
    # ------------------------------------------------------------------

    @staticmethod
    def _build_branching_tree(prompt_ids: list[int]) -> BranchNode:
        """Construct the root of the branching tree."""
        return BranchNode(
            prefix_tokens=list(prompt_ids),
            segment_tokens=[],
            segment_entropies=[],
            segment_logprobs=[],
            segment_branch_mask=[],
            depth=0,
            is_leaf=True,
        )

    def _flatten_tree_to_leaves(self, root: BranchNode) -> list[AgentLoopOutput]:
        """Walk the tree and emit one AgentLoopOutput per leaf with shared prefix
        token vectors assembled along the path. Used by 1c.2.

        Each leaf's response_ids = (root prefix excluded) all segment_tokens
        concatenated along the root → leaf path. The branch token chosen by
        the teacher at each split is included in the next segment's first
        position with a 1 in segment_branch_mask.
        """
        outputs: list[AgentLoopOutput] = []
        # DFS, accumulating along the path.
        path_tokens: list[int] = []
        path_entropies: list[float] = []
        path_logprobs: list[float] = []
        path_branch_mask: list[int] = []

        def _visit(node: BranchNode) -> None:
            nonlocal path_tokens, path_entropies, path_logprobs, path_branch_mask
            saved = (
                len(path_tokens), len(path_entropies),
                len(path_logprobs), len(path_branch_mask),
            )
            path_tokens.extend(node.segment_tokens)
            path_entropies.extend(node.segment_entropies)
            path_logprobs.extend(node.segment_logprobs)
            path_branch_mask.extend(node.segment_branch_mask)

            if node.is_leaf:
                outputs.append(
                    AgentLoopOutput(
                        prompt_ids=root.prefix_tokens,
                        response_ids=list(path_tokens[: self.response_length]),
                        response_mask=[1] * min(len(path_tokens), self.response_length),
                        response_logprobs=list(path_logprobs[: self.response_length]),
                        multi_modal_data=None,
                        num_turns=2,
                        metrics=AgentLoopMetrics(),
                        extra_fields={
                            "branch_token_mask": list(path_branch_mask[: self.response_length]),
                            "leaf_id": node.leaf_id,
                            "leaf_depth": node.depth,
                        },
                    )
                )
            else:
                for child in node.children:
                    _visit(child)

            # Backtrack.
            (lt, le, lp, lb) = saved
            del path_tokens[lt:]
            del path_entropies[le:]
            del path_logprobs[lp:]
            del path_branch_mask[lb:]

        _visit(root)
        # Stable left-to-right leaf order (DFS already ensures this; sanity check).
        leaves = collect_leaves(root)
        assert len(outputs) == len(leaves), (
            f"Tree flatten produced {len(outputs)} outputs but tree has {len(leaves)} leaves"
        )
        return outputs
