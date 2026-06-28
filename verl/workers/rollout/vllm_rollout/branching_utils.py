# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pure-python building blocks for teacher-guided branching rollout.

Three concerns, intentionally separated so they can be unit-tested without vLLM:

1. RunningEntropyDetector — rolling z-score of per-token entropy with a
   protect window. Caller asks `is_decision(t, H_t)`, gets a boolean.
2. entropy_from_topk_logprobs — truncated entropy estimator from a vLLM
   `Logprobs` dict (token_id -> logprob).
3. pick_teacher_branches — given student top-K and teacher top-K logprobs at
   a candidate branch position, select teacher's argmax and argmin within
   the intersection. Handles missing-from-teacher tokens and ties.

See research/teacher_branching_rollout.md for the broader design.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Tuple


# ---------------------------------------------------------------------------
# Entropy estimator
# ---------------------------------------------------------------------------


def entropy_from_topk_logprobs(topk_logprobs: Mapping[int, float]) -> float:
    """Truncated entropy from a top-K logprob dict.

    vLLM returns the realized-token logprob plus its top-K alternatives as a
    dict ``{token_id: logprob}``. We treat these K probabilities as a
    sub-distribution and compute the entropy on the renormalised support. This
    underestimates the true entropy (the tail is dropped) but the underestimate
    is monotone in K, so the running z-score detector still works as long as K
    is held constant across all positions.

    Args:
        topk_logprobs: token_id -> logprob (natural log).

    Returns:
        Truncated entropy in nats (>= 0). Returns 0.0 if input is empty.
    """
    if not topk_logprobs:
        return 0.0
    # log-sum-exp normalisation
    logprobs = list(topk_logprobs.values())
    max_lp = max(logprobs)
    norm = math.log(sum(math.exp(lp - max_lp) for lp in logprobs)) + max_lp
    h = 0.0
    for lp in logprobs:
        p = math.exp(lp - norm)
        if p > 0.0:
            h -= p * (lp - norm)
    return h


# ---------------------------------------------------------------------------
# Running z-score detector
# ---------------------------------------------------------------------------


@dataclass
class RunningEntropyDetector:
    """Rolling-window z-score detector for decision tokens.

    A token at position t (with entropy H_t) is called a *decision token* iff
    ``t >= protect_window`` AND ``H_t > running_mean + sigma * running_std``,
    where the running statistics are computed over the trailing
    ``window_size`` tokens (or all tokens seen so far, whichever is smaller).

    The detector is purely incremental — `update(H_t)` is O(1).

    The caller controls the sigma threshold; an adaptive σ-relaxation policy
    (lower σ until enough decision tokens are found) lives in the rollout
    orchestrator, not here.
    """

    window_size: int = 20
    protect_window: int = 20
    sigma: float = 2.0

    _entropies: deque = field(default_factory=lambda: deque())
    _sum: float = 0.0
    _sum_sq: float = 0.0
    _t: int = 0  # number of tokens observed so far

    def __post_init__(self) -> None:
        if self.window_size <= 1:
            raise ValueError(f"window_size must be > 1, got {self.window_size}")
        if self.protect_window < 0:
            raise ValueError(f"protect_window must be >= 0, got {self.protect_window}")
        # Re-create deque with proper maxlen (dataclass default_factory can't capture self).
        self._entropies = deque(maxlen=self.window_size)

    def reset(self) -> None:
        self._entropies.clear()
        self._sum = 0.0
        self._sum_sq = 0.0
        self._t = 0

    def is_decision(self, h_t: float) -> bool:
        """Update the rolling stats with H_t and return whether t is a decision token.

        Note: stats are updated AFTER the threshold test, so the test for
        token t uses statistics over tokens 0..t-1 (i.e., excludes the current
        token). This matches the user's spec: "第 t 个 token 熵 > running_mean
        + k * running_std", where running_* is over previously seen tokens.
        """
        t = self._t
        if t < self.protect_window or len(self._entropies) < 2:
            decision = False
        else:
            n = len(self._entropies)
            mean = self._sum / n
            var = max(0.0, self._sum_sq / n - mean * mean)
            std = math.sqrt(var)
            decision = h_t > mean + self.sigma * std

        # Update rolling stats with h_t.
        if len(self._entropies) == self.window_size:
            old = self._entropies[0]  # will be evicted by appendleft semantics
            self._sum -= old
            self._sum_sq -= old * old
        self._entropies.append(h_t)
        self._sum += h_t
        self._sum_sq += h_t * h_t
        self._t += 1
        return decision

    @property
    def num_observed(self) -> int:
        return self._t


def find_decision_positions(
    entropies: list[float],
    window_size: int = 20,
    protect_window: int = 20,
    sigma: float = 2.0,
) -> list[int]:
    """Functional convenience: scan a full entropy sequence and return all
    positions flagged as decision tokens at the given σ.

    Used by the σ-relaxation loop: scan once at high σ, if too few hits drop σ
    by half a step and rescan, repeating until the target count is reached.
    """
    det = RunningEntropyDetector(
        window_size=window_size,
        protect_window=protect_window,
        sigma=sigma,
    )
    return [t for t, h in enumerate(entropies) if det.is_decision(h)]


def find_decision_positions_with_sigma_relaxation(
    entropies: list[float],
    target_count: int,
    window_size: int = 20,
    protect_window: int = 20,
    sigma_start: float = 2.0,
    sigma_step: float = 0.5,
    sigma_floor: float = 0.5,
) -> Tuple[list[int], float, int]:
    """σ-relaxation loop: scan with sigma_start, lower σ by sigma_step until
    at least ``target_count`` decision positions are found, or σ ≤ sigma_floor.

    Returns the top-``target_count`` positions ranked by (relative entropy
    excess) descending, plus the σ value that produced the final list and the
    number of relaxation steps used.

    If even at sigma_floor we can't find target_count positions, return what
    we have (may be < target_count). Caller decides the fallback.
    """
    sigma = sigma_start
    relaxations = 0
    last_positions: list[int] = []
    while sigma >= sigma_floor:
        positions = find_decision_positions(
            entropies, window_size=window_size, protect_window=protect_window, sigma=sigma,
        )
        last_positions = positions
        if len(positions) >= target_count:
            break
        sigma -= sigma_step
        relaxations += 1
    # If we found more than target_count, rank by descending entropy and take the top.
    if len(last_positions) > target_count:
        last_positions = sorted(last_positions, key=lambda p: -entropies[p])[:target_count]
        last_positions.sort()  # restore chronological order
    return last_positions, max(sigma, sigma_floor), relaxations


# ---------------------------------------------------------------------------
# Teacher branch-point selector
# ---------------------------------------------------------------------------


def pick_teacher_branches(
    student_topk: Iterable[Tuple[int, float]],
    teacher_topk: Mapping[int, float],
    *,
    fallback_to_student: bool = True,
) -> Tuple[Optional[Tuple[int, float]], Optional[Tuple[int, float]]]:
    """Pick teacher's argmax and argmin token from student's top-K candidates.

    Args:
        student_topk: iterable of (token_id, student_logprob) pairs — student's
            top-K candidates at the branch position.
        teacher_topk: dict token_id -> teacher_logprob — teacher's top-K' at the
            same position under the privileged context (K' is typically >= K
            so the intersection has at least 2 entries).
        fallback_to_student: if True and the intersection has < 2 valid
            entries, fall back to student's top-1 (as argmax) and top-2 (as
            argmin) so the rollout can still split. If False, return (None, None)
            in that case and let the caller decide.

    Returns:
        ((tok_max, teacher_logprob_max), (tok_min, teacher_logprob_min)) —
        always two distinct token ids, or (None, None) if the fallback is off
        and the intersection is too small.
    """
    student_list = list(student_topk)
    if len(student_list) < 2:
        return None, None

    # Score each student candidate by teacher logprob (or -inf if absent).
    scored: list[tuple[int, float, float, int]] = []  # (tok, teacher_lp, student_lp, idx)
    for idx, (tok, student_lp) in enumerate(student_list):
        if tok in teacher_topk:
            scored.append((tok, teacher_topk[tok], student_lp, idx))
    if len(scored) >= 2:
        # Sort by teacher logprob; argmax = highest, argmin = lowest within intersection.
        scored.sort(key=lambda x: x[1])
        tok_min, lp_min = scored[0][0], scored[0][1]
        tok_max, lp_max = scored[-1][0], scored[-1][1]
        if tok_min != tok_max:
            return (tok_max, lp_max), (tok_min, lp_min)

    if not fallback_to_student:
        return None, None

    # Fallback: student top-1 and top-2 (by student logprob).
    student_list.sort(key=lambda x: -x[1])
    tok_max = student_list[0][0]
    tok_min = student_list[1][0]
    if tok_max == tok_min:
        return None, None
    lp_max = teacher_topk.get(tok_max, float("-inf"))
    lp_min = teacher_topk.get(tok_min, float("-inf"))
    return (tok_max, lp_max), (tok_min, lp_min)


# ---------------------------------------------------------------------------
# Branching tree state
# ---------------------------------------------------------------------------


@dataclass
class BranchNode:
    """One node of the branching tree.

    A leaf is a node with ``is_leaf=True``; the rollout orchestrator collects
    these into the final batch. An internal node has ``branch_position`` and
    ``children`` (length 2: argmax, argmin).
    """

    # Token sequence from prompt end up to but not including this node's first generated token.
    # For the root this is empty; for a child it is parent.prefix_tokens + parent.segment_tokens
    # up to (and including) the branch token chosen for this child.
    prefix_tokens: list[int] = field(default_factory=list)
    # Tokens generated in this segment (after the prefix), excluding the prefix's branch token.
    segment_tokens: list[int] = field(default_factory=list)
    # Per-token entropy for tokens in segment_tokens (same length).
    segment_entropies: list[float] = field(default_factory=list)
    # Per-token realized-token logprob for tokens in segment_tokens (same length).
    segment_logprobs: list[float] = field(default_factory=list)
    # Indicator vector: 1 iff the token at this position was teacher-injected
    # (i.e., this token is the branch token chosen by the teacher rather than
    # sampled by the student). Same length as segment_tokens. Always 0 except
    # possibly for position 0 of a child node.
    segment_branch_mask: list[int] = field(default_factory=list)
    # Branch position within segment_tokens that triggered the children (None for leaves).
    branch_position: Optional[int] = None
    # Children (length 2: [argmax_child, argmin_child]) or empty if leaf.
    children: list["BranchNode"] = field(default_factory=list)
    is_leaf: bool = True
    depth: int = 0
    leaf_id: Optional[int] = None  # assigned in finalize()
    # DPO: teacher logprobs at the branch point for reward shaping.
    # teacher_branch_logprob = teacher logprob of THIS child's branch token.
    # teacher_sibling_logprob = teacher logprob of the SIBLING's branch token.
    # DPO preference signal = teacher_branch_logprob - teacher_sibling_logprob.
    teacher_branch_logprob: Optional[float] = None
    teacher_sibling_logprob: Optional[float] = None

    def full_token_sequence(self) -> list[int]:
        return list(self.prefix_tokens) + list(self.segment_tokens)

    def full_entropy_sequence(self, parent_entropies: Optional[list[float]] = None) -> list[float]:
        if parent_entropies is None:
            return list(self.segment_entropies)
        return list(parent_entropies) + list(self.segment_entropies)


def collect_leaves(root: BranchNode) -> list[BranchNode]:
    """DFS collection of all leaf nodes in left-to-right (argmax-before-argmin) order."""
    out: list[BranchNode] = []
    stack: list[BranchNode] = [root]
    # Iterate manually to control order.
    def _visit(node: BranchNode) -> None:
        if node.is_leaf:
            out.append(node)
        else:
            for child in node.children:
                _visit(child)

    _visit(root)
    return out


def assemble_leaf_token_sequence(leaf: BranchNode) -> tuple[list[int], list[float], list[float], list[int]]:
    """Walk from root down to ``leaf`` and assemble the full per-token vectors.

    Returns:
        (token_ids, entropies, realized_logprobs, branch_mask) — all of the
        same length, covering only the response (not the prompt).
    """
    # ``prefix_tokens`` already encodes the path from prompt-end down to this leaf's
    # first segment token, and segment_tokens carries the rest.
    tokens = list(leaf.prefix_tokens) + list(leaf.segment_tokens)
    # Reconstruct entropies/logprobs/mask along the same path.
    # NOTE: prefix_tokens is opaque to the leaf; the orchestrator must keep a
    # parallel list during construction. Rather than walking the tree here, we
    # require the orchestrator to populate full_* fields explicitly. This
    # function is therefore a pure structural helper; see BranchingAgentLoop
    # for the actual assembly.
    raise NotImplementedError(
        "assemble_leaf_token_sequence is a placeholder; the orchestrator builds "
        "the per-leaf vectors directly during BranchNode construction."
    )
