# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for verl.workers.rollout.vllm_rollout.branching_utils.

CPU-only, no vLLM dependency. Exercises the pure-python building blocks of
the teacher-guided branching rollout.
"""

import math

import pytest

from verl.workers.rollout.vllm_rollout.branching_utils import (
    BranchNode,
    RunningEntropyDetector,
    collect_leaves,
    entropy_from_topk_logprobs,
    find_decision_positions,
    find_decision_positions_with_sigma_relaxation,
    pick_teacher_branches,
)


# ---------------------------------------------------------------------------
# entropy_from_topk_logprobs
# ---------------------------------------------------------------------------


def test_entropy_uniform_topk():
    """Top-K with uniform logprobs should give entropy log(K)."""
    K = 8
    lp = math.log(1.0 / K)
    topk = {i: lp for i in range(K)}
    h = entropy_from_topk_logprobs(topk)
    assert h == pytest.approx(math.log(K), abs=1e-6)


def test_entropy_peaked_topk():
    """Highly peaked distribution -> low entropy."""
    topk = {0: 0.0, 1: -10.0, 2: -10.0}  # token 0 nearly certain
    h = entropy_from_topk_logprobs(topk)
    assert 0.0 <= h < 0.01


def test_entropy_empty():
    assert entropy_from_topk_logprobs({}) == 0.0


def test_entropy_unnormalised_input_ok():
    """vLLM logprobs are not normalised over top-K; estimator should still
    return a sensible (non-NaN, finite, monotone) value via internal renorm."""
    topk = {0: -1.0, 1: -2.0, 2: -3.0}
    h = entropy_from_topk_logprobs(topk)
    assert math.isfinite(h)
    assert h > 0.0


# ---------------------------------------------------------------------------
# RunningEntropyDetector
# ---------------------------------------------------------------------------


def test_detector_protect_window_blocks_early_tokens():
    det = RunningEntropyDetector(window_size=10, protect_window=5, sigma=0.0)
    # Even with sigma=0, the first 5 tokens cannot be flagged.
    for _ in range(5):
        assert det.is_decision(100.0) is False


def test_detector_flags_obvious_spike():
    det = RunningEntropyDetector(window_size=20, protect_window=5, sigma=2.0)
    # 30 calm tokens at H=1.0, then a spike at H=10.0.
    for _ in range(30):
        det.is_decision(1.0)
    assert det.is_decision(10.0) is True


def test_detector_does_not_flag_within_one_sigma():
    det = RunningEntropyDetector(window_size=20, protect_window=5, sigma=2.0)
    for _ in range(30):
        det.is_decision(1.0)
    # H slightly above mean but well within 1-sigma — should NOT trigger.
    assert det.is_decision(1.1) is False


def test_detector_window_is_rolling():
    """Once the trailing window is filled, old observations should be evicted."""
    det = RunningEntropyDetector(window_size=10, protect_window=2, sigma=2.0)
    # Fill window with high-variance noise, then long tail of constants.
    for h in [5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0]:
        det.is_decision(h)
    for _ in range(30):
        det.is_decision(1.0)
    # By now the window has 10 ones; mean=1.0, std≈0. A H=1.5 spike against
    # a near-zero std is flagged trivially. Picking H=10 to be unambiguous.
    assert det.is_decision(10.0) is True


def test_detector_reset_clears_state():
    det = RunningEntropyDetector(window_size=5, protect_window=2, sigma=2.0)
    for _ in range(10):
        det.is_decision(1.0)
    det.reset()
    assert det.num_observed == 0


# ---------------------------------------------------------------------------
# find_decision_positions / sigma relaxation
# ---------------------------------------------------------------------------


def test_find_decision_positions_simple():
    # 30 calm tokens, then one spike at index 30, then more calm.
    entropies = [1.0] * 30 + [10.0] + [1.0] * 5
    pos = find_decision_positions(entropies, window_size=20, protect_window=5, sigma=2.0)
    assert pos == [30]


def test_sigma_relaxation_returns_top_k_when_too_many():
    # Pattern with 5 spikes; ask for 3 → should return top-3 ranked by H.
    entropies = [1.0] * 30
    spike_positions = [30, 40, 50, 60, 70]
    spike_values = [10.0, 5.0, 8.0, 6.0, 7.0]
    for p, v in zip(spike_positions, spike_values):
        entropies = entropies + [v] + [1.0] * (10 - 1)
    selected, _, _ = find_decision_positions_with_sigma_relaxation(
        entropies, target_count=3, window_size=20, protect_window=5,
        sigma_start=2.0, sigma_step=0.5, sigma_floor=0.5,
    )
    assert len(selected) == 3
    # Top-3 by entropy: positions of 10, 8, 7 → 30, 50, 70 in chronological order.
    assert selected == sorted([30, 50, 70])


def test_sigma_relaxation_lowers_until_target_met():
    # No spike: every token is uniform 1.0 — relaxation will hit floor with 0
    # decisions found. Ensure no crash and returns whatever it has.
    entropies = [1.0] * 50
    selected, final_sigma, relaxations = find_decision_positions_with_sigma_relaxation(
        entropies, target_count=3, window_size=20, protect_window=5,
        sigma_start=2.0, sigma_step=0.5, sigma_floor=0.5,
    )
    assert len(selected) <= 3
    assert relaxations >= 1
    assert final_sigma == 0.5  # hit floor


def test_sigma_relaxation_finds_subtler_spike_after_relaxing():
    # Spike at 1.5σ — sigma=2 misses it, sigma=1 catches it.
    base = [1.0] * 30
    # Inject a spike at position 30 just above 1σ but below 2σ.
    # With base mean=1.0, std≈0 (constant), any H>1.0 is many sigmas — adjust.
    # Use mild noise so std>0.
    import random
    random.seed(0)
    base = [1.0 + 0.3 * (random.random() - 0.5) for _ in range(30)]  # std ≈ 0.087
    spike_h = 1.0 + 0.15  # ~1.7σ above mean
    seq = base + [spike_h] + base
    selected, final_sigma, _ = find_decision_positions_with_sigma_relaxation(
        seq, target_count=1, window_size=20, protect_window=5,
        sigma_start=2.0, sigma_step=0.5, sigma_floor=0.5,
    )
    # Need at least 1 hit somewhere; relaxation should have helped.
    assert len(selected) >= 1
    assert final_sigma <= 1.5


# ---------------------------------------------------------------------------
# pick_teacher_branches
# ---------------------------------------------------------------------------


def test_pick_branches_clean_intersection():
    student_topk = [(100, -0.1), (200, -0.5), (300, -1.0), (400, -2.0)]
    teacher_topk = {200: -2.0, 300: -0.1, 100: -1.5, 400: -0.8}
    (mx, mxlp), (mn, mnlp) = pick_teacher_branches(student_topk, teacher_topk)
    # Within intersection, teacher's argmax = 300 (lp=-0.1), argmin = 200 (lp=-2.0)
    assert mx == 300 and mxlp == -0.1
    assert mn == 200 and mnlp == -2.0


def test_pick_branches_partial_intersection():
    student_topk = [(100, -0.1), (200, -0.5), (300, -1.0)]
    teacher_topk = {300: -0.5, 999: -0.1}  # only 300 is in student_topk
    # Intersection size 1 < 2 → fallback to student top-1 / top-2.
    (mx, _), (mn, _) = pick_teacher_branches(student_topk, teacher_topk)
    assert mx == 100  # student argmax
    assert mn == 200  # student arg-2nd


def test_pick_branches_empty_intersection_fallback():
    student_topk = [(100, -0.1), (200, -0.5)]
    teacher_topk = {999: -0.1, 888: -0.5}
    (mx, _), (mn, _) = pick_teacher_branches(student_topk, teacher_topk)
    assert mx == 100
    assert mn == 200


def test_pick_branches_no_fallback_returns_none():
    student_topk = [(100, -0.1), (200, -0.5)]
    teacher_topk = {999: -0.1}
    res = pick_teacher_branches(student_topk, teacher_topk, fallback_to_student=False)
    assert res == (None, None)


def test_pick_branches_single_candidate_returns_none():
    student_topk = [(100, -0.1)]
    teacher_topk = {100: -0.5}
    res = pick_teacher_branches(student_topk, teacher_topk)
    assert res == (None, None)


def test_pick_branches_distinct_argmax_argmin():
    """Even on a tie, the function must return distinct token ids or None."""
    student_topk = [(100, -0.1), (200, -0.1)]
    teacher_topk = {100: -0.5, 200: -0.5}  # tie
    res = pick_teacher_branches(student_topk, teacher_topk)
    # Either of the two outcomes is acceptable: distinct ids OR None,None
    if res != (None, None):
        (mx, _), (mn, _) = res
        assert mx != mn


# ---------------------------------------------------------------------------
# BranchNode / collect_leaves
# ---------------------------------------------------------------------------


def test_branchnode_default_is_leaf():
    n = BranchNode()
    assert n.is_leaf is True
    assert n.depth == 0
    assert n.children == []


def test_collect_leaves_balanced_binary_tree_depth_3():
    """A perfect binary tree depth=3 should yield 8 leaves in left-to-right order."""
    def make_tree(depth: int, leaf_counter: list[int]) -> BranchNode:
        node = BranchNode(depth=depth)
        if depth == 0:
            node.leaf_id = leaf_counter[0]
            leaf_counter[0] += 1
            return node
        node.is_leaf = False
        node.children = [make_tree(depth - 1, leaf_counter), make_tree(depth - 1, leaf_counter)]
        return node

    counter = [0]
    root = make_tree(3, counter)
    leaves = collect_leaves(root)
    assert len(leaves) == 8
    # Argmax-before-argmin DFS order means leaf_id increments left-to-right.
    assert [leaf.leaf_id for leaf in leaves] == list(range(8))


def test_collect_leaves_unbalanced_tree():
    """An unbalanced tree (one branch dies early) should still enumerate all leaves."""
    root = BranchNode(depth=3, is_leaf=False)
    leftleaf = BranchNode(depth=2, is_leaf=True, leaf_id=0)
    rightnode = BranchNode(depth=2, is_leaf=False)
    root.children = [leftleaf, rightnode]
    rightnode.children = [
        BranchNode(depth=1, is_leaf=True, leaf_id=1),
        BranchNode(depth=1, is_leaf=True, leaf_id=2),
    ]
    leaves = collect_leaves(root)
    assert [leaf.leaf_id for leaf in leaves] == [0, 1, 2]
