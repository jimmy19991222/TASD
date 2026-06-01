# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CPU-only tests for the Phase 2 branching loss / advantage hooks.

Pure-python (stdlib only) to keep the unit suite runnable on a developer
laptop without torch installed. The actual torch-using code paths in
verl/workers/actor/dp_actor.py and verl/trainer/ppo/core_algos.py are
exercised with element-wise equivalent logic here.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. branch_token_loss_mode ablation logic
#    Replicates dp_actor.py:effective_response_mask construction element-wise.
# ---------------------------------------------------------------------------


def _apply_branch_loss_mode(response_mask, branch_token_mask, mode):
    if mode == "all":
        return [list(row) for row in response_mask]
    out = []
    for rm_row, btm_row in zip(response_mask, branch_token_mask):
        if mode == "mask":
            out.append([rm * (1 - btm) for rm, btm in zip(rm_row, btm_row)])
        elif mode == "only":
            out.append([rm * btm for rm, btm in zip(rm_row, btm_row)])
        else:
            raise ValueError(mode)
    return out


def test_branch_token_loss_mode_all():
    rm = [[1, 1, 1, 1, 1, 0, 0]]
    btm = [[0, 0, 1, 0, 1, 0, 0]]
    eff = _apply_branch_loss_mode(rm, btm, "all")
    assert eff == rm
    print("test_branch_token_loss_mode_all PASS")


def test_branch_token_loss_mode_mask_zeros_branch_tokens():
    rm = [[1, 1, 1, 1, 1, 0, 0]]
    btm = [[0, 0, 1, 0, 1, 0, 0]]
    eff = _apply_branch_loss_mode(rm, btm, "mask")
    assert eff == [[1, 1, 0, 1, 0, 0, 0]], eff
    print("test_branch_token_loss_mode_mask_zeros_branch_tokens PASS")


def test_branch_token_loss_mode_only_keeps_only_branch_tokens():
    rm = [[1, 1, 1, 1, 1, 0, 0]]
    btm = [[0, 0, 1, 0, 1, 0, 0]]
    eff = _apply_branch_loss_mode(rm, btm, "only")
    assert eff == [[0, 0, 1, 0, 1, 0, 0]], eff
    print("test_branch_token_loss_mode_only_keeps_only_branch_tokens PASS")


def test_branch_token_loss_mode_pad_positions_stay_zero():
    """response_mask=0 (pad) positions must remain 0 in every mode, even when
    branch_token_mask flags them."""
    rm = [[1, 1, 0, 0]]
    btm = [[0, 1, 1, 1]]
    for m in ("all", "mask", "only"):
        eff = _apply_branch_loss_mode(rm, btm, m)
        assert eff[0][2] == 0 and eff[0][3] == 0, (m, eff)
    print("test_branch_token_loss_mode_pad_positions_stay_zero PASS")


def test_branch_token_loss_mode_invalid_rejected():
    try:
        _apply_branch_loss_mode([[1]], [[0]], "weighted")
        raise AssertionError("should have rejected 'weighted'")
    except ValueError:
        pass
    print("test_branch_token_loss_mode_invalid_rejected PASS")


# ---------------------------------------------------------------------------
# 2. adv_std_floor in compute_grpo_outcome_advantage
# ---------------------------------------------------------------------------


def _grpo_advantage_with_floor(
    rewards_per_row,
    index,
    *,
    norm_adv_by_std_in_grpo=True,
    adv_std_floor=0.0,
    epsilon=1e-6,
):
    """Pure-python equivalent of compute_grpo_outcome_advantage.

    rewards_per_row: list[float] (already summed over response axis)
    index: list of group keys, same length
    """
    id2score = defaultdict(list)
    for i, r in enumerate(rewards_per_row):
        id2score[index[i]].append(r)
    id2mean, id2std = {}, {}
    for k, vs in id2score.items():
        if len(vs) == 1:
            id2mean[k] = 0.0
            id2std[k] = 1.0
        else:
            id2mean[k] = statistics.mean(vs)
            id2std[k] = statistics.stdev(vs)  # n-1 (matches torch.std default)
            if adv_std_floor > 0.0:
                id2std[k] = max(id2std[k], adv_std_floor)
    out = []
    for i, r in enumerate(rewards_per_row):
        if norm_adv_by_std_in_grpo:
            out.append((r - id2mean[index[i]]) / (id2std[index[i]] + epsilon))
        else:
            out.append(r - id2mean[index[i]])
    return out


def test_adv_std_floor_zero_collapsed_group_is_bounded():
    rewards = [1.0] * 8 + [0.5] * 8
    idx = ["uA"] * 8 + ["uB"] * 8
    adv = _grpo_advantage_with_floor(rewards, idx, adv_std_floor=0.0)
    # All siblings of uA share reward 1.0 -> std=0 -> advantage=(1-1)/eps=0.
    # Same for uB. No blowup, just zero advantage.
    for a in adv:
        assert abs(a) < 1e3, a
    print("test_adv_std_floor_zero_collapsed_group_is_bounded PASS")


def test_adv_std_floor_downweights_low_variance_groups():
    """The (score-mean)/std ratio is naturally bounded by sqrt(n) for any
    finite group, so adv_std_floor does NOT prevent magnitude blowup —
    instead, it DOWNWEIGHTS low-variance groups by replacing std with floor
    when std < floor. Verify that downweighting:

      - without floor : outlier_adv ≈ sqrt(7) ≈ 2.65 (close to the ceiling)
      - with floor=0.05 : outlier_adv ≈ dev / floor ≈ 0.0175 (much smaller)

    This is the desired behaviour: noisy-but-tight reward groups (siblings
    that almost agree) contribute proportionally less to the gradient than
    genuinely high-variance groups.
    """
    rewards = [1.0] * 7 + [1.001]
    idx = ["uA"] * 8
    adv_no = _grpo_advantage_with_floor(rewards, idx, adv_std_floor=0.0)
    adv_fl = _grpo_advantage_with_floor(rewards, idx, adv_std_floor=0.05)
    outlier_no = abs(adv_no[7])
    outlier_fl = abs(adv_fl[7])
    # Without floor: bounded by sqrt(n)=2.83, but >= 1.0 — i.e., the low-std
    # group's lone outlier still gets a strong gradient signal.
    assert 1.0 < outlier_no < 3.0, ("no-floor outlier in [1, 3]", outlier_no)
    # With floor: the gradient signal is downweighted ~100x.
    assert outlier_fl < 0.1, ("floor: should be downweighted", outlier_fl)
    # Ratio = (std_actual + eps) / (floor + eps) — confirm the downweight is
    # proportional to std/floor.
    expected_ratio = (statistics.stdev(rewards) + 1e-6) / (0.05 + 1e-6)
    measured_ratio = outlier_fl / outlier_no
    assert math.isclose(measured_ratio, expected_ratio, rel_tol=0.01), (
        measured_ratio, expected_ratio,
    )
    print(
        f"test_adv_std_floor_downweights_low_variance_groups PASS  "
        f"(no_floor={outlier_no:.3f}, floor={outlier_fl:.4f}, "
        f"ratio={measured_ratio:.4f} ≈ {expected_ratio:.4f})"
    )


def test_adv_std_floor_zero_preserves_legacy():
    rewards = [1.0, 0.5, 0.0, 0.5]
    idx = ["uA"] * 4
    adv = _grpo_advantage_with_floor(rewards, idx, adv_std_floor=0.0)
    mean = statistics.mean(rewards)
    std = statistics.stdev(rewards)
    expected = [(r - mean) / (std + 1e-6) for r in rewards]
    for a, e in zip(adv, expected):
        assert math.isclose(a, e, abs_tol=1e-7), (a, e)
    print("test_adv_std_floor_zero_preserves_legacy PASS")


def test_adv_std_floor_does_not_dampen_high_variance_groups():
    """When std >> floor, the floor must be a no-op."""
    rewards = [0.0, 0.5, 1.0, 0.5, 0.2, 0.8]
    idx = ["uA"] * 6
    adv_no = _grpo_advantage_with_floor(rewards, idx, adv_std_floor=0.0)
    adv_fl = _grpo_advantage_with_floor(rewards, idx, adv_std_floor=0.05)
    for a, b in zip(adv_no, adv_fl):
        assert math.isclose(a, b, abs_tol=1e-6), (a, b)
    print("test_adv_std_floor_does_not_dampen_high_variance_groups PASS")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    test_branch_token_loss_mode_all()
    test_branch_token_loss_mode_mask_zeros_branch_tokens()
    test_branch_token_loss_mode_only_keeps_only_branch_tokens()
    test_branch_token_loss_mode_pad_positions_stay_zero()
    test_branch_token_loss_mode_invalid_rejected()

    test_adv_std_floor_zero_collapsed_group_is_bounded()
    test_adv_std_floor_downweights_low_variance_groups()
    test_adv_std_floor_zero_preserves_legacy()
    test_adv_std_floor_does_not_dampen_high_variance_groups()

    print()
    print("ALL 9 PHASE-2 TESTS PASS")


if __name__ == "__main__":
    main()
