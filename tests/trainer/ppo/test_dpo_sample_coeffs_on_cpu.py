# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gradient-equivalence test for the branching-DPO per-sample coefficient refactor.

The coupled pairwise DPO loss ``(1/P)Σ_i -logσ(β_i z_i)`` is replaced by a per-sample
weighted-logp surrogate ``Σ_s c_s·avg_logp_s`` whose coefficients (``dpo_coeff``) are
computed trainer-side by ``compute_dpo_sample_coeffs``. This test asserts the two have
the SAME gradient w.r.t. the policy logprobs when the coefficient's behavior policy
(``old_log_probs``) equals the current policy (the on-policy step condition) and N=1.
"""

from types import SimpleNamespace

import torch

from verl.trainer.ppo.utils import compute_dpo_sample_coeffs, suffix_avg_logp


def _make_batch(B, T, *, ref=False, teacher=False, stage1=False, seed=0):
    """Synthetic two-stage batch. Layout per prompt: [stage1, stage1, chosen, rejected]."""
    g = torch.Generator().manual_seed(seed)
    logp = torch.randn(B, T, dtype=torch.float64, generator=g)
    response_mask = torch.ones(B, T, dtype=torch.float64)
    # one branch point at t=1 for every row -> suffix is tokens t>1
    branch = torch.zeros(B, T, dtype=torch.float64)
    branch[:, 1] = 1.0

    n_prompts = B // 4
    is_s1 = torch.zeros(B, dtype=torch.long)
    pair_id = torch.full((B,), -1, dtype=torch.long)
    role = torch.full((B,), -1, dtype=torch.long)
    uid = []
    for p in range(n_prompts):
        base = p * 4
        is_s1[base] = 1
        is_s1[base + 1] = 1
        pair_id[base + 2] = 0
        role[base + 2] = 0  # chosen
        pair_id[base + 3] = 0
        role[base + 3] = 1  # rejected
        uid += [f"p{p}"] * 4

    tb = {
        "old_log_probs": logp.clone(),
        "response_mask": response_mask,
        "branch_token_mask": branch,
        "is_two_stage_stage1": is_s1,
        "dpo_pair_id": pair_id,
        "dpo_role": role,
    }
    if ref:
        tb["ref_log_prob"] = torch.randn(B, T, dtype=torch.float64, generator=g)
    if teacher:
        tb["teacher_branch_logprob"] = torch.randn(B, dtype=torch.float64, generator=g)
        tb["teacher_sibling_logprob"] = torch.randn(B, dtype=torch.float64, generator=g)
    batch = SimpleNamespace(batch=tb, non_tensor_batch={"uid": uid})
    return batch, logp


def _ref_loss(logp, batch, cfg):
    """Reference: the original coupled DPO loss (1/P)Σ -logσ(β z) [+ stage1 3-way]."""
    tb = batch.batch
    beta = float(cfg["dpo_coefficient"])
    avg_logp, suffix_mask = suffix_avg_logp(logp, tb["branch_token_mask"], tb["response_mask"])
    ref_avg = None
    if cfg.get("dpo_use_ref") and "ref_log_prob" in tb:
        ref_avg = (tb["ref_log_prob"] * suffix_mask).sum(1) / suffix_mask.sum(1).clamp_min(1.0)

    def adj(i):
        return avg_logp[i] - ref_avg[i] if ref_avg is not None else avg_logp[i]

    uid = batch.non_tensor_batch["uid"]
    role, pid, is_s1 = tb["dpo_role"], tb["dpo_pair_id"], tb["is_two_stage_stage1"]
    pairs = {}
    for i in range(logp.shape[0]):
        if is_s1[i] == 0 and int(pid[i]) >= 0 and int(role[i]) >= 0:
            pairs.setdefault((uid[i], int(pid[i])), {})[int(role[i])] = i
    cp = [(r[0], r[1]) for r in pairs.values() if 0 in r and 1 in r]

    losses = []
    for c_i, r_i in cp:
        z = adj(c_i) - adj(r_i)
        b = beta
        if cfg.get("dpo_teacher_guided_beta") and "teacher_branch_logprob" in tb:
            margin = (tb["teacher_branch_logprob"][c_i] - tb["teacher_sibling_logprob"][c_i]).detach()
            b = beta * float(torch.clamp(cfg["dpo_teacher_beta_alpha"] * margin,
                                         min=cfg["dpo_teacher_beta_min"], max=cfg["dpo_teacher_beta_max"]))
        losses.append(-torch.nn.functional.logsigmoid(b * z))
    total = torch.stack(losses).mean()

    if cfg.get("dpo_stage1_pair"):
        rep = {}
        for i in range(logp.shape[0]):
            if is_s1[i] == 1 and uid[i] not in rep:
                rep[uid[i]] = i
        best, worst = [], []
        for c_i, r_i in cp:
            if uid[c_i] in rep:
                m = rep[uid[c_i]]
                best.append(-torch.nn.functional.logsigmoid(beta * (adj(c_i) - adj(m))))
                worst.append(-torch.nn.functional.logsigmoid(beta * (adj(m) - adj(r_i))))
        if best:
            s1 = torch.stack(best).mean() + torch.stack(worst).mean()
            total = total + float(cfg["dpo_stage1_pair_weight"]) * s1
    return total


def _grad(fn, logp):
    x = logp.clone().requires_grad_(True)
    fn(x).backward()
    return x.grad


def _run_case(**kw):
    cfg = {
        "dpo_coefficient": 2.0,
        "dpo_use_ref": kw.get("ref", False),
        "dpo_teacher_guided_beta": kw.get("teacher", False),
        "dpo_teacher_beta_alpha": 1.0,
        "dpo_teacher_beta_min": 0.1,
        "dpo_teacher_beta_max": 3.0,
        "dpo_stage1_pair": kw.get("stage1", False),
        "dpo_stage1_pair_weight": 0.5,
    }
    batch, logp = _make_batch(12, 5, ref=kw.get("ref", False), teacher=kw.get("teacher", False),
                              stage1=kw.get("stage1", False), seed=kw.get("seed", 0))
    coeff, pair_member, metrics = compute_dpo_sample_coeffs(batch, cfg)
    assert coeff is not None
    P = pair_member.sum().item() / 2.0

    # Reference is the mean-over-pairs loss (1/P)Σ L_i; the actor forms the same
    # objective as (coeff·avg_logp).sum() / P (single mini-batch => P_mb == P).
    g_old = _grad(lambda x: _ref_loss(x, batch, cfg), logp)

    def surrogate(x):
        avg_logp, _ = suffix_avg_logp(x, batch.batch["branch_token_mask"], batch.batch["response_mask"])
        return (coeff * avg_logp).sum() / P

    g_new = _grad(surrogate, logp)
    torch.testing.assert_close(g_old, g_new, atol=1e-5, rtol=1e-4)
    return batch, cfg, metrics, pair_member


def test_grad_equiv_basic():
    _run_case()


def test_grad_equiv_with_ref():
    _run_case(ref=True)


def test_grad_equiv_teacher_beta():
    _run_case(teacher=True)


def test_grad_equiv_stage1_threeway():
    _, _, metrics, _ = _run_case(stage1=True)
    assert metrics["actor/dpo_stage1_pairs"] == 3.0  # 3 prompts each with a stage-1 rep


def test_grad_equiv_full():
    _run_case(ref=True, teacher=True, stage1=True, seed=7)


def test_pair_member_counts_pairs():
    _, _, metrics, pair_member = _run_case()
    assert pair_member.sum().item() / 2.0 == 3.0  # 3 stage-2 pairs
    assert metrics["actor/dpo_n_pairs"] == 3.0  # trainer-global count


def _cfg(**over):
    c = {"dpo_coefficient": 2.0, "dpo_use_ref": False, "dpo_teacher_guided_beta": False,
         "dpo_teacher_beta_alpha": 1.0, "dpo_teacher_beta_min": 0.1, "dpo_teacher_beta_max": 3.0,
         "dpo_stage1_pair": False, "dpo_stage1_pair_weight": 1.0}
    c.update(over)
    return c


def test_reward_filter_drops_inconsistent_pair():
    # Layout per prompt: [stage1, stage1, chosen@base+2, rejected@base+3]; 3 prompts.
    batch, logp = _make_batch(12, 5, seed=0)
    B, T = logp.shape
    tls = torch.full((B, T), 0.5, dtype=torch.float64)  # equal reward by default -> keep
    tls[2] = 0.0   # prompt 0 chosen: low reward
    tls[3] = 1.0   # prompt 0 rejected: high reward  => chosen < rejected -> drop
    batch.batch["token_level_scores"] = tls

    coeff, pair_member, metrics = compute_dpo_sample_coeffs(batch, _cfg(dpo_reward_filter=True))
    assert metrics["actor/dpo_n_pairs"] == 2.0
    assert metrics["actor/dpo_pairs_filtered"] == 1.0
    # dropped pair contributes nothing
    assert coeff[2].item() == 0.0 and coeff[3].item() == 0.0
    assert pair_member[2].item() == 0.0 and pair_member[3].item() == 0.0
    # a kept pair still active
    assert pair_member[6].item() == 1.0 and pair_member[7].item() == 1.0


def test_reward_filter_off_keeps_all():
    batch, logp = _make_batch(12, 5, seed=0)
    B, T = logp.shape
    tls = torch.full((B, T), 0.5, dtype=torch.float64)
    tls[2] = 0.0
    tls[3] = 1.0
    batch.batch["token_level_scores"] = tls
    coeff, pair_member, metrics = compute_dpo_sample_coeffs(batch, _cfg(dpo_reward_filter=False))
    assert metrics["actor/dpo_n_pairs"] == 3.0
    assert metrics["actor/dpo_pairs_filtered"] == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
