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

import warnings
from collections import defaultdict
from enum import Enum

import torch
from omegaconf import DictConfig

from verl.single_controller.base import Worker
from verl.trainer.ppo.core_algos import AdvantageEstimator

WorkerType = type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    Env = 7

    def __str__(self):
        return self._get_role_string()

    def _get_role_string(self):
        role_mapping = {
            Role.Actor: "actor",
            Role.Rollout: "rollout",
            Role.ActorRollout: "actor_rollout",
            Role.Critic: "critic",
            Role.RefPolicy: "ref",
            Role.RewardModel: "rm",
            Role.ActorRolloutRef: "actor_rollout_ref",
        }
        return role_mapping.get(self, self.name.lower())

    @classmethod
    def from_string(cls, name: str):
        string_mapping = {
            "actor": cls.Actor,
            "rollout": cls.Rollout,
            "actor_rollout": cls.ActorRollout,
            "critic": cls.Critic,
            "ref": cls.RefPolicy,
            "rm": cls.RewardModel,
            "actor_rollout_ref": cls.ActorRolloutRef,
        }
        role = string_mapping.get(name.lower())
        if role is None:
            raise ValueError(f"No Role found for string: {name}")
        return role


def need_reference_policy(
    config: DictConfig,
) -> bool:
    return (config.algorithm.use_kl_in_reward
            or config.actor_rollout_ref.actor.use_kl_loss
            or config.actor_rollout_ref.actor.policy_loss.get('dpo_use_ref', False))


def need_reward_model(
    role_worker_mapping: dict[Role, WorkerType],
) -> bool:
    """Given a role worker mapping, do we need reward model."""
    return Role.RewardModel in role_worker_mapping


def suffix_avg_logp(
    log_prob: torch.Tensor,
    branch_token_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average logprob over each sample's suffix (tokens after its last branch point).

    Shared by the trainer-side DPO coefficient computation (fed rollout/old logprobs)
    and the actor-side DPO loss (fed current-policy logprobs), so the two definitions
    can never drift. Mirrors the suffix mask that the original on-policy DPO loss used.

    Args:
        log_prob: [B, T] per-token logprobs.
        branch_token_mask: [B, T] 1 at branch positions (0 if no branch).
        response_mask: [B, T] valid response-token mask.

    Returns:
        (avg_logp[B], suffix_mask[B, T]) where avg_logp is the mean logprob over
        the suffix (positions strictly after the last branch position) intersected
        with the response mask.
    """
    T = log_prob.shape[1]
    device = log_prob.device
    positions = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
    has_branch = branch_token_mask.any(dim=1)  # [B]
    weighted = branch_token_mask * positions  # [B, T]
    last_branch_pos = weighted.max(dim=1).values.long()  # [B]
    last_branch_pos = torch.where(
        has_branch,
        last_branch_pos,
        torch.tensor(-1, device=device, dtype=last_branch_pos.dtype),
    )
    suffix_mask = (positions > last_branch_pos.unsqueeze(1)).to(log_prob.dtype)  # [B, T]
    suffix_mask = suffix_mask * response_mask
    logp_sum = (log_prob * suffix_mask).sum(dim=1)  # [B]
    suffix_len = suffix_mask.sum(dim=1).clamp_min(1.0)  # [B]
    avg_logp = logp_sum / suffix_len  # [B]
    return avg_logp, suffix_mask


def compute_dpo_sample_coeffs(batch, policy_loss_cfg):
    """Trainer-side pairing for branching DPO, reduced to a per-sample coefficient.

    The pairwise DPO loss ``L_i = -log σ(β_i z_i)`` has gradient
    ``∂L_i/∂θ = -w_i·β_i·(∂a_i/∂θ - ∂b_i/∂θ)`` with the detached weight
    ``w_i = σ(-β_i z_i)``. So the loss shares its gradient with the per-sample
    surrogate ``Σ_s c_s·avg_logp_s`` where a sample's coefficient sums
    ``∓ w_i β_i`` over the pairs it belongs to (``-`` for the chosen member, ``+``
    for the rejected one). Computing this here on the full driver-side batch
    removes the actor-side co-location requirement entirely: each sample carries
    its own weight, so a pair's two members may live on different DP ranks /
    micro-batches (100% pair coverage, no NCCL desync).

    ``w_i`` is evaluated from the rollout/behavior logprobs (``old_log_probs``),
    which equal the current policy at the start of an on-policy step — consistent
    with how the actor sets ``old_log_prob = log_prob.detach()`` on-policy.

    Coefficients here are UN-normalized (no 1/P, no world-size factor): the actor
    turns them into a per-mini-batch mean by dividing the summed ``coeff·avg_logp``
    by the number of pairs in the mini-batch (``dpo_pair_member`` counts them), and
    FSDP's cross-rank gradient averaging then yields the global pair-mean — exactly
    how ``pg_loss`` (a per-rank token-mean) already behaves.

    Args:
        batch: DataProto whose ``.batch`` holds ``old_log_probs``, ``response_mask``,
            ``branch_token_mask``, ``is_two_stage_stage1``, ``dpo_pair_id``,
            ``dpo_role`` (and optionally ``ref_log_prob``, ``teacher_branch_logprob``,
            ``teacher_sibling_logprob``, ``token_level_scores`` for the reward
            filter); ``.non_tensor_batch`` holds ``uid``.
        policy_loss_cfg: the ``actor.policy_loss`` config (``.get`` access) with the
            ``dpo_*`` keys. ``dpo_reward_filter`` (default True) drops any pair whose
            chosen leaf scored a lower sequence reward than its rejected leaf.

    Returns:
        (coeff[B] float tensor, pair_member[B] float tensor, metrics dict), or
        (None, None, {}) if required fields are missing. ``coeff`` is zero for
        Stage-1 non-anchor and unpaired rows; ``pair_member`` is 1.0 for the two
        members of each formed Stage-2 pair (so ``pair_member.sum()/2`` = #pairs).
    """
    tb = batch.batch
    required = ("old_log_probs", "response_mask", "branch_token_mask", "is_two_stage_stage1",
                "dpo_pair_id", "dpo_role")
    if any(k not in tb for k in required):
        return None, None, {}

    old_log_probs = tb["old_log_probs"]
    response_mask = tb["response_mask"]
    branch_token_mask = tb["branch_token_mask"]
    is_two_stage_stage1 = tb["is_two_stage_stage1"]
    dpo_pair_id = tb["dpo_pair_id"]
    dpo_role = tb["dpo_role"]
    device = old_log_probs.device
    B = old_log_probs.shape[0]

    dpo_beta = float(policy_loss_cfg.get("dpo_coefficient", 0.0))
    use_ref = bool(policy_loss_cfg.get("dpo_use_ref", False))
    teacher_guided = bool(policy_loss_cfg.get("dpo_teacher_guided_beta", False))
    beta_alpha = float(policy_loss_cfg.get("dpo_teacher_beta_alpha", 1.0))
    beta_min = float(policy_loss_cfg.get("dpo_teacher_beta_min", 0.1))
    beta_max = float(policy_loss_cfg.get("dpo_teacher_beta_max", 3.0))
    stage1_pair = bool(policy_loss_cfg.get("dpo_stage1_pair", False))
    stage1_pair_weight = float(policy_loss_cfg.get("dpo_stage1_pair_weight", 1.0))

    ref_log_prob = tb.get("ref_log_prob") if use_ref else None
    teacher_branch = tb.get("teacher_branch_logprob") if teacher_guided else None
    teacher_sibling = tb.get("teacher_sibling_logprob") if teacher_guided else None
    uid = batch.non_tensor_batch.get("uid")

    coeff = torch.zeros(B, dtype=torch.float32, device=device)
    pair_member = torch.zeros(B, dtype=torch.float32, device=device)
    empty_metrics = {"actor/dpo_loss": 0.0, "actor/dpo_n_pairs": 0.0}

    # avg logprob over suffix (ref-adjusted) for ALL samples, from behavior policy.
    avg_logp, suffix_mask = suffix_avg_logp(old_log_probs, branch_token_mask, response_mask)

    # Fix: Stage-1 samples have no meaningful branch point; their suffix_mask
    # should cover the full response. When branch_token_mask is all-zero for a
    # Stage-1 row, suffix_avg_logp happens to give the correct result (last_branch_pos
    # = -1 → suffix covers everything). But if a Stage-1 row carries any residual
    # branch_token_mask bits (e.g., from padding reuse), we explicitly override with
    # the full response_mask to guarantee correctness.
    is_s1 = is_two_stage_stage1.bool()
    if is_s1.any():
        s1_logp = (old_log_probs * response_mask).sum(dim=1) / response_mask.sum(dim=1).clamp_min(1.0)
        avg_logp = torch.where(is_s1, s1_logp, avg_logp)

    ref_avg_logp = None
    if ref_log_prob is not None:
        ref_avg_logp_base = (ref_log_prob * suffix_mask).sum(dim=1) / suffix_mask.sum(dim=1).clamp_min(1.0)
        if is_s1.any():
            ref_s1_logp = (ref_log_prob * response_mask).sum(dim=1) / response_mask.sum(dim=1).clamp_min(1.0)
            ref_avg_logp = torch.where(is_s1, ref_s1_logp, ref_avg_logp_base)
        else:
            ref_avg_logp = ref_avg_logp_base

    def adj(i):
        # ref-adjusted suffix-avg logp of sample i (scalar tensor)
        return avg_logp[i] - ref_avg_logp[i] if ref_avg_logp is not None else avg_logp[i]

    # Reward-consistency filter: the teacher picks chosen/rejected from its own
    # branch logprobs, but that preference can disagree with the actual rollout
    # outcome. Drop any pair whose chosen leaf scored a LOWER sequence reward than
    # its rejected leaf, so we never train the policy to prefer a branch the
    # environment rated worse. Uses token_level_scores (raw outcome reward, before
    # KL), available on the driver batch by the time this runs.
    reward_filter = bool(policy_loss_cfg.get("dpo_reward_filter", True))
    seq_reward = None
    if reward_filter and "token_level_scores" in tb:
        seq_reward = tb["token_level_scores"].sum(dim=-1)  # [B]

    # ---- Build (chosen, rejected) Stage-2 pairs from stable (uid, pair_id, role) ----
    is_stage2 = (is_two_stage_stage1 == 0)
    stage2_indices = torch.where(is_stage2)[0].tolist()
    pair_map: dict = defaultdict(dict)
    for i in stage2_indices:
        pid = int(dpo_pair_id[i].item())
        role = int(dpo_role[i].item())
        if pid < 0 or role < 0:
            continue
        pkey = (str(uid[i]) if uid is not None else "0", pid)
        pair_map[pkey][role] = i
    chosen_pos, rejected_pos = [], []
    n_reward_filtered = 0
    for _pkey, roles in pair_map.items():
        if 0 in roles and 1 in roles:
            c_i, r_i = roles[0], roles[1]
            if seq_reward is not None and float(seq_reward[c_i]) < float(seq_reward[r_i]):
                n_reward_filtered += 1
                # Explicitly zero coeff and pair_member for filtered pairs so that
                # actor-side normalization (pair_member.sum()/2) is never inflated.
                coeff[c_i] = 0.0
                coeff[r_i] = 0.0
                pair_member[c_i] = 0.0
                pair_member[r_i] = 0.0
                continue  # chosen scored worse than rejected -> drop this pair
            chosen_pos.append(c_i)
            rejected_pos.append(r_i)

    P = len(chosen_pos)
    if P == 0:
        empty = dict(empty_metrics)
        empty["actor/dpo_pairs_filtered"] = float(n_reward_filtered)
        return coeff, pair_member, empty

    margins, betas, zs = [], [], []
    for c_i, r_i in zip(chosen_pos, rejected_pos):
        z = adj(c_i) - adj(r_i)  # relative preference (chosen vs rejected)
        beta_i = dpo_beta
        if teacher_guided and teacher_branch is not None and teacher_sibling is not None:
            margin = (teacher_branch[c_i] - teacher_sibling[c_i]).detach()
            beta_i = dpo_beta * float(torch.clamp(beta_alpha * margin, min=beta_min, max=beta_max))
            margins.append(float(margin))
        w = torch.sigmoid(-beta_i * z).detach()  # difficulty weight, detached
        # un-normalized DPO gradient coefficient, mirrors -w β (∂a - ∂b)
        coeff[c_i] += -beta_i * w
        coeff[r_i] += beta_i * w
        pair_member[c_i] = 1.0
        pair_member[r_i] = 1.0
        betas.append(float(beta_i))
        zs.append(float(z))

    logits = torch.tensor([b * z for b, z in zip(betas, zs)])
    dpo_loss_val = float(-torch.nn.functional.logsigmoid(logits).mean())

    metrics = {
        "actor/dpo_loss": dpo_loss_val,
        "actor/dpo_n_pairs": float(P),
        "actor/dpo_pairs_filtered": float(n_reward_filtered),
        "actor/dpo_margin": float(sum(zs) / len(zs)),
        "actor/dpo_beta_effective_mean": float(sum(betas) / len(betas)),
    }
    if teacher_guided and margins:
        metrics["actor/dpo_teacher_margin_mean"] = float(sum(margins) / len(margins))

    # ---- Stage-1 3-way ranking (argmax > mid > argmin), same per-sample treatment ----
    # In two-stage branching every prompt has Stage-1 rollouts, so n_s1 == P and the
    # actor's /pairs normalization reproduces the original per-term means.
    # Bug-fix (2026-07-03): stage1_rep is now a (uid, tree_idx) → i mapping so each
    # DPO pair uses the stage1 mid-anchor from its OWN tree, not a shared first-seen.
    if stage1_pair and uid is not None:
        batch_tree_idx = tb.get("tree_idx")  # [B] LongTensor, -1 if absent
        stage1_rep: dict = {}  # (uid_str, tree_idx_int) -> batch index i
        for i in range(B):
            if int(is_two_stage_stage1[i].item()) == 1:
                uid_key = str(uid[i])
                ti = int(batch_tree_idx[i].item()) if batch_tree_idx is not None else 0
                rep_key = (uid_key, ti)
                if rep_key not in stage1_rep:
                    stage1_rep[rep_key] = i
        s1_units = []  # (argmax_i, mid_i, argmin_i)
        for c_i, r_i in zip(chosen_pos, rejected_pos):
            uid_key = str(uid[c_i])
            # Determine tree_idx for this DPO pair from chosen sample
            ti = int(batch_tree_idx[c_i].item()) if batch_tree_idx is not None else 0
            rep_key = (uid_key, ti)
            if rep_key in stage1_rep:
                s1_units.append((c_i, stage1_rep[rep_key], r_i))
            else:
                # Fallback: try any stage1 for this uid (backward compat)
                fallback_mid = None
                for k, v in stage1_rep.items():
                    if k[0] == uid_key:
                        fallback_mid = v
                        break
                if fallback_mid is not None:
                    s1_units.append((c_i, fallback_mid, r_i))
        n_s1 = len(s1_units)
        if n_s1 > 0:
            for amax, mid, amin in s1_units:
                z_best = adj(amax) - adj(mid)   # argmax > mid
                z_worst = adj(mid) - adj(amin)  # mid > argmin
                w_best = torch.sigmoid(-dpo_beta * z_best).detach()
                w_worst = torch.sigmoid(-dpo_beta * z_worst).detach()
                coeff[amax] += -stage1_pair_weight * dpo_beta * w_best
                coeff[mid] += stage1_pair_weight * dpo_beta * w_best
                coeff[mid] += -stage1_pair_weight * dpo_beta * w_worst
                coeff[amin] += stage1_pair_weight * dpo_beta * w_worst
            metrics["actor/dpo_stage1_pairs"] = float(n_s1)

    return coeff, pair_member, metrics


def need_critic(config: DictConfig) -> bool:
    """Given a config, do we need critic."""
    if config.critic.enable is not None:
        return bool(config.critic.enable)
    elif config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        return True
    else:
        warnings.warn(
            "Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True",
            stacklevel=2,
        )
        return False
