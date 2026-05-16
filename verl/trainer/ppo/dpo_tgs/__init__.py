# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""DPO-TGS: On-Policy DPO + Teacher-Guided Sampling.

Reuses the chain-rollout infrastructure from TCCA-Lite (`tcca_chain.py`):
the same B*chain_length augmented batch becomes the source of (chosen, rejected)
pairs instead of an additive ΔR modulation.

V1 (this implementation): linearized DPO encoded as a per-token advantage that
flows through the standard PPO clipped surrogate — no actor-side changes.

Design doc: research/dpo_teacher_guided_sampling.md
"""

from verl.trainer.ppo.dpo_tgs.pair_collector import (
    DPOPairCollector,
    collect_chain_consecutive_pairs,
    collect_hybrid_init_chain_pairs,
    collect_dpo_pairs,
    compute_dpo_metrics,
    write_pair_info_to_batch,
)
from verl.trainer.ppo.dpo_tgs.dpo_loss import compute_dpo_tgs_advantage  # registers "dpo_teacher_guided"
from verl.trainer.ppo.dpo_tgs.adaptive_rollout import dpo_tgs_adaptive_rollout

__all__ = [
    "DPOPairCollector",
    "collect_chain_consecutive_pairs",
    "collect_hybrid_init_chain_pairs",
    "collect_dpo_pairs",
    "compute_dpo_metrics",
    "write_pair_info_to_batch",
    "compute_dpo_tgs_advantage",
    "dpo_tgs_adaptive_rollout",
]
