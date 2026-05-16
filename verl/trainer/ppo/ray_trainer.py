# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import random
import re
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from string import Template
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.model import compute_position_id_with_mask
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.torch_functional import postprocess_data
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def concat_dataproto_with_padding(data_list: list["DataProto"]) -> "DataProto":
    """Concat DataProto list with padding to handle different seq lengths.
    
    Different batches may have different response lengths, so we need to pad
    shorter sequences before concatenating.
    
    Args:
        data_list: List of DataProto with potentially different seq_len (dim=1)
        
    Returns:
        Single DataProto with all batches concatenated along dim=0
    """
    if len(data_list) == 1:
        return data_list[0]
    
    # Collect all keys and find max seq_len for each key separately
    all_keys = set()
    for data in data_list:
        if data.batch is not None:
            all_keys.update(data.batch.keys())
    
    # Find max seq_len for each key separately (different keys may have different shapes)
    key_to_max_len = {}
    for key in all_keys:
        max_len = 0
        for data in data_list:
            if data.batch is not None and key in data.batch:
                tensor = data.batch[key]
                if len(tensor.shape) >= 2:
                    max_len = max(max_len, tensor.shape[1])
        if max_len > 0:
            key_to_max_len[key] = max_len
    
    # Pad each batch's tensors to their respective max lengths
    padded_batches = []
    for data in data_list:
        if data.batch is None:
            continue
        padded_batch = {}
        for key, tensor in data.batch.items():
            if key in key_to_max_len and len(tensor.shape) >= 2:
                target_len = key_to_max_len[key]
                curr_len = tensor.shape[1]
                if curr_len < target_len:
                    pad_len = target_len - curr_len
                    padded_batch[key] = torch.nn.functional.pad(tensor, (0, pad_len), value=0)
                elif len(tensor.shape) == 3:
                    # 3D tensor (e.g., teacher_topk_log_probs: B, T, K)
                    # Pad along seq_len dimension (dim=1)
                    padded_batch[key] = torch.nn.functional.pad(tensor, (0, 0, 0, pad_len), value=0)
                else:
                    padded_batch[key] = tensor
            else:
                padded_batch[key] = tensor
        padded_batches.append(padded_batch)
    
    # Now concat all padded batches
    from tensordict import TensorDict
    concatenated = {}
    for key in all_keys:
        tensors = [pb[key] for pb in padded_batches if key in pb]
        if tensors:
            concatenated[key] = torch.cat(tensors, dim=0)
    
    # Handle non_tensor_batch
    non_tensor_batch = {}
    all_non_tensor_keys = set()
    for data in data_list:
        if data.non_tensor_batch:
            all_non_tensor_keys.update(data.non_tensor_batch.keys())
    
    for key in all_non_tensor_keys:
        arrays = [data.non_tensor_batch[key] for data in data_list if data.non_tensor_batch and key in data.non_tensor_batch]
        if arrays:
            non_tensor_batch[key] = np.concatenate(arrays, axis=0)
    
    # Merge meta_info (take from first, they should be consistent)
    merged_meta_info = {}
    for data in data_list:
        if data.meta_info:
            merged_meta_info.update(data.meta_info)
    
    return DataProto(
        batch=TensorDict(concatenated, batch_size=[concatenated[next(iter(concatenated.keys()))].shape[0]]),
        non_tensor_batch=non_tensor_batch,
        meta_info=merged_meta_info
    )


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, using max_colocate_count=3: actor_critic_ref, rollout, reward model (optional)
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=3, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.TASD:
        # TASD advantage estimation（清爽版）
        # self_distillation_mask 标识哪些样本有 teacher context
        sdist_mask = None
        if "self_distillation_mask" in data.batch:
            sdist_mask = data.batch["self_distillation_mask"]

        # tasd_gate_mask: entropy gate 后有效的 token 位置（hard/soft 模式下存在）
        # 被 gate 掉的 token 不参与 group_mean 计算，advantage 直接置 0
        gate_mask = data.batch.get("tasd_gate_mask", None)

        # tasd_teacher/student_entropy_norm: 熵信息（adv_entropy_weight 需要）
        teacher_entropy_norm = data.batch.get("tasd_teacher_entropy_norm", None)
        student_entropy_norm = data.batch.get("tasd_student_entropy_norm", None)

        # self-teacher advantage 需要的参数
        teacher_log_probs = data.batch.get("tasd_teacher_log_probs", None)
        student_log_probs = data.batch.get("old_log_probs", None)  # (B,T) student log prob on sampled token
        student_topk_log_probs = data.batch.get("tasd_student_topk_log_probs", None)
        teacher_at_student_topk = data.batch.get("tasd_teacher_at_student_topk", None)

        advantages, returns, filtered_response_mask = core_algos.compute_tasd_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            config=config,
            self_distillation_mask=sdist_mask,
            gate_mask=gate_mask,
            teacher_entropy_norm=teacher_entropy_norm,
            student_entropy_norm=student_entropy_norm,
            teacher_log_probs=teacher_log_probs,
            student_log_probs=student_log_probs,
            student_topk_log_probs=student_topk_log_probs,
            student_topk_indices=None,  # 不需要（已有 teacher_at_student_topk）
            teacher_at_student_topk=teacher_at_student_topk,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        # filtered_response_mask 包含 effective_mask 的过滤（gate_mask + self_distillation_mask）
        # 被过滤的 token → advantage=0 且 response_mask=False → 不贡献梯度也不计入分母
        data.batch["response_mask"] = filtered_response_mask
    elif adv_estimator == "rlsd":
        # RLSD baseline (arXiv:2604.03128v2): A_t = A_seq · clip(exp(sign(A_seq)·(logp_T - logp_S)), 1±eps_w)
        adv_estimator_fn = core_algos.get_adv_estimator_fn("rlsd")
        advantages, returns = adv_estimator_fn(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            teacher_log_probs=data.batch.get("bc_teacher_log_probs"),
            student_log_probs=data.batch.get("old_log_probs"),
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == "prior_shift":
        # Prior-Shift (ours): A_t = A_seq · KL(P_T(·|y_≤t) ‖ P_T(·|y_<t)) / mean_t
        adv_estimator_fn = core_algos.get_adv_estimator_fn("prior_shift")
        advantages, returns = adv_estimator_fn(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            teacher_prior_shift_surprise=data.batch.get("bc_teacher_prior_shift_surprise"),
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == "intervention_credit":
        # TCCA-Lite (Token-level Causal Credit Assignment):
        #   A_t = (A_seq + λ_div · c_t) · response_mask · length_scale  (additive)
        # c_t (divergence_credit) 由 intervention_rollout._do_real_intervention 在失败样本上写入:
        #   composite 上 c_t = +ΔR at t* (teacher's choice positive credit)
        #   原 y 上 c_t = -ΔR at t* (student's wrong choice mirror)
        # response_mask 在 composite 上 prefix [0, t*) 已置 0 (Layer 2)
        adv_estimator_fn = core_algos.get_adv_estimator_fn("intervention_credit")
        advantages, returns = adv_estimator_fn(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            divergence_credit=data.batch.get("divergence_credit"),          # TCCA per-token c_t
            # legacy compat (will be ignored if divergence_credit set):
            token_causal_credit=data.batch.get("token_causal_credit"),
            intervention_delta_reward=data.batch.get("intervention_delta_reward"),
            intervention_used=data.batch.get("intervention_used"),
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = config.actor_rollout_ref.actor.get("self_distillation", {}).get("reprompt_truncation", "error")
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        return_dict: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor | dict[str, Any]:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            return_dict: Whether to return dict format with reward_extra_info (for validation)
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
            If return_dict=False and sum_reward=True: summed reward_tensor (1D tensor)
            If return_dict=False and sum_reward=False: reward_tensor (2D tensor)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if return_dict:
                # Extract reward_extra_info if available
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                # If sum_reward=True, only return tensor (for REMAX baseline)
                if sum_reward:
                    return reward_tensor
                # Otherwise, return tuple with reward_extra_info (for training loop)
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_infos_dict = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return reward_tensor, reward_extra_infos_dict

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if return_dict:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_info = result.get("reward_extra_info", {})
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    @staticmethod
    def _collect_feedback(
        include_environment_feedback: bool,
        reward_extra_infos_dict: Optional[dict[str, Any]],
        batch_size: int
    ) -> list[Any]:
        """
        Collect environment feedback from reward_extra_infos_dict.

        Args:
            include_environment_feedback: Whether to include environment feedback
            reward_extra_infos_dict: Dictionary containing reward extra information
            batch_size: Size of the batch

        Returns:
            List of feedback strings (or None for entries without feedback)
        """
        feedback_list: list[Any] = [None] * batch_size
        if include_environment_feedback and reward_extra_infos_dict is not None:
            raw_feedback = reward_extra_infos_dict.get("feedback", [])
            for i in range(min(len(raw_feedback), batch_size)):
                # Only include non-empty feedback strings
                if raw_feedback[i] and isinstance(raw_feedback[i], str) and raw_feedback[i].strip():
                    feedback_list[i] = raw_feedback[i]
        return feedback_list

    def _collect_solutions_by_uid(self, batch: DataProto, reward_tensor: torch.Tensor, success_reward_threshold: float) -> dict[Any, list[int]]:
        seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()
        uids = batch.non_tensor_batch["uid"]
        success_by_uid: dict[Any, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            if seq_scores[idx] >= success_reward_threshold:
                success_by_uid[uid].append(idx)
        return success_by_uid

    @staticmethod
    def _remove_thinking_trace(text: str) -> str:
        """Remove <think>...</think> tags and their content from text."""
        return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)

    def _get_solution(
        self,
        idx: int,
        success_by_uid: dict[Any, list[int]],
        uids: list[Any],
        response_texts: list[str],
        dont_reprompt_on_self_success: bool = False,
        remove_thinking_from_demonstration: bool = False,
        fallback_to_self: bool = False,
        use_self_as_teacher_on_success: bool = False,
    ) -> Optional[str]:
        uid = uids[idx]
        solution_idxs = success_by_uid[uid]
        is_self_success = idx in solution_idxs

        # 成功 rollout 用自己的 response 作为 teacher context
        if use_self_as_teacher_on_success and is_self_success:
            solution = response_texts[idx]
            if remove_thinking_from_demonstration:
                solution = self._remove_thinking_trace(solution)
            return solution

        if dont_reprompt_on_self_success:
            solution_idxs = [j for j in solution_idxs if j != idx]

        if len(solution_idxs) == 0:
            if fallback_to_self and is_self_success:
                return response_texts[idx]
            return None

        solution_idx = random.choice(solution_idxs)  # randomly select from successful demonstrations
        solution_str = response_texts[solution_idx]
        if remove_thinking_from_demonstration:
            solution_str = self._remove_thinking_trace(solution_str)
        return solution_str


    def _maybe_build_self_distillation_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, list]] = None,
    ) -> Optional[tuple[DataProto, dict[str, float]]]:
        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        adv_estimator = self.config.algorithm.adv_estimator

        is_sdpo = loss_mode == "sdpo"
        is_tasd = adv_estimator == "tasd"
        # bayesian_credit estimators (rlsd / prior_shift / posterior_shift) all require
        # the same self-distillation re-prompt machinery to assemble teacher inputs.
        # bayesian_credit estimators (rlsd / prior_shift / posterior_shift) reuse the same
        # self-distillation re-prompt machinery to assemble teacher inputs.
        # Note: intervention_credit (TCCA-Lite) does its own OPSD teacher fwd inside
        # intervention_rollout, doesn't need ray_trainer's SD batch construction. Skip.
        is_bayesian_credit = adv_estimator in ("rlsd", "prior_shift", "posterior_shift")
        if self_distillation_cfg is None or (not is_sdpo and not is_tasd and not is_bayesian_credit):
            return None

        tasd_cfg = self.config.algorithm.get("tasd", {})

        # use_self_as_teacher_on_success：只对TASD生效
        use_self_as_teacher_on_success = (
            is_tasd
            and tasd_cfg.get("use_self_as_teacher_on_success", False)
        )

        device = batch.batch["input_ids"].device
        response_mask = batch.batch["response_mask"]
        responses = batch.batch["responses"]
        response_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]
        prompt_texts = [msgs[-1]["content"] for msgs in batch.non_tensor_batch["raw_prompt"]]
        batch_size = batch.batch.batch_size[0]

        # Extract feedback if available and include_environment_feedback is enabled
        feedback_list = self._collect_feedback(
            include_environment_feedback=self_distillation_cfg.include_environment_feedback,
            reward_extra_infos_dict=reward_extra_infos_dict,
            batch_size=batch_size,
        )

        success_by_uid = self._collect_solutions_by_uid(batch, reward_tensor, success_reward_threshold=self_distillation_cfg.success_reward_threshold)
        solution_strs = [
            self._get_solution(
                i,
                success_by_uid,
                batch.non_tensor_batch["uid"],
                response_texts,
                self_distillation_cfg.dont_reprompt_on_self_success,
                self_distillation_cfg.get("remove_thinking_from_demonstration", False),
                fallback_to_self=False,
                use_self_as_teacher_on_success=use_self_as_teacher_on_success,
            )
            for i in range(batch_size)
        ]

        def _build_teacher_message(i: int) -> list[dict]:
            system_messages = batch.non_tensor_batch["raw_prompt"][i][:-1]
            has_solution = solution_strs[i] is not None
            has_feedback = feedback_list[i] is not None
            feedback_only_without_solution = self_distillation_cfg.get("environment_feedback_only_without_solution", False)

            # If feedback_only_without_solution is True, only use feedback when no solution exists
            use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)

            # build solution section
            solution_section = ""
            if has_solution:
                solution_section = self_distillation_cfg.solution_template.format(
                    successful_previous_attempt=solution_strs[i]
                )

            # build feedback section
            feedback_section = ""
            if use_feedback:
                feedback_section = self_distillation_cfg.feedback_template.format(
                    feedback_raw=feedback_list[i]
                )

            # combine solution and feedback sections
            if use_feedback or has_solution:
                reprompt_text = self_distillation_cfg.reprompt_template.format(
                    prompt=prompt_texts[i],
                    solution=solution_section,
                    feedback=feedback_section,
                )
            else:
                reprompt_text = prompt_texts[i]

            return system_messages + [
                {"role": "user", "content": reprompt_text},
            ]


        messages = [_build_teacher_message(i) for i in range(batch_size)]
        enable_thinking = self.config.data.apply_chat_template_kwargs.get("enable_thinking", True) if self.config.data.apply_chat_template_kwargs else True
        teacher_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            continue_final_message=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            max_length=self_distillation_cfg.max_reprompt_len,
            padding=True,
            truncation=True,
        )
        teacher_input_ids = torch.cat([teacher_prompt["input_ids"].to(device), responses], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt["attention_mask"].to(device), response_mask], dim=1)
        teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

        # Compute which samples actually use feedback (accounting for environment_feedback_only_without_solution)
        feedback_only_without_solution = self_distillation_cfg.get("environment_feedback_only_without_solution", False)
        feedback_used = [
            feedback_list[i] is not None and (not feedback_only_without_solution or solution_strs[i] is None)
            for i in range(batch_size)
        ]

        # self_distillation_mask is True if sample has a solution OR feedback is used (i.e., will get a reprompted message)
        self_distillation_mask = torch.tensor(
            [solution_strs[i] is not None or feedback_used[i] for i in range(batch_size)],
            dtype=torch.float32,
            device=device
        )
        
        # Debug: self_distillation_mask 统计
        num_with_solution = sum(1 for s in solution_strs if s is not None)
        num_with_feedback = sum(feedback_used)
        print(f"[TASD Debug] self_distillation_mask: {self_distillation_mask.sum().item()}/{batch_size} samples, "
              f"with_solution={num_with_solution}, with_feedback={num_with_feedback}")

        uids = set(batch.non_tensor_batch["uid"])
        num_with_feedback_available = sum(1 for f in feedback_list if f is not None)
        num_with_feedback_used = sum(1 for f in feedback_used if f)
        num_with_solution = sum(1 for s in solution_strs if s is not None)
        metrics = {
            "self_distillation/success_group_fraction": len([uid for uid in uids if len(success_by_uid[uid]) > 0]) / len(uids),
            "self_distillation/success_sample_fraction": num_with_solution / batch_size,
            "self_distillation/feedback_available_fraction": num_with_feedback_available / batch_size,
            "self_distillation/feedback_used_fraction": num_with_feedback_used / batch_size,
            "self_distillation/reprompt_sample_fraction": self_distillation_mask.float().mean().item(),
        }
        return DataProto.from_dict(tensors={
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "teacher_position_ids": teacher_position_ids,
            "self_distillation_mask": self_distillation_mask,
        }), metrics

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid", "raw_prompt"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            result = self._compute_or_extract_reward(test_batch, reward_fn=self.val_reward_fn, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_info = result.get("reward_extra_info", {})
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            # Note: mode is always "async" since sync mode is deprecated
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        else:
            rm_resource_pool = None

        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rm_resource_pool=rm_resource_pool,
        )

    def _get_best_metric_value(self, val_metrics: dict):
        """Extract the metric value used for best checkpoint tracking.

        Priority:
        1. trainer.save_best_metric (explicit config)
        2. First key containing 'mean@' in val-core namespace
        3. First key containing 'mean@' in any namespace
        Returns None if no suitable metric found.
        """
        explicit = self.config.trainer.get("save_best_metric", None)
        if explicit:
            return val_metrics.get(explicit, None), explicit
        # auto-detect: prefer val-core/*mean@*
        for k, v in val_metrics.items():
            if "val-core" in k and "mean@" in k:
                return v, k
        for k, v in val_metrics.items():
            if "mean@" in k:
                return v, k
        return None, None

    def _maybe_save_best_checkpoint(self, val_metrics: dict):
        """Save best checkpoint when the monitored metric improves."""
        metric_val, metric_key = self._get_best_metric_value(val_metrics)
        if metric_val is None:
            return
        if self._best_val_metric is None or metric_val > self._best_val_metric:
            self._best_val_metric = metric_val
            self._best_val_step = self.global_steps
            print(
                f"[BestCkpt] New best {metric_key}={metric_val:.4f} at step {self.global_steps}, saving best checkpoint."
            )
            self._save_best_checkpoint()

    def _save_best_checkpoint(self):
        """Save actor checkpoint to <default_local_dir>/best/actor, always overwriting."""
        from verl.utils.fs import local_mkdir_safe

        best_folder = os.path.join(self.config.trainer.default_local_dir, "best")
        actor_local_path = os.path.join(best_folder, "actor")
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, "best", "actor")
        )
        # max_ckpt_to_keep=1 ensures old best is removed when a new best arrives
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=1
        )
        # write a metadata file so we know which step the best is from
        local_mkdir_safe(best_folder)
        best_meta_path = os.path.join(best_folder, "best_step.txt")
        with open(best_meta_path, "w") as f:
            f.write(f"step={self.global_steps}\nbest_metric={self._best_val_metric}\n")
        print(f"[BestCkpt] Best checkpoint saved to {best_folder}")

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        # 设置全局随机种子（可复现性）
        seed = self.config.trainer.get("seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"[TASD] Global seed set to {seed}")

        # 从实验名称自动提取 SwanLab tags 和 group
        # 实验名称格式：TASD-simple-sciknoweval-biology-rtteacher_prob-gate_soft-clip2.0-Qwen3-8B-时间戳
        # group: 前面的方法+数据集部分（如 TASD-simple-sciknoweval-biology）
        # tags: 后面的超参部分（如 rtteacher_prob, gate_soft, clip2.0）
        import os
        _experiment_name = self.config.trainer.experiment_name
        _swanlab_tags = None
        _swanlab_group = None

        if _experiment_name:
            # 按 - 分割实验名称
            _parts = _experiment_name.split("-")
            # 找到数据集结束位置：通常是 前缀-数据集类型-数据集名（如 TASD-simple-sciknoweval-biology）
            _dataset_keywords = {"sciknoweval", "lcb", "tooluse"}
            _group_end_idx = 0
            for i, part in enumerate(_parts):
                if part in _dataset_keywords:
                    # 如果下一个不是超参关键词，则继续包含（如 sciknoweval 后面可能有 biology/material）
                    if i + 1 < len(_parts) and _parts[i + 1] not in {"rt", "gate", "clip", "lr", "topk", "rep"}:
                        _group_end_idx = i + 2
                    else:
                        _group_end_idx = i + 1
                    break

            if _group_end_idx > 0:
                _swanlab_group = "-".join(_parts[:_group_end_idx])
                # 剩余部分全部作为 tags（包括模型名、超参等，只排除时间戳）
                _tag_parts = _parts[_group_end_idx:]
                _filtered_tags = []
                for p in _tag_parts:
                    # 只跳过时间戳（包含下划线）
                    if "_" in p:
                        continue
                    _filtered_tags.append(p)
                if _filtered_tags:
                    _swanlab_tags = _filtered_tags

        # 环境变量优先级更高（允许手动覆盖）
        _env_tags = os.environ.get("SWANLAB_TAGS", "")
        if _env_tags:
            _swanlab_tags = [t.strip() for t in _env_tags.split(",") if t.strip()]
        _env_group = os.environ.get("SWANLAB_GROUP", "")
        if _env_group:
            _swanlab_group = _env_group
        # 最后回退到 config
        if not _swanlab_group:
            _swanlab_group = self.config.trainer.get("group_name", None)

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
            group_name=_swanlab_group,
            tags=_swanlab_tags,
        )

        self.global_steps = 0
        self._best_val_metric = None  # track best val metric for best checkpoint saving
        self._best_val_step = None     # step at which best metric was achieved

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # Send DingTalk notification: training started
        from verl.utils.tracking import send_dingtalk_alert
        _exp_name = self.config.trainer.get("experiment_name", "")
        _total_steps = self.total_training_steps
        _total_epochs = self.config.trainer.get("total_epochs", "?")
        send_dingtalk_alert(
            f"🟢 Training started!\n"
            f"  experiment: {_exp_name}\n"
            f"  total_steps: {_total_steps}, total_epochs: {_total_epochs}"
        )

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # ── TASD / RLSD 配置解析（循环外，避免每step重复读取）──────────
        is_tasd = self.config.algorithm.adv_estimator == AdvantageEstimator.TASD
        # bayesian_credit estimators share the same lightweight teacher-forward path
        # (only need teacher_log_probs_on_response, no entropy gate / token rewards).
        # bayesian_credit estimators that share the lightweight teacher-forward path
        # (only need teacher_log_probs_on_response + optional g_t, no entropy gate).
        # Note: intervention_credit (TCCA-Lite) skipped - does OPSD teacher fwd inside intervention_rollout.
        is_rlsd_like = self.config.algorithm.adv_estimator in ("rlsd", "prior_shift", "posterior_shift")
        if is_tasd:
            _tasd_cfg = self.config.algorithm.get("tasd", {})
            tasd_reward_type = _tasd_cfg.get("reward_type", "teacher_prob")
            tasd_entropy_gate = _tasd_cfg.get("entropy_gate", "none")
            tasd_entropy_gate_ratio = _tasd_cfg.get("entropy_gate_ratio", 1.0)
            tasd_adv_entropy_weight = _tasd_cfg.get("adv_entropy_weight", "none")
            tasd_adv_mode = _tasd_cfg.get("adv_mode", "token")
            tasd_use_vce = _tasd_cfg.get("use_vce", False)
            tasd_temperature = _tasd_cfg.get("distill_temperature", None) or self.config.actor_rollout_ref.rollout.temperature
            tasd_micro_batch_size = self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
            # entropy_gate / adv_entropy_weight / self_teacher + use_vce 都需要 topk 信息
            tasd_need_topk = (
                tasd_entropy_gate != "none"
                or tasd_adv_entropy_weight != "none"
                or (tasd_adv_mode == "self_teacher" and tasd_use_vce)
            )
            tasd_distill_topk = _tasd_cfg.get("distill_topk", 100) if tasd_need_topk else None
        # ────────────────────────────────────────────────────────

        # ── DAPO filter_groups 流式累积状态（循环外，跨 dataloader step 持继）
        _fg_accumulated_batch = None   # 累积过滤后的 batch
        _fg_num_prompt_in_batch = 0    # 已累积的合格 prompt 数
        _fg_num_gen_batches = 0        # 当前 step 已生成次数

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Compute or extract reward for training
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, return_dict=False
                            )

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        self_distillation_data = self._maybe_build_self_distillation_batch(batch, reward_tensor, reward_extra_infos_dict)
                        if self_distillation_data is not None:
                            self_distillation_batch, self_distillation_metrics = self_distillation_data
                            batch = batch.union(self_distillation_batch)
                            metrics.update(self_distillation_metrics)

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # ── RLSD/Prior-Shift/Posterior-Shift: lightweight teacher forward ──
                        # 只取 teacher_log_probs_on_response，不动 token_level_rewards、不算 entropy gate。
                        if is_rlsd_like and "teacher_input_ids" in batch.batch:
                            _bc_temperature = self.config.actor_rollout_ref.rollout.temperature
                            _bc_micro_batch_size = self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
                            # 何时计算 g_t (KL Bayes surprise)?
                            #   - adv_estimator=prior_shift          : 必需
                            #   - adv_estimator=intervention_credit + base_estimator=prior_shift : 必需
                            #   - 其他                                : 不需要 (节省 forward 开销)
                            _ic_cfg = self.config.algorithm.get("intervention_credit", {}) or {}
                            _ic_base = str(_ic_cfg.get("base_estimator", "grpo")).lower()
                            _bc_needs_g_t = (
                                self.config.algorithm.adv_estimator == "prior_shift"
                                or (self.config.algorithm.adv_estimator == "intervention_credit" and _ic_base == "prior_shift")
                            )
                            _bc_is_prior_shift = _bc_needs_g_t
                            bc_fwd_batch = DataProto.from_dict(
                                tensors={
                                    "teacher_input_ids":      batch.batch["teacher_input_ids"],
                                    "teacher_attention_mask": batch.batch["teacher_attention_mask"],
                                    "teacher_position_ids":   batch.batch["teacher_position_ids"],
                                    "responses":              batch.batch["responses"],
                                    "input_ids":              batch.batch["input_ids"],
                                    "attention_mask":         batch.batch["attention_mask"],
                                    "position_ids":           batch.batch["position_ids"],
                                }
                            )
                            bc_fwd_batch.meta_info = {
                                "temperature":      _bc_temperature,
                                "micro_batch_size": _bc_micro_batch_size,
                                "pad_token_id":     self.tokenizer.pad_token_id,
                                "distill_topk":     None,   # bayesian_credit baseline 不需要 top-K
                                # Prior-Shift 需要在 teacher forward 内部算 KL(D_t ‖ D_{t-1})
                                "compute_prior_shift_surprise": _bc_is_prior_shift,
                            }
                            with marked_timer("bc_teacher_fwd", timing_raw, color="cyan"):
                                bc_teacher_result = self.actor_rollout_wg.compute_teacher_log_probs(
                                    bc_fwd_batch
                                )
                            batch.batch["bc_teacher_log_probs"] = bc_teacher_result.batch[
                                "teacher_log_probs_on_response"
                            ]
                            if _bc_is_prior_shift and "prior_shift_surprise" in bc_teacher_result.batch:
                                batch.batch["bc_teacher_prior_shift_surprise"] = bc_teacher_result.batch[
                                    "prior_shift_surprise"
                                ]

                        # ── Intervention-Credit (Tier 3): 失败样本 t* 检测 + 真实 intervention rollout ──
                        # Phase 1: enable_intervention=False 时只做 divergence 度量，batch 不变
                        # Phase 2: enable_intervention=True 时调用 teacher generate + student tail
                        if self.config.algorithm.adv_estimator == "intervention_credit":
                            from verl.trainer.ppo.bayesian_credit.intervention_rollout import (
                                run_teacher_intervention_rollout,
                            )
                            with marked_timer("intervention_rollout", timing_raw, color="magenta"):
                                ic_result = run_teacher_intervention_rollout(
                                    batch=batch,
                                    actor_rollout_wg=self.actor_rollout_wg,
                                    async_rollout_manager=getattr(self, "async_rollout_manager", None),
                                    reward_fn=self.reward_fn,
                                    config=self.config,
                                    tokenizer=self.tokenizer,
                                )
                            batch = ic_result.batch
                            metrics.update(ic_result.metrics)

                        # ── TASD: 计算 teacher token-level rewards ─────────────────────
                        if is_tasd and "teacher_input_ids" in batch.batch:
                            teacher_fwd_batch = DataProto.from_dict(
                                tensors={
                                    "teacher_input_ids":      batch.batch["teacher_input_ids"],
                                    "teacher_attention_mask": batch.batch["teacher_attention_mask"],
                                    "teacher_position_ids":   batch.batch["teacher_position_ids"],
                                    "responses":              batch.batch["responses"],
                                    "input_ids":              batch.batch["input_ids"],
                                    "attention_mask":         batch.batch["attention_mask"],
                                    "position_ids":           batch.batch["position_ids"],
                                }
                            )
                            teacher_fwd_batch.meta_info = {
                                "temperature":      tasd_temperature,
                                "micro_batch_size": tasd_micro_batch_size,
                                "pad_token_id":     self.tokenizer.pad_token_id,
                                "distill_topk":     tasd_distill_topk,
                            }

                            with marked_timer("tasd_teacher_fwd", timing_raw, color="cyan"):
                                teacher_result = self.actor_rollout_wg.compute_teacher_log_probs(
                                    teacher_fwd_batch
                                )

                            token_rewards, gate_mask, teacher_entropy_norm, student_entropy_norm = core_algos.compute_tasd_token_rewards(
                                student_log_probs=batch.batch["old_log_probs"],
                                teacher_log_probs=teacher_result.batch["teacher_log_probs_on_response"],
                                student_topk_log_probs=teacher_result.batch.get("student_topk_log_probs"),
                                teacher_topk_log_probs=teacher_result.batch.get("teacher_topk_log_probs"),
                                reward_type=tasd_reward_type,
                                entropy_gate=tasd_entropy_gate,
                                entropy_gate_ratio=tasd_entropy_gate_ratio,
                                adv_entropy_weight=tasd_adv_entropy_weight,
                            )

                            # Mask out padding positions
                            response_mask_float = batch.batch["response_mask"].float()
                            token_rewards = token_rewards * response_mask_float

                            # 覆盖 token_level_rewards
                            batch.batch["token_level_rewards"] = token_rewards

                            # 保存 gate_mask，供 compute_tasd_advantage 排除 gate 掉的 token
                            # gate_mask=None 表示无熵门控（noGate），gate_mask!=None 表示 hard/soft 门控
                            if gate_mask is not None:
                                batch.batch["tasd_gate_mask"] = (gate_mask * response_mask_float).bool()

                            # 保存熵信息，供 adv_entropy_weight 使用
                            if teacher_entropy_norm is not None:
                                batch.batch["tasd_teacher_entropy_norm"] = teacher_entropy_norm
                            if student_entropy_norm is not None:
                                batch.batch["tasd_student_entropy_norm"] = student_entropy_norm

                            # ── 保存 self-teacher advantage 需要的参数 ────────────────────
                            # teacher_log_probs_on_response: (B, T) teacher 对实际 token 的评分
                            batch.batch["tasd_teacher_log_probs"] = teacher_result.batch["teacher_log_probs_on_response"]
                            # student_topk_log_probs: (B, T, K) student 的 top-K log probs
                            if "student_topk_log_probs" in teacher_result.batch:
                                batch.batch["tasd_student_topk_log_probs"] = teacher_result.batch["student_topk_log_probs"]
                            # teacher_topk_log_probs: (B, T, K) teacher 在 student top-K 位置的 log probs
                            if "teacher_topk_log_probs" in teacher_result.batch:
                                batch.batch["tasd_teacher_at_student_topk"] = teacher_result.batch["teacher_topk_log_probs"]
                            # student_topk_indices: 需要从 student forward 获取
                            # 注意：当前 teacher forward 没有返回 student_topk_indices
                            # 但 compute_self_teacher_advantage 不需要它（已经有 teacher_at_student_topk）

                            # ── 记录 TASD token reward 指标 ─────────────────────────────
                            response_mask_bool = batch.batch["response_mask"].bool()
                            valid_token_rewards = token_rewards[response_mask_bool]

                            metrics["tasd/token_reward_mean"] = valid_token_rewards.mean().item()
                            metrics["tasd/token_reward_std"] = valid_token_rewards.std().item() if valid_token_rewards.numel() > 1 else 0.0
                            metrics["tasd/token_reward_pos_rate"] = (valid_token_rewards > 0).float().mean().item()

                            # ── 记录 entropy gate 指标 ─────────────────────────────
                            # 生成 token 数量（每条 response）
                            response_token_counts = response_mask_float.sum(dim=-1)  # (B,)
                            metrics["tasd/gen_token_count_mean"] = response_token_counts.mean().item()
                            metrics["tasd/gen_token_count_max"] = response_token_counts.max().item()
                            metrics["tasd/gen_token_count_min"] = response_token_counts.min().item()

                            # 熵门控统计：使用 gate_mask 计算
                            # gate_mask: None (无门控) | (B,T) float [0,1]
                            # - hard: 0/1 二值，1=保留，0=丢弃
                            # - soft: 连续权重，表示保留程度
                            if gate_mask is not None:
                                # 只统计 response 内的 token
                                gate_mask_masked = gate_mask * response_mask_float  # (B, T)
                                
                                # 平均保留权重（soft 模式下是连续值）
                                gate_retention_ratio = gate_mask_masked.sum(dim=-1) / response_token_counts.clamp(min=1.0)  # (B,)
                                metrics["tasd/gate_retention_ratio_mean"] = gate_retention_ratio.mean().item()
                                metrics["tasd/gate_retention_ratio_max"] = gate_retention_ratio.max().item()
                                metrics["tasd/gate_retention_ratio_min"] = gate_retention_ratio.min().item()
                                
                                # hard 模式：统计被丢弃的 token 数量
                                if tasd_entropy_gate == "hard":
                                    dropped_counts = (gate_mask_masked < 0.5).float().sum(dim=-1)  # (B,)
                                    metrics["tasd/gate_dropped_count_mean"] = dropped_counts.mean().item()
                                    metrics["tasd/gate_dropped_count_max"] = dropped_counts.max().item()
                                    metrics["tasd/gate_dropped_count_min"] = dropped_counts.min().item()
                            else:
                                # 无熵门控时，所有 token 都保留
                                metrics["tasd/gate_retention_ratio_mean"] = 1.0
                                metrics["tasd/gate_retention_ratio_max"] = 1.0
                                metrics["tasd/gate_retention_ratio_min"] = 1.0

                            # 按 success/fail 分组统计（需要 acc 字段）
                            if "acc" in batch.batch:
                                acc = batch.batch["acc"]
                            elif "acc" in batch.non_tensor_batch:
                                acc = torch.tensor(
                                    batch.non_tensor_batch["acc"],
                                    dtype=torch.float32,
                                    device=token_rewards.device,
                                )
                            else:
                                acc = None

                            if acc is not None:
                                tasd_success_threshold = _tasd_cfg.get("success_reward_threshold", 1.0)
                                success_mask_1d = acc >= tasd_success_threshold
                                fail_mask_1d = ~success_mask_1d

                                def _masked_valid_tokens_2d(tensor_2d, seq_mask_1d, token_mask):
                                    combined_mask = token_mask * seq_mask_1d.unsqueeze(-1).float()
                                    return tensor_2d[combined_mask.bool()]

                                success_rewards = _masked_valid_tokens_2d(token_rewards, success_mask_1d, response_mask_bool)
                                fail_rewards = _masked_valid_tokens_2d(token_rewards, fail_mask_1d, response_mask_bool)

                                if success_rewards.numel() > 0:
                                    metrics["tasd/token_reward_mean_success"] = success_rewards.mean().item()
                                    metrics["tasd/token_reward_std_success"] = (
                                        success_rewards.std().item() if success_rewards.numel() > 1 else 0.0
                                    )
                                if fail_rewards.numel() > 0:
                                    metrics["tasd/token_reward_mean_fail"] = fail_rewards.mean().item()
                                    metrics["tasd/token_reward_std_fail"] = (
                                        fail_rewards.std().item() if fail_rewards.numel() > 1 else 0.0
                                    )
                                metrics["tasd/success_rate"] = success_mask_1d.float().mean().item()
                        # ────────────────────────────────────────────────────────────

                        # ── DAPO filter_groups: 动态采样，过滤全对/全错的 group ────────
                        # 参考 dapo_ray_trainer.py 实现：
                        #   - 过滤后样本累积到 _fg_accumulated_batch
                        #   - 合格 prompt 不够则 continue 继续下一个 dataloader batch
                        #   - 有上限时（max_num_gen_batches>0）超限则 raise ValueError
                        _filter_groups_cfg = self.config.algorithm.get("filter_groups", None)
                        _filter_groups_enabled = (
                            _filter_groups_cfg is not None
                            and getattr(_filter_groups_cfg, "enable", False)
                        )
                        if _filter_groups_enabled:
                            _fg_metric = getattr(_filter_groups_cfg, "metric", "acc")
                            _fg_max_gen = getattr(_filter_groups_cfg, "max_num_gen_batches", 0)
                            _fg_num_gen_batches += 1

                            # 构建 metric 字段
                            if _fg_metric == "seq_final_reward":
                                batch.non_tensor_batch["seq_final_reward"] = (
                                    batch.batch["token_level_rewards"].sum(dim=-1).cpu().numpy()
                                )
                            elif _fg_metric == "seq_reward":
                                batch.non_tensor_batch["seq_reward"] = (
                                    batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
                                )

                            # 按 uid 统计 metric std，过滤全同的 group
                            uid_to_metric_vals = defaultdict(list)
                            for uid_val, metric_val in zip(
                                batch.non_tensor_batch["uid"],
                                batch.non_tensor_batch[_fg_metric],
                            ):
                                uid_to_metric_vals[uid_val].append(float(metric_val))

                            kept_uids = [
                                uid for uid, vals in uid_to_metric_vals.items()
                                if np.std(vals) > 0 or len(vals) == 1
                            ]
                            kept_idxs = [
                                i for i, uid_val in enumerate(batch.non_tensor_batch["uid"])
                                if uid_val in kept_uids
                            ]

                            _fg_num_prompt_in_batch += len(kept_uids)
                            filtered_batch = batch[kept_idxs]
                            _fg_accumulated_batch = (
                                filtered_batch if _fg_accumulated_batch is None
                                else concat_dataproto_with_padding([_fg_accumulated_batch, filtered_batch])
                            )

                            metrics["filter_groups/n_prompts_before"] = len(uid_to_metric_vals)
                            metrics["filter_groups/n_prompts_kept"] = len(kept_uids)
                            metrics["filter_groups/keep_ratio"] = len(kept_uids) / max(len(uid_to_metric_vals), 1)
                            metrics["filter_groups/num_gen_batches"] = _fg_num_gen_batches
                            metrics["filter_groups/num_prompt_accumulated"] = _fg_num_prompt_in_batch

                            prompt_bsz = self.config.data.train_batch_size
                            if _fg_num_prompt_in_batch < prompt_bsz:
                                print(f"[filter_groups] {_fg_num_prompt_in_batch=} < {prompt_bsz=}")
                                if _fg_max_gen <= 0 or _fg_num_gen_batches < _fg_max_gen:
                                    print(f"[filter_groups] {_fg_num_gen_batches=}. Keep generating...")
                                    continue
                                else:
                                    raise ValueError(
                                        f"[filter_groups] {_fg_num_gen_batches=} >= {_fg_max_gen=}. "
                                        "Generated too many batches. Check if data are too easy/hard, "
                                        "or set max_num_gen_batches=0 for unlimited retries."
                                    )

                            # 样本足够，裁剪到 traj_bsz 并重置累积状态
                            traj_bsz = prompt_bsz * self.config.actor_rollout_ref.rollout.n
                            batch = _fg_accumulated_batch[:traj_bsz]
                            _fg_accumulated_batch = None
                            _fg_num_prompt_in_batch = 0
                            _fg_num_gen_batches = 0
                        # ────────────────────────────────────────────────────────────

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        # ── TASD: 记录 advantage 分布指标 ───────────────────────────────
                        if is_tasd:
                            valid_adv = batch.batch["advantages"][batch.batch["response_mask"].bool()]
                            metrics["tasd/advantage_mean"] = valid_adv.mean().item()
                            metrics["tasd/advantage_std"] = valid_adv.std().item() if valid_adv.numel() > 1 else 0.0
                            metrics["tasd/advantage_pos_rate"] = (valid_adv > 0).float().mean().item()

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
                    # save best checkpoint if val metric improved
                    self._maybe_save_best_checkpoint(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                # Check for training anomalies and send DingTalk alerts
                from verl.utils.tracking import check_training_anomalies
                check_training_anomalies(
                    metrics=metrics,
                    step=self.global_steps,
                    experiment_name=self.config.trainer.get("experiment_name", ""),
                )

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()

                    # Send DingTalk notification: training completed
                    from verl.utils.tracking import send_dingtalk_alert
                    _exp_name = self.config.trainer.get("experiment_name", "")
                    _msg_lines = [f"✅ Training completed!", f"  experiment: {_exp_name}", f"  total_steps: {self.global_steps}"]
                    # Add last val metrics
                    if last_val_metrics:
                        _msg_lines.append("  ── Last val metrics ──")
                        for _k in sorted(last_val_metrics.keys()):
                            if "val-core" in _k or "val-aux/sciknoweval/acc/mean" in _k:
                                _v = last_val_metrics[_k]
                                if isinstance(_v, float):
                                    _msg_lines.append(f"  {_k}: {_v:.4f}")
                                else:
                                    _msg_lines.append(f"  {_k}: {_v}")
                    # Add best historical metric
                    if self._best_val_metric is not None:
                        _best_metric_key = self.config.trainer.get("save_best_metric", "val-core/acc/mean@16")
                        _msg_lines.append(f"  ── Best historical ──")
                        _msg_lines.append(f"  {_best_metric_key}: {self._best_val_metric:.4f} (step {self._best_val_step})")
                    # Add training time
                    _timing = metrics.get("timing_s/step", None)
                    if _timing is not None:
                        _total_time_h = _timing * self.global_steps / 3600
                        _msg_lines.append(f"  total_time: ~{_total_time_h:.1f}h")
                    send_dingtalk_alert("\n".join(_msg_lines))

                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

        # Fallback: training ended by epoch exhaustion (not is_last_step path)
        # Send DingTalk notification here so we never miss a completion
        from verl.utils.tracking import send_dingtalk_alert
        _exp_name = self.config.trainer.get("experiment_name", "")
        _fb_msg_lines = [
            f"✅ Training completed (epoch end)!",
            f"  experiment: {_exp_name}",
            f"  total_steps: {self.global_steps - 1}",
        ]
        if last_val_metrics:
            _fb_msg_lines.append("  ── Last val metrics ──")
            for _k in sorted(last_val_metrics.keys()):
                if "val-core" in _k or "val-aux/sciknoweval/acc/mean" in _k:
                    _v = last_val_metrics[_k]
                    _fb_msg_lines.append(f"  {_k}: {_v:.4f}" if isinstance(_v, float) else f"  {_k}: {_v}")
        if self._best_val_metric is not None:
            _best_metric_key = self.config.trainer.get("save_best_metric", "val-core/acc/mean@16")
            _fb_msg_lines.append(f"  ── Best historical ──")
            _fb_msg_lines.append(f"  {_best_metric_key}: {self._best_val_metric:.4f} (step {self._best_val_step})")
        send_dingtalk_alert("\n".join(_fb_msg_lines))
