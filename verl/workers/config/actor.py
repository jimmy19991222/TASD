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

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "SelfDistillationConfig",
    "PolicyLossConfig",
    "RouterReplayConfig",
    "ActorConfig",
    "FSDPActorConfig",
    "McoreActorConfig",
]


@dataclass
class SelfDistillationConfig(BaseConfig):
    """Configuration for self-distillation loss.

    Args:
        Distillation is enabled when policy_loss.loss_mode == "sdpo".
        full_logit_distillation (bool): Whether to use full-logit KL distillation.
        alpha (float): KL interpolation coefficient. 0.0=forward KL, 1.0=reverse KL, in-between=JSD.
        success_reward_threshold (float): Minimum sequence reward to be considered successful.
        teacher_regularization (str): Teacher regularization mode. Options: "ema", "trust-region".
        teacher_update_rate (float): EMA update rate for teacher weights, or trust-region mixing coefficient.
        distillation_topk (Optional[int]): If set, use top-k logits for distillation.
        distillation_add_tail (bool): Whether to add a tail bucket for top-k distillation.
        max_reprompt_len (int): Maximum length of the reprompted prompt.
        reprompt_truncation (str): Truncation method for the reprompted prompt (recommended to use "right" or "error").
        dont_reprompt_on_self_success (bool): Whether to not reprompt on self-success.
        remove_thinking_from_demonstration (bool): Whether to remove <think>...</think> tags from successful demonstrations before reprompting.
        is_clip (Optional[float]): Clip value for distillation IS ratio; None disables IS weighting.
        reprompt_template (str): Template for reprompting. Uses {prompt}, {solution}, {feedback} placeholders.
        solution_template (str): Template for formatting solution section. Uses {successful_previous_attempt} placeholder.
        feedback_template (str): Template for formatting feedback section. Uses {feedback_raw} placeholder.
        include_environment_feedback (bool): Whether to include environment feedback in reprompting for wrong attempts.
        environment_feedback_only_without_solution (bool): If True, only use feedback when no solution is available (ignore feedback when solution exists).
        reprompt_template_feedback (str): Template for reprompting with feedback but no solution.
        reprompt_template_feedback_solution (str): Template for reprompting with both feedback and solution.
        teacher_context_mode (str): How the teacher's privileged context is constructed.
            - "ref": peer-rollout response from same uid (default; legacy SDPO behaviour).
            - "marker": prepend a static verdict marker to the assistant response.
            - "gt_marker": prepend a per-sample marker derived from the parquet ``ground_truth`` field.
            - "ref_gt": use the parquet ``ground_truth`` field as the ref text (does not require peer rollouts).
            - "cdm": dual prompts (positive/negative verdict markers) for TG-GRPO contrastive
              distillation modeling. Emits two teacher input tensor sets (pos/neg).
        verdict_right_marker (Optional[str]): Override for the verified-correct marker (default: VERDICT_RIGHT_MARKER).
        verdict_wrong_marker (Optional[str]): Override for the verified-incorrect marker (default: VERDICT_WRONG_MARKER).
        gt_marker_template (str): Template used by ``gt_marker`` mode. Available placeholders: ``{ground_truth}``.
        loss_method (str): Distillation loss family. ``"sdpo"`` (default) keeps the legacy
            self-distillation KL/IS path. ``"cdm"`` switches to TG-GRPO's contrastive
            distillation modeling — the trainer emits dual (positive/negative) teacher
            inputs and the actor computes |δ_t| = |log π_T+ - log π_T-| as the per-token
            weight selector for the policy loss.
        cdm_positive_template (str): Verdict text used to construct the positive teacher
            prompt under ``teacher_context_mode="cdm"`` (e.g. "This answer is verified correct.").
            Supports ``{ground_truth}`` placeholder.
        cdm_negative_template (str): Verdict text used to construct the negative teacher
            prompt under ``teacher_context_mode="cdm"`` (e.g. "This answer is verified incorrect.").
            Supports ``{ground_truth}`` placeholder.
        cdm_use_ref (bool): If True, also concatenate a sibling-success peer-rollout response
            (same-uid ref text, like ``teacher_context_mode="ref"``) into the teacher prompt
            in addition to the verdict marker ("ref+marker" variant). If False, marker-only.
    """

    full_logit_distillation: bool = True
    alpha: float = 0.0
    success_reward_threshold: float = 1.0
    teacher_regularization: str = "ema"
    teacher_update_rate: float = 0.05
    distillation_topk: Optional[int] = None
    distillation_add_tail: bool = True
    max_reprompt_len: int = 10240
    reprompt_truncation: str = "right"
    dont_reprompt_on_self_success: bool = False
    remove_thinking_from_demonstration: bool = False
    is_clip: Optional[float] = None
    reprompt_template: str = (
        "{prompt}{solution}{feedback}\n\n"
        "Correctly solve the original question.\n"
    )
    solution_template: str = (
        "\n"
        "Correct solution:\n\n"
        "{successful_previous_attempt}\n\n"
    )
    feedback_template: str = (
        "\n"
        "The following is feedback from your unsuccessful earlier attempt:\n\n"
        "{feedback_raw}\n\n"
    )
    include_environment_feedback: bool = False
    environment_feedback_only_without_solution: bool = False
    teacher_context_mode: str = "ref"
    verdict_right_marker: Optional[str] = None
    verdict_wrong_marker: Optional[str] = None
    gt_marker_template: str = (
        "[Meta: the assistant response below is verified to correctly answer the question. "
        "Reference answer: {ground_truth}]"
    )
    # ── TG-GRPO / CDM dual-teacher fields ─────────────────────────────────
    loss_method: str = "sdpo"
    cdm_positive_template: str = "This answer is verified correct."
    cdm_negative_template: str = "This answer is verified incorrect."
    cdm_use_ref: bool = True

    def __post_init__(self):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"self_distillation.alpha must be in [0,1], got {self.alpha}")
        valid_teacher_regularization = ["ema", "trust-region"]
        if self.teacher_regularization not in valid_teacher_regularization:
            raise ValueError(
                "self_distillation.teacher_regularization must be one of "
                f"{valid_teacher_regularization}, got {self.teacher_regularization}"
            )
        if not 0.0 <= self.teacher_update_rate <= 1.0:
            raise ValueError(
                f"self_distillation.teacher_update_rate must be in [0,1], got {self.teacher_update_rate}"
            )
        if self.distillation_topk is not None and self.distillation_topk <= 0:
            raise ValueError(
                f"self_distillation.distillation_topk must be a positive integer, got {self.distillation_topk}"
            )
        if self.is_clip is not None and self.is_clip <= 0:
            raise ValueError(f"self_distillation.is_clip must be positive, got {self.is_clip}")
        valid_teacher_context_modes = ["ref", "marker", "gt_marker", "ref_gt", "cdm"]
        if self.teacher_context_mode not in valid_teacher_context_modes:
            raise ValueError(
                "self_distillation.teacher_context_mode must be one of "
                f"{valid_teacher_context_modes}, got {self.teacher_context_mode}"
            )
        valid_loss_methods = ["sdpo", "cdm"]
        if self.loss_method not in valid_loss_methods:
            raise ValueError(
                "self_distillation.loss_method must be one of "
                f"{valid_loss_methods}, got {self.loss_method}"
            )
        if self.loss_method == "cdm" and self.teacher_context_mode != "cdm":
            raise ValueError(
                "self_distillation.loss_method='cdm' requires teacher_context_mode='cdm', "
                f"got teacher_context_mode={self.teacher_context_mode!r}"
            )
        if self.teacher_context_mode == "cdm" and self.loss_method != "cdm":
            raise ValueError(
                "self_distillation.teacher_context_mode='cdm' requires loss_method='cdm', "
                f"got loss_method={self.loss_method!r}"
            )


@dataclass
class RouterReplayConfig(BaseConfig):
    """Configuration for router replay in MoE models.

    This configuration controls the routing behavior for Mixture of Experts (MoE) models,
    allowing for deterministic training through route recording and replay.

    Args:
        mode (str): Router replay mode. Options: 'disabled', 'R2', 'R3'.
            - 'disabled': No router replay functionality
            - 'R2': Use Router Replay routing strategy
            - 'R3': Use Rollout Router Replay routing strategy
        record_file (Optional[str]): File path to save recorded routing decisions.
            Required when mode is 'record', 'R2', or 'R3'.
        replay_file (Optional[str]): File path to load recorded routing decisions for replay.
            Required when mode is 'replay'.
    """

    mode: str = "disabled"
    record_file: Optional[str] = None
    replay_file: Optional[str] = None

    def __post_init__(self):
        """Validate router replay configuration."""
        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg', 'sdpo', 'tg_grpo'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
        tg_top_k (float): Fraction of tokens to keep for TG-GRPO percentile-based modes
            (``topk_traj``, ``topk_batch``). E.g. 0.30 keeps top-30% by |δ_t|.
        tg_w_mode (str): TG-GRPO per-token weight mode w_t = f(|δ_t|). Options:

            - ``topk_traj`` (default, recommended): per-trajectory top-k·|response_mask| hard mask.
            - ``topk_batch``: global top-k across the whole batch (length-biased).
            - ``sigmoid``: soft weighting w_t = sigmoid(β · (|δ_t| - median(|δ|))).
            - ``hard``: per-token threshold w_t = 1{|δ_t| >= τ}.
        tg_w_sigmoid_beta (float): Sharpness β for sigmoid mode.
        tg_w_hard_threshold (float): Threshold τ for hard mode.
        branch_token_loss_mode (str): How the teacher-guided branching rollout's
            ``branch_token_mask`` weights the policy loss. Options:

            - ``all`` (default): mask is ignored; every response token contributes
              to the loss (response_mask only).
            - ``mask``: zero-out the branch tokens before PG aggregation
              (response_mask AND NOT branch_token_mask). Useful to remove
              off-policy bias of teacher-injected tokens.
            - ``only``: only branch tokens contribute (response_mask AND
              branch_token_mask). The "decision-token-only" hypothesis ablation.

            No-op when ``branch_token_mask`` is absent from the data batch
            (i.e., branching rollout is disabled).
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1
    # ── TG-GRPO per-token weight knobs ────────────────────────────────────
    tg_top_k: float = 0.30
    tg_w_mode: str = "topk_traj"
    tg_w_sigmoid_beta: float = 5.0
    tg_w_hard_threshold: float = 0.3
    branch_token_loss_mode: str = "all"

    def __post_init__(self):
        valid = {"all", "mask", "only"}
        if self.branch_token_loss_mode not in valid:
            raise ValueError(
                f"policy_loss.branch_token_loss_mode must be one of {valid}, "
                f"got {self.branch_token_loss_mode!r}"
            )
        valid_tg_modes = {"topk_traj", "topk_batch", "sigmoid", "hard"}
        if self.tg_w_mode not in valid_tg_modes:
            raise ValueError(
                f"policy_loss.tg_w_mode must be one of {valid_tg_modes}, "
                f"got {self.tg_w_mode!r}"
            )
        if not 0.0 < self.tg_top_k <= 1.0:
            raise ValueError(
                f"policy_loss.tg_top_k must be in (0,1], got {self.tg_top_k}"
            )
        if self.tg_w_sigmoid_beta <= 0.0:
            raise ValueError(
                f"policy_loss.tg_w_sigmoid_beta must be positive, got {self.tg_w_sigmoid_beta}"
            )


@dataclass
class ActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.
            If None, uses ppo_micro_batch_size_per_gpu.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.
        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.
        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_ratio_low (float): Lower bound for PPO clipping ratio.
        clip_ratio_high (float): Upper bound for PPO clipping ratio.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        clip_ratio_c (float): Clipping ratio for critic loss.
        loss_agg_mode (str): Loss aggregation mode. Options: 'token-mean', 'sample-mean'.
        loss_scale_factor (Optional[int]): Scale factor for 'seq-mean-token-sum-norm' loss aggregation mode.
            If None, uses response_length. Set to a constant to ensure consistent normalization.
        entropy_coeff (float): Entropy coefficient for regularization.
        tau_pos (float): Positive tau for SAPO smoothing (>= 1.0 keeps rewards stable).
        tau_neg (float): Negative tau for SAPO smoothing (> tau_pos for asymmetry).
        use_kl_loss (bool): Whether to use KL divergence loss.
        use_torch_compile (bool): Whether to use torch.compile for optimization.
        kl_loss_coef (float): KL divergence loss coefficient.
        kl_loss_type (str): Type of KL loss to use.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
        data_loader_seed (int): Seed for data loader. If None, uses global seed.
        router_replay (RouterReplayConfig): Configuration for router replay in MoE models.
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None  # deprecate
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[int] = None
    entropy_coeff: float = 0
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    calculate_entropy: bool = False
    use_kl_loss: bool = False
    # Whether to enable PrefixGrouper-based shared-prefix forward
    use_prefix_grouper: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 1
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING  # must be override by sampling config
    model_config: HFModelConfig = field(default_factory=BaseConfig)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)
    self_distillation: SelfDistillationConfig = field(default_factory=SelfDistillationConfig)

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        load_weight (bool): Whether to load model weights from checkpoint.
        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
    """

    strategy: str = "megatron"
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.megatron


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): [DEPRECATED] Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False
    calculate_sum_pi_squared: bool = False
    sum_pi_squared_checkpointing: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.fsdp_config

        # backward compatibility
        if self.ulysses_sequence_parallel_size > 1:
            self.fsdp_config.ulysses_sequence_parallel_size = self.ulysses_sequence_parallel_size

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )
