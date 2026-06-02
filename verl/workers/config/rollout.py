# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.utils.profiler import ProfilerConfig

__all__ = [
    "SamplingConfig",
    "MultiTurnConfig",
    "CustomAsyncServerConfig",
    "AgentLoopConfig",
    "BranchingConfig",
    "TraceConfig",
    "ServerConfig",
    "PrometheusConfig",
    "RolloutConfig",
]


@dataclass
class SamplingConfig(BaseConfig):
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    n: int = 1


@dataclass
class MultiTurnConfig(BaseConfig):
    _mutable_fields = {"max_assistant_turns", "max_user_turns"}

    enable: bool = False
    max_assistant_turns: Optional[int] = None
    tool_config_path: Optional[str] = None
    max_user_turns: Optional[int] = None
    max_parallel_calls: int = 1
    max_tool_response_length: int = 256
    tool_response_truncate_side: str = "middle"
    interaction_config_path: Optional[str] = None
    use_inference_chat_template: bool = False
    tokenization_sanity_check_mode: str = "strict"
    format: str = "hermes"
    num_repeat_rollouts: Optional[int] = None


@dataclass
class CustomAsyncServerConfig(BaseConfig):
    path: Optional[str] = None
    name: Optional[str] = None


@dataclass
class AgentLoopConfig(BaseConfig):
    num_workers: int = 8
    default_agent_loop: str = "single_turn_agent"
    agent_loop_config_path: Optional[str] = None
    custom_async_server: CustomAsyncServerConfig = field(default_factory=CustomAsyncServerConfig)
    # Fully qualified class name for custom AgentLoopManager (e.g., "mypackage.module.MyManager").
    # Security: This class will be dynamically imported via importlib. Only use trusted class paths.
    agent_loop_manager_class: Optional[str] = None


@dataclass
class BranchingConfig(BaseConfig):
    """Configuration for teacher-guided branching rollout.

    See research/teacher_branching_rollout.md.

    Attributes:
        enabled: Master switch. When False (default) the rollout falls back
            to the standard single_turn_agent behaviour.
        n_splits: Number of binary splits per prompt. With rollout_n=8 the
            balanced tree depth is 3, yielding 2^n_splits = 8 leaves.
        top_k: How many top logprobs vLLM should return per generated token.
            Used both for student entropy estimation and for the teacher
            branch-point query. K=50 is the default; raise if the student/
            teacher top-K intersection is too thin.
        entropy_window: Window size W for the rolling z-score.
        entropy_protect_window: Protect window — first P tokens are never
            flagged as decision tokens (running stats unstable). Defaults to W.
        entropy_sigma_start: Initial threshold k in H_t > mean + k*std.
        entropy_sigma_floor: Lower bound during σ-relaxation.
        entropy_sigma_step: How much to lower σ per relaxation iteration.
        teacher_context_mode: Which teacher context to use at branch points.
            One of {marker, gt_marker, ref_gt}. ref-from-peer is NOT supported
            here because peer rollouts haven't completed during branching.
        verdict_right_marker: Override for the verified-correct marker text.
            None falls back to verl.utils.verdict_markers.VERDICT_RIGHT_MARKER.
        gt_marker_template: Per-sample template; available placeholder
            ``{ground_truth}``. Used when teacher_context_mode == "gt_marker".
        reprompt_template: Used when teacher_context_mode == "ref_gt".
            Available placeholders: ``{prompt}``, ``{solution}``, ``{feedback}``.
        solution_template: Solution-section template inside ``reprompt_template``.
            Available placeholder: ``{successful_previous_attempt}``.
        max_reprompt_len: Cap on the rebuilt teacher prompt length under
            ref_gt mode. Mirrored from actor.self_distillation.max_reprompt_len.
        teacher_branch_query_max_tokens: How many tokens to generate when
            asking the teacher engine to score top-K (always 1 in V1).
        fallback_to_student_topk: If teacher top-K and student top-K intersect
            in fewer than 2 tokens, fall back to student's own top-1 / top-2.
        require_distinct_branches: If both children would carry the same
            token, treat the branch as failed (skip splitting at that
            position). Otherwise drop one child.
        max_branch_depth: Optional cap on tree depth (defaults to n_splits).
            Useful for ablations that want a deeper detector budget than the
            actual leaf count.
    """

    enabled: bool = False
    n_splits: int = 3
    # ``top_k`` = how many logprobs vLLM returns at each STUDENT generation
    # position. Defines (a) the depth used for truncated-entropy estimation
    # (z-score detector consumes this dict), and (b) the candidate pool the
    # teacher will choose argmax/argmin from at branch points.
    #
    # Lowered from 50 → 10 after the first pilot showed K=50 caused severe
    # OOD pressure: teacher's argmin within student's top-50 is by definition
    # a token the student would almost never sample (rank-50 student logprob
    # ~1e-3 to 1e-5), and forcing the chain to continue from it produces an
    # OOD prefix and noise-dominated PG. K=10 keeps both branches inside the
    # plausible student support.
    top_k: int = 10
    # ``teacher_top_k`` = how many logprobs vLLM returns from the TEACHER
    # branch-point query. This is independent from ``top_k``. We want it
    # *large* so that the teacher's top-K covers all of student's top-K
    # tokens — otherwise pick_teacher_branches sees only the
    # student∩teacher intersection, dropping (student-only) candidates from
    # the argmax/argmin selection. With teacher_top_k >> top_k the
    # intersection ≈ student top-K, so every student candidate effectively
    # gets a teacher score. Default 50 requires bumping vLLM's engine
    # ``max_logprobs`` ≥ 50 (parametric scripts do this automatically).
    teacher_top_k: int = 50
    # Optional sub-selection: how many of student's top-K we offer the
    # teacher for argmax/argmin. None ⇒ use top_k (consider all student
    # candidates). Set lower (e.g. 5) to keep both branches strictly in the
    # student's high-probability core.
    teacher_pick_top_k: Optional[int] = None
    entropy_window: int = 20
    entropy_protect_window: Optional[int] = None
    entropy_sigma_start: float = 2.0
    entropy_sigma_floor: float = 0.5
    entropy_sigma_step: float = 0.5
    teacher_context_mode: str = "gt_marker"
    verdict_right_marker: Optional[str] = None
    gt_marker_template: str = (
        "[Meta: the assistant response below is verified to correctly answer the question. "
        "Reference answer: {ground_truth}]"
    )
    reprompt_template: str = (
        "{prompt}{solution}{feedback}\n\n"
        "Correctly solve the original question.\n"
    )
    solution_template: str = (
        "\n"
        "Correct solution:\n\n"
        "{successful_previous_attempt}\n\n"
    )
    max_reprompt_len: int = 10240
    teacher_branch_query_max_tokens: int = 1
    fallback_to_student_topk: bool = True
    require_distinct_branches: bool = True
    max_branch_depth: Optional[int] = None

    def __post_init__(self):
        if self.enabled:
            if self.n_splits < 1:
                raise ValueError(f"branching.n_splits must be >= 1, got {self.n_splits}")
            if self.top_k < 2:
                raise ValueError(f"branching.top_k must be >= 2, got {self.top_k}")
            if self.entropy_window <= 1:
                raise ValueError(f"branching.entropy_window must be > 1, got {self.entropy_window}")
            if self.entropy_sigma_floor > self.entropy_sigma_start:
                raise ValueError(
                    "branching.entropy_sigma_floor must be <= entropy_sigma_start, "
                    f"got floor={self.entropy_sigma_floor} start={self.entropy_sigma_start}"
                )
            valid_modes = {"marker", "gt_marker", "ref_gt"}
            if self.teacher_context_mode not in valid_modes:
                raise ValueError(
                    f"branching.teacher_context_mode must be one of {valid_modes}, "
                    f"got {self.teacher_context_mode!r}"
                )

    # NOTE: ``@property`` accessors are NOT visible when the dataclass is read
    # back through OmegaConf as a DictConfig (which is what happens at runtime
    # in ``BranchingAgentLoop`` via ``rollout.get("branching")``). Callers that
    # only have a DictConfig MUST inline the equivalent logic (resolve
    # ``entropy_protect_window``/``max_branch_depth`` against their defaults
    # manually). These properties are kept here for tests and direct instances
    # that hold the live dataclass.

    @property
    def effective_protect_window(self) -> int:
        return self.entropy_protect_window if self.entropy_protect_window is not None else self.entropy_window

    @property
    def effective_max_branch_depth(self) -> int:
        return self.max_branch_depth if self.max_branch_depth is not None else self.n_splits


@dataclass
class TraceConfig(BaseConfig):
    backend: Optional[str] = None
    token2text: bool = False
    max_samples_per_step_per_worker: Optional[int] = None

    def __post_init__(self):
        if self.max_samples_per_step_per_worker is not None and self.max_samples_per_step_per_worker < 0:
            raise ValueError("`max_samples_per_step_per_worker` must be a non-negative integer or null.")


@dataclass
class ServerConfig(BaseConfig):
    """
    Configuration for SGLang server when running in server mode
    """

    timeout: float = 60.0
    max_attempts: int = 3
    retry_delay: float = 2.0
    max_connections: int = 1000
    max_start_wait_time: float = 300.0


@dataclass
class PrometheusConfig(BaseConfig):
    """
    Configuration for Prometheus server
    """

    # whether enable prometheus on server mode rollout
    enable: bool = False
    # Port number that Prometheus listens on, default is 9090
    port: int = 9090
    # Path to Prometheus configuration file
    file: str = "/tmp/ray/session_latest/metrics/prometheus/prometheus.yml"
    # Specify served_model_name to avoid displaying overly long model paths in Grafana
    served_model_name: Optional[str] = None


@dataclass
class RolloutConfig(BaseConfig):
    _mutable_fields = {"max_model_len", "load_format"}

    name: Optional[str] = MISSING
    mode: str = "async"

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    do_sample: bool = True
    n: int = 1
    repetition_penalty: float = 1.0

    # Early termination threshold for multi-turn rollout in sglang.
    # Abort remaining requests when (1 - over_sample_rate) * total_requests are completed.
    over_sample_rate: float = 0.0

    prompt_length: int = 512
    response_length: int = 512

    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    ignore_eos: bool = False
    enforce_eager: bool = True
    cudagraph_capture_sizes: Optional[list] = None
    free_cache_engine: bool = True
    data_parallel_size: int = 1
    expert_parallel_size: int = 1
    tensor_model_parallel_size: int = 2
    pipeline_model_parallel_size: int = 1
    max_num_batched_tokens: int = 8192
    logprobs_mode: Optional[str] = "processed_logprobs"
    scheduling_policy: Optional[str] = "fcfs"

    # TODO: enable train_kwargs
    # train_sampling_config: SamplingConfig = field(default_factory=SamplingConfig)

    val_kwargs: SamplingConfig = field(default_factory=SamplingConfig)

    max_model_len: Optional[int] = None
    max_num_seqs: int = 1024

    # note that the logprob computation should belong to the actor
    log_prob_micro_batch_size: Optional[int] = None
    log_prob_micro_batch_size_per_gpu: Optional[int] = None
    log_prob_use_dynamic_bsz: bool = False
    log_prob_max_token_len_per_gpu: int = 16384

    disable_log_stats: bool = True

    multi_stage_wake_up: bool = False
    engine_kwargs: dict = field(default_factory=dict)

    calculate_log_probs: bool = False

    agent: AgentLoopConfig = field(default_factory=AgentLoopConfig)

    trace: TraceConfig = field(default_factory=TraceConfig)

    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)

    # Teacher-guided branching rollout (off by default).
    branching: BranchingConfig = field(default_factory=BranchingConfig)

    # Server configuration for sglang server mode
    server: ServerConfig = field(default_factory=ServerConfig)

    # Use Prometheus to collect and monitor rollout statistics
    prometheus: PrometheusConfig = field(default_factory=PrometheusConfig)

    # Extension point for custom configurations
    custom: Optional[dict] = None

    update_weights_bucket_megabytes: int = 512

    skip_rollout: bool = False

    skip_dump_dir: str = "/tmp/rollout_dump"

    profiler: Optional[ProfilerConfig] = None

    enable_chunked_prefill: bool = True

    enable_prefix_caching: bool = True

    load_format: str = "dummy"

    layered_summon: bool = False

    layer_name_map: dict = field(default_factory=dict)

    sglang_engine_mode: str = "local"

    limit_images: Optional[int] = None

    skip_tokenizer_init: bool = False

    quantization: Optional[str] = None

    quantization_config_file: Optional[str] = None

    enable_rollout_routing_replay: bool = False

    enable_sleep_mode: bool = True

    def __post_init__(self):
        """Validate the rollout config"""
        # Deprecation warning for mode field - only async mode is supported
        if self.mode == "sync":
            raise ValueError(
                "Rollout mode 'sync' has been removed. Please set "
                "`actor_rollout_ref.rollout.mode=async` or remove the mode setting entirely."
            )
        if self.mode != "async":
            warnings.warn(
                f"Unknown rollout mode '{self.mode}'. Only 'async' mode is supported. "
                "The 'mode' field is deprecated and will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )

        if self.expert_parallel_size > 1:
            assert self.expert_parallel_size == (self.tensor_model_parallel_size * self.data_parallel_size), (
                "expert_parallel_size must be equal to tensor_model_parallel_size * data_parallel_size"
            )

        if self.pipeline_model_parallel_size > 1:
            if self.name == "vllm" or self.name == "sglang":
                raise NotImplementedError(
                    f"Current rollout {self.name=} not implemented pipeline_model_parallel_size > 1 yet."
                )
