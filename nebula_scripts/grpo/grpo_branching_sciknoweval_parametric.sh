#!/usr/bin/env bash
# =============================================================================
# Teacher-Guided Branching GRPO 参数化训练脚本（sciknoweval 数据集）
# 所有超参通过 nebulactl --env 注入。
# Mirrors grpo_sciknoweval_parametric.sh + 启用 actor_rollout_ref.rollout.branching.
# =============================================================================
set +xo pipefail

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"

# ── 从环境变量读取超参 ────────────────────────────────────────────────
check_env() { val=$(eval echo "\$$1"); [ -n "$val" ] || { echo "ERROR: $1 is not set. Aborting."; exit 1; }; }
check_env DATASET
check_env LR
check_env MINI_BATCH_SIZE
check_env TRAIN_BATCH_SIZE
check_env ROLLOUT_N
check_env MODEL_NAME

SEED="${SEED:-42}"

# Checkpoint / validation cadence (env-overridable)
TEST_FREQ="${TEST_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-10}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
SAVE_HF_ONLY="${SAVE_HF_ONLY:-True}"
if [ "${SAVE_HF_ONLY}" = "True" ]; then
    SAVE_CONTENTS_HYDRA="[hf_model]"
else
    SAVE_CONTENTS_HYDRA="[model,optimizer,extra,hf_model]"
fi

# ── Branching 专属超参 ────────────────────────────────────────────────
BRANCHING_ENABLED="${BRANCHING_ENABLED:-True}"
N_SPLITS="${N_SPLITS:-3}"
# N-trees topology: 1 prompt -> N_TREES independent trees, each tree has
# 2**N_SPLITS leaves. ROLLOUT_N must equal N_TREES * 2**N_SPLITS.
N_TREES="${N_TREES:-1}"
# split_trigger in {entropy, entropy_disagreement}. The latter only splits at
# positions where teacher's argmax-in-student-topK disagrees with student's
# actual sampled token (requires teacher_guided_rollout).
SPLIT_TRIGGER="${SPLIT_TRIGGER:-entropy}"
# Asymmetric K: student ``top_k`` is the candidate pool depth (used for
# entropy estimation + teacher pick), kept tight to avoid OOD branches.
# ``teacher_top_k`` is the teacher's logprob depth — kept LARGE so the
# teacher's top-K covers all of student's top-K candidates with high
# probability (otherwise pick_teacher_branches sees a thin intersection).
TOP_K="${TOP_K:-10}"
TEACHER_TOP_K="${TEACHER_TOP_K:-100}"
# vLLM engine max_logprobs must be >= max(TOP_K, TEACHER_TOP_K). The engine
# default is 20 so anything above triggers VLLMValidationError.
VLLM_MAX_LOGPROBS_REQUIRED=$(( TOP_K > TEACHER_TOP_K ? TOP_K : TEACHER_TOP_K ))
VLLM_MAX_LOGPROBS="${VLLM_MAX_LOGPROBS:-${VLLM_MAX_LOGPROBS_REQUIRED}}"
ENTROPY_WINDOW="${ENTROPY_WINDOW:-20}"
ENTROPY_SIGMA_START="${ENTROPY_SIGMA_START:-2.0}"
ENTROPY_SIGMA_FLOOR="${ENTROPY_SIGMA_FLOOR:-0.5}"
ENTROPY_SIGMA_STEP="${ENTROPY_SIGMA_STEP:-0.5}"
TEACHER_CONTEXT_MODE="${TEACHER_CONTEXT_MODE:-gt_marker}"
BRANCH_TOKEN_LOSS_MODE="${BRANCH_TOKEN_LOSS_MODE:-mask}"
ADV_STD_FLOOR="${ADV_STD_FLOOR:-0.05}"
DEFAULT_AGENT_LOOP="${DEFAULT_AGENT_LOOP:-branching_agent}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"
# Validation rollout count (val_kwargs.n). Override per-dataset for fair comparison.
VAL_N="${VAL_N:-16}"
# Two-stage branching (Stage 1 normal + Stage 2 branching)
TWO_STAGE="${TWO_STAGE:-False}"
STAGE1_N="${STAGE1_N:-4}"
TWO_STAGE_TEACHER_MODE="${TWO_STAGE_TEACHER_MODE:-ref_or_marker}"
# Success threshold for Stage 1 scoring (default 1.0 = perfect score required)
SUCCESS_THRESHOLD="${SUCCESS_THRESHOLD:-1.0}"
# Entropy regularization (nonzero adds entropy bonus to policy loss)
ENTROPY_COEFF="${ENTROPY_COEFF:-0}"
# DPO reward shaping: teacher preference signal for Stage 2 samples
DPO_COEFFICIENT="${DPO_COEFFICIENT:-0}"
DPO_USE_REF="${DPO_USE_REF:-False}"
# Teacher-Guided β: adaptive β based on teacher margin at branch points
DPO_TEACHER_GUIDED_BETA="${DPO_TEACHER_GUIDED_BETA:-False}"
DPO_TEACHER_BETA_ALPHA="${DPO_TEACHER_BETA_ALPHA:-1.0}"
DPO_TEACHER_BETA_MIN="${DPO_TEACHER_BETA_MIN:-0.1}"
DPO_TEACHER_BETA_MAX="${DPO_TEACHER_BETA_MAX:-3.0}"
# Stage 1 middle-version DPO pairing: use Stage 1 student chain as "middle"
# version in 3-way ranking (argmax > stage1 > argmin), turning 1 pair into 2
DPO_STAGE1_PAIR="${DPO_STAGE1_PAIR:-False}"
STAGE1_PAIR_WEIGHT="${STAGE1_PAIR_WEIGHT:-1.0}"
# Reward-consistency filter: drop pairs where chosen leaf's reward < rejected's
DPO_REWARD_FILTER="${DPO_REWARD_FILTER:-True}"

# Validation metric — derived from DATASET by default
# sciknoweval/biology → data_source=sciknoweval → val-core/sciknoweval/acc/mean@16
# math500            → data_source=math500     → val-core/math500/acc/mean@16
_DATASET_BASENAME="${DATASET##*/}"
_DEFAULT_METRIC_DS="${_DATASET_BASENAME}"
if [[ "$DATASET" == sciknoweval/* ]]; then
    _DEFAULT_METRIC_DS="sciknoweval"
fi
SAVE_BEST_METRIC="${SAVE_BEST_METRIC:-val-core/${_DEFAULT_METRIC_DS}/acc/mean@${VAL_N}}"

# 数据集路径
train_data_path="${OSS_ROOT}/datasets/${DATASET}/train.parquet"
val_data_path="${OSS_ROOT}/datasets/${DATASET}/test.parquet"
model_path="${OSS_ROOT}/base_models/${MODEL_NAME}"
save_path="${OSS_ROOT}/rl_models/${JOB_NAME:-tg_branching_sweep}"

# ── 环境 ──────────────────────────────────────────────────────────────
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARN
export WANDB_MODE=offline
export WANDB_ENTITY=oh-my-team
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal}"
export SWANLAB_LOG_DIR="${OSS_ROOT}/logs/swanlab_logs"
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0

pip install -e . --no-deps --no-build-isolation --quiet 2>/dev/null || true

mkdir -p "${SWANLAB_LOG_DIR}" 2>/dev/null || true

ENTROPY_HYDRA_ARGS=""
if [ "${ENTROPY_COEFF}" != "0" ]; then
    ENTROPY_HYDRA_ARGS="actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} actor_rollout_ref.actor.calculate_entropy=True"
fi

python -m verl.trainer.main_ppo \
    --config-name baseline_grpo \
    seed=${SEED} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.train_files="${train_data_path}" \
    data.val_files="${val_data_path}" \
    custom_reward_function.path="$(pwd)/verl/utils/reward_score/feedback/__init__.py" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.checkpoint.save_contents=${SAVE_CONTENTS_HYDRA} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.policy_loss.branch_token_loss_mode=${BRANCH_TOKEN_LOSS_MODE} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE:-1} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.enable_prefix_caching=True \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs=${VLLM_MAX_LOGPROBS} \
    actor_rollout_ref.rollout.agent.default_agent_loop=${DEFAULT_AGENT_LOOP} \
    actor_rollout_ref.rollout.branching.enabled=${BRANCHING_ENABLED} \
    actor_rollout_ref.rollout.branching.n_splits=${N_SPLITS} \
    actor_rollout_ref.rollout.branching.n_trees=${N_TREES} \
    actor_rollout_ref.rollout.branching.split_trigger=${SPLIT_TRIGGER} \
    actor_rollout_ref.rollout.branching.top_k=${TOP_K} \
    actor_rollout_ref.rollout.branching.teacher_top_k=${TEACHER_TOP_K} \
    actor_rollout_ref.rollout.branching.entropy_window=${ENTROPY_WINDOW} \
    actor_rollout_ref.rollout.branching.entropy_sigma_start=${ENTROPY_SIGMA_START} \
    actor_rollout_ref.rollout.branching.entropy_sigma_floor=${ENTROPY_SIGMA_FLOOR} \
    actor_rollout_ref.rollout.branching.entropy_sigma_step=${ENTROPY_SIGMA_STEP} \
    actor_rollout_ref.rollout.branching.teacher_context_mode=${TEACHER_CONTEXT_MODE} \
    actor_rollout_ref.rollout.branching.two_stage=${TWO_STAGE} \
    actor_rollout_ref.rollout.branching.stage1_n=${STAGE1_N} \
    actor_rollout_ref.rollout.branching.two_stage_teacher_mode=${TWO_STAGE_TEACHER_MODE} \
    actor_rollout_ref.rollout.branching.success_reward_threshold=${SUCCESS_THRESHOLD} \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.adv_std_floor=${ADV_STD_FLOOR} \
    +actor_rollout_ref.actor.policy_loss.dpo_coefficient=${DPO_COEFFICIENT} \
    +actor_rollout_ref.actor.policy_loss.dpo_use_ref=${DPO_USE_REF} \
    +actor_rollout_ref.actor.policy_loss.dpo_teacher_guided_beta=${DPO_TEACHER_GUIDED_BETA} \
    +actor_rollout_ref.actor.policy_loss.dpo_teacher_beta_alpha=${DPO_TEACHER_BETA_ALPHA} \
    +actor_rollout_ref.actor.policy_loss.dpo_teacher_beta_min=${DPO_TEACHER_BETA_MIN} \
    +actor_rollout_ref.actor.policy_loss.dpo_teacher_beta_max=${DPO_TEACHER_BETA_MAX} \
    +actor_rollout_ref.actor.policy_loss.dpo_stage1_pair=${DPO_STAGE1_PAIR} \
    +actor_rollout_ref.actor.policy_loss.dpo_stage1_pair_weight=${STAGE1_PAIR_WEIGHT} \
    +actor_rollout_ref.actor.policy_loss.dpo_reward_filter=${DPO_REWARD_FILTER} \
    trainer.total_epochs=30 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.max_actor_ckpt_to_keep=null \
    trainer.test_freq=${TEST_FREQ} \
    trainer.save_best_metric="${SAVE_BEST_METRIC}" \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME:-TG-Branching}" \
    trainer.experiment_name="${JOB_NAME:-tg_branching_sweep}" \
    trainer.group_name="${GROUP_NAME:-TG-Branching-${DATASET//\//-}}" \
    "trainer.logger=[console,swanlab]" \
    ${ENTROPY_HYDRA_ARGS}
