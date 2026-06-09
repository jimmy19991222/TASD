#!/usr/bin/env bash
# =============================================================================
# Teacher-Guided Branching SDPO 参数化训练脚本（sciknoweval 数据集）
# Mirrors sdpo_sciknoweval_parametric.sh + enables actor_rollout_ref.rollout.branching
# AND aligns actor.self_distillation.teacher_context_mode with the branching
# teacher so the rollout-time and training-time teacher distributions are
# byte-identical.
# =============================================================================
set +xo pipefail

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"

# ── 从环境变量读取超参 ────────────────────────────────────────────────
check_env() { val=$(eval echo "\$$1"); [ -n "$val" ] || { echo "ERROR: $1 is not set. Aborting."; exit 1; }; }
check_env DATASET
check_env LR
check_env ALPHA
check_env DONT_REPROMPT_ON_SELF_SUCCESS
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
# N-trees topology: 1 prompt -> N_TREES independent trees, each with 2**N_SPLITS leaves.
# ROLLOUT_N must equal N_TREES * 2**N_SPLITS.
N_TREES="${N_TREES:-1}"
# split_trigger in {entropy, entropy_disagreement}. The latter only splits at
# positions where teacher's argmax-in-student-topK disagrees with the student's
# actual sampled token (requires teacher_guided_rollout).
SPLIT_TRIGGER="${SPLIT_TRIGGER:-entropy}"
# Asymmetric K: student top_k=10 keeps branches in-distribution; teacher
# top_k=50 ensures teacher returns enough logprobs to cover all of student's
# 10 candidates (otherwise the intersection drops candidates silently).
TOP_K="${TOP_K:-10}"
TEACHER_TOP_K="${TEACHER_TOP_K:-100}"
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

# 数据集路径
train_data_path="${OSS_ROOT}/datasets/${DATASET}/train.parquet"
val_data_path="${OSS_ROOT}/datasets/${DATASET}/test.parquet"
model_path="${OSS_ROOT}/base_models/${MODEL_NAME}"
save_path="${OSS_ROOT}/rl_models/${JOB_NAME:-tg_branching_sdpo_sweep}"

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

# IMPORTANT: when branching is enabled we align actor.self_distillation.teacher_context_mode
# with rollout.branching.teacher_context_mode so the SDPO training-time teacher
# distribution matches the one the branching rollout used to pick branch tokens.
# Without this alignment the loss-time teacher would be the legacy peer-rollout
# 'ref' teacher while the rollout-time teacher would be e.g. 'gt_marker' — the
# distributions would drift and the on-policy assumption would break.
python -m verl.trainer.main_ppo \
    --config-name sdpo \
    seed=${SEED} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.train_files="${train_data_path}" \
    data.val_files="${val_data_path}" \
    custom_reward_function.path="$(pwd)/verl/utils/reward_score/feedback/__init__.py" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
    actor_rollout_ref.actor.self_distillation.alpha=${ALPHA} \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONT_REPROMPT_ON_SELF_SUCCESS} \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.actor.self_distillation.teacher_context_mode=${TEACHER_CONTEXT_MODE} \
    actor_rollout_ref.actor.checkpoint.save_contents=${SAVE_CONTENTS_HYDRA} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.policy_loss.branch_token_loss_mode=${BRANCH_TOKEN_LOSS_MODE} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
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
    algorithm.rollout_correction.rollout_is=token \
    algorithm.adv_std_floor=${ADV_STD_FLOOR} \
    trainer.total_epochs=30 \
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.max_actor_ckpt_to_keep=null \
    trainer.test_freq=${TEST_FREQ} \
    trainer.save_best_metric="val-core/sciknoweval/acc/mean@16" \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME:-TG-Branching}" \
    trainer.experiment_name="${JOB_NAME:-tg_branching_sdpo_sweep}" \
    trainer.group_name="TG-Branching-SDPO-${DATASET//\//-}" \
    "trainer.logger=[console,swanlab]"
