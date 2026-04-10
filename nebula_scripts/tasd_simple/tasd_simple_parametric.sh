#!/usr/bin/env bash
# =============================================================================
# TASD 清爽版参数化训练脚本（供 Nebula sweep 调用）
#
# 核心功能：
#   - reward_type: teacher_prob | teacher_log_prob
#   - entropy_gate: none | hard | soft
#   - clip_adv
# =============================================================================
set +xo pipefail

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"

# ── 必需参数 ─────────────────────────────────────────────────────────────
: "${DATASET:?DATASET is not set}"
: "${REWARD_TYPE:?REWARD_TYPE is not set}"      # teacher_prob | teacher_log_prob
: "${ENTROPY_GATE:?ENTROPY_GATE is not set}"     # none | hard | soft
: "${CLIP_ADV_VALUE:?CLIP_ADV_VALUE is not set}"
: "${MODEL_PATH:?MODEL_PATH is not set}"

# ── 可选参数（有默认值）─────────────────────────────────────────────────
LR="${LR:-1e-5}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.0}"
SEED="${SEED:-42}"
TEACHER_REG="${TEACHER_REG:-none}"
TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"
CLIP_ADV="${CLIP_ADV:-true}"
DISTILL_TOPK="${DISTILL_TOPK:-100}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.05}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-8}"
INCLUDE_SUCCESSFUL_ROLLOUTS="${INCLUDE_SUCCESSFUL_ROLLOUTS:-True}"

# ── 路径 ────────────────────────────────────────────────────────────────
train_data_path="${OSS_ROOT}/datasets/${DATASET}/train.parquet"
val_data_path="${OSS_ROOT}/datasets/${DATASET}/test.parquet"
model_path="${MODEL_PATH}"
save_path="${OSS_ROOT}/models/${JOB_NAME:-tasd_simple}"

# ── 环境 ────────────────────────────────────────────────────────────────
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

# 清理残留的 Ray session
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true
rm -rf /tmp/ray_session_* 2>/dev/null || true
rm -rf ~/.ray 2>/dev/null || true
sleep 3

echo "============================================"
echo "TASD 清爽版实验配置："
echo "  DATASET: ${DATASET}"
echo "  REWARD_TYPE: ${REWARD_TYPE}"
echo "  ENTROPY_GATE: ${ENTROPY_GATE}"
echo "  CLIP_ADV: ${CLIP_ADV}, VALUE: ${CLIP_ADV_VALUE}"
echo "  DISTILL_TOPK: ${DISTILL_TOPK}"
echo "  SEED: ${SEED}"
echo "============================================"

python -m verl.trainer.main_ppo \
    --config-name tasd_simple \
    seed=${SEED} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.train_files="${train_data_path}" \
    data.val_files="${val_data_path}" \
    custom_reward_function.path="$(pwd)/verl/utils/reward_score/feedback/__init__.py" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=${TEACHER_REG} \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=${TEACHER_UPDATE_RATE} \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.repetition_penalty=${REPETITION_PENALTY} \
    algorithm.tasd.reward_type=${REWARD_TYPE} \
    algorithm.tasd.entropy_gate=${ENTROPY_GATE} \
    algorithm.tasd.distill_topk=${DISTILL_TOPK} \
    algorithm.tasd.distill_temperature=${DISTILL_TEMPERATURE} \
    algorithm.tasd.clip_adv=${CLIP_ADV} \
    algorithm.tasd.clip_adv_value=${CLIP_ADV_VALUE} \
    algorithm.tasd.use_self_as_teacher_on_success=${INCLUDE_SUCCESSFUL_ROLLOUTS} \
    algorithm.tasd.include_successful_rollouts=${INCLUDE_SUCCESSFUL_ROLLOUTS} \
    algorithm.tasd.success_reward_threshold=1.0 \
    trainer.total_epochs=30 \
    trainer.total_training_steps=250 \
    trainer.save_freq=-1 \
    trainer.save_best_metric="val-core/sciknoweval/acc/mean@16" \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=False \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME:-TASD_simple}" \
    trainer.experiment_name="${JOB_NAME:-tasd_simple}" \
    trainer.group_name="TASD-simple" \
    "trainer.logger=[console,swanlab]"
