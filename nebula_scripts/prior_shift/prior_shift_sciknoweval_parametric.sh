#!/usr/bin/env bash
# =============================================================================
# Prior-Shift 参数化训练脚本（供 Nebula sweep / 本地 notebook 调用）
#
# Prior-Shift (Bayesian Credit Assignment, ours, Tier 1 首发):
#   g_t = KL( P_T(·|x, y_≤t) ‖ P_T(·|x, y_<t) )
#   A_t = A_seq · g_t / mean_t(g_t)
#
# 与 rlsd_sciknoweval_parametric.sh 同款结构，区别：
#   - --config-name prior_shift
#   - 注入 algorithm.prior_shift.{eps_norm, max_ratio, uniform_fallback}
#   - 触发 ray_trainer 在 lightweight teacher forward 内算 KL，logits 不外传
# =============================================================================
set +xo pipefail

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"

# ── 从环境变量读取超参 ────────────────────────────────────────────────
check_env() { val=$(eval echo "\$$1"); [ -n "$val" ] || { echo "ERROR: $1 is not set. Aborting."; exit 1; }; }
check_env DATASET
check_env LR
check_env TRAIN_BATCH_SIZE
check_env MINI_BATCH_SIZE
check_env ROLLOUT_N
check_env MODEL_NAME
check_env TEACHER_REGULARIZATION
check_env TEACHER_UPDATE_RATE

# Prior-Shift 专属超参（带默认值）
PS_EPS_NORM="${PS_EPS_NORM:-1.0e-6}"
PS_MAX_RATIO="${PS_MAX_RATIO:-3.0}"                   # v2: 10→3 防 length collapse
PS_UNIFORM_FALLBACK="${PS_UNIFORM_FALLBACK:-True}"
PS_RENORMALIZE_AFTER_CLIP="${PS_RENORMALIZE_AFTER_CLIP:-True}"   # v2 新增
PS_MIN_RESPONSE_LENGTH="${PS_MIN_RESPONSE_LENGTH:-50}"           # v2 新增
PS_LENGTH_PENALTY_TYPE="${PS_LENGTH_PENALTY_TYPE:-linear}"       # v2 新增: linear / zero

# 数据集路径：优先用 wrapper 已 export 的 TRAIN_DATA_PATH/VAL_DATA_PATH，
# 否则按 SUBJECT（若提供）回退到 OSS 路径
if [ -n "${TRAIN_DATA_PATH:-}" ]; then
    train_data_path="${TRAIN_DATA_PATH}"
elif [ -n "${SUBJECT:-}" ]; then
    train_data_path="${OSS_ROOT}/datasets/${DATASET}/${SUBJECT}/train.parquet"
else
    train_data_path="${OSS_ROOT}/datasets/${DATASET}/train.parquet"
fi
if [ -n "${VAL_DATA_PATH:-}" ]; then
    val_data_path="${VAL_DATA_PATH}"
elif [ -n "${SUBJECT:-}" ]; then
    val_data_path="${OSS_ROOT}/datasets/${DATASET}/${SUBJECT}/test.parquet"
else
    val_data_path="${OSS_ROOT}/datasets/${DATASET}/test.parquet"
fi
model_path="${MODEL_PATH:-${OSS_ROOT}/base_models/${MODEL_NAME}}"
save_path="${OSS_ROOT}/models/${JOB_NAME:-prior_shift_sweep}"

# ── 动态 save_best_metric（按 DATASET 区分 val 指标路径）──────────────
if [[ "${DATASET}" == tooluse* ]]; then
    SAVE_BEST_METRIC="val-core/tooluse/acc/mean@16"
else
    SAVE_BEST_METRIC="val-core/sciknoweval/acc/mean@16"
fi

# ── 环境 ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if git -C "${SCRIPT_DIR}" rev-parse --show-toplevel &>/dev/null; then
    GIT_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
    export PYTHONPATH="${GIT_ROOT}:${PYTHONPATH:-}"
    echo "[PYTHONPATH] Using git root: ${GIT_ROOT}"
elif [ -d "${SCRIPT_DIR}/../../verl" ]; then
    export PYTHONPATH="${SCRIPT_DIR}/../..:${PYTHONPATH:-}"
    echo "[PYTHONPATH] Using script relative path: ${SCRIPT_DIR}/../.."
else
    export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
    echo "[PYTHONPATH] Using pwd: $(pwd)"
fi
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

python -m verl.trainer.main_ppo \
    --config-name prior_shift \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.train_files="${train_data_path}" \
    data.val_files="${val_data_path}" \
    custom_reward_function.path="$(pwd)/verl/utils/reward_score/feedback/__init__.py" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=${TEACHER_REGULARIZATION} \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=${TEACHER_UPDATE_RATE} \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    algorithm.adv_estimator=prior_shift \
    algorithm.prior_shift.eps_norm=${PS_EPS_NORM} \
    algorithm.prior_shift.max_ratio=${PS_MAX_RATIO} \
    algorithm.prior_shift.uniform_fallback=${PS_UNIFORM_FALLBACK} \
    algorithm.prior_shift.renormalize_after_clip=${PS_RENORMALIZE_AFTER_CLIP} \
    algorithm.prior_shift.min_response_length=${PS_MIN_RESPONSE_LENGTH} \
    algorithm.prior_shift.length_penalty_type=${PS_LENGTH_PENALTY_TYPE} \
    algorithm.rollout_correction.rollout_is=token \
    trainer.total_epochs=30 \
    trainer.total_training_steps=250 \
    trainer.save_freq=-1 \
    trainer.save_best_metric="${SAVE_BEST_METRIC}" \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=False \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME:-Baselines_v3}" \
    trainer.experiment_name="${JOB_NAME:-prior_shift_sweep}" \
    trainer.group_name="PriorShift-generalization" \
    "trainer.logger=[console,swanlab]"
