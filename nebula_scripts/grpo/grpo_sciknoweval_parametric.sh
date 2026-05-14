#!/usr/bin/env bash
# =============================================================================
# GRPO Baseline 参数化训练脚本（供 Nebula sweep 调用）
# 所有超参通过 nebulactl --env 注入
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
save_path="${OSS_ROOT}/models/${JOB_NAME:-grpo_sweep}"

# ── 动态 save_best_metric（按 DATASET 区分 val 指标路径）──────────────
if [[ "${DATASET}" == tooluse* ]]; then
    SAVE_BEST_METRIC="val-core/tooluse/acc/mean@16"
else
    # sciknoweval/biology → sciknoweval/acc/mean@16（通用）
    SAVE_BEST_METRIC="val-core/sciknoweval/acc/mean@16"
fi

# ── 环境 ──────────────────────────────────────────────────────────────
# 优先使用 git 仓库根目录作为 PYTHONPATH，确保加载最新代码（而非 train_package 中的旧代码）
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
    --config-name baseline_grpo \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.train_files="${train_data_path}" \
    data.val_files="${val_data_path}" \
    custom_reward_function.path="$(pwd)/verl/utils/reward_score/feedback/__init__.py" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    algorithm.rollout_correction.rollout_is=token \
    trainer.total_epochs=30 \
    trainer.total_training_steps=250 \
    trainer.save_freq=-1 \
    trainer.save_best_metric="${SAVE_BEST_METRIC}" \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=False \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME:-Baselines_v3}" \
    trainer.experiment_name="${JOB_NAME:-grpo_sweep}" \
    trainer.group_name="GRPO-generalization" \
    "trainer.logger=[console,swanlab]"
