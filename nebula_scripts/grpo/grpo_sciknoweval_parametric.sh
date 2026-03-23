#!/usr/bin/env bash
# =============================================================================
# GRPO Baseline 参数化训练脚本（供 Nebula sweep 调用）
# 所有超参通过 nebulactl --env 注入
# =============================================================================
set +xeuo pipefail

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"

# ── 从环境变量读取超参 ────────────────────────────────────────────────
DATASET="${DATASET:-sciknoweval/biology}"    # 相对名称，如 sciknoweval/biology
LR="${LR:-1e-5}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"         # OSS base_models 下的目录名

train_data_path="${OSS_ROOT}/datasets/${DATASET}/train.parquet"
model_path="${OSS_ROOT}/base_models/${MODEL_NAME}"
save_path="${OSS_ROOT}/models/${JOB_NAME:-grpo_sweep}"

# ── 环境 ──────────────────────────────────────────────────────────────
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARN
export WANDB_MODE=offline
export WANDB_ENTITY=oh-my-team
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY=M5oC00EEt8G1wC0XaHkal
export SWANLAB_LOG_DIR="${OSS_ROOT}/logs/swanlab_logs"
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0

pip install -e . --no-deps --no-build-isolation --quiet 2>/dev/null || true

python -m verl.trainer.main_ppo \
    --config-name baseline_grpo \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.train_files="${train_data_path}" \
    data.val_files=null \
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
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=False \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="SDPO-GRPO-Baseline" \
    trainer.experiment_name="${JOB_NAME:-grpo_sweep}" \
    trainer.group_name="GRPO-generalization" \
    "trainer.logger=[console,swanlab]"
