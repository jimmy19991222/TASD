#!/usr/bin/env bash
# =============================================================================
# GRPO Baseline 参数化训练脚本（lcb_v6 数据集，供 Nebula sweep 调用）
# 所有超参通过 nebulactl --env 注入
# =============================================================================
set +xo pipefail

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"

# ── 从环境变量读取超参 ────────────────────────────────────────────────
check_env() { val=$(eval echo "\$$1"); [ -n "$val" ] || { echo "ERROR: $1 is not set. Aborting."; exit 1; }; }
check_env LR
check_env MINI_BATCH_SIZE
check_env TRAIN_BATCH_SIZE
check_env ROLLOUT_N
check_env MODEL_NAME

SEED="${SEED:-42}"

# Checkpoint / validation cadence (env-overridable, mirrors sciknoweval parametric script)
TEST_FREQ="${TEST_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-10}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
SAVE_HF_ONLY="${SAVE_HF_ONLY:-True}"
if [ "${SAVE_HF_ONLY}" = "True" ]; then
    SAVE_CONTENTS_HYDRA="[hf_model]"
else
    SAVE_CONTENTS_HYDRA="[model,optimizer,extra,hf_model]"
fi

train_data_path="${OSS_ROOT}/datasets/lcb_v6/train.parquet"
val_data_path="${OSS_ROOT}/datasets/lcb_v6/test.parquet"
model_path="${OSS_ROOT}/base_models/${MODEL_NAME}"
save_path="${OSS_ROOT}/rl_models/${JOB_NAME:-grpo_lcb_sweep}"

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

python -m verl.trainer.main_ppo \
    --config-name baseline_grpo \
    seed=${SEED} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.train_files="${train_data_path}" \
    data.val_files="${val_data_path}" \
    custom_reward_function.path="$(pwd)/verl/utils/reward_score/feedback/__init__.py" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.checkpoint.save_contents=${SAVE_CONTENTS_HYDRA} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    algorithm.rollout_correction.rollout_is=token \
    trainer.total_epochs=30 \
    trainer.total_training_steps=250 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.max_actor_ckpt_to_keep=null \
    trainer.test_freq=${TEST_FREQ} \
    trainer.save_best_metric="val-core/livecodebench/acc/mean@16" \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME:-Baselines_clean}" \
    trainer.experiment_name="${JOB_NAME:-grpo_lcb_sweep}" \
    trainer.group_name="GRPO-lcb" \
    "trainer.logger=[console,swanlab]"
