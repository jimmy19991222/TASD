#!/usr/bin/env bash
# =============================================================================
# TASD sciknoweval 参数化训练脚本（供 Nebula sweep 调用）
#
# 所有超参通过 nebulactl --env 注入，由 submit_tasd_ema_sweep.sh 设置
# =============================================================================
set +xeuo pipefail

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"

# ── 从环境变量读取超参（均有默认值）─────────────────────────────────────
REWARD_TYPE="${REWARD_TYPE:-teacher_prob}"
LR="${LR:-1e-5}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.0}"
TEACHER_REG="${TEACHER_REG:-none}"
TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"
NORM_ADV_BY_STD="${NORM_ADV_BY_STD:-False}"
CLIP_ADV="${CLIP_ADV:-True}"
CLIP_ADV_VALUE="${CLIP_ADV_VALUE:-5.0}"
ROLLOUT_IS="${ROLLOUT_IS:-token}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-8}"

# ── 路径 ────────────────────────────────────────────────────────────────
train_data_path="${OSS_ROOT}/datasets/sciknoweval/biology/train.parquet"
val_data_path="${OSS_ROOT}/datasets/sciknoweval/biology/test.parquet"
model_path="${OSS_ROOT}/base_models/Qwen3-8B"
save_path="${OSS_ROOT}/models/${JOB_NAME:-tasd_sweep}"

# ── 环境 ────────────────────────────────────────────────────────────────
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARN
export WANDB_MODE=offline
export WANDB_ENTITY=oh-my-team
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY="${SWANLAB_API_KEY:?SWANLAB_API_KEY not set}"
export SWANLAB_LOG_DIR="${OSS_ROOT}/logs/swanlab_logs"
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0

pip install -e . --no-deps --no-build-isolation --quiet 2>/dev/null || true

python -m verl.trainer.main_ppo \
    --config-name tasd \
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
    algorithm.tasd.reward_type=${REWARD_TYPE} \
    algorithm.tasd.reward_transform=none \
    algorithm.tasd.reward_scale=1.0 \
    algorithm.tasd.distill_topk=100 \
    algorithm.tasd.use_self_as_teacher_on_success=True \
    algorithm.tasd.include_successful_rollouts=True \
    algorithm.tasd.success_reward_threshold=1.0 \
    algorithm.tasd.norm_adv_by_std=${NORM_ADV_BY_STD} \
    algorithm.tasd.clip_adv=${CLIP_ADV} \
    algorithm.tasd.clip_adv_value=${CLIP_ADV_VALUE} \
    algorithm.rollout_correction.rollout_is=${ROLLOUT_IS} \
    trainer.total_epochs=30 \
    trainer.total_training_steps=250 \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=False \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="SDPO-TASD" \
    trainer.experiment_name="${JOB_NAME:-tasd_sweep}" \
    trainer.group_name="TASD-ema-teacher" \
    "trainer.logger=[console,swanlab]"
