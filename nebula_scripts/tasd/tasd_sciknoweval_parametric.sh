#!/usr/bin/env bash
# =============================================================================
# TASD sciknoweval 参数化训练脚本（供 Nebula sweep 调用）
#
# 所有超参通过 nebulactl --env 注入，由 submit_tasd_ema_sweep.sh 设置
# =============================================================================
set +xo pipefail

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"

# ── 从环境变量读取超参（必传，未设置时立即报错退出）─────────────────
: "${REWARD_TYPE:?REWARD_TYPE is not set}"
: "${LR:?LR is not set}"
: "${ENTROPY_COEFF:?ENTROPY_COEFF is not set}"
: "${TEACHER_REG:?TEACHER_REG is not set}"
: "${TEACHER_UPDATE_RATE:?TEACHER_UPDATE_RATE is not set}"
: "${NORM_ADV_BY_STD:?NORM_ADV_BY_STD is not set}"
: "${CLIP_ADV:?CLIP_ADV is not set}"
: "${CLIP_ADV_VALUE:?CLIP_ADV_VALUE is not set}"
: "${ROLLOUT_IS:?ROLLOUT_IS is not set}"
: "${TRAIN_BATCH_SIZE:?TRAIN_BATCH_SIZE is not set}"
: "${MINI_BATCH_SIZE:?MINI_BATCH_SIZE is not set}"
: "${ROLLOUT_N:?ROLLOUT_N is not set}"
: "${INCLUDE_SUCCESSFUL_ROLLOUTS:?INCLUDE_SUCCESSFUL_ROLLOUTS is not set}"
DISTILL_TOPK="${DISTILL_TOPK:-100}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
USE_TEACHER_GAE="${USE_TEACHER_GAE:-False}"
TEACHER_GAE_GAMMA="${TEACHER_GAE_GAMMA:-1.0}"
TEACHER_GAE_LAMBDA="${TEACHER_GAE_LAMBDA:-0.95}"

# ── 路径 ────────────────────────────────────────────────────────────────
# 数据集列表（目前只跑第一个，扩展时直接往数组里加）
DATA_PATHS=(
    "${OSS_ROOT}/datasets/sciknoweval/biology"
)
train_data_path="${DATA_PATHS[0]}/train.parquet"
val_data_path="${DATA_PATHS[0]}/test.parquet"
model_path="${OSS_ROOT}/base_models/Qwen3-8B"
save_path="${OSS_ROOT}/models/${JOB_NAME:-tasd_sweep}"

# ── 环境 ────────────────────────────────────────────────────────────────
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
unset VLLM_ATTENTION_BACKEND  # 与 verl_training.sh 行为一致，避免平台注入的值影响 attention 计算
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARN
export WANDB_MODE=offline
export WANDB_ENTITY=oh-my-team
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal}"
export SWANLAB_LOG_DIR="${OSS_ROOT}/logs/swanlab_logs"
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0

pip install -e . --no-deps --no-build-isolation --quiet 2>/dev/null || true

# 创建 SwanLab 日志目录（SwanLab 初始化时要求目录存在）
mkdir -p "${SWANLAB_LOG_DIR}" 2>/dev/null || true

# 清理残留的 Ray session（避免 session name 冲突）
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true

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
    algorithm.tasd.distill_topk=${DISTILL_TOPK} \
    algorithm.tasd.distill_temperature=${DISTILL_TEMPERATURE} \
    algorithm.tasd.use_self_as_teacher_on_success=${INCLUDE_SUCCESSFUL_ROLLOUTS} \
    algorithm.tasd.include_successful_rollouts=${INCLUDE_SUCCESSFUL_ROLLOUTS} \
    algorithm.tasd.success_reward_threshold=1.0 \
    algorithm.tasd.norm_adv_by_std=${NORM_ADV_BY_STD} \
    algorithm.tasd.clip_adv=${CLIP_ADV} \
    algorithm.tasd.clip_adv_value=${CLIP_ADV_VALUE} \
    algorithm.tasd.use_teacher_gae=${USE_TEACHER_GAE} \
    algorithm.tasd.teacher_gae_gamma=${TEACHER_GAE_GAMMA} \
    algorithm.tasd.teacher_gae_lambda=${TEACHER_GAE_LAMBDA} \
    algorithm.rollout_correction.rollout_is=${ROLLOUT_IS} \
    trainer.total_epochs=30 \
    trainer.total_training_steps=250 \
    trainer.save_freq=-1 \
    trainer.save_best_metric="val-core/sciknoweval/acc/mean@16" \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=False \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME:-TASD}" \
    trainer.experiment_name="${JOB_NAME:-tasd_sweep}" \
    trainer.group_name="TASD-ema-teacher" \
    "trainer.logger=[console,swanlab]"
