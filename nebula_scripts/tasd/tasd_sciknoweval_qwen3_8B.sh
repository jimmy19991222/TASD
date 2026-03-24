#!/usr/bin/env bash
set +xeuo pipefail

# =============================================================================
# TASD - sciknoweval biology - Qwen3-8B - Nebula 训练脚本
#
# 使用方式：
#   由 nebula_scripts/submit_job.sh 通过 launch_ray_cluster.sh 自动调用
#   也可以本地直接运行：bash nebula_scripts/tasd/tasd_sciknoweval_qwen3_8B.sh
# =============================================================================

CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
SCRIPT_NAME=$(basename "$0")
JOB_NAME="${JOB_NAME:-${SCRIPT_NAME%.*}_${CURRENT_TIME}}"

# ── 路径配置（Nebula OSS 挂载路径）────────────────────────────────────────
OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"

# 数据集：sciknoweval biology（已提前上传到 OSS）
# 本地对应：datasets/sciknoweval/biology/
train_data_path="${OSS_ROOT}/datasets/sciknoweval/biology/train.parquet"
val_data_path="${OSS_ROOT}/datasets/sciknoweval/biology/test.parquet"

# 基底模型
model_path="${OSS_ROOT}/base_models/Qwen3-8B"

# checkpoint 保存路径
save_path="${OSS_ROOT}/models/${JOB_NAME}"

# ── 训练超参 ──────────────────────────────────────────────────────────────
# 以下为单实验配置，如需扫描多个超参，请参考 submit_job.sh 的循环逻辑

REWARD_TYPE="teacher_prob"        # teacher_prob | log_teacher_prob
LR="1e-5"
ENTROPY_COEFF="0.03"              # >0 对抗熵崩溃
TEACHER_REG="ema"                 # none | ema
TEACHER_UPDATE_RATE="0.05"        # 0.0=固定初始模型，0.05=EMA跟随

# ── 规模参数 ──────────────────────────────────────────────────────────────
n_gpus_per_node=8                 # H20 节点 8 卡
train_batch_size=32
mini_batch_size=32
rollout_n=8
total_training_steps=250
lr_warmup_steps=10

# ── 序列长度 ──────────────────────────────────────────────────────────────
max_model_len=18944
gpu_memory_utilization=0.85

# =============================================================================
# 环境初始化
# =============================================================================
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARN
export WANDB_MODE=offline
export WANDB_ENTITY=oh-my-team
export WANDB_DIR="${OSS_ROOT}/logs/wandb_logs"
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY="${SWANLAB_API_KEY:?SWANLAB_API_KEY not set}"
export SWANLAB_LOG_DIR="${OSS_ROOT}/logs/swanlab_logs"
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0

# 安装 SDPO 包本身（Nebula 不自动执行 pip install -e .)
pip install -e . --no-deps --no-build-isolation --quiet 2>/dev/null || true

python -m verl.trainer.main_ppo \
    --config-name tasd \
    data.train_batch_size=${train_batch_size} \
    data.train_files="${train_data_path}" \
    data.val_files="${val_data_path}" \
    custom_reward_function.path="$(pwd)/verl/utils/reward_score/feedback/__init__.py" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${lr_warmup_steps} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=${TEACHER_REG} \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=${TEACHER_UPDATE_RATE} \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    algorithm.tasd.reward_type=${REWARD_TYPE} \
    algorithm.tasd.reward_transform=none \
    algorithm.tasd.reward_scale=1.0 \
    algorithm.tasd.distill_topk=100 \
    algorithm.tasd.use_self_as_teacher_on_success=True \
    algorithm.tasd.include_successful_rollouts=True \
    algorithm.tasd.success_reward_threshold=1.0 \
    algorithm.tasd.norm_adv_by_std=False \
    algorithm.tasd.clip_adv=True \
    algorithm.tasd.clip_adv_value=5.0 \
    algorithm.rollout_correction.rollout_is=token \
    trainer.total_epochs=30 \
    trainer.total_training_steps=${total_training_steps} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.val_before_train=False \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="SDPO-TASD" \
    trainer.experiment_name="${JOB_NAME}" \
    trainer.group_name="TASD-nebula" \
    "trainer.logger=[console,swanlab]"
