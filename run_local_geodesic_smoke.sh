#!/bin/bash
# =============================================================================
# Geodesic-VCE 本地 Smoke Test（A100 Notebook 环境）
#
# 目的：快速验证代码是否能正常运行（10 步即可看到 loss 下降）
#
# 配置：
#   - 单卡 A100（notebook 环境）
#   - 10 training steps
#   - batch_size=8, rollout_n=2
#   - SDPO baseline（最简单配置）
#
# 使用方式：
#   ./run_local_geodesic_smoke.sh
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="sdpo"

# 数据集路径（notebook 环境 OSS 挂载）
OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
DATA_PATH="${OSS_ROOT}/datasets/sciknoweval/biology"

# 直接指定 parquet 文件路径（避免 user.yaml 中的路径拼接问题）
TRAIN_FILE="${DATA_PATH}/train.parquet"
VAL_FILE="${DATA_PATH}/test.parquet"

# Smoke Test 超参（极简配置，快速验证）
TRAIN_BATCH_SIZE=8          # 减小 batch size 适配单卡
ROLLOUT_BATCH_SIZE=2        # 减少 rollout 数量
LR=1e-5
MAX_STEPS=10                # 只跑 10 步
SEED=42

# SDPO 参数
ALPHA=0.5
DISTILL_TOPK=100
DONT_REPROMPT_ON_SELF_SUCCESS=True

# Geodesic 参数（关闭，只测试基础 SDPO）
USE_GEODESIC=False
GEODESIC_TRUST_REGION=5.0
GEODESIC_BETA_SCALE=0.5

# 模型路径
MODEL_PATH="${OSS_ROOT}/base_models/Qwen3-8B"

# 实验名称后缀
SUFFIX=${1:-"geodesic_smoke"}

# =============================================================================
# SETUP
# =============================================================================

# 获取脚本所在目录
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# 定义 USER（Hydra config 需要）
export USER=${USER:-$(whoami)}

# 单卡配置
export N_GPUS_PER_NODE=1

# 环境配置
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARN
export WANDB_MODE=offline
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0

# =============================================================================
# EXECUTION
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="LOCAL-GEODESIC-SMOKE-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-steps${MAX_STEPS}-seed${SEED}-${MODEL_NAME}-${SUFFIX}"

# 构建 Hydra 参数
ARGS="data.train_files=['${TRAIN_FILE}'] \
data.val_files=['${VAL_FILE}'] \
data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=Geodesic-SmokeTest \
trainer.total_training_steps=$MAX_STEPS \
trainer.save_freq=-1 \
trainer.val_before_train=False \
seed=$SEED \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
actor_rollout_ref.rollout.val_kwargs.n=4 \
actor_rollout_ref.actor.self_distillation.distillation_topk=$DISTILL_TOPK \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONT_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
actor_rollout_ref.actor.self_distillation.use_geodesic=${USE_GEODESIC} \
actor_rollout_ref.actor.self_distillation.geodesic_trust_region=${GEODESIC_TRUST_REGION} \
actor_rollout_ref.actor.self_distillation.geodesic_beta_scale=${GEODESIC_BETA_SCALE}"

echo "========================================================================="
echo "🧪 启动 Geodesic-VCE 本地 Smoke Test"
echo "========================================================================="
echo "实验: $EXP_NAME"
echo "数据集: $DATA_PATH"
echo "模型: $MODEL_PATH"
echo "配置: SDPO baseline, $MAX_STEPS steps, batch_size=$TRAIN_BATCH_SIZE, rollout_n=$ROLLOUT_BATCH_SIZE"
echo "GPU: 1x A100 (notebook 环境)"
echo "========================================================================="

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
