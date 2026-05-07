#!/bin/bash
# TASD 实验配置预检脚本
# 在提交前运行，检查常见配置问题

set -e

echo "=========================================="
echo "TASD 实验配置验证"
echo "=========================================="

# 1. 检查必要环境变量
: "${DATASET:?DATASET is not set}"
: "${REWARD_TYPE:?REWARD_TYPE is not set}"
: "${ENTROPY_GATE:?ENTROPY_GATE is not set}"

# 1.5 检查 adv_mode
ADV_MODE="${ADV_MODE:-grpo}"
VALID_AM="grpo self_teacher"
if ! echo "$VALID_AM" | grep -qw "$ADV_MODE"; then
    echo "❌ 错误: adv_mode='$ADV_MODE' 无效"
    echo "   有效值: $VALID_AM"
    exit 1
fi

# 2. 检查参数冲突
if [ "$ENTROPY_GATE" = "none" ] && [ "${ENTROPY_GATE_RATIO:-1.0}" != "1.0" ]; then
    echo "❌ 错误: entropy_gate=none 时 entropy_gate_ratio 必须为 1.0"
    exit 1
fi

if [ "$ENTROPY_GATE" = "none" ] && [ "$ADV_ENTROPY_WEIGHT" = "none" ]; then
    echo "⚠️ 警告: entropy_gate=none 且 adv_entropy_weight=none，无熵控制"
fi

# 2.5 self_teacher 模式特殊检查
if [ "$ADV_MODE" = "self_teacher" ]; then
    echo "ℹ️  Self-Teacher Advantage 模式"
    echo "   beta=${BETA:-0.7}, ema_alpha=${EMA_ALPHA:-0.9}, clip_value=${CLIP_VALUE:-5.0}"
    
    # self_teacher 模式下，以下配置会被忽略
    if [ "${NORM_ADV_BY_STD:-false}" = "true" ]; then
        echo "⚠️  警告: self_teacher 模式不需要 norm_adv_by_std，已自动忽略"
    fi
    if [ "${ADV_ENTROPY_WEIGHT:-none}" != "none" ]; then
        echo "⚠️  警告: self_teacher 模式不需要 adv_entropy_weight，已自动忽略"
    fi
    if [ "${ENTROPY_GATE:-none}" != "none" ]; then
        echo "⚠️  警告: self_teacher 模式不需要 entropy_gate，已自动忽略"
    fi
    
    # self_teacher 模式需要 distill_topk
    if [ "${DISTILL_TOPK:-100}" -lt 50 ]; then
        echo "⚠️  警告: self_teacher 模式建议 distill_topk >= 50（当前=${DISTILL_TOPK:-100}）"
    fi
fi

# 3. 检查 adv_entropy_weight 值是否有效
VALID_AEW="none teacher_conf certainty_diff"
if ! echo "$VALID_AEW" | grep -qw "$ADV_ENTROPY_WEIGHT"; then
    echo "❌ 错误: adv_entropy_weight='$ADV_ENTROPY_WEIGHT' 无效"
    echo "   有效值: $VALID_AEW"
    exit 1
fi

# 4. 检查 clip_adv_value
if [ "${CLIP_ADV:-true}" = "true" ] && [ "${CLIP_ADV_VALUE:-2.0}" = "0" ]; then
    echo "⚠️ 警告: clip_adv=true 但 clip_adv_value=0，等效于不 clip"
fi

# 5. 检查 success_reward_threshold
if [ "${INCLUDE_SUCCESSFUL_ROLLOUTS:-True}" = "True" ]; then
    THRESHOLD="${SUCCESS_REWARD_THRESHOLD:-1.0}"
    echo "ℹ️  include_successful_rollouts=True"
    echo "   success_reward_threshold=$THRESHOLD"
    echo "   ⚠️  确保数据集中有样本的 reward >= $THRESHOLD"
fi

# 6. 检查数据集存在
TRAIN_DATA="/data/oss_bucket_0/ad/loujieming.ljm/datasets/${DATASET}/train.parquet"
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ 错误: 训练数据不存在: $TRAIN_DATA"
    exit 1
fi

echo "✅ 配置检查通过"
echo "=========================================="
echo "实验配置摘要:"
echo "  DATASET: $DATASET"
echo "  REWARD_TYPE: $REWARD_TYPE"
echo "  ENTROPY_GATE: $ENTROPY_GATE (ratio=${ENTROPY_GATE_RATIO:-1.0})"
echo "  ADV_ENTROPY_WEIGHT: $ADV_ENTROPY_WEIGHT"
echo "  CLIP_ADV: ${CLIP_ADV:-true} (value=${CLIP_ADV_VALUE:-2.0})"
echo "  GROUP_MEAN_MODE: ${GROUP_MEAN_MODE:-token}"
echo "  ROLLOUT_N: ${ROLLOUT_N:-8}"
echo "  ADV_MODE: ${ADV_MODE:-grpo}"
if [ "${ADV_MODE:-grpo}" = "self_teacher" ]; then
    echo "  BETA: ${BETA:-0.7}, EMA_ALPHA: ${EMA_ALPHA:-0.9}, CLIP_VALUE: ${CLIP_VALUE:-5.0}"
fi
echo "=========================================="
