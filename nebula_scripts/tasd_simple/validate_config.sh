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

# 2. 检查参数冲突
if [ "$ENTROPY_GATE" = "none" ] && [ "${ENTROPY_GATE_RATIO:-1.0}" != "1.0" ]; then
    echo "❌ 错误: entropy_gate=none 时 entropy_gate_ratio 必须为 1.0"
    exit 1
fi

if [ "$ENTROPY_GATE" = "none" ] && [ "$ADV_ENTROPY_WEIGHT" = "none" ]; then
    echo "⚠️ 警告: entropy_gate=none 且 adv_entropy_weight=none，无熵控制"
fi

# 3. 检查 adv_entropy_weight 值是否有效
VALID_AEW="none teacher_prob teacher_conf certainty_diff"
if ! echo "$VALID_AEW" | grep -qw "$ADV_ENTROPY_WEIGHT"; then
    echo "❌ 错误: adv_entropy_weight='$ADV_ENTROPY_WEIGHT' 无效"
    echo "   有效值: $VALID_AEW"
    exit 1
fi

# 3b. 检查 adv_baseline_mode 值是否有效
VALID_ABM="none causal_ema teacher_ce"
if ! echo "$VALID_ABM" | grep -qw "$ADV_BASELINE_MODE"; then
    echo "❌ 错误: adv_baseline_mode='$ADV_BASELINE_MODE' 无效"
    echo "   有效值: $VALID_ABM"
    exit 1
fi

# 3c. 检查 causal_ema_alpha 范围（仅 causal_ema 模式）
if [ "$ADV_BASELINE_MODE" = "causal_ema" ]; then
    ALPHA="${CAUSAL_EMA_ALPHA:-0.021}"
    # 简单检查：必须在 (0, 1) 范围内
    if python3 -c "import sys; sys.exit(0 if 0 < float('$ALPHA') < 1 else 1)" 2>/dev/null; then
        echo "ℹ️  causal_ema_alpha=$ALPHA (half-life ≈ $(python3 -c "import math; print(f'{math.log(0.5)/math.log(float(\"$ALPHA\")):.1f}') tokens")")"
    else
        echo "❌ 错误: causal_ema_alpha=$ALPHA 必须在 (0, 1) 范围内"
        exit 1
    fi
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
echo "=========================================="
