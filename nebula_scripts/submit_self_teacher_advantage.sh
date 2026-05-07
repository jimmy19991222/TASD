#!/bin/bash
# =============================================================================
# Self-Teacher Advantage with Bidirectional Baselines - Nebula 批量提交脚本
#
# 核心思想：
#   - 完全抛弃 GRPO seq-level advantage
#   - 使用纯 teacher advantage: A_t = Q_t - V_t
#   - V_t = β·V_CE + (1-β)·V_EMA（双向 baseline 融合）
#   - 无需 z-score 归一化，无需 entropy gate，无需 token filter
#
# 实验设计：
#   - beta 消融：1.0 / 0.7 / 0.5 / 0.0
#   - 对比 GRPO baseline
#
# 使用方式：
#   bash nebula_scripts/submit_self_teacher_advantage.sh [--dry-run]
# =============================================================================

# ── Nebula 账号配置 ──────────────────────────────────────────────────────
QUEUE="lazada_llm_ad_h20"
WORLD_SIZE=1
OPENLM_TOKEN="${OPENLM_TOKEN:?OPENLM_TOKEN not set}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:?OSS_ACCESS_ID not set}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?OSS_ACCESS_KEY not set}"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
CLUSTER_FILE="nebula_scripts/cluster.json"
SCRIPT_PATH="nebula_scripts/tasd_simple/tasd_simple_parametric.sh"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="Self-Teacher-Advantage"

# ── 数据集配置 ──────────────────────────────────────────────────────
DATASETS=(
    "sciknoweval/biology"
    # "sciknoweval/chemistry"
)

# ── dry-run 模式 ─────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 超参配置
# =============================================================================

# ── Advantage 模式 ─────────────────────────────────────────────────────
ADV_MODE="self_teacher"

# ── Beta 消融（V_CE vs V_EMA 融合系数）─────────────────────────────────
# 1.0 = 纯 V_CE（最安全，variance reduction 最优）
# 0.7 = 推荐默认值（V_CE 主导，V_EMA 辅助捕捉转折）
# 0.5 = 等权融合（信号最强）
# 0.0 = 纯 V_EMA（不稳定，验证 baseline 必要性）
BETA_LIST=(
    "1.0"
    "0.7"
    "0.5"
    "0.0"
)

# ── EMA Alpha ─────────────────────────────────────────────────────────
EMA_ALPHA="0.9"  # half-life ≈ 6.6 tokens

# ── Clip Value ────────────────────────────────────────────────────────
CLIP_VALUE="5.0"

# ── Reward Type ─────────────────────────────────────────────────────
# outcome: 仅用于构建 teacher context，不参与 advantage 计算
REWARD_TYPE="outcome"

# ── Entropy Gate ─────────────────────────────────────────────────────
# self_teacher 模式不需要 entropy gate
ENTROPY_GATE="none"
ENTROPY_GATE_RATIO="1.0"

# ── Distill Topk ──────────────────────────────────────────────────────
# 保证 top-K 覆盖充分（V_CE 计算需要）
DISTILL_TOPK="256"
DISTILL_TEMPERATURE="1.0"

# ── 其他固定参数 ─────────────────────────────────────────────────────
CLIP_ADV="true"
CLIP_ADV_VALUE="2.0"  # GRPO 模式使用，self_teacher 模式使用 clip_value
NORM_ADV_BY_STD="false"
ADV_STD_FLOOR="0.0"
ADV_ENTROPY_WEIGHT="none"
GROUP_MEAN_MODE="seq"
CLIP_RATIO_HIGH="0.28"
REPETITION_PENALTY="1.05"
ROLLOUT_N="8"
ROLLOUT_TEMPERATURE="1.0"
TRAIN_BATCH_SIZE="32"
GEN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
INCLUDE_SUCCESSFUL_ROLLOUTS="True"
TEACHER_REG="ema"
TEACHER_UPDATE_RATE="0.05"
LR="1e-5"
ENTROPY_COEFF="0.001"
SEED="42"
FILTER_GROUPS_ENABLE="false"
FILTER_GROUPS_METRIC="acc"
FILTER_GROUPS_MAX_GEN="0"

# ── 模型路径 ─────────────────────────────────────────────────────────
MODEL_PATH="${OSS_ROOT:-/data/oss_bucket_0/ad/loujieming.ljm}/models/Qwen3-8B"

# =============================================================================
# 实验矩阵
# =============================================================================

EXPERIMENTS=()

for DATASET in "${DATASETS[@]}"; do
    DATASET_NAME=$(echo "$DATASET" | sed 's|/|_|g')
    
    for BETA in "${BETA_LIST[@]}"; do
        # 生成实验标签
        BETA_LABEL=$(echo "$BETA" | sed 's/\.//g')  # 1.0 -> 10, 0.7 -> 07
        EXP_LABEL="beta${BETA_LABEL}"
        
        EXPERIMENTS+=(
            "${DATASET}|${ADV_MODE}|${BETA}|${EMA_ALPHA}|${CLIP_VALUE}|${REWARD_TYPE}|${ENTROPY_GATE}|${DISTILL_TOPK}|${EXP_LABEL}"
        )
    done
    
    # GRPO baseline 对照（adv_mode=grpo，其他参数相同）
    EXPERIMENTS+=(
        "${DATASET}|grpo|N/A|N/A|N/A|${REWARD_TYPE}|${ENTROPY_GATE}|${DISTILL_TOPK}|grpo_baseline"
    )
done

# =============================================================================
# 提交实验
# =============================================================================

echo "============================================"
echo "Self-Teacher Advantage 实验提交"
echo "============================================"
echo "实验数量: ${#EXPERIMENTS[@]}"
echo "============================================"

TASK_IDS=()
FAILED=0

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r DATASET ADV_MODE BETA EMA_ALPHA CLIP_VAL REWARD_TYPE ENTROPY_GATE DISTILL_TOPK EXP_LABEL <<< "$EXP"
    
    JOB_NAME="self_teacher_${EXP_LABEL}"
    DATASET_NAME=$(echo "$DATASET" | sed 's|/|_|g')
    
    echo ""
    echo "────────────────────────────────────────"
    echo "提交实验: ${JOB_NAME}"
    echo "  DATASET: ${DATASET}"
    echo "  ADV_MODE: ${ADV_MODE}"
    if [ "$ADV_MODE" = "self_teacher" ]; then
        echo "  BETA: ${BETA}, EMA_ALPHA: ${EMA_ALPHA}, CLIP_VALUE: ${CLIP_VAL}"
    fi
    echo "  REWARD_TYPE: ${REWARD_TYPE}"
    echo "  DISTILL_TOPK: ${DISTILL_TOPK}"
    
    # 构建环境变量
    export DATASET="$DATASET"
    export ADV_MODE="$ADV_MODE"
    export BETA="$BETA"
    export EMA_ALPHA="$EMA_ALPHA"
    export CLIP_VALUE="$CLIP_VAL"
    export REWARD_TYPE="$REWARD_TYPE"
    export ENTROPY_GATE="$ENTROPY_GATE"
    export ENTROPY_GATE_RATIO="$ENTROPY_GATE_RATIO"
    export DISTILL_TOPK="$DISTILL_TOPK"
    export DISTILL_TEMPERATURE="$DISTILL_TEMPERATURE"
    export CLIP_ADV="$CLIP_ADV"
    export CLIP_ADV_VALUE="$CLIP_ADV_VALUE"
    export NORM_ADV_BY_STD="$NORM_ADV_BY_STD"
    export ADV_STD_FLOOR="$ADV_STD_FLOOR"
    export ADV_ENTROPY_WEIGHT="$ADV_ENTROPY_WEIGHT"
    export GROUP_MEAN_MODE="$GROUP_MEAN_MODE"
    export CLIP_RATIO_HIGH="$CLIP_RATIO_HIGH"
    export REPETITION_PENALTY="$REPETITION_PENALTY"
    export ROLLOUT_N="$ROLLOUT_N"
    export ROLLOUT_TEMPERATURE="$ROLLOUT_TEMPERATURE"
    export TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE"
    export GEN_BATCH_SIZE="$GEN_BATCH_SIZE"
    export MINI_BATCH_SIZE="$MINI_BATCH_SIZE"
    export INCLUDE_SUCCESSFUL_ROLLOUTS="$INCLUDE_SUCCESSFUL_ROLLOUTS"
    export TEACHER_REG="$TEACHER_REG"
    export TEACHER_UPDATE_RATE="$TEACHER_UPDATE_RATE"
    export LR="$LR"
    export ENTROPY_COEFF="$ENTROPY_COEFF"
    export SEED="$SEED"
    export FILTER_GROUPS_ENABLE="$FILTER_GROUPS_ENABLE"
    export FILTER_GROUPS_METRIC="$FILTER_GROUPS_METRIC"
    export FILTER_GROUPS_MAX_GEN="$FILTER_GROUPS_MAX_GEN"
    export MODEL_PATH="$MODEL_PATH"
    export JOB_NAME="$JOB_NAME"
    export PROJECT_NAME="$PROJECT_NAME"
    export GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
    export GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    
    # 构建 nebula 命令
    CMD="nebula submit \
        --queue $QUEUE \
        --world_size $WORLD_SIZE \
        --cluster_file $CLUSTER_FILE \
        --script_path $SCRIPT_PATH \
        --docker_image $CUSTOM_DOCKER_IMAGE \
        --env OPENLM_TOKEN=$OPENLM_TOKEN \
        --env OSS_ACCESS_ID=$OSS_ACCESS_ID \
        --env OSS_ACCESS_KEY=$OSS_ACCESS_KEY \
        --env OSS_ENDPOINT=$OSS_ENDPOINT \
        --env OSS_BUCKET=$OSS_BUCKET \
        --env DATASET=$DATASET \
        --env ADV_MODE=$ADV_MODE \
        --env BETA=$BETA \
        --env EMA_ALPHA=$EMA_ALPHA \
        --env CLIP_VALUE=$CLIP_VAL \
        --env REWARD_TYPE=$REWARD_TYPE \
        --env ENTROPY_GATE=$ENTROPY_GATE \
        --env ENTROPY_GATE_RATIO=$ENTROPY_GATE_RATIO \
        --env DISTILL_TOPK=$DISTILL_TOPK \
        --env DISTILL_TEMPERATURE=$DISTILL_TEMPERATURE \
        --env CLIP_ADV=$CLIP_ADV \
        --env CLIP_ADV_VALUE=$CLIP_ADV_VALUE \
        --env NORM_ADV_BY_STD=$NORM_ADV_BY_STD \
        --env ADV_STD_FLOOR=$ADV_STD_FLOOR \
        --env ADV_ENTROPY_WEIGHT=$ADV_ENTROPY_WEIGHT \
        --env GROUP_MEAN_MODE=$GROUP_MEAN_MODE \
        --env CLIP_RATIO_HIGH=$CLIP_RATIO_HIGH \
        --env REPETITION_PENALTY=$REPETITION_PENALTY \
        --env ROLLOUT_N=$ROLLOUT_N \
        --env ROLLOUT_TEMPERATURE=$ROLLOUT_TEMPERATURE \
        --env TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE \
        --env GEN_BATCH_SIZE=$GEN_BATCH_SIZE \
        --env MINI_BATCH_SIZE=$MINI_BATCH_SIZE \
        --env INCLUDE_SUCCESSFUL_ROLLOUTS=$INCLUDE_SUCCESSFUL_ROLLOUTS \
        --env TEACHER_REG=$TEACHER_REG \
        --env TEACHER_UPDATE_RATE=$TEACHER_UPDATE_RATE \
        --env LR=$LR \
        --env ENTROPY_COEFF=$ENTROPY_COEFF \
        --env SEED=$SEED \
        --env FILTER_GROUPS_ENABLE=$FILTER_GROUPS_ENABLE \
        --env FILTER_GROUPS_METRIC=$FILTER_GROUPS_METRIC \
        --env FILTER_GROUPS_MAX_GEN=$FILTER_GROUPS_MAX_GEN \
        --env MODEL_PATH=$MODEL_PATH \
        --env JOB_NAME=$JOB_NAME \
        --env PROJECT_NAME=$PROJECT_NAME \
        --env GIT_BRANCH=$GIT_BRANCH \
        --env GIT_COMMIT=$GIT_COMMIT"
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] 命令:"
        echo "$CMD"
    else
        echo "提交中..."
        OUTPUT=$(eval "$CMD" 2>&1)
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            # 提取 task_id
            TASK_ID=$(echo "$OUTPUT" | grep -oP 'task_id["\s:]+\K[a-f0-9]+' | head -1)
            if [ -n "$TASK_ID" ]; then
                echo "✅ 提交成功: task_id=$TASK_ID"
                TASK_IDS+=("$TASK_ID")
            else
                echo "✅ 提交成功（未提取到 task_id）"
                echo "输出: $OUTPUT"
            fi
        else
            echo "❌ 提交失败 (exit code: $EXIT_CODE)"
            echo "输出: $OUTPUT"
            FAILED=$((FAILED + 1))
        fi
    fi
    
    sleep 2
done

# =============================================================================
# 总结
# =============================================================================

echo ""
echo "============================================"
echo "提交完成"
echo "============================================"
echo "成功: $((${#EXPERIMENTS[@]} - FAILED))"
echo "失败: $FAILED"
echo ""

if [ ${#TASK_IDS[@]} -gt 0 ]; then
    echo "Task IDs:"
    for TID in "${TASK_IDS[@]}"; do
        echo "  - $TID"
    done
    echo ""
    echo "监控命令:"
    echo "  nebula logs <task_id>"
    echo "  nebula status <task_id>"
fi

echo "============================================"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
