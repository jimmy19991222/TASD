#!/bin/bash
# =============================================================================
# 算法对比实验 - Nebula 批量提交脚本
#
# 对比算法（7 个实验）：
#   Baseline 算法（3 个）：
#     1. GRPO: 标准 GRPO baseline
#     2. SDPO: Self-Distillation Policy Optimization
#     3. FIPO: Future-KL Influenced Policy Optimization
#
#   Self-Teacher Advantage beta 消融（4 个）：
#     4. beta=1.0: 纯 V_CE（横向 baseline，variance reduction 最优）
#     5. beta=0.7: V_CE 主导（推荐配置，V_EMA 辅助捕捉转折）
#     6. beta=0.5: 等权融合（V_CE 和 V_EMA 信号最强）
#     7. beta=0.0: 纯 V_EMA（纵向 baseline，验证 EMA 必要性）
#
# 使用方式：
#   bash nebula_scripts/submit_algorithm_comparison.sh [--dry-run]
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
SCRIPT_PATH="nebula_scripts/algorithm_comparison/algorithm_comparison_parametric.sh"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="Algorithm-Comparison-v1"

# ── 数据集配置 ──────────────────────────────────────────────────────
DATASET="sciknoweval/biology"

# ── dry-run 模式 ─────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 实验矩阵（4 个算法对比）
# =============================================================================

declare -a EXPERIMENTS=(
    # 算法 | 特殊参数 | 实验标签
    "grpo|mini_batch_size=8|grpo_offpolicy_mbs8"
    "sdpo|alpha=0.5,dont_reprompt=True,mini_batch_size=32|sdpo_js_alpha0.5"
    "fipo|mini_batch_size=8|fipo_offpolicy_mbs8"
    "self_teacher|beta=1.0,ema_alpha=0.9,clip_value=5.0,mini_batch_size=32|self_teacher_beta1.0_Vce_only"
    "self_teacher|beta=0.7,ema_alpha=0.9,clip_value=5.0,mini_batch_size=32|self_teacher_beta0.7_recommended"
    "self_teacher|beta=0.5,ema_alpha=0.9,clip_value=5.0,mini_batch_size=32|self_teacher_beta0.5_equal"
    "self_teacher|beta=0.0,ema_alpha=0.9,clip_value=5.0,mini_batch_size=32|self_teacher_beta0.0_Vema_only"
)

# ── 共享超参 ─────────────────────────────────────────────────────────
LR="1e-5"
ENTROPY_COEFF="0.001"
SEED="42"
TRAIN_BATCH_SIZE="32"
GEN_BATCH_SIZE="32"
# MINI_BATCH_SIZE 由各实验单独指定（GRPO/FIPO=8, SDPO/Self-Teacher=32）
ROLLOUT_N="8"
ROLLOUT_TEMPERATURE="1.0"
REPETITION_PENALTY="1.05"
CLIP_RATIO_HIGH="0.28"
MODEL_PATH="/data/oss_bucket_0/ad/loujieming.ljm/models/Qwen3-8B"

# =============================================================================
# 提交实验
# =============================================================================

echo "============================================"
echo "算法对比实验提交"
echo "============================================"
echo "实验数量: ${#EXPERIMENTS[@]}"
echo "  - GRPO: off-policy, mini_batch=8 (论文参数)"
echo "  - SDPO: on-policy, alpha=0.5 (JS divergence, 论文 Table 3)"
echo "  - FIPO: off-policy, mini_batch=8 (与 GRPO 共享参数)"
echo "  - Self-Teacher: on-policy, beta 消融 (1.0/0.7/0.5/0.0)"
echo "数据集: ${DATASET}"
echo "模型: Qwen3-8B"
echo "SwanLab组: Algorithm-Comparison-v1"
echo "============================================"

TASK_IDS=()
FAILED=0

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r ALGORITHM SPECIAL_PARAMS EXP_LABEL <<< "$EXP"
    
    JOB_NAME="alg_comp_${EXP_LABEL}"
    
    echo ""
    echo "────────────────────────────────────────"
    echo "提交实验: ${JOB_NAME}"
    echo "  ALGORITHM: ${ALGORITHM}"
    if [ -n "$SPECIAL_PARAMS" ]; then
        echo "  参数: ${SPECIAL_PARAMS}"
    fi
    
    # 解析特殊参数
    MINI_BATCH_SIZE="32"  # 默认值（会被算法逻辑覆盖）
    SDPO_ALPHA="0.5"  # 默认 Jensen-Shannon
    SDPO_DONT_REPROMPT="True"
    ADV_MODE="self_teacher"
    BETA="0.7"
    EMA_ALPHA="0.9"
    CLIP_VALUE="5.0"
    
    if [[ "$SPECIAL_PARAMS" == *"mini_batch_size="* ]]; then
        MINI_BATCH_SIZE=$(echo "$SPECIAL_PARAMS" | sed -n 's/.*mini_batch_size=\([0-9]*\).*/\1/p')
    fi
    if [[ "$SPECIAL_PARAMS" == *"alpha="* ]]; then
        SDPO_ALPHA=$(echo "$SPECIAL_PARAMS" | sed -n 's/.*alpha=\([0-9.]*\).*/\1/p')
    fi
    if [[ "$SPECIAL_PARAMS" == *"dont_reprompt="* ]]; then
        SDPO_DONT_REPROMPT=$(echo "$SPECIAL_PARAMS" | sed -n 's/.*dont_reprompt=\([A-Za-z]*\).*/\1/p')
    fi
    if [[ "$SPECIAL_PARAMS" == *"beta="* ]]; then
        BETA=$(echo "$SPECIAL_PARAMS" | sed -n 's/.*beta=\([0-9.]*\).*/\1/p')
    fi
    if [[ "$SPECIAL_PARAMS" == *"ema_alpha="* ]]; then
        EMA_ALPHA=$(echo "$SPECIAL_PARAMS" | sed -n 's/.*ema_alpha=\([0-9.]*\).*/\1/p')
    fi
    if [[ "$SPECIAL_PARAMS" == *"clip_value="* ]]; then
        CLIP_VALUE=$(echo "$SPECIAL_PARAMS" | sed -n 's/.*clip_value=\([0-9.]*\).*/\1/p')
    fi
    
    # 构建环境变量
    export DATASET="$DATASET"
    export ALGORITHM="$ALGORITHM"
    export MODEL_PATH="$MODEL_PATH"
    export LR="$LR"
    export ENTROPY_COEFF="$ENTROPY_COEFF"
    export SEED="$SEED"
    export TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE"
    export GEN_BATCH_SIZE="$GEN_BATCH_SIZE"
    export MINI_BATCH_SIZE="$MINI_BATCH_SIZE"
    export ROLLOUT_N="$ROLLOUT_N"
    export ROLLOUT_TEMPERATURE="$ROLLOUT_TEMPERATURE"
    export REPETITION_PENALTY="$REPETITION_PENALTY"
    export CLIP_RATIO_HIGH="$CLIP_RATIO_HIGH"
    export SDPO_ALPHA="$SDPO_ALPHA"
    export SDPO_DONT_REPROMPT="$SDPO_DONT_REPROMPT"
    export ADV_MODE="$ADV_MODE"
    export BETA="$BETA"
    export EMA_ALPHA="$EMA_ALPHA"
    export CLIP_VALUE="$CLIP_VALUE"
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
        --env ALGORITHM=$ALGORITHM \
        --env MODEL_PATH=$MODEL_PATH \
        --env LR=$LR \
        --env ENTROPY_COEFF=$ENTROPY_COEFF \
        --env SEED=$SEED \
        --env TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE \
        --env GEN_BATCH_SIZE=$GEN_BATCH_SIZE \
        --env MINI_BATCH_SIZE=$MINI_BATCH_SIZE \
        --env ROLLOUT_N=$ROLLOUT_N \
        --env ROLLOUT_TEMPERATURE=$ROLLOUT_TEMPERATURE \
        --env REPETITION_PENALTY=$REPETITION_PENALTY \
        --env CLIP_RATIO_HIGH=$CLIP_RATIO_HIGH \
        --env SDPO_ALPHA=$SDPO_ALPHA \
        --env SDPO_DONT_REPROMPT=$SDPO_DONT_REPROMPT \
        --env ADV_MODE=$ADV_MODE \
        --env BETA=$BETA \
        --env EMA_ALPHA=$EMA_ALPHA \
        --env CLIP_VALUE=$CLIP_VALUE \
        --env JOB_NAME=$JOB_NAME \
        --env PROJECT_NAME=$PROJECT_NAME \
        --env GIT_BRANCH=$GIT_BRANCH \
        --env GIT_COMMIT=$GIT_COMMIT"
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] 命令:"
        echo "$CMD" | tr ' ' '\n' | grep "^--env" | sed 's/^--env /  /'
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
    echo ""
    echo "SwanLab 查看:"
    echo "  https://swanlab.cn/@oh-my-team/Algorithm-Comparison-v1"
fi

echo "============================================"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
