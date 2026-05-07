#!/bin/bash
# =============================================================================
# GRPO Baseline 实验 - Nebula 批量提交脚本
#
# 算法：GRPO（Group Relative Policy Optimization）
# 特点：off-policy, multi-step updates, mini_batch_size=8
# 参考：SDPO 论文 Table 3 + run_baseline_grpo_all.sh
#
# 使用方式：
#   bash nebula_scripts/submit_grpo_comparison.sh [--dry-run]
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
# 实验矩阵（GRPO off-policy 和 on-policy 配置）
# =============================================================================

declare -a EXPERIMENTS=(
    # mini_batch_size | 更新步数 | 实验标签
    "8|4|grpo_offpolicy_mbs8"
    "32|1|grpo_onpolicy_mbs32"
)

# ── 共享超参（仅包含原始论文实验中明确指定的参数）────────────────────────
LR="1e-5"
SEED="42"
TRAIN_BATCH_SIZE="32"
GEN_BATCH_SIZE="32"
ROLLOUT_N="8"
MODEL_PATH="/data/oss_bucket_0/ad/loujieming.ljm/models/Qwen3-8B"

# =============================================================================
# 提交实验
# =============================================================================

echo "============================================"
echo "GRPO Baseline 实验提交"
echo "============================================"
echo "实验数量: ${#EXPERIMENTS[@]}"
echo "  - GRPO (off-policy): mini_batch=8, 4步更新 (32/8)"
echo "  - GRPO (on-policy): mini_batch=32, 1步更新 (32/32)"
echo "数据集: ${DATASET}"
echo "模型: Qwen3-8B"
echo "SwanLab组: Algorithm-Comparison-v1"
echo "============================================"

TASK_IDS=()
FAILED=0

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r MINI_BATCH_SIZE NUM_UPDATES EXP_LABEL <<< "$EXP"
    
    JOB_NAME="grpo_${EXP_LABEL}"
    
    echo ""
    echo "────────────────────────────────────────"
    echo "提交实验: ${JOB_NAME}"
    echo "  ALGORITHM: grpo"
    echo "  MINI_BATCH_SIZE: ${MINI_BATCH_SIZE}"
    echo "  更新步数: ${NUM_UPDATES} (train_batch=32 / mini_batch=${MINI_BATCH_SIZE})"
    
    # 构建环境变量
    export DATASET="$DATASET"
    export ALGORITHM="grpo"
    export MODEL_PATH="$MODEL_PATH"
    export LR="$LR"
    export SEED="$SEED"
    export TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE"
    export GEN_BATCH_SIZE="$GEN_BATCH_SIZE"
    export MINI_BATCH_SIZE="$MINI_BATCH_SIZE"
    export ROLLOUT_N="$ROLLOUT_N"
    export JOB_NAME="$JOB_NAME"
    export PROJECT_NAME="$PROJECT_NAME"
    export GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
    export GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    
    # 构建 nebulactl 命令
    CMD="nebulactl run mdl \
        --force \
        --engine=xdl \
        --queue=$QUEUE \
        --entry=nebula_scripts/entry.py \
        --worker_count=$WORLD_SIZE \
        --file.cluster_file=$CLUSTER_FILE \
        --job_name=$JOB_NAME \
        --env=OPENLM_TOKEN=$OPENLM_TOKEN \
        --env=SWANLAB_API_KEY=\${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal} \
        --custom_docker_image=$CUSTOM_DOCKER_IMAGE \
        --requirements_file_name=requirements_nebula.txt \
        --oss_access_id=$OSS_ACCESS_ID \
        --oss_access_key=$OSS_ACCESS_KEY \
        --oss_bucket=$OSS_BUCKET \
        --oss_endpoint=$OSS_ENDPOINT \
        --user_params=\"DATASET=$DATASET ALGORITHM=grpo MODEL_PATH=$MODEL_PATH LR=$LR SEED=$SEED TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE GEN_BATCH_SIZE=$GEN_BATCH_SIZE MINI_BATCH_SIZE=$MINI_BATCH_SIZE ROLLOUT_N=$ROLLOUT_N JOB_NAME=$JOB_NAME PROJECT_NAME=$PROJECT_NAME GIT_BRANCH=$GIT_BRANCH GIT_COMMIT=$GIT_COMMIT\""
    
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
