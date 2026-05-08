#!/bin/bash
# =============================================================================
# SDPO Baseline 实验 - Nebula 批量提交脚本
#
# 算法：SDPO（Self-Distillation Policy Optimization）
# 特点：on-policy, single-step update, Jensen-Shannon divergence
# 参考：SDPO 论文 Table 3 + run_sdpo_all.sh
#
# 使用方式：
#   bash nebula_scripts/submit_sdpo_comparison.sh [--dry-run]
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
# 实验矩阵（SDPO on-policy 配置，论文 Table 3）
# =============================================================================

declare -a EXPERIMENTS=(
    # alpha | dont_reprompt | 实验标签
    "0.5|True|sdpo_js_alpha0.5"
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
echo "SDPO Baseline 实验提交"
echo "============================================"
echo "实验数量: ${#EXPERIMENTS[@]}"
echo "  - SDPO: on-policy, alpha=0.5 (JS divergence, 论文 Table 3)"
echo "数据集: ${DATASET}"
echo "模型: Qwen3-8B"
echo "SwanLab组: Algorithm-Comparison-v1"
echo "============================================"

TASK_IDS=()
FAILED=0

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r ALPHA DONT_REPROMPT EXP_LABEL <<< "$EXP"
    
    JOB_NAME="${EXP_LABEL}"
    
    echo ""
    echo "────────────────────────────────────────"
    echo "提交实验: ${JOB_NAME}"
    echo "  ALGORITHM: sdpo"
    echo "  ALPHA: ${ALPHA} (Jensen-Shannon)"
    echo "  DONT_REPROMPT: ${DONT_REPROMPT}"
    
    # Git 信息
    GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
    GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    
    # ── 提交 ────────────────────────────────────────────────────────
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] 准备提交"
        echo "  JOB_NAME: ${JOB_NAME}"
        echo "  ALPHA: ${ALPHA}"
    else
        echo "提交中..."
        
        SUBMIT_OUTPUT=$(nebulactl run mdl \
            --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=ALGORITHM=sdpo --env=MODEL_PATH=${MODEL_PATH} --env=LR=${LR} --env=SEED=${SEED} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=GEN_BATCH_SIZE=${GEN_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=SDPO_ALPHA=${ALPHA} --env=SDPO_DONT_REPROMPT=${DONT_REPROMPT} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=f598ad33b071751bf79d2484d8e1acefe8df9d879e129cae40340a158854f9cb --env=DINGTALK_SECRET=SECc5b9e4f61f56b32b46abf1ecedc11bdcba10dc35fbba8fa0ff62c084a1cc6ad3" \
            --worker_count=${WORLD_SIZE} \
            --file.cluster_file=${CLUSTER_FILE} \
            --job_name=${JOB_NAME} \
            --env=OPENLM_TOKEN=${OPENLM_TOKEN} \
            --env=SWANLAB_API_KEY=${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal} \
            --custom_docker_image=${CUSTOM_DOCKER_IMAGE} \
            --requirements_file_name=requirements_nebula.txt \
            --oss_access_id=${OSS_ACCESS_ID} \
            --oss_access_key=${OSS_ACCESS_KEY} \
            --oss_bucket=${OSS_BUCKET} \
            --oss_endpoint=${OSS_ENDPOINT} \
            2>&1)
        SUBMIT_EXIT=$?
        echo "$SUBMIT_OUTPUT"
        if [ $SUBMIT_EXIT -ne 0 ]; then
            echo "❌ 提交失败 (exit code: $SUBMIT_EXIT)"
            FAILED=$((FAILED + 1))
        else
            # 提取 task_id（兼容 macOS grep）
            TASK_ID=$(echo "$SUBMIT_OUTPUT" | grep -o 'task_id[": ]*[a-f0-9]\+' | grep -o '[a-f0-9]\{20,\}' | head -1)
            if [ -n "$TASK_ID" ]; then
                echo "✅ 提交成功: task_id=$TASK_ID"
                TASK_IDS+=("$TASK_ID")
            else
                echo "✅ 提交成功（未提取到 task_id）"
            fi
        fi
        sleep 2
    fi
done

# =============================================================================
# 总结
# =============================================================================

echo ""
echo "============================================"
echo "提交完成"
echo "============================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 模式：共 ${#EXPERIMENTS[@]} 个实验"
else
    echo "成功: $((${#EXPERIMENTS[@]} - FAILED))"
    echo "失败: ${FAILED}"
    echo ""
    if [ ${#TASK_IDS[@]} -gt 0 ]; then
        echo "Task IDs:"
        for tid in "${TASK_IDS[@]}"; do
            echo "  - $tid"
        done
    fi
fi
echo "============================================"
