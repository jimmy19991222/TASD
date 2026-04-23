#!/bin/bash
# =============================================================================
# DAPO - Nebula 批量提交脚本
#
# 使用方式：
#   bash nebula_scripts/submit_dapo_sweep.sh [--dry-run]
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
SCRIPT_PATH="nebula_scripts/dapo/dapo_sciknoweval_parametric.sh"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="TASD-v3"

# ── 数据集配置 ──────────────────────────────────────────────────────
DATASETS=(
    "sciknoweval/biology"
    # "sciknoweval/chemistry"
    # "sciknoweval/material"
    # "sciknoweval/physics"
    # "tooluse"
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

# ── Entropy Coeff ────────────────────────────────────────────────────
# DAPO 原论文推荐 0.001，0 退回标准 GRPO
ENTROPY_COEFF_LIST=(
    "0.001"
    # "0.0"
)

# ── Clip Ratio High ──────────────────────────────────────────────────
# DAPO 原论文推荐 clip_high >> clip_low（如 0.28 或 10000）
CLIP_RATIO_HIGH_LIST=(
    # "10000"   # 等效于不 clip 上界
    "0.28"  # DAPO 原论文设置
    # "0.2"   # 退回标准 PPO（上下对称 clip）
)

# ── Filter Groups ────────────────────────────────────────────────────
# 是否启用 Dynamic Sampling（过滤全对/全错 group）
FILTER_GROUPS_ENABLE_LIST=(
    "true"
    # "false"
)

# ── Repetition Penalty ───────────────────────────────────────────────
REPETITION_PENALTY_LIST=(
    "1.05"
    "1.0"
)

# ── 固定参数 ────────────────────────────────────────────────────────────
LR="1e-5"
SEED="42"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
ROLLOUT_N="8"
FILTER_GROUPS_METRIC="acc"
FILTER_GROUPS_MAX_GEN="0"
MODEL="Qwen3-8B"
# Git 信息（在本地获取，传递给 Nebula）
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# =============================================================================
# 提交循环
# =============================================================================
TOTAL=0
SUBMITTED=0

for DATASET in "${DATASETS[@]}"; do
for ENTROPY_COEFF in "${ENTROPY_COEFF_LIST[@]}"; do
for CLIP_RATIO_HIGH in "${CLIP_RATIO_HIGH_LIST[@]}"; do
for FILTER_GROUPS_ENABLE in "${FILTER_GROUPS_ENABLE_LIST[@]}"; do
for REPETITION_PENALTY in "${REPETITION_PENALTY_LIST[@]}"; do

    TOTAL=$((TOTAL + 1))

    DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
    MODEL_SHORT=$(echo "$MODEL" | tr '.' '-')
    OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
    MODEL_PATH="${OSS_ROOT}/base_models/${MODEL}"

    # ── 构建实验名 ───────────────────────────────────────────────────
    # filter_groups 标签
    if [ "$FILTER_GROUPS_ENABLE" = "true" ]; then
        FG_TAG="-dynSamp"
    else
        FG_TAG=""
    fi

    # clip_ratio_high 标签
    if [ "$CLIP_RATIO_HIGH" = "10000" ]; then
        CLIP_TAG="-clipHigh"
    else
        CLIP_TAG="-clip_high${CLIP_RATIO_HIGH}"
    fi

    # entropy 标签
    ENT_TAG="-ent${ENTROPY_COEFF}"

    # repetition penalty 标签
    REP_TAG="-rep${REPETITION_PENALTY}"

    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="DAPO-${DATASET_SHORT}${CLIP_TAG}${FG_TAG}${ENT_TAG}${REP_TAG}-${MODEL_SHORT}-${CURRENT_TIME}"

    # ── 提交 ────────────────────────────────────────────────────────
    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  ENTROPY_COEFF=$ENTROPY_COEFF CLIP_RATIO_HIGH=$CLIP_RATIO_HIGH"
        echo "  FILTER_GROUPS=$FILTER_GROUPS_ENABLE REPETITION_PENALTY=$REPETITION_PENALTY"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
            --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL=${MODEL} --env=MODEL_PATH=${MODEL_PATH} --env=LR=${LR} --env=SEED=${SEED} --env=ENTROPY_COEFF=${ENTROPY_COEFF} --env=CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH} --env=FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE} --env=FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC} --env=FILTER_GROUPS_MAX_GEN=${FILTER_GROUPS_MAX_GEN} --env=REPETITION_PENALTY=${REPETITION_PENALTY} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT}" \
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
        else
            SUBMITTED=$((SUBMITTED + 1))
            echo "✅ 已提交 (${SUBMITTED}/${TOTAL})"
        fi
        sleep 2
    fi

done
done
done
done
done

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
