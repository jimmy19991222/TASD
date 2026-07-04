#!/usr/bin/env bash
# =============================================================================
# Token-level DPO v2 实验批量提交
#
# 实验矩阵：4 sciknoweval 数据集 × 3 配置 = 12 个实验
#   全部使用 reference-free DPO (TOKEN_DPO_USE_REF=False)
#
# 维度：
#   数据集: biology, chemistry, material, physics
#   配置1: β=0.5, entropy_filter=False
#   配置2: β=0.1, entropy_filter=False
#   配置3: β=0.5, entropy_filter=True
#   (去掉 β=0.1 + entropy_filter=True 最激进组合)
#
# 使用方式：
#   bash nebula_scripts/submit_token_dpo_v2.sh [--dry-run]
#
# 环境变量（需提前设置）：
#   OPENLM_TOKEN, OSS_ACCESS_ID, OSS_ACCESS_KEY
# 可选覆盖：
#   PROJECT_NAME, SWANLAB_API_KEY, CUSTOM_DOCKER_IMAGE, TOTAL_TRAINING_STEPS
# =============================================================================
set -euo pipefail

# ── Nebula 账号配置 ──────────────────────────────────────────────────────
QUEUE="lazada_llm_ad_h20"
WORLD_SIZE=1
OPENLM_TOKEN="${OPENLM_TOKEN:?OPENLM_TOKEN not set}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:?OSS_ACCESS_ID not set}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?OSS_ACCESS_KEY not set}"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
CLUSTER_FILE="nebula_scripts/cluster.json"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="${PROJECT_NAME:-TokenDPO}"

# Capture branch / commit so each Nebula job logs them into SwanLab
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"

# Checkpoint / validation cadence env defaults forwarded into every job
TEST_FREQ="${TEST_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-10}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
SAVE_HF_ONLY="${SAVE_HF_ONLY:-True}"

# ── 命令行参数解析 ──────────────────────────────────────────────────────
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 实验参数
# =============================================================================
MODEL_NAME="Qwen3-8B"
SEED="42"
LR="1e-5"
TRAIN_BATCH_SIZE="32"
ROLLOUT_N="8"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-250}"
DONT_REPROMPT_ON_SELF_SUCCESS="True"

# 实验矩阵：4 数据集 × 3 配置 = 12 个实验
DATASETS=(
    "sciknoweval/biology"
    "sciknoweval/chemistry"
    "sciknoweval/material"
    "sciknoweval/physics"
)

# 3 组配置：BETA|ENTROPY_FILTER|BETA_TAG|ENTROPY_TAG
# (去掉 β=0.1 + entropy_filter=True 最激进组合)
CONFIGS=(
    "0.5|False|b05|off"
    "0.1|False|b01|off"
    "0.5|True|b05|on"
)

SCRIPT_PATH="nebula_scripts/sdpo/token_dpo_parametric.sh"

# =============================================================================
TOTAL=0
SUBMITTED=0

# 辅助函数：提交单个 job
_submit_job() {
    local JOB_NAME="$1"
    local USER_PARAMS="$2"

    TOTAL=$((TOTAL + 1))

    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}  [script: ${SCRIPT_PATH}]"
        echo "  $USER_PARAMS"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"
        SUBMIT_OUTPUT=$(nebulactl run mdl \
            --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} ${USER_PARAMS}" \
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
}

# 辅助函数：数据集 → 短名映射
_dataset_short() {
    case "$1" in
        sciknoweval/biology)    echo "bio" ;;
        sciknoweval/chemistry)  echo "chem" ;;
        sciknoweval/material)   echo "mat" ;;
        sciknoweval/physics)    echo "phy" ;;
        *)                      echo "$1" | tr '/' '-' ;;
    esac
}

# 辅助函数：β 值 → 标签
_beta_tag() {
    case "$1" in
        0.5)  echo "b05" ;;
        0.1)  echo "b01" ;;
        *)    echo "b${1//./}" ;;
    esac
}

# 辅助函数：熵过滤 → 标签
_entropy_tag() {
    case "$1" in
        True)  echo "on" ;;
        False) echo "off" ;;
        *)     echo "$1" ;;
    esac
}

# ─────────────────────────────────────────────────────────────────────────────
# 提交实验：4 数据集 × 3 配置 = 12 个实验 (noref only)
# ─────────────────────────────────────────────────────────────────────────────
GROUP_NAME="TokenDPO-v2"

for DATASET in "${DATASETS[@]}"; do
    for CONFIG in "${CONFIGS[@]}"; do
        IFS='|' read -r BETA ENTROPY_FILTER BETA_TAG ENTROPY_TAG <<< "$CONFIG"
        DS_SHORT=$(_dataset_short "$DATASET")
        CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
        JOB_NAME="token-dpo-v2-${DS_SHORT}-${BETA_TAG}-ef${ENTROPY_TAG}-${MODEL_NAME}-${CURRENT_TIME}"

        _submit_job "$JOB_NAME" \
            "--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=GROUP_NAME=${GROUP_NAME} --env=DATASET=${DATASET} --env=MODEL_NAME=${MODEL_NAME} --env=LR=${LR} --env=DONT_REPROMPT_ON_SELF_SUCCESS=${DONT_REPROMPT_ON_SELF_SUCCESS} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=SEED=${SEED} --env=TOKEN_DPO_USE_REF=False --env=TOKEN_DPO_BETA=${BETA} --env=TOKEN_DPO_ENTROPY_FILTER=${ENTROPY_FILTER} --env=TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ} --env=VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN} --env=SAVE_HF_ONLY=${SAVE_HF_ONLY} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT}"
    done
done

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job → project=${PROJECT_NAME}"
fi
echo "============================================================"
