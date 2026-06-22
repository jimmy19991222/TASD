#!/bin/bash
# =============================================================================
# Vanilla SDPO baseline on MATH500 & GSM8K
#
# Usage:
#   bash nebula_scripts/submit_sdpo_math_baseline.sh [--dry-run]
# =============================================================================
set -euo pipefail

QUEUE="lazada_llm_ad_h20"
WORLD_SIZE=1
OPENLM_TOKEN="${OPENLM_TOKEN:?OPENLM_TOKEN not set}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:?OSS_ACCESS_ID not set}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?OSS_ACCESS_KEY not set}"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
CLUSTER_FILE="nebula_scripts/cluster.json"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="SDPO-Math-Baseline"

GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 模式：只打印命令，不提交"
fi

# ── 超参 ─────────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
LR="${LR:-1e-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-16}"
ALPHA="${ALPHA:-0.5}"
DONT_REPROMPT="${DONT_REPROMPT:-False}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
SEED="${SEED:-42}"
TEST_FREQ=10
SAVE_FREQ=10

TOTAL=0
SUBMITTED=0

SDPO_SCRIPT="nebula_scripts/sdpo/sdpo_sciknoweval_parametric.sh"

_submit_job() {
    local SCRIPT_PATH="$1"
    local JOB_NAME="$2"
    local USER_PARAMS="$3"

    TOTAL=$((TOTAL + 1))

    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  script: ${SCRIPT_PATH}"
        echo "  ${USER_PARAMS}"
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

_base_env() {
    local JN="$1"
    local DS="$2"
    echo "--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JN} --env=DATASET=${DS} --env=MODEL_NAME=${MODEL_NAME} --env=LR=${LR} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=ALPHA=${ALPHA} --env=DONT_REPROMPT_ON_SELF_SUCCESS=${DONT_REPROMPT} --env=TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS} --env=SEED=${SEED} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ} --env=VAL_BEFORE_TRAIN=True --env=SAVE_HF_ONLY=True --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT}"
}

# =============================================================================
# Job 1: SDPO baseline on MATH-500
# =============================================================================
T=$(date +%Y%m%d_%H%M%S)
JN="SDPO-baseline-math500-${T}"
_submit_job "$SDPO_SCRIPT" "$JN" "$(_base_env "$JN" "math500")"

# =============================================================================
# Job 2: SDPO baseline on GSM8K
# =============================================================================
T=$(date +%Y%m%d_%H%M%S)
JN="SDPO-baseline-gsm8k-${T}"
_submit_job "$SDPO_SCRIPT" "$JN" "$(_base_env "$JN" "gsm8k")"

# =============================================================================
echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
