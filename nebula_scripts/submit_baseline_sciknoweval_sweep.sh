#!/bin/bash
# =============================================================================
# Baseline 超参扫描 - Sciknoweval 数据集（GRPO + SDPO）
#
# 使用方式：
#   bash nebula_scripts/submit_baseline_sciknoweval_sweep.sh [--dry-run]
#   bash nebula_scripts/submit_baseline_sciknoweval_sweep.sh --algo grpo [--dry-run]
#   bash nebula_scripts/submit_baseline_sciknoweval_sweep.sh --algo sdpo [--dry-run]
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
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="TASD-v3"

# ── 命令行参数解析 ──────────────────────────────────────────────────────
DRY_RUN=false
ALGO="all"    # all | grpo | sdpo

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --algo=*) ALGO="${arg#*=}" ;;
        --algo) shift; ALGO="$1" ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 超参配置
# =============================================================================
MODEL_NAMES=(
    "Qwen3-8B"
)

DATASETS=(
    "sciknoweval/biology"
    # "sciknoweval/chemistry"
    # "sciknoweval/physics"
    # "sciknoweval/material"
    # "tooluse"
)

# 固定参数
SEED="42"
TRAIN_BATCH_SIZE="32"
ROLLOUT_N="8"

# ── GRPO 专属超参 ──────────────────────────────────────────────────────
GRPO_LRS=("1e-5")
GRPO_MINI_BATCH_SIZES=("32")

# ── SDPO 专属超参 ──────────────────────────────────────────────────────
SDPO_LRS=("1e-5")
SDPO_ALPHAS=("0.5")
SDPO_DONT_REPROMPT_LIST=("False")

# =============================================================================
TOTAL=0
SUBMITTED=0

# 辅助函数：提交单个 job
_submit_job() {
    local SCRIPT_PATH="$1"
    local JOB_NAME="$2"
    local USER_PARAMS="$3"

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
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} ${USER_PARAMS} --env=DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=f598ad33b071751bf79d2484d8e1acefe8df9d879e129cae40340a158854f9cb --env=DINGTALK_SECRET=SECc5b9e4f61f56b32b46abf1ecedc11bdcba10dc35fbba8fa0ff62c084a1cc6ad3" \
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

# ─────────────────────────────────────────────────────────────────────────────
# GRPO - sciknoweval
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$ALGO" == "all" || "$ALGO" == "grpo" ]]; then
    SCRIPT_PATH="nebula_scripts/grpo/grpo_sciknoweval_parametric.sh"
    for DATASET in "${DATASETS[@]}"; do
    for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for LR in "${GRPO_LRS[@]}"; do
    for MINI_BATCH_SIZE in "${GRPO_MINI_BATCH_SIZES[@]}"; do
        DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
        LR_TAG=$(echo "$LR" | tr '-' '_')
        CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
        JOB_NAME="GRPO-${DATASET_SHORT}-mbs${MINI_BATCH_SIZE}-lr${LR_TAG}-${MODEL_NAME}-${CURRENT_TIME}"
        _submit_job "$SCRIPT_PATH" "$JOB_NAME" \
            "--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL_NAME=${MODEL_NAME} --env=LR=${LR} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=SEED=${SEED}"
    done; done; done; done
fi

# ─────────────────────────────────────────────────────────────────────────────
# SDPO - sciknoweval
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$ALGO" == "all" || "$ALGO" == "sdpo" ]]; then
    SCRIPT_PATH="nebula_scripts/sdpo/sdpo_sciknoweval_parametric.sh"
    for DATASET in "${DATASETS[@]}"; do
    for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for LR in "${SDPO_LRS[@]}"; do
    for ALPHA in "${SDPO_ALPHAS[@]}"; do
    for DONT_REPROMPT_ON_SELF_SUCCESS in "${SDPO_DONT_REPROMPT_LIST[@]}"; do
        DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
        LR_TAG=$(echo "$LR" | tr '-' '_')
        REPROMPT_TAG=$([ "$DONT_REPROMPT_ON_SELF_SUCCESS" = "True" ] && echo "noReprompt" || echo "reprompt")
        CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
        JOB_NAME="SDPO-${DATASET_SHORT}-alpha${ALPHA}-lr${LR_TAG}-${REPROMPT_TAG}-${MODEL_NAME}-${CURRENT_TIME}"
        _submit_job "$SCRIPT_PATH" "$JOB_NAME" \
            "--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL_NAME=${MODEL_NAME} --env=LR=${LR} --env=ALPHA=${ALPHA} --env=DONT_REPROMPT_ON_SELF_SUCCESS=${DONT_REPROMPT_ON_SELF_SUCCESS} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=SEED=${SEED}"
    done; done; done; done; done
fi

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job (algo=${ALGO}, dataset=sciknoweval)"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job (algo=${ALGO}, dataset=sciknoweval)"
fi
echo "============================================================"
