#!/bin/bash
# =============================================================================
# VCAC (Verdict-Conditioned Advantage Correction) sweep — Nebula batch submitter
#
# VCAC adds per-token credit shaping to the GRPO PG loss pathway:
#     A_t = A_GRPO + vcac_lambda · δ_t
# where δ_t = log π_T^+(y_t|x,y_<t) − log π_T^-(y_t|x,y_<t) is the CDM
# dual-teacher sampled-token log-prob differential (Bayes-clean verdict
# log-odds shift). Marker design: SDPO sibling-success ref reprompt
# (cdm_use_ref=True) + verdict-only markers (no GT leakage at train time).
#
# Usage:
#   bash nebula_scripts/submit_vcac_sweep.sh [--dry-run]
#   bash nebula_scripts/submit_vcac_sweep.sh --dataset sciknoweval [--dry-run]
#   bash nebula_scripts/submit_vcac_sweep.sh --dataset lcb [--dry-run]
#   bash nebula_scripts/submit_vcac_sweep.sh --dataset tooluse [--dry-run]
#   bash nebula_scripts/submit_vcac_sweep.sh --dataset all [--dry-run]
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
PROJECT_NAME="${PROJECT_NAME:-VCAC}"

# ── 命令行参数解析 ──────────────────────────────────────────────────────
DRY_RUN=false
DATASET_GROUP="all"  # all | sciknoweval | lcb | tooluse

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --dataset=*) DATASET_GROUP="${arg#*=}" ;;
        --dataset) shift; DATASET_GROUP="$1" ;;
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

# 固定参数
SEED="42"
TRAIN_BATCH_SIZE="32"
ROLLOUT_N="8"
LR="1e-5"

# Best+last only:default SAVE_FREQ=9999 (in parametric script). Override here
# if periodic ckpts wanted.
SAVE_FREQ="${SAVE_FREQ:-9999}"
TEST_FREQ="${TEST_FREQ:-10}"

# ── VCAC sweep 超参 ────────────────────────────────────────────────────
# vcac_lambda: mixing weight on δ_t. Range 0.1~1.0 per research note.
# Start with three points to span the regime.
VCAC_LAMBDAS=("0.1" "0.3" "1.0")
# Optional: per-token clip on |δ_t|. "null" disables.
VCAC_CLIPS=("null")
# Optional: per-batch standardization of δ_t before λ scale.
VCAC_NORMALIZES=("False")
# CDM marker style; default matches SDPO "ref_mk" preset (no GT leakage).
CDM_TEMPLATE_VARIANTS=("ref_mk")

# =============================================================================
TOTAL=0
SUBMITTED=0

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

_build_common_env() {
    # $1: JOB_NAME, $2: VCAC_LAMBDA, $3: VCAC_CLIP, $4: VCAC_NORMALIZE,
    # $5: CDM_TEMPLATE_VARIANT, $6: MODEL_NAME, [$7: DATASET]
    local jn="$1" vl="$2" vc="$3" vn="$4" cv="$5" mn="$6" ds="${7:-}"
    local base="--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${jn} --env=MODEL_NAME=${mn} --env=LR=${LR} --env=VCAC_LAMBDA=${vl} --env=VCAC_CLIP=${vc} --env=VCAC_NORMALIZE=${vn} --env=CDM_TEMPLATE_VARIANT=${cv} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=SEED=${SEED} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ}"
    if [ -n "$ds" ]; then
        echo "$base --env=DATASET=${ds}"
    else
        echo "$base"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# sciknoweval
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$DATASET_GROUP" == "all" || "$DATASET_GROUP" == "sciknoweval" ]]; then
    SCRIPT_PATH="nebula_scripts/vcac/vcac_sciknoweval_parametric.sh"
    DATASETS=("sciknoweval/biology")
    for DATASET in "${DATASETS[@]}"; do
    for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for VCAC_LAMBDA in "${VCAC_LAMBDAS[@]}"; do
    for VCAC_CLIP in "${VCAC_CLIPS[@]}"; do
    for VCAC_NORMALIZE in "${VCAC_NORMALIZES[@]}"; do
    for CDM_VARIANT in "${CDM_TEMPLATE_VARIANTS[@]}"; do
        DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
        LAM_TAG=$(echo "$VCAC_LAMBDA" | tr '.' '_')
        CLIP_TAG=$([ "$VCAC_CLIP" = "null" ] && echo "" || echo "-clip${VCAC_CLIP}")
        NORM_TAG=$([ "$VCAC_NORMALIZE" = "True" ] && echo "-norm" || echo "")
        CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
        JOB_NAME="VCAC-${DATASET_SHORT}-lam${LAM_TAG}${CLIP_TAG}${NORM_TAG}-${CDM_VARIANT}-${MODEL_NAME}-${CURRENT_TIME}"
        ENV_STR=$(_build_common_env "$JOB_NAME" "$VCAC_LAMBDA" "$VCAC_CLIP" "$VCAC_NORMALIZE" "$CDM_VARIANT" "$MODEL_NAME" "$DATASET")
        _submit_job "$SCRIPT_PATH" "$JOB_NAME" "$ENV_STR"
    done; done; done; done; done; done
fi

# ─────────────────────────────────────────────────────────────────────────────
# lcb_v6
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$DATASET_GROUP" == "all" || "$DATASET_GROUP" == "lcb" ]]; then
    SCRIPT_PATH="nebula_scripts/vcac/vcac_lcb_parametric.sh"
    for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for VCAC_LAMBDA in "${VCAC_LAMBDAS[@]}"; do
    for VCAC_CLIP in "${VCAC_CLIPS[@]}"; do
    for VCAC_NORMALIZE in "${VCAC_NORMALIZES[@]}"; do
    for CDM_VARIANT in "${CDM_TEMPLATE_VARIANTS[@]}"; do
        LAM_TAG=$(echo "$VCAC_LAMBDA" | tr '.' '_')
        CLIP_TAG=$([ "$VCAC_CLIP" = "null" ] && echo "" || echo "-clip${VCAC_CLIP}")
        NORM_TAG=$([ "$VCAC_NORMALIZE" = "True" ] && echo "-norm" || echo "")
        CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
        JOB_NAME="VCAC-lcb_v6-lam${LAM_TAG}${CLIP_TAG}${NORM_TAG}-${CDM_VARIANT}-${MODEL_NAME}-${CURRENT_TIME}"
        ENV_STR=$(_build_common_env "$JOB_NAME" "$VCAC_LAMBDA" "$VCAC_CLIP" "$VCAC_NORMALIZE" "$CDM_VARIANT" "$MODEL_NAME")
        _submit_job "$SCRIPT_PATH" "$JOB_NAME" "$ENV_STR"
    done; done; done; done; done
fi

# ─────────────────────────────────────────────────────────────────────────────
# tooluse
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$DATASET_GROUP" == "all" || "$DATASET_GROUP" == "tooluse" ]]; then
    SCRIPT_PATH="nebula_scripts/vcac/vcac_tooluse_parametric.sh"
    for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for VCAC_LAMBDA in "${VCAC_LAMBDAS[@]}"; do
    for VCAC_CLIP in "${VCAC_CLIPS[@]}"; do
    for VCAC_NORMALIZE in "${VCAC_NORMALIZES[@]}"; do
    for CDM_VARIANT in "${CDM_TEMPLATE_VARIANTS[@]}"; do
        LAM_TAG=$(echo "$VCAC_LAMBDA" | tr '.' '_')
        CLIP_TAG=$([ "$VCAC_CLIP" = "null" ] && echo "" || echo "-clip${VCAC_CLIP}")
        NORM_TAG=$([ "$VCAC_NORMALIZE" = "True" ] && echo "-norm" || echo "")
        CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
        JOB_NAME="VCAC-tooluse-lam${LAM_TAG}${CLIP_TAG}${NORM_TAG}-${CDM_VARIANT}-${MODEL_NAME}-${CURRENT_TIME}"
        ENV_STR=$(_build_common_env "$JOB_NAME" "$VCAC_LAMBDA" "$VCAC_CLIP" "$VCAC_NORMALIZE" "$CDM_VARIANT" "$MODEL_NAME")
        _submit_job "$SCRIPT_PATH" "$JOB_NAME" "$ENV_STR"
    done; done; done; done; done
fi

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job (dataset=${DATASET_GROUP})"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job (dataset=${DATASET_GROUP})"
fi
echo "============================================================"
