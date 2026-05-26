#!/bin/bash
# =============================================================================
# Bayes-DR (Bayes Doubly-Robust) sweep — Nebula batch submitter
#
# Bayes-DR adds a per-token Rao-Blackwell baseline to the GRPO PG loss pathway:
#     b_t = σ( Σ_{s<t} δ_s + logit ḡ_GRPO(x) )
#     A_t = R − b_t
# where δ_t = log π_T^+(y_t|x,y_<t) − log π_T^-(y_t|x,y_<t) is the CDM
# dual-teacher sampled-token log-prob differential. Because b_t depends only
# on y_<t, A_t is an unbiased control variate regardless of δ calibration.
# Marker design: SDPO sibling-success ref reprompt (cdm_use_ref=True)
# + verdict-only markers (no GT leakage at train time).
#
# Usage:
#   bash nebula_scripts/submit_bayes_dr_sweep.sh [--dry-run]
#   bash nebula_scripts/submit_bayes_dr_sweep.sh --dataset sciknoweval [--dry-run]
#   bash nebula_scripts/submit_bayes_dr_sweep.sh --dataset lcb [--dry-run]
#   bash nebula_scripts/submit_bayes_dr_sweep.sh --dataset tooluse [--dry-run]
#   bash nebula_scripts/submit_bayes_dr_sweep.sh --dataset all [--dry-run]
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
PROJECT_NAME="${PROJECT_NAME:-BayesDR}"

# ── 命令行参数解析 ──────────────────────────────────────────────────────
DRY_RUN=false
DATASET_GROUP="sciknoweval"  # default: bio only, per current research focus

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

# every-10-step checkpoints for analysis
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"

# ── Bayes-DR sweep 超参 ────────────────────────────────────────────────
# Default: single point with no clip, standard prior floor, seq R anchor.
# Add ablation points by extending these arrays.
BDR_DELTA_CLIPS=("null")
BDR_PRIOR_FLOORS=("0.05")
BDR_USE_SEQ_ANCHORS=("True")
BDR_NORMALIZE_ADVS=("False")
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
    # $1: JOB_NAME, $2: BDR_DELTA_CLIP, $3: BDR_PRIOR_FLOOR,
    # $4: BDR_USE_SEQ_ANCHOR, $5: BDR_NORMALIZE_ADV,
    # $6: CDM_TEMPLATE_VARIANT, $7: MODEL_NAME, [$8: DATASET]
    local jn="$1" bdc="$2" bpf="$3" bsa="$4" bna="$5" cv="$6" mn="$7" ds="${8:-}"
    local base="--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${jn} --env=MODEL_NAME=${mn} --env=LR=${LR} --env=BDR_DELTA_CLIP=${bdc} --env=BDR_PRIOR_FLOOR=${bpf} --env=BDR_USE_SEQ_ANCHOR=${bsa} --env=BDR_NORMALIZE_ADV=${bna} --env=CDM_TEMPLATE_VARIANT=${cv} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=SEED=${SEED} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ}"
    if [ -n "$ds" ]; then
        echo "$base --env=DATASET=${ds}"
    else
        echo "$base"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# sciknoweval/biology
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$DATASET_GROUP" == "all" || "$DATASET_GROUP" == "sciknoweval" ]]; then
    SCRIPT_PATH="nebula_scripts/bayes_dr/bayes_dr_sciknoweval_parametric.sh"
    DATASETS=("sciknoweval/biology")
    for DATASET in "${DATASETS[@]}"; do
    for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for BDR_DELTA_CLIP in "${BDR_DELTA_CLIPS[@]}"; do
    for BDR_PRIOR_FLOOR in "${BDR_PRIOR_FLOORS[@]}"; do
    for BDR_USE_SEQ_ANCHOR in "${BDR_USE_SEQ_ANCHORS[@]}"; do
    for BDR_NORMALIZE_ADV in "${BDR_NORMALIZE_ADVS[@]}"; do
    for CDM_VARIANT in "${CDM_TEMPLATE_VARIANTS[@]}"; do
        DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
        CLIP_TAG=$([ "$BDR_DELTA_CLIP" = "null" ] && echo "" || echo "-clip${BDR_DELTA_CLIP}")
        FLOOR_TAG=$(echo "$BDR_PRIOR_FLOOR" | tr '.' '_')
        ANCHOR_TAG=$([ "$BDR_USE_SEQ_ANCHOR" = "True" ] && echo "Ranchor" || echo "signAnchor")
        NORM_TAG=$([ "$BDR_NORMALIZE_ADV" = "True" ] && echo "-norm" || echo "")
        CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
        JOB_NAME="BayesDR-${DATASET_SHORT}-${ANCHOR_TAG}-floor${FLOOR_TAG}${CLIP_TAG}${NORM_TAG}-${CDM_VARIANT}-${MODEL_NAME}-${CURRENT_TIME}"
        ENV_STR=$(_build_common_env "$JOB_NAME" "$BDR_DELTA_CLIP" "$BDR_PRIOR_FLOOR" "$BDR_USE_SEQ_ANCHOR" "$BDR_NORMALIZE_ADV" "$CDM_VARIANT" "$MODEL_NAME" "$DATASET")
        _submit_job "$SCRIPT_PATH" "$JOB_NAME" "$ENV_STR"
    done; done; done; done; done; done; done
fi

# ─────────────────────────────────────────────────────────────────────────────
# lcb_v6 (placeholder — parametric script TBD; uncomment after adding it)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$DATASET_GROUP" == "lcb" || "$DATASET_GROUP" == "all" ]]; then
    SCRIPT_PATH="nebula_scripts/bayes_dr/bayes_dr_lcb_parametric.sh"
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "WARNING: ${SCRIPT_PATH} not yet created — skipping lcb sweep."
    else
        for MODEL_NAME in "${MODEL_NAMES[@]}"; do
        for BDR_DELTA_CLIP in "${BDR_DELTA_CLIPS[@]}"; do
        for BDR_PRIOR_FLOOR in "${BDR_PRIOR_FLOORS[@]}"; do
        for BDR_USE_SEQ_ANCHOR in "${BDR_USE_SEQ_ANCHORS[@]}"; do
        for BDR_NORMALIZE_ADV in "${BDR_NORMALIZE_ADVS[@]}"; do
        for CDM_VARIANT in "${CDM_TEMPLATE_VARIANTS[@]}"; do
            CLIP_TAG=$([ "$BDR_DELTA_CLIP" = "null" ] && echo "" || echo "-clip${BDR_DELTA_CLIP}")
            FLOOR_TAG=$(echo "$BDR_PRIOR_FLOOR" | tr '.' '_')
            ANCHOR_TAG=$([ "$BDR_USE_SEQ_ANCHOR" = "True" ] && echo "Ranchor" || echo "signAnchor")
            NORM_TAG=$([ "$BDR_NORMALIZE_ADV" = "True" ] && echo "-norm" || echo "")
            CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
            JOB_NAME="BayesDR-lcb_v6-${ANCHOR_TAG}-floor${FLOOR_TAG}${CLIP_TAG}${NORM_TAG}-${CDM_VARIANT}-${MODEL_NAME}-${CURRENT_TIME}"
            ENV_STR=$(_build_common_env "$JOB_NAME" "$BDR_DELTA_CLIP" "$BDR_PRIOR_FLOOR" "$BDR_USE_SEQ_ANCHOR" "$BDR_NORMALIZE_ADV" "$CDM_VARIANT" "$MODEL_NAME")
            _submit_job "$SCRIPT_PATH" "$JOB_NAME" "$ENV_STR"
        done; done; done; done; done; done
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# tooluse (placeholder — parametric script TBD; uncomment after adding it)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$DATASET_GROUP" == "tooluse" || "$DATASET_GROUP" == "all" ]]; then
    SCRIPT_PATH="nebula_scripts/bayes_dr/bayes_dr_tooluse_parametric.sh"
    if [ ! -f "$SCRIPT_PATH" ]; then
        echo "WARNING: ${SCRIPT_PATH} not yet created — skipping tooluse sweep."
    else
        for MODEL_NAME in "${MODEL_NAMES[@]}"; do
        for BDR_DELTA_CLIP in "${BDR_DELTA_CLIPS[@]}"; do
        for BDR_PRIOR_FLOOR in "${BDR_PRIOR_FLOORS[@]}"; do
        for BDR_USE_SEQ_ANCHOR in "${BDR_USE_SEQ_ANCHORS[@]}"; do
        for BDR_NORMALIZE_ADV in "${BDR_NORMALIZE_ADVS[@]}"; do
        for CDM_VARIANT in "${CDM_TEMPLATE_VARIANTS[@]}"; do
            CLIP_TAG=$([ "$BDR_DELTA_CLIP" = "null" ] && echo "" || echo "-clip${BDR_DELTA_CLIP}")
            FLOOR_TAG=$(echo "$BDR_PRIOR_FLOOR" | tr '.' '_')
            ANCHOR_TAG=$([ "$BDR_USE_SEQ_ANCHOR" = "True" ] && echo "Ranchor" || echo "signAnchor")
            NORM_TAG=$([ "$BDR_NORMALIZE_ADV" = "True" ] && echo "-norm" || echo "")
            CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
            JOB_NAME="BayesDR-tooluse-${ANCHOR_TAG}-floor${FLOOR_TAG}${CLIP_TAG}${NORM_TAG}-${CDM_VARIANT}-${MODEL_NAME}-${CURRENT_TIME}"
            ENV_STR=$(_build_common_env "$JOB_NAME" "$BDR_DELTA_CLIP" "$BDR_PRIOR_FLOOR" "$BDR_USE_SEQ_ANCHOR" "$BDR_NORMALIZE_ADV" "$CDM_VARIANT" "$MODEL_NAME")
            _submit_job "$SCRIPT_PATH" "$JOB_NAME" "$ENV_STR"
        done; done; done; done; done; done
    fi
fi

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job (dataset=${DATASET_GROUP})"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job (dataset=${DATASET_GROUP})"
fi
echo "============================================================"
