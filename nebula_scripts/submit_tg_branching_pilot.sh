#!/bin/bash
# =============================================================================
# Teacher-Guided Branching GRPO 首轮 Pilot — Nebula 批量提交
#
# Submits BOTH a vanilla GRPO baseline (branching disabled) AND the
# teacher-guided branching variants on sciknoweval. Defaults are chosen so
# the pilot is ~6 GPU-hours per run × 4 runs ≈ a single 4-GPU H20 day.
#
# Usage:
#   bash nebula_scripts/submit_tg_branching_pilot.sh [--dry-run]
#   bash nebula_scripts/submit_tg_branching_pilot.sh --variant baseline [--dry-run]
#   bash nebula_scripts/submit_tg_branching_pilot.sh --variant branching --loss-mode mask [--dry-run]
#   bash nebula_scripts/submit_tg_branching_pilot.sh --variant sdpo [--loss-mode mask] [--dry-run]
#
# Variants:
#   baseline   : rollout.branching.enabled=False, vanilla GRPO (control).
#   branching  : full pipeline (GRPO loss) with branch_token_loss_mode in
#                {all, mask, only}. Use --loss-mode all|mask|only|all_three
#                (default all_three).
#   sdpo       : full pipeline (SDPO loss) with branch_token_loss_mode in
#                {all, mask, only}. Aligns
#                actor.self_distillation.teacher_context_mode with
#                rollout.branching.teacher_context_mode so the on-policy
#                teacher distribution is byte-identical between rollout and
#                training time. Use --loss-mode same as above.
#   all        : baseline + branching (does NOT include sdpo by default —
#                that's a heavier sweep; pass --variant sdpo explicitly).
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
PROJECT_NAME="${PROJECT_NAME:-TG-Branching-Pilot}"

GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"

# Cadence env defaults forwarded into every job
TEST_FREQ="${TEST_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-10}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
SAVE_HF_ONLY="${SAVE_HF_ONLY:-True}"

# ── 参数解析 ──────────────────────────────────────────────────────────
DRY_RUN=false
VARIANT="all"        # all | baseline | branching
LOSS_MODE="all_three" # all | mask | only | all_three (only for branching)

for ((i=1; i<=$#; i++)); do
    arg="${!i}"
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --variant=*) VARIANT="${arg#*=}" ;;
        --variant) i=$((i+1)); VARIANT="${!i}" ;;
        --loss-mode=*) LOSS_MODE="${arg#*=}" ;;
        --loss-mode) i=$((i+1)); LOSS_MODE="${!i}" ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 超参配置
# =============================================================================
MODEL_NAMES=("Qwen3-8B")
DATASETS=("sciknoweval/biology")
SEED="42"
TRAIN_BATCH_SIZE="32"
ROLLOUT_N="8"   # MUST equal 2 ** branching.n_splits when branching is enabled
LRS=("1e-5")
MINI_BATCH_SIZES=("32")

# SDPO-specific defaults (only used by the sdpo variant)
SDPO_ALPHA="${SDPO_ALPHA:-0.5}"
SDPO_DONT_REPROMPT_ON_SELF_SUCCESS="${SDPO_DONT_REPROMPT_ON_SELF_SUCCESS:-False}"

# Branching defaults
N_SPLITS="3"
TOP_K="50"
ENTROPY_WINDOW="20"
ENTROPY_SIGMA_START="2.0"
ENTROPY_SIGMA_FLOOR="0.5"
ENTROPY_SIGMA_STEP="0.5"
TEACHER_CONTEXT_MODE="${TEACHER_CONTEXT_MODE:-gt_marker}"
ADV_STD_FLOOR="0.05"

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

# Common --env block forwarded to every job.
_common_env() {
    local JOB_NAME="$1"
    local DATASET="$2"
    local MODEL_NAME="$3"
    local LR="$4"
    local MINI_BATCH_SIZE="$5"
    echo "--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL_NAME=${MODEL_NAME} --env=LR=${LR} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=SEED=${SEED} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ} --env=VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN} --env=SAVE_HF_ONLY=${SAVE_HF_ONLY} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Variant: baseline (vanilla GRPO control). Reuses the existing parametric.
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$VARIANT" == "all" || "$VARIANT" == "baseline" ]]; then
    SCRIPT_PATH="nebula_scripts/grpo/grpo_sciknoweval_parametric.sh"
    for DATASET in "${DATASETS[@]}"; do
    for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for LR in "${LRS[@]}"; do
    for MINI_BATCH_SIZE in "${MINI_BATCH_SIZES[@]}"; do
        DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
        LR_TAG=$(echo "$LR" | tr '-' '_')
        CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
        JOB_NAME="GRPO-${DATASET_SHORT}-vanilla-mbs${MINI_BATCH_SIZE}-lr${LR_TAG}-${MODEL_NAME}-${CURRENT_TIME}"
        _submit_job "$SCRIPT_PATH" "$JOB_NAME" \
            "$(_common_env "$JOB_NAME" "$DATASET" "$MODEL_NAME" "$LR" "$MINI_BATCH_SIZE")"
    done; done; done; done
fi

# ─────────────────────────────────────────────────────────────────────────────
# Variant: branching (1, 2, or 3 loss-mode flavours)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$VARIANT" == "all" || "$VARIANT" == "branching" ]]; then
    if [[ "$LOSS_MODE" == "all_three" ]]; then
        LOSS_MODES=("all" "mask" "only")
    else
        LOSS_MODES=("$LOSS_MODE")
    fi
    SCRIPT_PATH="nebula_scripts/grpo/grpo_branching_sciknoweval_parametric.sh"
    for DATASET in "${DATASETS[@]}"; do
    for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for LR in "${LRS[@]}"; do
    for MINI_BATCH_SIZE in "${MINI_BATCH_SIZES[@]}"; do
    for BTM in "${LOSS_MODES[@]}"; do
        DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
        LR_TAG=$(echo "$LR" | tr '-' '_')
        CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
        JOB_NAME="TGB-${DATASET_SHORT}-${TEACHER_CONTEXT_MODE}-btm${BTM}-mbs${MINI_BATCH_SIZE}-lr${LR_TAG}-${MODEL_NAME}-${CURRENT_TIME}"
        _submit_job "$SCRIPT_PATH" "$JOB_NAME" \
            "$(_common_env "$JOB_NAME" "$DATASET" "$MODEL_NAME" "$LR" "$MINI_BATCH_SIZE") --env=BRANCHING_ENABLED=True --env=N_SPLITS=${N_SPLITS} --env=TOP_K=${TOP_K} --env=ENTROPY_WINDOW=${ENTROPY_WINDOW} --env=ENTROPY_SIGMA_START=${ENTROPY_SIGMA_START} --env=ENTROPY_SIGMA_FLOOR=${ENTROPY_SIGMA_FLOOR} --env=ENTROPY_SIGMA_STEP=${ENTROPY_SIGMA_STEP} --env=TEACHER_CONTEXT_MODE=${TEACHER_CONTEXT_MODE} --env=BRANCH_TOKEN_LOSS_MODE=${BTM} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR}"
    done; done; done; done; done
fi

# ─────────────────────────────────────────────────────────────────────────────
# Variant: sdpo (SDPO loss + branching rollout)
# Run only when explicitly requested (NOT included in --variant=all).
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$VARIANT" == "sdpo" ]]; then
    if [[ "$LOSS_MODE" == "all_three" ]]; then
        LOSS_MODES=("all" "mask" "only")
    else
        LOSS_MODES=("$LOSS_MODE")
    fi
    SCRIPT_PATH="nebula_scripts/sdpo/sdpo_branching_sciknoweval_parametric.sh"
    for DATASET in "${DATASETS[@]}"; do
    for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for LR in "${LRS[@]}"; do
    for BTM in "${LOSS_MODES[@]}"; do
        DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
        LR_TAG=$(echo "$LR" | tr '-' '_')
        CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
        JOB_NAME="TGB-SDPO-${DATASET_SHORT}-${TEACHER_CONTEXT_MODE}-alpha${SDPO_ALPHA}-btm${BTM}-lr${LR_TAG}-${MODEL_NAME}-${CURRENT_TIME}"
        # SDPO-specific env: ALPHA + DONT_REPROMPT_ON_SELF_SUCCESS. We use
        # the GRPO-side _common_env then append SDPO and branching extras.
        # Note: MINI_BATCH_SIZE is not threaded through SDPO parametric (it
        # hardcodes ppo_mini_batch_size=32 for stability) but we still pass
        # it for the env-block consistency.
        _submit_job "$SCRIPT_PATH" "$JOB_NAME" \
            "$(_common_env "$JOB_NAME" "$DATASET" "$MODEL_NAME" "$LR" "32") --env=ALPHA=${SDPO_ALPHA} --env=DONT_REPROMPT_ON_SELF_SUCCESS=${SDPO_DONT_REPROMPT_ON_SELF_SUCCESS} --env=BRANCHING_ENABLED=True --env=N_SPLITS=${N_SPLITS} --env=TOP_K=${TOP_K} --env=ENTROPY_WINDOW=${ENTROPY_WINDOW} --env=ENTROPY_SIGMA_START=${ENTROPY_SIGMA_START} --env=ENTROPY_SIGMA_FLOOR=${ENTROPY_SIGMA_FLOOR} --env=ENTROPY_SIGMA_STEP=${ENTROPY_SIGMA_STEP} --env=TEACHER_CONTEXT_MODE=${TEACHER_CONTEXT_MODE} --env=BRANCH_TOKEN_LOSS_MODE=${BTM} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR}"
    done; done; done; done
fi

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job (variant=${VARIANT}, loss_mode=${LOSS_MODE})"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job (variant=${VARIANT}, loss_mode=${LOSS_MODE})"
fi
echo "============================================================"
