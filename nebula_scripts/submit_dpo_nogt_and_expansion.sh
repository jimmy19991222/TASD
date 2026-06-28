#!/bin/bash
# =============================================================================
# DPO-2S (no GT) vs Vanilla GRPO vs Vanilla SDPO — Qwen3-4B × 4 Sciknoweval domains
#
# All 12 jobs in one SwanLab project: DPO-Comparison-4B
#
# Key change from previous round:
#   TEACHER_CONTEXT_MODE=marker  (was gt_marker — removes GT leakage in DPO-2S)
#   MODEL_NAME=Qwen3-4B         (was 8B — avoids GSM8K saturation)
#
# Config: rollout_n=8, steps=250, lr=1e-5, batch=32
#
# Usage:
#   bash nebula_scripts/submit_dpo_nogt_and_expansion.sh [--dry-run] [--biology] [--chemistry] [--physics] [--material] [--all]
#
# Pre-flight checklist:
#   1. Qwen3-4B uploaded to OSS: base_models/Qwen3-4B
#   2. Parquets exist on OSS for each domain:
#      - datasets/sciknoweval/biology/{train,test}.parquet    (confirmed)
#      - datasets/sciknoweval/chemistry/{train,test}.parquet  (likely exists from V2)
#      - datasets/sciknoweval/physics/{train,test}.parquet    (check!)
#      - datasets/sciknoweval/material/{train,test}.parquet   (check!)
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

GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"

DRY_RUN=false
RUN_BIOLOGY=false
RUN_CHEMISTRY=false
RUN_PHYSICS=false
RUN_MATERIAL=false
RUN_MATH=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)    DRY_RUN=true ;;
        --biology)    RUN_BIOLOGY=true ;;
        --chemistry)  RUN_CHEMISTRY=true ;;
        --physics)    RUN_PHYSICS=true ;;
        --material)   RUN_MATERIAL=true ;;
        --math)       RUN_MATH=true ;;
        --all)        RUN_BIOLOGY=true; RUN_CHEMISTRY=true; RUN_PHYSICS=true; RUN_MATERIAL=true; RUN_MATH=true ;;
    esac
done
if [ "$RUN_BIOLOGY" = false ] && [ "$RUN_CHEMISTRY" = false ] && [ "$RUN_PHYSICS" = false ] && [ "$RUN_MATERIAL" = false ] && [ "$RUN_MATH" = false ]; then
    RUN_BIOLOGY=true; RUN_CHEMISTRY=true; RUN_PHYSICS=true; RUN_MATERIAL=true; RUN_MATH=true
fi

if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 模式：只打印命令，不提交"
fi

# ── 固定超参 ─────────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"
SEED="${SEED:-42}"
LR="${LR:-1e-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
DPO_BETA="${DPO_BETA:-2.0}"
ROLLOUT_N=8
TOTAL_TRAINING_STEPS=250
PROJECT_NAME="DPO-Comparison-4B"

# Branching defaults (shared by DPO-2S jobs)
TOP_K=10
TEACHER_TOP_K=100
ENTROPY_WINDOW=20
ENTROPY_SIGMA_START=2.0
ENTROPY_SIGMA_FLOOR=0.5
ENTROPY_SIGMA_STEP=0.5
TEACHER_CONTEXT_MODE=marker    # ← KEY CHANGE: was gt_marker
ADV_STD_FLOOR=0.05

TEST_FREQ=10
SAVE_FREQ=10

TOTAL=0
SUBMITTED=0

# ── Scripts ──────────────────────────────────────────────────────────────
GRPO_SCRIPT="nebula_scripts/grpo/grpo_sciknoweval_parametric.sh"
GRPO_BRANCHING_SCRIPT="nebula_scripts/grpo/grpo_branching_sciknoweval_parametric.sh"
SDPO_SCRIPT="nebula_scripts/sdpo/sdpo_sciknoweval_parametric.sh"

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

# ── Helper: base env for GRPO (vanilla) ──────────────────────────────────
_grpo_base_env() {
    local JN="$1" DS="$2"
    echo "--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JN} --env=DATASET=${DS} --env=MODEL_NAME=${MODEL_NAME} --env=LR=${LR} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=SEED=${SEED} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ} --env=VAL_BEFORE_TRAIN=True --env=SAVE_HF_ONLY=True --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}"
}

# ── Helper: base env for SDPO (vanilla) ──────────────────────────────────
_sdpo_base_env() {
    local JN="$1" DS="$2"
    echo "--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JN} --env=DATASET=${DS} --env=MODEL_NAME=${MODEL_NAME} --env=LR=${LR} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=ALPHA=0.5 --env=DONT_REPROMPT_ON_SELF_SUCCESS=False --env=SEED=${SEED} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ} --env=VAL_BEFORE_TRAIN=True --env=SAVE_HF_ONLY=True --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}"
}

# ── Helper: branching env ────────────────────────────────────────────────
_branching_env() {
    echo "--env=BRANCHING_ENABLED=True --env=TOP_K=${TOP_K} --env=TEACHER_TOP_K=${TEACHER_TOP_K} --env=ENTROPY_WINDOW=${ENTROPY_WINDOW} --env=ENTROPY_SIGMA_START=${ENTROPY_SIGMA_START} --env=ENTROPY_SIGMA_FLOOR=${ENTROPY_SIGMA_FLOOR} --env=ENTROPY_SIGMA_STEP=${ENTROPY_SIGMA_STEP} --env=TEACHER_CONTEXT_MODE=${TEACHER_CONTEXT_MODE} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR}"
}

# ── Helper: DPO-2S specific env ──────────────────────────────────────────
# Usage: _dpo2s_env [stage1_n] [n_trees]
_dpo2s_env() {
    local S1="${1:-4}" NT="${2:-2}"
    echo "--env=TWO_STAGE=True --env=STAGE1_N=${S1} --env=N_SPLITS=1 --env=N_TREES=${NT} --env=BRANCH_TOKEN_LOSS_MODE=suffix --env=TWO_STAGE_TEACHER_MODE=ref_or_marker --env=SUCCESS_THRESHOLD=0.3 --env=DPO_COEFFICIENT=${DPO_BETA}"
}

BETA_TAG=$(echo "$DPO_BETA" | tr '.' '_')

# ── Submit 3-way comparison for a single domain ─────────────────────────
_submit_domain() {
    local DOMAIN="$1"
    local DS="sciknoweval/${DOMAIN}"
    local DOMAIN_UPPER=$(echo "$DOMAIN" | tr '[:lower:]' '[:upper:]' | head -c1)$(echo "$DOMAIN" | tail -c+2)

    echo ""
    echo "============================================================"
    echo "${DOMAIN_UPPER}: GRPO + SDPO + DPO-2S (marker, β=${DPO_BETA})"
    echo "  model=${MODEL_NAME}  n=${ROLLOUT_N}  steps=${TOTAL_TRAINING_STEPS}"
    echo "============================================================"

    # Job: Vanilla GRPO
    T=$(date +%Y%m%d_%H%M%S)
    JN="GRPO-4B-${DOMAIN}-${T}"
    _submit_job "$GRPO_SCRIPT" "$JN" \
        "$(_grpo_base_env "$JN" "${DS}")"

    # Job: Vanilla SDPO
    T=$(date +%Y%m%d_%H%M%S)
    JN="SDPO-4B-${DOMAIN}-${T}"
    _submit_job "$SDPO_SCRIPT" "$JN" \
        "$(_sdpo_base_env "$JN" "${DS}")"

    # Job: DPO-2S (marker, no GT)
    T=$(date +%Y%m%d_%H%M%S)
    JN="DPO-2S-4B-marker-beta${BETA_TAG}-${DOMAIN}-${T}"
    _submit_job "$GRPO_BRANCHING_SCRIPT" "$JN" \
        "$(_grpo_base_env "$JN" "${DS}") $(_branching_env) $(_dpo2s_env)"
}

# ── Submit 3-way comparison for a math dataset ────────────────────────────
# Math uses n=16, steps=150, DPO-2S stage1=8 + 4 trees × 2 leaves
_submit_math_domain() {
    local DS="$1"
    local DS_UPPER=$(echo "$DS" | tr '[:lower:]' '[:upper:]')
    local MATH_N=16
    local MATH_STEPS=150

    echo ""
    echo "============================================================"
    echo "${DS_UPPER}: GRPO + SDPO + DPO-2S (marker, β=${DPO_BETA})"
    echo "  model=${MODEL_NAME}  n=${MATH_N}  steps=${MATH_STEPS}"
    echo "============================================================"

    # Job: Vanilla GRPO
    T=$(date +%Y%m%d_%H%M%S)
    JN="GRPO-4B-${DS}-${T}"
    _submit_job "$GRPO_SCRIPT" "$JN" \
        "--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JN} --env=DATASET=${DS} --env=MODEL_NAME=${MODEL_NAME} --env=LR=${LR} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${MATH_N} --env=SEED=${SEED} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ} --env=VAL_BEFORE_TRAIN=True --env=SAVE_HF_ONLY=True --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=TOTAL_TRAINING_STEPS=${MATH_STEPS}"

    # Job: Vanilla SDPO
    T=$(date +%Y%m%d_%H%M%S)
    JN="SDPO-4B-${DS}-${T}"
    _submit_job "$SDPO_SCRIPT" "$JN" \
        "--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JN} --env=DATASET=${DS} --env=MODEL_NAME=${MODEL_NAME} --env=LR=${LR} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${MATH_N} --env=ALPHA=0.5 --env=DONT_REPROMPT_ON_SELF_SUCCESS=False --env=SEED=${SEED} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ} --env=VAL_BEFORE_TRAIN=True --env=SAVE_HF_ONLY=True --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=TOTAL_TRAINING_STEPS=${MATH_STEPS}"

    # Job: DPO-2S (marker, no GT) — n=16: stage1=8, 4 trees × 2 leaves
    T=$(date +%Y%m%d_%H%M%S)
    JN="DPO-2S-4B-marker-beta${BETA_TAG}-${DS}-${T}"
    _submit_job "$GRPO_BRANCHING_SCRIPT" "$JN" \
        "--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JN} --env=DATASET=${DS} --env=MODEL_NAME=${MODEL_NAME} --env=LR=${LR} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${MATH_N} --env=SEED=${SEED} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ} --env=VAL_BEFORE_TRAIN=True --env=SAVE_HF_ONLY=True --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=TOTAL_TRAINING_STEPS=${MATH_STEPS} $(_branching_env) $(_dpo2s_env 8 4)"
}

# =============================================================================
# Submit all selected domains
# =============================================================================
[ "$RUN_BIOLOGY" = true ]   && _submit_domain "biology"
[ "$RUN_CHEMISTRY" = true ] && _submit_domain "chemistry"
[ "$RUN_PHYSICS" = true ]   && _submit_domain "physics"
[ "$RUN_MATERIAL" = true ]  && _submit_domain "material"

# Math datasets (n=16, steps=150)
if [ "$RUN_MATH" = true ]; then
    _submit_math_domain "math500"
    _submit_math_domain "gsm8k"
fi

# =============================================================================
echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo ""
echo "实验对比矩阵 (model=${MODEL_NAME}, project=${PROJECT_NAME}):"
echo "┌────────────────────────┬──────────┬──────┬──────────┐"
echo "│ Dataset                │ Method   │ n    │ Steps    │"
echo "├────────────────────────┼──────────┼──────┼──────────┤"
echo "│ sciknoweval/biology    │ GRPO     │  8   │ 250      │"
echo "│ sciknoweval/biology    │ SDPO     │  8   │ 250      │"
echo "│ sciknoweval/biology    │ DPO-2S   │  8   │ 250      │"
echo "├────────────────────────┼──────────┼──────┼──────────┤"
echo "│ sciknoweval/chemistry  │ GRPO     │  8   │ 250      │"
echo "│ sciknoweval/chemistry  │ SDPO     │  8   │ 250      │"
echo "│ sciknoweval/chemistry  │ DPO-2S   │  8   │ 250      │"
echo "├────────────────────────┼──────────┼──────┼──────────┤"
echo "│ sciknoweval/physics    │ GRPO     │  8   │ 250      │"
echo "│ sciknoweval/physics    │ SDPO     │  8   │ 250      │"
echo "│ sciknoweval/physics    │ DPO-2S   │  8   │ 250      │"
echo "├────────────────────────┼──────────┼──────┼──────────┤"
echo "│ sciknoweval/material   │ GRPO     │  8   │ 250      │"
echo "│ sciknoweval/material   │ SDPO     │  8   │ 250      │"
echo "│ sciknoweval/material   │ DPO-2S   │  8   │ 250      │"
echo "├────────────────────────┼──────────┼──────┼──────────┤"
echo "│ MATH500                │ GRPO     │ 16   │ 150      │"
echo "│ MATH500                │ SDPO     │ 16   │ 150      │"
echo "│ MATH500                │ DPO-2S   │ 16   │ 150      │"
echo "├────────────────────────┼──────────┼──────┼──────────┤"
echo "│ GSM8K                  │ GRPO     │ 16   │ 150      │"
echo "│ GSM8K                  │ SDPO     │ 16   │ 150      │"
echo "│ GSM8K                  │ DPO-2S   │ 16   │ 150      │"
echo "└────────────────────────┴──────────┴──────┴──────────┘"
echo ""
echo "DPO-2S config: TEACHER_CONTEXT_MODE=marker (no GT), TWO_STAGE_TEACHER_MODE=ref_or_marker"
echo "============================================================"
