#!/bin/bash
# =============================================================================
# DPO-2S vs Vanilla GRPO vs Vanilla SDPO — Fair Comparison
#
# Submits all needed jobs for a fair 3-way comparison:
#   1. Biology (sciknoweval): DPO-2S β=2.0 at 250 steps (baselines already ran)
#   2. Math (GSM8K + MATH500): Vanilla GRPO + DPO-2S β=2.0 at 150 steps
#      (Vanilla SDPO math baselines already ran in SDPO-Math-Baseline project)
#
# Config alignment:
#   Biology: rollout_n=8,  steps=250, lr=1e-5, batch=32
#   Math:    rollout_n=16, steps=150, lr=1e-5, batch=32  (matches existing SDPO baselines)
#
# Usage:
#   bash nebula_scripts/submit_dpo_comparison.sh [--dry-run] [--biology] [--math] [--all]
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
RUN_MATH=false
for arg in "$@"; do
    case "$arg" in
        --dry-run)  DRY_RUN=true ;;
        --biology)  RUN_BIOLOGY=true ;;
        --math)     RUN_MATH=true ;;
        --all)      RUN_BIOLOGY=true; RUN_MATH=true ;;
    esac
done
# Default: run all if no subset specified
if [ "$RUN_BIOLOGY" = false ] && [ "$RUN_MATH" = false ]; then
    RUN_BIOLOGY=true
    RUN_MATH=true
fi

if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 模式：只打印命令，不提交"
fi

# ── 固定超参 ─────────────────────────────────────────────────────────────
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
SEED="${SEED:-42}"
LR="${LR:-1e-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
DPO_BETA="${DPO_BETA:-2.0}"

# Branching defaults (shared by DPO-2S jobs)
TOP_K=10
TEACHER_TOP_K=100
ENTROPY_WINDOW=20
ENTROPY_SIGMA_START=2.0
ENTROPY_SIGMA_FLOOR=0.5
ENTROPY_SIGMA_STEP=0.5
TEACHER_CONTEXT_MODE=gt_marker
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
    local JN="$1" DS="$2" ROLLOUT="$3" STEPS="$4" PROJECT="$5"
    echo "--env=PROJECT_NAME=${PROJECT} --env=JOB_NAME=${JN} --env=DATASET=${DS} --env=MODEL_NAME=${MODEL_NAME} --env=LR=${LR} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT} --env=SEED=${SEED} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ} --env=VAL_BEFORE_TRAIN=True --env=SAVE_HF_ONLY=True --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=TOTAL_TRAINING_STEPS=${STEPS}"
}

# ── Helper: base env for SDPO (vanilla) ──────────────────────────────────
_sdpo_base_env() {
    local JN="$1" DS="$2" ROLLOUT="$3" STEPS="$4" PROJECT="$5"
    echo "--env=PROJECT_NAME=${PROJECT} --env=JOB_NAME=${JN} --env=DATASET=${DS} --env=MODEL_NAME=${MODEL_NAME} --env=LR=${LR} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT} --env=ALPHA=0.5 --env=DONT_REPROMPT_ON_SELF_SUCCESS=False --env=SEED=${SEED} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ} --env=VAL_BEFORE_TRAIN=True --env=SAVE_HF_ONLY=True --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=TOTAL_TRAINING_STEPS=${STEPS}"
}

# ── Helper: branching env ────────────────────────────────────────────────
_branching_env() {
    echo "--env=BRANCHING_ENABLED=True --env=TOP_K=${TOP_K} --env=TEACHER_TOP_K=${TEACHER_TOP_K} --env=ENTROPY_WINDOW=${ENTROPY_WINDOW} --env=ENTROPY_SIGMA_START=${ENTROPY_SIGMA_START} --env=ENTROPY_SIGMA_FLOOR=${ENTROPY_SIGMA_FLOOR} --env=ENTROPY_SIGMA_STEP=${ENTROPY_SIGMA_STEP} --env=TEACHER_CONTEXT_MODE=${TEACHER_CONTEXT_MODE} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR}"
}

# ── Helper: DPO-2S specific env ──────────────────────────────────────────
_dpo2s_env() {
    local STAGE1="$1" NTREES="$2" BETA="$3"
    echo "--env=TWO_STAGE=True --env=STAGE1_N=${STAGE1} --env=N_SPLITS=1 --env=N_TREES=${NTREES} --env=BRANCH_TOKEN_LOSS_MODE=suffix --env=TWO_STAGE_TEACHER_MODE=ref_or_marker --env=SUCCESS_THRESHOLD=0.3 --env=DPO_COEFFICIENT=${BETA}"
}

BETA_TAG=$(echo "$DPO_BETA" | tr '.' '_')

# =============================================================================
# PART 1: Biology (sciknoweval) — DPO-2S β=2.0 at 250 steps
# Vanilla GRPO & SDPO baselines already exist at 250 steps in "Baselines" project
# =============================================================================
if [ "$RUN_BIOLOGY" = true ]; then
    echo ""
    echo "============================================================"
    echo "PART 1: Biology (sciknoweval) — DPO-2S re-run at 250 steps"
    echo "============================================================"

    T=$(date +%Y%m%d_%H%M%S)
    JN="V3-DPO-2S-beta${BETA_TAG}-sciknoweval-biology-250steps-${T}"
    _submit_job "$GRPO_BRANCHING_SCRIPT" "$JN" \
        "$(_grpo_base_env "$JN" "sciknoweval/biology" 8 250 "TG-Branching-V3") $(_branching_env) $(_dpo2s_env 4 2 ${DPO_BETA})"
fi

# =============================================================================
# PART 2: Math — Vanilla GRPO + DPO-2S β=2.0
# Config: rollout_n=16, steps=150 (matches existing SDPO math baselines)
# SDPO baselines already ran in "SDPO-Math-Baseline" project
#
# DPO-2S with rollout_n=16: Stage1_N=8 exploration, Stage2 = 8 rollouts
#   = 4 trees × 2 leaves (N_TREES=4, N_SPLITS=1)
# =============================================================================
if [ "$RUN_MATH" = true ]; then
    MATH_PROJECT="DPO-Comparison-Math"

    for DS in math500 gsm8k; do
        DS_UPPER=$(echo "$DS" | tr '[:lower:]' '[:upper:]')
        echo ""
        echo "============================================================"
        echo "PART 2: ${DS_UPPER} — Vanilla GRPO + DPO-2S"
        echo "  rollout_n=16, steps=150"
        echo "============================================================"

        # Job: Vanilla GRPO
        T=$(date +%Y%m%d_%H%M%S)
        JN="GRPO-vanilla-${DS}-${T}"
        _submit_job "$GRPO_SCRIPT" "$JN" \
            "$(_grpo_base_env "$JN" "${DS}" 16 150 "${MATH_PROJECT}")"

        # Job: DPO-2S β=2.0
        T=$(date +%Y%m%d_%H%M%S)
        JN="DPO-2S-beta${BETA_TAG}-${DS}-${T}"
        _submit_job "$GRPO_BRANCHING_SCRIPT" "$JN" \
            "$(_grpo_base_env "$JN" "${DS}" 16 150 "${MATH_PROJECT}") $(_branching_env) $(_dpo2s_env 8 4 ${DPO_BETA})"
    done
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
echo "实验对比矩阵:"
echo "┌──────────────────┬──────────┬──────┬──────────┬─────────┐"
echo "│ Dataset          │ Method   │ n    │ Steps    │ Status  │"
echo "├──────────────────┼──────────┼──────┼──────────┼─────────┤"
echo "│ sciknoweval/bio  │ GRPO     │  8   │ 250      │ ✅ done │"
echo "│ sciknoweval/bio  │ SDPO     │  8   │ 250      │ ✅ done │"
echo "│ sciknoweval/bio  │ DPO-2S   │  8   │ 250      │ 🚀 new  │"
echo "├──────────────────┼──────────┼──────┼──────────┼─────────┤"
echo "│ MATH500          │ GRPO     │ 16   │ 150      │ 🚀 new  │"
echo "│ MATH500          │ SDPO     │ 16   │ 150      │ ✅ done │"
echo "│ MATH500          │ DPO-2S   │ 16   │ 150      │ 🚀 new  │"
echo "├──────────────────┼──────────┼──────┼──────────┼─────────┤"
echo "│ GSM8K            │ GRPO     │ 16   │ 150      │ 🚀 new  │"
echo "│ GSM8K            │ SDPO     │ 16   │ 150      │ ✅ done │"
echo "│ GSM8K            │ DPO-2S   │ 16   │ 150      │ 🚀 new  │"
echo "└──────────────────┴──────────┴──────┴──────────┴─────────┘"
echo "============================================================"
