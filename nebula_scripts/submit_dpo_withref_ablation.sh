#!/bin/bash
# =============================================================================
# With-Ref DPO Ablation — Qwen3-4B × 4 Sciknoweval domains
#
# Validates that adding π_ref to DPO loss fixes the entropy collapse observed
# in no-ref DPO-2S (Chemistry: H=0.023, |y|=2.0).
#
# Experiment matrix (10 jobs):
#   Batch 1 — With-Ref DPO (β=2.0)              × 4 domains = 4 jobs
#   Batch 2 — With-Ref DPO + Entropy Reg (β=2.0) × 4 domains = 4 jobs
#   Batch 3 — With-Ref β tuning (β=1.0, β=0.5)   × chemistry = 2 jobs
#
# All in SwanLab project: DPO-Comparison-4B
# Same config as Round 2 baseline (marker, no GT) for fair comparison.
#
# Usage:
#   bash nebula_scripts/submit_dpo_withref_ablation.sh [--dry-run]
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
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 模式：只打印命令，不提交"
fi

# ── 固定超参 ─────────────────────────────────────────────────────────────
MODEL_NAME="Qwen3-4B"
SEED=42
LR="1e-5"
TRAIN_BATCH_SIZE=32
MINI_BATCH_SIZE=32
ROLLOUT_N=8
TOTAL_TRAINING_STEPS=250
PROJECT_NAME="DPO-Comparison-4B"

# Branching defaults
TOP_K=10
TEACHER_TOP_K=100
ENTROPY_WINDOW=20
ENTROPY_SIGMA_START=2.0
ENTROPY_SIGMA_FLOOR=0.5
ENTROPY_SIGMA_STEP=0.5
TEACHER_CONTEXT_MODE=marker
ADV_STD_FLOOR=0.05

TEST_FREQ=10
SAVE_FREQ=10

TOTAL=0
SUBMITTED=0

GRPO_BRANCHING_SCRIPT="nebula_scripts/grpo/grpo_branching_sciknoweval_parametric.sh"

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

# ── Helper: base env ─────────────────────────────────────────────────────
_base_env() {
    local JN="$1" DS="$2"
    echo "--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JN} --env=DATASET=${DS} --env=MODEL_NAME=${MODEL_NAME} --env=LR=${LR} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=SEED=${SEED} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ} --env=VAL_BEFORE_TRAIN=True --env=SAVE_HF_ONLY=True --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}"
}

# ── Helper: branching env ────────────────────────────────────────────────
_branching_env() {
    echo "--env=BRANCHING_ENABLED=True --env=TOP_K=${TOP_K} --env=TEACHER_TOP_K=${TEACHER_TOP_K} --env=ENTROPY_WINDOW=${ENTROPY_WINDOW} --env=ENTROPY_SIGMA_START=${ENTROPY_SIGMA_START} --env=ENTROPY_SIGMA_FLOOR=${ENTROPY_SIGMA_FLOOR} --env=ENTROPY_SIGMA_STEP=${ENTROPY_SIGMA_STEP} --env=TEACHER_CONTEXT_MODE=${TEACHER_CONTEXT_MODE} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR}"
}

# ── Helper: DPO-2S with-ref env ──────────────────────────────────────────
# Usage: _dpo2s_withref_env <beta> <entropy_coeff>
_dpo2s_withref_env() {
    local BETA="$1" ENT_COEFF="$2"
    echo "--env=TWO_STAGE=True --env=STAGE1_N=4 --env=N_SPLITS=1 --env=N_TREES=2 --env=BRANCH_TOKEN_LOSS_MODE=suffix --env=TWO_STAGE_TEACHER_MODE=ref_or_marker --env=SUCCESS_THRESHOLD=0.3 --env=DPO_COEFFICIENT=${BETA} --env=DPO_USE_REF=True --env=ENTROPY_COEFF=${ENT_COEFF}"
}

DOMAINS=("biology" "chemistry" "physics" "material")

# =============================================================================
# Batch 1: With-Ref DPO (β=2.0, no entropy reg) × 4 domains
# =============================================================================
echo ""
echo "============================================================"
echo "Batch 1: With-Ref DPO (β=2.0) × 4 domains"
echo "============================================================"

for DOMAIN in "${DOMAINS[@]}"; do
    T=$(date +%Y%m%d_%H%M%S)
    JN="DPO-2S-4B-withref-beta2_0-${DOMAIN}-${T}"
    _submit_job "$GRPO_BRANCHING_SCRIPT" "$JN" \
        "$(_base_env "$JN" "sciknoweval/${DOMAIN}") $(_branching_env) $(_dpo2s_withref_env 2.0 0)"
done

# =============================================================================
# Batch 2: With-Ref DPO + Entropy Reg (β=2.0, ent=0.01) × 4 domains
# =============================================================================
echo ""
echo "============================================================"
echo "Batch 2: With-Ref DPO + Entropy Reg (β=2.0, ent=0.01) × 4 domains"
echo "============================================================"

for DOMAIN in "${DOMAINS[@]}"; do
    T=$(date +%Y%m%d_%H%M%S)
    JN="DPO-2S-4B-withref-ent0_01-beta2_0-${DOMAIN}-${T}"
    _submit_job "$GRPO_BRANCHING_SCRIPT" "$JN" \
        "$(_base_env "$JN" "sciknoweval/${DOMAIN}") $(_branching_env) $(_dpo2s_withref_env 2.0 0.01)"
done

# =============================================================================
# Batch 3: β tuning on chemistry (with-ref, β=1.0 and β=0.5)
# =============================================================================
echo ""
echo "============================================================"
echo "Batch 3: β tuning on chemistry (with-ref, β=1.0 / β=0.5)"
echo "============================================================"

for BETA in 1.0 0.5; do
    BETA_TAG=$(echo "$BETA" | tr '.' '_')
    T=$(date +%Y%m%d_%H%M%S)
    JN="DPO-2S-4B-withref-beta${BETA_TAG}-chemistry-${T}"
    _submit_job "$GRPO_BRANCHING_SCRIPT" "$JN" \
        "$(_base_env "$JN" "sciknoweval/chemistry") $(_branching_env) $(_dpo2s_withref_env ${BETA} 0)"
done

# =============================================================================
echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo ""
echo "实验矩阵 (model=${MODEL_NAME}, project=${PROJECT_NAME}):"
echo "┌─────────────────────────┬──────────────────────────────┬──────┬───────┐"
echo "│ Dataset                 │ Variant                      │ β    │ Ent   │"
echo "├─────────────────────────┼──────────────────────────────┼──────┼───────┤"
echo "│ sciknoweval/biology     │ With-Ref DPO                 │ 2.0  │ 0     │"
echo "│ sciknoweval/chemistry   │ With-Ref DPO                 │ 2.0  │ 0     │"
echo "│ sciknoweval/physics     │ With-Ref DPO                 │ 2.0  │ 0     │"
echo "│ sciknoweval/material    │ With-Ref DPO                 │ 2.0  │ 0     │"
echo "├─────────────────────────┼──────────────────────────────┼──────┼───────┤"
echo "│ sciknoweval/biology     │ With-Ref DPO + Entropy Reg   │ 2.0  │ 0.01  │"
echo "│ sciknoweval/chemistry   │ With-Ref DPO + Entropy Reg   │ 2.0  │ 0.01  │"
echo "│ sciknoweval/physics     │ With-Ref DPO + Entropy Reg   │ 2.0  │ 0.01  │"
echo "│ sciknoweval/material    │ With-Ref DPO + Entropy Reg   │ 2.0  │ 0.01  │"
echo "├─────────────────────────┼──────────────────────────────┼──────┼───────┤"
echo "│ sciknoweval/chemistry   │ With-Ref DPO (β tuning)      │ 1.0  │ 0     │"
echo "│ sciknoweval/chemistry   │ With-Ref DPO (β tuning)      │ 0.5  │ 0     │"
echo "└─────────────────────────┴──────────────────────────────┴──────┴───────┘"
echo ""
echo "Baselines (already in project from Round 2):"
echo "  - No-Ref DPO-2S (β=2.0) × 4 domains"
echo "  - Vanilla GRPO × 4 domains"
echo "  - Vanilla SDPO × 4 domains"
echo ""
echo "Key metrics to watch:"
echo "  - actor/entropy (collapse: < 0.05)"
echo "  - response_length/mean (collapse: < 10)"
echo "  - actor/dpo_ref_chosen / actor/dpo_ref_rejected (NEW: ref logprob tracking)"
echo "  - val-core/sciknoweval/acc/mean@16"
echo "============================================================"
