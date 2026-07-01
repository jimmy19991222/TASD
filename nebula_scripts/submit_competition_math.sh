#!/bin/bash
# =============================================================================
# Competition Math (MATH 7500) — GRPO vs SDPO vs DPO-2S Comparison
#
# Dataset: EleutherAI/hendrycks_math (Level 1-5 competition math, 6750 train)
# Already uploaded to: oss://.../datasets/competition_math/
#
# Motivation: GSM8K/Math500 saturate at reward ~1.0 for all methods — can't
# distinguish DPO-2S from baselines. MATH has Level 4-5 problems that should
# provide meaningful separation.
#
# Performance: TENSOR_MODEL_PARALLEL_SIZE=2 for faster vLLM generation on
# long CoT sequences (math responses are typically 500~2000 tokens).
#
# Experiment matrix (4 jobs):
#   1. GRPO baseline                        (300 steps, n=8)
#   2. SDPO baseline                        (300 steps, n=8)
#   3. DPO-2S with-ref β=0.5               (300 steps, n=8)
#   4. DPO-2S with-ref β=2.0               (300 steps, n=8)
#
# Usage:
#   bash nebula_scripts/submit_competition_math.sh [--dry-run]
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
PROJECT_NAME="DPO-Comparison-4B"
DATASET="competition_math"
ROLLOUT_N=8
TOTAL_STEPS=300
TENSOR_MODEL_PARALLEL_SIZE=2   # TP=2 for faster vLLM on long math CoT

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

_base_env() {
    local JN="$1" RN="$2" STEPS="$3"
    echo "--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JN} --env=DATASET=${DATASET} --env=MODEL_NAME=${MODEL_NAME} --env=LR=${LR} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${RN} --env=SEED=${SEED} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ} --env=VAL_BEFORE_TRAIN=True --env=SAVE_HF_ONLY=True --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=TOTAL_TRAINING_STEPS=${STEPS} --env=TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE}"
}

_branching_env() {
    echo "--env=BRANCHING_ENABLED=True --env=TOP_K=${TOP_K} --env=TEACHER_TOP_K=${TEACHER_TOP_K} --env=ENTROPY_WINDOW=${ENTROPY_WINDOW} --env=ENTROPY_SIGMA_START=${ENTROPY_SIGMA_START} --env=ENTROPY_SIGMA_FLOOR=${ENTROPY_SIGMA_FLOOR} --env=ENTROPY_SIGMA_STEP=${ENTROPY_SIGMA_STEP} --env=TEACHER_CONTEXT_MODE=${TEACHER_CONTEXT_MODE} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR}"
}

_dpo2s_withref_env() {
    local BETA="$1"
    echo "--env=TWO_STAGE=True --env=STAGE1_N=4 --env=N_SPLITS=1 --env=N_TREES=2 --env=BRANCH_TOKEN_LOSS_MODE=suffix --env=TWO_STAGE_TEACHER_MODE=ref_or_marker --env=SUCCESS_THRESHOLD=0.3 --env=DPO_COEFFICIENT=${BETA} --env=DPO_USE_REF=True --env=ENTROPY_COEFF=0"
}

# =============================================================================
# Job 1: GRPO baseline
# =============================================================================
echo ""
echo "============================================================"
echo "Job 1: GRPO baseline on competition_math (${TOTAL_STEPS} steps)"
echo "============================================================"

T=$(date +%Y%m%d_%H%M%S)
JN="GRPO-4B-competition_math-${T}"
_submit_job "$GRPO_SCRIPT" "$JN" \
    "$(_base_env "$JN" ${ROLLOUT_N} ${TOTAL_STEPS})"

# =============================================================================
# Job 2: SDPO baseline
# =============================================================================
echo ""
echo "============================================================"
echo "Job 2: SDPO baseline on competition_math (${TOTAL_STEPS} steps)"
echo "============================================================"

T=$(date +%Y%m%d_%H%M%S)
JN="SDPO-4B-competition_math-${T}"
_submit_job "$SDPO_SCRIPT" "$JN" \
    "$(_base_env "$JN" ${ROLLOUT_N} ${TOTAL_STEPS}) --env=ALPHA=0.5 --env=DONT_REPROMPT_ON_SELF_SUCCESS=False"

# =============================================================================
# Jobs 3-4: DPO-2S with-ref × {β=0.5, β=2.0}
# =============================================================================
echo ""
echo "============================================================"
echo "Jobs 3-4: DPO-2S with-ref × {β=0.5, 2.0} (${TOTAL_STEPS} steps)"
echo "============================================================"

for BETA in 0.5 2.0; do
    BETA_TAG=$(echo "$BETA" | tr '.' '_')
    T=$(date +%Y%m%d_%H%M%S)
    JN="DPO-2S-4B-withref-beta${BETA_TAG}-competition_math-${T}"
    _submit_job "$GRPO_BRANCHING_SCRIPT" "$JN" \
        "$(_base_env "$JN" ${ROLLOUT_N} ${TOTAL_STEPS}) $(_branching_env) $(_dpo2s_withref_env $BETA)"
    sleep 1
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
echo "实验矩阵 (model=${MODEL_NAME}, dataset=competition_math, ${TOTAL_STEPS} steps, TP=${TENSOR_MODEL_PARALLEL_SIZE}):"
echo "┌─────────────────────────────────────┬───────┬──────┬───────┐"
echo "│ Method                              │ β     │ n    │ steps │"
echo "├─────────────────────────────────────┼───────┼──────┼───────┤"
echo "│ GRPO (baseline)                     │ —     │ 8    │ ${TOTAL_STEPS}   │"
echo "│ SDPO (baseline, α=0.5)              │ —     │ 8    │ ${TOTAL_STEPS}   │"
echo "│ DPO-2S with-ref                     │ 0.5   │ 8    │ ${TOTAL_STEPS}   │"
echo "│ DPO-2S with-ref                     │ 2.0   │ 8    │ ${TOTAL_STEPS}   │"
echo "└─────────────────────────────────────┴───────┴──────┴───────┘"
echo "Total: 4 jobs"
echo ""
echo "Key question: Does competition math (Level 1-5) provide enough"
echo "  difficulty spread to differentiate DPO-2S from GRPO/SDPO?"
echo "TP=${TENSOR_MODEL_PARALLEL_SIZE} for faster vLLM generation on long math CoT."
echo "============================================================"
