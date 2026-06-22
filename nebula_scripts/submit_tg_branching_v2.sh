#!/bin/bash
# =============================================================================
# TG-Branching V2 — Optimized experiment wave
#
# Based on Pilot analysis:
#   1. Two-Stage Stage 1 never succeeded (threshold=1.0 too strict) → lower to 0.3/0.5
#   2. SDPO + branching entropy explosion → add entropy_coeff regularization
#   3. GRPO + 2S not yet tested → wire up and run
#   4. Best GRPO variant (NT-suffix) peaked early → early-stop at 100 steps
#
# Usage:
#   bash nebula_scripts/submit_tg_branching_v2.sh [--dry-run]
#
# Env-overridable:
#   DATASET             default sciknoweval/biology
#   MODEL_NAME          default Qwen3-8B
#   LR                  default 1e-5
#   SEED                default 42
#   TRAIN_BATCH_SIZE    default 32
#   MINI_BATCH_SIZE     default 32
#   PROJECT_NAME        default TG-Branching-V2
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
PROJECT_NAME="${PROJECT_NAME:-TG-Branching-V2}"

GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"

# ── 参数解析 ──────────────────────────────────────────────────────────
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
    esac
done

if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 模式：只打印命令，不提交"
fi

# ── 超参配置 ──────────────────────────────────────────────────────────
DATASET="${DATASET:-sciknoweval/biology}"
MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
SEED="${SEED:-42}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
LR="${LR:-1e-5}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
SDPO_ALPHA="${SDPO_ALPHA:-0.5}"
SDPO_DONT_REPROMPT="${SDPO_DONT_REPROMPT_ON_SELF_SUCCESS:-False}"

# Branching defaults
TOP_K=10
TEACHER_TOP_K=100
ENTROPY_WINDOW=20
ENTROPY_SIGMA_START=2.0
ENTROPY_SIGMA_FLOOR=0.5
ENTROPY_SIGMA_STEP=0.5
TEACHER_CONTEXT_MODE=gt_marker
ADV_STD_FLOOR=0.0

TEST_FREQ=10
SAVE_FREQ=10
VAL_BEFORE_TRAIN=True
SAVE_HF_ONLY=True

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

_base_env() {
    local JOB_NAME="$1"
    echo "--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL_NAME=${MODEL_NAME} --env=LR=${LR} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${2} --env=SEED=${SEED} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ} --env=VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN} --env=SAVE_HF_ONLY=${SAVE_HF_ONLY} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT}"
}

_branching_env() {
    echo "--env=BRANCHING_ENABLED=True --env=TOP_K=${TOP_K} --env=TEACHER_TOP_K=${TEACHER_TOP_K} --env=ENTROPY_WINDOW=${ENTROPY_WINDOW} --env=ENTROPY_SIGMA_START=${ENTROPY_SIGMA_START} --env=ENTROPY_SIGMA_FLOOR=${ENTROPY_SIGMA_FLOOR} --env=ENTROPY_SIGMA_STEP=${ENTROPY_SIGMA_STEP} --env=TEACHER_CONTEXT_MODE=${TEACHER_CONTEXT_MODE} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR}"
}

DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
LR_TAG=$(echo "$LR" | tr '-' '_')

GRPO_SCRIPT="nebula_scripts/grpo/grpo_branching_sciknoweval_parametric.sh"
SDPO_SCRIPT="nebula_scripts/sdpo/sdpo_branching_sciknoweval_parametric.sh"

# =============================================================================
# Job 1: GRPO-2S with threshold=0.3 (ref_or_marker)
# Rationale: Test two-stage on the stable GRPO loss with an achievable threshold
# =============================================================================
T=$(date +%Y%m%d_%H%M%S)
JN="V2-GRPO-2S-ref-thr03-suffix-${DATASET_SHORT}-${T}"
_submit_job "$GRPO_SCRIPT" "$JN" \
    "$(_base_env "$JN" 8) $(_branching_env) --env=TOTAL_TRAINING_STEPS=150 --env=N_SPLITS=1 --env=N_TREES=2 --env=BRANCH_TOKEN_LOSS_MODE=suffix --env=TWO_STAGE=True --env=STAGE1_N=4 --env=TWO_STAGE_TEACHER_MODE=ref_or_marker --env=SUCCESS_THRESHOLD=0.3"

# =============================================================================
# Job 2: GRPO-2S with threshold=0.5 (ref_or_marker)
# Rationale: Higher bar — compare Stage 1 success rate vs 0.3
# =============================================================================
T=$(date +%Y%m%d_%H%M%S)
JN="V2-GRPO-2S-ref-thr05-suffix-${DATASET_SHORT}-${T}"
_submit_job "$GRPO_SCRIPT" "$JN" \
    "$(_base_env "$JN" 8) $(_branching_env) --env=TOTAL_TRAINING_STEPS=150 --env=N_SPLITS=1 --env=N_TREES=2 --env=BRANCH_TOKEN_LOSS_MODE=suffix --env=TWO_STAGE=True --env=STAGE1_N=4 --env=TWO_STAGE_TEACHER_MODE=ref_or_marker --env=SUCCESS_THRESHOLD=0.5"

# =============================================================================
# Job 3: SDPO-2S with threshold=0.3 (ref_or_marker)
# Rationale: Fix the root cause — Stage 1 should actually succeed sometimes
# =============================================================================
T=$(date +%Y%m%d_%H%M%S)
JN="V2-SDPO-2S-ref-thr03-suffix-${DATASET_SHORT}-${T}"
_submit_job "$SDPO_SCRIPT" "$JN" \
    "$(_base_env "$JN" 8) $(_branching_env) --env=TOTAL_TRAINING_STEPS=150 --env=N_SPLITS=1 --env=N_TREES=2 --env=BRANCH_TOKEN_LOSS_MODE=suffix --env=ALPHA=${SDPO_ALPHA} --env=DONT_REPROMPT_ON_SELF_SUCCESS=${SDPO_DONT_REPROMPT} --env=TWO_STAGE=True --env=STAGE1_N=4 --env=TWO_STAGE_TEACHER_MODE=ref_or_marker --env=SUCCESS_THRESHOLD=0.3"

# =============================================================================
# Job 4: SDPO-2S with threshold=0.3 + entropy regularization
# Rationale: Combine threshold fix with entropy reg to tame SDPO explosion
# =============================================================================
T=$(date +%Y%m%d_%H%M%S)
JN="V2-SDPO-2S-ref-thr03-ent001-suffix-${DATASET_SHORT}-${T}"
_submit_job "$SDPO_SCRIPT" "$JN" \
    "$(_base_env "$JN" 8) $(_branching_env) --env=TOTAL_TRAINING_STEPS=150 --env=N_SPLITS=1 --env=N_TREES=2 --env=BRANCH_TOKEN_LOSS_MODE=suffix --env=ALPHA=${SDPO_ALPHA} --env=DONT_REPROMPT_ON_SELF_SUCCESS=${SDPO_DONT_REPROMPT} --env=TWO_STAGE=True --env=STAGE1_N=4 --env=TWO_STAGE_TEACHER_MODE=ref_or_marker --env=SUCCESS_THRESHOLD=0.3 --env=ENTROPY_COEFF=0.01"

# =============================================================================
# Job 5: SDPO-NT with suffix + entropy regularization (biology)
# Rationale: Fix entropy explosion in the best SDPO N-trees topology
# =============================================================================
T=$(date +%Y%m%d_%H%M%S)
JN="V2-SDPO-NT-suffix-ent001-${DATASET_SHORT}-${T}"
_submit_job "$SDPO_SCRIPT" "$JN" \
    "$(_base_env "$JN" 8) $(_branching_env) --env=TOTAL_TRAINING_STEPS=150 --env=N_SPLITS=1 --env=N_TREES=4 --env=BRANCH_TOKEN_LOSS_MODE=suffix --env=SPLIT_TRIGGER=entropy_disagreement --env=ALPHA=${SDPO_ALPHA} --env=DONT_REPROMPT_ON_SELF_SUCCESS=${SDPO_DONT_REPROMPT} --env=ENTROPY_COEFF=0.01"

# =============================================================================
# Job 6: SDPO-NT with suffix + entropy reg — chemistry (cross-dataset)
# Rationale: Test generalization beyond biology
# =============================================================================
T=$(date +%Y%m%d_%H%M%S)
CHEM_DATASET="sciknoweval/chemistry"
CHEM_SHORT="sciknoweval-chemistry"
JN="V2-SDPO-NT-suffix-ent001-${CHEM_SHORT}-${T}"
_submit_job "$SDPO_SCRIPT" "$JN" \
    "--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JN} --env=DATASET=${CHEM_DATASET} --env=MODEL_NAME=${MODEL_NAME} --env=LR=${LR} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=8 --env=SEED=${SEED} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ} --env=VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN} --env=SAVE_HF_ONLY=${SAVE_HF_ONLY} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} $(_branching_env) --env=TOTAL_TRAINING_STEPS=150 --env=N_SPLITS=1 --env=N_TREES=4 --env=BRANCH_TOKEN_LOSS_MODE=suffix --env=SPLIT_TRIGGER=entropy_disagreement --env=ALPHA=${SDPO_ALPHA} --env=DONT_REPROMPT_ON_SELF_SUCCESS=${SDPO_DONT_REPROMPT} --env=ENTROPY_COEFF=0.01"

# =============================================================================
# Job 7: GRPO-NT with suffix — early stop at 100 steps
# Rationale: Pilot showed peak at ~step 80, 300 steps wasted compute
# =============================================================================
T=$(date +%Y%m%d_%H%M%S)
JN="V2-GRPO-NT-suffix-100steps-${DATASET_SHORT}-${T}"
_submit_job "$GRPO_SCRIPT" "$JN" \
    "$(_base_env "$JN" 8) $(_branching_env) --env=TOTAL_TRAINING_STEPS=100 --env=N_SPLITS=1 --env=N_TREES=4 --env=BRANCH_TOKEN_LOSS_MODE=suffix --env=SPLIT_TRIGGER=entropy_disagreement"

# =============================================================================
# Job 8: SDPO single-tree branching + suffix + entropy reg
# Rationale: Entropy reg on the original single-tree topology (3 splits, 8 leaves)
# =============================================================================
T=$(date +%Y%m%d_%H%M%S)
JN="V2-SDPO-1T3S-suffix-ent001-${DATASET_SHORT}-${T}"
_submit_job "$SDPO_SCRIPT" "$JN" \
    "$(_base_env "$JN" 8) $(_branching_env) --env=TOTAL_TRAINING_STEPS=150 --env=N_SPLITS=3 --env=N_TREES=1 --env=BRANCH_TOKEN_LOSS_MODE=suffix --env=ALPHA=${SDPO_ALPHA} --env=DONT_REPROMPT_ON_SELF_SUCCESS=${SDPO_DONT_REPROMPT} --env=ENTROPY_COEFF=0.01"

# =============================================================================
echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 V2 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 V2 job"
fi
echo "============================================================"
