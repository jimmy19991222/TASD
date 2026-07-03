#!/bin/bash
# =============================================================================
# DPO-8B Submission — re-run ONLY the DPO leg on Qwen3-8B
#
# 8B SDPO / GRPO baselines already exist offline; this submits the
# DPO-2S (stage1pair) leg on Qwen3-8B for the shared comparison matrix:
#   datasets: math500 + sciknoweval{biology,chemistry,physics,material} + tooluse
#   method  : DPO-2S stage1pair, withref, β=2.0, reward-consistency filter ON
#
# Config matched to the 8B baselines: TP=1, rollout.n=8, TBS=32, mbs=32, lr=1e-5.
#
# Usage: bash nebula_scripts/submit_dpo_8b.sh [--dry-run]
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
[ "$DRY_RUN" = true ] && echo "Dry-run 模式：只打印命令，不提交"

GROUP_NAME="DPO-8B-20260703"
PROJECT_NAME="DPO-8B"
TOTAL=0
SUBMITTED=0

_submit_job() {
    local SCRIPT_PATH="$1" JOB_NAME="$2" USER_PARAMS="$3"
    TOTAL=$((TOTAL + 1))
    if [ "$DRY_RUN" = true ]; then
        echo "Job #${TOTAL}: ${JOB_NAME}"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"
        nebulactl run mdl --force --engine=xdl --queue=${QUEUE} \
            ${NEBULA_WORK_DIR:+--nebula_work_dir ${NEBULA_WORK_DIR}} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} ${USER_PARAMS}" \
            --worker_count=${WORLD_SIZE} --file.cluster_file=${CLUSTER_FILE} \
            --job_name=${JOB_NAME} \
            --env=OPENLM_TOKEN=${OPENLM_TOKEN} \
            --env=SWANLAB_API_KEY=${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal} \
            --custom_docker_image=${CUSTOM_DOCKER_IMAGE} \
            --requirements_file_name=requirements_nebula.txt \
            --oss_access_id=${OSS_ACCESS_ID} --oss_access_key=${OSS_ACCESS_KEY} \
            --oss_bucket=${OSS_BUCKET} --oss_endpoint=${OSS_ENDPOINT} 2>&1 | tail -3
        SUBMITTED=$((SUBMITTED + 1))
        echo "✅ (${SUBMITTED}/${TOTAL})"
        sleep 2
    fi
}

# Common (matched to 8B baselines)
MODEL="Qwen3-8B"; SEED=42; LR="1e-5"; TBS=32; RN=8; MBS=32
TF=10; SF=10; TP=1; BETA="${BETA:-2.0}"
TK=10; TTK=100; EW=20; ESS=2.0; ESF=0.5; EST=0.5; TCM=marker; ASF=0.05

_base() {  # $1=JN $2=DATASET $3=STEPS
    echo "--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=$1 --env=DATASET=$2 --env=MODEL_NAME=${MODEL} --env=LR=${LR} --env=TRAIN_BATCH_SIZE=${TBS} --env=ROLLOUT_N=${RN} --env=SEED=${SEED} --env=TEST_FREQ=${TF} --env=SAVE_FREQ=${SF} --env=VAL_BEFORE_TRAIN=True --env=SAVE_HF_ONLY=True --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=TOTAL_TRAINING_STEPS=$3 --env=TENSOR_MODEL_PARALLEL_SIZE=${TP} --env=GROUP_NAME=${GROUP_NAME} --env=MINI_BATCH_SIZE=${MBS}"
}
_branch() {
    echo "--env=BRANCHING_ENABLED=True --env=TOP_K=${TK} --env=TEACHER_TOP_K=${TTK} --env=ENTROPY_WINDOW=${EW} --env=ENTROPY_SIGMA_START=${ESS} --env=ENTROPY_SIGMA_FLOOR=${ESF} --env=ENTROPY_SIGMA_STEP=${EST} --env=TEACHER_CONTEXT_MODE=${TCM} --env=ADV_STD_FLOOR=${ASF}"
}
_dpo() {  # stage1pair + withref + reward filter ON
    local TG="${DPO_TEACHER_GUIDED_BETA:-False}"
    local TGA="${DPO_TEACHER_BETA_ALPHA:-1.0}"
    local TGM="${DPO_TEACHER_BETA_MIN:-0.1}"
    local TGX="${DPO_TEACHER_BETA_MAX:-3.0}"
    echo "--env=TWO_STAGE=True --env=STAGE1_N=4 --env=N_SPLITS=1 --env=N_TREES=2 --env=BRANCH_TOKEN_LOSS_MODE=suffix --env=TWO_STAGE_TEACHER_MODE=ref_or_marker --env=SUCCESS_THRESHOLD=0.3 --env=DPO_COEFFICIENT=${BETA} --env=DPO_USE_REF=True --env=ENTROPY_COEFF=0 --env=DPO_STAGE1_PAIR=True --env=STAGE1_PAIR_WEIGHT=1.0 --env=DPO_REWARD_FILTER=True --env=DPO_TEACHER_GUIDED_BETA=${TG} --env=DPO_TEACHER_BETA_ALPHA=${TGA} --env=DPO_TEACHER_BETA_MIN=${TGM} --env=DPO_TEACHER_BETA_MAX=${TGX}"
}

DS="nebula_scripts/grpo/grpo_branching_sciknoweval_parametric.sh"

# dataset -> steps
echo "=== DPO-2S-8B stage1pair (β=${BETA}, reward-filter ON) ==="
INCLUDE="${INCLUDE:-all}"   # comma list of tags to submit, or "all"
_one() {  # $1=DATASET $2=TAG $3=STEPS
    if [ "$INCLUDE" != "all" ] && [[ ",$INCLUDE," != *",$2,"* ]]; then return; fi
    T=$(date +%Y%m%d_%H%M%S)
    JN="DPO-2S-8B-stage1pair-beta${BETA}-$2-${T}"
    _submit_job "$DS" "$JN" "$(_base "$JN" "$1" "$3") $(_branch) $(_dpo)"
    sleep 1
}

# All datasets 250 steps for a uniform comparison across the matrix.
_one "math500"               "math500"   250
_one "sciknoweval/biology"   "biology"   250
_one "sciknoweval/chemistry" "chemistry" 250
_one "sciknoweval/physics"   "physics"   250
_one "sciknoweval/material"  "material"  250
_one "tooluse"               "tooluse"   250

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job → project=${PROJECT_NAME}, group=${GROUP_NAME}"
fi
echo "============================================================"
