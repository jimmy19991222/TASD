#!/bin/bash
# =============================================================================
# DPO-FIX Batch Submission (final)
#
# Project: DPO-FIX | Group: DPO-FIX-20260702
# β=2.0, 每个数据集 nopair + stage1pair 各一个
# 数据集: tooluse + sciknoweval × 4 domains (competition_math 暂缓)
#
# Usage: bash nebula_scripts/submit_dpo_fix_experiments.sh [--dry-run]
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

GROUP_NAME="DPO-FIX-20260702"
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

# Common
MODEL="Qwen3-4B"; SEED=42; LR="1e-5"; TBS=32; RN=8; MBS=32
TF=10; SF=10; TP=1; BETA="2.0"
TK=10; TTK=100; EW=20; ESS=2.0; ESF=0.5; EST=0.5; TCM=marker; ASF=0.05

_base() {
    echo "--env=PROJECT_NAME=DPO-FIX --env=JOB_NAME=$1 --env=DATASET=$2 --env=MODEL_NAME=${MODEL} --env=LR=${LR} --env=TRAIN_BATCH_SIZE=${TBS} --env=ROLLOUT_N=${RN} --env=SEED=${SEED} --env=TEST_FREQ=${TF} --env=SAVE_FREQ=${SF} --env=VAL_BEFORE_TRAIN=True --env=SAVE_HF_ONLY=True --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=TOTAL_TRAINING_STEPS=$3 --env=TENSOR_MODEL_PARALLEL_SIZE=${TP} --env=GROUP_NAME=${GROUP_NAME} --env=MINI_BATCH_SIZE=${MBS}"
}
_branch() {
    echo "--env=BRANCHING_ENABLED=True --env=TOP_K=${TK} --env=TEACHER_TOP_K=${TTK} --env=ENTROPY_WINDOW=${EW} --env=ENTROPY_SIGMA_START=${ESS} --env=ENTROPY_SIGMA_FLOOR=${ESF} --env=ENTROPY_SIGMA_STEP=${EST} --env=TEACHER_CONTEXT_MODE=${TCM} --env=ADV_STD_FLOOR=${ASF}"
}
_dpo() {
    echo "--env=TWO_STAGE=True --env=STAGE1_N=4 --env=N_SPLITS=1 --env=N_TREES=2 --env=BRANCH_TOKEN_LOSS_MODE=suffix --env=TWO_STAGE_TEACHER_MODE=ref_or_marker --env=SUCCESS_THRESHOLD=0.3 --env=DPO_COEFFICIENT=${BETA} --env=DPO_USE_REF=True --env=ENTROPY_COEFF=0 --env=DPO_STAGE1_PAIR=$1 --env=STAGE1_PAIR_WEIGHT=1.0"
}

DS="nebula_scripts/grpo/grpo_branching_sciknoweval_parametric.sh"

# ── tooluse (250 steps) ──
echo "=== tooluse DPO-2S β=${BETA} ==="
for P in nopair stage1pair; do
    PARG=$([ "$P" = "nopair" ] && echo "False" || echo "True")
    T=$(date +%Y%m%d_%H%M%S)
    JN="DPO-2S-beta${BETA}-${P}-tooluse-DPOFIX-${T}"
    _submit_job "$DS" "$JN" "$(_base "$JN" tooluse 250) $(_branch) $(_dpo "$PARG")"
done

# ── sciknoweval 4 domains (300 steps) ──
echo "=== sciknoweval DPO-2S β=${BETA} ==="
for DOMAIN in biology chemistry physics material; do
    for P in nopair stage1pair; do
        PARG=$([ "$P" = "nopair" ] && echo "False" || echo "True")
        T=$(date +%Y%m%d_%H%M%S)
        JN="DPO-2S-beta${BETA}-${P}-sciknoweval-${DOMAIN}-DPOFIX-${T}"
        _submit_job "$DS" "$JN" "$(_base "$JN" "sciknoweval/${DOMAIN}" 300) $(_branch) $(_dpo "$PARG")"
        sleep 1
    done
done

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job → project=DPO-FIX, group=${GROUP_NAME}"
fi
echo "============================================================"
