#!/bin/bash
# =============================================================================
# TG-GRPO (Teacher-Guided GRPO) pilot — Nebula batch submitter
#
# TG-GRPO keeps the GRPO PG loss path with advantage A_t = R - R̄_group
# unchanged and multiplies per-token by a teacher-derived weight w_t derived
# from |δ_t| = |log π_T^+ - log π_T^-|. The teacher acts as a decision-token
# selector, not a critic — preserving GRPO unbiasedness while concentrating
# gradient on tokens that move the verdict posterior.
#
# Phase-1 pilot: 1 TG-GRPO run (top-30% per-traj) + 1 vanilla GRPO baseline
# on sciknoweval/biology, head-to-head with matched seed/batch-size/LR.
#
# Usage:
#   bash nebula_scripts/submit_tg_grpo_pilot.sh [--dry-run]
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
PROJECT_NAME="${PROJECT_NAME:-TG-GRPO}"

# ── 命令行参数解析 ──────────────────────────────────────────────────────
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --only-tg) ONLY_TG=1 ;;
        --only-grpo) ONLY_GRPO=1 ;;
    esac
done
ONLY_TG="${ONLY_TG:-0}"
ONLY_GRPO="${ONLY_GRPO:-0}"

if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 模式：只打印命令，不提交"
fi
if [ "$ONLY_TG" = "1" ]; then
    echo "ONLY_TG=1：仅提交 TG-GRPO，跳过 vanilla GRPO baseline"
fi
if [ "$ONLY_GRPO" = "1" ]; then
    echo "ONLY_GRPO=1：仅提交 vanilla GRPO baseline，跳过 TG-GRPO"
fi

# =============================================================================
# Common config (matched across both jobs for fair head-to-head)
# =============================================================================
MODEL_NAME="Qwen3-8B"
DATASET="sciknoweval/biology"
SEED="42"
TRAIN_BATCH_SIZE="32"
ROLLOUT_N="8"
LR="1e-5"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-10}"

# TG-GRPO pilot config
TG_TOP_K="0.30"
TG_W_MODE="topk_traj"
CDM_TEMPLATE_VARIANT="ref_mk"

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

# ─────────────────────────────────────────────────────────────────────────────
# Job 1: TG-GRPO  (top-30% per-traj, ref_mk markers)
# ─────────────────────────────────────────────────────────────────────────────
DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
TOPK_TAG=$(echo "$TG_TOP_K" | tr '.' '_')
CURRENT_TIME=$(date +%Y%m%d_%H%M%S)

JOB_NAME_TG="TGGRPO-${DATASET_SHORT}-k${TOPK_TAG}-${TG_W_MODE}-${CDM_TEMPLATE_VARIANT}-${MODEL_NAME}-${CURRENT_TIME}"
ENV_STR_TG="--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME_TG} --env=MODEL_NAME=${MODEL_NAME} --env=DATASET=${DATASET} --env=LR=${LR} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=SEED=${SEED} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ} --env=TG_TOP_K=${TG_TOP_K} --env=TG_W_MODE=${TG_W_MODE} --env=CDM_TEMPLATE_VARIANT=${CDM_TEMPLATE_VARIANT}"
if [ "$ONLY_GRPO" != "1" ]; then
    _submit_job "nebula_scripts/tg_grpo/tg_grpo_sciknoweval_parametric.sh" "$JOB_NAME_TG" "$ENV_STR_TG"
else
    echo "跳过 TG-GRPO job (ONLY_GRPO=1)"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Job 2: GRPO baseline  (matched config, vanilla loss_mode, single-teacher)
# ─────────────────────────────────────────────────────────────────────────────
sleep 1
CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
MINI_BATCH_SIZE_GRPO="32"
JOB_NAME_GRPO="GRPObase-${DATASET_SHORT}-mbs${TRAIN_BATCH_SIZE}-lr${LR/./_}-${MODEL_NAME}-${CURRENT_TIME}"
ENV_STR_GRPO="--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME_GRPO} --env=MODEL_NAME=${MODEL_NAME} --env=DATASET=${DATASET} --env=LR=${LR} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE_GRPO} --env=ROLLOUT_N=${ROLLOUT_N} --env=SEED=${SEED} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ}"
if [ "$ONLY_TG" != "1" ]; then
    _submit_job "nebula_scripts/grpo/grpo_sciknoweval_parametric.sh" "$JOB_NAME_GRPO" "$ENV_STR_GRPO"
else
    echo "跳过 vanilla GRPO baseline (ONLY_TG=1)"
fi

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
