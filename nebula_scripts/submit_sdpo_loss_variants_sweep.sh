#!/bin/bash
# =============================================================================
# SDPO Loss-Variant Sweep — 4 个 loss 形式横评
#   1) token-level   : full_logit=False, alpha=1.0  (token-PG; rKL 的 MC 估计)
#   2) reverse KL    : full_logit=True,  alpha=1.0  (mode-seeking)
#   3) forward KL    : full_logit=True,  alpha=0.0  (mode-covering)
#   4) JSD           : full_logit=True,  alpha=0.5  (rKL/fKL 加性内插)
#
# 所有变体共享 dataset / model / LR / batch / seed,以隔离 loss 形式效应。
# 所有变体都开 log_delta_w_stats=True,顺便采 ΔW SNR 诊断指标。
#
# 使用方式：
#   bash nebula_scripts/submit_sdpo_loss_variants_sweep.sh [--dry-run]
#   bash nebula_scripts/submit_sdpo_loss_variants_sweep.sh --variant rkl [--dry-run]
#   bash nebula_scripts/submit_sdpo_loss_variants_sweep.sh --variant token,fkl
#   bash nebula_scripts/submit_sdpo_loss_variants_sweep.sh --with-calibration   # 4 loss + 1 verdict calibration
#   bash nebula_scripts/submit_sdpo_loss_variants_sweep.sh --calibration-only   # only run calibration
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
PROJECT_NAME="${PROJECT_NAME:-SDPO_LossVariants}"

# ── 参数解析 ────────────────────────────────────────────────────────────
DRY_RUN=false
VARIANT_SELECT="all"   # all | token | rkl | fkl | jsd | comma-separated
WITH_CALIBRATION=false
CALIBRATION_ONLY=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --variant=*) VARIANT_SELECT="${arg#*=}" ;;
        --variant) shift; VARIANT_SELECT="$1" ;;
        --with-calibration) WITH_CALIBRATION=true ;;
        --calibration-only) CALIBRATION_ONLY=true; WITH_CALIBRATION=true ;;
    esac
done

[ "$DRY_RUN" = true ] && echo "Dry-run 模式：只打印命令,不提交"

# ── 公共固定超参 ────────────────────────────────────────────────────────
MODEL_NAME="Qwen3-8B"
DATASET="sciknoweval/biology"
LR="1e-5"
SEED="42"
TRAIN_BATCH_SIZE="32"
ROLLOUT_N="8"
DONT_REPROMPT_ON_SELF_SUCCESS="False"
SCRIPT_PATH="nebula_scripts/sdpo/sdpo_sciknoweval_parametric.sh"
DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
LR_TAG=$(echo "$LR" | tr '-' '_')

# ── Loss-variant 配置矩阵 ──────────────────────────────────────────────
# tag : full_logit_distillation : alpha
VARIANTS=(
    "token:False:1.0"
    "rkl:True:1.0"
    "fkl:True:0.0"
    "jsd:True:0.5"
)

_should_run() {
    local tag="$1"
    [ "$VARIANT_SELECT" = "all" ] && return 0
    case ",${VARIANT_SELECT}," in *",${tag},"*) return 0 ;; esac
    return 1
}

# ── 提交辅助 ────────────────────────────────────────────────────────────
TOTAL=0
SUBMITTED=0

_submit_job() {
    local JOB_NAME="$1"
    local USER_PARAMS="$2"

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

# ── 主循环 ──────────────────────────────────────────────────────────────
if [ "$CALIBRATION_ONLY" = true ]; then
    VARIANTS=()
fi

for spec in "${VARIANTS[@]}"; do
    IFS=':' read -r TAG FULL_LOGIT ALPHA <<< "$spec"
    _should_run "$TAG" || continue

    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="SDPO-loss-${TAG}-alpha${ALPHA}-fl${FULL_LOGIT}-${DATASET_SHORT}-lr${LR_TAG}-${MODEL_NAME}-${CURRENT_TIME}"

    USER_PARAMS="--env=PROJECT_NAME=${PROJECT_NAME} \
--env=JOB_NAME=${JOB_NAME} \
--env=DATASET=${DATASET} \
--env=MODEL_NAME=${MODEL_NAME} \
--env=LR=${LR} \
--env=ALPHA=${ALPHA} \
--env=FULL_LOGIT_DISTILLATION=${FULL_LOGIT} \
--env=LOG_DELTA_W_STATS=True \
--env=DONT_REPROMPT_ON_SELF_SUCCESS=${DONT_REPROMPT_ON_SELF_SUCCESS} \
--env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} \
--env=ROLLOUT_N=${ROLLOUT_N} \
--env=SEED=${SEED}"

    _submit_job "$JOB_NAME" "$USER_PARAMS"
done

# ── 可选: verdict calibration check (一次性 offline job, 不训练) ───────
if [ "$WITH_CALIBRATION" = true ]; then
    CALIB_SCRIPT_PATH="nebula_scripts/diagnostics/verdict_calibration.sh"
    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    CALIB_JOB_NAME="SDPO-loss-verdictCal-${DATASET_SHORT}-${MODEL_NAME}-${CURRENT_TIME}"
    CALIB_USER_PARAMS="--env=PROJECT_NAME=${PROJECT_NAME} \
--env=JOB_NAME=${CALIB_JOB_NAME} \
--env=DATASET=${DATASET} \
--env=MODEL_NAME=${MODEL_NAME} \
--env=N_EXAMPLES=50 \
--env=N_ROLLOUTS_PER_EXAMPLE=4 \
--env=MAX_NEW_TOKENS=512 \
--env=SEED=${SEED}"

    TOTAL=$((TOTAL + 1))
    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${CALIB_JOB_NAME}  [script: ${CALIB_SCRIPT_PATH}]"
        echo "  $CALIB_USER_PARAMS"
    else
        echo "提交 Job #${TOTAL}: ${CALIB_JOB_NAME}"
        SUBMIT_OUTPUT=$(nebulactl run mdl \
            --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${CALIB_SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${CALIB_JOB_NAME} ${CALIB_USER_PARAMS}" \
            --worker_count=${WORLD_SIZE} \
            --file.cluster_file=${CLUSTER_FILE} \
            --job_name=${CALIB_JOB_NAME} \
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
fi

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成,共 ${TOTAL} 个 job (variants: ${VARIANT_SELECT}, calibration: ${WITH_CALIBRATION})"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job (variants: ${VARIANT_SELECT}, calibration: ${WITH_CALIBRATION})"
fi
echo "============================================================"
