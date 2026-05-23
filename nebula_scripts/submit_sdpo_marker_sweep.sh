#!/bin/bash
# =============================================================================
# SDPO Self-Verified-Marker Sweep — 3 new teacher-context variants vs JSD baseline
#
# Hypothesis: prepending "This answer is verified correct." on the assistant
# turn provides a self-contained ΔW signal without needing the privileged ref.
#   * marker        : pure self-distill, teacher reads marker only (no ref)
#   * marker + damp : marker + ΔH overconfidence damping
#                     w_t = exp(-max(0, H_s - H_T) / τ), τ=1.0
#   * gt_marker     : per-sample marker with ground-truth substituted:
#                     "This answer is verified correct, correct answer is <gt>."
# All share the JSD vocab-level SDPO baseline (alpha=0.5, full_logit, topk=100).
#
# Usage:
#   bash nebula_scripts/submit_sdpo_marker_sweep.sh [--dry-run]
#   bash nebula_scripts/submit_sdpo_marker_sweep.sh --variant marker
#   bash nebula_scripts/submit_sdpo_marker_sweep.sh --variant marker,marker_damp
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
PROJECT_NAME="${PROJECT_NAME:-SDPO_Marker}"

# ── 参数解析 ────────────────────────────────────────────────────────────
DRY_RUN=false
VARIANT_SELECT="all"   # all | marker | marker_damp | gt_marker | comma-separated

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --variant=*) VARIANT_SELECT="${arg#*=}" ;;
        --variant) shift; VARIANT_SELECT="$1" ;;
    esac
done

[ "$DRY_RUN" = true ] && echo "Dry-run 模式：只打印命令,不提交"

# ── 公共固定超参 (与 LossVariants sweep JSD baseline 对齐) ─────────────
MODEL_NAME="Qwen3-8B"
DATASET="sciknoweval/biology"
LR="1e-5"
SEED="42"
TRAIN_BATCH_SIZE="32"
ROLLOUT_N="8"
ALPHA="0.5"                                  # JSD
FULL_LOGIT="True"
DISTILLATION_TOPK="100"
DONT_REPROMPT_ON_SELF_SUCCESS="False"
SCRIPT_PATH="nebula_scripts/sdpo/sdpo_sciknoweval_parametric.sh"
DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
LR_TAG=$(echo "$LR" | tr '-' '_')

# ── Marker 配置矩阵 ────────────────────────────────────────────────────
# tag : teacher_context_mode : overconfidence_damping
VARIANTS=(
    "marker:marker:0.0"
    "marker_damp:marker:1.0"
    "gt_marker:gt_marker:0.0"
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
for spec in "${VARIANTS[@]}"; do
    IFS=':' read -r TAG CTX_MODE DAMPING <<< "$spec"
    _should_run "$TAG" || continue

    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="SDPO-mk-${TAG}-${DATASET_SHORT}-lr${LR_TAG}-${MODEL_NAME}-${CURRENT_TIME}"

    USER_PARAMS="--env=PROJECT_NAME=${PROJECT_NAME} \
--env=JOB_NAME=${JOB_NAME} \
--env=DATASET=${DATASET} \
--env=MODEL_NAME=${MODEL_NAME} \
--env=LR=${LR} \
--env=ALPHA=${ALPHA} \
--env=FULL_LOGIT_DISTILLATION=${FULL_LOGIT} \
--env=DISTILLATION_TOPK=${DISTILLATION_TOPK} \
--env=TEACHER_CONTEXT_MODE=${CTX_MODE} \
--env=OVERCONFIDENCE_DAMPING=${DAMPING} \
--env=LOG_DELTA_W_STATS=True \
--env=DONT_REPROMPT_ON_SELF_SUCCESS=${DONT_REPROMPT_ON_SELF_SUCCESS} \
--env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} \
--env=ROLLOUT_N=${ROLLOUT_N} \
--env=SEED=${SEED}"

    _submit_job "$JOB_NAME" "$USER_PARAMS"
done

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成,共 ${TOTAL} 个 job (variants: ${VARIANT_SELECT})"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job (variants: ${VARIANT_SELECT})"
fi
echo "============================================================"
