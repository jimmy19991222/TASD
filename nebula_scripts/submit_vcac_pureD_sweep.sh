#!/bin/bash
# =============================================================================
# VCAC pure-δ ablation sweep — 4 jobs on sciknoweval/biology only.
#
# Advantage at PG loss is purely
#     A_t = λ · δ_t        (drops the GRPO baseline term)
# where δ_t = log π_T^+(y_t|x,y_<t) − log π_T^-(y_t|x,y_<t) is the CDM
# dual-teacher verdict log-odds shift. PPO clip / GRPO PG path is preserved
# (still routed through compute_policy_loss_vanilla). Toggled via
# policy_loss.vcac_use_grpo_advantage=False (env: VCAC_USE_GRPO_ADV=False).
#
# Design — λ as a meaningful "verdict trust" knob:
#   - 3 jobs with VCAC_NORMALIZE=True : δ_t standardized to unit variance per
#     batch → λ controls SNR-aware verdict strength. Sweep {0.1, 0.3, 1.0}.
#   - 1 job with VCAC_NORMALIZE=False, λ=1.0 : raw-scale control to see
#     whether the unnormalized δ swing matters.
#
# Scope: sciknoweval/biology only.
#
# Usage:
#   bash nebula_scripts/submit_vcac_pureD_sweep.sh [--dry-run]
# =============================================================================

QUEUE="lazada_llm_ad_h20"
WORLD_SIZE=1
OPENLM_TOKEN="${OPENLM_TOKEN:?OPENLM_TOKEN not set}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:?OSS_ACCESS_ID not set}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?OSS_ACCESS_KEY not set}"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
CLUSTER_FILE="nebula_scripts/cluster.json"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="${PROJECT_NAME:-VCAC}"

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
    esac
done

[ "$DRY_RUN" = true ] && echo "Dry-run 模式：只打印命令，不提交"

MODEL_NAME="Qwen3-8B"
SEED="42"
TRAIN_BATCH_SIZE="32"
ROLLOUT_N="8"
LR="1e-5"
SAVE_FREQ="${SAVE_FREQ:-9999}"
TEST_FREQ="${TEST_FREQ:-10}"

CDM_VARIANT="ref_mk"
VCAC_CLIP="null"
VCAC_USE_GRPO_ADV="False"   # ← the ablation toggle (drop GRPO adv)

# (lambda, normalize) tuples — 3 normalized + 1 raw control
CONFIGS=(
    "0.1:True"
    "0.3:True"
    "1.0:True"
    "1.0:False"
)

DATASET="sciknoweval/biology"
DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
SCRIPT_PATH="nebula_scripts/vcac/vcac_sciknoweval_parametric.sh"

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

for CFG in "${CONFIGS[@]}"; do
    VCAC_LAMBDA="${CFG%%:*}"
    VCAC_NORMALIZE="${CFG##*:}"
    LAM_TAG=$(echo "$VCAC_LAMBDA" | tr '.' '_')
    NORM_TAG=$([ "$VCAC_NORMALIZE" = "True" ] && echo "norm" || echo "raw")
    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="VCAC-pureD-${DATASET_SHORT}-lam${LAM_TAG}-${NORM_TAG}-${CDM_VARIANT}-${MODEL_NAME}-${CURRENT_TIME}"
    ENV_STR="--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=MODEL_NAME=${MODEL_NAME} --env=LR=${LR} --env=VCAC_LAMBDA=${VCAC_LAMBDA} --env=VCAC_CLIP=${VCAC_CLIP} --env=VCAC_NORMALIZE=${VCAC_NORMALIZE} --env=VCAC_USE_GRPO_ADV=${VCAC_USE_GRPO_ADV} --env=CDM_TEMPLATE_VARIANT=${CDM_VARIANT} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=SEED=${SEED} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ} --env=DATASET=${DATASET}"
    _submit_job "$SCRIPT_PATH" "$JOB_NAME" "$ENV_STR"
done

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job (pure-δ, bio only)"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job (pure-δ, bio only)"
fi
echo "============================================================"
