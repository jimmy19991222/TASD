#!/bin/bash
# =============================================================================
# SDPO + Entropy Weighting 超参扫描 - Nebula 批量提交
#
# 参考 experiments/generalization/run_sdpo_all_entropy_weighting_local_1.sh
# 使用方式：bash nebula_scripts/submit_sdpo_ew_sweep.sh [--dry-run]
# =============================================================================

# ── Nebula 账号配置 ──────────────────────────────────────────────────────
QUEUE="lazada_llm_ad_h20"
WORLD_SIZE=1
OPENLM_TOKEN="${OPENLM_TOKEN:?OPENLM_TOKEN not set}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:?OSS_ACCESS_ID not set}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?OSS_ACCESS_KEY not set}"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
CLUSTER_FILE="nebula_scripts/cluster_gpu_4.json"
SCRIPT_PATH="nebula_scripts/sdpo/sdpo_entropy_weighting_parametric.sh"
# 自定义镜像（留空则使用 --algo_name=pytorch260 默认镜像）
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324105345}"

DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 超参配置（尽量多覆盖 EW 相关维度）
# =============================================================================
DATASETS=(
    "sciknoweval/biology"
    # "sciknoweval/chemistry"
    # "sciknoweval/material"
    # "sciknoweval/physics"
)

MODEL_NAMES=("Qwen3-8B")

LRS=("1e-5")
ALPHAS=("0.5")
DONT_REPROMPT_LIST=("True")

# ── entropy weighting 核心扫描 ────────────────────────────────────────
# EW=True  扫描不同 temperature
ENTROPY_WEIGHTING_LIST=("True")
# temperature 仅在 EW=True 时有意义，EW=False 时跳过多余组合
ENTROPY_TEMPERATURE_LIST=("1.0" "0.5" "2.0")
ENTROPY_WEIGHTING_VERSION_LIST=(
    "v1" 
    "v4"
)

# 固定参数
TRAIN_BATCH_SIZE="32"
ROLLOUT_N="8"

# =============================================================================
TOTAL=0
SUBMITTED=0

for DATASET in "${DATASETS[@]}"; do
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
for LR in "${LRS[@]}"; do
for ALPHA in "${ALPHAS[@]}"; do
for DONT_REPROMPT in "${DONT_REPROMPT_LIST[@]}"; do
for EW in "${ENTROPY_WEIGHTING_LIST[@]}"; do
for ET in "${ENTROPY_TEMPERATURE_LIST[@]}"; do
for EW_VER in "${ENTROPY_WEIGHTING_VERSION_LIST[@]}"; do

    # EW=False 时 temperature 无意义，只跑一次（第一个 temperature）
    if [ "$EW" = "False" ] && [ "$ET" != "${ENTROPY_TEMPERATURE_LIST[0]}" ]; then
        continue
    fi

    TOTAL=$((TOTAL + 1))

    DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    EW_TAG="ew${EW}"
    ET_TAG=""
    if [ "$EW" = "True" ]; then
        ET_TAG="-et${ET}"
    fi
    JOB_NAME="SDPO-${DATASET_SHORT}-alpha${ALPHA}-lr${LR}-dross${DONT_REPROMPT}-${EW_VER}-${EW_TAG}${ET_TAG}-${MODEL_NAME}-${CURRENT_TIME}"

    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  DATASET=$DATASET LR=$LR ALPHA=$ALPHA"
        echo "  EW=$EW ET=$ET EW_VER=$EW_VER"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
                    --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME}" \
            --worker_count=${WORLD_SIZE} \
            --file.cluster_file=${CLUSTER_FILE} \
            --job_name=${JOB_NAME} \
            --access_id=${access_id} \
            --access_key=${access_key} \
            --env=OPENLM_TOKEN=${OPENLM_TOKEN} \
            --env=JOB_NAME=${JOB_NAME} \
            --env=DATASET=${DATASET} \
            --env=MODEL_NAME=${MODEL_NAME} \
            --env=LR=${LR} \
            --env=ALPHA=${ALPHA} \
            --env=DONT_REPROMPT_ON_SELF_SUCCESS=${DONT_REPROMPT} \
            --env=ENTROPY_WEIGHTING=${EW} \
            --env=ENTROPY_TEMPERATURE=${ET} \
            --env=ENTROPY_WEIGHTING_VERSION=${EW_VER} \
            --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} \
            --env=ROLLOUT_N=${ROLLOUT_N} \
            $([ -n "$CUSTOM_DOCKER_IMAGE" ] && echo "--custom_docker_image=${CUSTOM_DOCKER_IMAGE}" || echo "--algo_name=pytorch260") \
            --requirements_file_name=requirements_nebula.txt \
            --oss_access_id=${OSS_ACCESS_ID} \
            --oss_access_key=${OSS_ACCESS_KEY} \
            --oss_bucket=${OSS_BUCKET} \
            --oss_endpoint=${OSS_ENDPOINT} 2>&1)
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

done  # EW_VER
done  # ET
done  # EW
done  # DONT_REPROMPT
done  # ALPHA
done  # LR
done  # MODEL_NAME
done  # DATASET

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
