#!/bin/bash
# =============================================================================
# TASD 清爽版 - Nebula 批量提交脚本
#
# 实验设计：
#   - reward_type: teacher_prob | teacher_log_prob
#   - entropy_gate: none | hard | soft
#   - clip_adv_value: 2.0
#
# 使用方式：
#   bash nebula_scripts/submit_tasd_simple_sweep.sh [--dry-run]
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
SCRIPT_PATH="nebula_scripts/tasd_simple/tasd_simple_parametric.sh"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="TASD_simple"

# ── 数据集配置 ──────────────────────────────────────────────────────
DATASETS=(
    # "sciknoweval/biology"
    "sciknoweval/chemistry"
    "sciknoweval/material"
    "sciknoweval/physics"
    "tooluse"
)

# ── dry-run 模式 ─────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 超参配置
# =============================================================================

# ── Reward Type ─────────────────────────────────────────────────────
REWARD_TYPES=(
    "teacher_prob"
    # "teacher_log_prob"
)

# ── Entropy Gate ─────────────────────────────────────────────────────
ENTROPY_GATE_LIST=(
    "none"
    "hard"
    # "soft"
)

# ── Clip Adv Value ───────────────────────────────────────────────────
CLIP_ADV_VALUES=(
    "2.0"
)

# ── Distill Topk ──────────────────────────────────────────────────────
DISTILL_TOPK_LIST=(
    "100"
)

# ── Repetition Penalty ───────────────────────────────────────────────
REPETITION_PENALTY_LIST=(
    "1.0"
    "1.05"
)

# ── Norm Adv By Std ─────────────────────────────────────────────────
NORM_ADV_BY_STD_LIST=(
    "true"
    # "false"
)

# ── Adv Std Floor ───────────────────────────────────────────────────
# std下界：none | auto | float（仅在 norm_adv_by_std=true 时生效）
ADV_STD_FLOOR_LIST=(
    "none"
    "auto"
    # "0.1"
)

# ── 固定参数 ────────────────────────────────────────────────────────────
LR="1e-5"
SEED="42"
TEACHER_REG="ema"
TEACHER_UPDATE_RATE="0.1"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
ROLLOUT_N="8"
INCLUDE_SUCCESSFUL_ROLLOUTS="True"
MODEL="Qwen3-8B"

# =============================================================================
# 提交循环
# =============================================================================
TOTAL=0
SUBMITTED=0

for DATASET in "${DATASETS[@]}"; do
for REWARD_TYPE in "${REWARD_TYPES[@]}"; do
for ENTROPY_GATE in "${ENTROPY_GATE_LIST[@]}"; do
for CLIP_ADV_VALUE in "${CLIP_ADV_VALUES[@]}"; do
for DISTILL_TOPK in "${DISTILL_TOPK_LIST[@]}"; do
for REPETITION_PENALTY in "${REPETITION_PENALTY_LIST[@]}"; do
for NORM_ADV_BY_STD in "${NORM_ADV_BY_STD_LIST[@]}"; do
for ADV_STD_FLOOR in "${ADV_STD_FLOOR_LIST[@]}"; do

    TOTAL=$((TOTAL + 1))

    # 构建短数据集名
    DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')

    # 构建模型短名
    MODEL_SHORT=$(echo "$MODEL" | tr '.' '-')
    OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
    MODEL_PATH="${OSS_ROOT}/base_models/${MODEL}"

    # ── 构建实验名 ───────────────────────────────────────────────────
    # entropy gate 标签
    if [ "$ENTROPY_GATE" = "none" ]; then
        ENTROPY_TAG="-noGate"
        TOPK_TAG=""  # noGate 时 topk 无意义，不显示
    else
        ENTROPY_TAG="-gate_${ENTROPY_GATE}"
        TOPK_TAG="-topk${DISTILL_TOPK}"
    fi

    # repetition penalty 标签
    REP_TAG="-rep${REPETITION_PENALTY}"

    # norm_adv_by_std 标签（含 floor 信息）
    if [ "$NORM_ADV_BY_STD" = "true" ]; then
        if [ "$ADV_STD_FLOOR" = "auto" ]; then
            STD_TAG="-std_auto"
        elif [ "$ADV_STD_FLOOR" = "none" ] || [ "$ADV_STD_FLOOR" = "0.0" ] || [ "$ADV_STD_FLOOR" = "0" ]; then
            STD_TAG="-normStd"  # 无 floor 保护，纯 std 归一化
        else
            STD_TAG="-std_${ADV_STD_FLOOR}"
        fi
    else
        STD_TAG=""
    fi

    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="TASD-simple-${DATASET_SHORT}-rt_${REWARD_TYPE}${ENTROPY_TAG}-clip${CLIP_ADV_VALUE}${TOPK_TAG}${REP_TAG}${STD_TAG}-${MODEL_SHORT}-${CURRENT_TIME}"

    # ── 提交 ────────────────────────────────────────────────────────
    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  REWARD_TYPE=$REWARD_TYPE ENTROPY_GATE=$ENTROPY_GATE"
        echo "  CLIP_ADV_VALUE=$CLIP_ADV_VALUE DISTILL_TOPK=$DISTILL_TOPK"
        echo "  REPETITION_PENALTY=$REPETITION_PENALTY NORM_ADV_BY_STD=$NORM_ADV_BY_STD ADV_STD_FLOOR=$ADV_STD_FLOOR"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
            --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL=${MODEL} --env=MODEL_PATH=${MODEL_PATH} --env=REWARD_TYPE=${REWARD_TYPE} --env=ENTROPY_GATE=${ENTROPY_GATE} --env=CLIP_ADV_VALUE=${CLIP_ADV_VALUE} --env=DISTILL_TOPK=${DISTILL_TOPK} --env=REPETITION_PENALTY=${REPETITION_PENALTY} --env=LR=${LR} --env=SEED=${SEED} --env=TEACHER_REG=${TEACHER_REG} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=INCLUDE_SUCCESSFUL_ROLLOUTS=${INCLUDE_SUCCESSFUL_ROLLOUTS} --env=NORM_ADV_BY_STD=${NORM_ADV_BY_STD} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR}" \
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

done
done
done
done
done
done
done
done

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
