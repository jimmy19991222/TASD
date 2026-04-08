#!/bin/bash
# =============================================================================
# TASD + Future-KL Modulation 实验 - Nebula 批量提交
#
# 测试 Future-KL Modulation 组件对 TASD 训练的影响
# 组件默认关闭，通过 FUTURE_KL_ENABLED=True 启用
#
# 使用方式：
#   bash nebula_scripts/submit_tasd_future_kl_sweep.sh [--dry-run]
# =============================================================================

# ── Nebula 账号配置 ──────────────────────────────────────────────────────
QUEUE="lazada_llm_ad_h20"
WORLD_SIZE=1
OPENLM_TOKEN="${OPENLM_TOKEN:?OPENLM_TOKEN not set}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:?OSS_ACCESS_ID not set}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?OSS_ACCESS_KEY not set}"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
CLUSTER_FILE="nebula_scripts/cluster_gpu_4.json"    # 4 GPU
SCRIPT_PATH="nebula_scripts/tasd/tasd_sciknoweval_parametric.sh"
# 自定义镜像
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="TASD_FutureKL"

# ── 数据集配置 ──────────────────────────────────────────────────────
DATASETS=(
    "sciknoweval/biology"
    # "sciknoweval/chemistry"
)

# ── dry-run 模式 ─────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 基础超参配置（继承自 TASD EMA sweep）
# =============================================================================
REWARD_TYPES=(
    "teacher_prob"
    # "teacher_log_prob"
)
LRS=(
    "1e-5"
)
ENTROPY_COEFF_LIST=(
    "0.0"
)
TEACHER_REGULARIZATION_LIST=(
    "ema"
)
TEACHER_UPDATE_RATE_LIST=(
    "0.1"
)

# ── 固定参数 ──────────────────────────────────────────────────────
NORM_ADV_BY_STD="True"
ADV_STD_FLOOR="none"
CLIP_ADV="False"
CLIP_ADV_VALUE="None"
REPETITION_PENALTY="1.0"
ROLLOUT_IS="token"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
ROLLOUT_N="8"
INCLUDE_SUCCESSFUL_ROLLOUTS="True"

# ── 模型配置 ──────────────────────────────────────────────────────
MODELS=(
    "Qwen3-8B"
)

# =============================================================================
# Future-KL Modulation 参数（核心实验变量）
# =============================================================================
# 是否启用 Future-KL Modulation
FUTURE_KL_ENABLED_LIST=(
    # "False"     # 默认关闭（baseline）
    "True"      # 启用
)

# 信号类型（仅当 enabled=True 时有效）
FUTURE_KL_SIGNAL_TYPE_LIST=(
    "teacher_student_kl"    # KL(π_teacher ∥ π_student)，推荐
    # "teacher_prob"          # π_teacher(y_t)
    # "teacher_student_diff"  # π_teacher - π_student (clamp≥0)
    # "policy_change"         # FIPO 原版（需要 old_log_probs）
)

# 衰减率：gamma = 2^{-1/decay_rate}
# decay_rate=128 -> gamma≈0.9946（128 tokens 后衰减到 0.5）
# decay_rate=64  -> gamma≈0.9892（64 tokens 后衰减到 0.5）
# decay_rate=256 -> gamma≈0.9973（256 tokens 后衰减到 0.5）
FUTURE_KL_DECAY_RATE_LIST=(
    "128"
    # "64"
    # "256"
)

# 裁剪比例：weight ∈ [1-ratio, 1+ratio]
# clip_ratio=0.5 -> weight ∈ [0.5, 1.5]
# clip_ratio=0.3 -> weight ∈ [0.7, 1.3]
# clip_ratio=1.0 -> weight ∈ [0.0, 2.0]
FUTURE_KL_CLIP_RATIO_LIST=(
    "0.5"
    # "0.3"
    # "1.0"
)

# =============================================================================
# 提交循环
# =============================================================================
TOTAL=0
SUBMITTED=0

for DATASET in "${DATASETS[@]}"; do
for MODEL in "${MODELS[@]}"; do
for REWARD_TYPE in "${REWARD_TYPES[@]}"; do
for LR in "${LRS[@]}"; do
for ENTROPY_COEFF in "${ENTROPY_COEFF_LIST[@]}"; do
for TEACHER_REG in "${TEACHER_REGULARIZATION_LIST[@]}"; do
for TEACHER_UPDATE_RATE in "${TEACHER_UPDATE_RATE_LIST[@]}"; do
for FUTURE_KL_ENABLED in "${FUTURE_KL_ENABLED_LIST[@]}"; do
for FUTURE_KL_SIGNAL_TYPE in "${FUTURE_KL_SIGNAL_TYPE_LIST[@]}"; do
for FUTURE_KL_DECAY_RATE in "${FUTURE_KL_DECAY_RATE_LIST[@]}"; do
for FUTURE_KL_CLIP_RATIO in "${FUTURE_KL_CLIP_RATIO_LIST[@]}"; do

    TOTAL=$((TOTAL + 1))

    # 当 Future-KL 关闭时，跳过不同的 signal_type/decay_rate/clip_ratio 组合
    # 只跑一次 baseline
    if [ "$FUTURE_KL_ENABLED" = "False" ]; then
        # 检查是否是第一个组合（只跑一次 baseline）
        if [ "$FUTURE_KL_SIGNAL_TYPE" != "${FUTURE_KL_SIGNAL_TYPE_LIST[0]}" ] || \
           [ "$FUTURE_KL_DECAY_RATE" != "${FUTURE_KL_DECAY_RATE_LIST[0]}" ] || \
           [ "$FUTURE_KL_CLIP_RATIO" != "${FUTURE_KL_CLIP_RATIO_LIST[0]}" ]; then
            TOTAL=$((TOTAL - 1))
            continue
        fi
    fi

    # 构建短数据集名
    DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
    MODEL_SHORT=$(echo "$MODEL" | tr '.' '-')

    # 模型路径
    OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
    MODEL_PATH="${OSS_ROOT}/base_models/${MODEL}"

    # ── 构建实验名 ───────────────────────────────────────────────────
    ENT_TAG=""
    if [ "$(echo "$ENTROPY_COEFF > 0" | bc -l)" = "1" ]; then
        ENT_TAG="-ent${ENTROPY_COEFF}"
    fi
    EMA_TAG=""
    if [ "$TEACHER_REG" = "ema" ]; then
        EMA_TAG="-ema${TEACHER_UPDATE_RATE}"
    fi
    STD_TAG="-std_none"
    if [ "$NORM_ADV_BY_STD" = "True" ]; then
        if [ "$ADV_STD_FLOOR" = "auto" ]; then
            STD_TAG="-std_auto"
        elif [ "$ADV_STD_FLOOR" != "none" ] && [ "$ADV_STD_FLOOR" != "0.0" ]; then
            STD_TAG="-std_floor${ADV_STD_FLOOR}"
        fi
    fi

    # Future-KL 标签
    FKL_TAG=""
    if [ "$FUTURE_KL_ENABLED" = "True" ]; then
        # 简化 signal_type 名称
        SIGNAL_SHORT=$(echo "$FUTURE_KL_SIGNAL_TYPE" | sed 's/teacher_student_kl/tskl/g' | sed 's/teacher_prob/tprob/g' | sed 's/teacher_student_diff/tsdiff/g' | sed 's/policy_change/pc/g')
        FKL_TAG="-fkl_${SIGNAL_SHORT}_d${FUTURE_KL_DECAY_RATE}_c${FUTURE_KL_CLIP_RATIO}"
    else
        FKL_TAG="-fkl_off"
    fi

    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    LR_TAG=$(echo "$LR" | tr '-' '_')
    JOB_NAME="TASD-FKL-${DATASET_SHORT}-lr${LR_TAG}-rt${REWARD_TYPE}${STD_TAG}${ENT_TAG}${EMA_TAG}${FKL_TAG}-${MODEL_SHORT}-${CURRENT_TIME}"

    # ── 提交 ────────────────────────────────────────────────────────
    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  REWARD_TYPE=$REWARD_TYPE LR=$LR ENTROPY=$ENTROPY_COEFF"
        echo "  TEACHER_REG=$TEACHER_REG UPDATE_RATE=$TEACHER_UPDATE_RATE"
        echo "  Future-KL: ENABLED=$FUTURE_KL_ENABLED SIGNAL=$FUTURE_KL_SIGNAL_TYPE DECAY=$FUTURE_KL_DECAY_RATE CLIP=$FUTURE_KL_CLIP_RATIO"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
                    --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL=${MODEL} --env=MODEL_PATH=${MODEL_PATH} --env=REWARD_TYPE=${REWARD_TYPE} --env=LR=${LR} --env=ENTROPY_COEFF=${ENTROPY_COEFF} --env=TEACHER_REG=${TEACHER_REG} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=NORM_ADV_BY_STD=${NORM_ADV_BY_STD} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR} --env=CLIP_ADV=${CLIP_ADV} --env=CLIP_ADV_VALUE=${CLIP_ADV_VALUE} --env=REPETITION_PENALTY=${REPETITION_PENALTY} --env=ROLLOUT_IS=${ROLLOUT_IS} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=INCLUDE_SUCCESSFUL_ROLLOUTS=${INCLUDE_SUCCESSFUL_ROLLOUTS} --env=FUTURE_KL_ENABLED=${FUTURE_KL_ENABLED} --env=FUTURE_KL_SIGNAL_TYPE=${FUTURE_KL_SIGNAL_TYPE} --env=FUTURE_KL_DECAY_RATE=${FUTURE_KL_DECAY_RATE} --env=FUTURE_KL_CLIP_RATIO=${FUTURE_KL_CLIP_RATIO}" \
            --worker_count=${WORLD_SIZE} \
            --file.cluster_file=${CLUSTER_FILE} \
            --job_name=${JOB_NAME} \
            --access_id=${access_id} \
            --access_key=${access_key} \
            --env=OPENLM_TOKEN=${OPENLM_TOKEN} \
            --env=SWANLAB_API_KEY=${SWANLAB_API_KEY} \
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

done  # FUTURE_KL_CLIP_RATIO
done  # FUTURE_KL_DECAY_RATE
done  # FUTURE_KL_SIGNAL_TYPE
done  # FUTURE_KL_ENABLED
done  # TEACHER_UPDATE_RATE
done  # TEACHER_REG
done  # ENTROPY_COEFF
done  # LR
done  # REWARD_TYPE
done  # MODEL
done  # DATASET

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
