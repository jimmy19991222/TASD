#!/bin/bash
# =============================================================================
# TASD 稳定性增强实验 - Nebula 批量提交
#
# 目标：测试 teacher_log_prob + 安全保护机制
# 借鉴 FIPO 的稳定性设计：
#   1. Safety Threshold：负样本保护，避免 over-penalization
#   2. Log-prob Clamp：数值稳定性
#   3. Entropy Bonus：防止 entropy 崩溃
#
# 使用方式：
#   bash nebula_scripts/submit_tasd_stability_sweep.sh [--dry-run]
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
SCRIPT_PATH="nebula_scripts/tasd/tasd_stability_parametric.sh"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="TASD_stability"

# ── 数据集配置 ──────────────────────────────────────────────────────
DATASETS=(
    "sciknoweval/biology"
)

# ── dry-run 模式 ─────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 稳定性实验超参配置
# =============================================================================

# ── Reward Type ─────────────────────────────────────────────────────
REWARD_TYPES=(
    "teacher_log_prob"         # token-level，走 TASD 路径，safety_thresh 生效
    "teacher_seq_log_prob"     # sentence-level，走 GRPO 路径，天然有正有负
    # "teacher_prob"           # 对比组
)

# ── Entropy Coefficient（防止 collapse）────────────────────────────────
ENTROPY_COEFF_LIST=(
    # "0.05"
    # "0.1"
    # "0.3"
    "0.0"    # 对照组：不设 entropy
)

# ── Safety Threshold（借鉴 FIPO：负样本保护）──────────────────────────
SAFETY_THRESH_LIST=(
    "0.1"      # reward < 0.1 且 adv < 0 时触发保护
    # "0.05"     # 更激进的阈值
    # "0.0"   # 对照组：禁用 safety threshold
)
SAFETY_CLIP_VALUE_LIST=(
    "0.5"      # 限制 advantage 到 [-0.5, 0]
    # "1.0"      # 更宽松的限制
)

# ── Advantage Clipping ──────────────────────────────────────────────────
CLIP_ADV_LIST=(
    "True"
)
CLIP_ADV_VALUE_LIST=(
    # "3.0"      # 更严格
    "5.0"
)

# ── Std 归一化 ──────────────────────────────────────────────────────────
NORM_ADV_BY_STD="True"
ADV_STD_FLOOR_LIST=(
    "auto"
)

# ── 固定参数 ────────────────────────────────────────────────────────────
LR="1e-5"
TEACHER_REG="ema"
TEACHER_UPDATE_RATE="0.1"
REPETITION_PENALTY="1.1"    # 增大复读抑制
ROLLOUT_IS="token"
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
for ENTROPY_COEFF in "${ENTROPY_COEFF_LIST[@]}"; do
for SAFETY_THRESH in "${SAFETY_THRESH_LIST[@]}"; do
for SAFETY_CLIP_VALUE in "${SAFETY_CLIP_VALUE_LIST[@]}"; do
for CLIP_ADV in "${CLIP_ADV_LIST[@]}"; do
for CLIP_ADV_VALUE in "${CLIP_ADV_VALUE_LIST[@]}"; do
for ADV_STD_FLOOR in "${ADV_STD_FLOOR_LIST[@]}"; do

    TOTAL=$((TOTAL + 1))

    # 构建短数据集名
    DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')

    # 构建模型短名
    MODEL_SHORT=$(echo "$MODEL" | tr '.' '-')
    OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
    MODEL_PATH="${OSS_ROOT}/base_models/${MODEL}"

    # ── 构建实验名 ───────────────────────────────────────────────────
    ENT_TAG="-ent${ENTROPY_COEFF}"
    
    # Safety threshold 标签
    if [ "$SAFETY_THRESH" = "0.0" ] || [ "$SAFETY_THRESH" = "0" ]; then
        SAFETY_TAG="-noSafe"
    else
        SAFETY_TAG="-safe${SAFETY_THRESH}_${SAFETY_CLIP_VALUE}"
    fi

    # Std 标签
    if [ "$ADV_STD_FLOOR" = "auto" ]; then
        STD_TAG="-std_auto"
    else
        STD_TAG="-std_${ADV_STD_FLOOR}"
    fi

    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    LR_TAG=$(echo "$LR" | tr '-' '_')
    JOB_NAME="TASD-stable-${DATASET_SHORT}-lr${LR_TAG}-rt${REWARD_TYPE}${STD_TAG}-clip${CLIP_ADV_VALUE}${ENT_TAG}${SAFETY_TAG}-${MODEL_SHORT}-${CURRENT_TIME}"

    # ── 提交 ────────────────────────────────────────────────────────
    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  REWARD_TYPE=$REWARD_TYPE ENTROPY=$ENTROPY_COEFF"
        echo "  SAFETY_THRESH=$SAFETY_THRESH SAFETY_CLIP=$SAFETY_CLIP_VALUE"
        echo "  CLIP_ADV=$CLIP_ADV CLIP_ADV_VALUE=$CLIP_ADV_VALUE"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
                    --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL=${MODEL} --env=MODEL_PATH=${MODEL_PATH} --env=REWARD_TYPE=${REWARD_TYPE} --env=LR=${LR} --env=ENTROPY_COEFF=${ENTROPY_COEFF} --env=TEACHER_REG=${TEACHER_REG} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=NORM_ADV_BY_STD=${NORM_ADV_BY_STD} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR} --env=CLIP_ADV=${CLIP_ADV} --env=CLIP_ADV_VALUE=${CLIP_ADV_VALUE} --env=SAFETY_THRESH=${SAFETY_THRESH} --env=SAFETY_CLIP_VALUE=${SAFETY_CLIP_VALUE} --env=REPETITION_PENALTY=${REPETITION_PENALTY} --env=ROLLOUT_IS=${ROLLOUT_IS} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=INCLUDE_SUCCESSFUL_ROLLOUTS=${INCLUDE_SUCCESSFUL_ROLLOUTS}" \
            --worker_count=${WORLD_SIZE} \
            --file.cluster_file=${CLUSTER_FILE} \
            --job_name=${JOB_NAME} \
            --access_id=${OSS_ACCESS_ID} \
            --access_key=${OSS_ACCESS_KEY} \
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

done  # ADV_STD_FLOOR
done  # CLIP_ADV_VALUE
done  # CLIP_ADV
done  # SAFETY_CLIP_VALUE
done  # SAFETY_THRESH
done  # ENTROPY_COEFF
done  # REWARD_TYPE
done  # DATASET

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
