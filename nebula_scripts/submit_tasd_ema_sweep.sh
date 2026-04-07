#!/bin/bash
# =============================================================================
# TASD EMA Teacher 超参扫描 - Nebula 批量提交
#
# 参考 experiments/generalization/run_tasd_ema_teacher_local.sh 的超参配置
# 每组超参提交一个独立的 Nebula job（4 GPU / 节点）
#
# 使用方式：
#   bash nebula_scripts/submit_tasd_ema_sweep.sh [--dry-run]
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
# 自定义镜像（留空则使用 --algo_name=pytorch260 默认镜像）
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="TASD_param_search"

# ── 数据集配置 ──────────────────────────────────────────────────────
DATASETS=(
    "sciknoweval/biology"
    # "sciknoweval/chemistry"
    # "sciknoweval/material"
    # "sciknoweval/physics"
    # "tooluse"
)

# ── dry-run 模式 ─────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 超参配置
# 对比三种 reward type：teacher_log_prob / teacher_seq_log_prob / teacher_prob
# =============================================================================
REWARD_TYPES=(
    # "teacher_log_prob" x
    # "teacher_seq_log_prob" x
    "teacher_prob"
    # "teacher_sentence_prob"
    # "teacher_prob_binary" x
    # "top1_match" x
    # "teacher_prob_plus_verified"
    # "teacher_prob_relative" x
    # "teacher_prob_certainty"
    # "student_topk_teacher_prob" x
    # "student_topk_teacher_prob_weighted" x
    # "teacher_prob_diff_weighted" x
)
LRS=(
    "1e-5"
)
ENTROPY_COEFF_LIST=(
    "0.0"
    # "0.01"
    # "0.05"   # teacher_log_prob / teacher_prob 用 0.05
    # "0.1"
    # "1.0"    # 训练崩溃
)
TEACHER_REGULARIZATION_LIST=(
    "ema"
    # "none"
)
TEACHER_UPDATE_RATE_LIST=(
    # "0.3"
    # "0.2"
    "0.1"
    # "0.05"
    # "0.0"
)

# ── 扫描参数 ──────────────────────────────────────────────────────
NORM_ADV_BY_STD="True"   # 固定开启 std 归一化
ADV_STD_FLOOR_LIST=(
    "auto"     # auto=1/sqrt(n)
    # "none"     # 不使用下界
    # "0.1"
    # "0.5"
)
CLIP_ADV_LIST=(
    "False"
    # "True"
)
CLIP_ADV_VALUE_LIST=(
    "None"
    # "2"
    # "5"
)

# 固定参数（不扫描）
REPETITION_PENALTY="1.05"   # 复读抑制，防止entropy崩溃时产生超长重复序列
ROLLOUT_IS="token"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
ROLLOUT_N="8"
INCLUDE_SUCCESSFUL_ROLLOUTS_LIST=(
    "True"
    # "False"
)

# ── 模型配置 ──────────────────────────────────────────────────────
MODELS=(
    "Qwen3-8B"
    # "Qwen3.5-4B"
    # "Qwen3.5-9B"
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
for INCLUDE_SUCCESSFUL_ROLLOUTS in "${INCLUDE_SUCCESSFUL_ROLLOUTS_LIST[@]}"; do
for ADV_STD_FLOOR in "${ADV_STD_FLOOR_LIST[@]}"; do
for CLIP_ADV in "${CLIP_ADV_LIST[@]}"; do
for CLIP_ADV_VALUE in "${CLIP_ADV_VALUE_LIST[@]}"; do

    # teacher_regularization=none 时 update_rate 无意义，只跑一次
    # if [ "$TEACHER_REG" = "none" ] && [ "$TEACHER_UPDATE_RATE" != "${TEACHER_UPDATE_RATE_LIST[0]}" ]; then
    #     continue
    # fi

    # # teacher_seq_log_prob 用 ent1.0，其他 reward type 用 ent0.05
    # if [ "$REWARD_TYPE" = "teacher_seq_log_prob" ] && [ "$ENTROPY_COEFF" != "1.0" ]; then
    #     continue
    # fi
    # if [ "$REWARD_TYPE" != "teacher_seq_log_prob" ] && [ "$ENTROPY_COEFF" != "0.05" ]; then
    #     continue
    # fi

    TOTAL=$((TOTAL + 1))

    # 构建短数据集名（用于实验名）
    DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')

    # tooluse 使用 teacher_sentence_prob（seq-level GRPO 路径），其他数据集使用配置的 REWARD_TYPE
    # 原因：tooluse 是序列级正确/错误任务，token-level teacher_prob 方差极小（std≈0.027）无区分性
    #       teacher_sentence_prob = exp(mean log_prob) ∈ (0,1]，走 GRPO sequence-level 归一化，天然有正有负
    EFFECTIVE_REWARD_TYPE="${REWARD_TYPE}"
    # if [ "$DATASET" = "tooluse" ]; then
    #     EFFECTIVE_REWARD_TYPE="teacher_sentence_prob"
    # fi

    # 构建模型短名（去掉点，用于实验名）
    MODEL_SHORT=$(echo "$MODEL" | tr '.' '-')
    # 构建模型路径（和 Qwen3-8B 同目录规范）
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

    ISR_TAG="-isr1"
    if [ "$INCLUDE_SUCCESSFUL_ROLLOUTS" = "False" ]; then
        ISR_TAG="-isr0"
    fi

    STD_TAG="-no_std"
    if [ "$NORM_ADV_BY_STD" = "True" ]; then
        if [ "$ADV_STD_FLOOR" = "auto" ]; then
            STD_TAG="-std_auto"
        elif [ "$ADV_STD_FLOOR" != "0.0" ] && [ "$ADV_STD_FLOOR" != "0" ] && [ "$ADV_STD_FLOOR" != "none" ]; then
            STD_TAG="-std_floor${ADV_STD_FLOOR}"
        else
            STD_TAG="-std_none"
        fi
    fi

    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    REP_TAG=""
    if [ "$REPETITION_PENALTY" != "1.0" ] && [ "$REPETITION_PENALTY" != "1" ]; then
        REP_TAG="-rep${REPETITION_PENALTY}"
    fi
    LR_TAG=$(echo "$LR" | tr '-' '_')  # 把 lr 中的 - 替换成 _，便于按 - 分割
    JOB_NAME="TASD-${DATASET_SHORT}-lr${LR_TAG}-rt${EFFECTIVE_REWARD_TYPE}${STD_TAG}-clip${CLIP_ADV_VALUE}${ENT_TAG}-rctoken${ISR_TAG}${EMA_TAG}${REP_TAG}-${MODEL_SHORT}-${CURRENT_TIME}"

    # ── 提交 ────────────────────────────────────────────────────────
    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  REWARD_TYPE=$REWARD_TYPE LR=$LR ENTROPY=$ENTROPY_COEFF"
        echo "  TEACHER_REG=$TEACHER_REG UPDATE_RATE=$TEACHER_UPDATE_RATE"
        echo "  ADV_STD_FLOOR=$ADV_STD_FLOOR CLIP_ADV=$CLIP_ADV CLIP_ADV_VALUE=$CLIP_ADV_VALUE"
        echo "  INCLUDE_SUCCESSFUL_ROLLOUTS=$INCLUDE_SUCCESSFUL_ROLLOUTS"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
                    --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL=${MODEL} --env=MODEL_PATH=${MODEL_PATH} --env=REWARD_TYPE=${EFFECTIVE_REWARD_TYPE} --env=LR=${LR} --env=ENTROPY_COEFF=${ENTROPY_COEFF} --env=TEACHER_REG=${TEACHER_REG} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=NORM_ADV_BY_STD=${NORM_ADV_BY_STD} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR} --env=CLIP_ADV=${CLIP_ADV} --env=CLIP_ADV_VALUE=${CLIP_ADV_VALUE} --env=REPETITION_PENALTY=${REPETITION_PENALTY} --env=ROLLOUT_IS=${ROLLOUT_IS} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=INCLUDE_SUCCESSFUL_ROLLOUTS=${INCLUDE_SUCCESSFUL_ROLLOUTS}" \
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
        sleep 2    # 避免提交过快被限流
    fi

done  # CLIP_ADV_VALUE
done  # CLIP_ADV
done  # ADV_STD_FLOOR
done  # INCLUDE_SUCCESSFUL_ROLLOUTS
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
