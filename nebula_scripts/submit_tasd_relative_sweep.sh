#!/bin/bash
# =============================================================================
# TASD teacher_prob_relative 超参扫描 - Nebula 批量提交
#
# 目的：验证 teacher_prob_relative reward 是否能解决 entropy 崩溃
#   - reward 有正有负（log 差值），不系统性压缩 entropy
#   - 重点扫描 entropy_coeff（包括 0.0，验证 reward 本身是否稳定）
#   - 重点扫描 distill_topk（baseline 质量的关键）
#
# 使用方式：
#   bash nebula_scripts/submit_tasd_relative_sweep.sh [--dry-run]
# =============================================================================

# ── Nebula 账号配置 ──────────────────────────────────────────────────────
QUEUE="lazada_llm_ad_h20"
WORLD_SIZE=1
OPENLM_TOKEN="${OPENLM_TOKEN:?OPENLM_TOKEN not set}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:?OSS_ACCESS_ID not set}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?OSS_ACCESS_KEY not set}"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
CLUSTER_FILE="nebula_scripts/cluster_gpu_4.json"
SCRIPT_PATH="nebula_scripts/tasd/tasd_sciknoweval_parametric.sh"
PROJECT_NAME="TASD_param_search"

# ── dry-run 模式 ─────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 超参配置
#
# 核心思路：teacher_prob_relative reward 有正有负，不再需要大 entropy_coeff 对抗
#   reward = log(teacher_prob[sampled_token]) - log(mean(teacher_topk_probs))
#   - sampled token 比 teacher 期望更好 → 正 reward
#   - sampled token 比 teacher 期望更差 → 负 reward
#   - entropy 不再被系统性压缩
# =============================================================================
REWARD_TYPES=(
    "teacher_prob_relative"
)

LRS=(
    "1e-5"
)

ENTROPY_COEFF_LIST=(
    "0.0"    # 验证 teacher_prob_relative 本身是否能稳住 entropy（不需要额外 entropy 正则）
    "0.05"   # 加一点保险，对比 0.0
)

# distill_topk 决定 baseline 质量（teacher 期望概率的估计精度）
# 越大 baseline 越准，但计算量更大
DISTILL_TOPK_LIST=(
    "1000"   # 主力：覆盖 teacher 大部分概率质量，baseline 最准
    # "100"  # 备用：更快，baseline 质量略低
)

TEACHER_REGULARIZATION_LIST=(
    "ema"
)

TEACHER_UPDATE_RATE_LIST=(
    "0.1"    # 上轮实验最优
)

INCLUDE_SUCCESSFUL_ROLLOUTS_LIST=(
    "True"
)

# 固定参数
NORM_ADV_BY_STD="False"
CLIP_ADV="True"
CLIP_ADV_VALUE="5.0"
ROLLOUT_IS="token"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
ROLLOUT_N="8"
REPETITION_PENALTY="1.1"  # 抑制复读机

# =============================================================================
# 提交循环
# =============================================================================
TOTAL=0
SUBMITTED=0

for REWARD_TYPE in "${REWARD_TYPES[@]}"; do
for LR in "${LRS[@]}"; do
for ENTROPY_COEFF in "${ENTROPY_COEFF_LIST[@]}"; do
for DISTILL_TOPK in "${DISTILL_TOPK_LIST[@]}"; do
for TEACHER_REG in "${TEACHER_REGULARIZATION_LIST[@]}"; do
for TEACHER_UPDATE_RATE in "${TEACHER_UPDATE_RATE_LIST[@]}"; do
for INCLUDE_SUCCESSFUL_ROLLOUTS in "${INCLUDE_SUCCESSFUL_ROLLOUTS_LIST[@]}"; do

    # teacher_regularization=none 时 update_rate 无意义，只跑一次
    if [ "$TEACHER_REG" = "none" ] && [ "$TEACHER_UPDATE_RATE" != "${TEACHER_UPDATE_RATE_LIST[0]}" ]; then
        continue
    fi

    TOTAL=$((TOTAL + 1))

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

    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    LR_TAG=$(echo "$LR" | tr '-' '_')  # 把 lr 中的 - 替换成 _，便于按 - 分割
    JOB_NAME="TASD-bio-lr${LR_TAG}-rt${REWARD_TYPE}-topk${DISTILL_TOPK}-nostd-clip5.0${ENT_TAG}-rctoken${ISR_TAG}${EMA_TAG}-Qwen3-8B-${CURRENT_TIME}"

    # ── 提交 ────────────────────────────────────────────────────────
    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  REWARD_TYPE=$REWARD_TYPE  LR=$LR  ENTROPY=$ENTROPY_COEFF"
        echo "  DISTILL_TOPK=$DISTILL_TOPK"
        echo "  TEACHER_REG=$TEACHER_REG  UPDATE_RATE=$TEACHER_UPDATE_RATE"
        echo "  INCLUDE_SUCCESSFUL_ROLLOUTS=$INCLUDE_SUCCESSFUL_ROLLOUTS"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
                    --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=REWARD_TYPE=${REWARD_TYPE} --env=LR=${LR} --env=ENTROPY_COEFF=${ENTROPY_COEFF} --env=DISTILL_TOPK=${DISTILL_TOPK} --env=TEACHER_REG=${TEACHER_REG} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=NORM_ADV_BY_STD=${NORM_ADV_BY_STD} --env=CLIP_ADV=${CLIP_ADV} --env=CLIP_ADV_VALUE=${CLIP_ADV_VALUE} --env=ROLLOUT_IS=${ROLLOUT_IS} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=INCLUDE_SUCCESSFUL_ROLLOUTS=${INCLUDE_SUCCESSFUL_ROLLOUTS}" \
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

done  # INCLUDE_SUCCESSFUL_ROLLOUTS
done  # TEACHER_UPDATE_RATE
done  # TEACHER_REG
done  # DISTILL_TOPK
done  # ENTROPY_COEFF
done  # LR
done  # REWARD_TYPE

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
