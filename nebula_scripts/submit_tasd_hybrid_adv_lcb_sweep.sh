#!/bin/bash
# =============================================================================
# TASD teacher_prob_plus_sentence Hybrid Advantage 扫描 - LCB 数据集
#
# 实验目的：在 LCB 数据集上对比不同 alpha 的混合 advantage 效果
#   A_t = alpha * A_sent + (1-alpha) * A_delta
#   alpha=1.0 → 退化为纯 teacher_sentence_prob (GRPO 路径)
#   alpha=0.0 → 纯 within-response token delta
#   alpha=0.25/0.5/0.75 → 混合
#
# 参考 submit_tasd_hybrid_adv_sweep.sh 和 submit_tasd_lcb_sweep.sh
# 使用方式：
#   bash nebula_scripts/submit_tasd_hybrid_adv_lcb_sweep.sh [--dry-run]
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
SCRIPT_PATH="nebula_scripts/tasd/tasd_lcb_parametric.sh"
# 自定义镜像（留空则使用 --algo_name=pytorch260 默认镜像）
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="TASD_hybrid_adv_lcb"

# ── dry-run 模式 ─────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 超参配置
# =============================================================================

# 固定 reward_type = teacher_prob_plus_sentence
REWARD_TYPE="teacher_prob_plus_sentence"

# 核心扫描维度：alpha
ALPHA_LIST=(
    "0.25"
    "0.5"
    "0.75"
    "0.0"   # 纯 token-delta（不走 GRPO）
    "1.0"   # 纯 sentence（等价于 teacher_sentence_prob，作为基线可单独对比）
)

# 固定超参（参考 submit_tasd_lcb_sweep.sh）
LR="1e-6"
ENTROPY_COEFF="0.0"
TEACHER_REG="ema"
TEACHER_UPDATE_RATE="0.1"
NORM_ADV_BY_STD="True"
ADV_STD_FLOOR="auto"     # std下界：auto=1/sqrt(n)
CLIP_ADV="False"
CLIP_ADV_VALUE="None"
REPETITION_PENALTY="1.05"
ROLLOUT_IS="token"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="8"      # LCB 用 8
ROLLOUT_N="8"
INCLUDE_SUCCESSFUL_ROLLOUTS="True"

# ── 模型配置 ──────────────────────────────────────────────────────
MODEL_NAMES=(
    "Qwen3-8B"
    # "Qwen3.5-4B"
    # "Qwen3.5-9B"
)

# =============================================================================
# 提交循环
# =============================================================================
TOTAL=0
SUBMITTED=0

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
for ALPHA in "${ALPHA_LIST[@]}"; do

    TOTAL=$((TOTAL + 1))

    # 构建模型短名
    MODEL_SHORT=$(echo "$MODEL_NAME" | tr '.' '-')

    # ── 构建实验名 ───────────────────────────────────────────────────
    # alpha 转为标签（去掉小数点）
    ALPHA_TAG=$(echo "$ALPHA" | tr -d '.')
    LR_TAG=$(echo "$LR" | tr '-' '_')  # 把 lr 中的 - 替换成 _，便于按 - 分割
    STD_TAG="-std_auto"   # NORM_ADV_BY_STD=True + ADV_STD_FLOOR=auto 固定
    EMA_TAG="-ema${TEACHER_UPDATE_RATE}"

    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="TASD-hybrid-lcb-lr${LR_TAG}-alpha${ALPHA_TAG}${STD_TAG}-ema${TEACHER_UPDATE_RATE}-rep${REPETITION_PENALTY}-${MODEL_SHORT}-${CURRENT_TIME}"

    # ── 提交 ────────────────────────────────────────────────────────
    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  MODEL_NAME=$MODEL_NAME"
        echo "  REWARD_TYPE=$REWARD_TYPE ALPHA=$ALPHA"
        echo "  LR=$LR TEACHER_REG=$TEACHER_REG UPDATE_RATE=$TEACHER_UPDATE_RATE"
        echo "  NORM_ADV_BY_STD=$NORM_ADV_BY_STD ADV_STD_FLOOR=$ADV_STD_FLOOR"
        echo "  MINI_BATCH_SIZE=$MINI_BATCH_SIZE"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
                    --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=MODEL_NAME=${MODEL_NAME} --env=REWARD_TYPE=${REWARD_TYPE} --env=TASD_ALPHA=${ALPHA} --env=LR=${LR} --env=ENTROPY_COEFF=${ENTROPY_COEFF} --env=TEACHER_REG=${TEACHER_REG} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=NORM_ADV_BY_STD=${NORM_ADV_BY_STD} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR} --env=CLIP_ADV=${CLIP_ADV} --env=CLIP_ADV_VALUE=${CLIP_ADV_VALUE} --env=REPETITION_PENALTY=${REPETITION_PENALTY} --env=ROLLOUT_IS=${ROLLOUT_IS} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=INCLUDE_SUCCESSFUL_ROLLOUTS=${INCLUDE_SUCCESSFUL_ROLLOUTS}" \
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

done  # ALPHA
done  # MODEL_NAME

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
