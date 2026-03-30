#!/bin/bash
# =============================================================================
# [P0] TASD 可行性验证 - 验证 teacher_log_prob / teacher_seq_log_prob 能正常驱动学习
#
# 目标：确认 reward 有方差、pg_loss 不为 0，是后续实验的前提
#
# 跑 2 个 job：
#   1. teacher_log_prob  + ent0.0 + std + ema0.1 + isr1
#   2. teacher_seq_log_prob + ent0.0 + std + ema0.1 + isr1
#
# 监控指标（训练开始后 ~10 steps）：
#   - critic/rewards/mean 应为负值（teacher_log_prob）或负值广播（teacher_seq_log_prob）
#   - actor/pg_loss 应有明显非零值
#
# 使用方式：bash nebula_scripts/submit_tasd_p0_feasibility.sh [--dry-run]
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

# ── dry-run ───────────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# [P0] 实验列表：只跑两个 reward type，ent=0.0 排除 entropy 干扰
# =============================================================================
REWARD_TYPES=(
    "teacher_prob" 
    # "teacher_log_prob" 
    # "teacher_seq_log_prob"
)

# 固定参数
LR="1e-5"
ENTROPY_COEFF="0.1"
TEACHER_REG="ema"
TEACHER_UPDATE_RATE="0.1"
NORM_ADV_BY_STD="True"
CLIP_ADV="True"
CLIP_ADV_VALUE="5.0"
ROLLOUT_IS="token"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
ROLLOUT_N="8"
INCLUDE_SUCCESSFUL_ROLLOUTS="True"

# =============================================================================
TOTAL=0
SUBMITTED=0

for REWARD_TYPE in "${REWARD_TYPES[@]}"; do

    TOTAL=$((TOTAL + 1))
    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="TASD-bio-lr${LR}-rt${REWARD_TYPE}-std-clip5.0-rctoken-isr1-ema${TEACHER_UPDATE_RATE}-Qwen3-8B-${CURRENT_TIME}"

    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  REWARD_TYPE=$REWARD_TYPE  ENTROPY_COEFF=$ENTROPY_COEFF"
        echo "  TEACHER_REG=$TEACHER_REG  UPDATE_RATE=$TEACHER_UPDATE_RATE"
        echo "  NORM_ADV_BY_STD=$NORM_ADV_BY_STD  ISR=$INCLUDE_SUCCESSFUL_ROLLOUTS"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
                --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=REWARD_TYPE=${REWARD_TYPE} --env=LR=${LR} --env=ENTROPY_COEFF=${ENTROPY_COEFF} --env=TEACHER_REG=${TEACHER_REG} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=NORM_ADV_BY_STD=${NORM_ADV_BY_STD} --env=CLIP_ADV=${CLIP_ADV} --env=CLIP_ADV_VALUE=${CLIP_ADV_VALUE} --env=ROLLOUT_IS=${ROLLOUT_IS} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=INCLUDE_SUCCESSFUL_ROLLOUTS=${INCLUDE_SUCCESSFUL_ROLLOUTS}" \
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

done  # REWARD_TYPE

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
