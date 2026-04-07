#!/bin/bash
# =============================================================================
# TASD distill_temperature sweep
#
# 基础配置：teacher_prob + std + clip1.0 + rctoken + isr1 + ema0.1
# 参考实验：TASD-bio-lr1e-5-rtteacher_prob-std-clip1.0-rctoken-isr1-ema0.1-Qwen3-8B-20260327_211649
#
# 目标：验证提高 teacher forward temperature 能否增大 reward 方差，
#       避免 std 归一化时 advantage 爆炸（critic/advantages/min 被 clip 到 -5）
#
# 扫描参数：
#   DISTILL_TEMPERATURE ∈ {2.0, 3.0, 5.0}
#   （1.0 = 原始对照，复用已有实验）
#
# 使用方式：bash nebula_scripts/submit_tasd_distill_temp_sweep.sh [--dry-run]
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
# 固定参数（与对照实验 clip1.0 保持一致）
# =============================================================================
REWARD_TYPE="teacher_prob"
LR="1e-5"
ENTROPY_COEFF="0.0"
TEACHER_REG="ema"
TEACHER_UPDATE_RATE="0.1"
NORM_ADV_BY_STD="True"
CLIP_ADV="True"
CLIP_ADV_VALUE="2.0"
ROLLOUT_IS="token"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
ROLLOUT_N="8"
INCLUDE_SUCCESSFUL_ROLLOUTS="True"
DISTILL_TOPK="100"

# 扫描的 temperature 档位（1.0=对照已有，不重复提交）
DISTILL_TEMPERATURES=("0.5" "2.0" "3.0")

# =============================================================================
TOTAL=0
SUBMITTED=0

for DISTILL_TEMPERATURE in "${DISTILL_TEMPERATURES[@]}"; do

    TOTAL=$((TOTAL + 1))
    LR_TAG=$(echo "$LR" | tr '-' '_')  # 把 lr 中的 - 替换成 _，便于按 - 分割
    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="TASD-bio-lr${LR_TAG}-rtteacher_prob-std-clip${CLIP_ADV_VALUE}-dtemp${DISTILL_TEMPERATURE}-rctoken-isr1-ema${TEACHER_UPDATE_RATE}-Qwen3-8B-${CURRENT_TIME}"

    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  REWARD_TYPE=$REWARD_TYPE  LR=$LR  ENTROPY_COEFF=$ENTROPY_COEFF"
        echo "  DISTILL_TEMPERATURE=$DISTILL_TEMPERATURE  DISTILL_TOPK=$DISTILL_TOPK"
        echo "  NORM_ADV_BY_STD=$NORM_ADV_BY_STD  CLIP_ADV_VALUE=$CLIP_ADV_VALUE"
        echo "  TEACHER_REG=$TEACHER_REG  UPDATE_RATE=$TEACHER_UPDATE_RATE"
        echo "  INCLUDE_SUCCESSFUL_ROLLOUTS=$INCLUDE_SUCCESSFUL_ROLLOUTS"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
                --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=REWARD_TYPE=${REWARD_TYPE} --env=LR=${LR} --env=ENTROPY_COEFF=${ENTROPY_COEFF} --env=DISTILL_TOPK=${DISTILL_TOPK} --env=DISTILL_TEMPERATURE=${DISTILL_TEMPERATURE} --env=TEACHER_REG=${TEACHER_REG} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=NORM_ADV_BY_STD=${NORM_ADV_BY_STD} --env=CLIP_ADV=${CLIP_ADV} --env=CLIP_ADV_VALUE=${CLIP_ADV_VALUE} --env=ROLLOUT_IS=${ROLLOUT_IS} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=INCLUDE_SUCCESSFUL_ROLLOUTS=${INCLUDE_SUCCESSFUL_ROLLOUTS}" \
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

done  # DISTILL_TEMPERATURE

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
