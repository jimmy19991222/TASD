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
# 自定义镜像（留空则使用 --algo_name=pytorch260 默认镜像）
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
CLUSTER_FILE="nebula_scripts/cluster_gpu_4.json"    # 4 GPU
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
# 上轮实验发现 entropy_coeff=0.1 时所有 TASD job entropy 极速崩溃至 ~0.02
# 本轮大幅提升 entropy_coeff，并加入 none/ema 对照 + 更多 update_rate
# =============================================================================
REWARD_TYPES=(
    # "teacher_prob"
    # "log_teacher_prob"
    # "top1_match"
    # "student_topk_teacher_prob"           # student topk 位置上 teacher prob 均值，更平滑
    # "student_topk_teacher_prob_weighted"  # student prob 加权的 teacher prob，双向对称
    # "teacher_prob_kl_weighted"            # teacher认可度 × KL(teacher∥student)，分歧大才给强信号，有平衡点
    "teacher_prob_jsd_weighted"           # teacher认可度 × JSD(teacher,student)，对称分歧，天然有界[0,log2]
    "teacher_prob_diff_weighted"          # teacher认可度 × (teacher_prob-student_prob).clamp(0)，最简单，无需topk
)
LRS=(
    "1e-5"
    # "5e-6"
)
ENTROPY_COEFF_LIST=(
    # "0.0"
    # "0.01"
    # "0.03"
    "0.05"
    # "0.1"   # 上轮实验：entropy 早期崩溃（~0.016），val mean@16 < 0.41，效果差
    # "0.5"    # 本轮：大幅提升，防止早期 entropy 崩溃
    # "1.0"    # 本轮：激进版，进一步抑制坍缩
)
TEACHER_REGULARIZATION_LIST=(
    # "none"   # 先跑无 EMA 的干净基线，排除 EMA 的影响
    "ema"
)
TEACHER_UPDATE_RATE_LIST=(
    # "0.0"    # EMA 极慢更新（teacher ≈ 初始模型）
    # "0.05"   # EMA 适中更新
    "0.1"    # 上轮实验 step45 最佳（0.476 vs ema0.0 的 0.361），加回
)

# 固定参数（不扫描）
NORM_ADV_BY_STD="False"
CLIP_ADV="True"
CLIP_ADV_VALUE="5.0"
ROLLOUT_IS="token"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
ROLLOUT_N="8"
INCLUDE_SUCCESSFUL_ROLLOUTS_LIST=(
    "True"   # 成功rollout也参与TASD reward
    # "False"  # 只让失败rollout参与TASD reward，信号更干净
)

# =============================================================================
# 提交循环
# =============================================================================
TOTAL=0
SUBMITTED=0

for REWARD_TYPE in "${REWARD_TYPES[@]}"; do
for LR in "${LRS[@]}"; do
for ENTROPY_COEFF in "${ENTROPY_COEFF_LIST[@]}"; do
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
    JOB_NAME="TASD-bio-lr${LR}-rt${REWARD_TYPE}-nostd-clip5.0${ENT_TAG}-rctoken${ISR_TAG}${EMA_TAG}-Qwen3-8B-${CURRENT_TIME}"

    # ── 提交 ────────────────────────────────────────────────────────
    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  REWARD_TYPE=$REWARD_TYPE LR=$LR ENTROPY=$ENTROPY_COEFF"
        echo "  TEACHER_REG=$TEACHER_REG UPDATE_RATE=$TEACHER_UPDATE_RATE"
        echo "  INCLUDE_SUCCESSFUL_ROLLOUTS=$INCLUDE_SUCCESSFUL_ROLLOUTS"
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
            --env=SWANLAB_API_KEY=${SWANLAB_API_KEY} \
            --env=PROJECT_NAME=${PROJECT_NAME} \
            --env=JOB_NAME=${JOB_NAME} \
            --env=REWARD_TYPE=${REWARD_TYPE} \
            --env=LR=${LR} \
            --env=ENTROPY_COEFF=${ENTROPY_COEFF} \
            --env=TEACHER_REG=${TEACHER_REG} \
            --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} \
            --env=NORM_ADV_BY_STD=${NORM_ADV_BY_STD} \
            --env=CLIP_ADV=${CLIP_ADV} \
            --env=CLIP_ADV_VALUE=${CLIP_ADV_VALUE} \
            --env=ROLLOUT_IS=${ROLLOUT_IS} \
            --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} \
            --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} \
            --env=ROLLOUT_N=${ROLLOUT_N} \
            --env=INCLUDE_SUCCESSFUL_ROLLOUTS=${INCLUDE_SUCCESSFUL_ROLLOUTS} \
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

done  # INCLUDE_SUCCESSFUL_ROLLOUTS
done  # TEACHER_UPDATE_RATE
done  # TEACHER_REG
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
