#!/bin/bash
# =============================================================================
# [P2] TASD 消融实验 - 验证各组件贡献
#
# 目标：理解 EMA teacher、std 归一化、isr（成功 rollout 参与）的具体作用
#
# 基线：P0/P1 跑出的最优配置（默认用 teacher_log_prob + ent0.05 作为 anchor）
#
# 跑 4 个 job（与 P0 基线对照）：
#   消融 EMA：
#     1. teacher_log_prob + ent0.05 + std + ema=none (固定 teacher)
#   消融 std：
#     2. teacher_log_prob + ent0.05 + nostd + ema0.1
#   消融 isr：
#     3. teacher_log_prob + ent0.05 + std  + ema0.1 + isr=False
#   基线对照（建议复用 P0/P1 结果，不重复提交）：
#     - teacher_log_prob + ent0.05 + std + ema0.1 + isr=True  (P1 已有)
#
# 使用方式：bash nebula_scripts/submit_tasd_p2_ablation.sh [--dry-run]
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
# [P2] 消融实验矩阵
# 格式：REWARD_TYPE:ENTROPY_COEFF:TEACHER_REG:TEACHER_UPDATE_RATE:NORM_ADV_BY_STD:ISR:ABLATION_DESC
# =============================================================================
# 基线（P1 已有，此处只补充消融变体）
# anchor: teacher_log_prob:0.05:ema:0.1:True:True

EXPERIMENTS=(
    # 消融 EMA：teacher_update_rate=0.0 等价于固定 teacher（初始 actor 权重不再更新）
    "teacher_log_prob:0.05:none:0.1:True:True:ablate-ema"
    # 消融 std 归一化
    "teacher_log_prob:0.05:ema:0.1:False:True:ablate-std"
    # 消融 isr（成功 rollout 不参与训练）
    "teacher_log_prob:0.05:ema:0.1:True:False:ablate-isr"
)

# 固定参数
LR="1e-5"
CLIP_ADV="True"
CLIP_ADV_VALUE="5.0"
ROLLOUT_IS="token"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
ROLLOUT_N="8"

# =============================================================================
TOTAL=0
SUBMITTED=0

for EXP in "${EXPERIMENTS[@]}"; do
    IFS=':' read -r REWARD_TYPE ENTROPY_COEFF TEACHER_REG TEACHER_UPDATE_RATE NORM_ADV_BY_STD INCLUDE_SUCCESSFUL_ROLLOUTS ABLATION_DESC <<< "$EXP"

    TOTAL=$((TOTAL + 1))

    # 构建 job name 中的各 tag
    ENT_TAG=""
    if [ "$(echo "$ENTROPY_COEFF > 0" | bc -l)" = "1" ]; then
        ENT_TAG="-ent${ENTROPY_COEFF}"
    fi
    EMA_TAG="-ema-none"
    if [ "$TEACHER_REG" = "ema" ]; then
        EMA_TAG="-ema${TEACHER_UPDATE_RATE}"
    fi
    STD_TAG="-std"
    if [ "$NORM_ADV_BY_STD" = "False" ]; then
        STD_TAG="-no_std"
    fi
    ISR_TAG="-isr1"
    if [ "$INCLUDE_SUCCESSFUL_ROLLOUTS" = "False" ]; then
        ISR_TAG="-isr0"
    fi

    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    LR_TAG=$(echo "$LR" | tr '-' '_')  # 把 lr 中的 - 替换成 _，便于按 - 分割
    JOB_NAME="TASD-bio-lr${LR_TAG}-rt${REWARD_TYPE}${STD_TAG}-clip5.0${ENT_TAG}-rctoken${ISR_TAG}${EMA_TAG}-Qwen3-8B-${CURRENT_TIME}"

    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  [消融] $ABLATION_DESC"
        echo "  REWARD_TYPE=$REWARD_TYPE  ENTROPY_COEFF=$ENTROPY_COEFF"
        echo "  TEACHER_REG=$TEACHER_REG  UPDATE_RATE=$TEACHER_UPDATE_RATE"
        echo "  NORM_ADV_BY_STD=$NORM_ADV_BY_STD  ISR=$INCLUDE_SUCCESSFUL_ROLLOUTS"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}  [$ABLATION_DESC]"

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

done  # EXP

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
