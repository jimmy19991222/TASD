#!/bin/bash
# =============================================================================
# TASD - Entropy Coeff 扫描实验
#
# 基准配置（当前 best）：
#   - reward_type=teacher_log_prob
#   - entropy_gate=hard_keep_reward, ratio=1.0, topk=256
#   - rep=1.05, normStd, clipH0.28, clipAdv2.0, ema0.1, inclSucc, gmSeq
#   - ec=0.001 (baseline)
#
# 变量：entropy_coeff = [0.001, 0.005, 0.01]
# =============================================================================

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
PROJECT_NAME="TASD-v5"

DATASETS=(
    "sciknoweval/biology"
)

DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# ── 固定参数（best 配置）─────────────────────────────────────────────
REWARD_TYPE="teacher_log_prob"
ENTROPY_GATE="hard_keep_reward"
ENTROPY_GATE_RATIO="1.0"
DISTILL_TOPK="256"
REPETITION_PENALTY="1.05"
NORM_ADV_BY_STD="true"
ADV_STD_FLOOR="none"
CLIP_RATIO_HIGH="0.28"
CLIP_ADV="true"
CLIP_ADV_VALUE="2.0"
TEACHER_REG="ema"
TEACHER_UPDATE_RATE="0.1"
INCLUDE_SUCCESSFUL_ROLLOUTS="True"
GROUP_MEAN_MODE="seq"
ADV_ENTROPY_WEIGHT="none"
TEMPERATURE="1.0"
ENTROPY_FLOOR="0.0"
ENTROPY_PENALTY_COEFF="0.0"
LR="1e-5"
SEED="42"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
ROLLOUT_N="8"
MODEL="Qwen3-8B"
FILTER_GROUPS_ENABLE="false"
FILTER_GROUPS_METRIC="acc"
FILTER_GROUPS_MAX_GEN="0"

# ── 扫描变量 ─────────────────────────────────────────────────────────
ENTROPY_COEFF_LIST=(
    "0.001"   # baseline（当前 best）
    "0.005"
    "0.01"
)

GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# =============================================================================
TOTAL=0
SUBMITTED=0

for DATASET in "${DATASETS[@]}"; do
for ENTROPY_COEFF in "${ENTROPY_COEFF_LIST[@]}"; do
    TOTAL=$((TOTAL + 1))

    DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
    MODEL_SHORT=$(echo "$MODEL" | tr '.' '-')
    OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
    MODEL_PATH="${OSS_ROOT}/base_models/${MODEL}"

    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="TASD-${DATASET_SHORT}-rt_${REWARD_TYPE}-gate_${ENTROPY_GATE}-topk${DISTILL_TOPK}-rep${REPETITION_PENALTY}-normStd-clipH${CLIP_RATIO_HIGH}-clipAdv${CLIP_ADV_VALUE}-ema${TEACHER_UPDATE_RATE}-inclSucc-gmSeq-ec${ENTROPY_COEFF}-v2-${MODEL_SHORT}-${CURRENT_TIME}"

    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  ENTROPY_COEFF=${ENTROPY_COEFF}"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
            --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL=${MODEL} --env=MODEL_PATH=${MODEL_PATH} --env=REWARD_TYPE=${REWARD_TYPE} --env=ENTROPY_GATE=${ENTROPY_GATE} --env=ENTROPY_GATE_RATIO=${ENTROPY_GATE_RATIO} --env=CLIP_ADV=${CLIP_ADV} --env=CLIP_ADV_VALUE=${CLIP_ADV_VALUE} --env=DISTILL_TOPK=${DISTILL_TOPK} --env=REPETITION_PENALTY=${REPETITION_PENALTY} --env=LR=${LR} --env=SEED=${SEED} --env=ENTROPY_COEFF=${ENTROPY_COEFF} --env=ROLLOUT_TEMPERATURE=${TEMPERATURE} --env=TEACHER_REG=${TEACHER_REG} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=INCLUDE_SUCCESSFUL_ROLLOUTS=${INCLUDE_SUCCESSFUL_ROLLOUTS} --env=NORM_ADV_BY_STD=${NORM_ADV_BY_STD} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR} --env=ADV_ENTROPY_WEIGHT=${ADV_ENTROPY_WEIGHT} --env=GROUP_MEAN_MODE=${GROUP_MEAN_MODE} --env=CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH} --env=FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE} --env=FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC} --env=FILTER_GROUPS_MAX_GEN=${FILTER_GROUPS_MAX_GEN} --env=ENTROPY_FLOOR=${ENTROPY_FLOOR} --env=ENTROPY_PENALTY_COEFF=${ENTROPY_PENALTY_COEFF} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=f598ad33b071751bf79d2484d8e1acefe8df9d879e129cae40340a158854f9cb --env=DINGTALK_SECRET=SECc5b9e4f61f56b32b46abf1ecedc11bdcba10dc35fbba8fa0ff62c084a1cc6ad3" \
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

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
