#!/bin/bash
# =============================================================================
# TASD Bidirectional Credit 实验
#
# 核心思路：
#   用 Causal EMA baseline 解决 context pollution 问题
#   A_bidir_t = Q_t - V_t
#   Q_t = teacher_log_prob (已有)
#   V_t = causal_ema(Q_{<t})  # 用过去的位置平滑估计 baseline
#
# 实验矩阵（3 个）：
#   1. noGate:  无 gate 对照（纯 outcome + teacher_prob 加权）
#   2. gateWU:  gate + warmup baseline
#   3. bidir:   gate + warmup + causal_ema baseline (τ=32)
#
# 全部共享：
#   REWARD_TYPE=outcome, gmSeq, clipAdv=2.0, ema=0.1, rep=1.05, ec=0.001
#   GATE_WARMUP_STEPS=10
# =============================================================================

# ── Nebula 账号配置 ──────────────────────────────────────────────────────
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
PROJECT_NAME="TASD-v5-bidir"

# ── 数据集 ───────────────────────────────────────────────────────────────
DATASETS=(
    "sciknoweval/biology"
)

# ── dry-run ───────────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 共享超参（v5 最优）
# =============================================================================
REWARD_TYPE="outcome"
CLIP_ADV="true"
CLIP_ADV_VALUE="2.0"
REPETITION_PENALTY="1.05"
NORM_ADV_BY_STD="true"
ADV_STD_FLOOR="none"
GROUP_MEAN_MODE="seq"
CLIP_RATIO_HIGH="0.28"
FILTER_GROUPS_ENABLE="false"
FILTER_GROUPS_METRIC="acc"
FILTER_GROUPS_MAX_GEN="0"
INCLUDE_SUCCESSFUL_ROLLOUTS="True"
TEACHER_UPDATE_RATE="0.1"
ENTROPY_COEFF="0.001"
TEMPERATURE="1.0"
LR="1e-5"
SEED="42"
TEACHER_REG="ema"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
ROLLOUT_N="8"
MODEL="Qwen3-8B"
GATE_WARMUP_STEPS="10"
MAX_ERRORS_IN_POOL="8"
ERROR_ANSWER_MAX_CHARS="1024"

# =============================================================================
# 实验矩阵
# ADV_BASELINE_MODE | CAUSAL_EMA_ALPHA | ENTROPY_GATE | ADV_ENTROPY_WEIGHT | TAG
# =============================================================================
declare -a EXPERIMENTS=(
    "none|0.0|none|teacher_prob|noGate"
    "none|0.0|hard_keep_reward|none|gateWU"
    "causal_ema|0.021|hard_keep_reward|none|bidir_tau32"
)

GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# =============================================================================
# 提交
# =============================================================================
TOTAL=0
SUBMITTED=0

for DATASET in "${DATASETS[@]}"; do
for EXP_CFG in "${EXPERIMENTS[@]}"; do

    IFS='|' read -r ADV_BASELINE_MODE CAUSAL_EMA_ALPHA ENTROPY_GATE ADV_ENTROPY_WEIGHT TAG <<< "$EXP_CFG"

    TOTAL=$((TOTAL + 1))

    DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
    MODEL_SHORT=$(echo "$MODEL" | tr '.' '-')
    OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
    MODEL_PATH="${OSS_ROOT}/base_models/${MODEL}"

    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="TASD-${DATASET_SHORT}-bidir-${TAG}-topk256-gmSeq-clipAdv${CLIP_ADV_VALUE}-ema${TEACHER_UPDATE_RATE}-rep${REPETITION_PENALTY}-ec${ENTROPY_COEFF}-${MODEL_SHORT}-${CURRENT_TIME}"

    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  ADV_BASELINE_MODE=${ADV_BASELINE_MODE} CAUSAL_EMA_ALPHA=${CAUSAL_EMA_ALPHA} ENTROPY_GATE=${ENTROPY_GATE} ADV_ENTROPY_WEIGHT=${ADV_ENTROPY_WEIGHT}"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
            --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL=${MODEL} --env=MODEL_PATH=${MODEL_PATH} --env=REWARD_TYPE=${REWARD_TYPE} --env=ENTROPY_GATE=${ENTROPY_GATE} --env=ENTROPY_GATE_RATIO=1.0 --env=CLIP_ADV=${CLIP_ADV} --env=CLIP_ADV_VALUE=${CLIP_ADV_VALUE} --env=DISTILL_TOPK=256 --env=REPETITION_PENALTY=${REPETITION_PENALTY} --env=LR=${LR} --env=SEED=${SEED} --env=ENTROPY_COEFF=${ENTROPY_COEFF} --env=ROLLOUT_TEMPERATURE=${TEMPERATURE} --env=TEACHER_REG=${TEACHER_REG} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=INCLUDE_SUCCESSFUL_ROLLOUTS=${INCLUDE_SUCCESSFUL_ROLLOUTS} --env=NORM_ADV_BY_STD=${NORM_ADV_BY_STD} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR} --env=ADV_ENTROPY_WEIGHT=${ADV_ENTROPY_WEIGHT} --env=GROUP_MEAN_MODE=${GROUP_MEAN_MODE} --env=CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH} --env=FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE} --env=FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC} --env=FILTER_GROUPS_MAX_GEN=${FILTER_GROUPS_MAX_GEN} --env=TEACHER_CONTEXT_MODE=per_rollout --env=GATE_WARMUP_STEPS=${GATE_WARMUP_STEPS} --env=MAX_ERRORS_IN_POOL=${MAX_ERRORS_IN_POOL} --env=ERROR_ANSWER_MAX_CHARS=${ERROR_ANSWER_MAX_CHARS} --env=ERROR_POOL_FORMAT_ONLY=True --env=TOP_K_AGREEMENT_K=0 --env=ADV_BASELINE_MODE=${ADV_BASELINE_MODE} --env=CAUSAL_EMA_ALPHA=${CAUSAL_EMA_ALPHA} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=f598ad33b071751bf79d2484d8e1acefe8df9d879e129cae40340a158854f9cb --env=DINGTALK_SECRET=SECc5b9e4f61f56b32b46abf1ecedc11bdcba10dc35fbba8fa0ff62c084a1cc6ad3" \
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
