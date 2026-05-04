#!/bin/bash
# =============================================================================
# TASD v13 - v5 基准复现实验
#
# 目标：在干净的 v5-baseline 分支（commit 52149bd）上精确复现
#   TASD-sciknoweval-biology-rt_teacher_log_prob-gate_hard_keep_reward-topk256-
#   rep1.05-normStd-clipH0.28-clipAdv2.0-ema0.1-inclSucc-gmSeq-ec0.001-v2-Qwen3-8B
#   （best_step=120, best_metric=0.681）
#
# 为何要单独跑：
#   - HEAD (simplifed) 相对 52149bd 多 800+ 行 ray_trainer.py 改动，导致 v11.5
#     以相同超参只能达到 peak 0.574 ≠ v5 报告 0.681
#   - 切到 52149bd 干净分支排除"代码路径漂移"这个变量，验证数据/模型本身
#     仍能复现 0.68；成功后再在 v5-baseline 上增量叠加 error_pool / A / B
#
# 矩阵：固定 1 个 bio job（和原 v5 实验名完全一致的超参）
#
# 使用：
#   bash nebula_scripts/submit_tasd_v13_v5_replay_sweep.sh [--dry-run]
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
PROJECT_NAME="TASD-v13-v5Replay"

# ── dry-run 模式 ─────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 固定超参（严格对齐 v5 实验名）
# TASD-sciknoweval-biology-rt_teacher_log_prob-gate_hard_keep_reward-topk256-
# rep1.05-normStd-clipH0.28-clipAdv2.0-ema0.1-inclSucc-gmSeq-ec0.001-v2
# =============================================================================
DATASET="sciknoweval/biology"
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
TEACHER_UPDATE_RATE="0.1"
INCLUDE_SUCCESSFUL_ROLLOUTS="True"
ADV_ENTROPY_WEIGHT="none"
GROUP_MEAN_MODE="seq"
ENTROPY_COEFF="0.001"
TEMPERATURE="1.0"

FILTER_GROUPS_ENABLE="false"
FILTER_GROUPS_METRIC="acc"
FILTER_GROUPS_MAX_GEN="0"

LR="1e-5"
SEED="42"
TEACHER_REG="ema"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
ROLLOUT_N="8"
MODEL="Qwen3-8B"

GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# =============================================================================
# 实验名构造（完全对齐 v5 格式，保留 -v2 版本锚点便于 swanlab 筛选）
# =============================================================================
DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
MODEL_SHORT=$(echo "$MODEL" | tr '.' '-')
OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
MODEL_PATH="${OSS_ROOT}/base_models/${MODEL}"

ENTROPY_TAG="-gate_${ENTROPY_GATE}"
TOPK_TAG="-topk${DISTILL_TOPK}"
RATIO_TAG=""  # ratio=1.0 不显示
REP_TAG="-rep${REPETITION_PENALTY}"
STD_TAG="-normStd"  # norm_adv_by_std=true + floor=none
CLIP_TAG="-clipH${CLIP_RATIO_HIGH}"
CLIP_ADV_TAG="-clipAdv${CLIP_ADV_VALUE}"
EMA_TAG="-ema${TEACHER_UPDATE_RATE}"
ISR_TAG="-inclSucc"
AEW_TAG=""  # adv_entropy_weight=none 不显示
GMM_TAG="-gmSeq"
EC_TAG="-ec${ENTROPY_COEFF}"
TEMP_TAG=""  # temperature=1.0 不显示

CURRENT_TIME=$(date +%Y%m%d_%H%M%S)

# v13 tag 区分和 v5 原实验；保留 -v2 锚点让代码路径与 v5 实验名一致
JOB_NAME="TASD-${DATASET_SHORT}-rt_${REWARD_TYPE}${ENTROPY_TAG}${RATIO_TAG}${TOPK_TAG}${REP_TAG}${STD_TAG}${CLIP_TAG}${CLIP_ADV_TAG}${EMA_TAG}${ISR_TAG}${AEW_TAG}${GMM_TAG}${EC_TAG}${TEMP_TAG}-v13replay-${MODEL_SHORT}-${CURRENT_TIME}"

# =============================================================================
# 提交
# =============================================================================
echo "============================================================"
echo "v13 v5-replay baseline"
echo "  branch  : ${GIT_BRANCH}"
echo "  commit  : ${GIT_COMMIT}  (期望: 52149bd)"
echo "  dataset : ${DATASET}"
echo "  job_name: ${JOB_NAME}"
echo "  目标    : best_metric@val-core/sciknoweval/acc/mean@16 ≥ 0.65 (v5 ref=0.681)"
echo "============================================================"

if [ "$DRY_RUN" = true ]; then
    echo "Dry-run: 打印不提交"
    echo "  REWARD_TYPE=$REWARD_TYPE"
    echo "  ENTROPY_GATE=$ENTROPY_GATE  RATIO=$ENTROPY_GATE_RATIO  TOPK=$DISTILL_TOPK"
    echo "  REP=$REPETITION_PENALTY  NORM_STD=$NORM_ADV_BY_STD  FLOOR=$ADV_STD_FLOOR"
    echo "  CLIP_H=$CLIP_RATIO_HIGH  CLIP_ADV=$CLIP_ADV_VALUE"
    echo "  EMA=$TEACHER_UPDATE_RATE  AEW=$ADV_ENTROPY_WEIGHT  GMM=$GROUP_MEAN_MODE"
    echo "  EC=$ENTROPY_COEFF  TEMP=$TEMPERATURE  SEED=$SEED"
    exit 0
fi

echo "提交 Job: ${JOB_NAME}"

SUBMIT_OUTPUT=$(nebulactl run mdl \
    --force \
    --engine=xdl \
    --queue=${QUEUE} \
    --entry=nebula_scripts/entry.py \
    --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL=${MODEL} --env=MODEL_PATH=${MODEL_PATH} --env=REWARD_TYPE=${REWARD_TYPE} --env=ENTROPY_GATE=${ENTROPY_GATE} --env=ENTROPY_GATE_RATIO=${ENTROPY_GATE_RATIO} --env=CLIP_ADV=${CLIP_ADV} --env=CLIP_ADV_VALUE=${CLIP_ADV_VALUE} --env=DISTILL_TOPK=${DISTILL_TOPK} --env=REPETITION_PENALTY=${REPETITION_PENALTY} --env=LR=${LR} --env=SEED=${SEED} --env=ENTROPY_COEFF=${ENTROPY_COEFF} --env=ROLLOUT_TEMPERATURE=${TEMPERATURE} --env=TEACHER_REG=${TEACHER_REG} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=INCLUDE_SUCCESSFUL_ROLLOUTS=${INCLUDE_SUCCESSFUL_ROLLOUTS} --env=NORM_ADV_BY_STD=${NORM_ADV_BY_STD} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR} --env=ADV_ENTROPY_WEIGHT=${ADV_ENTROPY_WEIGHT} --env=GROUP_MEAN_MODE=${GROUP_MEAN_MODE} --env=CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH} --env=FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE} --env=FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC} --env=FILTER_GROUPS_MAX_GEN=${FILTER_GROUPS_MAX_GEN} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=f598ad33b071751bf79d2484d8e1acefe8df9d879e129cae40340a158854f9cb --env=DINGTALK_SECRET=SECc5b9e4f61f56b32b46abf1ecedc11bdcba10dc35fbba8fa0ff62c084a1cc6ad3" \
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

echo ""
echo "============================================================"
if [ $SUBMIT_EXIT -ne 0 ]; then
    echo "❌ 提交失败 (exit code: $SUBMIT_EXIT)"
    exit $SUBMIT_EXIT
else
    echo "✅ 已提交 v13 v5-replay job"
    echo "   在 SwanLab 项目 ${PROJECT_NAME} 查看；若 best@step~120 达到 ≥0.65"
    echo "   则说明 52149bd 代码可干净复现 v5，可继续叠加 v14/v15/v16"
fi
echo "============================================================"
