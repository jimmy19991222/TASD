#!/bin/bash
# =============================================================================
# Prior-Shift (Tier 1, Bayesian Credit Assignment 主菜) - Nebula 提交脚本
#
# 核心思想：
#   - g_t = KL( P_T(·|x, y_≤t) ‖ P_T(·|x, y_<t) )（teacher belief shift）
#   - A_t = A_seq · g_t / mean_t(g_t)  with clip(max=PS_MAX_RATIO)
#   - 退化保护：mean(g)<eps 时 fallback 为 GRPO 平摊
#
# 默认配置：1 个 smoke job（biology + 默认 PS_MAX_RATIO=10 + EMA r=0.05）
# 扩展：把对应 LIST 里注释掉的项打开即可 sweep
#
# 使用方式：
#   bash nebula_scripts/submit_prior_shift_sweep.sh [--dry-run]
# =============================================================================
set -euo pipefail

# ── Nebula 账号配置 ──────────────────────────────────────────────────────
QUEUE="lazada_llm_ad_h20"
WORLD_SIZE=1
OPENLM_TOKEN="${OPENLM_TOKEN:?OPENLM_TOKEN not set}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:?OSS_ACCESS_ID not set}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?OSS_ACCESS_KEY not set}"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
CLUSTER_FILE="nebula_scripts/cluster.json"
SCRIPT_PATH="nebula_scripts/prior_shift/prior_shift_sciknoweval_parametric.sh"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="PriorShift-Tier1"

# ── 数据集配置 ──────────────────────────────────────────────────────
DATASETS=(
    "sciknoweval/biology"
    # "sciknoweval/chemistry"
)

# ── dry-run 模式 ─────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 超参配置
# =============================================================================

# ── Prior-Shift 专属 ─────────────────────────────────────────────────
PS_MAX_RATIO_LIST=(
    "10.0"
    # "5.0"
    # "20.0"
)
PS_EPS_NORM="1.0e-6"
PS_UNIFORM_FALLBACK="True"

# ── Teacher EMA ──────────────────────────────────────────────────────
TEACHER_UPDATE_RATE_LIST=(
    "0.05"
    # "0.1"
)
TEACHER_REGULARIZATION="ema"

# ── 固定参数 ────────────────────────────────────────────────────────────
LR="1e-5"
SEED="42"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
ROLLOUT_N="8"
MODEL_NAME="Qwen3-8B"

# Git 信息（在本地获取，传递给 Nebula）
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# =============================================================================
# 提交循环
# =============================================================================
TOTAL=0
SUBMITTED=0

for DATASET in "${DATASETS[@]}"; do
for PS_MAX_RATIO in "${PS_MAX_RATIO_LIST[@]}"; do
for TEACHER_UPDATE_RATE in "${TEACHER_UPDATE_RATE_LIST[@]}"; do
    TOTAL=$((TOTAL + 1))

    # 构建短数据集名
    DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
    MODEL_SHORT=$(echo "$MODEL_NAME" | tr '.' '-')
    OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
    MODEL_PATH="${OSS_ROOT}/base_models/${MODEL_NAME}"

    # Tags
    MR_TAG="-mr${PS_MAX_RATIO}"
    EMA_TAG="-ema${TEACHER_UPDATE_RATE}"
    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)

    JOB_NAME="PS-${DATASET_SHORT}${MR_TAG}${EMA_TAG}-${MODEL_SHORT}-${CURRENT_TIME}"

    # ── 提交 ────────────────────────────────────────────────────────
    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  DATASET=$DATASET"
        echo "  PS_MAX_RATIO=$PS_MAX_RATIO PS_EPS_NORM=$PS_EPS_NORM PS_UNIFORM_FALLBACK=$PS_UNIFORM_FALLBACK"
        echo "  TEACHER_REG=$TEACHER_REGULARIZATION TEACHER_UPDATE_RATE=$TEACHER_UPDATE_RATE"
        echo "  LR=$LR BS=$TRAIN_BATCH_SIZE/$MINI_BATCH_SIZE N=$ROLLOUT_N MODEL=$MODEL_NAME"
        echo "  GIT_BRANCH=$GIT_BRANCH GIT_COMMIT=$GIT_COMMIT"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
            --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL_NAME=${MODEL_NAME} --env=MODEL_PATH=${MODEL_PATH} --env=LR=${LR} --env=SEED=${SEED} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=TEACHER_REGULARIZATION=${TEACHER_REGULARIZATION} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=PS_MAX_RATIO=${PS_MAX_RATIO} --env=PS_EPS_NORM=${PS_EPS_NORM} --env=PS_UNIFORM_FALLBACK=${PS_UNIFORM_FALLBACK} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=f598ad33b071751bf79d2484d8e1acefe8df9d879e129cae40340a158854f9cb --env=DINGTALK_SECRET=SECc5b9e4f61f56b32b46abf1ecedc11bdcba10dc35fbba8fa0ff62c084a1cc6ad3" \
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
done

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
