#!/bin/bash
# =============================================================================
# GRPO Baseline 实验 - Nebula 提交脚本
#
# 目的：提供 GRPO 作为 baseline 对比
#
# 配置：
#   - 4 卡训练
#   - 250 training steps
#   - batch_size=32, mini_batch_size=8, rollout_n=8
#   - seed=42（与 Geodesic-VCE 实验一致）
#
# 使用方式：
#   bash nebula_scripts/submit_grpo_baseline.sh [--dry-run]
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
SCRIPT_PATH="nebula_scripts/grpo/grpo_sciknoweval_parametric.sh"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="Geodesic-VCE-Ablation"

# ── 数据集配置 ──────────────────────────────────────────────────────
DATASET="sciknoweval/biology"

# ── dry-run 模式 ─────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# GRPO 超参（与 Geodesic-VCE 实验保持一致）
# =============================================================================

SEED="42"
LR="1e-5"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="8"
ROLLOUT_N="8"
MODEL_NAME="Qwen3-8B"

# Git 信息
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# =============================================================================
# 提交逻辑
# =============================================================================

# 构建任务名称
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
JOB_NAME="GRPO-${DATASET_SHORT}-baseline"

echo "========================================================================="
echo "🚀 提交 GRPO Baseline 实验"
echo "========================================================================="
echo "项目: $PROJECT_NAME"
echo "任务: $JOB_NAME"
echo "数据集: $DATASET"
echo "配置: GRPO baseline, 250 steps, batch_size=32, mini_batch=8, rollout_n=8"
echo "GPU: 4x A100"
echo "Seed: $SEED"
echo "========================================================================="

# 构建 ENV_PARAMS（所有业务参数通过 --env 传递，参考 reference_submit.sh）
ENV_PARAMS="--env=PROJECT_NAME=${PROJECT_NAME} \
    --env=JOB_NAME=${JOB_NAME} \
    --env=DATASET=${DATASET} \
    --env=SEED=${SEED} \
    --env=LR=${LR} \
    --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} \
    --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} \
    --env=ROLLOUT_N=${ROLLOUT_N} \
    --env=MODEL_NAME=${MODEL_NAME} \
    --env=GIT_BRANCH=${GIT_BRANCH} \
    --env=GIT_COMMIT=${GIT_COMMIT}"

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] 将执行:"
    echo "nebulactl run mdl --force --engine=xdl --queue=$QUEUE ..."
else
    echo "🚀 提交中..."
    SUBMIT_OUTPUT=$(nebulactl run mdl \
        --force \
        --engine=xdl \
        --queue=$QUEUE \
        --entry=nebula_scripts/entry.py \
        --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} ${ENV_PARAMS}" \
        --worker_count=$WORLD_SIZE \
        --file.cluster_file=$CLUSTER_FILE \
        --job_name=$JOB_NAME \
        --env=OPENLM_TOKEN=$OPENLM_TOKEN \
        --env=SWANLAB_API_KEY=${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal} \
        --custom_docker_image=$CUSTOM_DOCKER_IMAGE \
        --requirements_file_name=requirements_nebula.txt \
        --oss_access_id=$OSS_ACCESS_ID \
        --oss_access_key=$OSS_ACCESS_KEY \
        --oss_bucket=$OSS_BUCKET \
        --oss_endpoint=$OSS_ENDPOINT \
        2>&1)
    SUBMIT_EXIT=$?
    echo "$SUBMIT_OUTPUT"
    if [ $SUBMIT_EXIT -ne 0 ]; then
        echo "❌ 提交失败 (exit code: $SUBMIT_EXIT)"
    else
        echo "✅ 提交完成: $JOB_NAME"
        echo ""
        echo "📊 GRPO Baseline 配置:"
        echo "  - Training steps: 250"
        echo "  - Batch size: 32"
        echo "  - Mini batch size: 8"
        echo "  - Rollout n: 8"
        echo "  - GPU: 4x A100"
        echo "  - Seed: 42"
        echo ""
        echo "🔍 验证标准:"
        echo "  - 训练正常启动，无 crash"
        echo "  - actor/loss 稳定下降"
        echo "  - val/accuracy 逐步提升"
        echo "  - SwanLab 正常记录指标"
    fi
fi

echo "========================================================================="
