#!/bin/bash
# =============================================================================
# Geodesic-VCE 单卡 A100 Smoke Test
#
# 目的：快速验证代码是否能正常运行（10 步即可看到 loss 下降）
#
# 配置：
#   - 单卡 A100
#   - 10 training steps
#   - batch_size=8, rollout_n=2
#   - SDPO baseline（最简单配置）
#
# 使用方式：
#   bash nebula_scripts/submit_geodesic_smoke_test.sh [--dry-run]
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
SCRIPT_PATH="nebula_scripts/sdpo/geodesic_vce_ablation_parametric.sh"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="Geodesic-SmokeTest"

# ── 数据集配置 ──────────────────────────────────────────────────────
DATASET="sciknoweval/biology"

# ── dry-run 模式 ─────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# Smoke Test 超参（极简配置，快速验证）
# =============================================================================

SEED="42"
LR="1e-5"
TRAIN_BATCH_SIZE="8"        # 减小 batch size 适配单卡
ROLLOUT_N="2"               # 减少 rollout 数量
MODEL_NAME="Qwen3-8B"
MAX_STEPS="10"              # 只跑 10 步

# SDPO 参数
ALPHA="0.5"
DISTILL_TOPK="100"
DONT_REPROMPT_ON_SELF_SUCCESS="True"

# Geodesic 参数（关闭，只测试基础 SDPO）
USE_GEODESIC="False"
GEODESIC_TRUST_REGION="5.0"
GEODESIC_BETA_SCALE="0.5"

# V_CE 参数（不使用）
USE_VCE="False"
CLIP_VALUE="3.0"
ADV_STD_FLOOR="0.0"

# Git 信息
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# =============================================================================
# 提交逻辑
# =============================================================================

# 构建任务名称
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
JOB_NAME="GSMOKE-${DATASET_SHORT}-sdpo_baseline-${TIMESTAMP}"

echo "========================================================================="
echo "🚀 提交 Geodesic-VCE 单卡 A100 Smoke Test"
echo "========================================================================="
echo "项目: $PROJECT_NAME"
echo "任务: $JOB_NAME"
echo "数据集: $DATASET"
echo "配置: SDPO baseline, 10 steps, batch_size=8, rollout_n=2, 单卡A100"
echo "========================================================================="

# 构建 ENV_PARAMS（所有业务参数通过 --env 传递，参考 reference_submit.sh）
ENV_PARAMS="--env=PROJECT_NAME=${PROJECT_NAME} \
    --env=JOB_NAME=${JOB_NAME} \
    --env=DATASET=${DATASET} \
    --env=SEED=${SEED} \
    --env=LOSS_MODE=sdpo \
    --env=USE_VCE=${USE_VCE} \
    --env=USE_GEODESIC=${USE_GEODESIC} \
    --env=LR=${LR} \
    --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} \
    --env=ROLLOUT_N=${ROLLOUT_N} \
    --env=MODEL_NAME=${MODEL_NAME} \
    --env=MAX_STEPS=${MAX_STEPS} \
    --env=ALPHA=${ALPHA} \
    --env=DISTILL_TOPK=${DISTILL_TOPK} \
    --env=DONT_REPROMPT_ON_SELF_SUCCESS=${DONT_REPROMPT_ON_SELF_SUCCESS} \
    --env=CLIP_VALUE=${CLIP_VALUE} \
    --env=ADV_STD_FLOOR=${ADV_STD_FLOOR} \
    --env=GEODESIC_TRUST_REGION=${GEODESIC_TRUST_REGION} \
    --env=GEODESIC_BETA_SCALE=${GEODESIC_BETA_SCALE} \
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
        echo "📊 Smoke Test 配置:"
        echo "  - Training steps: 10"
        echo "  - Batch size: 8"
        echo "  - Rollout n: 2"
        echo "  - GPU: 1x A100"
        echo "  - Loss mode: SDPO baseline"
        echo "  - Geodesic: False"
        echo ""
        echo "🔍 验证标准:"
        echo "  - 训练正常启动，无 crash"
        echo "  - actor/loss 在前 10 步内下降"
        echo "  - val/accuracy 有输出"
        echo "  - SwanLab 正常记录指标"
    fi
fi

echo "========================================================================="
