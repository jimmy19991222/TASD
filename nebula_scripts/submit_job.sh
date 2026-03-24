#!/bin/bash
# =============================================================================
# SDPO Nebula 任务提交脚本
#
# 使用方式：
#   bash nebula_scripts/submit_job.sh [script_path] [world_size] [queue]
#
# 示例：
#   # 单节点 8 卡，提交 TASD sciknoweval 实验
#   bash nebula_scripts/submit_job.sh \
#       nebula_scripts/tasd/tasd_sciknoweval_qwen3_8B.sh \
#       1 \
#       lazada_llm_ad_h20
#
#   # 多节点，2 节点 × 8 卡 = 16 卡
#   bash nebula_scripts/submit_job.sh \
#       nebula_scripts/tasd/tasd_sciknoweval_qwen3_8B.sh \
#       2 \
#       lazada_llm_ad_h20
# =============================================================================

# ── Nebula 账号配置 ────────────────────────────────────────────────────────
QUEUE="lazada_llm_ad_h20"         # 默认队列（H20），可改为 ae_h100
WORLD_SIZE="${2:-1}"                     # 节点数，单节点填 1
OPENLM_TOKEN="${OPENLM_TOKEN:?OPENLM_TOKEN not set}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:?OSS_ACCESS_ID not set}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?OSS_ACCESS_KEY not set}"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
# 自定义镜像（留空则使用 --algo_name=pytorch260 默认镜像）
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324105345}"

# ── 训练脚本路径 ───────────────────────────────────────────────────────────
script_dir_path="${1:-nebula_scripts/tasd/tasd_sciknoweval_qwen3_8B.sh}"

# ── 根据节点数选择 cluster 配置文件 ──────────────────────────────────────
if [ "$WORLD_SIZE" -gt 1 ]; then
    CLUSTER_FILE="nebula_scripts/cluster.json"          # 8 GPU × N 节点
else
    CLUSTER_FILE="nebula_scripts/cluster_gpu_4.json"    # 单节点 4 GPU（省钱）
    # 若要单节点 8 GPU，改为：
    # CLUSTER_FILE="nebula_scripts/cluster.json"
fi

# ── 任务命名 ──────────────────────────────────────────────────────────────
CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
JOB_NAME="$(basename "${script_dir_path%.sh}")_${CURRENT_TIME}"

# ── 提交参数 ──────────────────────────────────────────────────────────────
options="--script_path=${script_dir_path} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME}"

echo "============================================================"
echo "提交 Nebula 任务"
echo "  脚本       : $script_dir_path"
echo "  节点数     : $WORLD_SIZE"
echo "  队列       : $QUEUE"
echo "  任务名     : $JOB_NAME"
echo "  Cluster    : $CLUSTER_FILE"
[ -n "$CUSTOM_DOCKER_IMAGE" ] && echo "  镜像       : $CUSTOM_DOCKER_IMAGE"
echo "============================================================"

SUBMIT_OUTPUT=$(nebulactl run mdl \
    --force \
    --engine=xdl \
    --queue=${QUEUE} \
    --entry=nebula_scripts/entry.py \
    --user_params="${options}" \
    --worker_count=${WORLD_SIZE} \
    --file.cluster_file=${CLUSTER_FILE} \
    --job_name=${JOB_NAME} \
    --access_id=${access_id} \
    --access_key=${access_key} \
    --env=OPENLM_TOKEN=${OPENLM_TOKEN} \
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
    echo "✅ 提交成功"
fi
