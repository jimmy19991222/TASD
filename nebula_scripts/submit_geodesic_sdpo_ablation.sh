#!/bin/bash
# =============================================================================
# Geodesic SDPO 消融实验 - Nebula 批量提交脚本
#
# 算法：Geodesic SDPO（测地线偏好优化）
# 核心思想：
#   - 在 SDPO 的 KL/JSD 蒸馏损失上引入 Fisher 度量加权
#   - 等价于在概率流形上做 Natural Gradient 更新
#   - 内生保护熵结构，免疫 Epistemic Collapse
#
# 实验设计（三组）：
#   A: SDPO baseline (alpha=0.5, use_geodesic=False)
#   B: Geodesic SDPO (alpha=0.5, use_geodesic=True, clip_max=10.0)
#   C: Geodesic SDPO Conservative (alpha=0.5, use_geodesic=True, clip_max=5.0)
#
# 使用方式：
#   bash nebula_scripts/submit_geodesic_sdpo_ablation.sh [--dry-run]
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
SCRIPT_PATH="nebula_scripts/sdpo/sdpo_sciknoweval_parametric.sh"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="Geodesic-SDPO-Ablation"

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
# 固定超参配置
# =============================================================================

# ── SDPO 核心参数 ──────────────────────────────────────────────
ALPHA="0.5"  # Jensen-Shannon divergence
DISTILL_TOPK="100"
ROLLOUT_N="8"
MINI_BATCH_SIZE="32"
LR="1e-5"

# ── 训练参数 ──────────────────────────────────────────────────
TRAIN_BATCH_SIZE="32"
MAX_STEPS="500"

# =============================================================================
# 实验矩阵
# =============================================================================

EXPERIMENTS=(
    "sdpo_baseline:use_geodesic=False"
    "geodesic_sdpo:use_geodesic=True,geodesic_trust_region=5.0,geodesic_beta_scale=0.5"
    "geodesic_sdpo_conservative:use_geodesic=True,geodesic_trust_region=3.0,geodesic_beta_scale=0.5"
    "geodesic_sdpo_aggressive:use_geodesic=True,geodesic_trust_region=8.0,geodesic_beta_scale=0.3"
)

# =============================================================================
# 提交逻辑
# =============================================================================

echo "========================================================================="
echo "🚀 开始提交 Geodesic SDPO 消融实验"
echo "========================================================================="
echo "项目: $PROJECT_NAME"
echo "队列: $QUEUE"
echo "数据集: ${DATASETS[*]}"
echo "实验数: ${#EXPERIMENTS[@]}"
echo "========================================================================="

for dataset in "${DATASETS[@]}"; do
    dataset_name=$(echo "$dataset" | tr '/' '_')
    
    for exp in "${EXPERIMENTS[@]}"; do
        exp_name="${exp%%:*}"
        exp_params="${exp#*:}"
        
        # 解析实验参数
        use_geodesic=$(echo "$exp_params" | grep -o 'use_geodesic=[^,]*' | cut -d'=' -f2)
        geodesic_trust_region=$(echo "$exp_params" | grep -o 'geodesic_trust_region=[^,]*' | cut -d'=' -f2)
        geodesic_beta_scale=$(echo "$exp_params" | grep -o 'geodesic_beta_scale=[^,]*' | cut -d'=' -f2)
        
        # 如果没有指定 trust_region，使用默认值
        if [ -z "$geodesic_trust_region" ]; then
            geodesic_trust_region="5.0"
        fi
        
        # 如果没有指定 beta_scale，使用默认值
        if [ -z "$geodesic_beta_scale" ]; then
            geodesic_beta_scale="0.5"
        fi
        
        # 构建任务名称
        TASK_NAME="${PROJECT_NAME}-${dataset_name}-${exp_name}"
        
        echo ""
        echo "📝 准备提交: $TASK_NAME"
        echo "   数据集: $dataset"
        echo "   参数: use_geodesic=$use_geodesic, trust_region=$geodesic_trust_region, beta_scale=$geodesic_beta_scale"
        
        # 构建 nebulactl 命令
        CMD="nebulactl run mdl \
            --engine=xdl \
            --entry=nebula_scripts/entry.py \
            --file.cluster_file=$CLUSTER_FILE \
            --queue=$QUEUE \
            --name=$TASK_NAME \
            --world_size=$WORLD_SIZE \
            --resource_type=GU20 \
            --image=$CUSTOM_DOCKER_IMAGE \
            --user_params \"
                openlm_token=$OPENLM_TOKEN \
                oss_access_id=$OSS_ACCESS_ID \
                oss_access_key=$OSS_ACCESS_KEY \
                oss_endpoint=$OSS_ENDPOINT \
                oss_bucket=$OSS_BUCKET \
                script_path=$SCRIPT_PATH \
                dataset=$dataset \
                alpha=$ALPHA \
                distill_topk=$DISTILL_TOPK \
                rollout_n=$ROLLOUT_N \
                mini_batch_size=$MINI_BATCH_SIZE \
                lr=$LR \
                train_batch_size=$TRAIN_BATCH_SIZE \
                max_steps=$MAX_STEPS \
                use_geodesic=$use_geodesic \
                geodesic_trust_region=$geodesic_trust_region \
                geodesic_beta_scale=$geodesic_beta_scale \
                exp_name=$exp_name \
                project_name=$PROJECT_NAME
            \""
        
        if [ "$DRY_RUN" = true ]; then
            echo "🔍 [Dry-run] 命令:"
            echo "$CMD"
            echo ""
        else
            echo "🚀 提交中..."
            eval $CMD
            echo "✅ 提交完成: $TASK_NAME"
            echo ""
            # 避免过快提交
            sleep 2
        fi
    done
done

echo ""
echo "========================================================================="
echo "✅ 所有实验提交完成！"
echo "========================================================================="
echo ""
echo "📊 实验矩阵:"
echo "  A: SDPO baseline (use_geodesic=False)"
echo "  B: Geodesic SDPO (trust_region=5.0, beta_scale=0.5)"
echo "  C: Geodesic SDPO Conservative (trust_region=3.0, beta_scale=0.5)"
echo "  D: Geodesic SDPO Aggressive (trust_region=8.0, beta_scale=0.3)"
echo ""
echo "🔍 监控方式:"
echo "  - SwanLab: 查看 actor/geodesic_* 指标"
echo "  - 钉钉通知: 训练异常自动告警"
echo "========================================================================="
