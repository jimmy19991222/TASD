#!/bin/bash
# =============================================================================
# Geodesic 粒度对比实验 - Token vs Vocabulary
#
# 科学问题：Geodesic 约束在 Token 粒度下是否仍然有效？
#
# 实验设计（4 组，与现有 4 组形成对比）：
#   C3: V_CE + Geodesic (Token 粒度)
#   D3: SDPO + Geodesic (Token 粒度)
#
# 对比目标：
#   C1 (vocab) vs C3 (token)：V_CE+Geodesic 的粒度敏感性
#   D1 (vocab) vs D3 (token)：SDPO+Geodesic 的粒度敏感性
#
# 使用方式：
#   bash nebula_scripts/submit_geodesic_granularity_comparison.sh [--dry-run]
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
PROJECT_NAME="Geodesic-Granularity"

# ── 数据集配置 ──────────────────────────────────────────────────────
DATASETS=(
    "sciknoweval/biology"
)

# ── dry-run 模式 ─────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 固定超参配置（所有实验共享）
# =============================================================================

# ── 训练参数 ──────────────────────────────────────────────────
SEED="42"
LR="1e-5"
TRAIN_BATCH_SIZE="32"
ROLLOUT_N="8"
MODEL_NAME="Qwen3-8B"
MAX_STEPS="250"

# ── SDPO 参数 ────────────────────────────────────────────────
ALPHA="0.5"              # Vocabulary 粒度默认值（JSD）
ALPHA_TOKEN="1.0"        # Token 粒度必须用 1.0（reverse KL）
DISTILL_TOPK="100"
DONT_REPROMPT_ON_SELF_SUCCESS="True"

# ── V_CE 参数 ────────────────────────────────────────────────
CLIP_VALUE="3.0"
ADV_STD_FLOOR="0.0"

# ── Geodesic 参数 ────────────────────────────────────────────
GEODESIC_TRUST_REGION="5.0"
GEODESIC_BETA_SCALE="0.5"

# =============================================================================
# 实验矩阵（2 组 Token 粒度实验）
# =============================================================================

EXPERIMENTS=(
    # C3: V_CE + Geodesic (Token 粒度)
    "vce_geodesic_token:loss_mode=self_teacher,use_vce=True,use_geodesic=True,full_logit=False"
    
    # D3: SDPO + Geodesic (Token 粒度)
    "sdpo_geodesic_token:loss_mode=sdpo,use_geodesic=True,full_logit=False"
)

# =============================================================================
# 提交逻辑
# =============================================================================

echo "========================================================================="
echo "🚀 开始提交 Geodesic 粒度对比实验（Token vs Vocabulary）"
echo "========================================================================="
echo "项目: $PROJECT_NAME"
echo "队列: $QUEUE"
echo "数据集: ${DATASETS[*]}"
echo "实验数: ${#EXPERIMENTS[@]}"
echo "Seed: $SEED (严格控制)"
echo "========================================================================="

for dataset in "${DATASETS[@]}"; do
    dataset_name=$(echo "$dataset" | tr '/' '_')
    
    for exp in "${EXPERIMENTS[@]}"; do
        exp_name="${exp%%:*}"
        exp_params="${exp#*:}"
        
        # 解析实验参数
        loss_mode=$(echo "$exp_params" | grep -o 'loss_mode=[^,]*' | cut -d'=' -f2)
        use_geodesic=$(echo "$exp_params" | grep -o 'use_geodesic=[^,]*' | cut -d'=' -f2)
        use_vce=$(echo "$exp_params" | grep -o 'use_vce=[^,]*' | cut -d'=' -f2 || echo "False")
        full_logit=$(echo "$exp_params" | grep -o 'full_logit=[^,]*' | cut -d'=' -f2)
        
        # 转换 full_logit 格式（false -> False, true -> True）
        if [ "$full_logit" = "False" ]; then
            full_logit_distillation="False"
            # Token 粒度必须用 alpha=1.0（reverse KL）
            current_alpha="$ALPHA_TOKEN"
        else
            full_logit_distillation="True"
            current_alpha="$ALPHA"
        fi
        
        # 如果没有指定 use_vce，默认为 False
        if [ -z "$use_vce" ]; then
            use_vce="False"
        fi
        
        # 构建任务名称（包含 timestamp 保证唯一性）
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        TASK_NAME="${PROJECT_NAME}-${dataset_name}-${exp_name}-${TIMESTAMP}"
        
        # 构建 JOB_NAME（用于 OSS 路径）
        JOB_NAME="GGRAN-${dataset_name}-${exp_name}"
        
        echo ""
        echo "📝 准备提交: $TASK_NAME"
        echo "   数据集: $dataset"
        echo "   模式: loss_mode=$loss_mode, full_logit=$full_logit, use_geodesic=$use_geodesic, alpha=$current_alpha"
        echo "   Seed: $SEED"
        
        # 构建 ENV_PARAMS（所有业务参数通过 --env 传递，参考 reference_submit.sh）
        ENV_PARAMS="--env=PROJECT_NAME=${PROJECT_NAME} \
            --env=JOB_NAME=${JOB_NAME} \
            --env=DATASET=${dataset} \
            --env=SEED=${SEED} \
            --env=LOSS_MODE=${loss_mode} \
            --env=USE_VCE=${use_vce} \
            --env=USE_GEODESIC=${use_geodesic} \
            --env=FULL_LOGIT_DISTILLATION=${full_logit_distillation} \
            --env=LR=${LR} \
            --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} \
            --env=ROLLOUT_N=${ROLLOUT_N} \
            --env=MODEL_NAME=${MODEL_NAME} \
            --env=ALPHA=${current_alpha} \
            --env=DISTILL_TOPK=${DISTILL_TOPK} \
            --env=DONT_REPROMPT_ON_SELF_SUCCESS=${DONT_REPROMPT_ON_SELF_SUCCESS} \
            --env=CLIP_VALUE=${CLIP_VALUE} \
            --env=ADV_STD_FLOOR=${ADV_STD_FLOOR} \
            --env=GEODESIC_TRUST_REGION=${GEODESIC_TRUST_REGION} \
            --env=GEODESIC_BETA_SCALE=${GEODESIC_BETA_SCALE}"
        
        # 构建 nebulactl 命令（标准化格式，参考 reference_submit.sh）
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
                echo "✅ 提交完成: $TASK_NAME"
            fi
            sleep 2
        fi
    done
done

echo ""
echo "========================================================================="
echo "✅ 所有实验提交完成！"
echo "========================================================================="
echo ""
echo "📊 实验矩阵（Token 粒度）:"
echo "  C3: V_CE + Geodesic (full_logit=False)"
echo "  D3: SDPO + Geodesic (full_logit=False)"
echo ""
echo "🔬 对比目标:"
echo "  C1 (vocab) vs C3 (token)：V_CE+Geodesic 的粒度敏感性"
echo "  D1 (vocab) vs D3 (token)：SDPO+Geodesic 的粒度敏感性"
echo ""
echo "🔍 监控方式:"
echo "  - SwanLab: actor/entropy, actor/geodesic_*, val/accuracy"
echo "  - 重点关注: Token vs Vocab 的 geodesic_weight 分布差异"
echo "  - Seed: 42 (所有实验统一)"
echo "========================================================================="
