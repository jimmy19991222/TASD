#!/bin/bash
# =============================================================================
# Geodesic-VCE 融合消融实验 - Nebula 批量提交脚本
#
# 实验设计（四组，严格控制 seed=42）：
#   A: SDPO baseline (loss_mode=sdpo, use_geodesic=False)
#   B: V_CE baseline (loss_mode=self_teacher, use_vce=True, use_geodesic=False)
#   C: V_CE + Geodesic (loss_mode=self_teacher, use_vce=True, use_geodesic=True)
#   D: SDPO + Geodesic (loss_mode=sdpo, use_geodesic=True)
#
# 核心科学问题：
#   1. V_CE 是否真的导致 entropy collapse？（复现 B vs A）
#   2. Geodesic 能否免疫 V_CE 的熵崩溃？（验证 C vs B）
#   3. SDPO+Geodesic vs VCE+Geodesic 哪个更强？（对比 D vs C）
#
# 使用方式：
#   bash nebula_scripts/submit_geodesic_vce_ablation.sh [--dry-run]
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
PROJECT_NAME="Geodesic-VCE-Ablation"

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
# 固定超参配置（所有实验共享，保证唯一变量是算法设计）
# =============================================================================

# ── 训练参数 ──────────────────────────────────────────────────
SEED="42"  # 严格控制 seed 保证可复现性
LR="1e-5"
TRAIN_BATCH_SIZE="32"
ROLLOUT_N="8"
MODEL_NAME="Qwen3-8B"
MAX_STEPS="250"

# ── SDPO 参数 ────────────────────────────────────────────────
ALPHA="0.5"  # Jensen-Shannon divergence
DISTILL_TOPK="100"
DONT_REPROMPT_ON_SELF_SUCCESS="True"

# ── V_CE 参数 ────────────────────────────────────────────────
CLIP_VALUE="3.0"  # advantage clip threshold
ADV_STD_FLOOR="0.0"

# ── Geodesic 参数 ────────────────────────────────────────────
GEODESIC_TRUST_REGION="5.0"
GEODESIC_BETA_SCALE="0.5"

# =============================================================================
# 实验矩阵（4 组）
# =============================================================================

EXPERIMENTS=(
    # A: SDPO baseline（对照组）
    "sdpo_baseline:loss_mode=sdpo,use_geodesic=False"
    
    # B: V_CE baseline（验证熵崩溃复现）
    "vce_baseline:loss_mode=self_teacher,use_vce=True,use_geodesic=False"
    
    # C: V_CE + Geodesic（验证 Geodesic 能否免疫熵崩溃）
    "vce_plus_geodesic:loss_mode=self_teacher,use_vce=True,use_geodesic=True"
    
    # D: SDPO + Geodesic（验证 Geodesic 在 SDPO 上的效果）
    "sdpo_plus_geodesic:loss_mode=sdpo,use_geodesic=True"
)

# =============================================================================
# 提交逻辑
# =============================================================================

echo "========================================================================="
echo "🚀 开始提交 Geodesic-VCE 融合消融实验"
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
        
        # 如果没有指定 use_vce，默认为 False
        if [ -z "$use_vce" ]; then
            use_vce="False"
        fi
        
        # 构建任务名称（包含 timestamp 保证唯一性）
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        TASK_NAME="${PROJECT_NAME}-${dataset_name}-${exp_name}-${TIMESTAMP}"
        
        # 构建 JOB_NAME（用于 OSS 路径）
        JOB_NAME="GVCE-${dataset_name}-${exp_name}"
        
        echo ""
        echo "📝 准备提交: $TASK_NAME"
        echo "   数据集: $dataset"
        echo "   模式: loss_mode=$loss_mode, use_vce=$use_vce, use_geodesic=$use_geodesic"
        echo "   Seed: $SEED"
        
        # 构建 USER_PARAMS（所有训练超参）
        USER_PARAMS="--dataset=${dataset} \
            --seed=${SEED} \
            --loss_mode=${loss_mode} \
            --use_vce=${use_vce} \
            --use_geodesic=${use_geodesic} \
            --lr=${LR} \
            --train_batch_size=${TRAIN_BATCH_SIZE} \
            --rollout_n=${ROLLOUT_N} \
            --model_name=${MODEL_NAME} \
            --alpha=${ALPHA} \
            --distill_topk=${DISTILL_TOPK} \
            --dont_reprompt_on_self_success=${DONT_REPROMPT_ON_SELF_SUCCESS} \
            --clip_value=${CLIP_VALUE} \
            --adv_std_floor=${ADV_STD_FLOOR} \
            --geodesic_trust_region=${GEODESIC_TRUST_REGION} \
            --geodesic_beta_scale=${GEODESIC_BETA_SCALE} \
            --project_name=${PROJECT_NAME}"
        
        # 构建 nebulactl 命令（标准化格式）
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
                --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} ${USER_PARAMS}" \
                --worker_count=$WORLD_SIZE \
                --file.cluster_file=$CLUSTER_FILE \
                --job_name=$JOB_NAME \
                --env=OPENLM_TOKEN=$OPENLM_TOKEN \
                --env=OSS_ACCESS_ID=$OSS_ACCESS_ID \
                --env=OSS_ACCESS_KEY=$OSS_ACCESS_KEY \
                --env=OSS_ENDPOINT=$OSS_ENDPOINT \
                --env=OSS_BUCKET=$OSS_BUCKET \
                --env=SWANLAB_API_KEY=${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal} \
                --custom_docker_image=$CUSTOM_DOCKER_IMAGE \
                --requirements_file_name=requirements_nebula.txt \
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
echo "📊 实验矩阵:"
echo "  A: SDPO baseline (loss_mode=sdpo, use_geodesic=False)"
echo "  B: V_CE baseline (loss_mode=self_teacher, use_vce=True, use_geodesic=False)"
echo "  C: V_CE + Geodesic (loss_mode=self_teacher, use_vce=True, use_geodesic=True)"
echo "  D: SDPO + Geodesic (loss_mode=sdpo, use_geodesic=True)"
echo ""
echo "🔬 核心验证:"
echo "  1. V_CE entropy collapse 复现 (B vs A)"
echo "  2. Geodesic 免疫熵崩溃 (C vs B)"
echo "  3. SDPO+Geodesic vs VCE+Geodesic (D vs C)"
echo ""
echo "🔍 监控方式:"
echo "  - SwanLab: actor/entropy, actor/geodesic_*, val/accuracy"
echo "  - 钉钉通知: 训练异常自动告警"
echo "  - Seed: 42 (所有实验统一)"
echo "========================================================================="
