#!/bin/bash
# =============================================================================
# Teacher-QV Ablation - Phase 3: Group baseline 双层归一 vs 单层
#
# 跑 research/geodesic_vce_ablation.md §IV Phase 3 的 3 组实验：
#   G:  qv_group_mean        baseline_type=group_mean,  norm_by_std=True
#   H:  qv_group_hier        baseline_type=group_hier,  norm_by_std=True
#   H': qv_group_hier_no_std baseline_type=group_hier,  norm_by_std=False  (仅中心化)
#
# 目的：
#   - H vs G : 双层（within-seq z + across-seq z）相比单层全局 z 是否更稳更准
#   - H vs H': std 归一化是必要的还是 over-normalization
#
# 与 Phase 1 共用同一份 parametric 脚本，q_source=log_teacher，
# 全程 use_geodesic=False（这一组先不引入 Geodesic 这个轴）。
#
# 用法：
#   bash nebula_scripts/submit_teacher_qv_phase3.sh [--dry-run]
# =============================================================================

QUEUE="lazada_llm_ad_h20"
WORLD_SIZE=1
OPENLM_TOKEN="${OPENLM_TOKEN:?OPENLM_TOKEN not set}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:?OSS_ACCESS_ID not set}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?OSS_ACCESS_KEY not set}"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
CLUSTER_FILE="nebula_scripts/cluster.json"
SCRIPT_PATH="nebula_scripts/sdpo/teacher_qv_sciknoweval_parametric.sh"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
# Unified SwanLab project — Phase 1 + Phase 3 share one dashboard.
# JOB_NAME prefixes (QV1- / QV3-) distinguish the phase.
PROJECT_NAME="Teacher-QV-Ablation"

GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

DATASETS=(
    "sciknoweval/biology"
)

DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# ── 共享超参（与 Phase 1 一致；唯一变量是 baseline_type / norm_by_std） ─
SEED="42"
LR="1e-5"
TRAIN_BATCH_SIZE="32"
ROLLOUT_N="8"
MODEL_NAME="Qwen3-8B"
DONT_REPROMPT_ON_SELF_SUCCESS="True"
CLIP_VALUE="null"
STD_FLOOR="1e-3"
USE_GEODESIC="False"
GRADIENT_MODE="full_logit"   # vocab-summed PG (V drops out for group baselines, gradient = ∇CE)
GEODESIC_TRUST_REGION="5.0"

# 实验矩阵：(name, baseline_type, norm_by_std)
EXPERIMENTS=(
    "qv_group_mean:group_mean:True"
    "qv_group_hier:group_hier:True"
    "qv_group_hier_no_std:group_hier:False"
)

echo "========================================================================="
echo "🚀 Teacher-QV Phase 3 提交（group baseline 双层 vs 单层）"
echo "========================================================================="
echo "项目: $PROJECT_NAME    队列: $QUEUE    数据集: ${DATASETS[*]}"
echo "实验数: ${#EXPERIMENTS[@]}    Seed: $SEED    Git: $GIT_BRANCH@$GIT_COMMIT"
echo "========================================================================="

for dataset in "${DATASETS[@]}"; do
    dataset_name=$(echo "$dataset" | tr '/' '_')

    for exp in "${EXPERIMENTS[@]}"; do
        IFS=':' read -r exp_name baseline_type norm_by_std <<< "$exp"

        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        TASK_NAME="${PROJECT_NAME}-${dataset_name}-${exp_name}-fl-${TIMESTAMP}"
        # -fl suffix marks the new full_logit gradient_mode runs.
        JOB_NAME="QV3-${dataset_name}-${exp_name}-fl"

        echo ""
        echo "📝 $TASK_NAME"
        echo "   baseline_type=${baseline_type}  norm_by_std=${norm_by_std}"

        ENV_PARAMS="--env=PROJECT_NAME=${PROJECT_NAME} \
            --env=JOB_NAME=${JOB_NAME} \
            --env=DATASET=${dataset} \
            --env=SEED=${SEED} \
            --env=LR=${LR} \
            --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} \
            --env=ROLLOUT_N=${ROLLOUT_N} \
            --env=MODEL_NAME=${MODEL_NAME} \
            --env=DONT_REPROMPT_ON_SELF_SUCCESS=${DONT_REPROMPT_ON_SELF_SUCCESS} \
            --env=BASELINE_TYPE=${baseline_type} \
            --env=NORM_BY_STD=${norm_by_std} \
            --env=USE_GEODESIC=${USE_GEODESIC} \
            --env=GEODESIC_TRUST_REGION=${GEODESIC_TRUST_REGION} \
            --env=CLIP_VALUE=${CLIP_VALUE} \
            --env=STD_FLOOR=${STD_FLOOR} \
            --env=GRADIENT_MODE=${GRADIENT_MODE} \
            --env=GIT_BRANCH=${GIT_BRANCH} \
            --env=GIT_COMMIT=${GIT_COMMIT}"

        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] nebulactl run mdl --queue=$QUEUE --job_name=$JOB_NAME"
            echo "          script=$SCRIPT_PATH"
            echo "          ENV: BASELINE_TYPE=${baseline_type} NORM_BY_STD=${norm_by_std}"
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
                --env=OSS_ACCESS_ID=$OSS_ACCESS_ID \
                --env=OSS_ACCESS_KEY=$OSS_ACCESS_KEY \
                --env=OSS_ENDPOINT=$OSS_ENDPOINT \
                --env=OSS_BUCKET=$OSS_BUCKET \
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
echo "✅ Phase 3 提交完成"
echo "========================================================================="
echo "实验矩阵 (research/geodesic_vce_ablation.md §IV Phase 3):"
echo "  G:  qv_group_mean         baseline=group_mean, norm=True   (单层全局 z)"
echo "  H:  qv_group_hier         baseline=group_hier, norm=True   (双层: within-seq z + across-seq z)"
echo "  H': qv_group_hier_no_std  baseline=group_hier, norm=False  (双层但只中心化、不除 std)"
echo ""
echo "关键监控指标 (SwanLab):"
echo "  - actor/teacher_qv_adv_std       三组的 advantage 量纲是否可比"
echo "  - actor/entropy                  双层是否比单层熵更稳"
echo "  - val-core/sciknoweval/acc/mean@16  终端准确率：H 是否 ≥ G ≥ H'"
echo "========================================================================="
