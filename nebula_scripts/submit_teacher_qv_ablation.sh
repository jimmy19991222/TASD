#!/bin/bash
# =============================================================================
# Teacher-QV Ablation - Phase 1: 真·V_CE 与 SDPO ± Geodesic
#
# 跑 research/geodesic_vce_ablation.md 第 IV 节 Phase 1 的 4 组实验：
#   A: qv_student        (SDPO sampled-token, baseline_type=student, geodesic=False)
#   B: qv_ce             (真·V_CE,            baseline_type=ce,      geodesic=False)
#   C: qv_ce_geodesic    (V_CE + Geodesic,    baseline_type=ce,      geodesic=True)
#   D: qv_student_geodesic (SDPO + Geodesic,  baseline_type=student, geodesic=True)
#
# 核心科学问题：
#   - B 是否复现 entropy collapse？（验证 V_CE 假说）
#   - C 是否被 Geodesic 救回？（验证 Geodesic 免疫熵崩溃）
#   - D vs A 的差异 = Geodesic 在干净 SDPO 上的净增量
#
# 用法：
#   bash nebula_scripts/submit_teacher_qv_ablation.sh [--dry-run]
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
PROJECT_NAME="Teacher-QV-Phase1"

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

# ── 共享超参（严格统一，唯一变量是 baseline_type / use_geodesic） ─────
SEED="42"
LR="1e-5"
TRAIN_BATCH_SIZE="32"
ROLLOUT_N="8"
MODEL_NAME="Qwen3-8B"
DONT_REPROMPT_ON_SELF_SUCCESS="True"
NORM_BY_STD="False"
CLIP_VALUE="null"
STD_FLOOR="1e-3"
DETACH_Q="True"
DETACH_V="True"
GEODESIC_TRUST_REGION="5.0"

# 实验矩阵：(name, baseline_type, use_geodesic)
EXPERIMENTS=(
    "qv_student:student:False"
    "qv_ce:ce:False"
    "qv_ce_geodesic:ce:True"
    "qv_student_geodesic:student:True"
)

echo "========================================================================="
echo "🚀 Teacher-QV Phase 1 提交（V_CE × Geodesic 4 组）"
echo "========================================================================="
echo "项目: $PROJECT_NAME    队列: $QUEUE    数据集: ${DATASETS[*]}"
echo "实验数: ${#EXPERIMENTS[@]}    Seed: $SEED    Git: $GIT_BRANCH@$GIT_COMMIT"
echo "========================================================================="

for dataset in "${DATASETS[@]}"; do
    dataset_name=$(echo "$dataset" | tr '/' '_')

    for exp in "${EXPERIMENTS[@]}"; do
        IFS=':' read -r exp_name baseline_type use_geodesic <<< "$exp"

        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        TASK_NAME="${PROJECT_NAME}-${dataset_name}-${exp_name}-${TIMESTAMP}"
        JOB_NAME="QV1-${dataset_name}-${exp_name}"

        echo ""
        echo "📝 $TASK_NAME"
        echo "   baseline_type=${baseline_type}  use_geodesic=${use_geodesic}"

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
            --env=USE_GEODESIC=${use_geodesic} \
            --env=GEODESIC_TRUST_REGION=${GEODESIC_TRUST_REGION} \
            --env=NORM_BY_STD=${NORM_BY_STD} \
            --env=CLIP_VALUE=${CLIP_VALUE} \
            --env=STD_FLOOR=${STD_FLOOR} \
            --env=DETACH_Q=${DETACH_Q} \
            --env=DETACH_V=${DETACH_V} \
            --env=GIT_BRANCH=${GIT_BRANCH} \
            --env=GIT_COMMIT=${GIT_COMMIT}"

        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] nebulactl run mdl --queue=$QUEUE --job_name=$JOB_NAME"
            echo "          script=$SCRIPT_PATH"
            echo "          ENV: BASELINE_TYPE=${baseline_type} USE_GEODESIC=${use_geodesic}"
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
echo "✅ Phase 1 提交完成"
echo "========================================================================="
echo "实验矩阵 (research/geodesic_vce_ablation.md §IV Phase 1):"
echo "  A: qv_student         baseline_type=student, use_geodesic=False  (SDPO baseline)"
echo "  B: qv_ce              baseline_type=ce,      use_geodesic=False  (真·V_CE，预计崩溃)"
echo "  C: qv_ce_geodesic     baseline_type=ce,      use_geodesic=True   (V_CE + Geodesic 救回)"
echo "  D: qv_student_geodesic baseline_type=student, use_geodesic=True  (SDPO + Geodesic 控制组)"
echo ""
echo "关键监控指标 (SwanLab):"
echo "  - actor/entropy                         B 是否前 50 步内崩到 ~0"
echo "  - actor/teacher_qv_adv_mean             B/C 应 ≈ 0 (zero-mean baseline)"
echo "  - actor/teacher_qv_adv_std              A vs B 量纲差异"
echo "  - actor/geodesic_mean_weight            C/D 才有，应稳定在 2-8"
echo "  - val-core/sciknoweval/acc/mean@16      终端准确率"
echo "========================================================================="
