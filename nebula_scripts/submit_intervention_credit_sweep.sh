#!/bin/bash
# =============================================================================
# Teacher-Guided Dynamic Intervention (TGDI / v3) - Nebula 提交脚本
#
# 论文叙事：ΔR (causal counterfactual credit) 是 base-agnostic 的，
# 在最强 baseline (RLSD, val=0.585) 之上叠加，看是否还能涨。
#
# 默认 sweep (Phase 3, 2 jobs) — 验证 ΔR 因果层在 GRPO / RLSD 上的增益:
#   Job 1: GRPO + ΔR (base_estimator=grpo, enable_intervention=True)
#   Job 2: RLSD + ΔR (base_estimator=rlsd, enable_intervention=True)
#
# 通过环境变量切换其他 mode:
#   # 跑 prior_shift 历史路径（已知 v2 best 0.33 < RLSD 0.585）
#   IC_BASE_ESTIMATORS="prior_shift" bash submit_intervention_credit_sweep.sh
#
#   # 关 intervention 跑纯 base baseline (用于补对照)
#   IC_ENABLE_INTERVENTION_LIST="False" bash submit_intervention_credit_sweep.sh
#
#   # 多 t* 策略消融（仅 enable_intervention=True 时有意义）
#   IC_DIVERGENCE_METRIC_LIST="argmax_excl_eos argmax_raw g_t_argmax" bash submit_intervention_credit_sweep.sh
#
# 使用方式:
#   bash nebula_scripts/submit_intervention_credit_sweep.sh [--dry-run]
# =============================================================================
# set -euo pipefail

# ── Nebula 账号配置 ──────────────────────────────────────────────────────
QUEUE="lazada_llm_ad_h20"
WORLD_SIZE=1
OPENLM_TOKEN="${OPENLM_TOKEN:?OPENLM_TOKEN not set}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:?OSS_ACCESS_ID not set}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?OSS_ACCESS_KEY not set}"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
CLUSTER_FILE="nebula_scripts/cluster.json"
SCRIPT_PATH="nebula_scripts/intervention_credit/intervention_credit_sciknoweval_parametric.sh"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="TGDI-Tier3"

# ── 数据集配置 (可通过 DATASETS_OVERRIDE env var 覆盖, 空格分隔) ────────
# 示例:
#   DATASETS_OVERRIDE="sciknoweval/biology sciknoweval/chemistry" bash submit_..sh
DATASETS_DEFAULT="sciknoweval/biology"
read -r -a DATASETS <<< "${DATASETS_OVERRIDE:-$DATASETS_DEFAULT}"

# ── dry-run 模式 ─────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 超参配置（默认 Phase 3：base × {grpo, rlsd}，single t* strategy）
# =============================================================================

# ── 是否开 intervention (默认开，跑论文主结果) ──────────────────────────
IC_ENABLE_INTERVENTION_LIST_DEFAULT="True"
read -r -a IC_ENABLE_INTERVENTION_LIST <<< "${IC_ENABLE_INTERVENTION_LIST:-$IC_ENABLE_INTERVENTION_LIST_DEFAULT}"

# ── λ_div_credit ablation (论文核心) ─────────────────────────────────
IC_LAMBDA_DIV_CREDIT_LIST_DEFAULT="1.0"
read -r -a IC_LAMBDA_DIV_CREDIT_LIST <<< "${IC_LAMBDA_DIV_CREDIT_LIST:-$IC_LAMBDA_DIV_CREDIT_LIST_DEFAULT}"

# ── TCCA-Lite 默认超参 ─────────────────────────────────────────────
IC_DIVERGENCE_METRIC="argmax_excl_eos"
IC_EXCLUDE_TAIL_TOKENS="8"
IC_FAILED_THRESHOLD="0.5"
IC_MAX_INTERVENTION_PER_PROMPT="4"
IC_TEACHER_DECODE_TEMPERATURE="0.0"
IC_DIVERGENCE_CREDIT_CLIP="1.0"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"   # 共享 GPU 时降到 0.7

# ── Length floor (复用 PS v2 防护) ──────────────────────────────────
IC_MIN_RESPONSE_LENGTH="50"
IC_LENGTH_PENALTY_TYPE="linear"

# ── Teacher EMA ─────────────────────────────────────────────────────
TEACHER_UPDATE_RATE_LIST=(
    "0.05"
)
TEACHER_REGULARIZATION="ema"

# ── 固定参数 ───────────────────────────────────────────────────────────
LR="1e-5"
SEED="42"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
ROLLOUT_N="7"   # n_initial=7（比 baseline n=8 少 1，腾出 append 配额）
MODEL_NAME="Qwen3-8B"

# Git 信息
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# =============================================================================
# 提交循环
# =============================================================================
TOTAL=0
SUBMITTED=0

for DATASET in "${DATASETS[@]}"; do
for IC_ENABLE_INTERVENTION in "${IC_ENABLE_INTERVENTION_LIST[@]}"; do
for IC_LAMBDA_DIV_CREDIT in "${IC_LAMBDA_DIV_CREDIT_LIST[@]}"; do
for TEACHER_UPDATE_RATE in "${TEACHER_UPDATE_RATE_LIST[@]}"; do
    TOTAL=$((TOTAL + 1))

    DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
    MODEL_SHORT=$(echo "$MODEL_NAME" | tr '.' '-')
    OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
    MODEL_PATH="${OSS_ROOT}/base_models/${MODEL_NAME}"

    # Tags: 体现核心维度
    if [ "$IC_ENABLE_INTERVENTION" = "True" ]; then
        DR_TAG="dR-l${IC_LAMBDA_DIV_CREDIT}"
        VERSION_TAG="TCCA-Lite"
    else
        DR_TAG="noDR"
        VERSION_TAG="GRPO-base"
    fi
    EMA_TAG="ema${TEACHER_UPDATE_RATE}"
    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)

    JOB_NAME="${VERSION_TAG}-${DR_TAG}-${DATASET_SHORT}-${EMA_TAG}-${MODEL_SHORT}-${CURRENT_TIME}"

    # ── 提交 ────────────────────────────────────────────────────────
    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  DATASET=$DATASET MODEL=$MODEL_NAME"
        echo "  IC_ENABLE_INTERVENTION=$IC_ENABLE_INTERVENTION  IC_LAMBDA_DIV_CREDIT=$IC_LAMBDA_DIV_CREDIT"
        echo "  IC_DIVERGENCE_METRIC=$IC_DIVERGENCE_METRIC  IC_FAILED_THRESHOLD=$IC_FAILED_THRESHOLD  IC_MAX_INT_PER_PROMPT=$IC_MAX_INTERVENTION_PER_PROMPT"
        echo "  IC_DIVERGENCE_CREDIT_CLIP=$IC_DIVERGENCE_CREDIT_CLIP"
        echo "  GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
        echo "  TEACHER_REG=$TEACHER_REGULARIZATION TEACHER_UPDATE_RATE=$TEACHER_UPDATE_RATE"
        echo "  LR=$LR BS=$TRAIN_BATCH_SIZE/$MINI_BATCH_SIZE N=$ROLLOUT_N"
        echo "  GIT_BRANCH=$GIT_BRANCH GIT_COMMIT=$GIT_COMMIT"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
            --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL_NAME=${MODEL_NAME} --env=MODEL_PATH=${MODEL_PATH} --env=LR=${LR} --env=SEED=${SEED} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=TEACHER_REGULARIZATION=${TEACHER_REGULARIZATION} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=IC_ENABLE_INTERVENTION=${IC_ENABLE_INTERVENTION} --env=IC_DIVERGENCE_METRIC=${IC_DIVERGENCE_METRIC} --env=IC_EXCLUDE_TAIL_TOKENS=${IC_EXCLUDE_TAIL_TOKENS} --env=IC_FAILED_THRESHOLD=${IC_FAILED_THRESHOLD} --env=IC_MAX_INTERVENTION_PER_PROMPT=${IC_MAX_INTERVENTION_PER_PROMPT} --env=IC_TEACHER_DECODE_TEMPERATURE=${IC_TEACHER_DECODE_TEMPERATURE} --env=IC_LAMBDA_DIV_CREDIT=${IC_LAMBDA_DIV_CREDIT} --env=IC_DIVERGENCE_CREDIT_CLIP=${IC_DIVERGENCE_CREDIT_CLIP} --env=IC_MIN_RESPONSE_LENGTH=${IC_MIN_RESPONSE_LENGTH} --env=IC_LENGTH_PENALTY_TYPE=${IC_LENGTH_PENALTY_TYPE} --env=GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=f598ad33b071751bf79d2484d8e1acefe8df9d879e129cae40340a158854f9cb --env=DINGTALK_SECRET=SECc5b9e4f61f56b32b46abf1ecedc11bdcba10dc35fbba8fa0ff62c084a1cc6ad3" \
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
done

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
    echo ""
    echo "[TCCA-Lite 默认 sweep] enable_intervention=True × lambda_div_credit=1.0 × 1 dataset = 1 job"
    echo "  → 论文核心: GRPO base + TCCA-Lite divergence-point credit modulation"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
