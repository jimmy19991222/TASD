#!/bin/bash
# =============================================================================
# Teacher-Guided Dynamic Intervention (TGDI / v3) - Nebula 提交脚本
#
# Phase 1 (本提交): IC_ENABLE_INTERVENTION=False → 退化为 prior_shift-equivalent
#                    用于验证 plumbing；SwanLab 上能看到 failed_rate / t* 分布
# Phase 2 (待实现): IC_ENABLE_INTERVENTION=True → 真实 teacher generation + ΔR
#
# 提交 3 个 job (t* 选择策略消融):
#   Job 1: argmax_excl_eos  — 排除尾部 EOS/punctuation 后取 |logp_T - logp_S| argmax
#   Job 2: argmax_raw       — 纯 |logp_T - logp_S| argmax (对照)
#   Job 3: g_t_argmax       — 复用 prior_shift 的 g_t = KL(D_t‖D_{t-1}) 的 argmax
#
# 使用方式:
#   bash nebula_scripts/submit_intervention_credit_sweep.sh [--dry-run]
#   IC_ENABLE_INTERVENTION=True bash ...sh   # Phase 2 接通后切到真 intervention
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

# ── Phase 控制 ────────────────────────────────────────────────────────
# Phase 1 默认 False（退化 prior_shift），Phase 2 实现完后改为 True
IC_ENABLE_INTERVENTION="${IC_ENABLE_INTERVENTION:-False}"

# ── t* 选择策略消融维度 (3 个 job) ──────────────────────────────────
IC_DIVERGENCE_METRIC_LIST=(
    "argmax_excl_eos"   # 推荐：排除尾部 N=8 token 后取 argmax
    "argmax_raw"        # 对照：纯 argmax，可能选到 EOS/punctuation
    "g_t_argmax"        # 复用 prior_shift 的 g_t argmax
)
IC_EXCLUDE_TAIL_TOKENS="8"

# ── intervention 长度 / 失败阈值 / λ ────────────────────────────────
IC_K="2"
IC_FAILED_THRESHOLD="0.5"
IC_MAX_INTERVENTION_PER_GROUP="7"
IC_TEACHER_DECODE_TEMPERATURE="0.0"
IC_LAMBDA_DR="1.0"

# ── g_t 防护（与 PS v2 对齐）──────────────────────────────────────────
IC_GT_EPS_NORM="1.0e-6"
IC_GT_MAX_RATIO="3.0"
IC_GT_RENORMALIZE_AFTER_CLIP="True"
IC_GT_UNIFORM_FALLBACK="True"

# ── Length floor (与 PS v2 对齐) ──────────────────────────────────────
IC_MIN_RESPONSE_LENGTH="50"
IC_LENGTH_PENALTY_TYPE="linear"

# ── Teacher EMA ──────────────────────────────────────────────────────
TEACHER_UPDATE_RATE_LIST=(
    "0.05"
)
TEACHER_REGULARIZATION="ema"

# ── 固定参数 ────────────────────────────────────────────────────────────
LR="1e-5"
SEED="42"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
ROLLOUT_N="7"   # n_initial=7（比 baseline n=8 少 1，腾出 append 配额）
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
for IC_DIVERGENCE_METRIC in "${IC_DIVERGENCE_METRIC_LIST[@]}"; do
for TEACHER_UPDATE_RATE in "${TEACHER_UPDATE_RATE_LIST[@]}"; do
    TOTAL=$((TOTAL + 1))

    # 构建短数据集名
    DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
    MODEL_SHORT=$(echo "$MODEL_NAME" | tr '.' '-')
    OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
    MODEL_PATH="${OSS_ROOT}/base_models/${MODEL_NAME}"

    # Tags
    if [ "$IC_ENABLE_INTERVENTION" = "True" ]; then
        VERSION_TAG="v3"
    else
        VERSION_TAG="v3p1"   # Phase 1: plumbing-only
    fi
    TSTAR_TAG=$(echo "$IC_DIVERGENCE_METRIC" | sed 's/argmax_excl_eos/aexEOS/; s/argmax_raw/araw/; s/g_t_argmax/gtarg/')
    K_TAG="-k${IC_K}"
    EMA_TAG="-ema${TEACHER_UPDATE_RATE}"
    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)

    JOB_NAME="TGDI-${VERSION_TAG}-${DATASET_SHORT}-${TSTAR_TAG}${K_TAG}${EMA_TAG}-${MODEL_SHORT}-${CURRENT_TIME}"

    # ── 提交 ────────────────────────────────────────────────────────
    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  DATASET=$DATASET MODEL=$MODEL_NAME"
        echo "  IC_ENABLE_INTERVENTION=$IC_ENABLE_INTERVENTION"
        echo "  IC_DIVERGENCE_METRIC=$IC_DIVERGENCE_METRIC IC_K=$IC_K IC_FAILED_THRESHOLD=$IC_FAILED_THRESHOLD"
        echo "  IC_LAMBDA_DR=$IC_LAMBDA_DR IC_MAX_INT_PER_GRP=$IC_MAX_INTERVENTION_PER_GROUP"
        echo "  IC_GT: max_ratio=$IC_GT_MAX_RATIO renorm=$IC_GT_RENORMALIZE_AFTER_CLIP fallback=$IC_GT_UNIFORM_FALLBACK"
        echo "  IC_MIN_RESP_LEN=$IC_MIN_RESPONSE_LENGTH IC_LEN_PENALTY=$IC_LENGTH_PENALTY_TYPE"
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
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL_NAME=${MODEL_NAME} --env=MODEL_PATH=${MODEL_PATH} --env=LR=${LR} --env=SEED=${SEED} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=TEACHER_REGULARIZATION=${TEACHER_REGULARIZATION} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=IC_ENABLE_INTERVENTION=${IC_ENABLE_INTERVENTION} --env=IC_DIVERGENCE_METRIC=${IC_DIVERGENCE_METRIC} --env=IC_EXCLUDE_TAIL_TOKENS=${IC_EXCLUDE_TAIL_TOKENS} --env=IC_K=${IC_K} --env=IC_FAILED_THRESHOLD=${IC_FAILED_THRESHOLD} --env=IC_MAX_INTERVENTION_PER_GROUP=${IC_MAX_INTERVENTION_PER_GROUP} --env=IC_TEACHER_DECODE_TEMPERATURE=${IC_TEACHER_DECODE_TEMPERATURE} --env=IC_LAMBDA_DR=${IC_LAMBDA_DR} --env=IC_GT_EPS_NORM=${IC_GT_EPS_NORM} --env=IC_GT_MAX_RATIO=${IC_GT_MAX_RATIO} --env=IC_GT_RENORMALIZE_AFTER_CLIP=${IC_GT_RENORMALIZE_AFTER_CLIP} --env=IC_GT_UNIFORM_FALLBACK=${IC_GT_UNIFORM_FALLBACK} --env=IC_MIN_RESPONSE_LENGTH=${IC_MIN_RESPONSE_LENGTH} --env=IC_LENGTH_PENALTY_TYPE=${IC_LENGTH_PENALTY_TYPE} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=f598ad33b071751bf79d2484d8e1acefe8df9d879e129cae40340a158854f9cb --env=DINGTALK_SECRET=SECc5b9e4f61f56b32b46abf1ecedc11bdcba10dc35fbba8fa0ff62c084a1cc6ad3" \
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
    echo ""
    echo "[Phase 注] 当前 IC_ENABLE_INTERVENTION=${IC_ENABLE_INTERVENTION}"
    if [ "$IC_ENABLE_INTERVENTION" = "False" ]; then
        echo "  → 这些 job 跑出来等价于 prior_shift（ΔR=0），用于验证 plumbing"
        echo "  → Phase 2 实现真 intervention 后，重提时设 IC_ENABLE_INTERVENTION=True"
    else
        echo "  → 这些 job 是真实 v3-full 训练（teacher generation + ΔR）"
    fi
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
