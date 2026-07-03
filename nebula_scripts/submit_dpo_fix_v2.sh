#!/bin/bash
# =============================================================================
# DPO-FIX-V2 Batch Submission — 验证 DPO branching 修复后的效果
#
# 三组实验：
#   Group 1: Smoke Test（快速验证修复正确性，小配置不崩溃即可）
#   Group 2: Full DPO（修复后完整实验，3-way ranking via stage1_pair=True）
#   Group 3: Ablation（去掉 3-way ranking，仅 2-way DPO，对照 Group 2）
#
# Usage:
#   bash nebula_scripts/submit_dpo_fix_v2.sh [--dry-run]
#   bash nebula_scripts/submit_dpo_fix_v2.sh --group smoke [--dry-run]
#   bash nebula_scripts/submit_dpo_fix_v2.sh --group full [--dry-run]
#   bash nebula_scripts/submit_dpo_fix_v2.sh --group ablation [--dry-run]
# =============================================================================
set -euo pipefail

# ── Nebula 平台配置 ──────────────────────────────────────────────────────
QUEUE="lazada_llm_ad_h20"
WORLD_SIZE=1
OPENLM_TOKEN="${OPENLM_TOKEN:?OPENLM_TOKEN not set}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:?OSS_ACCESS_ID not set}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?OSS_ACCESS_KEY not set}"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
CLUSTER_FILE="nebula_scripts/cluster.json"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"

# ── Git 信息自动获取 ─────────────────────────────────────────────────────
GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo unknown)}"

# ── SwanLab API Key（带默认值回退）──────────────────────────────────────
SWANLAB_API_KEY="${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal}"

# ── 命令行参数解析 ───────────────────────────────────────────────────────
DRY_RUN=false
GROUP_FILTER="all"  # all | smoke | full | ablation

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --group=*) GROUP_FILTER="${arg#*=}" ;;
    esac
done

[ "$DRY_RUN" = true ] && echo "🔍 Dry-run 模式：只打印命令，不提交"

# =============================================================================
# 公共超参
# =============================================================================
MODEL="Qwen3-8B"
SEED=42
LR="1e-5"
TRAIN_BATCH_SIZE=32
ROLLOUT_N=8
MINI_BATCH_SIZE=32
TEST_FREQ=10
SAVE_FREQ=10
TP=1
BETA="1.0"  # DPO coefficient

# Branching 公共参数
TOP_K=10
TEACHER_TOP_K=100
ENTROPY_WINDOW=20
ENTROPY_SIGMA_START=2.0
ENTROPY_SIGMA_FLOOR=0.5
ENTROPY_SIGMA_STEP=0.5
TEACHER_CONTEXT_MODE=marker
ADV_STD_FLOOR=0.05
TWO_STAGE_TEACHER_MODE=ref_or_marker
SUCCESS_THRESHOLD=0.3

# 入口脚本（branching 参数化脚本）
SCRIPT_PATH="nebula_scripts/grpo/grpo_branching_sciknoweval_parametric.sh"

PROJECT_NAME="DPO-FIX-V2"
GROUP_NAME="DPO-FIX-V2-$(date +%Y%m%d)"

# =============================================================================
# 计数器
# =============================================================================
TOTAL=0
SUBMITTED=0
FAILED=0

# =============================================================================
# 核心提交函数
# =============================================================================
_submit_job() {
    local SCRIPT="$1"
    local JOB_NAME="$2"
    local USER_PARAMS="$3"

    TOTAL=$((TOTAL + 1))

    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  script: ${SCRIPT}"
        echo "  params: ${USER_PARAMS}"
        return 0
    fi

    echo "提交 Job #${TOTAL}: ${JOB_NAME}"
    SUBMIT_OUTPUT=$(nebulactl run mdl \
        --force \
        --engine=xdl \
        --queue=${QUEUE} \
        --entry=nebula_scripts/entry.py \
        --user_params="--script_path=${SCRIPT} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} ${USER_PARAMS}" \
        --worker_count=${WORLD_SIZE} \
        --file.cluster_file=${CLUSTER_FILE} \
        --job_name=${JOB_NAME} \
        --env=OPENLM_TOKEN=${OPENLM_TOKEN} \
        --env=SWANLAB_API_KEY=${SWANLAB_API_KEY} \
        --env=GIT_BRANCH=${GIT_BRANCH} \
        --env=GIT_COMMIT=${GIT_COMMIT} \
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
        FAILED=$((FAILED + 1))
    else
        SUBMITTED=$((SUBMITTED + 1))
        echo "✅ 已提交 (${SUBMITTED}/${TOTAL})"
    fi
    sleep 2
}

# =============================================================================
# 参数构建辅助函数
# =============================================================================

# 基础训练参数
_base_params() {
    local JOB_NAME="$1" DATASET="$2" TOTAL_STEPS="$3"
    echo "--env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL_NAME=${MODEL} --env=LR=${LR} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=SEED=${SEED} --env=TEST_FREQ=${TEST_FREQ} --env=SAVE_FREQ=${SAVE_FREQ} --env=VAL_BEFORE_TRAIN=True --env=SAVE_HF_ONLY=True --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=TOTAL_TRAINING_STEPS=${TOTAL_STEPS} --env=TENSOR_MODEL_PARALLEL_SIZE=${TP} --env=GROUP_NAME=${GROUP_NAME} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE}"
}

# Branching 公共参数
_branch_params() {
    echo "--env=BRANCHING_ENABLED=True --env=TOP_K=${TOP_K} --env=TEACHER_TOP_K=${TEACHER_TOP_K} --env=ENTROPY_WINDOW=${ENTROPY_WINDOW} --env=ENTROPY_SIGMA_START=${ENTROPY_SIGMA_START} --env=ENTROPY_SIGMA_FLOOR=${ENTROPY_SIGMA_FLOOR} --env=ENTROPY_SIGMA_STEP=${ENTROPY_SIGMA_STEP} --env=TEACHER_CONTEXT_MODE=${TEACHER_CONTEXT_MODE} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR}"
}

# DPO 两阶段参数
# $1: N_TREES, $2: STAGE1_N, $3: N_SPLITS, $4: DPO_STAGE1_PAIR (True/False)
_dpo_params() {
    local N_TREES="$1" STAGE1_N="$2" N_SPLITS="$3" DPO_STAGE1_PAIR="$4"
    echo "--env=TWO_STAGE=True --env=STAGE1_N=${STAGE1_N} --env=N_SPLITS=${N_SPLITS} --env=N_TREES=${N_TREES} --env=BRANCH_TOKEN_LOSS_MODE=suffix --env=TWO_STAGE_TEACHER_MODE=${TWO_STAGE_TEACHER_MODE} --env=SUCCESS_THRESHOLD=${SUCCESS_THRESHOLD} --env=DPO_COEFFICIENT=${BETA} --env=DPO_USE_REF=True --env=ENTROPY_COEFF=0 --env=DPO_STAGE1_PAIR=${DPO_STAGE1_PAIR} --env=STAGE1_PAIR_WEIGHT=1.0"
}

# =============================================================================
# Group 1: Smoke Test — 快速验证修复正确性
# 小配置: n_trees=2, stage1_n=2, n_splits=1
# 只跑少量 steps，验证不崩溃
# =============================================================================
_smoke_test() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Group 1: Smoke Test — 快速验证修复正确性                   ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    local N_TREES=2 STAGE1_N=2 N_SPLITS=1
    local TOTAL_STEPS=30  # 少量 steps 足以验证不崩溃
    local DATASET="tooluse"
    local DATE_TAG=$(date +%Y%m%d)

    local JN="DPO-fix-v2-${DATASET}-smoke-${MODEL}-${DATE_TAG}"
    _submit_job "$SCRIPT_PATH" "$JN" \
        "$(_base_params "$JN" "$DATASET" "$TOTAL_STEPS") $(_branch_params) $(_dpo_params "$N_TREES" "$STAGE1_N" "$N_SPLITS" "True")"
}

# =============================================================================
# Group 2: Full DPO — 修复后完整实验（3-way ranking）
# 完整配置: n_trees=4, stage1_n=4, n_splits=1
# dpo_stage1_pair=True（启用 3-way ranking）
# 数据集: sciknoweval-biology + tooluse
# =============================================================================
_full_dpo() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Group 2: Full DPO — 修复后完整实验 (3-way ranking)        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    local N_TREES=4 STAGE1_N=4 N_SPLITS=1
    local DATE_TAG=$(date +%Y%m%d)

    # tooluse: 250 steps
    local DATASET="tooluse"
    local JN="DPO-fix-v2-${DATASET}-full3way-${MODEL}-${DATE_TAG}"
    _submit_job "$SCRIPT_PATH" "$JN" \
        "$(_base_params "$JN" "$DATASET" 250) $(_branch_params) $(_dpo_params "$N_TREES" "$STAGE1_N" "$N_SPLITS" "True")"

    # sciknoweval-biology: 300 steps
    DATASET="sciknoweval/biology"
    JN="DPO-fix-v2-sciknoweval-biology-full3way-${MODEL}-${DATE_TAG}"
    _submit_job "$SCRIPT_PATH" "$JN" \
        "$(_base_params "$JN" "$DATASET" 300) $(_branch_params) $(_dpo_params "$N_TREES" "$STAGE1_N" "$N_SPLITS" "True")"
}

# =============================================================================
# Group 3: Ablation — 去掉 3-way ranking（对照组）
# 同样 n_trees=4, stage1_n=4, n_splits=1
# dpo_stage1_pair=False（只用标准 2-way DPO）
# 验证 3-way ranking 是否有额外收益
# =============================================================================
_ablation_no_s1pair() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  Group 3: Ablation — 标准 2-way DPO（无 3-way ranking）    ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    local N_TREES=4 STAGE1_N=4 N_SPLITS=1
    local DATE_TAG=$(date +%Y%m%d)

    # tooluse: 250 steps
    local DATASET="tooluse"
    local JN="DPO-fix-v2-${DATASET}-2way-${MODEL}-${DATE_TAG}"
    _submit_job "$SCRIPT_PATH" "$JN" \
        "$(_base_params "$JN" "$DATASET" 250) $(_branch_params) $(_dpo_params "$N_TREES" "$STAGE1_N" "$N_SPLITS" "False")"

    # sciknoweval-biology: 300 steps
    DATASET="sciknoweval/biology"
    JN="DPO-fix-v2-sciknoweval-biology-2way-${MODEL}-${DATE_TAG}"
    _submit_job "$SCRIPT_PATH" "$JN" \
        "$(_base_params "$JN" "$DATASET" 300) $(_branch_params) $(_dpo_params "$N_TREES" "$STAGE1_N" "$N_SPLITS" "False")"
}

# =============================================================================
# 执行
# =============================================================================
echo "============================================================"
echo "DPO-FIX-V2 实验提交"
echo "  Model: ${MODEL} | LR: ${LR} | DPO β: ${BETA}"
echo "  Branch: ${GIT_BRANCH} | Commit: ${GIT_COMMIT}"
echo "  Group: ${GROUP_NAME}"
echo "  Filter: ${GROUP_FILTER}"
echo "============================================================"

if [[ "$GROUP_FILTER" == "all" || "$GROUP_FILTER" == "smoke" ]]; then
    _smoke_test
fi

if [[ "$GROUP_FILTER" == "all" || "$GROUP_FILTER" == "full" ]]; then
    _full_dpo
fi

if [[ "$GROUP_FILTER" == "all" || "$GROUP_FILTER" == "ablation" ]]; then
    _ablation_no_s1pair
fi

# =============================================================================
# 提交统计
# =============================================================================
echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "🔍 Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "📊 提交统计："
    echo "   总数:   ${TOTAL}"
    echo "   成功:   ${SUBMITTED}"
    echo "   失败:   ${FAILED}"
fi
echo "============================================================"
