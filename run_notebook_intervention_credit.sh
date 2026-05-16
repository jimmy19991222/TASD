#!/usr/bin/env bash
# =============================================================================
# Local Teacher-Guided Dynamic Intervention (TGDI / v3) 训练脚本
# 4 卡 notebook 单机版 — 用于 Nebula 任务的本地 plumbing 验证和快速 t* 消融
#
# 设计依据: Bayesian Credit Assignment Tier 3 (ours)
#   A_seq = (R - mean_group(R)) + λ·ΔR · 𝟙[intervention sample]
#   A_t   = A_seq · ĝ_t · length_scale
#
# 三种模式:
#   smoke   (默认): 10 step bs=4 n=4 ~15 min  → 仅验证 plumbing 不崩
#   tstar   :       30 step bs=8 n=4 ~45 min  → 串行 3 个 t* 策略消融
#   full    :       250 step bs=32 n=7 ~7h    → 与 Nebula 等价的完整运行
#
# 用法:
#   ./run_notebook_intervention_credit.sh                    # smoke (默认)
#   ./run_notebook_intervention_credit.sh smoke
#   ./run_notebook_intervention_credit.sh tstar              # 3 策略消融
#   ./run_notebook_intervention_credit.sh full               # 完整 250 step
#
#   # 自定义:
#   IC_DIVERGENCE_METRIC=g_t_argmax ./run_notebook_intervention_credit.sh smoke
#   TOTAL_STEPS=20 TRAIN_BATCH_SIZE=8 ./run_notebook_intervention_credit.sh smoke
#
# 后端调用 nebula_scripts/intervention_credit/intervention_credit_sciknoweval_parametric.sh
# =============================================================================
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

# ── 模式选择 ────────────────────────────────────────────────────────────
MODE="${1:-smoke}"

case "${MODE}" in
    smoke)
        # Plumbing 验证：~15 min，只验证不崩、新指标出现
        DEFAULT_TOTAL_STEPS=10
        DEFAULT_TRAIN_BATCH_SIZE=4
        DEFAULT_MINI_BATCH_SIZE=4
        DEFAULT_ROLLOUT_N=4
        DEFAULT_VAL_N=4
        DEFAULT_VAL_BEFORE_TRAIN=False
        ;;
    tstar)
        # t* 策略消融：每个策略 30 step，串行跑完 3 个，~45 min
        DEFAULT_TOTAL_STEPS=30
        DEFAULT_TRAIN_BATCH_SIZE=8
        DEFAULT_MINI_BATCH_SIZE=8
        DEFAULT_ROLLOUT_N=4
        DEFAULT_VAL_N=8
        DEFAULT_VAL_BEFORE_TRAIN=False
        ;;
    full)
        # 与 Nebula 等价配置
        DEFAULT_TOTAL_STEPS=250
        DEFAULT_TRAIN_BATCH_SIZE=32
        DEFAULT_MINI_BATCH_SIZE=32
        DEFAULT_ROLLOUT_N=7
        DEFAULT_VAL_N=16
        DEFAULT_VAL_BEFORE_TRAIN=False
        ;;
    *)
        echo "ERROR: unknown mode '${MODE}'. Use: smoke | tstar | full"
        exit 1
        ;;
esac

# ── OSS 挂载根目录（notebook 通常已挂为本地）─────────────────────────
export OSS_ROOT="${OSS_ROOT:-/data/oss_bucket_0/ad/loujieming.ljm}"

# ── 数据集 / 模型 ───────────────────────────────────────────────────
export DATASET="${DATASET:-sciknoweval}"
export SUBJECT="${SUBJECT:-biology}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
export MODEL_PATH="${MODEL_PATH:-${OSS_ROOT}/base_models/${MODEL_NAME}}"

# ── 训练超参 (默认值由模式决定，env 可覆盖) ──────────────────────────
export TOTAL_STEPS="${TOTAL_STEPS:-${DEFAULT_TOTAL_STEPS}}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-${DEFAULT_MINI_BATCH_SIZE}}"
export ROLLOUT_N="${ROLLOUT_N:-${DEFAULT_ROLLOUT_N}}"
export VAL_N="${VAL_N:-${DEFAULT_VAL_N}}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-${DEFAULT_VAL_BEFORE_TRAIN}}"
export LR="${LR:-1e-5}"

# ── teacher 装配 ─────────────────────────────────────────────────────
export TEACHER_REGULARIZATION="${TEACHER_REGULARIZATION:-ema}"
export TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"

# ── intervention_credit 专属 (单 t* 模式默认) ─────────────────────────
export IC_ENABLE_INTERVENTION="${IC_ENABLE_INTERVENTION:-False}"
export IC_BASE_ESTIMATOR="${IC_BASE_ESTIMATOR:-grpo}"            # grpo | rlsd | prior_shift
export IC_RLSD_EPS_W="${IC_RLSD_EPS_W:-0.2}"
export IC_DIVERGENCE_METRIC="${IC_DIVERGENCE_METRIC:-argmax_excl_eos}"
export IC_EXCLUDE_TAIL_TOKENS="${IC_EXCLUDE_TAIL_TOKENS:-8}"
export IC_K="${IC_K:-2}"
export IC_TOP_K="${IC_TOP_K:-3}"                                  # TCCA top-K positions
export IC_FAILED_THRESHOLD="${IC_FAILED_THRESHOLD:-0.5}"
export IC_MAX_INTERVENTION_PER_GROUP="${IC_MAX_INTERVENTION_PER_GROUP:-9}"
export IC_TEACHER_DECODE_TEMPERATURE="${IC_TEACHER_DECODE_TEMPERATURE:-0.0}"
export IC_LAMBDA_DR="${IC_LAMBDA_DR:-0.0}"                        # legacy seq-level (TCCA 默认关)
export IC_LAMBDA_TOKEN_CREDIT="${IC_LAMBDA_TOKEN_CREDIT:-1.0}"    # TCCA 核心
export IC_TOKEN_CREDIT_CLIP="${IC_TOKEN_CREDIT_CLIP:-2.0}"
export IC_GT_EPS_NORM="${IC_GT_EPS_NORM:-1.0e-6}"
export IC_GT_MAX_RATIO="${IC_GT_MAX_RATIO:-3.0}"
export IC_GT_RENORMALIZE_AFTER_CLIP="${IC_GT_RENORMALIZE_AFTER_CLIP:-True}"
export IC_GT_UNIFORM_FALLBACK="${IC_GT_UNIFORM_FALLBACK:-True}"
export IC_MIN_RESPONSE_LENGTH="${IC_MIN_RESPONSE_LENGTH:-50}"
export IC_LENGTH_PENALTY_TYPE="${IC_LENGTH_PENALTY_TYPE:-linear}"

# ── 数据路径解析（4 级 fallback）────────────────────────────────────
export DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets}"
_candidates_train=(
    "${DATA_ROOT}/${DATASET}/${SUBJECT}/train.parquet"
    "${DATA_ROOT}/${DATASET}/train.parquet"
    "${OSS_ROOT}/datasets/${DATASET}/${SUBJECT}/train.parquet"
    "${OSS_ROOT}/datasets/${DATASET}/train.parquet"
)
if [ -z "${TRAIN_DATA_PATH:-}" ]; then
    for _cand in "${_candidates_train[@]}"; do
        if [ -f "${_cand}" ]; then
            export TRAIN_DATA_PATH="${_cand}"
            export VAL_DATA_PATH="${VAL_DATA_PATH:-${_cand%/train.parquet}/test.parquet}"
            break
        fi
    done
fi

# ── 兜底校验 ────────────────────────────────────────────────────────
if [ ! -d "${OSS_ROOT}" ]; then
    echo "⚠️  OSS_ROOT=${OSS_ROOT} 不存在（未挂载）。save_path/model_path 依赖它。"
fi
if [ -z "${TRAIN_DATA_PATH:-}" ]; then
    echo "❌ 未找到数据（试过以下候选）："
    for _cand in "${_candidates_train[@]}"; do echo "   - ${_cand}"; done
    echo "   请 export DATA_ROOT=/your/datasets 或 export TRAIN_DATA_PATH=/abs/path/train.parquet"
    exit 1
fi

# ── 单机 4 卡 ───────────────────────────────────────────────────────
export RAY_ADDRESS=""
export N_GPUS_PER_NODE=4

# ── Git 信息 ────────────────────────────────────────────────────────
export GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
export GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
export PROJECT_NAME="${PROJECT_NAME:-TGDI-local}"

# ─────────────────────────────────────────────────────────────────────
# 单 t* 策略：smoke / full 模式
# ─────────────────────────────────────────────────────────────────────
run_single_strategy() {
    local strategy="$1"
    export IC_DIVERGENCE_METRIC="${strategy}"
    local strategy_short
    case "${strategy}" in
        argmax_excl_eos) strategy_short="aexEOS" ;;
        argmax_raw)      strategy_short="araw" ;;
        g_t_argmax)      strategy_short="gtarg" ;;
        *)               strategy_short="${strategy}" ;;
    esac

    SUFFIX="${SUFFIX:-${MODE}}"
    if [ "$IC_ENABLE_INTERVENTION" = "True" ]; then
        VERSION_TAG="v3"
    else
        VERSION_TAG="v3p1"
    fi
    export JOB_NAME="${JOB_NAME:-LOCAL-TGDI-${VERSION_TAG}-${IC_BASE_ESTIMATOR}-${SUBJECT}-${strategy_short}-bs${TRAIN_BATCH_SIZE}-n${ROLLOUT_N}-step${TOTAL_STEPS}-${SUFFIX}}"

    echo "========================================================================"
    echo "Local TGDI launch (notebook, ${N_GPUS_PER_NODE}×GPU) — MODE=${MODE}"
    echo "  DATASET        : ${DATASET} (subject=${SUBJECT})"
    echo "  TRAIN_DATA     : ${TRAIN_DATA_PATH}"
    echo "  MODEL          : ${MODEL_PATH}"
    echo "  TOTAL_STEPS    : ${TOTAL_STEPS}"
    echo "  TRAIN_BS / N   : ${TRAIN_BATCH_SIZE} / ${ROLLOUT_N}    LR=${LR}"
    echo "  TEACHER        : ${TEACHER_REGULARIZATION} (rate=${TEACHER_UPDATE_RATE})"
    echo "  BASE_ESTIMATOR : ${IC_BASE_ESTIMATOR}    rlsd_eps_w=${IC_RLSD_EPS_W} (used iff base=rlsd)"
    echo "  INTERVENTION   : enable=${IC_ENABLE_INTERVENTION} divergence=${IC_DIVERGENCE_METRIC} k=${IC_K} λ=${IC_LAMBDA_DR}"
    echo "  G_T            : max_ratio=${IC_GT_MAX_RATIO} renorm=${IC_GT_RENORMALIZE_AFTER_CLIP}"
    echo "  LENGTH         : min_resp=${IC_MIN_RESPONSE_LENGTH} penalty=${IC_LENGTH_PENALTY_TYPE}"
    echo "  GIT            : ${GIT_BRANCH} @ ${GIT_COMMIT}"
    echo "  JOB_NAME       : ${JOB_NAME}"
    echo "========================================================================"

    # TOTAL_STEPS / VAL_N / VAL_BEFORE_TRAIN / N_GPUS_PER_NODE 已经通过 env 导出，
    # parametric.sh 会读取并注入 hydra (本地 notebook smoke/tstar 模式专用)
    bash "${PROJECT_ROOT}/nebula_scripts/intervention_credit/intervention_credit_sciknoweval_parametric.sh"
    unset JOB_NAME  # let next strategy pick a fresh name
}

# ─────────────────────────────────────────────────────────────────────
# 调度
# ─────────────────────────────────────────────────────────────────────
case "${MODE}" in
    smoke|full)
        run_single_strategy "${IC_DIVERGENCE_METRIC}"
        ;;
    tstar)
        echo "[tstar 模式] 串行运行 3 个 t* 策略消融..."
        echo "  预计总耗时: 3 × ~15min = ~45min"
        echo ""
        for strategy in argmax_excl_eos argmax_raw g_t_argmax; do
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "  开始策略: ${strategy}"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            run_single_strategy "${strategy}"
        done
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "[tstar 模式] 3 策略全部完成"
        echo ""
        echo "重点比对的 SwanLab 指标 (在 awesome_jimmy/TGDI-local 项目)："
        echo "  - intervention/divergence_in_last_quarter_rate"
        echo "      argmax_excl_eos 应 ≤ 0.20"
        echo "      argmax_raw      可能 > 0.30 (EOS bias)"
        echo "      g_t_argmax      中间值"
        echo "  - intervention/divergence_position_normalized_mean"
        echo "      argmax_excl_eos 应 < 0.5 (关注前半段)"
        echo "  - intervention/failed_sample_rate (~0.4-0.7 合理)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ;;
esac
