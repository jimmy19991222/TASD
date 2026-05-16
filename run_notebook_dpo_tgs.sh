#!/usr/bin/env bash
# =============================================================================
# Local DPO-TGS V2 (On-Policy DPO + Teacher-Guided Sampling) 训练脚本
# 4 卡 notebook 单机版 — 用于 Nebula 任务的本地 plumbing 验证 + 快速 ablation
#
# 设计依据: research/dpo_teacher_guided_sampling.md
# 理论锚点: 5 篇 (OAIF + OFS-DPO + Samplers + RPO + Meta Bridging) 见
#           papers/raw/opd_papers/OPD_Deep_Analysis.html
#
# Pipeline:
#   Phase 1: 标准 rollout n_init per prompt
#   Phase 2: 同 prompt correct sibling → SDPO teacher ctx
#   Phase 3: failed sample 上 n_attempts 次干预 (z_T != y_S[t*] reselect_t)
#   Phase 4: concat + lineage tag → DPO pair → linearized DPO advantage
#
# 三种模式:
#   smoke   (默认): 10 step bs=4 n_init=2 n_attempts=2 ~15 min → plumbing 验证
#   pair    :       30 step bs=8 chain_consecutive vs hybrid_init_chain 串行   ~45 min
#   full    :       100 step bs=32 n_init=2 n_attempts=2 ~4h → 与 Nebula 一致
#
# 用法:
#   ./run_notebook_dpo_tgs.sh                    # smoke (默认)
#   ./run_notebook_dpo_tgs.sh smoke
#   ./run_notebook_dpo_tgs.sh pair               # 2 策略 ablation
#   ./run_notebook_dpo_tgs.sh full               # 完整 100 step
#
#   # 自定义:
#   DPO_BETA=0.5 ./run_notebook_dpo_tgs.sh smoke
#   DPO_N_INIT=4 DPO_N_ATTEMPTS=4 ./run_notebook_dpo_tgs.sh smoke
#   DPO_PAIR_STRATEGY=hybrid_init_chain ./run_notebook_dpo_tgs.sh smoke
#   TOTAL_STEPS=20 TRAIN_BATCH_SIZE=8 ./run_notebook_dpo_tgs.sh smoke
#
# 后端调用 nebula_scripts/dpo_tgs/dpo_tgs_sciknoweval_parametric.sh
# =============================================================================
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

# ── 模式选择 ────────────────────────────────────────────────────────────
MODE="${1:-smoke}"

case "${MODE}" in
    smoke)
        # Plumbing 验证: ~15 min, 只验证 plumbing + 关注 dpo/* 指标出现
        DEFAULT_TOTAL_STEPS=10
        DEFAULT_TRAIN_BATCH_SIZE=4
        DEFAULT_MINI_BATCH_SIZE=4
        DEFAULT_N_INIT=2
        DEFAULT_N_ATTEMPTS=2
        DEFAULT_VAL_N=4
        DEFAULT_VAL_BEFORE_TRAIN=False
        DEFAULT_PAIR_STRATEGY="chain_consecutive"
        ;;
    pair)
        # pair_strategy ablation: chain_consecutive vs hybrid_init_chain, 各 30 step
        DEFAULT_TOTAL_STEPS=30
        DEFAULT_TRAIN_BATCH_SIZE=8
        DEFAULT_MINI_BATCH_SIZE=8
        DEFAULT_N_INIT=2
        DEFAULT_N_ATTEMPTS=2
        DEFAULT_VAL_N=8
        DEFAULT_VAL_BEFORE_TRAIN=False
        DEFAULT_PAIR_STRATEGY="chain_consecutive"   # overridden in loop
        ;;
    full)
        # 与 Nebula 等价配置
        DEFAULT_TOTAL_STEPS=100
        DEFAULT_TRAIN_BATCH_SIZE=32
        DEFAULT_MINI_BATCH_SIZE=32
        DEFAULT_N_INIT=2
        DEFAULT_N_ATTEMPTS=2
        DEFAULT_VAL_N=16
        DEFAULT_VAL_BEFORE_TRAIN=False
        DEFAULT_PAIR_STRATEGY="chain_consecutive"
        ;;
    *)
        echo "ERROR: unknown mode '${MODE}'. Use: smoke | pair | full"
        exit 1
        ;;
esac

# ── OSS 挂载根目录 (notebook 通常已挂为本地) ─────────────────────────
export OSS_ROOT="${OSS_ROOT:-/data/oss_bucket_0/ad/loujieming.ljm}"

# ── 数据集 / 模型 ───────────────────────────────────────────────────
export DATASET="${DATASET:-sciknoweval}"
export SUBJECT="${SUBJECT:-biology}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
export MODEL_PATH="${MODEL_PATH:-${OSS_ROOT}/base_models/${MODEL_NAME}}"

# ── 训练超参 (默认由模式决定, env 可覆盖) ──────────────────────────
export TOTAL_STEPS="${TOTAL_STEPS:-${DEFAULT_TOTAL_STEPS}}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-${DEFAULT_MINI_BATCH_SIZE}}"
export VAL_N="${VAL_N:-${DEFAULT_VAL_N}}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-${DEFAULT_VAL_BEFORE_TRAIN}}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export LR="${LR:-1e-5}"

# ── teacher 装配 (EMA 复用 self_distillation 基础设施) ────────────────
export TEACHER_REGULARIZATION="${TEACHER_REGULARIZATION:-ema}"
export TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"

# ── DPO-TGS 专属超参 ──────────────────────────────────────────────────
export DPO_N_INIT="${DPO_N_INIT:-${DEFAULT_N_INIT}}"
export DPO_N_ATTEMPTS="${DPO_N_ATTEMPTS:-${DEFAULT_N_ATTEMPTS}}"
export DPO_CORRECT_THRESHOLD="${DPO_CORRECT_THRESHOLD:-1.0}"
export DPO_SDPO_CTX_SOURCE="${DPO_SDPO_CTX_SOURCE:-sibling_correct}"   # sibling_correct | gt
export DPO_ALL_FAILED_STRATEGY="${DPO_ALL_FAILED_STRATEGY:-skip}"      # skip | gt_fallback
export DPO_MAX_RESELECT="${DPO_MAX_RESELECT:-3}"
export DPO_EXCLUDE_TAIL="${DPO_EXCLUDE_TAIL:-8}"
export DPO_BETA="${DPO_BETA:-0.1}"
export DPO_ALPHA="${DPO_ALPHA:-1.0}"
export DPO_PAIR_STRATEGY="${DPO_PAIR_STRATEGY:-${DEFAULT_PAIR_STRATEGY}}"  # chain_consecutive | hybrid_init_chain
export DPO_PAIR_MARGIN="${DPO_PAIR_MARGIN:-0.0}"
export DPO_MIN_RESP_LEN="${DPO_MIN_RESP_LEN:-50}"
export DPO_LEN_PENALTY="${DPO_LEN_PENALTY:-linear}"

# rollout.n MUST equal n_init (Phase 1 baseline count). parametric.sh handles default.
export ROLLOUT_N="${ROLLOUT_N:-${DPO_N_INIT}}"

# Legacy intervention_credit knobs (mostly unused by adaptive_rollout — kept for compat)
export IC_DIVERGENCE_METRIC="${IC_DIVERGENCE_METRIC:-argmax_excl_eos}"
export IC_TEACHER_DECODE_TEMPERATURE="${IC_TEACHER_DECODE_TEMPERATURE:-0.0}"

# ── 数据路径解析 (4 级 fallback,与 intervention_credit 一致) ─────────
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

# ── 兜底校验 ───────────────────────────────────────────────────────
if [ ! -d "${OSS_ROOT}" ]; then
    echo "⚠️  OSS_ROOT=${OSS_ROOT} 不存在 (未挂载)。save_path / model_path 依赖它。"
fi
if [ -z "${TRAIN_DATA_PATH:-}" ]; then
    echo "❌ 未找到数据 (试过以下候选):"
    for _cand in "${_candidates_train[@]}"; do echo "   - ${_cand}"; done
    echo "   请 export DATA_ROOT=/your/datasets 或 export TRAIN_DATA_PATH=/abs/path/train.parquet"
    exit 1
fi

# ── 单机 4 卡 ───────────────────────────────────────────────────────
export RAY_ADDRESS=""
export N_GPUS_PER_NODE=4

# ── Git 信息 ───────────────────────────────────────────────────────
export GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
export GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
export PROJECT_NAME="${PROJECT_NAME:-DPO-TGS-local}"

# ─────────────────────────────────────────────────────────────────────
# 单策略 launcher (smoke / full / pair-内部调用)
# ─────────────────────────────────────────────────────────────────────
run_single_strategy() {
    local pair_strategy="$1"
    export DPO_PAIR_STRATEGY="${pair_strategy}"
    local ps_short
    case "${pair_strategy}" in
        chain_consecutive) ps_short="chcons" ;;
        hybrid_init_chain) ps_short="hybrid" ;;
        *)                 ps_short="${pair_strategy}" ;;
    esac

    SUFFIX="${SUFFIX:-${MODE}}"
    export JOB_NAME="${JOB_NAME:-LOCAL-DPO-TGS-${ps_short}-ni${DPO_N_INIT}na${DPO_N_ATTEMPTS}-b${DPO_BETA}-a${DPO_ALPHA}-${SUBJECT}-step${TOTAL_STEPS}-${SUFFIX}}"

    echo "========================================================================"
    echo "Local DPO-TGS launch (notebook, ${N_GPUS_PER_NODE}×GPU) — MODE=${MODE}"
    echo "  DATASET           : ${DATASET} (subject=${SUBJECT})"
    echo "  TRAIN_DATA        : ${TRAIN_DATA_PATH}"
    echo "  MODEL             : ${MODEL_PATH}"
    echo "  TOTAL_STEPS       : ${TOTAL_STEPS}"
    echo "  TRAIN_BS          : ${TRAIN_BATCH_SIZE}    LR=${LR}"
    echo "  TEACHER (EMA)     : rate=${TEACHER_UPDATE_RATE}"
    echo "  ── DPO-TGS V2 ────────────────────────────────────"
    echo "  ROLLOUT_N=N_INIT  : ${DPO_N_INIT}"
    echo "  N_ATTEMPTS        : ${DPO_N_ATTEMPTS}"
    echo "  CORRECT_THRESHOLD : ${DPO_CORRECT_THRESHOLD}"
    echo "  SDPO_CTX_SOURCE   : ${DPO_SDPO_CTX_SOURCE}    all_failed=${DPO_ALL_FAILED_STRATEGY}"
    echo "  MAX_RESELECT      : ${DPO_MAX_RESELECT}    exclude_tail=${DPO_EXCLUDE_TAIL}"
    echo "  BETA              : ${DPO_BETA}    ALPHA=${DPO_ALPHA} (1.0=pure DPO, 0.0=pure GRPO)"
    echo "  PAIR_STRATEGY     : ${DPO_PAIR_STRATEGY}    margin=${DPO_PAIR_MARGIN}"
    echo "  LENGTH FLOOR      : min=${DPO_MIN_RESP_LEN} penalty=${DPO_LEN_PENALTY}"
    echo "  ──────────────────────────────────────────────────"
    echo "  GIT               : ${GIT_BRANCH} @ ${GIT_COMMIT}"
    echo "  JOB_NAME          : ${JOB_NAME}"
    echo "========================================================================"

    bash "${PROJECT_ROOT}/nebula_scripts/dpo_tgs/dpo_tgs_sciknoweval_parametric.sh"
    unset JOB_NAME  # let next strategy pick a fresh name
}

# ─────────────────────────────────────────────────────────────────────
# 调度
# ─────────────────────────────────────────────────────────────────────
case "${MODE}" in
    smoke|full)
        run_single_strategy "${DPO_PAIR_STRATEGY}"
        ;;
    pair)
        # ablation: chain_consecutive vs hybrid_init_chain
        for ps in chain_consecutive hybrid_init_chain; do
            echo ""
            echo "════════ Pair-strategy ablation: ${ps} ════════"
            run_single_strategy "${ps}"
        done
        ;;
esac

echo ""
echo "========================================================================"
echo "✅ Local DPO-TGS run complete (MODE=${MODE})"
echo ""
echo "📊 SwanLab 关注指标:"
echo "  P0 (DPO 标准 + 早期预警):"
echo "    val-core/dpo_implicit_reward_accuracy  → 目标 > 0.6"
echo "    dpo/sigma_neg_margin_mean              → 梯度幅度 (萎缩=训练 stall)"
echo "    dpo/kl_to_ref_chosen_mean              → 涨太快=length collapse"
echo "    dpo/length_ratio_chosen_over_rejected  → ≈1 健康 >1.5 警告"
echo "  P1 (DPO-TGS V2 独有):"
echo "    dpo/prompts_with_no_correct_pct        → 高=SDPO ctx 缺失"
echo "    dpo/chain_attempt_success_rate@k       → n_attempts 边际收益"
echo "    dpo/grpo_fallback_rate                 → DPO 信号占比 = 1 - this"
echo "  主指标:"
echo "    val-core/sciknoweval/${SUBJECT}/acc/mean@${VAL_N}"
echo "========================================================================"
