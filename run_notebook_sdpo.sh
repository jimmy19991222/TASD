#!/usr/bin/env bash
# =============================================================================
# Local SDPO Baseline 训练脚本（4 卡 notebook 单机版）
#
# 用法:
#   ./run_notebook_sdpo.sh                    # 默认配置
#   ./run_notebook_sdpo.sh my_exp_suffix      # 指定实验名后缀
#
# 后端调用 nebula_scripts/sdpo/sdpo_sciknoweval_parametric.sh
# =============================================================================
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

# ── OSS 挂载根目录（notebook 通常已挂为本地）─────────────────────────
export OSS_ROOT="${OSS_ROOT:-/data/oss_bucket_0/ad/loujieming.ljm}"

# ── 数据集 / 模型 ───────────────────────────────────────────────────
export DATASET="${DATASET:-sciknoweval}"
export SUBJECT="${SUBJECT:-biology}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
export MODEL_PATH="${MODEL_PATH:-${OSS_ROOT}/base_models/${MODEL_NAME}}"

# ── 4 卡 H20-3e (144GB) 推荐档位 ─────────────────────────────────────
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-32}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export LR="${LR:-1e-5}"

# ── SDPO 专属 ────────────────────────────────────────────────────────
export ALPHA="${ALPHA:-0.5}"
export DONT_REPROMPT_ON_SELF_SUCCESS="${DONT_REPROMPT_ON_SELF_SUCCESS:-True}"

# ── 数据路径解析 ────────────────────────────────────────────────────
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

# ── 实验命名 ────────────────────────────────────────────────────────
SUFFIX="${1:-local_smoke}"
export JOB_NAME="${JOB_NAME:-LOCAL-SDPO-${DATASET}-bs${TRAIN_BATCH_SIZE}-n${ROLLOUT_N}-lr${LR}-a${ALPHA}-${SUFFIX}}"
export PROJECT_NAME="${PROJECT_NAME:-Baselines_local}"
export GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
export GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

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

echo "========================================================================"
echo "Local SDPO launch (notebook, 4×H20-3e):"
echo "  DATASET        : ${DATASET} (subject=${SUBJECT})"
echo "  TRAIN_DATA     : ${TRAIN_DATA_PATH}"
echo "  MODEL          : ${MODEL_PATH}"
echo "  TRAIN_BS / N   : ${TRAIN_BATCH_SIZE} / ${ROLLOUT_N}    LR=${LR}"
echo "  ALPHA          : ${ALPHA}"
echo "  DONT_REPROMPT  : ${DONT_REPROMPT_ON_SELF_SUCCESS}"
echo "  GIT            : ${GIT_BRANCH} @ ${GIT_COMMIT}"
echo "  JOB_NAME       : ${JOB_NAME}"
echo "========================================================================"

bash "${PROJECT_ROOT}/nebula_scripts/sdpo/sdpo_sciknoweval_parametric.sh"
