#!/usr/bin/env bash
# =============================================================================
# Local TASD 训练脚本（4 卡 notebook 单机版）
#
# 用法:
#   ./run_local_tasd.sh                    # 默认配置 smoke run
#   ./run_local_tasd.sh my_exp_suffix      # 指定实验名后缀
#
# 复用 nebula_scripts/tasd_simple/tasd_simple_parametric.sh，
# 只在本地补齐它要的环境变量；改超参直接改下面这一段或 export 后再调用。
# =============================================================================
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

# ── 必须：路径 ────────────────────────────────────────────────────────────
# OSS bucket 在 notebook 上一般已挂为本地目录；若没挂或想用别处，改这里
export OSS_ROOT="${OSS_ROOT:-/data/oss_bucket_0/ad/loujieming.ljm}"

# ── 必须参数（parametric 脚本要求显式传入）──────────────────────────────
export DATASET="${DATASET:-sciknoweval}"                  # 数据集名
export SUBJECT="${SUBJECT:-biology}"                      # sciknoweval 下的 subject（biology/chemistry/material/physics）
export MODEL_PATH="${MODEL_PATH:-${OSS_ROOT}/models/Qwen3-8B}"
export REWARD_TYPE="${REWARD_TYPE:-teacher_log_prob}"     # teacher_prob | teacher_log_prob
export ENTROPY_GATE="${ENTROPY_GATE:-none}"               # none | hard | hard_keep_reward | soft
export ENTROPY_GATE_RATIO="${ENTROPY_GATE_RATIO:-1.0}"
export CLIP_ADV_VALUE="${CLIP_ADV_VALUE:-3.0}"

# ── 4 卡 H20-3e (144GB) 推荐档位 ─────────────────────────────────────────
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-32}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export LR="${LR:-1e-5}"

# ── Self-Teacher Advantage 路径（默认走最新 V_CE）─────────────────────────
export ADV_MODE="${ADV_MODE:-self_teacher}"               # grpo | self_teacher
export USE_VCE="${USE_VCE:-true}"
export USE_LOG_PI_S="${USE_LOG_PI_S:-false}"
export CLIP_VALUE="${CLIP_VALUE:-3.0}"

# ── teacher EMA（修复后会真正生效；rate=0.1 即原 SDPO 配置）─────────────
export TEACHER_REG="${TEACHER_REG:-ema}"
export TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.1}"

# ── 校验脚本会读的可选变量，全部显式 export 兜底 ──────────────────────
export ADV_ENTROPY_WEIGHT="${ADV_ENTROPY_WEIGHT:-none}"
export NORM_ADV_BY_STD="${NORM_ADV_BY_STD:-false}"
export DISTILL_TOPK="${DISTILL_TOPK:-100}"
export CLIP_ADV="${CLIP_ADV:-true}"
export GROUP_MEAN_MODE="${GROUP_MEAN_MODE:-token}"
export INCLUDE_SUCCESSFUL_ROLLOUTS="${INCLUDE_SUCCESSFUL_ROLLOUTS:-True}"

# ── 数据路径解析（优先级：TRAIN_DATA_PATH > DATA_ROOT 拼接 > OSS_ROOT 拼接）─────────
export DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets}"
# 优先试 subject 子目录（如 sciknoweval/biology/），退化到 flat （如 sciknoweval/）
_subject_train="${DATA_ROOT}/${DATASET}/${SUBJECT}/train.parquet"
_flat_train="${DATA_ROOT}/${DATASET}/train.parquet"
if [ -z "${TRAIN_DATA_PATH:-}" ]; then
    if [ -f "${_subject_train}" ]; then
        export TRAIN_DATA_PATH="${_subject_train}"
        export VAL_DATA_PATH="${VAL_DATA_PATH:-${DATA_ROOT}/${DATASET}/${SUBJECT}/test.parquet}"
    elif [ -f "${_flat_train}" ]; then
        export TRAIN_DATA_PATH="${_flat_train}"
        export VAL_DATA_PATH="${VAL_DATA_PATH:-${DATA_ROOT}/${DATASET}/test.parquet}"
    fi
fi

# ── 实验命名（PROJECT_NAME 控制 SwanLab project）────────────────────────
SUFFIX="${1:-local_smoke}"
export JOB_NAME="${JOB_NAME:-LOCAL-TASD-adv${ADV_MODE}-vce${USE_VCE}-emar${TEACHER_UPDATE_RATE}-${SUFFIX}}"
export PROJECT_NAME="${PROJECT_NAME:-TASD_local}"
export GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
export GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# ── 关掉 OSS 依赖时的兜底（如果 OSS 没挂）────────────────────────────────
if [ ! -d "${OSS_ROOT}" ]; then
    echo "⚠️  OSS_ROOT=${OSS_ROOT} 不存在（未挂载）。save_path 依赖它，但本地 smoke 可忽略。"
fi
if [ -z "${TRAIN_DATA_PATH:-}" ]; then
    echo "❌ 未找到本地数据："
    echo "   - ${_subject_train}"
    echo "   - ${_flat_train}"
    echo "   请 export DATA_ROOT=/your/datasets 或直接 export TRAIN_DATA_PATH=/abs/path/train.parquet"
    exit 1
fi

# ── 单机 4 卡：禁用 Ray 多机调度，使用本地 Ray ─────────────────────────
export RAY_ADDRESS=""
export N_GPUS_PER_NODE=4

echo "========================================================================"
echo "Local TASD launch:"
echo "  GPUs           : 4 × H20-3e (144GB each)"
echo "  DATASET        : ${DATASET} (subject=${SUBJECT})"
echo "  TRAIN_DATA     : ${TRAIN_DATA_PATH:-${OSS_ROOT}/datasets/${DATASET}/train.parquet}"
echo "  MODEL          : ${MODEL_PATH}"
echo "  ADV_MODE       : ${ADV_MODE} (use_vce=${USE_VCE}, use_log_pi_s=${USE_LOG_PI_S})"
echo "  EMA rate       : ${TEACHER_UPDATE_RATE}  (修复后真生效)"
echo "  GIT            : ${GIT_BRANCH} @ ${GIT_COMMIT}"
echo "  JOB_NAME       : ${JOB_NAME}"
echo "========================================================================"

bash "${PROJECT_ROOT}/nebula_scripts/tasd_simple/tasd_simple_parametric.sh"
