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
export DATASET="${DATASET:-sciknoweval}"                  # 数据集名（${OSS_ROOT}/datasets/${DATASET}/{train,test}.parquet）
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
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

# ── 实验命名（PROJECT_NAME 控制 SwanLab project）────────────────────────
SUFFIX="${1:-local_smoke}"
export JOB_NAME="${JOB_NAME:-LOCAL-TASD-adv${ADV_MODE}-vce${USE_VCE}-emar${TEACHER_UPDATE_RATE}-${SUFFIX}}"
export PROJECT_NAME="${PROJECT_NAME:-TASD_local}"
export GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
export GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# ── 关掉 OSS 依赖时的兜底（如果 OSS 没挂）────────────────────────────────
if [ ! -d "${OSS_ROOT}" ]; then
    echo "⚠️  OSS_ROOT=${OSS_ROOT} 不存在，请先挂载 OSS 或改本地路径"
    echo "    需要存在: ${OSS_ROOT}/datasets/${DATASET}/{train,test}.parquet"
    exit 1
fi

# ── 单机 4 卡：禁用 Ray 多机调度，使用本地 Ray ─────────────────────────
export RAY_ADDRESS=""
export N_GPUS_PER_NODE=4

echo "========================================================================"
echo "Local TASD launch:"
echo "  GPUs           : 4 × H20-3e"
echo "  DATASET        : ${DATASET}"
echo "  MODEL          : ${MODEL_PATH}"
echo "  ADV_MODE       : ${ADV_MODE} (use_vce=${USE_VCE}, use_log_pi_s=${USE_LOG_PI_S})"
echo "  EMA rate       : ${TEACHER_UPDATE_RATE}  (修复后真生效)"
echo "  GIT            : ${GIT_BRANCH} @ ${GIT_COMMIT}"
echo "  JOB_NAME       : ${JOB_NAME}"
echo "========================================================================"

bash "${PROJECT_ROOT}/nebula_scripts/tasd_simple/tasd_simple_parametric.sh"
