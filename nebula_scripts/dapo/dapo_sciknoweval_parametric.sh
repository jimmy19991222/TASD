#!/usr/bin/env bash
# =============================================================================
# DAPO 参数化训练脚本（供 Nebula sweep 调用）
#
# 核心特性:
#   1. Clip-Higher: clip_ratio_high >> clip_ratio_low
#   2. Dynamic Sampling: filter_groups 过滤全对/全错 group
#   3. Token-level Loss: loss_agg_mode=token-mean
#   4. Entropy coeff: 保持探索
# =============================================================================
set +xo pipefail

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"

# ── 必需参数 ─────────────────────────────────────────────────────────────
: "${DATASET:?DATASET is not set}"
: "${MODEL_PATH:?MODEL_PATH is not set}"

# ── 可选参数（有默认值）─────────────────────────────────────────────────
LR="${LR:-1e-5}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.001}"       # DAPO: 保持探索
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-10000}"   # DAPO Clip-Higher: 不限上界
SEED="${SEED:-42}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.05}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-8}"
# ── DAPO Dynamic Sampling ────────────────────────────────────────────────
FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-false}"  # 是否启用 filter_groups
FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-acc}"    # acc / seq_reward / seq_final_reward
FILTER_GROUPS_MAX_GEN="${FILTER_GROUPS_MAX_GEN:-0}"    # 最大重采样次数，0=不限制

# ── 路径 ────────────────────────────────────────────────────────────────
train_data_path="${OSS_ROOT}/datasets/${DATASET}/train.parquet"
val_data_path="${OSS_ROOT}/datasets/${DATASET}/test.parquet"
model_path="${MODEL_PATH}"
save_path="${OSS_ROOT}/models/${JOB_NAME:-dapo}"

# ── 环境 ────────────────────────────────────────────────────────────────
# 优先使用 git 仓库根目录作为 PYTHONPATH，确保加载最新代码（而非 train_package 中的旧代码）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if git -C "${SCRIPT_DIR}" rev-parse --show-toplevel &>/dev/null; then
    GIT_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
    export PYTHONPATH="${GIT_ROOT}:${PYTHONPATH:-}"
    echo "[PYTHONPATH] Using git root: ${GIT_ROOT}"
elif [ -d "${SCRIPT_DIR}/../../verl" ]; then
    export PYTHONPATH="${SCRIPT_DIR}/../..:${PYTHONPATH:-}"
    echo "[PYTHONPATH] Using script relative path: ${SCRIPT_DIR}/../.."
else
    export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
    echo "[PYTHONPATH] Using pwd: $(pwd)"
fi
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARN
export WANDB_MODE=offline
export WANDB_ENTITY=oh-my-team
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal}"
export SWANLAB_LOG_DIR="${OSS_ROOT}/logs/swanlab_logs"
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0
# git 分支信息（由 submit 脚本传入，若未设置则尝试本地获取）
export GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')}"
export GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}"

pip install -e . --no-deps --no-build-isolation --quiet 2>/dev/null || true

mkdir -p "${SWANLAB_LOG_DIR}" 2>/dev/null || true

# 清理残留的 Ray session
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true
rm -rf /tmp/ray_session_* 2>/dev/null || true
rm -rf ~/.ray 2>/dev/null || true
sleep 3

echo "============================================"
echo "DAPO 实验配置："
echo "  DATASET: ${DATASET}"
echo "  LR: ${LR}, ENTROPY_COEFF: ${ENTROPY_COEFF}"
echo "  CLIP_RATIO_HIGH: ${CLIP_RATIO_HIGH}"
echo "  FILTER_GROUPS: enable=${FILTER_GROUPS_ENABLE}, metric=${FILTER_GROUPS_METRIC}, max_gen=${FILTER_GROUPS_MAX_GEN}"
echo "  REPETITION_PENALTY: ${REPETITION_PENALTY}"
echo "  SEED: ${SEED}"
echo "============================================"

python -m verl.trainer.main_ppo \
    --config-name dapo \
    seed=${SEED} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.train_files="${train_data_path}" \
    data.val_files="${val_data_path}" \
    custom_reward_function.path="$(pwd)/verl/utils/reward_score/feedback/__init__.py" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.repetition_penalty=${REPETITION_PENALTY} \
    algorithm.filter_groups.enable=${FILTER_GROUPS_ENABLE} \
    algorithm.filter_groups.metric=${FILTER_GROUPS_METRIC} \
    algorithm.filter_groups.max_num_gen_batches=${FILTER_GROUPS_MAX_GEN} \
    trainer.total_epochs=30 \
    trainer.total_training_steps=250 \
    trainer.save_freq=-1 \
    trainer.save_best_metric="val-core/sciknoweval/acc/mean@16" \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=False \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME:-DAPO}" \
    trainer.experiment_name="${JOB_NAME:-dapo}" \
    trainer.group_name="DAPO" \
    "trainer.logger=[console,swanlab]"
