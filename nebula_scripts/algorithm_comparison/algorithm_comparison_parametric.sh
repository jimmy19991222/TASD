#!/usr/bin/env bash
# =============================================================================
# 算法对比实验 - 统一参数化训练脚本
# 支持：GRPO / SDPO / FIPO / Self-Teacher Advantage
# =============================================================================
set +xo pipefail

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"

# ── 必需参数 ─────────────────────────────────────────────────────────────
: "${DATASET:?DATASET is not set}"
: "${ALGORITHM:?ALGORITHM is not set}"  # grpo | sdpo | fipo | self_teacher
: "${MODEL_PATH:?MODEL_PATH is not set}"

# ── 可选参数（有默认值）─────────────────────────────────────────────────
LR="${LR:-1e-5}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.001}"
SEED="${SEED:-42}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-32}"
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.05}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"

# ── 算法特定参数 ─────────────────────────────────────────────────────
# SDPO 参数
SDPO_ALPHA="${SDPO_ALPHA:-1.0}"
SDPO_DONT_REPROMPT="${SDPO_DONT_REPROMPT:-True}"

# Self-Teacher Advantage 参数
ADV_MODE="${ADV_MODE:-grpo}"  # grpo | self_teacher
BETA="${BETA:-0.7}"
EMA_ALPHA="${EMA_ALPHA:-0.9}"
CLIP_VALUE="${CLIP_VALUE:-5.0}"
REWARD_TYPE="${REWARD_TYPE:-outcome}"
ENTROPY_GATE="${ENTROPY_GATE:-none}"
ENTROPY_GATE_RATIO="${ENTROPY_GATE_RATIO:-1.0}"
DISTILL_TOPK="${DISTILL_TOPK:-256}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"

# ── 路径 ────────────────────────────────────────────────────────────────
train_data_path="${OSS_ROOT}/datasets/${DATASET}/train.parquet"
val_data_path="${OSS_ROOT}/datasets/${DATASET}/test.parquet"
model_path="${MODEL_PATH}"
save_path="${OSS_ROOT}/models/${JOB_NAME:-algorithm_comparison}"

# ── 动态 save_best_metric ──────────────────────────────────────────────
if [[ "${DATASET}" == tooluse* ]]; then
    SAVE_BEST_METRIC="val-core/tooluse/acc/mean@16"
else
    SAVE_BEST_METRIC="val-core/sciknoweval/acc/mean@16"
fi

# ── 环境 ────────────────────────────────────────────────────────────────
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
export VLLM_LOGGING_LEVEL=WARN
export WANDB_MODE=offline
export WANDB_ENTITY=oh-my-team
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal}"
export SWANLAB_LOG_DIR="${OSS_ROOT}/logs/swanlab_logs"
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0
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

# ── 配置预检 ────────────────────────────────────────────────────────────
VALID_ALG="grpo sdpo fipo self_teacher"
if ! echo "$VALID_ALG" | grep -qw "$ALGORITHM"; then
    echo "❌ 错误: ALGORITHM='$ALGORITHM' 无效"
    echo "   有效值: $VALID_ALG"
    exit 1
fi

echo "============================================"
echo "算法对比实验配置："
echo "  ALGORITHM: ${ALGORITHM}"
echo "  DATASET: ${DATASET}"
echo "  MODEL: ${model_path}"
echo "  LR: ${LR}, ENTROPY_COEFF: ${ENTROPY_COEFF}"
echo "  ROLLOUT_N: ${ROLLOUT_N}, TEMPERATURE: ${ROLLOUT_TEMPERATURE}"
echo "  CLIP_RATIO_HIGH: ${CLIP_RATIO_HIGH}"
echo "============================================"

# ── 根据算法选择配置文件和参数 ──────────────────────────────────────────
case "$ALGORITHM" in
    grpo)
        CONFIG_NAME="baseline_grpo"
        EXTRA_ARGS=""
        echo "  → GRPO Baseline"
        ;;
    
    sdpo)
        CONFIG_NAME="sdpo"
        EXTRA_ARGS="
    actor_rollout_ref.actor.self_distillation.alpha=${SDPO_ALPHA} \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${SDPO_DONT_REPROMPT} \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"
        echo "  → SDPO (alpha=${SDPO_ALPHA})"
        ;;
    
    fipo)
        # FIPO 使用 GRPO 配置 + fipo advantage estimator
        CONFIG_NAME="baseline_grpo"
        EXTRA_ARGS="
    algorithm.adv_estimator=fipo"
        echo "  → FIPO"
        ;;
    
    self_teacher)
        # Self-Teacher Advantage 使用 tasd_simple 配置
        CONFIG_NAME="tasd_simple"
        EXTRA_ARGS="
    algorithm.tasd.adv_mode=${ADV_MODE} \
    algorithm.tasd.beta=${BETA} \
    algorithm.tasd.ema_alpha=${EMA_ALPHA} \
    algorithm.tasd.clip_value=${CLIP_VALUE} \
    algorithm.tasd.reward_type=${REWARD_TYPE} \
    algorithm.tasd.entropy_gate=${ENTROPY_GATE} \
    algorithm.tasd.entropy_gate_ratio=${ENTROPY_GATE_RATIO} \
    algorithm.tasd.distill_topk=${DISTILL_TOPK} \
    algorithm.tasd.distill_temperature=${DISTILL_TEMPERATURE} \
    algorithm.tasd.norm_adv_by_std=false \
    algorithm.tasd.adv_entropy_weight=none \
    algorithm.tasd.group_mean_mode=seq \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.05"
        echo "  → Self-Teacher Advantage"
        echo "    adv_mode=${ADV_MODE}, beta=${BETA}, ema_alpha=${EMA_ALPHA}"
        ;;
esac

# ── 启动训练 ────────────────────────────────────────────────────────────
python -m verl.trainer.main_ppo \
    --config-name ${CONFIG_NAME} \
    seed=${SEED} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.gen_batch_size=${GEN_BATCH_SIZE} \
    data.train_files="${train_data_path}" \
    data.val_files="${val_data_path}" \
    custom_reward_function.path="$(pwd)/verl/utils/reward_score/feedback/__init__.py" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.data_loader_seed=${SEED} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.repetition_penalty=${REPETITION_PENALTY} \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
    actor_rollout_ref.rollout.seed=${SEED} \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    ${EXTRA_ARGS} \
    trainer.total_epochs=30 \
    trainer.total_training_steps=250 \
    trainer.save_freq=-1 \
    trainer.save_best_metric="${SAVE_BEST_METRIC}" \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=False \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME:-Algorithm-Comparison-v1}" \
    trainer.experiment_name="${JOB_NAME:-algorithm_comparison}" \
    trainer.group_name="Algorithm-Comparison-v1" \
    "trainer.logger=[console,swanlab]"
