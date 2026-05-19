#!/usr/bin/env bash
# =============================================================================
# Geodesic-VCE 融合实验参数化脚本（支持 SDPO 和 V_CE 两种模式）
# 所有超参通过 nebulactl --env 注入
# =============================================================================
set +xo pipefail

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"

# ── 从环境变量读取超参 ────────────────────────────────────────────────
check_env() { val=$(eval echo "\$$1"); [ -n "$val" ] || { echo "ERROR: $1 is not set. Aborting."; exit 1; }; }
check_env DATASET
check_env LR
check_env TRAIN_BATCH_SIZE
check_env ROLLOUT_N
check_env MODEL_NAME
check_env LOSS_MODE  # sdpo | self_teacher

SEED="${SEED:-42}"

# SDPO 参数（仅 loss_mode=sdpo 时生效）
ALPHA="${ALPHA:-0.5}"
DISTILL_TOPK="${DISTILL_TOPK:-100}"
DONT_REPROMPT_ON_SELF_SUCCESS="${DONT_REPROMPT_ON_SELF_SUCCESS:-True}"

# V_CE 参数（仅 loss_mode=self_teacher 时生效）
USE_VCE="${USE_VCE:-False}"
USE_LOG_PI_S="${USE_LOG_PI_S:-False}"
CLIP_VALUE="${CLIP_VALUE:-3.0}"
ADV_STD_FLOOR="${ADV_STD_FLOOR:-0.0}"

# Geodesic 参数（两种模式都可用）
USE_GEODESIC="${USE_GEODESIC:-False}"
GEODESIC_TRUST_REGION="${GEODESIC_TRUST_REGION:-5.0}"
GEODESIC_BETA_SCALE="${GEODESIC_BETA_SCALE:-0.5}"

# 粒度控制参数
FULL_LOGIT_DISTILLATION="${FULL_LOGIT_DISTILLATION:-True}"  # True=vocab粒度, False=token粒度

# 训练步数（支持 smoke test）
MAX_STEPS="${MAX_STEPS:-250}"  # 默认250步，smoke test可设为10

# 数据集路径
train_data_path="${OSS_ROOT}/datasets/${DATASET}/train.parquet"
val_data_path="${OSS_ROOT}/datasets/${DATASET}/test.parquet"
model_path="${OSS_ROOT}/base_models/${MODEL_NAME}"
save_path="${OSS_ROOT}/models/${JOB_NAME:-geodesic_vce_ablation}"

# ── 环境 ──────────────────────────────────────────────────────────────
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARN
export WANDB_MODE=offline
export WANDB_ENTITY=oh-my-team
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal}"
export SWANLAB_LOG_DIR="${OSS_ROOT}/logs/swanlab_logs"
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0

pip install -e . --no-deps --no-build-isolation --quiet 2>/dev/null || true

mkdir -p "${SWANLAB_LOG_DIR}" 2>/dev/null || true

# ── 构建 Hydra 命令 ─────────────────────────────────────────────────
HYDRA_ARGS=(
    --config-name sdpo
    seed=${SEED}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.train_files="${train_data_path}"
    data.val_files="${val_data_path}"
    custom_reward_function.path="$(pwd)/verl/utils/reward_score/feedback/__init__.py"
    actor_rollout_ref.model.path="${model_path}"
    actor_rollout_ref.actor.optim.lr=${LR}
    actor_rollout_ref.actor.optim.lr_warmup_steps=10
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.val_kwargs.n=16
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85
    algorithm.rollout_correction.rollout_is=token
    trainer.total_epochs=30
    trainer.total_training_steps=${MAX_STEPS}
    trainer.save_freq=-1
    trainer.save_best_metric="val-core/sciknoweval/acc/mean@16"
    trainer.n_gpus_per_node=4
    trainer.val_before_train=False
    trainer.default_local_dir="${save_path}"
    trainer.project_name="${PROJECT_NAME:-Geodesic-VCE-Ablation}"
    trainer.experiment_name="${JOB_NAME:-geodesic_vce_ablation}"
    trainer.group_name="Geodesic-VCE"
    "trainer.logger=[console,swanlab]"
)

# ── 根据 loss_mode 添加特定参数 ─────────────────────────────────────
if [ "$LOSS_MODE" = "sdpo" ]; then
    # SDPO 模式：使用 self_distillation loss
    HYDRA_ARGS+=(
        actor_rollout_ref.actor.policy_loss.loss_mode=sdpo
        actor_rollout_ref.actor.self_distillation.full_logit_distillation=${FULL_LOGIT_DISTILLATION}
        actor_rollout_ref.actor.self_distillation.distillation_topk=${DISTILL_TOPK}
        actor_rollout_ref.actor.self_distillation.alpha=${ALPHA}
        actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONT_REPROMPT_ON_SELF_SUCCESS}
        actor_rollout_ref.actor.self_distillation.include_environment_feedback=False
        actor_rollout_ref.actor.self_distillation.use_geodesic=${USE_GEODESIC}
        actor_rollout_ref.actor.self_distillation.geodesic_trust_region=${GEODESIC_TRUST_REGION}
        actor_rollout_ref.actor.self_distillation.geodesic_beta_scale=${GEODESIC_BETA_SCALE}
    )
elif [ "$LOSS_MODE" = "self_teacher" ]; then
    # Self-Teacher 模式：使用 advantage + policy gradient
    HYDRA_ARGS+=(
        actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
        algorithm.adv_estimator=self_teacher
        algorithm.use_vce=${USE_VCE}
        algorithm.use_log_pi_s=${USE_LOG_PI_S}
        algorithm.clip_value=${CLIP_VALUE}
        algorithm.adv_std_floor=${ADV_STD_FLOOR}
    )
else
    echo "ERROR: LOSS_MODE must be 'sdpo' or 'self_teacher'"
    exit 1
fi

# ── 执行训练 ─────────────────────────────────────────────────────────
python -m verl.trainer.main_ppo "${HYDRA_ARGS[@]}"
