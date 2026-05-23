#!/usr/bin/env bash
# =============================================================================
# SDPO Baseline 参数化训练脚本（供 Nebula sweep 调用）
# 所有超参通过 nebulactl --env 注入
# =============================================================================
set +xo pipefail

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"

# ── 从环境变量读取超参 ────────────────────────────────────────────────
check_env() { val=$(eval echo "\$$1"); [ -n "$val" ] || { echo "ERROR: $1 is not set. Aborting."; exit 1; }; }
check_env DATASET
check_env LR
check_env ALPHA
check_env DONT_REPROMPT_ON_SELF_SUCCESS
check_env TRAIN_BATCH_SIZE
check_env ROLLOUT_N
check_env MODEL_NAME

SEED="${SEED:-42}"

# Geodesic SDPO 参数（可选）
USE_GEODESIC="${USE_GEODESIC:-False}"
GEODESIC_TRUST_REGION="${GEODESIC_TRUST_REGION:-5.0}"
GEODESIC_BETA_SCALE="${GEODESIC_BETA_SCALE:-0.5}"

# Loss-variant 参数（可选,默认保持历史行为: full-logit + 不打 SNR）
FULL_LOGIT_DISTILLATION="${FULL_LOGIT_DISTILLATION:-True}"
LOG_DELTA_W_STATS="${LOG_DELTA_W_STATS:-False}"
# token-level (full_logit=False) 时禁用 topk(走 token-PG 分支,不需要)
if [ "${FULL_LOGIT_DISTILLATION}" = "False" ]; then
    DISTILLATION_TOPK="${DISTILLATION_TOPK:-null}"
else
    DISTILLATION_TOPK="${DISTILLATION_TOPK:-100}"
fi

# Verdict-conditioned self-distillation (reward_bayes_v2)
LOSS_METHOD="${LOSS_METHOD:-sdpo}"                    # sdpo / vc_opsd_sign / opd_bayes / vec
VERDICT_PRIOR_MODE="${VERDICT_PRIOR_MODE:-uniform}"   # uniform / logit / empirical
VERDICT_PRIOR_LOGIT="${VERDICT_PRIOR_LOGIT:-0.0}"
CALIBRATION_WEIGHT="${CALIBRATION_WEIGHT:-0.0}"
KL_DIRECTION="${KL_DIRECTION:-reverse}"               # reverse / forward (opd_bayes)
# opd_bayes requires full-logit; if user picks it, snap on full_logit & a reasonable topk
if [ "${LOSS_METHOD}" = "opd_bayes" ]; then
    FULL_LOGIT_DISTILLATION="True"
    DISTILLATION_TOPK="${DISTILLATION_TOPK:-100}"
fi

# Self-verified marker SDPO
TEACHER_CONTEXT_MODE="${TEACHER_CONTEXT_MODE:-ref}"               # ref / marker / ref_and_marker / gt_marker
SELF_VERIFIED_MARKER="${SELF_VERIFIED_MARKER:-This answer is verified correct.}"
GT_MARKER_TEMPLATE="${GT_MARKER_TEMPLATE:-This answer is verified correct, correct answer is {ground_truth}.}"
OVERCONFIDENCE_DAMPING="${OVERCONFIDENCE_DAMPING:-0.0}"           # 0.0 disables; sensible 0.5~2.0
# Marker modes require full-logit (for ΔH if damped; for clean signal anyway)
if [ "${TEACHER_CONTEXT_MODE}" != "ref" ]; then
    FULL_LOGIT_DISTILLATION="True"
    DISTILLATION_TOPK="${DISTILLATION_TOPK:-100}"
fi

# 数据集路径
train_data_path="${OSS_ROOT}/datasets/${DATASET}/train.parquet"
val_data_path="${OSS_ROOT}/datasets/${DATASET}/test.parquet"
model_path="${OSS_ROOT}/base_models/${MODEL_NAME}"
save_path="${OSS_ROOT}/models/${JOB_NAME:-sdpo_sweep}"

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

python -m verl.trainer.main_ppo \
    --config-name sdpo \
    seed=${SEED} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.train_files="${train_data_path}" \
    data.val_files="${val_data_path}" \
    custom_reward_function.path="$(pwd)/verl/utils/reward_score/feedback/__init__.py" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=${FULL_LOGIT_DISTILLATION} \
    actor_rollout_ref.actor.self_distillation.distillation_topk=${DISTILLATION_TOPK} \
    actor_rollout_ref.actor.self_distillation.alpha=${ALPHA} \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONT_REPROMPT_ON_SELF_SUCCESS} \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.actor.self_distillation.use_geodesic=${USE_GEODESIC} \
    actor_rollout_ref.actor.self_distillation.geodesic_trust_region=${GEODESIC_TRUST_REGION} \
    actor_rollout_ref.actor.self_distillation.geodesic_beta_scale=${GEODESIC_BETA_SCALE} \
    actor_rollout_ref.actor.self_distillation.log_delta_w_stats=${LOG_DELTA_W_STATS} \
    actor_rollout_ref.actor.self_distillation.loss_method=${LOSS_METHOD} \
    actor_rollout_ref.actor.self_distillation.verdict_prior_mode=${VERDICT_PRIOR_MODE} \
    actor_rollout_ref.actor.self_distillation.verdict_prior_logit=${VERDICT_PRIOR_LOGIT} \
    actor_rollout_ref.actor.self_distillation.calibration_weight=${CALIBRATION_WEIGHT} \
    actor_rollout_ref.actor.self_distillation.kl_direction=${KL_DIRECTION} \
    actor_rollout_ref.actor.self_distillation.teacher_context_mode=${TEACHER_CONTEXT_MODE} \
    actor_rollout_ref.actor.self_distillation.self_verified_marker="${SELF_VERIFIED_MARKER}" \
    actor_rollout_ref.actor.self_distillation.gt_marker_template="${GT_MARKER_TEMPLATE}" \
    actor_rollout_ref.actor.self_distillation.overconfidence_damping=${OVERCONFIDENCE_DAMPING} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    algorithm.rollout_correction.rollout_is=token \
    trainer.total_epochs=30 \
    trainer.total_training_steps=250 \
    trainer.save_freq=-1 \
    trainer.save_best_metric="val-core/sciknoweval/acc/mean@16" \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=False \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME:-Baselines}" \
    trainer.experiment_name="${JOB_NAME:-sdpo_sweep}" \
    trainer.group_name="SDPO-baseline" \
    "trainer.logger=[console,swanlab]"
