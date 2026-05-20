#!/usr/bin/env bash
# =============================================================================
# Teacher-QV 参数化训练脚本（供 Nebula sweep 调用）
#
# Token-level A = Q - V，PG loss：
#   Q_t = log p_teacher(y_t)
#   V_t 由 BASELINE_TYPE 选择：
#     student     : V_t = log p_student(y_t)
#     ce          : V_t = -log p_student(y_t)
#     group_mean  : V   = batch-mean(Q over valid tokens)
#     group_hier  : within-seq z-score + across-seq z-score (uid 分组)
# =============================================================================
set +xo pipefail

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"

check_env() { val=$(eval echo "\$$1"); [ -n "$val" ] || { echo "ERROR: $1 is not set. Aborting."; exit 1; }; }
check_env DATASET
check_env LR
check_env DONT_REPROMPT_ON_SELF_SUCCESS
check_env TRAIN_BATCH_SIZE
check_env ROLLOUT_N
check_env MODEL_NAME
check_env BASELINE_TYPE   # student | ce | group_mean | group_hier

SEED="${SEED:-42}"
NORM_BY_STD="${NORM_BY_STD:-False}"
CLIP_VALUE="${CLIP_VALUE:-null}"
STD_FLOOR="${STD_FLOOR:-1e-3}"
DETACH_Q="${DETACH_Q:-True}"
DETACH_V="${DETACH_V:-True}"

# Geodesic Fisher manifold weighting (orthogonal to baseline_type)
USE_GEODESIC="${USE_GEODESIC:-False}"
GEODESIC_TRUST_REGION="${GEODESIC_TRUST_REGION:-5.0}"

# baseline_type='ce' needs full or topk logits on both sides.
# Other baselines can leave FULL_LOGIT=False to save memory.
if [ "${BASELINE_TYPE}" = "ce" ]; then
    FULL_LOGIT="${FULL_LOGIT:-True}"
    DISTILL_TOPK="${DISTILL_TOPK:-100}"
else
    FULL_LOGIT="${FULL_LOGIT:-False}"
    DISTILL_TOPK="${DISTILL_TOPK:-null}"
fi

train_data_path="${OSS_ROOT}/datasets/${DATASET}/train.parquet"
val_data_path="${OSS_ROOT}/datasets/${DATASET}/test.parquet"
model_path="${OSS_ROOT}/base_models/${MODEL_NAME}"
save_path="${OSS_ROOT}/models/${JOB_NAME:-teacher_qv_sweep}"

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
    --config-name teacher_qv \
    seed=${SEED} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.train_files="${train_data_path}" \
    data.val_files="${val_data_path}" \
    custom_reward_function.path="$(pwd)/verl/utils/reward_score/feedback/__init__.py" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.policy_loss.loss_mode=teacher_qv \
    actor_rollout_ref.actor.policy_loss.teacher_qv.baseline_type=${BASELINE_TYPE} \
    actor_rollout_ref.actor.policy_loss.teacher_qv.norm_by_std=${NORM_BY_STD} \
    actor_rollout_ref.actor.policy_loss.teacher_qv.clip_value=${CLIP_VALUE} \
    actor_rollout_ref.actor.policy_loss.teacher_qv.std_floor=${STD_FLOOR} \
    actor_rollout_ref.actor.policy_loss.teacher_qv.detach_q=${DETACH_Q} \
    actor_rollout_ref.actor.policy_loss.teacher_qv.detach_v=${DETACH_V} \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=${FULL_LOGIT} \
    actor_rollout_ref.actor.self_distillation.distillation_topk=${DISTILL_TOPK} \
    actor_rollout_ref.actor.self_distillation.use_geodesic=${USE_GEODESIC} \
    actor_rollout_ref.actor.self_distillation.geodesic_trust_region=${GEODESIC_TRUST_REGION} \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONT_REPROMPT_ON_SELF_SUCCESS} \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
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
    trainer.experiment_name="${JOB_NAME:-teacher_qv_sweep}" \
    trainer.group_name="Teacher-QV-${BASELINE_TYPE}" \
    "trainer.logger=[console,swanlab]"
