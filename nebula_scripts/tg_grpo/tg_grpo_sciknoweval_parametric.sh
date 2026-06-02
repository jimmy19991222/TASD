#!/usr/bin/env bash
# =============================================================================
# TG-GRPO (Teacher-Guided GRPO) — sciknoweval parametric script.
#
# TG-GRPO keeps the standard GRPO PG loss path (compute_policy_loss_vanilla,
# PPO clip) with advantage A_t = R - R̄_group unchanged. The teacher's only
# role is a per-token weight w_t derived from |δ_t|, the CDM dual-teacher
# log-prob differential. response_mask in the pg_loss call is replaced by w_t,
# concentrating gradient on decision tokens (large |δ|) while masking out
# fluency tokens (small |δ|).
#
#   loss = -Σ_t w_t · A_t^GRPO · log π_θ(y_t | s_t)
#
# Compared to Bayes-DR (which makes the teacher do both V estimation and per-
# token credit allocation), TG-GRPO keeps only the credit-allocation role,
# avoiding the V_πT ≠ V_πS structural bias. Cost is identical (2× teacher
# forward + 1× student) but PG unbiasedness is preserved.
#
# All hyperparameters are injected via nebulactl --env.
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

SEED="${SEED:-42}"

# ── TG-GRPO specific knobs ────────────────────────────────────────────
# TG_TOP_K            : fraction of tokens to keep (percentile modes).
# TG_W_MODE           : topk_traj (recommended) / topk_batch / sigmoid / hard.
# TG_W_SIGMOID_BETA   : sharpness β for sigmoid mode.
# TG_W_HARD_THRESHOLD : threshold τ for hard mode.
TG_TOP_K="${TG_TOP_K:-0.30}"
TG_W_MODE="${TG_W_MODE:-topk_traj}"
TG_W_SIGMOID_BETA="${TG_W_SIGMOID_BETA:-5.0}"
TG_W_HARD_THRESHOLD="${TG_W_HARD_THRESHOLD:-0.3}"

# ── CDM dual-teacher marker design (sibling-success ref + verdict-only) ──
# TG-GRPO requires teacher_context_mode=cdm and loss_method=cdm so the batch
# builder emits dual (positive / negative) marker-injected inputs.
CDM_TEMPLATE_VARIANT="${CDM_TEMPLATE_VARIANT:-ref_mk}"

_DEFAULT_CDM_POS_TEMPLATE='This answer is verified correct.'
_DEFAULT_CDM_NEG_TEMPLATE='This answer is verified incorrect.'
CDM_POSITIVE_TEMPLATE="${CDM_POSITIVE_TEMPLATE:-$_DEFAULT_CDM_POS_TEMPLATE}"
CDM_NEGATIVE_TEMPLATE="${CDM_NEGATIVE_TEMPLATE:-$_DEFAULT_CDM_NEG_TEMPLATE}"
CDM_USE_REF="${CDM_USE_REF:-True}"
case "${CDM_TEMPLATE_VARIANT}" in
    ref_mk)
        CDM_POSITIVE_TEMPLATE='This answer is verified correct.'
        CDM_NEGATIVE_TEMPLATE='This answer is verified incorrect.'
        CDM_USE_REF="True"
        ;;
    mkonly)
        CDM_POSITIVE_TEMPLATE='This answer is verified correct.'
        CDM_NEGATIVE_TEMPLATE='This answer is verified incorrect.'
        CDM_USE_REF="False"
        ;;
    default)
        _DEFAULT_CDM_POS_TEMPLATE_GT='This answer is verified correct, reference answer is {ground_truth}.'
        _DEFAULT_CDM_NEG_TEMPLATE_GT='This answer is verified incorrect, reference answer is {ground_truth}.'
        CDM_POSITIVE_TEMPLATE="$_DEFAULT_CDM_POS_TEMPLATE_GT"
        CDM_NEGATIVE_TEMPLATE="$_DEFAULT_CDM_NEG_TEMPLATE_GT"
        ;;
    *) echo "WARNING: unknown CDM_TEMPLATE_VARIANT=${CDM_TEMPLATE_VARIANT}, using current values" ;;
esac

# ── Reprompt / data construction defaults ────────────────────────────
DONT_REPROMPT_ON_SELF_SUCCESS="${DONT_REPROMPT_ON_SELF_SUCCESS:-False}"
# TG-GRPO does not use JSD distillation but the batch builder needs alpha
# for the SelfDistillationConfig schema. Pin a sane default.
ALPHA="${ALPHA:-0.5}"

# ── Checkpoint / validation cadence ──────────────────────────────────
TEST_FREQ="${TEST_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-10}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
SAVE_HF_ONLY="${SAVE_HF_ONLY:-True}"
if [ "${SAVE_HF_ONLY}" = "True" ]; then
    SAVE_CONTENTS_HYDRA="[hf_model]"
else
    SAVE_CONTENTS_HYDRA="[model,optimizer,extra,hf_model]"
fi

train_data_path="${OSS_ROOT}/datasets/${DATASET}/train.parquet"
val_data_path="${OSS_ROOT}/datasets/${DATASET}/test.parquet"
model_path="${OSS_ROOT}/base_models/${MODEL_NAME}"
save_path="${OSS_ROOT}/rl_models/${JOB_NAME:-tg_grpo_sciknoweval_pilot}"

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
    actor_rollout_ref.actor.policy_loss.loss_mode=tg_grpo \
    actor_rollout_ref.actor.policy_loss.tg_top_k=${TG_TOP_K} \
    actor_rollout_ref.actor.policy_loss.tg_w_mode=${TG_W_MODE} \
    actor_rollout_ref.actor.policy_loss.tg_w_sigmoid_beta=${TG_W_SIGMOID_BETA} \
    actor_rollout_ref.actor.policy_loss.tg_w_hard_threshold=${TG_W_HARD_THRESHOLD} \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=True \
    actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
    actor_rollout_ref.actor.self_distillation.alpha=${ALPHA} \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONT_REPROMPT_ON_SELF_SUCCESS} \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.actor.self_distillation.loss_method=cdm \
    actor_rollout_ref.actor.self_distillation.teacher_context_mode=cdm \
    actor_rollout_ref.actor.self_distillation.cdm_positive_template="'${CDM_POSITIVE_TEMPLATE}'" \
    actor_rollout_ref.actor.self_distillation.cdm_negative_template="'${CDM_NEGATIVE_TEMPLATE}'" \
    actor_rollout_ref.actor.self_distillation.cdm_use_ref=${CDM_USE_REF} \
    actor_rollout_ref.actor.checkpoint.save_contents=${SAVE_CONTENTS_HYDRA} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    algorithm.rollout_correction.rollout_is=token \
    trainer.total_epochs=30 \
    trainer.total_training_steps=250 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.max_actor_ckpt_to_keep=null \
    trainer.test_freq=${TEST_FREQ} \
    trainer.save_best_metric="val-core/sciknoweval/acc/mean@16" \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME:-TG-GRPO}" \
    trainer.experiment_name="${JOB_NAME:-tg_grpo_sciknoweval_pilot}" \
    trainer.group_name="TG-GRPO-sciknoweval" \
    "trainer.logger=[console,swanlab]"
