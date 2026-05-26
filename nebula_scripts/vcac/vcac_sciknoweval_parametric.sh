#!/usr/bin/env bash
# =============================================================================
# VCAC (Verdict-Conditioned Advantage Correction) — sciknoweval parametric script.
#
# VCAC runs the GRPO PG loss pathway (compute_policy_loss_vanilla, PPO clip)
# with per-token credit shaping:
#     A_t = A_GRPO + vcac_lambda · δ_t
# where δ_t = log π_T^+(y_t|x,y_<t) − log π_T^-(y_t|x,y_<t) is the sampled-token
# log-prob differential from the CDM dual-teacher forward (self-as-teacher,
# different marker-injected inputs). See research/teacher_primed_value_grpo.md
# §3.3 (Ng-Harada-Russell potential-shaping safety) for the theoretical basis,
# and research/verdict_credit_assignment.md for the Bayes-clean derivation of
# δ_t as a verdict log-odds shift.
#
# Marker design here mirrors SDPO sibling-success ref reprompt + verdict-only
# markers (cdm_use_ref=True + marker-only templates) so that no GT is leaked at
# training time. The two teachers share an identical reprompted user message;
# they differ ONLY in the assistant-side verdict marker.
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
check_env VCAC_LAMBDA

SEED="${SEED:-42}"

# ── VCAC-specific knobs ───────────────────────────────────────────────
# VCAC_LAMBDA  : mixing weight on δ_t. Sensible range 0.1 ~ 1.0.
# VCAC_CLIP    : per-token clip on |δ_t| before mixing (null disables).
# VCAC_NORMALIZE : standardize δ_t per-batch before λ scale (False default).
VCAC_CLIP="${VCAC_CLIP:-null}"
VCAC_NORMALIZE="${VCAC_NORMALIZE:-False}"
# Pure-δ ablation toggle. True = A_t = A_GRPO + λ·δ_t (default, additive
# shaping). False = A_t = λ·δ_t only (drop the GRPO baseline term).
VCAC_USE_GRPO_ADV="${VCAC_USE_GRPO_ADV:-True}"

# ── CDM dual-teacher marker design (sibling-success ref + verdict-only) ──
# VCAC requires teacher_context_mode=cdm and loss_method=cdm so the batch
# builder emits dual (positive / negative) marker-injected inputs. We force
# cdm_use_ref=True and override the templates to verdict-only — no GT leakage
# at train time. CDM_TEMPLATE_VARIANT can be overridden to explore alt variants.
CDM_TEMPLATE_VARIANT="${CDM_TEMPLATE_VARIANT:-ref_mk}"

# Defaults below match SDPO's "ref_mk" preset; kept overridable for sweeps.
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
# VCAC does not use the JSD distillation loss, but the batch builder still
# needs alpha for the SelfDistillationConfig schema. Pin a sane default.
ALPHA="${ALPHA:-0.5}"

# ── Checkpoint / validation cadence (best + last only by default) ─────
# SAVE_FREQ=9999 means periodic save never fires within total_training_steps=250;
# the only checkpoints written are (a) best (via save_best_metric) and
# (b) last step (force-saved by trainer on is_last_step). This is the
# requested "best + last only" save policy.
TEST_FREQ="${TEST_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-9999}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
SAVE_HF_ONLY="${SAVE_HF_ONLY:-True}"
if [ "${SAVE_HF_ONLY}" = "True" ]; then
    SAVE_CONTENTS_HYDRA="[hf_model]"
else
    SAVE_CONTENTS_HYDRA="[model,optimizer,extra,hf_model]"
fi

# 数据集路径
train_data_path="${OSS_ROOT}/datasets/${DATASET}/train.parquet"
val_data_path="${OSS_ROOT}/datasets/${DATASET}/test.parquet"
model_path="${OSS_ROOT}/base_models/${MODEL_NAME}"
save_path="${OSS_ROOT}/rl_models/${JOB_NAME:-vcac_sciknoweval_sweep}"

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
    actor_rollout_ref.actor.policy_loss.loss_mode=vcac \
    actor_rollout_ref.actor.policy_loss.vcac_lambda=${VCAC_LAMBDA} \
    actor_rollout_ref.actor.policy_loss.vcac_clip=${VCAC_CLIP} \
    actor_rollout_ref.actor.policy_loss.vcac_normalize=${VCAC_NORMALIZE} \
    actor_rollout_ref.actor.policy_loss.vcac_use_grpo_advantage=${VCAC_USE_GRPO_ADV} \
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
    trainer.project_name="${PROJECT_NAME:-VCAC}" \
    trainer.experiment_name="${JOB_NAME:-vcac_sciknoweval_sweep}" \
    trainer.group_name="VCAC-sciknoweval" \
    "trainer.logger=[console,swanlab]"
