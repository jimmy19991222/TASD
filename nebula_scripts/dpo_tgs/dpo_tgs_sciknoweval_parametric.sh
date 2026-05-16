#!/usr/bin/env bash
# =============================================================================
# DPO-TGS (On-Policy DPO + Teacher-Guided Sampling) 参数化训练脚本
#
# 设计文档: research/dpo_teacher_guided_sampling.md
# 理论锚点: research/OPD_Deep_Analysis.html "Online DPO 理论锚点"
#
# Pipeline:
#   1. tcca_v2_chain_rollout (复用 TCCA-Lite) → chain_length 个 sample / prompt
#   2. reference model forward → ref_log_prob
#   3. collect_dpo_pairs (chain_consecutive) → (chosen, rejected) pairs
#   4. compute_dpo_tgs_advantage (linearized DPO + GRPO fallback)
#   5. standard PPO surrogate update
# =============================================================================
set +xo pipefail

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"

# ── 从环境变量读取超参 ────────────────────────────────────────────────
check_env() { val=$(eval echo "\$$1"); [ -n "$val" ] || { echo "ERROR: $1 is not set. Aborting."; exit 1; }; }
check_env DATASET
check_env LR
check_env TRAIN_BATCH_SIZE
check_env MINI_BATCH_SIZE
check_env MODEL_NAME
check_env TEACHER_REGULARIZATION
check_env TEACHER_UPDATE_RATE

TOTAL_STEPS="${TOTAL_STEPS:-250}"
VAL_N="${VAL_N:-16}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

# DPO-TGS V2 (adaptive rollout) — n_init MUST match rollout.n
DPO_N_INIT="${DPO_N_INIT:-2}"
DPO_N_ATTEMPTS="${DPO_N_ATTEMPTS:-2}"
DPO_CORRECT_THRESHOLD="${DPO_CORRECT_THRESHOLD:-1.0}"
DPO_SDPO_CTX_SOURCE="${DPO_SDPO_CTX_SOURCE:-sibling_correct}"
# NOTE: sdpo_ref_template is set in dpo_tgs.yaml (default: "Refer to this correct answer: {r}\n").
# We don't pass it via CLI because Hydra's grammar treats `{...}` as variable interpolation,
# causing "mismatched input '{'" parse errors. To customize, edit dpo_tgs.yaml directly.
DPO_ALL_FAILED_STRATEGY="${DPO_ALL_FAILED_STRATEGY:-skip}"
DPO_MAX_RESELECT="${DPO_MAX_RESELECT:-3}"
DPO_EXCLUDE_TAIL="${DPO_EXCLUDE_TAIL:-8}"

# rollout.n must equal n_init (Phase 1 baseline rollout count)
ROLLOUT_N="${ROLLOUT_N:-${DPO_N_INIT}}"

DPO_BETA="${DPO_BETA:-0.1}"
DPO_ALPHA="${DPO_ALPHA:-1.0}"
DPO_PAIR_STRATEGY="${DPO_PAIR_STRATEGY:-chain_consecutive}"   # chain_consecutive | hybrid_init_chain
DPO_PAIR_MARGIN="${DPO_PAIR_MARGIN:-0.0}"
DPO_MIN_RESP_LEN="${DPO_MIN_RESP_LEN:-50}"
DPO_LEN_PENALTY="${DPO_LEN_PENALTY:-linear}"

# Legacy intervention_credit knobs (mostly unused by adaptive_rollout)
IC_DIVERGENCE_METRIC="${IC_DIVERGENCE_METRIC:-argmax_excl_eos}"
IC_TEACHER_DECODE_TEMPERATURE="${IC_TEACHER_DECODE_TEMPERATURE:-0.0}"

# ── 数据集路径 ────────────────────────────────────────────────────────
if [ -n "${TRAIN_DATA_PATH:-}" ]; then
    train_data_path="${TRAIN_DATA_PATH}"
elif [ -n "${SUBJECT:-}" ]; then
    train_data_path="${OSS_ROOT}/datasets/${DATASET}/${SUBJECT}/train.parquet"
else
    train_data_path="${OSS_ROOT}/datasets/${DATASET}/train.parquet"
fi
if [ -n "${VAL_DATA_PATH:-}" ]; then
    val_data_path="${VAL_DATA_PATH}"
elif [ -n "${SUBJECT:-}" ]; then
    val_data_path="${OSS_ROOT}/datasets/${DATASET}/${SUBJECT}/test.parquet"
else
    val_data_path="${OSS_ROOT}/datasets/${DATASET}/test.parquet"
fi
model_path="${MODEL_PATH:-${OSS_ROOT}/base_models/${MODEL_NAME}}"
save_path="${OSS_ROOT}/models/${JOB_NAME:-dpo_tgs_sweep}"

# ── save best metric (DATASET 区分) ──────────────────────────────────
if [[ "${DATASET}" == tooluse* ]]; then
    SAVE_BEST_METRIC="val-core/tooluse/acc/mean@16"
else
    SAVE_BEST_METRIC="val-core/sciknoweval/acc/mean@16"
fi

# ── 环境 ──────────────────────────────────────────────────────────────
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

pip install -e . --no-deps --no-build-isolation --quiet 2>/dev/null || true

python -m verl.trainer.main_ppo \
    --config-name dpo_tgs \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.train_files="${train_data_path}" \
    data.val_files="${val_data_path}" \
    custom_reward_function.path="$(pwd)/verl/utils/reward_score/feedback/__init__.py" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=${TEACHER_REGULARIZATION} \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=${TEACHER_UPDATE_RATE} \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    algorithm.adv_estimator=dpo_teacher_guided \
    algorithm.dpo.n_init=${DPO_N_INIT} \
    algorithm.dpo.n_attempts=${DPO_N_ATTEMPTS} \
    algorithm.dpo.correct_threshold=${DPO_CORRECT_THRESHOLD} \
    algorithm.dpo.sdpo_ctx_source=${DPO_SDPO_CTX_SOURCE} \
    algorithm.dpo.all_failed_strategy=${DPO_ALL_FAILED_STRATEGY} \
    algorithm.dpo.max_reselect_attempts=${DPO_MAX_RESELECT} \
    algorithm.dpo.exclude_tail_tokens=${DPO_EXCLUDE_TAIL} \
    algorithm.dpo.beta=${DPO_BETA} \
    algorithm.dpo.alpha=${DPO_ALPHA} \
    algorithm.dpo.pair_strategy=${DPO_PAIR_STRATEGY} \
    algorithm.dpo.pair_margin=${DPO_PAIR_MARGIN} \
    algorithm.dpo.min_response_length=${DPO_MIN_RESP_LEN} \
    algorithm.dpo.length_penalty_type=${DPO_LEN_PENALTY} \
    algorithm.intervention_credit.enable_intervention=True \
    algorithm.intervention_credit.divergence_metric=${IC_DIVERGENCE_METRIC} \
    algorithm.intervention_credit.teacher_decode_temperature=${IC_TEACHER_DECODE_TEMPERATURE} \
    algorithm.rollout_correction.rollout_is=token \
    trainer.total_epochs=30 \
    trainer.total_training_steps=${TOTAL_STEPS} \
    trainer.save_freq=-1 \
    trainer.save_best_metric="${SAVE_BEST_METRIC}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME:-DPO-TGS}" \
    trainer.experiment_name="${JOB_NAME:-dpo_tgs_sweep}" \
    trainer.group_name="DPO-TGS-v1" \
    "trainer.logger=[console,swanlab]"
