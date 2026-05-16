#!/usr/bin/env bash
# =============================================================================
# Teacher-Guided Dynamic Intervention (TGDI / v3) 参数化训练脚本
#
# Tier 3 of Bayesian Credit Assignment (ours):
#   1. Student rollout y_s (n_initial 个)
#   2. Teacher forward → g_t = KL(D_t‖D_{t-1}) + teacher_log_probs
#   3. failed = R(y_s) < threshold；t* = divergence(logp_T, logp_S, g_t)
#   4. (Phase 2) Teacher 接管 k 个 token → y_int；student 续 y_tail
#   5. (Phase 2) y' = (y_<t*, y_int, y_tail) append 同 uid → ΔR = R(y') - R(y_s)
#   6. A_seq = (R - mean_group(R)) + λ·ΔR (仅 intervention 样本)
#      A_t   = A_seq · ĝ_t · length_scale
#
# Phase 1 (本提交): IC_ENABLE_INTERVENTION=False → 退化为 prior_shift-equivalent
#                    用于验证 plumbing；SwanLab 上能看到 failed_rate / t*分布
# Phase 2 (待实现): 真实 teacher generation + student tail rollout
# =============================================================================
set +xo pipefail

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"

# ── 从环境变量读取超参 ────────────────────────────────────────────────
check_env() { val=$(eval echo "\$$1"); [ -n "$val" ] || { echo "ERROR: $1 is not set. Aborting."; exit 1; }; }
check_env DATASET
check_env LR
check_env TRAIN_BATCH_SIZE
check_env MINI_BATCH_SIZE
check_env ROLLOUT_N
check_env MODEL_NAME
check_env TEACHER_REGULARIZATION
check_env TEACHER_UPDATE_RATE

# 可选 env: 用于本地 notebook smoke / tstar 模式覆盖 (默认值与 Nebula 任务一致)
TOTAL_STEPS="${TOTAL_STEPS:-250}"
VAL_N="${VAL_N:-16}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-4}"

# Intervention-Credit 专属超参（带默认值）
IC_ENABLE_INTERVENTION="${IC_ENABLE_INTERVENTION:-False}"        # Phase 1 默认 False
IC_BASE_ESTIMATOR="${IC_BASE_ESTIMATOR:-grpo}"                    # grpo | rlsd | prior_shift
IC_RLSD_EPS_W="${IC_RLSD_EPS_W:-0.2}"                             # base=rlsd 专属
IC_DIVERGENCE_METRIC="${IC_DIVERGENCE_METRIC:-argmax_excl_eos}"   # 3 种策略消融
IC_EXCLUDE_TAIL_TOKENS="${IC_EXCLUDE_TAIL_TOKENS:-8}"
IC_K="${IC_K:-2}"                                                 # intervention 长度
IC_FAILED_THRESHOLD="${IC_FAILED_THRESHOLD:-0.5}"
IC_MAX_INTERVENTION_PER_GROUP="${IC_MAX_INTERVENTION_PER_GROUP:-7}"
IC_TEACHER_DECODE_TEMPERATURE="${IC_TEACHER_DECODE_TEMPERATURE:-0.0}"
IC_LAMBDA_DR="${IC_LAMBDA_DR:-1.0}"

# g_t 防护参数（复用 prior_shift v2）
IC_GT_EPS_NORM="${IC_GT_EPS_NORM:-1.0e-6}"
IC_GT_MAX_RATIO="${IC_GT_MAX_RATIO:-3.0}"
IC_GT_RENORMALIZE_AFTER_CLIP="${IC_GT_RENORMALIZE_AFTER_CLIP:-True}"
IC_GT_UNIFORM_FALLBACK="${IC_GT_UNIFORM_FALLBACK:-True}"

# Length floor (复用 prior_shift v2)
IC_MIN_RESPONSE_LENGTH="${IC_MIN_RESPONSE_LENGTH:-50}"
IC_LENGTH_PENALTY_TYPE="${IC_LENGTH_PENALTY_TYPE:-linear}"

# 数据集路径：优先用 wrapper 已 export 的 TRAIN_DATA_PATH/VAL_DATA_PATH，
# 否则按 SUBJECT（若提供）回退到 OSS 路径
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
save_path="${OSS_ROOT}/models/${JOB_NAME:-intervention_credit_sweep}"

# ── 动态 save_best_metric（按 DATASET 区分 val 指标路径）──────────────
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
    --config-name intervention_credit \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    algorithm.adv_estimator=intervention_credit \
    algorithm.intervention_credit.enable_intervention=${IC_ENABLE_INTERVENTION} \
    algorithm.intervention_credit.base_estimator=${IC_BASE_ESTIMATOR} \
    algorithm.intervention_credit.rlsd_eps_w=${IC_RLSD_EPS_W} \
    algorithm.intervention_credit.divergence_metric=${IC_DIVERGENCE_METRIC} \
    algorithm.intervention_credit.exclude_tail_tokens=${IC_EXCLUDE_TAIL_TOKENS} \
    algorithm.intervention_credit.intervention_length_k=${IC_K} \
    algorithm.intervention_credit.failed_threshold=${IC_FAILED_THRESHOLD} \
    algorithm.intervention_credit.max_intervention_per_group=${IC_MAX_INTERVENTION_PER_GROUP} \
    algorithm.intervention_credit.teacher_decode_temperature=${IC_TEACHER_DECODE_TEMPERATURE} \
    algorithm.intervention_credit.lambda_delta_r=${IC_LAMBDA_DR} \
    algorithm.intervention_credit.g_t.eps_norm=${IC_GT_EPS_NORM} \
    algorithm.intervention_credit.g_t.max_ratio=${IC_GT_MAX_RATIO} \
    algorithm.intervention_credit.g_t.renormalize_after_clip=${IC_GT_RENORMALIZE_AFTER_CLIP} \
    algorithm.intervention_credit.g_t.uniform_fallback=${IC_GT_UNIFORM_FALLBACK} \
    algorithm.intervention_credit.min_response_length=${IC_MIN_RESPONSE_LENGTH} \
    algorithm.intervention_credit.length_penalty_type=${IC_LENGTH_PENALTY_TYPE} \
    algorithm.rollout_correction.rollout_is=token \
    trainer.total_epochs=30 \
    trainer.total_training_steps=${TOTAL_STEPS} \
    trainer.save_freq=-1 \
    trainer.save_best_metric="${SAVE_BEST_METRIC}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME:-TGDI-Tier3}" \
    trainer.experiment_name="${JOB_NAME:-intervention_credit_sweep}" \
    trainer.group_name="TGDI-v3" \
    "trainer.logger=[console,swanlab]"
