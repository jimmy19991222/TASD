#!/usr/bin/env bash
# =============================================================================
# Verdict-Calibration Diagnostic (one-shot offline job, no training)
#
# Runs scripts/diagnostics/verdict_calibration.py:
#   - generates N rollouts × M examples on the val set
#   - scores each via verifier (binary R)
#   - reads `delta_seq = log p_T(rollout|ctx_right) - log p_T(rollout|ctx_wrong)`
#   - reports AUC(delta_seq, R) + per-token AUC + Cohen's d + verdict
#
# Verdict: AUC ≥ 0.8 PROCEED to verdict_contrast baseline_type. AUC < 0.7 STOP.
# =============================================================================
set +xo pipefail

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"

check_env() { val=$(eval echo "\$$1"); [ -n "$val" ] || { echo "ERROR: $1 is not set. Aborting."; exit 1; }; }
check_env DATASET
check_env MODEL_NAME

N_EXAMPLES="${N_EXAMPLES:-50}"
N_ROLLOUTS_PER_EXAMPLE="${N_ROLLOUTS_PER_EXAMPLE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
SEED="${SEED:-42}"
DATA_SOURCE="${DATA_SOURCE:-sciknoweval}"

val_data_path="${OSS_ROOT}/datasets/${DATASET}/test.parquet"
model_path="${OSS_ROOT}/base_models/${MODEL_NAME}"
output_root="${OSS_ROOT}/diagnostics/${JOB_NAME:-verdict_calibration}"
output_json="${output_root}/result.json"

mkdir -p "${output_root}" 2>/dev/null || true

# ── 环境 ──────────────────────────────────────────────────────────────
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal}"
export SWANLAB_LOG_DIR="${OSS_ROOT}/logs/swanlab_logs"
mkdir -p "${SWANLAB_LOG_DIR}" 2>/dev/null || true

pip install -e . --no-deps --no-build-isolation --quiet 2>/dev/null || true

python scripts/diagnostics/verdict_calibration.py \
    --model_path "${model_path}" \
    --val_parquet "${val_data_path}" \
    --data_source "${DATA_SOURCE}" \
    --n_examples ${N_EXAMPLES} \
    --n_rollouts_per_example ${N_ROLLOUTS_PER_EXAMPLE} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --seed ${SEED} \
    --output_json "${output_json}" \
    --swanlab_project "${PROJECT_NAME:-SDPO_LossVariants}" \
    --swanlab_experiment "${JOB_NAME:-verdict_calibration}"
