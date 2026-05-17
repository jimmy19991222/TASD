#!/bin/bash
# =============================================================================
# DPO-TGS (On-Policy DPO + Teacher-Guided Sampling) — Nebula 提交脚本
#
# 论文叙事: 把 OAIF on-policy + Samplers-DPO reward-aware + OFS-DPO gradient
# stabilization + RPO 外部信号注入 四者合一, 用 token-level OPSD splice + verifiable
# env reward 替代 LLM judge/RM/hint quality.
#
# 默认 sweep (smoke): 1 job, chain_length=2, beta=0.1, alpha=1.0
#   → 验证 plumbing 是否通, 关注 dpo/pair_win_rate / dpo/margin_mean / val acc
#
# 论文主结果 sweep (待 smoke 通过后启动):
#   CHAIN_LENGTH_LIST="2 4" DPO_BETA_LIST="0.05 0.1 0.5" bash submit_dpo_tgs_sweep.sh
#
# 与 GRPO baseline / TCCA-Lite 对比时, 复用其各自 submit 脚本即可 (相同 dataset/model).
#
# 使用方式:
#   bash nebula_scripts/submit_dpo_tgs_sweep.sh [--dry-run]
# =============================================================================
# set -euo pipefail

# ── Nebula 账号配置 ──────────────────────────────────────────────────────
QUEUE="lazada_llm_ad_h20"
WORLD_SIZE=1
OPENLM_TOKEN="${OPENLM_TOKEN:?OPENLM_TOKEN not set}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:?OSS_ACCESS_ID not set}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?OSS_ACCESS_KEY not set}"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
CLUSTER_FILE="nebula_scripts/cluster.json"
SCRIPT_PATH="nebula_scripts/dpo_tgs/dpo_tgs_sciknoweval_parametric.sh"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="DPO-TGS"

# ── 数据集配置 ────────────────────────────────────────────────────────
DATASETS_DEFAULT="sciknoweval/biology"
read -r -a DATASETS <<< "${DATASETS_OVERRIDE:-$DATASETS_DEFAULT}"

# ── dry-run ──────────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式: 只打印命令, 不提交"
fi

# =============================================================================
# Sweep 超参 (默认 smoke = 1 job)
# =============================================================================
N_INIT_LIST_DEFAULT="2"
read -r -a N_INIT_LIST <<< "${N_INIT_LIST:-$N_INIT_LIST_DEFAULT}"

N_ATTEMPTS_LIST_DEFAULT="2"
read -r -a N_ATTEMPTS_LIST <<< "${N_ATTEMPTS_LIST:-$N_ATTEMPTS_LIST_DEFAULT}"

DPO_BETA_LIST_DEFAULT="0.1"
read -r -a DPO_BETA_LIST <<< "${DPO_BETA_LIST:-$DPO_BETA_LIST_DEFAULT}"

DPO_ALPHA_LIST_DEFAULT="1.0"
read -r -a DPO_ALPHA_LIST <<< "${DPO_ALPHA_LIST:-$DPO_ALPHA_LIST_DEFAULT}"

DPO_PAIR_STRATEGY_LIST_DEFAULT="chain_consecutive"
read -r -a DPO_PAIR_STRATEGY_LIST <<< "${DPO_PAIR_STRATEGY_LIST:-$DPO_PAIR_STRATEGY_LIST_DEFAULT}"

# 固定参数
DPO_CORRECT_THRESHOLD="${DPO_CORRECT_THRESHOLD:-1.0}"
DPO_SDPO_CTX_SOURCE="${DPO_SDPO_CTX_SOURCE:-sibling_correct}"
DPO_ALL_FAILED_STRATEGY="${DPO_ALL_FAILED_STRATEGY:-skip}"
DPO_MAX_RESELECT="${DPO_MAX_RESELECT:-3}"
DPO_EXCLUDE_TAIL="${DPO_EXCLUDE_TAIL:-8}"
DPO_PAIR_MARGIN="0.0"
DPO_MIN_RESP_LEN="50"
DPO_LEN_PENALTY="linear"

# ── 3 innovation knobs (default OFF, override via env to ablate) ──
DPO_CAUSAL_LOCALIZE="${DPO_CAUSAL_LOCALIZE:-False}"
DPO_BETA_TOKEN="${DPO_BETA_TOKEN:-}"
DPO_BETA_CONTINUATION="${DPO_BETA_CONTINUATION:-}"
DPO_USE_TEACHER_ANCHORED_REF="${DPO_USE_TEACHER_ANCHORED_REF:-False}"
DPO_DELTA_R_WEIGHT_MODE="${DPO_DELTA_R_WEIGHT_MODE:-none}"
DPO_DELTA_R_WEIGHT_TAU="${DPO_DELTA_R_WEIGHT_TAU:-1.0}"
IC_DIVERGENCE_METRIC="argmax_excl_eos"
IC_TEACHER_DECODE_TEMPERATURE="0.0"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

# Teacher EMA
TEACHER_UPDATE_RATE_LIST=(
    "0.05"
)
TEACHER_REGULARIZATION="ema"

# 固定参数
LR="1e-5"
SEED="42"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
MODEL_NAME="Qwen3-8B"

# Smoke run knobs: 减少 step / disable val before train
TOTAL_STEPS="${TOTAL_STEPS:-100}"   # smoke = 100 step ~ 4h
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# =============================================================================
# 提交循环
# =============================================================================
TOTAL=0
SUBMITTED=0

for DATASET in "${DATASETS[@]}"; do
for N_INIT in "${N_INIT_LIST[@]}"; do
for N_ATTEMPTS in "${N_ATTEMPTS_LIST[@]}"; do
for DPO_BETA in "${DPO_BETA_LIST[@]}"; do
for DPO_ALPHA in "${DPO_ALPHA_LIST[@]}"; do
for DPO_PAIR_STRATEGY in "${DPO_PAIR_STRATEGY_LIST[@]}"; do
for TEACHER_UPDATE_RATE in "${TEACHER_UPDATE_RATE_LIST[@]}"; do
    TOTAL=$((TOTAL + 1))

    DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
    MODEL_SHORT=$(echo "$MODEL_NAME" | tr '.' '-')
    OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
    MODEL_PATH="${OSS_ROOT}/base_models/${MODEL_NAME}"
    ROLLOUT_N="${N_INIT}"   # rollout.n MUST equal n_init (Phase 1 rollouts)
    DPO_N_INIT="${N_INIT}"
    DPO_N_ATTEMPTS="${N_ATTEMPTS}"

    BETA_TAG="b${DPO_BETA}"
    ALPHA_TAG="a${DPO_ALPHA}"
    CHAIN_TAG="ni${N_INIT}na${N_ATTEMPTS}"
    PS_TAG=$(echo "${DPO_PAIR_STRATEGY}" | tr '_' '-')
    EMA_TAG="ema${TEACHER_UPDATE_RATE}"
    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="DPO-TGS-${CHAIN_TAG}-${BETA_TAG}-${ALPHA_TAG}-${PS_TAG}-${DATASET_SHORT}-${EMA_TAG}-${MODEL_SHORT}-${CURRENT_TIME}"

    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  DATASET=$DATASET MODEL=$MODEL_NAME"
        echo "  N_INIT=$N_INIT N_ATTEMPTS=$N_ATTEMPTS ROLLOUT_N=$ROLLOUT_N"
        echo "  DPO_BETA=$DPO_BETA DPO_ALPHA=$DPO_ALPHA"
        echo "  DPO_PAIR_STRATEGY=$DPO_PAIR_STRATEGY DPO_PAIR_MARGIN=$DPO_PAIR_MARGIN"
        echo "  DPO_SDPO_CTX_SOURCE=$DPO_SDPO_CTX_SOURCE DPO_ALL_FAILED_STRATEGY=$DPO_ALL_FAILED_STRATEGY"
        echo "  DPO_CORRECT_THRESHOLD=$DPO_CORRECT_THRESHOLD DPO_MAX_RESELECT=$DPO_MAX_RESELECT"
        echo "  ── INNOVATIONS ──"
        echo "  ① CAUSAL_LOCALIZE=$DPO_CAUSAL_LOCALIZE  beta_token=$DPO_BETA_TOKEN beta_cont=$DPO_BETA_CONTINUATION"
        echo "  ② USE_TEACHER_ANCHORED_REF=$DPO_USE_TEACHER_ANCHORED_REF"
        echo "  ③ DELTA_R_WEIGHT_MODE=$DPO_DELTA_R_WEIGHT_MODE  tau=$DPO_DELTA_R_WEIGHT_TAU"
        echo "  TEACHER_REG=$TEACHER_REGULARIZATION TEACHER_UPDATE_RATE=$TEACHER_UPDATE_RATE"
        echo "  LR=$LR BS=$TRAIN_BATCH_SIZE/$MINI_BATCH_SIZE TOTAL_STEPS=$TOTAL_STEPS"
        echo "  GIT_BRANCH=$GIT_BRANCH GIT_COMMIT=$GIT_COMMIT"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
            --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL_NAME=${MODEL_NAME} --env=MODEL_PATH=${MODEL_PATH} --env=LR=${LR} --env=SEED=${SEED} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=DPO_N_INIT=${DPO_N_INIT} --env=DPO_N_ATTEMPTS=${DPO_N_ATTEMPTS} --env=DPO_CORRECT_THRESHOLD=${DPO_CORRECT_THRESHOLD} --env=DPO_SDPO_CTX_SOURCE=${DPO_SDPO_CTX_SOURCE} --env=DPO_ALL_FAILED_STRATEGY=${DPO_ALL_FAILED_STRATEGY} --env=DPO_MAX_RESELECT=${DPO_MAX_RESELECT} --env=DPO_EXCLUDE_TAIL=${DPO_EXCLUDE_TAIL} --env=TEACHER_REGULARIZATION=${TEACHER_REGULARIZATION} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=DPO_BETA=${DPO_BETA} --env=DPO_ALPHA=${DPO_ALPHA} --env=DPO_CAUSAL_LOCALIZE=${DPO_CAUSAL_LOCALIZE} --env=DPO_BETA_TOKEN=${DPO_BETA_TOKEN} --env=DPO_BETA_CONTINUATION=${DPO_BETA_CONTINUATION} --env=DPO_USE_TEACHER_ANCHORED_REF=${DPO_USE_TEACHER_ANCHORED_REF} --env=DPO_DELTA_R_WEIGHT_MODE=${DPO_DELTA_R_WEIGHT_MODE} --env=DPO_DELTA_R_WEIGHT_TAU=${DPO_DELTA_R_WEIGHT_TAU} --env=DPO_PAIR_STRATEGY=${DPO_PAIR_STRATEGY} --env=DPO_PAIR_MARGIN=${DPO_PAIR_MARGIN} --env=DPO_MIN_RESP_LEN=${DPO_MIN_RESP_LEN} --env=DPO_LEN_PENALTY=${DPO_LEN_PENALTY} --env=IC_DIVERGENCE_METRIC=${IC_DIVERGENCE_METRIC} --env=IC_TEACHER_DECODE_TEMPERATURE=${IC_TEACHER_DECODE_TEMPERATURE} --env=GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION} --env=TOTAL_STEPS=${TOTAL_STEPS} --env=VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=f598ad33b071751bf79d2484d8e1acefe8df9d879e129cae40340a158854f9cb --env=DINGTALK_SECRET=SECc5b9e4f61f56b32b46abf1ecedc11bdcba10dc35fbba8fa0ff62c084a1cc6ad3" \
            --worker_count=${WORLD_SIZE} \
            --file.cluster_file=${CLUSTER_FILE} \
            --job_name=${JOB_NAME} \
            --env=OPENLM_TOKEN=${OPENLM_TOKEN} \
            --env=SWANLAB_API_KEY=${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal} \
            --custom_docker_image=${CUSTOM_DOCKER_IMAGE} \
            --requirements_file_name=requirements_nebula.txt \
            --oss_access_id=${OSS_ACCESS_ID} \
            --oss_access_key=${OSS_ACCESS_KEY} \
            --oss_bucket=${OSS_BUCKET} \
            --oss_endpoint=${OSS_ENDPOINT} \
            2>&1)
        SUBMIT_EXIT=$?
        echo "$SUBMIT_OUTPUT"
        if [ $SUBMIT_EXIT -ne 0 ]; then
            echo "❌ 提交失败 (exit code: $SUBMIT_EXIT)"
        else
            SUBMITTED=$((SUBMITTED + 1))
            echo "✅ 已提交 (${SUBMITTED}/${TOTAL})"
        fi
        sleep 2
    fi
done
done
done
done
done
done
done

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成, 共 ${TOTAL} 个 job"
    echo ""
    echo "[smoke 默认] n_init=${N_INIT_LIST[*]} × n_attempts=${N_ATTEMPTS_LIST[*]} × beta=${DPO_BETA_LIST[*]} × alpha=${DPO_ALPHA_LIST[*]} × pair_strategy=${DPO_PAIR_STRATEGY_LIST[*]} = ${TOTAL} job"
    echo "  → 验证 plumbing + 关注 dpo/pair_win_rate / dpo/margin_mean / val acc"
else
    echo "提交完成: ${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
