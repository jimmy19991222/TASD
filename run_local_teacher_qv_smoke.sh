#!/bin/bash
# =============================================================================
# Teacher-QV 本地 Smoke Test（A100 Notebook 环境）
#
# 验证新增的 token-level A = Q - V 的 4 个 baseline 变体能否跑通：
#   student | ce | group_mean | group_hier
#
# 用法：
#   ./run_local_teacher_qv_smoke.sh                       # 默认 baseline_type=student
#   ./run_local_teacher_qv_smoke.sh ce                    # 指定 baseline_type
#   ./run_local_teacher_qv_smoke.sh group_hier qv_hier    # 同时指定后缀
# =============================================================================

set -euo pipefail

CONFIG_NAME="teacher_qv"

# 数据集路径（notebook 环境 OSS 挂载）
OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
DATA_PATH="${OSS_ROOT}/datasets/sciknoweval/biology"
TRAIN_FILE="${DATA_PATH}/train.parquet"
VAL_FILE="${DATA_PATH}/test.parquet"

TRAIN_BATCH_SIZE=8
ROLLOUT_BATCH_SIZE=2
LR=1e-5
MAX_STEPS=10
SEED=42

# Teacher-QV 参数
BASELINE_TYPE=${1:-"student"}     # student | ce | group_mean | group_hier
NORM_BY_STD=${NORM_BY_STD:-False}
CLIP_VALUE=${CLIP_VALUE:-"null"}
STD_FLOOR=${STD_FLOOR:-1e-3}

# baseline_type='ce' 需要 full / topk logits；其它 baseline 不需要
if [ "${BASELINE_TYPE}" = "ce" ]; then
    FULL_LOGIT="${FULL_LOGIT:-True}"
    DISTILL_TOPK="${DISTILL_TOPK:-100}"
else
    FULL_LOGIT="${FULL_LOGIT:-False}"
    DISTILL_TOPK="${DISTILL_TOPK:-null}"
fi

# Reuse SDPO 的 reprompt 设置（teacher 仍用 self-distillation 那套数据流）
DONT_REPROMPT_ON_SELF_SUCCESS=True

MODEL_PATH="${OSS_ROOT}/base_models/Qwen3-8B"

SUFFIX=${2:-"qv_${BASELINE_TYPE}"}

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}
export N_GPUS_PER_NODE=1

export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARN
export WANDB_MODE=offline
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="LOCAL-TEACHER-QV-${BASELINE_TYPE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-steps${MAX_STEPS}-seed${SEED}-${MODEL_NAME}-${SUFFIX}"

ARGS="data.train_files=['${TRAIN_FILE}'] \
data.val_files=['${VAL_FILE}'] \
data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
trainer.nnodes=1 \
trainer.group_name=Teacher-QV-SmokeTest \
trainer.total_training_steps=$MAX_STEPS \
trainer.save_freq=-1 \
trainer.val_before_train=False \
seed=$SEED \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
actor_rollout_ref.rollout.val_kwargs.n=4 \
actor_rollout_ref.actor.policy_loss.loss_mode=teacher_qv \
actor_rollout_ref.actor.policy_loss.teacher_qv.baseline_type=${BASELINE_TYPE} \
actor_rollout_ref.actor.policy_loss.teacher_qv.norm_by_std=${NORM_BY_STD} \
actor_rollout_ref.actor.policy_loss.teacher_qv.clip_value=${CLIP_VALUE} \
actor_rollout_ref.actor.policy_loss.teacher_qv.std_floor=${STD_FLOOR} \
actor_rollout_ref.actor.self_distillation.full_logit_distillation=${FULL_LOGIT} \
actor_rollout_ref.actor.self_distillation.distillation_topk=${DISTILL_TOPK} \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONT_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
algorithm.rollout_correction.rollout_is=token"

echo "========================================================================="
echo "🧪 Teacher-QV Smoke Test"
echo "  baseline_type = ${BASELINE_TYPE}"
echo "  norm_by_std   = ${NORM_BY_STD}"
echo "  clip_value    = ${CLIP_VALUE}"
echo "  exp           = ${EXP_NAME}"
echo "========================================================================="

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
