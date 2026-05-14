#!/bin/bash

# Usage: ./run_local_rlsd.sh [experiment_name_suffix]
# RLSD baseline (arXiv:2604.03128v2) on Bayesian Credit Assignment branch.

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="rlsd"

# Default to ToolUse dataset
DATA_PATH="datasets/tooluse"

# Hyperparameters
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
MINI_BATCH_SIZE=32
LR=1e-5

# RLSD-specific
EPS_W=0.2
TEACHER_UPDATE_RATE=0.05
TEACHER_REGULARIZATION=ema   # ema | hard_sync (TODO)

MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
export N_GPUS_PER_NODE=1

# Allow overriding experiment name suffix
SUFFIX=${1:-"local_rlsd"}

# =============================================================================
# SETUP
# =============================================================================

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}

# =============================================================================
# EXECUTION
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="LOCAL-RLSD-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-epsW${EPS_W}-tch${TEACHER_REGULARIZATION}${TEACHER_UPDATE_RATE}-${MODEL_NAME}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=RLSD-local \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.actor.self_distillation.teacher_regularization=$TEACHER_REGULARIZATION \
actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE \
actor_rollout_ref.model.path=$MODEL_PATH \
algorithm.rlsd.eps_w=$EPS_W \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=16"

echo "----------------------------------------------------------------"
echo "Starting Local RLSD Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "RLSD: eps_w=$EPS_W, teacher=$TEACHER_REGULARIZATION (rate=$TEACHER_UPDATE_RATE)"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
