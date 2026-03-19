#!/bin/bash

# ── 环境变量 ─────────────────────────────────────────────────
export HF_ENDPOINT=https://hf-mirror.com
export _NEBULA_USER_ID=435371
export WANDB_MODE=offline
export WANDB_ENTITY=oh-my-team
export WANDB_API_KEY=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
export WANDB_DIR=/home/loujieming.ljm/SDPO/logs/wandb_logs
export VLLM_LOGGING_LEVEL=WARN
export TENSORBOARD_DIR=/home/loujieming.ljm/SDPO/logs/tensorboard_logs
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY=M5oC00EEt8G1wC0XaHkal
export SWANLAB_LOG_DIR=/home/loujieming.ljm/SDPO/logs/swanlab_logs
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0
export CUDA_VISIBLE_DEVICES=4,5,6,7

# ── Dry-run模式 ───────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled. Commands will be printed but not executed."
fi

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG_NAME="tasd"

DATA_PATHS=(
    "datasets/sciknoweval/biology/"
    # "datasets/sciknoweval/chemistry/"
    # "datasets/sciknoweval/material/"
    # "datasets/sciknoweval/physics/"
    # "datasets/tooluse"
)

TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
MINI_BATCH_SIZES=(32)
LRS=(1e-5)

# ── TASD 超参扫描 ──────────────────────────────────────────────
REWARD_TYPES=("log_ratio")
REWARD_TRANSFORMS=("tanh01")
REWARD_SCALES=(1.0)
DISTILL_TOPKS=(100)
DONT_REPROMPT_ON_SELF_SUCCESS_LIST=(True)
INCLUDE_SUCCESSFUL_ROLLOUTS_LIST=(True False)

MODEL_PATHS=(
    "Qwen/Qwen3-8B"
    # "allenai/Olmo-3-7B-Instruct"
)

# =============================================================================
# JOB SUBMISSION FUNCTION
# =============================================================================
submit_job() {
    local exp_name="$1"
    local data_path="$2"
    shift 2
    local script_args=("$@")

    local dataset_name
    dataset_name=$(echo "$data_path" \
        | sed 's|datasets/||'  \
        | tr '/' '-'           \
        | sed 's|-*$||')

    local args_string=""
    for arg in "${script_args[@]}"; do
        args_string+=" $arg"
    done

    local setup_cmds="export PYTHONPATH=/home/loujieming.ljm/SDPO:\$PYTHONPATH"

    mkdir -p ./logs
    local log_file="./logs/job_${exp_name}_${dataset_name}_$(date +%Y-%m-%d_%H-%M-%S).log"
    local run_cmd="CUDA_VISIBLE_DEVICES=4,5,6,7 bash /home/loujieming.ljm/SDPO/training/verl_training.sh '$exp_name' '$CONFIG_NAME' '$data_path' $args_string 2>&1 | tee -a $log_file"
    local full_command="bash -c '$setup_cmds; $run_cmd'"

    if [ "$DRY_RUN" = true ]; then
        echo "--------------------------------------------------------------"
        echo "Dry-run would start process:"
        echo "  EXP_NAME : $exp_name"
        echo "  DATASET  : $dataset_name"
        echo "  CONFIG   : $CONFIG_NAME"
        echo "  LOG FILE : $log_file"
        echo "  COMMAND  : $full_command"
    else
        echo "========================================================"
        echo "Starting job : $exp_name"
        echo "Dataset      : $dataset_name"
        echo "Log          : $log_file"
        echo "========================================================"

        eval "$full_command"

        echo "========================================================"
        echo "Job finished : $exp_name"
        echo "Waiting 30s for GPU memory cleanup..."
        echo "========================================================"
        sleep 30
        nvidia-smi --query-gpu=memory.free,memory.total --format=csv
    fi
}

# =============================================================================
# MAIN LOOP
# 外层：DATA_PATH（先把一个数据集的所有setting跑完，再换下一个）
# 内层：所有超参组合
# =============================================================================
for DATA_PATH in "${DATA_PATHS[@]}"; do
    for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
        for ROLLOUT_BATCH_SIZE in "${ROLLOUT_BATCH_SIZES[@]}"; do
            for LR in "${LRS[@]}"; do
                for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                    for MINI_BATCH_SIZE in "${MINI_BATCH_SIZES[@]}"; do
                        for REWARD_TYPE in "${REWARD_TYPES[@]}"; do
                            for REWARD_TRANSFORM in "${REWARD_TRANSFORMS[@]}"; do
                                for REWARD_SCALE in "${REWARD_SCALES[@]}"; do
                                    for DISTILL_TOPK in "${DISTILL_TOPKS[@]}"; do
                                        for DONT_REPROMPT in "${DONT_REPROMPT_ON_SELF_SUCCESS_LIST[@]}"; do
                                            for INCLUDE_SUCCESSFUL_ROLLOUTS in "${INCLUDE_SUCCESSFUL_ROLLOUTS_LIST[@]}"; do

                                                MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
                                                DATASET_NAME=$(echo "$DATA_PATH" \
                                                    | sed 's|datasets/||' \
                                                    | tr '/' '-'          \
                                                    | sed 's|-*$||')

                                                EXP_NAME="TASD-${DATASET_NAME}-mbs${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-rt${REWARD_TYPE}-tf${REWARD_TRANSFORM}-topk${DISTILL_TOPK}-dross${DONT_REPROMPT}-isr${INCLUDE_SUCCESSFUL_ROLLOUTS}-${MODEL_NAME}-$(date +%Y-%m-%d_%H-%M-%S)"

                                                SCRIPT_ARGS=(
                                                    # ── 基础参数 ──────────────────────────
                                                    "data.train_batch_size=$TRAIN_BATCH_SIZE"
                                                    "trainer.total_epochs=30"
                                                    "trainer.total_training_steps=200"
                                                    "trainer.group_name=TASD-generalization"
                                                    "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
                                                    "actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE"
                                                    "actor_rollout_ref.actor.optim.lr=$LR"
                                                    "actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE"
                                                    "actor_rollout_ref.model.path=$MODEL_PATH"
                                                    "actor_rollout_ref.rollout.val_kwargs.n=16"

                                                    # ── TASD 特有参数 ──────────────────────
                                                    "algorithm.tasd.reward_type=$REWARD_TYPE"
                                                    "algorithm.tasd.reward_transform=$REWARD_TRANSFORM"
                                                    "algorithm.tasd.reward_scale=$REWARD_SCALE"
                                                    "algorithm.tasd.distill_topk=$DISTILL_TOPK"
                                                    "algorithm.tasd.include_successful_rollouts=$INCLUDE_SUCCESSFUL_ROLLOUTS"
                                                    "algorithm.tasd.success_reward_threshold=1.0"

                                                    # ── teacher_input_ids构建 ──────────────
                                                    "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=$DONT_REPROMPT"
                                                    "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"

                                                    # ── 本地运行参数 ───────────────────────
                                                    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
                                                    "actor_rollout_ref.rollout.gpu_memory_utilization=0.85"
                                                    "trainer.n_gpus_per_node=4"
                                                    "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"
                                                )

                                                submit_job "$EXP_NAME" "$DATA_PATH" "${SCRIPT_ARGS[@]}"

                                            done  # INCLUDE_SUCCESSFUL_ROLLOUTS
                                        done      # DONT_REPROMPT
                                    done          # DISTILL_TOPK
                                done              # REWARD_SCALE
                            done                  # REWARD_TRANSFORM
                        done                      # REWARD_TYPE
                    done                          # MINI_BATCH_SIZE
                done                              # MODEL_PATH
            done                                  # LR
        done                                      # ROLLOUT_BATCH_SIZE
    done                                          # TRAIN_BATCH_SIZE
done                                              # DATA_PATH
