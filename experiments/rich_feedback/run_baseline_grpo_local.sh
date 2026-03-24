# run_baseline_grpo_local.sh
#!/bin/bash

export HF_ENDPOINT=https://hf-mirror.com
export _NEBULA_USER_ID=435371
export WANDB_MODE=offline
export WANDB_ENTITY=oh-my-team
export WANDB_API_KEY="${WANDB_API_KEY:?WANDB_API_KEY not set}"
export WANDB_DIR=/home/loujieming.ljm/SDPO/logs/wandb_logs
export VLLM_LOGGING_LEVEL=WARN
export TENSORBOARD_DIR=/home/loujieming.ljm/SDPO/logs/tensorboard_logs
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY="${SWANLAB_API_KEY:?SWANLAB_API_KEY not set}"
export SWANLAB_LOG_DIR=/home/loujieming.ljm/SDPO/logs/swanlab_logs
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0

export CUDA_VISIBLE_DEVICES=0,1,2,3

DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run mode enabled. Commands will be printed but not executed."
fi

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG_NAME="baseline_grpo"

DATA_PATHS=(
    "datasets/lcb_v6"
)

TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
MINI_BATCH_SIZES=(8 32)       # off-policy / on-policy
LRS=(1e-5 1e-6)
MODEL_PATHS=(
    "Qwen/Qwen3-8B"
)

GPU_TYPE="H20"
GPU_MEMORY_UTIL=0.85
MAX_BATCHED_TOKENS=65536
TENSOR_PARALLEL=1
TOTAL_EPOCHS=5

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
        | sed 's|datasets/||' \
        | tr '/' '-'          \
        | sed 's|-*$||')

    local args_string=""
    for arg in "${script_args[@]}"; do
        args_string+=" $arg"
    done

    local setup_cmds="export PYTHONPATH=/home/loujieming.ljm/SDPO:\$PYTHONPATH"

    mkdir -p ./logs
    local log_file="./logs/job_${exp_name}_${dataset_name}_$(date +%Y-%m-%d_%H-%M-%S).log"
    local run_cmd="CUDA_VISIBLE_DEVICES=0,1,2,3 bash /home/loujieming.ljm/SDPO/training/verl_training.sh '$exp_name' '$CONFIG_NAME' '$data_path' $args_string 2>&1 | tee -a $log_file"
    local full_command="bash -c '$setup_cmds; $run_cmd'"

    if [ "$DRY_RUN" = true ]; then
        echo "--------------------------------------------------------------"
        echo "Dry-run would start process:"
        echo "  EXP_NAME   : $exp_name"
        echo "  DATASET    : $dataset_name"
        echo "  GPU        : $GPU_TYPE"
        echo "  EPOCHS     : $TOTAL_EPOCHS"
        echo "  LOG FILE   : $log_file"
        echo "  COMMAND    : $full_command"
    else
        echo "========================================================"
        echo "Starting job : $exp_name"
        echo "Dataset      : $dataset_name"
        echo "GPU          : $GPU_TYPE"
        echo "Epochs       : $TOTAL_EPOCHS"
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
# =============================================================================
for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    for ROLLOUT_BATCH_SIZE in "${ROLLOUT_BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do
            for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                for MINI_BATCH_SIZE in "${MINI_BATCH_SIZES[@]}"; do
                    for DATA_PATH in "${DATA_PATHS[@]}"; do

                        MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
                        DATASET_NAME=$(echo "$DATA_PATH" \
                            | sed 's|datasets/||' \
                            | tr '/' '-'          \
                            | sed 's|-*$||')

                        EXP_NAME="FINAL-GRPO-${GPU_TYPE}-${DATASET_NAME}-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-model${MODEL_NAME}-$(date +%Y-%m-%d_%H-%M-%S)"

                        SCRIPT_ARGS=(
                            # ── 基础参数 ──────────────────────────────────
                            "data.train_batch_size=$TRAIN_BATCH_SIZE"
                            "trainer.group_name=GRPO-rich-feedback-${MODEL_NAME}"
                            "actor_rollout_ref.actor.optim.lr_warmup_steps=0"
                            "actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE"
                            "actor_rollout_ref.actor.optim.lr=$LR"
                            "actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE"
                            "actor_rollout_ref.model.path=$MODEL_PATH"
                            "algorithm.rollout_correction.rollout_is=token"
                            "actor_rollout_ref.rollout.val_kwargs.n=4"
                            "trainer.n_gpus_per_node=4"
                            "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"

                            # ── H20 专项参数 ───────────────────────────────
                            "trainer.total_epochs=$TOTAL_EPOCHS"
                            "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL"
                            "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_BATCHED_TOKENS"
                            "actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL"
                        )

                        submit_job "$EXP_NAME" "$DATA_PATH" "${SCRIPT_ARGS[@]}"

                    done
                done
            done
        done
    done
done
