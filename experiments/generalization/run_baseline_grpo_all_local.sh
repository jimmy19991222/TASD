#!/bin/bash

# 设置 HF 镜像和其他环境变量
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

# 在环境变量设置区域添加
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0

# GPU 设置
export CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS_PER_NODE=4

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
    "datasets/sciknoweval/biology/"
    "datasets/sciknoweval/chemistry/"
    "datasets/sciknoweval/material/"
    "datasets/sciknoweval/physics/"
    "datasets/tooluse"
)

TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
MINI_BATCH_SIZES=(32)

LRS=(1e-5)
MODEL_PATHS=(
    "Qwen/Qwen3-8B"
    # "allenai/Olmo-3-7B-Instruct"
)

# Job Submission Function
submit_job() {
    local exp_name="$1"
    local data_path="$2"
    shift 2
    local script_args=("$@")

    # =========================================================
    # 从 data_path 提取数据集名称，用于 log 文件名
    # 例如：datasets/sciknoweval/biology/ -> sciknoweval-biology
    #        datasets/tooluse             -> tooluse
    # =========================================================
    local dataset_name
    dataset_name=$(echo "$data_path" \
        | sed 's|datasets/||'  \
        | tr '/' '-'           \
        | sed 's|-*$||')       # 去掉末尾多余的 '-'

    local args_string=""
    for arg in "${script_args[@]}"; do
        args_string+=" $arg"
    done

    local setup_cmds="export PYTHONPATH=/home/loujieming.ljm/SDPO:\$PYTHONPATH"

    mkdir -p ./logs
    # log 文件名格式：job_<exp_name>_<dataset>_<timestamp>.log
    local log_file="./logs/job_${exp_name}_${dataset_name}_$(date +%Y-%m-%d_%H-%M-%S).log"
    # local run_cmd="CUDA_VISIBLE_DEVICES=0,1,2,3 bash /home/loujieming.ljm/SDPO/training/verl_training.sh '$exp_name' '$CONFIG_NAME' '$data_path' $args_string > $log_file 2>&1"
    local run_cmd="CUDA_VISIBLE_DEVICES=0,1,2,3 bash /home/loujieming.ljm/SDPO/training/verl_training.sh '$exp_name' '$CONFIG_NAME' '$data_path' $args_string 2>&1 | tee -a $log_file"
    
    local full_command="bash -c '$setup_cmds; $run_cmd'"

    if [ "$DRY_RUN" = true ]; then
        echo "--------------------------------------------------------------"
        echo "Dry-run would start process:"
        echo "  EXP_NAME   : $exp_name"
        echo "  DATASET    : $dataset_name"   # 新增：dry-run 时也打印数据集
        echo "  LOG FILE   : $log_file"
        echo "  COMMAND    : $full_command"
    else
        echo "========================================================"
        echo "Starting job : $exp_name"
        echo "Dataset      : $dataset_name"   # 新增：运行时打印数据集
        echo "Log          : $log_file"
        echo "========================================================"

        eval "$full_command"

        echo "========================================================"
        echo "Job finished: $exp_name"
        echo "Waiting 30s for GPU memory cleanup..."
        echo "========================================================"
        sleep 30
        nvidia-smi --query-gpu=memory.free,memory.total --format=csv
    fi
}

# Main Loop
for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    for ROLLOUT_BATCH_SIZE in "${ROLLOUT_BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do
            for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                for MINI_BATCH_SIZE in "${MINI_BATCH_SIZES[@]}"; do
                    for DATA_PATH in "${DATA_PATHS[@]}"; do

                        MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')

                        # =====================================================
                        # EXP_NAME 中也加入数据集信息，与 log 文件名对应
                        # =====================================================
                        DATASET_NAME=$(echo "$DATA_PATH" \
                            | sed 's|datasets/||' \
                            | tr '/' '-'          \
                            | sed 's|-*$||')

                        EXP_NAME="FINAL-GRPO-${DATASET_NAME}-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-model${MODEL_NAME}-$(date +%Y-%m-%d_%H-%M-%S)"

                        SCRIPT_ARGS=(
                            "data.train_batch_size=$TRAIN_BATCH_SIZE"
                            "trainer.total_epochs=30"
                            "trainer.total_training_steps=250"
                            "trainer.group_name=GRPO-generalization"
                            "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
                            "actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE"
                            "actor_rollout_ref.actor.optim.lr=$LR"
                            "actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE"
                            "actor_rollout_ref.model.path=$MODEL_PATH"
                            "algorithm.rollout_correction.rollout_is=token"
                            "actor_rollout_ref.rollout.val_kwargs.n=16"
                            "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
                            "actor_rollout_ref.rollout.gpu_memory_utilization=0.85"
                            "trainer.n_gpus_per_node=4"
                            "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"
                        )

                        submit_job "$EXP_NAME" "$DATA_PATH" "${SCRIPT_ARGS[@]}"

                    done
                done
            done
        done
    done
done
