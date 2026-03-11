#!/bin/bash

# 设置 HF 镜像和其他环境变量
export HF_ENDPOINT=https://hf-mirror.com
export _NEBULA_USER_ID=435371
export VLLM_LOGGING_LEVEL=WARN
export WANDB_MODE=offline

export TORCH_NCCL_AVOID_RECORD_STREAMS=1

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
    local run_cmd="CUDA_VISIBLE_DEVICES=0,1,2,3 bash /home/loujieming.ljm/SDPO/training/verl_training.sh '$exp_name' '$CONFIG_NAME' '$data_path' $args_string > $log_file 2>&1"

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

                        EXP_NAME="FINAL-GRPO-${DATASET_NAME}-mbs-${MINI_BATCH_SIZE}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-model${MODEL_NAME}"

                        SCRIPT_ARGS=(
                            "data.train_batch_size=$TRAIN_BATCH_SIZE"
                            "trainer.group_name=GRPO-generalization"
                            "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
                            "actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE"
                            "actor_rollout_ref.actor.optim.lr=$LR"
                            "actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE"
                            "actor_rollout_ref.model.path=$MODEL_PATH"
                            "algorithm.rollout_correction.rollout_is=token"
                            "actor_rollout_ref.rollout.val_kwargs.n=4"
                            "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
                            "actor_rollout_ref.rollout.gpu_memory_utilization=0.85"
                            "actor_rollout_ref.rollout.max_num_batched_tokens=65536"
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
