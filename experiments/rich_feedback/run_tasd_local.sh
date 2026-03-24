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
CONFIG_NAME="tasd"

DATA_PATHS=(
    "datasets/lcb_v6"
)

TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
MINI_BATCH_SIZES=(32)
LRS=(1e-5 5e-6)

# ── TASD 超参扫描 ──────────────────────────────────────────────
REWARD_TYPES=("teacher_prob" "log_teacher_prob")
DISTILL_TOPKS=(100)
INCLUDE_SUCCESSFUL_ROLLOUTSS=(True)

# ── advantage 配置 ─────────────────────────────────────────────
NORM_ADV_BY_STD_LIST=(False)
CLIP_ADV_LIST=(True)
CLIP_ADV_VALUES=(5.0)

# ── entropy bonus ──────────────────────────────────────────────
ENTROPY_COEFF_LIST=(0.0 0.01)

MODEL_PATHS=(
    "Qwen/Qwen3-8B"
)

# =============================================================================
# H20 GPU 参数
# =============================================================================
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
        echo "  CONFIG     : $CONFIG_NAME"
        echo "  GPU        : $GPU_TYPE"
        echo "  EPOCHS     : $TOTAL_EPOCHS"
        echo "  LOG FILE   : $log_file"
        echo "  COMMAND    : $full_command"
    else
        echo "========================================================"
        echo "Starting job : $exp_name"
        echo "Dataset      : $dataset_name"
        echo "Config       : $CONFIG_NAME"
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
# DATA_PATH在外层：先把一个数据集所有setting跑完，再换下一个
# =============================================================================
for DATA_PATH in "${DATA_PATHS[@]}"; do
    for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
        for ROLLOUT_BATCH_SIZE in "${ROLLOUT_BATCH_SIZES[@]}"; do
            for LR in "${LRS[@]}"; do
                for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                    for MINI_BATCH_SIZE in "${MINI_BATCH_SIZES[@]}"; do
                        for REWARD_TYPE in "${REWARD_TYPES[@]}"; do
                            for DISTILL_TOPK in "${DISTILL_TOPKS[@]}"; do
                                for INCLUDE_SUCCESSFUL_ROLLOUTS in "${INCLUDE_SUCCESSFUL_ROLLOUTSS[@]}"; do
                                    for NORM_ADV_BY_STD in "${NORM_ADV_BY_STD_LIST[@]}"; do
                                        for CLIP_ADV in "${CLIP_ADV_LIST[@]}"; do
                                            for CLIP_ADV_VALUE in "${CLIP_ADV_VALUES[@]}"; do
                                                for ENTROPY_COEFF in "${ENTROPY_COEFF_LIST[@]}"; do

                                        MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
                                        DATASET_NAME=$(echo "$DATA_PATH" \
                                            | sed 's|datasets/||' \
                                            | tr '/' '-'          \
                                            | sed 's|-*$||')

                                        ENT_TAG=""
                                        if [ "$(echo "$ENTROPY_COEFF > 0" | bc -l)" = "1" ]; then
                                            ENT_TAG="-ent${ENTROPY_COEFF}"
                                        fi

                                        EXP_NAME="TASD-${GPU_TYPE}-${DATASET_NAME}-mbs${MINI_BATCH_SIZE}-lr${LR}-rt${REWARD_TYPE}-nostd-clip${CLIP_ADV_VALUE}${ENT_TAG}-${MODEL_NAME}-$(date +%Y-%m-%d_%H-%M-%S)"

                                        SCRIPT_ARGS=(
                                            # ── 基础参数（与grpo对齐）────────────────────
                                            "data.train_batch_size=$TRAIN_BATCH_SIZE"
                                            "trainer.group_name=TASD-rich-feedback-${MODEL_NAME}"
                                            "actor_rollout_ref.actor.optim.lr_warmup_steps=0"
                                            "actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE"
                                            "actor_rollout_ref.actor.optim.lr=$LR"
                                            "actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE"
                                            "actor_rollout_ref.model.path=$MODEL_PATH"
                                            "algorithm.rollout_correction.rollout_is=token"
                                            "actor_rollout_ref.rollout.val_kwargs.n=4"
                                            "trainer.n_gpus_per_node=4"
                                            "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"

                                            # ── TASD reward 参数 ───────────────────────
                                            "algorithm.tasd.reward_type=$REWARD_TYPE"
                                            "algorithm.tasd.distill_topk=$DISTILL_TOPK"
                                            "algorithm.tasd.include_successful_rollouts=$INCLUDE_SUCCESSFUL_ROLLOUTS"
                                            "algorithm.tasd.use_self_as_teacher_on_success=True"
                                            "algorithm.tasd.success_reward_threshold=1.0"

                                            # ── TASD advantage 参数 ──────────────────────
                                            "algorithm.tasd.norm_adv_by_std=$NORM_ADV_BY_STD"
                                            "algorithm.tasd.clip_adv=$CLIP_ADV"
                                            "algorithm.tasd.clip_adv_value=$CLIP_ADV_VALUE"

                                            # ── entropy bonus ────────────────────────────
                                            "actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF"

                                            # ── teacher_input_ids构建 ──────────────────
                                            "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"

                                            # ── H20 专项参数（与grpo完全相同）────────────
                                            "trainer.total_epochs=$TOTAL_EPOCHS"
                                            "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL"
                                            "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_BATCHED_TOKENS"
                                            "actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL"
                                        )

                                        submit_job "$EXP_NAME" "$DATA_PATH" "${SCRIPT_ARGS[@]}"

                                                done  # ENTROPY_COEFF
                                            done      # CLIP_ADV_VALUE
                                        done          # CLIP_ADV
                                    done              # NORM_ADV_BY_STD
                                done                  # INCLUDE_SUCCESSFUL_ROLLOUTS
                            done                      # DISTILL_TOPK
                        done                          # REWARD_TYPE
                    done                              # MINI_BATCH_SIZE
                done                                  # MODEL_PATH
            done                                      # LR
        done                                          # ROLLOUT_BATCH_SIZE
    done                                              # TRAIN_BATCH_SIZE
done                                                  # DATA_PATH
