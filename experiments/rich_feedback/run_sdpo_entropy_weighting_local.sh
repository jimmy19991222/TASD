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
CONFIG_NAME="sdpo"

DATA_PATHS=(
    "datasets/lcb_v6"
)

TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
LRS=(1e-6)
DONTS_REPROMPT_ON_SELF_SUCCESSS=(True)
ALPHAS=(1.0)
LAMBDAS=(0.0)
CLIP_ADV_HIGHS=(null)

# ── 新增：entropy weighting 配置 ──────────────────────────────
ENTROPY_WEIGHTINGS=(True)      # 对比实验可改成 (True False)
ENTROPY_TEMPERATURES=(0.5 1.0 2.0)     # 对比实验可改成 (0.5 1.0 2.0)

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
        echo "  EXP_NAME        : $exp_name"
        echo "  DATASET         : $dataset_name"
        echo "  CONFIG          : $CONFIG_NAME"
        echo "  GPU             : $GPU_TYPE"
        echo "  EPOCHS          : $TOTAL_EPOCHS"
        echo "  LOG FILE        : $log_file"
        echo "  COMMAND         : $full_command"
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
# =============================================================================
for TRAIN_BATCH_SIZE in "${TRAIN_BATCH_SIZES[@]}"; do
    for ROLLOUT_BATCH_SIZE in "${ROLLOUT_BATCH_SIZES[@]}"; do
        for LR in "${LRS[@]}"; do
            for DONTS_REPROMPT_ON_SELF_SUCCESS in "${DONTS_REPROMPT_ON_SELF_SUCCESSS[@]}"; do
                for MODEL_PATH in "${MODEL_PATHS[@]}"; do
                    for ALPHA in "${ALPHAS[@]}"; do
                        for LAMBDA in "${LAMBDAS[@]}"; do
                            for CLIP_ADV_HIGH in "${CLIP_ADV_HIGHS[@]}"; do
                                for ENTROPY_WEIGHTING in "${ENTROPY_WEIGHTINGS[@]}"; do        # ← 新增
                                    for ENTROPY_TEMPERATURE in "${ENTROPY_TEMPERATURES[@]}"; do # ← 新增
                                        for DATA_PATH in "${DATA_PATHS[@]}"; do

                                            MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
                                            DATASET_NAME=$(echo "$DATA_PATH" \
                                                | sed 's|datasets/||' \
                                                | tr '/' '-'          \
                                                | sed 's|-*$||')

                                            # EXP_NAME 加入 ew 和 et 标识方便区分实验
                                            EXP_NAME="FINAL-SDPO-${GPU_TYPE}-${DATASET_NAME}-train${TRAIN_BATCH_SIZE}-alpha${ALPHA}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-lambda${LAMBDA}-clip${CLIP_ADV_HIGH}-dross${DONTS_REPROMPT_ON_SELF_SUCCESS}-ew${ENTROPY_WEIGHTING}-et${ENTROPY_TEMPERATURE}-${MODEL_NAME}-$(date +%Y-%m-%d_%H-%M-%S)"

                                            SCRIPT_ARGS=(
                                                # ── 基础参数 ──────────────────────────────────
                                                "data.train_batch_size=$TRAIN_BATCH_SIZE"
                                                "trainer.group_name=SDPO-rich-feedback-${MODEL_NAME}"
                                                "actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE"
                                                "actor_rollout_ref.model.path=$MODEL_PATH"
                                                "actor_rollout_ref.actor.optim.lr=$LR"
                                                "actor_rollout_ref.actor.optim.lr_warmup_steps=0"
                                                "actor_rollout_ref.rollout.val_kwargs.n=4"
                                                "trainer.n_gpus_per_node=4"
                                                "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"

                                                # ── SDPO 特有参数 ──────────────────────────────
                                                "actor_rollout_ref.actor.ppo_mini_batch_size=1"
                                                "actor_rollout_ref.actor.self_distillation.distillation_topk=20"
                                                "actor_rollout_ref.actor.self_distillation.alpha=$ALPHA"
                                                "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=$DONTS_REPROMPT_ON_SELF_SUCCESS"
                                                "actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01"
                                                "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"
                                                "actor_rollout_ref.actor.self_distillation.entropy_weighting=$ENTROPY_WEIGHTING"       # ← 新增
                                                "actor_rollout_ref.actor.self_distillation.entropy_temperature=$ENTROPY_TEMPERATURE"   # ← 新增

                                                # ── IS correction ──────────────────────────────
                                                "algorithm.rollout_correction.rollout_is=token"

                                                # ── H20 专项参数 ───────────────────────────────
                                                "trainer.total_epochs=$TOTAL_EPOCHS"
                                                "actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL"
                                                "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_BATCHED_TOKENS"
                                                "actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_PARALLEL"
                                            )

                                            submit_job "$EXP_NAME" "$DATA_PATH" "${SCRIPT_ARGS[@]}"

                                        done
                                    done  # ← 新增
                                done      # ← 新增
                            done
                        done
                    done
                done
            done
        done
    done
done
