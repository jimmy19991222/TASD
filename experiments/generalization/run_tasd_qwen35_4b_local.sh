#!/bin/bash

# =============================================================================
# TASD 本地实验 - Qwen3.5-4B
# 
# 基于 run_tasd_ema_teacher_local.sh，针对 Qwen3.5-4B 优化
# 包含新的 advantage 配置：adv_std_floor
# =============================================================================

# ── 环境变量 ─────────────────────────────────────────────────
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

# ── Dry-run 模式 ───────────────────────────────────────────────
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
)

TRAIN_BATCH_SIZES=(32)
ROLLOUT_BATCH_SIZES=(8)
MINI_BATCH_SIZES=(32)
LRS=(1e-5)

# ── reward 类型 ─────────────────────────────────────────────
REWARD_TYPES=(
    "teacher_prob"
)

REWARD_TRANSFORMS=("none")
REWARD_SCALES=(1.0)
DISTILL_TOPKS=(100)

USE_SELF_AS_TEACHER_LIST=(True)
INCLUDE_SUCCESSFUL_ROLLOUTS_LIST=(True)

# ── advantage 配置 ──────────────────────────────────────────
# 新增 adv_std_floor：防止 group_std 过小导致 adv 爆炸
# norm_adv_by_std=True + adv_std_floor=0.1 → adv 量级约 ±10
NORM_ADV_BY_STD_LIST=(True)
ADV_STD_FLOOR_LIST=(0.1)       # std 下界
CLIP_ADV_LIST=(True)
CLIP_ADV_VALUES=(5.0)          # 放宽作为防御性兜底

# ── entropy bonus ─────────────────────────────────────────
ENTROPY_COEFF_LIST=(0.0 0.01)

# ── rollout correction ───────────────────────────────────
ROLLOUT_IS_LIST=("token")

# ── EMA teacher ─────────────────────────────────────────
TEACHER_REGULARIZATION_LIST=("ema")
TEACHER_UPDATE_RATE_LIST=(0.1)

# ── 模型配置 ─────────────────────────────────────────────
MODEL_PATHS=(
    "Qwen/Qwen3.5-4B"
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
    local run_cmd="CUDA_VISIBLE_DEVICES=0,1,2,3 bash /home/loujieming.ljm/SDPO/training/verl_training.sh '$exp_name' '$CONFIG_NAME' '$data_path' $args_string 2>&1 | tee -a $log_file"
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
                                        for USE_SELF_AS_TEACHER in "${USE_SELF_AS_TEACHER_LIST[@]}"; do
                                            for INCLUDE_SUCCESSFUL_ROLLOUTS in "${INCLUDE_SUCCESSFUL_ROLLOUTS_LIST[@]}"; do
                                                for NORM_ADV_BY_STD in "${NORM_ADV_BY_STD_LIST[@]}"; do
                                                    for ADV_STD_FLOOR in "${ADV_STD_FLOOR_LIST[@]}"; do
                                                        for CLIP_ADV in "${CLIP_ADV_LIST[@]}"; do
                                                            for CLIP_ADV_VALUE in "${CLIP_ADV_VALUES[@]}"; do
                                                                for ENTROPY_COEFF in "${ENTROPY_COEFF_LIST[@]}"; do
                                                                    for ROLLOUT_IS in "${ROLLOUT_IS_LIST[@]}"; do
                                                                        for TEACHER_REG in "${TEACHER_REGULARIZATION_LIST[@]}"; do
                                                                            for TEACHER_UPDATE_RATE in "${TEACHER_UPDATE_RATE_LIST[@]}"; do

                                                MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
                                                DATASET_NAME=$(echo "$DATA_PATH" \
                                                    | sed 's|datasets/||' \
                                                    | tr '/' '-'          \
                                                    | sed 's|-*$||')

                                                # 构建实验名：包含 advantage 配置
                                                NORM_TAG="nostd"
                                                if [ "$NORM_ADV_BY_STD" = "True" ]; then
                                                    if [ "$ADV_STD_FLOOR" != "0.0" ] && [ "$ADV_STD_FLOOR" != "0" ]; then
                                                        NORM_TAG="std-floor${ADV_STD_FLOOR}"
                                                    else
                                                        NORM_TAG="std"
                                                    fi
                                                fi
                                                CLIP_TAG="noclip"
                                                if [ "$CLIP_ADV" = "True" ]; then
                                                    CLIP_TAG="clip${CLIP_ADV_VALUE}"
                                                fi
                                                ENT_TAG=""
                                                if [ "$(echo "$ENTROPY_COEFF > 0" | bc -l)" = "1" ]; then
                                                    ENT_TAG="-ent${ENTROPY_COEFF}"
                                                fi
                                                EMA_TAG=""
                                                if [ "$TEACHER_REG" = "ema" ]; then
                                                    EMA_TAG="-ema${TEACHER_UPDATE_RATE}"
                                                fi

                                                EXP_NAME="TASD-${DATASET_NAME}-lr${LR}-rt${REWARD_TYPE}-${NORM_TAG}-${CLIP_TAG}${ENT_TAG}${EMA_TAG}-${MODEL_NAME}-$(date +%Y-%m-%d_%H-%M-%S)"

                                                SCRIPT_ARGS=(
                                                    # ── 基础参数 ──────────────────────────
                                                    "data.train_batch_size=$TRAIN_BATCH_SIZE"
                                                    "trainer.total_epochs=30"
                                                    "trainer.total_training_steps=250"
                                                    "trainer.group_name=TASD-Qwen3.5-4B"
                                                    "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
                                                    "actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE"
                                                    "actor_rollout_ref.actor.optim.lr=$LR"
                                                    "actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE"
                                                    "actor_rollout_ref.model.path=$MODEL_PATH"
                                                    "actor_rollout_ref.rollout.val_kwargs.n=16"

                                                    # ── TASD reward 参数 ──────────────────
                                                    "algorithm.tasd.reward_type=$REWARD_TYPE"
                                                    "algorithm.tasd.reward_transform=$REWARD_TRANSFORM"
                                                    "algorithm.tasd.reward_scale=$REWARD_SCALE"
                                                    "algorithm.tasd.distill_topk=$DISTILL_TOPK"
                                                    "algorithm.tasd.use_self_as_teacher_on_success=$USE_SELF_AS_TEACHER"
                                                    "algorithm.tasd.include_successful_rollouts=$INCLUDE_SUCCESSFUL_ROLLOUTS"
                                                    "algorithm.tasd.success_reward_threshold=1.0"

                                                    # ── TASD advantage 参数 ───────────────
                                                    "algorithm.tasd.norm_adv_by_std=$NORM_ADV_BY_STD"
                                                    "algorithm.tasd.adv_std_floor=$ADV_STD_FLOOR"
                                                    "algorithm.tasd.clip_adv=$CLIP_ADV"
                                                    "algorithm.tasd.clip_adv_value=$CLIP_ADV_VALUE"

                                                    # ── entropy bonus ─────────────────────
                                                    "actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF"

                                                    # ── rollout correction ────────────────
                                                    "algorithm.rollout_correction.rollout_is=$ROLLOUT_IS"

                                                    # ── EMA teacher ───────────────────────
                                                    "actor_rollout_ref.actor.self_distillation.teacher_regularization=$TEACHER_REG"
                                                    "actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE"

                                                    # ── teacher_input_ids 构建 ──────────────
                                                    "actor_rollout_ref.actor.self_distillation.include_environment_feedback=False"

                                                    # ── 本地运行参数 ───────────────────────
                                                    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
                                                    "actor_rollout_ref.rollout.gpu_memory_utilization=0.85"
                                                    "trainer.n_gpus_per_node=4"
                                                    "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16"
                                                )

                                                submit_job "$EXP_NAME" "$DATA_PATH" "${SCRIPT_ARGS[@]}"

                                                                            done  # TEACHER_UPDATE_RATE
                                                                        done  # TEACHER_REG
                                                                    done  # ROLLOUT_IS
                                                                done  # ENTROPY_COEFF
                                                            done  # CLIP_ADV_VALUE
                                                        done  # CLIP_ADV
                                                    done  # ADV_STD_FLOOR
                                                done  # NORM_ADV_BY_STD
                                            done  # INCLUDE_SUCCESSFUL_ROLLOUTS
                                        done  # USE_SELF_AS_TEACHER
                                    done  # DISTILL_TOPK
                                done  # REWARD_SCALE
                            done  # REWARD_TRANSFORM
                        done  # REWARD_TYPE
                    done  # MINI_BATCH_SIZE
                done  # MODEL_PATH
            done  # LR
        done  # ROLLOUT_BATCH_SIZE
    done  # TRAIN_BATCH_SIZE
done  # DATA_PATH
