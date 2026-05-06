#!/usr/bin/env bash
# =============================================================================
# TASD 清爽版参数化训练脚本（供 Nebula sweep 调用）
#
# 核心功能：
#   - reward_type: teacher_prob | teacher_log_prob
#   - entropy_gate: none | hard | soft
#   - clip_adv
# =============================================================================
set +xo pipefail

OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"

# ── 必需参数 ─────────────────────────────────────────────────────────────
: "${DATASET:?DATASET is not set}"
: "${REWARD_TYPE:?REWARD_TYPE is not set}"      # teacher_prob | teacher_log_prob | outcome
: "${ENTROPY_GATE:?ENTROPY_GATE is not set}"     # none | hard | hard_keep_reward | soft | top_k_agreement
: "${ENTROPY_GATE_RATIO:?ENTROPY_GATE_RATIO is not set}"  # hard gate 保留比例：1.0=原始 | 0.8=top80% | 0.5=top50%
: "${CLIP_ADV_VALUE:?CLIP_ADV_VALUE is not set}"
: "${MODEL_PATH:?MODEL_PATH is not set}"

# ── 可选参数（有默认值）─────────────────────────────────────────────────
LR="${LR:-1e-5}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.001}"
SEED="${SEED:-42}"
TEACHER_REG="${TEACHER_REG:-ema}"
TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.1}"
CLIP_ADV="${CLIP_ADV:-true}"
NORM_ADV_BY_STD="${NORM_ADV_BY_STD:-false}"
ADV_STD_FLOOR="${ADV_STD_FLOOR:-0.0}"  # std下界：0 | auto | float
ADV_ENTROPY_WEIGHT="${ADV_ENTROPY_WEIGHT:-none}"  # advantage 熵加权：none | teacher_conf | certainty_diff（纯加权，过滤由 ENTROPY_GATE 控制）
GROUP_MEAN_MODE="${GROUP_MEAN_MODE:-token}"  # group mean/std 统计粒度：token（原有，存在length bias）| seq（per-seq均值后统计，消除length bias）
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-10000}"  # DAPO 风格：不 clip 上界；用 0.2 可退回标准 PPO
DISTILL_TOPK="${DISTILL_TOPK:-100}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
# ── top_k_agreement gate 参数（ENTROPY_GATE=top_k_agreement 时生效）──────
# TOP_K_AGREEMENT_K: student 候选集大小（在该集合内取 teacher argmax 与 student 采样对比）
# TOP_K_AGREEMENT_EPS: 一条 response 全部 token 一致（全被过滤）时整条赋值的 gate 权重兜底
TOP_K_AGREEMENT_K="${TOP_K_AGREEMENT_K:-512}"
TOP_K_AGREEMENT_EPS="${TOP_K_AGREEMENT_EPS:-0.1}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"   # vLLM rollout 采样温度
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"   # 1.0=不启用重复惩罚（等效禁用），与tasd_v2保持一致
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-32}"   # 不用 filter_groups 时应等于 train_batch_size；用 filter_groups 时可设更大值（如64）
MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-8}"
INCLUDE_SUCCESSFUL_ROLLOUTS="${INCLUDE_SUCCESSFUL_ROLLOUTS:-True}"
# ── DAPO 动态采样配置 ────────────────────────────────────────────────
FILTER_GROUPS_ENABLE="${FILTER_GROUPS_ENABLE:-false}"  # 是否启用 filter_groups
FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-acc}"    # 过滤指标：acc / seq_reward / seq_final_reward
FILTER_GROUPS_MAX_GEN="${FILTER_GROUPS_MAX_GEN:-0}"    # 最大重采样次数，0=不限制

# ── 训练详情 dump 开关 ───────────────────────────────────────────────
DUMP_DETAILED="${DUMP_DETAILED:-true}"            # true=开启每 step 详细日志
DUMP_SAMPLE_SIZE="${DUMP_SAMPLE_SIZE:-4}"          # 每 step 采样条数
DUMP_TOP_K="${DUMP_TOP_K:-5}"                      # teacher top-K
DUMP_DISAGREEMENT="${DUMP_DISAGREEMENT:-true}"      # 额外 disagreement-only JSONL
DUMP_OSS_PATH="${DUMP_OSS_PATH:-${OSS_ROOT}/logs/rollout_details/${JOB_NAME:-tasd_simple}}"

# ── stdout/stderr 日志镜像到 OSS ────────────────────────────────────
STDOUT_LOG_DIR="${OSS_ROOT}/logs/stdout"
mkdir -p "${STDOUT_LOG_DIR}" 2>/dev/null || true
STDOUT_LOG_FILE="${STDOUT_LOG_DIR}/${JOB_NAME:-tasd_simple}_$(date +%Y%m%d_%H%M%S).log"

# ── 路径 ────────────────────────────────────────────────────────────────
train_data_path="${OSS_ROOT}/datasets/${DATASET}/train.parquet"
val_data_path="${OSS_ROOT}/datasets/${DATASET}/test.parquet"
model_path="${MODEL_PATH}"
save_path="${OSS_ROOT}/models/${JOB_NAME:-tasd_simple}"

# ── 环境 ────────────────────────────────────────────────────────────────
# 优先使用 git 仓库根目录作为 PYTHONPATH，确保加载最新代码（而非 train_package 中的旧代码）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if git -C "${SCRIPT_DIR}" rev-parse --show-toplevel &>/dev/null; then
    GIT_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
    export PYTHONPATH="${GIT_ROOT}:${PYTHONPATH:-}"
    echo "[PYTHONPATH] Using git root: ${GIT_ROOT}"
elif [ -d "${SCRIPT_DIR}/../../verl" ]; then
    # 如果脚本在 nebula_scripts/tasd_simple/ 下，向上两级是 git 根目录
    export PYTHONPATH="${SCRIPT_DIR}/../..:${PYTHONPATH:-}"
    echo "[PYTHONPATH] Using script relative path: ${SCRIPT_DIR}/../.."
else
    export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
    echo "[PYTHONPATH] Using pwd: $(pwd)"
fi
unset VLLM_ATTENTION_BACKEND
# VLLM_USE_V1 在 vLLM 0.8.4 不支持，移除避免警告
# export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARN
export WANDB_MODE=offline
export WANDB_ENTITY=oh-my-team
export SWANLAB_MODE=cloud
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal}"
export SWANLAB_LOG_DIR="${OSS_ROOT}/logs/swanlab_logs"
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0
# git 分支信息（由 submit 脚本传入，若未设置则尝试本地获取）
export GIT_BRANCH="${GIT_BRANCH:-$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')}"
export GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}"

pip install -e . --no-deps --no-build-isolation --quiet 2>/dev/null || true

mkdir -p "${SWANLAB_LOG_DIR}" 2>/dev/null || true

# 清理残留的 Ray session
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true
rm -rf /tmp/ray_session_* 2>/dev/null || true
rm -rf ~/.ray 2>/dev/null || true
sleep 3

# ── 配置预检 ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/validate_config.sh" || {
    echo "❌ 配置验证失败，终止实验"
    exit 1
}

echo "============================================"
echo "TASD 清爽版实验配置："
echo "  DATASET: ${DATASET}"
echo "  REWARD_TYPE: ${REWARD_TYPE}"
echo "  ENTROPY_GATE: ${ENTROPY_GATE}"
echo "  CLIP_ADV: ${CLIP_ADV}, VALUE: ${CLIP_ADV_VALUE}, NORM_BY_STD: ${NORM_ADV_BY_STD}"
echo "  CLIP_RATIO_HIGH: ${CLIP_RATIO_HIGH}, ENTROPY_COEFF: ${ENTROPY_COEFF}"
echo "  ROLLOUT_TEMPERATURE: ${ROLLOUT_TEMPERATURE}, DISTILL_TEMPERATURE: ${DISTILL_TEMPERATURE}"
echo "  ADV_ENTROPY_WEIGHT: ${ADV_ENTROPY_WEIGHT}"
echo "  FILTER_GROUPS: enable=${FILTER_GROUPS_ENABLE}, metric=${FILTER_GROUPS_METRIC}, max_gen=${FILTER_GROUPS_MAX_GEN}"
echo "  DISTILL_TOPK: ${DISTILL_TOPK}"
echo "  TOP_K_AGREEMENT_K: ${TOP_K_AGREEMENT_K}, TOP_K_AGREEMENT_EPS: ${TOP_K_AGREEMENT_EPS}"
echo "  SEED: ${SEED}"
echo "============================================"

python -m verl.trainer.main_ppo \
    --config-name tasd_simple \
    seed=${SEED} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.gen_batch_size=${GEN_BATCH_SIZE} \
    data.train_files="${train_data_path}" \
    data.val_files="${val_data_path}" \
    custom_reward_function.path="$(pwd)/verl/utils/reward_score/feedback/__init__.py" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH} \
    actor_rollout_ref.actor.data_loader_seed=${SEED} \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=${TEACHER_REG} \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=${TEACHER_UPDATE_RATE} \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.repetition_penalty=${REPETITION_PENALTY} \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE} \
    actor_rollout_ref.rollout.seed=${SEED} \
    algorithm.tasd.reward_type=${REWARD_TYPE} \
    algorithm.tasd.entropy_gate=${ENTROPY_GATE} \
    algorithm.tasd.entropy_gate_ratio=${ENTROPY_GATE_RATIO} \
    algorithm.tasd.distill_topk=${DISTILL_TOPK} \
    algorithm.tasd.distill_temperature=${DISTILL_TEMPERATURE} \
    algorithm.tasd.top_k_agreement_k=${TOP_K_AGREEMENT_K} \
    algorithm.tasd.top_k_agreement_eps=${TOP_K_AGREEMENT_EPS} \
    algorithm.tasd.norm_adv_by_std=${NORM_ADV_BY_STD} \
    algorithm.tasd.adv_std_floor=${ADV_STD_FLOOR} \
    algorithm.tasd.clip_adv=${CLIP_ADV} \
    algorithm.tasd.clip_adv_value=${CLIP_ADV_VALUE} \
    algorithm.tasd.adv_entropy_weight=${ADV_ENTROPY_WEIGHT} \
    algorithm.tasd.group_mean_mode=${GROUP_MEAN_MODE} \
    algorithm.tasd.use_self_as_teacher_on_success=${INCLUDE_SUCCESSFUL_ROLLOUTS} \
    algorithm.tasd.include_successful_rollouts=${INCLUDE_SUCCESSFUL_ROLLOUTS} \
    algorithm.tasd.success_reward_threshold=1.0 \
    algorithm.filter_groups.enable=${FILTER_GROUPS_ENABLE} \
    algorithm.filter_groups.metric=${FILTER_GROUPS_METRIC} \
    algorithm.filter_groups.max_num_gen_batches=${FILTER_GROUPS_MAX_GEN} \
    trainer.total_epochs=30 \
    trainer.total_training_steps=250 \
    trainer.save_freq=-1 \
    trainer.save_best_metric="val-core/sciknoweval/acc/mean@16" \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-True} \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME:-TASD_simple}" \
    trainer.experiment_name="${JOB_NAME:-tasd_simple}" \
    trainer.group_name="TASD-simple" \
    trainer.rollout_dump_detailed=${DUMP_DETAILED} \
    trainer.rollout_dump_sample_size=${DUMP_SAMPLE_SIZE} \
    trainer.rollout_dump_top_k=${DUMP_TOP_K} \
    trainer.rollout_dump_disagreement=${DUMP_DISAGREEMENT} \
    trainer.rollout_dump_oss_path="${DUMP_OSS_PATH}" \
    "trainer.logger=[console,swanlab]" 2>&1 | tee -a "${STDOUT_LOG_FILE}"
