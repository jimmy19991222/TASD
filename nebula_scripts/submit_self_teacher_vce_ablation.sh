#!/bin/bash
# =============================================================================
# Self-Teacher Advantage V_CE 消融实验 - Nebula 批量提交脚本
#
# 核心思想：
#   - 去除 V_EMA（已实验证伪：beta=0.5/0.7/1.0 熵崩溃速度完全一致）
#   - 改为 group-level z-score（seq 模式）修正 top-k 截断偏置
#   - post-normalization clip（±3σ）
#
# 实验设计（四组）：
#   A: use_vce=true,     use_log_pi_s=false, clip=3.0  → Q - V_CE（主实验）
#   B: use_vce=true,     use_log_pi_s=false, clip=100  → Q - V_CE + noclip
#   C: use_vce=false,    use_log_pi_s=false, clip=3.0  → Q only（消融基线，clip 与 A 对齐）
#   D: use_log_pi_s=true, use_vce=false,     clip=3.0  → Q - logπ_s（SDPO-style，熵保护）
#
# 使用方式：
#   bash nebula_scripts/submit_self_teacher_vce_ablation.sh [--dry-run]
# =============================================================================

# ── Nebula 账号配置 ──────────────────────────────────────────────────────
QUEUE="lazada_llm_ad_h20"
WORLD_SIZE=1
OPENLM_TOKEN="${OPENLM_TOKEN:?OPENLM_TOKEN not set}"
OSS_ACCESS_ID="${OSS_ACCESS_ID:?OSS_ACCESS_ID not set}"
OSS_ACCESS_KEY="${OSS_ACCESS_KEY:?OSS_ACCESS_KEY not set}"
OSS_ENDPOINT="oss-cn-hangzhou-zmf.aliyuncs.com"
OSS_BUCKET="lazada-ai-model"
CLUSTER_FILE="nebula_scripts/cluster.json"
SCRIPT_PATH="nebula_scripts/tasd_simple/tasd_simple_parametric.sh"
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="Self-Teacher-VCE-Ablation"

# ── 数据集配置 ──────────────────────────────────────────────────────
DATASETS=(
    "sciknoweval/biology"
    # "sciknoweval/chemistry"
)

# ── dry-run 模式 ─────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 固定超参配置
# =============================================================================

# ── Self-Teacher 核心参数 ──────────────────────────────────────────────
REWARD_TYPE="teacher_log_prob"
ENTROPY_GATE="none"
ENTROPY_GATE_RATIO="1.0"
DISTILL_TOPK="256"
DISTILL_TEMPERATURE="1.0"

# ── Normalization 参数 ────────────────────────────────────────────────
NORM_ADV_BY_STD="true"         # group-level z-score（seq 模式）
ADV_STD_FLOOR="0.0"
CLIP_VALUE="3.0"               # post-normalization clip（≈±3σ）
GROUP_MEAN_MODE="seq"          # seq 模式：消除 length bias

# ── GRPO 模式兼容参数（self_teacher 模式不使用，但脚本必传）─────────────
CLIP_ADV="true"
CLIP_ADV_VALUE="2.0"
ADV_ENTROPY_WEIGHT="none"

# ── 其他固定参数 ─────────────────────────────────────────────────────
CLIP_RATIO_HIGH="0.28"
REPETITION_PENALTY="1.05"
ROLLOUT_N="8"
ROLLOUT_TEMPERATURE="1.0"
TRAIN_BATCH_SIZE="32"
GEN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
INCLUDE_SUCCESSFUL_ROLLOUTS="True"
TEACHER_REG="ema"
TEACHER_UPDATE_RATE="0.05"
LR="1e-5"
ENTROPY_COEFF="0.001"
SEED="42"
FILTER_GROUPS_ENABLE="false"
FILTER_GROUPS_METRIC="acc"
FILTER_GROUPS_MAX_GEN="0"

# ── 模型路径 ─────────────────────────────────────────────────────────
MODEL_PATH="${OSS_ROOT:-/data/oss_bucket_0/ad/loujieming.ljm}/models/Qwen3-8B"

# =============================================================================
# 实验矩阵
# =============================================================================
# 格式: DATASET|ADV_MODE|USE_VCE|USE_LOG_PI_S|CLIP_VALUE|EXP_LABEL
EXPERIMENTS=()

for DATASET in "${DATASETS[@]}"; do
    DATASET_NAME=$(echo "$DATASET" | sed 's|/|_|g')

    # ── use_vce=true：clip 消融（3.0 / no_clip）────────────────────
    EXPERIMENTS+=("${DATASET}|self_teacher|true|false|3.0|vce_clip3")
    EXPERIMENTS+=("${DATASET}|self_teacher|true|false|100|vce_noclip")  # 100 >> N(0,1)，等效无截断

    # ── use_vce=false：Q only 消融基线（clip=3.0，与 vce_clip3 对齐）───────────────────
    EXPERIMENTS+=("${DATASET}|self_teacher|false|false|3.0|q_only_clip3")

    # ── use_log_pi_s=true：SDPO-style，隐式熵保护（固定 clip=3.0）─────────────────────────────
    EXPERIMENTS+=("${DATASET}|self_teacher|false|true|3.0|q_logpis_clip3")
done

# =============================================================================
# 提交实验
# =============================================================================

echo "============================================"
echo "Self-Teacher VCE 消融实验提交"
echo "PROJECT_NAME: ${PROJECT_NAME}"
echo "============================================"
echo "实验数量: ${#EXPERIMENTS[@]}"
for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r _DS _MODE _VCE _LOGPIS _CLIP _LABEL <<< "$EXP"
    echo "  - self_teacher_${_LABEL} [dataset=${_DS}, adv_mode=${_MODE}, use_vce=${_VCE}, use_log_pi_s=${_LOGPIS}, clip=${_CLIP}]"
done
echo "============================================"

TASK_IDS=()
FAILED=0

for EXP in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r EXP_DATASET EXP_ADV_MODE EXP_USE_VCE EXP_USE_LOG_PI_S EXP_CLIP_VALUE EXP_LABEL <<< "$EXP"

    JOB_NAME="self_teacher_${EXP_LABEL}"

    echo ""
    echo "────────────────────────────────────────"
    echo "提交实验: ${JOB_NAME}"
    echo "  DATASET:    ${EXP_DATASET}"
    echo "  ADV_MODE:   ${EXP_ADV_MODE}"
    if [ "${EXP_ADV_MODE}" = "self_teacher" ]; then
        echo "  USE_VCE:         ${EXP_USE_VCE}"
        echo "  USE_LOG_PI_S:    ${EXP_USE_LOG_PI_S}"
        echo "  CLIP_VALUE: ${EXP_CLIP_VALUE}, NORM_BY_STD: ${NORM_ADV_BY_STD}"
        echo "  GROUP_MEAN_MODE: ${GROUP_MEAN_MODE}"
    fi
    echo "  REWARD_TYPE: ${REWARD_TYPE}"
    echo "  DISTILL_TOPK: ${DISTILL_TOPK}"

    # 预计算所有变量（参考 submit_self_teacher_advantage.sh 的 export 模式）
    export DATASET="$EXP_DATASET"
    export ADV_MODE="$EXP_ADV_MODE"
    export USE_VCE="$EXP_USE_VCE"
    export USE_LOG_PI_S="$EXP_USE_LOG_PI_S"
    export CLIP_VALUE="$EXP_CLIP_VALUE"
    export REWARD_TYPE="$REWARD_TYPE"
    export ENTROPY_GATE="$ENTROPY_GATE"
    export ENTROPY_GATE_RATIO="$ENTROPY_GATE_RATIO"
    export DISTILL_TOPK="$DISTILL_TOPK"
    export DISTILL_TEMPERATURE="$DISTILL_TEMPERATURE"
    export CLIP_ADV="$CLIP_ADV"
    export CLIP_ADV_VALUE="$CLIP_ADV_VALUE"
    export NORM_ADV_BY_STD="$NORM_ADV_BY_STD"
    export ADV_STD_FLOOR="$ADV_STD_FLOOR"
    export ADV_ENTROPY_WEIGHT="$ADV_ENTROPY_WEIGHT"
    export GROUP_MEAN_MODE="$GROUP_MEAN_MODE"
    export CLIP_RATIO_HIGH="$CLIP_RATIO_HIGH"
    export REPETITION_PENALTY="$REPETITION_PENALTY"
    export ROLLOUT_N="$ROLLOUT_N"
    export ROLLOUT_TEMPERATURE="$ROLLOUT_TEMPERATURE"
    export TRAIN_BATCH_SIZE="$TRAIN_BATCH_SIZE"
    export GEN_BATCH_SIZE="$GEN_BATCH_SIZE"
    export MINI_BATCH_SIZE="$MINI_BATCH_SIZE"
    export INCLUDE_SUCCESSFUL_ROLLOUTS="$INCLUDE_SUCCESSFUL_ROLLOUTS"
    export TEACHER_REG="$TEACHER_REG"
    export TEACHER_UPDATE_RATE="$TEACHER_UPDATE_RATE"
    export LR="$LR"
    export ENTROPY_COEFF="$ENTROPY_COEFF"
    export SEED="$SEED"
    export FILTER_GROUPS_ENABLE="$FILTER_GROUPS_ENABLE"
    export FILTER_GROUPS_METRIC="$FILTER_GROUPS_METRIC"
    export FILTER_GROUPS_MAX_GEN="$FILTER_GROUPS_MAX_GEN"
    export MODEL_PATH="$MODEL_PATH"
    export JOB_NAME="$JOB_NAME"
    export PROJECT_NAME="$PROJECT_NAME"
    GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
    GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] 准备提交"
        echo "  JOB_NAME: ${JOB_NAME}"
        echo "  ADV_MODE: ${ADV_MODE}, USE_VCE: ${USE_VCE}, USE_LOG_PI_S: ${USE_LOG_PI_S}, CLIP_VALUE: ${CLIP_VALUE}"
    else
        echo "提交中..."
        SUBMIT_OUTPUT=$(nebulactl run mdl \
            --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=ADV_MODE=${ADV_MODE} --env=USE_VCE=${USE_VCE} --env=USE_LOG_PI_S=${USE_LOG_PI_S} --env=CLIP_VALUE=${CLIP_VALUE} --env=REWARD_TYPE=${REWARD_TYPE} --env=ENTROPY_GATE=${ENTROPY_GATE} --env=ENTROPY_GATE_RATIO=${ENTROPY_GATE_RATIO} --env=DISTILL_TOPK=${DISTILL_TOPK} --env=DISTILL_TEMPERATURE=${DISTILL_TEMPERATURE} --env=CLIP_ADV=${CLIP_ADV} --env=CLIP_ADV_VALUE=${CLIP_ADV_VALUE} --env=NORM_ADV_BY_STD=${NORM_ADV_BY_STD} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR} --env=ADV_ENTROPY_WEIGHT=${ADV_ENTROPY_WEIGHT} --env=GROUP_MEAN_MODE=${GROUP_MEAN_MODE} --env=CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH} --env=REPETITION_PENALTY=${REPETITION_PENALTY} --env=ROLLOUT_N=${ROLLOUT_N} --env=ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=GEN_BATCH_SIZE=${GEN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=INCLUDE_SUCCESSFUL_ROLLOUTS=${INCLUDE_SUCCESSFUL_ROLLOUTS} --env=TEACHER_REG=${TEACHER_REG} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=LR=${LR} --env=ENTROPY_COEFF=${ENTROPY_COEFF} --env=SEED=${SEED} --env=FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE} --env=FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC} --env=FILTER_GROUPS_MAX_GEN=${FILTER_GROUPS_MAX_GEN} --env=MODEL_PATH=${MODEL_PATH} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=f598ad33b071751bf79d2484d8e1acefe8df9d879e129cae40340a158854f9cb --env=DINGTALK_SECRET=SECc5b9e4f61f56b32b46abf1ecedc11bdcba10dc35fbba8fa0ff62c084a1cc6ad3" \
            --worker_count=${WORLD_SIZE} \
            --file.cluster_file=${CLUSTER_FILE} \
            --job_name=${JOB_NAME} \
            --env=OPENLM_TOKEN=${OPENLM_TOKEN} \
            --env=SWANLAB_API_KEY=${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal} \
            --custom_docker_image=${CUSTOM_DOCKER_IMAGE} \
            --requirements_file_name=requirements_nebula.txt \
            --oss_access_id=${OSS_ACCESS_ID} \
            --oss_access_key=${OSS_ACCESS_KEY} \
            --oss_bucket=${OSS_BUCKET} \
            --oss_endpoint=${OSS_ENDPOINT} \
            2>&1)
        SUBMIT_EXIT=$?
        echo "$SUBMIT_OUTPUT"
        if [ $SUBMIT_EXIT -ne 0 ]; then
            echo "❌ 提交失败 (exit code: $SUBMIT_EXIT)"
            FAILED=$((FAILED + 1))
        else
            TASK_ID=$(echo "$SUBMIT_OUTPUT" | grep -o 'task_id[":\s]*[a-f0-9]\+' | grep -o '[a-f0-9]\{20,\}' | head -1)
            if [ -n "$TASK_ID" ]; then
                echo "✅ 提交成功: task_id=$TASK_ID"
                TASK_IDS+=("$TASK_ID")
            else
                echo "✅ 提交成功（未提取到 task_id）"
            fi
        fi
    fi

    sleep 2
done

# =============================================================================
# 总结
# =============================================================================

echo ""
echo "============================================"
echo "提交完成"
echo "============================================"
echo "成功: $((${#EXPERIMENTS[@]} - FAILED))"
echo "失败: $FAILED"
echo ""

if [ ${#TASK_IDS[@]} -gt 0 ]; then
    echo "Task IDs:"
    for TID in "${TASK_IDS[@]}"; do
        echo "  - $TID"
    done
fi

echo "============================================"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
