#!/bin/bash
# =============================================================================
# TASD Tooluse Feedback Enhanced - Nebula 批量提交脚本
#
# 核心变化（对比基线 TASD）：
#   1. Feedback 细粒度增强：tooluse reward 从简单 mismatch 升级为三级反馈
#      - 格式错误（Format error）：缺失 Action/Action Input 字段
#      - Action 错误（Action error）：动作类型/数量/顺序不匹配
#      - Input 错误（Input error）：参数 key/value 不匹配
#   2. Teacher context 展示错误答案：feedback_template 新增 failed_attempt 段落
#      失败 rollout 的 teacher context 同时包含：错误答案 + 细粒度 feedback + 正确示范
#   3. 成功 rollout 固定看到自己：_get_solution 简化，成功 rollout 永远返回自己
#      不再随机选别人的答案，避免成功 rollout 学习错误示范
#
# 使用方式：
#   bash nebula_scripts/submit_tasd_tooluse_feedback_enhanced_sweep.sh [--dry-run]
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
PROJECT_NAME="TASD-v6"

# ── 数据集配置 ──────────────────────────────────────────────────────
# 跑 tooluse + bio，验证 feedback 增强效果
DATASETS=(
    "sciknoweval/biology"
    # "sciknoweval/chemistry"
    "sciknoweval/material"
    # "sciknoweval/physics"
    "tooluse"
)

# ── dry-run 模式 ─────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 超参配置（使用 bio 最优参数作为基线）
# =============================================================================

# ── Reward Type ─────────────────────────────────────────────────────
REWARD_TYPES=(
    "teacher_log_prob"
)

# ── Entropy Gate ─────────────────────────────────────────────────────
ENTROPY_GATE_LIST=(
    "hard_keep_reward"
)
ENTROPY_GATE_RATIO_LIST=(
    "1.0"
)

# ── Clip Adv ─────────────────────────────────────────────────────────
CLIP_ADV_LIST=(
    "true"
)
CLIP_ADV_VALUE_LIST=(
    "2.0"
)

# ── Distill Topk ──────────────────────────────────────────────────────
DISTILL_TOPK_LIST=(
    "256"
)

# ── Repetition Penalty ───────────────────────────────────────────────
REPETITION_PENALTY_LIST=(
    "1.05"
)

# ── Norm Adv By Std ─────────────────────────────────────────────────
NORM_ADV_BY_STD_LIST=(
    "true"
)

# ── Adv Std Floor ───────────────────────────────────────────────────
ADV_STD_FLOOR_LIST=(
    "none"
)

# ── Adv Entropy Weight ──────────────────────────────────────────────
ADV_ENTROPY_WEIGHT_LIST=(
    "none"
)

# ── Group Mean Mode ───────────────────────────────────────────────────
GROUP_MEAN_MODE_LIST=(
    "seq"
)

# ── Clip Ratio High ──────────────────────────────────────────────────
CLIP_RATIO_HIGH_LIST=(
    "0.28"
)

# Filter Groups: 关闭动态采样
FILTER_GROUPS_ENABLE="false"
FILTER_GROUPS_METRIC="acc"
FILTER_GROUPS_MAX_GEN="0"

# Include Successful Rollouts: 成功的 rollout 也参与训练
INCLUDE_SUCCESSFUL_ROLLOUTS_LIST=(
    "True"
)

# Remove Thinking from Demonstration: 是否从正确示范中移除 <think>...</think> 标签
REMOVE_THINKING_LIST=(
    "True"
    "False"
)

# Include Environment Feedback: 是否将环境反馈（错误答案 + 细粒度 feedback）注入 teacher context
# True = 启用 fbEnhanced 核心功能（failed_attempt + feedback_raw）
INCLUDE_ENVIRONMENT_FEEDBACK="True"

# ── Teacher Update Rate ─────────────────────────────────────────────
TEACHER_UPDATE_RATE_LIST=(
    "0.1"
)

# ── 固定参数 ────────────────────────────────────────────────────────────
LR="1e-5"
SEED="42"
# Git 信息（在本地获取，传递给 Nebula）
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
# ── Entropy Gate Tolerance ────────────────────────────────────────────
ENTROPY_GATE_TOLERANCE="0.0"
# entropy loss 系数
ENTROPY_COEFF_LIST=(
    "0.001"
)

# ── Rollout Temperature ─────────────────────────────────────────────
TEMPERATURE_LIST=(
    "1.0"
)

# ── Entropy Floor Penalty ───────────────────────────────────────────────
ENTROPY_FLOOR_LIST=(
    "0.0"
)
ENTROPY_PENALTY_COEFF_LIST=(
    "0.0"
)

# 固定参数
TEACHER_REG="ema"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
ROLLOUT_N="8"
MODEL="Qwen3-8B"

# =============================================================================
# 提交循环
# =============================================================================
TOTAL=0
SUBMITTED=0

for DATASET in "${DATASETS[@]}"; do
for REWARD_TYPE in "${REWARD_TYPES[@]}"; do
for ENTROPY_GATE in "${ENTROPY_GATE_LIST[@]}"; do
for ENTROPY_GATE_RATIO in "${ENTROPY_GATE_RATIO_LIST[@]}"; do
for CLIP_ADV in "${CLIP_ADV_LIST[@]}"; do
for CLIP_ADV_VALUE in "${CLIP_ADV_VALUE_LIST[@]}"; do
for DISTILL_TOPK in "${DISTILL_TOPK_LIST[@]}"; do
for REPETITION_PENALTY in "${REPETITION_PENALTY_LIST[@]}"; do
for NORM_ADV_BY_STD in "${NORM_ADV_BY_STD_LIST[@]}"; do
for ADV_STD_FLOOR in "${ADV_STD_FLOOR_LIST[@]}"; do
for ADV_ENTROPY_WEIGHT in "${ADV_ENTROPY_WEIGHT_LIST[@]}"; do
for CLIP_RATIO_HIGH in "${CLIP_RATIO_HIGH_LIST[@]}"; do
for TEACHER_UPDATE_RATE in "${TEACHER_UPDATE_RATE_LIST[@]}"; do
for INCLUDE_SUCCESSFUL_ROLLOUTS in "${INCLUDE_SUCCESSFUL_ROLLOUTS_LIST[@]}"; do
for ENTROPY_COEFF in "${ENTROPY_COEFF_LIST[@]}"; do
for TEMPERATURE in "${TEMPERATURE_LIST[@]}"; do
for GROUP_MEAN_MODE in "${GROUP_MEAN_MODE_LIST[@]}"; do
for ENTROPY_FLOOR in "${ENTROPY_FLOOR_LIST[@]}"; do
for ENTROPY_PENALTY_COEFF in "${ENTROPY_PENALTY_COEFF_LIST[@]}"; do
for REMOVE_THINKING in "${REMOVE_THINKING_LIST[@]}"; do
    # 当 gate=none 时，跳过非 1.0 的 ratio
    if [ "$ENTROPY_GATE" = "none" ] && [ "$ENTROPY_GATE_RATIO" != "1.0" ]; then
        continue
    fi

    TOTAL=$((TOTAL + 1))

    # 构建短数据集名
    DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')

    # 构建模型短名
    MODEL_SHORT=$(echo "$MODEL" | tr '.' '-')
    OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
    MODEL_PATH="${OSS_ROOT}/base_models/${MODEL}"

    # ── 构建实验名 ───────────────────────────────────────────────────
    # entropy gate 标签
    if [ "$ENTROPY_GATE" = "none" ]; then
        ENTROPY_TAG="-noGate"
        TOPK_TAG=""
        RATIO_TAG=""
    else
        ENTROPY_TAG="-gate_${ENTROPY_GATE}"
        TOPK_TAG="-topk${DISTILL_TOPK}"
        if [ "$ENTROPY_GATE_RATIO" = "1.0" ]; then
            RATIO_TAG=""
        else
            RATIO_TAG="-r${ENTROPY_GATE_RATIO}"
        fi
    fi

    REP_TAG="-rep${REPETITION_PENALTY}"

    if [ "$NORM_ADV_BY_STD" = "true" ]; then
        if [ "$ADV_STD_FLOOR" = "auto" ]; then
            STD_TAG="-std_auto"
        elif [ "$ADV_STD_FLOOR" = "none" ] || [ "$ADV_STD_FLOOR" = "0.0" ] || [ "$ADV_STD_FLOOR" = "0" ]; then
            STD_TAG="-normStd"
        else
            STD_TAG="-std_${ADV_STD_FLOOR}"
        fi
    else
        STD_TAG=""
    fi

    if [ "$CLIP_RATIO_HIGH" = "10000" ]; then
        CLIP_TAG="-clipHigh"
    else
        CLIP_TAG="-clipH${CLIP_RATIO_HIGH}"
    fi

    if [ "$CLIP_ADV" = "true" ]; then
        CLIP_ADV_TAG="-clipAdv${CLIP_ADV_VALUE}"
    else
        CLIP_ADV_TAG="-noClipAdv"
    fi

    EMA_TAG="-ema${TEACHER_UPDATE_RATE}"

    if [ "$INCLUDE_SUCCESSFUL_ROLLOUTS" = "True" ]; then
        ISR_TAG="-inclSucc"
    else
        ISR_TAG=""
    fi

    if [ "$ADV_ENTROPY_WEIGHT" = "none" ]; then
        AEW_TAG=""
    else
        AEW_TAG="-aew_${ADV_ENTROPY_WEIGHT}"
    fi

    if [ "$GROUP_MEAN_MODE" = "seq" ]; then
        GMM_TAG="-gmSeq"
    else
        GMM_TAG=""
    fi

    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    EC_TAG="-ec${ENTROPY_COEFF}"
    if [ "$TEMPERATURE" = "1.0" ]; then
        TEMP_TAG=""
    else
        TEMP_TAG="-temp${TEMPERATURE}"
    fi
    if [ "$ENTROPY_FLOOR" = "0.0" ] || [ "$ENTROPY_FLOOR" = "0" ]; then
        DIV_TAG=""
    else
        DIV_TAG="-ef${ENTROPY_FLOOR}_pc${ENTROPY_PENALTY_COEFF}"
    fi

    # 核心标签：fbEnhanced = feedback enhanced（细粒度反馈 + 错误答案展示 + 成功固定看到自己）
    FB_TAG="-fbEnhanced"

    JOB_NAME="TASD-${DATASET_SHORT}-rt_${REWARD_TYPE}${ENTROPY_TAG}${RATIO_TAG}${TOPK_TAG}${REP_TAG}${STD_TAG}${CLIP_TAG}${CLIP_ADV_TAG}${EMA_TAG}${ISR_TAG}${AEW_TAG}${GMM_TAG}${EC_TAG}${TEMP_TAG}${DIV_TAG}${FB_TAG}-v2-${MODEL_SHORT}-${CURRENT_TIME}"

    # ── 提交 ────────────────────────────────────────────────────────
    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  REWARD_TYPE=$REWARD_TYPE ENTROPY_GATE=$ENTROPY_GATE ENTROPY_GATE_RATIO=$ENTROPY_GATE_RATIO"
        echo "  CLIP_ADV_VALUE=$CLIP_ADV_VALUE DISTILL_TOPK=$DISTILL_TOPK"
        echo "  REPETITION_PENALTY=$REPETITION_PENALTY NORM_ADV_BY_STD=$NORM_ADV_BY_STD ADV_STD_FLOOR=$ADV_STD_FLOOR"
        echo "  ADV_ENTROPY_WEIGHT=$ADV_ENTROPY_WEIGHT"
        echo "  TEACHER_UPDATE_RATE=$TEACHER_UPDATE_RATE ENTROPY_COEFF=$ENTROPY_COEFF TEMPERATURE=$TEMPERATURE"
        echo "  ENTROPY_FLOOR=$ENTROPY_FLOOR ENTROPY_PENALTY_COEFF=$ENTROPY_PENALTY_COEFF"
        echo "  REMOVE_THINKING_FROM_DEMONSTRATION=$REMOVE_THINKING"
        echo "  INCLUDE_ENVIRONMENT_FEEDBACK=$INCLUDE_ENVIRONMENT_FEEDBACK"
        echo "  FEEDBACK_ENHANCED=true (细粒度反馈 + 错误答案展示 + 成功固定看到自己)"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
            --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL=${MODEL} --env=MODEL_PATH=${MODEL_PATH} --env=REWARD_TYPE=${REWARD_TYPE} --env=ENTROPY_GATE=${ENTROPY_GATE} --env=ENTROPY_GATE_RATIO=${ENTROPY_GATE_RATIO} --env=ENTROPY_GATE_TOLERANCE=${ENTROPY_GATE_TOLERANCE} --env=CLIP_ADV=${CLIP_ADV} --env=CLIP_ADV_VALUE=${CLIP_ADV_VALUE} --env=DISTILL_TOPK=${DISTILL_TOPK} --env=REPETITION_PENALTY=${REPETITION_PENALTY} --env=LR=${LR} --env=SEED=${SEED} --env=ENTROPY_COEFF=${ENTROPY_COEFF} --env=ROLLOUT_TEMPERATURE=${TEMPERATURE} --env=TEACHER_REG=${TEACHER_REG} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=INCLUDE_SUCCESSFUL_ROLLOUTS=${INCLUDE_SUCCESSFUL_ROLLOUTS} --env=NORM_ADV_BY_STD=${NORM_ADV_BY_STD} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR} --env=ADV_ENTROPY_WEIGHT=${ADV_ENTROPY_WEIGHT} --env=GROUP_MEAN_MODE=${GROUP_MEAN_MODE} --env=CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH} --env=FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE} --env=FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC} --env=FILTER_GROUPS_MAX_GEN=${FILTER_GROUPS_MAX_GEN} --env=ENTROPY_FLOOR=${ENTROPY_FLOOR} --env=ENTROPY_PENALTY_COEFF=${ENTROPY_PENALTY_COEFF} --env=REMOVE_THINKING_FROM_DEMONSTRATION=${REMOVE_THINKING} --env=INCLUDE_ENVIRONMENT_FEEDBACK=${INCLUDE_ENVIRONMENT_FEEDBACK} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=f598ad33b071751bf79d2484d8e1acefe8df9d879e129cae40340a158854f9cb --env=DINGTALK_SECRET=SECc5b9e4f61f56b32b46abf1ecedc11bdcba10dc35fbba8fa0ff62c084a1cc6ad3" \
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
        else
            SUBMITTED=$((SUBMITTED + 1))
            echo "✅ 已提交 (${SUBMITTED}/${TOTAL})"
        fi
        sleep 2
    fi

done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done
done

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
