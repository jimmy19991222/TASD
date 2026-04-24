#!/bin/bash
# =============================================================================
# TASD - Nebula 批量提交脚本
#
# TASD 核心：teacher distillation + token-level advantage
# 稳定性：clip_ratio_high (Clip-Higher) + clip_adv
#
# 实验设计：
#   - reward_type: teacher_prob | teacher_log_prob
#   - entropy_gate: none | hard | soft_v2（筛选有效训练信号）
#   - clip_ratio_high: 10000 (Clip-Higher)
#   - clip_adv_value: advantage clip 范围扫描
#   - ema update_rate: 0.1 | 1.0
#
# 使用方式：
#   bash nebula_scripts/submit_tasd_simple_sweep.sh [--dry-run]
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
PROJECT_NAME="TASD-v5"

# ── 数据集配置 ──────────────────────────────────────────────────────
DATASETS=(
    "sciknoweval/biology"
    # "sciknoweval/chemistry"
    # "sciknoweval/material"
    # "sciknoweval/physics"
    # "tooluse"
)

# ── dry-run 模式 ─────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 超参配置
# =============================================================================

# ── Reward Type ─────────────────────────────────────────────────────
REWARD_TYPES=(
    # "teacher_prob"
    "teacher_log_prob"
)

# ── Entropy Gate ─────────────────────────────────────────────────────
# 核心测试变量：entropy_gate + entropy_gate_ratio
# ratio 仅在 hard/hard_keep_reward 模式生效：1.0=原始hard | 0.8=保留top80% | 0.5=保留top50%
# hard: 过滤影响 reward（置零）和 advantage
# hard_keep_reward: 过滤只影响 advantage，reward 保持不变（group_mean/std 基于所有token）
ENTROPY_GATE_LIST=(
    "none"
    "hard"
    # "hard_keep_reward"
)
ENTROPY_GATE_RATIO_LIST=(
    "1.0"
    "0.8"
    # "0.5"
)

# ── Clip Adv ─────────────────────────────────────────────────────────
# v5 最优：clipAdv=true + value=2.0
CLIP_ADV_LIST=(
    "true"
)
CLIP_ADV_VALUE_LIST=(
    "2.0"
)

# ── Distill Topk ──────────────────────────────────────────────────────
DISTILL_TOPK_LIST=(
    # "100"
    "256"
    # "512"
)

# ── Repetition Penalty ───────────────────────────────────────────────
# v5 最优：rep=1.05（gmSeq + aew=none 下 best_acc=0.6325）
REPETITION_PENALTY_LIST=(
    "1.05"
)

# ── Norm Adv By Std ─────────────────────────────────────────────────
NORM_ADV_BY_STD_LIST=(
    "true"
    # "false"
)

# ── Adv Std Floor ───────────────────────────────────────────────────
# std下界：none | auto | float（仅在 norm_adv_by_std=true 时生效）
ADV_STD_FLOOR_LIST=(
    "none"
    # "auto"
    # "0.1"
)

# ── Adv Entropy Weight ──────────────────────────────────────────────
# advantage 阶段熵加权（纯加权，不过滤；过滤由 ENTROPY_GATE 统一控制）
# none: 不加权
# teacher_conf: 按 teacher 确定性加权 w = 1 - H_t_norm，关注 teacher 更确定的token
# certainty_diff: 按 teacher-student 熵差加权 w = norm(H_s - H_t)，关注 teacher 和 student 熵差较大的token
ADV_ENTROPY_WEIGHT_LIST=(
    # "none"
    # "teacher_conf"
    "certainty_diff"
)

# ── Group Mean Mode ───────────────────────────────────────────────────
# group mean/std 的统计粒度：
#   "token"：所有有效 token 打平统计（原有方式，存在 length bias）
#   "seq"  ：先 per-seq 取 token 均值，再在 seq 上统计（消除 length bias）
GROUP_MEAN_MODE_LIST=(
    # "token"
    "seq"
)

# ── Clip Ratio High ──────────────────────────────────────────────────
# 10000 = Clip-Higher（不 clip 上界），0.2 = 标准 PPO（关闭 Clip-Higher）
CLIP_RATIO_HIGH_LIST=(
    # "10000"
    "0.28"
    # "0.2"
)

# Filter Groups: 关闭动态采样
FILTER_GROUPS_ENABLE="false"
FILTER_GROUPS_METRIC="acc"
FILTER_GROUPS_MAX_GEN="0"

# Include Successful Rollouts: 成功的 rollout 也参与训练
INCLUDE_SUCCESSFUL_ROLLOUTS_LIST=(
    "True"
    # "False"
)

# ── Teacher Update Rate ─────────────────────────────────────────────
# EMA 更新率：1.0 = 完全跟随 student，0.1 = 缓慢跟踪
TEACHER_UPDATE_RATE_LIST=(
    "0.1"
    # "0.5"
    # "1.0"
)

# ── 固定参数 ────────────────────────────────────────────────────────────
LR="1e-5"
SEED="42"
# Git 信息（在本地获取，传递给 Nebula）
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
ENTROPY_COEFF="0.001"
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
for GROUP_MEAN_MODE in "${GROUP_MEAN_MODE_LIST[@]}"; do
    # 当 gate=none 时，跳过非 1.0 的 ratio（ratio 仅在 hard 时生效）
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
        TOPK_TAG=""  # noGate 时 topk 无意义，不显示
        RATIO_TAG=""
    else
        ENTROPY_TAG="-gate_${ENTROPY_GATE}"
        TOPK_TAG="-topk${DISTILL_TOPK}"
        # ratio 标签：1.0 为原始 hard gate，不加标记；其他值显示比例
        if [ "$ENTROPY_GATE_RATIO" = "1.0" ]; then
            RATIO_TAG=""
        else
            RATIO_TAG="-r${ENTROPY_GATE_RATIO}"
        fi
    fi

    # repetition penalty 标签
    REP_TAG="-rep${REPETITION_PENALTY}"

    # norm_adv_by_std 标签（含 floor 信息）
    if [ "$NORM_ADV_BY_STD" = "true" ]; then
        if [ "$ADV_STD_FLOOR" = "auto" ]; then
            STD_TAG="-std_auto"
        elif [ "$ADV_STD_FLOOR" = "none" ] || [ "$ADV_STD_FLOOR" = "0.0" ] || [ "$ADV_STD_FLOOR" = "0" ]; then
            STD_TAG="-normStd"  # 无 floor 保护，纯 std 归一化
        else
            STD_TAG="-std_${ADV_STD_FLOOR}"
        fi
    else
        STD_TAG=""
    fi

    # clip_ratio_high 标签
    if [ "$CLIP_RATIO_HIGH" = "10000" ]; then
        CLIP_TAG="-clipHigh"
    else
        CLIP_TAG="-clipH${CLIP_RATIO_HIGH}"
    fi

    # clip_adv 标签
    if [ "$CLIP_ADV" = "true" ]; then
        CLIP_ADV_TAG="-clipAdv${CLIP_ADV_VALUE}"
    else
        CLIP_ADV_TAG="-noClipAdv"
    fi

    # EMA update rate 标签
    EMA_TAG="-ema${TEACHER_UPDATE_RATE}"

    # include_successful_rollouts 标签
    if [ "$INCLUDE_SUCCESSFUL_ROLLOUTS" = "True" ]; then
        ISR_TAG="-inclSucc"
    else
        ISR_TAG=""
    fi

    # adv_entropy_weight 标签
    if [ "$ADV_ENTROPY_WEIGHT" = "none" ]; then
        AEW_TAG=""
    else
        AEW_TAG="-aew_${ADV_ENTROPY_WEIGHT}"
    fi

    # group_mean_mode 标签
    if [ "$GROUP_MEAN_MODE" = "seq" ]; then
        GMM_TAG="-gmSeq"
    else
        GMM_TAG=""  # token 模式为默认，不加标签
    fi

    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    EC_TAG="-ec${ENTROPY_COEFF}"
    # v2: clip_adv 移到 adv_entropy_weight 之后，修复归一化放大突破 clip 上界的问题
    JOB_NAME="TASD-${DATASET_SHORT}-rt_${REWARD_TYPE}${ENTROPY_TAG}${RATIO_TAG}${TOPK_TAG}${REP_TAG}${STD_TAG}${CLIP_TAG}${CLIP_ADV_TAG}${EMA_TAG}${ISR_TAG}${AEW_TAG}${GMM_TAG}${EC_TAG}-v2-${MODEL_SHORT}-${CURRENT_TIME}"

    # ── 提交 ────────────────────────────────────────────────────────
    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  REWARD_TYPE=$REWARD_TYPE ENTROPY_GATE=$ENTROPY_GATE ENTROPY_GATE_RATIO=$ENTROPY_GATE_RATIO"
        echo "  CLIP_ADV_VALUE=$CLIP_ADV_VALUE DISTILL_TOPK=$DISTILL_TOPK"
        echo "  REPETITION_PENALTY=$REPETITION_PENALTY NORM_ADV_BY_STD=$NORM_ADV_BY_STD ADV_STD_FLOOR=$ADV_STD_FLOOR"
        echo "  ADV_ENTROPY_WEIGHT=$ADV_ENTROPY_WEIGHT"
        echo "  TEACHER_UPDATE_RATE=$TEACHER_UPDATE_RATE"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
            --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL=${MODEL} --env=MODEL_PATH=${MODEL_PATH} --env=REWARD_TYPE=${REWARD_TYPE} --env=ENTROPY_GATE=${ENTROPY_GATE} --env=ENTROPY_GATE_RATIO=${ENTROPY_GATE_RATIO} --env=CLIP_ADV=${CLIP_ADV} --env=CLIP_ADV_VALUE=${CLIP_ADV_VALUE} --env=DISTILL_TOPK=${DISTILL_TOPK} --env=REPETITION_PENALTY=${REPETITION_PENALTY} --env=LR=${LR} --env=SEED=${SEED} --env=ENTROPY_COEFF=${ENTROPY_COEFF} --env=TEACHER_REG=${TEACHER_REG} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=INCLUDE_SUCCESSFUL_ROLLOUTS=${INCLUDE_SUCCESSFUL_ROLLOUTS} --env=NORM_ADV_BY_STD=${NORM_ADV_BY_STD} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR} --env=ADV_ENTROPY_WEIGHT=${ADV_ENTROPY_WEIGHT} --env=GROUP_MEAN_MODE=${GROUP_MEAN_MODE} --env=CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH} --env=FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE} --env=FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC} --env=FILTER_GROUPS_MAX_GEN=${FILTER_GROUPS_MAX_GEN} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=f598ad33b071751bf79d2484d8e1acefe8df9d879e129cae40340a158854f9cb --env=DINGTALK_SECRET=SECc5b9e4f61f56b32b46abf1ecedc11bdcba10dc35fbba8fa0ff62c084a1cc6ad3" \
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

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
