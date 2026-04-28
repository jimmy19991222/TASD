#!/bin/bash
# =============================================================================
# TASD - Entropy Floor Penalty 实验组
#
# 目标：验证 teacher_prob reward + entropy_floor penalty 能否缓解 entropy collapse
# 核心变量：
#   - reward_type: teacher_prob（区别于 baseline 的 teacher_log_prob）
#   - entropy_floor: student 归一化熵下界，低于此值的 token 被惩罚
#   - entropy_penalty_coeff: 惩罚强度
# 其他参数锁定为当前已知最优配置
#
# 使用方式：
#   bash nebula_scripts/submit_tasd_entropy_floor_sweep.sh [--dry-run]
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
# teacher_prob：scale 较小（0~1），entropy_floor penalty 相对更显著，先验证
# teacher_log_prob：对照组，仅跑 baseline（无 floor）作参照
REWARD_TYPES=(
    "teacher_prob"
    # "teacher_log_prob"  # 对照组：只跑 floor=0.0 的 baseline
)

# ── Entropy Floor ────────────────────────────────────────────────────
# 0.0  = 不启用（baseline）
# 0.1  = 最有希望：只惩罚真正低熵 token，有选择性（当前均值 ~0.055，低于 0.1 的 token 会被惩罚）
# 跳过 0.05（几乎无条件惩罚，等同 entropy_coeff）和 0.2（过激进）
ENTROPY_FLOOR_LIST=(
    "0.0"   # baseline：不惩罚
    "0.1"   # 最有希望：选择性惩罚低熵 token
    "0.2"
)

# ── Entropy Penalty Coeff ────────────────────────────────────────────
# floor=0.1 时 deficit 最大约 0.1；coeff=3.0 → penalty_max ≈ 0.3，量级合理
# 只跑 3.0，避免 1.0 太弱（penalty_max=0.1，可能仍被淹没）
ENTROPY_PENALTY_COEFF_LIST=(
    # "1.0"
    "3.0"
)

# ── 固定参数（当前已知最优）────────────────────────────────────────────
ENTROPY_GATE="none"
ENTROPY_GATE_RATIO="1.0"
CLIP_ADV="true"
CLIP_ADV_VALUE="2.0"
DISTILL_TOPK="256"          # entropy_floor 需要 topk 计算熵
REPETITION_PENALTY="1.05"
NORM_ADV_BY_STD="true"
ADV_STD_FLOOR="none"
ADV_ENTROPY_WEIGHT="none"
GROUP_MEAN_MODE="seq"
CLIP_RATIO_HIGH="0.28"
INCLUDE_SUCCESSFUL_ROLLOUTS="True"
TEACHER_UPDATE_RATE="0.1"
ENTROPY_COEFF="0.001"
TEMPERATURE="1.0"
TEACHER_REG="ema"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
ROLLOUT_N="8"
MODEL="Qwen3-8B"
FILTER_GROUPS_ENABLE="false"
FILTER_GROUPS_METRIC="acc"
FILTER_GROUPS_MAX_GEN="0"
LR="1e-5"
SEED="42"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# =============================================================================
# 提交循环
# =============================================================================
TOTAL=0
SUBMITTED=0

for DATASET in "${DATASETS[@]}"; do
for REWARD_TYPE in "${REWARD_TYPES[@]}"; do
for ENTROPY_FLOOR in "${ENTROPY_FLOOR_LIST[@]}"; do
for ENTROPY_PENALTY_COEFF in "${ENTROPY_PENALTY_COEFF_LIST[@]}"; do
    # teacher_log_prob 只跑 baseline（floor=0.0），不重复验证 entropy_floor
    if [ "$REWARD_TYPE" = "teacher_log_prob" ] && [ "$ENTROPY_FLOOR" != "0.0" ]; then
        continue
    fi

    # floor=0.0 时，coeff 无意义，只跑一次（用第一个 coeff 值占位，实际传 0.0）
    if [ "$ENTROPY_FLOOR" = "0.0" ] || [ "$ENTROPY_FLOOR" = "0" ]; then
        # 只让第一个 coeff 值通过，避免重复提交 baseline
        if [ "$ENTROPY_PENALTY_COEFF" != "3.0" ]; then
            continue
        fi
        ENTROPY_PENALTY_COEFF_ACTUAL="0.0"
    else
        ENTROPY_PENALTY_COEFF_ACTUAL="${ENTROPY_PENALTY_COEFF}"
    fi

    TOTAL=$((TOTAL + 1))

    DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
    MODEL_SHORT=$(echo "$MODEL" | tr '.' '-')
    OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
    MODEL_PATH="${OSS_ROOT}/base_models/${MODEL}"

    # ── 构建实验名 ───────────────────────────────────────────────────
    # entropy floor 标签
    if [ "$ENTROPY_FLOOR" = "0.0" ] || [ "$ENTROPY_FLOOR" = "0" ]; then
        EF_TAG=""
    else
        EF_TAG="-ef${ENTROPY_FLOOR}_pc${ENTROPY_PENALTY_COEFF_ACTUAL}"
    fi

    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="TASD-${DATASET_SHORT}-rt_${REWARD_TYPE}-noGate-rep1.05-normStd-clipH0.28-clipAdv2.0-ema0.1-inclSucc-gmSeq-ec0.001${EF_TAG}-v2-${MODEL_SHORT}-${CURRENT_TIME}"

    # ── 提交 ────────────────────────────────────────────────────────
    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  REWARD_TYPE=$REWARD_TYPE"
        echo "  ENTROPY_FLOOR=$ENTROPY_FLOOR, ENTROPY_PENALTY_COEFF=$ENTROPY_PENALTY_COEFF_ACTUAL"
        echo "  DISTILL_TOPK=$DISTILL_TOPK (必须>0 以支持熵计算)"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
            --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL=${MODEL} --env=MODEL_PATH=${MODEL_PATH} --env=REWARD_TYPE=${REWARD_TYPE} --env=ENTROPY_GATE=${ENTROPY_GATE} --env=ENTROPY_GATE_RATIO=${ENTROPY_GATE_RATIO} --env=CLIP_ADV=${CLIP_ADV} --env=CLIP_ADV_VALUE=${CLIP_ADV_VALUE} --env=DISTILL_TOPK=${DISTILL_TOPK} --env=REPETITION_PENALTY=${REPETITION_PENALTY} --env=LR=${LR} --env=SEED=${SEED} --env=ENTROPY_COEFF=${ENTROPY_COEFF} --env=ROLLOUT_TEMPERATURE=${TEMPERATURE} --env=TEACHER_REG=${TEACHER_REG} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=INCLUDE_SUCCESSFUL_ROLLOUTS=${INCLUDE_SUCCESSFUL_ROLLOUTS} --env=NORM_ADV_BY_STD=${NORM_ADV_BY_STD} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR} --env=ADV_ENTROPY_WEIGHT=${ADV_ENTROPY_WEIGHT} --env=GROUP_MEAN_MODE=${GROUP_MEAN_MODE} --env=CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH} --env=FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE} --env=FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC} --env=FILTER_GROUPS_MAX_GEN=${FILTER_GROUPS_MAX_GEN} --env=ENTROPY_FLOOR=${ENTROPY_FLOOR} --env=ENTROPY_PENALTY_COEFF=${ENTROPY_PENALTY_COEFF_ACTUAL} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=f598ad33b071751bf79d2484d8e1acefe8df9d879e129cae40340a158854f9cb --env=DINGTALK_SECRET=SECc5b9e4f61f56b32b46abf1ecedc11bdcba10dc35fbba8fa0ff62c084a1cc6ad3" \
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

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
