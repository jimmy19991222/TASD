#!/bin/bash
# =============================================================================
# TASD top_k_agreement gate sweep
#
# 核心：teacher-student 共识过滤。逐位判定：
#   - 候选集 = student top-K（K 由 TOP_K_AGREEMENT_K 控制，默认 512）
#   - 在候选集内取 teacher argmax
#   - 若 student 采样 token == teacher argmax → 过滤（两者一致 = 无修改意愿）
#   - 全过滤的 response 整条 ε 兜底（默认 0.1）
#
# 作用层：等价 hard_keep_reward（只影响 advantage 的 effective_mask，不乘 reward）
#
# 固定其他超参为 v5 最优组合：
#   gmSeq + aew=none + clipAdv=2.0 + ema=0.1 + rep=1.05 + ec=0.001
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

# ── 数据集 ───────────────────────────────────────────────────────────────
DATASETS=(
    "sciknoweval/biology"
)

# ── dry-run ───────────────────────────────────────────────────────────────
DRY_RUN=false
if [ $# -gt 0 ] && [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry-run 模式：只打印命令，不提交"
fi

# =============================================================================
# 超参（top_k_agreement 专属）
# =============================================================================
REWARD_TYPE="teacher_log_prob"
ENTROPY_GATE="top_k_agreement"
ENTROPY_GATE_RATIO="1.0"   # top_k_agreement 模式下忽略，仅为参数齐全

# ── TOP_K_AGREEMENT 参数扫描 ─────────────────────────────────────────────
TOP_K_AGREEMENT_K_LIST=(
    "512"
)
TOP_K_AGREEMENT_EPS_LIST=(
    "0.1"
)

# ── 其他参数（锁定 v5 最优）────────────────────────────────────────────
CLIP_ADV="true"
CLIP_ADV_VALUE="2.0"
REPETITION_PENALTY="1.05"
NORM_ADV_BY_STD="true"
ADV_STD_FLOOR="none"
ADV_ENTROPY_WEIGHT="none"
GROUP_MEAN_MODE="seq"
CLIP_RATIO_HIGH="0.28"
FILTER_GROUPS_ENABLE="false"
FILTER_GROUPS_METRIC="acc"
FILTER_GROUPS_MAX_GEN="0"
INCLUDE_SUCCESSFUL_ROLLOUTS="True"
TEACHER_UPDATE_RATE="0.1"
ENTROPY_COEFF="0.001"
TEMPERATURE="1.0"

# ── 固定参数 ────────────────────────────────────────────────────────────
LR="1e-5"
SEED="42"
TEACHER_REG="ema"
TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="32"
ROLLOUT_N="8"
MODEL="Qwen3-8B"
# DISTILL_TOPK 名义上会被 top_k_agreement_k 覆盖（见 ray_trainer.py 逻辑），此处占位
DISTILL_TOPK="256"

GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# =============================================================================
# 提交循环
# =============================================================================
TOTAL=0
SUBMITTED=0

for DATASET in "${DATASETS[@]}"; do
for TOP_K_AGREEMENT_K in "${TOP_K_AGREEMENT_K_LIST[@]}"; do
for TOP_K_AGREEMENT_EPS in "${TOP_K_AGREEMENT_EPS_LIST[@]}"; do

    TOTAL=$((TOTAL + 1))

    DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
    MODEL_SHORT=$(echo "$MODEL" | tr '.' '-')
    OSS_ROOT="/data/oss_bucket_0/ad/loujieming.ljm"
    MODEL_PATH="${OSS_ROOT}/base_models/${MODEL}"

    # 实验名：TASD-{ds}-tka_K{K}_eps{eps}-gmSeq-clipAdv2.0-ema0.1-rep1.05-v2-{model}-{ts}
    TKA_TAG="-tka_K${TOP_K_AGREEMENT_K}_eps${TOP_K_AGREEMENT_EPS}"
    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)

    JOB_NAME="TASD-${DATASET_SHORT}${TKA_TAG}-gmSeq-clipAdv${CLIP_ADV_VALUE}-ema${TEACHER_UPDATE_RATE}-rep${REPETITION_PENALTY}-ec${ENTROPY_COEFF}-clipH${CLIP_RATIO_HIGH}-v2-${MODEL_SHORT}-${CURRENT_TIME}"

    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}"
        echo "  ENTROPY_GATE=${ENTROPY_GATE} TOP_K_AGREEMENT_K=${TOP_K_AGREEMENT_K} TOP_K_AGREEMENT_EPS=${TOP_K_AGREEMENT_EPS}"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"

        SUBMIT_OUTPUT=$(nebulactl run mdl \
            --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} --env=PROJECT_NAME=${PROJECT_NAME} --env=JOB_NAME=${JOB_NAME} --env=DATASET=${DATASET} --env=MODEL=${MODEL} --env=MODEL_PATH=${MODEL_PATH} --env=REWARD_TYPE=${REWARD_TYPE} --env=ENTROPY_GATE=${ENTROPY_GATE} --env=ENTROPY_GATE_RATIO=${ENTROPY_GATE_RATIO} --env=TOP_K_AGREEMENT_K=${TOP_K_AGREEMENT_K} --env=TOP_K_AGREEMENT_EPS=${TOP_K_AGREEMENT_EPS} --env=CLIP_ADV=${CLIP_ADV} --env=CLIP_ADV_VALUE=${CLIP_ADV_VALUE} --env=DISTILL_TOPK=${DISTILL_TOPK} --env=REPETITION_PENALTY=${REPETITION_PENALTY} --env=LR=${LR} --env=SEED=${SEED} --env=ENTROPY_COEFF=${ENTROPY_COEFF} --env=ROLLOUT_TEMPERATURE=${TEMPERATURE} --env=TEACHER_REG=${TEACHER_REG} --env=TEACHER_UPDATE_RATE=${TEACHER_UPDATE_RATE} --env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} --env=MINI_BATCH_SIZE=${MINI_BATCH_SIZE} --env=ROLLOUT_N=${ROLLOUT_N} --env=INCLUDE_SUCCESSFUL_ROLLOUTS=${INCLUDE_SUCCESSFUL_ROLLOUTS} --env=NORM_ADV_BY_STD=${NORM_ADV_BY_STD} --env=ADV_STD_FLOOR=${ADV_STD_FLOOR} --env=ADV_ENTROPY_WEIGHT=${ADV_ENTROPY_WEIGHT} --env=GROUP_MEAN_MODE=${GROUP_MEAN_MODE} --env=CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH} --env=FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE} --env=FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC} --env=FILTER_GROUPS_MAX_GEN=${FILTER_GROUPS_MAX_GEN} --env=GIT_BRANCH=${GIT_BRANCH} --env=GIT_COMMIT=${GIT_COMMIT} --env=DINGTALK_WEBHOOK=https://oapi.dingtalk.com/robot/send?access_token=f598ad33b071751bf79d2484d8e1acefe8df9d879e129cae40340a158854f9cb --env=DINGTALK_SECRET=SECc5b9e4f61f56b32b46abf1ecedc11bdcba10dc35fbba8fa0ff62c084a1cc6ad3" \
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

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成，共 ${TOTAL} 个 job"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job"
fi
echo "============================================================"
