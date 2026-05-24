#!/bin/bash
# =============================================================================
# SDPO CDM Ablations — orthogonal-axis sweep alongside the λ-sweep
#
# 主 λ-sweep (submit_sdpo_cdm_sweep.sh) 覆盖 cdm_neg_weight ∈ {0.3, 0.5, 1.0}。
# 这里跑 6 个正交方向的消融,每个回答一个具体问题,不重复 λ 轴:
#
#   ── 超参轴 (4 jobs) ─────────────────────────────────────────────
#   l00  (λ=0)             : 关掉负教师 → 等价单 +gt teacher SDPO
#                            ⇒ "负教师到底贡献了多少 lift"
#   null (NULL_MODE)       : +gt / -gt 模板换成同一份 "Reference answer is <gt>"
#                            ⇒ "cdm_jsd_diff_mean 是否在 ε 噪声内 (sanity)"
#                            (见 research/cdm_jsd_combination.md §6.4 / §A)
#   damp (damping=1.0)     : λ=0.5 + overconfidence_damping=1.0
#                            ⇒ "λ<1 残留的 (1−λ)·Δ_copy 项是否伤害收敛"
#   a1   (α=1.0)           : 把 JSD 换成纯 forward-KL 几何 (α=1)
#                            ⇒ "前向 KL 的 CDM 是否更稳 / 更激进"
#
#   ── 模板轴 (2 jobs, λ=0.5 固定 — 模仿 SDPO teacher_context_mode 各档) ──
#   mkonly (verdict-only)  : 模板剥掉 GT,只剩 "This answer is verified ..."
#                            模仿 SDPO `marker` 模式 (无 GT 泄漏)
#                            ⇒ "GT 字符串是否必要,还是 verdict 本身就够?"
#                            (Δ_copy 项天然为 0,因为没有 GT 可抄)
#   bare   (no "verified") : 模板换成最简 "This answer is correct/incorrect."
#                            模仿 SDPO `ref_and_marker` 极简形态
#                            ⇒ "信号是否依赖 'verified' 这个词,还是只看极性?"
#
# 共享:LOSS_METHOD=cdm, TEACHER_CONTEXT_MODE=cdm, FULL_LOGIT=True, topk=100,
# 数据/模型/lr/seed 与主 sweep 完全对齐,可直接横向比较。
#
# Usage:
#   bash nebula_scripts/submit_sdpo_cdm_ablations.sh [--dry-run]
#   bash nebula_scripts/submit_sdpo_cdm_ablations.sh --variant l00
#   bash nebula_scripts/submit_sdpo_cdm_ablations.sh --variant mkonly,bare
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
CUSTOM_DOCKER_IMAGE="${CUSTOM_DOCKER_IMAGE:-hub.docker.alibaba-inc.com/mdl/notebook_saved:loujieming.ljm_yueqiu_sdpo_env_torch260_20260324155942}"
PROJECT_NAME="${PROJECT_NAME:-SDPO_CDM}"

# ── 参数解析 ────────────────────────────────────────────────────────────
DRY_RUN=false
VARIANT_SELECT="all"   # all | l00 | null | damp | a1 | mkonly | bare | comma-separated

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --variant=*) VARIANT_SELECT="${arg#*=}" ;;
        --variant) shift; VARIANT_SELECT="$1" ;;
    esac
done

[ "$DRY_RUN" = true ] && echo "Dry-run 模式：只打印命令,不提交"

# ── 公共固定超参 (与 λ-sweep 对齐) ─────────────────────────────────────
MODEL_NAME="Qwen3-8B"
DATASET="sciknoweval/biology"
LR="1e-5"
SEED="42"
TRAIN_BATCH_SIZE="32"
ROLLOUT_N="8"
FULL_LOGIT="True"
DISTILLATION_TOPK=100
DONT_REPROMPT_ON_SELF_SUCCESS="False"
SCRIPT_PATH="nebula_scripts/sdpo/sdpo_sciknoweval_parametric.sh"
DATASET_SHORT=$(echo "$DATASET" | tr '/' '-')
LR_TAG=$(echo "$LR" | tr '-' '_')

# ── Ablation 配置矩阵 ──────────────────────────────────────────────────
# 字段顺序: tag : ALPHA : CDM_NEG_WEIGHT : CDM_NULL_MODE : OVERCONFIDENCE_DAMPING : CDM_TEMPLATE_VARIANT
VARIANTS=(
    "l00:0.5:0.0:False:0.0:default"
    "null:0.5:1.0:True:0.0:default"
    "damp:0.5:0.5:False:1.0:default"
    "a1:1.0:0.5:False:0.0:default"
    "mkonly:0.5:0.5:False:0.0:mkonly"
    "bare:0.5:0.5:False:0.0:bare"
)

_should_run() {
    local tag="$1"
    [ "$VARIANT_SELECT" = "all" ] && return 0
    case ",${VARIANT_SELECT}," in *",${tag},"*) return 0 ;; esac
    return 1
}

# ── 提交辅助 ────────────────────────────────────────────────────────────
TOTAL=0
SUBMITTED=0

_submit_job() {
    local JOB_NAME="$1"
    local USER_PARAMS="$2"

    TOTAL=$((TOTAL + 1))

    if [ "$DRY_RUN" = true ]; then
        echo "------------------------------------------------------------"
        echo "Job #${TOTAL}: ${JOB_NAME}  [script: ${SCRIPT_PATH}]"
        echo "  $USER_PARAMS"
    else
        echo "提交 Job #${TOTAL}: ${JOB_NAME}"
        SUBMIT_OUTPUT=$(nebulactl run mdl \
            --force \
            --engine=xdl \
            --queue=${QUEUE} \
            --entry=nebula_scripts/entry.py \
            --user_params="--script_path=${SCRIPT_PATH} --world_size=${WORLD_SIZE} --job_name=${JOB_NAME} ${USER_PARAMS}" \
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
}

# ── 主循环 ──────────────────────────────────────────────────────────────
for spec in "${VARIANTS[@]}"; do
    IFS=':' read -r TAG ALPHA LAMBDA NULL_MODE DAMPING TEMPLATE_VARIANT <<< "$spec"
    _should_run "$TAG" || continue

    CURRENT_TIME=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="SDPO-cdm-${TAG}-${DATASET_SHORT}-lr${LR_TAG}-${MODEL_NAME}-${CURRENT_TIME}"

    USER_PARAMS="--env=PROJECT_NAME=${PROJECT_NAME} \
--env=JOB_NAME=${JOB_NAME} \
--env=DATASET=${DATASET} \
--env=MODEL_NAME=${MODEL_NAME} \
--env=LR=${LR} \
--env=ALPHA=${ALPHA} \
--env=FULL_LOGIT_DISTILLATION=${FULL_LOGIT} \
--env=DISTILLATION_TOPK=${DISTILLATION_TOPK} \
--env=LOSS_METHOD=cdm \
--env=TEACHER_CONTEXT_MODE=cdm \
--env=CDM_NEG_WEIGHT=${LAMBDA} \
--env=CDM_NULL_MODE=${NULL_MODE} \
--env=CDM_TEMPLATE_VARIANT=${TEMPLATE_VARIANT} \
--env=OVERCONFIDENCE_DAMPING=${DAMPING} \
--env=LOG_DELTA_W_STATS=True \
--env=DONT_REPROMPT_ON_SELF_SUCCESS=${DONT_REPROMPT_ON_SELF_SUCCESS} \
--env=TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} \
--env=ROLLOUT_N=${ROLLOUT_N} \
--env=SEED=${SEED}"

    _submit_job "$JOB_NAME" "$USER_PARAMS"
done

echo ""
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "Dry-run 完成,共 ${TOTAL} 个 job (variants: ${VARIANT_SELECT})"
else
    echo "提交完成：${SUBMITTED} / ${TOTAL} 个 job (variants: ${VARIANT_SELECT})"
fi
echo "============================================================"
