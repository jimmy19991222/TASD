#!/bin/bash
# =============================================================================
# SDPO CDM Ablations — orthogonal-axis sweep alongside the λ-sweep
#
# 主 λ-sweep (submit_sdpo_cdm_sweep.sh) 覆盖 cdm_neg_weight ∈ {0.3, 0.5, 1.0}。
# 这里跑 3 个正交方向的消融,每个回答一个具体问题,不重复 λ 轴:
#
#   null   (NULL_MODE) : +gt / -gt 模板换成同一份 "Reference answer is <gt>"
#                        ⇒ "cdm_jsd_diff_mean 是否在 ε 噪声内 (math sanity)"
#                        (见 research/cdm_jsd_combination.md §6.4 / §A)
#   a1     (α=1.0)     : 把 JSD 换成纯 reverse-KL 几何 KL(π_s‖π_T) (α=1), λ=0.5
#                        ⚠️ α=1 时 D 已不是 JSD,严格说"双教师 JSD 组合"是误称;
#                        框架结构(差分 + Δ_copy 抵消)仍成立。
#                        ⇒ "reverse-KL (mode-seeking / zero-forcing) 的 CDM 是否更稳"
#   mkonly (verdict)   : 模板剥掉 GT,只剩 "This answer is verified ...", λ=0.5
#                        模仿 SDPO `marker` 模式 (无 GT 泄漏)
#                        ⇒ "GT 字符串是否必要,还是 verdict 本身就够?"
#                        (Δ_copy 项天然为 0,因为没有 GT 可抄)
#   refmk  (ref+marker): SDPO `ref` 用户侧 reprompt(注入 sibling 成功 rollout) +
#                        verdict-only assistant marker。两个 teacher 共享相同
#                        user message,只在 assistant 侧的 verdict 词不同。
#                        ⇒ "用 sibling rollout 替代 GT,user 侧句法 copy bias
#                           可被对称结构清掉,语义追随信号被差分锐化"
#                        (无需 train-time GT,部署友好)
#
# 砍掉的:
#   - l00  (λ=0)        : 主 λ-sweep 的 l03 邻近覆盖,边际信息量低
#   - damp (damping=1.0): (1−λ)·Δ_copy 残留是二阶问题,先看 λ-sweep 主结果
#   - bare (no verified): 跟 mkonly 信息高度重叠 (都是模板剥离方向)
#
# 共享:LOSS_METHOD=cdm, TEACHER_CONTEXT_MODE=cdm, FULL_LOGIT=True, topk=100,
# 数据/模型/lr/seed 与主 sweep 完全对齐,可直接横向比较。
#
# Usage:
#   bash nebula_scripts/submit_sdpo_cdm_ablations.sh [--dry-run]
#   bash nebula_scripts/submit_sdpo_cdm_ablations.sh --variant null
#   bash nebula_scripts/submit_sdpo_cdm_ablations.sh --variant a1,mkonly
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
VARIANT_SELECT="all"   # all | null | a1 | mkonly | refmk | comma-separated

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
# ref_mk: SDPO `ref` reprompt 注入 sibling success rollout 到 user 侧, +/- teacher
#         共享相同 reprompted user message, 仅 assistant 侧 verdict 标记不同。
#         不依赖 train-time GT; 是部署友好版 CDM。
VARIANTS=(
    "null:0.5:1.0:True:0.0:default"
    "a1:1.0:0.5:False:0.0:default"
    "mkonly:0.5:0.5:False:0.0:mkonly"
    "refmk:0.5:0.5:False:0.0:ref_mk"
    # 2026-05-24 batch 2: 完成 a1 (reverse-KL) 和 refmk (无 GT) 的 λ-sweep
    "a1l03:1.0:0.3:False:0.0:default"
    "a1l10:1.0:1.0:False:0.0:default"
    "refmkl03:0.5:0.3:False:0.0:ref_mk"
    "refmkl10:0.5:1.0:False:0.0:ref_mk"
    # 2026-05-25 batch 3: α 几何三角形封口 + Δ_copy 残留消解机制验证
    # a0  (α=0.0 forward-KL):mode-covering,push 触达 neg teacher 所有有质量位置;
    #                       理论上比 JSD/reverse-KL 更激进,可能更猛或更易爆
    # damp (damping=1.0)   :消掉 (1−λ)·Δ_copy 残留,验证 l05 |y| 收缩是否来自该项
    "a0:0.0:0.5:False:0.0:default"
    "damp:0.5:0.5:False:1.0:default"
    # 2026-05-25 batch 4: 拆分 cdm-a1 崩溃驱动 — GT 泄漏 (default) vs α=1 zero-forcing
    # a1refmk: reverse-KL + ref-style sibling reprompt + verdict-only marker;
    #          剥掉 GT 泄漏路径,只留 verdict 完成性信号 + α=1 zero-forcing。
    #          若仍崩 → 罪魁是 α=1;若稳 → GT 泄漏是 cdm-a1 主因。
    "a1refmk:1.0:0.5:False:0.0:ref_mk"
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
--env=SEED=${SEED} \
--env=TEST_FREQ=${TEST_FREQ:-10} \
--env=VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}"

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
