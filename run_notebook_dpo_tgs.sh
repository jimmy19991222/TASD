#!/usr/bin/env bash
# =============================================================================
# Local DPO-TGS V2 (On-Policy DPO + Teacher-Guided Sampling) 训练脚本
# 4 卡 notebook 单机版 — 用于 Nebula 任务的本地 plumbing 验证 + 快速 ablation
#
# 设计依据: research/dpo_teacher_guided_sampling.md
# 理论锚点: 5 篇 (OAIF + OFS-DPO + Samplers + RPO + Meta Bridging) 见
#           papers/raw/opd_papers/OPD_Deep_Analysis.html
#
# Pipeline:
#   Phase 1: 标准 rollout n_init per prompt
#   Phase 2: 同 prompt correct sibling → SDPO teacher ctx
#   Phase 3: failed sample 上 n_attempts 次干预 (z_T != y_S[t*] reselect_t)
#   Phase 4: concat + lineage tag → DPO pair → linearized DPO advantage
#
# 5 种模式:
#   tiny          :        3 step bs=2 n_init=2 n_attempts=1 ~5 min → 1-GPU 极简 plumbing 验证
#   smoke         (默认): 10 step bs=4 n_init=2 n_attempts=2 ~15 min → 4-GPU plumbing 验证
#   innov         :        10 step + 全 3 innovations 开 (①+②+③) → 4-GPU 验证新代码 plumbing
#   pair          :        30 step chain_consecutive vs hybrid_init_chain 串行 ~45 min
#   full          :        100 step bs=32 n_init=2 n_attempts=2 ~4h → 与 Nebula 一致
#
# GPU 数量自动适配 (env 可覆盖):
#   tiny 模式默认 N_GPUS_PER_NODE=1, gpu_memory_utilization=0.7 (留 OPSD teacher fwd 空间)
#   其他模式默认 N_GPUS_PER_NODE=4
#   覆盖: N_GPUS_PER_NODE=2 ./run_notebook_dpo_tgs.sh smoke
#
# 用法:
#   ./run_notebook_dpo_tgs.sh tiny               # 1-GPU 极简 (新增, 队列紧张时用)
#   ./run_notebook_dpo_tgs.sh                    # smoke (默认 v1 baseline, 4-GPU)
#   ./run_notebook_dpo_tgs.sh smoke              # 同上
#   ./run_notebook_dpo_tgs.sh innov              # 全 3 innovations 开
#   ./run_notebook_dpo_tgs.sh pair               # pair_strategy 2 路 ablation
#   ./run_notebook_dpo_tgs.sh full               # 完整 100 step
#
#   # 单独 toggle innovations:
#   DPO_USE_TEACHER_ANCHORED_REF=True ./run_notebook_dpo_tgs.sh smoke   # 仅 ②
#   DPO_DELTA_R_WEIGHT_MODE=linear ./run_notebook_dpo_tgs.sh smoke      # 仅 ③
#   DPO_CAUSAL_LOCALIZE=True DPO_BETA_TOKEN=0.2 DPO_BETA_CONTINUATION=0.05 \
#     ./run_notebook_dpo_tgs.sh smoke                                    # 仅 ①
#
#   # 自定义其他超参:
#   DPO_BETA=0.5 ./run_notebook_dpo_tgs.sh smoke
#   DPO_N_INIT=4 DPO_N_ATTEMPTS=4 ./run_notebook_dpo_tgs.sh smoke
#   TOTAL_STEPS=20 TRAIN_BATCH_SIZE=8 ./run_notebook_dpo_tgs.sh smoke
#
# 后端调用 nebula_scripts/dpo_tgs/dpo_tgs_sciknoweval_parametric.sh
# =============================================================================
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

# ── 模式选择 ────────────────────────────────────────────────────────────
MODE="${1:-smoke}"

case "${MODE}" in
    tiny)
        # 1-GPU 极简 smoke: 3 step bs=2 n_init=2 n_attempts=1 ~5 min
        # 仅验证 dispatch/rollout/reward_fn/pair_collector/dpo_loss plumbing 不崩
        DEFAULT_TOTAL_STEPS=3
        DEFAULT_TRAIN_BATCH_SIZE=2
        DEFAULT_MINI_BATCH_SIZE=2
        DEFAULT_N_INIT=2
        DEFAULT_N_ATTEMPTS=1                 # 最少
        DEFAULT_VAL_N=2
        DEFAULT_VAL_BEFORE_TRAIN=False
        DEFAULT_PAIR_STRATEGY="chain_consecutive"
        DEFAULT_CAUSAL_LOCALIZE=False
        DEFAULT_USE_TEACHER_ANCHORED_REF=False
        DEFAULT_DELTA_R_WEIGHT_MODE=none
        DEFAULT_N_GPUS_PER_NODE=1
        # 1-GPU 时 FSDP shard size = world (无分片), Qwen3-8B bf16 params (16GB) +
        # AdamW fp32 optimizer state (32GB) ≈ 48GB,vLLM 只能拿剩下 ~30GB / 80GB ≈ 0.30。
        # 0.65 是 4-GPU 假设 (actor 分 4 片) 的值,单卡必须降。
        DEFAULT_GPU_MEM_UTIL=0.30
        # tiny 是 plumbing 验证,不在乎吞吐。打开 optimizer_offload 把 32GB AdamW 推到
        # CPU,GPU 给 vLLM 留更多空间。Nebula / smoke 4-GPU 不需要,只 tiny 模式打开。
        export FSDP_OPTIMIZER_OFFLOAD="${FSDP_OPTIMIZER_OFFLOAD:-True}"
        export FSDP_PARAM_OFFLOAD="${FSDP_PARAM_OFFLOAD:-False}"
        # async rollout 默认 num_workers=8; tiny 的 effective batch = bs(2) * n_init(2) = 4
        # 个 prompt, 要 chunk 给 8 个 worker 会触发 "size 4 % 8 != 0" assert. 单卡只用 1 个
        # worker 即可。smoke / innov / full (bs ≥ 4) 用默认 8 不会卡。
        export ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-1}"
        ;;
    smoke)
        # Plumbing 验证: ~15 min, 只验证 plumbing + 关注 dpo/* 指标出现 (v1 baseline, 4-GPU)
        DEFAULT_TOTAL_STEPS=10
        DEFAULT_TRAIN_BATCH_SIZE=4
        DEFAULT_MINI_BATCH_SIZE=4
        DEFAULT_N_INIT=2
        DEFAULT_N_ATTEMPTS=2
        DEFAULT_VAL_N=4
        DEFAULT_VAL_BEFORE_TRAIN=False
        DEFAULT_PAIR_STRATEGY="chain_consecutive"
        DEFAULT_CAUSAL_LOCALIZE=False
        DEFAULT_USE_TEACHER_ANCHORED_REF=False
        DEFAULT_DELTA_R_WEIGHT_MODE=none
        DEFAULT_N_GPUS_PER_NODE=4
        DEFAULT_GPU_MEM_UTIL=0.85
        ;;
    innov)
        # V2.5 plumbing 验证: ~15 min, 全 3 innovations 一起开 (4-GPU)
        DEFAULT_TOTAL_STEPS=10
        DEFAULT_TRAIN_BATCH_SIZE=4
        DEFAULT_MINI_BATCH_SIZE=4
        DEFAULT_N_INIT=2
        DEFAULT_N_ATTEMPTS=2
        DEFAULT_VAL_N=4
        DEFAULT_VAL_BEFORE_TRAIN=False
        DEFAULT_PAIR_STRATEGY="chain_consecutive"
        DEFAULT_CAUSAL_LOCALIZE=True
        DEFAULT_USE_TEACHER_ANCHORED_REF=True
        DEFAULT_DELTA_R_WEIGHT_MODE=linear
        DEFAULT_N_GPUS_PER_NODE=4
        DEFAULT_GPU_MEM_UTIL=0.85
        # 推荐 default β_token = 2β, β_continuation = 0.5β (decisive token weighted higher)
        export DPO_BETA_TOKEN="${DPO_BETA_TOKEN:-0.2}"
        export DPO_BETA_CONTINUATION="${DPO_BETA_CONTINUATION:-0.05}"
        ;;
    pair)
        # pair_strategy ablation: chain_consecutive vs hybrid_init_chain, 各 30 step
        DEFAULT_TOTAL_STEPS=30
        DEFAULT_TRAIN_BATCH_SIZE=8
        DEFAULT_MINI_BATCH_SIZE=8
        DEFAULT_N_INIT=2
        DEFAULT_N_ATTEMPTS=2
        DEFAULT_VAL_N=8
        DEFAULT_VAL_BEFORE_TRAIN=False
        DEFAULT_PAIR_STRATEGY="chain_consecutive"   # overridden in loop
        DEFAULT_CAUSAL_LOCALIZE=False
        DEFAULT_USE_TEACHER_ANCHORED_REF=False
        DEFAULT_DELTA_R_WEIGHT_MODE=none
        DEFAULT_N_GPUS_PER_NODE=4
        DEFAULT_GPU_MEM_UTIL=0.85
        ;;
    full)
        # 与 Nebula 等价配置
        DEFAULT_TOTAL_STEPS=100
        DEFAULT_TRAIN_BATCH_SIZE=32
        DEFAULT_MINI_BATCH_SIZE=32
        DEFAULT_N_INIT=2
        DEFAULT_N_ATTEMPTS=2
        DEFAULT_VAL_N=16
        DEFAULT_VAL_BEFORE_TRAIN=False
        DEFAULT_PAIR_STRATEGY="chain_consecutive"
        DEFAULT_CAUSAL_LOCALIZE=False
        DEFAULT_USE_TEACHER_ANCHORED_REF=False
        DEFAULT_DELTA_R_WEIGHT_MODE=none
        DEFAULT_N_GPUS_PER_NODE=4
        DEFAULT_GPU_MEM_UTIL=0.85
        ;;
    *)
        echo "ERROR: unknown mode '${MODE}'. Use: tiny | smoke | innov | pair | full"
        exit 1
        ;;
esac

# ── OSS 挂载根目录 (notebook 通常已挂为本地) ─────────────────────────
export OSS_ROOT="${OSS_ROOT:-/data/oss_bucket_0/ad/loujieming.ljm}"

# ── 数据集 / 模型 ───────────────────────────────────────────────────
export DATASET="${DATASET:-sciknoweval}"
export SUBJECT="${SUBJECT:-biology}"
export MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
export MODEL_PATH="${MODEL_PATH:-${OSS_ROOT}/base_models/${MODEL_NAME}}"

# ── 训练超参 (默认由模式决定, env 可覆盖) ──────────────────────────
export TOTAL_STEPS="${TOTAL_STEPS:-${DEFAULT_TOTAL_STEPS}}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${DEFAULT_TRAIN_BATCH_SIZE}}"
export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-${DEFAULT_MINI_BATCH_SIZE}}"
export VAL_N="${VAL_N:-${DEFAULT_VAL_N}}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-${DEFAULT_VAL_BEFORE_TRAIN}}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-${DEFAULT_GPU_MEM_UTIL:-0.85}}"
export LR="${LR:-1e-5}"

# ── teacher 装配 (EMA 复用 self_distillation 基础设施) ────────────────
export TEACHER_REGULARIZATION="${TEACHER_REGULARIZATION:-ema}"
export TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"

# ── DPO-TGS 专属超参 ──────────────────────────────────────────────────
export DPO_N_INIT="${DPO_N_INIT:-${DEFAULT_N_INIT}}"
export DPO_N_ATTEMPTS="${DPO_N_ATTEMPTS:-${DEFAULT_N_ATTEMPTS}}"
export DPO_CORRECT_THRESHOLD="${DPO_CORRECT_THRESHOLD:-1.0}"
export DPO_SDPO_CTX_SOURCE="${DPO_SDPO_CTX_SOURCE:-sibling_correct}"   # sibling_correct | gt
export DPO_ALL_FAILED_STRATEGY="${DPO_ALL_FAILED_STRATEGY:-skip}"      # skip | gt_fallback
export DPO_MAX_RESELECT="${DPO_MAX_RESELECT:-3}"
export DPO_EXCLUDE_TAIL="${DPO_EXCLUDE_TAIL:-8}"
export DPO_BETA="${DPO_BETA:-0.1}"
export DPO_ALPHA="${DPO_ALPHA:-1.0}"
export DPO_PAIR_STRATEGY="${DPO_PAIR_STRATEGY:-${DEFAULT_PAIR_STRATEGY}}"  # chain_consecutive | hybrid_init_chain
export DPO_PAIR_MARGIN="${DPO_PAIR_MARGIN:-0.0}"
export DPO_MIN_RESP_LEN="${DPO_MIN_RESP_LEN:-50}"
export DPO_LEN_PENALTY="${DPO_LEN_PENALTY:-linear}"

# ── 3 V2.5 innovations (mode 默认值在 case 块,这里 env 覆盖优先) ────────
export DPO_CAUSAL_LOCALIZE="${DPO_CAUSAL_LOCALIZE:-${DEFAULT_CAUSAL_LOCALIZE}}"
export DPO_USE_TEACHER_ANCHORED_REF="${DPO_USE_TEACHER_ANCHORED_REF:-${DEFAULT_USE_TEACHER_ANCHORED_REF}}"
export DPO_DELTA_R_WEIGHT_MODE="${DPO_DELTA_R_WEIGHT_MODE:-${DEFAULT_DELTA_R_WEIGHT_MODE}}"
export DPO_DELTA_R_WEIGHT_TAU="${DPO_DELTA_R_WEIGHT_TAU:-1.0}"
# beta_token / beta_continuation: empty = use beta default (yaml null)
export DPO_BETA_TOKEN="${DPO_BETA_TOKEN:-}"
export DPO_BETA_CONTINUATION="${DPO_BETA_CONTINUATION:-}"

# rollout.n MUST equal n_init (Phase 1 baseline count). parametric.sh handles default.
export ROLLOUT_N="${ROLLOUT_N:-${DPO_N_INIT}}"

# Legacy intervention_credit knobs (mostly unused by adaptive_rollout — kept for compat)
export IC_DIVERGENCE_METRIC="${IC_DIVERGENCE_METRIC:-argmax_excl_eos}"
export IC_TEACHER_DECODE_TEMPERATURE="${IC_TEACHER_DECODE_TEMPERATURE:-0.0}"

# ── 数据路径解析 (4 级 fallback,与 intervention_credit 一致) ─────────
export DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets}"
_candidates_train=(
    "${DATA_ROOT}/${DATASET}/${SUBJECT}/train.parquet"
    "${DATA_ROOT}/${DATASET}/train.parquet"
    "${OSS_ROOT}/datasets/${DATASET}/${SUBJECT}/train.parquet"
    "${OSS_ROOT}/datasets/${DATASET}/train.parquet"
)
if [ -z "${TRAIN_DATA_PATH:-}" ]; then
    for _cand in "${_candidates_train[@]}"; do
        if [ -f "${_cand}" ]; then
            export TRAIN_DATA_PATH="${_cand}"
            export VAL_DATA_PATH="${VAL_DATA_PATH:-${_cand%/train.parquet}/test.parquet}"
            break
        fi
    done
fi

# ── 兜底校验 ───────────────────────────────────────────────────────
if [ ! -d "${OSS_ROOT}" ]; then
    echo "⚠️  OSS_ROOT=${OSS_ROOT} 不存在 (未挂载)。save_path / model_path 依赖它。"
fi
if [ -z "${TRAIN_DATA_PATH:-}" ]; then
    echo "❌ 未找到数据 (试过以下候选):"
    for _cand in "${_candidates_train[@]}"; do echo "   - ${_cand}"; done
    echo "   请 export DATA_ROOT=/your/datasets 或 export TRAIN_DATA_PATH=/abs/path/train.parquet"
    exit 1
fi

# ── GPU 数量 (mode 默认在 case 块,env 覆盖优先) ─────────────────────
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${DEFAULT_N_GPUS_PER_NODE:-4}}"

# ── Ray 启动健壮化 (tiny / 1-GPU 资源紧张时关键) ────────────────────
# In-process ray.init() 在 notebook /tmp NFS + dashboard 启动慢的环境会 timeout。
# 解决方案: 这里 shell 显式 ray start --head + disable dashboard,
# 然后设 RAY_ADDRESS=auto 让 ray.init() attach 到 already-running cluster (秒级)。
export RAY_TMPDIR="${RAY_TMPDIR:-/dev/shm/ray_tmp_${USER}}"
[ ! -d /dev/shm ] && export RAY_TMPDIR="/tmp/ray_tmp_${USER}"
mkdir -p "${RAY_TMPDIR}" 2>/dev/null
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"

# RAY_NODE_IP_ADDRESS env 用户可显式设;默认空 = 让 ray 自动检测 (强制 127.0.0.1
# 会让某些 A100 容器内 ray 启动后无输出,反而更难诊断)。
# 用真实内网 IP: RAY_NODE_IP_ADDRESS=$(hostname -I | awk '{print $1}')
# 用 loopback: RAY_NODE_IP_ADDRESS=127.0.0.1
RAY_NODE_IP_ADDRESS="${RAY_NODE_IP_ADDRESS:-}"

# 大幅放宽 raylet 启动等待 (默认 10s 在容器里不够)
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-300}"
export RAY_gcs_rpc_server_reconnect_timeout_s="${RAY_gcs_rpc_server_reconnect_timeout_s:-120}"
export RAY_gcs_server_request_timeout_seconds="${RAY_gcs_server_request_timeout_seconds:-300}"
export RAY_DISABLE_USAGE_STATS=1
# 关键: 强制 Python 不 buffer stdout/stderr,否则 ray hang 5 min 后被 SIGKILL,
# 所有 buffer 在内存里的 log 全丢,导致 /tmp/ray_start.log 是空的。
export PYTHONUNBUFFERED=1

# tiny mode 进一步限制 ray object store 大小 (1 GPU 机器内存紧张)
if [ "${MODE}" = "tiny" ]; then
    _RAY_OBJ_MEM="${RAY_object_store_memory:-1000000000}"   # 1 GB
else
    _RAY_OBJ_MEM="${RAY_object_store_memory:-4000000000}"   # 4 GB
fi
export RAY_object_store_memory="${_RAY_OBJ_MEM}"

# ── NCCL / GLOO bootstrap 接口 (notebook eth1 不允许自连接, 必须走 lo) ─
# 现象: NCCL 默认选 eth1 = 202.x 内网, listen 后自 connect-back timeout 30s × 34 retry.
# 单节点 (含 4-GPU) bootstrap 走 loopback 安全;实际 GPU 间通信走 NVLink/PCIe 不受影响.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# ── 预启动 Ray cluster (绕开 ray.init() in-process slow startup) ──
echo "[ray] stopping any existing cluster..."
ray stop --force >/dev/null 2>&1 || true
pkill -9 -f raylet 2>/dev/null || true
pkill -9 -f gcs_server 2>/dev/null || true
sleep 1
rm -rf "${RAY_TMPDIR}"/* 2>/dev/null || true

echo "[ray] starting head node (no dashboard, ${N_GPUS_PER_NODE} GPU, $(nproc) CPU, ${_RAY_OBJ_MEM} bytes obj store, ip=${RAY_NODE_IP_ADDRESS:-auto})..."
# Don't pass --num-gpus to `ray start` — CUDA init can stall raylet registration.
# Resources are still discoverable in workers via CUDA_VISIBLE_DEVICES + ray.get_runtime_context().
# Build ray start args (omit --node-ip-address when empty so ray auto-detects)
_RAY_ARGS=(--head --include-dashboard=false --disable-usage-stats
    --num-cpus=$(nproc) --temp-dir="${RAY_TMPDIR}"
    --object-store-memory=${_RAY_OBJ_MEM})
if [ -n "${RAY_NODE_IP_ADDRESS}" ]; then
    _RAY_ARGS+=(--node-ip-address="${RAY_NODE_IP_ADDRESS}")
fi
# stdbuf -oL -eL 强制 line-buffering;PYTHONUNBUFFERED 已 export 在前面
stdbuf -oL -eL ray start "${_RAY_ARGS[@]}" >/tmp/ray_start.log 2>&1 &
_RAY_START_PID=$!

# 等 ray runtime 就绪 (通过 grep "Ray runtime started" 判定),最多 5 min
echo -n "[ray] waiting for runtime up ..."
for _i in $(seq 1 60); do
    if grep -q "Ray runtime started" /tmp/ray_start.log 2>/dev/null; then
        echo " ✓ (${_i}×5s)"
        break
    fi
    if ! kill -0 $_RAY_START_PID 2>/dev/null; then
        echo ""
        echo "❌ ray start exited unexpectedly. Last 40 lines of /tmp/ray_start.log:"
        tail -40 /tmp/ray_start.log
        exit 1
    fi
    sleep 5
    echo -n "."
done

if ! grep -q "Ray runtime started" /tmp/ray_start.log 2>/dev/null; then
    echo ""
    echo "❌ ray start timed out after 5 min."
    echo "  /tmp/ray_start.log size: $(wc -c < /tmp/ray_start.log 2>/dev/null || echo 0) bytes"
    echo "  Last 40 lines:"
    tail -40 /tmp/ray_start.log 2>/dev/null
    echo ""
    echo "  Active ray processes (likely hung):"
    ps -ef | grep -E "ray|raylet|gcs" | grep -v grep | head -20
    kill -9 $_RAY_START_PID 2>/dev/null
    exit 1
fi

if [ -n "${RAY_NODE_IP_ADDRESS}" ]; then
    export RAY_ADDRESS="${RAY_NODE_IP_ADDRESS}:6379"
else
    # ray start 自动选 IP,从 log 里抓 (不同 ray 版本 log 格式不同, grep 必须容错)
    # 不加 || true 时, grep 没匹配 → pipefail+set -e 会让脚本静默 exit
    _DETECTED_IP=$(grep -oP "Local node IP: \K[^ ]+" /tmp/ray_start.log 2>/dev/null | head -1 || true)
    if [ -z "${_DETECTED_IP}" ]; then
        # fallback: 从 "ray start" 命令输出的 "Started a local Ray instance" 段抓
        _DETECTED_IP=$(grep -oE "[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:6379" /tmp/ray_start.log 2>/dev/null | head -1 | cut -d: -f1 || true)
    fi
    export RAY_ADDRESS="${_DETECTED_IP:-127.0.0.1}:6379"
fi
echo "[ray] cluster up. RAY_ADDRESS=${RAY_ADDRESS}, RAY_TMPDIR=${RAY_TMPDIR}"

# 在脚本退出时清理 ray (避免下次有残留进程)
trap 'echo "[ray] stopping..."; ray stop --force >/dev/null 2>&1 || true' EXIT

# ── Git 信息 ───────────────────────────────────────────────────────
export GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
export GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
export PROJECT_NAME="${PROJECT_NAME:-DPO-TGS-local}"

# ─────────────────────────────────────────────────────────────────────
# 单策略 launcher (smoke / full / pair-内部调用)
# ─────────────────────────────────────────────────────────────────────
run_single_strategy() {
    local pair_strategy="$1"
    export DPO_PAIR_STRATEGY="${pair_strategy}"
    local ps_short
    case "${pair_strategy}" in
        chain_consecutive) ps_short="chcons" ;;
        hybrid_init_chain) ps_short="hybrid" ;;
        *)                 ps_short="${pair_strategy}" ;;
    esac

    SUFFIX="${SUFFIX:-${MODE}}"
    export JOB_NAME="${JOB_NAME:-LOCAL-DPO-TGS-${ps_short}-ni${DPO_N_INIT}na${DPO_N_ATTEMPTS}-b${DPO_BETA}-a${DPO_ALPHA}-${SUBJECT}-step${TOTAL_STEPS}-${SUFFIX}}"

    echo "========================================================================"
    echo "Local DPO-TGS launch (notebook, ${N_GPUS_PER_NODE}×GPU) — MODE=${MODE}"
    echo "  DATASET           : ${DATASET} (subject=${SUBJECT})"
    echo "  TRAIN_DATA        : ${TRAIN_DATA_PATH}"
    echo "  MODEL             : ${MODEL_PATH}"
    echo "  TOTAL_STEPS       : ${TOTAL_STEPS}"
    echo "  TRAIN_BS          : ${TRAIN_BATCH_SIZE}    LR=${LR}"
    echo "  TEACHER (EMA)     : rate=${TEACHER_UPDATE_RATE}"
    echo "  ── DPO-TGS V2 ────────────────────────────────────"
    echo "  ROLLOUT_N=N_INIT  : ${DPO_N_INIT}"
    echo "  N_ATTEMPTS        : ${DPO_N_ATTEMPTS}"
    echo "  CORRECT_THRESHOLD : ${DPO_CORRECT_THRESHOLD}"
    echo "  SDPO_CTX_SOURCE   : ${DPO_SDPO_CTX_SOURCE}    all_failed=${DPO_ALL_FAILED_STRATEGY}"
    echo "  MAX_RESELECT      : ${DPO_MAX_RESELECT}    exclude_tail=${DPO_EXCLUDE_TAIL}"
    echo "  BETA              : ${DPO_BETA}    ALPHA=${DPO_ALPHA} (1.0=pure DPO, 0.0=pure GRPO)"
    echo "  PAIR_STRATEGY     : ${DPO_PAIR_STRATEGY}    margin=${DPO_PAIR_MARGIN}"
    echo "  LENGTH FLOOR      : min=${DPO_MIN_RESP_LEN} penalty=${DPO_LEN_PENALTY}"
    echo "  ── INNOVATIONS (V2.5) ────────────────────────────"
    echo "  ① CAUSAL_LOCALIZE : ${DPO_CAUSAL_LOCALIZE}  beta_token=${DPO_BETA_TOKEN:-(beta)} beta_cont=${DPO_BETA_CONTINUATION:-(0.5*beta)}"
    echo "  ② TEACHER_ANCHORED: ${DPO_USE_TEACHER_ANCHORED_REF}"
    echo "  ③ DELTA_R_WEIGHT  : ${DPO_DELTA_R_WEIGHT_MODE}  tau=${DPO_DELTA_R_WEIGHT_TAU}"
    echo "  ──────────────────────────────────────────────────"
    echo "  GIT               : ${GIT_BRANCH} @ ${GIT_COMMIT}"
    echo "  JOB_NAME          : ${JOB_NAME}"
    echo "========================================================================"

    bash "${PROJECT_ROOT}/nebula_scripts/dpo_tgs/dpo_tgs_sciknoweval_parametric.sh"
    unset JOB_NAME  # let next strategy pick a fresh name
}

# ─────────────────────────────────────────────────────────────────────
# 调度
# ─────────────────────────────────────────────────────────────────────
case "${MODE}" in
    tiny|smoke|innov|full)
        run_single_strategy "${DPO_PAIR_STRATEGY}"
        ;;
    pair)
        # ablation: chain_consecutive vs hybrid_init_chain
        for ps in chain_consecutive hybrid_init_chain; do
            echo ""
            echo "════════ Pair-strategy ablation: ${ps} ════════"
            run_single_strategy "${ps}"
        done
        ;;
esac

echo ""
echo "========================================================================"
echo "✅ Local DPO-TGS run complete (MODE=${MODE})"
echo ""
echo "📊 SwanLab 关注指标:"
echo "  P0 (DPO 标准 + 早期预警):"
echo "    val-core/dpo_implicit_reward_accuracy  → 目标 > 0.6"
echo "    dpo/sigma_neg_margin_mean              → 梯度幅度 (萎缩=训练 stall)"
echo "    dpo/kl_to_ref_chosen_mean              → 涨太快=length collapse"
echo "    dpo/length_ratio_chosen_over_rejected  → ≈1 健康 >1.5 警告"
echo "  P1 (DPO-TGS V2 独有):"
echo "    dpo/prompts_with_no_correct_pct        → 高=SDPO ctx 缺失"
echo "    dpo/chain_attempt_success_rate@k       → n_attempts 边际收益"
echo "    dpo/grpo_fallback_rate                 → DPO 信号占比 = 1 - this"
echo "  主指标:"
echo "    val-core/sciknoweval/${SUBJECT}/acc/mean@${VAL_N}"
echo "========================================================================"
