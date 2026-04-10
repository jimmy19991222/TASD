#!/bin/bash
set -e
set -x

# ========== 参数解析 ==========
SCRIPT_PATH=""
WORLD_SIZE=""
JOB_NAME=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --script_path=*) SCRIPT_PATH="${1#*=}" ;;
        --world_size=*) WORLD_SIZE="${1#*=}" ;;
        --job_name=*) JOB_NAME="${1#*=}" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

export WORLD_SIZE=$WORLD_SIZE
export JOB_NAME=$JOB_NAME

echo "SCRIPT_PATH = $SCRIPT_PATH"
echo "WORLD_SIZE  = $WORLD_SIZE"
echo "JOB_NAME    = $JOB_NAME"

# ── 激活自定义 conda 环境（若存在）───────────────────────────────
# 必须在 Ray start 之前激活，让 Ray worker 继承正确的 Python 环境
# 直接 export PATH 而非 conda activate，避免非交互式 shell 下 conda hook 未初始化问题
CONDA_ENV_NAME="sdpo_env"
CONDA_ENV_BIN="/opt/conda/envs/${CONDA_ENV_NAME}/bin"
if [ -d "${CONDA_ENV_BIN}" ]; then
    export PATH="${CONDA_ENV_BIN}:${PATH}"
    echo "Activated conda env: ${CONDA_ENV_NAME} (${CONDA_ENV_BIN})"
else
    echo "[WARN] conda env '${CONDA_ENV_NAME}' not found at ${CONDA_ENV_BIN}, using system Python"
fi

# ── 在 ray start 之前设置环境变量 ──
# Ray worker 进程从 ray daemon 继承环境变量，必须在 ray start 前设置

# 1. PYTHONPATH: 让 worker 加载 SDPO 自定义的 verl
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# 2. CUDA 环境 ───────────────────────────────────────────────────
#    确保 Ray worker 继承正确的 CUDA 环境
#    动态检测 GPU 数量并设置 CUDA_VISIBLE_DEVICES
GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 0)
if [ "$GPU_COUNT" -gt 0 ]; then
    GPU_INDICES=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
    export CUDA_VISIBLE_DEVICES="$GPU_INDICES"
    echo "Detected $GPU_COUNT GPUs, setting CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# 3. 禁用 deepspeed triton 功能 ──────────────────────────────────────────
#    transformers 在导入时会无条件 import deepspeed，
#    deepspeed 的 triton ops 在模块加载时需要 CUDA driver，
#    但 Ray worker 在导入阶段无法正确检测 GPU，导致 "0 active drivers" 错误。
#    设置以下环境变量来绕过这个问题：
export DS_BUILD_OPS=0                           # 禁止编译 deepspeed ops
export DS_SKIP_CUDA_CHECK=1                     # 跳过 CUDA 检查
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas  # 设置正确的 ptxas 路径
# TRITON_INTERPRET=1 已移除：自定义镜像 sdpo_env 中 triton 正常，无需解释器模式（会极大拖慢训练）

# 4. vLLM / PyTorch 配置
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARN
export TORCH_WARN_ACCUMULATE_GRAD_STREAM=0

# 5. SwanLab 配置（fallback 到硬编码 key，确保 Ray worker 进程继承）
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-M5oC00EEt8G1wC0XaHkal}"

echo "PYTHONPATH = $PYTHONPATH"

# ========== 清理超长环境变量，防止 Ray ray start 因 std::length_error 崩溃 ==========
# LD_LIBRARY_PATH / PATH 可能被 Nebula 初始化脚本反复追加导致极长（含大量重复路径）
# Ray 内部 C++ basic_string 在环境变量超长时会抛出 std::length_error
clean_path() {
    echo "$1" | tr ':' '\n' | awk '!seen[$0]++' | tr '\n' ':' | sed 's/:$//'
}
export LD_LIBRARY_PATH=$(clean_path "$LD_LIBRARY_PATH") || true
export PATH=$(clean_path "$PATH") || true

# ========== 启动 Ray 集群 ==========
if [ "$RANK" -eq 0 ]; then
    # 清理残留的 Ray 进程，防止 session name 冲突（上次任务异常中断时触发）
    ray stop --force 2>/dev/null || true
    rm -rf /tmp/ray 2>/dev/null || true
    rm -rf /tmp/ray_session_* 2>/dev/null || true
    rm -rf ~/.ray 2>/dev/null || true
    sleep 3  # 等待 Ray 完全停止
    ray start --head --dashboard-host=0.0.0.0
    sleep 20
    echo "Ray head started"
else
    ray start --address="$MASTER_ADDR:6379"
fi

ray status

# ========== Rank 0 执行训练 ==========
if [ "$RANK" -eq 0 ]; then
    bash $SCRIPT_PATH
fi
