#!/bin/bash
# =============================================================================
# Ray 集群启动脚本（简化版）
# 用于 Nebula 任务入口
# =============================================================================

set -e

# 解析参数
SCRIPT_PATH=""
WORLD_SIZE=1
JOB_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --script_path=*)
            SCRIPT_PATH="${1#*=}"
            shift
            ;;
        --world_size=*)
            WORLD_SIZE="${1#*=}"
            shift
            ;;
        --job_name=*)
            JOB_NAME="${1#*=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "============================================"
echo "[launch_ray_cluster.sh] 启动参数:"
echo "  SCRIPT_PATH: ${SCRIPT_PATH}"
echo "  WORLD_SIZE: ${WORLD_SIZE}"
echo "  JOB_NAME: ${JOB_NAME}"
echo "============================================"

# 清理残留的 Ray session
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true
rm -rf ~/.ray 2>/dev/null || true
sleep 3

# 执行实际训练脚本
if [ -n "$SCRIPT_PATH" ]; then
    echo "[launch_ray_cluster.sh] 执行脚本: ${SCRIPT_PATH}"
    bash "${SCRIPT_PATH}"
else
    echo "[launch_ray_cluster.sh] ERROR: SCRIPT_PATH is empty"
    exit 1
fi
