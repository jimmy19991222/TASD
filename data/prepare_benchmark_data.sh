#!/bin/bash
# =============================================================================
# 准备 MATH500 / GSM8K / Code 数据集用于 RL 训练
#
# 用法:
#   bash data/prepare_benchmark_data.sh math500
#   bash data/prepare_benchmark_data.sh gsm8k
#   bash data/prepare_benchmark_data.sh livecodebench
#   bash data/prepare_benchmark_data.sh all
#
# 输出:
#   data/<dataset>/train.parquet  — RL rollout 用的 prompt 集合
#   data/<dataset>/test.parquet   — validation 用的 prompt 集合
#
# 之后需要上传到 OSS:
#   ossutil cp -r data/<dataset>/ oss://lazada-ai-model/ad/loujieming.ljm/datasets/<dataset>/
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Use project venv if available
if [ -f "${PROJECT_DIR}/sdpo_env/bin/python" ]; then
    export PATH="${PROJECT_DIR}/sdpo_env/bin:${PATH}"
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

DATASET="${1:?用法: $0 <math500|gsm8k|livecodebench|all>}"

prepare_dataset() {
    local name="$1"
    local hf_name="$2"
    local out_dir="data/${name}"

    echo "============================================================"
    echo "准备数据集: ${name} (${hf_name})"
    echo "============================================================"

    mkdir -p "${out_dir}"

    # Step 1: 从 HuggingFace 下载并格式化
    echo "[1/3] 下载并格式化..."
    python data/load_dataset.py \
        --dataset_name "${hf_name}" \
        --output_path "${out_dir}/all.json"

    # Step 2: 拆分 train/test (90/10)
    echo "[2/3] 拆分 train/test (90/10)..."
    python data/split_tasks.py \
        --json_path "${out_dir}/all.json" \
        --output_dir "${out_dir}" \
        --test_ratio 0.1 \
        --seed 42

    # Step 3: 转成 parquet
    echo "[3/3] 转换为 parquet..."
    python data/preprocess.py --data_source "${out_dir}"

    echo ""
    echo "完成! 输出文件:"
    ls -lh "${out_dir}"/*.parquet 2>/dev/null || echo "  (未找到 parquet 文件)"
    echo ""
}

case "$DATASET" in
    math500)
        prepare_dataset "math500" "math-ai/math500"
        ;;
    gsm8k)
        prepare_dataset "gsm8k" "openai/gsm8k"
        ;;
    livecodebench)
        prepare_dataset "livecodebench" "livecodebench/code_generation_lite-v6"
        ;;
    all)
        prepare_dataset "math500" "math-ai/math500"
        prepare_dataset "gsm8k" "openai/gsm8k"
        prepare_dataset "livecodebench" "livecodebench/code_generation_lite-v6"
        ;;
    *)
        echo "不支持的数据集: ${DATASET}"
        echo "可选: math500, gsm8k, livecodebench, all"
        exit 1
        ;;
esac

echo "============================================================"
echo "数据准备完成!"
echo ""
echo "下一步: 上传到 OSS"
echo "  ossutil cp -r data/<dataset>/ oss://lazada-ai-model/ad/loujieming.ljm/datasets/<dataset>/"
echo "============================================================"
