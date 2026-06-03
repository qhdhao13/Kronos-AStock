#!/usr/bin/env bash
# 潍柴动力 000338 微调启动脚本（Mac MPS / CPU）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/finetune_csv"

source "$ROOT/.venv/bin/activate"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

CONFIG="configs/config_000338_daily_qfq.yaml"
LOG="$ROOT/finetune_csv/finetuned/000338_daily_qfq/train.log"

mkdir -p "$(dirname "$LOG")"

echo "=========================================="
echo " Kronos 微调：000338 日 K 前复权"
echo " 配置：$CONFIG"
echo " 日志：$LOG"
echo " 设备：MPS（Apple GPU）或 CPU"
echo "=========================================="

python train_sequential.py --config "$CONFIG" 2>&1 | tee "$LOG"

echo ""
echo "微调完成。模型目录："
echo "  Tokenizer: $ROOT/finetune_csv/finetuned/000338_daily_qfq/tokenizer/best_model"
echo "  Predictor: $ROOT/finetune_csv/finetuned/000338_daily_qfq/basemodel/best_model"
echo ""
echo "运行预测："
echo "  python examples/predict_000338_finetuned.py"
