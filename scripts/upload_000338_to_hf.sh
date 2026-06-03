#!/usr/bin/env bash
# 上传 000338 微调权重到 Hugging Face（需先 huggingface-cli login）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

source "$ROOT/.venv/bin/activate"

if ! python -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
  echo "❌ 请先登录 Hugging Face："
  echo "   huggingface-cli login"
  echo "或：export HF_TOKEN=hf_xxxxxxxx"
  exit 1
fi

python scripts/upload_000338_to_hf.py "$@"
