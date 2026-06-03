#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 000338 微调后的 Kronos 模型做日 K 前瞻预测。

用法：
    cd /Volumes/disk-hfm/Kronos
    source .venv/bin/activate
    export HF_ENDPOINT=https://hf-mirror.com

    # 优先本地微调权重，没有则自动从 Hugging Face 下载
    python examples/predict_000338_finetuned.py

    # 强制从 Hugging Face 加载
    python examples/predict_000338_finetuned.py --from-hub
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import Kronos, KronosTokenizer, KronosPredictor

# 本地微调产物
FINETUNED_DIR = os.path.join(ROOT, "finetune_csv", "finetuned", "000338_daily_qfq")
TOKENIZER_LOCAL = os.path.join(FINETUNED_DIR, "tokenizer", "best_model")
MODEL_LOCAL = os.path.join(FINETUNED_DIR, "basemodel", "best_model")

# Hugging Face 公开权重（upload_000338_to_hf.py 上传）
HF_TOKENIZER_ID = os.environ.get("HF_TOKENIZER_REPO", "qhdhao13/Kronos-000338-Tokenizer")
HF_MODEL_ID = os.environ.get("HF_MODEL_REPO", "qhdhao13/Kronos-000338-small")

DATA_PATH = os.path.join(ROOT, "data", "000338_daily_qfq.csv")
OUTPUT_DIR = os.path.join(ROOT, "examples", "outputs")

LOOKBACK = 400
PRED_LEN = 120
DEVICE = "mps"  # Mac 可用 mps；无 GPU 改为 cpu


def pick_device():
    import torch

    if DEVICE == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if DEVICE == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def local_weights_ready() -> bool:
    return (
        os.path.isfile(os.path.join(TOKENIZER_LOCAL, "model.safetensors"))
        and os.path.isfile(os.path.join(MODEL_LOCAL, "model.safetensors"))
    )


def resolve_model_source(from_hub: bool) -> tuple[str, str, str]:
    """返回 (tokenizer_id, model_id, source_label)"""
    if from_hub:
        return HF_TOKENIZER_ID, HF_MODEL_ID, "Hugging Face Hub"
    if local_weights_ready():
        return TOKENIZER_LOCAL, MODEL_LOCAL, "本地微调"
    return HF_TOKENIZER_ID, HF_MODEL_ID, "Hugging Face Hub（本地权重不存在，自动回退）"


def main():
    parser = argparse.ArgumentParser(description="000338 微调 Kronos 前瞻预测")
    parser.add_argument(
        "--from-hub",
        action="store_true",
        help="从 Hugging Face 加载 qhdhao13/Kronos-000338-* 权重",
    )
    args = parser.parse_args()

    tokenizer_src, model_src, source_label = resolve_model_source(args.from_hub)
    if not args.from_hub and not local_weights_ready():
        print("ℹ️  未找到本地微调权重，将从 Hugging Face 下载（约 110MB）")
        print("   本地微调：bash scripts/finetune_000338.sh")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = pick_device()
    print(f"📦 加载微调模型（{source_label}，device={device}）...")
    print(f"   Tokenizer: {tokenizer_src}")
    print(f"   Model:     {model_src}")

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_src)
    model = Kronos.from_pretrained(model_src)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    df = pd.read_csv(DATA_PATH)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    if len(df) < LOOKBACK + PRED_LEN:
        raise ValueError(f"数据不足，需要至少 {LOOKBACK + PRED_LEN} 行")

    # 用最新 LOOKBACK 根 K 线预测未来 PRED_LEN 个交易日
    hist = df.iloc[-LOOKBACK:].copy()
    last_ts = hist["timestamps"].iloc[-1]
    freq = df["timestamps"].diff().dropna()
    freq = freq[freq > pd.Timedelta(0)].median()
    if pd.isna(freq):
        freq = pd.Timedelta(days=1)

    future_ts = pd.date_range(start=last_ts + freq, periods=PRED_LEN, freq=freq)

    cols = ["open", "high", "low", "close"]
    if "volume" in hist.columns:
        cols.append("volume")

    x_df = hist[cols]
    x_timestamp = hist["timestamps"]
    y_timestamp = pd.Series(future_ts, name="timestamps")

    print(f"📈 输入：{x_timestamp.iloc[0].date()} ~ {x_timestamp.iloc[-1].date()}，最新收盘 {hist['close'].iloc[-1]:.2f}")
    print(f"🔮 预测：{y_timestamp.iloc[0].date()} ~ {y_timestamp.iloc[-1].date()}（{PRED_LEN} 根）")

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=1.0,
        top_p=0.9,
        sample_count=1,
    )

    out_csv = os.path.join(OUTPUT_DIR, "pred_000338_finetuned_forward.csv")
    result = pred_df.copy()
    result.insert(0, "timestamps", y_timestamp.values)
    result.to_csv(out_csv, index=False)
    print(f"✅ 已保存：{out_csv}")

    # 简单图表：历史收盘 + 预测收盘
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist["timestamps"], hist["close"], label="历史收盘", color="#2563eb")
    ax.plot(y_timestamp, pred_df["close"], label="微调模型预测收盘", color="#dc2626", linestyle="--")
    ax.axvline(last_ts, color="#94a3b8", linestyle=":", label="预测起点")
    ax.set_title("000338 潍柴动力 — 微调 Kronos-small 前瞻预测")
    ax.set_xlabel("日期")
    ax.set_ylabel("价格（前复权）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    out_png = os.path.join(OUTPUT_DIR, "pred_000338_finetuned_forward_chart.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 已保存：{out_png}")
    print(f"末日预测收盘：{pred_df['close'].iloc[-1]:.2f}")


if __name__ == "__main__":
    main()
