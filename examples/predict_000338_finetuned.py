#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 000338 本地微调后的 Kronos 模型做日 K 前瞻预测。

用法：
    cd /Volumes/disk-hfm/Kronos
    source .venv/bin/activate
    export HF_ENDPOINT=https://hf-mirror.com
    python examples/predict_000338_finetuned.py
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from model import Kronos, KronosTokenizer, KronosPredictor

FINETUNED_DIR = os.path.join(ROOT, "finetune_csv", "finetuned", "000338_daily_qfq")
TOKENIZER_PATH = os.path.join(FINETUNED_DIR, "tokenizer", "best_model")
MODEL_PATH = os.path.join(FINETUNED_DIR, "basemodel", "best_model")
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


def main():
    if not os.path.isfile(os.path.join(TOKENIZER_PATH, "model.safetensors")):
        print("❌ 未找到本地微调模型，请先运行：")
        print("   bash scripts/finetune_000338.sh")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = pick_device()
    print(f"📦 加载本地微调模型（device={device}）...")

    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PATH)
    model = Kronos.from_pretrained(MODEL_PATH)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    df = pd.read_csv(DATA_PATH)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)

    if len(df) < LOOKBACK + PRED_LEN:
        raise ValueError(f"数据不足，需要至少 {LOOKBACK + PRED_LEN} 行")

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
