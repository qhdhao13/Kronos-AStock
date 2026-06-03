---
license: mit
tags:
- kronos
- financial
- stock-prediction
- a-share
- china-stock
- weichai
- 000338
- time-series
library_name: kronos
base_model: NeoQuasar/Kronos-small
datasets:
- custom
language:
- zh
---

# Kronos-000338-small

潍柴动力（**000338**）日 K 前复权数据上微调后的 **Kronos-small** 预测模型（24.7M），基于 [NeoQuasar/Kronos-small](https://huggingface.co/NeoQuasar/Kronos-small)。

## 配套 Tokenizer

- [qhdhao13/Kronos-000338-Tokenizer](https://huggingface.co/qhdhao13/Kronos-000338-Tokenizer)

## 训练概要

| 项 | 值 |
|----|-----|
| 股票 | 000338 潍柴动力 |
| 数据 | 日 K 前复权，约 4500+ 根 |
| 输入 / 预测 | 400 + 120 |
| 验证损失 | ~2.63 |
| 设备 | Apple MPS |

## 快速预测

```bash
git clone https://github.com/qhdhao13/Kronos-AStock.git
cd Kronos-AStock
pip install -r requirements.txt
export HF_ENDPOINT=https://hf-mirror.com   # 国内可选

python examples/predict_000338_finetuned.py --from-hub
```

```python
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer = KronosTokenizer.from_pretrained("qhdhao13/Kronos-000338-Tokenizer")
model = Kronos.from_pretrained("qhdhao13/Kronos-000338-small")
predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)
```

## 免责声明

输出为统计意义上的 K 线预测，**不构成投资建议**。

## 引用

请同时引用上游 [Kronos](https://github.com/shiyu-coder/Kronos) 论文与模型。
