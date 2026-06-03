---
license: mit
tags:
- kronos
- financial
- tokenizer
- a-share
- china-stock
- weichai
- 000338
library_name: kronos
base_model: NeoQuasar/Kronos-Tokenizer-base
datasets:
- custom
language:
- zh
---

# Kronos-000338-Tokenizer

潍柴动力（**000338**）日 K 前复权数据上微调后的 **Kronos Tokenizer**，基于 [NeoQuasar/Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base)。

## 配套模型

需与预测模型一起使用：

- **Predictor**: [qhdhao13/Kronos-000338-small](https://huggingface.co/qhdhao13/Kronos-000338-small)

## 训练数据

- 股票：000338（潍柴动力）
- 周期：日 K，前复权
- 窗口：lookback 400 + predict 120
- 上游代码：[qhdhao13/Kronos-AStock](https://github.com/qhdhao13/Kronos-AStock)

## 用法

```python
from model import KronosTokenizer

tokenizer = KronosTokenizer.from_pretrained("qhdhao13/Kronos-000338-Tokenizer")
```

## 免责声明

仅供学习研究，不构成投资建议。
