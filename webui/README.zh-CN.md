# Kronos 中文 Web UI

面向 A 股用户的 Kronos 图形化预测界面（本分支汉化版）。

## 功能

- 全中文操作界面与错误提示
- A 股代码一键下载（东方财富 / akshare）
- 固定 520 根 K 线窗口（400 输入 + 120 对比）
- 时间滑块按**行号**对齐，支持「对齐最新数据」
- K 线图表、预测 vs 真实对比、分析结论
- 支持 CPU / CUDA / MPS

## 启动

```bash
cd webui
source ../.venv/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
python app.py
```

访问 http://localhost:7070

## 推荐流程（以 000338 为例）

1. 股票代码填 `000338`，周期选「日 K」，复权选「前复权」→ 下载
2. 加载 `000338_daily_qfq.csv`
3. 点击「对齐最新数据」
4. 加载模型 → 选择 Kronos-small
5. 开始预测 → 查看图表与对比表

## 说明

- 本 Web UI 使用 **HuggingFace 预训练模型**，用于**历史回测对比**
- **微调后的单股前瞻预测**请使用：`python examples/predict_000338_finetuned.py`

完整文档：[docs/000338使用指南.md](../docs/000338使用指南.md)
