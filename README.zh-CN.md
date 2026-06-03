<div align="center">

# Kronos · A 股中文版

**基于 [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos) 的汉化与 A 股实践分支**

[![License](https://img.shields.io/github/license/shiyu-coder/Kronos?color=green)](./LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗-Kronos_Model-yellow)](https://huggingface.co/NeoQuasar/Kronos-small)

[English README](./README.md) · [000338 使用指南](./docs/000338使用指南.md)

</div>

---

## 这是什么？

本仓库在官方 Kronos 金融 K 线基础模型之上，增加了 **面向中国 A 股用户** 的完整工具链：

| 模块 | 说明 |
|------|------|
| **中文 Web UI** | 全中文界面、A 股代码一键下载、回测对比、分析结论 |
| **000338 微调流程** | 潍柴动力日 K 前复权微调配置 + 一键脚本 |
| **前瞻预测脚本** | 微调模型从最新价预测未来 120 个交易日 |
| **中文文档** | [docs/000338使用指南.md](./docs/000338使用指南.md) 从环境到微调全流程 |

> 上游项目：Kronos 在 45+ 全球交易所 K 线上预训练，**并非 A 股专用**。本分支通过汉化、数据下载、单股微调，降低 A 股用户上手成本。

---

## 核心改造一览

### 1. Web UI 汉化（`webui/`）

- 界面、错误提示、图表悬停、分析结论 **全中文**
- **A 股数据下载**：输入 6 位代码（如 `000338`），支持日 K / 5 分钟 K、前复权 / 后复权
- **时间窗口修复**：按 K 线行号选 520 根窗口，默认对齐最新数据
- **预测 vs 真实对比**：400 输入 + 120 对比，MAE / RMSE / MAPE
- Plotly 中文 locale、错误信息精简（避免刷屏）

### 2. 000338（潍柴动力）微调（`finetune_csv/`）

- 配置：`finetune_csv/configs/config_000338_daily_qfq.yaml`
- 一键微调：`bash scripts/finetune_000338.sh`
- 支持 **Apple MPS** GPU 加速（Mac）
- 基于 Kronos-small 继续训练，更贴近单股走势

### 3. 前瞻预测（`examples/`）

```bash
python examples/predict_000338_finetuned.py
```

需先完成本地微调（`bash scripts/finetune_000338.sh`）。输出未来 120 根 K 线 CSV 与趋势图。

---

## 快速开始

### 环境

```bash
git clone https://github.com/qhdhao13/Kronos-AStock.git
cd Kronos-AStock

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install flask flask-cors plotly akshare pyyaml

export HF_ENDPOINT=https://hf-mirror.com   # 国内建议
```

### 启动中文 Web UI

```bash
cd webui
python app.py
# 浏览器打开 http://localhost:7070
```

### 000338 完整流程

详见 **[docs/000338使用指南.md](./docs/000338使用指南.md)**，简要步骤：

1. Web UI 下载 `000338` 日 K 前复权 → `data/000338_daily_qfq.csv`
2. Web UI 回测：滑块「对齐最新数据」→ 加载 Kronos-small → 开始预测
3. **二选一**：本地微调 `bash scripts/finetune_000338.sh`（约 2 小时）
4. 前瞻：`python examples/predict_000338_finetuned.py`

---

## 目录结构（本分支新增/改动）

```
├── docs/000338使用指南.md          # 中文详细教程
├── scripts/finetune_000338.sh      # 微调一键脚本
├── data/000338_daily_qfq.csv       # 示例数据（可 Web UI 更新）
├── examples/predict_000338_finetuned.py
├── finetune_csv/configs/config_000338_daily_qfq.yaml
└── webui/                          # 汉化 Web 界面
    ├── app.py
    └── templates/index.html
```

---

## 两种使用方式

| 方式 | 工具 | 模型 | 用途 |
|------|------|------|------|
| **回测对比** | Web UI | 预训练 Kronos-small | 看模型在历史 120 天上准不准 |
| **前瞻预测** | `predict_000338_finetuned.py` | 本地微调模型 | 从最新价往后猜 120 个交易日 |

**⚠️ 免责声明**：所有预测仅供学习研究，不构成投资建议。

---

## 扩展到其他 A 股

1. Web UI 下载目标股票 CSV  
2. 复制 `config_000338_daily_qfq.yaml`，修改 `data_path` 与 `exp_name`  
3. 复制 `predict_000338_finetuned.py` 并改路径  
4. 运行微调与预测  

---

## 致谢与引用

- 上游项目：[shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)
- 论文：[Kronos: A Foundation Model for the Language of Financial Markets](https://arxiv.org/abs/2508.02739)

如使用 Kronos 模型，请引用上游 README 中的 BibTeX。

---

## License

与上游相同，见 [LICENSE](./LICENSE)。
