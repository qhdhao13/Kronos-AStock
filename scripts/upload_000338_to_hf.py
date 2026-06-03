#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 000338 微调权重上传到 Hugging Face Hub。

使用前请先登录（任选其一）：
  huggingface-cli login
  export HF_TOKEN=hf_xxxxxxxx

国内上传可配合镜像（下载侧）：
  export HF_ENDPOINT=https://hf-mirror.com

用法：
  python scripts/upload_000338_to_hf.py
  python scripts/upload_000338_to_hf.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TOKENIZER_LOCAL = os.path.join(
    ROOT, "finetune_csv", "finetuned", "000338_daily_qfq", "tokenizer", "best_model"
)
MODEL_LOCAL = os.path.join(
    ROOT, "finetune_csv", "finetuned", "000338_daily_qfq", "basemodel", "best_model"
)
CARD_TOKENIZER = os.path.join(ROOT, "scripts", "hf_model_cards", "tokenizer", "README.md")
CARD_MODEL = os.path.join(ROOT, "scripts", "hf_model_cards", "basemodel", "README.md")

HF_TOKENIZER_REPO = os.environ.get("HF_TOKENIZER_REPO", "qhdhao13/Kronos-000338-Tokenizer")
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "qhdhao13/Kronos-000338-small")


def _check_local(path: str, label: str) -> None:
    weights = os.path.join(path, "model.safetensors")
    if not os.path.isfile(weights):
        raise FileNotFoundError(
            f"未找到 {label} 权重：{weights}\n请先运行：bash scripts/finetune_000338.sh"
        )


def _stage_upload_dir(local_dir: str, card_path: str) -> str:
    """复制权重目录并用 Model Card 替换 README"""
    tmp = tempfile.mkdtemp(prefix="kronos_hf_upload_")
    for name in os.listdir(local_dir):
        src = os.path.join(local_dir, name)
        dst = os.path.join(tmp, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    shutil.copy2(card_path, os.path.join(tmp, "README.md"))
    return tmp


def upload_one(api, local_dir: str, card_path: str, repo_id: str, dry_run: bool) -> None:
    _check_local(local_dir, repo_id)
    staged = _stage_upload_dir(local_dir, card_path)
    try:
        print(f"\n{'[DRY-RUN] ' if dry_run else ''}上传 → {repo_id}")
        if dry_run:
            print(f"  本地目录: {local_dir}")
            print(f"  文件: {os.listdir(staged)}")
            return
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=staged,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload 000338 daily qfq finetuned weights",
        )
        print(f"  ✅ https://huggingface.co/{repo_id}")
    finally:
        shutil.rmtree(staged, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload 000338 finetuned Kronos weights to Hugging Face")
    parser.add_argument("--dry-run", action="store_true", help="只检查路径，不上传")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import LocalTokenNotFoundError
    except ImportError as e:
        raise SystemExit("请先安装：pip install huggingface_hub") from e

    if not args.dry_run:
        try:
            user = HfApi().whoami()["name"]
            print(f"已登录 Hugging Face：{user}")
        except LocalTokenNotFoundError:
            raise SystemExit(
                "未登录 Hugging Face。请先执行：\n"
                "  huggingface-cli login\n"
                "或设置环境变量：\n"
                "  export HF_TOKEN=hf_xxxxxxxx"
            )

    api = HfApi()
    upload_one(api, TOKENIZER_LOCAL, CARD_TOKENIZER, HF_TOKENIZER_REPO, args.dry_run)
    upload_one(api, MODEL_LOCAL, CARD_MODEL, HF_MODEL_REPO, args.dry_run)

    if not args.dry_run:
        print("\n全部上传完成。他人可使用：")
        print(f"  Tokenizer: {HF_TOKENIZER_REPO}")
        print(f"  Model:     {HF_MODEL_REPO}")
        print("  python examples/predict_000338_finetuned.py --from-hub")


if __name__ == "__main__":
    main()
