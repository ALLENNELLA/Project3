#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对 model_train 中各「训练 N 天」的 Model A，在按时间排序的第 (N+1)…24 天上，
计算单天 pickle 中 train 划分的平均 SLPE。

- 日历天顺序使用 config.SESSION_NAMES_CHRONOLOGICAL（第 1 天 = data0428）。
- 迁移跨度 transfer_span = chronological_day_index - n_train_days（未见过该天的模型在该天上评估）。
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.model_a.config import SESSION_NAMES_CHRONOLOGICAL
from src.model_a.evaluate import get_eval_dataset_loaders, load_trained_model_a
from src.utils.slpe import compute_slpe_batch


def session_to_single_day_dataset(base_dir: str, session_name: str) -> str:
    parts = session_name.split(".")
    if len(parts) < 3:
        raise ValueError(f"无法解析 session: {session_name}")
    return os.path.join(base_dir, "data", f"data{parts[-2]}{parts[-1]}")


def mean_slpe_train_split(
    model,
    dataset_path: str,
    batch_size: int,
    device: str,
    show_progress: bool,
) -> tuple[float, int]:
    """在单个 session 的 pickle 上，对 train 划分计算样本级 SLPE 的均值。"""
    loader, _ = get_eval_dataset_loaders(
        dataset_path, batch_size, eval_split="train", num_workers=0
    )
    scores = compute_slpe_batch(
        model, loader, device=device, blank=0, show_progress=show_progress
    )
    n = int(scores.shape[0])
    return float(np.mean(scores)), n


def parse_ndays_from_dirname(folder_name: str) -> int | None:
    m = re.match(r"^.+-(\d+)days-seed\d+$", folder_name)
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser(
        description="train 划分平均 SLPE × 训练天数 × 日历天（仅评估天序号 > 训练天数）"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/root/25S151115/project3",
        help="项目根目录",
    )
    parser.add_argument(
        "--model_train_dir",
        type=str,
        default=None,
        help="默认 {base_dir}/outputs/model_train",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="默认 {base_dir}/outputs/slpe_train_transfer_by_day.csv",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--n_days_list",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5, 7, 9, 10, 12],
        help="训练天数（与目录 conformer-Ndays-seed* 一致）",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="随机种子编号",
    )
    parser.add_argument(
        "--model_substr",
        type=str,
        default="conformer",
        help="只跑目录名包含该子串的模型（默认 conformer）",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="追加写入 CSV",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="关闭 SLPE 批次内 tqdm",
    )
    parser.add_argument(
        "--chronological_days",
        type=int,
        nargs="+",
        default=None,
        help="只评估这些日历天序号（1-based）；默认从 n_train+1 到 24",
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.expanduser(args.base_dir))
    model_train_root = args.model_train_dir or os.path.join(
        base_dir, "outputs", "model_train"
    )
    output_csv = args.output_csv or os.path.join(
        base_dir, "outputs", "slpe_train_transfer_by_day.csv"
    )

    sessions = list(SESSION_NAMES_CHRONOLOGICAL)
    n_cal = len(sessions)
    if n_cal != 24:
        print(f"警告: SESSION_NAMES_CHRONOLOGICAL 长度为 {n_cal}，预期 24。")

    fieldnames = [
        "n_train_days",
        "seed",
        "chronological_day",
        "session_name",
        "transfer_span",
        "mean_slpe_train",
        "n_samples",
        "dataset_path",
        "model_dir",
    ]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    write_header = not (args.append and os.path.isfile(output_csv))
    mode = "a" if args.append else "w"

    total_rows = 0
    with open(output_csv, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        for n_train in args.n_days_list:
            for seed in args.seeds:
                folder = f"{args.model_substr}-{n_train}days-seed{seed}"
                model_dir = os.path.join(model_train_root, folder)
                wpath = os.path.join(model_dir, "modelWeights.pth")
                cpath = os.path.join(model_dir, "config.pkl")
                if not (os.path.isfile(wpath) and os.path.isfile(cpath)):
                    print(f"跳过（缺少权重或配置）: {model_dir}")
                    continue

                print(f"加载模型: {model_dir}")
                model, _cfg = load_trained_model_a(model_dir, device=args.device)

                if args.chronological_days is not None:
                    days_loop = sorted(set(args.chronological_days))
                else:
                    days_loop = list(range(n_train + 1, n_cal + 1))

                for day_idx in days_loop:
                    if day_idx <= n_train:
                        print(
                            f"  跳过 day {day_idx}（需 > 训练天数 {n_train}）"
                        )
                        continue
                    if day_idx < 1 or day_idx > n_cal:
                        print(f"  跳过非法 day {day_idx}（有效 1..{n_cal}）")
                        continue
                    session = sessions[day_idx - 1]
                    ds_path = session_to_single_day_dataset(base_dir, session)
                    if not os.path.isfile(ds_path):
                        print(f"  跳过无文件: {ds_path}")
                        continue

                    mean_slpe, n_samp = mean_slpe_train_split(
                        model,
                        ds_path,
                        args.batch_size,
                        args.device,
                        show_progress=not args.quiet,
                    )
                    transfer_span = day_idx - n_train
                    row = {
                        "n_train_days": n_train,
                        "seed": seed,
                        "chronological_day": day_idx,
                        "session_name": session,
                        "transfer_span": transfer_span,
                        "mean_slpe_train": f"{mean_slpe:.8f}",
                        "n_samples": n_samp,
                        "dataset_path": ds_path,
                        "model_dir": model_dir,
                    }
                    w.writerow(row)
                    f.flush()
                    total_rows += 1
                    print(
                        f"  day {day_idx}/{n_cal} span={transfer_span} "
                        f"mean_slpe(train)={mean_slpe:.6f} n={n_samp}"
                    )

                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"完成，写入 {total_rows} 行 -> {output_csv}")


if __name__ == "__main__":
    main()
