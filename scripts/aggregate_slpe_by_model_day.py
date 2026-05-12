#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 slpe_train_transfer_by_day*.csv 按 (n_train_days, chronological_day) 对 seed 聚合。
输出列：n_train_days, chronological_day, mean_slpe_train, std_slpe_train。
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict


def main():
    p = argparse.ArgumentParser(description="按模型训练天数 + 日历天聚合 SLPE（跨 seed）")
    p.add_argument(
        "input_csv",
        nargs="?",
        default="/root/25S151115/project3/outputs/slpe_train_transfer_by_day_merged.csv",
        help="原始按 seed 展开的 CSV",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="默认与输入同目录，文件名为 <stem>_agg.csv",
    )
    args = p.parse_args()

    inp = os.path.abspath(args.input_csv)
    out = args.output
    if out is None:
        stem, _ = os.path.splitext(inp)
        out = stem + "_agg.csv"

    groups: dict[tuple[int, int], list[float]] = defaultdict(list)

    with open(inp, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            nd = int(row["n_train_days"])
            day = int(row["chronological_day"])
            key = (nd, day)
            groups[key].append(float(row["mean_slpe_train"]))

    fieldnames = [
        "n_train_days",
        "chronological_day",
        "mean_slpe_train",
        "std_slpe_train",
    ]

    rows_out = []
    for (nd, day) in sorted(groups.keys()):
        xs = groups[(nd, day)]
        m = statistics.mean(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        rows_out.append(
            {
                "n_train_days": nd,
                "chronological_day": day,
                "mean_slpe_train": f"{m:.8f}",
                "std_slpe_train": f"{sd:.8f}",
            }
        )

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print(f"写入 {len(rows_out)} 行 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
