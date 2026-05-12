#!/usr/bin/env python3
"""汇总 model_test 下各 7->N 迁移实验中指定 model_b 策略的 Fine-tuned CER（按 seed 平均）。"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# 用户关心的策略目录名后缀（与 outputs/model_test/<pair>/ 下子目录一致）
DEFAULT_STRATEGIES = [
    "model_b_modelb_sigma001_cosine_tmax1p0_3p0e-4_to_1p0e-5_noclip_abn256_top100",
    "model_b_modelb_sigma001_cosine_tmax1p0_3p0e-4_to_1p0e-5_noclip_abn256_top80_20",
    "model_b_modelb_sigma001_cosine_tmax1p0_3p0e-4_to_1p0e-5_noclip_abn256_top60_40",
    "model_b_modelb_sigma001_linear_3p0e-4_to_1p0e-5_noclip_abn256_top60_40",
    "model_b_modelb_sigma001_linear_3p0e-4_to_1p0e-5_noclip_abn256_top80_20",
    "model_b_modelb_sigma001_linear_3p0e-4_to_1p0e-5_noclip_abn256_top100",
]

SHORT_NAMES = {
    "model_b_modelb_sigma001_cosine_tmax1p0_3p0e-4_to_1p0e-5_noclip_abn256_top100": "cosine + top100",
    "model_b_modelb_sigma001_cosine_tmax1p0_3p0e-4_to_1p0e-5_noclip_abn256_top80_20": "cosine + top80/20",
    "model_b_modelb_sigma001_cosine_tmax1p0_3p0e-4_to_1p0e-5_noclip_abn256_top60_40": "cosine + top60/40",
    "model_b_modelb_sigma001_linear_3p0e-4_to_1p0e-5_noclip_abn256_top60_40": "linear + top60/40",
    "model_b_modelb_sigma001_linear_3p0e-4_to_1p0e-5_noclip_abn256_top80_20": "linear + top80/20",
    "model_b_modelb_sigma001_linear_3p0e-4_to_1p0e-5_noclip_abn256_top100": "linear + top100",
}


def parse_cer(path: Path) -> float | None:
    with path.open() as f:
        for line in f:
            if "Fine-tuned CER:" in line:
                return float(line.split(":")[-1].strip())
    return None


def collect_strategy(
    root: Path, pair: str, strategy_dir: str
) -> tuple[list[float], int]:
    """返回 (各 seed 的 CER 列表, 缺失 finetuned_cer 的 seed 数)."""
    exp = root / pair / strategy_dir
    cers: list[float] = []
    missing = 0
    if not exp.is_dir():
        return [], 0
    for seed_dir in sorted(exp.glob("seed*")):
        cer_path = seed_dir / "results" / "finetuned_cer.txt"
        if not cer_path.is_file():
            missing += 1
            continue
        cer = parse_cer(cer_path)
        if cer is not None:
            cers.append(cer)
    return cers, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "model_test",
    )
    ap.add_argument(
        "--pairs",
        nargs="*",
        default=["7-8", "7-9", "7-10", "7-11", "7-12"],
    )
    ap.add_argument("--csv", type=Path, help="可选：写入 CSV")
    args = ap.parse_args()
    root: Path = args.root

    rows: list[dict[str, str | float | int]] = []
    for pair in args.pairs:
        for strat in DEFAULT_STRATEGIES:
            cers, missing = collect_strategy(root, pair, strat)
            n = len(cers)
            if n == 0:
                mean = float("nan")
                cmin = float("nan")
                cmax = float("nan")
            else:
                mean = sum(cers) / n
                cmin = min(cers)
                cmax = max(cers)
            rows.append(
                {
                    "pair": pair,
                    "strategy_short": SHORT_NAMES.get(strat, strat),
                    "strategy_dir": strat,
                    "n_seeds": n,
                    "mean_cer": mean,
                    "min_cer": cmin,
                    "max_cer": cmax,
                    "missing_cer_files": missing,
                }
            )

    # 控制台：按 pair 分组打印
    for pair in args.pairs:
        sub = [r for r in rows if r["pair"] == pair]
        print(f"\n=== {pair} (7天 -> {pair.split('-')[1]}天) ===")
        sub.sort(key=lambda r: (r["mean_cer"] if r["mean_cer"] == r["mean_cer"] else 999))
        for r in sub:
            m = r["mean_cer"]
            mean_s = f"{m:.6f}" if m == m else "nan"
            print(
                f"  {r['strategy_short']:<22}  mean={mean_s}  n={r['n_seeds']}  "
                f"min={r['min_cer']:.6f}  max={r['max_cer']:.6f}"
            )

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
