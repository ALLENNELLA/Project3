#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对 outputs/model_train 下每个 Model A：若训练天数为 N，仅在「排序后 sessionNames」的
第 N+1、N+2 天（1-based）单天 test 上算 CER，用于直接迁移指标 cer_{N+1}+cer_{N+2}。
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.model_a.config import get_base_config
from src.model_a.evaluate import evaluate_cer_on_dataset, load_trained_model_a


def session_to_single_day_dataset(base_dir: str, session_name: str) -> str:
    parts = session_name.split(".")
    if len(parts) < 3:
        raise ValueError(f"无法解析 session: {session_name}")
    return os.path.join(base_dir, "data", f"data{parts[-2]}{parts[-1]}")


def list_model_dirs(model_train_root: str) -> list[str]:
    out = []
    if not os.path.isdir(model_train_root):
        return out
    for name in sorted(os.listdir(model_train_root)):
        d = os.path.join(model_train_root, name)
        if not os.path.isdir(d):
            continue
        if os.path.isfile(os.path.join(d, "modelWeights.pth")) and os.path.isfile(
            os.path.join(d, "config.pkl")
        ):
            out.append(d)
    return out


def parse_model_dir_name(folder_name: str) -> dict | None:
    m = re.match(r"^(.+)-(\d+)days-seed(\d+)$", folder_name)
    if not m:
        return None
    return {
        "model_name": m.group(1),
        "n_days_train": int(m.group(2)),
        "seed": int(m.group(3)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Model A (N days train): CER on days N+1 and N+2 only (test split)"
    )
    parser.add_argument("--base_dir", type=str, default="/root/25S151115/project3")
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
        help="默认 {base_dir}/outputs/model_train_next2_transfer_cer.csv",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--model_glob",
        type=str,
        default=None,
        help="只评估目录名包含该子串的模型",
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    model_train_root = args.model_train_dir or os.path.join(base_dir, "outputs", "model_train")
    output_csv = args.output_csv or os.path.join(
        base_dir, "outputs", "model_train_next2_transfer_cer.csv"
    )

    cfg = get_base_config()
    session_names: list[str] = list(cfg["sessionNames"])
    n_sessions = len(session_names)

    model_dirs = list_model_dirs(model_train_root)
    if args.model_glob:
        model_dirs = [p for p in model_dirs if args.model_glob in os.path.basename(p)]

    model_dirs = [p for p in model_dirs if parse_model_dir_name(os.path.basename(p))]

    if not model_dirs:
        print(f"未在 {model_train_root} 找到符合条件的模型目录", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    fieldnames = [
        "timestamp",
        "model_dir",
        "folder_name",
        "parsed_model_name",
        "parsed_n_days_train",
        "parsed_seed",
        "day_index_1based",
        "session_name",
        "dataset_path",
        "test_cer",
        "test_avg_loss",
        "error",
    ]
    ts = datetime.now().isoformat(timespec="seconds")

    # per-model: two days only
    total_evals = 0
    for model_dir in model_dirs:
        meta = parse_model_dir_name(os.path.basename(model_dir))
        assert meta is not None
        n = meta["n_days_train"]
        d1, d2 = n + 1, n + 2
        if d2 > n_sessions:
            print(
                f"SKIP {os.path.basename(model_dir)}: 需要 day{d1},day{d2} 但只有 {n_sessions} 个 session",
                file=sys.stderr,
            )
            continue
        total_evals += 2

    done = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for model_dir in model_dirs:
            folder = os.path.basename(model_dir)
            meta = parse_model_dir_name(folder)
            assert meta is not None
            n = meta["n_days_train"]
            days_pair = (n + 1, n + 2)
            if days_pair[1] > n_sessions:
                continue

            try:
                model, cfg_m = load_trained_model_a(model_dir, args.device)
            except Exception as e:
                for day_1 in days_pair:
                    done += 1
                    session_name = session_names[day_1 - 1]
                    dataset_path = session_to_single_day_dataset(base_dir, session_name)
                    w.writerow(
                        {
                            "timestamp": ts,
                            "model_dir": model_dir,
                            "folder_name": folder,
                            "parsed_model_name": meta["model_name"],
                            "parsed_n_days_train": n,
                            "parsed_seed": meta["seed"],
                            "day_index_1based": day_1,
                            "session_name": session_name,
                            "dataset_path": dataset_path,
                            "test_cer": "",
                            "test_avg_loss": "",
                            "error": f"load_model:{repr(e)}",
                        }
                    )
                    f.flush()
                    print(f"[{done}/{total_evals}] FAIL load {folder}: {e}", flush=True)
                continue

            for day_1 in days_pair:
                done += 1
                session_name = session_names[day_1 - 1]
                dataset_path = session_to_single_day_dataset(base_dir, session_name)
                row = {
                    "timestamp": ts,
                    "model_dir": model_dir,
                    "folder_name": folder,
                    "parsed_model_name": meta["model_name"],
                    "parsed_n_days_train": n,
                    "parsed_seed": meta["seed"],
                    "day_index_1based": day_1,
                    "session_name": session_name,
                    "dataset_path": dataset_path,
                    "test_cer": "",
                    "test_avg_loss": "",
                    "error": "",
                }
                if not os.path.isfile(dataset_path):
                    row["error"] = "dataset_missing"
                    w.writerow(row)
                    f.flush()
                    print(f"[{done}/{total_evals}] SKIP {folder} day{day_1}: 无数据", flush=True)
                    continue
                try:
                    avg_loss, cer = evaluate_cer_on_dataset(
                        model,
                        cfg_m,
                        dataset_path=dataset_path,
                        batch_size=args.batch_size,
                        eval_split="test",
                        device=args.device,
                        verbose=False,
                    )
                    row["test_cer"] = f"{cer:.8f}"
                    row["test_avg_loss"] = f"{avg_loss:.8f}"
                except Exception as e:
                    row["error"] = repr(e)
                w.writerow(row)
                f.flush()
                if row["error"]:
                    print(f"[{done}/{total_evals}] FAIL {folder} day{day_1}: {row['error']}", flush=True)
                else:
                    print(
                        f"[{done}/{total_evals}] OK {folder} day{day_1} CER={row['test_cer']}",
                        flush=True,
                    )

            del model
            gc.collect()
            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"已写入: {output_csv}")

    # 汇总：每个 N 的 cer 之和，top2 seed
    by_n_seed: dict[int, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    with open(output_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("error"):
                continue
            if not row.get("test_cer"):
                continue
            nn = int(row["parsed_n_days_train"])
            seed = int(row["parsed_seed"])
            d = int(row["day_index_1based"])
            by_n_seed[nn][seed][d] = float(row["test_cer"])

    print("\n=== 各训练天数 N：在 day N+1、N+2 上 CER 之和，40 seed 中最好 / 次好 ===\n")
    for n in sorted(by_n_seed.keys()):
        sums: list[tuple[float, int, float, float]] = []
        for seed, dm in sorted(by_n_seed[n].items()):
            need = (n + 1, n + 2)
            if need[0] not in dm or need[1] not in dm:
                continue
            c1, c2 = dm[need[0]], dm[need[1]]
            sums.append((c1 + c2, seed, c1, c2))
        sums.sort(key=lambda x: x[0])
        if len(sums) < 2:
            print(f"N={n}: 有效 seed 数不足2 (got {len(sums)})")
            continue
        (s0, seed0, a0, b0), (s1, seed1, a1, b1) = sums[0], sums[1]
        print(
            f"N={n:2d}  (评 day{n+1:2d}+day{n+2:2d})  "
            f"最佳 seed{seed0}: sum={s0:.6f} (cer{n+1}={a0:.6f}, cer{n+2}={b0:.6f})  "
            f"| 次佳 seed{seed1}: sum={s1:.6f} (cer{n+1}={a1:.6f}, cer{n+2}={b1:.6f})"
        )


if __name__ == "__main__":
    main()
