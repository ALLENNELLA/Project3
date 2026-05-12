#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对 outputs/model_train 下每个已训练 Model A，在全部单天数据上计算 test 集 CER。

单天数据路径与 config.get_base_config() 中排序后的 sessionNames 一致：
  t12.2022.MM.DD -> {base_dir}/data/dataMMDD
聚合数据 ptDecoder_ctcN 对应前 N 个 session（排序后），本脚本不读聚合文件，只读单天文件。
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import re
import sys
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
    """例如 conformer-7days-seed3 -> model_name, n_days, seed"""
    m = re.match(r"^(.+)-(\d+)days-seed(\d+)$", folder_name)
    if not m:
        return None
    return {
        "model_name": m.group(1),
        "n_days_train": int(m.group(2)),
        "seed": int(m.group(3)),
    }


def main():
    parser = argparse.ArgumentParser(description="batch test CER (test split) for all model_train models × all days")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/root/25S151115/project3",
        help="项目根目录（含 data/ 与 outputs/）",
    )
    parser.add_argument(
        "--model_train_dir",
        type=str,
        default=None,
        help="model_train 目录，默认 {base_dir}/outputs/model_train",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="结果 CSV，默认 {base_dir}/outputs/model_train_all_days_cer.csv",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--day_indices",
        type=int,
        nargs="+",
        default=None,
        help="仅评估这些 day 编号（1-based，与排序后 sessionNames 下标+1 一致）；默认全部天",
    )
    parser.add_argument(
        "--model_glob",
        type=str,
        default=None,
        help="若设置，只评估目录名包含该子串的模型（例如 conformer-7days）",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="追加写入 CSV（需与已有表头一致）；用于断点续跑时避免重复行请自行删重复或换输出文件",
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    model_train_root = args.model_train_dir or os.path.join(base_dir, "outputs", "model_train")
    output_csv = args.output_csv or os.path.join(base_dir, "outputs", "model_train_all_days_cer.csv")

    cfg = get_base_config()
    session_names: list[str] = list(cfg["sessionNames"])
    n_sessions = len(session_names)

    if args.day_indices:
        days_to_run = sorted(set(args.day_indices))
        for d in days_to_run:
            if d < 1 or d > n_sessions:
                raise SystemExit(f"day_indices 必须在 1..{n_sessions} 之间，收到 {d}")
    else:
        days_to_run = list(range(1, n_sessions + 1))

    model_dirs = list_model_dirs(model_train_root)
    if args.model_glob:
        model_dirs = [p for p in model_dirs if args.model_glob in os.path.basename(p)]

    if not model_dirs:
        print(f"未在 {model_train_root} 找到含 modelWeights.pth 与 config.pkl 的子目录", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    write_header = not (args.append and os.path.isfile(output_csv))
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

    mode = "a" if args.append and os.path.isfile(output_csv) else "w"
    with open(output_csv, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        total = len(model_dirs) * len(days_to_run)
        done = 0
        ts = datetime.now().isoformat(timespec="seconds")

        for model_dir in model_dirs:
            folder = os.path.basename(model_dir)
            meta = parse_model_dir_name(folder) or {}
            try:
                model, cfg_m = load_trained_model_a(model_dir, args.device)
            except Exception as e:
                for day_1 in days_to_run:
                    done += 1
                    session_name = session_names[day_1 - 1]
                    dataset_path = session_to_single_day_dataset(base_dir, session_name)
                    w.writerow(
                        {
                            "timestamp": ts,
                            "model_dir": model_dir,
                            "folder_name": folder,
                            "parsed_model_name": meta.get("model_name", ""),
                            "parsed_n_days_train": meta.get("n_days_train", ""),
                            "parsed_seed": meta.get("seed", ""),
                            "day_index_1based": day_1,
                            "session_name": session_name,
                            "dataset_path": dataset_path,
                            "test_cer": "",
                            "test_avg_loss": "",
                            "error": f"load_model:{repr(e)}",
                        }
                    )
                    f.flush()
                    print(f"[{done}/{total}] FAIL load {folder}: {e}", flush=True)
                continue

            for day_1 in days_to_run:
                done += 1
                session_name = session_names[day_1 - 1]
                dataset_path = session_to_single_day_dataset(base_dir, session_name)
                row = {
                    "timestamp": ts,
                    "model_dir": model_dir,
                    "folder_name": folder,
                    "parsed_model_name": meta.get("model_name", ""),
                    "parsed_n_days_train": meta.get("n_days_train", ""),
                    "parsed_seed": meta.get("seed", ""),
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
                    print(f"[{done}/{total}] SKIP {folder} day{day_1} {session_name}: 无数据文件", flush=True)
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
                err = row["error"]
                if err:
                    print(f"[{done}/{total}] FAIL {folder} day{day_1}: {err}", flush=True)
                else:
                    print(
                        f"[{done}/{total}] OK {folder} day{day_1} {session_name} CER={row['test_cer']}",
                        flush=True,
                    )

            del model
            gc.collect()
            if args.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"已写入: {output_csv}")


if __name__ == "__main__":
    main()
