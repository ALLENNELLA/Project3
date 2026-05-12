#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 ABCD 组导出 SLPE 明细表：
- 使用对应组的 Model A（conformer-{N}days-best）
- 对组内每个 target day 的 train split 计算 SLPE
- 导出每句的 SLPE 分数、排名、文本标签、音素序列
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from model_b_data_module import _resolve_data_file_by_day
from src.model_a.config import SESSION_NAMES_CHRONOLOGICAL
from src.model_a.get_model import get_model
from src.utils.dataset import SpeechDataset
from src.utils.sample_selection import PHONEME_VOCAB
from src.utils.slpe import compute_slpe_batch


GROUPS = {
    "A": {"eval_ndays": 2, "days": [3]},
    "B": {"eval_ndays": 5, "days": [6, 7]},
    "C": {"eval_ndays": 7, "days": [8, 9, 10, 11, 12]},
    "D": {"eval_ndays": 12, "days": [13, 14, 15, 16, 17, 18, 19, 20, 21]},
}


def _padding(batch):
    x, y, x_lens, y_lens, days = zip(*batch)
    x_padded = pad_sequence(x, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)
    return (
        x_padded,
        y_padded,
        torch.stack(x_lens),
        torch.stack(y_lens),
        torch.stack(days),
    )


def _load_model_a(model_a_path: Path, device: str):
    config_path = model_a_path / "config.pkl"
    weights_path = model_a_path / "modelWeights.pth"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Model A 文件缺失: {model_a_path}")

    with open(config_path, "rb") as f:
        config = pickle.load(f)
    model = get_model(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model.eval()
    return model


def _ids_to_phoneme_text(phoneme_ids: np.ndarray) -> str:
    tokens: List[str] = []
    for idx in phoneme_ids:
        idx_int = int(idx)
        if 0 <= idx_int < len(PHONEME_VOCAB):
            tokens.append(PHONEME_VOCAB[idx_int])
        else:
            tokens.append(f"UNK_{idx_int}")
    return " ".join(tokens)


def _compute_day_rows(
    *,
    group: str,
    day: int,
    eval_ndays: int,
    model,
    dataset_path: Path,
    batch_size: int,
    device: str,
) -> List[Dict]:
    with open(dataset_path, "rb") as f:
        loaded = pickle.load(f)
    train_dataset = SpeechDataset(loaded["train"], transform=None)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_padding,
    )

    slpe_scores = compute_slpe_batch(model, loader, device=device, blank=0)

    # 从 loader 中按同一顺序抽取截断后的标签音素序列
    phoneme_ids_list: List[np.ndarray] = []
    with torch.no_grad():
        for _, y, _, y_len, _ in loader:
            for i in range(len(y)):
                phoneme_ids_list.append(y[i, : y_len[i]].cpu().numpy())

    if len(slpe_scores) != len(phoneme_ids_list) or len(slpe_scores) != len(train_dataset):
        raise RuntimeError(
            f"样本数量不一致: slpe={len(slpe_scores)}, phoneme={len(phoneme_ids_list)}, dataset={len(train_dataset)}"
        )

    order = np.argsort(-slpe_scores)  # 分数越高排名越前
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(slpe_scores) + 1)

    session_names_sorted = sorted(list(SESSION_NAMES_CHRONOLOGICAL))
    session_name = session_names_sorted[day - 1] if 1 <= day <= len(session_names_sorted) else "unknown"

    rows: List[Dict] = []
    for idx in range(len(slpe_scores)):
        transcription = train_dataset.get_transcription(idx)
        phoneme_ids = phoneme_ids_list[idx]
        rows.append(
            {
                "group": group,
                "target_day": day,
                "session_name": session_name,
                "model_a_eval_ndays": eval_ndays,
                "sample_index": idx,
                "slpe_rank_desc": int(ranks[idx]),
                "slpe_score": float(slpe_scores[idx]),
                "transcription": "" if transcription is None else str(transcription),
                "phoneme_ids": " ".join(str(int(x)) for x in phoneme_ids),
                "phoneme_text": _ids_to_phoneme_text(phoneme_ids),
            }
        )
    return rows


def _write_rows_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "group",
        "target_day",
        "session_name",
        "model_a_eval_ndays",
        "sample_index",
        "slpe_rank_desc",
        "slpe_score",
        "transcription",
        "phoneme_ids",
        "phoneme_text",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ABCD SLPE score tables.")
    parser.add_argument("--groups", nargs="+", choices=sorted(GROUPS.keys()), default=sorted(GROUPS.keys()))
    parser.add_argument(
        "--model_train_selected_root",
        type=str,
        default="/root/25S151115/project3/outputs/model_train_selected_3",
        help="Model A 根目录，内含 conformer-{N}days-best",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/root/25S151115/project3",
        help="项目根目录（用于解析 data/dataMMDD）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/25S151115/project3/outputs/analysis/slpe_tables",
        help="SLPE 表格输出目录",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model_root = Path(args.model_train_selected_root)
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    device = args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu"

    all_rows: List[Dict] = []
    for group in args.groups:
        gcfg = GROUPS[group]
        eval_ndays = gcfg["eval_ndays"]
        model_a_path = model_root / f"conformer-{eval_ndays}days-best"
        print(f"\n=== Group {group} | Model A: {model_a_path} ===")
        model = _load_model_a(model_a_path, device)

        group_rows: List[Dict] = []
        for day in gcfg["days"]:
            data_file = _resolve_data_file_by_day(day)
            dataset_path = base_dir / "data" / data_file
            if not dataset_path.exists():
                raise FileNotFoundError(f"数据文件不存在: {dataset_path}")

            print(f"  - Day {day}: computing SLPE on {dataset_path.name} ...")
            day_rows = _compute_day_rows(
                group=group,
                day=day,
                eval_ndays=eval_ndays,
                model=model,
                dataset_path=dataset_path,
                batch_size=args.batch_size,
                device=device,
            )
            group_rows.extend(day_rows)

            day_csv = output_dir / f"group_{group}" / f"day_{day}_slpe_table.csv"
            _write_rows_csv(day_rows, day_csv)
            print(f"    saved: {day_csv} ({len(day_rows)} rows)")

        group_csv = output_dir / f"group_{group}" / f"group_{group}_all_days_slpe_table.csv"
        _write_rows_csv(group_rows, group_csv)
        print(f"  group merged saved: {group_csv} ({len(group_rows)} rows)")
        all_rows.extend(group_rows)

    all_csv = output_dir / "abcd_all_groups_slpe_table.csv"
    _write_rows_csv(all_rows, all_csv)
    print(f"\n✅ all groups merged saved: {all_csv} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()

