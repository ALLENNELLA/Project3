#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 logs_abcd_5seeds 解析 random100 / full 的 Fine-tuned CER（按组、target_day、seed 聚合）。"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import DefaultDict, Dict, List, Optional, Tuple

CER_RE = re.compile(r"✅ Fine-tuned CER:\s*([\d.]+)")
NAME_RANDOM = re.compile(r"^random100_([A-D])_seed(\d+)_day(\d+)\.log$")
NAME_FULL = re.compile(r"^full_([A-D])_seed(\d+)_day(\d+)\.log$")

Key = Tuple[str, int]  # (group, target_day)

ORDER: List[Tuple[str, List[int]]] = [
    ("A", [3]),
    ("B", [6, 7]),
    ("C", [8, 9, 10, 11, 12]),
    ("D", [13, 14, 15, 16, 17, 18, 19, 20, 21]),
]


def _parse_cer(log_text: str) -> Optional[float]:
    m = CER_RE.search(log_text)
    return float(m.group(1)) if m else None


def collect(log_dir: Path) -> Tuple[Dict[str, DefaultDict[Key, List[float]]], Dict[str, List[str]]]:
    data: Dict[str, DefaultDict[Key, List[float]]] = {
        "random100": defaultdict(list),
        "full": defaultdict(list),
    }
    missing = {"random100": [], "full": []}

    for p in sorted(log_dir.glob("*.log")):
        mr, mf = NAME_RANDOM.match(p.name), NAME_FULL.match(p.name)
        text = p.read_text(encoding="utf-8", errors="replace")
        if mr:
            g, day = mr.group(1), int(mr.group(3))
            cer = _parse_cer(text)
            if cer is None:
                missing["random100"].append(p.name)
            else:
                data["random100"][(g, day)].append(cer)
        elif mf:
            g, day = mf.group(1), int(mf.group(3))
            cer = _parse_cer(text)
            if cer is None:
                missing["full"].append(p.name)
            else:
                data["full"][(g, day)].append(cer)

    return data, missing


def fmt(cers: List[float]) -> str:
    if not cers:
        return "—"
    m = mean(cers)
    s = stdev(cers) if len(cers) > 1 else 0.0
    return f"{m:.4f} ± {s:.4f} (n={len(cers)})"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize random100/full from logs_abcd_5seeds.")
    ap.add_argument(
        "--log_dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "outputs"
        / "automated_experiments"
        / "logs_abcd_5seeds",
    )
    ap.add_argument("--md", action="store_true", help="Print Markdown tables to stdout.")
    args = ap.parse_args()
    data, missing = collect(args.log_dir)

    if missing["random100"] or missing["full"]:
        print("Missing CER in logs:")
        if missing["random100"]:
            print("  random100:", ", ".join(missing["random100"]))
        if missing["full"]:
            print("  full:", ", ".join(missing["full"]))
        print()

    if not args.md:
        for gname, days in ORDER:
            for d in days:
                k = (gname, d)
                print(
                    f"{gname} day{d}: random100 {fmt(data['random100'][k])} | full {fmt(data['full'][k])}"
                )
        return

    print(
        "Fine-tuned CER 来自各任务日志中的 `✅ Fine-tuned CER:` 行；"
        "与 `outputs/model_test/.../random/` 无冲突问题。"
    )
    print()
    for gname, days in ORDER:
        print(f"#### 组 {gname}")
        print()
        print("| target_day | random100 (Mean ± Std, n) | full (Mean ± Std, n) |")
        print("|------------|---------------------------|----------------------|")
        for d in days:
            k = (gname, d)
            print(f"| {d} | {fmt(data['random100'][k])} | {fmt(data['full'][k])} |")
        print()


if __name__ == "__main__":
    main()
