#!/usr/bin/env python3
"""从 repro 日志目录解析「训练最佳CER」，按天×数据量×模型对 selection_seed(ssel) 聚合（不读测试集 CER，不写文件）。"""
from __future__ import annotations

import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path

# 例如 random150_A_day3_ssel0.log、random150_B1_day8_ssel0.log、random400_C2_day14_ssel0.log
LOG_NAME_RE = re.compile(
    r"^random(\d+)_([A-Za-z0-9]+)_day(\d+)_ssel(\d+)\.log$", re.IGNORECASE
)
TRAIN_BEST_RE = re.compile(r"训练最佳CER:\s*([\d.]+)")


def extract_train_best_cer(text: str) -> float | None:
    matches = TRAIN_BEST_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "log_root",
        type=Path,
        nargs="?",
        default=Path(
            "/root/25S151115/project3/outputs/repro_runs/my_repro_v6/logs/run_20260414_025738"
        ),
        help="含 random*/random*_*.log 的日志根目录",
    )
    p.add_argument(
        "--brief",
        action="store_true",
        help="只输出每组聚合一行，不列出各 ssel 明细",
    )
    args = p.parse_args()
    root: Path = args.log_root

    # (day, n_samples, model) -> list of (ssel, cer, path)
    grouped: dict[tuple[int, int, str], list[tuple[int, float, str]]] = defaultdict(list)
    bad: list[str] = []

    for log_path in sorted(root.rglob("*.log")):
        m = LOG_NAME_RE.match(log_path.name)
        if not m:
            continue
        n_samp = int(m.group(1))
        model = m.group(2).upper()
        day = int(m.group(3))
        ssel = int(m.group(4))
        text = log_path.read_text(encoding="utf-8", errors="replace")
        cer = extract_train_best_cer(text)
        key = (day, n_samp, model)
        if cer is None:
            bad.append(f"[缺训练最佳CER] {log_path}")
            continue
        grouped[key].append((ssel, cer, str(log_path)))

    print("=" * 88)
    print("训练最佳 CER 汇总（仅解析日志行「训练最佳CER」，忽略 Evaluation / 最终微调CER）")
    print(f"目录: {root.resolve()}")
    print("=" * 88)

    keys = sorted(grouped.keys(), key=lambda k: (k[0], k[1], k[2]))
    for (day, n_samp, model) in keys:
        items = sorted(grouped[(day, n_samp, model)], key=lambda x: x[0])
        cers = [c for _, c, _ in items]
        mean = statistics.fmean(cers)
        std = statistics.stdev(cers) if len(cers) > 1 else 0.0
        print(
            f"day={day:2d}  n={n_samp:3d}  model={model}  "
            f"n_seed={len(cers)}  mean={mean:.4f}  std={std:.4f}  "
            f"min={min(cers):.4f}  max={max(cers):.4f}"
        )
        if not args.brief:
            for ssel, cer, path in items:
                print(f"    ssel{ssel}: {cer:.4f}  ({path})")

    if bad:
        print("\n" + "-" * 88)
        print(f"解析失败或未找到「训练最佳CER」: {len(bad)} 个文件")
        for line in bad[:50]:
            print(line)
        if len(bad) > 50:
            print(f"    ... 另有 {len(bad) - 50} 条省略")


if __name__ == "__main__":
    main()
