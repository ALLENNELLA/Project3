#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键脚本：运行带 PEFT (CABlock + AdaptFFN) 的 Model B 微调实验，并汇总 Day 8-12 结果。

说明：
- 并行与任务调度仍由现有的 `run_automated_experiments.py` 完成；
- 本脚本只是：
  1) 调用自动化实验脚本；
  2) 实验结束后扫描日志，生成 Day 8-12 的 CER + 重合率统计。
"""

import argparse
import subprocess
from pathlib import Path
from collections import defaultdict
import statistics
import re
import sys


def run_automated_experiments(args):
    """调用现有自动化脚本，执行训练 + 微调（含 PEFT 逻辑）"""
    scripts_dir = Path(__file__).parent
    cmd = [
        sys.executable,
        str(scripts_dir / "run_automated_experiments.py"),
        "--skip_model_a" if args.skip_model_a else "",
        "--num_gpus",
        str(args.num_gpus),
        "--num_samples",
        str(args.num_samples),
    ]

    # seeds
    if args.seeds:
        cmd.extend(["--seeds"] + [str(s) for s in args.seeds])
    # pretrained_ndays
    if args.pretrained_ndays:
        cmd.extend(["--pretrained_ndays"] + [str(n) for n in args.pretrained_ndays])
    # finetune days
    if args.finetune_days:
        cmd.extend(["--finetune_days"] + [str(d) for d in args.finetune_days])

    # 过滤掉空字符串
    cmd = [c for c in cmd if c]

    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=str(scripts_dir))
    if result.returncode != 0:
        raise RuntimeError(f"run_automated_experiments.py 运行失败 (exit={result.returncode})")


def extract_cer_and_overlap(days):
    """
    从 `outputs/automated_experiments/logs` 中提取
    - 各方法 (real_slpe / model_b / random) 的 CER
    - model_b 与 real_slpe 前 100 句的重合率
    """
    log_dir = Path("outputs/automated_experiments/logs")

    # CER: results[day][method] = [cer1, cer2, ...]
    results = defaultdict(lambda: defaultdict(list))
    # overlap: overlap_results[day] = [overlap%, ...]
    overlap_results = defaultdict(list)

    for day in days:
        for log_file in sorted(log_dir.glob(f"finetune_*_7_{day}_*.log")):
            content = log_file.read_text(encoding="utf-8", errors="ignore")

            # 方法名
            method = None
            if "finetune_real_slpe" in log_file.name:
                method = "real_slpe"
            elif "finetune_model_b" in log_file.name:
                method = "model_b"
                # 提取重合率
                overlap_match = re.search(
                    r"Model B选择的100句与真实SLPE前100句的重合率:\s*(\d+)/100 = ([\d.]+)%",
                    content,
                )
                if overlap_match:
                    overlap_ratio = float(overlap_match.group(2))
                    overlap_results[day].append(overlap_ratio)
            elif "finetune_random" in log_file.name:
                method = "random"
            else:
                continue

            # 提取 CER（兼容不同打印格式）
            cer = None
            m1 = re.search(r"\*\*Character Error Rate \(CER\):\s*([\d.]+)\*\*", content)
            if m1:
                cer = float(m1.group(1))
            else:
                m2 = re.search(r"Character Error Rate \(CER\):\s*([\d.]+)", content)
                if m2:
                    cer = float(m2.group(1))
                else:
                    m3 = re.search(r"最终微调CER:\s*([\d.]+)", content)
                    if m3:
                        cer = float(m3.group(1))

            if cer is not None and method is not None:
                results[day][method].append(cer)

    return results, overlap_results


def summarize_peft_results(days, results, overlap_results, output_path):
    """生成 Day 8-12 的 CER + 重合率汇总（Mean±Std），写入 markdown 文件。"""
    lines = []
    lines.append("=" * 80)
    lines.append("📊 Day 8-12 PEFT (CABlock + AdaptFFN) 微调结果统计")
    lines.append("=" * 80)
    lines.append("")

    # 完成情况
    lines.append("=" * 80)
    lines.append("📋 完成情况检查")
    lines.append("=" * 80)
    lines.append("")
    for day in days:
        lines.append(f"Day {day}:")
        for method in ["real_slpe", "model_b", "random"]:
            count = len(results[day][method])
            expected = 10 if method != "random" else 50
            status = "✅" if count == expected else f"⚠️ ({count}/{expected})"
            lines.append(f"  {method.upper().replace('_', ' '):<15s}: {status}")
        lines.append("")

    # CER Mean±Std 表
    lines.append("=" * 80)
    lines.append("📊 CER统计 (Mean±Std)")
    lines.append("=" * 80)
    lines.append("")

    header = ["方法"] + [f"Day {d}" for d in days]
    lines.append(f"{header[0]:<15s} " + " ".join([f"{h:<20s}" for h in header[1:]]))
    lines.append("-" * (15 + 20 * len(days)))

    for method in ["real_slpe", "model_b", "random"]:
        row = [method.upper().replace("_", " ")]
        for day in days:
            cers = results[day][method]
            if cers:
                avg = sum(cers) / len(cers)
                std = statistics.stdev(cers) if len(cers) > 1 else 0.0
                expected = 10 if method != "random" else 50
                if len(cers) < expected:
                    row.append(f"{avg:.4f}±{std:.4f} ({len(cers)}/{expected})")
                else:
                    row.append(f"{avg:.4f}±{std:.4f} (n={len(cers)})")
            else:
                row.append("N/A")
        lines.append(f"{row[0]:<15s} " + " ".join([f"{c:<20s}" for c in row[1:]]))

    # 重合率
    lines.append("")
    lines.append("=" * 80)
    lines.append("📊 Model B 与真实 SLPE 前100句重合率统计")
    lines.append("=" * 80)
    lines.append("")

    all_overlaps = []
    for day in days:
        overlaps = overlap_results[day]
        if overlaps:
            avg = sum(overlaps) / len(overlaps)
            std = statistics.stdev(overlaps) if len(overlaps) > 1 else 0.0
            expected = 10
            status = "" if len(overlaps) == expected else f" ({len(overlaps)}/{expected})"
            lines.append(f"Day {day}{status}:")
            lines.append(f"  平均重合率: {avg:.2f}% ± {std:.2f}%")
            lines.append(f"  范围: [{min(overlaps):.2f}%, {max(overlaps):.2f}%]")
            lines.append(f"  样本数: {len(overlaps)}")
            lines.append("")
            all_overlaps.extend(overlaps)

    if all_overlaps:
        avg_all = sum(all_overlaps) / len(all_overlaps)
        std_all = statistics.stdev(all_overlaps) if len(all_overlaps) > 1 else 0.0
        lines.append("总体 (Day 8-12):")
        lines.append(f"  平均重合率: {avg_all:.2f}% ± {std_all:.2f}%")
        lines.append(f"  范围: [{min(all_overlaps):.2f}%, {max(all_overlaps):.2f}%]")
        lines.append(f"  总样本数: {len(all_overlaps)}")

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    print(f"\n✅ PEFT 结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="运行 PEFT 微调实验并汇总 Day 8-12 结果")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--pretrained_ndays", type=int, nargs="+", default=[5, 7])
    parser.add_argument("--finetune_days", type=int, nargs="+", default=[8, 9, 10, 11, 12])
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_gpus", type=int, default=5)
    parser.add_argument("--skip_model_a", action="store_true", default=True)
    parser.add_argument("--only_summary", action="store_true",
                        help="不重新跑实验，只基于现有日志汇总结果")
    args = parser.parse_args()

    if not args.only_summary:
        run_automated_experiments(args)

    days = args.finetune_days
    results, overlap_results = extract_cer_and_overlap(days)
    base_dir = Path(__file__).resolve().parent.parent
    output_path = base_dir / "results_peft_day8_12.md"
    summarize_peft_results(days, results, overlap_results, output_path)


if __name__ == "__main__":
    main()

