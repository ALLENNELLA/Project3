#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汇总前7天模型在Day 8-12上的基线CER结果
"""
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

def summarize_baseline_results(results_file: str, output_file: str = None):
    """
    汇总基线评估结果
    
    Args:
        results_file: 结果pickle文件路径
        output_file: 输出markdown文件路径（如果为None，则打印到控制台）
    """
    # 加载结果
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    # 提取所有seed和day
    seeds = sorted([s for s in results.keys() if results[s]])
    test_days = [8, 9, 10, 11, 12]
    
    # 收集所有CER值
    day_cers = defaultdict(list)
    for seed in seeds:
        for day in test_days:
            if day in results[seed] and results[seed][day] is not None:
                cer = results[seed][day]['cer']
                day_cers[day].append(cer)
    
    # 生成报告
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("📊 前7天模型在Day 8-12上的基线CER评估结果（不微调）")
    report_lines.append("="*80)
    report_lines.append("")
    
    # 1. 完成情况
    report_lines.append("="*80)
    report_lines.append("📋 完成情况")
    report_lines.append("="*80)
    report_lines.append("")
    for day in test_days:
        count = len(day_cers[day])
        expected = len(seeds)
        status = "✅" if count == expected else f"⚠️ ({count}/{expected})"
        report_lines.append(f"Day {day}: {status}")
    report_lines.append("")
    
    # 2. 详细结果表格
    report_lines.append("="*80)
    report_lines.append("📊 详细结果 (CER)")
    report_lines.append("="*80)
    report_lines.append("")
    
    # 表头
    header = ["Seed"] + [f"Day {day}" for day in test_days]
    report_lines.append(f"{header[0]:<8s} {' '.join([f'{h:<12s}' for h in header[1:]])}")
    report_lines.append("-" * (8 + 12 * len(test_days)))
    
    # 每个seed的结果
    for seed in seeds:
        row = [f"Seed{seed}"]
        for day in test_days:
            if day in results[seed] and results[seed][day] is not None:
                cer = results[seed][day]['cer']
                row.append(f"{cer:.4f}")
            else:
                row.append("N/A")
        report_lines.append(f"{row[0]:<8s} {' '.join([f'{r:<12s}' for r in row[1:]])}")
    report_lines.append("")
    
    # 3. 统计结果
    report_lines.append("="*80)
    report_lines.append("📊 统计结果 (Mean±Std)")
    report_lines.append("="*80)
    report_lines.append("")
    
    for day in test_days:
        cers = day_cers[day]
        if cers:
            mean_cer = np.mean(cers)
            std_cer = np.std(cers) if len(cers) > 1 else 0
            min_cer = np.min(cers)
            max_cer = np.max(cers)
            report_lines.append(f"Day {day}:")
            report_lines.append(f"  Mean±Std: {mean_cer:.4f} ± {std_cer:.4f}")
            report_lines.append(f"  范围: [{min_cer:.4f}, {max_cer:.4f}]")
            report_lines.append(f"  样本数: {len(cers)}")
            report_lines.append("")
        else:
            report_lines.append(f"Day {day}: N/A")
            report_lines.append("")
    
    # 4. 趋势分析
    report_lines.append("="*80)
    report_lines.append("📊 趋势分析")
    report_lines.append("="*80)
    report_lines.append("")
    
    day_means = []
    for day in test_days:
        cers = day_cers[day]
        if cers:
            mean_cer = np.mean(cers)
            day_means.append((day, mean_cer))
            report_lines.append(f"Day {day}: {mean_cer:.4f}")
    
    if len(day_means) >= 2:
        first_day, first_cer = day_means[0]
        last_day, last_cer = day_means[-1]
        trend = ((last_cer - first_cer) / first_cer) * 100
        report_lines.append("")
        report_lines.append(f"趋势 (Day {first_day} → Day {last_day}): {trend:+.2f}%")
        report_lines.append(f"  Day {first_day}: {first_cer:.4f}")
        report_lines.append(f"  Day {last_day}: {last_cer:.4f}")
        report_lines.append(f"  绝对增长: {last_cer - first_cer:+.4f}")
    
    report_lines.append("")
    report_lines.append("="*80)
    
    # 输出报告
    report_text = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"✅ 报告已保存到: {output_file}")
    else:
        print(report_text)
    
    return report_text


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='汇总基线评估结果')
    parser.add_argument('--results_file', type=str, 
                        default='/root/25S151115/project3/scripts/outputs/baseline_7days_results.pkl',
                        help='结果pickle文件路径')
    parser.add_argument('--output_file', type=str, default=None,
                        help='输出markdown文件路径（如果为None，则打印到控制台）')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"❌ 结果文件不存在: {args.results_file}")
        print("请等待评估完成后再运行此脚本")
        sys.exit(1)
    
    summarize_baseline_results(args.results_file, args.output_file)
