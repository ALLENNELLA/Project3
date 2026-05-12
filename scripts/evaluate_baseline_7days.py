#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评估前7天训练的模型在Day 8-12上的CER（不微调）
"""
import os
import sys
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from src.model_a.evaluate import evaluate_cer

def get_day_dataset_path(day: int, base_dir: str = '/root/25S151115/project3') -> str:
    """
    获取特定天的数据集路径
    
    Args:
        day: 天数（8, 9, 10, 11, 12），对应sessionNames的索引（day-1）
        base_dir: 基础目录
    
    Returns:
        数据集路径
    """
    # sessionNames列表（从config.py）
    sessionNames = [
        't12.2022.04.28', 't12.2022.05.26', 't12.2022.06.21', 't12.2022.07.21', 't12.2022.08.13',
        't12.2022.05.05', 't12.2022.06.02', 't12.2022.06.23', 't12.2022.07.27', 't12.2022.08.18',
        't12.2022.05.17', 't12.2022.06.07', 't12.2022.06.28', 't12.2022.07.29', 't12.2022.08.23',
        't12.2022.05.19', 't12.2022.06.14', 't12.2022.07.05', 't12.2022.08.02', 't12.2022.08.25',
        't12.2022.05.24', 't12.2022.06.16', 't12.2022.07.14', 't12.2022.08.11'
    ]
    sessionNames.sort()
    
    if day < 1 or day > len(sessionNames):
        raise ValueError(f"天数 {day} 超出范围，只支持 1-{len(sessionNames)}")
    
    # 获取对应的session名称
    session_name = sessionNames[day - 1]
    
    # 从session名称提取日期部分（格式：t12.2022.MM.DD -> dataMMDD）
    parts = session_name.split('.')
    if len(parts) >= 3:
        month_day = parts[-2] + parts[-1]  # 例如 "0628"
        data_folder = f'data{month_day}'
    else:
        raise ValueError(f"无法解析session名称: {session_name}")
    
    # 构建数据集路径（数据集文件直接是data文件夹下的文件）
    dataset_path = os.path.join(base_dir, 'data', data_folder)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    return dataset_path


def evaluate_baseline_7days(
    seeds: list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    test_days: list = [8, 9, 10, 11, 12],
    base_dir: str = '/root/25S151115/project3',
    batch_size: int = 64,
    device: str = 'cuda'
):
    """
    评估前7天训练的模型在Day 8-12上的CER
    
    Args:
        seeds: 要评估的seed列表
        test_days: 要测试的天数列表
        base_dir: 基础目录
        batch_size: 批次大小
        device: 设备
    
    Returns:
        results: 字典，格式为 {seed: {day: cer}}
    """
    print("="*80)
    print("🚀 评估前7天训练的模型在Day 8-12上的CER（不微调）")
    print("="*80)
    print(f"Seeds: {seeds}")
    print(f"Test Days: {test_days}")
    print("="*80)
    print()
    
    results = defaultdict(dict)
    
    for seed in seeds:
        print("\n" + "="*80)
        print(f"📊 评估 Seed {seed}")
        print("="*80)
        
        # 模型路径
        model_dir = os.path.join(base_dir, 'outputs', 'model_train', f'conformer-7days-seed{seed}')
        
        if not os.path.exists(model_dir):
            print(f"⚠️  模型目录不存在: {model_dir}，跳过")
            continue
        
        if not os.path.exists(os.path.join(model_dir, 'modelWeights.pth')):
            print(f"⚠️  模型权重文件不存在: {os.path.join(model_dir, 'modelWeights.pth')}，跳过")
            continue
        
        print(f"✅ 加载模型: {model_dir}")
        
        # 对每个测试天进行评估
        for day in test_days:
            print(f"\n--- Day {day} ---")
            
            try:
                # 获取数据集路径
                dataset_path = get_day_dataset_path(day, base_dir)
                print(f"数据集路径: {dataset_path}")
                
                # 评估CER
                avg_loss, cer = evaluate_cer(
                    model_dir=model_dir,
                    dataset_path=dataset_path,
                    batch_size=batch_size,
                    eval_split="test",
                    device=device
                )
                
                results[seed][day] = {
                    'cer': cer,
                    'avg_loss': avg_loss
                }
                
                print(f"✅ Day {day} CER: {cer:.4f}")
                
            except FileNotFoundError as e:
                print(f"❌ 数据集文件不存在: {e}")
                results[seed][day] = None
            except Exception as e:
                print(f"❌ 评估失败: {e}")
                import traceback
                traceback.print_exc()
                results[seed][day] = None
    
    # 打印汇总结果
    print("\n" + "="*80)
    print("📊 汇总结果")
    print("="*80)
    
    # 表头
    header = ["Seed"] + [f"Day {day}" for day in test_days]
    print(f"{header[0]:<8s} {' '.join([f'{h:<12s}' for h in header[1:]])}")
    print("-" * (8 + 12 * len(test_days)))
    
    # 每个seed的结果
    for seed in seeds:
        if seed not in results:
            continue
        row = [f"Seed{seed}"]
        for day in test_days:
            if day in results[seed] and results[seed][day] is not None:
                cer = results[seed][day]['cer']
                row.append(f"{cer:.4f}")
            else:
                row.append("N/A")
        print(f"{row[0]:<8s} {' '.join([f'{r:<12s}' for r in row[1:]])}")
    
    # 计算每个天的平均值和标准差
    print("\n" + "="*80)
    print("📊 统计结果 (Mean±Std)")
    print("="*80)
    
    for day in test_days:
        cers = []
        for seed in seeds:
            if seed in results and day in results[seed] and results[seed][day] is not None:
                cers.append(results[seed][day]['cer'])
        
        if cers:
            mean_cer = np.mean(cers)
            std_cer = np.std(cers) if len(cers) > 1 else 0
            print(f"Day {day}: {mean_cer:.4f} ± {std_cer:.4f} (n={len(cers)})")
        else:
            print(f"Day {day}: N/A")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='评估前7天训练的模型在Day 8-12上的CER')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help='要评估的seed列表')
    parser.add_argument('--test_days', type=int, nargs='+', default=[8, 9, 10, 11, 12],
                        help='要测试的天数列表')
    parser.add_argument('--base_dir', type=str, default='/root/25S151115/project3',
                        help='基础目录')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    
    args = parser.parse_args()
    
    results = evaluate_baseline_7days(
        seeds=args.seeds,
        test_days=args.test_days,
        base_dir=args.base_dir,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # 保存结果
    output_file = os.path.join(args.base_dir, 'scripts', 'outputs', 'baseline_7days_results.pkl')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✅ 结果已保存到: {output_file}")
