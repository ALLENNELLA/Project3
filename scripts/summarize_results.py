#!/usr/bin/env python3
"""总结微调结果"""
import re
from pathlib import Path
from collections import defaultdict
import statistics

def extract_cer_from_log(log_file):
    """从日志文件中提取CER"""
    content = log_file.read_text()
    
    # 提取最终CER - 匹配格式: "  - **Character Error Rate (CER): 0.3316**"
    cer_match = re.search(r'Character Error Rate \(CER\): ([\d.]+)', content)
    if not cer_match:
        return None
    
    return float(cer_match.group(1))

def summarize_finetune_results(days=[8, 9]):
    """总结指定天数的微调结果"""
    log_dir = Path("outputs/automated_experiments/logs")
    
    results = defaultdict(lambda: defaultdict(list))
    
    for day in days:
        # 解析日志文件
        for log_file in log_dir.glob(f"finetune_*_7_{day}_*.log"):
            # 提取方法、seed等信息
            # 匹配格式: finetune_{method}_7_{day}_seed{seed}_sel{sel_seed}.log 或 finetune_{method}_7_{day}_seed{seed}_run{run_id}.log
            pattern = rf"finetune_(\w+)_7_{day}_seed(\d+)(?:_sel(\d+)|_run(\d+))?\.log"
            match = re.match(pattern, log_file.name)
            if not match:
                continue
            
            method = match.group(1)
            seed = int(match.group(2))
            sel_seed = match.group(3)
            run_id = match.group(4)
            
            cer = extract_cer_from_log(log_file)
            if cer is None:
                continue
            
            # 构建key
            if sel_seed:
                key = f"{method}_seed{seed}_sel{sel_seed}"
            else:
                key = f"{method}_seed{seed}"
            
            results[day][key].append(cer)
    
    # 生成总结
    print("=" * 80)
    print("📊 Day 8 & Day 9 微调结果总结")
    print("=" * 80)
    
    for day in days:
        print(f"\n{'='*80}")
        print(f"📅 Day {day} 微调结果")
        print(f"{'='*80}\n")
        
        # 按方法分组
        method_results = defaultdict(lambda: defaultdict(list))
        
        for key, cers in results[day].items():
            # 提取方法名（可能是real_slpe, model_b, random）
            if key.startswith('real_slpe'):
                method = 'real_slpe'
                remaining = key[len('real_slpe')+1:]  # 去掉 'real_slpe_'
            elif key.startswith('model_b'):
                method = 'model_b'
                remaining = key[len('model_b')+1:]  # 去掉 'model_b_'
            elif key.startswith('random'):
                method = 'random'
                remaining = key[len('random')+1:]  # 去掉 'random_'
            else:
                continue
            
            # 解析剩余部分：seed{num} 或 seed{num}_sel{num}
            parts = remaining.split('_')
            seed = parts[0]  # seed{num}
            
            if len(parts) > 1 and parts[1].startswith('sel'):
                sel_seed = parts[1]
                method_results[method][f"{seed}_{sel_seed}"].extend(cers)
            else:
                method_results[method][seed].extend(cers)
        
        # 输出每个方法的结果
        for method in ['real_slpe', 'model_b', 'random']:
            if method not in method_results:
                continue
            
            print(f"\n🔹 方法: {method.upper()}")
            print("-" * 80)
            
            all_cers = []
            seed_cers = defaultdict(list)
            
            for key, cers in method_results[method].items():
                if cers:
                    cer = cers[0]  # 每个配置只有一个结果
                    all_cers.append(cer)
                    
                    # 提取seed（去掉sel部分）
                    if '_sel' in key:
                        seed_part = key.split('_sel')[0]
                    else:
                        seed_part = key
                    seed_cers[seed_part].append(cer)
            
            # 按seed输出
            def extract_seed_num(seed_str):
                """提取seed数字用于排序"""
                match = re.search(r'seed(\d+)', seed_str)
                return int(match.group(1)) if match else 0
            
            for seed in sorted(seed_cers.keys(), key=extract_seed_num):
                cers = seed_cers[seed]
                if method == 'random':
                    # random方法有多个selection_seed
                    avg_cer = sum(cers) / len(cers)
                    min_cer = min(cers)
                    max_cer = max(cers)
                    std_cer = statistics.stdev(cers) if len(cers) > 1 else 0
                    print(f"  {seed}: 平均CER={avg_cer:.4f}, 最小={min_cer:.4f}, 最大={max_cer:.4f}, 标准差={std_cer:.4f} (共{len(cers)}次)")
                else:
                    # real_slpe和model_b每个seed只有一个结果
                    print(f"  {seed}: CER={cers[0]:.4f}")
            
            # 总体统计
            if all_cers:
                avg_cer = sum(all_cers) / len(all_cers)
                min_cer = min(all_cers)
                max_cer = max(all_cers)
                std_cer = statistics.stdev(all_cers) if len(all_cers) > 1 else 0
                print(f"\n  📈 总体统计 (共{len(all_cers)}个结果):")
                print(f"     平均CER: {avg_cer:.4f}")
                print(f"     最小CER: {min_cer:.4f}")
                print(f"     最大CER: {max_cer:.4f}")
                if len(all_cers) > 1:
                    print(f"     标准差: {std_cer:.4f}")
    
    # 每一天每个方法的Mean±Std
    print(f"\n{'='*80}")
    print("📊 每一天每个方法的 Mean±Std")
    print(f"{'='*80}\n")
    
    # 按天和方法组织数据
    day_method_stats = defaultdict(lambda: defaultdict(list))
    
    for day in days:
        for key, cers in results[day].items():
            if cers:
                # 提取方法名
                if key.startswith('real_slpe'):
                    method = 'real_slpe'
                elif key.startswith('model_b'):
                    method = 'model_b'
                elif key.startswith('random'):
                    method = 'random'
                else:
                    continue
                # 对于每个配置，取第一个CER值
                day_method_stats[day][method].append(cers[0])
    
    # 输出表格格式
    print(f"{'方法':<15s} {'Day 8':<20s} {'Day 9':<20s}")
    print("-" * 55)
    
    for method in ['real_slpe', 'model_b', 'random']:
        method_name = method.upper().replace('_', ' ')
        day8_cers = day_method_stats[8].get(method, [])
        day9_cers = day_method_stats[9].get(method, [])
        
        if day8_cers:
            day8_mean = sum(day8_cers) / len(day8_cers)
            day8_std = statistics.stdev(day8_cers) if len(day8_cers) > 1 else 0
            day8_str = f"{day8_mean:.4f}±{day8_std:.4f} (n={len(day8_cers)})"
        else:
            day8_str = "N/A"
        
        if day9_cers:
            day9_mean = sum(day9_cers) / len(day9_cers)
            day9_std = statistics.stdev(day9_cers) if len(day9_cers) > 1 else 0
            day9_str = f"{day9_mean:.4f}±{day9_std:.4f} (n={len(day9_cers)})"
        else:
            day9_str = "N/A"
        
        print(f"{method_name:<15s} {day8_str:<20s} {day9_str:<20s}")
    
    # 方法对比（所有天数的平均）
    print(f"\n{'='*80}")
    print("📊 方法对比 (Day 8 & Day 9 平均)")
    print(f"{'='*80}\n")
    
    method_avg = defaultdict(list)
    for day in days:
        for key, cers in results[day].items():
            if cers:
                # 提取方法名
                if key.startswith('real_slpe'):
                    method = 'real_slpe'
                elif key.startswith('model_b'):
                    method = 'model_b'
                elif key.startswith('random'):
                    method = 'random'
                else:
                    continue
                method_avg[method].append(cers[0])
    
    for method in ['real_slpe', 'model_b', 'random']:
        if method in method_avg and method_avg[method]:
            cers = method_avg[method]
            avg = sum(cers) / len(cers)
            min_cer = min(cers)
            max_cer = max(cers)
            std = statistics.stdev(cers) if len(cers) > 1 else 0
            print(f"  {method.upper():12s}: 平均CER={avg:.4f}±{std:.4f}, 范围=[{min_cer:.4f}, {max_cer:.4f}] (n={len(cers)})")
    
    print("\n" + "=" * 80)
    print("✅ 总结完成")
    print("=" * 80)

if __name__ == "__main__":
    summarize_finetune_results([8, 9])
