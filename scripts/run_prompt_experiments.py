#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
并行跑不同 prompt_format 的实验脚本
功能：
1. 遍历指定的多种 prompt_format（当前为 native_pair, feature_injection）。
2. 每种 format 跑 10 个 seed 的训练 (Model B) 和测试 (Model B Test / 重合度)。
3. 使用指定 GPU [0, 1, 3, 4]，每张卡最多并行 2 个任务。
4. 结果将统一汇总并保存。
"""
import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# 实验配置
PROMPT_FORMATS = ['native_pair', 'feature_injection']
SEEDS = list(range(10))  # 10 个 seeds: 0~9
GPUS = [0, 1, 3, 4]      # 可用 GPU
TASKS_PER_GPU = 2
MAX_WORKERS = len(GPUS) * TASKS_PER_GPU

# 全局变量和锁，用于管理 GPU 分配
gpu_lock = Lock()
# 记录每个 GPU 当前分配的任务数，以支持 MAX_WORKERS > len(GPUS) 的情况
gpu_usage = {gpu: 0 for gpu in GPUS}

def get_available_gpu():
    """获取当前负载最轻且未超过并发上限的 GPU"""
    while True:
        with gpu_lock:
            candidates = [gpu for gpu, usage in gpu_usage.items() if usage < TASKS_PER_GPU]
            if candidates:
                best_gpu = min(candidates, key=lambda g: gpu_usage[g])
                gpu_usage[best_gpu] += 1
                return best_gpu
        time.sleep(0.2)

def release_gpu(gpu_id):
    """释放 GPU 负载"""
    with gpu_lock:
        gpu_usage[gpu_id] -= 1

def run_command(cmd, gpu_id, log_file, env_vars=None):
    """运行子进程命令"""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    if env_vars:
        env.update(env_vars)
        
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=36000  # 10小时超时
            )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        with open(log_file, 'a') as f:
            f.write("\n\n[ERROR] Command timed out after 10 hours.\n")
        return False
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"\n\n[ERROR] Execution failed: {str(e)}\n")
        return False

def run_task(task, base_dir, scripts_dir, logs_dir, outputs_dir):
    """运行单个任务（包含训练和测试）"""
    prompt_format = task['prompt_format']
    seed = task['seed']
    task_name = f"{prompt_format}_seed{seed}"
    
    # 获取 GPU
    gpu_id = get_available_gpu()
    print(f"🚀 [Start] 任务: {task_name} -> 分配 GPU: {gpu_id}")
    
    # 路径配置
    model_b_output_dir = os.path.join(outputs_dir, 'model_b', f"{prompt_format}_seed{seed}")
    train_log = os.path.join(logs_dir, f"train_{task_name}.log")
    test_log = os.path.join(logs_dir, f"test_{task_name}.log")
    
    # 假设 Model A 的预训练天数和路径使用之前代码的默认设置（前 5 天训练 B，前 7 天测试）
    pretrained_ndays_train = 5
    pretrained_ndays_eval = 7
    
    model_a_path_train = os.path.join(base_dir, 'outputs', 'model_train', f'conformer-{pretrained_ndays_train}days-seed{seed}')
    model_a_path_eval = os.path.join(base_dir, 'outputs', 'model_train', f'conformer-{pretrained_ndays_eval}days-seed{seed}')
    
    # 构建训练命令
    train_cmd = [
        sys.executable, os.path.join(scripts_dir, 'main_pipeline.py'),
        '--mode', 'train_only',
        '--metric', 'slpe',
        '--pretrained_ndays_train_b', str(pretrained_ndays_train),
        '--model_a_path_train_b', model_a_path_train,
        '--output_dir', model_b_output_dir,
        '--base_dir', base_dir,
        '--seed', str(seed)
    ]
    
    # 执行训练
    train_success = run_command(
        train_cmd,
        gpu_id,
        train_log,
        env_vars={
            'PYTHONPATH': scripts_dir,
            'MODEL_B_PROMPT_FORMAT': prompt_format
        }
    )
    
    if not train_success:
        print(f"❌ [Failed] 任务: {task_name} (训练阶段失败，跳过测试)")
        release_gpu(gpu_id)
        return {'task': task_name, 'prompt_format': prompt_format, 'seed': seed, 'status': 'train_failed'}
    
    print(f"✅ [Train Done] 任务: {task_name} 训练完成，开始测试...")
    
    # 构建测试命令
    test_cmd = [
        sys.executable, os.path.join(scripts_dir, 'main_pipeline.py'),
        '--mode', 'test_only',
        '--metric', 'slpe',
        '--model_b_path', model_b_output_dir,  # 指向刚训练出的模型
        '--pretrained_ndays_eval', str(pretrained_ndays_eval),
        '--model_a_path_eval', model_a_path_eval,
        '--base_dir', base_dir,
        '--seed', str(seed)
    ]
    
    # 执行测试
    test_success = run_command(
        test_cmd,
        gpu_id,
        test_log,
        env_vars={
            'PYTHONPATH': scripts_dir,
            'MODEL_B_PROMPT_FORMAT': prompt_format
        }
    )
    
    # 释放 GPU
    release_gpu(gpu_id)
    
    if test_success:
        print(f"🎉 [Success] 任务: {task_name} (训练和测试全部完成)")
        return {'task': task_name, 'prompt_format': prompt_format, 'seed': seed, 'status': 'success'}
    else:
        print(f"⚠️ [Failed] 任务: {task_name} (测试阶段失败)")
        return {'task': task_name, 'prompt_format': prompt_format, 'seed': seed, 'status': 'test_failed'}


def main():
    parser = argparse.ArgumentParser(description='Prompt Format 自动化对比实验')
    parser.add_argument('--base_dir', type=str, default='/root/25S151115/project3', help='项目基础目录')
    args = parser.parse_args()
    
    base_dir = args.base_dir
    scripts_dir = os.path.join(base_dir, 'scripts')
    
    # 实验输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(scripts_dir, 'outputs', 'prompt_experiments', f"exp_{timestamp}")
    logs_dir = os.path.join(exp_dir, 'logs')
    outputs_dir = os.path.join(exp_dir, 'models')
    
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
    print("="*80)
    print("🚀 启动 Prompt Format 并行实验")
    print(f"   Formats: {PROMPT_FORMATS}")
    print(f"   Seeds: {SEEDS}")
    print(f"   GPUs: {GPUS}")
    print(f"   Tasks Per GPU: {TASKS_PER_GPU}")
    print(f"   Max Workers (并行度): {MAX_WORKERS}")
    print(f"   Logs & Outputs: {exp_dir}")
    print("="*80)
    
    # 构建所有任务列表
    tasks = []
    for fmt in PROMPT_FORMATS:
        for seed in SEEDS:
            tasks.append({'prompt_format': fmt, 'seed': seed})
            
    print(f"总计任务数: {len(tasks)}")
    
    # 并行执行
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for task in tasks:
            future = executor.submit(run_task, task, base_dir, scripts_dir, logs_dir, outputs_dir)
            futures.append(future)
            
        for future in futures:
            results.append(future.result())
            
    elapsed = time.time() - start_time
    
    # 汇总结果
    print("\n" + "="*80)
    print(f"✅ 所有实验执行完毕！总耗时: {elapsed/3600:.2f} 小时")
    
    summary = {
        'total_tasks': len(tasks),
        'success': sum(1 for r in results if r['status'] == 'success'),
        'train_failed': sum(1 for r in results if r['status'] == 'train_failed'),
        'test_failed': sum(1 for r in results if r['status'] == 'test_failed'),
        'details': results
    }
    
    summary_path = os.path.join(exp_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
        
    print(f"统计汇总:\n  成功: {summary['success']}\n  训练失败: {summary['train_failed']}\n  测试失败: {summary['test_failed']}")
    print(f"详细报告已保存至: {summary_path}")
    print("="*80)


if __name__ == '__main__':
    main()
