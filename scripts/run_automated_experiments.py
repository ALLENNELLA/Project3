#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自动化实验脚本
功能：多GPU并行执行多seed实验
- 训练Model A（前5天和前7天）
- 微调实验（real_slpe, model_b, random）
"""
import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(
        self,
        base_dir: str = '/root/25S151115/project3',
        scripts_dir: str = None,
        num_gpus: int = None,
        seeds: List[int] = None,
        pretrained_ndays_list: List[int] = None,
        finetune_days: List[int] = None,
        num_samples: int = 100
    ):
        self.base_dir = Path(base_dir)
        if scripts_dir is None:
            self.scripts_dir = self.base_dir / 'scripts'
        else:
            self.scripts_dir = Path(scripts_dir)
        
        # 检测可用GPU数量
        if num_gpus is None:
            self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        else:
            self.num_gpus = min(num_gpus, torch.cuda.device_count() if torch.cuda.is_available() else 1)
        
        self.seeds = seeds if seeds is not None else list(range(10))  # 0-9
        self.pretrained_ndays_list = pretrained_ndays_list if pretrained_ndays_list else [5, 7]
        self.finetune_days = finetune_days if finetune_days else [8, 9, 10, 11, 12]
        self.num_samples = num_samples
        
        # 实验状态记录
        self.experiment_log = []
        self.results_dir = self.scripts_dir / 'outputs' / 'automated_experiments'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🔧 实验配置:")
        print(f"   - 基础目录: {self.base_dir}")
        print(f"   - 脚本目录: {self.scripts_dir}")
        print(f"   - 可用GPU数: {self.num_gpus}")
        print(f"   - Seeds: {self.seeds}")
        print(f"   - 预训练天数: {self.pretrained_ndays_list}")
        print(f"   - 微调天数: {self.finetune_days}")
        print(f"   - 样本数量: {self.num_samples}")
        print()
    
    def generate_tasks(self, skip_model_a: bool = False) -> List[Dict]:
        """生成所有实验任务"""
        tasks = []
        
        # 1. 训练Model A任务（如果未跳过且Model A不存在）
        if not skip_model_a:
            for n_days in self.pretrained_ndays_list:
                for seed in self.seeds:
                    model_a_path = self.base_dir / 'outputs' / 'model_train' / f'conformer-{n_days}days-seed{seed}' / 'modelWeights.pth'
                    # 如果Model A已存在，跳过训练任务，但将其标记为已完成
                    if model_a_path.exists():
                        task_id = f"train_model_a_{n_days}_{seed}"
                        print(f"✅ Model A已存在，跳过训练: conformer-{n_days}days-seed{seed}")
                        continue
                    
                    task = {
                        'type': 'train_model_a',
                        'n_days': n_days,
                        'seed': seed,
                        'gpu': None,  # 待分配
                        'status': 'pending',
                        'output_dir': self.base_dir / 'outputs' / 'model_train' / f'conformer-{n_days}days-seed{seed}',
                        'model_name': 'conformer'
                    }
                    tasks.append(task)
        
        # 2. 训练Model B任务（基于SLPE，只用前5天模型预测6、7天的SLPE）
        # 只用前5天模型，训练10个seed
        for seed in self.seeds:
            task = {
                'type': 'train_model_b',
                'pretrained_ndays': 5,  # 固定使用前5天模型
                'seed': seed,
                'metric': 'slpe',
                'gpu': None,
                'status': 'pending',
                'output_dir': self.scripts_dir / 'outputs' / 'model_b' / f'slpe-5days-seed{seed}',
                'depends_on': [f'train_model_a_5_{seed}']
            }
            tasks.append(task)
        
        # 3. 微调任务（只用前7天模型，微调到8、9、10、11、12天）
        pretrained_ndays = 7  # 固定使用前7天模型
        for day in self.finetune_days:
            # real_slpe: 每个seed跑1次
            for seed in self.seeds:
                task = {
                    'type': 'finetune',
                    'method': 'real_slpe',
                    'pretrained_ndays': pretrained_ndays,
                    'target_day': day,
                    'seed': seed,
                    'num_samples': self.num_samples,
                    'gpu': None,
                    'status': 'pending',
                    'output_dir': self.base_dir / 'outputs' / 'model_test' / f'{pretrained_ndays}-{day}' / 'real_slpe' / f'seed{seed}',
                    'depends_on': [f'train_model_a_{pretrained_ndays}_{seed}']
                }
                tasks.append(task)
            
            # model_b: 每个seed跑1次（需要Model B，基于前5天模型）
            for seed in self.seeds:
                task = {
                    'type': 'finetune',
                    'method': 'model_b',
                    'pretrained_ndays': pretrained_ndays,
                    'target_day': day,
                    'seed': seed,
                    'num_samples': self.num_samples,
                    'gpu': None,
                    'status': 'pending',
                    'output_dir': self.base_dir / 'outputs' / 'model_test' / f'{pretrained_ndays}-{day}' / 'model_b' / f'seed{seed}',
                    'depends_on': [
                        f'train_model_a_{pretrained_ndays}_{seed}',
                        f'train_model_b_5_{seed}'  # Model B基于前5天模型
                    ]
                }
                tasks.append(task)
            
            # random: 每个seed跑5次（模型seed + 选句子seed）
            for seed in self.seeds:
                for selection_seed in range(5):  # 5个不同的选句子seed
                    task = {
                        'type': 'finetune',
                        'method': 'random',
                        'pretrained_ndays': pretrained_ndays,
                        'target_day': day,
                        'seed': seed,  # 模型seed
                        'selection_seed': selection_seed,  # 选句子seed
                        'run_id': selection_seed,
                        'num_samples': self.num_samples,
                        'gpu': None,
                        'status': 'pending',
                        'output_dir': self.base_dir / 'outputs' / 'model_test' / f'{pretrained_ndays}-{day}' / 'random' / f'seed{seed}_sel{selection_seed}',
                        'depends_on': [f'train_model_a_{pretrained_ndays}_{seed}']
                    }
                    tasks.append(task)
        
        print(f"📋 生成了 {len(tasks)} 个实验任务")
        return tasks
    
    def get_task_id(self, task: Dict) -> str:
        """获取任务ID"""
        if task['type'] == 'train_model_a':
            return f"train_model_a_{task['n_days']}_{task['seed']}"
        elif task['type'] == 'train_model_b':
            return f"train_model_b_{task['pretrained_ndays']}_{task['seed']}"
        elif task['type'] == 'finetune':
            run_id = task.get('run_id', 0)
            return f"finetune_{task['method']}_{task['pretrained_ndays']}_{task['target_day']}_{task['seed']}_{run_id}"
        return f"{task['type']}_{task.get('seed', 0)}"
    
    def check_dependencies(self, task: Dict, completed_tasks: set) -> bool:
        """检查任务依赖是否满足"""
        depends_on = task.get('depends_on', [])
        if not depends_on:
            return True
        return all(dep in completed_tasks for dep in depends_on)
    
    def run_train_model_a(self, task: Dict, gpu_id: int) -> Tuple[bool, str]:
        """运行Model A训练任务"""
        n_days = task['n_days']
        seed = task['seed']
        output_dir = task['output_dir']
        
        # 需要修改main_pipeline.py来支持output_dir参数
        # 暂时通过修改config来实现，或者直接调用train_model_a
        # 这里我们直接调用train_model_a函数
        import sys
        sys.path.insert(0, str(self.scripts_dir))
        from model_a_train_module import train_model_a
        
        # 直接调用函数（在同一进程中，但使用不同的GPU）
        # 由于需要隔离GPU，我们还是用subprocess，但需要修改main_pipeline支持output_dir
        # 暂时先不传output_dir，让config自动生成，然后手动移动（不推荐）
        # 更好的方法是修改get_train_config支持seed和自定义outputDir
        
        cmd = [
            'python', str(self.scripts_dir / 'main_pipeline.py'),
            '--mode', 'train_model_a',
            '--model_a_n_days', str(n_days),
            '--model_a_name', 'conformer',
            '--base_dir', str(self.base_dir),
            '--seed', str(seed)
        ]
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['PYTHONPATH'] = str(self.scripts_dir)
        
        log_file = self.results_dir / 'logs' / f"train_model_a_{n_days}_seed{seed}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    cwd=str(self.scripts_dir),
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=36000  # 10小时超时
                )
            
            success = result.returncode == 0
            msg = f"Model A训练完成 (n_days={n_days}, seed={seed})"
            return success, msg
        except subprocess.TimeoutExpired:
            return False, f"Model A训练超时 (n_days={n_days}, seed={seed})"
        except Exception as e:
            return False, f"Model A训练失败: {str(e)}"
    
    def run_train_model_b(self, task: Dict, gpu_id: int) -> Tuple[bool, str]:
        """运行Model B训练任务"""
        pretrained_ndays = task['pretrained_ndays']
        seed = task['seed']
        metric = task['metric']
        
        # Model A路径（需要对应seed）
        model_a_path = self.base_dir / 'outputs' / 'model_train' / f'conformer-{pretrained_ndays}days-seed{seed}'
        
        # 检查Model A是否存在
        if not model_a_path.exists():
            return False, f"Model A不存在: {model_a_path}，请先训练Model A"
        
        # 训练天数（假设用6、7天）
        train_days = [6, 7] if pretrained_ndays == 5 else [6, 7]
        
        cmd = [
            'python', str(self.scripts_dir / 'main_pipeline.py'),
            '--mode', 'train_only',
            '--metric', metric,
            '--train_days'] + [str(d) for d in train_days] + [
            '--pretrained_ndays', str(pretrained_ndays),
            '--model_a_path', str(model_a_path),
            '--output_dir', str(task['output_dir']),
            '--base_dir', str(self.base_dir),
            '--seed', str(seed)
        ]
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['PYTHONPATH'] = str(self.scripts_dir)
        
        log_file = self.results_dir / 'logs' / f"train_model_b_{pretrained_ndays}_seed{seed}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    cwd=str(self.scripts_dir),
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=36000
                )
            
            success = result.returncode == 0
            msg = f"Model B训练完成 (pretrained_ndays={pretrained_ndays}, seed={seed}, metric={metric})"
            return success, msg
        except subprocess.TimeoutExpired:
            return False, f"Model B训练超时 (pretrained_ndays={pretrained_ndays}, seed={seed})"
        except Exception as e:
            return False, f"Model B训练失败: {str(e)}"
    
    def run_finetune(self, task: Dict, gpu_id: int) -> Tuple[bool, str]:
        """运行微调任务"""
        method = task['method']
        pretrained_ndays = task['pretrained_ndays']
        target_day = task['target_day']
        seed = task['seed']
        num_samples = task['num_samples']
        
        # Model A路径
        model_a_path = self.base_dir / 'outputs' / 'model_train' / f'conformer-{pretrained_ndays}days-seed{seed}'
        
        cmd = [
            'python', str(self.scripts_dir / 'main_pipeline.py'),
            '--mode', 'finetune_only',
            '--finetune_method', method,
            '--finetune_target_days', str(target_day),
            '--num_samples', str(num_samples),
            '--pretrained_ndays', str(pretrained_ndays),
            '--model_a_path', str(model_a_path),
            '--base_dir', str(self.base_dir),
            '--seed', str(seed)
        ]
        
        # 如果是model_b方法，需要提供model_b_path（Model B基于前5天模型训练）
        if method == 'model_b':
            model_b_path = self.scripts_dir / 'outputs' / 'model_b' / f'slpe-5days-seed{seed}'
            cmd.extend(['--model_b_path', str(model_b_path)])
        
        # 如果是random方法，需要传递selection_seed（选句子的seed）
        if method == 'random' and 'selection_seed' in task:
            cmd.extend(['--selection_seed', str(task['selection_seed'])])
        
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['PYTHONPATH'] = str(self.scripts_dir)
        
        run_id = task.get('run_id', 0)
        selection_seed = task.get('selection_seed', '')
        if selection_seed != '':
            log_file = self.results_dir / 'logs' / f"finetune_{method}_{pretrained_ndays}_{target_day}_seed{seed}_sel{selection_seed}.log"
        else:
            log_file = self.results_dir / 'logs' / f"finetune_{method}_{pretrained_ndays}_{target_day}_seed{seed}_run{run_id}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(log_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    cwd=str(self.scripts_dir),
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=36000
                )
            
            success = result.returncode == 0
            msg = f"微调完成 (method={method}, pretrained_ndays={pretrained_ndays}, day={target_day}, seed={seed}, run={run_id})"
            return success, msg
        except subprocess.TimeoutExpired:
            return False, f"微调超时 (method={method}, pretrained_ndays={pretrained_ndays}, day={target_day}, seed={seed})"
        except Exception as e:
            return False, f"微调失败: {str(e)}"
    
    def run_task(self, task: Dict, gpu_id: int) -> Tuple[str, bool, str]:
        """运行单个任务"""
        task_id = self.get_task_id(task)
        
        try:
            if task['type'] == 'train_model_a':
                success, msg = self.run_train_model_a(task, gpu_id)
            elif task['type'] == 'train_model_b':
                success, msg = self.run_train_model_b(task, gpu_id)
            elif task['type'] == 'finetune':
                success, msg = self.run_finetune(task, gpu_id)
            else:
                return task_id, False, f"未知任务类型: {task['type']}"
            
            return task_id, success, msg
        except Exception as e:
            return task_id, False, f"任务执行异常: {str(e)}"
    
    def run_experiments(self, max_workers: int = None, skip_model_a: bool = False):
        """运行所有实验（多GPU并行）"""
        if max_workers is None:
            max_workers = self.num_gpus
        
        tasks = self.generate_tasks(skip_model_a=skip_model_a)
        completed_tasks = set()
        running_tasks = {}  # {task_id: (task, gpu_id)}
        gpu_available = {i: True for i in range(self.num_gpus)}
        
        # 如果跳过Model A训练，将所有已存在的Model A标记为已完成
        if skip_model_a:
            for n_days in self.pretrained_ndays_list:
                for seed in self.seeds:
                    model_a_path = self.base_dir / 'outputs' / 'model_train' / f'conformer-{n_days}days-seed{seed}' / 'modelWeights.pth'
                    if model_a_path.exists():
                        task_id = f"train_model_a_{n_days}_{seed}"
                        completed_tasks.add(task_id)
                        print(f"✅ Model A已存在，标记为已完成: {task_id}")
        
        # 任务队列
        pending_tasks = tasks.copy()
        
        print(f"🚀 开始执行 {len(tasks)} 个实验任务（使用 {max_workers} 个GPU）")
        print("="*80)
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            
            while pending_tasks or running_tasks:
                # 分配新任务
                for task in pending_tasks[:]:
                    task_id = self.get_task_id(task)
                    
                    # 检查依赖
                    if not self.check_dependencies(task, completed_tasks):
                        continue
                    
                    # 分配GPU
                    gpu_id = None
                    for gpu in range(self.num_gpus):
                        if gpu_available[gpu]:
                            gpu_id = gpu
                            gpu_available[gpu] = False
                            break
                    
                    if gpu_id is None:
                        continue  # 没有可用GPU，等待
                    
                    # 提交任务
                    future = executor.submit(self.run_task, task, gpu_id)
                    futures[future] = (task_id, task, gpu_id)
                    running_tasks[task_id] = (task, gpu_id)
                    pending_tasks.remove(task)
                    
                    print(f"📌 提交任务: {task_id} (GPU {gpu_id})")
                
                # 检查完成的任务
                done_futures = []
                for future in futures:
                    if future.done():
                        done_futures.append(future)
                
                for future in done_futures:
                    task_id, task, gpu_id = futures.pop(future)
                    success, status, msg = future.result()
                    
                    # 释放GPU
                    gpu_available[gpu_id] = True
                    running_tasks.pop(task_id, None)
                    
                    # 记录结果
                    if success:
                        completed_tasks.add(task_id)
                        task['status'] = 'completed'
                        print(f"✅ {msg}")
                    else:
                        task['status'] = 'failed'
                        print(f"❌ {msg}")
                    
                    self.experiment_log.append({
                        'task_id': task_id,
                        'task': task,
                        'success': success,
                        'message': msg,
                        'gpu': gpu_id,
                        'timestamp': datetime.now().isoformat()
                    })
                
                # 如果没有任务在运行且还有待处理任务，等待一下
                if not running_tasks and pending_tasks:
                    time.sleep(1)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print("="*80)
        print(f"✅ 所有实验完成！总耗时: {elapsed/3600:.2f} 小时")
        
        # 保存实验日志
        self.save_experiment_log()
        
        # 生成总结报告
        self.generate_summary()
    
    def save_experiment_log(self):
        """保存实验日志"""
        log_file = self.results_dir / f"experiment_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(log_file, 'w') as f:
            json.dump({
                'config': {
                    'base_dir': str(self.base_dir),
                    'scripts_dir': str(self.scripts_dir),
                    'num_gpus': self.num_gpus,
                    'seeds': self.seeds,
                    'pretrained_ndays_list': self.pretrained_ndays_list,
                    'finetune_days': self.finetune_days,
                    'num_samples': self.num_samples
                },
                'experiments': self.experiment_log
            }, f, indent=2)
        
        print(f"📝 实验日志已保存: {log_file}")
    
    def generate_summary(self):
        """生成实验总结"""
        summary = {
            'total_tasks': len(self.experiment_log),
            'completed': sum(1 for exp in self.experiment_log if exp['success']),
            'failed': sum(1 for exp in self.experiment_log if not exp['success']),
            'by_type': {},
            'by_method': {}
        }
        
        for exp in self.experiment_log:
            task = exp['task']
            task_type = task['type']
            
            if task_type not in summary['by_type']:
                summary['by_type'][task_type] = {'total': 0, 'completed': 0, 'failed': 0}
            
            summary['by_type'][task_type]['total'] += 1
            if exp['success']:
                summary['by_type'][task_type]['completed'] += 1
            else:
                summary['by_type'][task_type]['failed'] += 1
            
            if task_type == 'finetune':
                method = task['method']
                if method not in summary['by_method']:
                    summary['by_method'][method] = {'total': 0, 'completed': 0, 'failed': 0}
                
                summary['by_method'][method]['total'] += 1
                if exp['success']:
                    summary['by_method'][method]['completed'] += 1
                else:
                    summary['by_method'][method]['failed'] += 1
        
        summary_file = self.results_dir / f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 实验总结已保存: {summary_file}")
        print("\n实验总结:")
        print(f"  总任务数: {summary['total_tasks']}")
        print(f"  成功: {summary['completed']}")
        print(f"  失败: {summary['failed']}")
        print("\n按类型统计:")
        for task_type, stats in summary['by_type'].items():
            print(f"  {task_type}: {stats['completed']}/{stats['total']} 成功")
        if summary['by_method']:
            print("\n按方法统计（微调）:")
            for method, stats in summary['by_method'].items():
                print(f"  {method}: {stats['completed']}/{stats['total']} 成功")


def main():
    parser = argparse.ArgumentParser(description='自动化实验脚本')
    parser.add_argument('--base_dir', type=str, default='/root/25S151115/project3',
                       help='项目基础目录')
    parser.add_argument('--scripts_dir', type=str, default=None,
                       help='脚本目录（默认: base_dir/scripts）')
    parser.add_argument('--num_gpus', type=int, default=None,
                       help='使用的GPU数量（默认: 自动检测）')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(10)),
                       help='实验种子列表（默认: 0-9）')
    parser.add_argument('--pretrained_ndays', type=int, nargs='+', default=[5, 7],
                       help='预训练天数列表（默认: 5, 7）')
    parser.add_argument('--finetune_days', type=int, nargs='+', default=[8, 9, 10, 11, 12],
                       help='微调目标天数（默认: 8, 9, 10, 11, 12）')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='每个微调实验选择的样本数（默认: 100）')
    parser.add_argument('--max_workers', type=int, default=None,
                       help='最大并行任务数（默认: GPU数量）')
    parser.add_argument('--skip_model_a', action='store_true',
                       help='跳过Model A训练（假设已训练完成）')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(
        base_dir=args.base_dir,
        scripts_dir=args.scripts_dir,
        num_gpus=args.num_gpus,
        seeds=args.seeds,
        pretrained_ndays_list=args.pretrained_ndays,
        finetune_days=args.finetune_days,
        num_samples=args.num_samples
    )
    
    runner.run_experiments(max_workers=args.max_workers, skip_model_a=args.skip_model_a)


if __name__ == '__main__':
    main()
