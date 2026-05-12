#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从已有的微调结果目录中读取 selected_indices，只改变微调方式为：
Conformer 全参数 + CABlock + AdaptFFN，然后重新微调并记录结果。

覆盖范围：
- pretrained_ndays 固定 7
- Days: 8,9,10,11,12
- Methods:
    - real_slpe: 10 seeds (seed0-9)
    - model_b:   10 seeds (seed0-9)
    - random:    10 seeds × 5 selection_seed -> 50 runs/day
"""

import argparse
import os
import pickle
import sys
import subprocess
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def build_tasks(base_dir: Path, days, seeds):
    tasks = []
    pretrained_ndays = 7

    for day in days:
        day_root = base_dir / "outputs" / "model_test" / f"{pretrained_ndays}-{day}"

        # real_slpe / model_b: 每个 seed 一条
        for method in ["real_slpe", "model_b"]:
            for seed in seeds:
                old_dir = day_root / method / f"seed{seed}"
                cfg_path = old_dir / "config.pkl"
                if not cfg_path.exists():
                    continue
                new_dir = day_root / f"{method}_peft" / f"seed{seed}"
                tasks.append(
                    {
                        "day": day,
                        "method": method,
                        "seed": seed,
                        "sel_seed": None,
                        "old_dir": old_dir,
                        "new_dir": new_dir,
                        "config_path": cfg_path,
                    }
                )

        # random: 每个 seed 有 5 个 selection_seed
        method = "random"
        for seed in seeds:
            for sel in range(5):
                old_dir = day_root / method / f"seed{seed}_sel{sel}"
                cfg_path = old_dir / "config.pkl"
                if not cfg_path.exists():
                    continue
                new_dir = day_root / f"{method}_peft" / f"seed{seed}_sel{sel}"
                tasks.append(
                    {
                        "day": day,
                        "method": method,
                        "seed": seed,
                        "sel_seed": sel,
                        "old_dir": old_dir,
                        "new_dir": new_dir,
                        "config_path": cfg_path,
                    }
                )

    return tasks


def prepare_peft_config(task, scripts_dir: Path):
    """从旧 config.pkl 生成一个新的带 PEFT 开关的 config 文件，并返回其路径。"""
    cfg_path = task["config_path"]
    new_dir: Path = task["new_dir"]
    new_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path, "rb") as f:
        config = pickle.load(f)

    # 复用原来的 selected_indices / 数据路径 / 训练参数，只改输出路径和 PEFT 相关开关
    config["pretrainedModelOutputPath"] = str(new_dir)

    # 打开 PEFT：全参数 + CA Block + AdaptFFN（adapter bottleneck=64）
    config["use_adapter"] = True
    config["use_ca_block"] = True
    config["adapter_bottleneck"] = 64
    # 不再调用 enable_adapter_mode，因此不会冻结主干；这里 train_output_head 仅保留向后兼容
    config.setdefault("train_output_head", True)

    new_cfg_path = new_dir / "config_peft.pkl"
    with open(new_cfg_path, "wb") as f:
        pickle.dump(config, f)

    return new_cfg_path


def run_single_task(task, gpu_id, scripts_dir: Path, logs_dir: Path, gpu_semaphores, physical_gpu_start=2):
    """在指定 GPU 上运行一条 PEFT 微调任务，并将日志写入单独的 log 文件。

    使用 gpu_semaphores[gpu_id] 保证同一块 GPU 上最多并行 1 个任务。
    gpu_id 是逻辑 ID (0-4)，物理 GPU ID = physical_gpu_start + gpu_id (2-6)
    """
    sem = gpu_semaphores[gpu_id]
    sem.acquire()

    new_cfg_path = prepare_peft_config(task, scripts_dir)

    # 日志文件命名：finetune_{method}_7_{day}_seed{seed}[_sel{sel}]_peft.log
    day = task["day"]
    method = task["method"]
    seed = task["seed"]
    sel = task["sel_seed"]
    if sel is None:
        log_name = f"finetune_{method}_7_{day}_seed{seed}_peft.log"
    else:
        log_name = f"finetune_{method}_7_{day}_seed{seed}_sel{sel}_peft.log"
    log_file = logs_dir / log_name

    cmd = [
        sys.executable,
        str(scripts_dir / "run_finetune_from_config.py"),
        "--config_path",
        str(new_cfg_path),
    ]

    try:
        env = os.environ.copy()
        # 将逻辑 GPU ID (0-4) 映射到物理 GPU ID (2-6)
        physical_gpu_id = physical_gpu_start + gpu_id
        env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
        with open(log_file, "w", encoding="utf-8") as f:
            print(
                f"[RUN] Day {day}, method={method}, seed={seed}, sel={sel} on Physical GPU {physical_gpu_id} (logical {gpu_id})",
                file=f,
                flush=True,
            )
            proc = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(scripts_dir),
                env=env,
            )

        if proc.returncode != 0:
            print(
                f"[FAIL] Day {day}, method={method}, seed={seed}, sel={sel} exit={proc.returncode}"
            )
        else:
            print(f"[OK] Day {day}, method={method}, seed={seed}, sel={sel}")
        sys.stdout.flush()
    finally:
        sem.release()


def main():
    parser = argparse.ArgumentParser(
        description="复用旧 selected_indices，使用 Conformer+PEFT 重新微调 Day 8-12。"
    )
    parser.add_argument("--days", type=int, nargs="+", default=[8, 9, 10, 11, 12])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--num_gpus", type=int, default=5, help="使用的GPU数量（默认5，对应物理GPU 2-6）")
    parser.add_argument("--gpu_start", type=int, default=2, help="起始物理GPU ID（默认2，即使用GPU 2-6）")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    scripts_dir = Path(__file__).resolve().parent
    logs_dir = scripts_dir / "outputs" / "automated_experiments" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(base_dir, args.days, args.seeds)
    if not tasks:
        print("没有找到任何旧的微调 config.pkl，请确认老实验已经跑完。")
        return

    print(f"共发现 {len(tasks)} 条需要重新微调的任务。")
    print(f"使用物理 GPU {args.gpu_start} 到 {args.gpu_start + args.num_gpus - 1}（共 {args.num_gpus} 张卡）")
    print(f"每张卡只跑一个任务（不并行）")

    # 为每块 GPU 设置最多 1 个并行任务的信号量（每卡只跑一个任务）
    num_gpus = max(1, args.num_gpus)
    gpu_semaphores = {gid: threading.Semaphore(1) for gid in range(num_gpus)}

    # 线程池最大并行数 = GPU 数（每卡一个任务）
    futures = []
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        for idx, task in enumerate(tasks):
            gpu_id = idx % num_gpus
            futures.append(
                executor.submit(
                    run_single_task, task, gpu_id, scripts_dir, logs_dir, gpu_semaphores, args.gpu_start
                )
            )

        # 实时进度
        done = 0
        total = len(futures)
        for f in as_completed(futures):
            done += 1
            print(f"[PROGRESS] {done}/{total} 任务完成")
            sys.stdout.flush()

    print("✅ 所有 PEFT 微调任务已结束。")


if __name__ == "__main__":
    main()

