#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
复用旧 selected_indices，使用 Conformer + PEFT（CA Block + AdaptFFN）重新微调。

默认只跑 Day 8 的 real_slpe/model_b，各 10 个 seed。
"""

import argparse
import os
import pickle
import sys
import subprocess
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def build_tasks(base_dir: Path, days, seeds, exp_suffix: str):
    tasks = []
    pretrained_ndays = 7

    for day in days:
        day_root = base_dir / "outputs" / "model_test" / f"{pretrained_ndays}-{day}"
        for method in ["real_slpe", "model_b"]:
            for seed in seeds:
                old_dir = day_root / method / f"seed{seed}"
                cfg_path = old_dir / "config.pkl"
                if not cfg_path.exists():
                    continue
                new_dir = day_root / f"{method}_{exp_suffix}" / f"seed{seed}"
                tasks.append(
                    {
                        "day": day,
                        "method": method,
                        "seed": seed,
                        "old_dir": old_dir,
                        "new_dir": new_dir,
                        "config_path": cfg_path,
                    }
                )
    return tasks


def prepare_config(task, cfg_name: str):
    cfg_path = task["config_path"]
    new_dir: Path = task["new_dir"]
    new_dir.mkdir(parents=True, exist_ok=True)

    with open(cfg_path, "rb") as f:
        config = pickle.load(f)

    config["pretrainedModelOutputPath"] = str(new_dir)
    config["use_adapter"] = True
    config["use_ca_block"] = True
    config["adapter_bottleneck"] = 64
    config["ca_bottleneck"] = 64
    config["adapter_init_scale"] = float(task["sigma_init"])
    config["ca_init_scale"] = float(task["sigma_init"])
    config["scheduler_type"] = str(task["scheduler_type"])
    config["use_grad_clip"] = bool(task["use_grad_clip"])
    config["grad_clip_max_norm"] = float(task["grad_clip_max_norm"])
    config.setdefault("train_output_head", True)

    new_cfg_path = new_dir / cfg_name
    with open(new_cfg_path, "wb") as f:
        pickle.dump(config, f)
    return new_cfg_path


def run_single_task(
    task,
    gpu_id,
    scripts_dir: Path,
    logs_dir: Path,
    gpu_semaphores,
    physical_gpu_start=2,
    cfg_name: str = "config_peft_nol2sp.pkl",
    log_suffix: str = "peft_nol2sp",
):
    sem = gpu_semaphores[gpu_id]
    sem.acquire()

    try:
        new_cfg_path = prepare_config(task, cfg_name)

        day = task["day"]
        method = task["method"]
        seed = task["seed"]
        log_name = f"finetune_{method}_7_{day}_seed{seed}_{log_suffix}.log"
        log_file = logs_dir / log_name

        cmd = [
            sys.executable,
            str(scripts_dir / "run_finetune_from_config.py"),
            "--config_path",
            str(new_cfg_path),
        ]

        env = os.environ.copy()
        physical_gpu_id = physical_gpu_start + gpu_id
        env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)

        with open(log_file, "w", encoding="utf-8") as f:
            print(
                f"[RUN] Day {day}, method={method}, seed={seed} on Physical GPU {physical_gpu_id} (logical {gpu_id})",
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
            print(f"[FAIL] Day {day}, method={method}, seed={seed}, exit={proc.returncode}")
        else:
            print(f"[OK] Day {day}, method={method}, seed={seed}")
        sys.stdout.flush()
    finally:
        sem.release()


def main():
    parser = argparse.ArgumentParser(
        description="复用旧 selected_indices，使用 Conformer+PEFT（CA Block + AdaptFFN）重新微调 Day8。"
    )
    parser.add_argument("--days", type=int, nargs="+", default=[8])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--num_gpus", type=int, default=5, help="使用的GPU数量（默认5，对应物理GPU 2-6）")
    parser.add_argument("--gpu_start", type=int, default=2, help="起始物理GPU ID（默认2，即使用GPU 2-6）")
    parser.add_argument(
        "--exp_tag",
        type=str,
        default="peft_nol2sp_schemeA",
        help="实验标签，用于输出目录与日志后缀",
    )
    parser.add_argument("--sigma_init", type=float, default=0.01, help="CA/AdaptFFN 的 sigma 初始值")
    parser.add_argument("--scheduler_type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--use_grad_clip", action="store_true", help="启用梯度范数裁剪")
    parser.add_argument("--grad_clip_max_norm", type=float, default=1.0, help="梯度裁剪 max_norm")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    scripts_dir = Path(__file__).resolve().parent
    logs_dir = scripts_dir / "outputs" / "automated_experiments" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    exp_suffix = args.exp_tag
    cfg_name = f"config_{exp_suffix}.pkl"
    log_suffix = exp_suffix

    tasks = build_tasks(base_dir, args.days, args.seeds, exp_suffix)
    for t in tasks:
        t["sigma_init"] = args.sigma_init
        t["scheduler_type"] = args.scheduler_type
        t["use_grad_clip"] = bool(args.use_grad_clip)
        t["grad_clip_max_norm"] = float(args.grad_clip_max_norm)
    if not tasks:
        print("没有找到任何旧的微调 config.pkl，请确认老实验已经跑完。")
        return

    print(f"共发现 {len(tasks)} 条任务（days={args.days}, seeds={args.seeds}）。")
    print(
        f"sigma_init={args.sigma_init}, scheduler_type={args.scheduler_type}, "
        f"use_grad_clip={args.use_grad_clip}, grad_clip_max_norm={args.grad_clip_max_norm}"
    )
    print(f"使用物理 GPU {args.gpu_start} 到 {args.gpu_start + args.num_gpus - 1}（共 {args.num_gpus} 张卡）")
    print("每张卡只跑一个任务（不并行）")

    num_gpus = max(1, args.num_gpus)
    gpu_semaphores = {gid: threading.Semaphore(1) for gid in range(num_gpus)}

    futures = []
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        for idx, task in enumerate(tasks):
            gpu_id = idx % num_gpus
            futures.append(
                executor.submit(
                    run_single_task,
                    task,
                    gpu_id,
                    scripts_dir,
                    logs_dir,
                    gpu_semaphores,
                    args.gpu_start,
                    cfg_name,
                    log_suffix,
                )
            )

        done = 0
        total = len(futures)
        for _ in as_completed(futures):
            done += 1
            print(f"[PROGRESS] {done}/{total} 任务完成")
            sys.stdout.flush()

    print("✅ 所有 PEFT（CA Block + AdaptFFN）微调任务已结束。")


if __name__ == "__main__":
    main()

