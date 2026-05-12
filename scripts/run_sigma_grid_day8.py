#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量运行 Day8 的 sigma 网格实验（PEFT: CA Block + AdaptFFN, no L2SP）。

网格：
- sigma_init in {1.0, 0.1, 0.01}

任务：
- methods: real_slpe / model_b
- seeds: 0..9

调度策略：
- 默认 GPU 2-6（可改 --gpu_ids）
- 每张卡同时仅 1 个任务，尽量避免 OOM
- 全局队列混排所有实验组合，提升总体并行吞吐
"""

import argparse
import os
import pickle
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def fmt_num(x: float) -> str:
    s = str(x).replace(".", "p")
    return s


def build_tasks(base_dir: Path, day: int, seeds, sigma_inits):
    tasks = []
    day_root = base_dir / "outputs" / "model_test" / f"7-{day}"
    for sigma_init in sigma_inits:
        exp_tag = f"peft_sigma{fmt_num(sigma_init)}"
        for method in ["real_slpe", "model_b"]:
            for seed in seeds:
                old_dir = day_root / method / f"seed{seed}"
                cfg_path = old_dir / "config.pkl"
                if not cfg_path.exists():
                    continue
                new_dir = day_root / f"{method}_{exp_tag}" / f"seed{seed}"
                tasks.append(
                    {
                        "day": day,
                        "method": method,
                        "seed": seed,
                        "config_path": cfg_path,
                        "new_dir": new_dir,
                        "exp_tag": exp_tag,
                        "sigma_init": float(sigma_init),
                    }
                )
    return tasks


def prepare_config(task):
    task["new_dir"].mkdir(parents=True, exist_ok=True)
    with open(task["config_path"], "rb") as f:
        cfg = pickle.load(f)

    cfg["pretrainedModelOutputPath"] = str(task["new_dir"])
    cfg["use_adapter"] = True
    cfg["use_ca_block"] = True
    cfg["adapter_bottleneck"] = 64
    cfg["ca_bottleneck"] = 64
    cfg["adapter_init_scale"] = task["sigma_init"]
    cfg["ca_init_scale"] = task["sigma_init"]
    cfg.setdefault("train_output_head", True)

    cfg_name = f"config_{task['exp_tag']}.pkl"
    cfg_path = task["new_dir"] / cfg_name
    with open(cfg_path, "wb") as f:
        pickle.dump(cfg, f)
    return cfg_path


def already_done(task) -> bool:
    p = task["new_dir"] / "results" / "finetuned_cer.txt"
    return p.exists()


def run_one(task, gpu_idx, gpu_ids, scripts_dir: Path, logs_dir: Path, semaphores):
    sem = semaphores[gpu_idx]
    sem.acquire()
    try:
        if already_done(task):
            return (task["exp_tag"], task["method"], task["seed"], 0, "SKIP_DONE")

        cfg_path = prepare_config(task)
        gpu_id = gpu_ids[gpu_idx]
        log_file = logs_dir / (
            f"finetune_{task['method']}_7_{task['day']}_seed{task['seed']}_{task['exp_tag']}.log"
        )
        cmd = [
            sys.executable,
            str(scripts_dir / "run_finetune_from_config.py"),
            "--config_path",
            str(cfg_path),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        with open(log_file, "w", encoding="utf-8") as f:
            print(
                (
                    f"[RUN] exp={task['exp_tag']} day={task['day']} method={task['method']} "
                    f"seed={task['seed']} gpu={gpu_id} sigma_init={task['sigma_init']}"
                ),
                file=f,
                flush=True,
            )
            proc = subprocess.run(
                cmd,
                cwd=str(scripts_dir),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )
        status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
        return (task["exp_tag"], task["method"], task["seed"], proc.returncode, status)
    finally:
        sem.release()


def main():
    parser = argparse.ArgumentParser(description="Run Day8 sigma grid experiments.")
    parser.add_argument("--day", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--sigma_inits", type=float, nargs="+", default=[1.0, 0.1, 0.01])
    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6],
        help="物理 GPU IDs",
    )
    parser.add_argument(
        "--per_gpu_concurrency",
        type=int,
        default=1,
        help="每张 GPU 的并发任务数",
    )
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    base_dir = scripts_dir.parent
    logs_dir = scripts_dir / "outputs" / "automated_experiments" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(
        base_dir=base_dir,
        day=args.day,
        seeds=args.seeds,
        sigma_inits=args.sigma_inits,
    )
    if not tasks:
        print("未找到可运行任务，请检查基础 config.pkl 是否存在。")
        return

    print(
        f"总任务数: {len(tasks)} | day={args.day} | seeds={args.seeds} | "
        f"sigma_inits={args.sigma_inits}"
    )
    per_gpu = max(1, int(args.per_gpu_concurrency))
    print(f"GPU IDs: {args.gpu_ids} | 每卡并发={per_gpu}")

    semaphores = {i: threading.Semaphore(per_gpu) for i in range(len(args.gpu_ids))}
    futures = []
    with ThreadPoolExecutor(max_workers=len(args.gpu_ids) * per_gpu) as ex:
        for idx, task in enumerate(tasks):
            gpu_idx = idx % len(args.gpu_ids)
            futures.append(
                ex.submit(
                    run_one,
                    task,
                    gpu_idx,
                    args.gpu_ids,
                    scripts_dir,
                    logs_dir,
                    semaphores,
                )
            )

        done = 0
        total = len(futures)
        for fu in as_completed(futures):
            done += 1
            exp_tag, method, seed, _, status = fu.result()
            print(f"[{done}/{total}] {exp_tag} {method} seed{seed} {status}")
            sys.stdout.flush()

    print("✅ sigma 网格实验全部调度完成。")


if __name__ == "__main__":
    main()

