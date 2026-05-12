#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行跑 real_slpe 选样微调：
- method = real_slpe
- num_samples = 100（Top-难句等由 sample_selection 内 hard 策略决定）
- days = 8,9,10,11,12
- seeds = 0..9（可改）
- 关闭 adapter / ca_block（与 model_b 复现脚本一致）

特性：
- 支持 --gpus / --per_gpu
- 结果与日志写到独立 run 目录（--base_dir=run_base）
- run 目录内自动链接 data -> 原项目 data
- 默认不按共享缓存算 SLPE（与 launch 一致）；需要可加 --use_slpe_cache
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class Task:
    seed: int
    day: int
    gpu: int
    model_a_path: Path
    log_file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real_slpe (100 samples) finetune for day8-12 and multiple seeds in parallel."
    )
    parser.add_argument("--orig_base", type=str, default="/root/25S151115/project3")
    parser.add_argument("--run_root", type=str, default=None, help="Default: <orig_base>/outputs/repro_runs")
    parser.add_argument("--run_tag", type=str, default=None, help="Default: realslpe100_repro_<timestamp>")
    parser.add_argument("--scripts_dir", type=str, default=None, help="Default: <orig_base>/scripts")

    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--days", type=int, nargs="+", default=[8, 9, 10, 11, 12])
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--per_gpu", type=int, default=2)

    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--pretrained_ndays_eval", type=int, default=7)
    parser.add_argument("--use_slpe_cache", action="store_true", default=False)
    parser.add_argument("--python_bin", type=str, default="python")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args for main_pipeline.py. Use: --extra_args -- --lrStart 1e-4",
    )
    return parser.parse_args()


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src, target_is_directory=True)


def build_tasks(
    seeds: List[int],
    days: List[int],
    gpus: List[int],
    per_gpu: int,
    orig_base: Path,
    run_base: Path,
) -> List[Task]:
    gpu_slots: List[int] = []
    for g in gpus:
        for _ in range(max(1, per_gpu)):
            gpu_slots.append(g)
    if not gpu_slots:
        raise ValueError("No GPU slots available, check --gpus and --per_gpu")

    logs_dir = run_base / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[Task] = []
    slot_idx = 0
    for seed in seeds:
        for day in days:
            gpu = gpu_slots[slot_idx % len(gpu_slots)]
            slot_idx += 1
            model_a_path = orig_base / "outputs" / "model_train" / f"conformer-7days-seed{seed}"
            log_file = logs_dir / f"realslpe100_seed{seed}_day{day}.log"
            tasks.append(
                Task(seed=seed, day=day, gpu=gpu, model_a_path=model_a_path, log_file=log_file)
            )
    return tasks


def run_one_task(
    task: Task,
    *,
    python_bin: str,
    scripts_dir: Path,
    run_base: Path,
    pretrained_ndays_eval: int,
    num_samples: int,
    use_slpe_cache: bool,
    extra_args: List[str],
    dry_run: bool,
    semaphores: Dict[int, threading.Semaphore],
) -> Tuple[bool, str]:
    sem = semaphores[task.gpu]
    sem.acquire()
    try:
        if not task.model_a_path.exists():
            return False, f"missing model_a_path: {task.model_a_path}"

        cmd = [
            python_bin,
            str(scripts_dir / "main_pipeline.py"),
            "--mode",
            "finetune_only",
            "--finetune_method",
            "real_slpe",
            "--finetune_target_days",
            str(task.day),
            "--num_samples",
            str(num_samples),
            "--pretrained_ndays_eval",
            str(pretrained_ndays_eval),
            "--model_a_path_eval",
            str(task.model_a_path),
            "--base_dir",
            str(run_base),
            "--seed",
            str(task.seed),
            "--no-use_adapter",
            "--no-use_ca_block",
        ]
        if use_slpe_cache:
            cmd.append("--use_slpe_cache")
        if extra_args:
            cmd.extend(extra_args)

        if dry_run:
            with open(task.log_file, "w", encoding="utf-8") as f:
                f.write("[DRY_RUN] " + " ".join(cmd) + "\n")
            return True, f"dry_run seed={task.seed} day={task.day} gpu={task.gpu}"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(task.gpu)
        env["PYTHONPATH"] = str(scripts_dir)

        with open(task.log_file, "w", encoding="utf-8") as f:
            p = subprocess.run(cmd, cwd=str(scripts_dir), env=env, stdout=f, stderr=subprocess.STDOUT)

        if p.returncode == 0:
            return True, f"ok seed={task.seed} day={task.day} gpu={task.gpu}"
        return False, f"fail rc={p.returncode} seed={task.seed} day={task.day} gpu={task.gpu} log={task.log_file}"
    finally:
        sem.release()


def write_tasks_csv(tasks: List[Task], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "day", "gpu", "model_a_path", "log_file"])
        for t in tasks:
            writer.writerow([t.seed, t.day, t.gpu, str(t.model_a_path), str(t.log_file)])


def main() -> None:
    args = parse_args()

    orig_base = Path(args.orig_base).resolve()
    scripts_dir = Path(args.scripts_dir).resolve() if args.scripts_dir else (orig_base / "scripts")
    run_root = Path(args.run_root).resolve() if args.run_root else (orig_base / "outputs" / "repro_runs")
    run_tag = args.run_tag or f"realslpe100_repro_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_base = run_root / run_tag

    (run_base / "outputs" / "model_test").mkdir(parents=True, exist_ok=True)
    (run_base / "outputs" / "meta").mkdir(parents=True, exist_ok=True)
    (run_base / "logs").mkdir(parents=True, exist_ok=True)

    ensure_symlink(orig_base / "data", run_base / "data")

    tasks = build_tasks(
        seeds=args.seeds,
        days=args.days,
        gpus=args.gpus,
        per_gpu=args.per_gpu,
        orig_base=orig_base,
        run_base=run_base,
    )
    write_tasks_csv(tasks, run_base / "outputs" / "meta" / "tasks.csv")

    print(f"run_base={run_base}")
    print(f"method=real_slpe, num_samples={args.num_samples}, use_slpe_cache={args.use_slpe_cache}")
    print(f"gpus={args.gpus}, per_gpu={args.per_gpu}, max_workers={len(args.gpus) * max(1, args.per_gpu)}")
    print(f"tasks={len(tasks)} (seeds={args.seeds}, days={args.days})")
    if args.extra_args:
        print(f"extra_args={args.extra_args}")

    semaphores = {g: threading.Semaphore(max(1, args.per_gpu)) for g in args.gpus}

    ok, fail = 0, 0
    failed_msgs: List[str] = []
    with ThreadPoolExecutor(max_workers=len(args.gpus) * max(1, args.per_gpu)) as ex:
        futures = [
            ex.submit(
                run_one_task,
                t,
                python_bin=args.python_bin,
                scripts_dir=scripts_dir,
                run_base=run_base,
                pretrained_ndays_eval=args.pretrained_ndays_eval,
                num_samples=args.num_samples,
                use_slpe_cache=bool(args.use_slpe_cache),
                extra_args=args.extra_args,
                dry_run=bool(args.dry_run),
                semaphores=semaphores,
            )
            for t in tasks
        ]

        for fu in as_completed(futures):
            is_ok, msg = fu.result()
            if is_ok:
                ok += 1
            else:
                fail += 1
                failed_msgs.append(msg)
                print(f"[FAILED] {msg}")

    summary = {
        "run_base": str(run_base),
        "method": "real_slpe",
        "ok": ok,
        "failed": fail,
        "total": len(tasks),
        "dry_run": bool(args.dry_run),
        "use_slpe_cache": bool(args.use_slpe_cache),
        "gpus": args.gpus,
        "per_gpu": args.per_gpu,
        "seeds": args.seeds,
        "days": args.days,
        "num_samples": args.num_samples,
        "pretrained_ndays_eval": args.pretrained_ndays_eval,
        "failed_msgs": failed_msgs,
    }
    with open(run_base / "outputs" / "meta" / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"done. ok={ok}, failed={fail}, total={len(tasks)}")
    print(f"logs_dir={run_base / 'logs'}")
    print(f"results_dir={run_base / 'outputs' / 'model_test'}")
    print(f"summary={run_base / 'outputs' / 'meta' / 'summary.json'}")


if __name__ == "__main__":
    main()
