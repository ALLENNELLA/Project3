#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量训练 Model A：多组训练天数 × 多种子。

默认：前 1、2、3、9、10、12 天 × seed 0–9，共 60 次。

用法（在 project3/scripts 目录下）:
  python run_model_a_grid.py
  python run_model_a_grid.py --dry-run
  python run_model_a_grid.py --skip-existing
  python run_model_a_grid.py --n-days 1 2 3 --seeds 0 1 2

多卡并行（每卡同时只跑 1 个任务，任务队列动态分配）:
  python run_model_a_grid.py --gpus 0,1,3,4,6
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


DEFAULT_N_DAYS = [1, 2, 3, 9, 10, 12]
DEFAULT_SEEDS = list(range(10))


def model_a_weights_path(base_dir: Path, model_name: str, n_days: int, seed: int) -> Path:
    """与 get_train_config 中 outputDir 命名一致：{model}-{n}days-seed{seed}/modelWeights.pth"""
    return (
        base_dir
        / "outputs"
        / "model_train"
        / f"{model_name}-{n_days}days-seed{seed}"
        / "modelWeights.pth"
    )


def parse_gpus(s: str) -> List[int]:
    parts = [p.strip() for p in s.replace(" ", ",").split(",") if p.strip()]
    return [int(p) for p in parts]


def run_train_subprocess(
    cmd: List[str],
    log_file: Path,
    scripts_dir: Path,
    cuda_visible: Optional[str] = None,
) -> int:
    env = os.environ.copy()
    if cuda_visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"# {' '.join(cmd)}\n")
        if cuda_visible is not None:
            lf.write(f"# CUDA_VISIBLE_DEVICES={cuda_visible}\n")
        lf.flush()
        p = subprocess.run(
            cmd,
            cwd=str(scripts_dir),
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env,
        )
    return p.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="批量训练 Model A（多天数 × 多种子）")
    parser.add_argument(
        "--n-days",
        type=int,
        nargs="+",
        default=DEFAULT_N_DAYS,
        help=f"训练使用的天数（默认: {DEFAULT_N_DAYS}）",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="随机种子列表（默认: 0..9）",
    )
    parser.add_argument(
        "--model-a-name",
        type=str,
        default="conformer",
        choices=["gru", "moganet", "conformer", "conformer1"],
        help="Model A 架构名",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/root/25S151115/project3",
        help="项目根目录（含 scripts/、data/）",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="单条实验日志目录（默认: scripts/outputs/model_a_grid_logs/<时间戳>）",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="若该组合下 modelWeights.pth 已存在则跳过",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将执行的命令，不真正训练",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="逗号分隔的物理 GPU 编号（如 0,1,3,4,6）。设置后每张卡并行 1 个任务，"
        "通过 CUDA_VISIBLE_DEVICES 绑定；不设则顺序单进程跑（可用当前默认可见 GPU）",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    scripts_dir = base_dir / "scripts"
    main_py = scripts_dir / "main_pipeline.py"
    if not main_py.is_file():
        print(f"找不到入口: {main_py}", file=sys.stderr)
        return 1

    gpus: Optional[List[int]] = None
    if args.gpus is not None:
        gpus = parse_gpus(args.gpus)
        if not gpus:
            print("--gpus 解析结果为空", file=sys.stderr)
            return 1

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = (
        Path(args.log_dir).resolve()
        if args.log_dir
        else scripts_dir / "outputs" / "model_a_grid_logs" / ts
    )
    log_root.mkdir(parents=True, exist_ok=True)

    tasks = [(n, s) for n in args.n_days for s in args.seeds]
    total = len(tasks)
    print(f"共 {total} 个任务 | base_dir={base_dir}")
    print(f"日志目录: {log_root}")
    if gpus:
        print(f"多卡并行: {len(gpus)} 个工作线程，GPU 绑定 CUDA_VISIBLE_DEVICES = {gpus}（每卡同时 1 任务）")

    ok, skipped, failed = 0, 0, 0
    print_lock = threading.Lock()

    # 先收集需要真正运行的任务（跳过已存在的）
    pending: List[Tuple[int, int, int, Path, List[str]]] = []
    for idx, (n_days, seed) in enumerate(tasks, start=1):
        wpath = model_a_weights_path(base_dir, args.model_a_name, n_days, seed)
        if args.skip_existing and wpath.is_file():
            print(f"[{idx}/{total}] 跳过（已存在）: {wpath}")
            skipped += 1
            continue

        cmd = [
            sys.executable,
            str(main_py),
            "--mode",
            "train_model_a",
            "--base_dir",
            str(base_dir),
            "--model_a_n_days",
            str(n_days),
            "--model_a_name",
            args.model_a_name,
            "--seed",
            str(seed),
        ]
        log_file = log_root / f"model_a_{args.model_a_name}_{n_days}d_seed{seed}.log"
        pending.append((idx, n_days, seed, log_file, cmd))

    def handle_one(
        idx: int, n_days: int, seed: int, log_file: Path, cmd: List[str], gpu: Optional[int]
    ) -> None:
        nonlocal ok, failed
        gpu_s = str(gpu) if gpu is not None else None
        with print_lock:
            extra = f" GPU={gpu}" if gpu is not None else ""
            print(f"[{idx}/{total}] n_days={n_days} seed={seed}{extra} -> {log_file.name}")

        if args.dry_run:
            with print_lock:
                if gpu_s is not None:
                    print(f"   CUDA_VISIBLE_DEVICES={gpu_s}", " ".join(cmd))
                else:
                    print("  ", " ".join(cmd))
            with print_lock:
                ok += 1
            return

        rc = run_train_subprocess(cmd, log_file, scripts_dir, cuda_visible=gpu_s)
        with print_lock:
            if rc == 0:
                ok += 1
            else:
                failed += 1
                print(
                    f"  !! 失败 exit={rc} (GPU={gpu})，见日志: {log_file}",
                    file=sys.stderr,
                )

    if not pending:
        print(f"结束 | 成功={ok} 跳过={skipped} 失败={failed} | 日志: {log_root}")
        return 0 if failed == 0 else 1

    if gpus is None:
        for idx, n_days, seed, log_file, cmd in pending:
            handle_one(idx, n_days, seed, log_file, cmd, gpu=None)
    elif args.dry_run:
        # 仅展示：按轮询标明若多卡并行时各任务会绑到哪张卡
        for i, (idx, n_days, seed, log_file, cmd) in enumerate(pending):
            gpu = gpus[i % len(gpus)]
            handle_one(idx, n_days, seed, log_file, cmd, gpu=gpu)
    else:
        # 每 GPU 一个线程，共享任务队列：任意空闲卡取下一个任务
        task_q: queue.Queue[Optional[Tuple[int, int, int, Path, List[str]]]] = queue.Queue()
        for item in pending:
            task_q.put(item)
        for _ in gpus:
            task_q.put(None)

        def worker(phys_gpu: int) -> None:
            while True:
                item = task_q.get()
                if item is None:
                    break
                idx, n_days, seed, log_file, cmd = item
                handle_one(idx, n_days, seed, log_file, cmd, gpu=phys_gpu)

        threads = [
            threading.Thread(target=worker, args=(gpu,), name=f"gpu{gpu}")
            for gpu in gpus
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(
        f"结束 | 成功={ok} 跳过={skipped} 失败={failed} | 日志: {log_root}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
