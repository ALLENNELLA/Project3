#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行批量训练 Model A，落盘到 <base_dir>/outputs/model_train/<model>-{n}days-seed{s}/。

特性：
- 多 GPU：--gpus；每卡并发：--per_gpu（同一 GPU 上多进程分时，注意显存）
- 默认可将每种 n_days 扩展到 40 个 seed（0..39）；已存在 modelWeights.pth 可跳过
- 可选从现有 model_train 目录扫描 n_days（--discover_ndays）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Set, Tuple


@dataclass
class Task:
    n_days: int
    seed: int
    gpu: int
    log_file: Path
    out_dir: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="并行训练 Model A（多 seed / 多天数据）。")
    p.add_argument("--base_dir", type=str, default="/root/25S151115/project3")
    p.add_argument("--scripts_dir", type=str, default=None, help="默认 <base_dir>/scripts")
    p.add_argument("--model_a_name", type=str, default="conformer", choices=["gru", "moganet", "conformer", "conformer1"])

    p.add_argument("--ndays", type=int, nargs="+", default=None, help="训练天数列表，如 5 7；不传则结合 discover / 数据目录推断")
    p.add_argument("--discover_ndays", action="store_true", help="从 outputs/model_train 下已有目录解析 n_days（仅含当前 model_a_name）")
    p.add_argument(
        "--require_dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="仅调度 data/ptDecoder_ctc{n} 存在的 n（默认开启）",
    )

    p.add_argument("--seeds", type=int, nargs="+", default=None, help="显式 seed 列表；与 num_seeds 二选一")
    p.add_argument("--num_seeds", type=int, default=40, help="使用 seed 0..num_seeds-1（当未传 --seeds）")
    p.add_argument("--seed_start", type=int, default=0, help="与 num_seeds 联用：seed_start..seed_start+num_seeds-1")

    p.add_argument("--gpus", type=int, nargs="+", default=[0])
    p.add_argument("--per_gpu", type=int, default=1, help="每块 GPU 上同时运行的训练进程数")

    p.add_argument("--python_bin", type=str, default="python")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="即使已有 modelWeights.pth 也重训")

    p.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="任务日志目录；默认 <base_dir>/scripts/outputs/model_a_parallel_logs/<tag>",
    )
    p.add_argument("--log_tag", type=str, default=None, help="日志子目录名；默认时间戳")
    return p.parse_args()


def model_train_dir_name(model_name: str, n_days: int, seed: int) -> str:
    return f"{model_name}-{n_days}days-seed{seed}"


def discover_ndays_from_model_train(model_train_root: Path, model_name: str) -> List[int]:
    pat = re.compile(rf"^{re.escape(model_name)}-(\d+)days-seed\d+$")
    found: Set[int] = set()
    if not model_train_root.is_dir():
        return []
    for p in model_train_root.iterdir():
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if m:
            found.add(int(m.group(1)))
    return sorted(found)


def default_ndays_from_data(base_dir: Path) -> List[int]:
    data = base_dir / "data"
    out: List[int] = []
    if not data.is_dir():
        return out
    pat = re.compile(r"^ptDecoder_ctc(\d+)$")
    for p in data.iterdir():
        if not p.exists():
            continue
        m = pat.match(p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def build_tasks(
    *,
    ndays_list: List[int],
    seeds: List[int],
    gpus: List[int],
    per_gpu: int,
    base_dir: Path,
    model_name: str,
    log_dir: Path,
    require_dataset: bool,
    overwrite: bool,
) -> Tuple[List[Task], List[str]]:
    gpu_slots: List[int] = []
    for g in gpus:
        for _ in range(max(1, per_gpu)):
            gpu_slots.append(g)
    if not gpu_slots:
        raise ValueError("检查 --gpus 与 --per_gpu")

    log_dir.mkdir(parents=True, exist_ok=True)
    model_train = base_dir / "outputs" / "model_train"

    skipped: List[str] = []
    tasks: List[Task] = []
    slot_idx = 0

    for n_days in ndays_list:
        ds = base_dir / "data" / f"ptDecoder_ctc{n_days}"
        if require_dataset and not ds.exists():
            skipped.append(f"n_days={n_days} skip: missing dataset {ds}")
            continue

        for seed in seeds:
            out_dir = model_train / model_train_dir_name(model_name, n_days, seed)
            if (out_dir / "modelWeights.pth").exists() and not overwrite:
                skipped.append(f"skip existing {out_dir}")
                continue

            gpu = gpu_slots[slot_idx % len(gpu_slots)]
            slot_idx += 1
            log_file = log_dir / f"modela_{model_name}_ndays{n_days}_seed{seed}.log"
            tasks.append(Task(n_days=n_days, seed=seed, gpu=gpu, log_file=log_file, out_dir=out_dir))

    return tasks, skipped


def run_one_task(
    task: Task,
    *,
    python_bin: str,
    scripts_dir: Path,
    base_dir: Path,
    model_name: str,
    dry_run: bool,
    semaphores: dict,
) -> Tuple[bool, str]:
    sem = semaphores[task.gpu]
    sem.acquire()
    try:
        cmd = [
            python_bin,
            str(scripts_dir / "main_pipeline.py"),
            "--mode",
            "train_model_a",
            "--model_a_n_days",
            str(task.n_days),
            "--model_a_name",
            model_name,
            "--base_dir",
            str(base_dir),
            "--seed",
            str(task.seed),
        ]
        if dry_run:
            with open(task.log_file, "w", encoding="utf-8") as f:
                f.write("[DRY_RUN] " + " ".join(cmd) + "\n")
            return True, f"dry_run ndays={task.n_days} seed={task.seed} gpu={task.gpu}"

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(task.gpu)
        env["PYTHONPATH"] = str(scripts_dir)

        with open(task.log_file, "w", encoding="utf-8") as f:
            p = subprocess.run(cmd, cwd=str(scripts_dir), env=env, stdout=f, stderr=subprocess.STDOUT)

        if p.returncode == 0:
            return True, f"ok ndays={task.n_days} seed={task.seed} gpu={task.gpu}"
        return False, f"fail rc={p.returncode} ndays={task.n_days} seed={task.seed} gpu={task.gpu} log={task.log_file}"
    finally:
        sem.release()


def write_tasks_csv(tasks: List[Task], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_days", "seed", "gpu", "out_dir", "log_file"])
        for t in tasks:
            w.writerow([t.n_days, t.seed, t.gpu, str(t.out_dir), str(t.log_file)])


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    scripts_dir = Path(args.scripts_dir).resolve() if args.scripts_dir else (base_dir / "scripts")

    tag = args.log_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir).resolve() if args.log_dir else (scripts_dir / "outputs" / "model_a_parallel_logs" / tag)

    if args.seeds is not None:
        seeds = list(args.seeds)
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))

    model_train_root = base_dir / "outputs" / "model_train"

    if args.ndays is not None:
        ndays_list = sorted(set(args.ndays))
    elif args.discover_ndays:
        ndays_list = discover_ndays_from_model_train(model_train_root, args.model_a_name)
        if not ndays_list:
            ndays_list = default_ndays_from_data(base_dir)
    else:
        ndays_list = default_ndays_from_data(base_dir)

    if not ndays_list:
        raise SystemExit("没有可用的 n_days：请指定 --ndays 或确保 data/ptDecoder_ctc* 存在，或使用 --discover_ndays")

    tasks, skipped = build_tasks(
        ndays_list=ndays_list,
        seeds=seeds,
        gpus=args.gpus,
        per_gpu=args.per_gpu,
        base_dir=base_dir,
        model_name=args.model_a_name,
        log_dir=log_dir,
        require_dataset=args.require_dataset,
        overwrite=args.overwrite,
    )

    meta_dir = log_dir / "meta"
    write_tasks_csv(tasks, meta_dir / "tasks.csv")
    with open(meta_dir / "skipped.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(skipped) + ("\n" if skipped else ""))

    print(f"base_dir={base_dir}")
    print(f"scripts_dir={scripts_dir}")
    print(f"log_dir={log_dir}")
    print(f"ndays={ndays_list} (count={len(ndays_list)})")
    print(f"seeds: {seeds[0]}..{seeds[-1]} (n={len(seeds)})" if seeds else "seeds: (empty)")
    print(f"gpus={args.gpus}, per_gpu={args.per_gpu}, workers={len(args.gpus) * max(1, args.per_gpu)}")
    print(f"tasks_to_run={len(tasks)}, skipped_lines={len(skipped)}")
    for s in skipped[:20]:
        print(f"  {s}")
    if len(skipped) > 20:
        print(f"  ... and {len(skipped) - 20} more (see {meta_dir / 'skipped.txt'})")

    semaphores = {g: threading.Semaphore(max(1, args.per_gpu)) for g in args.gpus}

    ok, fail = 0, 0
    failed_msgs: List[str] = []
    with ThreadPoolExecutor(max_workers=len(args.gpus) * max(1, args.per_gpu)) as ex:
        futs = [
            ex.submit(
                run_one_task,
                t,
                python_bin=args.python_bin,
                scripts_dir=scripts_dir,
                base_dir=base_dir,
                model_name=args.model_a_name,
                dry_run=bool(args.dry_run),
                semaphores=semaphores,
            )
            for t in tasks
        ]
        for fu in as_completed(futs):
            is_ok, msg = fu.result()
            if is_ok:
                ok += 1
            else:
                fail += 1
                failed_msgs.append(msg)
                print(f"[FAILED] {msg}")

    summary = {
        "base_dir": str(base_dir),
        "model_a_name": args.model_a_name,
        "ndays": ndays_list,
        "seeds": seeds,
        "gpus": args.gpus,
        "per_gpu": args.per_gpu,
        "log_dir": str(log_dir),
        "ok": ok,
        "failed": fail,
        "total_ran": len(tasks),
        "skipped_count": len(skipped),
        "dry_run": bool(args.dry_run),
        "overwrite": bool(args.overwrite),
        "failed_msgs": failed_msgs,
    }
    with open(meta_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"done. ok={ok}, failed={fail}, total_ran={len(tasks)}")
    print(f"summary={meta_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
