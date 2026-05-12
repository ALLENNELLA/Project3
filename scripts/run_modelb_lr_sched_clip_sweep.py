#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
只跑 Day8 model_b 的微调参数扫描：
- 不使用梯度裁剪（代码内固定关闭）
- scheduler: linear + cosine
- cosine 形状固定：T_max = 1.0 * nBatch

`--task_filter lr_rich`：更丰富的学习率区间（跨不同数量级与区间宽度）。
`--task_filter compare_table4`：仅 results.md 表中 1/2/5/6 四组。
`--task_filter lr_high`：仅高学习率4组（3e-4→1e-5, 4e-4→1e-5, 5e-4→1e-5, 1e-3→5e-5）。
"""

import argparse
import os
import pickle
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def fmt_float(x: float) -> str:
    return f"{x:.1e}".replace("+0", "").replace("-0", "-").replace(".", "p")


def build_tasks(
    base_dir: Path,
    day: int,
    seeds,
    *,
    task_filter: str,
):
    tasks = []
    day_root = base_dir / "outputs" / "model_test" / f"7-{day}"
    method = "model_b"
    clip_tag = "noclip"

    if task_filter == "compare_table4":
        lr_pairs = [(1e-4, 5e-6), (5e-5, 1e-6)]
    elif task_filter == "all":
        lr_pairs = [(1e-4, 5e-6), (5e-5, 1e-6), (1e-5, 2e-7)]
    elif task_filter == "lr_rich":
        lr_pairs = [
            (2e-4, 2e-5),
            (1.5e-4, 1e-5),
            (1.2e-4, 8e-6),
            (1e-4, 1e-5),
            (1e-4, 5e-6),
            (8e-5, 5e-6),
            (5e-5, 5e-6),
            (5e-5, 1e-6),
            (3e-5, 1e-6),
            (1e-5, 1e-6),
            (1e-5, 2e-7),
            (5e-6, 1e-7),
        ]
    elif task_filter == "lr_high":
        lr_pairs = [
            (3e-4, 1e-5),
            (4e-4, 1e-5),
            (5e-4, 1e-5),
            (1e-3, 5e-5),
        ]
    else:
        raise ValueError(f"unknown task_filter: {task_filter}")

    for seed in seeds:
        old_cfg = day_root / method / f"seed{seed}" / "config.pkl"
        if not old_cfg.exists():
            continue

        # linear
        for lr_start, lr_end in lr_pairs:
            exp_tag = f"modelb_sigma001_linear_{fmt_float(lr_start)}_to_{fmt_float(lr_end)}_{clip_tag}"
            tasks.append(
                {
                    "seed": seed,
                    "config_path": old_cfg,
                    "scheduler_type": "linear",
                    "lr_start": lr_start,
                    "lr_end": lr_end,
                    "cosine_ratio": None,
                    "exp_tag": exp_tag,
                    "new_dir": day_root / f"{method}_{exp_tag}" / f"seed{seed}",
                }
            )

        # cosine（固定形状：T_max=1.0*nBatch）
        for lr_start, lr_end in lr_pairs:
            exp_tag = (
                f"modelb_sigma001_cosine_tmax1p0_"
                f"{fmt_float(lr_start)}_to_{fmt_float(lr_end)}_{clip_tag}"
            )
            tasks.append(
                {
                    "seed": seed,
                    "config_path": old_cfg,
                    "scheduler_type": "cosine",
                    "lr_start": lr_start,
                    "lr_end": lr_end,
                    "cosine_ratio": 1.0,
                    "exp_tag": exp_tag,
                    "new_dir": day_root / f"{method}_{exp_tag}" / f"seed{seed}",
                }
            )
    return tasks


def already_done(task) -> bool:
    return (task["new_dir"] / "results" / "finetuned_cer.txt").exists()


def prepare_config(task, day_root: Path):
    task["new_dir"].mkdir(parents=True, exist_ok=True)
    with open(task["config_path"], "rb") as f:
        cfg = pickle.load(f)

    cfg["pretrainedModelOutputPath"] = str(task["new_dir"])
    cfg["use_adapter"] = True
    cfg["use_ca_block"] = True
    cfg["adapter_bottleneck"] = 64
    cfg["ca_bottleneck"] = 64
    cfg["adapter_init_scale"] = 0.01
    cfg["ca_init_scale"] = 0.01

    cfg["lrStart"] = float(task["lr_start"])
    cfg["lrEnd"] = float(task["lr_end"])
    cfg["scheduler_type"] = task["scheduler_type"]
    if task["scheduler_type"] == "cosine":
        cfg["cosine_tmax"] = max(1, int(cfg["nBatch"] * float(task["cosine_ratio"])))

    # 梯度裁剪固定关闭
    cfg["use_grad_clip"] = False
    cfg.setdefault("train_output_head", True)

    cfg_path = task["new_dir"] / f"config_{task['exp_tag']}.pkl"
    with open(cfg_path, "wb") as f:
        pickle.dump(cfg, f)
    return cfg_path


def run_one(task, gpu_slot: int, gpu_ids, scripts_dir: Path, logs_dir: Path, semaphores, day_root: Path, force_rerun: bool):
    sem = semaphores[gpu_slot]
    sem.acquire()
    try:
        if (not force_rerun) and already_done(task):
            return task["exp_tag"], task["seed"], "SKIP_DONE", 0

        cfg_path = prepare_config(task, day_root)
        gpu_id = gpu_ids[gpu_slot]
        log_file = logs_dir / f"finetune_model_b_7_8_seed{task['seed']}_{task['exp_tag']}.log"
        cmd = [sys.executable, str(scripts_dir / "run_finetune_from_config.py"), "--config_path", str(cfg_path)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        with open(log_file, "w", encoding="utf-8") as f:
            print(
                f"[RUN] exp={task['exp_tag']} seed={task['seed']} gpu={gpu_id} "
                f"scheduler={task['scheduler_type']} lr=({task['lr_start']}->{task['lr_end']}) "
                f"grad_clip=off tmax=1.0",
                file=f,
                flush=True,
            )
            p = subprocess.run(cmd, cwd=str(scripts_dir), env=env, stdout=f, stderr=subprocess.STDOUT)
        status = "OK" if p.returncode == 0 else f"FAIL({p.returncode})"
        return task["exp_tag"], task["seed"], status, p.returncode
    finally:
        sem.release()


def main():
    parser = argparse.ArgumentParser(description="Run model_b LR/scheduler sweep (no grad-clip, cosine tmax fixed 1.0).")
    parser.add_argument("--day", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    parser.add_argument("--per_gpu_concurrency", type=int, default=4)
    parser.add_argument("--force_rerun", action="store_true", help="忽略已有结果并强制重跑")
    parser.add_argument(
        "--task_filter",
        choices=["all", "compare_table4", "lr_rich", "lr_high"],
        default="lr_rich",
        help="all=旧三档lr；compare_table4=四组对照；lr_rich=扩展多数量级lr区间；lr_high=高学习率4组",
    )
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    base_dir = scripts_dir.parent
    day_root = base_dir / "outputs" / "model_test" / f"7-{args.day}"
    logs_dir = scripts_dir / "outputs" / "automated_experiments" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(
        base_dir,
        args.day,
        args.seeds,
        task_filter=args.task_filter,
    )
    if not tasks:
        print("未找到任务，请检查基础 model_b/seed*/config.pkl 是否存在。")
        return

    per_gpu = max(1, int(args.per_gpu_concurrency))
    print(f"total tasks={len(tasks)}, day={args.day}, seeds={args.seeds}, gpu_ids={args.gpu_ids}, per_gpu={per_gpu}")
    semaphores = {i: threading.Semaphore(per_gpu) for i in range(len(args.gpu_ids))}

    done = 0
    with ThreadPoolExecutor(max_workers=len(args.gpu_ids) * per_gpu) as ex:
        futures = []
        for idx, t in enumerate(tasks):
            gpu_slot = idx % len(args.gpu_ids)
            futures.append(
                ex.submit(
                    run_one,
                    t,
                    gpu_slot,
                    args.gpu_ids,
                    scripts_dir,
                    logs_dir,
                    semaphores,
                    day_root,
                    bool(args.force_rerun),
                )
            )

        for fu in as_completed(futures):
            done += 1
            exp_tag, seed, status, _ = fu.result()
            print(f"[{done}/{len(futures)}] {exp_tag} seed{seed} {status}")
            sys.stdout.flush()

    print("✅ model_b lr/scheduler sweep finished.")


if __name__ == "__main__":
    main()

