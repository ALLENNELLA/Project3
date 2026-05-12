#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量运行跨天迁移 A/B/D 组实验：
- 先训练对应的 Model B
- 再跑 model_b 微调
- 再跑 random 微调（每个 seed 默认 5 个 selection_seed）

设计与当前项目约定：
- A: N=2, K=1, M=3   -> train_b_ndays=1, train_days=[2], eval_ndays=2, target_days=[3]
- B: N=5, K=2, M=7   -> train_b_ndays=3, train_days=[4,5], eval_ndays=5, target_days=[6,7]
- D: N=12, K=3, M=21 -> train_b_ndays=9, train_days=[10,11,12], eval_ndays=12, target_days=[13..21]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List


EXPERIMENT_GROUPS: Dict[str, Dict] = {
    "A": {
        "train_b_ndays": 1,
        "train_days": [2],
        "eval_ndays": 2,
        "target_days": [3],
    },
    "B": {
        "train_b_ndays": 3,
        "train_days": [4, 5],
        "eval_ndays": 5,
        "target_days": [6, 7],
    },
    "D": {
        "train_b_ndays": 9,
        "train_days": [10, 11, 12],
        "eval_ndays": 12,
        "target_days": [13, 14, 15, 16, 17, 18, 19, 20, 21],
    },
}


def build_tasks(
    base_dir: Path,
    scripts_dir: Path,
    groups: List[str],
    seeds: List[int],
    num_samples: int,
    random_selection_seeds: List[int],
) -> List[Dict]:
    tasks: List[Dict] = []
    for g in groups:
        cfg = EXPERIMENT_GROUPS[g]
        train_b_ndays = cfg["train_b_ndays"]
        eval_ndays = cfg["eval_ndays"]
        train_days = cfg["train_days"]
        target_days = cfg["target_days"]

        for seed in seeds:
            model_a_train_b = base_dir / "outputs" / "model_train" / f"conformer-{train_b_ndays}days-seed{seed}"
            model_a_eval = base_dir / "outputs" / "model_train" / f"conformer-{eval_ndays}days-seed{seed}"
            model_b_out = scripts_dir / "outputs" / "model_b" / f"slpe-group{g}-{train_b_ndays}days-seed{seed}"

            tasks.append(
                {
                    "type": "train_b",
                    "group": g,
                    "seed": seed,
                    "cmd": [
                        "python",
                        str(scripts_dir / "main_pipeline.py"),
                        "--mode",
                        "train_only",
                        "--metric",
                        "slpe",
                        "--train_days",
                        *[str(d) for d in train_days],
                        "--pretrained_ndays_train_b",
                        str(train_b_ndays),
                        "--model_a_path_train_b",
                        str(model_a_train_b),
                        "--output_dir",
                        str(model_b_out),
                        "--base_dir",
                        str(base_dir),
                        "--seed",
                        str(seed),
                        "--prompt_format",
                        "feature_injection",
                    ],
                    "ok_path": model_b_out / "config.pkl",
                }
            )

            for day in target_days:
                finetune_model_b_dir = (
                    base_dir / "outputs" / "model_test" / f"{eval_ndays}-{day}" / f"model_b_group{g}" / f"seed{seed}"
                )
                tasks.append(
                    {
                        "type": "finetune_model_b",
                        "group": g,
                        "seed": seed,
                        "day": day,
                        "cmd": [
                            "python",
                            str(scripts_dir / "main_pipeline.py"),
                            "--mode",
                            "finetune_only",
                            "--finetune_method",
                            "model_b",
                            "--finetune_target_days",
                            str(day),
                            "--num_samples",
                            str(num_samples),
                            "--pretrained_ndays_eval",
                            str(eval_ndays),
                            "--model_a_path_eval",
                            str(model_a_eval),
                            "--model_b_path",
                            str(model_b_out),
                            "--base_dir",
                            str(base_dir),
                            "--seed",
                            str(seed),
                        ],
                        "ok_path": finetune_model_b_dir / "results" / "finetuned_cer.txt",
                    }
                )

                for sel in random_selection_seeds:
                    finetune_random_dir = (
                        base_dir
                        / "outputs"
                        / "model_test"
                        / f"{eval_ndays}-{day}"
                        / f"random_group{g}"
                        / f"seed{seed}_sel{sel}"
                    )
                    tasks.append(
                        {
                            "type": "finetune_random",
                            "group": g,
                            "seed": seed,
                            "day": day,
                            "selection_seed": sel,
                            "cmd": [
                                "python",
                                str(scripts_dir / "main_pipeline.py"),
                                "--mode",
                                "finetune_only",
                                "--finetune_method",
                                "random",
                                "--finetune_target_days",
                                str(day),
                                "--num_samples",
                                str(num_samples),
                                "--pretrained_ndays_eval",
                                str(eval_ndays),
                                "--model_a_path_eval",
                                str(model_a_eval),
                                "--base_dir",
                                str(base_dir),
                                "--seed",
                                str(seed),
                                "--selection_seed",
                                str(sel),
                            "--no-use_adapter",
                            "--no-use_ca_block",
                            ],
                            "ok_path": finetune_random_dir / "results" / "finetuned_cer.txt",
                        }
                    )
    return tasks


def run_one_task(
    task: Dict,
    gpu_id: int,
    scripts_dir: Path,
    logs_dir: Path,
    skip_done: bool,
    gpu_semaphores: Dict[int, threading.Semaphore],
) -> Dict:
    if skip_done and task["ok_path"].exists():
        return {"ok": True, "skipped": True, "task": task, "msg": "already_done"}

    sem = gpu_semaphores[gpu_id]
    sem.acquire()
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = str(scripts_dir)

        logs_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"group{task['group']}_seed{task['seed']}"
        if "day" in task:
            suffix += f"_day{task['day']}"
        if "selection_seed" in task:
            suffix += f"_sel{task['selection_seed']}"
        log_file = logs_dir / f"{task['type']}_{suffix}.log"

        with open(log_file, "w", encoding="utf-8") as f:
            p = subprocess.run(task["cmd"], cwd=str(scripts_dir), env=env, stdout=f, stderr=subprocess.STDOUT)
        return {"ok": p.returncode == 0, "skipped": False, "task": task, "msg": str(log_file)}
    finally:
        sem.release()


def execute_tasks(
    tasks: List[Dict],
    scripts_dir: Path,
    logs_dir: Path,
    skip_done: bool,
    gpus: List[int],
    max_workers: int,
    gpu_semaphores: Dict[int, threading.Semaphore],
) -> Dict[str, int]:
    ok_count = 0
    fail_count = 0
    skip_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for i, task in enumerate(tasks):
            gpu_id = gpus[i % len(gpus)]
            futures.append(
                ex.submit(run_one_task, task, gpu_id, scripts_dir, logs_dir, skip_done, gpu_semaphores)
            )
        for fu in as_completed(futures):
            r = fu.result()
            if r["skipped"]:
                skip_count += 1
                continue
            if r["ok"]:
                ok_count += 1
            else:
                fail_count += 1
                t = r["task"]
                print(f"[FAILED] {t['type']} group={t['group']} seed={t['seed']} day={t.get('day')} -> {r['msg']}")
    return {"ok": ok_count, "skipped": skip_count, "failed": fail_count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A/B/D cross-day experiments for model_b + random finetuning.")
    parser.add_argument("--base_dir", type=str, default="/root/25S151115/project3")
    parser.add_argument("--scripts_dir", type=str, default="/root/25S151115/project3/scripts")
    parser.add_argument("--groups", nargs="+", choices=["A", "B", "D"], default=["A", "B", "D"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--random_selection_seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--gpus", nargs="+", type=int, default=[5, 2, 0])
    parser.add_argument("--tasks_per_gpu", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--skip_done", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    scripts_dir = Path(args.scripts_dir)
    logs_dir = scripts_dir / "outputs" / "automated_experiments" / "logs_abd"

    tasks = build_tasks(
        base_dir=base_dir,
        scripts_dir=scripts_dir,
        groups=args.groups,
        seeds=args.seeds,
        num_samples=args.num_samples,
        random_selection_seeds=args.random_selection_seeds,
    )
    # 确保每个 seed 的 train_b 先执行，再执行对应 seed 的微调任务。
    tasks.sort(
        key=lambda x: (
            x["seed"],
            0 if x["type"] == "train_b" else 1,
            x.get("day", 0),
            x.get("selection_seed", -1),
        )
    )
    train_b_tasks = [t for t in tasks if t["type"] == "train_b"]
    finetune_tasks = [t for t in tasks if t["type"] != "train_b"]

    max_workers = args.max_workers
    if max_workers is None:
        max_workers = len(args.gpus) * args.tasks_per_gpu
    gpu_semaphores = {g: threading.Semaphore(args.tasks_per_gpu) for g in args.gpus}

    print(f"Total tasks: {len(tasks)}")
    print(f"Stage1 train_b tasks: {len(train_b_tasks)}")
    print(f"Stage2 finetune tasks: {len(finetune_tasks)}")
    print(f"Groups: {args.groups}, Seeds: {args.seeds}")
    print(
        f"Using GPUs: {args.gpus}, tasks_per_gpu={args.tasks_per_gpu}, max_workers={max_workers}"
    )

    stage1 = execute_tasks(
        train_b_tasks, scripts_dir, logs_dir, args.skip_done, args.gpus, max_workers, gpu_semaphores
    )
    print(
        f"Stage1 done. ok={stage1['ok']}, skipped={stage1['skipped']}, failed={stage1['failed']}"
    )

    stage2 = execute_tasks(
        finetune_tasks, scripts_dir, logs_dir, args.skip_done, args.gpus, max_workers, gpu_semaphores
    )
    print(
        f"Stage2 done. ok={stage2['ok']}, skipped={stage2['skipped']}, failed={stage2['failed']}"
    )
    print(
        "Done. "
        f"ok={stage1['ok'] + stage2['ok']}, "
        f"skipped={stage1['skipped'] + stage2['skipped']}, "
        f"failed={stage1['failed'] + stage2['failed']}"
    )


if __name__ == "__main__":
    main()
