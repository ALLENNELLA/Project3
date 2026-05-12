#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
严格绑卡调度器：GPU {1,3,4,6}，每卡并发 4 个任务
任务包含：
1) ABD: real_slpe 最难100句
2) ABD: 全量样本微调（num_samples = -1）
3) ABD: random 学习率 1e-5→1e-6（禁用适配器）
默认 seed=0；如需多 seed 可自行扩展 seeds 列表。
"""
from __future__ import annotations
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

BASE_DIR = Path("/root/25S151115/project3")
SCRIPTS_DIR = BASE_DIR / "scripts"
PY = "python"
GPUS = [1, 3, 4, 6]
TASKS_PER_GPU = 4
SEEDS = [0]

def cmd(*args: str) -> List[str]:
    return [PY, str(SCRIPTS_DIR / "main_pipeline.py"), *args]

def build_tasks() -> List[Dict]:
    tasks: List[Dict] = []
    # 组 A: 2->3
    A = dict(eval_nd=2, days=[3])
    # 组 B: 5->6,7
    B = dict(eval_nd=5, days=[6, 7])
    # 组 D: 12->13..21
    D = dict(eval_nd=12, days=list(range(13, 22)))
    groups = [("A", A), ("B", B), ("D", D)]

    for seed in SEEDS:
        for gname, gcfg in groups:
            eval_nd = gcfg["eval_nd"]
            days = gcfg["days"]
            # 1) real_slpe 最难100句
            for day in days:
                tasks.append({
                    "name": f"real_slpe_{gname}_seed{seed}_day{day}",
                    "cmd": cmd(
                        "--mode","finetune_only",
                        "--finetune_method","real_slpe",
                        "--finetune_target_days", str(day),
                        "--pretrained_ndays_eval", str(eval_nd),
                        "--num_samples","100",
                        "--base_dir", str(BASE_DIR),
                        "--seed", str(seed)
                    )
                })
            # 2) 全量样本（num_samples=-1）
            for day in days:
                tasks.append({
                    "name": f"full_random_{gname}_seed{seed}_day{day}",
                    "cmd": cmd(
                        "--mode","finetune_only",
                        "--finetune_method","random",
                        "--finetune_target_days", str(day),
                        "--pretrained_ndays_eval", str(eval_nd),
                        "--num_samples","-1",
                        "--base_dir", str(BASE_DIR),
                        "--seed", str(seed),
                        "--no-use_adapter",
                        "--no-use_ca_block",
                    )
                })
            # 3) random(新lr)
            for day in days:
                tasks.append({
                    "name": f"random_lr_{gname}_seed{seed}_day{day}",
                    "cmd": cmd(
                        "--mode","finetune_only",
                        "--finetune_method","random",
                        "--finetune_target_days", str(day),
                        "--pretrained_ndays_eval", str(eval_nd),
                        "--num_samples","100",
                        "--base_dir", str(BASE_DIR),
                        "--seed", str(seed),
                        "--lrStart","1e-5",
                        "--lrEnd","1e-6",
                        "--no-use_adapter",
                        "--no-use_ca_block",
                    )
                })
    return tasks

def run_task(task: Dict, gpu_id: int, logs_dir: Path, semaphores: Dict[int, threading.Semaphore]) -> Dict:
    sem = semaphores[gpu_id]
    sem.acquire()
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = str(SCRIPTS_DIR)
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / f"{task['name']}.log"
        with open(log_file, "w", encoding="utf-8") as f:
            p = subprocess.run(task["cmd"], cwd=str(SCRIPTS_DIR), env=env, stdout=f, stderr=subprocess.STDOUT)
        return {"ok": p.returncode == 0, "task": task, "log": str(log_file)}
    finally:
        sem.release()

def main():
    logs_dir = SCRIPTS_DIR / "outputs" / "automated_experiments" / "logs_abd_strict"
    tasks = build_tasks()
    semaphores = {g: threading.Semaphore(TASKS_PER_GPU) for g in GPUS}
    print(f"Total tasks: {len(tasks)}; GPUs={GPUS}; per_gpu={TASKS_PER_GPU}")
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=len(GPUS)*TASKS_PER_GPU) as ex:
        futures = []
        for i, t in enumerate(tasks):
            gpu = GPUS[i % len(GPUS)]
            futures.append(ex.submit(run_task, t, gpu, logs_dir, semaphores))
        for fu in as_completed(futures):
            r = fu.result()
            if r["ok"]:
                ok += 1
            else:
                fail += 1
    print(f"Done. ok={ok}, failed={fail}")

if __name__ == "__main__":
    main()

