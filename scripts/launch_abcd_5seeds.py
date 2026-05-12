#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABCD 组批量微调调度器（selected best 模型版）：
- random50（默认学习率）：每个 day 跑 selection_seed（默认 0-9）共 10 次
- random100（默认学习率）：每个 day 跑 selection_seed（默认 0-9）共 10 次
- random150/random200/random300/random400（默认学习率）：每个 day 跑 selection_seed（默认 0-9）共 10 次
- full_data（num_samples=-1）：每个 day 跑 1 次
- real_slpe100 / real_cer100（默认学习率）：每个 day 跑 1 次
- length100（默认学习率）：每个 day 跑 1 次（按音素数量最多的100句）
- model_b50/model_b100/model_b150/model_b200/model_b300（默认学习率）：每个 day 跑 1 次（自动推断 model_b_path）

约束：
- 所有任务统一使用 <model_train_selected_root>/conformer-{N}days-best（默认 outputs/model_train_selected，可用 CLI 改）
- 关闭 adapter/ca_block
- 输出目录通过 output_tag + run_tag 隔离，避免覆盖
- 日志目录支持 log_tag，并按任务类型分子目录
- 支持多 GPU 并发（默认 GPU 0/1/2/3/4，每卡并发4任务）
"""
from __future__ import annotations

import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import argparse
import re


BASE_DIR = Path("/root/25S151115/project3")
SCRIPTS_DIR = BASE_DIR / "scripts"
PY = "python3"


GROUPS = {
    "A": {
        "eval_ndays": 2,
        "days": [3],
        "model_b_tpl": "slpe-groupA-1days-seed{seed}",
        "model_b_pretrained_ndays": 1,
        "model_b_train_days": [2],
        "model_b_val_days": [3],
    },
    "B": {
        "eval_ndays": 5,
        "days": [6, 7, 8],
        "model_b_tpl": "slpe-groupB-3days-seed{seed}",
        "model_b_pretrained_ndays": 3,
        "model_b_train_days": [4, 5],
        "model_b_val_days": [6, 7, 8],
    },
    "C": {
        "eval_ndays": 7,
        "days": [10, 11, 12, 13, 14],
        "model_b_tpl": "slpe-5days-seed{seed}",
        "model_b_pretrained_ndays": 5,
        "model_b_train_days": [6, 7],
        "model_b_val_days": [10, 11, 12, 13, 14],
    },
    "D": {
        "eval_ndays": 12,
        "days": [15, 16, 17, 18, 19, 20, 21],
        "model_b_tpl": "slpe-groupD-9days-seed{seed}",
        "model_b_pretrained_ndays": 9,
        "model_b_train_days": [10, 11, 12],
        "model_b_val_days": [15, 16, 17, 18, 19, 20, 21],
    },
}


@dataclass
class Task:
    name: str
    cmd: List[str]
    gpu: int
    log_file: Path


@dataclass(frozen=True)
class Variant:
    key: str
    method: str
    num_samples: int


ALL_VARIANTS = [
    Variant(key="random50", method="random", num_samples=50),
    Variant(key="random100", method="random", num_samples=100),
    Variant(key="random150", method="random", num_samples=150),
    Variant(key="random200", method="random", num_samples=200),
    Variant(key="random300", method="random", num_samples=300),
    Variant(key="random400", method="random", num_samples=400),
    Variant(key="full_data", method="random", num_samples=-1),
    Variant(key="real_slpe100", method="real_slpe", num_samples=100),
    Variant(key="real_cer100", method="real_cer", num_samples=100),
    Variant(key="length100", method="length", num_samples=100),
    Variant(key="model_b50", method="model_b", num_samples=50),
    Variant(key="model_b100", method="model_b", num_samples=100),
    Variant(key="model_b150", method="model_b", num_samples=150),
    Variant(key="model_b200", method="model_b", num_samples=200),
    Variant(key="model_b300", method="model_b", num_samples=300),
    Variant(key="badge100", method="badge", num_samples=100),
]


def cmd(*args: str) -> List[str]:
    return [PY, str(SCRIPTS_DIR / "main_pipeline.py"), *args]

def _normalize_strategy(strategy: str) -> str:
    if strategy is None:
        return "hard_top100"
    strategy = strategy.strip()
    m = re.fullmatch(r"ran_?(\d+)_(\d+)_(\d+)", strategy)
    if m:
        return f"ran_{m.group(1)}_{m.group(2)}_{m.group(3)}"
    return strategy

def _is_randomized_strategy(strategy: str) -> bool:
    return re.fullmatch(r"ran_\d+_\d+_\d+", _normalize_strategy(strategy)) is not None

def _validate_strategy(strategy: str) -> None:
    strategy = _normalize_strategy(strategy)
    if strategy in {"hard_top100", "down100"}:
        return
    ran_match = re.fullmatch(r"ran_(\d+)_(\d+)_(\d+)", strategy)
    if ran_match is None:
        raise ValueError(
            f"Invalid strategy '{strategy}': supported hard_top100 / down100 / ran_x_y_z (or ranx_y_z)"
        )
    x, y, z = [int(v) for v in ran_match.groups()]
    if x + y + z != 100:
        raise ValueError(f"Invalid strategy '{strategy}': x+y+z must equal 100")


def _resolve_task_run_seed(
    base_run_seed: int,
    method: str,
    num_samples: int,
    strategy: str | None,
    selection_seed: int | None,
) -> int:
    if selection_seed is None:
        return base_run_seed
    if method == "random" and num_samples > 0:
        return selection_seed
    if strategy is not None and _is_randomized_strategy(strategy):
        return selection_seed
    return base_run_seed


def build_tasks(
    gpus: List[int],
    groups: List[str],
    variant_keys: List[str],
    logs_dir: Path,
    base_dir: Path,
    run_tag: str,
    run_seeds: List[int] | None = None,
    model_b_seed: int = 0,
    model_b_root: Path | None = None,
    retrain_model_b: bool = False,
    model_train_selected_root: Path | None = None,
    use_slpe_cache: bool = False,
    strategies: List[str] | None = None,
    selection_seeds: List[int] | None = None,
    extra_args: List[str] | None = None,
) -> List[Task]:
    tasks: List[Task] = []
    gpu_idx = 0
    variants = [v for v in ALL_VARIANTS if v.key in set(variant_keys)]
    if model_b_root is None:
        model_b_root = SCRIPTS_DIR / "outputs" / "model_b"
    if model_train_selected_root is None:
        model_train_selected_root = BASE_DIR / "outputs" / "model_train_selected"
    if strategies is None or len(strategies) == 0:
        strategies = ["hard_top100"]
    strategies = [_normalize_strategy(s) for s in strategies]
    # random 类方法额外循环的 selection seeds，默认 0-9
    if selection_seeds is None:
        selection_seeds = list(range(10))
    if run_seeds is None or len(run_seeds) == 0:
        run_seeds = [0]

    for run_seed in run_seeds:
        run_seed_tag = f"rseed{run_seed}"
        for gname in groups:
            gcfg = GROUPS[gname]
            eval_nd = gcfg["eval_ndays"]
            model_a_eval = model_train_selected_root / f"conformer-{eval_nd}days-best"
            model_b_path = model_b_root / gcfg["model_b_tpl"].format(seed=model_b_seed)

            for day in gcfg["days"]:
                common = [
                    "--mode", "finetune_only",
                    "--finetune_target_days", str(day),
                    "--pretrained_ndays_eval", str(eval_nd),
                    "--model_a_path_eval", str(model_a_eval),
                    "--base_dir", str(base_dir),
                ]
                for v in variants:
                    tag = v.key if not run_tag else f"{v.key}_{run_tag}"
                    # random 类方法（num_samples > 0）：对每个 selection_seed 生成独立任务
                    if v.method == "random" and v.num_samples > 0:
                        for sel_seed in selection_seeds:
                            task_run_seed = _resolve_task_run_seed(
                                base_run_seed=run_seed,
                                method=v.method,
                                num_samples=v.num_samples,
                                strategy=None,
                                selection_seed=sel_seed,
                            )
                            c = cmd(
                                *common,
                                "--seed", str(task_run_seed),
                                "--finetune_method", v.method,
                                "--num_samples", str(v.num_samples),
                                "--output_tag", tag,
                                "--no-use_adapter",
                                "--no-use_ca_block",
                                "--selection_seed", str(sel_seed),
                            )
                            if extra_args:
                                c.extend(list(extra_args))
                            task_run_seed_tag = f"rseed{task_run_seed}"
                            name = f"{v.key}_{gname}_day{day}_{task_run_seed_tag}_ssel{sel_seed}"
                            tasks.append(
                                Task(
                                    name=name,
                                    cmd=c,
                                    gpu=gpus[gpu_idx % len(gpus)],
                                    log_file=logs_dir / f"run_seed{task_run_seed}" / v.key / f"{name}.log",
                                )
                            )
                            gpu_idx += 1
                    else:
                        target_strategies = strategies if v.method in {"model_b", "real_slpe"} else ["hard_top100"]
                        for strategy in target_strategies:
                            strat_tag = f"{tag}_{strategy}" if v.method in {"model_b", "real_slpe"} else tag
                            strat_name_suffix = f"_strat{strategy}" if v.method in {"model_b", "real_slpe"} else ""
                            per_strategy_seeds = selection_seeds if _is_randomized_strategy(strategy) else [None]
                            for sel_seed in per_strategy_seeds:
                                task_run_seed = _resolve_task_run_seed(
                                    base_run_seed=run_seed,
                                    method=v.method,
                                    num_samples=v.num_samples,
                                    strategy=strategy,
                                    selection_seed=sel_seed,
                                )
                                task_run_seed_tag = f"rseed{task_run_seed}"
                                name = f"{v.key}_{gname}_day{day}_{task_run_seed_tag}{strat_name_suffix}"
                                c = cmd(
                                    *common,
                                    "--seed", str(task_run_seed),
                                    "--finetune_method", v.method,
                                    "--num_samples", str(v.num_samples),
                                    "--output_tag", strat_tag,
                                    "--no-use_adapter",
                                    "--no-use_ca_block",
                                )
                                if v.method in {"model_b", "real_slpe"}:
                                    c.extend(["--selection_strategy", strategy])
                                if sel_seed is not None:
                                    c.extend(["--selection_seed", str(sel_seed)])
                                    name = f"{name}_ssel{sel_seed}"
                                if v.method == "model_b":
                                    if retrain_model_b:
                                        auto_model_b_out = base_dir / "outputs" / "model_b_auto" / name
                                        model_a_train_b = model_train_selected_root / f"conformer-{gcfg['model_b_pretrained_ndays']}days-best"
                                        c.extend([
                                            "--auto_train_model_b",
                                            "--force_retrain_model_b",
                                            "--model_b_output_dir", str(auto_model_b_out),
                                            "--model_a_path_train_b", str(model_a_train_b),
                                            "--pretrained_ndays_train_b", str(gcfg["model_b_pretrained_ndays"]),
                                            "--train_days", *[str(x) for x in gcfg["model_b_train_days"]],
                                            "--val_days", *[str(x) for x in gcfg["model_b_val_days"]],
                                        ])
                                    else:
                                        c.extend(["--model_b_path", str(model_b_path)])
                                if v.method == "real_slpe" and use_slpe_cache:
                                    c.append("--use_slpe_cache")
                                if extra_args:
                                    c.extend(list(extra_args))
                                tasks.append(
                                    Task(
                                        name=name,
                                        cmd=c,
                                        gpu=gpus[gpu_idx % len(gpus)],
                                        log_file=logs_dir / f"run_seed{task_run_seed}" / v.key / f"{name}.log",
                                    )
                                )
                                gpu_idx += 1

    return tasks


def run_task(task: Task, semaphores: Dict[int, threading.Semaphore], dry_run: bool = False) -> Dict:
    sem = semaphores[task.gpu]
    sem.acquire()
    try:
        task.log_file.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            return {"ok": True, "task": task.name, "gpu": task.gpu, "log": str(task.log_file), "dry_run": True}

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(task.gpu)
        env["PYTHONPATH"] = str(SCRIPTS_DIR)

        with open(task.log_file, "w", encoding="utf-8") as f:
            p = subprocess.run(task.cmd, cwd=str(SCRIPTS_DIR), env=env, stdout=f, stderr=subprocess.STDOUT)
        return {"ok": p.returncode == 0, "task": task.name, "gpu": task.gpu, "log": str(task.log_file), "dry_run": False}
    finally:
        sem.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ABCD finetune tasks with selected best model and multi-GPU.")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--tasks_per_gpu", type=int, default=4)
    parser.add_argument("--groups", nargs="+", choices=sorted(GROUPS.keys()), default=sorted(GROUPS.keys()))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[v.key for v in ALL_VARIANTS],
        default=[v.key for v in ALL_VARIANTS],
    )
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--log_tag", type=str, default="")
    parser.add_argument("--run_seed", nargs="+", type=int, default=[0],
                        help="微调训练随机种子列表（可传多个，自动展开任务）")
    parser.add_argument("--model_b_seed", type=int, default=0,
                        help="自动推断 model_b_path 时使用的 model_b seed（默认 0）")
    parser.add_argument("--model_b_root", type=str, default=str(SCRIPTS_DIR / "outputs" / "model_b"),
                        help="Model B 根目录（默认 scripts/outputs/model_b）")
    parser.add_argument("--retrain_model_b", action="store_true", default=False,
                        help="model_b* 任务不加载已有 Model B，改为每个任务先自动重训再用于选样")
    parser.add_argument("--run_base_dir", type=str, default=str(BASE_DIR),
                        help="实验运行根目录。日志写入 <run_base_dir>/logs，结果由 main_pipeline 写入 <run_base_dir>/outputs。")
    parser.add_argument(
        "--model_train_selected_root",
        type=str,
        default=str(BASE_DIR / "outputs" / "model_train_selected"),
        help="预训练 Model A（各 N days best）根目录，内含 conformer-{N}days-best（默认 project3/outputs/model_train_selected）",
    )
    parser.add_argument("--use_slpe_cache", action="store_true", default=False,
                        help="real_slpe 方法是否允许读共享 SLPE 缓存（默认关闭，按 model_a_path 重算）")
    parser.add_argument(
        "--strategy",
        nargs="+",
        type=str,
        default=["hard_top100"],
        help="model_b/real_slpe 的选样策略列表。默认 hard_top100；支持 down100、ran_x_y_z（x+y+z=100）或 ranx_y_z",
    )
    parser.add_argument("--selection_seeds", nargs="+", type=int, default=list(range(10)),
                        help="random 和 ran_x_y_z 的句子挑选 seed 列表（默认 0-9，共10个）")
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="附加传递给 main_pipeline.py 的参数（例如 --lr_start 3e-4 --lr_end 1e-5 --lr_scheduler linear）。"
             "使用方式：... --extra_args -- --flag1 val1 --flag2 val2",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    for s in args.strategy:
        _validate_strategy(s)

    run_base_dir = Path(args.run_base_dir)
    run_base_dir.mkdir(parents=True, exist_ok=True)
    # 使用独立 run_base_dir 时，确保能访问数据目录（软链接到项目主 data）
    data_link = run_base_dir / "data"
    source_data = BASE_DIR / "data"
    if not data_link.exists():
        data_link.symlink_to(source_data, target_is_directory=True)

    logs_dir = run_base_dir / "logs"
    if args.log_tag:
        logs_dir = logs_dir / args.log_tag
    run_seeds = list(dict.fromkeys(args.run_seed))
    tasks = build_tasks(
        gpus=args.gpus,
        groups=args.groups,
        variant_keys=args.variants,
        logs_dir=logs_dir,
        base_dir=run_base_dir,
        run_tag=args.run_tag,
        run_seeds=run_seeds,
        model_b_seed=args.model_b_seed,
        model_b_root=Path(args.model_b_root),
        retrain_model_b=args.retrain_model_b,
        model_train_selected_root=Path(args.model_train_selected_root),
        use_slpe_cache=args.use_slpe_cache,
        strategies=args.strategy,
        selection_seeds=args.selection_seeds,
        extra_args=args.extra_args,
    )

    max_workers = len(args.gpus) * args.tasks_per_gpu
    semaphores = {g: threading.Semaphore(args.tasks_per_gpu) for g in args.gpus}

    print(f"Total tasks: {len(tasks)}")
    print(f"GPUs: {args.gpus}, tasks_per_gpu={args.tasks_per_gpu}, max_workers={max_workers}")
    print(f"Logs dir: {logs_dir}")
    print(f"Dry run: {args.dry_run}")
    print(f"Groups: {args.groups}")
    print(f"Variants: {args.variants}")
    print(f"Run tag: {args.run_tag or 'none'}")
    print(f"Log tag: {args.log_tag or 'none'}")
    print(f"Run seeds (expanded): {run_seeds}")
    print(f"Model B seed (auto path): {args.model_b_seed}")
    print(f"Model B root: {args.model_b_root}")
    print(f"Retrain Model B for model_b*: {args.retrain_model_b}")
    print(f"Model train selected root: {args.model_train_selected_root}")
    print(f"Run base dir: {run_base_dir}")
    print(f"Use SLPE cache: {args.use_slpe_cache}")
    print(f"Strategies (model_b/real_slpe): {args.strategy}")
    print(f"Selection seeds (random/ran_x_y_z): {args.selection_seeds}")
    if args.extra_args:
        print(f"Extra args -> main_pipeline: {args.extra_args}")

    ok = 0
    fail = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_task, t, semaphores, args.dry_run) for t in tasks]
        for fu in as_completed(futures):
            r = fu.result()
            if r["ok"]:
                ok += 1
            else:
                fail += 1
                print(f"[FAILED] task={r['task']} gpu={r['gpu']} log={r['log']}")

    print(f"Done. ok={ok}, failed={fail}")


if __name__ == "__main__":
    main()

