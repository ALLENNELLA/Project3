#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主入口文件
功能：统一入口，协调所有模块
"""
import os
import argparse
import re
from typing import Optional, List, Dict

from model_a_train_module import train_model_a, get_model_a_path
from model_b_data_module import compute_train_scores, compute_val_scores
from model_b_train_module import train_model_b
from model_b_test_module import compute_overlap_analysis
from finetune_module import finetune_model_a
from model_b_utils import extract_transcriptions


def normalize_selection_strategy(strategy: Optional[str]) -> str:
    if strategy is None:
        return 'hard_top100'
    strategy = strategy.strip()
    m = re.fullmatch(r"ran_?(\d+)_(\d+)_(\d+)", strategy)
    if m:
        return f"ran_{m.group(1)}_{m.group(2)}_{m.group(3)}"
    return strategy


def resolve_model_a_paths(
    pretrained_ndays_train_b: int,
    pretrained_ndays_eval: int,
    model_a_name: str = 'conformer',
    base_dir: str = '/root/25S151115/project3',
    model_a_path: Optional[str] = None,
    model_a_path_train_b: Optional[str] = None,
    model_a_path_eval: Optional[str] = None,
    trained_model_a_path: Optional[str] = None,
    trained_model_a_n_days: Optional[int] = None
) -> Dict[str, str]:
    train_b_path = model_a_path_train_b if model_a_path_train_b is not None else model_a_path
    eval_path = model_a_path_eval if model_a_path_eval is not None else model_a_path

    if train_b_path is None:
        if trained_model_a_path is not None and trained_model_a_n_days == pretrained_ndays_train_b:
            train_b_path = trained_model_a_path
        else:
            train_b_path = get_model_a_path(pretrained_ndays_train_b, model_a_name, base_dir)

    if eval_path is None:
        if trained_model_a_path is not None and trained_model_a_n_days == pretrained_ndays_eval:
            eval_path = trained_model_a_path
        else:
            eval_path = get_model_a_path(pretrained_ndays_eval, model_a_name, base_dir)

    return {
        'train_b': train_b_path,
        'eval': eval_path
    }


def run_model_b_training_pipeline(
    metric: str = 'slpe',  # 'slpe' or 'cer'
    train_days: List[int] = [6, 7],
    val_days: List[int] = [8, 9, 10, 11, 12],
    pretrained_ndays: int = 5,  # 训练B时：用前N天Model A计算SLPE/CER标签（默认前5天）
    model_a_path: Optional[str] = None,
    data_path: Optional[str] = None,
    output_dir: str = 'outputs/model_b',
    base_dir: str = '/root/25S151115/project3',
    **train_kwargs
) -> Dict:
    """
    运行Model B训练流程
    
    Returns:
        Dict包含训练结果和模型路径
    """
    print("="*80)
    print("🚀 Model B训练流程")
    print("="*80)
    
    # 自动生成路径（如果未指定）
    if data_path is None:
        data_path = os.path.join(base_dir, 'data', f'ptDecoder_day6_7')
    
    if model_a_path is None:
        model_a_path = get_model_a_path(pretrained_ndays, 'conformer', base_dir)
    
    # 计算训练数据分数
    print("\n📊 [Step 1/4] 计算训练数据分数...")
    train_scores, train_phoneme_seqs, train_day_indices = compute_train_scores(
        model_a_path=model_a_path,
        data_path=data_path,
        train_days=train_days,
        metric=metric,
        pretrained_ndays=pretrained_ndays,
        batch_size=32,
        device='cuda'
    )
    
    # 计算验证数据分数
    print("\n📊 [Step 2/4] 计算验证数据分数...")
    val_scores, val_phoneme_seqs, val_day_indices = compute_val_scores(
        model_a_path=model_a_path,
        data_path=data_path,
        train_days=train_days,
        metric=metric,
        pretrained_ndays=pretrained_ndays,
        batch_size=32,
        device='cuda'
    )
    
    # 提取transcriptions
    print("\n📊 [Step 3/4] 提取transcriptions...")
    import pickle
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    train_transcriptions = extract_transcriptions(data['train'])
    val_transcriptions = extract_transcriptions(data['val']) if 'val' in data else (extract_transcriptions(data['test']) if 'test' in data else None)
    
    # 训练Model B
    print("\n📊 [Step 4/4] 训练Model B...")
    results = train_model_b(
        train_scores=train_scores,
        train_phoneme_seqs=train_phoneme_seqs,
        train_day_indices=train_day_indices,
        val_scores=val_scores,
        val_phoneme_seqs=val_phoneme_seqs,
        val_day_indices=val_day_indices,
        train_transcriptions=train_transcriptions,
        val_transcriptions=val_transcriptions,
        output_dir=output_dir,
        **train_kwargs
    )
    
    print("\n✅ Model B训练流程完成！")
    return results


def run_test_pipeline(
    model_b_path: str,
    model_a_path: str,
    metric: str = 'slpe',
    val_days: List[int] = [8, 9, 10, 11, 12],
    pretrained_ndays: int = 7,
    top_k_list: List[int] = [50, 100],
    base_dir: str = '/root/25S151115/project3',
    **test_kwargs
) -> Dict:
    """
    运行测试流程（重合度分析）
    """
    print("="*80)
    print("🔬 Model B测试流程")
    print("="*80)
    
    results = compute_overlap_analysis(
        model_a_path=model_a_path,
        model_b_path=model_b_path,
        val_days=val_days,
        metric=metric,
        pretrained_ndays=pretrained_ndays,
        top_k_list=top_k_list,
        base_dir=base_dir,
        **test_kwargs
    )
    
    print("\n✅ 测试流程完成！")
    return results


def run_finetune_pipeline(
    method: str,
    target_days: List[int],
    num_samples: int,
    model_a_path: str,
    metric: str = 'slpe',
    model_b_path: Optional[str] = None,
    pretrained_ndays: int = 7,
    base_dir: str = '/root/25S151115/project3',
    seed: Optional[int] = None,
    selection_seed: Optional[int] = None,
    selection_strategy: str = 'hard_top100',
    **finetune_kwargs
) -> Dict:
    """
    运行微调流程
    """
    print("="*80)
    print("🚀 微调流程")
    print("="*80)
    
    results = {}
    for day in target_days:
        print(f"\n📅 微调Day {day}...")
        day_results = finetune_model_a(
            method=method,
            model_a_path=model_a_path,
            target_day=day,
            num_samples=num_samples,
            model_b_path=model_b_path,
            pretrained_ndays=pretrained_ndays,
            base_dir=base_dir,
            seed=seed if seed is not None else finetune_kwargs.get('seed', 0),
            selection_seed=selection_seed if selection_seed is not None else finetune_kwargs.get('selection_seed'),
            selection_strategy=selection_strategy,
            **finetune_kwargs
        )
        results[day] = day_results
    
    print("\n✅ 微调流程完成！")
    return results


def run_model_a_training_pipeline(
    n_days: int = 7,
    model_name: str = 'conformer',
    base_dir: str = '/root/25S151115/project3',
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    **train_kwargs
) -> Dict:
    """
    运行Model A训练流程
    
    Args:
        n_days: 训练使用的天数
        model_name: 模型名称
        base_dir: 基础目录
        **train_kwargs: 其他训练参数
    
    Returns:
        Dict包含训练结果和模型路径
    """
    print("="*80)
    print("🚀 Model A训练流程")
    print("="*80)
    
    if seed is not None:
        train_kwargs['seed'] = seed
    if output_dir is not None:
        train_kwargs['output_dir'] = output_dir
    results = train_model_a(
        n_days=n_days,
        model_name=model_name,
        base_dir=base_dir,
        **train_kwargs
    )
    
    print("\n✅ Model A训练流程完成！")
    return results


def run_full_pipeline(
    train_model_a_first: bool = False,
    model_a_n_days: int = 7,
    model_a_name: str = 'conformer',
    do_finetune: bool = False,
    metric: str = 'slpe',
    train_days: List[int] = [6, 7],
    val_days: List[int] = [8, 9, 10, 11, 12],
    pretrained_ndays: int = 5,
    pretrained_ndays_eval: Optional[int] = None,
    finetune_method: Optional[str] = None,
    finetune_target_days: Optional[List[int]] = None,
    num_samples: int = 100,
    model_a_path_train_b: Optional[str] = None,
    model_a_path_eval: Optional[str] = None,
    data_path: Optional[str] = None,
    output_dir: str = 'outputs/model_b',
    top_k_list: List[int] = [50, 100],
    base_dir: str = '/root/25S151115/project3',
    selection_strategy: str = 'hard_top100',
    **kwargs
) -> Dict:
    """
    运行完整流程
    
    Args:
        do_finetune: 是否进行微调
        metric: 使用的指标（'slpe'或'cer'）
        train_days: 训练Model B使用的天数
        val_days: 验证/测试使用的天数
        pretrained_ndays: 训练Model B时用于打分的Model A天数（默认5）
        pretrained_ndays_eval: 评测/微调时用的Model A天数（默认7，未传时由 run_full_pipeline 内补全）
        finetune_method: 微调方法（'random', 'length', 'model_b', 'real_cer', 'real_slpe'）
        finetune_target_days: 微调目标天数
        num_samples: 微调时选择的样本数量
    """
    print("="*80)
    print("🚀 完整流程")
    print("="*80)
    
    pretrained_ndays_train_b = pretrained_ndays
    pretrained_ndays_eval = 7 if pretrained_ndays_eval is None else pretrained_ndays_eval

    trained_model_a_path = None
    if train_model_a_first:
        print("\n[Step 0/4] 训练Model A...")
        model_a_results = run_model_a_training_pipeline(
            n_days=model_a_n_days,
            model_name=model_a_name,
            base_dir=base_dir
        )
        trained_model_a_path = model_a_results['model_path']
        print(f"✅ Model A训练完成，路径: {trained_model_a_path}")

    model_a_paths = resolve_model_a_paths(
        pretrained_ndays_train_b=pretrained_ndays_train_b,
        pretrained_ndays_eval=pretrained_ndays_eval,
        model_a_name=model_a_name,
        base_dir=base_dir,
        model_a_path_train_b=model_a_path_train_b,
        model_a_path_eval=model_a_path_eval,
        trained_model_a_path=trained_model_a_path,
        trained_model_a_n_days=model_a_n_days if trained_model_a_path is not None else None
    )
    model_a_path_train_b = model_a_paths['train_b']
    model_a_path_eval = model_a_paths['eval']

    print(f"\n📦 Model A(训练B)路径: {model_a_path_train_b}")
    print(f"📦 Model A(评估)路径: {model_a_path_eval}")
    
    # Step 1: 训练Model B
    print("\n[Step 1/4] 训练Model B...")
    train_results = run_model_b_training_pipeline(
        metric=metric,
        train_days=train_days,
        val_days=val_days,
        pretrained_ndays=pretrained_ndays_train_b,
        model_a_path=model_a_path_train_b,
        data_path=data_path,
        output_dir=output_dir,
        base_dir=base_dir,
        **kwargs
    )
    
    model_b_path = train_results['best_model_path']
    
    # Step 2: 测试Model B
    print("\n[Step 2/4] 测试Model B...")
    test_results = run_test_pipeline(
        model_b_path=model_b_path,
        model_a_path=model_a_path_eval,
        metric=metric,
        val_days=val_days,
        pretrained_ndays=pretrained_ndays_eval,
        top_k_list=top_k_list,
        base_dir=base_dir
    )
    
    # Step 3: 微调（可选）
    finetune_results = None
    if do_finetune and finetune_method and finetune_target_days:
        print("\n[Step 3/4] 微调Model A...")
        finetune_results = run_finetune_pipeline(
            method=finetune_method,
            target_days=finetune_target_days,
            num_samples=num_samples,
            model_a_path=model_a_path_eval,
            model_b_path=model_b_path if finetune_method == 'model_b' else None,
            pretrained_ndays=pretrained_ndays_eval,
            base_dir=base_dir,
            selection_strategy=selection_strategy
        )
    
    return {
        'train_results': train_results,
        'test_results': test_results,
        'finetune_results': finetune_results
    }


def main():
    """主函数，解析参数并执行相应流程"""
    parser = argparse.ArgumentParser(description='Model B Pipeline')
    parser.add_argument('--mode', type=str, choices=['train_model_a', 'train_only', 'test_only', 'finetune_only', 'full'],
                       default='train_only', help='运行模式')
    parser.add_argument('--metric', type=str, choices=['slpe', 'cer'], default='slpe',
                       help='使用的指标')
    parser.add_argument('--train_days', type=int, nargs='+', default=[6, 7],
                       help='训练Model B使用的天数')
    parser.add_argument('--val_days', type=int, nargs='+', default=[8, 9, 10, 11, 12],
                       help='验证/测试使用的天数')
    parser.add_argument('--pretrained_ndays', type=int, default=None,
                       help='若指定：在未单独指定 train_b/eval 时，同时作为两者天数；默认不设表示 train_b=5、eval=7')
    parser.add_argument('--pretrained_ndays_train_b', type=int, default=None,
                       help='训练Model B时用于打分的Model A天数（默认5；与--pretrained_ndays 二选一逻辑见帮助）')
    parser.add_argument('--pretrained_ndays_eval', type=int, default=None,
                       help='评测/微调时Model A天数，用于真实SLPE路径与 conformer-Ndays（默认7）')
    parser.add_argument('--do_finetune', action='store_true',
                       help='是否进行微调')
    parser.add_argument('--finetune_method', type=str, choices=['random', 'length', 'model_b', 'real_cer', 'real_slpe', 'badge'],
                       help='微调方法')
    parser.add_argument('--finetune_target_days', type=int, nargs='+',
                       help='微调目标天数')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='微调时选择的样本数量')
    parser.add_argument('--model_b_path', type=str, default=None,
                       help='Model B路径（test_only或finetune_only时需要）')
    parser.add_argument('--auto_train_model_b', action='store_true', default=False,
                       help='finetune_only + model_b 时，若未提供可用 model_b_path，则先自动训练一个 Model B')
    parser.add_argument('--force_retrain_model_b', action='store_true', default=False,
                       help='finetune_only + model_b + auto_train 时，强制重训 Model B（忽略已传 model_b_path）')
    parser.add_argument('--model_b_output_dir', type=str, default=None,
                       help='自动训练 Model B 时的输出目录（默认自动生成）')
    parser.add_argument('--model_a_path', type=str, default=None,
                       help='Model A路径（如果为None，自动生成）')
    parser.add_argument('--model_a_path_train_b', type=str, default=None,
                       help='训练Model B时使用的Model A路径（优先级高于--model_a_path）')
    parser.add_argument('--model_a_path_eval', type=str, default=None,
                       help='评估/微调时使用的Model A路径（优先级高于--model_a_path）')
    parser.add_argument('--data_path', type=str, default=None,
                       help='数据路径（如果为None，自动生成）')
    parser.add_argument('--output_dir', type=str, default='outputs/model_b',
                       help='输出目录')
    parser.add_argument('--base_dir', type=str, default='/root/25S151115/project3',
                       help='基础目录')
    parser.add_argument('--model_a_n_days', type=int, default=7,
                       help='Model A训练使用的天数')
    parser.add_argument('--model_a_name', type=str, default='conformer',
                       choices=['gru', 'moganet', 'conformer', 'conformer1'],
                       help='Model A模型名称')
    parser.add_argument('--train_model_a_first', action='store_true',
                       help='在完整流程中先训练Model A')
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子（用于可重复实验）')
    parser.add_argument('--selection_seed', type=int, default=None,
                       help='选句子的随机种子（random 与 ran_x_y_z 使用；未指定则回退seed）')
    parser.add_argument(
        '--selection_strategy',
        type=str,
        default='hard_top100',
        help="选样策略：默认 hard_top100；支持 down100、ran_x_y_z（x+y+z必须=100，仅支持100句）或 ranx_y_z。",
    )
    parser.add_argument('--use_adapter', action=argparse.BooleanOptionalAction, default=False,
                       help='微调时是否启用 AdaptFFN（默认关闭，可用 --no-use_adapter 显式关闭）')
    parser.add_argument('--use_ca_block', action=argparse.BooleanOptionalAction, default=False,
                       help='微调时是否启用 CA Block（默认关闭，可用 --no-use_ca_block 显式关闭）')
    parser.add_argument('--top_k_list', type=int, nargs='+', default=[50, 100],
                       help='测试重合率时计算的Top-K列表（如: --top_k_list 100）')
    parser.add_argument('--prompt_format', type=str, default=None,
                       choices=['combined_zh', 'combined_en', 'instruction', 'phoneme_only', 'native_pair', 'feature_injection'],
                       help='指定构建 prompt 时使用的格式（覆盖默认配置）')
    parser.add_argument('--lrStart', type=float, default=None,
                       help='微调起始学习率（覆盖默认配置）')
    parser.add_argument('--lrEnd', type=float, default=None,
                       help='微调结束学习率（覆盖默认配置）')
    parser.add_argument('--output_tag', type=str, default=None,
                       help='微调输出目录标签（用于隔离不同实验，例如 random100 / random_all）')
    parser.add_argument('--use_slpe_cache', action='store_true', default=False,
                       help='real_slpe 方法是否允许读共享 SLPE 缓存（默认关闭，每次按 model_a_path 重算）')
    
    args = parser.parse_args()
    args.selection_strategy = normalize_selection_strategy(args.selection_strategy)
    ran_match = re.fullmatch(r"ran_(\d+)_(\d+)_(\d+)", args.selection_strategy or "")
    if ran_match:
        x, y, z = [int(v) for v in ran_match.groups()]
        if x + y + z != 100:
            raise ValueError(
                f"--selection_strategy={args.selection_strategy} 非法：x+y+z 必须等于 100"
            )
    if args.pretrained_ndays is not None:
        pretrained_ndays_train_b = (
            args.pretrained_ndays_train_b if args.pretrained_ndays_train_b is not None else args.pretrained_ndays
        )
        pretrained_ndays_eval = (
            args.pretrained_ndays_eval if args.pretrained_ndays_eval is not None else args.pretrained_ndays
        )
    else:
        pretrained_ndays_train_b = args.pretrained_ndays_train_b if args.pretrained_ndays_train_b is not None else 5
        pretrained_ndays_eval = args.pretrained_ndays_eval if args.pretrained_ndays_eval is not None else 7
    model_a_paths = resolve_model_a_paths(
        pretrained_ndays_train_b=pretrained_ndays_train_b,
        pretrained_ndays_eval=pretrained_ndays_eval,
        model_a_name=args.model_a_name,
        base_dir=args.base_dir,
        model_a_path=args.model_a_path,
        model_a_path_train_b=args.model_a_path_train_b,
        model_a_path_eval=args.model_a_path_eval
    )
    model_a_path_train_b = model_a_paths['train_b']
    model_a_path_eval = model_a_paths['eval']
    
    if args.mode == 'train_model_a':
        run_model_a_training_pipeline(
            n_days=args.model_a_n_days,
            model_name=args.model_a_name,
            base_dir=args.base_dir,
            seed=args.seed
        )
    
    elif args.mode == 'train_only':
        run_model_b_training_pipeline(
            metric=args.metric,
            train_days=args.train_days,
            val_days=args.val_days,
            pretrained_ndays=pretrained_ndays_train_b,
            model_a_path=model_a_path_train_b,
            data_path=args.data_path,
            output_dir=args.output_dir,
            base_dir=args.base_dir,
            seed=args.seed if args.seed is not None else 42,
            prompt_format=args.prompt_format
        )
    
    elif args.mode == 'test_only':
        if args.model_b_path is None:
            raise ValueError("test_only模式需要提供--model_b_path")

        run_test_pipeline(
            model_b_path=args.model_b_path,
            model_a_path=model_a_path_eval,
            metric=args.metric,
            val_days=args.val_days,
            pretrained_ndays=pretrained_ndays_eval,
            top_k_list=args.top_k_list,
            base_dir=args.base_dir
        )
    
    elif args.mode == 'finetune_only':
        if args.finetune_method is None:
            raise ValueError("finetune_only模式需要提供--finetune_method")
        random_related_selection = (
            args.finetune_method == 'random'
            or (
                args.selection_strategy is not None
                and args.selection_strategy.startswith('ran_')
            )
        )
        effective_seed = (
            args.selection_seed
            if (args.selection_seed is not None and random_related_selection)
            else args.seed
        )
        effective_model_b_path = args.model_b_path
        if args.finetune_method == 'model_b':
            if args.auto_train_model_b:
                should_train_model_b = args.force_retrain_model_b or effective_model_b_path is None
                if should_train_model_b:
                    target_days = args.finetune_target_days or args.val_days
                    target_day_tag = f"day{target_days[0]}" if target_days else "dayNA"
                    seed_value = args.seed if args.seed is not None else 42
                    auto_model_b_output_dir = args.model_b_output_dir or os.path.join(
                        args.base_dir,
                        'outputs',
                        'model_b_auto',
                        f"{target_day_tag}_seed{seed_value}"
                    )
                    print(f"🔧 finetune_only(model_b): 自动训练 Model B 到 {auto_model_b_output_dir}")
                    train_results = run_model_b_training_pipeline(
                        metric=args.metric,
                        train_days=args.train_days,
                        val_days=args.val_days,
                        pretrained_ndays=pretrained_ndays_train_b,
                        model_a_path=model_a_path_train_b,
                        data_path=args.data_path,
                        output_dir=auto_model_b_output_dir,
                        base_dir=args.base_dir,
                        seed=seed_value,
                        prompt_format=args.prompt_format
                    )
                    effective_model_b_path = train_results['best_model_path']
                    print(f"✅ 自动训练完成，Model B 路径: {effective_model_b_path}")
            if effective_model_b_path is None:
                raise ValueError(
                    "finetune_only模式在finetune_method=model_b时需要提供--model_b_path，"
                    "或启用 --auto_train_model_b"
                )

        run_finetune_pipeline(
            method=args.finetune_method,
            target_days=args.finetune_target_days or args.val_days,
            num_samples=args.num_samples,
            model_a_path=model_a_path_eval,
            model_b_path=effective_model_b_path,
            pretrained_ndays=pretrained_ndays_eval,
            base_dir=args.base_dir,
            seed=effective_seed,
            selection_seed=args.selection_seed,
            selection_strategy=args.selection_strategy,
            use_adapter=args.use_adapter,
            use_ca_block=args.use_ca_block,
            lrStart=args.lrStart if args.lrStart is not None else None,
            lrEnd=args.lrEnd if args.lrEnd is not None else None,
            output_tag=args.output_tag,
            use_slpe_cache=args.use_slpe_cache,
        )
    
    elif args.mode == 'full':
        run_full_pipeline(
            train_model_a_first=args.train_model_a_first,
            model_a_n_days=args.model_a_n_days,
            model_a_name=args.model_a_name,
            do_finetune=args.do_finetune,
            metric=args.metric,
            train_days=args.train_days,
            val_days=args.val_days,
            pretrained_ndays=pretrained_ndays_train_b,
            pretrained_ndays_eval=pretrained_ndays_eval,
            finetune_method=args.finetune_method,
            finetune_target_days=args.finetune_target_days,
            num_samples=args.num_samples,
            model_a_path_train_b=model_a_path_train_b,
            model_a_path_eval=model_a_path_eval,
            data_path=args.data_path,
            output_dir=args.output_dir,
            top_k_list=args.top_k_list,
            base_dir=args.base_dir,
            seed=args.seed if args.seed is not None else 42,
            selection_strategy=args.selection_strategy
        )


if __name__ == '__main__':
    main()
