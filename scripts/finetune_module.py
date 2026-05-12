#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
微调模块
功能：使用不同方法进行微调（随机、model b、real cer、real slpe）
"""
import os
import pickle
import random
import numpy as np
import torch
from typing import Optional, List, Dict
from pathlib import Path

from src.model_a.config import get_finetune_config
from src.model_a.finetune import finetune_model
from src.utils.sample_selection import (
    random_sample_selection,
    length_sample_selection,
    model_b_sample_selection,
    real_cer_sample_selection,
    real_slpe_sample_selection,
)
from model_b_utils import build_prompt


def finetune_model_a(
    method: str,  # 'random', 'length', 'model_b', 'real_cer', 'real_slpe'
    model_a_path: str,
    target_day: int,
    num_samples: int,
    model_b_path: Optional[str] = None,  # method='model_b'时需要
    pretrained_ndays: int = 7,
    batch_size: int = 32,
    device: str = 'cuda',
    output_dir: Optional[str] = None,
    seed: int = 0,
    selection_seed: Optional[int] = None,  # 选句子的seed（random 与 ran_x_y_z 使用）
    selection_strategy: str = 'hard',  # 'hard'/'easy'/'diverse' 或 ran_x_y_z
    use_slpe_cache: bool = False,  # False=强制按model_a_path重算SLPE，True=允许读共享缓存
    base_dir: str = '/root/25S151115/project3',
    **finetune_kwargs
) -> Dict:
    """
    微调Model A
    
    Args:
        method: 样本选择方法
        model_a_path: Model A路径
        target_day: 目标天数
        num_samples: 选择的样本数量
        model_b_path: Model B路径（method='model_b'时需要）
        ... (其他微调参数)
    
    Returns:
        Dict包含：
            - finetuned_model_path: 微调后的模型路径
            - selected_indices: 选择的样本索引
            - metrics: 微调后的指标
    """
    display_method = "full_data" if (method == "random" and isinstance(num_samples, int) and num_samples <= 0) else method
    print("="*80)
    print("🚀 Model A微调")
    print("="*80)
    print(f"方法: {display_method}")
    print(f"目标天数: {target_day}")
    print(f"样本数量: {num_samples}")
    print("="*80)
    print()
    
    extra_cfg = {k: v for k, v in finetune_kwargs.items() if v is not None}
    output_tag = extra_cfg.pop("output_tag", None)

    # 获取微调配置
    config = get_finetune_config(
        day=target_day,
        num_samples=num_samples,
        model_name='conformer',
        pretrained_ndays=pretrained_ndays,
        base_dir=base_dir,
        method=method,
        seed=seed,
        selection_seed=selection_seed,
        selection_strategy=selection_strategy,
        output_tag=output_tag,
        **extra_cfg,
    )
    # 选句与微调初始化必须使用同一份 Model A，避免 seed 不一致（例如错误回落到 seed0）。
    if model_a_path:
        config["pretrainedModelPath"] = model_a_path
    
    selected_indices = None
    # 选择样本；当 num_samples <= 0 时，表示使用全部训练样本，不进行采样
    if isinstance(num_samples, int) and num_samples <= 0:
        print("📊 使用全部训练样本进行微调（不进行子集采样）")
        config["selected_indices"] = None
    else:
        print(f"📊 使用 {method} 方法选择 {num_samples} 个样本...")
        # ran_x_y_z 是随机分层采样，也复用 selection_seed
        use_selection_seed = (
            selection_seed is not None
            and (
                method == 'random'
                or (selection_strategy is not None and selection_strategy.startswith('ran_'))
            )
        )
        sample_seed = selection_seed if use_selection_seed else seed
        selected_indices = select_samples_for_finetune(
            method=method,
            dataset_path=config["finetuneDataPath"],
            num_samples=num_samples,
            model_a_path=model_a_path if method in ['real_cer', 'real_slpe', 'model_b', 'badge'] else None,  # model_b也需要model_a_path来计算重合率
            model_b_path=model_b_path if method == 'model_b' else None,
            day=target_day,
            pretrained_ndays=pretrained_ndays,
            seed=sample_seed,
            selection_strategy=selection_strategy,
            use_slpe_cache=use_slpe_cache,
            base_dir=base_dir
        )
        # 将选择的样本索引保存到config中
        config["selected_indices"] = selected_indices
        print(f"✅ 选择了 {len(selected_indices)} 个样本")
    
    # 微调模型
    print(f"\n🏋️  开始微调Model A...")
    best_cer, finetuned_cer = finetune_model(config)
    
    print(f"\n✅ 微调完成！")
    print(f"   - 训练最佳CER: {best_cer:.4f}")
    print(f"   - 最终微调CER: {finetuned_cer:.4f}")
    
    return {
        'finetuned_model_path': config["pretrainedModelOutputPath"],
        'selected_indices': selected_indices,
        'best_cer': best_cer,
        'finetuned_cer': finetuned_cer
    }


def select_samples_for_finetune(
    method: str,
    dataset_path: str,
    num_samples: int,
    model_a_path: Optional[str] = None,
    model_b_path: Optional[str] = None,
    day: Optional[int] = None,
    pretrained_ndays: int = 7,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    selection_strategy: str = 'hard',
    use_slpe_cache: bool = False,  # False=强制重算，True=允许读共享缓存
    base_dir: str = '/root/25S151115/project3'
) -> List[int]:
    """
    根据方法选择样本
    
    Returns:
        选中的样本索引列表
    """
    if method == 'random':
        selected_indices = random_sample_selection(
            dataset_path=dataset_path,
            num_samples=num_samples,
            seed=seed
        )
        return selected_indices

    elif method == 'length':
        selected_indices = length_sample_selection(
            dataset_path=dataset_path,
            num_samples=num_samples,
            selection_strategy=selection_strategy
        )
        return selected_indices

    elif method == 'model_b':
        if model_b_path is None:
            raise ValueError("model_b_path 必须提供（当method='model_b'时）")
        # 仅 ran_x_y_z 需要随机种子；hard/easy/diverse 对 Model B 选样是确定性的
        strategy_seed = seed if (selection_strategy is not None and selection_strategy.startswith('ran_')) else 0
        selected_indices = model_b_sample_selection(
            dataset_path=dataset_path,
            model_b_path=model_b_path,
            num_samples=num_samples,
            selection_strategy=selection_strategy,
            seed=strategy_seed,
            model_a_path=model_a_path  # 传递model_a_path以计算重合率
        )
        return selected_indices
    
    elif method == 'real_cer':
        if model_a_path is None:
            raise ValueError("model_a_path 必须提供（当method='real_cer'时）")
        
        selected_indices = real_cer_sample_selection(
            dataset_path=dataset_path,
            model_a_path=model_a_path,
            num_samples=num_samples,
            selection_strategy=selection_strategy,
            batch_size=32,
            device='cuda',
            cache_dir=cache_dir
        )
        return selected_indices
    
    elif method == 'real_slpe':
        if model_a_path is None:
            raise ValueError("model_a_path 必须提供（当method='real_slpe'时）")

        # use_slpe_cache=False（默认）时强制重算，确保每个 seed 用自己的 model_a 打分
        # use_slpe_cache=True 时才允许读共享缓存（缓存中 model_a 路径无 seed 后缀，会导致所有 seed 共享同一组样本）
        effective_cache_dir = None
        if use_slpe_cache:
            effective_cache_dir = cache_dir or os.path.join(base_dir, 'outputs', 'slpe_scores', f'{pretrained_ndays}days')

        selected_indices = real_slpe_sample_selection(
            dataset_path=dataset_path,
            model_a_path=model_a_path,
            num_samples=num_samples,
            selection_strategy=selection_strategy,
            seed=seed,
            batch_size=32,
            device='cuda',
            cache_dir=effective_cache_dir,
            day=day,
            pretrained_ndays=pretrained_ndays
        )
        return selected_indices
    
    elif method == 'badge':
        if model_a_path is None:
            raise ValueError("model_a_path 必须提供（当method='badge'时）")
        selected_indices = badge_sample_selection(
            dataset_path=dataset_path,
            num_samples=num_samples,
            model_a_path=model_a_path,
            batch_size=32,
            device='cuda'
        )
        return selected_indices

    else:
        raise ValueError(
            f"不支持的方法: {method}，可选: random / length / model_b / real_cer / real_slpe / badge"
        )


# 便捷函数（保持向后兼容）
def random_sample_selection(
    dataset_path: str,
    num_samples: int,
    seed: int = 42
) -> List[int]:
    """
    随机选择样本
    """
    from src.utils.sample_selection import random_sample_selection as _random_sample_selection
    indices, _ = _random_sample_selection(dataset_path, num_samples, seed)
    return indices


def length_sample_selection(
    dataset_path: str,
    num_samples: int,
    selection_strategy: str = 'hard'
) -> List[int]:
    """
    按音素数量选择样本（hard=最多，easy=最少）
    """
    from src.utils.sample_selection import length_sample_selection as _length_sample_selection
    indices, _ = _length_sample_selection(
        dataset_path=dataset_path,
        num_samples=num_samples,
        selection_strategy=selection_strategy
    )
    return indices


def model_b_sample_selection(
    dataset_path: str,
    model_b_path: str,
    num_samples: int,
    selection_strategy: str = 'hard',  # 'hard' or 'easy'
    seed: int = 0,
    model_a_path: Optional[str] = None  # 用于计算重合率
) -> List[int]:
    """
    使用Model B选择样本
    """
    from src.utils.sample_selection import model_b_sample_selection as _model_b_sample_selection
    indices, _ = _model_b_sample_selection(
        dataset_path=dataset_path,
        model_b_path=model_b_path,
        num_samples=num_samples,
        selection_strategy=selection_strategy,
        seed=seed,
        model_a_path=model_a_path  # 传递model_a_path以计算重合率
    )
    return indices


def real_cer_sample_selection(
    dataset_path: str,
    model_a_path: str,
    num_samples: int,
    selection_strategy: str = 'hard',
    batch_size: int = 32,
    device: str = 'cuda',
    cache_dir: Optional[str] = None
) -> List[int]:
    """
    使用真实CER选择样本
    """
    from src.utils.sample_selection import real_cer_sample_selection as _real_cer_sample_selection
    indices, _ = _real_cer_sample_selection(
        dataset_path=dataset_path,
        model_a_path=model_a_path,
        num_samples=num_samples,
        selection_strategy=selection_strategy,
        batch_size=batch_size,
        device=device,
        save_dir=cache_dir
    )
    return indices


def real_slpe_sample_selection(
    dataset_path: str,
    model_a_path: str,
    num_samples: int,
    selection_strategy: str = 'hard',
    seed: int = 0,
    batch_size: int = 32,
    device: str = 'cuda',
    cache_dir: Optional[str] = None,
    day: Optional[int] = None,
    pretrained_ndays: int = 7,
    base_dir: str = '/root/25S151115/project3'
) -> List[int]:
    """
    使用真实SLPE选择样本
    """
    from src.utils.sample_selection import real_slpe_sample_selection as _real_slpe_sample_selection
    indices, _ = _real_slpe_sample_selection(
        dataset_path=dataset_path,
        model_a_path=model_a_path,
        num_samples=num_samples,
        selection_strategy=selection_strategy,
        seed=seed,
        batch_size=batch_size,
        device=device,
        save_dir=None,  # 不保存选择结果
        slpe_cache_dir=cache_dir,  # 使用slpe_cache_dir参数名
        day=day,
        pretrained_ndays=pretrained_ndays,
        base_dir=base_dir
    )
    return indices


def badge_sample_selection(
    dataset_path: str,
    model_a_path: str,
    num_samples: int,
    batch_size: int = 32,
    device: str = 'cuda'
) -> List[int]:
    """
    使用 BADGE 选择样本（CTC 梯度嵌入 + k-means++）
    """
    from src.utils.sample_selection import badge_sample_selection as _badge_sample_selection
    indices, _ = _badge_sample_selection(
        dataset_path=dataset_path,
        num_samples=num_samples,
        model_a_path=model_a_path,
        batch_size=batch_size,
        device=device,
        save_dir=None
    )
    return indices
