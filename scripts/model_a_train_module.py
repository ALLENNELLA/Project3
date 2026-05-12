#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model A训练模块
功能：训练Model A（脑电解码模型）
"""
import os
from typing import Optional, Dict
from pathlib import Path

from src.model_a.config import get_train_config
from src.model_a.trainer import train_model


def train_model_a(
    n_days: int = 7,
    model_name: str = 'conformer',
    base_dir: str = '/root/25S151115/project3',
    output_dir: Optional[str] = None,
    **train_kwargs
) -> Dict:
    """
    训练Model A模型
    
    Args:
        n_days: 训练使用的天数
        model_name: 模型名称（'gru', 'moganet', 'conformer', 'conformer1'）
        base_dir: 基础目录
        output_dir: 输出目录（如果为None，自动生成）
        **train_kwargs: 其他训练参数（会传递给config）
    
    Returns:
        Dict包含：
            - model_path: 训练好的模型路径
            - config: 训练配置
    """
    print("="*80)
    print("🚀 Model A训练")
    print("="*80)
    print(f"模型名称: {model_name}")
    print(f"训练天数: {n_days}")
    print("="*80)
    print()
    
    # 获取seed（如果提供）
    seed = train_kwargs.pop('seed', None)
    
    # 获取训练配置（传递seed以生成正确的输出路径）
    config = get_train_config(
        nDays=n_days,
        model_name=model_name,
        base_dir=base_dir,
        seed=seed,
        **train_kwargs
    )
    
    # 更新配置（如果提供了额外参数）
    for key, value in train_kwargs.items():
        if key in config:
            config[key] = value
    
    # 如果指定了输出目录，更新配置
    if output_dir is not None:
        config['outputDir'] = str(output_dir)
    
    # 确保seed已设置到config中（get_train_config应该已经设置了）
    if seed is not None and 'seed' not in config:
        config['seed'] = seed
    
    print(f"📁 输出目录: {config['outputDir']}")
    print(f"📁 数据路径: {config['datasetPath']}")
    print()
    
    # 训练模型
    print("🏋️  开始训练Model A...")
    model = train_model(config)
    
    print("\n✅ Model A训练完成！")
    print(f"   - 模型保存路径: {config['outputDir']}")
    
    return {
        'model_path': config['outputDir'],
        'config': config,
        'model': model
    }


def get_model_a_path(
    n_days: int,
    model_name: str = 'conformer',
    base_dir: str = '/root/25S151115/project3',
    seed: int = 0,
) -> str:
    """
    获取Model A的路径（根据训练参数）
    
    Args:
        n_days: 训练使用的天数
        model_name: 模型名称
        base_dir: 基础目录（项目根，含 outputs/model_train/）
        seed: 与训练时一致，默认 0，对应目录 .../conformer-{n}days-seed{seed}/
    
    Returns:
        Model A路径
    """
    model_name_map = {
        'gru': 'gru',
        'moganet': 'moganet',
        'conformer': 'conformer',
        'conformer1': 'conformer1'
    }
    model_name_str = model_name_map.get(model_name, model_name)
    root = os.path.abspath(os.path.expanduser(base_dir))
    return os.path.join(
        root, 'outputs', 'model_train', f'{model_name_str}-{n_days}days-seed{seed}'
    )
