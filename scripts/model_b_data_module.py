#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model B数据获取模块
功能：计算SLPE或CER分数，使用前N天的Model A计算
"""
import os
import pickle
import numpy as np
import torch
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.utils.dataset import SpeechDataset
from src.utils.slpe import compute_slpe_batch
from src.model_a.get_model import get_model
from src.model_a.config import SESSION_NAMES_CHRONOLOGICAL


def _resolve_data_file_by_day(day: int) -> str:
    """将实验 day 编号映射到 dataMMDD 文件名（与微调配置保持一致）"""
    sessions = list(SESSION_NAMES_CHRONOLOGICAL)
    sessions.sort()
    if day < 1 or day > len(sessions):
        raise ValueError(f"day 超出范围: {day}, 有效范围 1..{len(sessions)}")
    parts = sessions[day - 1].split('.')
    return f"data{parts[-2]}{parts[-1]}"


def compute_scores_for_model_b(
    model_a_path: str,
    dataset_path: str,
    metric: str = 'slpe',  # 'slpe' or 'cer'
    batch_size: int = 32,
    device: str = 'cuda',
    cache_dir: Optional[str] = None,
    save_cache: bool = True,
    days: Optional[List[int]] = None,
    data_type: str = 'train'  # 'train' or 'test'
) -> Tuple[np.ndarray, List, np.ndarray]:
    """
    计算用于训练Model B的分数（SLPE或CER）
    
    Returns:
        scores: 分数数组（SLPE或CER）
        phoneme_seqs: 音素序列列表
        day_indices: 天数索引数组
    """
    print(f"📊 计算{data_type}数据的{metric.upper()}分数...")
    print(f"   Model A路径: {model_a_path}")
    print(f"   数据集路径: {dataset_path}")
    
    # 尝试从缓存加载
    if cache_dir and days:
        cache_file = _get_cache_file_path(cache_dir, days, metric, data_type)
        if os.path.exists(cache_file):
            print(f"📦 从缓存加载: {cache_file}")
            return _load_from_cache(cache_file)
    
    # 加载Model A
    model = _load_model_a(model_a_path, device)
    
    # 加载数据集
    dataset = _load_dataset(dataset_path, data_type)
    
    # 计算分数
    if metric == 'slpe':
        scores, phoneme_seqs, day_indices = _compute_slpe_scores(
            model, dataset, batch_size, device
        )
    elif metric == 'cer':
        scores, phoneme_seqs, day_indices = _compute_cer_scores(
            model, dataset, batch_size, device
        )
    else:
        raise ValueError(f"不支持的指标类型: {metric}")
    
    # 保存缓存
    if save_cache and cache_dir and days:
        _save_to_cache(cache_file, scores, phoneme_seqs, day_indices, 
                      metric, data_type, days)
    
    return scores, phoneme_seqs, day_indices


def compute_train_scores(
    model_a_path: str,
    data_path: str,
    train_days: List[int],
    metric: str = 'slpe',
    pretrained_ndays: int = 5,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    device: str = 'cuda'
) -> Tuple[np.ndarray, List, np.ndarray]:
    """
    计算训练数据的分数（day 6-7的train数据）
    """
    if cache_dir is None:
        base_dir = os.path.dirname(os.path.dirname(data_path))
        cache_dir = os.path.join(base_dir, 'outputs', 'slpe_scores', f'{pretrained_ndays}days')
    
    return compute_scores_for_model_b(
        model_a_path=model_a_path,
        dataset_path=data_path,
        metric=metric,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
        days=train_days,
        data_type='train'
    )


def compute_val_scores(
    model_a_path: str,
    data_path: str,
    train_days: List[int],
    metric: str = 'slpe',
    pretrained_ndays: int = 5,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    device: str = 'cuda'
) -> Tuple[np.ndarray, List, np.ndarray]:
    """
    计算验证数据的分数（day 6-7的test数据）
    """
    if cache_dir is None:
        base_dir = os.path.dirname(os.path.dirname(data_path))
        cache_dir = os.path.join(base_dir, 'outputs', 'slpe_scores', f'{pretrained_ndays}days')
    
    return compute_scores_for_model_b(
        model_a_path=model_a_path,
        dataset_path=data_path,
        metric=metric,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
        days=train_days,
        data_type='test'
    )


def compute_final_test_scores(
    model_a_path: str,
    val_days: List[int],
    metric: str = 'slpe',
    pretrained_ndays: int = 7,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    device: str = 'cuda',
    data_type: str = 'train',  # 最终测试用train数据
    base_dir: str = '/root/25S151115/project3'
) -> Dict[int, Tuple[np.ndarray, List, np.ndarray]]:
    """
    计算最终测试数据的分数（day 8-12的train数据）
    
    Returns:
        Dict[day, (scores, phoneme_seqs, day_indices)]
    """
    if cache_dir is None:
        cache_dir = os.path.join(base_dir, 'outputs', 'slpe_scores', f'{pretrained_ndays}days')
    
    results = {}
    for day in val_days:
        try:
            data_file = _resolve_data_file_by_day(day)
        except Exception as e:
            print(f"⚠️  无法解析 day {day} 对应数据文件: {e}")
            continue
        dataset_path = os.path.join(base_dir, 'data', data_file)
        
        if not os.path.exists(dataset_path):
            print(f"⚠️  数据路径不存在: {dataset_path}，跳过day {day}")
            continue
        
        print(f"\n📅 计算day {day}的{metric.upper()}分数...")
        scores, phoneme_seqs, day_indices = compute_scores_for_model_b(
            model_a_path=model_a_path,
            dataset_path=dataset_path,
            metric=metric,
            batch_size=batch_size,
            device=device,
            cache_dir=cache_dir,
            days=[day],
            data_type=data_type
        )
        results[day] = (scores, phoneme_seqs, day_indices)
    
    return results


# 辅助函数
def _get_cache_file_path(cache_dir: str, days: List[int], 
                         metric: str, data_type: str) -> str:
    """获取缓存文件路径"""
    os.makedirs(cache_dir, exist_ok=True)
    if len(days) == 1:
        filename = f'day_{days[0]}_{data_type}_{metric}_scores.pkl'
    else:
        days_str = '_'.join(map(str, sorted(days)))
        filename = f'days_{days_str}_{data_type}_{metric}_scores.pkl'
    return os.path.join(cache_dir, filename)


def _load_from_cache(cache_file: str) -> Tuple[np.ndarray, List, np.ndarray]:
    """从缓存加载"""
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)
    return (
        np.array(cached_data['scores']),
        cached_data['phoneme_seqs'],
        np.array(cached_data['day_indices'])
    )


def _save_to_cache(cache_file: str, scores: np.ndarray, 
                   phoneme_seqs: List, day_indices: np.ndarray,
                   metric: str, data_type: str, days: List[int]):
    """保存到缓存"""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    cache_data = {
        'scores': scores,
        'phoneme_seqs': phoneme_seqs,
        'day_indices': day_indices,
        'metric': metric,
        'data_type': data_type,
        'days': days
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"💾 已保存到缓存: {cache_file}")


def _load_model_a(model_a_path: str, device: str):
    """加载Model A"""
    config_path = os.path.join(model_a_path, "config.pkl")
    weights_path = os.path.join(model_a_path, "modelWeights.pth")
    
    if not os.path.exists(config_path):
        config_path = os.path.join(model_a_path, "args")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(model_a_path, "modelWeights")
    
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    
    model = get_model(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model.eval()
    return model


def _load_dataset(dataset_path: str, data_type: str) -> SpeechDataset:
    """加载数据集"""
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    if data_type == 'train':
        return SpeechDataset(data['train'])
    elif data_type == 'test':
        return SpeechDataset(data['test'])
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")


def _compute_slpe_scores(model, dataset, batch_size, device) -> Tuple[np.ndarray, List, np.ndarray]:
    """计算SLPE分数"""
    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)
        return (X_padded, y_padded, torch.stack(X_lens), 
                torch.stack(y_lens), torch.stack(days))
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=0, collate_fn=_padding)
    
    # 提取音素序列和天数
    phoneme_seqs = []
    day_indices = []
    with torch.no_grad():
        for X, y, X_len, y_len, days in loader:
            for i in range(len(y)):
                true_seq = y[i, :y_len[i]].cpu().numpy()
                phoneme_seqs.append(true_seq)
                day_indices.append(days[i].item())
    
    # 计算SLPE
    slpe_scores = compute_slpe_batch(model, loader, device=device, blank=0)
    
    return slpe_scores, phoneme_seqs, np.array(day_indices)


def _compute_cer_scores(model, dataset, batch_size, device) -> Tuple[np.ndarray, List, np.ndarray]:
    """计算CER分数"""
    # TODO: 实现CER计算逻辑
    raise NotImplementedError("CER计算功能待实现")
