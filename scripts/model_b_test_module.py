#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model B测试模块
功能：计算Model A前7天模型预测8-12天的CER或SLPE的top50、100的重合度
"""
import os
import pickle
import numpy as np
import torch
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from transformers import AutoTokenizer

from model_b_data_module import compute_final_test_scores
from model_b_train_module import load_trained_model_b
from model_b_utils import build_prompt, extract_transcriptions, normalize_scores, filter_nan_samples
from src.utils.dataset import SpeechDataset
from src.model_a.config import SESSION_NAMES_CHRONOLOGICAL


def _load_tokenizer_with_fallback(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception as e_local:
        print(f"⚠️ 本地缓存加载tokenizer失败，尝试在线加载: {e_local}")
        return AutoTokenizer.from_pretrained(model_name)


def _resolve_data_file_by_day(day: int) -> str:
    """将实验 day 编号映射到 dataMMDD 文件名（与微调配置保持一致）"""
    sessions = list(SESSION_NAMES_CHRONOLOGICAL)
    sessions.sort()
    if day < 1 or day > len(sessions):
        raise ValueError(f"day 超出范围: {day}, 有效范围 1..{len(sessions)}")
    parts = sessions[day - 1].split('.')
    return f"data{parts[-2]}{parts[-1]}"


def compute_overlap_analysis(
    model_a_path: str,
    model_b_path: str,
    val_days: List[int],  # [8, 9, 10, 11, 12]
    metric: str = 'slpe',  # 'slpe' or 'cer'
    pretrained_ndays: int = 7,
    top_k_list: List[int] = [50, 100],
    batch_size: int = 32,
    device: str = 'cuda',
    cache_dir: Optional[str] = None,
    base_dir: str = '/root/25S151115/project3'
) -> Dict:
    """
    计算重合度分析
    
    Returns:
        Dict包含：
            - overlap_results: Dict[day, Dict[top_k, overlap_rate]]
            - model_b_predictions: Dict[day, np.ndarray]  # Model B预测的分数
            - real_scores: Dict[day, np.ndarray]  # 真实分数（SLPE或CER）
            - model_b_top_indices: Dict[day, Dict[top_k, List[int]]]
            - real_top_indices: Dict[day, Dict[top_k, List[int]]]
    """
    print("="*80)
    print("🔬 Model B重合度分析")
    print("="*80)
    print(f"Model A路径: {model_a_path}")
    print(f"Model B路径: {model_b_path}")
    print(f"验证天数: {val_days}")
    print(f"指标: {metric.upper()}")
    print("="*80)
    print()
    
    # 计算真实分数
    print("📊 [1/2] 计算真实分数...")
    real_scores_dict = compute_real_scores(
        model_a_path=model_a_path,
        val_days=val_days,
        metric=metric,
        pretrained_ndays=pretrained_ndays,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
        base_dir=base_dir
    )
    
    # 计算Model B预测
    print("\n📊 [2/2] 计算Model B预测...")
    model_b_predictions_dict = compute_model_b_predictions(
        model_b_path=model_b_path,
        val_days=val_days,
        batch_size=batch_size,
        device=device,
        base_dir=base_dir
    )
    
    # 计算重合度
    print("\n📊 [3/3] 计算重合度...")
    overlap_results = {}
    model_b_top_indices_dict = {}
    real_top_indices_dict = {}
    
    for day in val_days:
        if day not in real_scores_dict or day not in model_b_predictions_dict:
            print(f"⚠️  Day {day} 数据不完整，跳过")
            continue
        
        real_scores = real_scores_dict[day]
        model_b_scores = model_b_predictions_dict[day]
        
        # 确保长度一致
        min_len = min(len(real_scores), len(model_b_scores))
        real_scores = real_scores[:min_len]
        model_b_scores = model_b_scores[:min_len]
        
        # 计算Top-K重合度
        day_overlap = {}
        day_model_b_top = {}
        day_real_top = {}
        
        for top_k in top_k_list:
            if top_k > min_len:
                continue
            
            # 获取Top-K索引（困难样本，分数最大）
            model_b_top = _get_top_k_indices(model_b_scores, top_k, strategy='hard')
            real_top = _get_top_k_indices(real_scores, top_k, strategy='hard')
            
            overlap_rate = calculate_overlap(model_b_top, real_top)
            day_overlap[top_k] = overlap_rate
            day_model_b_top[top_k] = model_b_top
            day_real_top[top_k] = real_top
            
            print(f"   Day {day} Top-{top_k}: {overlap_rate:.2%} ({len(set(model_b_top) & set(real_top))}/{top_k})")
        
        overlap_results[day] = day_overlap
        model_b_top_indices_dict[day] = day_model_b_top
        real_top_indices_dict[day] = day_real_top
    
    return {
        'overlap_results': overlap_results,
        'model_b_predictions': model_b_predictions_dict,
        'real_scores': real_scores_dict,
        'model_b_top_indices': model_b_top_indices_dict,
        'real_top_indices': real_top_indices_dict
    }


def compute_model_b_predictions(
    model_b_path: str,
    val_days: List[int],
    batch_size: int = 32,
    device: str = 'cuda',
    base_dir: str = '/root/25S151115/project3'
) -> Dict[int, np.ndarray]:
    """
    使用Model B预测8-12天的分数
    """
    # 加载Model B
    print("   加载Model B...")
    # 尝试从results.pkl中读取模型配置
    model_dir = Path(model_b_path).parent
    results_path = model_dir / 'results.pkl'
    
    model_name = 'roberta-base'
    model_type = 'roberta'
    prompt_format = None
    
    if results_path.exists():
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
            # 从顶层或args层尝试获取prompt_format
            prompt_format = results.get('prompt_format')
            if 'args' in results:
                args = results['args']
                model_name = args.get('model_name', 'roberta-base')
                model_type = args.get('model_type', 'roberta')
                if prompt_format is None:
                    prompt_format = args.get('prompt_format')
    
    model = load_trained_model_b(model_b_path, model_name, model_type, device)
    tokenizer = _load_tokenizer_with_fallback(model_name)
    
    # 预测每个天的分数
    predictions_dict = {}
    for day in val_days:
        try:
            data_file = _resolve_data_file_by_day(day)
        except Exception as e:
            print(f"   ⚠️  无法解析 day {day} 对应数据文件: {e}")
            continue
        dataset_path = os.path.join(base_dir, 'data', data_file)
        
        if not os.path.exists(dataset_path):
            print(f"   ⚠️  Day {day} 数据路径不存在: {dataset_path}")
            continue
        
        print(f"   预测Day {day}...")
        
        # 加载数据
        with open(dataset_path, 'rb') as f:
            day_data = pickle.load(f)
        
        if 'train' not in day_data or len(day_data['train']) == 0:
            print(f"   ⚠️  Day {day} 没有train数据")
            continue
        
        dataset = SpeechDataset(day_data['train'])
        
        # 提取音素序列和transcriptions
        phoneme_seqs = []
        transcriptions = []
        for i in range(len(dataset)):
            _, y, _, y_len, _ = dataset[i]
            phoneme_seq = y[:y_len].cpu().numpy()
            phoneme_seqs.append(phoneme_seq)
            
            # 获取transcription
            if hasattr(dataset, 'get_transcription'):
                trans = dataset.get_transcription(i)
            else:
                trans = None
            transcriptions.append(trans)
        
        # 构建prompt
        prompts = []
        for i, seq in enumerate(phoneme_seqs):
            trans = transcriptions[i] if i < len(transcriptions) else None
            # 使用训练时保存的格式；若老模型未保存则回退到系统默认格式
            fmt = prompt_format
            prompt = build_prompt(seq, trans, prompt_format=fmt)
            prompts.append(prompt)
        
        # 批量预测
        model.eval()
        predictions = []
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i+batch_size]
                batch_scores = model.predict(batch_prompts, batch_size=len(batch_prompts), device=device)
                predictions.extend(batch_scores)
        
        predictions_dict[day] = np.array(predictions)
        print(f"   ✅ Day {day}: {len(predictions)} 个预测")
    
    return predictions_dict


def compute_real_scores(
    model_a_path: str,
    val_days: List[int],
    metric: str = 'slpe',
    pretrained_ndays: int = 7,
    batch_size: int = 32,
    device: str = 'cuda',
    cache_dir: Optional[str] = None,
    data_type: str = 'train',
    base_dir: str = '/root/25S151115/project3'
) -> Dict[int, np.ndarray]:
    """
    使用Model A计算8-12天的真实分数
    """
    if cache_dir is None:
        cache_dir = os.path.join(base_dir, 'outputs', 'slpe_scores', f'{pretrained_ndays}days')
    
    results = compute_final_test_scores(
        model_a_path=model_a_path,
        val_days=val_days,
        metric=metric,
        pretrained_ndays=pretrained_ndays,
        cache_dir=cache_dir,
        batch_size=batch_size,
        device=device,
        data_type=data_type,
        base_dir=base_dir
    )
    
    # 提取分数
    scores_dict = {}
    for day, (scores, _, _) in results.items():
        scores_dict[day] = scores
    
    return scores_dict


def calculate_overlap(
    indices1: List[int],
    indices2: List[int]
) -> float:
    """
    计算两个索引列表的重合度
    """
    set1 = set(indices1)
    set2 = set(indices2)
    overlap = len(set1 & set2)
    total = len(set1)
    return overlap / total if total > 0 else 0.0


# 辅助函数
def _get_top_k_indices(scores: np.ndarray, k: int, strategy: str = 'hard') -> List[int]:
    """
    获取Top-K索引
    
    Args:
        scores: 分数数组
        k: Top-K值
        strategy: 'hard' (困难样本，分数最大) 或 'easy' (简单样本，分数最小)
    """
    if strategy == 'hard':
        # 困难样本：分数最大
        return np.argsort(scores)[-k:].tolist()
    else:
        # 简单样本：分数最小
        return np.argsort(scores)[:k].tolist()
