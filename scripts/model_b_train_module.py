#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model B训练模块
功能：训练Model B模型
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr, kendalltau
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType

from src.model_b.models.phoneme_cer_predictor import PhonemeCERPredictor
from src.model_b.utils.phoneme_dataset import PhonemeTextDataset
from src.model_b.utils.ranking_loss import MarginRankingLoss
from model_b_utils import build_prompt, extract_transcriptions, normalize_scores as normalize_scores_func, filter_nan_samples


def _load_tokenizer_with_fallback(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception as e_local:
        print(f"⚠️ 本地缓存加载tokenizer失败，尝试在线加载: {e_local}")
        return AutoTokenizer.from_pretrained(model_name)


def train_model_b(
    train_scores: np.ndarray,
    train_phoneme_seqs: List,
    val_scores: np.ndarray,
    val_phoneme_seqs: List,
    train_day_indices: np.ndarray = None,
    val_day_indices: np.ndarray = None,
    train_transcriptions: Optional[List] = None,
    val_transcriptions: Optional[List] = None,
    model_name: str = 'roberta-base',
    model_type: str = 'roberta',
    output_dir: str = 'outputs/model_b',
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    normalize_scores: str = 'rank',
    use_ranking_loss: bool = True,
    use_mse_as_auxiliary: bool = False,
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    head_learning_rate: float = 1e-3,
    device: str = 'cuda',
    seed: int = 42,
    prompt_format: Optional[str] = None
) -> Dict:
    """
    训练Model B模型
    
    Args:
        train_scores: 训练集分数（SLPE或CER）
        train_phoneme_seqs: 训练集音素序列
        val_scores: 验证集分数
        val_phoneme_seqs: 验证集音素序列
        train_transcriptions: 训练集转录文本（用于prompt）
        val_transcriptions: 验证集转录文本（用于prompt）
        ... (其他训练参数)
    
    Returns:
        Dict包含：
            - best_model_path: 最佳模型路径
            - train_history: 训练历史
            - val_history: 验证历史
            - best_epoch: 最佳epoch
            - best_spearman: 最佳Spearman相关系数
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() and device != 'cpu' else 'cpu')
    print(f"🔧 使用设备: {device}\n")
    
    # 过滤NaN样本
    train_scores, train_phoneme_seqs, _, train_transcriptions = filter_nan_samples(
        train_scores, train_phoneme_seqs, np.zeros(len(train_scores)), train_transcriptions
    )
    val_scores, val_phoneme_seqs, _, val_transcriptions = filter_nan_samples(
        val_scores, val_phoneme_seqs, np.zeros(len(val_scores)), val_transcriptions
    )
    
    # 归一化分数
    if normalize_scores != 'none':
        # 需要day_indices，如果没传，回退到全部是0
        if train_day_indices is None:
            train_day_indices = np.zeros(len(train_scores))
        if val_day_indices is None:
            val_day_indices = np.zeros(len(val_scores))
        
        train_scores = normalize_scores_func(train_scores, train_day_indices, method=normalize_scores)
        val_scores = normalize_scores_func(val_scores, val_day_indices, method=normalize_scores)
    
    # 转换音素序列为文本并构建prompt
    print(f"🔄 将音素序列转换为文本并构建prompt (format={prompt_format or 'default'})...")
    train_phoneme_texts = []
    for i, seq in enumerate(tqdm(train_phoneme_seqs, desc='转换训练集音素')):
        trans = train_transcriptions[i] if train_transcriptions and i < len(train_transcriptions) else None
        text = build_prompt(seq, trans, prompt_format=prompt_format)
        train_phoneme_texts.append(text)
    
    val_phoneme_texts = []
    for i, seq in enumerate(tqdm(val_phoneme_seqs, desc='转换验证集音素')):
        trans = val_transcriptions[i] if val_transcriptions and i < len(val_transcriptions) else None
        text = build_prompt(seq, trans, prompt_format=prompt_format)
        val_phoneme_texts.append(text)
    
    # 创建数据集
    print("📦 创建数据集...")
    tokenizer = _load_tokenizer_with_fallback(model_name)
    
    train_dataset = PhonemeTextDataset(
        train_phoneme_texts, train_scores, tokenizer, max_length=512,
        text_contents=train_transcriptions
    )
    val_dataset = PhonemeTextDataset(
        val_phoneme_texts, val_scores, tokenizer, max_length=512,
        text_contents=val_transcriptions
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 创建模型
    print("🏗️  创建模型...")
    model = PhonemeCERPredictor(
        model_name=model_name,
        model_type=model_type,
        dropout=0.1,
        hidden_dim=256
    )
    
    # 应用LoRA
    if use_lora:
        print(f"   应用LoRA: r={lora_r}, alpha={lora_alpha}")
        target_modules = ["query", "value", "key"]
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )
        model.bert = get_peft_model(model.bert, lora_config)
        model.bert.print_trainable_parameters()
    
    model.to(device)
    
    # 设置优化器
    print("⚙️  设置优化器和调度器...")
    if use_lora:
        lora_params = [p for n, p in model.named_parameters() if 'lora' in n.lower() and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if 'regressor' in n.lower()]
    else:
        lora_params = []
        head_params = list(model.regressor.parameters())
    
    bert_params = [p for n, p in model.bert.named_parameters() if p.requires_grad]
    
    optimizer_params = [
        {'params': bert_params, 'lr': learning_rate},
        {'params': head_params, 'lr': head_learning_rate}
    ]
    
    optimizer = AdamW(optimizer_params, weight_decay=0.01)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,
        num_training_steps=total_steps
    )
    
    # 训练循环
    print("="*80)
    print("🏋️  开始训练Model B...")
    print("="*80)
    print(f"📊 训练轮数: {num_epochs}")
    print(f"📦 Batch大小: {batch_size}")
    print("="*80)
    print()
    
    best_spearman = -1.0
    best_epoch = 0
    train_history = []
    val_history = []
    
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*80}")
        
        # 训练
        train_metrics = _train_epoch(
            model, train_loader, optimizer, scheduler, device,
            use_ranking_loss=use_ranking_loss,
            use_mse_as_auxiliary=use_mse_as_auxiliary
        )
        train_history.append(train_metrics)
        
        print(f"训练损失: {train_metrics['total_loss']:.4f}")
        if use_ranking_loss:
            print(f"  排序损失: {train_metrics['ranking_loss']:.4f}")
        if use_mse_as_auxiliary:
            print(f"  MSE辅助损失: {train_metrics['mse_loss']:.4f}")
        
        # 验证
        val_metrics, val_predictions, val_labels = _evaluate(model, val_loader, device)
        val_history.append(val_metrics)
        
        print(f"\n验证集指标:")
        print(f"  Spearman: {val_metrics['spearman']:.4f}")
        print(f"  Kendall:  {val_metrics['kendall']:.4f}")
        print(f"  MSE:      {val_metrics['mse']:.4f}")
        print(f"  MAE:      {val_metrics['mae']:.4f}")
        
        # 保存最佳模型 / LoRA 适配器
        if val_metrics['spearman'] > best_spearman:
            best_spearman = val_metrics['spearman']
            best_epoch = epoch + 1

            if use_lora:
                # LoRA 模型：保存 adapter + head
                adapter_dir = output_path  # 直接使用输出目录存放adapter
                print(f"  ✅ 保存最佳LoRA模型 (Spearman={best_spearman:.4f})")
                # 保存LoRA适配器（PEFT格式）
                try:
                    try:
                        model.bert.save_pretrained(adapter_dir, save_embedding_layers=False)
                    except TypeError:
                        model.bert.save_pretrained(adapter_dir)
                except Exception as e:
                    print(f"  ⚠️ 保存LoRA适配器失败: {e}")
                # 保存回归头
                torch.save(model.regressor.state_dict(), output_path / 'head.pt')
            else:
                # 非LoRA：保存完整模型权重
                torch.save(model.state_dict(), output_path / 'best_model.pt')
                print(f"  ✅ 保存最佳模型 (Spearman={best_spearman:.4f})")
    
    print(f"\n{'='*80}")
    print(f"✅ 训练完成！")
    print(f"   - 最佳模型: Epoch {best_epoch}/{num_epochs}")
    print(f"   - 最佳Spearman相关系数: {best_spearman:.4f}")
    print("="*80)
    
    # 保存训练历史
    results = {
        'best_epoch': best_epoch,
        'best_spearman': best_spearman,
        'train_history': train_history,
        'val_history': val_history,
        'use_lora': use_lora,
        'model_name': model_name,
        'model_type': model_type,
        'prompt_format': prompt_format,  # 保存超参数
        'args': {
            'model_name': model_name,
            'model_type': model_type,
            'prompt_format': prompt_format
        }
    }
    
    with open(output_path / 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # 返回的 best_model_path：
    # - LoRA: 返回目录路径，load 时按adapter+head方式加载
    # - 非LoRA: 返回best_model.pt路径，load 时按完整模型加载
    if use_lora:
        best_model_path = str(output_path)
    else:
        best_model_path = str(output_path / 'best_model.pt')
    
    return {
        'best_model_path': best_model_path,
        'train_history': train_history,
        'val_history': val_history,
        'best_epoch': best_epoch,
        'best_spearman': best_spearman
    }


def load_trained_model_b(
    model_path: str,
    model_name: str = 'roberta-base',
    model_type: str = 'roberta',
    device: str = 'cuda'
) -> torch.nn.Module:
    """
    加载训练好的Model B
    """
    """
    加载训练好的Model B
    
    支持两种保存格式：
    1）非LoRA：best_model.pt（完整PhonemeCERPredictor的state_dict）
    2）LoRA：目录下包含 adapter_config.json + head.pt
    """
    model_dir = Path(model_path)

    # 如果传入的是文件路径，则取其父目录作为checkpoint目录
    if model_dir.is_file():
        checkpoint_dir = model_dir.parent
        weight_path = model_dir
    else:
        checkpoint_dir = model_dir
        weight_path = checkpoint_dir / 'best_model.pt'

    # 构建基础模型
    model = PhonemeCERPredictor(
        model_name=model_name,
        model_type=model_type,
        dropout=0.1,
        hidden_dim=256
    )
    
    # LoRA 模型加载：目录中存在 adapter_config.json & head.pt
    adapter_config = checkpoint_dir / 'adapter_config.json'
    head_path = checkpoint_dir / 'head.pt'

    if adapter_config.exists() and head_path.exists():
        from peft import PeftModel
        print(f"   检测到LoRA适配器，按LoRA方式加载: {checkpoint_dir}")
        # 将基础RoBERTa包装为PeftModel，并加载adapter权重
        model.bert = PeftModel.from_pretrained(model.bert, str(checkpoint_dir))
        # 加载回归头
        head_state = torch.load(head_path, map_location=device)
        model.regressor.load_state_dict(head_state)
    else:
        # 非LoRA：按完整state_dict加载
        print(f"   按完整模型权重方式加载: {weight_path}")
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    
    return model


# 辅助函数
def _train_epoch(
    model, train_loader, optimizer, scheduler, device,
    use_ranking_loss=True,
    use_mse_as_auxiliary=False,
    ranking_margin=1.0,
    mse_alpha=0.1
):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_ranking_loss = 0.0
    n_batches = 0
    
    if use_mse_as_auxiliary:
        mse_loss_fn = nn.MSELoss()
    if use_ranking_loss:
        ranking_loss_fn = MarginRankingLoss(margin=ranking_margin)
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc='训练')):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        outputs = model.bert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        predictions = model.regressor(cls_embedding).squeeze(-1)
        
        # 计算损失
        loss = torch.tensor(0.0, device=device)
        
        if use_ranking_loss:
            batch_size = predictions.size(0)
            if batch_size >= 2:
                # 构造样本对
                max_pairs = min(batch_size * (batch_size - 1) // 2, 100)
                pairs = []
                for i in range(batch_size):
                    for j in range(i+1, batch_size):
                        pairs.append((i, j))
                
                if len(pairs) > max_pairs:
                    import random
                    pairs = random.sample(pairs, max_pairs)
                
                pair_predictions_easy = []
                pair_predictions_hard = []
                pair_labels = []
                
                for i, j in pairs:
                    if labels[i] < labels[j]:
                        pair_predictions_easy.append(predictions[i])
                        pair_predictions_hard.append(predictions[j])
                        pair_labels.append(1.0)
                    elif labels[j] < labels[i]:
                        pair_predictions_easy.append(predictions[j])
                        pair_predictions_hard.append(predictions[i])
                        pair_labels.append(1.0)
                
                if len(pair_predictions_easy) > 0:
                    pair_predictions_easy = torch.stack(pair_predictions_easy)
                    pair_predictions_hard = torch.stack(pair_predictions_hard)
                    pair_labels = torch.tensor(pair_labels, device=device, dtype=torch.float32)
                    
                    ranking_loss = ranking_loss_fn(
                        pair_predictions_easy, 
                        pair_predictions_hard, 
                        pair_labels
                    )
                    
                    loss = loss + ranking_loss
                    total_ranking_loss += ranking_loss.item()
        
        if use_mse_as_auxiliary:
            mse_loss = mse_loss_fn(predictions, labels)
            loss = loss + mse_alpha * mse_loss
            total_mse_loss += mse_loss.item()
        
        if loss.item() == 0.0:
            continue
        
        # 反向传播
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        n_batches += 1
    
    if n_batches == 0:
        return {
            'total_loss': 0.0,
            'mse_loss': 0.0,
            'ranking_loss': 0.0
        }
    
    return {
        'total_loss': total_loss / n_batches,
        'mse_loss': total_mse_loss / n_batches if use_mse_as_auxiliary else 0.0,
        'ranking_loss': total_ranking_loss / n_batches if use_ranking_loss else 0.0
    }


def _evaluate(model, val_loader, device):
    """评估模型"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='评估'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            outputs = model.bert(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            predictions = model.regressor(cls_embedding).squeeze(-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # 计算评估指标
    spearman, _ = spearmanr(all_predictions, all_labels)
    kendall, _ = kendalltau(all_predictions, all_labels)
    
    mse = np.mean((all_predictions - all_labels) ** 2)
    mae = np.mean(np.abs(all_predictions - all_labels))
    
    metrics = {
        'spearman': spearman,
        'kendall': kendall,
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse)
    }
    
    return metrics, all_predictions, all_labels
