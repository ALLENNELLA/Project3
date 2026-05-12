#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model B辅助工具模块
提供各种辅助功能，包括prompt构建、音素序列转换、文本处理等
"""
import os
import numpy as np
from typing import List, Optional, Tuple, Dict
from scipy.stats import rankdata
from sklearn.preprocessing import QuantileTransformer

from src.model_b.utils.phoneme_converter import phoneme_seq_to_text as _phoneme_seq_to_text

# ==========================================
# 提示词格式默认值
# ==========================================
# 可选值: 'combined_zh', 'combined_en', 'instruction', 'phoneme_only', 'native_pair', 'feature_injection'
DEFAULT_PROMPT_FORMAT = 'combined_zh'
# ==========================================

def build_prompt(
    phoneme_seq: List[int] or np.ndarray,
    transcription: Optional[str] = None,
    prompt_format: Optional[str] = None  # 如果不传，则使用 DEFAULT_PROMPT_FORMAT
) -> str:
    """
    构建prompt，结合音素序列和自然语言转录
    
    Args:
        phoneme_seq: 音素ID序列
        transcription: 自然语言转录文本（可选）
        prompt_format: prompt格式，如果为None则使用DEFAULT_PROMPT_FORMAT
            - 'phoneme_only': 只使用音素序列
            - 'combined_en': 改进的英文版本（音素序列 + 转录文本）
            - 'combined_zh': 原始翻译的中文版本（音素序列 + 转录文本）
            - 'instruction': 带指令的格式
    
    Returns:
        prompt字符串
    """
    # 如果未指定格式，则使用总控设置（可由环境变量覆盖）
    if prompt_format is None:
        fmt = os.environ.get('MODEL_B_PROMPT_FORMAT', DEFAULT_PROMPT_FORMAT)
    else:
        fmt = prompt_format

    # 将音素序列转换为文本
    phoneme_text = _phoneme_seq_to_text(phoneme_seq, remove_padding=True)
    transcription_text = transcription.strip() if transcription is not None else ""
    word_count = len(transcription_text.split()) if transcription_text else 0
    phoneme_count = len([tok for tok in phoneme_text.split() if tok])
    
    if fmt == 'phoneme_only':
        return f"Phonemes: {phoneme_text}"
    elif fmt == 'combined_en':
        return build_combined_prompt_en(phoneme_text, transcription)
    elif fmt == 'combined_zh':
        return build_combined_prompt_zh(phoneme_text, transcription)
    elif fmt == 'instruction':
        return format_prompt_instruction(phoneme_text, transcription)
    elif fmt == 'native_pair':
        return format_prompt_native_pair(phoneme_text, transcription_text)
    elif fmt == 'feature_injection':
        return format_prompt_feature_injection(
            phoneme_text=phoneme_text,
            transcription_text=transcription_text,
            word_count=word_count,
            phoneme_count=phoneme_count
        )
    else:
        raise ValueError(f"不支持的prompt格式: {fmt}")


# phoneme_seq_to_text 函数已从 src.model_b.utils.phoneme_converter 导入


def extract_transcriptions(
    data_split: List[Dict],
    field_name: str = 'transcriptions'
) -> List[Optional[str]]:
    """
    从数据中提取转录文本
    
    Args:
        data_split: 数据分割（如data['train']），是一个列表，每个元素是一天的数据
        field_name: 转录字段名称
    
    Returns:
        transcriptions列表，顺序与SpeechDataset一致
    """
    transcriptions = []
    for day_data in data_split:
        if field_name in day_data:
            day_transcriptions = day_data[field_name]
            transcriptions.extend(day_transcriptions)
        else:
            # 如果没有transcriptions，使用None占位
            num_samples = len(day_data.get('sentenceDat', []))
            transcriptions.extend([None] * num_samples)
    return transcriptions


def normalize_scores(
    scores: np.ndarray,
    day_indices: np.ndarray,
    method: str = 'rank'  # 'rank', 'minmax', 'zscore', 'quantile', 'none'
) -> np.ndarray:
    """
    归一化分数（按天分别归一化或全局归一化）
    
    Args:
        scores: 分数数组
        day_indices: 天数索引数组
        method: 归一化方法
            - 'rank': 排序归一化（按天分别）
            - 'minmax': 最小最大归一化（按天分别）
            - 'zscore': Z-score归一化（按天分别）
            - 'quantile': 分位数归一化（全局）
            - 'none': 不归一化
    
    Returns:
        归一化后的分数数组
    """
    if method == 'none':
        return scores
    
    if method == 'rank':
        # 按天分别排序归一化
        normalized_scores = np.zeros_like(scores, dtype=np.float32)
        unique_days = np.unique(day_indices)
        
        for day in unique_days:
            day_mask = (day_indices == day)
            day_scores = scores[day_mask]
            
            if len(day_scores) > 1:
                ranked = rankdata(day_scores, method='average')
                normalized = (ranked - 1) / (len(ranked) - 1)
                normalized_scores[day_mask] = normalized.astype(np.float32)
            elif len(day_scores) == 1:
                normalized_scores[day_mask] = 0.5
            else:
                normalized_scores[day_mask] = 0.0
        
        return normalized_scores
    
    elif method == 'minmax':
        # 按天分别最小最大归一化
        normalized_scores = np.zeros_like(scores, dtype=np.float32)
        unique_days = np.unique(day_indices)
        
        for day in unique_days:
            day_mask = (day_indices == day)
            day_scores = scores[day_mask]
            
            if len(day_scores) > 1:
                min_score = day_scores.min()
                max_score = day_scores.max()
                if max_score > min_score:
                    normalized = (day_scores - min_score) / (max_score - min_score)
                else:
                    normalized = np.zeros_like(day_scores)
                normalized_scores[day_mask] = normalized.astype(np.float32)
            elif len(day_scores) == 1:
                normalized_scores[day_mask] = 0.5
            else:
                normalized_scores[day_mask] = 0.0
        
        return normalized_scores
    
    elif method == 'zscore':
        # 按天分别Z-score归一化
        normalized_scores = np.zeros_like(scores, dtype=np.float32)
        unique_days = np.unique(day_indices)
        
        for day in unique_days:
            day_mask = (day_indices == day)
            day_scores = scores[day_mask]
            
            if len(day_scores) > 1:
                mean_score = day_scores.mean()
                std_score = day_scores.std() + 1e-8
                normalized = (day_scores - mean_score) / std_score
                normalized_scores[day_mask] = normalized.astype(np.float32)
            elif len(day_scores) == 1:
                normalized_scores[day_mask] = 0.0
            else:
                normalized_scores[day_mask] = 0.0
        
        return normalized_scores
    
    elif method == 'quantile':
        # 全局分位数归一化
        qt = QuantileTransformer(output_distribution='uniform', random_state=0)
        normalized = qt.fit_transform(scores.reshape(-1, 1)).flatten().astype(np.float32)
        return normalized
    
    else:
        raise ValueError(f"不支持的归一化方法: {method}")


def filter_nan_samples(
    scores: np.ndarray,
    phoneme_seqs: List,
    day_indices: np.ndarray,
    transcriptions: Optional[List] = None
) -> Tuple[np.ndarray, List, np.ndarray, Optional[List]]:
    """
    过滤包含NaN的异常样本
    
    Returns:
        (过滤后的scores, phoneme_seqs, day_indices, transcriptions)
    """
    valid_mask = ~np.isnan(scores)
    
    if np.all(valid_mask):
        return scores, phoneme_seqs, day_indices, transcriptions
    
    num_invalid = np.sum(~valid_mask)
    print(f"   ⚠️  发现 {num_invalid} 个包含NaN的异常样本，将自动过滤")
    
    filtered_scores = scores[valid_mask]
    filtered_phoneme_seqs = [phoneme_seqs[i] for i in range(len(phoneme_seqs)) if valid_mask[i]]
    filtered_day_indices = day_indices[valid_mask]
    
    if transcriptions is not None:
        filtered_transcriptions = [transcriptions[i] for i in range(len(transcriptions)) if valid_mask[i]]
    else:
        filtered_transcriptions = None
    
    return filtered_scores, filtered_phoneme_seqs, filtered_day_indices, filtered_transcriptions


def create_phoneme_text_dataset(
    phoneme_texts: List[str],
    scores: np.ndarray,
    tokenizer,
    max_length: int = 512,
    text_contents: Optional[List[str]] = None
):
    """
    创建用于Model B训练的数据集（包含prompt）
    
    Args:
        phoneme_texts: 音素文本列表
        scores: 分数数组
        tokenizer: tokenizer
        max_length: 最大长度
        text_contents: 转录文本列表（用于构建prompt）
    
    Returns:
        Dataset对象
    """
    from src.model_b.utils.phoneme_dataset import PhonemeTextDataset
    return PhonemeTextDataset(
        phoneme_texts, scores, tokenizer, max_length=max_length, text_contents=text_contents
    )


def format_prompt_instruction(
    phoneme_text: str,
    transcription: Optional[str] = None
) -> str:
    """
    格式化带指令的prompt
    
    Returns:
        格式化的prompt字符串
    """
    sep_token = " [SEP] "

    if transcription is not None and transcription.strip():
        prompt = (
            f"Task: Predict brain-speech decoding error rate{sep_token}"
            f"Focus: Pay special attention to confusable phonemes and long sequences{sep_token}"
            f"Phonemes: {phoneme_text}{sep_token}"
            f"Transcription: {transcription}"
        )
    else:
        prompt = (
            f"Task: Predict brain-speech decoding error rate{sep_token}"
            f"Focus: Pay special attention to confusable phonemes and long sequences{sep_token}"
            f"Phonemes: {phoneme_text}"
        )
    
    return prompt


def build_combined_prompt_zh(
    phoneme_text: str,
    transcription: Optional[str] = None
) -> str:
    """
    原始版本的中文翻译（音素 + 转录）
    """
    if transcription is not None and transcription.strip():
        return f"任务：预测脑语音解码错误率\n音素：{phoneme_text}\n转录文本：{transcription}"
    else:
        return f"任务：预测脑语音解码错误率\n音素：{phoneme_text}"


def build_combined_prompt_en(
    phoneme_text: str,
    transcription: Optional[str] = None
) -> str:
    """
    改进的英文版本（音素 + 转录）
    """
    if transcription is not None and transcription.strip():
        return f"Task: Evaluate the phoneme sequence and its transcription to predict the decoding error rate.\nPhonemes: {phoneme_text}\nTranscription: {transcription}"
    else:
        return f"Task: Evaluate the phoneme sequence to predict the decoding error rate.\nPhonemes: {phoneme_text}"


def format_prompt_native_pair(
    phoneme_text: str,
    transcription_text: str
) -> str:
    return f"<s> {transcription_text} </s></s> {phoneme_text} </s>"


def format_prompt_feature_injection(
    phoneme_text: str,
    transcription_text: str,
    word_count: int,
    phoneme_count: int
) -> str:
    return (
        f"[Length: {word_count}] {transcription_text} "
        f"[SEP] [Phonemes: {phoneme_count}] {phoneme_text}"
    )


def get_expert_task_prefix() -> str:
    # 按照要求移除前缀，如果其他地方需要保留兼容性可返回空字符串
    return ""


def _get_phoneme_vocab() -> List[str]:
    """获取音素词汇表"""
    from src.model_b.utils.phoneme_converter import PHONEME_VOCAB
    return PHONEME_VOCAB


def _validate_phoneme_seq(phoneme_seq: np.ndarray) -> bool:
    """验证音素序列是否有效"""
    vocab_size = len(_get_phoneme_vocab())
    valid_indices = (phoneme_seq >= 0) & (phoneme_seq < vocab_size)
    return np.all(valid_indices)
