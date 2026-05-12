"""
音素序列转换模块
用于将音素ID序列转换为文本格式
"""
import numpy as np

# 音素ID到文本的映射（与make_dataset.py中的定义一致）
PHONEME_VOCAB = [
    "SIL",  # 0: 静音/padding
    "AA", "AE", "AH", "AO", "AW",  # 1-5
    "AY", "B", "CH", "D", "DH",    # 6-10
    "EH", "ER", "EY", "F", "G",    # 11-15
    "HH", "IH", "IY", "JH", "K",   # 16-20
    "L", "M", "N", "NG", "OW",     # 21-25
    "OY", "P", "R", "S", "SH",     # 26-30
    "T", "TH", "UH", "UW", "V",    # 31-35
    "W", "Y", "Z", "ZH"            # 36-39
]


def phoneme_seq_to_text(phoneme_indices, phone_len=None, remove_padding=True):
    """
    将音素ID序列转换为文本格式
    
    注意：此函数与训练Model B时使用的格式完全一致（remove_padding=True时移除所有0）
    
    Args:
        phoneme_indices: 音素ID序列（numpy array或list）
        phone_len: 音素序列长度（如果提供，只取前phone_len个，然后再移除0）
        remove_padding: 是否移除所有padding（索引0），默认为True，与训练时一致
    
    Returns:
        str: 音素文本序列，如 "AA AE AH"（移除了所有0）
    """
    # 转换为numpy array
    if isinstance(phoneme_indices, (list, tuple)):
        phoneme_indices = np.array(phoneme_indices)
    
    # 如果提供了长度，先只取前phone_len个
    if phone_len is not None:
        phoneme_indices = phoneme_indices[:phone_len]
    
    # 移除所有padding（索引0），与训练时使用的remove_padding=True行为一致
    # 这是关键：训练时使用remove_padding=True，会移除所有0，包括序列中间的0
    if remove_padding:
        phoneme_indices = phoneme_indices[phoneme_indices != 0]
    
    # 确保索引在有效范围内
    valid_indices = phoneme_indices[(phoneme_indices >= 0) & (phoneme_indices < len(PHONEME_VOCAB))]
    
    # 转换为文本
    phonemes = [PHONEME_VOCAB[int(idx)] for idx in valid_indices]
    return " ".join(phonemes)
