"""
音素ID序列数据集类（用于CNN训练）
"""
import torch
from torch.utils.data import Dataset
import numpy as np


class PhonemeIDDataset(Dataset):
    """
    音素ID序列数据集，用于CNN模型训练
    """
    
    def __init__(self, phoneme_seqs, scores, max_len=None):
        """
        Args:
            phoneme_seqs: 音素ID序列列表，每个序列是numpy array或list
            scores: CER分数列表
            max_len: 最大序列长度（如果为None，则使用所有序列的最大长度）
        """
        self.phoneme_seqs = phoneme_seqs
        self.scores = np.array(scores, dtype=np.float32)
        
        if max_len is None:
            self.max_len = max([len(seq) for seq in phoneme_seqs]) if len(phoneme_seqs) > 0 else 512
        else:
            self.max_len = max_len
        
        print(f"   数据集大小: {len(self.phoneme_seqs)}")
        print(f"   最大序列长度: {self.max_len}")
        if len(self.phoneme_seqs) > 0:
            print(f"   平均序列长度: {np.mean([len(seq) for seq in self.phoneme_seqs]):.1f}")
        print(f"   CER范围: [{self.scores.min():.4f}, {self.scores.max():.4f}]")
        print(f"   CER均值: {self.scores.mean():.4f}")
    
    def __len__(self):
        return len(self.phoneme_seqs)
    
    def __getitem__(self, idx):
        seq = self.phoneme_seqs[idx]
        score = self.scores[idx]
        
        # 转换为numpy array
        if isinstance(seq, (list, tuple)):
            seq = np.array(seq, dtype=np.int64)
        else:
            seq = seq.astype(np.int64)
        
        # Padding
        padded_seq = np.zeros(self.max_len, dtype=np.int64)
        seq_len = min(len(seq), self.max_len)
        padded_seq[:seq_len] = seq[:seq_len]
        
        return torch.LongTensor(padded_seq), torch.FloatTensor([score]), seq_len
