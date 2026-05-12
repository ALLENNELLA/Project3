"""
音素序列数据集类（用于HuggingFace训练）
"""
import torch
from torch.utils.data import Dataset
import numpy as np


class PhonemeTextDataset(Dataset):
    """
    音素文本数据集，用于LLM微调（带prompt格式）
    """
    
    def __init__(self, phoneme_texts, cer_scores, tokenizer, max_length=512, text_contents=None):
        """
        Args:
            phoneme_texts: 音素文本列表，如 ["SIL AA R K", "B EH T"]
            cer_scores: CER分数列表
            tokenizer: HuggingFace tokenizer
            max_length: 最大序列长度
            text_contents: 原始文本内容列表（transcriptions），可选
        """
        self.phoneme_texts = phoneme_texts
        self.cer_scores = np.array(cer_scores, dtype=np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_contents = text_contents if text_contents is not None else [None] * len(phoneme_texts)
        
        print(f"   数据集大小: {len(self.phoneme_texts)}")
        print(f"   CER范围: [{self.cer_scores.min():.4f}, {self.cer_scores.max():.4f}]")
        print(f"   CER均值: {self.cer_scores.mean():.4f}")
        # 检查prompt格式（instruction格式应该包含"任务:预测"）
        if len(self.phoneme_texts) > 0 and "任务:预测" in self.phoneme_texts[0]:
            print(f"   使用instruction格式prompt（已包含完整指令）")
        elif any(t is not None for t in self.text_contents):
            print(f"   使用prompt格式输入（包含音素和文本）")
        else:
            print(f"   使用prompt格式输入（仅音素）")
    
    def __len__(self):
        return len(self.phoneme_texts)
    
    def __getitem__(self, idx):
        # phoneme_texts 已经是完整的 instruction 格式 prompt，直接使用
        prompt_text = self.phoneme_texts[idx]
        score = self.cer_scores[idx]
        
        # Tokenize（prompt_text 已经是完整的 instruction 格式）
        encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(score, dtype=torch.float32)
        }
