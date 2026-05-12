# dataset.py
import torch
from torch.utils.data import Dataset


class SpeechDataset(Dataset):
    """语音数据集类，用于加载脑电数据和对应的音素序列"""
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.n_days = len(data)
        self.n_trials = sum([len(d["sentenceDat"]) for d in data])

        self.neural_feats = []
        self.phone_seqs = []
        self.transcriptions = []  # 添加文本标签
        self.neural_time_bins = []
        self.phone_seq_lens = []
        self.days = []
        for day in range(self.n_days):
            for trial in range(len(data[day]["sentenceDat"])):
                self.neural_feats.append(data[day]["sentenceDat"][trial])
                self.phone_seqs.append(data[day]["phonemes"][trial])
                # 获取文本标签（如果存在）
                if "transcriptions" in data[day] and trial < len(data[day]["transcriptions"]):
                    self.transcriptions.append(data[day]["transcriptions"][trial])
                else:
                    self.transcriptions.append(None)
                self.neural_time_bins.append(data[day]["sentenceDat"][trial].shape[0])
                self.phone_seq_lens.append(data[day]["phoneLens"][trial])
                self.days.append(day)

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        neural_feats = torch.tensor(self.neural_feats[idx], dtype=torch.float32)

        if self.transform:
            neural_feats = self.transform(neural_feats)

        return (
            neural_feats,
            torch.tensor(self.phone_seqs[idx], dtype=torch.int32),
            torch.tensor(self.neural_time_bins[idx], dtype=torch.int32),
            torch.tensor(self.phone_seq_lens[idx], dtype=torch.int32),
            torch.tensor(self.days[idx], dtype=torch.int64),
        )
    
    def get_transcription(self, idx):
        """获取指定索引的文本标签"""
        return self.transcriptions[idx] if idx < len(self.transcriptions) else None
    
    def get_phoneme_seq(self, idx):
        """获取指定索引的音素序列（整数ID序列）"""
        if idx < len(self.phone_seqs):
            return self.phone_seqs[idx]
        return None
    
    def get_phoneme_len(self, idx):
        """获取指定索引的音素序列长度"""
        if idx < len(self.phone_seq_lens):
            return self.phone_seq_lens[idx]
        return None