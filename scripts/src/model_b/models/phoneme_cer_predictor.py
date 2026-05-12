"""
RoBERTa/CANINE/GPT-2微调模型定义 - 用于预测音素序列的CER分数
"""
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer, GPT2Model, GPT2Tokenizer


def _load_hf_component(loader_fn, model_name: str, component_name: str):
    """
    优先本地缓存加载；若本地不可用则回退到在线加载。
    """
    try:
        return loader_fn(model_name, local_files_only=True)
    except Exception as e_local:
        print(f"⚠️ 本地缓存加载{component_name}失败，尝试在线加载: {e_local}")
        try:
            return loader_fn(model_name)
        except Exception as e_online:
            raise RuntimeError(
                f"无法加载{component_name} '{model_name}'。"
                f"本地缓存加载失败: {e_local}; 在线加载也失败: {e_online}"
            ) from e_online


class PhonemeCERPredictor(nn.Module):
    """
    RoBERTa/CANINE/GPT-2微调模型，用于预测音素序列的CER分数
    支持三种模型类型：
    - RoBERTa: 基于BPE tokenization的encoder模型
    - CANINE: 无分词器的字符级模型，更适合音素序列
    - GPT-2: Decoder-only模型，使用最后一个token的表示
    """
    
    def __init__(self, model_name='roberta-base', model_type='roberta', dropout=0.1, hidden_dim=256):
        """
        Args:
            model_name: 预训练模型名称 
                - RoBERTa: 'roberta-base', 'distilroberta-base'
                - CANINE: 'google/canine-s', 'google/canine-c'
                - GPT-2: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
            model_type: 模型类型 ('roberta', 'canine', 或 'gpt2')
            dropout: Dropout率
            hidden_dim: 回归头隐藏层维度
        """
        super().__init__()
        
        # 加载预训练模型和tokenizer
        self.model_name = model_name
        self.model_type = model_type.lower()
        
        # 根据模型类型加载模型
        if self.model_type == 'canine':
            # CANINE模型
            self.bert = _load_hf_component(AutoModel.from_pretrained, model_name, "CANINE模型")
            self.tokenizer = _load_hf_component(AutoTokenizer.from_pretrained, model_name, "CANINE分词器")
        elif self.model_type == 'roberta':
            # RoBERTa模型
            self.bert = _load_hf_component(AutoModel.from_pretrained, model_name, "RoBERTa模型")
            self.tokenizer = _load_hf_component(AutoTokenizer.from_pretrained, model_name, "RoBERTa分词器")
        elif self.model_type == 'gpt2':
            # GPT-2模型（decoder-only）
            self.bert = _load_hf_component(GPT2Model.from_pretrained, model_name, "GPT2模型")
            self.tokenizer = _load_hf_component(GPT2Tokenizer.from_pretrained, model_name, "GPT2分词器")
            # GPT-2没有pad_token，需要设置
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise ValueError(f"不支持的模型类型: {model_type}，请选择 'roberta', 'canine' 或 'gpt2'")
        
        # 获取隐藏层维度
        hidden_size = self.bert.config.hidden_size
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
            # 注意：不使用Sigmoid，因为CER可能不在0-1范围
        )
        
        # 初始化回归头
        self._init_regressor()
    
    def _init_regressor(self):
        """初始化回归头权重"""
        for module in self.regressor.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, texts, return_hidden=False):
        """
        前向传播
        
        Args:
            texts: 音素文本列表，如 ["SIL AA R K", "B EH T"]
            return_hidden: 是否返回隐藏表示
        
        Returns:
            scores: CER分数预测 [batch_size]
            hidden (可选): 隐藏表示 [batch_size, hidden_size]
        """
        # Tokenize（这里假设texts已经是tokenized的，或者需要在这里tokenize）
        if isinstance(texts, list):
            # 如果输入是文本列表，需要先tokenize
            inputs = self.tokenizer(
                texts, 
                return_tensors='pt', 
                padding=True, 
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(next(self.bert.parameters()).device) for k, v in inputs.items()}
        else:
            # 如果已经是tokenized的，直接使用
            inputs = texts
        
        # 通过编码器（BERT/CANINE）或解码器（GPT-2）
        outputs = self.bert(**inputs)
        
        # 获取表示向量
        if self.model_type == 'gpt2':
            # GPT-2是decoder-only模型，使用最后一个非padding token的表示
            last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            attention_mask = inputs.get('attention_mask', None)
            if attention_mask is not None:
                # 找到每个序列的最后一个非padding位置
                seq_lengths = attention_mask.sum(dim=1) - 1  # -1因为索引从0开始
                batch_size = last_hidden_state.size(0)
                embedding = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), seq_lengths]
            else:
                # 如果没有attention_mask，使用最后一个token
                embedding = last_hidden_state[:, -1, :]
        else:
            # RoBERTa和CANINE使用[CLS] token的表示（第一个token）
            embedding = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # 预测分数
        score = self.regressor(embedding).squeeze(-1)  # [batch_size]
        
        if return_hidden:
            return score, embedding
        else:
            return score
    
    def predict(self, texts, batch_size=32, device='cuda'):
        """
        批量预测
        
        Args:
            texts: 音素文本列表
            batch_size: batch大小
            device: 设备
        
        Returns:
            scores: CER分数预测数组
        """
        self.eval()
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_scores = self.forward(batch_texts)
                scores.extend(batch_scores.cpu().numpy())
        
        return np.array(scores)
