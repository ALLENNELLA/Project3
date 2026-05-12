"""
CNN模型定义 - 用于预测音素序列的CER分数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PhonemeCNNPredictor(nn.Module):
    """
    1D CNN模型，用于预测音素序列的CER分数
    Embedding → 1D Conv (多尺度) → Global Pooling → MLP
    """
    def __init__(self, n_phonemes=41, embedding_dim=64, 
                 num_filters=128, kernel_sizes=[3, 5, 7], dropout=0.3,
                 mlp_dims=[128, 64], feature_mode='embedding',
                 feature_extractor=None):
        super().__init__()
        
        self.n_phonemes = n_phonemes
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.feature_mode = feature_mode
        
        # 特征提取
        if feature_mode == 'embedding':
            self.embedding = nn.Embedding(n_phonemes, embedding_dim, padding_idx=0)
            input_dim = embedding_dim
            self.feature_extractor = None
        elif feature_mode == 'features':
            assert feature_extractor is not None
            self.feature_extractor = feature_extractor
            feature_matrix = torch.FloatTensor(feature_extractor.get_feature_matrix())
            feature_matrix = torch.nan_to_num(feature_matrix, nan=0.0)
            feature_mean = feature_matrix.mean(dim=0, keepdim=True)
            feature_std = feature_matrix.std(dim=0, keepdim=True) + 1e-8
            feature_matrix = (feature_matrix - feature_mean) / feature_std
            self.register_buffer('feature_matrix', feature_matrix)
            input_dim = feature_extractor.feature_dim
            self.embedding = None
        elif feature_mode == 'hybrid':
            assert feature_extractor is not None
            self.feature_extractor = feature_extractor
            feature_matrix = torch.FloatTensor(feature_extractor.get_feature_matrix())
            feature_matrix = torch.nan_to_num(feature_matrix, nan=0.0)
            feature_mean = feature_matrix.mean(dim=0, keepdim=True)
            feature_std = feature_matrix.std(dim=0, keepdim=True) + 1e-8
            feature_matrix = (feature_matrix - feature_mean) / feature_std
            self.register_buffer('feature_matrix', feature_matrix)
            self.embedding = nn.Embedding(n_phonemes, embedding_dim, padding_idx=0)
            input_dim = feature_extractor.feature_dim + embedding_dim
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")
        
        # 多尺度1D卷积
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # MLP head
        mlp_layers = []
        in_dim = num_filters * len(kernel_sizes)
        for mlp_dim in mlp_dims:
            mlp_layers.append(nn.Linear(in_dim, mlp_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = mlp_dim
        mlp_layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
    
    def forward(self, x, lengths=None):
        """
        前向传播
        
        Args:
            x: 音素ID序列 [batch_size, seq_len]
            lengths: 序列长度（可选，当前未使用）
        
        Returns:
            output: CER分数预测 [batch_size, 1]
        """
        x = x.long()
        
        # 获取表示
        if self.feature_mode == 'embedding':
            representation = self.embedding(x)  # [batch, seq_len, embedding_dim]
        elif self.feature_mode == 'features':
            representation = self.feature_matrix[x]
            representation = torch.nan_to_num(representation, nan=0.0)
        elif self.feature_mode == 'hybrid':
            features = self.feature_matrix[x]
            features = torch.nan_to_num(features, nan=0.0)
            embedded = self.embedding(x)
            representation = torch.cat([features, embedded], dim=-1)
        
        # 转置为 [batch, channels, seq_len] (Conv1d要求)
        representation = representation.transpose(1, 2)
        
        # 多尺度卷积 + ReLU + Global Max Pooling
        conv_outs = []
        for conv in self.convs:
            conv_out = F.relu(conv(representation))  # [batch, num_filters, seq_len]
            pooled = F.adaptive_max_pool1d(conv_out, 1).squeeze(2)  # [batch, num_filters]
            conv_outs.append(pooled)
        
        # 拼接所有尺度
        concat = torch.cat(conv_outs, dim=1)  # [batch, num_filters * len(kernel_sizes)]
        concat = self.dropout(concat)
        
        # MLP预测
        output = self.mlp(concat)
        
        return output.squeeze(-1)  # [batch_size]
