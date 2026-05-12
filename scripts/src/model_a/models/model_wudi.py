import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from decoder import BeamSearchDecoder

class ConformerBlock(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, ff_mult: int = 4,
                 conv_kernel_size: int = 31, conv_expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, ff_mult * input_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * input_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, ff_mult * input_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * input_dim, input_dim),
            nn.Dropout(dropout)
        )
        self.attn_norm = nn.LayerNorm(input_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads,
            batch_first=True, dropout=dropout
        )
        self.conv_norm = nn.LayerNorm(input_dim)
        self.pointwise1 = nn.Conv1d(input_dim, conv_expansion * input_dim, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            input_dim, input_dim, kernel_size=conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2, groups=input_dim
        )
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.act = nn.SiLU()
        self.pointwise2 = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        self.conv_dropout = nn.Dropout(dropout)
        self._init_weights()

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        attn_out, _ = self.self_attn(
            self.attn_norm(x), self.attn_norm(x), self.attn_norm(x),
            key_padding_mask=key_padding_mask
        )
        x = x + attn_out
        conv_in = self.conv_norm(x).transpose(1, 2)
        c = self.conv_dropout(
            self.pointwise2(
                self.act(
                    self.batch_norm(
                        self.depthwise(self.glu(self.pointwise1(conv_in)))
                    )
                )
            )
        ).transpose(1, 2)
        x = x + c
        x = x + 0.5 * self.ffn2(x)
        return x

    def _init_weights(self):
        for m in [self.ffn1, self.ffn2]:
            nn.init.xavier_uniform_(m[1].weight)
            nn.init.xavier_uniform_(m[4].weight)
        nn.init.xavier_uniform_(self.pointwise1.weight)
        nn.init.xavier_uniform_(self.pointwise2.weight)


class DayLayer(nn.Module):
    """
    为每日神经信号特征学习一个独立的线性变换
    """

    def __init__(self, n_days: int, input_dim: int):
        super().__init__()
        self.n_days = n_days
        self.input_dim = input_dim

        # 参考官方代码的初始化方式
        self.dayWeights = nn.Parameter(torch.randn(n_days, input_dim, input_dim))
        self.dayBias = nn.Parameter(torch.zeros(n_days, 1, input_dim))

        # 初始化为单位矩阵
        for x in range(n_days):
            self.dayWeights.data[x, :, :] = torch.eye(input_dim)

        self.inputLayerNonlinearity = nn.Softsign()

    def forward(self, x: torch.Tensor, day_indices: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, Time, Dim]
        day_indices: [Batch] - 每个样本对应的day index
        """
        dayWeights = torch.index_select(self.dayWeights, 0, day_indices)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", x, dayWeights
        ) + torch.index_select(self.dayBias, 0, day_indices)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        return transformedNeural


class ConformerEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, num_layers: int, num_heads: int,
                 conv_kernel_size: int = 31, ff_expansion_factor: int = 4,
                 dropout: float = 0.1, subsampling_factor: int = 4, conv_expansion: int = 2):
        super().__init__()
        self.subsample = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads, ff_mult=ff_expansion_factor,
                           conv_kernel_size=conv_kernel_size, conv_expansion=conv_expansion,
                           dropout=dropout) for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self._init_weights()

    def _generate_key_padding_mask(self, seq_len: int, lengths: torch.Tensor) -> torch.Tensor:
        batch_size = lengths.size(0)
        arange = torch.arange(seq_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
        return arange >= lengths.unsqueeze(1)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]]:
        x = self.subsample(x.transpose(1, 2)).transpose(1, 2)
        if lengths is not None:
            output_lengths = ((lengths - 1) // 2 + 1 - 1) // 2 + 1
            output_lengths = torch.clamp(output_lengths, min=1)
            key_padding_mask = self._generate_key_padding_mask(x.size(1), output_lengths)
        else:
            output_lengths = None
            key_padding_mask = None
        for block in self.conformer_blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return self.output_norm(x), output_lengths

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.subsample[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.subsample[2].weight, mode='fan_in', nonlinearity='relu')

'''
config = {
        'input_dim': 256,
        'vocab_size': vocab_size,
        'blank_idx': tokenizer.token_to_id('<blank>'),
        'eos_idx': tokenizer.token_to_id('<eos>'),
        'n_days': n_days,
        'encoder_dim': 512,
        'encoder_layers': 2,
        'encoder_heads': 4,
        'conv_kernel_size': 31,
        'ff_expansion_factor': 4,
        'subsampling_factor': 4,
        'embed_dim': 256,
        'predictor_dim': 512,
        'predictor_layers': 2,
        'joint_dim': 512,
        'dropout': 0.4,
        'learning_rate': 1e-3,
        'num_epochs': 50,
        'steps_per_epoch': len(train_loader),
    }
'''