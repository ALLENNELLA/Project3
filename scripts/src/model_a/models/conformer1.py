import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

# ==========================================
# 1. 辅助模块
# ==========================================

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, input_channels=256, max_len=5000):
        super().__init__()
        # 1. 时间维度卷积
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, emb_size, kernel_size=(1, 25), stride=(1, 1), padding=(0, 12)),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
        )
        # 2. 空间维度卷积
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(emb_size, emb_size, kernel_size=(input_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
        )
        # 3. 降采样
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 4), padding=(0, 37))
        # 4. 位置编码
        self.positions = nn.Parameter(torch.randn(max_len, emb_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: Tensor):
        x = self.temporal_conv(x) 
        x = self.spatial_conv(x)  
        x = self.pool(x)          
        x = x.squeeze(2).permute(0, 2, 1) 
        
        seq_len = x.shape[1]
        if seq_len > self.positions.shape[0]:
             x = x + self.positions[:self.positions.shape[0], :].unsqueeze(0)[:, :seq_len, :]
        else:
             x = x + self.positions[:seq_len, :].unsqueeze(0)
        
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = self.queries(x)
        keys = self.keys(x)
        values = self.values(x)
        
        # Reshape: [Batch, Time, Heads, HeadDim] -> [Batch, Heads, Time, HeadDim]
        queries = queries.view(queries.shape[0], queries.shape[1], self.num_heads, -1).transpose(1, 2)
        keys = keys.view(keys.shape[0], keys.shape[1], self.num_heads, -1).transpose(1, 2)
        values = values.view(values.shape[0], values.shape[1], self.num_heads, -1).transpose(1, 2)
        
        # Attention
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        
        # Output: [Batch, Heads, Time, HeadDim]
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        
        # --- 关键修复点 ---
        # 1. 先转置回 [Batch, Time, Heads, HeadDim]
        out = out.transpose(1, 2).contiguous()
        # 2. 再 Flatten 为 [Batch, Time, EmbSize]
        # 此时 out.shape[1] 是 Time，逻辑正确
        out = out.view(out.shape[0], out.shape[1], -1)
        
        out = self.projection(out)
        return out

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, dropout):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
        )

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=8, drop_p=0.5, forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, dropout=forward_drop_p),
                nn.Dropout(drop_p)
            ))
        )

# ==========================================
# 2. 主模型类
# ==========================================

class ECoGConformer(nn.Module):
    def __init__(self, 
                 input_channels=256, 
                 num_classes=40, 
                 emb_size=128,      
                 depth=3,           
                 num_heads=8,
                 strideLen = 8,
                 kernelLen = 16,
                 **kwargs):
        super().__init__()
        
        # 1. Day Adapter
        self.day_adapter = nn.Linear(input_channels, input_channels)
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        # 2. 特征提取
        self.patch_embedding = PatchEmbedding(emb_size, input_channels, **kwargs)
        
        # 3. Transformer
        self.transformer = nn.Sequential(
            *[TransformerEncoderBlock(emb_size, num_heads=num_heads) for _ in range(depth)]
        )
        
        # 4. 输出头
        self.layer_norm = nn.LayerNorm(emb_size)
        self.fc_out = nn.Linear(emb_size, num_classes)

    def forward(self, x: Tensor):
        # 维度清洗
        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
            
        # 跨天对齐
        x = self.day_adapter(x)
        
        # 转换格式 [Batch, Time, Channels] -> [Batch, 1, Channels, Time]
        x = x.unsqueeze(1).permute(0, 1, 3, 2)
        
        # 模型主干
        x = self.patch_embedding(x)
        x = self.transformer(x)
        x = self.layer_norm(x)
        output = self.fc_out(x)
        
        return output
