import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

from .augmentations import GaussianSmoothing


# ─────────────────────────────────────────────────────────────────
# PEFT 模块：CABlock 和 AdaptFFN
# 结构均为：x_out = x + scale * Up(GELU(Down(x)))
# CABlock   → 插入 ConformerBlock 的 FFN2 之前
# AdaptFFN  → 插入 ConformerDecoder 的 fc_decoder_out 之前
# ─────────────────────────────────────────────────────────────────


class CABlock(nn.Module):
    """
    Channel Aggregation Block（通道聚合块）
    参考 ChannelAggregationFFN 的通道聚合思路：
    x_out = x + sigma * (x - act(decompose(x)))
    插入位置：ConformerBlock 中 FFN2 之前

    Args:
        embed_dim   : 输入/输出维度 d（与 hidden_dim 一致）
        init_scale  : 可学习缩放因子 σ 的初始值，默认 1e-2
        dropout     : Dropout 概率
    """

    def __init__(
        self,
        embed_dim: int,
        bottleneck: int = 64,
        init_scale: float = 1e-2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.bottleneck = bottleneck  # kept for interface compatibility
        self.decompose = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=1,
            kernel_size=1,
            bias=True,
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        # 逐通道 sigma（[1, C, 1]），而非单标量
        self.scale = nn.Parameter(torch.ones(1, embed_dim, 1) * init_scale)

        nn.init.kaiming_normal_(self.decompose.weight, nonlinearity="relu")
        nn.init.zeros_(self.decompose.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        return: [B, T, D]
        """
        # 转为 [B, D, T] 以做按通道聚合，再转回 [B, T, D]
        x_c = x.transpose(1, 2)
        x_decomp = self.act(self.decompose(x_c))  # [B, 1, T]
        out = x_c + self.scale * (x_c - x_decomp)
        out = self.drop(out)
        return out.transpose(1, 2)


class AdaptFFN(nn.Module):
    """
    AdaptFormer 风格输出适配器
    公式：x_out = x + scale * Up(GELU(Down(x)))
    插入位置：ConformerDecoder 的 fc_decoder_out 之前

    Args:
        embed_dim   : 输入/输出维度（与 hidden_dim 一致）
        bottleneck  : 瓶颈维度 r，默认 64
        init_scale  : 可学习缩放因子 σ 的初始值，默认 1e-2
        dropout     : Dropout 概率
    """

    def __init__(
        self,
        embed_dim: int,
        bottleneck: int = 64,
        init_scale: float = 1e-2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.down_proj = nn.Linear(embed_dim, bottleneck, bias=True)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(bottleneck, embed_dim, bias=True)
        self.drop = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(1) * init_scale)

        nn.init.kaiming_normal_(self.down_proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        return: [B, T, D]
        """
        residual = x
        x = self.layer_norm(x)
        x = self.down_proj(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.up_proj(x)
        return residual + self.scale * x


class RelativePositionalEncoding(nn.Module):
    """相对位置编码 - 更适合语音/神经信号"""
    def __init__(self, d_model: int, max_relative_position: int = 100):
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # 可学习的相对位置嵌入
        self.relative_position_k = nn.Embedding(
            2 * max_relative_position + 1, d_model
        )
        self.relative_position_v = nn.Embedding(
            2 * max_relative_position + 1, d_model
        )
        
    def forward(self, length: int):
        """生成相对位置索引"""
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(0).expand(length, length)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # 裁剪到最大相对位置
        distance_mat_clipped = torch.clamp(
            distance_mat, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # 转换为正索引
        final_mat = distance_mat_clipped + self.max_relative_position
        return final_mat


class LocalMultiheadAttention(nn.Module):
    """局部窗口注意力 - 减少计算量，增强局部特征"""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int = 64,
        dropout: float = 0.1,
        use_attn_adapter: bool = False,
        attn_adapter_bottleneck: int = 64,
        attn_adapter_init_scale: float = 1e-2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        self.use_attn_adapter = use_attn_adapter
        
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        if use_attn_adapter:
            # 线性层侧边接入：y = linear(x) + (adaptffn(x) - y)
            # qkv 由 q/k/v 三个分支组成，分别从同一输入 x 走 adapter 后再拼接
            self.q_attn_adapter = AdaptFFN(
                embed_dim=embed_dim,
                bottleneck=attn_adapter_bottleneck,
                init_scale=attn_adapter_init_scale,
                dropout=dropout,
            )
            self.k_attn_adapter = AdaptFFN(
                embed_dim=embed_dim,
                bottleneck=attn_adapter_bottleneck,
                init_scale=attn_adapter_init_scale,
                dropout=dropout,
            )
            self.v_attn_adapter = AdaptFFN(
                embed_dim=embed_dim,
                bottleneck=attn_adapter_bottleneck,
                init_scale=attn_adapter_init_scale,
                dropout=dropout,
            )
            self.out_attn_adapter = AdaptFFN(
                embed_dim=embed_dim,
                bottleneck=attn_adapter_bottleneck,
                init_scale=attn_adapter_init_scale,
                dropout=dropout,
            )
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None):
        """
        x: [Batch, Time, Dim]
        """
        B, T, D = x.shape
        
        # QKV projection
        qkv = self.qkv_proj(x)
        if self.use_attn_adapter:
            qkv_side = torch.cat(
                [
                    self.q_attn_adapter(x),
                    self.k_attn_adapter(x),
                    self.v_attn_adapter(x),
                ],
                dim=-1,
            )
            x_triplet = torch.cat([x, x, x], dim=-1)
            # 方案A：y = linear(x) + (adaptffn(x) - x)
            qkv = qkv + (qkv_side - x_triplet)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D_h]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 局部窗口注意力
        attn_output = self._local_attention(q, k, v, self.window_size)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, T, D)
        
        output = self.out_proj(attn_output)
        if self.use_attn_adapter:
            # 方案A：y = linear(x) + (adaptffn(x) - x)
            output = output + (self.out_attn_adapter(attn_output) - attn_output)
        output = self.dropout(output)
        
        return output, None
    
    def _local_attention(self, q, k, v, window_size):
        """实现局部窗口注意力"""
        B, H, T, D_h = q.shape
        
        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 创建局部窗口mask
        mask = self._create_local_mask(T, window_size, q.device)
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        return output
    
    def _create_local_mask(self, length, window_size, device):
        """创建局部窗口mask"""
        mask = torch.zeros(length, length, device=device)
        for i in range(length):
            start = max(0, i - window_size // 2)
            end = min(length, i + window_size // 2 + 1)
            mask[i, start:end] = 1
        return mask


class PositionalEncoding(nn.Module):
    """正弦位置编码 - 保留用于兼容性"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [Batch, Time, Dim]"""
        return x + self.pe[:, :x.size(1), :]


class ConvolutionModule(nn.Module):
    """Conformer卷积模块"""
    def __init__(self, input_dim: int, kernel_size: int = 31, 
                 expansion_factor: int = 2, dropout: float = 0.1,
                 use_group_norm: bool = True):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Pointwise Conv 1 (GLU需要2倍通道)
        self.pointwise_conv1 = nn.Conv1d(
            input_dim, 
            2 * expansion_factor * input_dim,
            kernel_size=1
        )
        
        # GLU激活
        self.glu = nn.GLU(dim=1)
        
        # Depthwise Conv
        self.depthwise_conv = nn.Conv1d(
            expansion_factor * input_dim,
            expansion_factor * input_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=expansion_factor * input_dim
        )
        
        # Normalization
        if use_group_norm:
            self.norm = nn.GroupNorm(num_groups=1, num_channels=expansion_factor * input_dim)
        else:
            self.norm = nn.BatchNorm1d(expansion_factor * input_dim)
        
        self.activation = nn.SiLU()
        
        # Pointwise Conv 2
        self.pointwise_conv2 = nn.Conv1d(
            expansion_factor * input_dim,
            input_dim,
            kernel_size=1
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [Batch, Time, Dim]"""
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, T, D] -> [B, D, T]
        
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        x = x.transpose(1, 2)  # [B, D, T] -> [B, T, D]
        return x


class FeedForwardModule(nn.Module):
    """前馈模块"""
    def __init__(self, input_dim: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, expansion_factor * input_dim)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(expansion_factor * input_dim, input_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class ConformerBlock(nn.Module):
    """Conformer Block: FFN(1/2) + MHSA + Conv + FFN(1/2)"""
    
    def __init__(self, input_dim: int, num_heads: int, ff_expansion_factor: int = 4,
                 conv_kernel_size: int = 31, conv_expansion_factor: int = 2, 
                 dropout: float = 0.1, use_group_norm: bool = True,
                 window_size: int = 64, max_relative_position: int = 100,
                 use_local_attention: bool = True,
                 use_ca_block: bool = False,
                 ca_bottleneck: int = 64,
                 ca_init_scale: float = 1e-2,
                 # 注意力子层旁的 AdaptFFN（与输出层侧边 AdaptFFN 共享结构）
                 use_attn_adapter: bool = False,
                 attn_adapter_bottleneck: int = 64,
                 attn_adapter_init_scale: float = 1e-2):
        super().__init__()
        
        self.use_local_attention = use_local_attention
        self.use_attn_adapter = use_attn_adapter
        
        # Feed Forward Module 1
        self.ffn1 = FeedForwardModule(input_dim, ff_expansion_factor, dropout)
        
        # Multi-Head Self-Attention (局部或全局)
        self.attn_layer_norm = nn.LayerNorm(input_dim)
        if use_local_attention:
            self.self_attn = LocalMultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                window_size=window_size,
                dropout=dropout,
                use_attn_adapter=use_attn_adapter,
                attn_adapter_bottleneck=attn_adapter_bottleneck,
                attn_adapter_init_scale=attn_adapter_init_scale,
            )
        else:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        self.attn_dropout = nn.Dropout(dropout)

        # 全局注意力分支无法直接接到 qkv/out 线性层侧边，保留输出侧旁路作为兼容回退
        if use_attn_adapter and (not use_local_attention):
            self.attn_adapter = AdaptFFN(
                embed_dim=input_dim,
                bottleneck=attn_adapter_bottleneck,
                init_scale=attn_adapter_init_scale,
                dropout=dropout,
            )
        
        # Relative Position Encoding
        self.rel_pos_enc = RelativePositionalEncoding(
            input_dim, max_relative_position
        )
        
        # Convolution Module
        self.conv_module = ConvolutionModule(
            input_dim, 
            conv_kernel_size, 
            conv_expansion_factor,
            dropout,
            use_group_norm
        )
        
        # 可选：CABlock（在 FFN2 之前）
        self.use_ca_block = use_ca_block
        if use_ca_block:
            self.ca_block = CABlock(
                embed_dim=input_dim,
                bottleneck=ca_bottleneck,
                init_scale=ca_init_scale,
                dropout=dropout,
        )
        
        # Feed Forward Module 2
        self.ffn2 = FeedForwardModule(input_dim, ff_expansion_factor, dropout)
        
        # Layer Normalization
        self.final_layer_norm = nn.LayerNorm(input_dim)
        
        # 可学习的残差权重
        self.alpha_ffn1 = nn.Parameter(torch.ones(1) * 0.5)
        self.alpha_attn = nn.Parameter(torch.ones(1) * 1.0)
        self.alpha_conv = nn.Parameter(torch.ones(1) * 1.0)
        self.alpha_ffn2 = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [Batch, Time, Dim]
            key_padding_mask: [Batch, Time] - True for padding
        """
        # FFN1 (adaptive residual)
        x = x + self.alpha_ffn1 * self.ffn1(x)
        
        # Multi-Head Self-Attention
        residual = x
        x = self.attn_layer_norm(x)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask) if not self.use_local_attention \
                      else self.self_attn(x, key_padding_mask=key_padding_mask)

        # 仅全局注意力回退路径保留输出侧 adapter
        if self.use_attn_adapter and (not self.use_local_attention):
            attn_out = self.attn_adapter(attn_out)

        x = residual + self.alpha_attn * attn_out
        
        # Convolution Module
        x = x + self.alpha_conv * self.conv_module(x)
        
        # CA Block（可选，在 FFN2 之前）
        if self.use_ca_block:
            x = self.ca_block(x)
        
        # FFN2 (adaptive residual)
        x = x + self.alpha_ffn2 * self.ffn2(x)
        
        # Final Layer Norm
        x = self.final_layer_norm(x)
        
        return x


class ConformerDecoder(nn.Module):
    """
    Conformer-based Neural Decoder for Speech BCI
    
    仿照GRUDecoder的预处理流程:
    Input -> Gaussian Smoothing -> Day-specific Layer -> Unfold -> 
    Conformer Blocks -> Output Layer
    """
    
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        num_heads=8,
        dropout=0.1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        strideLen=2,  # 从4改为2，保留更多时序信息
        kernelLen=16,  # 从32改为16，减少信息丢失
        gaussianSmoothWidth=2.0,
        conv_kernel_size=31,
        ff_expansion_factor=4,
        conv_expansion_factor=2,
        use_group_norm=True,
        bidirectional=False,  # 保留兼容性
        window_size=64,  # 局部窗口大小
        max_relative_position=100,
        use_local_attention=True,  # 是否使用局部注意力
        use_positional_encoding=False,  # 是否使用绝对位置编码（默认使用相对位置）
        use_adapter: bool = False,
        adapter_bottleneck: int = 64,
        use_ca_block: bool = False,
        ca_bottleneck: int = 64,
        adapter_init_scale: float = 1e-2,
        ca_init_scale: float = 1e-2,
        **kwargs
    ):
        super(ConformerDecoder, self).__init__()
        
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.num_heads = num_heads
        self.use_positional_encoding = use_positional_encoding
        self.use_local_attention = use_local_attention
        self.use_adapter = use_adapter
        self.use_ca_block = use_ca_block
        self.adapter_bottleneck = adapter_bottleneck
        self.ca_bottleneck = ca_bottleneck
        self.adapter_init_scale = adapter_init_scale
        self.ca_init_scale = ca_init_scale
        
        # Gaussian Smoothing (与GRU一致)
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        
        # Unfold (与GRU一致)
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        
        # Day-specific Input Layer (与GRU一致)
        self.inpLayer = nn.Linear(neural_dim, neural_dim)
        self.inpLayer.weight = nn.Parameter(
            self.inpLayer.weight + torch.eye(neural_dim)
        )
        self.inputLayerNonlinearity = torch.nn.Softsign()
        
        # Input Projection (将unfold后的特征投影到hidden_dim)
        self.input_projection = nn.Linear(neural_dim * self.kernelLen, hidden_dim)
        
        # Positional Encoding (可选)
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(hidden_dim)
        else:
            self.pos_encoding = None
        
        # Conformer Blocks（透传 CA / attention-adapter 相关参数）
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                hidden_dim,
                num_heads,
                ff_expansion_factor,
                conv_kernel_size,
                conv_expansion_factor,
                dropout,
                use_group_norm,
                window_size,
                max_relative_position,
                use_local_attention,
                use_ca_block=use_ca_block,
                ca_bottleneck=ca_bottleneck,
                ca_init_scale=ca_init_scale,
                # AdaptFFN 仅接入 attention 线性层侧边
                use_attn_adapter=use_adapter,
                attn_adapter_bottleneck=adapter_bottleneck,
                attn_adapter_init_scale=adapter_init_scale,
            ) for _ in range(layer_dim)
        ])
        
        # 添加中间层dropout (防止过拟合)
        self.layer_dropout = nn.Dropout(dropout * 0.5)
        
        # Output Layer (CTC)
        self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)
        
        self._init_weights()

    def forward(self, neuralInput):
        """
        Args:
            neuralInput: [Batch, Time, neural_dim]
        Returns:
            output: [Batch, Time', n_classes+1]
        """
        # Gaussian Smoothing (与GRU一致)
        # B, T, D → B, D, T
        x = torch.permute(neuralInput, (0, 2, 1))
        x = self.gaussianSmoother(x)
        x = torch.permute(x, (0, 2, 1))
        
        # Day-specific Input Layer (与GRU一致)
        x = self.inputLayerNonlinearity(self.inpLayer(x))
        
        # Unfold (与GRU一致)
        # [B, T, D] -> [B, D, T] -> [B, D, T, 1] -> unfold -> [B, num_windows, D*kernelLen]
        stridedInputs = torch.permute(
            self.unfolder(torch.unsqueeze(torch.permute(x, (0, 2, 1)), 3)),
            (0, 2, 1)
        )
        
        # Input Projection
        x = self.input_projection(stridedInputs)  # [B, num_windows, hidden_dim]
        
        # Positional Encoding (可选)
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        
        # Conformer Blocks
        for i, block in enumerate(self.conformer_blocks):
            x = block(x, key_padding_mask=None)
            # 在中间层添加dropout
            if i < len(self.conformer_blocks) - 1:
                x = self.layer_dropout(x)
        
        # Output Layer
        seq_out = self.fc_decoder_out(x)
        
        return seq_out

    def forward_features(self, neuralInput):
        """
        提取倒数第二层的特征 (embedding)
        输出维度: (batch, hidden_dim)
        """
        # Gaussian Smoothing
        x = torch.permute(neuralInput, (0, 2, 1))
        x = self.gaussianSmoother(x)
        x = torch.permute(x, (0, 2, 1))
        
        # Day-specific Input Layer
        x = self.inputLayerNonlinearity(self.inpLayer(x))
        
        # Unfold
        stridedInputs = torch.permute(
            self.unfolder(torch.unsqueeze(torch.permute(x, (0, 2, 1)), 3)),
            (0, 2, 1)
        )
        
        # Input Projection
        x = self.input_projection(stridedInputs)
        
        # Positional Encoding
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        
        # Conformer Blocks
        for block in self.conformer_blocks:
            x = block(x, key_padding_mask=None)
        
        # Global Average Pooling (去掉时间维度)
        pooled = torch.mean(x, dim=1)  # (batch, hidden_dim)
        
        return pooled

    def forward_features_seq(self, neuralInput):
        """
        提取序列级别的特征（保留时间步）
        输出维度: (batch, num_windows, hidden_dim)
        """
        # Gaussian Smoothing
        x = torch.permute(neuralInput, (0, 2, 1))
        x = self.gaussianSmoother(x)
        x = torch.permute(x, (0, 2, 1))
        
        # Day-specific Input Layer
        x = self.inputLayerNonlinearity(self.inpLayer(x))
        
        # Unfold
        stridedInputs = torch.permute(
            self.unfolder(torch.unsqueeze(torch.permute(x, (0, 2, 1)), 3)),
            (0, 2, 1)
        )
        
        # Input Projection
        x = self.input_projection(stridedInputs)
        
        # Positional Encoding
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        
        # Conformer Blocks
        for block in self.conformer_blocks:
            x = block(x, key_padding_mask=None)
        
        return x  # 保留时间步维度

    def _init_weights(self):
        """
        改进的初始化策略 - 针对深层网络
        使用深度缩放因子防止梯度消失/爆炸
        """
        # 计算深度缩放因子
        depth_scale = (2 * self.layer_dim) ** -0.5
        
        for name, module in self.named_modules():
            # 跳过 PEFT 模块（CABlock / AdaptFFN），保留其在自身 __init__ 中的初始化
            if any(key in name for key in ["output_adapter", "ca_block", "attn_adapter"]):
                continue
            if isinstance(module, nn.Linear):
                if 'fc_decoder_out' in name:
                    # 输出层：使用较小的初始化
                    nn.init.normal_(module.weight, mean=0, std=0.01)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                
                elif 'input_projection' in name:
                    # 输入投影层
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                
                elif 'linear2' in name or 'out_proj' in name:
                    # 残差连接前的最后一层：使用深度缩放
                    nn.init.xavier_uniform_(module.weight, gain=depth_scale)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                
                elif 'qkv_proj' in name:
                    # QKV投影：标准初始化
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                
                else:
                    # 其他Linear层：标准初始化
                    nn.init.xavier_uniform_(module.weight, gain=1.0)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.Conv1d):
                # 卷积层：使用Kaiming初始化（针对SiLU）
                if 'pointwise_conv2' in name:
                    # 残差连接前的最后一层：使用深度缩放
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
                    module.weight.data *= depth_scale
                else:
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='linear')
                
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
                # 归一化层
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            
            elif isinstance(module, nn.Embedding):
                # Embedding层（相对位置编码）
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def get_lr_scale(self, step, warmup_steps=4000):
        """
        学习率预热调度 (Noam Scheduler)
        用于训练时的学习率调整
        
        Args:
            step: 当前训练步数
            warmup_steps: 预热步数
        
        Returns:
            学习率缩放因子
        """
        if step == 0:
            return 0
        return min(step ** -0.5, step * (warmup_steps ** -1.5))

    # ══════════════════════════════════════════════════════════════
    # PEFT 参数管理：enable_adapter_mode / save_adapter / load_adapter
    # ══════════════════════════════════════════════════════════════

    def enable_adapter_mode(self, also_train_output_head: bool = False):
        """
        冻结所有原始参数，只保留 adapter / ca_block 相关参数可训练。

        Args:
            also_train_output_head: 是否同时解冻 fc_decoder_out（输出层）
        """
        # Step 1：全部冻结
        for param in self.parameters():
            param.requires_grad = False

        # Step 2：解冻 adapter 相关参数（通过参数名过滤）
        adapter_keywords = ['attn_adapter', 'ca_block']
        for name, param in self.named_parameters():
            if any(kw in name for kw in adapter_keywords):
                param.requires_grad = True

        # Step 3：可选解冻输出层
        if also_train_output_head:
            for param in self.fc_decoder_out.parameters():
                param.requires_grad = True

        # 打印可训练参数统计
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[enable_adapter_mode] 可训练参数: {trainable:,} / {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

    def save_adapter(self, path: str):
        """
        只保存 attention adapter / ca_block 相关参数到文件。
        """
        adapter_keywords = ['attn_adapter', 'ca_block']
        adapter_state = {
            name: param.detach().cpu()
            for name, param in self.named_parameters()
            if any(kw in name for kw in adapter_keywords)
        }

        config = {
            'use_adapter': self.use_adapter,
            'use_ca_block': self.use_ca_block,
            'hidden_dim': self.hidden_dim,
            'adapter_bottleneck': self.adapter_bottleneck,
            'ca_bottleneck': self.ca_bottleneck,
            'adapter_init_scale': self.adapter_init_scale,
            'ca_init_scale': self.ca_init_scale,
        }

        torch.save({'adapter_state': adapter_state, 'config': config}, path)
        print(f"[save_adapter] 已保存 {len(adapter_state)} 个参数张量 → {path}")
        print(f"[save_adapter] config: {config}")

    def load_adapter(self, path: str, strict: bool = True):
        """
        从文件加载 attention adapter / ca_block 参数，注入当前模型。
        原始模型权重不受影响。
        """
        checkpoint = torch.load(path, map_location='cpu')
        adapter_state = checkpoint.get('adapter_state', {})
        config = checkpoint.get('config', {})

        print(f"[load_adapter] 正在加载 {len(adapter_state)} 个参数张量 ← {path}")
        print(f"[load_adapter] 保存时 config: {config}")

        current_state = dict(self.named_parameters())

        missing_keys = []
        unexpected_keys = []
        loaded_count = 0

        for name, saved_tensor in adapter_state.items():
            if name in current_state:
                if current_state[name].shape == saved_tensor.shape:
                    current_state[name].data.copy_(saved_tensor.to(current_state[name].device))
                    loaded_count += 1
                else:
                    msg = (
                        f"[load_adapter] 形状不匹配: {name} "
                        f"模型={current_state[name].shape}, 文件={saved_tensor.shape}"
                    )
                    if strict:
                        raise RuntimeError(msg)
                    else:
                        print(f"[WARNING] {msg}，已跳过")
            else:
                unexpected_keys.append(name)
                if strict:
                    raise RuntimeError(
                        f"[load_adapter] 文件中存在模型没有的参数: {name}"
                    )

        for name in current_state:
            if any(kw in name for kw in ['attn_adapter', 'ca_block']):
                if name not in adapter_state:
                    missing_keys.append(name)

        if missing_keys and strict:
            raise RuntimeError(
                f"[load_adapter] 模型中存在文件没有的 adapter 参数: {missing_keys}"
            )

        print(f"[load_adapter] 成功加载 {loaded_count} 个参数张量")
        if missing_keys:
            print(f"[load_adapter] 缺失参数（未加载）: {missing_keys}")
        if unexpected_keys:
            print(f"[load_adapter] 多余参数（已忽略）: {unexpected_keys}")


# ============= 训练辅助工具 =============

class ConformerTrainingConfig:
    """
    Conformer训练配置 - 提供优化器、调度器等配置
    """
    
    @staticmethod
    def get_optimizer(model, base_lr=5e-4, weight_decay=1e-4):
        """
        分层学习率 + 权重衰减
        
        Args:
            model: ConformerDecoder模型
            base_lr: 基础学习率
            weight_decay: 权重衰减系数
        
        Returns:
            optimizer: AdamW优化器
        """
        from torch.optim import AdamW
        
        # 不同层使用不同学习率
        param_groups = []
        
        # 输入层：较大学习率
        input_params = [
            p
            for n, p in model.named_parameters()
            if ("inpLayer" in n or "input_projection" in n) and p.requires_grad
        ]
        if input_params:
            param_groups.append({
                'params': input_params,
                'lr': base_lr * 2.0,
                'weight_decay': weight_decay
            })
        
        # Conformer blocks：标准学习率
        conformer_params = [
            p
            for n, p in model.named_parameters()
            if "conformer_blocks" in n and p.requires_grad
        ]
        if conformer_params:
            param_groups.append({
                'params': conformer_params,
                'lr': base_lr,
                'weight_decay': weight_decay
            })
        
        # 输出层：较小学习率
        output_params = [
            p
            for n, p in model.named_parameters()
            if "fc_decoder_out" in n and p.requires_grad
        ]
        if output_params:
            param_groups.append({
                'params': output_params,
                'lr': base_lr * 0.5,
                'weight_decay': weight_decay * 0.1
            })
        
        # 其他参数
        other_params = [
            p
            for n, p in model.named_parameters()
            if not any(
                key in n
                for key in ["inpLayer", "input_projection", "conformer_blocks", "fc_decoder_out"]
            )
            and p.requires_grad
        ]
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr,
                'weight_decay': weight_decay
            })
        
        optimizer = AdamW(param_groups, betas=(0.9, 0.98), eps=1e-9)
        return optimizer
    
    @staticmethod
    def get_scheduler(optimizer, warmup_steps=4000):
        """
        Noam学习率调度器 (Transformer原论文使用)
        
        Args:
            optimizer: 优化器
            warmup_steps: 预热步数
        
        Returns:
            scheduler: LambdaLR调度器
        """
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(step):
            if step == 0:
                return 0
            return min(step ** -0.5, step * (warmup_steps ** -1.5))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        return scheduler


class ProgressiveTraining:
    """
    渐进式训练：先训练浅层，再逐步解冻深层
    用于提升训练稳定性
    """
    def __init__(self, model, total_epochs=100):
        self.model = model
        self.total_epochs = total_epochs
        self.num_blocks = len(model.conformer_blocks)
    
    def freeze_blocks(self, num_frozen_blocks):
        """冻结前N个Conformer blocks"""
        for i in range(num_frozen_blocks):
            for param in self.model.conformer_blocks[i].parameters():
                param.requires_grad = False
    
    def unfreeze_all(self):
        """解冻所有层"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def get_frozen_blocks(self, epoch):
        """
        根据epoch决定冻结多少层
        
        Args:
            epoch: 当前epoch
        
        Returns:
            num_frozen: 需要冻结的block数量
        """
        if epoch < self.total_epochs * 0.2:
            # 前20% epoch: 只训练最后2层
            return max(0, self.num_blocks - 2)
        elif epoch < self.total_epochs * 0.5:
            # 20%-50% epoch: 训练最后4层
            return max(0, self.num_blocks - 4)
        else:
            # 50%之后: 全部训练
            return 0
