# models/MogaNet1D.py
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 辅助函数 ====================

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """截断正态分布初始化"""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class DropPath(nn.Module):
    """随机深度（Stochastic Depth）"""
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


def build_act_layer(act_type):
    """构建激活层"""
    if act_type is None:
        return nn.Identity()
    act_dict = {
        'GELU': nn.GELU,
        'ReLU': nn.ReLU,
        'SiLU': nn.SiLU,
        'LeakyReLU': nn.LeakyReLU,
        'ELU': nn.ELU,
        'Tanh': nn.Tanh,
        'Sigmoid': nn.Sigmoid,
        'Mish': nn.Mish,
    }
    assert act_type in act_dict, f'Activation {act_type} not supported'
    return act_dict[act_type]()


def build_norm_layer(norm_type, embed_dims):
    """构建归一化层"""
    assert norm_type in ['BN', 'GN', 'LN', 'IN']
    if norm_type == 'GN':
        num_groups = min(32, embed_dims)
        while embed_dims % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, embed_dims, eps=1e-5)
    elif norm_type == 'LN':
        return nn.LayerNorm(embed_dims, eps=1e-6)
    elif norm_type == 'IN':
        return nn.InstanceNorm1d(embed_dims, eps=1e-5)
    else:
        return nn.BatchNorm1d(embed_dims, eps=1e-5)


class ElementScale(nn.Module):
    """可学习的逐元素缩放"""
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    """通道聚合FFN"""
    def __init__(self, embed_dims, feedforward_channels, kernel_size=3, act_type='GELU', ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.fc1 = nn.Conv1d(embed_dims, feedforward_channels, kernel_size=1)
        self.dwconv = nn.Conv1d(
            feedforward_channels, feedforward_channels, kernel_size=kernel_size,
            stride=1, padding=kernel_size // 2, bias=True, groups=feedforward_channels
        )
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv1d(feedforward_channels, embed_dims, kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)
        self.decompose = nn.Conv1d(feedforward_channels, 1, kernel_size=1)
        self.sigma = ElementScale(feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_act_layer(act_type)

    def feat_decompose(self, x):
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiOrderDWConv(nn.Module):
    """多阶深度卷积"""
    def __init__(self, embed_dims, dw_dilation=[1, 2, 3], channel_split=[1, 3, 4], kernel_sizes=[5, 5, 7]):
        super(MultiOrderDWConv, self).__init__()

        self.split_ratio = [i / sum(channel_split) for i in channel_split]
        self.embed_dims_1 = int(self.split_ratio[1] * embed_dims)
        self.embed_dims_2 = int(self.split_ratio[2] * embed_dims)
        self.embed_dims_0 = embed_dims - self.embed_dims_1 - self.embed_dims_2
        self.embed_dims = embed_dims

        self.DW_conv0 = nn.Conv1d(
            self.embed_dims, self.embed_dims, kernel_size=kernel_sizes[0],
            padding=self._get_padding(kernel_sizes[0], dw_dilation[0]),
            groups=self.embed_dims, stride=1, dilation=dw_dilation[0]
        )
        self.DW_conv1 = nn.Conv1d(
            self.embed_dims_1, self.embed_dims_1, kernel_size=kernel_sizes[1],
            padding=self._get_padding(kernel_sizes[1], dw_dilation[1]),
            groups=self.embed_dims_1, stride=1, dilation=dw_dilation[1]
        )
        self.DW_conv2 = nn.Conv1d(
            self.embed_dims_2, self.embed_dims_2, kernel_size=kernel_sizes[2],
            padding=self._get_padding(kernel_sizes[2], dw_dilation[2]),
            groups=self.embed_dims_2, stride=1, dilation=dw_dilation[2]
        )
        self.PW_conv = nn.Conv1d(embed_dims, embed_dims, kernel_size=1)

    def _get_padding(self, kernel_size, dilation):
        return (kernel_size - 1) * dilation // 2

    def forward(self, x):
        x_0 = self.DW_conv0(x)
        x_1 = self.DW_conv1(x_0[:, self.embed_dims_0: self.embed_dims_0+self.embed_dims_1, ...])
        x_2 = self.DW_conv2(x_0[:, self.embed_dims-self.embed_dims_2:, ...])
        x = torch.cat([x_0[:, :self.embed_dims_0, ...], x_1, x_2], dim=1)
        x = self.PW_conv(x)
        return x


class MultiOrderGatedAggregation(nn.Module):
    """多阶门控聚合"""
    def __init__(self, embed_dims, attn_dw_dilation=[1, 2, 3], attn_channel_split=[1, 3, 4],
                 attn_kernel_sizes=[5, 5, 7], attn_act_type='SiLU', attn_force_fp32=False):
        super(MultiOrderGatedAggregation, self).__init__()

        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv1d(embed_dims, embed_dims, kernel_size=1)
        self.gate = nn.Conv1d(embed_dims, embed_dims, kernel_size=1)
        self.value = MultiOrderDWConv(
            embed_dims, dw_dilation=attn_dw_dilation,
            channel_split=attn_channel_split, kernel_sizes=attn_kernel_sizes
        )
        self.proj_2 = nn.Conv1d(embed_dims, embed_dims, kernel_size=1)
        self.act_value = build_act_layer(attn_act_type)
        self.act_gate = build_act_layer(attn_act_type)
        self.sigma = ElementScale(embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        x_d = F.adaptive_avg_pool1d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward(self, x):
        shortcut = x.clone()
        x = self.feat_decompose(x)
        g = self.gate(x)
        v = self.value(x)
        x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        x = x + shortcut
        return x


class MogaBlock(nn.Module):
    """MogaNet 基本块"""
    def __init__(self, embed_dims, ffn_ratio=4., drop_rate=0., drop_path_rate=0.,
                 act_type='GELU', norm_type='BN', init_value=1e-5,
                 attn_dw_dilation=[1, 2, 3], attn_channel_split=[1, 3, 4],
                 attn_kernel_sizes=[5, 5, 7], attn_act_type='SiLU',
                 ffn_kernel_size=3, attn_force_fp32=False):
        super(MogaBlock, self).__init__()

        self.norm1 = build_norm_layer(norm_type, embed_dims)
        self.attn = MultiOrderGatedAggregation(
            embed_dims, attn_dw_dilation=attn_dw_dilation,
            attn_channel_split=attn_channel_split, attn_kernel_sizes=attn_kernel_sizes,
            attn_act_type=attn_act_type, attn_force_fp32=attn_force_fp32
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_type, embed_dims)
        mlp_hidden_dim = int(embed_dims * ffn_ratio)
        self.mlp = ChannelAggregationFFN(
            embed_dims=embed_dims, feedforward_channels=mlp_hidden_dim,
            kernel_size=ffn_kernel_size, act_type=act_type, ffn_drop=drop_rate
        )
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((1, embed_dims, 1)), requires_grad=True)

    def forward(self, x):
        identity = x
        x = self.layer_scale_1 * self.attn(self.norm1(x))
        x = identity + self.drop_path(x)
        identity = x
        x = self.layer_scale_2 * self.mlp(self.norm2(x))
        x = identity + self.drop_path(x)
        return x


class ConvPatchEmbed(nn.Module):
    """卷积Patch Embedding"""
    def __init__(self, in_channels, embed_dims, kernel_size=3, stride=2, norm_type='BN'):
        super(ConvPatchEmbed, self).__init__()
        self.projection = nn.Conv1d(in_channels, embed_dims, kernel_size=kernel_size,
                                     stride=stride, padding=kernel_size // 2)
        self.norm = build_norm_layer(norm_type, embed_dims)

    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        return x


class StackConvPatchEmbed(nn.Module):
    """堆叠卷积Patch Embedding"""
    def __init__(self, in_channels, embed_dims, kernel_size=3, stride=2, act_type='GELU', norm_type='BN'):
        super(StackConvPatchEmbed, self).__init__()
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels, embed_dims // 2, kernel_size=kernel_size,
                      stride=stride, padding=kernel_size // 2),
            build_norm_layer(norm_type, embed_dims // 2),
            build_act_layer(act_type),
            nn.Conv1d(embed_dims // 2, embed_dims, kernel_size=kernel_size,
                      stride=1, padding=kernel_size // 2),
            build_norm_layer(norm_type, embed_dims),
        )

    def forward(self, x):
        x = self.projection(x)
        return x


# ==================== 主模型 ====================
class MogaNetDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        dropout=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
        # MogaNet 特有参数
        embed_dims=None,
        depths=None,
        ffn_ratios=None,
        act_type='GELU',
        attn_act_type='SiLU',
        drop_path_rate=0.1,
        patch_strides=None,
        patch_sizes=None,
        **kwargs
    ):
        super(MogaNetDecoder, self).__init__()
        
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.dropout_rate = dropout
        self.device = device
        self.bidirectional = bidirectional
        
        # ============ 保存这些属性以兼容训练代码 ============
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        
        # ============ 自动生成配置 ============
        if embed_dims is None:
            base_dim = max(32, hidden_dim // 8)
            embed_dims = [base_dim * (2 ** i) for i in range(layer_dim)]
        
        if depths is None:
            if layer_dim == 4:
                depths = [2, 2, 4, 2]
            elif layer_dim == 3:
                depths = [2, 3, 2]
            else:
                depths = [2] * layer_dim
        
        if ffn_ratios is None:
            ffn_ratios = [4] * layer_dim
        
        # ============ 下采样配置 ============
        if patch_strides is None:
            # 默认：前面下采样，最后不下采样
            patch_strides = [2] * (layer_dim - 1) + [1]
        
        if patch_sizes is None:
            patch_sizes = [3] * layer_dim
        
        self.embed_dims = embed_dims
        self.depths = depths
        self.num_stages = len(depths)
        self.patch_strides = patch_strides
        self.patch_sizes = patch_sizes
        
        # ============ 计算实际的总下采样率 ============
        self.total_downsample_rate = 1
        for stride in patch_strides:
            self.total_downsample_rate *= stride
        
        # 为了兼容训练代码，调整 strideLen 和 kernelLen
        # 使得 (X_len - kernelLen) / strideLen ≈ X_len / total_downsample_rate
        # 简化处理：设置 kernelLen=0，strideLen=total_downsample_rate
        self.strideLen = self.total_downsample_rate
        self.kernelLen = 0
        
        # ============ 构建 Backbone ============
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        
        self.stages = nn.ModuleList()
        cur_block_idx = 0
        
        for i, depth in enumerate(depths):
            # Patch Embedding
            if i == 0:
                patch_embed = StackConvPatchEmbed(
                    in_channels=neural_dim,
                    embed_dims=embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_strides[i],
                    act_type=act_type,
                    norm_type='BN'
                )
            else:
                patch_embed = ConvPatchEmbed(
                    in_channels=embed_dims[i - 1],
                    embed_dims=embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_strides[i],
                    norm_type='BN'
                )
            
            # MogaNet Blocks
            blocks = nn.ModuleList([
                MogaBlock(
                    embed_dims=embed_dims[i],
                    ffn_ratio=ffn_ratios[i],
                    drop_path_rate=dpr[cur_block_idx + j],
                    act_type=act_type,
                    attn_act_type=attn_act_type,
                    norm_type='BN'
                )
                for j in range(depth)
            ])
            
            cur_block_idx += depth
            
            stage = nn.ModuleDict({
                'patch_embed': patch_embed,
                'blocks': blocks,
            })
            self.stages.append(stage)
        
        # ============ 分类头 ============
        self.norm = nn.BatchNorm1d(embed_dims[-1])
        self.final_layer = nn.Linear(embed_dims[-1], n_classes + 1)
        
        # Dropout
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        print(f"MogaNetDecoder initialized:")
        print(f"  - Total downsample rate: {self.total_downsample_rate}x")
        print(f"  - Effective strideLen: {self.strideLen}")
        print(f"  - Effective kernelLen: {self.kernelLen}")
    
    def forward_features(self, x):
        """
        提取特征（全局池化）
        
        Args:
            x: (B, T, C)
        
        Returns:
            features: (B, embed_dims[-1])
        """
        # (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        
        # 通过所有 stage
        for stage in self.stages:
            x = stage['patch_embed'](x)
            for block in stage['blocks']:
                x = block(x)
        
        # 全局平均池化: (B, C, T) -> (B, C)
        x = x.mean(dim=-1)
        x = self.norm(x)
        
        return x
    
    def forward_features_seq(self, x):
        """
        提取序列特征（保留时间维度）
        
        Args:
            x: (B, T, C)
        
        Returns:
            features: (B, T', embed_dims[-1])
        """
        # (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        
        # 通过所有 stage
        for stage in self.stages:
            x = stage['patch_embed'](x)
            for block in stage['blocks']:
                x = block(x)
        
        # (B, C, T') -> (B, T', C)
        x = x.permute(0, 2, 1)
        
        # Batch Norm (需要转换维度)
        B, T, C = x.shape
        x = x.reshape(B * T, C)
        x = self.norm(x)
        x = x.reshape(B, T, C)
        
        return x
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (B, T, C)
        
        Returns:
            logits: (B, T', n_classes+1)
        """
        # 提取序列特征
        x = self.forward_features_seq(x)  # (B, T', embed_dims[-1])
        
        # Dropout
        if self.dropout is not None:
            x = self.dropout(x)
        
        # 分类
        logits = self.final_layer(x)  # (B, T', n_classes+1)
        
        return logits
