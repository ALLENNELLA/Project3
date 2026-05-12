"""
Ladder Side-Tuning for Conformer Decoder
实现参数高效的微调策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import math


class LightweightConformerBlock(nn.Module):
    """
    轻量级 Conformer Block（用于侧边网络）
    
    相比完整版的改动：
    1. 去掉卷积模块（减少参数）
    2. 简化 FFN（expansion_factor 从 4 降到 2）
    3. 可选：使用更少的注意力头
    """
    
    def __init__(self, input_dim: int, num_heads: int, 
                 ff_expansion_factor: int = 2,  # 从4降到2
                 dropout: float = 0.1,
                 use_local_attention: bool = False,
                 window_size: int = 64):
        super().__init__()
        
        self.use_local_attention = use_local_attention
        
        # Feed Forward Module 1 (简化版)
        self.ffn1 = self._make_ffn(input_dim, ff_expansion_factor, dropout)
        
        # Multi-Head Self-Attention
        self.attn_layer_norm = nn.LayerNorm(input_dim)
        if use_local_attention:
            # 如果需要，可以导入你的 LocalMultiheadAttention
            self.self_attn = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        else:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        self.attn_dropout = nn.Dropout(dropout)
        
        # Feed Forward Module 2
        self.ffn2 = self._make_ffn(input_dim, ff_expansion_factor, dropout)
        
        # Layer Normalization
        self.final_layer_norm = nn.LayerNorm(input_dim)
        
        # 残差权重（较小的初始值，让侧边网络初期贡献较小）
        self.alpha_ffn1 = nn.Parameter(torch.ones(1) * 0.3)
        self.alpha_attn = nn.Parameter(torch.ones(1) * 0.5)
        self.alpha_ffn2 = nn.Parameter(torch.ones(1) * 0.3)
    
    def _make_ffn(self, input_dim, expansion_factor, dropout):
        """构建前馈网络"""
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, expansion_factor * input_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(expansion_factor * input_dim, input_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, 
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [Batch, Time, Dim]
        """
        # FFN1
        x = x + self.alpha_ffn1 * self.ffn1(x)
        
        # Multi-Head Self-Attention
        residual = x
        x = self.attn_layer_norm(x)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + self.alpha_attn * attn_out
        
        # FFN2
        x = x + self.alpha_ffn2 * self.ffn2(x)
        
        # Final Layer Norm
        x = self.final_layer_norm(x)
        
        return x


class LadderFusion(nn.Module):
    """
    梯子融合模块：融合主干和侧边网络的输出
    
    支持三种融合策略：
    1. 'learnable_scalar': 每层一个可学习标量权重
    2. 'gated': 门控机制（更灵活但参数更多）
    3. 'fixed': 固定权重（不可学习）
    """
    
    def __init__(self, hidden_dim: int, num_layers: int, 
                 fusion_type: str = 'learnable_scalar',
                 init_alpha: float = 0.7):
        super().__init__()
        self.fusion_type = fusion_type
        self.num_layers = num_layers
        
        if fusion_type == 'learnable_scalar':
            # 每层一个可学习的 alpha（初始化为 init_alpha）
            # alpha 控制主干的贡献，(1-alpha) 控制侧边的贡献
            self.alphas = nn.Parameter(
                torch.ones(num_layers) * self._inverse_sigmoid(init_alpha)
            )
        
        elif fusion_type == 'gated':
            # 门控网络：根据特征动态决定融合权重
            self.gate_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, 1),
                    nn.Sigmoid()
                ) for _ in range(num_layers)
            ])
        
        elif fusion_type == 'fixed':
            # 固定权重（不可学习）
            self.register_buffer('alphas', torch.ones(num_layers) * init_alpha)
        
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
    
    def _inverse_sigmoid(self, x):
        """sigmoid 的反函数，用于初始化"""
        x = torch.clamp(torch.tensor(x), 0.01, 0.99)
        return torch.log(x / (1 - x))
    
    def forward(self, main_out: torch.Tensor, side_out: torch.Tensor, 
                layer_idx: int) -> torch.Tensor:
        """
        Args:
            main_out: 主干网络输出 [Batch, Time, Dim]
            side_out: 侧边网络输出 [Batch, Time, Dim]
            layer_idx: 当前层索引
        """
        if self.fusion_type == 'learnable_scalar':
            alpha = torch.sigmoid(self.alphas[layer_idx])
            output = alpha * main_out + (1 - alpha) * side_out
        
        elif self.fusion_type == 'gated':
            # 拼接特征，通过门控网络计算权重
            concat_features = torch.cat([main_out, side_out], dim=-1)
            gate = self.gate_nets[layer_idx](concat_features)  # [B, T, 1]
            output = gate * main_out + (1 - gate) * side_out
        
        elif self.fusion_type == 'fixed':
            alpha = self.alphas[layer_idx]
            output = alpha * main_out + (1 - alpha) * side_out
        
        return output
    
    def get_alpha_values(self):
        """获取当前的 alpha 值（用于监控）"""
        if self.fusion_type == 'learnable_scalar':
            return torch.sigmoid(self.alphas).detach().cpu().numpy()
        elif self.fusion_type == 'fixed':
            return self.alphas.cpu().numpy()
        else:
            return None  # 门控机制没有固定的 alpha


class ConformerLST(nn.Module):
    """
    Conformer with Ladder Side-Tuning
    
    架构：
    1. 主干网络（Backbone）：冻结的预训练 Conformer
    2. 侧边网络（Side Network）：轻量级可训练 Conformer
    3. 梯子融合（Ladder Fusion）：逐层融合主干和侧边的输出
    
    使用方式：
        # 从预训练模型创建 LST 模型
        lst_model = ConformerLST.from_pretrained(
            pretrained_model,
            side_hidden_dim=128,
            side_num_heads=4
        )
    """
    
    def __init__(self,
                 backbone_model,  # 预训练的 ConformerDecoder
                 side_hidden_dim: int = 128,  # 侧边网络隐藏层维度
                 side_num_heads: int = 4,     # 侧边网络注意力头数
                 side_ff_expansion: int = 2,  # 侧边网络 FFN 扩展因子
                 fusion_type: str = 'learnable_scalar',  # 融合策略
                 init_alpha: float = 0.7,     # 初始融合权重（主干占比）
                 side_dropout: float = 0.1,
                 ladder_mode: str = 'sequential',  # 'parallel' 或 'sequential'
                 use_side_projection: bool = True,  # 是否使用投影层
                 ):
        super().__init__()
        
        # ========== 保存配置 ==========
        self.backbone_hidden_dim = backbone_model.hidden_dim
        self.side_hidden_dim = side_hidden_dim
        self.num_layers = backbone_model.layer_dim
        self.n_classes = backbone_model.n_classes
        self.ladder_mode = ladder_mode
        self.use_side_projection = use_side_projection

        self.kernelLen = backbone_model.kernelLen
        self.strideLen = backbone_model.strideLen
        self.neural_dim = backbone_model.neural_dim
        self.hidden_dim = backbone_model.hidden_dim
        self.layer_dim = backbone_model.layer_dim
        self.dropout = backbone_model.dropout
        self.gaussianSmoothWidth = backbone_model.gaussianSmoothWidth
        
        # ========== 主干网络（冻结）==========
        self.backbone = backbone_model
        self._freeze_backbone()
        
        # ========== 维度投影层 ==========
        # 如果侧边网络维度与主干不同，需要投影
        if use_side_projection and side_hidden_dim != self.backbone_hidden_dim:
            # 主干 -> 侧边（在侧边网络输入前）
            self.main_to_side = nn.Linear(self.backbone_hidden_dim, side_hidden_dim)
            # 侧边 -> 主干（在融合前）
            self.side_to_main = nn.Linear(side_hidden_dim, self.backbone_hidden_dim)
        else:
            self.main_to_side = nn.Identity()
            self.side_to_main = nn.Identity()
            # 如果不使用投影，侧边维度必须等于主干维度
            if side_hidden_dim != self.backbone_hidden_dim:
                raise ValueError(
                    f"side_hidden_dim ({side_hidden_dim}) must equal "
                    f"backbone_hidden_dim ({self.backbone_hidden_dim}) "
                    f"when use_side_projection=False"
                )
        
        # ========== 侧边网络 ==========
        self.side_blocks = nn.ModuleList([
            LightweightConformerBlock(
                input_dim=side_hidden_dim,
                num_heads=side_num_heads,
                ff_expansion_factor=side_ff_expansion,
                dropout=side_dropout,
                use_local_attention=False,
            ) for _ in range(self.num_layers)
        ])
        
        # ========== 梯子融合 ==========
        self.fusion = LadderFusion(
            hidden_dim=self.backbone_hidden_dim,
            num_layers=self.num_layers,
            fusion_type=fusion_type,
            init_alpha=init_alpha
        )
        
        # ========== 输出层（复用主干的，但可选择重新训练）==========
        # 选项1：冻结输出层（使用预训练的）
        # for param in self.backbone.fc_decoder_out.parameters():
        #     param.requires_grad = False
        
        # 选项2：解冻输出层（推荐，让输出层适应新数据）
        for param in self.backbone.fc_decoder_out.parameters():
            param.requires_grad = True
        
        # ========== 初始化侧边网络 ==========
        self._init_side_network()
    
    def _freeze_backbone(self):
        """冻结主干网络的所有参数"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"✅ 主干网络已冻结（{sum(p.numel() for p in self.backbone.parameters())} 参数）")
    
    def _init_side_network(self):
        """初始化侧边网络（较小的初始化，让其初期贡献较小）"""
        for module in self.side_blocks.modules():
            if isinstance(module, nn.Linear):
                # 使用较小的标准差初始化
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)
        
        # 投影层也使用小初始化
        if isinstance(self.side_to_main, nn.Linear):
            nn.init.normal_(self.side_to_main.weight, mean=0, std=0.02)
            nn.init.constant_(self.side_to_main.bias, 0)
    
    def forward(self, neuralInput: torch.Tensor) -> torch.Tensor:
        """
        Args:
            neuralInput: [Batch, Time, neural_dim]
        Returns:
            output: [Batch, Time', n_classes+1]
        """
        # ========== 预处理（与主干网络相同）==========
        # Gaussian Smoothing + Day-specific Layer + Unfold
        x = self._preprocess(neuralInput)  # [B, num_windows, backbone_hidden_dim]
        
        # ========== 逐层处理（Ladder Side-Tuning）==========
        for i in range(self.num_layers):
            # 1. 主干网络前向传播（冻结，无梯度）
            with torch.no_grad():
                main_out = self.backbone.conformer_blocks[i](x)
            
            # 2. 侧边网络前向传播（可训练）
            if self.ladder_mode == 'sequential':
                # Sequential: 侧边网络接收主干的输出
                side_input = self.main_to_side(main_out)
            elif self.ladder_mode == 'parallel':
                # Parallel: 侧边网络接收相同的输入
                side_input = self.main_to_side(x)
            else:
                raise ValueError(f"Unknown ladder_mode: {self.ladder_mode}")
            
            side_out = self.side_blocks[i](side_input)
            side_out = self.side_to_main(side_out)  # 投影回主干维度
            
            # 3. 融合主干和侧边的输出
            x = self.fusion(main_out, side_out, layer_idx=i)
        
        # ========== 输出层 ==========
        output = self.backbone.fc_decoder_out(x)
        
        return output
    
    def _preprocess(self, neuralInput: torch.Tensor) -> torch.Tensor:
        """
        复用主干网络的预处理流程
        Gaussian Smoothing -> Day-specific Layer -> Unfold -> Input Projection
        """
        # Gaussian Smoothing
        x = torch.permute(neuralInput, (0, 2, 1))
        x = self.backbone.gaussianSmoother(x)
        x = torch.permute(x, (0, 2, 1))
        
        # Day-specific Input Layer
        x = self.backbone.inputLayerNonlinearity(self.backbone.inpLayer(x))
        
        # Unfold
        stridedInputs = torch.permute(
            self.backbone.unfolder(torch.unsqueeze(torch.permute(x, (0, 2, 1)), 3)),
            (0, 2, 1)
        )
        
        # Input Projection
        x = self.backbone.input_projection(stridedInputs)
        
        # Positional Encoding (如果有)
        if self.backbone.pos_encoding is not None:
            x = self.backbone.pos_encoding(x)
        
        return x
    
    def forward_features(self, neuralInput: torch.Tensor) -> torch.Tensor:
        """
        提取特征（用于分析）
        输出维度: (batch, backbone_hidden_dim)
        """
        x = self._preprocess(neuralInput)
        
        for i in range(self.num_layers):
            with torch.no_grad():
                main_out = self.backbone.conformer_blocks[i](x)
            
            side_input = self.main_to_side(main_out if self.ladder_mode == 'sequential' else x)
            side_out = self.side_blocks[i](side_input)
            side_out = self.side_to_main(side_out)
            
            x = self.fusion(main_out, side_out, layer_idx=i)
        
        # Global Average Pooling
        pooled = torch.mean(x, dim=1)
        return pooled
    
    def get_trainable_parameters(self):
        """获取可训练参数的统计信息"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        side_params = sum(p.numel() for p in self.side_blocks.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        output_params = sum(p.numel() for p in self.backbone.fc_decoder_out.parameters() 
                           if p.requires_grad)
        
        return {
            'trainable': trainable_params,
            'total': total_params,
            'ratio': trainable_params / total_params * 100,
            'side_network': side_params,
            'fusion': fusion_params,
            'output_layer': output_params,
        }
    
    def print_trainable_parameters(self):
        """打印可训练参数信息"""
        stats = self.get_trainable_parameters()
        print("\n" + "="*80)
        print("📊 Ladder Side-Tuning 参数统计")
        print("="*80)
        print(f"总参数量:          {stats['total']:,}")
        print(f"可训练参数:        {stats['trainable']:,}")
        print(f"训练参数占比:      {stats['ratio']:.2f}%")
        print(f"  ├─ 侧边网络:     {stats['side_network']:,}")
        print(f"  ├─ 融合模块:     {stats['fusion']:,}")
        print(f"  └─ 输出层:       {stats['output_layer']:,}")
        print("="*80 + "\n")
    
    def get_fusion_weights(self):
        """获取当前的融合权重（用于分析）"""
        alphas = self.fusion.get_alpha_values()
        if alphas is not None:
            print("\n📊 各层融合权重 (alpha):")
            print("   alpha=1.0 表示完全依赖主干，alpha=0.0 表示完全依赖侧边")
            for i, alpha in enumerate(alphas):
                bar_length = int(alpha * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                print(f"   Layer {i:2d}: {bar} {alpha:.3f}")
            print()
        return alphas
    
    @classmethod
    def from_pretrained(cls, pretrained_model, 
                       side_hidden_dim: int = 128,
                       side_num_heads: int = 4,
                       **kwargs):
        """
        从预训练的 ConformerDecoder 创建 LST 模型
        
        Args:
            pretrained_model: 预训练的 ConformerDecoder 实例
            side_hidden_dim: 侧边网络隐藏层维度（建议为主干的 1/4 到 1/2）
            side_num_heads: 侧边网络注意力头数
            **kwargs: 其他参数传递给 ConformerLST
        
        Returns:
            ConformerLST 实例
        """
        return cls(
            backbone_model=pretrained_model,
            side_hidden_dim=side_hidden_dim,
            side_num_heads=side_num_heads,
            **kwargs
        )


# ============= 训练辅助工具 =============

class LSTTrainingMonitor:
    """
    LST 训练监控器
    用于跟踪融合权重、侧边网络贡献等指标
    """
    
    def __init__(self, model: ConformerLST):
        self.model = model
        self.alpha_history = []
    
    def log_fusion_weights(self, epoch: int):
        """记录融合权重"""
        alphas = self.model.get_fusion_weights()
        if alphas is not None:
            self.alpha_history.append({
                'epoch': epoch,
                'alphas': alphas.copy()
            })
    
    def plot_alpha_evolution(self, save_path: str = None):
        """绘制融合权重随训练的变化"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        if not self.alpha_history:
            print("⚠️  没有记录的融合权重数据")
            return
        
        epochs = [h['epoch'] for h in self.alpha_history]
        alphas = np.array([h['alphas'] for h in self.alpha_history])  # [num_epochs, num_layers]
        
        plt.figure(figsize=(12, 6))
        for layer_idx in range(alphas.shape[1]):
            plt.plot(epochs, alphas[:, layer_idx], label=f'Layer {layer_idx}', alpha=0.7)
        
        plt.xlabel('Epoch')
        plt.ylabel('Alpha (主干贡献)')
        plt.title('融合权重随训练的变化')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ 融合权重图已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_side_contribution(self):
        """分析侧边网络的贡献"""
        if not self.alpha_history:
            return
        
        latest_alphas = self.alpha_history[-1]['alphas']
        avg_alpha = latest_alphas.mean()
        
        print("\n" + "="*80)
        print("📊 侧边网络贡献分析")
        print("="*80)
        print(f"平均主干贡献: {avg_alpha:.3f}")
        print(f"平均侧边贡献: {1 - avg_alpha:.3f}")
        print(f"最依赖主干的层: Layer {latest_alphas.argmax()} (alpha={latest_alphas.max():.3f})")
        print(f"最依赖侧边的层: Layer {latest_alphas.argmin()} (alpha={latest_alphas.min():.3f})")
        print("="*80 + "\n")
