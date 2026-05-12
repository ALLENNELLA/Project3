#GRUDecoder.py
import torch
from torch import nn

from .augmentations import GaussianSmoothing


class GRUDecoder(nn.Module):
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
    ):
        super(GRUDecoder, self).__init__()

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
        self.inputLayerNonlinearity = torch.nn.Softsign()

        # Gaussian smoothing
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )

        # unfold
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )

        # 一个统一的输入层（代替每个 day 的 inpLayerX）
        self.inpLayer = nn.Linear(neural_dim, neural_dim)
        self.inpLayer.weight = nn.Parameter(self.inpLayer.weight + torch.eye(neural_dim))

        # GRU
        self.gru_decoder = nn.GRU(
            neural_dim * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # 输出层
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(hidden_dim * 2, n_classes + 1)
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)

    def forward(self, neuralInput):
        # B, T, D → B, D, T
        x = torch.permute(neuralInput, (0, 2, 1))
        x = self.gaussianSmoother(x)
        x = torch.permute(x, (0, 2, 1))

        # 统一输入层
        x = self.inputLayerNonlinearity(self.inpLayer(x))

        # unfold
        stridedInputs = torch.permute(
            self.unfolder(torch.unsqueeze(torch.permute(x, (0, 2, 1)), 3)),
            (0, 2, 1)
        )

        # initial hidden
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2, x.size(0), self.hidden_dim, device=self.device
            )
        else:
            h0 = torch.zeros(
                self.layer_dim, x.size(0), self.hidden_dim, device=self.device
            )

        hid, _ = self.gru_decoder(stridedInputs, h0)

        seq_out = self.fc_decoder_out(hid)
        return seq_out
    
    def forward_features(self, neuralInput):
        """
        提取倒数第二层的特征 (embedding)
        输出维度: (batch, hidden_dim) 或 (batch, hidden_dim*2) if bidirectional
        """
        # B, T, D → B, D, T
        x = torch.permute(neuralInput, (0, 2, 1))
        x = self.gaussianSmoother(x)
        x = torch.permute(x, (0, 2, 1))

        # 统一输入层
        x = self.inputLayerNonlinearity(self.inpLayer(x))

        # unfold
        stridedInputs = torch.permute(
            self.unfolder(torch.unsqueeze(torch.permute(x, (0, 2, 1)), 3)),
            (0, 2, 1)
        )

        # initial hidden
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2, x.size(0), self.hidden_dim, device=self.device
            )
        else:
            h0 = torch.zeros(
                self.layer_dim, x.size(0), self.hidden_dim, device=self.device
            )

        hid, _ = self.gru_decoder(stridedInputs, h0)
        # hid: (batch, num_windows, hidden_dim*[1 or 2])

        # 全局平均池化，去掉时间维度
        pooled = torch.mean(hid, dim=1)  # (batch, hidden_dim*[1 or 2])

        return pooled

    def forward_features_seq(self, neuralInput):
        """
        提取序列级别的特征（保留时间步）
        输出维度: (batch, num_windows, hidden_dim) 或 (batch, num_windows, hidden_dim*2)
        """
        # B, T, D → B, D, T
        x = torch.permute(neuralInput, (0, 2, 1))
        x = self.gaussianSmoother(x)
        x = torch.permute(x, (0, 2, 1))
    
        # 统一输入层
        x = self.inputLayerNonlinearity(self.inpLayer(x))
    
        # unfold
        stridedInputs = torch.permute(
            self.unfolder(torch.unsqueeze(torch.permute(x, (0, 2, 1)), 3)),
            (0, 2, 1)
        )
    
        # initial hidden
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2, x.size(0), self.hidden_dim, device=self.device
            )
        else:
            h0 = torch.zeros(
                self.layer_dim, x.size(0), self.hidden_dim, device=self.device
            )
    
        hid, _ = self.gru_decoder(stridedInputs, h0)
        # hid: (batch, num_windows, hidden_dim*[1 or 2])
    
        return hid  # 保留时间步维度
    