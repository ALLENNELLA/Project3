"""
排序损失函数模块
用于模型B的训练，专注于学习样本的相对难度排序
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginRankingLoss(nn.Module):
    """
    Margin Ranking Loss
    用于学习样本对的相对排序关系
    
    对于一对样本 (easy, hard)，如果 easy 的 CER < hard 的 CER，
    我们希望预测分数也满足：pred_easy < pred_hard + margin
    """
    
    def __init__(self, margin=1.0):
        """
        Args:
            margin: margin值，控制排序对的间隔
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, pred_easy, pred_hard, labels):
        """
        计算margin ranking loss
        
        Args:
            pred_easy: 简单样本的预测分数 [batch_size]
            pred_hard: 困难样本的预测分数 [batch_size]
            labels: 标签（通常为1.0，表示pred_easy应该小于pred_hard）
        
        Returns:
            loss: 平均损失值
        """
        # labels=1表示pred_easy应该小于pred_hard
        # 损失 = max(0, margin - (pred_hard - pred_easy))
        # 如果pred_hard - pred_easy >= margin，损失为0
        # 否则损失 = margin - (pred_hard - pred_easy)
        loss = F.relu(self.margin - (pred_hard - pred_easy))
        return loss.mean()


class RankNetLoss(nn.Module):
    """
    RankNet Loss
    基于概率的排序损失，使用sigmoid函数将分数差转换为概率
    """
    
    def __init__(self, sigma=1.0):
        """
        Args:
            sigma: 缩放参数
        """
        super().__init__()
        self.sigma = sigma
    
    def forward(self, pred_easy, pred_hard, labels):
        """
        计算RankNet loss
        
        Args:
            pred_easy: 简单样本的预测分数 [batch_size]
            pred_hard: 困难样本的预测分数 [batch_size]
            labels: 标签（通常为1.0，表示pred_easy应该小于pred_hard）
        
        Returns:
            loss: 平均损失值
        """
        # 计算分数差
        score_diff = pred_hard - pred_easy
        
        # 使用sigmoid将分数差转换为概率
        # P(i > j) = sigmoid(sigma * (score_i - score_j))
        prob = torch.sigmoid(self.sigma * score_diff)
        
        # 交叉熵损失：-log(P)
        # 如果labels=1，表示pred_easy应该小于pred_hard，即P应该接近1
        loss = -torch.log(prob + 1e-8)
        
        return loss.mean()
