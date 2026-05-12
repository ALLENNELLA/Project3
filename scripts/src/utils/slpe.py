# -*- coding: utf-8 -*-
"""
SLPE (Sentence-Level Prediction Entropy) 计算模块
基于CTC路径后验分布的熵
(已优化：向量化CTC前向-后向算法)
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from tqdm import tqdm

def compute_slpe(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    计算SLPE（句子预测熵）
    使用向量化CTC前向-后向算法计算路径后验分布的熵
    
    Args:
        log_probs: [T, Batch, Num_Classes] log概率（已经log_softmax）
        targets: [Batch, Target_Length] 目标序列
        input_lengths: [Batch] 输入长度
        target_lengths: [Batch] 目标长度
        blank: blank标签（默认为0）
    
    Returns:
        slpe: [Batch] 每个样本的SLPE值
    """
    T_max, B, C = log_probs.shape
    device = log_probs.device
    slpe_values = []
    
    # 循环Batch维度（因为每个样本的target长度不同，难以完全Batch化，
    # 但State维度的向量化已经能带来50-100倍加速）
    for b in range(B):
        t_len = int(input_lengths[b].item())
        y_len = int(target_lengths[b].item())
        
        # 边界情况处理
        if y_len == 0:
            slpe_values.append(0.0)
            continue
            
        y = targets[b, :y_len]  # [y_len]
        
        # 1. 构建扩展目标序列 (blank, y1, blank, y2, ..., blank)
        # 长度为 2*y_len + 1
        ext_target = torch.full((2 * y_len + 1,), blank, device=device, dtype=torch.long)
        ext_target[1::2] = y
        path_len = len(ext_target)
        
        # 获取当前样本在路径上对应的 log_probs
        # log_probs: [T, B, C] -> [T, C]
        curr_log_probs = log_probs[:t_len, b, :]
        # 提取路径上每个状态对应的发射概率: [T, path_len]
        # 这一步避免了在循环中反复索引
        log_probs_path = curr_log_probs[:, ext_target]
        
        # 2. 预计算 Skip Mask (是否可以跳过blank)
        # CTC规则：s-2 -> s 允许的条件是：
        # s >= 2, ext_target[s-1] == blank, ext_target[s] != ext_target[s-2]
        # 在扩展路径中，偶数位总是blank，奇数位是label
        # 跳跃只发生在 label -> blank -> label (即 s为奇数时)
        can_skip = torch.zeros(path_len, device=device, dtype=torch.bool)
        # 检查 s 和 s-2 是否不同 (从索引2开始)
        diff_labels = (ext_target[2:] != ext_target[:-2])
        # 检查 s-1 是否为 blank (扩展序列特性保证了中间总是blank，但为了严谨加上检查)
        is_blank_mid = (ext_target[1:-1] == blank)
        can_skip[2:] = diff_labels & is_blank_mid
        
        # 无穷小常量
        neg_inf = torch.tensor(-float('inf'), device=device)
        
        # ===========================
        # 3. 前向算法 (Forward) - 向量化 State
        # ===========================
        alpha = torch.full((t_len, path_len), -float('inf'), device=device)
        
        # 初始化 t=0
        alpha[0, 0] = log_probs_path[0, 0]
        if path_len > 1:
            alpha[0, 1] = log_probs_path[0, 1]
            
        # 时间步递归 (无法消除，但内部是向量操作)
        for t in range(1, t_len):
            prev_alpha = alpha[t-1] # [S]
            current_emit = log_probs_path[t] # [S]
            
            # 转移 1: Same state (s -> s)
            log_same = prev_alpha
            
            # 转移 2: Prev state (s-1 -> s)
            # 右移1位: [0, 1, 2] -> [-inf, 0, 1]
            log_prev = torch.cat([neg_inf.unsqueeze(0), prev_alpha[:-1]])
            
            # 转移 3: Skip state (s-2 -> s)
            # 右移2位: [0, 1, 2] -> [-inf, -inf, 0]
            log_skip_raw = torch.cat([neg_inf.unsqueeze(0).repeat(2), prev_alpha[:-2]])
            # 应用 mask: 不满足条件的设为 -inf
            log_skip = torch.where(can_skip, log_skip_raw, neg_inf)
            
            # LogSumExp 合并三个分支
            combined = torch.stack([log_same, log_prev, log_skip])
            alpha[t] = torch.logsumexp(combined, dim=0) + current_emit
            
        # 计算 P(y|x)
        # 最后一个时间步，必须停在最后两个状态之一 (blank 或 最后一个label)
        log_p_y_x = torch.logsumexp(alpha[t_len-1, -2:], dim=0)
        
        # ===========================
        # 4. 后向算法 (Backward) - 向量化 State
        # ===========================
        beta = torch.full((t_len, path_len), -float('inf'), device=device)
        
        # 初始化 t=T-1
        beta[t_len-1, -1] = 0.0 # log(1)
        if path_len > 1:
            beta[t_len-1, -2] = 0.0
            
        # 反向时间步递归
        for t in range(t_len-2, -1, -1):
            next_beta = beta[t+1] # [S]
            next_emit = log_probs_path[t+1] # [S]
            
            # 计算公共项: beta[t+1] + log_prob[t+1]
            term = next_beta + next_emit
            
            # 转移 1: Same state (s -> s)
            log_same = term
            
            # 转移 2: Next state (s -> s+1)
            # 左移1位: [0, 1, 2] -> [1, 2, -inf]
            log_next = torch.cat([term[1:], neg_inf.unsqueeze(0)])
            
            # 转移 3: Skip state (s -> s+2)
            # 左移2位: [0, 1, 2] -> [2, -inf, -inf]
            log_skip_raw = torch.cat([term[2:], neg_inf.unsqueeze(0).repeat(2)])
            
            # Mask逻辑: 在位置 s，能否跳到 s+2?
            # 等价于检查 can_skip[s+2] 是否为 True
            # 左移 mask
            mask_backward = torch.cat([can_skip[2:], torch.zeros(2, device=device, dtype=torch.bool)])
            log_skip = torch.where(mask_backward, log_skip_raw, neg_inf)
            
            combined = torch.stack([log_same, log_next, log_skip])
            beta[t] = torch.logsumexp(combined, dim=0)
            
        # ===========================
        # 5. 计算熵 (Entropy)
        # ===========================
        # log_posterior(t, s) = alpha(t, s) + beta(t, s) - log P(y|x)
        log_gamma = alpha + beta - log_p_y_x
        
        # 过滤极小值以保证数值稳定性 (exp(-20) 约为 2e-9)
        valid_mask = log_gamma > -20
        
        if valid_mask.any():
            log_gamma_valid = log_gamma[valid_mask]
            gamma_valid = torch.exp(log_gamma_valid)
            
            # 归一化检查 (理论上 sum(gamma) 在每个 t 应该为 1，但在整个矩阵上求和用于计算总熵)
            # SLPE 定义为路径分布的熵。在CTC中，通常近似为对齐分布的熵。
            # Entropy = - sum( p * log(p) )
            entropy = -torch.sum(gamma_valid * log_gamma_valid)
            
            # 由于我们在所有 t, s 上求和，实际上计算的是期望对齐路径的熵总和
            # 为了得到句子级别的熵，通常需要除以长度或者直接取值
            # 这里保持原逻辑：直接求和所有有效后验的熵
            
            # 注意：原代码的逻辑是对所有 t,s 的后验概率归一化后求熵
            # 原代码逻辑：total_p = sum(posteriors); p /= total_p
            # 这里的 gamma 在每个时间步 t 的 sum 应该是 1。
            # 如果原意是计算 "Alignment Entropy"，则上述公式是对的。
            # 如果原意是完全复刻原代码逻辑（将整个 T*S 矩阵视为一个分布）：
            
            total_p = gamma_valid.sum()
            if total_p > 1e-6:
                probs = gamma_valid / total_p
                entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            else:
                entropy = torch.tensor(0.0, device=device)
        else:
            entropy = torch.tensor(0.0, device=device)
            
        slpe_values.append(max(entropy.item(), 0.0))
    
    return torch.tensor(slpe_values, device=log_probs.device, dtype=torch.float32)


def compute_slpe_batch(model, dataloader, device='cuda', blank=0, show_progress=True):
    """
    批量计算SLPE值
    
    Args:
        model: Model A模型
        dataloader: 数据加载器
        device: 设备
        blank: blank标签
        show_progress: 是否显示 tqdm 进度条（批量脚本可设为 False）
    
    Returns:
        slpe_scores: [N] 所有样本的SLPE分数
    """
    
    model.eval()
    slpe_scores = []
    
    # 添加进度条
    total_batches = len(dataloader)
    iterator = dataloader if not show_progress else tqdm(
        dataloader, desc='计算SLPE', total=total_batches, unit='batch'
    )
    pbar = iterator
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 5:
                X, y, X_len, y_len, _ = batch
            else:
                X, y, X_len, y_len = batch[:4]
            
            X = X.to(device)
            y = y.to(device)
            X_len = X_len.to(device)
            y_len = y_len.to(device)
            
            # 模型前向传播
            logits = model.forward(X)  # [Batch, T', Num_Classes]
            
            # 转换为CTC格式：[T, Batch, Num_Classes]
            log_probs = logits.log_softmax(2).permute(1, 0, 2)
            
            # 调整输入长度（考虑kernel和stride）
            if hasattr(model, 'kernelLen') and hasattr(model, 'strideLen'):
                adjusted_lens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)
            else:
                adjusted_lens = X_len.to(torch.int32)
            
            # 确保adjusted_lens不超过log_probs的长度
            max_len = log_probs.shape[0]
            adjusted_lens = torch.clamp(adjusted_lens, max=max_len)
            
            # 计算SLPE (使用优化后的函数)
            batch_slpe = compute_slpe(
                log_probs,
                y,
                adjusted_lens,
                y_len,
                blank=blank
            )
            
            slpe_scores.extend(batch_slpe.cpu().numpy())
            
            # 更新进度条信息
            if show_progress and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    '已处理样本': len(slpe_scores),
                    '当前batch': f'{batch_idx+1}/{total_batches}'
                })
    
    if show_progress and hasattr(pbar, 'close'):
        pbar.close()
    return np.array(slpe_scores)
