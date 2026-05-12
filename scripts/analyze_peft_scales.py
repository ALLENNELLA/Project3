#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析 PEFT（CA Block + AdaptFFN）微调后的模型权重，统计 scale (sigma) 参数
"""

import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_model_weights(weight_path: Path):
    """分析单个模型权重文件中的 PEFT 参数"""
    print(f"\n{'='*60}")
    print(f"分析模型: {weight_path}")
    print(f"{'='*60}")
    
    state_dict = torch.load(weight_path, map_location='cpu')
    
    # 收集所有 scale 参数
    ca_block_scales = []
    output_adapter_scales = []
    attn_adapter_scales = []
    
    # 统计结构信息
    ca_block_layers = []
    attn_adapter_layers = []
    has_output_adapter = False
    
    for name, param in state_dict.items():
        # CA Block 的 scale
        if 'ca_block.scale' in name:
            ca_block_scales.append(param.item())
            # 提取层号
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p == 'conformer_blocks':
                    layer_idx = int(parts[i+1])
                    ca_block_layers.append(layer_idx)
                    break
        
        # Output Adapter 的 scale
        elif 'output_adapter.scale' in name:
            output_adapter_scales.append(param.item())
            has_output_adapter = True
        
        # Attention Adapter 的 scale
        elif 'attn_adapter.scale' in name:
            attn_adapter_scales.append(param.item())
            # 提取层号
            parts = name.split('.')
            for i, p in enumerate(parts):
                if p == 'conformer_blocks':
                    layer_idx = int(parts[i+1])
                    attn_adapter_layers.append(layer_idx)
                    break
    
    results = {
        'ca_block': {
            'scales': ca_block_scales,
            'layers': sorted(set(ca_block_layers)),
            'count': len(ca_block_scales)
        },
        'output_adapter': {
            'scales': output_adapter_scales,
            'count': len(output_adapter_scales),
            'exists': has_output_adapter
        },
        'attn_adapter': {
            'scales': attn_adapter_scales,
            'layers': sorted(set(attn_adapter_layers)),
            'count': len(attn_adapter_scales)
        }
    }
    
    return results

def print_results(results, model_name):
    """打印分析结果"""
    print(f"\n【{model_name}】")
    
    # CA Block
    if results['ca_block']['count'] > 0:
        scales = results['ca_block']['scales']
        print(f"\n📦 CA Block (共 {results['ca_block']['count']} 个):")
        print(f"   层位置: {results['ca_block']['layers']}")
        print(f"   Scale 值: {[f'{s:.6f}' for s in scales]}")
        print(f"   平均值: {np.mean(scales):.6f}")
        print(f"   标准差: {np.std(scales):.6f}")
        print(f"   最小值: {np.min(scales):.6f}")
        print(f"   最大值: {np.max(scales):.6f}")
    else:
        print(f"\n📦 CA Block: 未找到")
    
    # Output Adapter
    if results['output_adapter']['exists']:
        scales = results['output_adapter']['scales']
        print(f"\n🔌 Output Adapter (共 {results['output_adapter']['count']} 个):")
        print(f"   Scale 值: {[f'{s:.6f}' for s in scales]}")
        print(f"   平均值: {np.mean(scales):.6f}")
        print(f"   标准差: {np.std(scales):.6f}")
        print(f"   最小值: {np.min(scales):.6f}")
        print(f"   最大值: {np.max(scales):.6f}")
    else:
        print(f"\n🔌 Output Adapter: 未找到")
    
    # Attention Adapter
    if results['attn_adapter']['count'] > 0:
        scales = results['attn_adapter']['scales']
        print(f"\n🎯 Attention Adapter (共 {results['attn_adapter']['count']} 个):")
        print(f"   层位置: {results['attn_adapter']['layers']}")
        print(f"   Scale 值: {[f'{s:.6f}' for s in scales]}")
        print(f"   平均值: {np.mean(scales):.6f}")
        print(f"   标准差: {np.std(scales):.6f}")
        print(f"   最小值: {np.min(scales):.6f}")
        print(f"   最大值: {np.max(scales):.6f}")
    else:
        print(f"\n🎯 Attention Adapter: 未找到")

def main():
    base_dir = Path("/root/25S151115/project3")
    
    # 查找所有 PEFT 模型权重（含 peft_nol2sp、peft_l2sp、schemeA 等）
    weight_files = list((base_dir / "outputs" / "model_test" / "7-8").glob("*peft*/seed*/modelWeights.pth"))
    
    if not weight_files:
        print("❌ 未找到模型权重文件")
        return
    
    print(f"找到 {len(weight_files)} 个模型权重文件")
    
    # 按方法分组
    all_results = {
        'real_slpe': [],
        'model_b': []
    }
    
    for weight_file in sorted(weight_files):
        method = 'real_slpe' if 'real_slpe' in str(weight_file) else 'model_b'
        seed = weight_file.parent.name  # seed0, seed1, etc.
        
        results = analyze_model_weights(weight_file)
        all_results[method].append({
            'seed': seed,
            'results': results
        })
        
        print_results(results, f"{method} - {seed}")
    
    # 汇总统计
    print(f"\n\n{'='*60}")
    print("📊 汇总统计（所有 seeds 的平均值）")
    print(f"{'='*60}")
    
    for method in ['real_slpe', 'model_b']:
        if not all_results[method]:
            continue
        
        print(f"\n【{method.upper()}】")
        
        # 汇总 CA Block
        all_ca_scales = []
        for item in all_results[method]:
            all_ca_scales.extend(item['results']['ca_block']['scales'])
        
        if all_ca_scales:
            print(f"\n📦 CA Block (所有层、所有 seeds):")
            print(f"   平均值: {np.mean(all_ca_scales):.6f}")
            print(f"   标准差: {np.std(all_ca_scales):.6f}")
            print(f"   最小值: {np.min(all_ca_scales):.6f}")
            print(f"   最大值: {np.max(all_ca_scales):.6f}")
            print(f"   样本数: {len(all_ca_scales)}")
        
        # 汇总 Output Adapter
        all_output_scales = []
        for item in all_results[method]:
            all_output_scales.extend(item['results']['output_adapter']['scales'])
        
        if all_output_scales:
            print(f"\n🔌 Output Adapter (所有 seeds):")
            print(f"   平均值: {np.mean(all_output_scales):.6f}")
            print(f"   标准差: {np.std(all_output_scales):.6f}")
            print(f"   最小值: {np.min(all_output_scales):.6f}")
            print(f"   最大值: {np.max(all_output_scales):.6f}")
            print(f"   样本数: {len(all_output_scales)}")
        
        # 汇总 Attention Adapter
        all_attn_scales = []
        for item in all_results[method]:
            all_attn_scales.extend(item['results']['attn_adapter']['scales'])
        
        if all_attn_scales:
            print(f"\n🎯 Attention Adapter (所有层、所有 seeds):")
            print(f"   平均值: {np.mean(all_attn_scales):.6f}")
            print(f"   标准差: {np.std(all_attn_scales):.6f}")
            print(f"   最小值: {np.min(all_attn_scales):.6f}")
            print(f"   最大值: {np.max(all_attn_scales):.6f}")
            print(f"   样本数: {len(all_attn_scales)}")

if __name__ == "__main__":
    main()
