#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
音素序列和文本标签示例演示脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.model_b.utils.phoneme_converter import phoneme_seq_to_text
from model_b_utils import build_prompt

def print_example(name, phoneme_seq, transcription):
    """打印一个完整的示例"""
    print("=" * 80)
    print(f"示例: {name}")
    print("=" * 80)
    
    # 转换为文本格式
    phoneme_text = phoneme_seq_to_text(phoneme_seq, remove_padding=True)
    
    print(f"\n1. 音素序列（ID格式）:")
    print(f"   {phoneme_seq}")
    
    print(f"\n2. 音素序列（文本格式，移除padding后）:")
    print(f"   {phoneme_text}")
    
    print(f"\n3. 文本标签:")
    print(f"   {transcription}")
    
    # 构建prompt
    prompt = build_prompt(phoneme_seq, transcription)
    
    print(f"\n4. 最终Prompt（当前默认总控格式）:")
    print(f"   {prompt}")
    print()

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("音素序列和文本标签示例演示")
    print("=" * 80 + "\n")
    
    # 示例 1: "hello world"
    phoneme_seq_1 = [15, 11, 21, 21, 25, 0, 0, 0, 23, 25, 18, 21, 3, 0, 0, 0]
    transcription_1 = "hello world"
    print_example("hello world", phoneme_seq_1, transcription_1)
    
    # 示例 2: "the quick brown fox"
    phoneme_seq_2 = [31, 10, 0, 20, 20, 6, 1, 9, 0, 2, 18, 25, 23, 0, 14, 25, 20, 29, 0, 0, 0]
    transcription_2 = "the quick brown fox"
    print_example("the quick brown fox", phoneme_seq_2, transcription_2)
    
    # 示例 3: "how are you"
    phoneme_seq_3 = [15, 25, 23, 0, 1, 18, 11, 0, 34, 25, 0, 0, 0, 0]
    transcription_3 = "how are you"
    print_example("how are you", phoneme_seq_3, transcription_3)
    
    print("=" * 80)
    print("演示完成！")
    print("=" * 80)
