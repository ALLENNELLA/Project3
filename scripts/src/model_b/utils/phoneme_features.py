# phoneme_features.py
"""
音素发音特征表
基于语言学的articulatory features
"""

import numpy as np
import torch
import torch.nn as nn

# ARPABET音素列表（39个）
PHONE_DEF = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]

# 添加静音标记
PHONE_DEF_SIL = PHONE_DEF + ['SIL']

# 创建音素到索引的映射（索引从1开始，0保留给padding）
PHONE_TO_IDX = {phone: idx + 1 for idx, phone in enumerate(PHONE_DEF)}
PHONE_TO_IDX['<pad>'] = 0
PHONE_TO_IDX['SIL'] = 40  # 🔧 添加 SIL 的索引映射


class PhonemeFeatureExtractor:
    """
    音素特征提取器
    将音素转换为语言学特征向量
    
    索引映射:
        0: <pad> (填充)
        1-39: PHONE_DEF 中的 39 个音素
        40: SIL (静音)
    """
    
    def __init__(self):
        # 定义特征维度
        self.feature_names = [
            # Type (2)
            'consonant', 'vowel',
            
            # Voicing (1)
            'voiced',
            
            # Place of Articulation (7) - for consonants
            'bilabial', 'labiodental', 'dental', 'alveolar', 
            'palatal', 'velar', 'glottal',
            
            # Manner of Articulation (6) - for consonants
            'stop', 'fricative', 'affricate', 'nasal', 'liquid', 'glide',
            
            # Vowel Height (3)
            'high', 'mid', 'low',
            
            # Vowel Backness (3)
            'front', 'central', 'back',
            
            # Vowel Features (2)
            'rounded', 'tense',
        ]
        
        self.feature_dim = len(self.feature_names)  # 24维
        
        # 构建特征表
        self.feature_table = self._build_feature_table()
        
        print(f"✅ 音素特征提取器初始化完成")
        print(f"   音素数量: {len(PHONE_DEF_SIL)} (包含 SIL)")
        print(f"   特征维度: {self.feature_dim}")
        print(f"   特征表形状: {self.feature_table.shape}")
        print(f"   索引范围: 0-{self.feature_table.shape[0]-1}")
        print(f"   特征名称: {self.feature_names[:5]}... (共{len(self.feature_names)}个)")
    
    def _build_feature_table(self):
        """构建完整的音素特征表"""
        # 🔧 关键修复：确保包含索引 40
        # 索引 0: padding
        # 索引 1-39: 39个音素
        # 索引 40: SIL
        max_idx = 40  # 显式设置最大索引为 40
        table = np.zeros((max_idx + 1, self.feature_dim), dtype=np.float32)
        
        print(f"   分配特征表: {table.shape[0]} 行（索引0-{max_idx}）")
        
        # 辅音特征
        consonants = {
            # Stops (塞音)
            'B':  {'voiced': 1, 'bilabial': 1, 'stop': 1},
            'P':  {'voiced': 0, 'bilabial': 1, 'stop': 1},
            'D':  {'voiced': 1, 'alveolar': 1, 'stop': 1},
            'T':  {'voiced': 0, 'alveolar': 1, 'stop': 1},
            'G':  {'voiced': 1, 'velar': 1, 'stop': 1},
            'K':  {'voiced': 0, 'velar': 1, 'stop': 1},
            
            # Affricates (塞擦音)
            'JH': {'voiced': 1, 'palatal': 1, 'affricate': 1},
            'CH': {'voiced': 0, 'palatal': 1, 'affricate': 1},
            
            # Fricatives (擦音)
            'V':  {'voiced': 1, 'labiodental': 1, 'fricative': 1},
            'F':  {'voiced': 0, 'labiodental': 1, 'fricative': 1},
            'DH': {'voiced': 1, 'dental': 1, 'fricative': 1},
            'TH': {'voiced': 0, 'dental': 1, 'fricative': 1},
            'Z':  {'voiced': 1, 'alveolar': 1, 'fricative': 1},
            'S':  {'voiced': 0, 'alveolar': 1, 'fricative': 1},
            'ZH': {'voiced': 1, 'palatal': 1, 'fricative': 1},
            'SH': {'voiced': 0, 'palatal': 1, 'fricative': 1},
            'HH': {'voiced': 0, 'glottal': 1, 'fricative': 1},
            
            # Nasals (鼻音)
            'M':  {'voiced': 1, 'bilabial': 1, 'nasal': 1},
            'N':  {'voiced': 1, 'alveolar': 1, 'nasal': 1},
            'NG': {'voiced': 1, 'velar': 1, 'nasal': 1},
            
            # Liquids (流音)
            'L':  {'voiced': 1, 'alveolar': 1, 'liquid': 1},
            'R':  {'voiced': 1, 'alveolar': 1, 'liquid': 1},
            
            # Glides (滑音)
            'W':  {'voiced': 1, 'bilabial': 1, 'glide': 1},
            'Y':  {'voiced': 1, 'palatal': 1, 'glide': 1},
        }
        
        # 元音特征
        vowels = {
            # High Front
            'IY': {'high': 1, 'front': 1, 'rounded': 0, 'tense': 1},
            'IH': {'high': 1, 'front': 1, 'rounded': 0, 'tense': 0},
            
            # Mid Front
            'EY': {'mid': 1, 'front': 1, 'rounded': 0, 'tense': 1},
            'EH': {'mid': 1, 'front': 1, 'rounded': 0, 'tense': 0},
            
            # Low Front
            'AE': {'low': 1, 'front': 1, 'rounded': 0, 'tense': 0},
            
            # Low Back
            'AA': {'low': 1, 'back': 1, 'rounded': 0, 'tense': 1},
            'AO': {'low': 1, 'back': 1, 'rounded': 1, 'tense': 1},
            
            # Mid Central
            'AH': {'mid': 1, 'central': 1, 'rounded': 0, 'tense': 0},
            'ER': {'mid': 1, 'central': 1, 'rounded': 0, 'tense': 1},
            
            # High Back
            'UH': {'high': 1, 'back': 1, 'rounded': 1, 'tense': 0},
            'UW': {'high': 1, 'back': 1, 'rounded': 1, 'tense': 1},
            
            # Mid Back
            'OW': {'mid': 1, 'back': 1, 'rounded': 1, 'tense': 1},
            
            # Diphthongs (双元音)
            'AW': {'low': 1, 'back': 1, 'rounded': 0, 'tense': 0},
            'AY': {'low': 1, 'front': 1, 'rounded': 0, 'tense': 0},
            'OY': {'mid': 1, 'back': 1, 'rounded': 1, 'tense': 0},
        }
        
        # 填充特征表
        for phone in PHONE_DEF:
            idx = PHONE_TO_IDX[phone]
            
            if phone in consonants:
                # 辅音
                table[idx, self.feature_names.index('consonant')] = 1
                table[idx, self.feature_names.index('vowel')] = 0
                
                for feat, val in consonants[phone].items():
                    table[idx, self.feature_names.index(feat)] = val
            
            elif phone in vowels:
                # 元音
                table[idx, self.feature_names.index('consonant')] = 0
                table[idx, self.feature_names.index('vowel')] = 1
                table[idx, self.feature_names.index('voiced')] = 1  # 元音都是浊音
                
                for feat, val in vowels[phone].items():
                    table[idx, self.feature_names.index(feat)] = val
        
        # 🔧 关键：为索引 40 (SIL) 设置特征
        # SIL (静音) 的特征：全零向量（表示没有发音特征）
        # 索引 40 已经在初始化时设为全零，所以不需要额外操作
        # 但为了明确，我们可以显式设置
        sil_idx = 40
        table[sil_idx, :] = 0  # 静音：所有特征都为0
        
        # 索引 0 (padding) 也保持全零
        table[0, :] = 0
        
        # 验证特征表
        print(f"   索引 0 (<pad>): 全零向量 = {np.all(table[0] == 0)}")
        print(f"   索引 40 (SIL): 全零向量 = {np.all(table[40] == 0)}")
        print(f"   索引 1-39: 已填充 {np.sum(np.any(table[1:40] != 0, axis=1))} 个音素特征")
        
        return table
    
    def get_features(self, phoneme_indices):
        """
        获取音素的特征向量
        
        Args:
            phoneme_indices: [batch, seq_len] 或 [seq_len]
        
        Returns:
            features: [batch, seq_len, feature_dim] 或 [seq_len, feature_dim]
        """
        if isinstance(phoneme_indices, torch.Tensor):
            phoneme_indices = phoneme_indices.cpu().numpy()
        
        # 检查索引范围
        if np.any(phoneme_indices < 0) or np.any(phoneme_indices >= self.feature_table.shape[0]):
            print(f"⚠️  警告: 音素索引超出范围 [0, {self.feature_table.shape[0]-1}]")
            print(f"   最小索引: {phoneme_indices.min()}")
            print(f"   最大索引: {phoneme_indices.max()}")
            # 裁剪到有效范围
            phoneme_indices = np.clip(phoneme_indices, 0, self.feature_table.shape[0] - 1)
        
        return self.feature_table[phoneme_indices]
    
    def get_feature_matrix(self):
        """获取完整的特征矩阵"""
        return self.feature_table
    
    def get_phoneme_name(self, idx):
        """根据索引获取音素名称"""
        if idx == 0:
            return '<pad>'
        elif idx == 40:
            return 'SIL'
        elif 1 <= idx <= 39:
            return PHONE_DEF[idx - 1]
        else:
            return '<unk>'
    
    def print_phoneme_features(self, phoneme):
        """打印某个音素的特征"""
        if phoneme not in PHONE_TO_IDX:
            print(f"❌ 音素 '{phoneme}' 不存在")
            return
        
        idx = PHONE_TO_IDX[phoneme]
        features = self.feature_table[idx]
        
        print(f"\n音素 '{phoneme}' (索引 {idx}) 的特征:")
        print("-" * 40)
        active_features = []
        for i, (name, val) in enumerate(zip(self.feature_names, features)):
            if val > 0:
                active_features.append(f"{name}: {val:.1f}")
        
        if active_features:
            for feat in active_features:
                print(f"  {feat}")
        else:
            print(f"  (全零向量 - 静音或填充)")
        print("-" * 40)
    
    def validate_indices(self, phoneme_indices):
        """验证音素索引是否在有效范围内"""
        if isinstance(phoneme_indices, torch.Tensor):
            phoneme_indices = phoneme_indices.cpu().numpy()
        
        min_idx = phoneme_indices.min()
        max_idx = phoneme_indices.max()
        unique_indices = np.unique(phoneme_indices)
        
        print(f"\n🔍 音素索引验证:")
        print(f"   索引范围: [{min_idx}, {max_idx}]")
        print(f"   唯一索引数: {len(unique_indices)}")
        print(f"   特征表大小: {self.feature_table.shape[0]} (索引 0-{self.feature_table.shape[0]-1})")
        
        if max_idx >= self.feature_table.shape[0]:
            print(f"   ❌ 错误: 最大索引 {max_idx} 超出特征表范围!")
            return False
        elif min_idx < 0:
            print(f"   ❌ 错误: 最小索引 {min_idx} 小于 0!")
            return False
        else:
            print(f"   ✅ 所有索引都在有效范围内")
            
            # 显示索引分布
            print(f"\n   索引分布:")
            if 0 in unique_indices:
                print(f"     0 (<pad>): {np.sum(phoneme_indices == 0)} 次")
            if 40 in unique_indices:
                print(f"     40 (SIL): {np.sum(phoneme_indices == 40)} 次")
            
            regular_indices = unique_indices[(unique_indices > 0) & (unique_indices < 40)]
            if len(regular_indices) > 0:
                print(f"     1-39 (音素): {len(regular_indices)} 种，共 {np.sum((phoneme_indices > 0) & (phoneme_indices < 40))} 次")
            
            return True