# sample_selection.py - 样本选择模块（支持随机采样和模型b采样）
import os
import pickle
import random
import re
import numpy as np
import torch
from typing import List, Optional
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dataset import SpeechDataset
from ..model_b.models.phoneme_cer_predictor import PhonemeCERPredictor
from ..model_b.models.phoneme_cnn_predictor import PhonemeCNNPredictor
from ..model_b.utils.phoneme_dataset import PhonemeTextDataset
from ..model_b.utils.phoneme_id_dataset import PhonemeIDDataset
from ..model_b.utils.phoneme_features import PhonemeFeatureExtractor
from .slpe import compute_slpe_batch
# 音素ID到文本的映射（与make_dataset.py中的定义一致）
PHONEME_VOCAB = [
    "SIL",  # 0: 静音/padding
    "AA", "AE", "AH", "AO", "AW",  # 1-5
    "AY", "B", "CH", "D", "DH",    # 6-10
    "EH", "ER", "EY", "F", "G",    # 11-15
    "HH", "IH", "IY", "JH", "K",   # 16-20
    "L", "M", "N", "NG", "OW",     # 21-25
    "OY", "P", "R", "S", "SH",     # 26-30
    "T", "TH", "UH", "UW", "V",    # 31-35
    "W", "Y", "Z", "ZH"            # 36-39
]

def _phoneme_seq_to_text(phoneme_indices, phone_len=None, remove_padding=True):
    """
    将音素ID序列转换为文本格式
    
    注意：此函数与训练Model B时使用的格式完全一致（remove_padding=True时移除所有0）
    
    Args:
        phoneme_indices: 音素ID序列（numpy array或list）
        phone_len: 音素序列长度（如果提供，只取前phone_len个，然后再移除0）
        remove_padding: 是否移除所有padding（索引0），默认为True，与训练时一致
    
    Returns:
        str: 音素文本序列，如 "AA AE AH"（移除了所有0）
    """
    # 转换为numpy array
    if isinstance(phoneme_indices, (list, tuple)):
        phoneme_indices = np.array(phoneme_indices)
    
    # 如果提供了长度，先只取前phone_len个
    if phone_len is not None:
        phoneme_indices = phoneme_indices[:phone_len]
    
    # 移除所有padding（索引0），与训练时使用的remove_padding=True行为一致
    # 这是关键：训练时使用remove_padding=True，会移除所有0，包括序列中间的0
    if remove_padding:
        phoneme_indices = phoneme_indices[phoneme_indices != 0]
    
    # 确保索引在有效范围内
    valid_indices = phoneme_indices[(phoneme_indices >= 0) & (phoneme_indices < len(PHONEME_VOCAB))]
    
    # 转换为文本
    phonemes = [PHONEME_VOCAB[int(idx)] for idx in valid_indices]
    return " ".join(phonemes)

# 延迟导入peft，避免版本兼容性问题
# 确保在导入peft之前transformers已经被导入
_peft_available = None
def _check_peft_availability():
    """检查peft是否可用"""
    global _peft_available
    if _peft_available is None:
        try:
            # 确保transformers先被导入
            import transformers
            import peft
            _peft_available = True
        except ImportError:
            _peft_available = False
    return _peft_available


def random_sample_selection(dataset_path: str, num_samples: int, seed: int = 0, save_dir: Optional[str] = None) -> tuple[List[int], None]:
    """
    随机选择样本
    
    Args:
        dataset_path: 数据集路径
        num_samples: 需要选择的样本数量
        seed: 随机种子
    
    Returns:
        选中的样本索引列表
    """
    with open(dataset_path, "rb") as handle:
        loaded_data = pickle.load(handle)
    
    # 计算总样本数
    total_samples = sum([len(d["sentenceDat"]) for d in loaded_data["train"]])
    
    random.seed(seed)
    full_indices = list(range(total_samples))
    num_samples = min(num_samples, total_samples)
    sampled_indices = random.sample(full_indices, num_samples)
    
    # 保存选择结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        result = {
            'method': 'random',
            'dataset_path': dataset_path,
            'num_samples': num_samples,
            'seed': seed,
            'selected_indices': sampled_indices
        }
        result_path = os.path.join(save_dir, f'random_selection_{num_samples}_seed{seed}.pkl')
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"✅ Selection results saved to: {result_path}\n")
    
    print(f"✅ Random sampling: selected {len(sampled_indices)} samples from {total_samples} total samples")
    return sampled_indices, None


def length_sample_selection(
    dataset_path: str,
    num_samples: int,
    selection_strategy: str = 'hard',
    save_dir: Optional[str] = None
) -> tuple[List[int], None]:
    """
    按音素数量选择样本。

    Args:
        dataset_path: 数据集路径
        num_samples: 需要选择的样本数量
        selection_strategy: 'hard'=音素最多，'easy'=音素最少
        save_dir: 可选保存目录
    """
    with open(dataset_path, "rb") as handle:
        loaded_data = pickle.load(handle)

    phone_lengths = []
    for day_data in loaded_data["train"]:
        if "phoneLens" in day_data and day_data["phoneLens"] is not None:
            day_phone_lens = day_data["phoneLens"]
            if hasattr(day_phone_lens, "tolist"):
                day_phone_lens = day_phone_lens.tolist()
            phone_lengths.extend([int(x) for x in day_phone_lens])
        elif "phonemes" in day_data and day_data["phonemes"] is not None:
            for phoneme_seq in day_data["phonemes"]:
                seq = np.array(phoneme_seq)
                phone_lengths.append(int(np.sum(seq != 0)))
        else:
            raise ValueError("数据中缺少 phoneLens/phonemes 字段，无法执行 length 选样")

    if len(phone_lengths) == 0:
        raise ValueError("训练集为空，无法执行 length 选样")

    length_scores = np.array(phone_lengths, dtype=np.int64)
    selected_indices = _select_samples_by_strategy(
        scores=length_scores,
        num_samples=num_samples,
        strategy=selection_strategy
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        result = {
            'method': 'length',
            'dataset_path': dataset_path,
            'num_samples': num_samples,
            'selection_strategy': selection_strategy,
            'selected_indices': selected_indices,
            'phone_lengths': length_scores.tolist()
        }
        result_path = os.path.join(save_dir, f'length_selection_{selection_strategy}_{num_samples}.pkl')
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"✅ Selection results saved to: {result_path}\n")

    selected_lengths = length_scores[selected_indices] if len(selected_indices) > 0 else np.array([], dtype=np.int64)
    min_len = int(selected_lengths.min()) if selected_lengths.size > 0 else 0
    max_len = int(selected_lengths.max()) if selected_lengths.size > 0 else 0
    print(f"✅ Length sampling: selected {len(selected_indices)} samples, phoneme count range [{min_len}, {max_len}]")
    return selected_indices, None


# 保留_text_to_phoneme_text函数作为备用（如果transcriptions需要转换）
def _text_to_phoneme_text(text):
    """将文本转换为音素文本（使用g2p）- 备用函数，现在主要使用标签中的音素序列"""
    try:
        from g2p_en import G2p
        g2p = G2p()
        # 清理文本
        clean_text = re.sub(r'[^a-zA-Z\- \']', '', str(text).strip())
        clean_text = clean_text.replace('--', '').lower()
        
        phonemes = []
        add_inter_word_symbol = True
        
        for p in g2p(clean_text):
            if add_inter_word_symbol and p == ' ':
                phonemes.append('SIL')
            p = re.sub(r'[0-9]', '', p)  # 移除重音标记
            if re.match(r'[A-Z]+', p):  # 只保留音素
                phonemes.append(p)
        
        # 在末尾添加一个SIL符号
        if add_inter_word_symbol:
            phonemes.append('SIL')
        
        return " ".join(phonemes)
    except ImportError:
        raise ImportError("g2p_en is required for text-to-phoneme conversion, but we should use phoneme sequences from labels instead")


def _load_model_b(model_b_path, device='cuda'):
    """加载训练好的模型b（支持RoBERTa、CNN和GPT2，包括LoRA格式）"""
    print(f"📦 Loading Model B: {model_b_path}")
    
    # 检查是否是LoRA格式（存在adapter_config.json和head.pt）
    adapter_config_path = os.path.join(model_b_path, 'adapter_config.json')
    head_path = os.path.join(model_b_path, 'head.pt')
    checkpoint_path = os.path.join(model_b_path, 'best_model.pt')
    
    is_lora = os.path.exists(adapter_config_path) and os.path.exists(head_path)
    
    if not is_lora and not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model B checkpoint not found: {checkpoint_path}")
    
    # 检测模型类型
    model_type = 'roberta'  # 默认是RoBERTa
    model_name = 'roberta-base'  # 默认模型名
    modelb_args = {}
    prompt_format = None
    
    # 尝试加载results.pkl获取模型配置信息
    results_path = os.path.join(model_b_path, 'results.pkl')
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        if isinstance(results, dict):
            # 从results中获取模型信息
            if 'model_type' in results:
                model_type = results.get('model_type', 'roberta')
            if 'model_name' in results:
                model_name = results.get('model_name', 'roberta-base')
            if 'prompt_format' in results:
                prompt_format = results.get('prompt_format')
            if 'args' in results:
                modelb_args = results['args']
                if isinstance(modelb_args, dict):
                    if 'model_type' in modelb_args:
                        model_type = modelb_args.get('model_type', 'roberta')
                    if 'model_name' in modelb_args:
                        model_name = modelb_args.get('model_name', 'roberta-base')
                    if prompt_format is None and 'prompt_format' in modelb_args:
                        prompt_format = modelb_args.get('prompt_format')
    
    # 尝试加载results.json（CNN使用）
    results_json_path = os.path.join(model_b_path, 'results.json')
    if os.path.exists(results_json_path):
        import json
        with open(results_json_path, 'r') as f:
            results = json.load(f)
        if isinstance(results, dict) and 'args' in results:
            modelb_args = results['args']
            model_type = modelb_args.get('model_type', 'roberta')
    
    # 如果未找到results.pkl，尝试args.pkl
    if not os.path.exists(results_path):
            args_path = os.path.join(model_b_path, 'args.pkl')
            if os.path.exists(args_path):
                with open(args_path, 'rb') as f:
                    modelb_args = pickle.load(f)
                    # 从args中获取model_type（如果存在）
                    if isinstance(modelb_args, dict) and 'model_type' in modelb_args:
                        model_type = modelb_args.get('model_type', 'roberta')
                if isinstance(modelb_args, dict) and 'model_name' in modelb_args:
                    model_name = modelb_args.get('model_name', 'roberta-base')
    
    # 如果是LoRA格式，使用load_trained_model_b加载
    if is_lora:
        print(f"   检测到LoRA格式，使用load_trained_model_b加载")
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
        from model_b_train_module import load_trained_model_b
        
        model = load_trained_model_b(
            model_path=model_b_path,
            model_name=model_name,
            model_type=model_type,
            device=device
        )
        
        # 获取tokenizer
        if model_type == 'roberta':
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif model_type == 'gpt2':
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
        elif model_type == 'canine':
            from transformers import CanineTokenizer
            tokenizer = CanineTokenizer.from_pretrained(model_name)
        else:
            tokenizer = None
        
        print(f"✅ Model B (LoRA, {model_type}) loaded successfully\n")
        return model, model_type, tokenizer, prompt_format
    
    # 根据模型类型创建模型
    if model_type == 'cnn' or 'cnn' in model_b_path.lower():
        print(f"   Detected CNN model")
        # CNN模型参数
        n_phonemes = modelb_args.get('n_phonemes', 41)
        embedding_dim = modelb_args.get('embedding_dim', 64)
        cnn_num_filters = modelb_args.get('cnn_num_filters', 128)
        cnn_kernel_sizes = modelb_args.get('cnn_kernel_sizes', [3, 5, 7])
        cnn_mlp_dims = modelb_args.get('cnn_mlp_dims', [128, 64])
        dropout = modelb_args.get('dropout', 0.3)
        feature_mode = modelb_args.get('feature_mode', 'embedding')
        
        # 初始化特征提取器（如果需要）
        feature_extractor = None
        if feature_mode in ['features', 'hybrid']:
            feature_extractor = PhonemeFeatureExtractor()
        
        # 创建CNN模型
        model = PhonemeCNNPredictor(
            n_phonemes=n_phonemes,
            embedding_dim=embedding_dim,
            num_filters=cnn_num_filters,
            kernel_sizes=cnn_kernel_sizes,
            dropout=dropout,
            mlp_dims=cnn_mlp_dims,
            feature_mode=feature_mode,
            feature_extractor=feature_extractor
        )
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        
        print(f"✅ Model B (CNN) loaded successfully\n")
        return model, 'cnn', None, prompt_format  # CNN不需要tokenizer
    
    else:
        # LLM模型（RoBERTa、GPT2或CANINE）
        # 从args中获取model_type，如果未指定则根据model_name推断
        if isinstance(modelb_args, dict):
            model_type_from_args = modelb_args.get('model_type', model_type)
            if model_type_from_args and model_type_from_args != 'roberta':
                model_type = model_type_from_args
            else:
                # 如果没有指定model_type，根据model_name推断
                model_name_from_args = modelb_args.get('model_name', 'roberta-base')
                if 'gpt2' in model_name_from_args.lower():
                    model_type = 'gpt2'
                elif 'canine' in model_name_from_args.lower():
                    model_type = 'canine'
        
        if model_type == 'gpt2':
            print(f"   Detected GPT2 model")
            model_name = 'gpt2'
        elif model_type == 'canine':
            print(f"   Detected CANINE model")
            model_name = 'google/canine-s'
        else:
            print(f"   Detected RoBERTa model")
            model_name = 'roberta-base'
        
        use_lora = False
        lora_r = 16
        lora_alpha = 32
        lora_dropout = 0.1
        
        if isinstance(modelb_args, dict):
            model_name = modelb_args.get('model_name', model_name)
            use_lora = modelb_args.get('use_lora', False)
            lora_r = modelb_args.get('lora_r', 16)
            lora_alpha = modelb_args.get('lora_alpha', 32)
            lora_dropout = modelb_args.get('lora_dropout', 0.1)
        
        # 创建模型，传入model_type参数
        model = PhonemeCERPredictor(
            model_name=model_name,
            model_type=model_type,
            dropout=0.1,
            hidden_dim=256
        )
        
        # 如果使用了LoRA，需要先应用LoRA
        if use_lora:
            print(f"   Detected LoRA, applying LoRA configuration...")
            
            # 检查peft是否可用
            if not _check_peft_availability():
                error_msg = (
                    f"❌ Error: peft library is not available.\n"
                    f"   LoRA requires the peft library to be installed.\n\n"
                    f"   Install it with:\n"
                    f"   pip install peft\n"
                )
                raise ImportError(error_msg)
            
            try:
                # 延迟导入，确保transformers先被导入
                import transformers  # 确保transformers先导入
                import importlib
                peft_module = importlib.import_module('peft')
                LoraConfig = peft_module.LoraConfig
                get_peft_model = peft_module.get_peft_model
                TaskType = peft_module.TaskType
            except ImportError as e:
                error_msg = (
                    f"❌ Error: Failed to import peft library.\n"
                    f"   This is likely due to version incompatibility between peft and transformers.\n"
                    f"   Error details: {e}\n\n"
                    f"   Solutions:\n"
                    f"   1. Update transformers: pip install --upgrade transformers\n"
                    f"   2. Or update peft: pip install --upgrade peft\n"
                    f"   3. Or install compatible versions:\n"
                    f"      pip install transformers>=4.30.0 peft>=0.3.0\n"
                )
                raise ImportError(error_msg) from e
            
            try:
                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=["query", "value", "key"],
                )
                model.bert = get_peft_model(model.bert, lora_config)
            except Exception as e:
                error_msg = (
                    f"❌ Error: Failed to apply LoRA configuration.\n"
                    f"   Error details: {e}\n\n"
                    f"   This might be due to:\n"
                    f"   1. Version incompatibility between peft and transformers\n"
                    f"   2. Model structure mismatch\n\n"
                    f"   Try updating libraries:\n"
                    f"   pip install --upgrade transformers peft\n"
                )
                raise RuntimeError(error_msg) from e
        
        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 如果使用了LoRA，需要处理键名映射
        if use_lora:
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('bert.base_model.model.'):
                    new_key = key.replace('bert.base_model.model.', 'bert.')
                    new_state_dict[new_key] = value
                elif key.startswith('bert.base_model.') and 'lora' in key:
                    new_state_dict[key] = value
                elif not key.startswith('bert.'):
                    new_state_dict[key] = value
                else:
                    new_state_dict[key] = value
            
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        
        model.to(device)
        model.eval()
        
        # 获取tokenizer
        if model_type == 'gpt2':
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model_type_display = model_type.upper() if model_type != 'roberta' else 'RoBERTa'
        print(f"✅ Model B ({model_type_display}) loaded successfully (model: {model_name}, LoRA: {use_lora})\n")
        return model, model_type, tokenizer, prompt_format


def _predict_difficulty_scores(model_b, model_type_or_name, tokenizer, dataset, batch_size=32, device='cuda', prompt_format=None):
    """使用模型b预测所有样本的难度分数（支持RoBERTa和CNN）"""
    model_b.eval()
    
    # 判断模型类型
    is_cnn = (model_type_or_name == 'cnn' or 'cnn' in str(model_type_or_name).lower())
    
    if is_cnn:
        # CNN模型：直接使用音素ID序列
        print("🔄 Extracting phoneme ID sequences (for CNN)...")
        phoneme_seqs = []
        
        for i in range(len(dataset)):
            # 获取音素序列（整数ID序列）
            phoneme_seq = dataset.get_phoneme_seq(i)
            phone_len = dataset.get_phoneme_len(i)
            
            if phoneme_seq is None:
                print(f"⚠️ Warning: No phoneme sequence for sample {i}, skipping...")
                continue
            
            # 转换为numpy array并移除padding
            if isinstance(phoneme_seq, (list, tuple)):
                phoneme_seq = np.array(phoneme_seq, dtype=np.int64)
            else:
                phoneme_seq = phoneme_seq.astype(np.int64)
            
            # 如果提供了长度，只取前phone_len个
            if phone_len is not None:
                phoneme_seq = phoneme_seq[:phone_len]
            
            # 移除所有padding（索引0），与训练时一致
            phoneme_seq = phoneme_seq[phoneme_seq != 0]
            
            phoneme_seqs.append(phoneme_seq)
        
        if len(phoneme_seqs) == 0:
            raise ValueError("No valid phoneme sequences found in dataset!")
        
        # 创建数据集
        dummy_scores = np.zeros(len(phoneme_seqs))
        pred_dataset = PhonemeIDDataset(phoneme_seqs, dummy_scores)
        
        pred_loader = DataLoader(
            pred_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # 预测
        print("🔮 Predicting difficulty scores using Model B (CNN)...")
        difficulty_scores = []
        
        with torch.no_grad():
            for seqs, _, lengths in tqdm(pred_loader, desc='Predicting'):
                seqs = seqs.to(device)
                predictions = model_b(seqs, lengths)
                difficulty_scores.extend(predictions.cpu().numpy())
        
        difficulty_scores = np.array(difficulty_scores)
    
    else:
        # LLM模型（RoBERTa、GPT2或CANINE）：转换为文本
        model_type_name = model_type_or_name if isinstance(model_type_or_name, str) and model_type_or_name != 'cnn' else 'RoBERTa'
        if model_type_or_name == 'gpt2':
            model_type_name = 'GPT2'
        elif model_type_or_name == 'canine':
            model_type_name = 'CANINE'
        
        print(f"🔄 Converting phoneme sequences to text and building prompts...")
        # 导入build_prompt函数
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
        from model_b_utils import build_prompt
        
        phoneme_texts = []
        
        for i in range(len(dataset)):
            # 获取音素序列（整数ID序列）
            phoneme_seq = dataset.get_phoneme_seq(i)
            phone_len = dataset.get_phoneme_len(i)
            
            if phoneme_seq is None:
                print(f"⚠️ Warning: No phoneme sequence for sample {i}, skipping...")
                continue
            
            # 将音素ID序列转换为文本
            phoneme_text = _phoneme_seq_to_text(phoneme_seq, phone_len)
            
            # 尝试获取transcription（如果有）
            transcription = None
            if hasattr(dataset, 'get_transcription'):
                transcription = dataset.get_transcription(i)
            elif hasattr(dataset, 'transcriptions') and i < len(dataset.transcriptions):
                transcription = dataset.transcriptions[i]
            
            # 使用统一格式构建prompt（与训练时保持一致）
            fmt = prompt_format
            prompt_text = build_prompt(
                phoneme_seq, 
                transcription,
                prompt_format=fmt
            )
            phoneme_texts.append(prompt_text)
        
        if len(phoneme_texts) == 0:
            raise ValueError("No valid phoneme sequences found in dataset!")
        
        # 创建数据集
        dummy_scores = np.zeros(len(phoneme_texts))
        pred_dataset = PhonemeTextDataset(
            phoneme_texts, dummy_scores, tokenizer, max_length=512
        )
        
        pred_loader = DataLoader(
            pred_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        # 预测
        print(f"🔮 Predicting difficulty scores using Model B ({model_type_name})...")
        difficulty_scores = []
        
        # 检查是否为GPT2模型
        is_gpt2 = (model_type_or_name == 'gpt2')
        
        with torch.no_grad():
            for batch in tqdm(pred_loader, desc='Predicting'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
                outputs = model_b.bert(**inputs)
                
                # 根据模型类型选择表示向量
                if is_gpt2:
                    # GPT-2使用最后一个非padding token的表示
                    last_hidden_state = outputs.last_hidden_state
                    seq_lengths = attention_mask.sum(dim=1) - 1  # -1因为索引从0开始
                    batch_size = last_hidden_state.size(0)
                    embedding = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), seq_lengths]
                else:
                    # RoBERTa和CANINE使用第一个token（[CLS]）的表示
                    embedding = outputs.last_hidden_state[:, 0, :]
                
                predictions = model_b.regressor(embedding).squeeze(-1)
                difficulty_scores.extend(predictions.cpu().numpy())
        
        difficulty_scores = np.array(difficulty_scores)
    print(f"✅ Prediction completed, score range: [{difficulty_scores.min():.4f}, {difficulty_scores.max():.4f}]\n")
    
    return difficulty_scores


def _select_samples_by_strategy(scores, num_samples, strategy='hard', seed: int = 0):
    """根据难度分数选择样本"""
    if num_samples > len(scores):
        print(f"⚠️ Warning: Requested {num_samples} samples but only {len(scores)} available, selecting all")
        num_samples = len(scores)

    strategy = strategy or 'hard'
    ran_alias = re.fullmatch(r"ran_?(\d+)_(\d+)_(\d+)", strategy)
    if ran_alias:
        strategy = f"ran_{ran_alias.group(1)}_{ran_alias.group(2)}_{ran_alias.group(3)}"
    if strategy == 'hard_top100':
        strategy = 'hard'
    elif strategy == 'down100':
        if num_samples != 100:
            raise ValueError(f"Strategy '{strategy}' requires num_samples=100, but got {num_samples}")
        strategy = 'easy'

    ran_match = re.fullmatch(r"ran_(\d+)_(\d+)_(\d+)", strategy)
    if ran_match:
        if num_samples != 100:
            raise ValueError(f"Strategy '{strategy}' requires num_samples=100, but got {num_samples}")
        hard_n, mid_n, easy_n = [int(x) for x in ran_match.groups()]
        if hard_n + mid_n + easy_n != 100:
            raise ValueError(
                f"Invalid strategy '{strategy}': x+y+z must equal 100, got {hard_n + mid_n + easy_n}"
            )

        sorted_indices = np.argsort(scores)[::-1]
        hard_pool, mid_pool, easy_pool = np.array_split(sorted_indices, 3)
        if hard_n > len(hard_pool) or mid_n > len(mid_pool) or easy_n > len(easy_pool):
            raise ValueError(
                f"Strategy '{strategy}' requests more samples than bucket size "
                f"(hard {hard_n}/{len(hard_pool)}, mid {mid_n}/{len(mid_pool)}, easy {easy_n}/{len(easy_pool)})"
            )

        rng = random.Random(seed)
        hard_selected = rng.sample(hard_pool.tolist(), hard_n)
        mid_selected = rng.sample(mid_pool.tolist(), mid_n)
        easy_selected = rng.sample(easy_pool.tolist(), easy_n)
        selected_indices = hard_selected + mid_selected + easy_selected
    elif strategy == 'hard':
        # 选择分数最高的（最困难的）
        selected_indices = np.argsort(scores)[-num_samples:][::-1]
    elif strategy == 'easy':
        # 选择分数最低的（最简单的）
        selected_indices = np.argsort(scores)[:num_samples]
    elif strategy == 'diverse':
        # 多样性采样：选择分数分布均匀的样本
        sorted_indices = np.argsort(scores)
        step = len(sorted_indices) / num_samples
        selected_indices = [sorted_indices[int(i * step)] for i in range(num_samples)]
    else:
        raise ValueError(
            f"Unknown strategy: {strategy}. Choose 'hard_top100', 'down100', 'hard', 'easy', 'diverse', or 'ran_x_y_z'"
        )
    
    print(f"✅ Selected {len(selected_indices)} {strategy} samples")
    print(f"   Score range: [{scores[selected_indices].min():.4f}, {scores[selected_indices].max():.4f}]\n")
    
    return selected_indices.tolist() if isinstance(selected_indices, np.ndarray) else selected_indices


def model_b_sample_selection(
    dataset_path: str, 
    num_samples: int,
    model_b_path: Optional[str] = None,
    selection_strategy: str = 'hard',
    seed: int = 0,
    batch_size: int = 32,
    device: str = 'cuda',
    model_a_path: Optional[str] = None,
    save_dir: Optional[str] = None,
    auto_train: bool = False,
    pretrained_ndays: int = 7,
    base_dir: str = '/root/25S151115/project3'
) -> tuple[List[int], Optional[dict]]:
    """
    使用模型b选择样本
    
    Args:
        dataset_path: 数据集路径
        num_samples: 需要选择的样本数量
        model_b_path: 模型b的路径（如果为None，则回退到随机采样或自动训练）
        selection_strategy: 选择策略 ('hard', 'easy', 'diverse')
        seed: 随机种子（注意：对于model_b方法，如果使用同一个训练好的模型，seed不影响结果）
        batch_size: 预测时的batch大小
        device: 设备
        auto_train: 如果model_b_path不存在，是否自动训练（默认False）
        pretrained_ndays: 预训练天数（用于自动训练）
        base_dir: 基础目录（用于自动训练）
    
    Returns:
        选中的样本索引列表
    """
    # 检查model_b_path
    if model_b_path is None or not os.path.exists(model_b_path):
        if auto_train:
            print("\n" + "="*80)
            print("🔧 Model B not found. Starting automatic training...")
            print("="*80)
            
            # 自动训练model_b
            model_b_path = os.path.join(base_dir, 'outputs', 'model_b')
            best_model_path = os.path.join(model_b_path, 'best_model.pt')
            
            if not os.path.exists(best_model_path):
                print(f"Training Model B to: {model_b_path}")
                import subprocess
                train_script = os.path.join(base_dir, 'scripts', 'train_model_b.py')
                cmd = [
                    'python', train_script,
                    '--n_days', str(pretrained_ndays),
                    '--output_dir', model_b_path,
                    '--base_dir', base_dir
                ]
                try:
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print(result.stdout)
                    if result.returncode != 0:
                        print(f"❌ Model B training failed. Falling back to random sampling.")
                        return random_sample_selection(dataset_path, num_samples, seed, save_dir)
                except Exception as e:
                    print(f"❌ Error during Model B training: {e}")
                    print("   Falling back to random sampling.")
                    return random_sample_selection(dataset_path, num_samples, seed, save_dir)
            else:
                print(f"✅ Found existing Model B at: {model_b_path}")
        else:
            print("⚠️ Model B path not provided or not found. Falling back to random sampling.")
            print("   Tip: Set auto_train=True to automatically train Model B, or provide model_b_path.")
            return random_sample_selection(dataset_path, num_samples, seed, save_dir)
    
    print("\n" + "="*80)
    print("🚀 Using Model B for sample selection")
    print("="*80)
    print(f"Model B path: {model_b_path}")
    print(f"Dataset path: {dataset_path}")
    print(f"Number of samples: {num_samples}")
    print(f"Selection strategy: {selection_strategy}")
    print("="*80 + "\n")
    
    # 加载模型b
    model_b, model_type_or_name, tokenizer, prompt_format = _load_model_b(model_b_path, device)
    
    # 加载数据集
    with open(dataset_path, "rb") as handle:
        loaded_data = pickle.load(handle)
    
    train_dataset = SpeechDataset(loaded_data['train'])
    print(f"✅ Dataset loaded successfully, {len(train_dataset)} training samples\n")
    
    # 预测难度分数
    difficulty_scores = _predict_difficulty_scores(
        model_b, model_type_or_name, tokenizer, train_dataset, 
        batch_size=batch_size, device=device, prompt_format=prompt_format
    )
    
    # 选择样本
    selected_indices = _select_samples_by_strategy(
        difficulty_scores, num_samples, selection_strategy, seed=seed
    )
    
    # 计算与真实SLPE前100句的重合率（如果提供了model_a_path）
    if model_a_path is not None and os.path.exists(model_a_path):
        try:
            print("\n📊 计算Model B选择的样本与真实SLPE前100句的重合率...")
            # 计算真实SLPE分数
            real_slpe_scores = compute_real_slpe_scores(
                dataset_path,
                model_a_path,
                batch_size=batch_size,
                device=device
            )
            
            # 获取真实SLPE前100句的索引（按SLPE分数从高到低排序）
            real_slpe_top100_indices = np.argsort(real_slpe_scores)[-100:][::-1]
            
            # 获取Model B选择的前100句索引
            model_b_top100_indices = set(selected_indices[:min(100, len(selected_indices))])
            real_slpe_top100_set = set(real_slpe_top100_indices)
            
            # 计算重合率
            overlap_count = len(model_b_top100_indices & real_slpe_top100_set)
            overlap_ratio = overlap_count / 100.0 if len(real_slpe_top100_indices) > 0 else 0.0
            
            print(f"✅ Model B选择的100句与真实SLPE前100句的重合率: {overlap_count}/100 = {overlap_ratio:.2%}")
            print()
        except Exception as e:
            print(f"⚠️ 计算真实SLPE重合率时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 计算与真实CER的重合度（如果不是diverse策略且提供了model_a_path）
    overlap_info = None
    if selection_strategy != 'diverse' and model_a_path is not None and os.path.exists(model_a_path):
        try:
            print("\n📊 Computing overlap with real CER selection...")
            real_cer_scores = compute_real_cer_scores(
                dataset_path, 
                model_a_path,
                batch_size=batch_size,
                device=device
            )
            
            # 检查Model B是否使用了rank归一化
            # 如果Model B训练时使用了rank归一化，预测分数应该是归一化后的
            # 为了公平比较，我们也对真实CER进行rank归一化
            # 注意：对于overlap计算，我们只需要排序，所以rank归一化不影响结果
            # 但为了保持一致性，我们仍然进行归一化
            from scipy.stats import rankdata
            ranked_cer = rankdata(real_cer_scores, method='average')
            normalized_cer = (ranked_cer - 1) / (len(ranked_cer) - 1) if len(ranked_cer) > 1 else ranked_cer
            
            # 使用归一化后的CER进行选择（排序不变，所以结果相同）
            # 但这样更符合Model B训练时的处理方式
            real_cer_indices = _select_samples_by_strategy(
                normalized_cer, num_samples, selection_strategy
            )
            
            # 计算重合度
            selected_set = set(selected_indices)
            real_cer_set = set(real_cer_indices)
            overlap = len(selected_set & real_cer_set)
            overlap_ratio = overlap / num_samples if num_samples > 0 else 0.0
            
            # 计算前n句的重合度（n = 10, 20, 50, 100等）
            top_n_overlaps = {}
            top_n_values = [10, 20, 50, 100, 200]
            for n in top_n_values:
                if n <= num_samples:
                    # 取前n句
                    selected_top_n = set(selected_indices[:n])
                    real_cer_top_n = set(real_cer_indices[:n] if isinstance(real_cer_indices, list) else real_cer_indices[:n])
                    overlap_n = len(selected_top_n & real_cer_top_n)
                    overlap_ratio_n = overlap_n / n if n > 0 else 0.0
                    top_n_overlaps[n] = {
                        'overlap_count': overlap_n,
                        'overlap_ratio': overlap_ratio_n
                    }
            
            overlap_info = {
                'overlap_count': overlap,
                'overlap_ratio': overlap_ratio,
                'selected_indices': selected_indices,
                'real_cer_indices': real_cer_indices.tolist() if isinstance(real_cer_indices, np.ndarray) else real_cer_indices,
                'top_n_overlaps': top_n_overlaps
            }
            
            # 打印重合度信息
            print(f"✅ Overlap with real CER: {overlap}/{num_samples} ({overlap_ratio:.2%})")
            print(f"\n📊 Top-N Overlap Analysis:")
            for n in sorted(top_n_overlaps.keys()):
                info = top_n_overlaps[n]
                print(f"   Top-{n:3d}: {info['overlap_count']:3d}/{n} ({info['overlap_ratio']:.2%})")
            print()
        except Exception as e:
            print(f"⚠️ Warning: Could not compute overlap with real CER: {e}\n")
    
    # 保存选择结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        result = {
            'method': 'model_b',
            'dataset_path': dataset_path,
            'model_b_path': model_b_path,
            'num_samples': num_samples,
            'selection_strategy': selection_strategy,
            'selected_indices': selected_indices,
            'difficulty_scores': difficulty_scores.tolist(),
            'overlap_info': overlap_info
        }
        result_path = os.path.join(save_dir, f'model_b_selection_{selection_strategy}_{num_samples}.pkl')
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"✅ Selection results saved to: {result_path}\n")
    
    print("="*80)
    print("✅ Sample selection completed!")
    print(f"   Selected {len(selected_indices)} samples")
    if overlap_info:
        print(f"   Overall Overlap: {overlap_info['overlap_count']}/{num_samples} ({overlap_info['overlap_ratio']:.2%})")
        if 'top_n_overlaps' in overlap_info:
            print(f"   Top-N Overlaps:")
            for n in sorted(overlap_info['top_n_overlaps'].keys()):
                info = overlap_info['top_n_overlaps'][n]
                print(f"     Top-{n:3d}: {info['overlap_count']:3d}/{n} ({info['overlap_ratio']:.2%})")
    print("="*80 + "\n")
    
    return selected_indices, overlap_info


def create_finetune_dataset(
    dataset_path: str,
    output_path: str,
    selected_indices: List[int],
    split: str = 'train'
):
    """
    根据选中的索引创建微调数据集
    
    Args:
        dataset_path: 原始数据集路径
        output_path: 输出数据集路径
        selected_indices: 选中的样本索引
        split: 数据集分割（'train'或'test'）
    """
    with open(dataset_path, "rb") as handle:
        loaded_data = pickle.load(handle)
    
    # 将索引映射到具体的day和trial
    selected_datasets = []
    current_idx = 0
    
    for day_idx, day_data in enumerate(loaded_data[split]):
        day_samples = []
        for trial_idx in range(len(day_data["sentenceDat"])):
            if current_idx in selected_indices:
                day_samples.append(trial_idx)
            current_idx += 1
        
        if day_samples:
            # 创建该天的子集
            day_subset = {
                "sentenceDat": [day_data["sentenceDat"][i] for i in day_samples],
                "transcriptions": [day_data["transcriptions"][i] for i in day_samples],
                "phonemes": [day_data["phonemes"][i] for i in day_samples],
                "timeSeriesLens": day_data["timeSeriesLens"][day_samples],
                "phoneLens": day_data["phoneLens"][day_samples],
                "phonePerTime": day_data["phonePerTime"][day_samples] if "phonePerTime" in day_data else None,
            }
            selected_datasets.append(day_subset)
    
    # 创建新的数据集
    new_dataset = {
        split: selected_datasets,
        "test": loaded_data.get("test", []),
        "competition": loaded_data.get("competition", []),
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump(new_dataset, handle)
    
    print(f"✅ Created finetune dataset with {len(selected_indices)} samples at {output_path}")


def compute_real_cer_scores(
    dataset_path: str,
    model_a_path: str,
    batch_size: int = 32,
    device: str = 'cuda'
) -> np.ndarray:
    """
    使用模型a计算每个样本的真实CER分数
    
    Args:
        dataset_path: 数据集路径
        model_a_path: 模型a的路径（包含modelWeights.pth和config.pkl）
        batch_size: 批次大小
        device: 设备
    
    Returns:
        cer_scores: 每个样本的CER分数数组
    """
    import pickle
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader
    from edit_distance import SequenceMatcher
    from ..model_a.get_model import get_model
    
    print(f"📊 Computing real CER scores using Model A...")
    print(f"   Model A path: {model_a_path}")
    print(f"   Dataset path: {dataset_path}\n")
    
    # 加载模型a
    model_weight_path = os.path.join(model_a_path, "modelWeights.pth")
    with open(os.path.join(model_a_path, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    
    model = get_model(config).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    print(f"✅ Model A loaded successfully\n")
    
    # 加载数据集
    with open(dataset_path, "rb") as handle:
        loaded_data = pickle.load(handle)
    
    train_dataset = SpeechDataset(loaded_data["train"], transform=None)
    
    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)
        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )
    
    loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=_padding
    )
    
    cer_scores = []
    
    with torch.no_grad():
        for X, y, X_len, y_len, _ in tqdm(loader, desc='Computing CER'):
            X, y = X.to(device), y.to(device)
            pred = model.forward(X)
            log_probs = pred.log_softmax(2)
            
            for i in range(pred.shape[0]):
                seq_len = ((X_len[i] - model.kernelLen) / model.strideLen).int().item()
                decoded_seq = torch.argmax(log_probs[i, :seq_len, :], dim=-1)
                decoded_seq = torch.unique_consecutive(decoded_seq, dim=-1)
                decoded_seq = decoded_seq.cpu().numpy()
                decoded_seq = np.array([idx for idx in decoded_seq if idx != 0])
                
                true_seq = y[i, :y_len[i]].cpu().numpy()
                matcher = SequenceMatcher(a=true_seq.tolist(), b=decoded_seq.tolist())
                edit_dist = matcher.distance()
                cer = edit_dist / len(true_seq) if len(true_seq) > 0 else 0.0
                
                cer_scores.append(cer)
    
    cer_scores = np.array(cer_scores)
    print(f"✅ CER computation completed, score range: [{cer_scores.min():.4f}, {cer_scores.max():.4f}]\n")
    
    return cer_scores


def compute_real_slpe_scores(
    dataset_path: str,
    model_a_path: str,
    batch_size: int = 32,
    device: str = 'cuda',
    slpe_cache_dir: Optional[str] = None,
    day: Optional[int] = None,
    pretrained_ndays: int = 7,
    base_dir: str = '/root/25S151115/project3'
) -> np.ndarray:
    """
    使用模型a计算每个样本的真实SLPE分数，支持从缓存加载
    
    Args:
        dataset_path: 数据集路径
        model_a_path: 模型a的路径（包含modelWeights.pth和config.pkl）
        batch_size: 批次大小
        device: 设备
        slpe_cache_dir: SLPE缓存目录（如果提供，会从缓存加载）
        day: 目标天数（用于构建缓存路径）
        pretrained_ndays: 预训练天数（用于构建缓存路径）
        base_dir: 基础目录（用于构建缓存路径）
    
    Returns:
        slpe_scores: 每个样本的SLPE分数数组
    """
    import pickle
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader
    from ..model_a.get_model import get_model
    
    # 尝试从缓存加载SLPE分数（slpe_cache_dir=None 时跳过缓存，强制重算）
    if day is not None and slpe_cache_dir is not None:
        cache_dir = slpe_cache_dir
        cache_file = os.path.join(cache_dir, f'day_{day}_slpe_scores.pkl')
        
        if os.path.exists(cache_file):
            print(f"📦 Loading SLPE scores from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    cached_result = pickle.load(f)
                if 'slpe_scores' in cached_result:
                    slpe_scores = np.array(cached_result['slpe_scores'])
                    print(f"✅ Loaded {len(slpe_scores)} SLPE scores from cache")
                    print(f"   Score range: [{slpe_scores.min():.4f}, {slpe_scores.max():.4f}]\n")
                    return slpe_scores
                else:
                    print(f"⚠️  Cache file format invalid, computing from scratch...")
            except Exception as e:
                print(f"⚠️  Failed to load cache: {e}, computing from scratch...")
    
    print(f"📊 Computing real SLPE scores using Model A...")
    print(f"   Model A path: {model_a_path}")
    print(f"   Dataset path: {dataset_path}\n")
    
    # 加载模型a
    model_weight_path = os.path.join(model_a_path, "modelWeights.pth")
    with open(os.path.join(model_a_path, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    
    model = get_model(config).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    print(f"✅ Model A loaded successfully\n")
    
    # 加载数据集
    with open(dataset_path, "rb") as handle:
        loaded_data = pickle.load(handle)
    
    train_dataset = SpeechDataset(loaded_data["train"], transform=None)
    
    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)
        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )
    
    loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=_padding
    )
    
    # 使用SLPE批量计算
    slpe_scores = compute_slpe_batch(model, loader, device=device, blank=0)
    
    print(f"✅ SLPE computation completed, score range: [{slpe_scores.min():.4f}, {slpe_scores.max():.4f}]\n")
    
    return slpe_scores


def real_cer_sample_selection(
    dataset_path: str,
    num_samples: int,
    model_a_path: str,
    selection_strategy: str = 'hard',
    batch_size: int = 32,
    device: str = 'cuda',
    save_dir: Optional[str] = None
) -> tuple[List[int], None]:
    """
    根据真实CER选择样本
    
    Args:
        dataset_path: 数据集路径
        num_samples: 需要选择的样本数量
        model_a_path: 模型a的路径（用于计算真实CER）
        selection_strategy: 选择策略 ('hard'=最高CER, 'easy'=最低CER)
        batch_size: 计算CER时的batch大小
        device: 设备
    
    Returns:
        选中的样本索引列表
    """
    print("\n" + "="*80)
    print("🚀 Using Real CER for sample selection")
    print("="*80)
    print(f"Dataset path: {dataset_path}")
    print(f"Model A path: {model_a_path}")
    print(f"Number of samples: {num_samples}")
    print(f"Selection strategy: {selection_strategy} ({'highest CER' if selection_strategy == 'hard' else 'lowest CER'})")
    print("="*80 + "\n")
    
    # 计算真实CER
    cer_scores = compute_real_cer_scores(
        dataset_path, model_a_path, batch_size, device
    )
    
    # 根据策略选择样本
    selected_indices = _select_samples_by_strategy(
        cer_scores, num_samples, selection_strategy, seed=0
    )
    
    # 保存选择结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        result = {
            'method': 'real_cer',
            'dataset_path': dataset_path,
            'model_a_path': model_a_path,
            'num_samples': num_samples,
            'selection_strategy': selection_strategy,
            'selected_indices': selected_indices,
            'cer_scores': cer_scores.tolist()
        }
        result_path = os.path.join(save_dir, f'real_cer_selection_{selection_strategy}_{num_samples}.pkl')
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"✅ Selection results saved to: {result_path}\n")
    
    print("="*80)
    print("✅ Sample selection completed!")
    print(f"   Selected {len(selected_indices)} samples")
    print("="*80 + "\n")
    
    return selected_indices, None


def real_slpe_sample_selection(
    dataset_path: str,
    num_samples: int,
    model_a_path: str,
    selection_strategy: str = 'hard',
    seed: int = 0,
    batch_size: int = 32,
    device: str = 'cuda',
    save_dir: Optional[str] = None,
    slpe_cache_dir: Optional[str] = None,
    day: Optional[int] = None,
    pretrained_ndays: int = 7,
    base_dir: str = '/root/25S151115/project3'
) -> tuple[List[int], None]:
    """
    根据真实SLPE选择样本
    
    Args:
        dataset_path: 数据集路径
        num_samples: 需要选择的样本数量
        model_a_path: 模型a的路径（用于计算真实SLPE）
        selection_strategy: 选择策略 ('hard'=最高SLPE, 'easy'=最低SLPE)
        batch_size: 计算SLPE时的batch大小（如果使用缓存则不会用到）
        device: 设备
        slpe_cache_dir: SLPE缓存目录（如果提供，会从缓存加载）
        day: 目标天数（用于构建缓存路径）
        pretrained_ndays: 预训练天数（用于构建缓存路径）
        base_dir: 基础目录（用于构建缓存路径）
    
    Returns:
        选中的样本索引列表
    """
    print("\n" + "="*80)
    print("🚀 Using Real SLPE for sample selection")
    print("="*80)
    print(f"Dataset path: {dataset_path}")
    print(f"Model A path: {model_a_path}")
    print(f"Number of samples: {num_samples}")
    print(f"Selection strategy: {selection_strategy} ({'highest SLPE' if selection_strategy == 'hard' else 'lowest SLPE'})")
    if slpe_cache_dir or day:
        cache_path = slpe_cache_dir or os.path.join(base_dir, 'outputs', 'slpe_scores', f'{pretrained_ndays}days')
        print(f"SLPE cache: {cache_path}")
    print("="*80 + "\n")
    
    # 计算真实SLPE（会尝试从缓存加载）
    slpe_scores = compute_real_slpe_scores(
        dataset_path, model_a_path, batch_size, device,
        slpe_cache_dir=slpe_cache_dir,
        day=day,
        pretrained_ndays=pretrained_ndays,
        base_dir=base_dir
    )
    
    # 根据策略选择样本
    selected_indices = _select_samples_by_strategy(
        slpe_scores, num_samples, selection_strategy, seed=seed
    )
    
    # 保存选择结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        result = {
            'method': 'real_slpe',
            'dataset_path': dataset_path,
            'model_a_path': model_a_path,
            'num_samples': num_samples,
            'selection_strategy': selection_strategy,
            'selected_indices': selected_indices,
            'slpe_scores': slpe_scores.tolist()
        }
        result_path = os.path.join(save_dir, f'real_slpe_selection_{selection_strategy}_{num_samples}.pkl')
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"✅ Selection results saved to: {result_path}\n")
    
    print("="*80)
    print("✅ Sample selection completed!")
    print(f"   Selected {len(selected_indices)} samples")
    print("="*80 + "\n")
    
    return selected_indices, None


def _badge_kmeanspp_select(grad_embs: np.ndarray, n_select: int) -> List[int]:
    """
    基于梯度嵌入的 k-means++ 多样性采样。
    与 project1/src/badge.py 中的逻辑一致，使用分解距离加速。

    Args:
        grad_embs: 梯度嵌入矩阵 [N, D]
        n_select: 需要选出的样本数

    Returns:
        选中的样本索引列表
    """
    from scipy import stats as scipy_stats

    N, D = grad_embs.shape
    n_select = min(n_select, N)

    norms_sq = np.sum(grad_embs ** 2, axis=-1)  # [N]

    chosen = set()
    chosen_list: List[int] = []
    D2: Optional[np.ndarray] = None

    for step in range(n_select):
        if len(chosen) == 0:
            ind = int(np.argmax(norms_sq))
        else:
            new_center = grad_embs[chosen_list[-1]]  # [D]
            new_norm_sq = norms_sq[chosen_list[-1]]
            dot = grad_embs @ new_center  # [N]
            new_dist = norms_sq * new_norm_sq - dot ** 2
            new_dist = np.sqrt(np.clip(new_dist, a_min=0, a_max=None))
            D2 = np.minimum(D2, new_dist) if D2 is not None else new_dist
            D2[chosen_list] = 0
            D2_sq = D2 ** 2
            total = D2_sq.sum()
            if total == 0:
                remaining = list(set(range(N)) - chosen)
                ind = remaining[0]
            else:
                probs = D2_sq / total
                custom_dist = scipy_stats.rv_discrete(
                    name='badge_kmpp', values=(np.arange(N), probs)
                )
                ind = custom_dist.rvs(size=1)[0]
                while ind in chosen:
                    ind = custom_dist.rvs(size=1)[0]

        chosen.add(ind)
        chosen_list.append(ind)

        if (step + 1) % 20 == 0 or step == n_select - 1:
            print(f"   k-means++ progress: {step + 1}/{n_select}")

    return chosen_list


def _compute_badge_gradient_embeddings(
    model,
    loader,
    device: str = 'cuda'
) -> np.ndarray:
    """
    逐样本计算基于 CTC Loss 的梯度嵌入。

    对每个样本:
      1) forward pass 得到 logits
      2) greedy decode 获取伪标签
      3) 用伪标签计算 CTC loss
      4) 反传得到 fc_decoder_out.weight 的梯度，展平为梯度嵌入

    Args:
        model: Conformer 模型（需要有 fc_decoder_out, kernelLen, strideLen）
        loader: DataLoader，返回 (X, y, X_len, y_len, days)
        device: 计算设备

    Returns:
        grad_embeddings: [N, C * hidden_dim] 的梯度嵌入矩阵
    """
    import torch.nn.functional as F

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    all_grad_embs = []

    for X, y, X_len, y_len, _ in tqdm(loader, desc='Computing BADGE gradient embeddings'):
        X = X.to(device)
        batch_size = X.shape[0]

        for i in range(batch_size):
            xi = X[i:i+1]  # [1, T, D]
            seq_len = max(1, ((X_len[i] - model.kernelLen) / model.strideLen).int().item())

            # Step 1: greedy decode → pseudo-label (no grad)
            with torch.no_grad():
                logits = model(xi)  # [1, T', C]
                decoded = torch.argmax(logits[0, :seq_len, :], dim=-1)
                decoded = torch.unique_consecutive(decoded, dim=-1)
                pseudo_label = decoded[decoded != 0]

            if len(pseudo_label) == 0:
                pseudo_label = torch.tensor([1], device=device, dtype=torch.int32)

            # Step 2: 只对 fc_decoder_out.weight 开启梯度，计算 CTC loss 梯度
            model.fc_decoder_out.weight.requires_grad_(True)
            model.fc_decoder_out.zero_grad()

            hidden = model.forward_features_seq(xi)[:, :seq_len, :]  # [1, T', hidden_dim]
            logits_grad = model.fc_decoder_out(hidden)  # [1, T', C]
            lp = logits_grad.log_softmax(2).permute(1, 0, 2)  # [T', 1, C]

            input_lengths = torch.tensor([seq_len], dtype=torch.long, device=device)
            target_lengths = torch.tensor([len(pseudo_label)], dtype=torch.long, device=device)

            loss = F.ctc_loss(
                lp, pseudo_label.unsqueeze(0).to(torch.long),
                input_lengths, target_lengths,
                blank=0, reduction='mean', zero_infinity=True
            )
            loss.backward()

            grad_w = model.fc_decoder_out.weight.grad  # [C, hidden_dim]
            all_grad_embs.append(grad_w.detach().cpu().flatten().numpy())
            model.fc_decoder_out.weight.requires_grad_(False)

    for p in model.parameters():
        p.requires_grad_(True)
    model.zero_grad()

    return np.stack(all_grad_embs)


def badge_sample_selection(
    dataset_path: str,
    num_samples: int,
    model_a_path: str,
    batch_size: int = 32,
    device: str = 'cuda',
    save_dir: Optional[str] = None
) -> tuple[List[int], None]:
    """
    使用 BADGE (Batch Active Learning by Diverse Gradient Embeddings) 选择样本。
    适配 CTC 模型：用 CTC Loss 的梯度嵌入 + k-means++ 多样性采样。

    Args:
        dataset_path: 数据集路径
        num_samples: 需要选择的样本数量
        model_a_path: 模型a的路径（包含modelWeights.pth和config.pkl）
        batch_size: 推理时的batch大小
        device: 设备
        save_dir: 保存选择结果的目录

    Returns:
        (selected_indices, None) 与其他方法签名一致
    """
    import pickle
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader
    from ..model_a.get_model import get_model

    print("\n" + "="*80)
    print("BADGE: Batch Active Learning by Diverse Gradient Embeddings (CTC)")
    print("="*80)
    print(f"Dataset path: {dataset_path}")
    print(f"Model A path: {model_a_path}")
    print(f"Number of samples: {num_samples}")
    print("="*80 + "\n")

    model_weight_path = os.path.join(model_a_path, "modelWeights.pth")
    with open(os.path.join(model_a_path, "config.pkl"), "rb") as f:
        config = pickle.load(f)

    model = get_model(config).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    print(f"Model A loaded successfully\n")

    with open(dataset_path, "rb") as handle:
        loaded_data = pickle.load(handle)

    train_dataset = SpeechDataset(loaded_data["train"], transform=None)
    total_samples = len(train_dataset)
    print(f"Total candidate samples: {total_samples}")

    if num_samples > total_samples:
        print(f"Warning: requested {num_samples} but only {total_samples} available, selecting all")
        num_samples = total_samples

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)
        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_padding
    )

    print(f"\nStep 1/2: Computing CTC gradient embeddings...")
    grad_embs = _compute_badge_gradient_embeddings(model, loader, device)
    emb_dim = grad_embs.shape[1]
    print(f"Gradient embedding shape: [{total_samples}, {emb_dim}]")
    norms = np.linalg.norm(grad_embs, axis=1)
    print(f"Gradient norm range: [{norms.min():.6f}, {norms.max():.6f}]\n")

    print(f"Step 2/2: Running k-means++ to select {num_samples} diverse samples...")
    selected_indices = _badge_kmeanspp_select(grad_embs, num_samples)

    selected_norms = norms[selected_indices]
    print(f"\nSelected samples gradient norm range: [{selected_norms.min():.6f}, {selected_norms.max():.6f}]")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        result = {
            'method': 'badge',
            'dataset_path': dataset_path,
            'model_a_path': model_a_path,
            'num_samples': num_samples,
            'selected_indices': selected_indices,
            'gradient_norms': norms.tolist(),
        }
        result_path = os.path.join(save_dir, f'badge_selection_{num_samples}.pkl')
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"Selection results saved to: {result_path}\n")

    print("="*80)
    print(f"BADGE selection completed! Selected {len(selected_indices)} samples")
    print("="*80 + "\n")

    return selected_indices, None


def select_samples_for_finetune(
    dataset_path: str,
    num_samples: int,
    method: str = 'random',
    model_b_path: Optional[str] = None,
    model_a_path: Optional[str] = None,
    selection_strategy: str = 'hard',
    seed: int = 0,
    batch_size: int = 32,
    device: str = 'cuda',
    save_dir: Optional[str] = None,
    auto_train_model_b: bool = False,
    pretrained_ndays: int = 7,
    base_dir: str = '/root/25S151115/project3',
    **kwargs
) -> tuple[List[int], Optional[dict]]:
    """
    统一的样本选择接口
    
    Args:
        dataset_path: 数据集路径
        num_samples: 需要选择的样本数量
        method: 选择方法 ('random', 'model_b', 'real_cer', 'real_slpe' 或 'length')
        model_b_path: 模型b的路径（当method='model_b'时需要）
        model_a_path: 模型a的路径（当method='real_cer'或'real_slpe'时需要）
        selection_strategy: 选择策略
            - 当method='model_b'时: 'hard', 'easy', 'diverse'
            - 当method='real_cer'或'real_slpe'时: 'hard'=最高分数, 'easy'=最低分数
        seed: 随机种子
            - 当method='random'时: 控制随机采样
            - 当method='model_b'时: 如果使用同一个训练好的model_b，seed不影响结果（预测是确定性的）
                                  如果要测试不同seed的效果，需要训练多个不同seed的model_b模型
            - 当method='real_cer'或'real_slpe'时: 不使用
        batch_size: 批次大小（当method='model_b'、'real_cer'或'real_slpe'时使用）
        device: 设备
        auto_train_model_b: 如果model_b_path不存在，是否自动训练（默认False）
        pretrained_ndays: 预训练天数（用于自动训练model_b）
        base_dir: 基础目录（用于自动训练model_b）
    
    Returns:
        选中的样本索引列表
    """
    if method == 'random':
        return random_sample_selection(dataset_path, num_samples, seed, save_dir)
    elif method == 'length':
        return length_sample_selection(
            dataset_path=dataset_path,
            num_samples=num_samples,
            selection_strategy=selection_strategy,
            save_dir=save_dir
        )
    elif method == 'model_b':
        return model_b_sample_selection(
            dataset_path, 
            num_samples, 
            model_b_path, 
            selection_strategy, 
            seed,
            batch_size,
            device,
            model_a_path,  # 用于计算重合度
            save_dir,
            auto_train=auto_train_model_b,
            pretrained_ndays=pretrained_ndays,
            base_dir=base_dir
        )
    elif method == 'real_cer':
        if model_a_path is None:
            raise ValueError("model_a_path is required when method='real_cer'")
        return real_cer_sample_selection(
            dataset_path,
            num_samples,
            model_a_path,
            selection_strategy,
            batch_size,
            device,
            save_dir
        )
    elif method == 'real_slpe':
        if model_a_path is None:
            raise ValueError("model_a_path is required when method='real_slpe'")
        # 尝试从kwargs中获取day和pretrained_ndays用于缓存
        # 注意：这里需要从调用方传递day参数，如果没有则尝试从dataset_path推断
        kwargs_slpe = {}
        if 'day' in kwargs:
            kwargs_slpe['day'] = kwargs['day']
            kwargs_slpe['pretrained_ndays'] = kwargs.get('pretrained_ndays', pretrained_ndays)
            kwargs_slpe['base_dir'] = kwargs.get('base_dir', base_dir)
            kwargs_slpe['slpe_cache_dir'] = kwargs.get('slpe_cache_dir')
        return real_slpe_sample_selection(
            dataset_path,
            num_samples,
            model_a_path,
            selection_strategy,
            seed,
            batch_size,
            device,
            save_dir,
            **kwargs_slpe
        )
    elif method == 'badge':
        if model_a_path is None:
            raise ValueError("model_a_path is required when method='badge'")
        return badge_sample_selection(
            dataset_path,
            num_samples,
            model_a_path,
            batch_size,
            device,
            save_dir
        )
    else:
        raise ValueError(
            f"Unknown selection method: {method}. "
            "Choose 'random', 'length', 'model_b', 'real_cer', 'real_slpe', or 'badge'"
        )
