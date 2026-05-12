# config.py - 模型a（解码模型）的配置管理
import torch

SESSION_NAMES_CHRONOLOGICAL = [
    't12.2022.04.28', 't12.2022.05.26', 't12.2022.06.21', 't12.2022.07.21', 't12.2022.08.13',
    't12.2022.05.05', 't12.2022.06.02', 't12.2022.06.23', 't12.2022.07.27', 't12.2022.08.18',
    't12.2022.05.17', 't12.2022.06.07', 't12.2022.06.28', 't12.2022.07.29', 't12.2022.08.23',
    't12.2022.05.19', 't12.2022.06.14', 't12.2022.07.05', 't12.2022.08.02', 't12.2022.08.25',
    't12.2022.05.24', 't12.2022.06.16', 't12.2022.07.14', 't12.2022.08.11'
]


def get_base_config(model_name='conformer'):
    """获取基础配置"""
    config = {}
    config["model_name"] = model_name

    config["sessionNames"] = list(SESSION_NAMES_CHRONOLOGICAL)
    config["sessionNames"].sort()
    
    config['seqLen'] = 150
    config['maxTimeSeriesLen'] = 1200
    config['batchSize'] = 64
    config['lrStart'] = 0.02
    config['lrEnd'] = 0.02
    config['nBatch'] = 20000
    config['seed'] = 0
    config['l2_decay'] = 1e-5
    config["patience"] = 15

    config['whiteNoiseSD'] = 0.8
    config['constantOffsetSD'] = 0.2

    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    
    return config


def get_model_config(model_name='conformer'):
    """获取模型专属参数"""
    if model_name == 'gru':
        return {
            'model': 'gru',
            'nInputFeatures': 256,
            'nClasses': 40,
            'nUnits': 512,
            'nLayers': 3,
            'dropout': 0.4,
            'strideLen': 4,
            'kernelLen': 32,
            'gaussianSmoothWidth': 2.0,
            'bidirectional': True,
        }
    
    elif model_name == 'moganet':
        return {
            'model': 'moganet',
            'nInputFeatures': 256,
            'nClasses': 40,
            'nUnits': 1024,
            'nLayers': 3,
            'dropout': 0.4,
            'strideLen': 4,
            'kernelLen': 32,
            'gaussianSmoothWidth': 2.0,
            'bidirectional': True,
            'embed_dims': [128, 256, 512],
            'depths': [2, 4, 6],
            'ffn_ratios': [4, 4, 4],
            'act_type': 'GELU',
            'attn_act_type': 'SiLU',
            'drop_path_rate': 0.1,
            'patch_strides': [2, 2, 1],
            'patch_sizes': [3, 3, 3],
        }
    
    elif model_name == 'conformer':
        return {
            'model': 'conformer',
            'nInputFeatures': 256,
            'nClasses': 40,
            'nUnits': 256,
            'nLayers': 3,
            'num_heads': 8,
            'dropout': 0.4,
            'strideLen': 4,
            'kernelLen': 32,
            'gaussianSmoothWidth': 2.0,
            'bidirectional': False,
            'conv_kernel_size': 31,
            'ff_expansion_factor': 4,
            'conv_expansion': 2,
            'use_group_norm': False,
            'window_size': 64,
            'use_local_attention': True,
        }
    elif model_name == 'conformer1':
        return {
            'model': 'conformer1',
            'nInputFeatures': 256,
            'nClasses': 40,
            'nUnits': 256,
            'nLayers': 3,
            'num_heads': 8,
            'dropout': 0.4,
            'strideLen': 4,
            'kernelLen': 32,
            'gaussianSmoothWidth': 2.0,
            'bidirectional': False,
            'conv_kernel_size': 31,
            'ff_expansion_factor': 4,
            'conv_expansion': 2,
            'use_group_norm': False,
            'window_size': 64,
            'use_local_attention': True,
        }
    else:
        raise ValueError(f"Model '{model_name}' not recognized. Choose 'gru', 'moganet', or 'conformer'.")


def get_train_config(nDays=7, model_name='conformer', base_dir='/root/25S151115/project3', seed=None, **kwargs):
    """获取训练配置"""
    config = get_base_config(model_name)
    config["nDays"] = nDays
    config["sessionNames_train"] = config["sessionNames"][:nDays]
    
    modelName = f'{model_name}-{nDays}days'
    if seed is not None:
        modelName = f'{modelName}-seed{seed}'
    config['outputDir'] = f'{base_dir}/outputs/model_train/{modelName}'
    config['datasetPath'] = f'{base_dir}/data/ptDecoder_ctc{nDays}'
    
    # 如果kwargs中有seed，设置到config（优先使用显式传递的seed）
    if seed is not None:
        config['seed'] = seed
    elif 'seed' in kwargs:
        config['seed'] = kwargs['seed']
    # 如果都没有，使用config中已有的seed（从get_base_config来的默认值0）
    
    # 添加模型参数
    model_config = get_model_config(model_name)
    config.update(model_config)
    
    # 统一参数名称
    config['neural_dim'] = config['nInputFeatures']
    config['n_classes'] = config['nClasses']
    config['hidden_dim'] = config['nUnits']
    config['layer_dim'] = config['nLayers']
    
    return config


def get_finetune_config(day, num_samples, model_name='conformer', 
                       pretrained_ndays=7,
                       base_dir='/root/25S151115/project3',
                       **kwargs):
    """获取微调配置"""
    config = get_train_config(pretrained_ndays, model_name, base_dir)
    config["sessionNames"].sort()
    
    # 设置预训练模型路径
    config["pretrainedModelPath"] = (
        f"{base_dir}/outputs/model_train/"
        f"{model_name}-{pretrained_ndays}days"
    )
    
    # 设置微调数据路径
    parts = config["sessionNames"][day-1].split('.')
    config["finetuneDataPath"] = f'{base_dir}/data/data{parts[-2]+ parts[-1]}'
    
    # 设置样本数量和输出路径
    config["pretrainedDataNum"] = num_samples
    config["eval_day"] = day
    # 新的文件夹结构：{pretrained_ndays}-{day}/{method_tag}/run_seed{seed}/[selection_seed{selection_seed}/]
    # 如果kwargs中有method和seed，使用新结构；否则使用旧结构
    if 'method' in kwargs and 'seed' in kwargs:
        method = kwargs['method']
        seed = kwargs['seed']
        selection_seed = kwargs.get('selection_seed')
        # 优先使用 output_tag 作为目录标签（例如 random100/random150/...），避免不同实验互相覆盖
        output_tag = kwargs.get("output_tag")
        if output_tag:
            method_tag = str(output_tag)
        else:
            # 对 full_data 场景（method=random 且 num_samples<=0）使用独立标签，避免与随机子集混淆/覆盖
            method_tag = "full_data" if (method == "random" and isinstance(num_samples, int) and num_samples <= 0) else method
        run_seed_tag = f"run_seed{seed}"
        output_root = (
            f"{base_dir}/outputs/model_test/"
            f"{pretrained_ndays}-{day}/{method_tag}/{run_seed_tag}"
        )
        if selection_seed is not None:
            output_root = f"{output_root}/selection_seed{selection_seed}"
        config["pretrainedModelOutputPath"] = output_root
        # 保存到config中
        config['selection_method'] = method_tag
        config['seed'] = seed
        if selection_seed is not None:
            config['selection_seed'] = selection_seed
        if 'selection_strategy' in kwargs:
            config['selection_strategy'] = kwargs['selection_strategy']
    else:
        config["pretrainedModelOutputPath"] = (
            f"{base_dir}/outputs/model_test/"
            f"{pretrained_ndays}days-to-day{day}-num={num_samples}"
        )
    
    # 微调专用的默认参数
    finetune_defaults = {
        'batchSize': 32,
        'lrStart': 5e-5,
        'lrEnd': 1e-6,
        'nBatch': 20000,
        'patience': 30,
        'num_workers': 6,
        'scheduler_type': 'linear',  # linear | cosine
    }
    config.update(finetune_defaults)
    config.update(kwargs)
    
    return config
