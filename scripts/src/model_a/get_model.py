# get_model.py - 获取模型a（解码模型）
from .models.GRUDecoder import GRUDecoder
from .models.MogaNet1D import MogaNetDecoder
from .models.Conformer import ConformerDecoder


def get_model(config):
    """根据配置获取模型实例"""
    model_type = config.get("model", "gru")
    
    if model_type == "gru":
        model = GRUDecoder(
            neural_dim=config["neural_dim"],
            n_classes=config["n_classes"],
            hidden_dim=config["hidden_dim"],
            layer_dim=config["layer_dim"],
            dropout=config["dropout"],
            device=config["device"],
            strideLen=config["strideLen"],
            kernelLen=config["kernelLen"],
            gaussianSmoothWidth=config["gaussianSmoothWidth"],
            bidirectional=config["bidirectional"],
        )
    
    elif model_type == "moganet":
        model = MogaNetDecoder(
            neural_dim=config["neural_dim"],
            n_classes=config["n_classes"],
            hidden_dim=config["hidden_dim"],
            layer_dim=config["layer_dim"],
            dropout=config["dropout"],
            device=config["device"],
            strideLen=config["strideLen"],
            kernelLen=config["kernelLen"],
            gaussianSmoothWidth=config["gaussianSmoothWidth"],
            bidirectional=config["bidirectional"],
            embed_dims=config.get("embed_dims", None),
            depths=config.get("depths", None),
            ffn_ratios=config.get("ffn_ratios", None),
            act_type=config.get("act_type", "GELU"),
            attn_act_type=config.get("attn_act_type", "SiLU"),
            drop_path_rate=config.get("drop_path_rate", 0.1),
            patch_strides=config.get("patch_strides", None),
            patch_sizes=config.get("patch_sizes", None),
        )
    
    elif model_type == "conformer":
        model = ConformerDecoder(
            neural_dim=config["neural_dim"],
            n_classes=config["n_classes"],
            hidden_dim=config["hidden_dim"],
            layer_dim=config["layer_dim"],
            num_heads=config.get("num_heads", 8),
            dropout=config["dropout"],
            device=config["device"],
            strideLen=config["strideLen"],
            kernelLen=config["kernelLen"],
            gaussianSmoothWidth=config["gaussianSmoothWidth"],
            conv_kernel_size=config.get("conv_kernel_size", 31),
            ff_expansion_factor=config.get("ff_expansion_factor", 4),
            conv_expansion_factor=config.get("conv_expansion", 2),
            bidirectional=config.get("bidirectional", False),
            use_group_norm=config.get("use_group_norm", False),
            window_size=config.get("window_size", 64),
            use_local_attention=config.get("use_local_attention", True),
            use_adapter=config.get("use_adapter", False),
            adapter_bottleneck=config.get("adapter_bottleneck", 64),
            use_ca_block=config.get("use_ca_block", False),
            ca_bottleneck=config.get("ca_bottleneck", 64),
            adapter_init_scale=config.get("adapter_init_scale", 1e-2),
            ca_init_scale=config.get("ca_init_scale", 1e-2),
        )
    elif model_type == "conformer1":
        from .models.conformer1 import ECoGConformer
        model = ECoGConformer(
            input_channels=config["neural_dim"],
            num_classes=config["n_classes"],
            emb_size=config["hidden_dim"],
            depth=config["layer_dim"],
            num_heads=config.get("num_heads", 8),
            strideLen=config["strideLen"],
            kernelLen=config["kernelLen"],
        )
    else:
        raise ValueError(f"Model '{model_type}' not recognized.")
    
    return model.to(config["device"])
