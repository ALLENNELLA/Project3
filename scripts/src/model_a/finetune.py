# finetune.py - 模型a的微调模块
import os
import pickle
import time
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from edit_distance import SequenceMatcher
from ..utils.dataset import SpeechDataset
from .get_model import get_model
from .trainer import evaluate_model, DataPrefetcher


def _padding(batch):
    """批次填充函数"""
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


def get_finetune_dataset_loaders(dataset_path, batch_size, num_samples=None, seed=0, num_workers=0, selected_indices=None):
    """创建微调数据加载器（支持子集采样）
    
    Args:
        dataset_path: 数据集路径
        batch_size: batch大小
        num_samples: 样本数量（如果selected_indices为None，则随机选择）
        seed: 随机种子（仅当selected_indices为None时使用）
        num_workers: 数据加载的worker数量
        selected_indices: 预选择的样本索引列表（如果提供，将优先使用）
    """
    with open(dataset_path, "rb") as handle:
        loaded_data = pickle.load(handle)

    full_train_ds = SpeechDataset(loaded_data["train"], transform=None)
    test_ds = SpeechDataset(loaded_data["test"])
    total_train_samples = len(full_train_ds)
    
    train_ds = full_train_ds
    if num_samples is not None and num_samples > 0 and num_samples >= total_train_samples:
        print(
            f"Finetune data request = {num_samples} sentences, "
            f"but only {total_train_samples} available; using {total_train_samples}."
        )
    if num_samples is not None and num_samples > 0 and num_samples < len(full_train_ds):
        if selected_indices is not None:
            # 使用预选择的样本索引
            print(f"Finetune data = {len(selected_indices)} sentences (using pre-selected indices).")
            # 验证索引有效性
            valid_indices = [idx for idx in selected_indices if 0 <= idx < len(full_train_ds)]
            if len(valid_indices) != len(selected_indices):
                print(f"⚠️ Warning: {len(selected_indices) - len(valid_indices)} invalid indices filtered out")
            if len(valid_indices) != num_samples:
                print(
                    f"Finetune data adjusted to {len(valid_indices)} sentences "
                    f"(requested {num_samples})."
                )
            train_ds = Subset(full_train_ds, valid_indices)
        else:
            # 随机选择（向后兼容）
            print(f"Finetune data = {num_samples} sentences (random selection).")
            random.seed(seed)
            full_indices = list(range(len(full_train_ds)))
            num_samples = min(num_samples, len(full_train_ds))
            sampled_indices = random.sample(full_indices, num_samples)
            train_ds = Subset(full_train_ds, sampled_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loaded_data


def finetune_model(config):
    """微调模型"""
    print("🚀 Starting fine-tuning process...")
    
    os.makedirs(config["pretrainedModelOutputPath"], exist_ok=True)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    device = config["device"]

    # 保存配置
    with open(os.path.join(config["pretrainedModelOutputPath"], "config.pkl"), "wb") as f:
        pickle.dump(config, f)
    
    # 保存实验信息标记
    info_file = os.path.join(config["pretrainedModelOutputPath"], "experiment_info.txt")
    with open(info_file, 'w') as f:
        f.write(f"Method: {config.get('selection_method', 'unknown')}\n")
        f.write(f"Seed: {config.get('seed', 'unknown')}\n")
        f.write(f"Pretrained Days: {config.get('nDays', 'unknown')}\n")
        f.write(f"Finetune Day: {config.get('eval_day', 'unknown')}\n")
        f.write(f"Num Samples: {config.get('pretrainedDataNum', 'unknown')}\n")
        f.write(f"Selection Strategy: {config.get('selection_strategy', 'unknown')}\n")
    print(f"✅ Experiment info saved to: {info_file}")
    
    # 创建结果保存目录
    results_dir = os.path.join(config["pretrainedModelOutputPath"], "results")
    os.makedirs(results_dir, exist_ok=True)

    # 加载数据
    # 如果config中有selected_indices，使用它；否则使用随机选择
    selected_indices = config.get("selected_indices")
    train_loader, test_loader, _ = get_finetune_dataset_loaders(
        config["finetuneDataPath"],
        config["batchSize"],
        num_samples=config.get("pretrainedDataNum"),
        seed=config.get('seed', 0),
        num_workers=config.get("num_workers", 0),
        selected_indices=selected_indices,
    )

    # 构建模型
    model = get_model(config).to(device)
    
    # 加载预训练权重
    loaded_pretrained = False
    pretrained_path = config.get("pretrainedModelPath")
    if pretrained_path and os.path.exists(pretrained_path):
        model_weight_path = os.path.join(pretrained_path, "modelWeights.pth")
        if os.path.exists(model_weight_path):
            print(f"✅ Loading pre-trained weights from: {pretrained_path}")
            try:
                # 先在CPU上加载，再以 strict=False 方式注入模型，忽略新加的 PEFT 参数
                state_dict = torch.load(model_weight_path, map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
                print("Loaded pre-trained weights with strict=False (ignored missing/extra keys).")
                loaded_pretrained = True
            except Exception as e:
                print(f"⚠️ Warning: Could not load full state_dict. Attempting partial load. Error: {e}")
                pretrained_dict = torch.load(model_weight_path, map_location="cpu")
                model_dict = model.state_dict()
                pretrained_dict = {
                    k: v for k, v in pretrained_dict.items() 
                    if k in model_dict and model_dict[k].shape == v.shape
                }
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=False)
                print("Partial weights loaded successfully (filtered by matching keys/shapes).")
                loaded_pretrained = True
        else:
            print("❌ Model weights file not found. Training from scratch.")
    else:
        print("❌ No valid pre-trained model path provided. Training from scratch.")

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lrStart"],
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=config["l2_decay"],
    )
    scheduler_type = str(config.get("scheduler_type", "linear")).lower()
    if scheduler_type == "cosine":
        cosine_tmax = int(config.get("cosine_tmax", config["nBatch"]))
        cosine_tmax = max(1, cosine_tmax)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_tmax,
            eta_min=float(config.get("lrEnd", 0.0)),
        )
    else:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=config["lrEnd"] / config["lrStart"],
            total_iters=config["nBatch"],
        )
        scheduler_type = "linear"

    print(f"[train] scheduler={scheduler_type}, grad_clip=off")
    if scheduler_type == "cosine":
        print(f"[train] cosine_tmax={cosine_tmax}")

    # 训练循环
    test_loss_history = []
    test_cer_history = []
    patience = config.get("patience", 30)
    trigger_times = 0
    best_test_cer = np.inf
    start_time = time.time()

    # 使用数据预取器（重要：加速数据加载）
    use_prefetcher = config.get("use_prefetcher", True) and torch.cuda.is_available()
    if use_prefetcher:
        prefetcher = DataPrefetcher(train_loader, device)
    else:
        train_iter = iter(train_loader)

    for batch in range(config["nBatch"]):
        model.train()

        # 数据加载（使用预取器加速）
        try:
            if use_prefetcher:
                batch_data = prefetcher.next()
                if batch_data is None:
                    prefetcher = DataPrefetcher(train_loader, device)
                    batch_data = prefetcher.next()
                # 使用prefetcher时，数据已经在GPU上了
                X, y, X_len, y_len, day_idx = batch_data
            else:
                try:
                    X, y, X_len, y_len, day_idx = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    X, y, X_len, y_len, day_idx = next(train_iter)
                
                # 不使用prefetcher时，需要手动转换到GPU
                X, y, X_len, y_len, day_idx = (
                    X.to(device),
                    y.to(device),
                    X_len.to(device),
                    y_len.to(device),
                    day_idx.to(device),
                )
        except Exception as e:
            continue

        # 数据增强
        if config["whiteNoiseSD"] > 0:
            X += torch.randn(X.shape, device=device) * config["whiteNoiseSD"]

        if config["constantOffsetSD"] > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=device)
                * config["constantOffsetSD"]
            )

        # 前向+反向传播
        pred = model.forward(X)
        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        optimizer.zero_grad()
        loss.backward()
        # grad clipping removed by design
        optimizer.step()
        scheduler.step()

        # 评估
        if batch % 100 == 0:
            avg_loss, cer = evaluate_model(model, test_loader, loss_ctc, device)
            
            end_time = time.time()
            print(
                f"batch {batch}, ctc loss: {avg_loss:>7f}, cer: {cer:>7f}, "
                f"time/batch: {(end_time - start_time)/100:>7.3f}"
            )
            start_time = time.time()

            # 保存最佳模型（首轮验证也必须落盘，否则可能出现从未更优导致无 checkpoint）
            improved = len(test_cer_history) == 0 or cer < np.min(test_cer_history)
            model_dir = config["pretrainedModelOutputPath"]
            os.makedirs(model_dir, exist_ok=True)
            if improved:
                torch.save(
                    model.state_dict(),
                    os.path.join(model_dir, "modelWeights.pth"),
                )
                trigger_times = 0
                best_test_cer = min(best_test_cer, cer)
            else:
                trigger_times += 1

            test_loss_history.append(avg_loss)
            test_cer_history.append(cer)

            # 保存训练统计
            if 'stats' not in locals():
                stats = {}
            stats["testLoss"] = np.array(test_loss_history)
            stats["testCER"] = np.array(test_cer_history)
            stats["bestCER"] = best_test_cer
            results_dir = os.path.join(config["pretrainedModelOutputPath"], "results")
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, "trainingStats.pkl"), "wb") as f:
                pickle.dump(stats, f)

            # 早停检查
            if trigger_times >= patience:
                print(f"### Early stopping triggered! Best CER: {best_test_cer:.4f} ###")
                break
    
    print("✨ Fine-tuning complete.")

    # 若训练全程未触发“更优”保存（旧逻辑漏洞），保证仍有权重可评估
    model_dir = config["pretrainedModelOutputPath"]
    weight_path = os.path.join(model_dir, "modelWeights.pth")
    if not os.path.isfile(weight_path):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), weight_path)
        print(f"⚠️ No best checkpoint was saved during training; wrote last-step weights to {weight_path}")
    
    # 评估微调后的模型并保存CER
    print("\n📊 Evaluating fine-tuned model...")
    from .evaluate import evaluate_cer
    try:
        finetuned_avg_loss, finetuned_cer = evaluate_cer(
            model_dir=config["pretrainedModelOutputPath"],
            dataset_path=config["finetuneDataPath"],
            batch_size=config["batchSize"],
            eval_split="test",
            device=device
        )
        
        # 保存微调后的CER到结果文件
        results_file = os.path.join(results_dir, "finetuned_cer.txt")
        with open(results_file, 'w') as f:
            f.write(f"Fine-tuned CER: {finetuned_cer:.6f}\n")
            f.write(f"Fine-tuned Loss: {finetuned_avg_loss:.6f}\n")
        
        print(f"✅ Fine-tuned CER: {finetuned_cer:.6f}")
        print(f"   Saved to: {results_file}")
        
        # 同时更新训练统计
        if 'stats' not in locals():
            stats = {
                "testLoss": np.array(test_loss_history),
                "testCER": np.array(test_cer_history),
                "bestCER": best_test_cer,
            }
        stats["finetunedCER"] = finetuned_cer
        stats["finetunedLoss"] = finetuned_avg_loss
        with open(os.path.join(results_dir, "trainingStats.pkl"), "wb") as f:
            pickle.dump(stats, f)
            
    except Exception as e:
        print(f"⚠️ Warning: Could not evaluate fine-tuned model: {e}")
        finetuned_cer = best_test_cer
    
    return best_test_cer, finetuned_cer
