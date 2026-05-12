# trainer.py - 模型a的训练器
import os
import pickle
import time
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from edit_distance import SequenceMatcher
from ..utils.dataset import SpeechDataset
from .get_model import get_model


class DataPrefetcher:
    """GPU数据预取器，在GPU计算时提前加载下一批数据"""
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.iter = iter(loader)
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.iter)
        except StopIteration:
            self.batch = None
            return
        
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.batch = tuple(
                    item.to(self.device, non_blocking=True) 
                    for item in self.batch
                )
        else:
            self.batch = tuple(item.to(self.device) for item in self.batch)

    def next(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            self.preload()
        return batch

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.next()
        if batch is None:
            raise StopIteration
        return batch


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


def get_dataset_loaders(dataset_path, batch_size, num_workers=6):
    """创建训练和测试数据加载器"""
    with open(dataset_path, "rb") as handle:
        loaded_data = pickle.load(handle)

    train_ds = SpeechDataset(loaded_data["train"], transform=None)
    test_ds = SpeechDataset(loaded_data["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_padding,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(2, num_workers // 3),
        pin_memory=True,
        collate_fn=_padding,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, test_loader, loaded_data


def evaluate_model(model, test_loader, loss_ctc, device):
    """评估模型性能"""
    model.eval()
    all_loss = []
    total_edit_distance = 0
    total_seq_length = 0
    
    with torch.no_grad():
        for X, y, X_len, y_len, test_day_idx in test_loader:
            X, y, X_len, y_len, test_day_idx = (
                X.to(device),
                y.to(device),
                X_len.to(device),
                y_len.to(device),
                test_day_idx.to(device),
            )

            pred = model.forward(X)
            loss = loss_ctc(
                torch.permute(pred.log_softmax(2), [1, 0, 2]),
                y,
                ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                y_len,
            )
            all_loss.append(loss.sum().cpu().item())

            adjusted_lens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)
            for iter_idx in range(pred.shape[0]):
                decoded_seq = torch.argmax(
                    pred[iter_idx, 0:adjusted_lens[iter_idx], :], 
                    dim=-1
                )
                decoded_seq = torch.unique_consecutive(decoded_seq, dim=-1)
                decoded_seq = decoded_seq.cpu().numpy()
                decoded_seq = np.array([i for i in decoded_seq if i != 0])

                true_seq = y[iter_idx][0:y_len[iter_idx]].cpu().numpy()

                matcher = SequenceMatcher(
                    a=true_seq.tolist(), 
                    b=decoded_seq.tolist()
                )
                total_edit_distance += matcher.distance()
                total_seq_length += len(true_seq)

    avg_loss = sum(all_loss) / len(test_loader)
    cer = total_edit_distance / total_seq_length if total_seq_length > 0 else 0
    
    return avg_loss, cer


def train_model(config):
    """训练模型"""
    os.makedirs(config["outputDir"], exist_ok=True)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    device = config["device"]

    # 保存配置
    with open(os.path.join(config["outputDir"], "config.pkl"), "wb") as f:
        pickle.dump(config, f)

    # 加载数据
    num_workers = config.get("num_workers", 6)
    train_loader, test_loader, _ = get_dataset_loaders(
        config["datasetPath"],
        config["batchSize"],
        num_workers=num_workers,
    )

    # 构建模型
    model = get_model(config)
    print(f"✅ Model created: {config['model_name']}")
    
    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lrStart"],
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=config["l2_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=config["lrEnd"] / config["lrStart"],
        total_iters=config["nBatch"],
    )

    # 训练准备
    test_loss_history = []
    test_cer_history = []
    patience = config.get("patience", 10)
    trigger_times = 0
    best_test_cer = np.inf
    best_epoch = 0

    # 使用数据预取器（重要：加速数据加载）
    use_prefetcher = config.get("use_prefetcher", True) and torch.cuda.is_available()
    if use_prefetcher:
        prefetcher = DataPrefetcher(train_loader, device)
    else:
        train_iter = iter(train_loader)

    # 训练循环
    num_epochs = config["nBatch"] // 100
    
    for epoch in range(num_epochs):
        pbar = tqdm(
            range(100), 
            desc=f"Epoch {epoch+1}/{num_epochs}",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        epoch_losses = []
        epoch_start_time = time.time()
        
        for step in pbar:
            batch_idx = epoch * 100 + step
            model.train()

            # 数据加载（使用预取器加速）
            try:
                if use_prefetcher:
                    batch_data = prefetcher.next()
                    if batch_data is None:
                        prefetcher = DataPrefetcher(train_loader, device)
                        batch_data = prefetcher.next()
                    # 使用prefetcher时，数据已经在GPU上了，无需再次转换
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
                X = X + torch.randn_like(X) * config["whiteNoiseSD"]

            if config["constantOffsetSD"] > 0:
                X = X + (
                    torch.randn(X.shape[0], 1, X.shape[2], device=device)
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
            loss = loss.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'avg_loss': f'{np.mean(epoch_losses):.3f}'
            })

        # Epoch结束后评估
        epoch_time = time.time() - epoch_start_time
        avg_loss, cer = evaluate_model(model, test_loader, loss_ctc, device)
        
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f} | CER: {cer:.4f} | Time: {epoch_time:.1f}s")
        
        # 保存最佳模型
        if cer < best_test_cer:
            best_test_cer = cer
            best_epoch = epoch + 1
            model_dir = config["outputDir"]
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, "modelWeights.pth"))
            print(f"✓ Best model saved (CER: {cer:.4f})")
            trigger_times = 0
        else:
            trigger_times += 1
        
        test_loss_history.append(avg_loss)
        test_cer_history.append(cer)

        # 保存训练统计
        stats = {
            "testLoss": np.array(test_loss_history),
            "testCER": np.array(test_cer_history),
            "bestEpoch": best_epoch,
            "bestCER": best_test_cer,
        }
        results_dir = os.path.join(config["outputDir"], "results")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "trainingStats.pkl"), "wb") as f:
            pickle.dump(stats, f)

        # 早停检查
        if trigger_times >= patience:
            print(f"\nEarly stopping! Best CER: {best_test_cer:.4f} at Epoch {best_epoch}")
            break
    
    print(f"\n✅ Training Complete! Best CER: {best_test_cer:.4f} at Epoch {best_epoch}")
    return model


def load_model(model_dir, device="cuda"):
    """加载训练好的模型"""
    model_weight_path = os.path.join(model_dir, "modelWeights.pth")
    with open(os.path.join(model_dir, "config.pkl"), "rb") as f:
        config = pickle.load(f)

    model = get_model(config).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    
    return model, config
