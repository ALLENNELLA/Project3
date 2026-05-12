# evaluate.py - 模型a的评估模块（计算CER）
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import datetime
import matplotlib
matplotlib.use('Agg')

from edit_distance import SequenceMatcher
from ..utils.dataset import SpeechDataset
from .get_model import get_model


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


def get_eval_dataset_loaders(dataset_path, batch_size, eval_split="test", num_workers=0):
    """加载评估数据集"""
    with open(dataset_path, "rb") as handle:
        loaded_data = pickle.load(handle)

    eval_ds = SpeechDataset(loaded_data[eval_split], transform=None) 

    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_padding,
    )

    return eval_loader, loaded_data


def load_trained_model_a(model_dir, device="cuda"):
    """从目录加载 Model A 权重与 config（供单次或多次数据集评估复用）。"""
    model_weight_path = os.path.join(model_dir, "modelWeights.pth")
    with open(os.path.join(model_dir, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    model = get_model(config).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    return model, config


def evaluate_cer_on_dataset(
    model,
    config,
    dataset_path,
    batch_size=64,
    eval_split="test",
    device="cuda",
    verbose=True,
):
    """
    对已加载的 Model A 在单个数据集文件上计算 CER（与 evaluate_cer 的指标一致）。
    """
    eval_loader, loaded_data = get_eval_dataset_loaders(
        dataset_path,
        batch_size,
        eval_split=eval_split,
    )

    n_days = len(loaded_data["train"])
    if verbose:
        print(f"✅ Dataset ({dataset_path}) contains {n_days} sessions for evaluation.")

    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    all_loss = []
    total_edit_distance = 0
    total_seq_length = 0

    with torch.no_grad():
        for X, y, X_len, y_len, day_idx in eval_loader:
            X, y, X_len, y_len, day_idx = (
                X.to(device),
                y.to(device),
                X_len.to(device),
                y_len.to(device),
                day_idx.to(device),
            )

            pred = model.forward(X)

            adjusted_lens_ctc = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)
            loss = loss_ctc(
                torch.permute(pred.log_softmax(2), [1, 0, 2]),
                y,
                adjusted_lens_ctc,
                y_len,
            )
            loss = torch.sum(loss)
            all_loss.append(loss.cpu().detach().numpy())

            for iter_idx in range(pred.shape[0]):
                adjusted_lens_seq = adjusted_lens_ctc[iter_idx].item()

                decoded_seq = torch.argmax(
                    pred[iter_idx, 0 : adjusted_lens_seq, :],
                    dim=-1,
                )

                decoded_seq = torch.unique_consecutive(decoded_seq, dim=-1)
                decoded_seq = decoded_seq.cpu().detach().numpy()

                decoded_seq = np.array([i for i in decoded_seq if i != 0])

                true_seq = y[iter_idx][0 : y_len[iter_idx]].cpu().detach().numpy()

                matcher = SequenceMatcher(
                    a=true_seq.tolist(), b=decoded_seq.tolist()
                )
                total_edit_distance += matcher.distance()
                total_seq_length += len(true_seq)

    avg_loss = np.sum(all_loss) / len(eval_loader)
    cer = total_edit_distance / total_seq_length if total_seq_length > 0 else 0

    if verbose:
        print("\n" + "=" * 50)
        print(f"🚀 Evaluation Results ({eval_split} split)")
        print(f"  - Total samples: {total_seq_length} phonemes")
        print(f"  - Average CTC Loss: {avg_loss:.6f}")
        print(f"  - **Character Error Rate (CER): {cer:.4f}**")
        print("=" * 50)

    return avg_loss, cer


def evaluate_cer(model_dir, dataset_path, batch_size=64, eval_split="test", device="cuda", verbose=True):
    """
    评估模型CER
    
    Args:
        model_dir: 模型目录（包含modelWeights.pth和config.pkl）
        dataset_path: 数据集路径
        batch_size: 批次大小
        eval_split: 评估的数据集分割（'train'或'test'）
        device: 设备
        verbose: 是否打印详细日志（批量评估时可设为 False）
    
    Returns:
        avg_loss: 平均损失
        cer: 字符错误率
    """
    model, config = load_trained_model_a(model_dir, device)
    if verbose:
        print(f"✅ Model loaded from {model_dir}")

    return evaluate_cer_on_dataset(
        model,
        config,
        dataset_path,
        batch_size=batch_size,
        eval_split=eval_split,
        device=device,
        verbose=verbose,
    )


def evaluate_across_days(model_dir, session_datasets, batch_size=64, device="cuda", output_path=None):
    """
    评估模型在训练最后一天和后续天的test表现，并绘制折线图
    
    Args:
        model_dir: 模型目录（包含modelWeights.pth和config.pkl）
        session_datasets: 字典，格式为 {session_name: dataset_path}
                         例如: {'t12.2022.05.24': '/path/to/data20220524', ...}
        batch_size: 批次大小
        device: 设备
        output_path: 图片保存路径（如果为None，则保存到model_dir）
    
    Returns:
        results: 评估结果列表，每个元素包含 {'session_name', 'date', 'cer', 'avg_loss', 'is_last_train_day'}
    """
    model, config = load_trained_model_a(model_dir, device)
    train_sessions = config.get("sessionNames_train", [])
    last_train_session = train_sessions[-1] if train_sessions else None

    print("="*80)
    print(f"🚀 跨天评估: 模型 {config.get('model_name', 'unknown')} (训练了 {config.get('nDays', 0)} 天)")
    print(f"最后训练天: {last_train_session}")
    print(f"✅ Model loaded from {model_dir}")
    print("="*80)
    
    results = []
    
    # 评估每个session
    for session_name, dataset_path in session_datasets.items():
        if not os.path.exists(dataset_path):
            print(f"⚠️ 跳过不存在的数据集: {session_name} -> {dataset_path}")
            continue
        
        is_last_train_day = (session_name == last_train_session)
        status = f" (DAY {config.get('nDays', 0)} / TRAIN)" if is_last_train_day else " (TEST SESSION)"
        
        print(f"\n评估 Session: {session_name}{status}")
        print(f"数据集路径: {dataset_path}")
        
        # 评估该session的test数据
        try:
            avg_loss, cer = evaluate_cer_on_dataset(
                model,
                config,
                dataset_path=dataset_path,
                batch_size=batch_size,
                eval_split="test",
                device=device,
                verbose=False,
            )
            print(f"  → CER: {cer:.4f} | avg CTC loss: {avg_loss:.6f}")
            
            # 从session_name提取日期
            parts = session_name.split('.')
            if len(parts) >= 3:
                date_str = f"{parts[-2]}.{parts[-1]}"
            else:
                date_str = session_name
            
            results.append({
                "session_name": session_name,
                "date": date_str,
                "cer": cer,
                "avg_loss": avg_loss,
                "is_last_train_day": is_last_train_day
            })
            
        except Exception as e:
            print(f"❌ 评估失败: {e}")
            continue
    
    # 打印汇总结果
    print("\n" + "="*80)
    print("📊 跨天评估汇总结果:")
    print("="*80)
    for item in results:
        tag = "[最后训练天]" if item['is_last_train_day'] else "[测试天]"
        print(f"{tag} {item['session_name']} | 日期: {item['date']} | CER: {item['cer']:.4f}")
    print("="*80)
    
    # 绘制折线图
    if results:
        plot_cer_across_days(config, results, output_path or model_dir)
    
    return results


def plot_cer_across_days(config, results, output_dir):
    """
    绘制CER跨天折线图
    
    Args:
        config: 模型配置
        results: 评估结果列表
        output_dir: 图片保存目录
    """
    # 提取日期和CER数据
    dates = []
    cers = []
    labels = []
    is_train_days = []
    
    for item in results:
        # 解析日期
        date_str = "2022." + item['date']  # 假设年份是2022
        try:
            date_obj = datetime.strptime(date_str, '%Y.%m.%d')
        except:
            # 如果解析失败，使用序号
            date_obj = datetime(2022, 1, 1)  # 默认日期
        
        dates.append(date_obj)
        cers.append(item['cer'])
        labels.append(item['session_name'])
        is_train_days.append(item['is_last_train_day'])
    
    # 根据日期排序
    sorted_data = sorted(zip(dates, cers, labels, is_train_days), key=lambda x: x[0])
    sorted_dates, sorted_cers, sorted_labels, sorted_is_train = zip(*sorted_data)
    
    # 创建图表
    plt.figure(figsize=(14, 8))
    
    # 绘制折线图，区分训练天和测试天
    train_dates = [d for d, is_train in zip(sorted_dates, sorted_is_train) if is_train]
    train_cers = [c for c, is_train in zip(sorted_cers, sorted_is_train) if is_train]
    test_dates = [d for d, is_train in zip(sorted_dates, sorted_is_train) if not is_train]
    test_cers = [c for c, is_train in zip(sorted_cers, sorted_is_train) if not is_train]
    
    # 绘制所有点的连线
    plt.plot(sorted_dates, sorted_cers, marker='o', linestyle='-', linewidth=2, 
             markersize=8, color='steelblue', label='CER')
    
    # 标记最后训练天
    if train_dates:
        plt.scatter(train_dates, train_cers, s=200, marker='*', color='red', 
                   zorder=5, label='Last Training Day', edgecolors='black', linewidths=1.5)
    
    # 标记测试天（使用点而不是方块）
    if test_dates:
        plt.scatter(test_dates, test_cers, s=150, marker='o', color='blue', 
                   zorder=5, label='Test Days', edgecolors='black', linewidths=1)
    
    # 设置标题和标签（使用英文）
    n_days = config.get('nDays', 0)
    model_name = config.get('model_name', 'unknown')
    plt.title(f'CER Across Days (Model: {model_name}, Trained on {n_days} days)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Character Error Rate (CER)', fontsize=12)
    
    # 在每个点显示CER值（使用英文格式）
    for date, cer, is_train in zip(sorted_dates, sorted_cers, sorted_is_train):
        cer_percent_str = f"{cer * 100:.1f}%"
        plt.text(
            date, 
            cer, 
            cer_percent_str,
            fontsize=9, 
            ha='left', 
            va='bottom',
            color='darkred',
            fontweight='bold'
        )
    
    # 设置X轴格式
    if len(sorted_dates) > 1:
        date_floats = mdates.date2num(sorted_dates)
        plt.gca().xaxis.set_major_locator(mticker.FixedLocator(date_floats))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m.%d'))
        plt.gcf().autofmt_xdate(rotation=45)
    
    # 设置Y轴范围（固定从0到1）
    plt.ylim(0, 1)
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    plt.legend(loc='best', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    save_filename = os.path.join(output_dir, f'CER_across_days_{model_name}_{n_days}days.png')
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ 图表已保存: {save_filename}")
    
    # 同时保存PDF版本
    pdf_filename = os.path.join(output_dir, f'CER_across_days_{model_name}_{n_days}days.pdf')
    plt.savefig(pdf_filename, bbox_inches='tight')
    print(f"✅ PDF版本已保存: {pdf_filename}")
    
    plt.close()
