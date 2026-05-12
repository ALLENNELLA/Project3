# 自动化实验脚本使用说明

## 概述

`run_automated_experiments.py` 是一个自动化实验脚本，可以：
- 多GPU并行执行实验
- 自动管理任务依赖关系
- 支持多seed实验（0-9）
- 自动训练Model A（前5天和前7天）
- 自动训练Model B（基于SLPE）
- 自动执行微调实验（real_slpe, model_b, random）

## 实验设计

### 实验任务

1. **Model A训练**（每个seed）
   - 前5天模型：`conformer-5days-seed{seed}`
   - 前7天模型：`conformer-7days-seed{seed}`

2. **Model B训练**（每个seed，基于SLPE）
   - 基于5天Model A：`slpe-5days-seed{seed}`
   - 基于7天Model A：`slpe-7days-seed{seed}`

3. **微调实验**（每个seed，每个目标天数）
   - **real_slpe**: 每个seed跑1次
   - **model_b**: 每个seed跑1次
   - **random**: 每个seed跑3次

### 实验规模

- **Seeds**: 0-9（10个）
- **预训练天数**: 5, 7（2个）
- **微调目标天数**: 8, 9, 10, 11, 12（5个）
- **总任务数**:
  - Model A训练: 10 seeds × 2 days = 20个
  - Model B训练: 10 seeds × 2 days = 20个
  - 微调任务: 10 seeds × 2 pretrained_days × 5 target_days × (1+1+3) methods = 500个
  - **总计**: 540个任务

## 使用方法

### 基本用法

```bash
conda activate env1
cd /root/25S151115/project3/scripts

# 使用默认配置（自动检测GPU数量，seeds 0-9）
python run_automated_experiments.py
```

### 自定义配置

```bash
# 指定GPU数量
python run_automated_experiments.py --num_gpus 4

# 指定seeds
python run_automated_experiments.py --seeds 0 1 2 3 4

# 指定预训练天数
python run_automated_experiments.py --pretrained_ndays 5 7

# 指定微调目标天数
python run_automated_experiments.py --finetune_days 8 9 10

# 指定样本数量
python run_automated_experiments.py --num_samples 100

# 组合使用
python run_automated_experiments.py \
    --num_gpus 4 \
    --seeds 0 1 2 3 4 \
    --pretrained_ndays 7 \
    --finetune_days 8 9 10 11 12 \
    --num_samples 100
```

### 完整参数列表

```bash
python run_automated_experiments.py --help
```

参数说明：
- `--base_dir`: 项目基础目录（默认: `/root/25S151115/project3`）
- `--scripts_dir`: 脚本目录（默认: `base_dir/scripts`）
- `--num_gpus`: 使用的GPU数量（默认: 自动检测）
- `--seeds`: 实验种子列表（默认: 0-9）
- `--pretrained_ndays`: 预训练天数列表（默认: 5, 7）
- `--finetune_days`: 微调目标天数（默认: 8, 9, 10, 11, 12）
- `--num_samples`: 每个微调实验选择的样本数（默认: 100）
- `--max_workers`: 最大并行任务数（默认: GPU数量）

## 输出结构

### 实验日志

所有实验日志保存在：
```
scripts/outputs/automated_experiments/
├── experiment_log_YYYYMMDD_HHMMSS.json      # 详细实验日志
├── experiment_summary_YYYYMMDD_HHMMSS.json  # 实验总结
└── *.log                                     # 各任务的执行日志
```

### Model A输出

```
outputs/model_train/
├── conformer-5days-seed0/
├── conformer-5days-seed1/
├── ...
├── conformer-7days-seed0/
└── conformer-7days-seed1/
```

### Model B输出

```
scripts/outputs/model_b/
├── slpe-5days-seed0/
├── slpe-5days-seed1/
├── ...
├── slpe-7days-seed0/
└── slpe-7days-seed1/
```

### 微调输出

```
outputs/model_test/
├── 5-8/
│   ├── real_slpe/seed0/
│   ├── real_slpe/seed1/
│   ├── model_b/seed0/
│   ├── model_b/seed1/
│   ├── random/seed0_run0/
│   ├── random/seed0_run1/
│   ├── random/seed0_run2/
│   └── ...
├── 5-9/
├── ...
├── 7-8/
└── ...
```

## 任务依赖关系

脚本会自动处理任务依赖：

1. **Model A训练** → 无依赖
2. **Model B训练** → 依赖对应的Model A训练完成
3. **微调（real_slpe）** → 依赖对应的Model A训练完成
4. **微调（model_b）** → 依赖对应的Model A和Model B训练完成
5. **微调（random）** → 依赖对应的Model A训练完成

## 运行监控

### 实时监控

脚本运行时会实时输出：
- 任务提交信息
- 任务完成状态（✅成功 / ❌失败）
- GPU使用情况

### 查看日志

```bash
# 查看特定任务的日志
tail -f scripts/outputs/automated_experiments/train_model_a_7_seed0_gpu0.log

# 查看所有失败的实验
grep -r "❌" scripts/outputs/automated_experiments/*.log
```

### 查看实验总结

```bash
# 查看最新的实验总结
cat scripts/outputs/automated_experiments/experiment_summary_*.json | tail -1 | python -m json.tool
```

## 注意事项

1. **磁盘空间**: 确保有足够的磁盘空间（每个实验会生成模型文件）
2. **运行时间**: 完整实验可能需要数天时间，建议使用screen或tmux
3. **GPU内存**: 确保每个GPU有足够的内存
4. **中断恢复**: 脚本不支持断点续传，如果中断需要重新运行（已完成的任务会跳过）

## 使用screen运行（推荐）

```bash
# 创建screen会话
screen -S automated_experiments

# 运行脚本
conda activate env1
cd /root/25S151115/project3/scripts
python run_automated_experiments.py

# 分离会话（按 Ctrl+A 然后 D）
# 重新连接会话
screen -r automated_experiments
```

## 故障排除

### GPU不可用

```bash
# 检查GPU状态
nvidia-smi

# 如果GPU被占用，可以指定使用特定GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
python run_automated_experiments.py --num_gpus 4
```

### 任务失败

1. 查看对应任务的日志文件
2. 检查依赖任务是否成功完成
3. 检查磁盘空间和内存
4. 手动重新运行失败的任务

### 内存不足

- 减少并行任务数：`--max_workers 2`
- 减少GPU数量：`--num_gpus 2`

## 示例：小规模测试

如果想先测试小规模实验：

```bash
# 只运行seed 0，只训练7天模型，只微调day 8
python run_automated_experiments.py \
    --seeds 0 \
    --pretrained_ndays 7 \
    --finetune_days 8 \
    --num_gpus 1
```

这会生成：
- 1个Model A训练任务
- 1个Model B训练任务
- 5个微调任务（real_slpe×1 + model_b×1 + random×3）
- 总计7个任务
