# Model B 模块化框架

本目录包含重构后的 Model B 模块化框架，完全独立于 `scripts_old`。

## 文件结构

```
scripts/
├── model_b_data_module.py          # Model B数据获取模块
├── model_b_train_module.py          # Model B训练模块
├── model_b_test_module.py           # Model B测试模块（重合度计算）
├── finetune_module.py               # Model A微调模块
├── model_b_utils.py                 # Model B辅助工具模块
├── main_pipeline.py                 # 主入口文件
├── MODULE_FRAMEWORK.md              # 框架设计文档
└── README.md                        # 本文件
```

## 快速开始

### 0. 训练 Model A（如果需要）

```bash
conda activate env1
cd /root/25S151115/project3/scripts
python main_pipeline.py --mode train_model_a \
    --model_a_n_days 7 \
    --model_a_name conformer
```

批量并行（多 GPU、每卡多进程）：将各 `n_days` 扩展到多个 seed，输出仍在 `outputs/model_train/{model}-{n}days-seed{s}/`。

```bash
cd /root/25S151115/project3/scripts
# 从现有 model_train 目录推断有哪些 n_days，并补全 seed 0–39（已存在 modelWeights.pth 会跳过）
python run_model_a_train_parallel.py --discover_ndays --gpus 0 1 2 3 --per_gpu 1
# 仅指定天数与 40 个种子
python run_model_a_train_parallel.py --ndays 5 7 --num_seeds 40 --gpus 0 1 --per_gpu 2
```

### 1. 训练 Model B

```bash
conda activate env1
cd /root/25S151115/project3/scripts
python main_pipeline.py --mode train_only \
    --metric slpe \
    --train_days 6 7 \
    --pretrained_ndays_train_b 5 \
    --model_a_path_train_b /root/25S151115/project3/outputs/model_train/conformer-5days \
    --output_dir outputs/model_b_new
```

### 2. 测试 Model B（重合度分析）

```bash
conda activate env1
python main_pipeline.py --mode test_only \
    --model_b_path outputs/model_b_new \
    --metric slpe \
    --val_days 8 9 10 11 12 \
    --pretrained_ndays_eval 7 \
    --model_a_path_eval /root/25S151115/project3/outputs/model_train/conformer-7days
```

### 3. 微调 Model A

```bash
conda activate env1
python main_pipeline.py --mode finetune_only \
    --finetune_method model_b \
    --finetune_target_days 8 9 10 11 12\
    --num_samples 100 \
    --model_b_path outputs/model_b_new \
    --pretrained_ndays_eval 7 \
    --model_a_path_eval /root/25S151115/project3/outputs/model_train/conformer-7days

# length100: 按当天音素数量最多的100句进行微调
python main_pipeline.py --mode finetune_only \
    --finetune_method length \
    --finetune_target_days 8 9 10 11 12 \
    --num_samples 100 \
    --pretrained_ndays_eval 7 \
    --model_a_path_eval /root/25S151115/project3/outputs/model_train/conformer-7days

# strategy: 默认 hard_top100（最难100句）
python main_pipeline.py --mode finetune_only \
    --finetune_method real_slpe \
    --finetune_target_days 8 \
    --num_samples 100 \
    --selection_strategy hard_top100 \
    --model_a_path_eval /root/25S151115/project3/outputs/model_train/conformer-7days

# strategy: 分层随机（按SLPE降序切三段，hard/mid/easy），x+y+z必须=100
# ran_70_30_0 表示：难段随机70 + 中段随机30 + 易段随机0
python main_pipeline.py --mode finetune_only \
    --finetune_method model_b \
    --finetune_target_days 8 \
    --num_samples 100 \
    --selection_strategy ran_70_30_0 \
    --selection_seed 0 \
    --model_b_path outputs/model_b_new \
    --model_a_path_eval /root/25S151115/project3/outputs/model_train/conformer-7days
```

> 说明：`ran_x_y_z` 仅支持 `num_samples=100`；`selection_seed` 用于每次随机分层抽样复现。

### 3.1 批量调度器多 strategy（`launch_abcd_5seeds.py`）

```bash
python launch_abcd_5seeds.py \
  --variants model_b100 real_slpe100 \
  --strategy ran_70_30_0 ran_60_20_20 hard_top100 \
  --selection_seeds 0 1 2 3 4
```

> 对 `ran_x_y_z` 会按 `selection_seeds` 展开多任务；`hard_top100` 保持单任务。

### 4. 完整流程

```bash
# 完整流程（不训练Model A，使用已有模型）
conda activate env1
python main_pipeline.py --mode full \
    --metric slpe \
    --train_days 6 7 \
    --val_days 8 9 10 11 12 \
    --pretrained_ndays_train_b 5 \
    --pretrained_ndays_eval 7 \
    --model_a_path_train_b /root/25S151115/project3/outputs/model_train/conformer-5days \
    --model_a_path_eval /root/25S151115/project3/outputs/model_train/conformer-7days \
    --do_finetune \
    --finetune_method model_b \
    --finetune_target_days 8 9 10 \
    --num_samples 100

# 完整流程（先训练7天Model A；训练B仍使用已有5天Model A）
conda activate env1
python main_pipeline.py --mode full \
    --train_model_a_first \
    --model_a_n_days 7 \
    --metric slpe \
    --train_days 6 7 \
    --val_days 8 9 10 11 12 \
    --pretrained_ndays_train_b 5 \
    --pretrained_ndays_eval 7 \
    --model_a_path_train_b /root/25S151115/project3/outputs/model_train/conformer-5days \
    --do_finetune \
    --finetune_method model_b \
    --finetune_target_days 8 9 10 \
    --num_samples 100
```

> 说明（路径优先级）：
> 1. `--model_a_path_train_b` / `--model_a_path_eval` 优先级最高，分别控制训练B与评估/微调路径  
> 2. 若未单独传入，上述两者都会回退到 `--model_a_path`  
> 3. 若三者都未传，自动按天数生成：`conformer-{pretrained_ndays_train_b}days` 与 `conformer-{pretrained_ndays_eval}days`  
>
> 默认是 **双路径**：训练 Model B 用前5天 Model A；评测 8–12 天与微调用前7天 Model A。  
> 如果只传 `--pretrained_ndays N`，则 train 与 eval 都会使用 N 天。  
> `--train_model_a_first --model_a_n_days 7` 时，完整流程会自动复用新训练的 7 天模型作为评估/微调路径；训练 B 仍按 `--model_a_path_train_b` 或自动生成的 5 天路径执行。  
> `--model_b_path` 始终指向“用 6/7 天标签训练出来”的 Model B 目录（标签来自前5天 Model A），例如 `outputs/model_b/slpe-5days-seed0`。

## 模块说明

### model_a_train_module.py

训练Model A（脑电解码模型）。

主要函数：
- `train_model_a()`: 训练Model A
- `get_model_a_path()`: 获取Model A路径（根据训练参数）

### model_b_data_module.py

计算SLPE或CER分数，使用前N天的Model A计算。

主要函数：
- `compute_train_scores()`: 计算训练数据分数
- `compute_val_scores()`: 计算验证数据分数
- `compute_final_test_scores()`: 计算最终测试数据分数

### model_b_train_module.py

训练Model B模型。

主要函数：
- `train_model_b()`: 训练Model B
- `load_trained_model_b()`: 加载训练好的Model B

### model_b_test_module.py

计算Model B预测与真实分数的重合度。

主要函数：
- `compute_overlap_analysis()`: 计算重合度分析
- `compute_model_b_predictions()`: 使用Model B预测分数
- `compute_real_scores()`: 计算真实分数

### finetune_module.py

使用不同方法进行Model A微调。

主要函数：
- `finetune_model_a()`: 微调Model A
- `select_samples_for_finetune()`: 根据方法选择样本

### model_b_utils.py

提供各种辅助功能。

主要函数：
- `build_prompt()`: 构建prompt
- `normalize_scores()`: 归一化分数
- `filter_nan_samples()`: 过滤NaN样本
- `extract_transcriptions()`: 提取转录文本

### main_pipeline.py

统一入口，协调所有模块。

主要函数：
- `run_model_b_training_pipeline()`: 运行Model B训练流程
- `run_test_pipeline()`: 运行测试流程
- `run_finetune_pipeline()`: 运行微调流程
- `run_full_pipeline()`: 运行完整流程

## 依赖关系

所有模块只依赖：
- `src/` 目录下的代码
- `scripts/` 目录内的其他模块

**完全不依赖 `scripts_old/` 目录**

## 注意事项

1. 数据路径：默认使用 `/root/25S151115/project3/data/` 下的数据
2. 模型路径：默认使用 `/root/25S151115/project3/outputs/` 下的模型
3. 缓存：SLPE分数会缓存到 `outputs/slpe_scores/` 目录
4. 输出：所有结果保存到 `outputs/` 目录

## 详细文档

请参考 `MODULE_FRAMEWORK.md` 了解详细的框架设计。
