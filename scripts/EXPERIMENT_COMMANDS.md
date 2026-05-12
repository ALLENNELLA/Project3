## 实验命令与参数修改速查表

> 建议：所有命令默认在 `env1` 环境、`project3/scripts` 目录下运行  
> ```bash
> conda activate env1
> cd /root/25S151115/project3/scripts
> ```

---

### 一、基础：Model A / Model B / 基本微调命令

- **1. 训练 Model A（Conformer，前 7 天）**

```bash
cd /root/25S151115/project3/scripts
conda activate env1

python main_pipeline.py \
  --mode train_model_a \
  --model_a_n_days 7 \
  --model_a_name conformer \
  --base_dir /root/25S151115/project3 \
  --seed 0
```

训练结果：`outputs/model_train/conformer-7days-seed0/`

---

- **2. 训练 Model B（SLPE 预测）**

> 一般只需要跑一次 5days→(6,7) 的训练，脚本里已经封装好。

```bash
cd /root/25S151115/project3/scripts
conda activate env1

python main_pipeline.py \
  --mode train_only \
  --metric slpe \
  --train_days 6 7 \
  --pretrained_ndays 5 \
  --base_dir /root/25S151115/project3
```

训练输出目录示例（每个 seed 一份）：  
`scripts/outputs/model_b/slpe-5days-seed{seed}/`

---

- **3. 单次经典微调（不带 PEFT）**

以 **real_slpe** 方法、7→8 天为例：

```bash
cd /root/25S151115/project3/scripts
conda activate env1

python main_pipeline.py \
  --mode finetune_only \
  --finetune_method real_slpe \
  --finetune_target_days 8 \
  --num_samples 100 \
  --pretrained_ndays 7 \
  --model_a_path /root/25S151115/project3/outputs/model_train/conformer-7days-seed0 \
  --base_dir /root/25S151115/project3 \
  --seed 0
```

输出目录示例：  
`outputs/model_test/7-8/real_slpe/seed0/`

---

### 二、使用 Model B 进行选样 + 微调

- **1. 使用 Model B 进行样本选择并微调（7→8 天）**

假设：
- Model A：`outputs/model_train/conformer-7days-seed0`
- Model B：`scripts/outputs/model_b/slpe-5days-seed0`

```bash
cd /root/25S151115/project3/scripts
conda activate env1

python main_pipeline.py \
  --mode finetune_only \
  --finetune_method model_b \
  --finetune_target_days 8 \
  --num_samples 100 \
  --pretrained_ndays 7 \
  --model_a_path /root/25S151115/project3/outputs/model_train/conformer-7days-seed0 \
  --model_b_path /root/25S151115/project3/scripts/outputs/model_b/slpe-5days-seed0 \
  --base_dir /root/25S151115/project3 \
  --seed 0
```

输出目录：  
`outputs/model_test/7-8/model_b/seed0/`

---

- **2. random 方法（指定 selection_seed）**

```bash
python main_pipeline.py \
  --mode finetune_only \
  --finetune_method random \
  --finetune_target_days 8 \
  --num_samples 100 \
  --pretrained_ndays 7 \
  --model_a_path /root/25S151115/project3/outputs/model_train/conformer-7days-seed0 \
  --base_dir /root/25S151115/project3 \
  --seed 0 \
  --selection_seed 3
```

输出目录：  
`outputs/model_test/7-8/random/seed0_sel3/`

---

### 三、PEFT 微调（CABlock + AdaptFFN）

#### 1. 关键配置参数位置

- **模型结构与 PEFT 模块定义**
  - 文件：`scripts/src/model_a/models/Conformer.py`
  - 相关内容：
    - `class CABlock`
    - `class AdaptFFN`
    - `class ConformerDecoder(..., use_adapter=False, adapter_bottleneck=64, use_ca_block=False, ca_bottleneck=64, ...)`

- **默认配置（非 PEFT 实验时的默认开关/瓶颈）**
  - 文件：`scripts/src/model_a/config.py`
  - 函数：`get_model_config('conformer')`
  - 这里可以设置全局默认：
    - `use_adapter` / `adapter_bottleneck`
    - `use_ca_block` / `ca_bottleneck`

- **自动化 PEFT 实验专用配置（从旧实验复用 config.pkl）**
  - 文件：`scripts/run_peft_from_saved_indices.py`
  - 函数：`prepare_peft_config(...)` 中：
    - **是否开启 PEFT：**
      - `config["use_adapter"] = True`
      - `config["use_ca_block"] = True`
    - **PEFT 瓶颈大小（你现在调到 512 的地方）：**
      - `config["adapter_bottleneck"] = 512`
      - `config["ca_bottleneck"] = 512`

> **以后如果想改 PEFT 容量（bottleneck 大小）**：  
> 只要改 `run_peft_from_saved_indices.py` 里这两行即可，所有新生成的 `config_peft.pkl` 都会自动更新。

---

#### 2. 单次 PEFT 微调（手动指定 config.pkl）

如果你已经有一个 `config_peft.pkl`（例如自动脚本生成的），可以直接用：

```bash
cd /root/25S151115/project3/scripts
conda activate env1

CUDA_VISIBLE_DEVICES=4 python run_finetune_from_config.py \
  --config_path /root/25S151115/project3/outputs/model_test/7-8/real_slpe_peft/seed0/config_peft.pkl
```

说明：
- `CUDA_VISIBLE_DEVICES=4` 控制使用哪一张物理 GPU
- `config_peft.pkl` 里已经写好了：
  - `use_adapter = True`
  - `use_ca_block = True`
  - `adapter_bottleneck = 512`
  - `ca_bottleneck = 512`

---

#### 3. 批量 PEFT 微调（复用旧 selected_indices）

统一从旧实验的 `config.pkl` 读出 `selected_indices`，生成新的 `config_peft.pkl`，然后批量微调。

- **脚本：** `scripts/run_peft_from_saved_indices.py`
- **核心作用：**
  - 遍历：`outputs/model_test/7-{day}/{method}/seed*/config.pkl`
  - 保留原来的 `selected_indices` 和超参数
  - 覆盖 / 增加：
    - `pretrainedModelOutputPath` → 写到新的 `*_peft` 目录
    - `use_adapter = True`, `use_ca_block = True`
    - `adapter_bottleneck = 512`, `ca_bottleneck = 512`
  - 调用 `run_finetune_from_config.py` 在指定 GPU 上跑

**推荐运行方式（GPU 2-6，每卡 1 个任务）：**

```bash
cd /root/25S151115/project3/scripts
conda activate env1

CUDA_VISIBLE_DEVICES=2,3,4,5,6 python run_peft_from_saved_indices.py \
  --days 8 9 10 11 12 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --num_gpus 5 \
  --gpu_start 2 \
  > outputs/automated_experiments/peft_bottleneck512_gpu2-6.log 2>&1 &
```

参数说明：
- `--days`：目标微调天（例如 `8 9 10 11 12`）
- `--seeds`：使用哪些 model A 的 seed（0–9）
- `--num_gpus`：脚本内部的“逻辑 GPU 数”（这里设 5）
- `--gpu_start`：起始物理 GPU ID（2 → 实际使用 2,3,4,5,6）
- 脚本内部已经保证：
  - 每个逻辑 GPU 上 `Semaphore(1)` → **每张卡只跑一个任务**

**相关日志：**
- 主控日志：`scripts/outputs/automated_experiments/peft_bottleneck512_gpu2-6.log`
- 单任务日志：`scripts/outputs/automated_experiments/logs/finetune_{method}_7_{day}_seed{seed}[_sel{sel}]_peft.log`

---

### 四、常用参数修改位置一览

- **1. Conformer / PEFT 结构本身**
  - `scripts/src/model_a/models/Conformer.py`
    - `CABlock` / `AdaptFFN` 的网络结构、激活函数、`init_scale`、`dropout`
    - `ConformerBlock(..., use_ca_block=False, ca_bottleneck=64, ...)`
    - `ConformerDecoder(..., use_adapter=False, adapter_bottleneck=64, use_ca_block=False, ca_bottleneck=64, ...)`

- **2. 默认模型配置（不通过 config.pkl 时的默认值）**
  - `scripts/src/model_a/config.py`
    - `get_model_config('conformer')` 里可以改：
      - `use_adapter` / `adapter_bottleneck`
      - `use_ca_block` / `ca_bottleneck`

- **3. 自动化实验（非 PEFT，原始 random / real_slpe / model_b）**
  - `scripts/run_automated_experiments.py`
    - 任务生成逻辑（哪些天、哪些 seeds、哪些方法）
    - GPU 数控制：命令行参数 `--num_gpus`、`--max_workers`
  - `scripts/finetune_module.py`
    - `finetune_model_a`：封装了 `finetune.py` 的调用与日志写入

- **4. PEFT 自动实验（使用旧 selected_indices）**
  - `scripts/run_peft_from_saved_indices.py`
    - 控制：
      - 使用哪些 `days`、`seeds`
      - GPU 映射与并行度
      - **PEFT 开关与 bottleneck 大小（当前统一设为 512）**

---

### 五、快速 FAQ

- **Q: 想只改 bottleneck，但不重写其他配置，应该改哪里？**  
  **A:** 改 `run_peft_from_saved_indices.py` 里的：
  - `config["adapter_bottleneck"] = 512`
  - `config["ca_bottleneck"] = 512`

- **Q: 想完全关闭 PEFT，跑一版纯 Conformer baseline？**  
  **A:** 确保 config 或默认配置中：
  - `use_adapter = False`
  - `use_ca_block = False`
  然后用 `main_pipeline.py --mode finetune_only` 或原来的 `run_automated_experiments.py` 即可。

- **Q: 想手动用某个 seed 的 config 重跑一次微调？**  
  **A:** 用：

```bash
CUDA_VISIBLE_DEVICES=4 python run_finetune_from_config.py \
  --config_path /root/25S151115/project3/outputs/model_test/7-8/real_slpe_peft/seed0/config_peft.pkl
```

以后如果你有新的需求（比如只跑某几天、只用某几个 seed），可以在这个文档基础上再追加对应命令。 

---

### 六、并行批量运行所有实验 & 结果查看

这一节主要回答四个问题：
- **怎么一次性并行跑完“原始实验”（real_slpe / model_b / random）？**
- **怎么选用哪些 GPU？**
- **怎么控制并行度（每张卡/总共同时跑多少任务）？**
- **结果到哪里看（日志 + CER）？**

---

#### 1. 并行批量运行“原始实验”（不带 PEFT）

批量实验入口脚本：`scripts/run_automated_experiments.py`

- **完整大规模实验（重跑全部）示例：**

```bash
cd /root/25S151115/project3/scripts
conda activate env1

# 假设你想用物理 GPU 0,1,2,3 共 4 张卡
export CUDA_VISIBLE_DEVICES=0,1,2,3

python run_automated_experiments.py \
  --num_gpus 4 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --pretrained_ndays 5 7 \
  --finetune_days 8 9 10 11 12 \
  --num_samples 100 \
  --max_workers 4
```

说明：
- `CUDA_VISIBLE_DEVICES=0,1,2,3`
  - 进程里看到的逻辑 GPU 为 0,1,2,3
  - `run_automated_experiments.py` 里的 `gpu_id` 就是对这 4 张卡取模后的结果
- `--num_gpus 4`
  - **逻辑 GPU 数量**，必须 ≤ `CUDA_VISIBLE_DEVICES` 的个数
- `--max_workers 4`
  - **最大并行任务数**，默认等于 `num_gpus`
  - 一般设成和 `num_gpus` 一样 → 每张卡一个任务
  - 如果显存紧张，可以设小一点，例如 `--max_workers 2`（总共同时只跑 2 个任务，其余任务排队）

> **如果 Model A 已经训练好，只想重跑微调**  
> 加上 `--skip_model_a`，脚本会自动检测已存在的 Model A 权重并标记为“已完成”，只跑微调部分：
>
> ```bash
> python run_automated_experiments.py \
>   --num_gpus 4 \
>   --skip_model_a
> ```

---

#### 2. 怎么选择 GPU & 并行度（原始 + PEFT 通用）

- **选择使用哪些物理卡：**
  - 通过环境变量 `CUDA_VISIBLE_DEVICES` 控制
  - 例如：
    - `export CUDA_VISIBLE_DEVICES=2` → 只用物理 GPU2（在脚本内部记作逻辑 GPU0）
    - `export CUDA_VISIBLE_DEVICES=2,4,5` → 只用物理 2、4、5 三张卡（逻辑 ID 为 0,1,2）

- **`run_automated_experiments.py` 中的 GPU 分配规则：**
  - 逻辑 GPU 数量 = `num_gpus`
  - 每次提交任务时，从 `0 .. num_gpus-1` 里找一张 **空闲 GPU** 分配
  - 再乘上 `CUDA_VISIBLE_DEVICES` 的映射，即可定位到物理卡

- **并行度控制：**
  - **原始实验（run_automated_experiments.py）**
    - `--max_workers`：ThreadPoolExecutor 的最大并行任务数
    - 通常设成 `max_workers = num_gpus` → 每张卡 1 个任务
    - 如果你希望“少任务但更稳定”，可以：
      - `CUDA_VISIBLE_DEVICES=2,3,4,5,6`
      - `--num_gpus 5 --max_workers 3`（5 张卡里最多只跑 3 个任务）
  - **PEFT 实验（run_peft_from_saved_indices.py）**
    - 外层：`--num_gpus` + `--gpu_start` 负责逻辑 GPU 到物理 GPU 的映射
    - 内部已经固定：每个逻辑 GPU 一个信号量 `Semaphore(1)` → **每张卡最多 1 个任务**
    - 因此你只需要选好卡即可：
      - `CUDA_VISIBLE_DEVICES=2,3,4,5,6`
      - `--num_gpus 5 --gpu_start 2`

---

#### 3. 一次性跑完“全部原先实验”的推荐命令

如果你想重现“全套 baseline + 三种微调方法”的设置，可以用：

```bash
cd /root/25S151115/project3/scripts
conda activate env1

export CUDA_VISIBLE_DEVICES=0,1,2,3  # 按你当前机器情况改

python run_automated_experiments.py \
  --num_gpus 4 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --pretrained_ndays 5 7 \
  --finetune_days 8 9 10 11 12 \
  --num_samples 100 \
  --max_workers 4
```

运行内容包括：
- **Model A 训练：**
  - 5 天模型：`outputs/model_train/conformer-5days-seed{seed}`
  - 7 天模型：`outputs/model_train/conformer-7days-seed{seed}`
- **Model B 训练：**
  - `scripts/outputs/model_b/slpe-5days-seed{seed}`
- **微调：**
  - `outputs/model_test/7-{day}/real_slpe/seed{seed}`
  - `outputs/model_test/7-{day}/model_b/seed{seed}`
  - `outputs/model_test/7-{day}/random/seed{seed}_sel{sel}`

如果只想**重跑微调**（假设所有 Model A / Model B 都已经存在），可以：

```bash
python run_automated_experiments.py \
  --num_gpus 4 \
  --skip_model_a
```

---

#### 4. 批量 PEFT 实验的并行运行（复习 + 汇总）

已经在“三、PEFT 微调”中给出，这里只做一次集中版本：

- **bottleneck=512，GPU 2–6，每卡 1 任务：**

```bash
cd /root/25S151115/project3/scripts
conda activate env1

export CUDA_VISIBLE_DEVICES=2,3,4,5,6

python run_peft_from_saved_indices.py \
  --days 8 9 10 11 12 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --num_gpus 5 \
  --gpu_start 2 \
  > outputs/automated_experiments/peft_bottleneck512_gpu2-6.log 2>&1 &
```

---

#### 5. 结果查看：日志 + CER 汇总

- **1）自动化实验的日志（原始实验 + PEFT）**

  - 主控日志：
    - 原始自动化：`scripts/outputs/automated_experiments/experiment_main.log`
    - PEFT 批量：`scripts/outputs/automated_experiments/peft_*.log`
  - 单任务日志：
    - 统一放在：`scripts/outputs/automated_experiments/logs/`
    - 命名规则类似：
      - 原始微调：`finetune_{method}_{pretrained_ndays}_{day}_seed{seed}_run{run_id}.log`
      - 带 selection_seed 的 random：`..._sel{sel}.log`
      - PEFT：`finetune_{method}_7_{day}_seed{seed}[_sel{sel}]_peft.log`

- **2）单个实验的 CER / Loss**

  - 每次微调结束后，会在对应的 `model_test` 目录里写入：
    - `outputs/model_test/7-{day}/{method}[_peft]/seed{seed}[_sel{sel}]/results/finetuned_cer.txt`
  - 内容类似：
    - `Fine-tuned CER: 0.329768`
    - `Fine-tuned Loss: 2.331248`

  - 你之前统计 Day 8 的 CER 就是从这里读出来的：

```bash
cd /root/25S151115/project3

python - << 'EOF'
import os, glob, statistics as stats

base = 'outputs/model_test/7-8'
methods = ['real_slpe', 'model_b', 'real_slpe_peft', 'model_b_peft']

for method in methods:
    cers = []
    for seed_dir in sorted(glob.glob(os.path.join(base, method, 'seed*'))):
        cer_file = os.path.join(seed_dir, 'results', 'finetuned_cer.txt')
        if not os.path.exists(cer_file):
            continue
        with open(cer_file) as f:
            for line in f:
                if 'Fine-tuned CER' in line:
                    cers.append(float(line.split(':')[-1]))
                    break
    if not cers:
        continue
    print(f'方法 {method}: n={len(cers)}, mean={stats.mean(cers):.4f}, std={stats.pstdev(cers):.4f}')
EOF
```

- **3）全局结果汇总文件**

  顶层 `outputs/` 目录下已经有若干你之前生成的汇总：
  - `outputs/finetune_results_summary.pkl / .md`
  - `outputs/parallel_finetune_results.pkl`
  - `outputs/results_day8_12_current.md` 等

  这些文件通常是你之前写的分析脚本输出的，如果后续要扩展，只需要在分析脚本里继续读取上述 `finetuned_cer.txt` 即可。

