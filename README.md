# Project3

基于主动获取（Active Acquisition）的小样本跨天脑电模型迁移方法。

## 项目简介

本项目面向跨天脑电语音解码场景，研究在标注样本有限时，如何通过样本选择策略提升模型迁移效果。核心思路是先使用预训练模型对候选样本打分，再按策略选择少量高价值样本进行微调，从而在控制标注成本的前提下提升目标天性能。

## 主要内容

- Model A 跨天迁移与微调流程
- 多种样本选择策略（如 `random`、`real_slpe`、`model_b` 等）
- 自动化批量实验脚本（按组、按天、按 seed 运行）
- 结果汇总与对比分析脚本

## 目录说明

- `scripts/`：主要训练、微调、评估与实验调度代码
- `scripts/src/model_a/`：Model A 相关实现
- `scripts/src/model_b/`：Model B 相关实现与打分模型
- `scripts/src/utils/`：数据与样本选择等公共工具
- `environment.yml`：复现实验的 Conda 环境配置

## 快速开始

1. 创建环境

```bash
conda env create -f environment.yml
conda activate env1
```

2. 查看主流程与参数

```bash
python3 scripts/main_pipeline.py --help
python3 scripts/launch_abcd_5seeds.py --help
```

3. 运行批量实验（示例）

```bash
python3 scripts/launch_abcd_5seeds.py --groups A B C D --dry_run
```

## 说明

- 为避免仓库体积过大，训练产物、日志和中间结果目录默认不纳入版本控制（见 `.gitignore`）。
- 如需复现实验，请确保本地已有对应数据，并按脚本参数指定路径。
