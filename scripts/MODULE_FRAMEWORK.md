任务描述：我现在在做跨天ECoG2Sentence小样本迁移任务。我的想法是，把脑电解码模型记为Model A，把样本选择模型记为Model B。我先训练Model A，根据Model A计算出的一些指标来训练Model B，再使用Model B只根据标签预测这些指标（这里模拟的是实际场景，我们选择标签，让患者去读并录制脑电，再微调模型），选择指标好的一部分微调Model A。

实验范式：我们先训练前N-K天的模型A1，然后用模型A1计算N-K+1天到N天数据的指标（字符错误率cer或者句子预测熵slpe）。之后我们使用N-K+1天到N天数据及其指标训练模型B。我们会进行一些辅助，比如构造prompt、对cer或者slpe进行排序归一化。之后做到N+1 到 N+K天的迁移。首先我们会先看模型B和前N天训练的模型A2计算的结果（真实结果）的符合度，比方说前100\50难样本的重合率。然后我们选择X句进行微调（可以根据模型B选的句子、或者按照真实标签或者随机），看一下最终的微调结果。

目前测试的参数：N=7 K=2 K=5 X=100

因为工程量较大，模块较多，我希望能进行模块化设计。希望能把模型A训练模块、指标计算模块、模型B训练模块、模型B验证模块（N+1 到 N+K天，与真实标签比）、模型A微调模块都独立出来。并提供借口进行递进地调用。

# Model B 模块化框架设计

## 文件结构

```
scripts/
├── model_b_data_module.py          # Model B数据获取模块
├── model_b_train_module.py          # Model B训练模块
├── model_b_test_module.py           # Model B测试模块（重合度计算）
├── finetune_module.py               # Model A微调模块
├── model_b_utils.py                 # Model B辅助工具模块
└── main_pipeline.py                 # 主入口文件
```

---

## 1. model_b_data_module.py - Model B数据获取模块

**功能**：计算SLPE或CER分数，使用前N天的Model A计算

### 主要函数：

```python
def compute_scores_for_model_b(
    model_a_path: str,
    dataset_path: str,
    metric: str = 'slpe',  # 'slpe' or 'cer'
    batch_size: int = 32,
    device: str = 'cuda',
    cache_dir: Optional[str] = None,
    save_cache: bool = True,
    days: Optional[List[int]] = None,
    data_type: str = 'train'  # 'train' or 'test'
) -> Tuple[np.ndarray, List, np.ndarray]:
    """
    计算用于训练Model B的分数（SLPE或CER）
    
    Returns:
        scores: 分数数组（SLPE或CER）
        phoneme_seqs: 音素序列列表
        day_indices: 天数索引数组
    """

def compute_train_scores(
    model_a_path: str,
    data_path: str,
    train_days: List[int],
    metric: str = 'slpe',
    pretrained_ndays: int = 5,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    device: str = 'cuda'
) -> Tuple[np.ndarray, List, np.ndarray]:
    """
    计算训练数据的分数（day 6-7的train数据）
    """

def compute_val_scores(
    model_a_path: str,
    data_path: str,
    train_days: List[int],
    metric: str = 'slpe',
    pretrained_ndays: int = 5,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    device: str = 'cuda'
) -> Tuple[np.ndarray, List, np.ndarray]:
    """
    计算验证数据的分数（day 6-7的test数据）
    """

def compute_final_test_scores(
    model_a_path: str,
    val_days: List[int],
    metric: str = 'slpe',
    pretrained_ndays: int = 7,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    device: str = 'cuda',
    data_type: str = 'train'  # 最终测试用train数据
) -> Dict[int, Tuple[np.ndarray, List, np.ndarray]]:
    """
    计算最终测试数据的分数（day 8-12的train数据）
    
    Returns:
        Dict[day, (scores, phoneme_seqs, day_indices)]
    """

# 辅助函数
def _load_model_a(model_a_path: str, device: str) -> torch.nn.Module
def _load_dataset(dataset_path: str, data_type: str) -> SpeechDataset
def _compute_slpe_scores(...) -> Tuple[np.ndarray, List, np.ndarray]
def _compute_cer_scores(...) -> Tuple[np.ndarray, List, np.ndarray]
def _get_cache_file_path(...) -> str
def _load_from_cache(...) -> Tuple[np.ndarray, List, np.ndarray]
def _save_to_cache(...) -> None
```

---

## 2. model_b_train_module.py - Model B训练模块

**功能**：训练Model B模型

### 主要函数：

```python
def train_model_b(
    train_scores: np.ndarray,
    train_phoneme_seqs: List,
    val_scores: np.ndarray,
    val_phoneme_seqs: List,
    train_transcriptions: Optional[List] = None,
    val_transcriptions: Optional[List] = None,
    model_name: str = 'roberta-base',
    model_type: str = 'roberta',
    output_dir: str = 'outputs/model_b',
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    normalize_scores: str = 'rank',
    use_ranking_loss: bool = True,
    use_mse_as_auxiliary: bool = False,
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    head_learning_rate: float = 1e-3,
    device: str = 'cuda',
    seed: int = 42
) -> Dict:
    """
    训练Model B模型
    
    Args:
        train_scores: 训练集分数（SLPE或CER）
        train_phoneme_seqs: 训练集音素序列
        val_scores: 验证集分数
        val_phoneme_seqs: 验证集音素序列
        train_transcriptions: 训练集转录文本（用于prompt）
        val_transcriptions: 验证集转录文本（用于prompt）
        ... (其他训练参数)
    
    Returns:
        Dict包含：
            - best_model_path: 最佳模型路径
            - train_history: 训练历史
            - val_history: 验证历史
            - best_epoch: 最佳epoch
            - best_spearman: 最佳Spearman相关系数
    """

def load_trained_model_b(
    model_path: str,
    model_name: str = 'roberta-base',
    model_type: str = 'roberta',
    device: str = 'cuda'
) -> torch.nn.Module:
    """
    加载训练好的Model B
    """

# 辅助函数
def _create_datasets(...) -> Tuple[Dataset, Dataset]
def _create_model(...) -> torch.nn.Module
def _train_epoch(...) -> Dict
def _evaluate(...) -> Tuple[Dict, np.ndarray, np.ndarray]
```

---

## 3. model_b_test_module.py - Model B测试模块

**功能**：计算Model A前7天模型预测8-12天的CER或SLPE的top50、100的重合度

### 主要函数：

```python
def compute_overlap_analysis(
    model_a_path: str,
    val_days: List[int],  # [8, 9, 10, 11, 12]
    metric: str = 'slpe',  # 'slpe' or 'cer'
    pretrained_ndays: int = 7,
    top_k_list: List[int] = [50, 100],
    batch_size: int = 32,
    device: str = 'cuda',
    cache_dir: Optional[str] = None
) -> Dict:
    """
    计算重合度分析
    
    Returns:
        Dict包含：
            - overlap_results: Dict[day, Dict[top_k, overlap_rate]]
            - model_b_predictions: Dict[day, np.ndarray]  # Model B预测的分数
            - real_scores: Dict[day, np.ndarray]  # 真实分数（SLPE或CER）
            - model_b_top_indices: Dict[day, Dict[top_k, List[int]]]
            - real_top_indices: Dict[day, Dict[top_k, List[int]]]
    """

def compute_model_b_predictions(
    model_b_path: str,
    val_days: List[int],
    data_path: str,
    batch_size: int = 32,
    device: str = 'cuda'
) -> Dict[int, np.ndarray]:
    """
    使用Model B预测8-12天的分数
    """

def compute_real_scores(
    model_a_path: str,
    val_days: List[int],
    metric: str = 'slpe',
    pretrained_ndays: int = 7,
    batch_size: int = 32,
    device: str = 'cuda',
    cache_dir: Optional[str] = None,
    data_type: str = 'train'
) -> Dict[int, np.ndarray]:
    """
    使用Model A计算8-12天的真实分数
    """

def calculate_overlap(
    indices1: List[int],
    indices2: List[int]
) -> float:
    """
    计算两个索引列表的重合度
    """

# 辅助函数
def _get_top_k_indices(scores: np.ndarray, k: int) -> List[int]
def _aggregate_overlap_results(...) -> Dict
```

---

## 4. finetune_module.py - 微调模块

**功能**：使用不同方法进行微调（随机、length、model b、real cer、real slpe）

### 主要函数：

```python
def finetune_model_a(
    method: str,  # 'random', 'length', 'model_b', 'real_cer', 'real_slpe'
    model_a_path: str,
    target_day: int,
    num_samples: int,
    model_b_path: Optional[str] = None,  # method='model_b'时需要
    pretrained_ndays: int = 7,
    batch_size: int = 32,
    device: str = 'cuda',
    output_dir: Optional[str] = None,
    **finetune_kwargs
) -> Dict:
    """
    微调Model A
    
    Args:
        method: 样本选择方法
        model_a_path: Model A路径
        target_day: 目标天数
        num_samples: 选择的样本数量
        model_b_path: Model B路径（method='model_b'时需要）
        ... (其他微调参数)
    
    Returns:
        Dict包含：
            - finetuned_model_path: 微调后的模型路径
            - selected_indices: 选择的样本索引
            - metrics: 微调后的指标
    """

def select_samples_for_finetune(
    method: str,
    dataset_path: str,
    num_samples: int,
    model_a_path: Optional[str] = None,
    model_b_path: Optional[str] = None,
    day: Optional[int] = None,
    pretrained_ndays: int = 7,
    seed: int = 42,
    cache_dir: Optional[str] = None
) -> List[int]:
    """
    根据方法选择样本
    
    Returns:
        选中的样本索引列表
    """

def random_sample_selection(
    dataset_path: str,
    num_samples: int,
    seed: int = 42
) -> List[int]:
    """
    随机选择样本
    """

def model_b_sample_selection(
    dataset_path: str,
    model_b_path: str,
    num_samples: int,
    selection_strategy: str = 'hard_top100'  # 默认最难100；也支持 ran_x_y_z
) -> List[int]:
    """
    使用Model B选择样本
    """

def real_cer_sample_selection(
    dataset_path: str,
    model_a_path: str,
    num_samples: int,
    selection_strategy: str = 'hard',
    batch_size: int = 32,
    device: str = 'cuda',
    cache_dir: Optional[str] = None
) -> List[int]:
    """
    使用真实CER选择样本
    """

def real_slpe_sample_selection(
    dataset_path: str,
    model_a_path: str,
    num_samples: int,
    selection_strategy: str = 'hard_top100',
    batch_size: int = 32,
    device: str = 'cuda',
    cache_dir: Optional[str] = None,
    day: Optional[int] = None,
    pretrained_ndays: int = 7
) -> List[int]:
    """
    使用真实SLPE选择样本
    """

# 辅助函数
def _run_finetune(...) -> Dict
def _save_finetune_results(...) -> None
```

---

## 5. model_b_utils.py - Model B辅助工具模块

**功能**：提供各种辅助功能，包括prompt构建、音素序列转换、文本处理等

### 主要函数：

```python
def build_prompt(
    phoneme_seq: List[int] or np.ndarray,
    transcription: Optional[str] = None,
    prompt_format: str = 'combined'  # 'phoneme_only', 'combined', 'instruction'
) -> str:
    """
    构建prompt，结合音素序列和自然语言转录
    
    Args:
        phoneme_seq: 音素ID序列
        transcription: 自然语言转录文本（可选）
        prompt_format: prompt格式
            - 'phoneme_only': 只使用音素序列
            - 'combined': 音素序列 + 转录文本
            - 'instruction': 带指令的格式
    
    Returns:
        prompt字符串
    """

def phoneme_seq_to_text(
    phoneme_seq: List[int] or np.ndarray,
    remove_padding: bool = True
) -> str:
    """
    将音素ID序列转换为文本格式
    
    Args:
        phoneme_seq: 音素ID序列
        remove_padding: 是否移除padding（索引0）
    
    Returns:
        音素文本序列，如 "AA AE AH"
    """

def extract_transcriptions(
    data_split: List[Dict],
    field_name: str = 'transcriptions'
) -> List[Optional[str]]:
    """
    从数据中提取转录文本
    
    Args:
        data_split: 数据分割（如data['train']），是一个列表，每个元素是一天的数据
        field_name: 转录字段名称
    
    Returns:
        transcriptions列表，顺序与SpeechDataset一致
    """

def normalize_scores(
    scores: np.ndarray,
    day_indices: np.ndarray,
    method: str = 'rank'  # 'rank', 'minmax', 'zscore', 'quantile', 'none'
) -> np.ndarray:
    """
    归一化分数（按天分别归一化或全局归一化）
    
    Args:
        scores: 分数数组
        day_indices: 天数索引数组
        method: 归一化方法
            - 'rank': 排序归一化（按天分别）
            - 'minmax': 最小最大归一化（按天分别）
            - 'zscore': Z-score归一化（按天分别）
            - 'quantile': 分位数归一化（全局）
            - 'none': 不归一化
    
    Returns:
        归一化后的分数数组
    """

def filter_nan_samples(
    scores: np.ndarray,
    phoneme_seqs: List,
    day_indices: np.ndarray,
    transcriptions: Optional[List] = None
) -> Tuple[np.ndarray, List, np.ndarray, Optional[List]]:
    """
    过滤包含NaN的异常样本
    
    Returns:
        (过滤后的scores, phoneme_seqs, day_indices, transcriptions)
    """

def create_phoneme_text_dataset(
    phoneme_texts: List[str],
    scores: np.ndarray,
    tokenizer,
    max_length: int = 512,
    text_contents: Optional[List[str]] = None
) -> Dataset:
    """
    创建用于Model B训练的数据集（包含prompt）
    
    Args:
        phoneme_texts: 音素文本列表
        scores: 分数数组
        tokenizer: tokenizer
        max_length: 最大长度
        text_contents: 转录文本列表（用于构建prompt）
    
    Returns:
        Dataset对象
    """

def format_prompt_instruction(
    phoneme_text: str,
    transcription: Optional[str] = None
) -> str:
    """
    格式化带指令的prompt
    
    Returns:
        格式化的prompt字符串
    """

def build_combined_prompt(
    phoneme_text: str,
    transcription: Optional[str] = None
) -> str:
    """
    构建组合prompt（音素 + 转录）
    
    Returns:
        组合后的prompt字符串
    """

# 辅助函数
def _get_phoneme_vocab() -> List[str]
def _validate_phoneme_seq(...) -> bool
```

---

## 6. main_pipeline.py - 主入口文件

**功能**：统一入口，协调所有模块

### 主要函数：

```python
def main():
    """
    主函数，解析参数并执行相应流程
    """

def run_model_b_training_pipeline(
    metric: str = 'slpe',  # 'slpe' or 'cer'
    train_days: List[int] = [6, 7],
    val_days: List[int] = [8, 9, 10, 11, 12],
    pretrained_ndays: int = 5,  # 用于计算分数的Model A天数
    model_a_path: Optional[str] = None,
    data_path: Optional[str] = None,
    output_dir: str = 'outputs/model_b',
    **train_kwargs
) -> Dict:
    """
    运行Model B训练流程
    
    Returns:
        Dict包含训练结果和模型路径
    """

def run_test_pipeline(
    model_b_path: str,
    metric: str = 'slpe',
    val_days: List[int] = [8, 9, 10, 11, 12],
    pretrained_ndays: int = 7,
    top_k_list: List[int] = [50, 100],
    **test_kwargs
) -> Dict:
    """
    运行测试流程（重合度分析）
    """

def run_finetune_pipeline(
    method: str,
    target_days: List[int],
    num_samples: int,
    metric: str = 'slpe',
    model_b_path: Optional[str] = None,
    pretrained_ndays: int = 7,
    **finetune_kwargs
) -> Dict:
    """
    运行微调流程
    """

def run_full_pipeline(
    do_finetune: bool = False,
    metric: str = 'slpe',
    train_days: List[int] = [6, 7],
    val_days: List[int] = [8, 9, 10, 11, 12],
    pretrained_ndays: int = 5,
    finetune_method: Optional[str] = None,
    finetune_target_days: Optional[List[int]] = None,
    num_samples: int = 100,
    **kwargs
) -> Dict:
    """
    运行完整流程
    
    Args:
        do_finetune: 是否进行微调
        metric: 使用的指标（'slpe'或'cer'）
        train_days: 训练Model B使用的天数
        val_days: 验证/测试使用的天数
        pretrained_ndays: 用于计算分数的Model A预训练天数
        finetune_method: 微调方法（'random', 'length', 'model_b', 'real_cer', 'real_slpe'）
        finetune_target_days: 微调目标天数
        num_samples: 微调时选择的样本数量
    """
```

---

## 命令行参数设计

```python
parser.add_argument('--mode', type=str, choices=['train_only', 'test_only', 'finetune_only', 'full'],
                   default='train_only', help='运行模式')
parser.add_argument('--metric', type=str, choices=['slpe', 'cer'], default='slpe',
                   help='使用的指标')
parser.add_argument('--train_days', type=int, nargs='+', default=[6, 7],
                   help='训练Model B使用的天数')
parser.add_argument('--val_days', type=int, nargs='+', default=[8, 9, 10, 11, 12],
                   help='验证/测试使用的天数')
parser.add_argument('--pretrained_ndays', type=int, default=5,
                   help='用于计算分数的Model A预训练天数')
parser.add_argument('--do_finetune', action='store_true',
                   help='是否进行微调')
parser.add_argument('--finetune_method', type=str, choices=['random', 'length', 'model_b', 'real_cer', 'real_slpe'],
                   help='微调方法')
parser.add_argument('--finetune_target_days', type=int, nargs='+',
                   help='微调目标天数')
parser.add_argument('--num_samples', type=int, default=100,
                   help='微调时选择的样本数量')
parser.add_argument('--seed', type=int, default=None,
                   help='微调训练随机种子；random/ran_x_y_z 且提供 selection_seed 时会自动对齐为 selection_seed')
parser.add_argument('--selection_seed', type=int, default=None,
                   help='选样随机种子（random 与 ran_x_y_z 使用）；生效时会同步作为微调 seed')
parser.add_argument('--selection_strategy', type=str, default='hard_top100',
                   help='选样策略：hard_top100 或 ran_x_y_z（x+y+z=100）')
```

---

## 数据流

```
1. Model B训练流程：
   model_b_data_module.compute_train_scores() 
   → model_b_utils.phoneme_seq_to_text()  # 转换音素序列
   → model_b_utils.extract_transcriptions()  # 提取转录文本
   → model_b_utils.build_prompt()  # 构建prompt
   → model_b_utils.normalize_scores()  # 归一化分数
   → model_b_data_module.compute_val_scores()
   → model_b_utils.build_prompt()  # 构建验证集prompt
   → model_b_train_module.train_model_b()
   → 保存模型

2. 测试流程：
   model_b_test_module.compute_model_b_predictions()
   → model_b_utils.build_prompt()  # 构建预测用的prompt
   → model_b_test_module.compute_real_scores()
   → model_b_test_module.compute_overlap_analysis()
   → 返回重合度结果

3. 微调流程：
   finetune_module.select_samples_for_finetune()
   → finetune_module.finetune_model_a()
   → 保存微调后的模型

4. 完整流程：
   run_model_b_training_pipeline()
   → run_test_pipeline()
   → (可选) run_finetune_pipeline()
```

## Prompt格式示例

### 1. phoneme_only（仅音素）
```
"AA AE AH B CH D"
```

### 2. combined（音素 + 转录）
```
"Phonemes: AA AE AH B CH D\nTranscription: The cat sat"
```

### 3. instruction（带指令）
```
"Given the phoneme sequence: AA AE AH B CH D\nPredict the difficulty score for this sample.\nTranscription: The cat sat"
```

或更简洁的格式：
```
"Phonemes: AA AE AH B CH D\nText: The cat sat\nScore:"
```

---

## 模块间依赖关系

```
main_pipeline.py
├── model_b_data_module.py
│   └── model_b_utils.py (用于数据处理)
├── model_b_train_module.py
│   └── model_b_utils.py (用于prompt构建、数据准备)
├── model_b_test_module.py
│   ├── model_b_train_module.py (加载Model B)
│   └── model_b_utils.py (用于prompt构建)
└── finetune_module.py
    ├── model_b_train_module.py (使用Model B)
    └── model_b_utils.py (用于样本选择)
```

## 配置文件

可以考虑添加 `config.py` 统一管理配置：

```python
# config.py
MODEL_B_CONFIG = {
    'default_model_name': 'roberta-base',
    'default_batch_size': 16,
    'default_num_epochs': 10,
    'cache_base_dir': 'outputs/scores_cache',
    'model_b_output_dir': 'outputs/model_b',
    'finetune_output_dir': 'outputs/finetuned_models'
}
```
