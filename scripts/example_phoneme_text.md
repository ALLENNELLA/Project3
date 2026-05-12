# 音素序列和文本标签示例

## 示例 1: "hello world"

### 音素序列（ID格式）
```python
phoneme_seq = [1, 11, 21, 21, 25, 0, 0, 0, 23, 25, 18, 21, 3, 0, 0, 0]
# 对应音素: HH EH L L OW SIL SIL SIL W OW R L D SIL SIL SIL
```

### 音素序列（文本格式，移除padding后）
```
HH EH L L OW W OW R L D
```

### 文本标签
```
transcription = "hello world"
```

### 最终Prompt（instruction格式）
```
任务:预测脑电解码错误率 [SEP] 难点:关注相似音素和长序列 [SEP] 音素: HH EH L L OW W OW R L D [SEP] 文本: hello world
```

---

## 示例 2: "the quick brown fox"

### 音素序列（ID格式）
```python
phoneme_seq = [31, 10, 0, 20, 20, 6, 1, 9, 0, 2, 18, 25, 23, 0, 15, 25, 24, 0, 0, 0]
# 对应音素: TH AH SIL K W IH K B R OW N SIL F OW K S SIL SIL SIL
```

### 音素序列（文本格式，移除padding后）
```
TH AH K W IH K B R OW N F OW K S
```

### 文本标签
```
transcription = "the quick brown fox"
```

### 最终Prompt（instruction格式）
```
任务:预测脑电解码错误率 [SEP] 难点:关注相似音素和长序列 [SEP] 音素: TH AH K W IH K B R OW N F OW K S [SEP] 文本: the quick brown fox
```

---

## 示例 3: "how are you"

### 音素序列（ID格式）
```python
phoneme_seq = [15, 25, 23, 0, 1, 18, 11, 0, 34, 25, 0, 0, 0, 0]
# 对应音素: HH OW W SIL AA R EH SIL Y OW SIL SIL SIL SIL
```

### 音素序列（文本格式，移除padding后）
```
HH OW W AA R EH Y OW
```

### 文本标签
```
transcription = "how are you"
```

### 最终Prompt（instruction格式）
```
任务:预测脑电解码错误率 [SEP] 难点:关注相似音素和长序列 [SEP] 音素: HH OW W AA R EH Y OW [SEP] 文本: how are you
```

---

## 音素ID映射表

| ID | 音素 | 说明 |
|----|------|------|
| 0  | SIL  | 静音/padding |
| 1  | AA   | 如 "father" 中的 a |
| 2  | AE   | 如 "cat" 中的 a |
| 3  | AH   | 如 "but" 中的 u |
| 4  | AO   | 如 "law" 中的 aw |
| 5  | AW   | 如 "cow" 中的 ow |
| 6  | AY   | 如 "hide" 中的 i |
| 7  | B    | 如 "be" 中的 b |
| 8  | CH   | 如 "cheese" 中的 ch |
| 9  | D    | 如 "deed" 中的 d |
| 10 | DH   | 如 "the" 中的 th |
| 11 | EH   | 如 "red" 中的 e |
| 12 | ER   | 如 "hurt" 中的 ur |
| 13 | EY   | 如 "ate" 中的 ay |
| 14 | F    | 如 "fee" 中的 f |
| 15 | G    | 如 "green" 中的 g |
| 16 | HH   | 如 "he" 中的 h |
| 17 | IH   | 如 "it" 中的 i |
| 18 | IY   | 如 "eat" 中的 ee |
| 19 | JH   | 如 "jeep" 中的 j |
| 20 | K    | 如 "key" 中的 k |
| 21 | L    | 如 "lee" 中的 l |
| 22 | M    | 如 "me" 中的 m |
| 23 | N    | 如 "knee" 中的 n |
| 24 | NG   | 如 "sing" 中的 ng |
| 25 | OW   | 如 "show" 中的 o |
| 26 | OY   | 如 "toy" 中的 oy |
| 27 | P    | 如 "pea" 中的 p |
| 28 | R    | 如 "read" 中的 r |
| 29 | S    | 如 "sea" 中的 s |
| 30 | SH   | 如 "she" 中的 sh |
| 31 | T    | 如 "tea" 中的 t |
| 32 | TH   | 如 "theta" 中的 th |
| 33 | UH   | 如 "hood" 中的 u |
| 34 | UW   | 如 "two" 中的 oo |
| 35 | V    | 如 "vee" 中的 v |
| 36 | W    | 如 "we" 中的 w |
| 37 | Y    | 如 "yield" 中的 y |
| 38 | Z    | 如 "zoo" 中的 z |
| 39 | ZH   | 如 "azure" 中的 zh |

---

## 使用说明

1. **音素序列（ID格式）**：是一个整数数组，每个整数对应一个音素ID
2. **音素序列（文本格式）**：将ID转换为音素符号，并用空格连接
3. **文本标签**：对应的自然语言转录文本
4. **最终Prompt**：使用 `instruction` 格式，包含任务描述、难点提示、音素序列和文本标签

## Python代码示例

```python
from src.model_b.utils.phoneme_converter import phoneme_seq_to_text
from model_b_utils import build_prompt

# 示例音素序列（ID格式）
phoneme_seq = [15, 25, 23, 0, 1, 18, 11, 0, 34, 25, 0, 0, 0, 0]

# 转换为文本格式
phoneme_text = phoneme_seq_to_text(phoneme_seq, remove_padding=True)
print(f"音素文本: {phoneme_text}")
# 输出: HH OW W AA R EH Y OW

# 文本标签
transcription = "how are you"

# 构建prompt
prompt = build_prompt(phoneme_seq, transcription, prompt_format='instruction')
print(f"Prompt: {prompt}")
# 输出: 任务:预测脑电解码错误率 [SEP] 难点:关注相似音素和长序列 [SEP] 音素: HH OW W AA R EH Y OW [SEP] 文本: how are you
```
