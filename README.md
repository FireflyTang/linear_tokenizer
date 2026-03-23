# linear_tokenizer

无需加载模型权重的 LLM token 数线性近似估算器，支持 GLM-5 / DeepSeek-V3.2 / Kimi-K2.5 / MiniMax-M2.5。

## 方法

将文本中每个字符归入五个互斥类别，各乘以独立系数后求和：

```
tokens ≈ cjk × F_cjk + letter × F_letter + digit × F_digit + punct × F_punct + space × F_space
```

| 特征 | 字符范围 | 默认系数 |
|------|---------|---------|
| `cjk` | CJK 汉字、标点、全角字符 | 0.6256 |
| `letter` | ASCII 字母 `[A-Za-z]` | 0.1828 |
| `digit` | 数字 `[0-9]` | 1.0820 |
| `punct` | 标点/符号（其余字符） | 0.8003 |
| `space` | 空白 `\s` | 0.1184 |

系数通过四个模型的真实 tokenizer 在 261 条 Wikipedia + GitHub 语料（每条约 4000 字符）上联合最小二乘拟合得到。

## 基准测试结果

### 默认系数（拟合前）

| 模型 | MAPE |
|------|------|
| GLM-5 | 28.7% |
| Kimi-K2.5 | 29.1% |
| DeepSeek-V3.2 | 25.1% |
| MiniMax-M2.5 | 16.0% |

### 共用拟合系数（四模型联合 NNLS）

```python
Coeffs(cjk=0.6256, letter=0.1828, digit=1.0820, punct=0.8003, space=0.1184)
```

| 模型 | MAPE | 偏差方向 | 偏差比例 |
|------|------|---------|---------|
| GLM-5 | 8.0% | 高估 | +4.6% |
| Kimi-K2.5 | 8.0% | 高估 | +5.0% |
| DeepSeek-V3.2 | 5.9% | 高估 | +1.3% |
| MiniMax-M2.5 | 8.5% | 低估 | −6.6% |
| **平均** | **7.6%** | — | **≈0%** |

### 各模型独立拟合系数

| 模型 | cjk | letter | digit | punct | space | MAPE | 偏差比例 |
|------|-----|--------|-------|-------|-------|------|---------|
| GLM-5 | 0.6615 | 0.1840 | 1.2459 | 0.6745 | 0.1185 | 6.6% | +0.87% |
| Kimi-K2.5 | 0.6485 | 0.1847 | 1.0885 | 0.6958 | 0.1121 | 6.6% | +0.83% |
| DeepSeek-V3.2 | 0.5761 | 0.1785 | 1.1767 | 0.8226 | 0.1186 | 5.8% | +0.83% |
| MiniMax-M2.5 | 0.6162 | 0.1838 | 0.8170 | 1.0085 | 0.1244 | 4.9% | +0.45% |

各自系数下四个模型偏差均压到 1% 以内。

## 安装

```bash
pip install -r requirements.txt
# 可选（用于 --fit / --fit-shared）
pip install scipy numpy
```

## 快速使用

```python
from tokenizer_approx import estimate, Coeffs

# 共用系数（默认）
estimate("这是一段中英文混合的 sample text。")

# 使用各模型独立系数
KIMI_COEFFS = Coeffs(cjk=0.6485, letter=0.1847, digit=1.0885, punct=0.6958, space=0.1121)
estimate("text", coeffs=KIMI_COEFFS)
```

## 基准测试

```bash
# 合成短样本（无需下载）
python benchmark.py --samples 20 --fit-shared

# 真实 k 级别语料（首次运行自动下载并缓存）
python benchmark.py --real-data --fit-shared

# 各模型分别拟合
python benchmark.py --real-data --fit

# 指定模型子集
python benchmark.py --real-data --models kimi,dsv --fit-shared --csv out.csv
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `tokenizer_approx.py` | 核心估算模块，`estimate()` / `extract_features()` |
| `real_tokenizers.py` | 四个真实 tokenizer 的薄封装（懒加载，仅下载词表文件） |
| `benchmark.py` | 误差评估 + NNLS 系数拟合 |
| `data_fetch.py` | 从 Wikipedia / GitHub 下载真实语料并缓存 |
| `sample_gen.py` | 合成测试样本生成器（12 类别） |
| `results.csv` | 上次基准测试的逐样本结果 |
