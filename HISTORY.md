# 算法演进历史

本文档记录了从 naive 基线到当前 LP 上界方案的完整研发过程，包含每个阶段的实现、数据集、实测结果和结论。所有实验代码均保留在仓库中作为证据。

---

## 阶段 0：Naive 基线（ASCII/5 + 非ASCII/2）

**实现**: `tokenizer_approx.py::estimate_naive()`

**思路**: 简单规则：ASCII 字符（英文字母、数字、标点）平均约 5 个字符合一个 token；非 ASCII 字符（汉字等）平均约 2 个字符合一个 token。

```python
def estimate_naive(text: str) -> int:
    ascii_count = sum(1 for c in text if ord(c) < 128)
    non_ascii_count = len(text) - ascii_count
    return max(1, round(ascii_count / 5 + non_ascii_count / 2))
```

**数据集**: 5 类中英文合成数据（pure_chinese, pure_english, chat_zh, chat_en, mixed），100 样本/类

**结果**（4 模型平均）:

| 统计量 | ASCII/5 + 非ASCII/2 |
|--------|---------------------|
| 保证率（不低估） | ~55%（**不满足要求**） |
| 最大低估 | −45% |
| 最大高估 | +189% |
| MAPE | ~35% |

**结论**: 两个方向误差都很大，且无法保证"绝对高估"。被否定。

---

## 阶段 1：5 特征 NNLS（无 word 计数）

**实现**: `tokenizer_approx.py::estimate_fast_noword()`、`estimate_fast()`

**思路**: 把文本按字符类别分成 5 个互斥类别（cjk / letter / digit / punct / space），各乘独立系数求和。系数用 NNLS（非负最小二乘）在真实 tokenizer 数据上拟合。

**特征定义**:
- `cjk`: U+4E00–U+9FFF（汉字）+ U+3000–U+303F（CJK 标点）+ U+FF00–U+FFEF（全角）
- `letter`: ASCII 字母 `[A-Za-z]`
- `digit`: 数字 `[0-9]`
- `punct`: 其余所有字符（符号、emoji、西里尔字母等）
- `space`: 空白 `[ \t\n\r]`

**数据集**: 261 条真实语料（Wikipedia 19 篇 + GitHub 35+ 文件，每条约 4000 字符）

**拟合结果**（共用系数，4 模型 stacked NNLS）:

```
Coeffs(cjk=0.6256, letter=0.1828, digit=1.0820, punct=0.8003, space=0.1184)
```

**评估**:

| 模型 | MAPE | 偏差 |
|------|------|------|
| GLM-5 | 8.0% | +4.6% |
| Kimi-K2.5 | 8.0% | +5.0% |
| DeepSeek-V3.2 | 5.9% | +1.3% |
| MiniMax-M2.5 | 8.5% | −6.6% |
| **平均** | **7.6%** | ≈0% |

**结论**: 精度尚可，但 NNLS 最小化平方误差，允许低估。不满足"绝对高估"需求，但已比 naive 好很多。

---

## 阶段 2：6 特征 NNLS（加入 word 计数）

**实现**: `tokenizer_approx.py::estimate()`（主函数）、`estimate_numpy_full()`、`estimate_ctypes()`

**思路**: 在 5 特征基础上加第 6 个特征 `word`（空格切词数），捕捉 BPE 词边界行为——英文文本中 token 数与词数的相关性比与字母数更紧密。

**各实现方式**（同一算法，不同后端）:

| 实现 | 文件位置 | 特点 |
|------|----------|------|
| `estimate()` | `tokenizer_approx.py` | 主函数，自动选 C 扩展或 regex |
| `estimate_numpy_full()` | `tokenizer_approx.py:163` | 全 NumPy，word 从 space→非space 跃变计数 |
| `estimate_ctypes()` | `tokenizer_approx.py:192` | array 模块 + memoryview，无 NumPy |
| `estimate_fast()` | `tokenizer_approx.py:122` | NumPy + text.split() 词计数 |

**拟合结果（DEFAULT_COEFFS，当前默认系数）**:

```python
DEFAULT_COEFFS = Coeffs(
    cjk    = 0.6330,
    letter = 0.1406,
    digit  = 0.7876,
    punct  = 0.7115,
    space  = 0.0995,
    word   = 0.3633,
)
```

**各模型独立拟合系数**:

| 模型 | cjk | letter | digit | punct | space | word | MAPE |
|------|-----|--------|-------|-------|-------|------|------|
| GLM-5 | 0.6615 | 0.1840 | 1.2459 | 0.6745 | 0.1185 | — | 6.6% |
| Kimi-K2.5 | 0.6485 | 0.1847 | 1.0885 | 0.6958 | 0.1121 | — | 6.6% |
| DeepSeek-V3.2 | 0.5761 | 0.1785 | 1.1767 | 0.8226 | 0.1186 | — | 5.8% |
| MiniMax-M2.5 | 0.6162 | 0.1838 | 0.8170 | 1.0085 | 0.1244 | — | 4.9% |

**结论**: word 特征在英文文本上有小幅提升（MAPE ~7.6% → ~7.0%），差异不大。仍不保证上界，但 NNLS 系数用作 DEFAULT_COEFFS 保留至今。

---

## 阶段 3：性能优化实验

**背景**: 在业务中要高频调用 token 估算，需要极低延迟。测试了各种实现的性能。

**实验代码**: `_bench_numba.py`（Numba JIT 对比）

**测试方法**: 文本长度 5K→160K token，每种方法独立计时（多次重复取平均）。

**实测结果（160K token 输入）**:

| 实现 | 耗时 | vs tiktoken |
|------|------|-------------|
| HF transformers Kimi（Python 封装，`is_fast=False`） | ~700ms | 0.07× |
| HF transformers GLM/DSV/MiniMax（Rust 后端） | ~4–6ms | ~8× |
| tiktoken cl100k_base（OpenAI Rust 实现） | 46ms | 1× |
| NumPy 全特征 `estimate_numpy_full()` | 4.6ms | **10×** |
| Numba JIT `_bench_numba.py` | ~3ms | **15×** |
| C 扩展 `_features.pyd` | 1.49ms | **43×** |

**关键发现**:
- Kimi-K2.5 内部使用 tiktoken cl100k_base，因此 tiktoken 可作为 Kimi 的精确快速代理
- HF transformers 的 Kimi 包装是纯 Python（`is_fast=False`），比 tiktoken 慢 15 倍
- C 扩展直接访问 CPython 内部 Unicode buffer（PEP 393），无内存拷贝，是最快方案

**C 扩展实现**: `_features.c`，使用 `PyUnicode_KIND` / `PyUnicode_DATA` / `PyUnicode_READ`，单次遍历完成全部 6 特征统计。

**结论**: C 扩展为最终主力后端，自动降级到 regex 方案（`_USE_C` 标志）。

---

## 阶段 4：扩展 24 类数据集 + LP 上界（失败案例）

**背景**: 为了保证"绝对高估"，改用 LP（线性规划）拟合系数。

**LP 问题形式**:
```
最小化  c · Σᵢ fᵢ          （最小化总高估）
约束    fᵢ · c ≥ bᵢ  ∀i   （每个样本不低估）
        c ≥ 0               （系数非负）

其中 bᵢ = max_m(real_{i,m})  （取 4 个模型中的最大值）
```

**数据集**: 24 类合成数据，含纯 emoji、UUID 密集、各种代码、LaTeX、日文等，100 样本/类

**LP 拟合结果（失败）**:

```
Coeffs(cjk=0.71, letter=0.05, digit=0.00, punct=2.07, space=0.00, word=0.98)
```

**评估**:

| 统计量 | 值 |
|--------|-----|
| 保证率 | 100% |
| 平均高估 | **+70%** |
| 最大高估 | **+569%** |

**失败原因**: 极端合成样本（纯 emoji、纯 UUID 序列等）字符全落在 `punct` 类别，token 数却很高。LP 被迫把 `punct` 系数推高到 2.07，对普通中英文造成巨大过高估。

**结论**: LP 对数据分布极端敏感。须把训练域限制在目标业务场景（中英文），删除极端样本。

---

## 阶段 5：收窄到中英文 5 类 + LP 上界

**数据集**: 仅 5 类（pure_chinese, pure_english, chat_zh, chat_en, mixed），100 样本/类，共 500 条

**LP 拟合结果**:

```python
UPPER_COEFFS = Coeffs(
    cjk    = 0.7151,
    letter = 0.0463,
    digit  = 0.0000,
    punct  = 0.9617,
    space  = 0.0000,
    word   = 0.9761,
)
```

**评估**（4 模型合计）:

| 统计量 | 值 |
|--------|-----|
| 保证率 | **100.0%** |
| 平均高估 | +23.5% |
| 最大高估 | +92.9% |
| MAPE | 23.5% |

**结论**: 首次实现 100% 保证率，且高估幅度可接受。但对代码类文本（含大量 digit/punct）可能低估——需扩充训练集。

---

## 阶段 6：21 类训练集（含代码/格式化文本）+ LP 上界（当前主力）

**数据集**: 21 类合成数据（sample_gen.py），100 样本/类，共 2100 条:
- 自然语言：pure_chinese, pure_english, chat_zh, chat_en, mixed
- 代码：code_py, code_js, code_shell, code_go, code_rust, code_cpp, code_java, code_sql
- 格式化：markdown, json, yaml, xml_html, latex
- 其他：numeric, log_output, url_code

**LP 拟合结果（当前 UPPER_COEFFS）**:

```python
UPPER_COEFFS = Coeffs(
    cjk    = 0.7177,
    letter = 0.3171,
    digit  = 1.0067,
    punct  = 0.5641,
    space  = 0.0000,
    word   = 0.9030,
)
```

**评估**（4 模型合计）:

| 统计量 | 值 |
|--------|-----|
| 保证率 | **100.0%** |
| 平均高估 | +43.7% |
| 最大高估 | ~194% |
| MAPE | 43.7% |

**高估代价**: 相比阶段 5（仅中英文），加入代码类后 `digit` 和 `letter` 系数升高，对纯中文文本引入额外高估（+20%→+44%）。这是覆盖代码类场景的必然代价。

**结论**: 当前版本。若业务数据以中英文为主，可退回阶段 5 的系数（更紧）；若含大量代码，使用当前系数。用 `fit_custom.py` 在业务数据上重新拟合可得到最优系数。

---

## 总结对比

| 阶段 | 算法 | 保证率 | MAPE | 最大高估 | 最大低估 |
|------|------|--------|------|----------|----------|
| 0 | Naive (ASCII/5 + 非ASCII/2) | ~55% | ~35% | +189% | −45% |
| 1 | NNLS 5 特征 | N/A | ~7.6% | — | −30%+ |
| 2 | NNLS 6 特征（DEFAULT_COEFFS） | N/A | ~7.0% | — | −25%+ |
| 4 | LP 24 类（失败） | 100% | 70% | +569% | 0% |
| 5 | LP 中英文 5 类 | **100%** | 23.5% | +92.9% | 0% |
| **6** | **LP 21 类（当前）** | **100%** | **43.7%** | **~194%** | **0%** |

**主力推荐**:
- 需要**精度**（允许低估）→ `DEFAULT_COEFFS`（NNLS，MAPE ~7%）
- 需要**安全上界**（不允许低估）→ `UPPER_COEFFS`（LP，100% 保证）
- 有**业务数据** → 用 `fit_custom.py` 重新拟合，得到更紧的 UPPER_COEFFS
