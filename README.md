# linear_tokenizer

无需加载模型权重的 LLM token 数线性近似估算器，支持 GLM-5 / DeepSeek-V3.2 / Kimi-K2.5 / MiniMax-M2.5。

---

## 算法原理

将文本中每个字符归入 5 个互斥类别，加上一个词级特征，各乘独立系数后求和：

```
tokens ≈ cjk×F_cjk + letter×F_letter + digit×F_digit + punct×F_punct + space×F_space + word×F_word
```

| 特征 | 字符范围 | 说明 |
|------|---------|------|
| `cjk` | U+4E00–U+9FFF, U+3000–U+303F, U+FF00–U+FFEF | 汉字、CJK 标点、全角字符 |
| `letter` | `[A-Za-z]` | ASCII 字母 |
| `digit` | `[0-9]` | 数字 |
| `punct` | 其余字符 | 标点、符号、emoji、西里尔字母等 |
| `space` | `[ \t\n\r…]` | 空白字符 |
| `word` | — | 空格切词数（捕捉 BPE 词边界行为） |

扩展 7 特征变体（`estimate7`）将 `digit` 拆分为 `digit_iso`（孤立，两侧非数字）和 `digit_run`（连续 2+ 位数字串），可降低最大高估幅度（见下方评测）。

### 三套系数

| 系数集 | 拟合方法 | 均高估 | P90 | 最大高估 | 保证率 | 最大低估 |
|--------|---------|--------|-----|---------|--------|---------|
| `DEFAULT_COEFFS` | NNLS（最小化平方误差） | +4.4% | — | — | 53% | 不保证 |
| `UPPER_COEFFS` | LP 严格上界（estimate ≥ real） | +23.9% | +37% | +98.7% | **99.9%** | 0% |
| `UPPER_COEFFS_RELAXED` | LP 宽松上界（estimate ≥ 0.9×real） | **+11.5%** | **+23.5%** | +78.8%* | 93% | 10% |

*最大高估由 7 个极端样本（Wikipedia 引用密集段 + 特殊库代码）驱动，这类内容在业务场景中罕见。去掉这 7 个样本后 P95 = +27.7%。

```python
# DEFAULT_COEFFS — 4 模型 stacked NNLS，1117 条真实语料（Wikipedia + GitHub）
DEFAULT_COEFFS = Coeffs(
    cjk=0.6223, letter=0.1766, digit=1.1655,
    punct=0.7246, space=0.0971, word=0.1947,
)

# UPPER_COEFFS — LP 严格上界，保证 estimate >= real（99.9%）
# 均高估 +23.9%，最大 +98.7%
UPPER_COEFFS = Coeffs(
    cjk=0.7591, letter=0.2135, digit=1.8696,
    punct=1.1526, space=0.0980, word=0.3077,
)

# UPPER_COEFFS_RELAXED — LP 宽松上界（estimate >= 0.9×real），推荐用于大多数场景
# 均高估 +11.5%，P90 +23.5%，93% 保证不低估，最大低估 10%
UPPER_COEFFS_RELAXED = Coeffs(
    cjk=0.6832, letter=0.1921, digit=1.6826,
    punct=1.0374, space=0.0882, word=0.2770,
)
```

---

## 基准评测（真实数据）

测试集：1117 条真实语料（Wikipedia 中英文 19 篇 + GitHub 代码 35+ 文件，每条约 4000 字符）。
语料存放在 `.sample_cache/`，可直接使用，无需下载。

### 跨模型 token 数相关性

各模型 token 数高度线性相关（r > 0.95），但单样本差异不可忽略：

| 模型对 | Pearson r |
|--------|---------|
| GLM-5 vs Kimi-K2.5 | **0.9978** |
| DSV-V3.2 vs Kimi-K2.5 | 0.9732 |
| MiniMax-M2.5 vs Kimi-K2.5 | 0.9515 |
| DSV-V3.2 vs MiniMax-M2.5 | 0.9826 |

以 Kimi 为基准，各模型的 token 数比值：

| 类别 | GLM-5/Kimi | DSV-V3.2/Kimi | MiniMax-M2.5/Kimi |
|------|-----------|--------------|-----------------|
| pure_english | 1.010±0.013 | 0.999±0.021 | 1.028±0.024 |
| pure_chinese | 1.029±0.032 | 0.950±0.075 | 1.006±0.066 |
| code_py | 0.996±0.012 | 1.036±0.042 | 1.160±0.061 |
| code_cpp | 1.003±0.011 | 1.072±0.035 | 1.176±0.060 |
| code_go | 1.002±0.008 | 1.058±0.020 | 1.197±0.063 |
| mixed | 0.997±0.007 | 1.084±0.027 | 1.251±0.064 |
| **ALL** | **1.001±0.014** | **1.049±0.045** | **1.158±0.074** |
| 极端范围 | 0.94~1.09 | 0.80~1.43 | 0.87~1.54 |

**用 Kimi token 数代替其他模型的误差：**

| 目标模型 | MAPE | 偏差 | 范围 |
|---------|------|------|------|
| GLM-5 | **0.9%** | −0.1% | −8.2% ~ +6.5% |
| DSV-V3.2 | 5.4% | −4.5% | −30.0% ~ +24.5% |
| MiniMax-M2.5 | 13.4% | −13.3% | −35.0% ~ +14.5% |

GLM-5 与 Kimi-K2.5 实际上使用了几乎相同的 tokenizer（MAPE 0.9%）。MiniMax 在代码类别上分词粒度明显更细（比 Kimi 多约 16–25%）。偏差为负意味着 Kimi 系统性低于目标模型——**用 Kimi 计数做预算保证会低估 DSV/MiniMax**。

### DEFAULT_COEFFS — 精度（MAPE / 偏差）

| 类别 | 样本数 | GLM-5 MAPE | GLM-5 偏差 | DSV-V3.2 MAPE | DSV-V3.2 偏差 | Kimi-K2.5 MAPE | Kimi-K2.5 偏差 | MiniMax-M2.5 MAPE | MiniMax-M2.5 偏差 |
|------|--------|-----------|-----------|--------------|--------------|---------------|---------------|------------------|------------------|
| code_cpp | 427 | 6.8% | +4.4% | 6.3% | −2.3% | 6.8% | +4.7% | 11.1% | −10.9% |
| code_py | 435 | 7.9% | +6.8% | 4.6% | +2.8% | 7.8% | +6.4% | 8.5% | −8.2% |
| code_rust | 59 | 12.1% | +11.7% | 6.6% | +4.8% | 12.0% | +11.6% | 5.1% | −4.3% |
| code_go | 46 | 3.7% | −1.3% | 7.4% | −6.5% | 3.6% | −1.1% | 17.1% | −17.0% |
| code_js | 14 | 12.3% | +12.3% | 5.4% | +5.3% | 11.2% | +11.2% | 9.8% | −9.8% |
| code_shell | 10 | 6.7% | −6.1% | 13.2% | −13.2% | 6.8% | −5.2% | 19.2% | −19.2% |
| pure_chinese | 23 | 7.8% | −4.8% | 5.9% | +3.2% | 9.0% | −1.8% | 6.9% | −2.5% |
| pure_english | 80 | 10.3% | +6.7% | 11.1% | +7.9% | 10.7% | +7.7% | 9.5% | +4.8% |
| mixed | 23 | 12.0% | +12.0% | 3.4% | +2.9% | 11.6% | +11.6% | 10.8% | −10.8% |
| **OVERALL** | **1117** | **7.8%** | **+5.6%** | **6.0%** | **+0.8%** | **7.8%** | **+5.7%** | **9.9%** | **−8.5%** |

### UPPER_COEFFS — 保证率 / 高估幅度

| 类别 | 样本数 | GLM-5 保证率 | 均高估 | 最大高估 | DSV-V3.2 保证率 | 均高估 | 最大高估 | Kimi-K2.5 保证率 | 均高估 | 最大高估 | MiniMax-M2.5 保证率 | 均高估 | 最大高估 |
|------|--------|------------|--------|---------|----------------|--------|---------|----------------|--------|---------|-------------------|--------|---------|
| code_cpp | 427 | 100.0% | +42.5% | +85.5% | 100.0% | +33.4% | +71.6% | 100.0% | +42.9% | +78.8% | 100.0% | +21.6% | +49.2% |
| code_py | 435 | 100.0% | +44.3% | +105.6% | 100.0% | +38.9% | +95.7% | 100.0% | +43.7% | +98.9% | 100.0% | +24.1% | +82.0% |
| code_rust | 59 | 100.0% | +59.7% | +82.8% | 100.0% | +49.9% | +71.4% | 100.0% | +59.5% | +79.0% | 100.0% | +36.8% | +55.8% |
| code_go | 46 | 100.0% | +35.1% | +50.5% | 100.0% | +28.1% | +46.1% | 100.0% | +35.4% | +51.2% | 100.0% | +13.5% | +37.1% |
| code_js | 14 | 100.0% | +53.3% | +61.7% | 100.0% | +43.9% | +52.2% | 100.0% | +51.9% | +59.7% | 100.0% | +23.2% | +28.6% |
| code_shell | 10 | 100.0% | +31.6% | +45.9% | 100.0% | +21.7% | +39.9% | 100.0% | +33.0% | +50.2% | 100.0% | +13.3% | +30.2% |
| pure_chinese | 23 | 100.0% | +20.1% | +40.6% | 100.0% | +30.2% | +45.7% | 100.0% | +23.8% | +48.3% | 100.0% | +22.9% | +38.6% |
| pure_english | 80 | 100.0% | +39.1% | +99.1% | 100.0% | +40.7% | +112.3% | 100.0% | +40.5% | +112.3% | 100.0% | +36.8% | +102.7% |
| mixed | 23 | 100.0% | +51.0% | +71.9% | 100.0% | +38.8% | +53.2% | 100.0% | +50.4% | +71.3% | 100.0% | +20.3% | +29.6% |
| **OVERALL** | **1117** | **100.0%** | **+43.3%** | **+105.6%** | **100.0%** | **+36.8%** | **+112.3%** | **100.0%** | **+43.4%** | **+112.3%** | **100.0%** | **+24.0%** | **+102.7%** |

### UPPER_COEFFS7（7 特征，digit 拆分）— 降低最大高估

将 `digit` 特征拆分为两类：`digit_iso`（孤立数字字符，两侧均为非数字）和 `digit_run`（连续 2+ 位的数字串），可解决 `[1]`、`[2]` 等引用编号被严重高估的问题。

| 指标 | 6 特征 UPPER | 7 特征 UPPER7 |
|------|------------|--------------|
| 保证率 | 100.0% | 100.0% |
| 平均高估 | +36.9% | +36.8% |
| **最大高估** | **+112.3%** | **+102.2%** |
| pure_english 最大 | +112.3% | **+62.6%** |
| code_py 最大 | +105.6% | +102.2% |

```python
from tokenizer_approx import estimate7, UPPER_COEFFS7

estimate7(text, coeffs=UPPER_COEFFS7)   # 7-feature upper bound
```

```python
UPPER_COEFFS7 = Coeffs7(
    cjk=0.7869, letter=0.2052,
    digit_iso=2.5475, digit_run=0.8148,
    punct=1.0791, space=0.0976, word=0.4461,
)
```

`digit_iso=2.55` 远大于 `digit_run=0.81`，符合"孤立数字如 `[1]` 通常独占一个 token，而数字串 `1234` 被合并"的 BPE 行为。

### 各模型单独 UPPER_COEFFS — 保证率 / 高估幅度

每个模型用专属系数，高估幅度比共用系数降低约 15–20 个百分点。

| 类别 | 样本数 | GLM-5 保证率 | 均高估 | 最大高估 | DSV-V3.2 保证率 | 均高估 | 最大高估 | Kimi-K2.5 保证率 | 均高估 | 最大高估 | MiniMax-M2.5 保证率 | 均高估 | 最大高估 |
|------|--------|------------|--------|---------|----------------|--------|---------|----------------|--------|---------|-------------------|--------|---------|
| code_cpp | 427 | 100.0% | +20.1% | +62.0% | 100.0% | +19.7% | +62.5% | 100.0% | +17.6% | +43.9% | 100.0% | +21.6% | +49.2% |
| code_py | 435 | 100.0% | +24.2% | +82.5% | 100.0% | +25.2% | +92.3% | 100.0% | +23.5% | +61.5% | 100.0% | +24.1% | +82.0% |
| code_rust | 59 | 100.0% | +37.4% | +55.2% | 100.0% | +39.9% | +58.8% | 100.0% | +31.3% | +45.6% | 100.0% | +36.8% | +55.8% |
| code_go | 46 | 100.0% | +10.5% | +21.0% | 100.0% | +10.4% | +22.6% | 100.0% | +13.9% | +28.6% | 100.0% | +13.5% | +37.1% |
| code_js | 14 | 100.0% | +30.9% | +36.0% | 100.0% | +30.4% | +37.0% | 100.0% | +27.2% | +30.4% | 100.0% | +23.2% | +28.6% |
| code_shell | 10 | 100.0% | +10.7% | +24.0% | 100.0% | +10.0% | +27.9% | 100.0% | +9.7% | +23.0% | 100.0% | +13.3% | +30.2% |
| pure_chinese | 23 | 100.0% | +16.7% | +38.4% | 100.0% | +10.2% | +21.2% | 100.0% | +26.2% | +54.2% | 100.0% | +13.1% | +25.7% |
| pure_english | 80 | 100.0% | +13.4% | +61.9% | 100.0% | +16.3% | +86.8% | 100.0% | +24.0% | +72.6% | 100.0% | +36.8% | +102.7% |
| mixed | 23 | 100.0% | +30.1% | +52.6% | 100.0% | +25.9% | +42.1% | 100.0% | +27.4% | +45.4% | 100.0% | +20.3% | +29.6% |
| **OVERALL** | **1117** | **100.0%** | **+21.9%** | **+82.5%** | **100.0%** | **+22.3%** | **+92.3%** | **100.0%** | **+21.4%** | **+72.6%** | **100.0%** | **+23.8%** | **+102.7%** |

各模型专属系数（可粘贴到代码中使用）：

```python
GLM_UPPER   = Coeffs(cjk=0.7605, letter=0.1610, digit=1.5260, punct=1.0537, space=0.1318, word=0.1657)
DSV_UPPER   = Coeffs(cjk=0.6195, letter=0.1667, digit=1.6694, punct=1.2273, space=0.1180, word=0.1244)
KIMI_UPPER  = Coeffs(cjk=0.8133, letter=0.1554, digit=1.4905, punct=0.8416, space=0.1284, word=0.4188)
MMAX_UPPER  = Coeffs(cjk=0.6773, letter=0.2135, digit=1.8696, punct=1.1526, space=0.0980, word=0.3077)
```

---

## 性能

C 扩展直接访问 CPython 内部 Unicode buffer（PEP 393），单次遍历完成全部 6 特征统计，无内存拷贝。

**精确计数（误差 0%，需加载 tokenizer 文件）**

| 实现 | 耗时（~160K token） | vs tiktoken |
|------|-------------------|-------------|
| HF transformers Kimi（Python 封装） | ~735ms | 0.09× |
| HF transformers GLM/DSV/MiniMax（Rust 后端） | ~250ms | 0.25× |
| tiktoken cl100k_base（Kimi 快速代理） | ~60ms | 1× |

**近似估算（误差 6–10% MAPE，零依赖）**

| 实现 | 耗时（~160K token） | vs tiktoken |
|------|-------------------|-------------|
| NumPy 全特征 `estimate_numpy_full()` | ~3.4ms | 18× |
| Numba JIT（`_bench_numba.py`） | ~1.8ms | 33× |
| **C 扩展 `_features.pyd`（默认后端）** | **~1.5ms** | **40×** |

> 测试规模：5K / 10K / 20K / 40K / 80K / 160K token，实测 O(n) 线性。单条文本顺序执行，耗时因文本类型有±30% 波动。

---

## 快速使用

```python
from tokenizer_approx import estimate, UPPER_COEFFS, UPPER_COEFFS_RELAXED, extract_features

text = "这是一段中英文混合的 sample text，包含代码 x = 42。"

# 默认系数（最小误差，~7% MAPE，允许低估）
estimate(text)

# 宽松上界（推荐）：93% 不低估，均高估 +11.5%，最大低估 10%
estimate(text, coeffs=UPPER_COEFFS_RELAXED)

# 严格上界：99.9% 不低估，均高估 +23.9%
estimate(text, coeffs=UPPER_COEFFS)

# 底层特征（用于自定义计算）
extract_features(text)
# → {"cjk": 12, "letter": 16, "digit": 2, "punct": 3, "space": 8, "word": 7}
```

### 各模型独立系数（精度更高）

```python
from tokenizer_approx import estimate, Coeffs

KIMI_COEFFS = Coeffs(cjk=0.6485, letter=0.1847, digit=1.0885, punct=0.6958, space=0.1121, word=0.0)
DSV_COEFFS  = Coeffs(cjk=0.5761, letter=0.1785, digit=1.1767, punct=0.8226, space=0.1186, word=0.0)
GLM_COEFFS  = Coeffs(cjk=0.6615, letter=0.1840, digit=1.2459, punct=0.6745, space=0.1185, word=0.0)
MMAX_COEFFS = Coeffs(cjk=0.6162, letter=0.1838, digit=0.8170, punct=1.0085, space=0.1244, word=0.0)

estimate(text, coeffs=KIMI_COEFFS)
```

---

## C 扩展编译（可选，获得最高性能）

不编译也能运行，会自动回退到 NumPy/regex 实现（精度相同，速度稍慢）。

### 一键编译（跨平台）

```bash
python build.py
```

`build.py` 自动检测平台，调用正确的编译器，并在编译后验证结果：

```
[build] 平台: Windows (AMD64)
[build] Python: C:\Python312\python.exe  (3.12.0)
[build] MinGW gcc: C:\msys64\mingw64\bin
[build] 运行: python setup.py build_ext --inplace --compiler=mingw32
...
[build] 验证通过: extract_features('hello 你好 123')
[build]   → cjk=2, letter=5, digit=3, punct=0, space=2, word=3
[build] 成功！_features 模块已就绪，estimate() 将自动使用 C 扩展后端。
```

### 平台依赖

| 平台 | 依赖 | 安装方式 |
|------|------|---------|
| Windows | MSYS2 + MinGW-w64 gcc | 安装 [MSYS2](https://www.msys2.org/)，然后 `pacman -S mingw-w64-x86_64-gcc` |
| Linux | gcc + python3-dev | `sudo apt install build-essential python3-dev` |
| macOS | Xcode CLT | `xcode-select --install` |

Windows 下 gcc 默认路径为 `C:\msys64\mingw64\bin\gcc.exe`；如安装到其他位置，设置：
```
set MINGW_BIN=C:\path\to\mingw64\bin
python build.py
```

编译成功后 `estimate()` 自动使用 C 扩展，无需任何代码修改。

---

## 系数训练

### 方法一：合成数据集（无需下载，最快）

21 类合成数据（代码、Markdown、JSON、中英文等），每类 N 个样本：

```bash
# 拟合 UPPER_COEFFS（LP 上界）
python benchmark.py --samples 100 --fit-upper

# 拟合 DEFAULT_COEFFS（NNLS）
python benchmark.py --samples 100 --fit-shared

# 同时拟合两套
python benchmark.py --samples 100 --fit-upper --fit-shared
```

### 方法二：真实 Wikipedia + GitHub 语料

首次运行自动下载并缓存到 `.sample_cache/`：

```bash
python benchmark.py --real-data --fit-upper --fit-shared

# 强制重新下载
python benchmark.py --real-data --refresh --fit-upper
```

### 方法三：自定义业务数据（推荐）

将你的业务文本给到 `fit_custom.py`，支持与内置数据集合并训练：

```bash
# 只用自定义数据
python fit_custom.py --text mydata.txt

# 自定义 + 合成数据集（获得更广的覆盖）
python fit_custom.py --text mydata.txt --with-synthetic --samples 50

# 自定义 + Wikipedia/GitHub 真实数据
python fit_custom.py --text mydata.txt --with-real

# 多个文件
python fit_custom.py --text file1.txt --text file2.txt --with-synthetic
```

输出示例：

```
数据来源汇总:
  mydata.txt: 143 块  (428,741 字符)
  合成数据集（21 类×50 样本）: 1050 块
  ──────────────────────────────────────
  合计: 1193 块  (3,214,082 字符)

── UPPER_COEFFS 拟合结果（LP，保证不低估）──
  整体指标（1 个模型 × 1193 块）:
    保证率:   100.0%  (1193/1193)
    平均高估: +21.3%
    最大高估: +78.4%
    MAPE:     21.3%

  按数据来源细分（模型: kimi_fast）:
  来源                            块数  保证率   均高估  最大高估   MAPE
  ──────────────────────────────  ────  ───────  ───────  ────────  ─────
  自定义/mydata.txt                143  100.0%   +18.2%   +54.1%   18.2%
  合成/pure_chinese                 50  100.0%   +22.1%   +63.4%   22.1%
  合成/code_py                      50  100.0%   +31.2%   +77.6%   31.2%
  ...

══════════════════════════════════════════════════════════════════════
  # 将以下内容粘贴到 tokenizer_approx.py 以更新系数:

UPPER_COEFFS = Coeffs(
    cjk    = 0.7251,
    ...
)
```

拟合完成后将输出的 `UPPER_COEFFS` 和 `DEFAULT_COEFFS` 粘贴到 `tokenizer_approx.py` 即可。

---

## 数据集说明

| 数据集 | 文件 | 类别数 | 说明 |
|--------|------|--------|------|
| 合成数据 | `sample_gen.py` | 21 类 | 代码（Python/JS/Go/Rust/C++/Java/SQL/Shell）、Markdown、JSON、YAML、XML、LaTeX、中英文、日志等 |
| 真实数据 | `data_fetch.py` | — | Wikipedia 19 篇（中英文 AI/ML）+ GitHub 35+ 文件（CPython/requests/NumPy/Flask 等） |
| 自定义数据 | `fit_custom.py --text` | — | 任意 UTF-8 文本文件，自动按行切块 |

---

## 算法历史

详见 [HISTORY.md](HISTORY.md)，记录了 6 个阶段的演进过程、失败案例和数字结果。

| 阶段 | 算法 | 保证率 | 均高估 | 最大高估 |
|------|------|--------|--------|---------|
| 0 | Naive（ASCII/5 + 非ASCII/2） | ~55% | ~35% | — |
| 1–2 | NNLS 5/6 特征（DEFAULT_COEFFS） | 不保证 | ~4% | — |
| 4 | LP 24 类（失败，punct 系数 2.07） | 100% | 70% | +569% |
| 5 | LP 中英文 5 类 | 100% | 23.5% | +92.9% |
| 6 | LP 21 类（当前 UPPER_COEFFS，真实数据） | 99.9% | +23.9% | +98.7% |
| 7 | 7 特征 LP（digit_iso/digit_run 拆分） | 99.9% | +36.8% | +102.2% |
| **8** | **LP 宽松约束 tol=10%（UPPER_COEFFS_RELAXED）** | **93%** | **+11.5%** | **+78.8%** |

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `tokenizer_approx.py` | 核心估算模块：`estimate()` / `extract_features()` / `UPPER_COEFFS` / `DEFAULT_COEFFS`；含 7 特征扩展 `estimate7()` / `UPPER_COEFFS7` |
| `_features.c` | C 扩展源码：单次遍历特征统计，访问 CPython 内部 Unicode buffer |
| `setup.py` | C 扩展编译配置（`python setup.py build_ext --inplace`） |
| `fit_custom.py` | 自定义数据集接入 + 系数重新拟合 |
| `benchmark.py` | 误差评估 + NNLS/LP 系数拟合 + 详细报告 |
| `sample_gen.py` | 21 类合成样本生成器（无需下载） |
| `data_fetch.py` | Wikipedia + GitHub 真实语料下载与缓存 |
| `real_tokenizers.py` | 4 个真实 tokenizer 的薄封装（懒加载，仅下载词表） |
| `bench_perf.py` | 性能测试：5K→160K token 输入，对比 C扩展/NumPy/tiktoken |
| `build.py` | 跨平台 C 扩展编译脚本（Windows MSYS2 / Linux gcc / macOS CLT） |
| `_fit7.py` | 7 特征模型拟合与评估（digit_iso/digit_run 拆分） |
| `_trim_analysis.py` | 高估分布分析：找约束最紧的样本，测试修剪上界的效果 |
| `_refit_filtered.py` | 改进上界系数的两种策略分析（过滤离群值 vs 放松 LP 约束） |
| `_bench_sampling.py` | 行采样估算方案实验（结论：中文/英文文本效果不佳） |
| `_cross_model_analysis.py` | 跨模型 token 数比值、Pearson r、用 Kimi 代替其他模型的误差 |
| `_eval_upper.py` | 逐类别评估 UPPER_COEFFS 保证率与高估幅度 |
| `_per_model_upper.py` | 各模型独立拟合 LP 上界系数 |
| `_bench_numba.py` | 实验性：Numba JIT vs NumPy vs tiktoken 性能对比 |
| `HISTORY.md` | 算法演进历史：6 个阶段的实现、数据、结果、结论 |
| `results.csv` | 上次完整 benchmark 的逐样本结果 |

---

## 安装

```bash
pip install -r requirements.txt
# 用于系数拟合（benchmark.py / fit_custom.py）
pip install scipy numpy
# 可选：Numba JIT 实验
pip install numba
```

`requirements.txt` 包含 transformers / sentencepiece / protobuf（真实 tokenizer 依赖）。
仅使用 `estimate()` 估算功能时无需安装以上依赖，只需 Python 标准库即可运行（C 扩展可选）。
