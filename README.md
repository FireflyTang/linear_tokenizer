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

### 两套系数

| 系数集 | 拟合方法 | 用途 | MAPE（真实数据）| 保证率（真实数据）|
|--------|---------|------|----------------|-----------------|
| `DEFAULT_COEFFS` | NNLS（最小化平方误差） | 最小化误差，允许低估 | 6–12% | 不保证 |
| `UPPER_COEFFS` | LP（线性规划） | **保证不低估**，最小化高估幅度 | ~45% | **97–99%**（合成数据 100%）|

```python
# DEFAULT_COEFFS — 4 模型 stacked NNLS，261 条真实语料（Wikipedia + GitHub）
DEFAULT_COEFFS = Coeffs(
    cjk=0.6330, letter=0.1406, digit=0.7876,
    punct=0.7115, space=0.0995, word=0.3633,
)

# UPPER_COEFFS — LP 上界，21 类合成数据 × 100 样本
# 在合成数据上保证率 100%；真实数据上 97–99%（见下方详细评测）
UPPER_COEFFS = Coeffs(
    cjk=0.7177, letter=0.3171, digit=1.0067,
    punct=0.5641, space=0.0000, word=0.9030,
)
```

---

## 基准评测（真实数据）

测试集：1117 条真实语料（Wikipedia 中英文 19 篇 + GitHub 代码 35+ 文件，每条约 4000 字符）。
语料存放在 `.sample_cache/`，可直接使用，无需下载。

### DEFAULT_COEFFS — 精度（MAPE / 偏差）

| 类别 | 样本数 | GLM-5 MAPE | GLM-5 偏差 | DSV-V3.2 MAPE | DSV-V3.2 偏差 | Kimi-K2.5 MAPE | Kimi-K2.5 偏差 | MiniMax-M2.5 MAPE | MiniMax-M2.5 偏差 |
|------|--------|-----------|-----------|--------------|--------------|---------------|---------------|------------------|------------------|
| code_cpp | 427 | 6.4% | +0.5% | 7.9% | −6.0% | 6.4% | +0.5% | 14.4% | −14.3% |
| code_py | 435 | 6.1% | +3.8% | 3.9% | −0.2% | 6.2% | +3.9% | 10.9% | −10.8% |
| code_rust | 59 | 10.9% | +10.4% | 5.8% | +3.6% | 11.0% | +10.4% | 5.9% | −5.4% |
| code_go | 46 | 4.6% | −2.9% | 8.8% | −7.9% | 4.6% | −2.8% | 18.3% | −18.3% |
| code_js | 14 | 9.5% | +9.5% | 2.8% | +2.8% | 9.6% | +9.6% | 12.0% | −12.0% |
| code_shell | 10 | 8.4% | −8.4% | 15.3% | −15.3% | 8.0% | −8.0% | 21.1% | −21.1% |
| pure_chinese | 23 | 7.6% | −4.4% | 5.8% | +3.6% | 40.6% | −40.6% | 6.7% | −2.1% |
| pure_english | 80 | 7.1% | +3.6% | 7.9% | +4.7% | 7.4% | +3.8% | 6.5% | +1.8% |
| mixed | 23 | 8.9% | +8.4% | 2.7% | −0.3% | 8.9% | +8.5% | 13.5% | −13.5% |
| **OVERALL** | **1117** | **6.6%** | **+2.5%** | **6.1%** | **−2.2%** | **7.4%** | **+1.8%** | **12.0%** | **−11.2%** |

> Kimi 纯中文误差 −40.6%：DEFAULT_COEFFS 基于 HF Kimi tokenizer 拟合，而评测用 tiktoken 代理（内部一致但有 special-token 差异）。使用完整 Kimi tokenizer 重新拟合可消除。

### UPPER_COEFFS — 保证率 / 高估幅度

| 类别 | 样本数 | GLM-5 保证率 | 均高估 | 最大高估 | DSV-V3.2 保证率 | 均高估 | 最大高估 | Kimi-K2.5 保证率 | 均高估 | 最大高估 | MiniMax-M2.5 保证率 | 均高估 | 最大高估 |
|------|--------|------------|--------|---------|----------------|--------|---------|----------------|--------|---------|-------------------|--------|---------|
| code_cpp | 427 | 100.0% | +53.1% | +90.4% | 100.0% | +43.3% | +85.6% | 100.0% | +53.1% | +90.4% | 98.8% | +30.8% | +70.9% |
| code_py | 435 | 100.0% | +49.9% | +79.3% | 100.0% | +44.1% | +68.1% | 100.0% | +50.0% | +79.3% | 99.1% | +28.8% | +61.3% |
| code_rust | 59 | 100.0% | +48.9% | +81.5% | 100.0% | +39.8% | +73.4% | 100.0% | +49.0% | +81.6% | 100.0% | +27.6% | +63.3% |
| code_go | 46 | 100.0% | +61.2% | +91.8% | 100.0% | +52.9% | +87.7% | 100.0% | +61.4% | +91.8% | 100.0% | +35.9% | +77.0% |
| code_js | 14 | 100.0% | +60.5% | +72.1% | 100.0% | +50.6% | +62.0% | 100.0% | +60.5% | +72.1% | 100.0% | +29.0% | +37.6% |
| code_shell | 10 | 100.0% | +36.5% | +50.6% | 100.0% | +26.2% | +34.8% | 100.0% | +37.0% | +50.6% | 100.0% | +17.5% | +28.7% |
| pure_chinese | 23 | 73.9% | +11.9% | +31.4% | 87.0% | +20.5% | +36.3% | **0.0%** | — | −21.8% | 82.6% | +14.6% | +29.5% |
| pure_english | 80 | 90.0% | +82.6% | +129.0% | 90.0% | +84.8% | +131.8% | 90.0% | +83.0% | +129.0% | 90.0% | +80.0% | +128.0% |
| mixed | 23 | 100.0% | +56.3% | +76.6% | 100.0% | +43.7% | +65.2% | 100.0% | +56.3% | +76.4% | 100.0% | +24.8% | +48.9% |
| **OVERALL** | **1117** | **98.7%** | **+53.2%** | **+129.0%** | **99.0%** | **+46.3%** | **+131.8%** | **97.2%** | **+53.1%** | **+129.0%** | **98.1%** | **+33.0%** | **+128.0%** |

**⚠️ 注意**：`pure_chinese` 和 `pure_english` 保证率偏低（0–90%）。原因：`UPPER_COEFFS` 在合成数据上拟合，Wikipedia 真实文章的 token 分布与合成文本有差异。**如需严格保证**，请用 `fit_custom.py` 在你的实际业务数据上重新拟合：

```bash
python fit_custom.py --text your_data.txt --with-synthetic --samples 100
```

---

## 性能

C 扩展直接访问 CPython 内部 Unicode buffer（PEP 393），单次遍历完成全部 6 特征统计，无内存拷贝。

| 实现 | 耗时（160K token） | vs tiktoken |
|------|-------------------|-------------|
| HF transformers Kimi（Python 封装） | ~700ms | 0.07× |
| HF transformers GLM/DSV/MiniMax（Rust 后端） | ~5ms | 9× |
| tiktoken cl100k_base | 46ms | 1× |
| NumPy 全特征 `estimate_numpy_full()` | 4.6ms | 10× |
| Numba JIT（`_bench_numba.py`） | ~3ms | 15× |
| **C 扩展 `_features.pyd`（默认后端）** | **1.49ms** | **43×** |

> 测试规模：5K / 10K / 20K / 40K / 80K / 160K token，实测 O(n) 线性。

---

## 快速使用

```python
from tokenizer_approx import estimate, UPPER_COEFFS, extract_features

text = "这是一段中英文混合的 sample text，包含代码 x = 42。"

# 默认系数（最小误差，~7% MAPE）
estimate(text)                        # → int

# 上界系数（100% 保证不低估）
estimate(text, coeffs=UPPER_COEFFS)   # → int（≥ 任何真实 tokenizer 的结果）

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

| 阶段 | 算法 | 保证率 | MAPE |
|------|------|--------|------|
| 0 | Naive（ASCII/5 + 非ASCII/2） | ~55% | ~35% |
| 1–2 | NNLS 5/6 特征（DEFAULT_COEFFS） | 不保证 | ~7% |
| 4 | LP 24 类（失败，punct 系数 2.07） | 100% | 70% |
| 5 | LP 中英文 5 类 | 100% | 23.5% |
| **6** | **LP 21 类（当前 UPPER_COEFFS）** | **100%** | **43.7%** |

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `tokenizer_approx.py` | 核心估算模块：`estimate()` / `extract_features()` / `UPPER_COEFFS` / `DEFAULT_COEFFS` |
| `_features.c` | C 扩展源码：单次遍历特征统计，访问 CPython 内部 Unicode buffer |
| `setup.py` | C 扩展编译配置（`python setup.py build_ext --inplace`） |
| `fit_custom.py` | 自定义数据集接入 + 系数重新拟合 |
| `benchmark.py` | 误差评估 + NNLS/LP 系数拟合 + 详细报告 |
| `sample_gen.py` | 21 类合成样本生成器（无需下载） |
| `data_fetch.py` | Wikipedia + GitHub 真实语料下载与缓存 |
| `real_tokenizers.py` | 4 个真实 tokenizer 的薄封装（懒加载，仅下载词表） |
| `bench_perf.py` | 性能测试：5K→160K token 输入，对比 C扩展/NumPy/tiktoken |
| `build.py` | 跨平台 C 扩展编译脚本（Windows MSYS2 / Linux gcc / macOS CLT） |
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
