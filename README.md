# linear_tokenizer

无需加载模型权重的 LLM token 数线性近似估算器，支持 GLM-5 / DeepSeek-V3.2 / Kimi-K2.5 / MiniMax-M2.5。

---

## 算法原理

将文本按字符类型分为 5 个互斥类别，加上一个词级特征，各乘系数后求和：

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

扩展 7 特征变体（`estimate7`）将 `digit` 拆分为 `digit_iso`（孤立，两侧非数字）和 `digit_run`（连续 2+ 位数字串），可降低最大高估幅度（见[算法历史](#算法历史)）。

### 系数选择

| 系数集 | 用途 | 均高估 | P90 | 最大高估 | 保证率 | 最大低估 |
|--------|------|--------|-----|---------|--------|---------|
| `DEFAULT_COEFFS` | 最小化误差，允许低估 | +4% | — | — | 53% | 不保证 |
| `UPPER_COEFFS` | 严格上界，几乎不低估 | +24% | +37% | +99% | 99.9% | ~0% |
| `UPPER_COEFFS_RELAXED` | **宽松上界（推荐）** | **+12%** | **+24%** | **+83%*** | **93%** | **10%** |

\* 最大高估由少数极端样本（Wikipedia 引用密集段、特殊库代码）驱动；业务场景中实际遇到的上限接近 P90。

---

## 快速使用

```python
from tokenizer_approx import estimate, UPPER_COEFFS, UPPER_COEFFS_RELAXED, extract_features

text = "这是一段中英文混合的 sample text，包含代码 x = 42。"

# 默认：最小化误差，约 7% MAPE，允许低估
estimate(text)

# 宽松上界（推荐）：93% 保证不低估，均高估约 +12%，最大低估 10%
estimate(text, coeffs=UPPER_COEFFS_RELAXED)

# 严格上界：99.9% 保证不低估，均高估约 +24%
estimate(text, coeffs=UPPER_COEFFS)

# 底层特征
extract_features(text)
# → {"cjk": 12, "letter": 16, "digit": 2, "punct": 3, "space": 8, "word": 7}
```

---

## 性能

C 扩展直接访问 CPython 内部 Unicode buffer（PEP 393），单次遍历完成全部 6 特征统计，无内存拷贝。

**精确计数（误差 0%，需加载 tokenizer 文件）**

| 实现 | 耗时（~160K token） | vs tiktoken |
|------|-------------------|-------------|
| HF transformers Kimi（Python 封装） | ~735ms | 0.09× |
| HF transformers GLM/DSV/MiniMax（Rust 后端） | ~250ms | 0.25× |
| tiktoken cl100k_base | ~60ms | 1× |

**近似估算（MAPE 6–10%，零依赖）**

| 实现 | 耗时（~160K token） | vs tiktoken |
|------|-------------------|-------------|
| NumPy 全特征 `estimate_numpy_full()` | ~3.4ms | 18× |
| Numba JIT（`_bench_numba.py`） | ~1.8ms | 33× |
| **C 扩展 `_features.pyd`（默认后端）** | **~1.5ms** | **40×** |

测试规模：5K–160K token，O(n) 线性。单条文本顺序执行，耗时因文本类型有 ±30% 波动。

---

## 评测结果

测试集：1117 条真实语料（Wikipedia 中英文 19 篇 + GitHub 代码 35+ 文件，每条约 4000 字符）。
语料存放在 `.sample_cache/`，可直接使用，无需下载。

### 跨模型 token 数相关性

| 模型对 | Pearson r | 用 Kimi 代替的 MAPE | 偏差 | 范围 |
|--------|---------|---------------------|------|------|
| GLM-5 vs Kimi-K2.5 | **0.9978** | **0.9%** | −0.1% | −8% ~ +7% |
| DSV-V3.2 vs Kimi-K2.5 | 0.9732 | 5.4% | −4.5% | −30% ~ +25% |
| MiniMax-M2.5 vs Kimi-K2.5 | 0.9515 | 13.4% | −13.3% | −35% ~ +15% |

GLM-5 与 Kimi-K2.5 实际上使用了几乎相同的 tokenizer（MAPE 0.9%）。用 Kimi 计数替代 DSV/MiniMax 会系统性低估（偏差为负），不适合做预算上界。

### DEFAULT_COEFFS — MAPE / 偏差

| 类别 | 样本数 | GLM-5 MAPE | 偏差 | DSV-V3.2 MAPE | 偏差 | Kimi-K2.5 MAPE | 偏差 | MiniMax MAPE | 偏差 |
|------|--------|-----------|------|--------------|------|---------------|------|-------------|------|
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

### UPPER_COEFFS_RELAXED — 共享系数（4 模型联合拟合，tol=10%）

共享系数以 4 个模型中的最大 token 数为目标（通常为 MiniMax），因此对 GLM/Kimi 偏保守。

| 模型 | 保证率 | 均高估 | P90 | 最大高估 | 低估样本数 | 最大低估 |
|------|--------|--------|-----|---------|-----------|---------|
| GLM-5 | 98.9% | +29.0% | +43.7% | +85.1% | 12 | −7.4% |
| DSV-V3.2 | 99.4% | +23.1% | +36.0% | +90.7% | 7 | −6.3% |
| Kimi-K2.5 | 99.0% | +29.1% | +43.5% | +90.7% | 11 | −10.0% |
| MiniMax-M2.5 | 93.4% | +11.6% | +23.5% | +82.1% | 74 | −10.0% |

### UPPER_COEFFS_RELAXED — 各模型独立拟合（tol=10%）

每个模型用专属系数，均高估降至 ~10%，代价是保证率降至 84–93%（84–87% 的低估量最多 −10%）。

| 模型 | 保证率 | 均高估 | P90 | 最大高估 | 低估样本数 | 最大低估 |
|------|--------|--------|-----|---------|-----------|---------|
| GLM-5 | 84.7% | +9.7% | +23.0% | +64.2% | 171 | −10.0% |
| DSV-V3.2 | 86.8% | +10.1% | +23.9% | +73.1% | 148 | −10.0% |
| Kimi-K2.5 | 84.4% | +9.2% | +19.0% | +55.3% | 174 | −10.0% |
| MiniMax-M2.5 | 92.7% | +11.5% | +23.5% | +82.1% | 82 | −10.0% |

各模型专属系数：

```python
# 各模型独立 UPPER_COEFFS_RELAXED（tol=10%，运行 _refit_filtered.py 重新拟合）
GLM_UPPER_RELAXED  = Coeffs(cjk=0.6845, letter=0.1449, digit=1.3734, punct=0.9483, space=0.1186, word=0.1491)
DSV_UPPER_RELAXED  = Coeffs(cjk=0.5576, letter=0.1500, digit=1.5024, punct=1.1046, space=0.1062, word=0.1120)
KIMI_UPPER_RELAXED = Coeffs(cjk=0.7320, letter=0.1398, digit=1.3414, punct=0.7574, space=0.1155, word=0.3769)
MMAX_UPPER_RELAXED = Coeffs(cjk=0.6096, letter=0.1921, digit=1.6826, punct=1.0374, space=0.0882, word=0.2770)
```

### UPPER_COEFFS — 严格上界（4 模型联合拟合，tol=0%）

| 类别 | 样本数 | GLM-5 保证率 | 均高估 | 最大高估 | DSV-V3.2 保证率 | 均高估 | 最大高估 | Kimi-K2.5 保证率 | 均高估 | 最大高估 | MiniMax 保证率 | 均高估 | 最大高估 |
|------|--------|------------|--------|---------|----------------|--------|---------|----------------|--------|---------|--------------|--------|---------|
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

---

## 系数选择建议

| 场景 | 推荐系数 | 说明 |
|------|---------|------|
| 需要 token 数估算，对精度无严格要求 | `DEFAULT_COEFFS` | MAPE ~7%，速度最快 |
| 保守预算：不能低估，可接受高估 | `UPPER_COEFFS_RELAXED` | 93% 不低估，均高估 ~12% |
| 严格预算：几乎不允许任何低估 | `UPPER_COEFFS` | 99.9% 不低估，均高估 ~24% |
| 已知目标模型，精度优先 | 各模型独立系数 | 均高估 ~10%，保证率 84–93% |

关于共享 vs 独立系数：共享系数以 MiniMax（token 数最多的模型）为基准，对 GLM/DSV/Kimi 偏保守（多高估 ~20 个百分点）。如果只服务单一模型，建议使用各模型独立拟合的系数。

---

## C 扩展编译（可选，获得最高性能）

不编译也能运行，会自动回退到 NumPy/regex 实现（精度相同，速度稍慢）。

```bash
python build.py   # 自动检测平台
```

`build.py` 自动检测平台，调用正确的编译器，并在编译后验证结果。

| 平台 | 依赖 | 安装方式 |
|------|------|---------|
| Windows | MSYS2 + MinGW-w64 gcc | 安装 [MSYS2](https://www.msys2.org/)，然后 `pacman -S mingw-w64-x86_64-gcc` |
| Linux | gcc + python3-dev | `sudo apt install build-essential python3-dev` |
| macOS | Xcode CLT | `xcode-select --install` |

Windows 下 gcc 默认路径为 `C:\msys64\mingw64\bin\gcc.exe`；如安装到其他位置，设置环境变量 `MINGW_BIN`。

---

## 系数训练

### 方法一：合成数据集（无需下载，最快）

```bash
python benchmark.py --samples 100 --fit-upper --fit-shared
```

### 方法二：真实 Wikipedia + GitHub 语料

```bash
python benchmark.py --real-data --fit-upper --fit-shared
```

首次运行自动下载并缓存到 `.sample_cache/`。

### 方法三：自定义业务数据（推荐）

```bash
# 只用自定义数据
python fit_custom.py --text mydata.txt

# 合并内置语料
python fit_custom.py --text mydata.txt --with-synthetic --samples 50
python fit_custom.py --text mydata.txt --with-real
```

输出可直接粘贴的系数块，拟合完成后将 `UPPER_COEFFS` / `DEFAULT_COEFFS` 粘贴到 `tokenizer_approx.py` 即可。

### 重新拟合宽松上界系数

```bash
python _refit_filtered.py          # 完整分析：策略对比 + 各 tol 值扫描
python _refit_filtered.py --tol 10 # 只拟合 tol=10%
```

---

## 数据集说明

| 数据集 | 文件 | 类别数 | 说明 |
|--------|------|--------|------|
| 合成数据 | `sample_gen.py` | 21 类 | 代码（Python/JS/Go/Rust/C++/Java/SQL/Shell）、Markdown、JSON、YAML、XML、LaTeX、中英文、日志等 |
| 真实数据 | `data_fetch.py` | — | Wikipedia 19 篇（中英文 AI/ML）+ GitHub 35+ 文件（CPython/requests/NumPy/Flask 等） |
| 自定义数据 | `fit_custom.py --text` | — | 任意 UTF-8 文本文件，自动按行切块 |

---

## 算法历史

详见 [HISTORY.md](HISTORY.md)，记录了每个阶段的实现、数据、结果和失败案例。

| 阶段 | 算法 | 保证率 | 均高估 | 最大高估 |
|------|------|--------|--------|---------|
| 0 | Naive（ASCII/5 + 非ASCII/2） | ~55% | ~35% | — |
| 1–2 | NNLS 5/6 特征（DEFAULT_COEFFS） | 不保证 | ~4% | — |
| 4 | LP 24 类（失败，punct 系数 2.07） | 100% | 70% | +569% |
| 5 | LP 中英文 5 类 | 100% | 23.5% | +92.9% |
| 6 | LP 21 类，真实数据（UPPER_COEFFS） | 99.9% | +23.9% | +98.7% |
| 7 | 7 特征 LP，digit_iso/digit_run 拆分 | 99.9% | +36.8% | +102.2% |
| **8** | **LP tol=10%（UPPER_COEFFS_RELAXED）** | **93%** | **+11.5%** | **+78.8%** |

阶段 7 通过拆分 digit 特征降低了 pure_english 的最大高估（+112% → +62%），但整体 max 变化不大。阶段 8 通过放松 LP 约束（允许 10% 低估）使均高估从 +24% 降至 +12%，>40% 高估样本从 71 个减至 7 个。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `tokenizer_approx.py` | 核心模块：`estimate()` / `extract_features()` / `DEFAULT_COEFFS` / `UPPER_COEFFS` / `UPPER_COEFFS_RELAXED`；含 7 特征扩展 `estimate7()` |
| `_features.c` | C 扩展源码：单次遍历特征统计，访问 CPython 内部 Unicode buffer |
| `setup.py` | C 扩展编译配置 |
| `build.py` | 跨平台 C 扩展编译脚本（Windows MSYS2 / Linux gcc / macOS CLT） |
| `benchmark.py` | 误差评估 + NNLS/LP 系数拟合 |
| `fit_custom.py` | 自定义数据集接入 + 系数重新拟合 |
| `sample_gen.py` | 21 类合成样本生成器（无需下载） |
| `data_fetch.py` | Wikipedia + GitHub 真实语料下载与缓存 |
| `real_tokenizers.py` | 4 个真实 tokenizer 的薄封装（懒加载） |
| `bench_perf.py` | 性能测试：5K→160K token，对比 C扩展/NumPy/tiktoken |
| `_refit_filtered.py` | 宽松上界系数分析与拟合（策略A/B对比，tol 扫描） |
| `_fit7.py` | 7 特征模型拟合与评估（digit_iso/digit_run 拆分） |
| `_trim_analysis.py` | 高估分布分析：binding constraint 识别与修剪测试 |
| `_bench_sampling.py` | 行采样估算方案实验（结论：中文/英文文本效果不佳） |
| `_cross_model_analysis.py` | 跨模型 token 数比值、Pearson r、代理误差分析 |
| `_eval_upper.py` | 逐类别评估 UPPER_COEFFS 保证率与高估幅度 |
| `_per_model_upper.py` | 各模型独立拟合 LP 严格上界系数 |
| `_bench_numba.py` | 实验性：Numba JIT vs NumPy vs tiktoken 性能对比 |
| `HISTORY.md` | 算法演进历史：8 个阶段的实现、数据、结论 |
| `results.csv` | 上次完整 benchmark 的逐样本结果（1117 条） |

---

## 安装

```bash
pip install -r requirements.txt
# 系数拟合依赖
pip install scipy numpy
# 可选：Numba JIT 实验
pip install numba
```

`requirements.txt` 包含 transformers / sentencepiece / protobuf（真实 tokenizer 依赖）。
仅使用 `estimate()` 估算功能时只需 Python 标准库，C 扩展可选。
