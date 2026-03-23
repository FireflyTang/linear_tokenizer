"""
Random text sample generator for benchmark purposes.

Categories:
  pure_chinese   – natural Chinese text (news, technical)
  pure_english   – natural English text (news, technical)
  mixed          – interleaved Chinese and English
  code_py        – Python code snippets
  code_js        – JavaScript / TypeScript snippets
  code_shell     – Shell / bash commands and scripts
  chat_zh        – Chinese daily conversation (short, colloquial)
  chat_en        – English daily conversation
  numeric        – numbers, math expressions, punctuation
  numeric_dense  – pure numeric: IPs, UUIDs, timestamps, hashes
  markdown       – markdown with headings, lists, code blocks
  url_code       – URLs, file paths, import statements, identifiers
"""

import random

random.seed(42)

# ── Chinese sentence pool ─────────────────────────────────────────────────────
_ZH_SENTENCES = [
    "人工智能正在深刻改变人类社会的方方面面，从医疗到教育，从交通到金融，无处不在。",
    "深度学习模型需要大量的标注数据和计算资源，这对中小型企业来说是一个不小的挑战。",
    "自然语言处理技术让机器能够理解和生成人类语言，这在十年前还被认为是遥不可及的梦想。",
    "大型语言模型通过预训练和微调的方式，在各种下游任务上取得了令人惊叹的表现。",
    "量子计算有望在未来十年内打破传统计算的瓶颈，为人工智能带来指数级的性能提升。",
    "今天天气真好，适合出去散步。你最近怎么样？有没有什么有趣的事情可以分享？",
    "这道菜的做法很简单，先把蔬菜洗干净切好，然后热锅下油，大火翻炒两分钟就好了。",
    "我最近在学习Python编程，感觉挺有意思的，但有些概念还是有点难以理解。",
    "北京的秋天是一年中最美的季节，天高云淡，金黄的银杏叶铺满了街道。",
    "这部电影的剧情非常感人，特效也很震撼，强烈推荐大家去电影院观看。",
    "随着电子商务的发展，越来越多的人选择在网上购物，这改变了传统零售业的格局。",
    "机器学习算法在医学影像诊断领域的应用，大大提高了疾病的早期发现率。",
    "区块链技术的去中心化特性，为数字资产的安全存储和交易提供了全新的解决方案。",
    "中国的高铁网络是世界上最发达的，连接了全国几乎所有主要城市，极大地方便了人们的出行。",
    "云计算平台为企业提供了灵活、可扩展的IT基础设施，降低了技术门槛和运营成本。",
    "多模态模型能够同时处理文本、图像和音频，为人机交互带来了全新的可能性。",
    "强化学习让智能体通过与环境的交互不断试错，最终学会完成复杂的序列决策任务。",
    "数据隐私保护越来越受到重视，各国纷纷出台相关法规来规范企业对用户数据的使用。",
    "开源社区的蓬勃发展使得高质量的软件工具得以免费共享，极大地降低了技术创新的门槛。",
    "语音识别技术的准确率已经超过了人类水平，这使得人机语音交互变得越来越自然。",
    "微服务架构将单体应用拆分为独立部署的小型服务，提升了系统的可维护性和扩展性。",
    "联邦学习允许多方在不共享原始数据的情况下联合训练模型，有效保护了数据隐私。",
    "图神经网络在社交网络分析、推荐系统和药物发现等领域展现出了强大的建模能力。",
    "边缘计算将计算任务从云端下放到靠近数据源的设备上，有效降低了网络延迟和带宽消耗。",
    "生成对抗网络通过生成器和判别器的博弈训练，能够生成以假乱真的图像和视频内容。",
]

_ZH_CHAT = [
    "你好啊，最近忙不忙？",
    "我刚吃完饭，感觉撑死了。",
    "这个周末有什么计划吗？",
    "帮我推荐一部好看的电影吧。",
    "你觉得这个方案怎么样？",
    "明天要开个会，好烦啊。",
    "这道题我不会做，能帮我解释一下吗？",
    "刚才地铁好挤，人太多了。",
    "你喜欢吃什么类型的食物？",
    "最近工作压力好大，感觉快撑不住了。",
    "哈哈哈这个视频太搞笑了！",
    "好的，没问题，我一会儿发给你。",
    "等等，你刚说的意思是什么？",
    "这个价格是不是有点贵？",
    "谢谢你告诉我，我不知道呢。",
    "下午三点开会，记得准时。",
    "这个bug找了我一下午，终于找到了。",
    "你有没有试过那家新开的火锅店？",
    "明天天气不好，要带伞出门。",
    "我觉得你说得很有道理，我赞同。",
]

# ── English sentence pool ─────────────────────────────────────────────────────
_EN_SENTENCES = [
    "Artificial intelligence is transforming industries at an unprecedented pace, raising both opportunities and ethical questions.",
    "Large language models have demonstrated remarkable capabilities in understanding and generating human-like text across diverse domains.",
    "The transformer architecture, introduced in the paper 'Attention Is All You Need', revolutionized natural language processing.",
    "Reinforcement learning from human feedback (RLHF) has proven to be a powerful technique for aligning AI models with human preferences.",
    "Cloud computing has democratized access to powerful computational resources, enabling startups to compete with established enterprises.",
    "The rapid advancement of semiconductor technology continues to follow Moore's Law, though at a slower pace than in previous decades.",
    "Data privacy regulations like GDPR and CCPA have significantly changed how companies collect and process personal information.",
    "Quantum computing promises to solve certain computational problems that are intractable for classical computers.",
    "The open-source movement has fundamentally changed software development, enabling collaborative innovation on a global scale.",
    "Machine learning models trained on biased data can perpetuate and amplify existing social inequalities.",
    "Edge computing brings computation closer to data sources, reducing latency and bandwidth requirements for IoT applications.",
    "The integration of AI into healthcare has shown promise in early disease detection, drug discovery, and personalized treatment.",
    "Cybersecurity threats are evolving rapidly, requiring organizations to adopt proactive defense strategies and zero-trust architectures.",
    "Natural language processing has reached a point where machines can engage in nuanced conversations that are difficult to distinguish from human interactions.",
    "The rise of remote work has accelerated digital transformation and changed the dynamics of urban development.",
    "Retrieval-augmented generation combines the strengths of parametric and non-parametric memory to improve factual accuracy.",
    "Sparse mixture-of-experts architectures allow models to scale parameters without proportionally increasing inference cost.",
    "Instruction fine-tuning teaches language models to follow directions rather than simply predict the next token.",
    "Constitutional AI introduces a self-critique mechanism where models evaluate and revise their own outputs against a set of principles.",
    "Speculative decoding uses a smaller draft model to propose candidate tokens that are then verified by the target model in parallel.",
    "Flash attention dramatically reduces memory usage by reordering operations in the attention computation to avoid materializing the full attention matrix.",
    "Mechanistic interpretability aims to reverse-engineer neural networks by identifying circuits responsible for specific capabilities.",
    "Multimodal models that process images and text together have enabled breakthroughs in visual question answering and image captioning.",
    "The context window of modern language models has grown from a few hundred tokens to hundreds of thousands, enabling new long-document use cases.",
    "Continual learning addresses the problem of catastrophic forgetting, allowing models to acquire new knowledge without erasing what they already know.",
]

_EN_CHAT = [
    "Hey, how's it going? Haven't heard from you in a while!",
    "Can you help me with something? I'm totally stuck.",
    "That sounds amazing, I'd love to try that restaurant.",
    "No worries, take your time. I'll wait.",
    "Wait, are you serious? That's incredible!",
    "I just got back from the gym, feeling exhausted but good.",
    "What are you up to this weekend? Any fun plans?",
    "Honestly, I have no idea what's happening in that meeting.",
    "Could you send me the file again? I can't find it.",
    "Thanks so much, that really helped me out!",
    "lol yeah same, this week has been crazy busy.",
    "I think you're overthinking it, just go with your gut.",
    "Btw, did you see that news about the new model release?",
    "Let me know if you need anything else, happy to help.",
    "Sounds good! See you tomorrow then.",
    "Oh wow, I didn't expect that at all.",
    "Yeah the PR looks good to me, I'll approve it.",
    "Have you tried restarting it? Classic first step.",
    "I'll be five minutes late, go ahead and start without me.",
    "That's a really good point, I hadn't thought of it that way.",
]

# ── Code snippets ─────────────────────────────────────────────────────────────
_CODE_PY = [
    """import numpy as np
import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df['score'] = (df['score'] - df['score'].mean()) / df['score'].std()
    return df
""",
    """class TokenEstimator:
    def __init__(self, zh_tpc: float = 1.0, en_tpc: float = 0.25):
        self.zh_tpc = zh_tpc
        self.en_tpc = en_tpc

    def estimate(self, text: str) -> int:
        import re
        cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        return round(cjk * self.zh_tpc + (len(text) - cjk) * self.en_tpc)
""",
    """async def fetch_data(url: str, session: aiohttp.ClientSession) -> dict:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            return await resp.json()
    except aiohttp.ClientError as e:
        logging.error(f"Request failed: {e}")
        return {}
""",
    """def quicksort(arr: list) -> list:
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid  = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)
""",
    """@app.route('/api/v1/tokenize', methods=['POST'])
def tokenize():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'text is required'}), 400
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return jsonify({'token_count': len(tokens), 'tokens': tokens})
""",
    """from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Config:
    model_id: str = "gpt-4o"
    max_tokens: int = 2048
    temperature: float = 0.7
    stop_sequences: list[str] = field(default_factory=list)
    system_prompt: Optional[str] = None

    def validate(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature {self.temperature} out of range [0, 2]")
""",
    """import sqlite3
from contextlib import contextmanager

@contextmanager
def get_db(path: str):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
""",
    """def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    import functools, time
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            d = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    time.sleep(d)
                    d *= backoff
        return wrapper
    return decorator
""",
]

_CODE_JS = [
    """const fetchTokenCount = async (text) => {
  const response = await fetch('/api/tokenize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  const { token_count } = await response.json();
  return token_count;
};
""",
    """interface TokenEstimate {
  total: number;
  cjkChars: number;
  nonCjkChars: number;
}

function estimate(text: string, zhTpc = 1.0, enTpc = 0.25): TokenEstimate {
  const cjkPattern = /[\u4e00-\u9fff]/g;
  const cjkChars = (text.match(cjkPattern) ?? []).length;
  return {
    total: Math.round(cjkChars * zhTpc + (text.length - cjkChars) * enTpc),
    cjkChars,
    nonCjkChars: text.length - cjkChars,
  };
}
""",
    """// React component for token counter
export function TokenCounter({ text }: { text: string }) {
  const count = useMemo(() => estimate(text), [text]);
  return (
    <div className="token-counter">
      <span>{count.total} tokens</span>
      <small>({count.cjkChars} CJK + {count.nonCjkChars} other chars)</small>
    </div>
  );
}
""",
    """class EventEmitter<T extends Record<string, unknown[]>> {
  private listeners = new Map<keyof T, Set<(...args: unknown[]) => void>>();

  on<K extends keyof T>(event: K, listener: (...args: T[K]) => void): this {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event)!.add(listener as (...args: unknown[]) => void);
    return this;
  }

  emit<K extends keyof T>(event: K, ...args: T[K]): void {
    this.listeners.get(event)?.forEach(fn => fn(...args));
  }
}
""",
    """export async function* streamTokens(prompt: string): AsyncGenerator<string> {
  const response = await fetch('/api/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt }),
  });
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    yield decoder.decode(value, { stream: true });
  }
}
""",
]

_CODE_SHELL = [
    """#!/bin/bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-gpt-4o}"
OUTPUT_DIR="./results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Running benchmark for model: $MODEL_ID"
python benchmark.py --samples 50 --models all --csv "$OUTPUT_DIR/results.csv" --fit
echo "Done. Results saved to $OUTPUT_DIR"
""",
    """git log --oneline --graph --decorate --all | head -20
git diff --stat HEAD~1 HEAD
git stash list
git remote -v
""",
    """# Install deps and run tests
pip install -r requirements.txt -q
pip install pytest pytest-cov -q
pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=80
""",
    """docker build -t myapp:latest --build-arg VERSION=1.2.3 .
docker run -d --name myapp \\
    -e DATABASE_URL=postgres://user:pass@db:5432/mydb \\
    -e SECRET_KEY=$(openssl rand -hex 32) \\
    -p 8080:8080 \\
    --restart unless-stopped \\
    myapp:latest
docker logs -f myapp
""",
    """curl -X POST https://api.example.com/v1/completions \\
    -H "Authorization: Bearer $API_KEY" \\
    -H "Content-Type: application/json" \\
    -d '{
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 256
    }' | jq '.choices[0].message.content'
""",
    """find . -name "*.py" -not -path "./.venv/*" | xargs wc -l | sort -rn | head -20
grep -rn "TODO\\|FIXME\\|HACK\\|XXX" --include="*.py" .
ls -lah dist/ 2>/dev/null || echo "No dist directory"
""",
]

_MARKDOWN = [
    """# AI Model Comparison

## Overview
This document compares token estimation accuracy across four major LLMs:
- **GLM-5** by Zhipu AI
- **DeepSeek V3.2** by DeepSeek
- **Kimi K2.5** by Moonshot AI
- **MiniMax M2.5** by MiniMax

## Methodology
1. Generate random samples across 8 categories
2. Run each sample through all four tokenizers
3. Compare against linear approximation: `tokens ≈ zh_chars × α + en_chars × β`

## Results
| Model | Overall MAE | MAPE | Best Category |
|-------|------------|------|---------------|
| GLM-5 | TBD | TBD | pure_chinese |
""",
    """## 安装与使用

### 环境要求
- Python >= 3.10
- transformers >= 4.40

### 快速开始

```python
from tokenizer_approx import estimate

text = "这是一段中英文混合的 sample text。"
print(estimate(text))  # 输出近似token数
```

### 配置说明
通过环境变量自定义模型 ID：
```bash
export GLM_MODEL_ID="zai-org/GLM-5"
export DSV_MODEL_ID="deepseek-ai/DeepSeek-V3.2"
```
""",
    """## API Reference

### `estimate(text, coeffs=DEFAULT_COEFFS) -> int`

Returns the estimated token count for `text`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | — | Input text |
| `coeffs` | `Coeffs` | `DEFAULT_COEFFS` | Per-feature coefficients |

**Example:**
```python
>>> estimate("Hello, world!")
4
>>> estimate("你好世界")
4
```

### `extract_features(text) -> dict`

Returns raw feature counts without applying coefficients.
Keys: `cjk`, `letter`, `digit`, `punct`, `space`.
""",
    """# Changelog

## v2.0.0 — 2026-03-24

### Breaking Changes
- `estimate()` now accepts a `Coeffs` named-tuple instead of `zh_tpc`/`en_tpc` kwargs
- `estimate_detail()` return dict keys changed (added `letter_chars`, `digit_chars`, etc.)

### New Features
- 5-feature linear model: CJK / letters / digits / punctuation / whitespace
- `extract_features()` public function for custom downstream use
- `--fit` now fits all 5 coefficients via non-negative least squares

### Bug Fixes
- Fixed systematic under-estimation of numeric-heavy text (digits now use 0.50 coeff)
- Fixed over-estimation of indented code (whitespace now uses 0.08 coeff)
""",
]

_NUMERIC = [
    "3.14159265358979 × 2.71828182845904 = 8.53973422",
    "SELECT id, name, score FROM users WHERE score > 90.5 ORDER BY score DESC LIMIT 100;",
    "IPv4: 192.168.1.254/24  IPv6: 2001:0db8:85a3:0000:0000:8a2e:0370:7334",
    "SHA256: a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
    "2026-03-23T15:04:05.999Z  |  ERROR  |  [req-7f3a]  timeout after 30000ms",
    "f(x) = ∫₀¹ x² dx = [x³/3]₀¹ = 1/3 ≈ 0.333...",
    "CPU: 98.7% | MEM: 14.2GB/32GB | DISK: 234GB/1TB | NET↑: 1.2Gbps ↓: 0.8Gbps",
    "(A ∪ B) ∩ C = {x | x∈A ∨ x∈B} ∩ {x | x∈C}",
    "P(A|B) = P(B|A)·P(A) / P(B)  →  posterior ∝ likelihood × prior",
    "Layer norm: y = (x − μ) / √(σ² + ε) × γ + β,  where ε = 1e-5",
    "loss = −∑ yᵢ log(ŷᵢ) + λ(‖W‖² + ‖b‖²)   [cross-entropy + L2 reg]",
]

_NUMERIC_DENSE = [
    "550e8400-e29b-41d4-a716-446655440000",
    "2026-03-24T08:15:32.047Z 2026-03-24T08:15:32.891Z 2026-03-24T08:15:33.204Z",
    "192.168.0.1 10.0.0.254 172.16.255.255 127.0.0.1 0.0.0.0 255.255.255.255",
    "a3f8c2d1e4b5 0xDEADBEEF 0o777 0b10110011 255 65535 4294967295",
    "+1 (415) 555-0192  +86 138 0013 8000  +44 20 7946 0958  +81 3-1234-5678",
    "v1.2.3 v2.0.0-rc.1 v3.14.159-beta+build.2026.03.24 v0.0.1-alpha",
    "40.7128,-74.0060  51.5074,-0.1278  35.6762,139.6503  -33.8688,151.2093",
    "1000000 999999 123456789 9876543210 1e8 2.5e-3 6.022e23 1.602e-19",
    "ping 8.8.8.8: 14ms 12ms 18ms 11ms 16ms  avg=14.2ms loss=0%",
    "md5: d41d8cd98f00b204e9800998ecf8427e sha1: da39a3ee5e6b4b0d3255bfef95601890afd80709",
    "port 22 80 443 3306 5432 6379 8080 8443 27017 50051",
    "2^32=4294967296 2^64=18446744073709551616 log2(1000000)≈19.93 sqrt(2)≈1.41421356",
]

_URL_CODE = [
    "https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizer.json",
    "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig",
    "import os, sys, re, json, csv, time, math, random, hashlib, itertools, functools",
    "/home/user/.cache/huggingface/hub/models--zai-org--GLM-5/snapshots/abc123def456/",
    "C:\\Users\\kylin\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\",
    "https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types.py",
    "model_id = os.getenv('MODEL_ID', 'claude-sonnet-4-6')  # or 'claude-opus-4-6'",
    "pip install 'transformers>=4.40.0' 'torch>=2.1.0' 'accelerate>=0.27.0' sentencepiece",
    "SELECT u.id, u.name, COUNT(o.id) AS order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id HAVING order_count > 5;",
    "curl -s 'https://api.openai.com/v1/models' -H 'Authorization: Bearer $OPENAI_API_KEY' | python3 -m json.tool",
    "Error: ENOENT: no such file or directory, open '/var/run/app.pid'\n    at Object.openSync (node:fs:600:3)\n    at Module._resolveFilename (node:internal/modules/cjs/loader:1039:15)",
    "git remote add origin git@github.com:username/linear-tokenizer.git && git push -u origin main",
    "ARG PYTHON_VERSION=3.12\nFROM python:${PYTHON_VERSION}-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt",
]

# ── Public API ────────────────────────────────────────────────────────────────

CATEGORIES = [
    "pure_chinese",
    "pure_english",
    "chat_zh",
    "chat_en",
    "mixed",
    "code_py",
    "code_js",
    "code_shell",
    "markdown",
    "numeric",
    "numeric_dense",
    "url_code",
]


def _join_sentences(pool, min_s=1, max_s=3):
    n = random.randint(min_s, max_s)
    return "".join(random.choices(pool, k=n))


def generate_samples(n_per_category: int = 20) -> list[dict]:
    """Return list of dicts: {category, text}."""
    samples = []

    def add(cat, text):
        samples.append({"category": cat, "text": text})

    for _ in range(n_per_category):
        add("pure_chinese", _join_sentences(_ZH_SENTENCES, 1, 3))
    for _ in range(n_per_category):
        add("pure_english", _join_sentences(_EN_SENTENCES, 1, 3))
    for _ in range(n_per_category):
        add("chat_zh", _join_sentences(_ZH_CHAT, 2, 5))
    for _ in range(n_per_category):
        add("chat_en", _join_sentences(_EN_CHAT, 2, 5))
    for _ in range(n_per_category):
        # mixed: interleave zh and en sentences
        parts = []
        for _ in range(random.randint(2, 4)):
            if random.random() < 0.5:
                parts.append(random.choice(_ZH_SENTENCES))
            else:
                parts.append(random.choice(_EN_SENTENCES))
        add("mixed", " ".join(parts))
    for _ in range(n_per_category):
        add("code_py", random.choice(_CODE_PY) * random.randint(1, 3))
    for _ in range(n_per_category):
        add("code_js", random.choice(_CODE_JS) * random.randint(1, 2))
    for _ in range(n_per_category):
        add("code_shell", random.choice(_CODE_SHELL) * random.randint(1, 2))
    for _ in range(n_per_category):
        add("markdown", random.choice(_MARKDOWN))
    for _ in range(n_per_category):
        add("numeric", random.choice(_NUMERIC) * random.randint(1, 3))
    for _ in range(n_per_category):
        add("numeric_dense", random.choice(_NUMERIC_DENSE) * random.randint(1, 3))
    for _ in range(n_per_category):
        add("url_code", random.choice(_URL_CODE) * random.randint(1, 2))

    random.shuffle(samples)
    return samples
