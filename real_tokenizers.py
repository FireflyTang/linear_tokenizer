"""
Thin wrappers around the real tokenizers.

Supported models (configurable via env vars):
  GLM_MODEL_ID   – default: "zai-org/GLM-5"
  DSV_MODEL_ID   – default: "deepseek-ai/DeepSeek-V3.2"
  KIMI_MODEL_ID  – default: "moonshotai/Kimi-K2.5"
  MMAX_MODEL_ID  – default: "MiniMaxAI/MiniMax-M2.5"

Only tokenizer files are downloaded (~few MB each). Model weights are NOT fetched.

Proxy: HTTP_PROXY / HTTPS_PROXY env vars are respected by huggingface_hub.
Default proxy applied: http://127.0.0.1:7897

Fast Kimi counter
-----------------
Kimi-K2.5's tokenizer wraps OpenAI's tiktoken cl100k_base internally
(see tokenization_kimi.py in the HF repo). kimi_count_fast() uses tiktoken
directly — no HF download, loads in <100ms, produces identical token counts.

    pip install tiktoken   # ~2 MB, no model weights

kimi_count() is the canonical reference (AutoTokenizer); kimi_count_fast()
is the drop-in replacement for production use.
"""

import os
from functools import lru_cache

GLM_MODEL_ID  = os.getenv("GLM_MODEL_ID",  "zai-org/GLM-5")
DSV_MODEL_ID  = os.getenv("DSV_MODEL_ID",  "deepseek-ai/DeepSeek-V3.2")
KIMI_MODEL_ID = os.getenv("KIMI_MODEL_ID", "moonshotai/Kimi-K2.5")
MMAX_MODEL_ID = os.getenv("MMAX_MODEL_ID", "MiniMaxAI/MiniMax-M2.5")

_DEFAULT_PROXY = os.getenv("HF_PROXY", "http://127.0.0.1:7897")


def _apply_proxy():
    if _DEFAULT_PROXY:
        os.environ.setdefault("HTTP_PROXY",  _DEFAULT_PROXY)
        os.environ.setdefault("HTTPS_PROXY", _DEFAULT_PROXY)


def _load_auto(model_id: str):
    """Load via AutoTokenizer (works for GLM, Kimi, MiniMax)."""
    _apply_proxy()
    from transformers import AutoTokenizer
    print(f"[tokenizer] loading {model_id} …")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"[tokenizer] {model_id} ready  (vocab={tok.vocab_size})")
    return tok


def _load_fast_from_cache(model_id: str):
    """
    Load via PreTrainedTokenizerFast directly from the cached tokenizer.json.

    Required for DeepSeek-V3.2 whose HuggingFace entry maps to LlamaTokenizer
    (SentencePiece) but the actual tokenizer file is BPE-based, causing
    AutoTokenizer to silently return empty token lists for non-ASCII text.
    """
    _apply_proxy()
    from huggingface_hub import snapshot_download
    from transformers import PreTrainedTokenizerFast

    print(f"[tokenizer] downloading tokenizer files for {model_id} …")
    local_dir = snapshot_download(
        repo_id=model_id,
        ignore_patterns=["*.bin", "*.safetensors", "*.pt", "*.gguf"],
    )
    tok_json = os.path.join(local_dir, "tokenizer.json")
    if not os.path.exists(tok_json):
        raise FileNotFoundError(f"tokenizer.json not found in {local_dir}")
    tok = PreTrainedTokenizerFast(tokenizer_file=tok_json)
    print(f"[tokenizer] {model_id} ready  (vocab={tok.vocab_size})")
    return tok


@lru_cache(maxsize=1)
def _glm():
    return _load_auto(GLM_MODEL_ID)

@lru_cache(maxsize=1)
def _dsv():
    # Must use PreTrainedTokenizerFast; AutoTokenizer picks the wrong backend.
    return _load_fast_from_cache(DSV_MODEL_ID)

@lru_cache(maxsize=1)
def _kimi():
    return _load_auto(KIMI_MODEL_ID)

@lru_cache(maxsize=1)
def _mmax():
    return _load_auto(MMAX_MODEL_ID)


def _count(tok, text: str) -> int:
    ids = tok.encode(text, add_special_tokens=False)
    return len(ids)


def glm_count(text: str)  -> int: return _count(_glm(),  text)
def dsv_count(text: str)  -> int: return _count(_dsv(),  text)
def kimi_count(text: str) -> int: return _count(_kimi(), text)
def mmax_count(text: str) -> int: return _count(_mmax(), text)


# ── Fast Kimi via tiktoken ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _kimi_tiktoken():
    """
    Load tiktoken cl100k_base — Kimi-K2.5 uses this encoding internally.
    Requires: pip install tiktoken
    """
    try:
        import tiktoken
    except ImportError:
        raise ImportError("tiktoken not installed. Run: pip install tiktoken")
    enc = tiktoken.get_encoding("cl100k_base")
    print(f"[tokenizer] tiktoken cl100k_base ready  (vocab={enc.n_vocab})")
    return enc


def kimi_count_fast(text: str) -> int:
    """
    Count Kimi-K2.5 tokens via tiktoken cl100k_base.

    Equivalent to kimi_count() but ~10x faster to initialise (no HF download)
    and ~3x faster per call. Safe to use as a drop-in replacement for Kimi.
    """
    return len(_kimi_tiktoken().encode(text))
