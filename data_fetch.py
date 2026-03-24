"""
Fetch real-world k-length text samples for benchmark use.

Sources
-------
  Wikipedia  – English and Chinese articles via MediaWiki API (plain text)
  GitHub     – Raw source files from popular open-source repos

All fetched texts are cached under .sample_cache/ as plain UTF-8 files so
that re-runs are instant.  Pass force_refresh=True (or delete the directory)
to re-download.

Each sample returned by fetch_all() is a dict:
    {"category": str, "source": str, "text": str}

The "category" mirrors sample_gen.CATEGORIES so benchmark.py can mix both
synthetic and real samples seamlessly.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

CACHE_DIR = Path(".sample_cache")
_UA = "linear-tokenizer-benchmark/2.0 (educational; github.com)"
_DELAY = 0.5   # seconds between requests (be a good citizen)

# ── Sources ───────────────────────────────────────────────────────────────────

# Wikipedia articles: (title_for_API, lang, category_label)
_WIKI_ARTICLES = [
    # English — tech/AI topics
    ("Transformer_(deep_learning_architecture)", "en", "pure_english"),
    ("Large_language_model",                     "en", "pure_english"),
    ("Attention_(machine_learning)",             "en", "pure_english"),
    ("Generative_pre-trained_transformer",       "en", "pure_english"),
    ("Convolutional_neural_network",             "en", "pure_english"),
    ("Python_(programming_language)",            "en", "pure_english"),
    ("Reinforcement_learning",                   "en", "pure_english"),
    ("Recurrent_neural_network",                 "en", "pure_english"),
    ("Byte_pair_encoding",                       "en", "pure_english"),
    ("Tokenization_(lexical_analysis)",          "en", "pure_english"),
    # Chinese Wikipedia
    ("大型语言模型",   "zh", "pure_chinese"),
    ("人工智能",       "zh", "pure_chinese"),
    ("深度学习",       "zh", "pure_chinese"),
    ("自然语言处理",   "zh", "pure_chinese"),
    ("卷积神经网络",   "zh", "pure_chinese"),
    ("强化学习",       "zh", "pure_chinese"),
    ("生成对抗网络",   "zh", "pure_chinese"),
    ("变换器模型",     "zh", "pure_chinese"),
    ("Python",         "zh", "pure_chinese"),
]

# GitHub raw files: (owner, repo, branch, path, category)
_GITHUB_FILES = [
    # CPython stdlib
    ("python", "cpython", "main", "Lib/ast.py",            "code_py"),
    ("python", "cpython", "main", "Lib/json/__init__.py",  "code_py"),
    ("python", "cpython", "main", "Lib/pathlib/_local.py", "code_py"),
    ("python", "cpython", "main", "Lib/typing.py",         "code_py"),
    ("python", "cpython", "main", "Lib/dataclasses.py",    "code_py"),
    # Popular Python libraries
    ("psf",      "requests", "main",   "src/requests/models.py",   "code_py"),
    ("psf",      "requests", "main",   "src/requests/adapters.py", "code_py"),
    ("pallets",  "flask",    "main",   "src/flask/app.py",         "code_py"),
    ("tiangolo", "fastapi",  "master", "fastapi/routing.py",       "code_py"),
    ("tiangolo", "fastapi",  "master", "fastapi/applications.py",  "code_py"),
    # JavaScript / TypeScript
    ("expressjs", "express", "master", "lib/application.js",        "code_js"),
    ("expressjs", "express", "master", "lib/router/index.js",       "code_js"),
    ("axios",     "axios",   "v1.x",   "lib/core/Axios.js",         "code_js"),
    ("axios",     "axios",   "v1.x",   "lib/adapters/http.js",      "code_js"),
    # Go — standard library (dense comments + complex logic)
    ("golang", "go", "master", "src/encoding/json/encode.go",   "code_go"),
    ("golang", "go", "master", "src/net/http/server.go",         "code_go"),
    ("golang", "go", "master", "src/sync/map.go",                "code_go"),
    # Rust — standard library
    ("rust-lang", "rust", "master", "library/std/src/collections/hash/map.rs", "code_rust"),
    ("rust-lang", "rust", "master", "library/core/src/iter/traits/iterator.rs","code_rust"),
    # Shell / DevOps
    ("nvm-sh",   "nvm",    "master", "install.sh",          "code_shell"),
    ("ohmyzsh",  "ohmyzsh","master", "tools/install.sh",    "code_shell"),
    # Chinese code with Chinese comments (PaddleNLP — well-commented Chinese ML code)
    ("PaddlePaddle", "PaddleNLP", "develop",
     "paddlenlp/transformers/tokenizer_utils.py", "mixed"),
    ("PaddlePaddle", "PaddleNLP", "develop",
     "paddlenlp/transformers/auto/tokenization.py", "mixed"),
    # More Python — numpy, pandas, scikit-learn
    ("numpy",        "numpy",     "main",   "numpy/core/fromnumeric.py",          "code_py"),
    ("numpy",        "numpy",     "main",   "numpy/lib/function_base.py",         "code_py"),
    ("pandas-dev",   "pandas",    "main",   "pandas/core/frame.py",               "code_py"),
    ("pandas-dev",   "pandas",    "main",   "pandas/core/groupby/groupby.py",     "code_py"),
    ("scikit-learn", "scikit-learn", "main","sklearn/ensemble/_forest.py",        "code_py"),
    # C++ — LLVM, protobuf, abseil, OpenCV
    ("llvm",         "llvm-project", "main", "llvm/lib/Analysis/LoopInfo.cpp",          "code_cpp"),
    ("llvm",         "llvm-project", "main", "clang/lib/Sema/SemaDecl.cpp",            "code_cpp"),
    ("protocolbuffers", "protobuf", "main",  "src/google/protobuf/descriptor.cc",       "code_cpp"),
    ("protocolbuffers", "protobuf", "main",  "src/google/protobuf/descriptor.h",        "code_cpp"),
    ("abseil",       "abseil-cpp", "master", "absl/container/internal/raw_hash_set.h",  "code_cpp"),
    ("abseil",       "abseil-cpp", "master", "absl/strings/str_format.h",               "code_cpp"),
    ("opencv",       "opencv",    "4.x",    "modules/core/src/matrix.cpp",             "code_cpp"),
    ("opencv",       "opencv",    "4.x",    "modules/imgproc/src/color.cpp",           "code_cpp"),
]


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(url: str, *, label: str = "") -> Optional[str]:
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        print(f"  [fetch] HTTP {e.code} — {label or url}")
        return None
    except Exception as e:
        print(f"  [fetch] Error — {label or url}: {e}")
        return None


# ── Wikipedia ────────────────────────────────────────────────────────────────

def _wiki_url(title: str, lang: str) -> str:
    t = urllib.parse.quote(title, safe="()")
    return (
        f"https://{lang}.wikipedia.org/w/api.php"
        f"?action=query&prop=extracts&explaintext=1&exlimit=1"
        f"&titles={t}&format=json&formatversion=2"
    )


def _fetch_wiki(title: str, lang: str) -> Optional[str]:
    raw = _get(_wiki_url(title, lang), label=f"wikipedia:{lang}:{title}")
    if not raw:
        return None
    try:
        data = json.loads(raw)
        pages = data["query"]["pages"]
        text = pages[0].get("extract", "")
        return text.strip() or None
    except Exception:
        return None


# ── GitHub raw ────────────────────────────────────────────────────────────────

def _github_raw_url(owner: str, repo: str, branch: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    safe = key.replace("/", "__").replace(":", "_").replace(" ", "_")[:180]
    return CACHE_DIR / f"{safe}.txt"


def _cached(key: str) -> Optional[str]:
    p = _cache_path(key)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return None


def _save(key: str, text: str) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    _cache_path(key).write_text(text, encoding="utf-8")


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk(text: str, size: int = 4000, overlap: int = 0) -> list[str]:
    """Split text into chunks of ~size characters (whole-line boundaries)."""
    lines = text.splitlines(keepends=True)
    chunks, buf = [], []
    n = 0
    for line in lines:
        buf.append(line)
        n += len(line)
        if n >= size:
            chunks.append("".join(buf))
            buf, n = [], 0
    if buf:
        chunks.append("".join(buf))
    return [c for c in chunks if len(c.strip()) > 200]


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_all(force_refresh: bool = False, chunk_size: int = 4000) -> list[dict]:
    """
    Fetch (or load from cache) all configured sources.

    Returns a list of dicts: {"category", "source", "text"}.
    Each chunk is ~chunk_size characters (a few thousand tokens).
    """
    samples: list[dict] = []

    # ── Wikipedia ─────────────────────────────────────────────────────────────
    print(f"[data_fetch] Fetching {len(_WIKI_ARTICLES)} Wikipedia articles …")
    for title, lang, cat in _WIKI_ARTICLES:
        key = f"wiki_{lang}_{title}"
        text = None if force_refresh else _cached(key)
        if text is None:
            print(f"  downloading: {lang}:{title}")
            text = _fetch_wiki(title, lang)
            if text:
                _save(key, text)
            time.sleep(_DELAY)
        else:
            print(f"  cached:      {lang}:{title}")
        if not text:
            continue
        for i, chunk in enumerate(_chunk(text, chunk_size)):
            samples.append({
                "category": cat,
                "source":   f"wikipedia:{lang}:{title}[{i}]",
                "text":     chunk,
            })

    # ── GitHub ────────────────────────────────────────────────────────────────
    print(f"[data_fetch] Fetching {len(_GITHUB_FILES)} GitHub files …")
    for owner, repo, branch, path, cat in _GITHUB_FILES:
        key = f"github_{owner}_{repo}_{branch}_{path}"
        text = None if force_refresh else _cached(key)
        if text is None:
            url = _github_raw_url(owner, repo, branch, path)
            print(f"  downloading: {owner}/{repo}/{path}")
            text = _get(url, label=f"{owner}/{repo}/{path}")
            if text:
                _save(key, text)
            time.sleep(_DELAY)
        else:
            print(f"  cached:      {owner}/{repo}/{path}")
        if not text:
            continue
        for i, chunk in enumerate(_chunk(text, chunk_size)):
            samples.append({
                "category": cat,
                "source":   f"github:{owner}/{repo}/{path}[{i}]",
                "text":     chunk,
            })

    print(f"[data_fetch] Total chunks: {len(samples)}  "
          f"(avg {sum(len(s['text']) for s in samples)//max(len(samples),1)} chars each)")
    return samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch and cache real-world benchmark data")
    parser.add_argument("--refresh", action="store_true", help="Force re-download (ignore cache)")
    parser.add_argument("--chunk-size", type=int, default=4000)
    args = parser.parse_args()

    samples = fetch_all(force_refresh=args.refresh, chunk_size=args.chunk_size)
    from collections import Counter
    cats = Counter(s["category"] for s in samples)
    print("\nChunks per category:")
    for cat, n in sorted(cats.items()):
        avg = sum(len(s["text"]) for s in samples if s["category"] == cat) // n
        print(f"  {cat:<16} {n:>3} chunks  avg {avg:>5} chars")
