"""
Linear token count approximation — 6-feature model.

Five character-class features (exhaustive, mutually exclusive):
  cjk    – CJK ideographs + CJK punctuation + fullwidth forms
  letter – ASCII letters  [A-Za-z]
  digit  – Decimal digits [0-9]
  punct  – Everything else: ASCII symbols, other Unicode (emoji, Cyrillic…)
  space  – Whitespace     [ \\t\\n\\r…]

Plus one word-level feature:
  word   – whitespace-split word count (captures BPE word-boundary behaviour;
           particularly useful for English prose where word count correlates
           more tightly with token count than letter-char count alone)

Default coefficients are fitted via NNLS on 261 real k-length samples
(Wikipedia + GitHub) across GLM-5, Kimi-K2.5, DeepSeek-V3.2, MiniMax-M2.5.
Run benchmark.py --real-data --fit-shared to refit.
"""

import re
from typing import NamedTuple

# ── Character-class patterns ──────────────────────────────────────────────────
# CJK: Unified Ideographs, Extension A/B, Compatibility Ideographs,
#      CJK Symbols & Punctuation, Halfwidth/Fullwidth Forms
_CJK_RE = re.compile(
    "["
    "\u4e00-\u9fff"          # CJK Unified Ideographs
    "\u3400-\u4dbf"          # CJK Extension A
    "\U00020000-\U0002a6df"  # CJK Extension B
    "\uf900-\ufaff"          # CJK Compatibility Ideographs
    "\u3000-\u303f"          # CJK Symbols and Punctuation  (。，、！？…)
    "\uff00-\uffef"          # Halfwidth and Fullwidth Forms (ａｂｃ１２３)
    "]"
)
_LETTER_RE  = re.compile(r"[A-Za-z]")
_DIGIT_RE   = re.compile(r"[0-9]")
_SPACE_RE   = re.compile(r"\s")


# ── Coefficient container ──────────────────────────────────────────────────────
class Coeffs(NamedTuple):
    """Tokens-per-unit for each feature. char features: tokens/char; word: tokens/word."""
    cjk:    float = 0.6330   # CJK chars
    letter: float = 0.1406   # ASCII letter chars
    digit:  float = 0.7876   # digit chars
    punct:  float = 0.7115   # other chars (punct/symbols)
    space:  float = 0.0995   # whitespace chars
    word:   float = 0.3633   # whitespace-split words

DEFAULT_COEFFS = Coeffs()

# Backward-compat aliases
ZH_TPC: float = DEFAULT_COEFFS.cjk
EN_TPC: float = DEFAULT_COEFFS.letter


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(text: str) -> dict[str, int]:
    """
    Return a dict with six features.

    The first five (cjk/letter/digit/punct/space) are mutually exclusive
    char counts that sum to len(text).  'word' is an independent word count.
    """
    cjk    = len(_CJK_RE.findall(text))
    letter = len(_LETTER_RE.findall(text))
    digit  = len(_DIGIT_RE.findall(text))
    space  = len(_SPACE_RE.findall(text))
    punct  = len(text) - cjk - letter - digit - space
    word   = len(text.split())
    return {
        "cjk":    cjk,
        "letter": letter,
        "digit":  digit,
        "punct":  punct,
        "space":  space,
        "word":   word,
    }


# ── Public API ────────────────────────────────────────────────────────────────
def estimate(text: str, coeffs: Coeffs = DEFAULT_COEFFS) -> int:
    """
    Estimate token count without any tokenizer.

    Uses a 5-feature linear model; see module docstring for details.
    Pass a custom Coeffs() to use fitted per-model coefficients.
    """
    if not text:
        return 0
    f = extract_features(text)
    total = (
        f["cjk"]    * coeffs.cjk    +
        f["letter"] * coeffs.letter +
        f["digit"]  * coeffs.digit  +
        f["punct"]  * coeffs.punct  +
        f["space"]  * coeffs.space  +
        f["word"]   * coeffs.word
    )
    return max(1, round(total))


def estimate_detail(text: str, coeffs: Coeffs = DEFAULT_COEFFS) -> dict:
    """Same as estimate() but returns a full breakdown dict."""
    if not text:
        return {
            "total": 0,
            "cjk_chars": 0, "letter_chars": 0, "digit_chars": 0,
            "punct_chars": 0, "space_chars": 0,
            "cjk_tokens": 0.0, "letter_tokens": 0.0, "digit_tokens": 0.0,
            "punct_tokens": 0.0, "space_tokens": 0.0,
        }
    f = extract_features(text)
    cjk_t    = f["cjk"]    * coeffs.cjk
    letter_t = f["letter"] * coeffs.letter
    digit_t  = f["digit"]  * coeffs.digit
    punct_t  = f["punct"]  * coeffs.punct
    space_t  = f["space"]  * coeffs.space
    word_t   = f["word"]   * coeffs.word
    return {
        "total":         max(1, round(cjk_t + letter_t + digit_t + punct_t + space_t + word_t)),
        "cjk_chars":     f["cjk"],
        "letter_chars":  f["letter"],
        "digit_chars":   f["digit"],
        "punct_chars":   f["punct"],
        "space_chars":   f["space"],
        "word_count":    f["word"],
        "cjk_tokens":    cjk_t,
        "letter_tokens": letter_t,
        "digit_tokens":  digit_t,
        "punct_tokens":  punct_t,
        "space_tokens":  space_t,
        "word_tokens":   word_t,
    }
