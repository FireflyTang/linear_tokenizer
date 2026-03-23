"""
Linear token count approximation — 5-feature model.

Decomposes every character into one of five mutually-exclusive buckets and
applies a separate tokens-per-character coefficient to each:

  cjk    – CJK ideographs + CJK punctuation + fullwidth forms
  letter – ASCII letters  [A-Za-z]
  digit  – Decimal digits [0-9]
  punct  – Everything else: ASCII symbols, other Unicode (emoji, Cyrillic…)
  space  – Whitespace     [ \\t\\n\\r…]

The five buckets are exhaustive: cjk + letter + digit + punct + space == len(text).

Why 5 features beat the old 2-feature model:
  • digits tokenise at ~2 chars/token (0.50), not 4 (0.25) — biggest win for numeric text
  • punctuation tokenises at ~1 char/token (0.90), not 4 — fixes code and symbol-heavy text
  • spaces are nearly free in BPE (merged into adjacent words) — fixes indented code over-estimate
  • letters stay close to 0.25–0.27 but are no longer contaminated by digits/punct

Default coefficients are reasonable starting points; run benchmark.py --fit to get
model-specific values via least-squares.
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
    """Tokens-per-character for each feature bucket."""
    cjk:    float = 1.00   # CJK chars:  ~1 token/char
    letter: float = 0.27   # ASCII letters: ~3.7 chars/token
    digit:  float = 0.50   # Digits:      ~2 chars/token
    punct:  float = 0.90   # Symbols/punct: ~1.1 chars/token
    space:  float = 0.08   # Whitespace: mostly merged in BPE

DEFAULT_COEFFS = Coeffs()

# Backward-compat aliases
ZH_TPC: float = DEFAULT_COEFFS.cjk
EN_TPC: float = DEFAULT_COEFFS.letter


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(text: str) -> dict[str, int]:
    """
    Return a dict with five mutually-exclusive character counts.
    The counts sum to len(text).
    """
    cjk    = len(_CJK_RE.findall(text))
    letter = len(_LETTER_RE.findall(text))
    digit  = len(_DIGIT_RE.findall(text))
    space  = len(_SPACE_RE.findall(text))
    punct  = len(text) - cjk - letter - digit - space
    return {
        "cjk":    cjk,
        "letter": letter,
        "digit":  digit,
        "punct":  punct,
        "space":  space,
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
        f["cjk"]    * coeffs.cjk  +
        f["letter"] * coeffs.letter +
        f["digit"]  * coeffs.digit +
        f["punct"]  * coeffs.punct +
        f["space"]  * coeffs.space
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
    return {
        "total":         max(1, round(cjk_t + letter_t + digit_t + punct_t + space_t)),
        "cjk_chars":     f["cjk"],
        "letter_chars":  f["letter"],
        "digit_chars":   f["digit"],
        "punct_chars":   f["punct"],
        "space_chars":   f["space"],
        "cjk_tokens":    cjk_t,
        "letter_tokens": letter_t,
        "digit_tokens":  digit_t,
        "punct_tokens":  punct_t,
        "space_tokens":  space_t,
    }
