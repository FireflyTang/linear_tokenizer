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

Default coefficients are fitted via NNLS on 1117 real samples
(Wikipedia + GitHub) across GLM-5, Kimi-K2.5, DeepSeek-V3.2, MiniMax-M2.5.
Run benchmark.py --real-data --fit-shared to refit.
"""

import re
from typing import NamedTuple

# ── C extension (optional fast backend) ───────────────────────────────────────
# _features.cp312-win_amd64.pyd (or .so on Linux/macOS) — built from _features.c.
# Provides single-pass char classification directly on CPython's internal buffer;
# ~43x faster than tiktoken, ~3x faster than numpy at 160K tokens.
# Falls back to regex if not available (no behavioural difference).
try:
    import _features as _C
    _USE_C = True
except ImportError:
    _C = None
    _USE_C = False

# ── Character-class patterns (regex fallback) ─────────────────────────────────
_CJK_RE    = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")
_LETTER_RE = re.compile(r"[A-Za-z]")
_DIGIT_RE  = re.compile(r"[0-9]")
_SPACE_RE  = re.compile(r"\s")


# ── Coefficient container ──────────────────────────────────────────────────────
class Coeffs(NamedTuple):
    """Tokens-per-unit for each feature. char features: tokens/char; word: tokens/word."""
    cjk:    float = 0.6223   # CJK chars
    letter: float = 0.1766   # ASCII letter chars
    digit:  float = 1.1655   # digit chars
    punct:  float = 0.7246   # other chars (punct/symbols)
    space:  float = 0.0971   # whitespace chars
    word:   float = 0.1947   # whitespace-split words

DEFAULT_COEFFS = Coeffs()

# Upper-bound coefficients — fitted via LP (scipy linprog/HiGHS) on 1117 real
# samples (Wikipedia + GitHub) across all 4 models.
# Guarantees estimate ≥ real on training corpus; mean overestimate ~37%, max ~112%.
# Run:  python benchmark.py --real-data --fit-upper   to refit.
UPPER_COEFFS = Coeffs(
    cjk    = 0.7591,
    letter = 0.2135,
    digit  = 1.8696,
    punct  = 1.1526,
    space  = 0.0980,
    word   = 0.3077,
)

# Backward-compat aliases
ZH_TPC: float = DEFAULT_COEFFS.cjk
EN_TPC: float = DEFAULT_COEFFS.letter


# ── 7-feature coefficient container ───────────────────────────────────────────
class Coeffs7(NamedTuple):
    """7-feature variant: digit split into digit_iso (isolated) and digit_run (runs 2+)."""
    cjk:       float = 0.0   # CJK chars
    letter:    float = 0.0   # ASCII letter chars
    digit_iso: float = 0.0   # isolated digit chars (neighbours are non-digit)
    digit_run: float = 0.0   # digit chars in runs of 2+ consecutive digits
    punct:     float = 0.0   # other chars (punct/symbols)
    space:     float = 0.0   # whitespace chars
    word:      float = 0.0   # whitespace-split words

# Upper-bound coefficients — fitted via LP on 1117 real samples (same corpus as
# UPPER_COEFFS). Guarantees estimate >= real; max overestimate ~102% vs ~112% for
# the 6-feature model. Run _fit7.py to refit.
UPPER_COEFFS7 = Coeffs7(
    cjk       = 0.7869,
    letter    = 0.2052,
    digit_iso = 2.5475,
    digit_run = 0.8148,
    punct     = 1.0791,
    space     = 0.0976,
    word      = 0.4461,
)

# Default coefficients — fitted via NNLS on same corpus. Run _fit7.py to refit.
DEFAULT_COEFFS7 = Coeffs7(
    cjk       = 0.6286,
    letter    = 0.1929,
    digit_iso = 2.1975,
    digit_run = 0.7170,
    punct     = 0.6911,
    space     = 0.0912,
    word      = 0.1308,
)


# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(text: str) -> dict[str, int]:
    """
    Return a dict with six features.

    The first five (cjk/letter/digit/punct/space) are mutually exclusive
    char counts that sum to len(text).  'word' is an independent word count.
    Uses the C extension when available, falls back to regex.
    """
    if _USE_C:
        cjk, letter, digit, punct, space, word = _C.extract_features(text)
        return {"cjk": cjk, "letter": letter, "digit": digit,
                "punct": punct, "space": space, "word": word}
    cjk    = len(_CJK_RE.findall(text))
    letter = len(_LETTER_RE.findall(text))
    digit  = len(_DIGIT_RE.findall(text))
    space  = len(_SPACE_RE.findall(text))
    return {
        "cjk":    cjk,
        "letter": letter,
        "digit":  digit,
        "punct":  len(text) - cjk - letter - digit - space,
        "space":  space,
        "word":   len(text.split()),
    }


def extract_features_fast(text: str) -> tuple[int, int, int, int, int, int]:
    """
    Same classification as extract_features() but returns a flat tuple
    (cjk, letter, digit, punct, space, word).
    Uses numpy vectorised ops on UTF-32 codepoints — avoids Python loops entirely.
    """
    import numpy as np
    cp = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    cjk = int(np.sum(
        ((cp >= 0x4e00) & (cp <= 0x9fff))
        | ((cp >= 0x3000) & (cp <= 0x303f))
        | ((cp >= 0xff00) & (cp <= 0xffef))
    ))
    letter = int(np.sum(((cp >= 0x41) & (cp <= 0x5a)) | ((cp >= 0x61) & (cp <= 0x7a))))
    digit  = int(np.sum((cp >= 0x30) & (cp <= 0x39)))
    space  = int(np.sum((cp == 0x20) | (cp == 0x09) | (cp == 0x0a) | (cp == 0x0d)))
    punct  = len(cp) - cjk - letter - digit - space
    word   = len(text.split())
    return cjk, letter, digit, punct, space, word


def estimate_fast(text: str, coeffs: Coeffs = DEFAULT_COEFFS) -> int:
    """estimate() using numpy-accelerated feature extraction."""
    if not text:
        return 0
    cjk, letter, digit, punct, space, word = extract_features_fast(text)
    total = (
        cjk    * coeffs.cjk    +
        letter * coeffs.letter +
        digit  * coeffs.digit  +
        punct  * coeffs.punct  +
        space  * coeffs.space  +
        word   * coeffs.word
    )
    return max(1, round(total))


def estimate_fast_noword(text: str, coeffs: Coeffs = DEFAULT_COEFFS) -> int:
    """estimate_fast without word count — skips text.split()."""
    if not text:
        return 0
    import numpy as np
    cp = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    cjk = int(np.sum(
        ((cp >= 0x4e00) & (cp <= 0x9fff))
        | ((cp >= 0x3000) & (cp <= 0x303f))
        | ((cp >= 0xff00) & (cp <= 0xffef))
    ))
    letter = int(np.sum(((cp >= 0x41) & (cp <= 0x5a)) | ((cp >= 0x61) & (cp <= 0x7a))))
    digit  = int(np.sum((cp >= 0x30) & (cp <= 0x39)))
    space  = int(np.sum((cp == 0x20) | (cp == 0x09) | (cp == 0x0a) | (cp == 0x0d)))
    punct  = len(cp) - cjk - letter - digit - space
    total = (
        cjk    * coeffs.cjk    +
        letter * coeffs.letter +
        digit  * coeffs.digit  +
        punct  * coeffs.punct  +
        space  * coeffs.space
    )
    return max(1, round(total))


def estimate_numpy_full(text: str, coeffs: Coeffs = DEFAULT_COEFFS) -> int:
    """All 6 features via numpy — word count from space→non-space transitions."""
    if not text:
        return 0
    import numpy as np
    cp = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    is_space = (cp == 0x20) | (cp == 0x09) | (cp == 0x0a) | (cp == 0x0d)
    cjk = int(np.sum(
        ((cp >= 0x4e00) & (cp <= 0x9fff))
        | ((cp >= 0x3000) & (cp <= 0x303f))
        | ((cp >= 0xff00) & (cp <= 0xffef))
    ))
    letter = int(np.sum(((cp >= 0x41) & (cp <= 0x5a)) | ((cp >= 0x61) & (cp <= 0x7a))))
    digit  = int(np.sum((cp >= 0x30) & (cp <= 0x39)))
    space  = int(np.sum(is_space))
    punct  = len(cp) - cjk - letter - digit - space
    # Word count: transitions from space to non-space, plus leading non-space
    word = int(np.sum(is_space[:-1] & ~is_space[1:])) + (0 if is_space[0] else 1) if len(cp) > 0 else 0
    total = (
        cjk    * coeffs.cjk    +
        letter * coeffs.letter +
        digit  * coeffs.digit  +
        punct  * coeffs.punct  +
        space  * coeffs.space  +
        word   * coeffs.word
    )
    return max(1, round(total))


def estimate_ctypes(text: str, coeffs: Coeffs = DEFAULT_COEFFS) -> int:
    """estimate using array module + memoryview — no numpy, no regex."""
    if not text:
        return 0
    import array as _array
    raw = text.encode("utf-32-le")
    cp = _array.array("I")
    cp.frombytes(raw)
    n = len(cp)
    # C-level iteration via memoryview is not vectorised,
    # but avoids Python object creation per character.
    cjk = letter = digit = space = word = 0
    mv = memoryview(cp)
    in_word = False
    for c in mv:
        if c <= 0x7f:
            if 0x41 <= c <= 0x5a or 0x61 <= c <= 0x7a:
                letter += 1
                if not in_word: word += 1; in_word = True
            elif 0x30 <= c <= 0x39:
                digit += 1
                if not in_word: word += 1; in_word = True
            elif c == 0x20 or c == 0x09 or c == 0x0a or c == 0x0d:
                space += 1
                in_word = False
            else:
                if not in_word: word += 1; in_word = True
        else:
            if (0x4e00 <= c <= 0x9fff or 0x3000 <= c <= 0x303f
                    or 0xff00 <= c <= 0xffef):
                cjk += 1
            if not in_word: word += 1; in_word = True
    punct = n - cjk - letter - digit - space
    total = (
        cjk    * coeffs.cjk    +
        letter * coeffs.letter +
        digit  * coeffs.digit  +
        punct  * coeffs.punct  +
        space  * coeffs.space  +
        word   * coeffs.word
    )
    return max(1, round(total))


# ── Public API ────────────────────────────────────────────────────────────────
def estimate(text: str, coeffs: Coeffs = DEFAULT_COEFFS) -> int:
    """
    Estimate token count without any tokenizer.

    Uses the C extension when available (43x faster than tiktoken at 160K tokens),
    falls back to regex. Pass a custom Coeffs() to use per-model coefficients.
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


def estimate_naive(text: str) -> int:
    """
    Naive baseline estimator: ASCII chars / 5, non-ASCII chars / 2.

    No fitting required — useful as a lower-bound on how well a trivial
    rule-of-thumb performs compared to the fitted linear model.
    """
    if not text:
        return 0
    ascii_count = sum(1 for c in text if ord(c) < 128)
    non_ascii_count = len(text) - ascii_count
    return max(1, round(ascii_count / 5 + non_ascii_count / 2))


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


# ── 7-feature extraction ───────────────────────────────────────────────────────

def extract_features7(text: str) -> dict[str, int]:
    """
    Return a dict with seven features.

    Same as extract_features() but the 'digit' feature is split into:
      digit_iso: digit chars where BOTH adjacent chars are non-digit (or boundary)
      digit_run: digit chars that are part of a run of 2+ consecutive digits

    The C extension gives only 6 features; extract_features7 applies the
    digit split on top of extract_features() output using a simple scan.
    """
    # Get base 6-feature dict (uses C ext if available)
    base = extract_features(text)

    # Count digit_iso via a simple character scan
    digit_iso = 0
    n = len(text)
    for i, ch in enumerate(text):
        if ch.isdigit():
            prev_is_digit = (i > 0 and text[i - 1].isdigit())
            next_is_digit = (i + 1 < n and text[i + 1].isdigit())
            if not prev_is_digit and not next_is_digit:
                digit_iso += 1

    digit_run = base["digit"] - digit_iso
    return {
        "cjk":       base["cjk"],
        "letter":    base["letter"],
        "digit_iso": digit_iso,
        "digit_run": digit_run,
        "punct":     base["punct"],
        "space":     base["space"],
        "word":      base["word"],
    }


def estimate7(text: str, coeffs: Coeffs7 = None) -> int:
    """
    Estimate token count using the 7-feature model.

    If coeffs is None, falls back to DEFAULT_COEFFS7.
    """
    if not text:
        return 0
    if coeffs is None:
        coeffs = DEFAULT_COEFFS7
    f = extract_features7(text)
    total = (
        f["cjk"]       * coeffs.cjk       +
        f["letter"]    * coeffs.letter     +
        f["digit_iso"] * coeffs.digit_iso  +
        f["digit_run"] * coeffs.digit_run  +
        f["punct"]     * coeffs.punct      +
        f["space"]     * coeffs.space      +
        f["word"]      * coeffs.word
    )
    return max(1, round(total))
