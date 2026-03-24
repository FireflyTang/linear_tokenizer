"""
Fit and evaluate the 7-feature model (digit_iso / digit_run split).

Usage:
    python _fit7.py 2>/dev/null

Reads:
  - real data via data_fetch.fetch_all() (uses cache, no network if already cached)
  - pre-computed token counts from results.csv  (avoids calling tokenizers again)

Outputs:
  - Fitted UPPER_COEFFS7 (LP) and DEFAULT_COEFFS7 (NNLS)
  - Side-by-side comparison table: 6-feature vs 7-feature
"""

import csv
import sys

import numpy as np

from data_fetch import fetch_all
from tokenizer_approx import (
    estimate, extract_features,
    UPPER_COEFFS, DEFAULT_COEFFS,
    Coeffs7, extract_features7, estimate7,
)
from benchmark import fit_upper_bound7, fit_shared7, ALL_MODELS, MODEL_LABELS

CSV_PATH = "results.csv"


# ── Load CSV rows (pre-computed token counts) ─────────────────────────────────

def load_csv(path: str) -> dict[str, dict]:
    """Return {source: row_dict} indexed by the 'source' column."""
    result = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result[row["source"]] = row
    return result


def mock_counter(model_key: str, csv_rows: dict[str, dict]):
    """
    Return a callable counter(text) -> int that looks up the pre-computed
    real token count from csv_rows by matching the sample's 'source' field.

    The returned callable is used with a cache keyed on id(text); callers
    pass text, so we need a way to map text -> source.  Instead, we expose
    a second helper make_counters() below that wires everything up by index.
    """
    col = f"{model_key}_real"
    def _counter(source: str) -> int:
        row = csv_rows.get(source)
        if row is None:
            return 0
        try:
            return int(row[col])
        except (KeyError, ValueError):
            return 0
    return _counter


# ── Evaluation helpers ────────────────────────────────────────────────────────

def eval_6feature(samples, csv_rows, coeffs, models):
    """Evaluate 6-feature estimate() against pre-computed counts."""
    all_pcts = []
    violations = 0
    per_cat = {}  # cat -> list of pct
    for s in samples:
        row = csv_rows.get(s["source"])
        if row is None:
            continue
        app = estimate(s["text"], coeffs=coeffs)
        for m in models:
            real = int(row.get(f"{m}_real", 0) or 0)
            if real <= 0:
                continue
            pct = (app - real) / real * 100
            all_pcts.append(pct)
            cat = s["category"]
            per_cat.setdefault(cat, []).append(pct)
            if app < real:
                violations += 1
    return all_pcts, violations, per_cat


def eval_7feature(samples, csv_rows, coeffs7, models):
    """Evaluate 7-feature estimate7() against pre-computed counts."""
    all_pcts = []
    violations = 0
    per_cat = {}
    for s in samples:
        row = csv_rows.get(s["source"])
        if row is None:
            continue
        app = estimate7(s["text"], coeffs=coeffs7)
        for m in models:
            real = int(row.get(f"{m}_real", 0) or 0)
            if real <= 0:
                continue
            pct = (app - real) / real * 100
            all_pcts.append(pct)
            cat = s["category"]
            per_cat.setdefault(cat, []).append(pct)
            if app < real:
                violations += 1
    return all_pcts, violations, per_cat


def summary_stats(pcts, violations):
    n = len(pcts)
    if n == 0:
        return dict(n=0, guarantee=0, mean_over=0, max_over=0, mape=0)
    guarantee = (n - violations) / n * 100
    over_pcts  = [p for p in pcts if p > 0]
    mean_over  = sum(over_pcts) / n if over_pcts else 0.0
    max_over   = max(pcts)
    mape       = sum(abs(p) for p in pcts) / n
    return dict(n=n, guarantee=guarantee, mean_over=mean_over,
                max_over=max_over, mape=mape)


# ── Build design matrices from (samples, csv_rows) ───────────────────────────

def build_matrices_6(samples, csv_rows, models):
    """Return stacked (A, b) for NNLS/LP — 6 features."""
    A_all, b_all = [], []
    for m in models:
        col = f"{m}_real"
        for s in samples:
            row = csv_rows.get(s["source"])
            if row is None:
                continue
            real = int(row.get(col, 0) or 0)
            if real <= 0:
                continue
            f = extract_features(s["text"])
            A_all.append([f["cjk"], f["letter"], f["digit"],
                           f["punct"], f["space"], f["word"]])
            b_all.append(float(real))
    return np.array(A_all, dtype=float), np.array(b_all, dtype=float)


def build_matrices_7(samples, csv_rows, models):
    """Return stacked (A, b) for NNLS/LP — 7 features."""
    A_all, b_all = [], []
    for m in models:
        col = f"{m}_real"
        for s in samples:
            row = csv_rows.get(s["source"])
            if row is None:
                continue
            real = int(row.get(col, 0) or 0)
            if real <= 0:
                continue
            f = extract_features7(s["text"])
            A_all.append([f["cjk"], f["letter"], f["digit_iso"], f["digit_run"],
                           f["punct"], f["space"], f["word"]])
            b_all.append(float(real))
    return np.array(A_all, dtype=float), np.array(b_all, dtype=float)


def fit_upper_7_from_matrices(samples, csv_rows, models):
    """LP fit for 7-feature upper bound, using CSV counts (no live tokenizers)."""
    from scipy.optimize import linprog
    A_rows, b_max = [], []
    for s in samples:
        row = csv_rows.get(s["source"])
        if row is None:
            continue
        reals = []
        for m in models:
            v = int(row.get(f"{m}_real", 0) or 0)
            if v > 0:
                reals.append(v)
        if not reals:
            continue
        real_max = max(reals)
        f = extract_features7(s["text"])
        A_rows.append([f["cjk"], f["letter"], f["digit_iso"], f["digit_run"],
                        f["punct"], f["space"], f["word"]])
        b_max.append(float(real_max))
    A = np.array(A_rows, dtype=float)
    b = np.array(b_max, dtype=float)
    feature_sums = A.sum(axis=0)
    res = linprog(
        c      = feature_sums,
        A_ub   = -A,
        b_ub   = -b,
        bounds = [(0, None)] * 7,
        method = "highs",
    )
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    coef = res.x
    return Coeffs7(cjk=coef[0], letter=coef[1], digit_iso=coef[2], digit_run=coef[3],
                   punct=coef[4], space=coef[5], word=coef[6])


def fit_upper_6_from_matrices(samples, csv_rows, models):
    """LP fit for 6-feature upper bound (verification vs hardcoded UPPER_COEFFS)."""
    from scipy.optimize import linprog
    A_rows, b_max = [], []
    for s in samples:
        row = csv_rows.get(s["source"])
        if row is None:
            continue
        reals = []
        for m in models:
            v = int(row.get(f"{m}_real", 0) or 0)
            if v > 0:
                reals.append(v)
        if not reals:
            continue
        real_max = max(reals)
        f = extract_features(s["text"])
        A_rows.append([f["cjk"], f["letter"], f["digit"],
                        f["punct"], f["space"], f["word"]])
        b_max.append(float(real_max))
    A = np.array(A_rows, dtype=float)
    b = np.array(b_max, dtype=float)
    feature_sums = A.sum(axis=0)
    res = linprog(
        c      = feature_sums,
        A_ub   = -A,
        b_ub   = -b,
        bounds = [(0, None)] * 6,
        method = "highs",
    )
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    coef = res.x
    from tokenizer_approx import Coeffs
    return Coeffs(cjk=coef[0], letter=coef[1], digit=coef[2],
                  punct=coef[3], space=coef[4], word=coef[5])


def fit_shared_6_from_matrices(samples, csv_rows, models):
    """NNLS fit for 6-feature shared coeffs."""
    from scipy.optimize import nnls
    from tokenizer_approx import Coeffs
    A, b = build_matrices_6(samples, csv_rows, models)
    coef, _ = nnls(A, b)
    return Coeffs(cjk=coef[0], letter=coef[1], digit=coef[2],
                  punct=coef[3], space=coef[4], word=coef[5])


def fit_shared_7_from_matrices(samples, csv_rows, models):
    """NNLS fit for 7-feature shared coeffs."""
    from scipy.optimize import nnls
    A, b = build_matrices_7(samples, csv_rows, models)
    coef, _ = nnls(A, b)
    return Coeffs7(cjk=coef[0], letter=coef[1], digit_iso=coef[2], digit_run=coef[3],
                   punct=coef[4], space=coef[5], word=coef[6])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("[_fit7] Loading samples from cache …", file=sys.stderr)
    samples = fetch_all()
    print(f"[_fit7] {len(samples)} samples loaded", file=sys.stderr)

    print("[_fit7] Loading pre-computed token counts from results.csv …", file=sys.stderr)
    csv_rows = load_csv(CSV_PATH)

    # Only keep samples that have a matching CSV row
    matched = [s for s in samples if s["source"] in csv_rows]
    print(f"[_fit7] {len(matched)} / {len(samples)} samples matched to CSV rows",
          file=sys.stderr)

    models = ALL_MODELS  # glm, dsv, kimi, mmax

    # ── Fit 6-feature models (re-fit from CSV for fair comparison) ────────────
    print("[_fit7] Fitting 6-feature models from CSV …", file=sys.stderr)
    upper6  = fit_upper_6_from_matrices(matched, csv_rows, models)
    shared6 = fit_shared_6_from_matrices(matched, csv_rows, models)

    # ── Fit 7-feature models ──────────────────────────────────────────────────
    print("[_fit7] Fitting 7-feature models from CSV …", file=sys.stderr)
    upper7  = fit_upper_7_from_matrices(matched, csv_rows, models)
    shared7 = fit_shared_7_from_matrices(matched, csv_rows, models)

    # ── Evaluate all four sets ────────────────────────────────────────────────
    print("[_fit7] Evaluating …", file=sys.stderr)

    pcts6u, viol6u, cat6u = eval_6feature(matched, csv_rows, upper6,  models)
    pcts6s, viol6s, cat6s = eval_6feature(matched, csv_rows, shared6, models)
    pcts7u, viol7u, cat7u = eval_7feature(matched, csv_rows, upper7,  models)
    pcts7s, viol7s, cat7s = eval_7feature(matched, csv_rows, shared7, models)

    stat6u = summary_stats(pcts6u, viol6u)
    stat6s = summary_stats(pcts6s, viol6s)
    stat7u = summary_stats(pcts7u, viol7u)
    stat7s = summary_stats(pcts7s, viol7s)

    # ── Print fitted coefficients ─────────────────────────────────────────────
    print()
    print("=" * 70)
    print("FITTED COEFFICIENTS")
    print("=" * 70)

    c = upper6
    print(f"\n6-feature UPPER_COEFFS (LP re-fit from CSV):")
    print(f"  Coeffs(cjk={c.cjk:.4f}, letter={c.letter:.4f}, digit={c.digit:.4f}, "
          f"punct={c.punct:.4f}, space={c.space:.4f}, word={c.word:.4f})")

    c = upper7
    print(f"\n7-feature UPPER_COEFFS7 (LP):")
    print(f"  Coeffs7(cjk={c.cjk:.4f}, letter={c.letter:.4f},")
    print(f"          digit_iso={c.digit_iso:.4f}, digit_run={c.digit_run:.4f},")
    print(f"          punct={c.punct:.4f}, space={c.space:.4f}, word={c.word:.4f})")

    c = shared6
    print(f"\n6-feature DEFAULT_COEFFS (NNLS re-fit from CSV):")
    print(f"  Coeffs(cjk={c.cjk:.4f}, letter={c.letter:.4f}, digit={c.digit:.4f}, "
          f"punct={c.punct:.4f}, space={c.space:.4f}, word={c.word:.4f})")

    c = shared7
    print(f"\n7-feature DEFAULT_COEFFS7 (NNLS):")
    print(f"  Coeffs7(cjk={c.cjk:.4f}, letter={c.letter:.4f},")
    print(f"          digit_iso={c.digit_iso:.4f}, digit_run={c.digit_run:.4f},")
    print(f"          punct={c.punct:.4f}, space={c.space:.4f}, word={c.word:.4f})")

    # ── Comparison table ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("OVERALL COMPARISON  (all models combined, all 1117 samples)")
    print("=" * 70)
    print(f"\n{'Model':<24}  {'Guarantee':>10}  {'MeanOver%':>10}  {'MaxOver%':>9}  {'MAPE':>7}")
    print("-" * 67)

    rows_data = [
        ("6-feat UPPER (LP)",    stat6u),
        ("7-feat UPPER7 (LP)",   stat7u),
        ("6-feat DEFAULT (NNLS)",stat6s),
        ("7-feat DEFAULT7 (NNLS)",stat7s),
    ]
    for label, st in rows_data:
        print(f"  {label:<22}  {st['guarantee']:>9.1f}%  "
              f"{st['mean_over']:>+9.1f}%  {st['max_over']:>+8.1f}%  "
              f"{st['mape']:>6.1f}%")

    # ── Per-category breakdown ────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("PER-CATEGORY  MaxOver%  (6-feat UPPER vs 7-feat UPPER7)")
    print("=" * 70)
    all_cats = sorted(set(cat6u.keys()) | set(cat7u.keys()))
    print(f"\n{'Category':<18}  {'6-feat MaxOver':>14}  {'7-feat MaxOver':>14}  {'Delta':>8}")
    print("-" * 62)
    for cat in all_cats:
        p6 = cat6u.get(cat, [0])
        p7 = cat7u.get(cat, [0])
        mo6 = max(p6) if p6 else 0
        mo7 = max(p7) if p7 else 0
        delta = mo7 - mo6
        sign = "+" if delta >= 0 else ""
        print(f"  {cat:<16}  {mo6:>+12.1f}%  {mo7:>+12.1f}%  {sign}{delta:.1f}%")

    # ── Paste-ready final coefficients ───────────────────────────────────────
    c7u = upper7
    c7s = shared7
    print()
    print("=" * 70)
    print("PASTE-READY — update tokenizer_approx.py UPPER_COEFFS7 / DEFAULT_COEFFS7")
    print("=" * 70)
    print(f"""
UPPER_COEFFS7 = Coeffs7(
    cjk       = {c7u.cjk:.4f},
    letter    = {c7u.letter:.4f},
    digit_iso = {c7u.digit_iso:.4f},
    digit_run = {c7u.digit_run:.4f},
    punct     = {c7u.punct:.4f},
    space     = {c7u.space:.4f},
    word      = {c7u.word:.4f},
)

DEFAULT_COEFFS7 = Coeffs7(
    cjk       = {c7s.cjk:.4f},
    letter    = {c7s.letter:.4f},
    digit_iso = {c7s.digit_iso:.4f},
    digit_run = {c7s.digit_run:.4f},
    punct     = {c7s.punct:.4f},
    space     = {c7s.space:.4f},
    word      = {c7s.word:.4f},
)
""")


if __name__ == "__main__":
    main()
