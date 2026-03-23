"""
Benchmark: compare linear approximation vs real tokenizers.

Supported models: GLM-5, DeepSeek-V3.2, Kimi-K2.5, MiniMax-M2.5

Usage:
    # Synthetic short samples (default)
    python benchmark.py [--samples N] [--models glm,dsv,kimi,mmax] [--csv out.csv]

    # Real k-length data from Wikipedia + GitHub (downloads on first run)
    python benchmark.py --real-data [--refresh] [--chunk-size 4000]

    # Fit per-model coefficients
    python benchmark.py --real-data --fit

    # Fit ONE shared set of coefficients across all available models (recommended)
    python benchmark.py --real-data --fit-shared

Metrics per category and overall:
  MAE   – mean absolute error (tokens)
  MAPE  – mean absolute percentage error (%)
  bias  – mean signed error (positive = approx over-estimates)
"""

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np

from sample_gen import generate_samples, CATEGORIES as SYNTH_CATEGORIES
from tokenizer_approx import (
    estimate, extract_features, DEFAULT_COEFFS, Coeffs,
)

ALL_MODELS = ["glm", "dsv", "kimi", "mmax"]
MODEL_LABELS = {
    "glm":  "GLM-5   ",
    "dsv":  "DSV-V3.2",
    "kimi": "Kimi-K2 ",
    "mmax": "MiniMax ",
}


# ── Tokenizer loading ─────────────────────────────────────────────────────────

def _load_counter(name: str):
    try:
        from real_tokenizers import glm_count, dsv_count, kimi_count, mmax_count
        return {"glm": glm_count, "dsv": dsv_count,
                "kimi": kimi_count, "mmax": mmax_count}[name]
    except Exception as exc:
        print(f"[warn] {name} tokenizer unavailable: {exc}", file=sys.stderr)
        return None


# ── Statistics ────────────────────────────────────────────────────────────────

def _stats(errors, pcts):
    if not errors:
        return "    N/A      N/A      N/A"
    n    = len(errors)
    mae  = sum(abs(e) for e in errors) / n
    bias = sum(errors) / n
    mape = sum(abs(p) for p in pcts) / n
    return f"{mae:>7.2f}  {bias:>+7.2f}  {mape:>7.1f}%"


# ── Coefficient fitting ───────────────────────────────────────────────────────

def _build_matrix(samples, counter_fn):
    """Return (A, b) design matrix for one model."""
    A_rows, b_rows = [], []
    for s in samples:
        f = extract_features(s["text"])
        real = counter_fn(s["text"])
        if real <= 0:
            continue
        A_rows.append([f["cjk"], f["letter"], f["digit"], f["punct"], f["space"]])
        b_rows.append(real)
    return np.array(A_rows, dtype=float), np.array(b_rows, dtype=float)


def fit_per_model(samples, counters) -> dict[str, Coeffs]:
    """Fit separate Coeffs for each model via NNLS."""
    from scipy.optimize import nnls
    result = {}
    for name, fn in counters.items():
        A, b = _build_matrix(samples, fn)
        if len(A) == 0:
            continue
        coef, _ = nnls(A, b)
        result[name] = Coeffs(cjk=coef[0], letter=coef[1], digit=coef[2],
                              punct=coef[3], space=coef[4])
    return result


def fit_shared(samples, counters) -> Coeffs:
    """
    Fit ONE set of coefficients that minimises squared error simultaneously
    across ALL available models.

    Each (sample, model) pair contributes one row to the stacked design matrix:
        A_stacked · c ≈ b_stacked
    so the NNLS solution minimises ∑_models ∑_samples (c·features - real_tokens)².
    """
    from scipy.optimize import nnls
    A_all, b_all = [], []
    for fn in counters.values():
        A, b = _build_matrix(samples, fn)
        A_all.append(A)
        b_all.append(b)
    A_stacked = np.vstack(A_all)
    b_stacked  = np.concatenate(b_all)
    coef, residual = nnls(A_stacked, b_stacked)
    return Coeffs(cjk=coef[0], letter=coef[1], digit=coef[2],
                  punct=coef[3], space=coef[4])


def _eval_coeffs(samples, fn, coeffs: Coeffs) -> tuple[float, float]:
    """Return (MAE, MAPE) when using given coeffs for one model."""
    errs, pcts = [], []
    for s in samples:
        real = fn(s["text"])
        if real <= 0:
            continue
        app = estimate(s["text"], coeffs=coeffs)
        err = app - real
        errs.append(err)
        pcts.append(err / real * 100)
    if not errs:
        return 0.0, 0.0
    return (sum(abs(e) for e in errs) / len(errs),
            sum(abs(p) for p in pcts) / len(pcts))


def _print_coeffs_table(label: str, coeffs_map: dict[str, Coeffs],
                         samples, counters):
    """Pretty-print a table of fitted coefficients + their errors."""
    hdr = (f"  {'Model':<12}  {'cjk':>7}  {'letter':>7}  {'digit':>7}  "
           f"{'punct':>7}  {'space':>7}  {'MAE':>8}  {'MAPE':>7}")
    print(f"\n── {label} ──")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, c in coeffs_map.items():
        fn = counters[name]
        mae, mape = _eval_coeffs(samples, fn, c)
        print(f"  {MODEL_LABELS[name]:<12}  {c.cjk:>7.4f}  {c.letter:>7.4f}  "
              f"{c.digit:>7.4f}  {c.punct:>7.4f}  {c.space:>7.4f}  "
              f"{mae:>8.2f}  {mape:>6.1f}%")


def _print_shared_table(label: str, shared: Coeffs, samples, counters):
    """Print shared coefficients and per-model error under them."""
    c = shared
    print(f"\n── {label} ──")
    print(f"  Coeffs(cjk={c.cjk:.4f}, letter={c.letter:.4f}, digit={c.digit:.4f}, "
          f"punct={c.punct:.4f}, space={c.space:.4f})")
    hdr = f"  {'Model':<12}  {'MAE':>8}  {'MAPE':>7}  {'bias':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    all_errs, all_pcts = [], []
    for name, fn in counters.items():
        errs, pcts = [], []
        for s in samples:
            real = fn(s["text"])
            if real <= 0:
                continue
            app = estimate(s["text"], coeffs=shared)
            e = app - real
            errs.append(e)
            pcts.append(e / real * 100)
        if not errs:
            continue
        mae  = sum(abs(e) for e in errs) / len(errs)
        mape = sum(abs(p) for p in pcts) / len(pcts)
        bias = sum(errs) / len(errs)
        print(f"  {MODEL_LABELS[name]:<12}  {mae:>8.2f}  {mape:>6.1f}%  {bias:>+8.2f}")
        all_errs.extend(errs)
        all_pcts.extend(pcts)
    if all_errs:
        mae  = sum(abs(e) for e in all_errs) / len(all_errs)
        mape = sum(abs(p) for p in all_pcts) / len(all_pcts)
        bias = sum(all_errs) / len(all_errs)
        print(f"  {'ALL MODELS':<12}  {mae:>8.2f}  {mape:>6.1f}%  {bias:>+8.2f}")


# ── Main benchmark loop ───────────────────────────────────────────────────────

def run_benchmark(
    samples: list[dict],
    active_models: list[str] | None = None,
    csv_path: str | None = None,
    do_fit: bool = False,
    do_fit_shared: bool = False,
    display_categories: list[str] | None = None,
):
    if active_models is None:
        active_models = ALL_MODELS

    counters = {m: _load_counter(m) for m in active_models}
    counters = {m: fn for m, fn in counters.items() if fn is not None}

    if not counters:
        print("No tokenizers available.", file=sys.stderr)
        return

    # ── collect per-sample data ───────────────────────────────────────────────
    rows = []
    errs  = {m: defaultdict(list) for m in counters}
    pcts  = {m: defaultdict(list) for m in counters}
    cats_seen: set[str] = set()

    n = len(samples)
    for i, s in enumerate(samples, 1):
        text = s["text"]
        cat  = s["category"]
        cats_seen.add(cat)
        approx = estimate(text)
        source = s.get("source", "")
        row = {"#": i, "category": cat, "chars": len(text),
               "approx": approx, "source": source}

        parts = []
        for m, fn in counters.items():
            real = fn(text)
            err  = approx - real
            pct  = err / real * 100 if real else 0.0
            row[f"{m}_real"] = real
            row[f"{m}_err"]  = err
            row[f"{m}_pct"]  = round(pct, 2)
            errs[m][cat].append(err)
            pcts[m][cat].append(pct)
            errs[m]["OVERALL"].append(err)
            pcts[m]["OVERALL"].append(pct)
            parts.append(f"{MODEL_LABELS[m]}={real:>5}")
        rows.append(row)

        progress = "  ".join(parts)
        src_label = f"  [{source[:40]}]" if source else ""
        print(f"[{i:>4}/{n}] {cat:<14} chars={len(text):>5}  approx={approx:>5}  "
              f"{progress}{src_label}")

    # ── summary table ─────────────────────────────────────────────────────────
    col_w = 32
    header_models = "  ".join(f"{MODEL_LABELS[m]} MAE    bias    MAPE" for m in counters)
    sep_w = 16 + 2 + len(counters) * col_w
    print("\n" + "=" * sep_w)
    print(f"{'Category':<16}  {header_models}")
    print("-" * sep_w)

    if display_categories is None:
        display_categories = sorted(cats_seen)
    for cat in display_categories + ["OVERALL"]:
        if cat not in cats_seen and cat != "OVERALL":
            continue
        if cat == "OVERALL":
            print("=" * sep_w)
        cols = "  ".join(
            _stats(errs[m].get(cat, []), pcts[m].get(cat, []))
            for m in counters
        )
        print(f"{cat:<16}  {cols}")

    print("=" * sep_w)
    c = DEFAULT_COEFFS
    print(f"\nDefault coefficients: cjk={c.cjk}  letter={c.letter}  "
          f"digit={c.digit}  punct={c.punct}  space={c.space}")

    # ── fitting ───────────────────────────────────────────────────────────────
    needs_scipy = do_fit or do_fit_shared
    if needs_scipy:
        try:
            from scipy.optimize import nnls  # noqa
        except ImportError:
            print("\n[fit] scipy not installed. Run: pip install scipy")
            needs_scipy = do_fit = do_fit_shared = False

    if do_fit:
        per_model = fit_per_model(samples, counters)
        _print_coeffs_table("Fitted coefficients — per model", per_model,
                             samples, counters)

    if do_fit_shared:
        shared = fit_shared(samples, counters)
        _print_shared_table(
            "Fitted coefficients — ONE shared set across all models",
            shared, samples, counters,
        )
        c = shared
        print(f"\n  Paste into tokenizer_approx.py DEFAULT_COEFFS:\n"
              f"  DEFAULT_COEFFS = Coeffs(cjk={c.cjk:.4f}, letter={c.letter:.4f}, "
              f"digit={c.digit:.4f}, punct={c.punct:.4f}, space={c.space:.4f})")

    # ── CSV export ────────────────────────────────────────────────────────────
    if csv_path and rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nDetailed results written to: {csv_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token-count approximation benchmark")

    # Sample source
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--real-data", action="store_true",
                     help="Use real k-length texts from Wikipedia + GitHub (via data_fetch.py)")
    src.add_argument("--samples", type=int, default=20,
                     help="Samples per category for synthetic data (default: 20)")

    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download even if cache exists (with --real-data)")
    parser.add_argument("--chunk-size", type=int, default=4000,
                        help="Target chars per chunk when splitting real texts (default: 4000)")
    parser.add_argument("--models", type=str, default=",".join(ALL_MODELS),
                        help=f"Comma-separated models (default: {','.join(ALL_MODELS)})")
    parser.add_argument("--csv",  type=str, default=None,
                        help="Write per-sample CSV results to this path")
    parser.add_argument("--fit", action="store_true",
                        help="Fit optimal coefficients separately per model")
    parser.add_argument("--fit-shared", action="store_true",
                        help="Fit ONE shared coefficient set across all models (recommended)")

    args = parser.parse_args()

    if args.real_data:
        from data_fetch import fetch_all
        samples = fetch_all(force_refresh=args.refresh, chunk_size=args.chunk_size)
    else:
        samples = generate_samples(args.samples)

    run_benchmark(
        samples=samples,
        active_models=[m.strip() for m in args.models.split(",")],
        csv_path=args.csv,
        do_fit=args.fit,
        do_fit_shared=args.fit_shared,
    )
