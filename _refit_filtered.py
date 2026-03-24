"""
Analysis: how to reduce max overestimate of UPPER_COEFFS.

Two strategies investigated:

  Strategy A — Remove high-overestimate outliers from LP training
    Result: LP coefficients do NOT change, because the outlier samples
    (Wikipedia citation paragraphs, specific library code) are NOT the
    binding constraints.  The LP is driven by other samples (dense math
    formulas, C++/Go code with many symbols); removing outliers has no
    effect on the solution.  Conclusion: filtering outliers alone is
    ineffective.

  Strategy B — Relax the LP lower bound (allow up to X% underestimate)
    Original LP:  estimate >= real           (0% tolerance)
    Relaxed LP:   estimate >= (1-tol)*real   (tol% tolerance)
    Result: allowing 10% underestimate dramatically reduces overestimate
    on the vast majority of inputs while only failing on rare edge cases.

    UPPER_COEFFS_RELAXED = tol=10% result:
      Guarantee (estimate >= real): 93.0%  (vs 99.9%)
      Mean overestimate:           +11.5%  (vs +23.9%)
      P90 overestimate:            +23.5%  (vs +37.2%)
      Max overestimate:            +78.8%  (vs +98.7%)
      Samples with >40% overestimate:  7   (vs 71)
      Max underestimate:           -10%

    The 7 remaining >40% outliers are all rare edge-case patterns:
      - Wikipedia citation-dense paragraphs ("[1][2][3]..." style)
      - Specific library code chunks (cpython dataclasses, Rust hashmap, pandas)

Usage:
    python _refit_filtered.py              # full analysis: Strategy A + B sweep
    python _refit_filtered.py --tol 10     # just refit with 10% tolerance
    python _refit_filtered.py --strategy a # only run Strategy A (filter outliers)
    python _refit_filtered.py --strategy b # only run Strategy B (relax LP)
"""

import argparse
import csv
from collections import Counter

import numpy as np
from scipy.optimize import linprog

from data_fetch import fetch_all
from tokenizer_approx import UPPER_COEFFS, extract_features, Coeffs

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--tol",      type=float, default=None,
                    help="Run Strategy B with this single tolerance value (0-100).")
parser.add_argument("--strategy", choices=["a", "b"], default=None,
                    help="Run only Strategy A or B. Default: run both.")
args = parser.parse_args()

# ── Load data ─────────────────────────────────────────────────────────────────
print("[refit] Loading samples from cache ...", flush=True)
samples = fetch_all()

print("[refit] Loading exact counts from results.csv ...", flush=True)
csv_rows: dict[int, dict] = {}
with open("results.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        csv_rows[int(row["#"])] = row

assert len(samples) == len(csv_rows), "Sample count mismatch between cache and CSV"

MODELS    = ["glm_real", "dsv_real", "kimi_real", "mmax_real"]
FEAT_NAMES = ["cjk", "letter", "digit", "punct", "space", "word"]

# ── Extract features ──────────────────────────────────────────────────────────
print("[refit] Extracting features ...", flush=True)
all_feats, all_reals, all_cats, all_srcs = [], [], [], []
for i, s in enumerate(samples):
    f    = extract_features(s["text"])
    fvec = np.array([f[k] for k in FEAT_NAMES], dtype=float)
    real = max(int(csv_rows[i + 1][m]) for m in MODELS)
    all_feats.append(fvec)
    all_reals.append(real)
    all_cats.append(s["category"])
    all_srcs.append(csv_rows[i + 1]["source"])

all_feats = np.array(all_feats)
all_reals = np.array(all_reals)


# ── Helpers ───────────────────────────────────────────────────────────────────
def fit_lp(feats: np.ndarray, reals: np.ndarray) -> np.ndarray:
    """LP: minimize sum estimates  s.t.  feats @ c >= reals,  c >= 0."""
    res = linprog(
        feats.sum(axis=0),
        A_ub=-feats, b_ub=-reals,
        bounds=[(0, None)] * feats.shape[1],
        method="highs",
    )
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    return res.x


def evaluate(c: np.ndarray, feats: np.ndarray, reals: np.ndarray,
             cats: list[str]) -> dict:
    over = (feats @ c / reals - 1.0) * 100.0
    per_cat = {}
    for cat in sorted(set(cats)):
        idx = [i for i, x in enumerate(cats) if x == cat]
        per_cat[cat] = {
            "n":       len(idx),
            "mean":    over[idx].mean(),
            "max":     over[idx].max(),
            "violate": int((over[idx] < 0).sum()),
        }
    viol = int((over < 0).sum())
    return {
        "guarantee": (1 - viol / len(reals)) * 100,
        "mean":      over.mean(),
        "p90":       np.percentile(over, 90),
        "p95":       np.percentile(over, 95),
        "max":       over.max(),
        "n_over40":  int((over > 40).sum()),
        "n_under":   viol,
        "max_under": float(-over[over < 0].min()) if viol else 0.0,
        "per_cat":   per_cat,
    }


def print_eval(ev: dict, label: str = ""):
    if label:
        print(f"  {label}")
    print(f"    Guarantee={ev['guarantee']:.1f}%  Mean={ev['mean']:+.1f}%"
          f"  P90={ev['p90']:+.1f}%  P95={ev['p95']:+.1f}%  Max={ev['max']:+.1f}%"
          f"  >40%={ev['n_over40']}  undercount={ev['n_under']}(max {ev['max_under']:.1f}%)")


def print_per_cat(ev: dict):
    print(f"  {'Category':<14}  {'N':>5}  {'Guarantee':>10}  {'Mean':>7}  {'Max':>7}")
    print("  " + "-" * 48)
    for cat, d in ev["per_cat"].items():
        guar = (1 - d["violate"] / d["n"]) * 100
        flag = "!!" if d["max"] > 40 else "OK"
        print(f"  {flag} {cat:<12}  {d['n']:>5}  {guar:>9.1f}%"
              f"  {d['mean']:>+6.1f}%  {d['max']:>+6.1f}%")


# ── Baseline ──────────────────────────────────────────────────────────────────
c_base = np.array([getattr(UPPER_COEFFS, k) for k in FEAT_NAMES])
ev_base = evaluate(c_base, all_feats, all_reals, all_cats)
current_over = (all_feats @ c_base / all_reals - 1.0) * 100.0

print()
print("=" * 72)
print("BASELINE  (current UPPER_COEFFS, all 1117 samples)")
print("=" * 72)
print_eval(ev_base)

# ── Strategy A: filter high-overestimate outliers ─────────────────────────────
run_a = args.strategy in (None, "a")
if run_a and args.tol is None:
    print()
    print("=" * 72)
    print("STRATEGY A — Remove high-overestimate outliers from training")
    print("  Hypothesis: outlier samples (e.g. [1][2][3] citation paragraphs)")
    print("  pull the LP coefficients upward.  Removing them should lower max.")
    print("=" * 72)

    thresholds = [40.0, 50.0, 60.0, 80.0]
    print()
    print(f"  {'Removed':>10}  {'Guarantee':>10}  {'Mean':>7}  {'Max':>7}  {'>40%':>5}  digit   punct  verdict")
    print("  " + "-" * 75)
    print(f"  {'(baseline)':>10}  {ev_base['guarantee']:>9.1f}%  {ev_base['mean']:>+6.1f}%"
          f"  {ev_base['max']:>+6.1f}%  {ev_base['n_over40']:>5}"
          f"  {c_base[2]:.3f}  {c_base[3]:.3f}")
    same_count = 0
    for thresh in thresholds:
        keep      = current_over <= thresh
        n_removed = int((~keep).sum())
        c_new     = fit_lp(all_feats[keep], all_reals[keep])
        ev_full   = evaluate(c_new, all_feats, all_reals, all_cats)
        same      = np.allclose(c_new, c_base, atol=1e-4)
        same_count += same
        verdict = "SAME coeffs (outliers not binding)" if same else "coeffs changed"
        cats_rm = dict(Counter(c for c, k in zip(all_cats, keep) if not k))
        print(f"  >={thresh:.0f}%: {n_removed:>3} rm  {ev_full['guarantee']:>9.1f}%"
              f"  {ev_full['mean']:>+6.1f}%  {ev_full['max']:>+6.1f}%  {ev_full['n_over40']:>5}"
              f"  {c_new[2]:.3f}  {c_new[3]:.3f}  {verdict}")
    print()
    print(f"  Finding: in all {len(thresholds)} threshold tests, LP coefficients {'UNCHANGED' if same_count == len(thresholds) else 'changed'}.")
    print("  The extreme-overestimate samples are NOT binding constraints.")
    print("  The binding constraints are dense code/math formula chunks that")
    print("  genuinely require high digit/punct coefficients to avoid undercount.")
    print("  Strategy A does not solve the problem.")

# ── Strategy B: relax LP lower bound ─────────────────────────────────────────
run_b = args.strategy in (None, "b")
if run_b:
    if args.tol is not None:
        tol_values = [args.tol / 100.0]
    else:
        tol_values = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]

    print()
    print("=" * 72)
    print("STRATEGY B — Relax LP lower bound: estimate >= (1-tol)*real")
    print("  Allows at most tol% underestimate per sample in exchange for")
    print("  lower overestimate on the majority of inputs.")
    print("=" * 72)
    print()
    print(f"  {'tol':>5}  {'Guarantee':>10}  {'Mean':>7}  {'P90':>7}  {'Max':>7}"
          f"  {'>40%':>5}  {'Undercount':>10}  digit   punct")
    print("  " + "-" * 85)

    best_c = None
    for tol in tol_values:
        c_new = fit_lp(all_feats, all_reals * (1.0 - tol))
        ev    = evaluate(c_new, all_feats, all_reals, all_cats)
        under_str = f"{ev['n_under']} (max {ev['max_under']:.1f}%)"
        print(f"  {tol*100:>4.0f}%  {ev['guarantee']:>9.1f}%  {ev['mean']:>+6.1f}%"
              f"  {ev['p90']:>+6.1f}%  {ev['max']:>+6.1f}%  {ev['n_over40']:>5}"
              f"  {under_str:>10}  {c_new[2]:.3f}  {c_new[3]:.3f}")
        if abs(tol - 0.10) < 1e-6 or (args.tol and abs(tol - args.tol / 100) < 1e-6):
            best_c  = c_new
            best_ev = ev
            best_tol = tol

    if best_c is not None:
        print()
        print(f"  SELECTED: tol={best_tol*100:.0f}%  — per-category breakdown:")
        print_per_cat(best_ev)

        print()
        best_over = (all_feats @ best_c / all_reals - 1.0) * 100.0
        bad_idx   = np.where(best_over > 40)[0]
        if len(bad_idx):
            print(f"  Remaining {len(bad_idx)} samples with >40% overestimate:")
            for i in bad_idx:
                print(f"    [{i+1}] {best_over[i]:+.1f}%  {all_cats[i]:<14}  {all_srcs[i][:60]}")

        print()
        print("  Paste-ready UPPER_COEFFS_RELAXED:")
        print(f"  UPPER_COEFFS_RELAXED = Coeffs(")
        for name, val in zip(FEAT_NAMES, best_c):
            print(f"      {name:<6} = {val:.4f},")
        print("  )")

# ── Summary ───────────────────────────────────────────────────────────────────
if args.tol is None and args.strategy is None:
    # Compute tol=10 result for summary
    c10 = fit_lp(all_feats, all_reals * 0.90)
    ev10 = evaluate(c10, all_feats, all_reals, all_cats)

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print()
    print(f"  {'Coeffs set':<30}  {'Guarantee':>10}  {'Mean':>7}  {'P90':>7}  {'Max':>7}  {'>40%':>5}")
    print("  " + "-" * 72)
    print(f"  {'UPPER_COEFFS (strict LP)':<30}  {ev_base['guarantee']:>9.1f}%"
          f"  {ev_base['mean']:>+6.1f}%  {ev_base['p90']:>+6.1f}%"
          f"  {ev_base['max']:>+6.1f}%  {ev_base['n_over40']:>5}")
    print(f"  {'UPPER_COEFFS_RELAXED (tol=10%)':<30}  {ev10['guarantee']:>9.1f}%"
          f"  {ev10['mean']:>+6.1f}%  {ev10['p90']:>+6.1f}%"
          f"  {ev10['max']:>+6.1f}%  {ev10['n_over40']:>5}")
    print()
    print("  Recommendation:")
    print("    - UPPER_COEFFS:         strict upper bound, 99.9% guarantee,")
    print("                            but +24% mean / +99% max overestimate.")
    print("    - UPPER_COEFFS_RELAXED: 93% guarantee, at most -10% undercount,")
    print("                            +11.5% mean / +24% P90 overestimate.")
    print("                            7 outlier patterns exceed 40% (rare in")
    print("                            business data).")
    print()
    print("  See tokenizer_approx.py for the fitted coefficient values.")

print()
print("[refit] Done.")
