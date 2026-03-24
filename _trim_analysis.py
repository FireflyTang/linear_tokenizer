"""
Trim analysis: understand why UPPER_COEFFS has high max overestimate (+105%)
and evaluate trimmed LP fitting at various trim levels.

Background
----------
The UPPER_COEFFS LP minimises total predicted tokens subject to:
    f_i · c >= b_i  for ALL i    (guarantee constraint)
    c >= 0

The high max overestimate happens because a few outlier samples (e.g. short
Wikipedia section headers, code chunks heavy in digits) have very high
token-per-char ratios that push one or more coefficients (especially 'digit')
to a high value, which then over-estimates all other samples.

Correct "trim" approach
-----------------------
Removing the most-overestimated samples doesn't help (they're not binding
constraints — they're already well above the floor).

Instead, we RELAX the hardest binding constraints by allowing a small fraction
X% of samples to be violated.  This is done by solving:

    minimise   c · Σᵢ fᵢ + M · Σᵢ sᵢ    (add slack penalty)
    subject to fᵢ · c + sᵢ >= bᵢ  ∀i    (each constraint can be violated with penalty)
               c, s >= 0

  OR equivalently as "hard trim": solve standard LP but exclude the bottom X%
  of samples sorted by (b_i / ||f_i||) — i.e. the most "demanding" samples
  in terms of tokens-per-feature-unit.

We implement the LP-with-budget approach: allow at most floor(X% * N) violations,
choosing to violate the constraints that cause the most tightening.

Implementation: we use a two-pass approach:
  Pass 1: solve full LP, find which samples have binding constraints
          (slack = estimate - real ≈ 0)
  Pass 2: remove the K tightest binding constraints (where K = floor(X% * N))
          and re-solve LP — this is the true "trimmed LP"
"""

import csv
import sys
import numpy as np
from scipy.optimize import linprog

from data_fetch import fetch_all
from tokenizer_approx import extract_features, UPPER_COEFFS, Coeffs


# ── 1. Load CSV ───────────────────────────────────────────────────────────────

def load_csv(path="results.csv"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "source":    row["source"],
                "glm_real":  int(row["glm_real"]),
                "dsv_real":  int(row["dsv_real"]),
                "kimi_real": int(row["kimi_real"]),
                "mmax_real": int(row["mmax_real"]),
                "category":  row["category"],
            })
    return rows


# ── 2. Load texts and join ────────────────────────────────────────────────────

def load_and_join():
    csv_rows = load_csv()

    # fetch_all() uses cache, so fast after first run
    raw_samples = fetch_all()
    text_by_source = {s["source"]: s["text"] for s in raw_samples}

    joined = []
    missing = 0
    for row in csv_rows:
        src = row["source"]
        text = text_by_source.get(src)
        if text is None:
            missing += 1
            continue
        joined.append({**row, "text": text})

    if missing:
        print(f"[warn] {missing} CSV rows had no matching text (source key mismatch)",
              file=sys.stderr)
    return joined


# ── 3. Compute per-sample stats ───────────────────────────────────────────────

def compute_stats(samples):
    """Add features, upper_est, max_real, overestimate_pct, tightness to each sample."""
    c = UPPER_COEFFS
    result = []
    for s in samples:
        f = extract_features(s["text"])
        frow = [f["cjk"], f["letter"], f["digit"], f["punct"], f["space"], f["word"]]
        upper_est = (frow[0]*c.cjk + frow[1]*c.letter + frow[2]*c.digit +
                     frow[3]*c.punct + frow[4]*c.space + frow[5]*c.word)
        max_real = max(s["glm_real"], s["dsv_real"], s["kimi_real"], s["mmax_real"])
        over_pct = (upper_est - max_real) / max_real * 100 if max_real > 0 else 0.0
        tightness = max_real / upper_est if upper_est > 0 else 0.0  # close to 1.0 = binding
        result.append({
            **s,
            "features":  frow,
            "upper_est": upper_est,
            "max_real":  max_real,
            "over_pct":  over_pct,
            "tightness": tightness,
        })
    return result


# ── 4. Distribution analysis ──────────────────────────────────────────────────

def print_distribution(samples):
    pcts = [s["over_pct"] for s in samples]
    n = len(pcts)

    print("\n=== UPPER_COEFFS overestimate distribution ===")
    print(f"  N samples  : {n}")
    print(f"  Min        : {min(pcts):+.1f}%  (negative = underestimate = violation)")
    print(f"  P5         : {np.percentile(pcts,  5):+.1f}%")
    print(f"  P25        : {np.percentile(pcts, 25):+.1f}%")
    print(f"  Median     : {np.percentile(pcts, 50):+.1f}%")
    print(f"  P75        : {np.percentile(pcts, 75):+.1f}%")
    print(f"  P90        : {np.percentile(pcts, 90):+.1f}%")
    print(f"  P95        : {np.percentile(pcts, 95):+.1f}%")
    print(f"  P99        : {np.percentile(pcts, 99):+.1f}%")
    print(f"  Max        : {max(pcts):+.1f}%")
    print(f"  Mean       : {np.mean(pcts):+.1f}%")
    print(f"  Violations : {sum(1 for p in pcts if p < 0)} ({sum(1 for p in pcts if p < 0)/n*100:.1f}%)")

    print("\n  Bucket breakdown:")
    buckets = [
        ("> 80% overestimate",  lambda p: p > 80),
        ("> 60% overestimate",  lambda p: p > 60),
        ("> 40% overestimate",  lambda p: p > 40),
        ("> 20% overestimate",  lambda p: p > 20),
        ("  0–20% overestimate", lambda p: 0 <= p <= 20),
        ("  underestimate (<0)", lambda p: p < 0),
    ]
    for label, fn in buckets:
        cnt = sum(1 for p in pcts if fn(p))
        print(f"    {label:<28}: {cnt:4d}  ({cnt/n*100:5.1f}%)")

    # Top-10 worst overestimates
    worst = sorted(samples, key=lambda s: s["over_pct"], reverse=True)[:10]
    print("\n  Top-10 worst overestimates:")
    print(f"  {'#':<4} {'over%':>7} {'max_real':>9} {'upper_est':>10} {'category':<18} source")
    print("  " + "-" * 90)
    for i, s in enumerate(worst, 1):
        print(f"  {i:<4} {s['over_pct']:>+7.1f}% {s['max_real']:>9} {s['upper_est']:>10.1f}"
              f"  {s['category']:<18} {s['source'][:50]}")

    # Feature profile of high-overestimate outliers vs normal samples
    high   = [s for s in samples if s["over_pct"] > 60]
    normal = [s for s in samples if 0 <= s["over_pct"] <= 40]
    if high:
        print(f"\n  Feature profile (per token) — high overestimate >60% vs normal 0–40%:")
        feat_names = ["cjk", "letter", "digit", "punct", "space", "word"]
        print(f"  {'Feature':<8} {'high (n=%d)' % len(high):>14} {'normal (n=%d)' % len(normal):>16}  ratio")
        for i, fn in enumerate(feat_names):
            h_mean = np.mean([s["features"][i] / max(s["max_real"], 1) for s in high])
            n_mean = np.mean([s["features"][i] / max(s["max_real"], 1) for s in normal])
            ratio  = h_mean / n_mean if n_mean > 0 else float("inf")
            print(f"  {fn:<8}  {h_mean:>12.3f}   {n_mean:>13.3f}  {ratio:>6.2f}x")

    print("\n  Root cause: high-overestimate samples have very high digit chars/token ratio")
    print("  (e.g. short sections with references [1][2][3]..., or code with many numbers).")
    print("  The LP must set digit coeff = 1.87 to cover these, which over-estimates")
    print("  all samples containing digits — the spill-over hits long code files.")

    # Binding constraint analysis
    tight = sorted(samples, key=lambda s: s["tightness"], reverse=True)[:5]
    print("\n  Most binding constraints (tightest: tightness = max_real/upper_est ≈ 1.0):")
    print(f"  {'tightness':>10} {'over_pct':>9} {'max_real':>9} {'category':<18} source")
    for s in tight:
        print(f"  {s['tightness']:>10.5f}  {s['over_pct']:>+8.2f}%  {s['max_real']:>8}"
              f"  {s['category']:<18}  {s['source'][:50]}")
    print("  These are the samples that FORCE the LP coefficients upward.")


# ── 5. Trimmed LP fitting ─────────────────────────────────────────────────────

def fit_lp_direct(samples_subset):
    """
    Fit LP directly using precomputed features and max_real values.
    Doesn't call extract_features or tokenizers again.
    Equivalent to fit_upper_bound but uses cached feature rows.
    """
    A_rows, b = [], []
    for s in samples_subset:
        if s["max_real"] <= 0:
            continue
        A_rows.append(s["features"])
        b.append(float(s["max_real"]))

    A = np.array(A_rows, dtype=float)
    b = np.array(b, dtype=float)
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
    return Coeffs(cjk=coef[0], letter=coef[1], digit=coef[2],
                  punct=coef[3], space=coef[4], word=coef[5])


def trim_fit_lp(samples, trim_pct):
    """
    True trimmed LP: remove the K tightest binding constraints
    (samples where current UPPER_COEFFS provides least slack).

    These are the samples that *force* the LP into high-coefficient territory.
    K = floor(trim_pct/100 * N).
    """
    n_trim = int(len(samples) * trim_pct / 100)
    if n_trim == 0:
        return fit_lp_direct(samples)

    # Sort by tightness descending (tightness=max_real/upper_est, closest to 1.0 = most binding)
    sorted_by_tightness = sorted(samples, key=lambda s: s["tightness"], reverse=True)

    # Remove the n_trim most binding constraints
    trimmed = sorted_by_tightness[n_trim:]

    return fit_lp_direct(trimmed)


def eval_coeffs_on_all(coeffs, samples):
    """Return (guarantee_rate, mean_over_pct, max_over_pct, mape)."""
    c = coeffs
    pcts = []
    for s in samples:
        f = s["features"]
        est = (f[0]*c.cjk + f[1]*c.letter + f[2]*c.digit +
               f[3]*c.punct + f[4]*c.space + f[5]*c.word)
        max_real = s["max_real"]
        if max_real <= 0:
            continue
        pct = (est - max_real) / max_real * 100
        pcts.append(pct)
    n = len(pcts)
    violations = sum(1 for p in pcts if p < 0)
    guarantee = (n - violations) / n * 100
    pos_pcts = [p for p in pcts if p > 0]
    mean_over = np.mean(pos_pcts) if pos_pcts else 0.0
    max_over  = max(pcts)
    mape      = np.mean([abs(p) for p in pcts])
    return guarantee, mean_over, max_over, mape


def print_tradeoff_table(samples, trim_levels):
    print("\n=== Trimmed LP tradeoff table ===")
    print("  Method: remove top X% most-BINDING constraints (tightest max_real/estimate)")
    print("  then refit LP — evaluate on ALL samples including the trimmed ones.")
    print()
    print(f"  {'Trim%':>6}  {'N_fit':>6}  {'Guarantee':>10}  "
          f"{'MeanOver%':>10}  {'MaxOver%':>9}  {'MAPE':>7}  "
          f"{'cjk':>7}  {'letter':>7}  {'digit':>7}  {'punct':>7}  {'space':>7}  {'word':>7}")
    print("  " + "-" * 120)

    results = []
    for trim_pct in trim_levels:
        n_trim = int(len(samples) * trim_pct / 100)
        n_fit  = len(samples) - n_trim

        coeffs = trim_fit_lp(samples, trim_pct)
        guarantee, mean_over, max_over, mape = eval_coeffs_on_all(coeffs, samples)
        c = coeffs

        marker = " <-- UPPER_COEFFS" if trim_pct == 0.0 else ""
        print(f"  {trim_pct:>5.1f}%  {n_fit:>6}  {guarantee:>9.2f}%  "
              f"  {mean_over:>+8.1f}%  {max_over:>+8.1f}%  {mape:>6.1f}%  "
              f"{c.cjk:>7.4f}  {c.letter:>7.4f}  {c.digit:>7.4f}  "
              f"{c.punct:>7.4f}  {c.space:>7.4f}  {c.word:>7.4f}{marker}")
        results.append((trim_pct, coeffs, guarantee, mean_over, max_over, mape))

    # Commentary
    print()
    print("  Interpretation:")
    print("  - Trim 0%  = original UPPER_COEFFS: 100% guarantee on training data,")
    print("               but max overestimate ~99% due to pathological digit-heavy outliers.")
    print("  - Trim 0.5-1%: removes the binding constraints from outliers, allowing")
    print("               tighter coefficients. Reduces max overestimate substantially")
    print("               at cost of a few guaranteed violations.")
    print("  - Trim 2-5%: further reduction. Guarantee drops, but mean overestimate")
    print("               can drop from 24% to a lower value for typical inputs.")
    print()
    print("  The digit coefficient dominates: high-digit samples (e.g. short reference")
    print("  lists '[1][2][3]') force digit_coeff=1.87 to handle 1 token per digit.")
    print("  With trim, the digit coeff relaxes, reducing overestimate on code/prose.")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...", flush=True)
    samples = load_and_join()
    print(f"Joined {len(samples)} samples with text.", flush=True)

    samples = compute_stats(samples)

    print_distribution(samples)

    trim_levels = [0.0, 0.5, 1.0, 2.0, 5.0]
    results = print_tradeoff_table(samples, trim_levels)

    print("\nDone.")


if __name__ == "__main__":
    main()
