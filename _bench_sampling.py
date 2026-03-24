"""
Benchmark: line-sampling estimator vs linear approx vs exact tokenizer.

Accuracy test:
  Load 1117 real samples from cache, match to results.csv exact counts.
  For each sampling fraction (5%/10%/20%/50%), run sampled tokenizer on
  each text and compare to the exact count.  Repeat with N_SEEDS random
  seeds and report mean +/- std of MAPE.

Speed test:
  Time the actual tokenizer (kimi) at full and sampled fractions on a
  concatenated long text (~160K tokens).

Usage:
    python _bench_sampling.py              # accuracy only (kimi)
    python _bench_sampling.py --speed      # add speed measurement
    python _bench_sampling.py --model glm  # use a different model
    python _bench_sampling.py --seeds 10   # more seeds (default 5)
"""

import argparse
import csv
import random
import sys
import time

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="kimi", choices=["glm", "dsv", "kimi", "mmax"])
parser.add_argument("--seeds",  type=int, default=5)
parser.add_argument("--speed",  action="store_true")
args = parser.parse_args()

FRACTIONS = [0.05, 0.10, 0.20, 0.50, 1.00]
N_SEEDS   = args.seeds

# ── Load exact counts from results.csv ───────────────────────────────────────
print(f"[sampling] Loading results.csv ...")
exact_by_idx: dict[int, int] = {}
with open("results.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        idx = int(row["#"])
        exact_by_idx[idx] = int(row[f"{args.model}_real"])
print(f"[sampling] {len(exact_by_idx)} exact counts loaded ({args.model})")

# ── Load texts from cache ─────────────────────────────────────────────────────
print(f"[sampling] Loading sample texts from cache ...")
from data_fetch import fetch_all
samples = fetch_all()          # reads from .sample_cache/, no network
assert len(samples) == len(exact_by_idx), (
    f"Sample count mismatch: cache={len(samples)}, csv={len(exact_by_idx)}"
)
# results.csv rows are 1-indexed in the same order as fetch_all()
texts    = [s["text"]     for s in samples]
cats     = [s["category"] for s in samples]
exact    = [exact_by_idx[i+1] for i in range(len(samples))]
print(f"[sampling] {len(texts)} texts loaded, avg {sum(len(t) for t in texts)//len(texts)} chars")

# ── Load tokenizer ────────────────────────────────────────────────────────────
print(f"[sampling] Loading tokenizer ({args.model}) ...")
from real_tokenizers import glm_count, dsv_count, kimi_count, mmax_count
_fn_map = {"glm": glm_count, "dsv": dsv_count, "kimi": kimi_count, "mmax": mmax_count}
tokenizer_fn = _fn_map[args.model]
# warm-up
tokenizer_fn("warm up")
print(f"[sampling] Tokenizer ready.")


# ── Sampling estimator ────────────────────────────────────────────────────────
import io, contextlib

@contextlib.contextmanager
def _quiet():
    """Suppress stdout/stderr noise from verbose tokenizer wrappers."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def estimate_by_sampling(text: str, frac: float, seed: int) -> int:
    """
    Randomly sample `frac` fraction of non-empty lines, run exact tokenizer,
    scale by 1/actual_frac.  Returns rounded token count.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    n = len(lines)
    if n == 0:
        return tokenizer_fn(text)
    k = max(1, round(n * frac))
    if k >= n:
        with _quiet():
            return tokenizer_fn(text)      # fraction >= 1, full tokenization
    rng = random.Random(seed)
    sampled = rng.sample(lines, k)
    with _quiet():
        sample_tokens = tokenizer_fn("\n".join(sampled))
    return round(sample_tokens / (k / n))  # scale by actual fraction


# ── Accuracy test ─────────────────────────────────────────────────────────────
print()
print("=" * 72)
print(f"ACCURACY  (model={args.model.upper()}, N={len(texts)}, seeds={N_SEEDS})")
print("=" * 72)

# collect per-category set names
all_cats = sorted(set(cats))

# For each fraction, run N_SEEDS passes and aggregate
results: dict[float, dict] = {}

for frac in FRACTIONS:
    label = "100% (exact)" if frac >= 1.0 else f"{frac*100:.0f}%"
    print(f"\n  Fraction {label} ...")

    # accumulate across seeds
    ape_all: list[list[float]] = [[] for _ in range(len(texts))]  # per-sample APE list

    for seed in range(N_SEEDS):
        for i, (text, real) in enumerate(zip(texts, exact)):
            est = estimate_by_sampling(text, frac, seed=seed)
            ape = abs(est - real) / real * 100
            ape_all[i].append(ape)

    # mean APE per sample across seeds, then aggregate
    mean_ape_per_sample = [sum(v)/len(v) for v in ape_all]

    # signed error (for bias)
    bias_all: list[list[float]] = [[] for _ in range(len(texts))]
    for seed in range(N_SEEDS):
        for i, (text, real) in enumerate(zip(texts, exact)):
            est = estimate_by_sampling(text, frac, seed=seed)
            bias_all[i].append((est - real) / real * 100)
    mean_bias_per_sample = [sum(v)/len(v) for v in bias_all]

    # guarantee: fraction of samples where mean estimate >= real
    guarantee_samples = 0
    max_under_samples = []
    max_over_samples  = []
    for i in range(len(texts)):
        b = mean_bias_per_sample[i]
        if b >= 0:
            guarantee_samples += 1
            max_over_samples.append(b)
        else:
            max_under_samples.append(b)

    results[frac] = {
        "mape":        sum(mean_ape_per_sample) / len(mean_ape_per_sample),
        "bias":        sum(mean_bias_per_sample) / len(mean_bias_per_sample),
        "max_over":    max(max_over_samples)  if max_over_samples  else 0.0,
        "max_under":   min(max_under_samples) if max_under_samples else 0.0,
        "guarantee":   guarantee_samples / len(texts) * 100,
        "per_cat":     {},
    }
    # per-category
    for cat in all_cats:
        idxs = [i for i, c in enumerate(cats) if c == cat]
        cat_ape  = [mean_ape_per_sample[i]  for i in idxs]
        cat_bias = [mean_bias_per_sample[i] for i in idxs]
        results[frac]["per_cat"][cat] = {
            "n":    len(idxs),
            "mape": sum(cat_ape)/len(cat_ape),
            "bias": sum(cat_bias)/len(cat_bias),
            "max":  max(cat_bias),
            "min":  min(cat_bias),
        }

# ── Print summary table ───────────────────────────────────────────────────────
print()
print(f"{'Fraction':<14}  {'MAPE':>7}  {'Bias':>7}  {'Guarantee':>10}  {'MaxOver':>8}  {'MaxUnder':>9}")
print("-" * 62)
for frac in FRACTIONS:
    r = results[frac]
    label = "100%(exact)" if frac >= 1.0 else f"{frac*100:.0f}%"
    print(
        f"  {label:<12}  {r['mape']:>6.1f}%  {r['bias']:>+6.1f}%"
        f"  {r['guarantee']:>9.1f}%  {r['max_over']:>+7.1f}%  {r['max_under']:>+8.1f}%"
    )

# ── Per-category for the most interesting fractions ──────────────────────────
for frac in [0.10, 0.20, 1.00]:
    label = "100% (exact)" if frac >= 1.0 else f"{frac*100:.0f}% sample"
    print()
    print(f"  Per-category — {label}:")
    print(f"  {'Category':<16}  {'N':>5}  {'MAPE':>7}  {'Bias':>7}  {'Max+':>7}  {'Max-':>7}")
    print("  " + "-" * 56)
    for cat in all_cats:
        d = results[frac]["per_cat"][cat]
        print(
            f"  {cat:<16}  {d['n']:>5}  {d['mape']:>6.1f}%"
            f"  {d['bias']:>+6.1f}%  {d['max']:>+6.1f}%  {d['min']:>+6.1f}%"
        )

# ── Speed test ────────────────────────────────────────────────────────────────
if args.speed:
    print()
    print("=" * 72)
    print("SPEED  (single long text, ~160K tokens)")
    print("=" * 72)

    # Build a ~160K token text by concatenating samples until we exceed target
    TARGET_TOKENS = 160_000
    long_text_parts = []
    total_tok = 0
    for text, tok_count in zip(texts, exact):
        if total_tok >= TARGET_TOKENS:
            break
        long_text_parts.append(text)
        total_tok += tok_count
    long_text = "\n\n".join(long_text_parts)
    actual_chars = len(long_text)
    # exact token count
    t0 = time.perf_counter()
    with _quiet():
        full_count = tokenizer_fn(long_text)
    t_full = (time.perf_counter() - t0) * 1000
    print(f"\n  Full text: {actual_chars:,} chars, {full_count:,} tokens, {t_full:.1f}ms")
    print()
    print(f"  {'Fraction':<12}  {'Time':>8}  {'Speedup':>8}  {'Error':>8}  {'Estimate':>10}")
    print("  " + "-" * 52)
    # time each fraction (3 repeats, best of 3)
    for frac in [0.05, 0.10, 0.20, 0.50]:
        times = []
        ests  = []
        for seed in range(3):
            t0 = time.perf_counter()
            est = estimate_by_sampling(long_text, frac, seed=seed)
            times.append((time.perf_counter() - t0) * 1000)
            ests.append(est)
        t_avg = sum(times) / len(times)
        e_avg = sum(ests)  / len(ests)
        err   = (e_avg - full_count) / full_count * 100
        print(
            f"  {frac*100:.0f}% sample    {t_avg:>7.1f}ms  {t_full/t_avg:>7.1f}x"
            f"  {err:>+7.1f}%  {int(e_avg):>10,}"
        )
    print(f"  {'100%(exact)':<12}  {t_full:>7.1f}ms  {'1.0x':>8}  {'0.0%':>8}  {full_count:>10,}")

print()
print("Done.")
