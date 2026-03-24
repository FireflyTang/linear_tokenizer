"""Fit and evaluate per-model LP upper-bound coefficients."""
import csv
from collections import defaultdict
from data_fetch import fetch_all
from benchmark import fit_upper_bound
from tokenizer_approx import estimate, Coeffs
from real_tokenizers import glm_count, dsv_count, kimi_count, mmax_count

samples = fetch_all()

MODELS = [
    ("glm",  "GLM-5",        glm_count),
    ("dsv",  "DSV-V3.2",     dsv_count),
    ("kimi", "Kimi-K2.5",    kimi_count),
    ("mmax", "MiniMax-M2.5", mmax_count),
]

# Pre-warm tokenizers
for _, name, fn in MODELS:
    fn("warmup")
    print(f"{name} ready")

# Fit LP upper-bound per model
per_model_coeffs = {}
for key, name, fn in MODELS:
    print(f"\nFitting LP upper bound for {name}...")
    coeffs = fit_upper_bound(samples, {key: fn})
    per_model_coeffs[key] = coeffs
    c = coeffs
    print(f"  {name}: cjk={c.cjk:.4f} letter={c.letter:.4f} digit={c.digit:.4f} "
          f"punct={c.punct:.4f} space={c.space:.4f} word={c.word:.4f}")

# Evaluate per-category for each model using its own coefficients
print("\n\n=== Per-model UPPER_COEFFS (each model uses its own coefficients) ===")
print()

def upper_fmt(ratios):
    guar = sum(1 for r in ratios if r >= 1.0) / len(ratios) * 100
    mean = (sum(r - 1 for r in ratios) / len(ratios)) * 100
    mx   = (max(ratios) - 1) * 100
    return f"{guar:.1f}%", f"+{mean:.1f}%", f"+{mx:.1f}%"

# Build per-category, per-model stats
stats = defaultdict(lambda: defaultdict(list))
all_ratios = defaultdict(list)

for s in samples:
    cat  = s["cat"] if "cat" in s else s["category"]
    text = s["text"]
    for key, name, fn in MODELS:
        coeffs = per_model_coeffs[key]
        real = fn(text)
        if real <= 0:
            continue
        upper_est = estimate(text, coeffs=coeffs)
        ratio = upper_est / real
        stats[cat][key].append(ratio)
        all_ratios[key].append(ratio)

cats = sorted(stats.keys())

# Header
hdr = f"| {'类别':<14} | {'N':>4}"
for _, mname, __ in MODELS:
    short = mname.split("-")[0]
    hdr += f" | {mname} 保证率 | 均高估 | 最大高估"
hdr += " |"
print(hdr)
sep = "|" + "-"*16 + "|" + "-"*6 + ("|" + "-"*14 + "|" + "-"*8 + "|" + "-"*10)*4 + "|"
print(sep)

for cat in cats:
    n = len(stats[cat]["glm"])
    line = f"| {cat:<14} | {n:>4}"
    for key, _, __ in MODELS:
        g, m, mx = upper_fmt(stats[cat][key])
        line += f" | {g:>12} | {m:>6} | {mx:>8}"
    line += " |"
    print(line)

n_all = len(all_ratios["glm"])
line = f"| **{'OVERALL':<12}** | **{n_all}**"
for key, _, __ in MODELS:
    g, m, mx = upper_fmt(all_ratios[key])
    line += f" | **{g:>10}** | **{m:>4}** | **{mx:>6}**"
line += " |"
print(line)

print()
print("=== Coefficient summary ===")
print(f"{'模型':<15} {'cjk':>7} {'letter':>7} {'digit':>7} {'punct':>7} {'space':>7} {'word':>7}")
for key, name, _ in MODELS:
    c = per_model_coeffs[key]
    print(f"{name:<15} {c.cjk:>7.4f} {c.letter:>7.4f} {c.digit:>7.4f} {c.punct:>7.4f} {c.space:>7.4f} {c.word:>7.4f}")
