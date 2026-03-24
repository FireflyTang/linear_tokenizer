"""Evaluate DEFAULT_COEFFS and UPPER_COEFFS per-category on real dataset."""
import csv
from collections import defaultdict
from data_fetch import fetch_all
from tokenizer_approx import estimate, UPPER_COEFFS

samples = fetch_all()
rows = list(csv.DictReader(open("results_new.csv", encoding="utf-8")))
assert len(samples) == len(rows)

MODEL_KEYS = [("glm", "GLM-5"), ("dsv", "DSV-V3.2"), ("kimi", "Kimi-K2.5"), ("mmax", "MiniMax-M2.5")]

# DEFAULT_COEFFS stats already in CSV (glm_pct = signed error %)
default_stats = defaultdict(lambda: defaultdict(list))
upper_stats   = defaultdict(lambda: defaultdict(list))
default_all   = defaultdict(list)
upper_all     = defaultdict(list)

for s, row in zip(samples, rows):
    cat  = s["category"]
    text = s["text"]
    upper_est = estimate(text, coeffs=UPPER_COEFFS)
    for key, _ in MODEL_KEYS:
        pct   = float(row[f"{key}_pct"])   # signed error %: (approx-real)/real*100
        real  = int(row[f"{key}_real"])
        ratio = upper_est / real if real > 0 else 1.0

        default_stats[cat][key].append(pct)
        default_all[key].append(pct)
        upper_stats[cat][key].append(ratio)
        upper_all[key].append(ratio)

def default_fmt(pcts):
    mape = sum(abs(p) for p in pcts) / len(pcts)
    bias = sum(pcts) / len(pcts)
    sign = "+" if bias >= 0 else ""
    return f"{mape:.1f}%", f"{sign}{bias:.1f}%"

def upper_fmt(ratios):
    guar = sum(1 for r in ratios if r >= 1.0) / len(ratios) * 100
    mean = (sum(r - 1 for r in ratios) / len(ratios)) * 100
    mx   = (max(ratios) - 1) * 100
    return f"{guar:.1f}%", f"+{mean:.1f}%", f"+{mx:.1f}%"

cats = sorted(default_stats.keys())

# ── DEFAULT_COEFFS table ──────────────────────────────────────────────────────
print("=== DEFAULT_COEFFS ===")
hdr = f"| {'类别':<14} | {'样本':>4}"
for _, mname in MODEL_KEYS:
    hdr += f" | {mname} MAPE | {mname} 偏差"
hdr += " |"
sep = "|" + "|".join(["-"*16, "-"*6] + ["-"*14, "-"*14]*4) + "|"
print(hdr)
print(sep)

for cat in cats:
    n = len(default_stats[cat]["glm"])
    line = f"| {cat:<14} | {n:>4}"
    for key, _ in MODEL_KEYS:
        mape, bias = default_fmt(default_stats[cat][key])
        line += f" | {mape:>10} | {bias:>10}"
    line += " |"
    print(line)

n_all = len(default_all["glm"])
line = f"| **{'OVERALL':<12}** | **{n_all:>4}**"
for key, _ in MODEL_KEYS:
    mape, bias = default_fmt(default_all[key])
    line += f" | **{mape:>8}** | **{bias:>8}**"
line += " |"
print(line)

# ── UPPER_COEFFS table ────────────────────────────────────────────────────────
print()
print("=== UPPER_COEFFS ===")
hdr2 = f"| {'类别':<14} | {'样本':>4}"
for _, mname in MODEL_KEYS:
    hdr2 += f" | {mname} 保证率 | 均高估 | 最大高估"
hdr2 += " |"
sep2 = "|" + "|".join(["-"*16, "-"*6] + ["-"*14, "-"*8, "-"*10]*4) + "|"
print(hdr2)
print(sep2)

for cat in cats:
    n = len(upper_stats[cat]["glm"])
    line = f"| {cat:<14} | {n:>4}"
    for key, _ in MODEL_KEYS:
        g, m, mx = upper_fmt(upper_stats[cat][key])
        line += f" | {g:>12} | {m:>6} | {mx:>8}"
    line += " |"
    print(line)

n_all = len(upper_all["glm"])
line = f"| **{'OVERALL':<12}** | **{n_all:>4}**"
for key, _ in MODEL_KEYS:
    g, m, mx = upper_fmt(upper_all[key])
    line += f" | **{g:>10}** | **{m:>4}** | **{mx:>6}**"
line += " |"
print(line)
