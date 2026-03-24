"""Cross-model token count correlation analysis on real data (results.csv)."""
import csv
import math
from collections import defaultdict

rows = list(csv.DictReader(open("results.csv", encoding="utf-8")))

MODELS = ["glm", "dsv", "kimi", "mmax"]
MODEL_NAMES = {"glm": "GLM-5", "dsv": "DSV-V3.2", "kimi": "Kimi-K2.5", "mmax": "MiniMax-M2.5"}

# Build per-category and global ratio stats (ratio relative to kimi)
cat_data = defaultdict(lambda: {m: [] for m in MODELS})
for row in rows:
    cat = row["category"]
    for m in MODELS:
        cat_data[cat][m].append(int(row[f"{m}_real"]))

def ratio_stats(a_vals, b_vals):
    """Returns mean ratio, std, min, max of a/b."""
    ratios = [a/b for a, b in zip(a_vals, b_vals) if b > 0]
    mean = sum(ratios) / len(ratios)
    std  = math.sqrt(sum((r - mean)**2 for r in ratios) / len(ratios))
    return mean, std, min(ratios), max(ratios)

def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    dx  = math.sqrt(sum((x-mx)**2 for x in xs))
    dy  = math.sqrt(sum((y-my)**2 for y in ys))
    return num / (dx * dy) if dx * dy > 0 else 1.0

# Global lists
all_vals = {m: [] for m in MODELS}
for cat, vals in cat_data.items():
    for m in MODELS:
        all_vals[m].extend(vals[m])

print("=== Ratios relative to Kimi (real data, 1117 samples) ===")
print()
print(f"{'类别':<15} {'N':>4}  {'glm/kimi':>12} {'dsv/kimi':>12} {'mmax/kimi':>12}")
print("-" * 60)
for cat in sorted(cat_data.keys()):
    vals = cat_data[cat]
    n = len(vals["kimi"])
    gm, gs, gmn, gmx = ratio_stats(vals["glm"],  vals["kimi"])
    dm, ds, dmn, dmx = ratio_stats(vals["dsv"],  vals["kimi"])
    mm, ms, mmn, mmx = ratio_stats(vals["mmax"], vals["kimi"])
    print(f"{cat:<15} {n:>4}  {gm:.3f}+/-{gs:.3f}   {dm:.3f}+/-{ds:.3f}   {mm:.3f}+/-{ms:.3f}")

gm, gs, gmn, gmx = ratio_stats(all_vals["glm"],  all_vals["kimi"])
dm, ds, dmn, dmx = ratio_stats(all_vals["dsv"],  all_vals["kimi"])
mm, ms, mmn, mmx = ratio_stats(all_vals["mmax"], all_vals["kimi"])
print("-" * 60)
print(f"{'ALL':<15} {len(all_vals['kimi']):>4}  {gm:.3f}+/-{gs:.3f}   {dm:.3f}+/-{ds:.3f}   {mm:.3f}+/-{ms:.3f}")
print()
print(f"  glm range: {gmn:.3f}~{gmx:.3f}")
print(f"  dsv range: {dmn:.3f}~{dmx:.3f}")
print(f" mmax range: {mmn:.3f}~{mmx:.3f}")

print()
print("=== Pearson r (relative to Kimi) ===")
print(f"  glm  vs kimi: {pearson(all_vals['glm'],  all_vals['kimi']):.4f}")
print(f"  dsv  vs kimi: {pearson(all_vals['dsv'],  all_vals['kimi']):.4f}")
print(f"  mmax vs kimi: {pearson(all_vals['mmax'], all_vals['kimi']):.4f}")
print(f"  glm  vs dsv:  {pearson(all_vals['glm'],  all_vals['dsv']):.4f}")
print(f"  glm  vs mmax: {pearson(all_vals['glm'],  all_vals['mmax']):.4f}")
print(f"  dsv  vs mmax: {pearson(all_vals['dsv'],  all_vals['mmax']):.4f}")

print()
print("=== 用 Kimi 代替其他模型的误差（MAPE） ===")
for target in ["glm", "dsv", "mmax"]:
    pcts = [(k - t) / t * 100 for k, t in zip(all_vals["kimi"], all_vals[target]) if t > 0]
    mape = sum(abs(p) for p in pcts) / len(pcts)
    bias = sum(pcts) / len(pcts)
    mx   = max(pcts)
    mn   = min(pcts)
    sign = "+" if bias >= 0 else ""
    print(f"  kimi → {MODEL_NAMES[target]:<14}: MAPE={mape:.1f}%  bias={sign}{bias:.1f}%  "
          f"range=[{mn:.1f}%, +{mx:.1f}%]")
print()
print("  注：MAPE 表示「把 kimi token 数当作目标模型计数」的误差。")
print("  range 右端为 kimi 高估目标模型的最大幅度（潜在低估风险）。")
