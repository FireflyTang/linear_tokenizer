"""
fit_custom.py — 自定义数据集接入 + 系数重新拟合

将自定义文本文件切分为块，与内置数据集合并，重新拟合 UPPER_COEFFS 和 DEFAULT_COEFFS。

用法示例:
    # 只用自定义文件
    python fit_custom.py --text mydata.txt

    # 自定义 + 合成数据集（21 类，50 样本/类）
    python fit_custom.py --text mydata.txt --with-synthetic --samples 50

    # 自定义 + Wikipedia/GitHub 真实数据
    python fit_custom.py --text mydata.txt --with-real

    # 全部合并
    python fit_custom.py --text mydata.txt --with-synthetic --with-real

    # 多个文件
    python fit_custom.py --text file1.txt --text file2.txt --with-synthetic

    # 只用合成数据（不传 --text，验证流程用）
    python fit_custom.py --with-synthetic --samples 20

    # 指定多个模型（需要 HF 权重）
    python fit_custom.py --text mydata.txt --models kimi_fast,glm,dsv

输出:
    - 数据来源汇总（含 chunk_size）
    - UPPER_COEFFS（LP，100% 保证不低估）：每个模型整体指标 + 按数据来源细分
    - DEFAULT_COEFFS（NNLS，最小化误差）：每个模型整体指标 + 按数据来源细分
    - 可直接粘贴到 tokenizer_approx.py 的系数代码块
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path


# ── 切块 ────────────────────────────────────────────────────────────────────

def _chunk_text(text: str, size: int = 3000) -> list[str]:
    """按行边界切分文本为 ~size 字符的块（与 data_fetch._chunk 逻辑相同）。"""
    lines = text.splitlines(keepends=True)
    chunks, buf, n = [], [], 0
    for line in lines:
        buf.append(line)
        n += len(line)
        if n >= size:
            chunks.append("".join(buf))
            buf, n = [], 0
    if buf:
        chunks.append("".join(buf))
    return [c for c in chunks if len(c.strip()) > 50]


def load_text_files(paths: list[str], chunk_size: int) -> list[dict]:
    """读入自定义文本文件，切块，返回 sample 列表。"""
    samples = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"[错误] 文件不存在: {p}", file=sys.stderr)
            sys.exit(1)
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = _chunk_text(text, chunk_size)
        for i, chunk in enumerate(chunks):
            samples.append({
                "text":     chunk,
                "category": f"custom/{path.name}",
                "source":   f"{path.name}[{i}]",
                "_file":    path.name,
            })
        print(f"  {path.name}: {len(chunks)} 块  "
              f"(chunk_size={chunk_size}, 共 {len(text):,} 字符)")
    return samples


# ── 统计工具 ─────────────────────────────────────────────────────────────────

def _eval_upper(samples: list[dict], counter_fn, coeffs) -> dict:
    """计算 UPPER_COEFFS 在一组样本上的详细统计。"""
    from tokenizer_approx import estimate
    pcts, violations = [], 0
    for s in samples:
        real = counter_fn(s["text"])
        if real <= 0:
            continue
        app = estimate(s["text"], coeffs=coeffs)
        pct = (app - real) / real * 100
        pcts.append(pct)
        if app < real:
            violations += 1
    if not pcts:
        return {}
    n = len(pcts)
    return {
        "n":          n,
        "guarantee":  (n - violations) / n * 100,
        "violations": violations,
        "mean_over":  sum(p for p in pcts if p > 0) / n,
        "max_over":   max(pcts),
        "mape":       sum(abs(p) for p in pcts) / n,
    }


def _eval_default(samples: list[dict], counter_fn, coeffs) -> dict:
    """计算 DEFAULT_COEFFS 在一组样本上的详细统计。"""
    from tokenizer_approx import estimate
    pcts = []
    for s in samples:
        real = counter_fn(s["text"])
        if real <= 0:
            continue
        app = estimate(s["text"], coeffs=coeffs)
        pcts.append((app - real) / real * 100)
    if not pcts:
        return {}
    n = len(pcts)
    return {
        "n":       n,
        "mape":    sum(abs(p) for p in pcts) / n,
        "bias":    sum(pcts) / n,
        "max_pos": max(pcts),
        "max_neg": min(pcts),
    }


# ── 打印 ─────────────────────────────────────────────────────────────────────

def _print_upper_section(coeffs, all_samples, by_source, counter_fns):
    """打印 UPPER_COEFFS 的完整报告（每模型整体指标 + 每模型按来源细分）。"""
    c = coeffs
    W = 72
    print("\n" + "─" * W)
    print("  UPPER_COEFFS 拟合结果（LP，保证不低估）")
    print("─" * W)
    print(f"""
  UPPER_COEFFS = Coeffs(
      cjk    = {c.cjk:.4f},
      letter = {c.letter:.4f},
      digit  = {c.digit:.4f},
      punct  = {c.punct:.4f},
      space  = {c.space:.4f},
      word   = {c.word:.4f},
  )
""")

    # ── 整体指标：每个模型一行 ─────────────────────────────────────────────
    print(f"  整体指标（{len(all_samples)} 块）:")
    hdr = f"  {'模型':<14}  {'保证率':>8}  {'违规数':>6}  {'均高估':>8}  {'最大高估':>9}  {'MAPE':>7}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for model_name, fn in counter_fns.items():
        st = _eval_upper(all_samples, fn, coeffs)
        if not st:
            continue
        print(f"  {model_name:<14}  {st['guarantee']:>7.1f}%  "
              f"{st['violations']:>6}  "
              f"+{st['mean_over']:>6.1f}%  "
              f"+{st['max_over']:>7.1f}%  "
              f"{st['mape']:>6.1f}%")

    # ── 按数据来源细分：每个模型单独一张表 ───────────────────────────────
    src_keys = sorted(by_source.keys())
    for model_name, fn in counter_fns.items():
        print(f"\n  按数据来源细分 — 模型: {model_name}")
        hdr = f"  {'来源':<32}  {'块数':>4}  {'保证率':>8}  {'均高估':>8}  {'最大高估':>9}  {'MAPE':>7}"
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for src_key in src_keys:
            s_list = by_source[src_key]
            st = _eval_upper(s_list, fn, coeffs)
            if not st:
                continue
            print(f"  {src_key:<32}  {st['n']:>4}  "
                  f"{st['guarantee']:>7.1f}%  "
                  f"+{st['mean_over']:>6.1f}%  "
                  f"+{st['max_over']:>7.1f}%  "
                  f"{st['mape']:>6.1f}%")


def _print_default_section(coeffs, all_samples, by_source, counter_fns):
    """打印 DEFAULT_COEFFS 的完整报告（每模型整体指标 + 每模型按来源细分）。"""
    c = coeffs
    W = 72
    print("\n" + "─" * W)
    print("  DEFAULT_COEFFS 拟合结果（NNLS，最小化误差）")
    print("─" * W)
    print(f"""
  DEFAULT_COEFFS = Coeffs(
      cjk    = {c.cjk:.4f},
      letter = {c.letter:.4f},
      digit  = {c.digit:.4f},
      punct  = {c.punct:.4f},
      space  = {c.space:.4f},
      word   = {c.word:.4f},
  )
""")

    # ── 整体指标：每个模型一行 ─────────────────────────────────────────────
    print(f"  整体指标（{len(all_samples)} 块）:")
    hdr = f"  {'模型':<14}  {'MAPE':>7}  {'偏差':>8}  {'最大正误差':>10}  {'最大负误差':>10}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for model_name, fn in counter_fns.items():
        st = _eval_default(all_samples, fn, coeffs)
        if not st:
            continue
        print(f"  {model_name:<14}  {st['mape']:>6.1f}%  "
              f"{st['bias']:>+7.1f}%  "
              f"{st['max_pos']:>+9.1f}%  "
              f"{st['max_neg']:>+9.1f}%")

    # ── 按数据来源细分：每个模型单独一张表 ───────────────────────────────
    src_keys = sorted(by_source.keys())
    for model_name, fn in counter_fns.items():
        print(f"\n  按数据来源细分 — 模型: {model_name}")
        hdr = f"  {'来源':<32}  {'块数':>4}  {'MAPE':>7}  {'偏差':>8}  {'最大正误差':>10}  {'最大负误差':>10}"
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for src_key in src_keys:
            s_list = by_source[src_key]
            st = _eval_default(s_list, fn, coeffs)
            if not st:
                continue
            print(f"  {src_key:<32}  {st['n']:>4}  "
                  f"{st['mape']:>6.1f}%  "
                  f"{st['bias']:>+7.1f}%  "
                  f"{st['max_pos']:>+9.1f}%  "
                  f"{st['max_neg']:>+9.1f}%")


def _print_paste_block(upper, default):
    """打印可直接粘贴到 tokenizer_approx.py 的系数块。"""
    u, d = upper, default
    W = 72
    print("\n" + "═" * W)
    print("  # 将以下内容粘贴到 tokenizer_approx.py 以更新系数:")
    print("═" * W)
    print(f"""
UPPER_COEFFS = Coeffs(
    cjk    = {u.cjk:.4f},
    letter = {u.letter:.4f},
    digit  = {u.digit:.4f},
    punct  = {u.punct:.4f},
    space  = {u.space:.4f},
    word   = {u.word:.4f},
)

DEFAULT_COEFFS = Coeffs(
    cjk    = {d.cjk:.4f},
    letter = {d.letter:.4f},
    digit  = {d.digit:.4f},
    punct  = {d.punct:.4f},
    space  = {d.space:.4f},
    word   = {d.word:.4f},
)""")
    print("═" * W)


# ── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="自定义数据集接入 + 系数重新拟合",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--text", metavar="FILE", action="append", default=[],
                        help="自定义文本文件（UTF-8），可多次指定")
    parser.add_argument("--chunk-size", type=int, default=3000,
                        help="切块大小（字符数，默认 3000）")
    parser.add_argument("--with-synthetic", action="store_true",
                        help="追加合成数据集（21 类）")
    parser.add_argument("--samples", type=int, default=20,
                        help="合成数据每类样本数（默认 20，--with-synthetic 时生效）")
    parser.add_argument("--with-real", action="store_true",
                        help="追加 Wikipedia + GitHub 真实数据（首次运行自动下载）")
    parser.add_argument("--refresh", action="store_true",
                        help="强制重新下载真实数据（--with-real 时生效）")
    parser.add_argument("--models", type=str, default="kimi_fast",
                        help="逗号分隔的模型列表：kimi_fast, kimi, glm, dsv, mmax（默认 kimi_fast）")
    args = parser.parse_args()

    if not args.text and not args.with_synthetic and not args.with_real:
        parser.error("至少指定一个数据来源：--text FILE、--with-synthetic 或 --with-real")

    # ── 加载 tokenizer ──────────────────────────────────────────────────────
    model_names = [m.strip() for m in args.models.split(",")]
    counter_fns = {}
    for name in model_names:
        if name == "kimi_fast":
            try:
                from real_tokenizers import kimi_count_fast
                counter_fns["kimi_fast"] = kimi_count_fast
                print(f"[tokenizer] kimi_fast (tiktoken cl100k_base) 加载成功")
            except Exception as e:
                print(f"[错误] kimi_fast 加载失败: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            try:
                from real_tokenizers import glm_count, dsv_count, kimi_count, mmax_count
                fn_map = {"glm": glm_count, "dsv": dsv_count,
                          "kimi": kimi_count, "mmax": mmax_count}
                counter_fns[name] = fn_map[name]
                print(f"[tokenizer] {name} 加载成功")
            except Exception as e:
                print(f"[警告] {name} 加载失败，跳过: {e}", file=sys.stderr)

    if not counter_fns:
        print("[错误] 没有可用的 tokenizer", file=sys.stderr)
        sys.exit(1)

    # ── 收集样本 ─────────────────────────────────────────────────────────────
    all_samples: list[dict] = []

    print("\n数据来源汇总:")
    if args.text:
        custom = load_text_files(args.text, args.chunk_size)
        all_samples.extend(custom)

    if args.with_synthetic:
        from sample_gen import generate_samples, CATEGORIES
        synth = generate_samples(args.samples)
        for s in synth:
            s.setdefault("source", f"synthetic/{s['category']}")
            s["_file"] = "synthetic"
        all_samples.extend(synth)
        print(f"  合成数据集（{len(CATEGORIES)} 类×{args.samples} 样本）: {len(synth)} 块")

    if args.with_real:
        from data_fetch import fetch_all
        real = fetch_all(force_refresh=args.refresh, chunk_size=args.chunk_size)
        for s in real:
            s["_file"] = "real_data"
        all_samples.extend(real)
        print(f"  真实数据（Wikipedia+GitHub，chunk_size={args.chunk_size}）: {len(real)} 块")

    total_chars = sum(len(s["text"]) for s in all_samples)
    print(f"  {'─'*42}")
    print(f"  合计: {len(all_samples)} 块  ({total_chars:,} 字符)")
    print(f"  模型: {', '.join(counter_fns.keys())}")

    # ── 建立按来源分组索引 ───────────────────────────────────────────────────
    by_source: dict[str, list[dict]] = defaultdict(list)
    for s in all_samples:
        if s.get("_file") == "synthetic":
            key = f"合成/{s['category']}"
        elif s.get("_file") == "real_data":
            key = f"真实/{s['category']}"
        else:
            key = f"自定义/{s.get('_file', '?')}"
        by_source[key].append(s)

    # ── 检查 scipy ───────────────────────────────────────────────────────────
    try:
        import scipy  # noqa
    except ImportError:
        print("\n[错误] 需要 scipy: pip install scipy", file=sys.stderr)
        sys.exit(1)

    # ── 拟合 ─────────────────────────────────────────────────────────────────
    from benchmark import fit_upper_bound, fit_shared

    print("\n正在拟合 UPPER_COEFFS（LP）…")
    upper = fit_upper_bound(all_samples, counter_fns)

    print("正在拟合 DEFAULT_COEFFS（NNLS）…")
    default = fit_shared(all_samples, counter_fns)

    # ── 打印报告 ──────────────────────────────────────────────────────────────
    W = 72
    print("\n" + "═" * W)
    print("  拟合报告")
    print("═" * W)

    _print_upper_section(upper, all_samples, by_source, counter_fns)
    _print_default_section(default, all_samples, by_source, counter_fns)
    _print_paste_block(upper, default)


if __name__ == "__main__":
    main()
