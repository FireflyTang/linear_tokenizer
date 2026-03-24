"""
bench_perf.py — estimate() 性能测试脚本

测试不同实现在 5K→160K token 输入规模下的耗时，与 tiktoken 对比。

测试的实现:
  tiktoken      — OpenAI tiktoken cl100k_base（基线）
  numpy_full    — estimate_numpy_full()：NumPy 全特征含 word 计数
  numpy_noword  — estimate_fast_noword()：NumPy 无 word 计数
  estimate      — estimate()：自动后端（C 扩展 or 回退 NumPy/regex）

用法:
    python bench_perf.py
    python bench_perf.py --sizes 5000,20000,80000
    python bench_perf.py --repeat 50   # 固定重复次数（默认自适应）
    python bench_perf.py --no-tiktoken # 跳过 tiktoken（无网络/无安装时）
"""

import argparse
import time
import sys


# ── 构造目标长度的测试文本 ──────────────────────────────────────────────────

def _build_corpus() -> str:
    """用合成中英文样本拼出足够长的文本（无需下载）。"""
    from sample_gen import generate_samples
    samples = generate_samples(n_per_category=50)
    # 只取中英文，token密度稳定
    texts = [s["text"] for s in samples
             if s["category"] in ("pure_chinese", "pure_english", "chat_zh", "chat_en", "mixed")]
    corpus = "\n".join(texts)
    # 重复到足够长（至少能覆盖 160K token）
    while len(corpus) < 1_200_000:
        corpus = corpus + "\n" + corpus
    return corpus


def _get_text_for_tokens(corpus: str, target_tokens: int, enc) -> str:
    """截取文本使其约等于 target_tokens 个 token（用 tiktoken 校准）。"""
    # 粗估：中英文混合约 2.5 字符/token
    est_chars = int(target_tokens * 2.5)
    text = corpus[:min(est_chars, len(corpus))]
    actual = len(enc.encode(text))

    # 如果偏差大于 5%，按比例调整一次
    if abs(actual - target_tokens) / target_tokens > 0.05:
        ratio = target_tokens / max(actual, 1)
        text = corpus[:min(int(est_chars * ratio), len(corpus))]

    return text


# ── 计时工具 ─────────────────────────────────────────────────────────────────

def _timeit(fn, text, min_total_ms: float = 300.0, fixed_repeat: int = 0) -> tuple[float, int]:
    """
    运行 fn(text) 若干次，返回 (每次平均耗时ms, 实际重复次数)。
    若 fixed_repeat > 0 则固定次数，否则自适应直到累计 ≥ min_total_ms。
    """
    # warmup
    fn(text)
    fn(text)

    if fixed_repeat > 0:
        t0 = time.perf_counter()
        for _ in range(fixed_repeat):
            fn(text)
        elapsed = (time.perf_counter() - t0) * 1000
        return elapsed / fixed_repeat, fixed_repeat

    # 自适应：先跑 3 次估算单次耗时
    t0 = time.perf_counter()
    for _ in range(3):
        fn(text)
    single_ms = (time.perf_counter() - t0) * 1000 / 3

    repeat = max(3, int(min_total_ms / max(single_ms, 0.01)))
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn(text)
    elapsed = (time.perf_counter() - t0) * 1000
    return elapsed / repeat, repeat


# ── 主逻辑 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="estimate() 性能测试")
    parser.add_argument("--sizes", type=str,
                        default="5000,10000,20000,40000,80000,160000",
                        help="测试的 token 规模（逗号分隔，默认 5K/10K/20K/40K/80K/160K）")
    parser.add_argument("--repeat", type=int, default=0,
                        help="固定重复次数（默认 0=自适应，目标每组 ≥ 300ms）")
    parser.add_argument("--no-tiktoken", action="store_true",
                        help="跳过 tiktoken 测试（无安装时使用）")
    args = parser.parse_args()

    targets = [int(x.strip()) for x in args.sizes.split(",")]

    # ── 导入被测函数 ──────────────────────────────────────────────────────────
    from tokenizer_approx import (
        estimate, estimate_numpy_full, estimate_fast_noword, _USE_C,
    )

    backend_label = "C扩展" if _USE_C else "NumPy回退"
    print(f"\n性能测试: estimate() vs tiktoken")
    print(f"estimate() 后端: {backend_label}")
    print(f"重复次数: {'自适应（每组 ≥ 300ms）' if args.repeat == 0 else args.repeat}")

    # ── 加载 tiktoken ─────────────────────────────────────────────────────────
    enc = None
    if not args.no_tiktoken:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            enc.encode("warmup")  # 触发初始化
        except ImportError:
            print("[警告] tiktoken 未安装，跳过（pip install tiktoken）")
            enc = None

    # ── 构建语料库 ────────────────────────────────────────────────────────────
    print("\n准备测试语料…")
    corpus = _build_corpus()
    print(f"语料库长度: {len(corpus):,} 字符\n")

    # ── 表头 ──────────────────────────────────────────────────────────────────
    col_w = 12
    backends = [
        ("tiktoken",     lambda t: enc.encode(t),         enc is not None),
        ("numpy_full",   estimate_numpy_full,              True),
        ("numpy_noword", estimate_fast_noword,             True),
        (f"estimate({backend_label})", estimate,           True),
    ]
    active = [(name, fn) for name, fn, ok in backends if ok]

    hdr_parts = [f"{'token规模':>10}"]
    for name, _ in active:
        hdr_parts.append(f"{name:>{col_w}}")
    if enc is not None:
        hdr_parts.append(f"{'vs_tiktoken':>{col_w}}")
    print("  " + "  ".join(hdr_parts))
    print("  " + "─" * (len("  ".join(hdr_parts)) + 2))

    # ── 逐规模测试 ────────────────────────────────────────────────────────────
    last_tiktoken_ms = None
    for target in targets:
        # 获取对应长度文本
        if enc is not None:
            text = _get_text_for_tokens(corpus, target, enc)
            actual_tokens = len(enc.encode(text))
        else:
            # 无 tiktoken 时用字符数估算
            text = corpus[:int(target * 2.5)]
            actual_tokens = target  # 标称值

        row_parts = [f"{actual_tokens:>10,}"]
        tiktoken_ms = None

        for name, fn in active:
            ms, n = _timeit(fn, text, fixed_repeat=args.repeat)
            row_parts.append(f"{ms:>{col_w-2}.2f}ms")
            if name == "tiktoken":
                tiktoken_ms = ms
                last_tiktoken_ms = ms

        if enc is not None and tiktoken_ms is not None:
            # estimate vs tiktoken 倍数
            estimate_ms = None
            for name, fn in active:
                if name.startswith("estimate"):
                    ms, _ = _timeit(fn, text, fixed_repeat=args.repeat)
                    estimate_ms = ms
                    break
            if estimate_ms and estimate_ms > 0:
                ratio = tiktoken_ms / estimate_ms
                row_parts.append(f"{ratio:>{col_w-1}.1f}×")
            else:
                row_parts.append(f"{'—':>{col_w}}")

        print("  " + "  ".join(row_parts))

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    print()
    if enc is not None:
        print(f"  最快实现: estimate() with {backend_label}")
    print(f"  测试语料: 中英文混合（pure_chinese / pure_english / chat / mixed）")
    print(f"  注: 耗时为多次重复的平均值，已排除 import 和 warmup 时间\n")


if __name__ == "__main__":
    main()
