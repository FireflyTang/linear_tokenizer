"""Benchmark: numba JIT vs numpy vs tiktoken."""
import numpy as np
import numba as nb
import tiktoken, time, sys
from tokenizer_approx import estimate_numpy_full

@nb.njit(cache=True)
def _count_features(cp):
    cjk = 0; letter = 0; digit = 0; space = 0; word = 0
    in_word = 0
    n = len(cp)
    for i in range(n):
        c = cp[i]
        if c <= 0x7f:
            if (c >= 0x41 and c <= 0x5a) or (c >= 0x61 and c <= 0x7a):
                letter += 1
                if not in_word: word += 1; in_word = 1
            elif c >= 0x30 and c <= 0x39:
                digit += 1
                if not in_word: word += 1; in_word = 1
            elif c == 0x20 or c == 0x09 or c == 0x0a or c == 0x0d:
                space += 1
                in_word = 0
            else:
                if not in_word: word += 1; in_word = 1
        else:
            if (c >= 0x4e00 and c <= 0x9fff) or (c >= 0x3000 and c <= 0x303f) or (c >= 0xff00 and c <= 0xffef):
                cjk += 1
            elif c == 0x85 or c == 0xa0 or (c >= 0x2000 and c <= 0x200b):
                space += 1
                in_word = 0
                continue
            if not in_word: word += 1; in_word = 1
    punct = n - cjk - letter - digit - space
    return cjk, letter, digit, punct, space, word

def estimate_numba(text, c_cjk=0.6330, c_letter=0.1406, c_digit=0.7876,
                   c_punct=0.7115, c_space=0.0995, c_word=0.3633):
    if not text:
        return 0
    cp = np.frombuffer(text.encode('utf-32-le'), dtype=np.uint32)
    cjk, letter, digit, punct, space, word = _count_features(cp)
    total = cjk*c_cjk + letter*c_letter + digit*c_digit + punct*c_punct + space*c_space + word*c_word
    return max(1, round(total))

# Also test: skip encode, pass raw buffer via ctypes
def estimate_numba_preenc(cp_array, c_cjk=0.6330, c_letter=0.1406, c_digit=0.7876,
                          c_punct=0.7115, c_space=0.0995, c_word=0.3633):
    """Version where the caller passes pre-encoded uint32 array."""
    cjk, letter, digit, punct, space, word = _count_features(cp_array)
    total = cjk*c_cjk + letter*c_letter + digit*c_digit + punct*c_punct + space*c_space + word*c_word
    return max(1, round(total))


if __name__ == "__main__":
    # Warm up JIT
    estimate_numba('hello world 你好')
    print('Numba JIT compiled.')

    enc = tiktoken.get_encoding('cl100k_base')

    from sample_gen import generate_samples
    samples = generate_samples(500)
    corpus = ' '.join(s['text'] for s in samples)
    while len(enc.encode(corpus)) < 200000:
        corpus = corpus + corpus

    targets = [5000, 10000, 20000, 40000, 80000, 160000]

    print(f"{'Tokens':>10} {'Chars':>10} {'tiktoken':>12} {'np_full':>12} {'numba':>12} {'numba_pre':>12}")
    print('-' * 74)

    last_times = {}
    for target in targets:
        est_chars = int(target * 3.5)
        text = corpus[:est_chars]
        actual = len(enc.encode(text))
        if actual < target * 0.95:
            text = corpus[:int(est_chars * target / actual * 1.02)]
            actual = len(enc.encode(text))

        # Pre-encode for the pre-encoded variant
        cp_pre = np.frombuffer(text.encode('utf-32-le'), dtype=np.uint32)

        # Warm
        len(enc.encode(text))
        estimate_numpy_full(text)
        estimate_numba(text)
        estimate_numba_preenc(cp_pre)

        N = max(3, 50000 // target)
        times = {}

        t0 = time.perf_counter()
        for _ in range(N): len(enc.encode(text))
        times['tiktoken'] = (time.perf_counter() - t0) / N * 1000

        t0 = time.perf_counter()
        for _ in range(N): estimate_numpy_full(text)
        times['np_full'] = (time.perf_counter() - t0) / N * 1000

        t0 = time.perf_counter()
        for _ in range(N): estimate_numba(text)
        times['numba'] = (time.perf_counter() - t0) / N * 1000

        t0 = time.perf_counter()
        for _ in range(N): estimate_numba_preenc(cp_pre)
        times['numba_pre'] = (time.perf_counter() - t0) / N * 1000

        print(f"{actual:>10,} {len(text):>10,} {times['tiktoken']:>10.2f}ms {times['np_full']:>10.2f}ms {times['numba']:>10.2f}ms {times['numba_pre']:>10.2f}ms")
        sys.stdout.flush()
        last_times = times

    print()
    for name in ['np_full', 'numba', 'numba_pre']:
        print(f"  {name}: {last_times['tiktoken']/last_times[name]:.1f}x vs tiktoken")

    # Correctness
    text = corpus[:50000]
    a = estimate_numpy_full(text)
    b = estimate_numba(text)
    print(f"\nCorrectness: np_full={a}  numba={b}  match={a==b}")
