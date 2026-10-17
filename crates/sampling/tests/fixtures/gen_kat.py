#!/usr/bin/env python3
"""Known-answer tests for hachi-co-sampling.

The PRG is recomputed with Python's hashlib.shake_128 (OpenSSL), an implementation of SHAKE128
independent of the RustCrypto `shake` crate, and every sampler is transcribed from the
byte-level rules in the crate's rustdoc. Python integers are unbounded, so none of the
overflow-sensitive paths (u64 masks, i64 offsets, 2^64-wide ranges) are shared with Rust.

Run from anywhere:  python3 crates/sampling/tests/fixtures/gen_kat.py
It rewrites kat.json next to this script; the Rust test tests/kat.rs replays it.
"""
import hashlib
import json
import math
import platform
import ssl
from pathlib import Path

PRG_TAG = b"HACHI-CO-v1/prg"


def frame(x: bytes) -> bytes:
    return len(x).to_bytes(8, "little") + x


class Prg:
    """SHAKE128 over f(tag) || f(seed) || f(domain) || f(LE64(index)), read in order."""

    def __init__(self, seed: bytes, domain: bytes, index: int):
        assert len(seed) == 32 and 0 <= index < 2**64
        self.input = frame(PRG_TAG) + frame(seed) + frame(domain) + frame(index.to_bytes(8, "little"))
        self.buf = b""
        self.pos = 0

    def read(self, n: int) -> bytes:
        end = self.pos + n
        if end > len(self.buf):
            self.buf = hashlib.shake_128(self.input).digest(max(end, 2 * len(self.buf), 4096))
        out = self.buf[self.pos:end]
        self.pos = end
        return out

    def u64(self) -> int:
        return int.from_bytes(self.read(8), "little")

    def u32(self) -> int:
        return int.from_bytes(self.read(4), "little")


def below(prg: Prg, n: int) -> int:
    """uniform_below: low bit_length(n-1) bits of one u64 per try, accepted if < n."""
    assert n >= 1
    if n == 1:
        return 0
    b = (n - 1).bit_length()
    while True:
        w = prg.u64() & ((1 << b) - 1)
        if w < n:
            return w


def below_big(prg: Prg, n: int) -> int:
    """uniform_below_big: L = ceil(b/64) words per try, little-endian limbs, masked to b bits."""
    assert n >= 1
    if n == 1:
        return 0
    b = (n - 1).bit_length()
    limbs = (b + 63) // 64
    while True:
        x = 0
        for k in range(limbs):
            x |= prg.u64() << (64 * k)
        x &= (1 << b) - 1
        if x < n:
            return x


def range_i64(prg, lo, hi):
    return lo + below(prg, hi - lo + 1)  # width 2^64 keeps all 64 bits: one word, always accepted


def mod_q(prg, q, n):
    return [below(prg, q) for _ in range(n)]


def boxed(prg, gamma, n):
    return [below(prg, 2 * gamma + 1) - gamma for _ in range(n)]


def short(prg, beta, n):
    return [below(prg, beta) - beta // 2 for _ in range(n)]


def floyd(prg, n, k):
    chosen = set()
    for j in range(n - k, n):
        t = below(prg, j + 1)
        chosen.add(j if t in chosen else t)
    return sorted(chosen)


def l1_weights(d, omega):
    return [2**i * math.comb(d, i) * math.comb(omega, i) for i in range(min(d, omega) + 1)]


def l1_sample(prg, d, omega):
    w = l1_weights(d, omega)
    u = below_big(prg, sum(w))
    i, acc = 0, w[0]
    while u >= acc:
        i += 1
        acc += w[i]
    positions = floyd(prg, d, i)
    sums = [s + 1 for s in floyd(prg, omega, i)]
    c, prev = [0] * d, 0
    for p, s in zip(positions, sums):
        a, prev = s - prev, s
        c[p] = a if below(prg, 2) == 0 else -a
    return c


def sparse_sample(prg, d, omega):
    c = [0] * d
    for p in floyd(prg, d, omega):
        c[p] = 1 if below(prg, 2) == 0 else -1
    return c


def sim_rej(prg, gamma, beta1, n):
    outer, inner = 2 * gamma + 1, 2 * gamma - 2 * beta1 + 1
    for _ in range(n):
        if below(prg, outer) >= inner:
            return None
    return boxed(prg, gamma - beta1, n)


def expand(seed, domain, rows, cols, d, q):
    return [[mod_q(Prg(seed, domain, i * cols + j), q, d) for j in range(cols)] for i in range(rows)]


SEED_A = bytes(range(32))
SEED_B = bytes([0xA5] * 32)


def stream(seed, domain, index):
    return {"seed": seed.hex(), "domain": domain.hex(), "index": index}


def with_tail(prg, case):
    """Record how many bytes the sampler read (always whole 64-bit words) and the next word, so
    that the test also checks the consumption of the stream."""
    assert prg.pos % 8 == 0
    case["words"] = prg.pos // 8
    case["next_u64"] = prg.u64()
    return case


def main():
    out = {
        "generator": {
            "script": "crates/sampling/tests/fixtures/gen_kat.py",
            "python": platform.python_version(),
            "openssl": ssl.OPENSSL_VERSION,
        }
    }

    out["prg_streams"] = []
    for seed, domain, index in [
        (SEED_A, b"", 0),
        (SEED_A, b"", 1),
        (SEED_A, b"matrix/A", 0),
        (SEED_A, b"matrix/A", 2**64 - 1),
        (SEED_B, bytes(range(256)) * 2, 7),
    ]:
        case = stream(seed, domain, index)
        case["bytes"] = Prg(seed, domain, index).read(400).hex()  # crosses the 168-byte rate
        out["prg_streams"].append(case)

    # Mixed u32/u64/byte reads: words are little-endian slices of one byte stream.
    prg = Prg(SEED_B, b"mixed", 3)
    calls = []
    for op, n in [("u32", 0), ("u64", 0), ("bytes", 5), ("u64", 0), ("bytes", 0), ("u32", 0), ("bytes", 200), ("u64", 0)]:
        if op == "u32":
            calls.append({"op": op, "value": prg.u32()})
        elif op == "u64":
            calls.append({"op": op, "value": prg.u64()})
        else:
            calls.append({"op": op, "len": n, "value": prg.read(n).hex()})
    out["prg_mixed"] = [dict(stream(SEED_B, b"mixed", 3), calls=calls)]

    out["uniform_below"] = []
    for i, n in enumerate([1, 2, 3, 5, 7, 10, 1000, 2**32 + 1, 2**63 - 1, 2**63, 2**63 + 1, 2**64 - 2, 2**64 - 1]):
        prg = Prg(SEED_A, b"uniform_below", i)
        values = [below(prg, n) for _ in range(40)]
        out["uniform_below"].append(with_tail(prg, dict(stream(SEED_A, b"uniform_below", i), n=n, values=values)))

    out["uniform_below_big"] = []
    for i, n in enumerate([1, 5, 2**64 - 1, 2**64, 2**64 + 1, 3 * 2**100 + 7, 2**128, 10**60, math.comb(1024, 40) * 2**40]):
        prg = Prg(SEED_A, b"uniform_below_big", i)
        values = [str(below_big(prg, n)) for _ in range(20)]
        out["uniform_below_big"].append(with_tail(prg, dict(stream(SEED_A, b"uniform_below_big", i), n=str(n), values=values)))

    out["uniform_range_i64"] = []
    for i, (lo, hi) in enumerate([(0, 0), (-1, 1), (-5, 3), (-(2**63), 2**63 - 1), (-(2**63), -(2**63) + 2), (2**63 - 3, 2**63 - 1), (-(2**62), 2**62)]):
        prg = Prg(SEED_A, b"uniform_range_i64", i)
        values = [range_i64(prg, lo, hi) for _ in range(30)]
        out["uniform_range_i64"].append(with_tail(prg, dict(stream(SEED_A, b"uniform_range_i64", i), lo=lo, hi=hi, values=values)))

    out["uniform_mod_q"] = []
    for i, q in enumerate([1, 3, 8388637, 33554501, 134217757, 2**61 - 1, 2**63 - 25]):
        prg = Prg(SEED_A, b"uniform_mod_q", i)
        case = dict(stream(SEED_A, b"uniform_mod_q", i), q=q, len=33, values=mod_q(prg, q, 33))
        out["uniform_mod_q"].append(with_tail(prg, case))

    out["boxed"] = []
    for i, gamma in enumerate([0, 1, 2, 49152, 2**40, 2**62, 2**63 - 1]):
        prg = Prg(SEED_A, b"boxed", i)
        case = dict(stream(SEED_A, b"boxed", i), gamma=gamma, len=25, values=boxed(prg, gamma, 25))
        out["boxed"].append(with_tail(prg, case))

    out["short"] = []
    for i, beta in enumerate([1, 2, 3, 4, 5, 16, 2**63, 2**64 - 1]):
        prg = Prg(SEED_A, b"short", i)
        case = dict(stream(SEED_A, b"short", i), beta=beta, len=25, values=short(prg, beta, 25))
        out["short"].append(with_tail(prg, case))

    out["l1ball_count"] = [
        {"d": d, "omega": w, "count": str(sum(l1_weights(d, w)))}
        for d, w in [(0, 0), (0, 5), (5, 0), (1, 7), (16, 2), (64, 10), (256, 23), (1024, 40), (1024, 1024), (2048, 61)]
    ]
    out["sparse_count"] = [
        {"d": d, "omega": w, "count": str(math.comb(d, w) * 2**w if w <= d else 0)}
        for d, w in [(0, 0), (5, 6), (16, 2), (64, 10), (1024, 40), (1024, 1024)]
    ]

    out["l1ball_sample"] = []
    for i, (d, omega, draws) in enumerate([(0, 3, 2), (4, 0, 2), (1, 9, 10), (16, 2, 20), (16, 5, 20), (8, 20, 10), (256, 23, 4), (1024, 40, 2)]):
        prg = Prg(SEED_B, b"l1ball", i)
        samples = [l1_sample(prg, d, omega) for _ in range(draws)]
        out["l1ball_sample"].append(with_tail(prg, dict(stream(SEED_B, b"l1ball", i), d=d, omega=omega, draws=samples)))

    out["sparse_sample"] = []
    for i, (d, omega, draws) in enumerate([(0, 0, 2), (4, 4, 3), (16, 2, 20), (16, 16, 2), (1024, 40, 2)]):
        prg = Prg(SEED_B, b"sparse", i)
        samples = [sparse_sample(prg, d, omega) for _ in range(draws)]
        out["sparse_sample"].append(with_tail(prg, dict(stream(SEED_B, b"sparse", i), d=d, omega=omega, draws=samples)))

    out["sim_rej"] = []
    for i, (gamma, beta1, n, draws) in enumerate([(0, 0, 3, 2), (2, 1, 2, 30), (3, 3, 1, 30), (49152, 32, 768, 6)]):
        prg = Prg(SEED_B, b"sim_rej", i)
        samples = [sim_rej(prg, gamma, beta1, n) for _ in range(draws)]
        assert any(s is None for s in samples) == (beta1 > 0), "both outcomes should occur"
        out["sim_rej"].append(with_tail(prg, dict(stream(SEED_B, b"sim_rej", i), gamma=gamma, beta1=beta1, n=n, draws=samples)))

    out["matrix"] = []
    for seed, domain, rows, cols, d, q in [
        (SEED_A, b"A", 1, 2, 16, 8388637),
        (SEED_B, b"D", 2, 3, 8, 2**61 - 1),
        (SEED_A, b"", 3, 1, 4, 5),
        (SEED_A, b"empty", 0, 4, 16, 17),
    ]:
        out["matrix"].append(
            {"seed": seed.hex(), "domain": domain.hex(), "rows": rows, "cols": cols, "d": d, "q": q,
             "entries": expand(seed, domain, rows, cols, d, q)}
        )

    # Rejections must actually occur in these streams, so that the KATs cover that path: a case
    # that read more words than it returned values rejected at least one word.
    for case in out["uniform_below"]:
        n, k, words = case["n"], len(case["values"]), case["words"]
        if n == 1:
            assert words == 0, "n = 1 reads nothing"
        elif n in (3, 5, 10, 2**32 + 1, 2**63 + 1):  # P(no rejection in 40 words) <= 0.75^40 < 1.01e-5
            assert words > k, f"no rejection for n = {n}"
        else:
            assert words >= k
    assert out["uniform_below_big"][4]["words"] > 2 * 20, "n = 2^64 + 1 must reject"
    assert out["uniform_range_i64"][3]["words"] == 30, "the 2^64-wide range never rejects"

    path = Path(__file__).with_name("kat.json")
    # One line per case, so that the file stays small and diffs stay readable.
    lines = []
    for key, value in out.items():
        if isinstance(value, list):
            body = ",\n".join("  " + json.dumps(case, separators=(",", ":")) for case in value)
            lines.append(f"{json.dumps(key)}: [\n{body}\n ]")
        else:
            lines.append(f"{json.dumps(key)}: {json.dumps(value)}")
    path.write_text("{\n " + ",\n ".join(lines) + "\n}\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
