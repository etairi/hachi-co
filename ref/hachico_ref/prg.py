"""Deterministic sampling for the oracle: a SHAKE128 counter-mode PRG and the samplers built on it.

Stream for (seed, domain, index):
    block_j = SHAKE128(TAG || u64le(len(seed)) || seed || u64le(len(domain)) || domain || u64le(index)
                       || u64le(j)).digest(168),   j = 0, 1, 2, ...
and the stream is block_0 || block_1 || ...  (TAG = b"HACHI-CO-REF-PRG-v1").

uniform_below(n): read ceil(bits(n-1)/8) bytes little-endian, mask to bits(n-1) bits, reject if >= n.

This PRG only makes the oracle reproducible.  The Rust side does not need to re-derive anything from it:
every sampled value is written to the fixtures.
"""
import hashlib
from math import comb

TAG = b"HACHI-CO-REF-PRG-v1"


def _u64(x):
    return int(x).to_bytes(8, "little")


class Prg:
    def __init__(self, seed, domain, index=0):
        if isinstance(seed, str):
            seed = seed.encode()
        dom = domain.encode()
        self.prefix = TAG + _u64(len(seed)) + seed + _u64(len(dom)) + dom + _u64(index)
        self.counter = 0
        self.buf = bytearray()
        self.pos = 0

    def _refill(self):
        self.buf = self.buf[self.pos:] + hashlib.shake_128(self.prefix + _u64(self.counter)).digest(168)
        self.pos = 0
        self.counter += 1

    def read(self, n):
        while len(self.buf) - self.pos < n:
            self._refill()
        out = bytes(self.buf[self.pos:self.pos + n])
        self.pos += n
        return out

    def below(self, n):
        """Uniform integer in [0, n), n >= 1, exact (rejection sampling)."""
        n = int(n)
        assert n >= 1
        if n == 1:
            return 0
        bits = (n - 1).bit_length()
        nbytes = (bits + 7) // 8
        mask = (1 << bits) - 1
        while True:
            x = int.from_bytes(self.read(nbytes), "little") & mask
            if x < n:
                return x

    def range(self, lo, hi):
        """Uniform integer in [lo, hi] (inclusive)."""
        return lo + self.below(hi - lo + 1)

    def mod_q(self, q, count):
        return [self.below(q) for _ in range(count)]

    def boxed(self, gamma, count):
        """Uniform in [-gamma, gamma]^count (the mask distribution chi_r, 4_P:117)."""
        return [self.below(2 * gamma + 1) - gamma for _ in range(count)]

    def short(self, beta, count):
        """Uniform coefficients of S_beta: [ceil(-beta/2), ceil(beta/2) - 1] (4_P:88)."""
        lo, hi = -(beta // 2), (beta + 1) // 2 - 1
        return [self.range(lo, hi) for _ in range(count)]

    def subset(self, size, k):
        """Uniform k-subset of [0, size), returned sorted (partial Fisher-Yates)."""
        pool = list(range(size))
        for j in range(k):
            t = j + self.below(size - j)
            pool[j], pool[t] = pool[t], pool[j]
        return sorted(pool[:k])


def l1ball_weights(d, omega):
    """Number of c in Z^d with ||c||_1 <= omega and support size i: 2^i C(d,i) C(omega,i)."""
    return [2 ** i * comb(d, i) * comb(omega, i) for i in range(min(d, omega) + 1)]


def l1ball_count(d, omega):
    return sum(l1ball_weights(d, omega))


def l1ball_sample(prg, d, omega):
    """Exactly uniform c in {c in Z^d : ||c||_1 <= omega} (the challenge set C of 5_BB:122).

    Support size i with weight 2^i C(d,i) C(omega,i) (exact integer draw); a uniform i-subset of positions;
    uniform signs; magnitudes = the gaps of a uniform i-subset {p_1 < ... < p_i} of [1, omega]
    (m_1 = p_1, m_j = p_j - p_{j-1}), assigned to the positions in increasing order."""
    w = l1ball_weights(d, omega)
    x = prg.below(sum(w))
    i = 0
    while x >= w[i]:
        x -= w[i]
        i += 1
    pos = prg.subset(d, i)
    signs = [prg.below(2) for _ in range(i)]
    pts = [p + 1 for p in prg.subset(omega, i)]
    mags = [pts[0]] + [pts[j] - pts[j - 1] for j in range(1, i)] if i else []
    c = [0] * d
    for j in range(i):
        c[pos[j]] = -mags[j] if signs[j] else mags[j]
    return c
