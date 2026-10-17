"""R_q^H, its field structure, the trace Tr_H, the packing map psi and the map E.

Definitions follow the draft (4_preliminaries.tex, "Galois Automorphisms and Trace Map";
field-to-ring-rok.tex) and Hachi (ePrint 2026/156, Lemma 5, Eq. (7), Theorem 2):

  R_q = Z_q[X]/(X^d + 1), d = 2^alpha, q prime = 5 mod 8, k = 2^kappa dividing d/2.
  sigma_i : X -> X^i for odd i, H = <sigma_{-1}, sigma_{4k+1}>, |H| = d/k.
  Tr_H(a) = sum_{sigma in H} sigma(a).
  psi(a_0..a_{d/k-1}) = sum_{i<d/2k} a_i X^i + X^{d/2} sum_{i<d/2k} a_{d/2k+i} X^i.
  Theorem 2: Tr_H(psi(a) * sigma_{-1}(psi(b))) = (d/k) <a, b>.
  E(u) = (k/d) Tr_H(u * sigma_{-1}(v)), v = psi(monomials of the point).

Field representation (Hachi Eq. (7)). With t = X^{d/(2k)} (a primitive 4k-th root of unity,
t^{2k} = -1), every element of R_q^H is a_0 + sum_{m=1}^{k-1} a_m (t^m - t^{2k-m}); the
coordinates (a_0, ..., a_{k-1}) in Z_q^k represent F_{q^k}.  The coordinates are read directly from
the ring coefficients at positions m*d/(2k).

Index convention for bit strings (used by `monomials`): a bit string (b_1, ..., b_s) has index
nat(b) = sum_j b_j 2^{s-j}, i.e. b_1 is the most significant bit, so nat(i||j) = nat(i) 2^{|j|} + nat(j).
"""
from sage.all import GF, Integer, ZZ, randint


class Subfield:
    def __init__(self, q, d, k):
        q, d, k = Integer(q), int(d), int(k)
        assert q.is_prime() and q % 8 == 5, "q must be a prime = 5 mod 8"
        assert d >= 2 and d & (d - 1) == 0, "d must be a power of two"
        assert k >= 1 and k & (k - 1) == 0 and (d // 2) % k == 0, "k must be a power of two dividing d/2"
        self.q, self.d, self.k = q, d, k
        self.F = GF(q)
        self.step = d // (2 * k)                      # t = X^step
        # H as a set of odd exponents mod 2d
        gens = [(2 * d - 1) % (2 * d), (4 * k + 1) % (2 * d)]
        H = {1}
        frontier = [1]
        while frontier:
            h = frontier.pop()
            for g in gens:
                x = (h * g) % (2 * d)
                if x not in H:
                    H.add(x)
                    frontier.append(x)
        self.H = sorted(H)
        assert len(self.H) == d // k

    # ---- ring elements: lists of d elements of GF(q) ----
    def zero(self):
        return [self.F(0)] * self.d

    def ring(self, coeffs):
        assert len(coeffs) == self.d
        return [self.F(c) for c in coeffs]

    def add(self, a, b):
        return [x + y for x, y in zip(a, b)]

    def sub(self, a, b):
        return [x - y for x, y in zip(a, b)]

    def scale(self, c, a):
        c = self.F(c)
        return [c * x for x in a]

    def mul(self, a, b):
        """Negacyclic product in R_q (schoolbook)."""
        d = self.d
        out = [self.F(0)] * d
        for i, x in enumerate(a):
            if x == 0:
                continue
            for j, y in enumerate(b):
                e = i + j
                if e < d:
                    out[e] += x * y
                else:
                    out[e - d] -= x * y
        return out

    def monomial(self, e, c=1):
        """c * X^e in R_q for any integer e (X^d = -1, X^{2d} = 1)."""
        e %= 2 * self.d
        out = self.zero()
        if e < self.d:
            out[e] = self.F(c)
        else:
            out[e - self.d] = -self.F(c)
        return out

    def sigma(self, a, i):
        """sigma_i(a) = a(X^i) mod X^d + 1, i odd."""
        assert i % 2 == 1
        d = self.d
        out = [self.F(0)] * d
        for j, x in enumerate(a):
            if x == 0:
                continue
            e = (i * j) % (2 * d)
            if e < d:
                out[e] += x
            else:
                out[e - d] -= x
        return out

    def trace(self, a):
        out = self.zero()
        for h in self.H:
            out = self.add(out, self.sigma(a, h))
        return out

    def in_subfield(self, a):
        return all(self.sigma(a, h) == a for h in (self.H[1:] if len(self.H) > 1 else []))

    # ---- F_{q^k} in Hachi's basis ----
    def basis(self, m):
        """1 for m = 0, t^m - t^{2k-m} for 1 <= m < k, with t = X^{d/(2k)}."""
        if m == 0:
            return self.monomial(0)
        return self.sub(self.monomial(m * self.step), self.monomial((2 * self.k - m) * self.step))

    def embed(self, x):
        """Coordinates (x_0, ..., x_{k-1}) -> the ring element of R_q^H."""
        assert len(x) == self.k
        out = self.zero()
        for m, c in enumerate(x):
            if c != 0:
                out = self.add(out, self.scale(c, self.basis(m)))
        return out

    def extract(self, a):
        """The coordinates of a ring element of R_q^H (asserts membership)."""
        x = [a[m * self.step] for m in range(self.k)]
        assert self.embed(x) == list(a), "not an element of R_q^H"
        return x

    def fmul(self, x, y):
        return self.extract(self.mul(self.embed(x), self.embed(y)))

    def fadd(self, x, y):
        return [a + b for a, b in zip(x, y)]

    def finv(self, x):
        Q = self.q ** self.k
        return self.fpow(x, Q - 2)

    def fpow(self, x, e):
        result = [self.F(1)] + [self.F(0)] * (self.k - 1)
        base = list(x)
        e = Integer(e)
        while e > 0:
            if e & 1:
                result = self.fmul(result, base)
            base = self.fmul(base, base)
            e >>= 1
        return result

    def frandom(self):
        return [self.F.random_element() for _ in range(self.k)]

    def fone(self):
        return [self.F(1)] + [self.F(0)] * (self.k - 1)

    def fzero(self):
        return [self.F(0)] * self.k

    def inner(self, a, b):
        acc = self.fzero()
        for x, y in zip(a, b):
            acc = self.fadd(acc, self.fmul(x, y))
        return acc

    # ---- psi, psi^{-1}, E ----
    def psi(self, avec):
        d, k = self.d, self.k
        assert len(avec) == d // k
        half = d // (2 * k)
        out = self.zero()
        for i in range(half):
            out = self.add(out, self.mul(self.embed(avec[i]), self.monomial(i)))
            out = self.add(out, self.mul(self.embed(avec[half + i]), self.monomial(d // 2 + i)))
        return out

    def psi_inv(self, u):
        """a_i = (k/d) Tr_H(u * sigma_{-1}(psi(e_i))), by Theorem 2."""
        d, k = self.d, self.k
        half = d // (2 * k)
        scale = self.F(k) / self.F(d)
        out = []
        for i in range(d // k):
            e = i if i < half else d // 2 + (i - half)
            w = self.sigma(self.monomial(e), 2 * d - 1)          # sigma_{-1}(X^e) = X^{-e}
            out.append(self.extract(self.scale(scale, self.trace(self.mul(u, w)))))
        return out

    def monomials(self, x):
        """(prod_j x_j^{b_j}) for b in {0,1}^s, indexed by nat(b) with b_1 the most significant bit."""
        s = len(x)
        out = []
        for idx in range(2 ** s):
            acc = self.fone()
            for j in range(s):
                if (idx >> (s - 1 - j)) & 1:
                    acc = self.fmul(acc, x[j])
            out.append(acc)
        return out

    def E(self, u, v):
        """E(u) = (k/d) Tr_H(u * sigma_{-1}(v)) in R_q^H (as a ring element)."""
        scale = self.F(self.k) / self.F(self.d)
        return self.scale(scale, self.trace(self.mul(u, self.sigma(v, 2 * self.d - 1))))

    def E_matrix(self, v):
        """The d x d matrix over Z_q with cf(E(u)) = M cf(u)."""
        cols = [self.E(self.monomial(j), v) for j in range(self.d)]
        return [[cols[j][i] for j in range(self.d)] for i in range(self.d)]


def self_test(q, d, k, trials=3):
    """Checks Theorem 2, psi^{-1}(psi(a)) = a, closure and E(psi(a)) = <a, monomials(x)>."""
    S = Subfield(q, d, k)
    for _ in range(trials):
        x, y = S.frandom(), S.frandom()
        assert S.in_subfield(S.embed(x))
        assert S.fmul(x, y) == S.fmul(y, x)
        if any(c != 0 for c in x):
            assert S.fmul(x, S.finv(x)) == S.fone()
        a = [S.frandom() for _ in range(d // k)]
        b = [S.frandom() for _ in range(d // k)]
        lhs = S.trace(S.mul(S.psi(a), S.sigma(S.psi(b), 2 * d - 1)))
        rhs = S.embed([S.F(d // k) * c for c in S.inner(a, b)])
        assert lhs == rhs, "Theorem 2 fails"
        assert S.psi_inv(S.psi(a)) == a
        s = (d // k).bit_length() - 1
        pt = [S.frandom() for _ in range(s)]
        mono = S.monomials(pt)
        v = S.psi(mono)
        assert S.extract(S.E(S.psi(a), v)) == S.inner(a, mono)
    return True
