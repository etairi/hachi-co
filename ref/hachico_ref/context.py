"""Shared context of one oracle instance, and an independent Sage-based checker for ring relations."""
import numpy as np
from sage.all import GF, PolynomialRing

from .fqk import Fqk
from .ring import Ring
from .subfield import Subfield


class Ctx:
    """Parameters plus the arithmetic objects: R (numpy R_q), S (subfield.py), F (vectorized F_{q^k}),
    and SR (Sage R_q used only to check relations)."""

    def __init__(self, P, seed, setup_seed=None):
        self.P = P
        self.seed = seed if isinstance(seed, bytes) else seed.encode()
        if setup_seed is None:
            setup_seed = ("hachi-co-ref/setup/" + P.name).encode()
        self.setup_seed = setup_seed if isinstance(setup_seed, bytes) else setup_seed.encode()
        self.R = Ring(P.q, P.d)
        self.S = Subfield(P.q, P.d, P.k)
        self.F = Fqk(self.S)
        self.SR = SageRing(P.q, P.d)

    # field element <-> ring element (the embedding of R_q^H into R_q, subfield.py)
    def embed(self, x):
        return np.array([int(c) for c in self.S.embed(self.F.to_sub(x))], dtype=np.int64)

    def extract(self, a):
        """Coordinates of a ring element of R_q^H (raises AssertionError if not in R_q^H)."""
        return self.F.from_sub(self.S.extract(self.S.ring([int(c) for c in a])))

    def in_subfield(self, a):
        return self.S.in_subfield(self.S.ring([int(c) for c in a]))


class SageRing:
    """R_q = GF(q)[X]/(X^d + 1) and Z_q[X]/(X^{2d} + 1) through Sage (FLINT), for independent checks."""

    def __init__(self, q, d):
        self.q, self.d = int(q), int(d)
        self.PR = PolynomialRing(GF(self.q), 'X')
        X = self.PR.gen()
        self.mod_d = X ** self.d + 1
        self.mod_2d = X ** (2 * self.d) + 1
        self.X = X

    def poly(self, coeffs):
        return self.PR([int(c) for c in coeffs])

    def coeffs(self, p, length):
        c = [int(x) for x in p.list()]
        assert len(c) <= length, "degree too large"
        return c + [0] * (length - len(c))

    def rq(self, coeffs):
        return self.poly(coeffs) % self.mod_d

    def matvec(self, M, v):
        """M (rows, cols, d) times v (cols, d) in R_q, as a list of coefficient lists."""
        rows, cols = len(M), len(M[0])
        vv = [self.poly(v[j]) for j in range(cols)]
        out = []
        for i in range(rows):
            acc = self.PR(0)
            for j in range(cols):
                if vv[j] != 0:
                    acc += self.poly(M[i][j]) * vv[j]
            out.append(self.coeffs(acc % self.mod_d, self.d))
        return out

    def is_unit(self, coeffs):
        """Whether a ring element is invertible in R_q (gcd with X^d + 1 is 1)."""
        return self.poly(coeffs).gcd(self.mod_d) == 1
