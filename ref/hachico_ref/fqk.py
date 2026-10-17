"""Vectorized F_{q^k} = R_q^H arithmetic in Hachi's basis (Hachi Eq. (7), m0-spec "Field").

An element is the int64 array (a_0, ..., a_{k-1}) of residues in [0, q) and stands for
a_0 + sum_{m>=1} a_m (t^m - t^{2k-m}), t = X^{d/(2k)}.  Arrays of elements have shape (..., k).

The multiplication table T[a][b][c] (coordinate c of basis_a * basis_b) is computed once from
`Subfield.fmul` (ref/hachico_ref/subfield.py, the main session's reference), so this module inherits
exactly its field structure; `self_check` compares random products with `Subfield.fmul`.

Also here: equality polynomials, multilinear extensions (MSB-first variable order, see `eq_table`),
and Lagrange interpolation at the integer nodes 1..D+1 used by the sumcheck messages.
"""
import numpy as np


class Fqk:
    def __init__(self, S):
        self.S = S
        self.q, self.k = int(S.q), int(S.k)
        q, k = self.q, self.k
        assert k * k * 2 * q < 2 ** 62 and q < 2 ** 31, "int64 headroom"
        T = np.zeros((k, k, k), dtype=np.int64)
        for a in range(k):
            for b in range(k):
                ea = [S.F(1 if j == a else 0) for j in range(k)]
                eb = [S.F(1 if j == b else 0) for j in range(k)]
                prod = S.fmul(ea, eb)
                for c in range(k):
                    v = int(prod[c])
                    T[a, b, c] = v - q if v > q // 2 else v
        self.T = T
        self.T2 = T.reshape(k * k, k)

    # ---- conversions ----
    def from_sub(self, x):
        """Subfield element (list of GF(q)) -> array (k,)."""
        return np.array([int(c) for c in x], dtype=np.int64)

    def to_sub(self, x):
        return [self.S.F(int(c)) for c in np.asarray(x).reshape(self.k)]

    def from_int(self, c):
        out = np.zeros(self.k, dtype=np.int64)
        out[0] = int(c) % self.q
        return out

    def lift(self, a):
        """Z_q array of any shape -> F array of shape (..., k) (embedding Z_q -> F_{q^k})."""
        a = np.asarray(a, dtype=np.int64) % self.q
        out = np.zeros(a.shape + (self.k,), dtype=np.int64)
        out[..., 0] = a
        return out

    def zeros(self, *shape):
        return np.zeros(shape + (self.k,), dtype=np.int64)

    def one(self):
        return self.from_int(1)

    # ---- arithmetic (broadcasting over leading axes) ----
    def add(self, x, y):
        return (np.asarray(x) + np.asarray(y)) % self.q

    def sub(self, x, y):
        return (np.asarray(x) - np.asarray(y)) % self.q

    def neg(self, x):
        return (-np.asarray(x)) % self.q

    def smul(self, s, x):
        """Z_q scalar(s) times F elements; s broadcasts against the leading axes of x."""
        s = np.asarray(s, dtype=np.int64) % self.q
        return (s[..., None] * np.asarray(x)) % self.q

    def mul(self, x, y):
        q, k = self.q, self.k
        x = np.asarray(x, dtype=np.int64)
        y = np.asarray(y, dtype=np.int64)
        P = (x[..., :, None] * y[..., None, :]) % q
        shp = P.shape[:-2]
        return (P.reshape(shp + (k * k,)) @ self.T2) % q

    def sum(self, x, axis=0):
        """Sum along an axis (exact: reduces in chunks to stay inside int64)."""
        x = np.moveaxis(np.asarray(x, dtype=np.int64) % self.q, axis, 0)
        acc = np.zeros(x.shape[1:], dtype=np.int64)
        chunk = 1 << 20   # 2^20 * q < 2^51
        for s in range(0, x.shape[0], chunk):
            acc = (acc + np.sum(x[s:s + chunk], axis=0)) % self.q
        return acc

    def pow(self, x, e):
        result = self.one()
        base = np.asarray(x, dtype=np.int64) % self.q
        e = int(e)
        while e > 0:
            if e & 1:
                result = self.mul(result, base)
            base = self.mul(base, base)
            e >>= 1
        return result

    def powers(self, x, count):
        """(x^0, ..., x^{count-1}) of shape (count, k)."""
        out = self.zeros(count)
        cur = self.one()
        for i in range(count):
            out[i] = cur
            cur = self.mul(cur, x)
        return out

    def inv(self, x):
        return self.pow(x, self.q ** self.k - 2)

    def eq(self, x, y):
        return bool(np.array_equal(np.asarray(x) % self.q, np.asarray(y) % self.q))

    # ---- multilinear machinery ----
    def eq_table(self, tau):
        """eq(tau, x) for all x in {0,1}^s, tau of shape (s, k); index nat(x) = sum_j x_j 2^{s-j}
        (x_1 most significant, m0-spec "Bit strings").  Output shape (2^s, k)."""
        tau = np.asarray(tau, dtype=np.int64) % self.q
        tab = self.one()[None, :]
        for j in range(tau.shape[0]):
            t = tau[j]
            f0 = self.sub(self.one(), t)
            new = self.zeros(tab.shape[0], 2)
            new[:, 0] = self.mul(tab, f0)
            new[:, 1] = self.mul(tab, t)
            tab = new.reshape(-1, self.k)
        return tab

    def eq_point(self, tau, bits):
        """eq(tau, b) for one bit string b (list of 0/1), computed as a product."""
        acc = self.one()
        for t, bit in zip(np.asarray(tau), bits):
            acc = self.mul(acc, t if bit else self.sub(self.one(), t))
        return acc

    def bind_top(self, tab, a):
        """Fix the most significant variable of a table (2^s, k) to a: T'[x] = T[x] + a (T[x+h] - T[x])."""
        h = tab.shape[0] // 2
        lo, hi = tab[:h], tab[h:]
        return self.add(lo, self.mul(self.sub(hi, lo), a))

    def mle_eval(self, tab, point):
        """Multilinear extension of a hypercube table (2^s, k) at point (s, k), MSB-first."""
        t = np.asarray(tab, dtype=np.int64) % self.q
        for a in np.asarray(point):
            t = self.bind_top(t, a)
        assert t.shape[0] == 1
        return t[0]

    # ---- univariate interpolation at nodes 1..D+1 ----
    def lagrange_eval(self, values, x):
        """values = (H(1), ..., H(D+1)) (shape (D+1, k)); returns H(x) for the unique polynomial of degree
        <= D through them, at x in F (shape (k,))."""
        q = self.q
        values = np.asarray(values, dtype=np.int64)
        D1 = values.shape[0]
        nodes = list(range(1, D1 + 1))
        x = np.asarray(x, dtype=np.int64) % q
        acc = self.zeros()
        for j, nj in enumerate(nodes):
            num = self.one()
            den = 1
            for mm in nodes:
                if mm == nj:
                    continue
                num = self.mul(num, self.sub(x, self.from_int(mm)))
                den = den * (nj - mm) % q
            coef = self.smul(pow(den, q - 2, q), num)
            acc = self.add(acc, self.mul(coef, values[j]))
        return acc

    def self_check(self, trials=50, seed=0):
        """Random products against Subfield.fmul (independent: via the ring embedding)."""
        import random
        rnd = random.Random(seed)
        for _ in range(trials):
            x = np.array([rnd.randrange(self.q) for _ in range(self.k)], dtype=np.int64)
            y = np.array([rnd.randrange(self.q) for _ in range(self.k)], dtype=np.int64)
            ref = self.from_sub(self.S.fmul(self.to_sub(x), self.to_sub(y)))
            assert np.array_equal(self.mul(x, y), ref)
        return True
