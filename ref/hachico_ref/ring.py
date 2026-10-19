"""R_q = Z_q[X]/(X^d + 1) on numpy int64 arrays, the gadgets G, J and their inverses.

Representation: a ring element is an int64 array of shape (d,) with residues in [0, q); index j is the
coefficient of X^j.  Vectors of ring elements have shape (len, d), matrices (rows, cols, d).

Overflow guard (exact int64 arithmetic): products are formed on residues in [0, q), and a sum of products
is reduced after at most d of them, with at most one more residue added (matvec adds the reduced
accumulator to a convolution).  Every intermediate is therefore at most d (q-1)^2 + (q-1), and `Ring`
asserts d (q-1)^2 + (q-1) <= 2^63 - 1, the int64 limit (numpy's integer convolve, einsum and @ accumulate
in int64; the partial sums of non-negative terms grow monotonically, so the final bound covers them).
`polymul_full` and `g_apply` assert their own counts (min(len a, len b) resp. delta products).  The same
bound covers the d-term sums outside this class, M(alpha) and y(alpha) in sumcheck.py and V's E cf(u) in
oracle.py: Ctx builds a Ring for every instance.  Every toy preset satisfies it; toy-n8 (q = 536871029,
just above 2^29) has d (q-1)^2 + (q-1) < 2^62.000001, and the bound allows q up to 759250125 (2^29.5) at
d = 16.  The guard used to be the stricter d (q-1)^2 < 2^62, which toy-n8 misses by a factor 1 + 4e-7;
the self-test `t_ring` checks the products at every toy modulus, all-(q-1) operands included.

Gadgets (draft 4_P:94-102, m0-spec "Digit sets"):
  * digits are balanced, in [ceil(-b/2), ceil(b/2) - 1];
  * G_{b,n} = I_n (x) (1, b, ..., b^(delta-1)): element-major, entry i's delta digits at positions
    i*delta, ..., i*delta + delta - 1 (digit e has weight b^e); same layout for J with tau digits;
  * G^{-1} of a residue x decomposes rep(x) = x if x <= top else x - q, top = (ceil(b/2) - 1) * S,
    S = (b^delta - 1)/(b - 1)  (the rule of the legacy `to_window`, crates/hachi/src/arithmetic/utils.rs:100;
    the m0-spec calls it "the unique representative in the window", but when b^delta > q a residue can have
    two representatives in the window, so a rule is needed: this one takes the non-negative one);
  * J^{-1} of a short integer z decomposes z itself into tau digits and raises ValueError if z is outside
    [ceil(-b/2) * S_tau, (ceil(b/2) - 1) * S_tau].
"""
import numpy as np

from .params import digit_bounds

INT64_MAX = 2 ** 63 - 1


def int64_sum_fits(terms, q, extra=0):
    """Whether a sum of `terms` products of residues in [0, q), plus `extra` more residues, fits in int64."""
    return terms * (q - 1) ** 2 + extra * (q - 1) <= INT64_MAX


class Ring:
    def __init__(self, q, d):
        q, d = int(q), int(d)
        assert int64_sum_fits(d, q, extra=1), "int64 headroom: d (q-1)^2 + (q-1) must be <= 2^63 - 1"
        self.q, self.d = q, d

    # ---- basic ----
    def zero(self, *shape):
        return np.zeros(shape + (self.d,), dtype=np.int64)

    def reduce(self, a):
        return np.asarray(a, dtype=np.int64) % self.q

    def from_signed(self, a):
        return np.asarray(a, dtype=np.int64) % self.q

    def centered(self, a):
        a = np.asarray(a, dtype=np.int64) % self.q
        return np.where(a > (self.q - 1) // 2, a - self.q, a)

    def inf_norm(self, a):
        """Infinity norm of centered representatives (4_P:86)."""
        a = self.centered(a)
        return int(np.max(np.abs(a))) if a.size else 0

    def scalar(self, c):
        """The constant ring element c."""
        out = self.zero()
        out[0] = int(c) % self.q
        return out

    # ---- products ----
    def polymul_full(self, a, b):
        """Product in Z_q[X] of two polynomials (degree < len each); no reduction by X^d + 1."""
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        assert int64_sum_fits(min(a.shape[-1], b.shape[-1]), self.q), "int64 headroom"
        return np.convolve(a, b) % self.q

    def negacyclic(self, full):
        """Reduce a polynomial of degree < 2d modulo X^d + 1."""
        d = self.d
        full = np.asarray(full, dtype=np.int64)
        out = full[:d].copy()
        hi = full[d:]
        out[:len(hi)] -= hi
        return out % self.q

    def mul(self, a, b):
        return self.negacyclic(self.polymul_full(a, b))

    def matvec(self, M, v):
        """M (rows, cols, d) times v (cols, d) -> (rows, d) in R_q."""
        M = np.asarray(M, dtype=np.int64)
        v = np.asarray(v, dtype=np.int64)
        rows, cols, d = M.shape
        assert v.shape == (cols, d)
        out = self.zero(rows)
        for i in range(rows):
            acc = np.zeros(2 * d - 1, dtype=np.int64)
            for j in range(cols):
                acc = (acc + np.convolve(M[i, j], v[j])) % self.q
            out[i] = self.negacyclic(acc)
        return out

    def matvec_full(self, M, v):
        """Like matvec but returns the products in Z_q[X] (degree <= 2d-2), padded to length 2d."""
        M = np.asarray(M, dtype=np.int64)
        v = np.asarray(v, dtype=np.int64)
        rows, cols, d = M.shape
        out = np.zeros((rows, 2 * d), dtype=np.int64)
        for i in range(rows):
            acc = np.zeros(2 * d - 1, dtype=np.int64)
            for j in range(cols):
                if not v[j].any() or not M[i, j].any():
                    continue
                acc = (acc + np.convolve(M[i, j], v[j])) % self.q
            out[i, :2 * d - 1] = acc
        return out

    def int_mul(self, a, b):
        """Exact product in Z[X]/(X^d + 1) of two integer polynomials (Python ints, no modulus)."""
        d = self.d
        out = [0] * d
        for i, x in enumerate(a):
            if x == 0:
                continue
            for j, y in enumerate(b):
                if y == 0:
                    continue
                e = i + j
                if e < d:
                    out[e] += x * y
                else:
                    out[e - d] -= x * y
        return out

    # ---- gadgets ----
    def window_top(self, b, delta):
        _, hi = digit_bounds(b)
        return hi * ((b ** delta - 1) // (b - 1))

    def g_inv(self, t, b, delta):
        """G^{-1}_{b,n}(t) for t of shape (n, d) (residues) -> (n*delta, d) digits stored as residues."""
        t = np.asarray(t, dtype=np.int64) % self.q
        assert b ** delta >= self.q
        top = self.window_top(b, delta)
        rep = np.where(t <= top, t, t - self.q)
        lo, _ = digit_bounds(b)
        n = t.shape[0]
        out = np.zeros((n, delta, self.d), dtype=np.int64)
        for e in range(delta):
            dig = (rep - lo) % b + lo
            out[:, e, :] = dig
            rep = (rep - dig) // b
        assert not rep.any(), "G^{-1}: representative outside the window"
        return out.reshape(n * delta, self.d) % self.q

    def g_apply(self, digits, b, delta):
        """G_{b,n} applied to (n*delta, d) residues -> (n, d) residues."""
        digits = np.asarray(digits, dtype=np.int64) % self.q
        assert int64_sum_fits(delta, self.q), "int64 headroom"
        n = digits.shape[0] // delta
        pw = np.array([pow(b, e, self.q) for e in range(delta)], dtype=np.int64)
        x = digits.reshape(n, delta, self.d)
        return np.einsum('ned,e->nd', x, pw) % self.q

    def j_inv(self, z, b, tau):
        """J^{-1}(z) for an integer array z of shape (n, d) (exact integers, not residues) -> (n*tau, d)
        digits as signed integers.  Raises ValueError if some entry is outside the tau-digit window."""
        z = np.asarray(z, dtype=np.int64)
        lo, hi = digit_bounds(b)
        S = (b ** tau - 1) // (b - 1)
        if z.size and (z.min() < lo * S or z.max() > hi * S):
            raise ValueError("J^{-1}: entry outside [%d, %d]" % (lo * S, hi * S))
        n = z.shape[0]
        out = np.zeros((n, tau, self.d), dtype=np.int64)
        rep = z.copy()
        for t in range(tau):
            dig = (rep - lo) % b + lo
            out[:, t, :] = dig
            rep = (rep - dig) // b
        assert not rep.any()
        return out.reshape(n * tau, self.d)

    def j_apply_int(self, digits, b, tau):
        """J applied over the integers to signed digits (n*tau, d) -> (n, d) integers."""
        digits = np.asarray(digits, dtype=np.int64)
        n = digits.shape[0] // tau
        pw = np.array([b ** t for t in range(tau)], dtype=np.int64)
        return np.einsum('ntd,t->nd', digits.reshape(n, tau, self.d), pw)


def int_digits(x, b, count):
    """Balanced base-b digits of the integer x (little-endian), or None if count digits do not suffice."""
    lo, _ = digit_bounds(b)
    out = []
    for _ in range(count):
        dig = (x - lo) % b + lo
        out.append(dig)
        x = (x - dig) // b
    return out if x == 0 else None
