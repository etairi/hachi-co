#!/usr/bin/env python3
"""Checks the shared instance `masked_spec.json` against `hachi-pcs.tex`, Appendix A.2.

The unit test `pcs::spec_fixture::the_masked_spec_fixture_is_current` of the `hachi` crate writes the
fixture: one masked opening of a polynomial over the field, of a vector and of a merged commitment,
at the toy choices with masks, with the prover's secrets, its Fiat-Shamir challenges, step 1 and the
masked level 0. That test replays each opening and requires the replayed first iteration to equal
the proof's, so these are the values of real openings.

This script recomputes, with Sage's own arithmetic and from the definitions of the appendix (the
field F_{q^k} as R_q^H, its basis Z_i, psi, sigma_t, Tr_H, E and the gadget G), every equation
those values must satisfy:
- Commit: G s_i = (f'_i, masking row), digits in S_b, t_i = A s_i + [A'_1 | I] rho_i, G t_hat = t,
  u = B t_hat, and the message laid out as psi of the coefficients.
- Step 1: the weights b, a, v from the point, y = f(x), y_m = E(Y_m), Y* = Y + x* Y_m, the
  verifier's check E(Y*) = y + x* y_m, and a* = (a, x*) (with lambda for a merged opening).
- Level 0: w_i = a*^T G s_i, v = D (w_hat, w0_hat, t0_hat), the masks s_0 = z - sum c_i s_i and
  rho_0 = z_rho - sum c_i rho_i over Z, w_0 = a*^T G s_0, t_0 = A s_0 + A' rho_0, the bound beta,
  the challenges, and the widths of the masks against sigma.

Indices: the Rust code orders a table with the first variable as the most significant bit, and the
appendix (nat, bits) as the least significant one. The layout and the weights are checked in the
Rust order, and the script also reports whether they equal the appendix's order exactly or up to
reversing the bits of each index (chunk, element, position in psi).

Run: sage -python crates/hachi/tests/fixtures/check_masked_spec.py
It prints one line per check and exits with status 1 if any check fails.
"""
import json
import statistics
import sys
from pathlib import Path

from sage.all import GF, ZZ, PolynomialRing

FAILS = []


def check(cond, name):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


def bitrev(i, n):
    return int("".join(reversed(format(i, f"0{n}b"))), 2) if n else 0


class Field:
    """F_{q^k} as the span of Z_0 = 1 and Z_i = Y^i - Y^{2k - i} in Z_q[Y]/(Y^{2k} + 1)."""

    def __init__(self, q, k):
        self.q, self.k = q, k
        self.base = GF(q)
        P = PolynomialRing(self.base, "Y")
        self.P = P
        self.S = P.quotient(P.gen() ** (2 * k) + 1, "y")
        y = self.S.gen()
        self.Z = [self.S(1)] + [y ** i - y ** (2 * k - i) for i in range(1, k)]
        self.zero, self.one = self.S(0), self.S(1)

    def of(self, coords):
        """A field element from its coordinates in the basis Z_i (the Rust `Ext`)."""
        return sum((self.base(c) * z for c, z in zip(coords, self.Z)), self.zero)

    def coeffs(self, a):
        c = list(a.lift())
        return c + [self.base(0)] * (2 * self.k - len(c))


class Ring:
    """R_q = Z_q[X]/(X^d + 1), with F embedded as R_q^H by Y = X^{d/2k}."""

    def __init__(self, F, d):
        self.F, self.d, self.q, self.k = F, d, F.q, F.k
        self.PX = PolynomialRing(F.base, "X")
        X = self.PX.gen()
        self.R = self.PX.quotient(X ** d + 1, "x")
        self.x = self.R.gen()
        self.step = d // (2 * self.k)
        k = self.k
        self.H = sorted(set([(4 * k * a + 1) % (2 * d) for a in range(d // (2 * k))]
                            + [(-(4 * k * a + 1)) % (2 * d) for a in range(d // (2 * k))]))
        assert len(self.H) == d // k
        self.ZX = PolynomialRing(ZZ, "X")

    def cf(self, a):
        c = list(a.lift())
        return c + [self.F.base(0)] * (self.d - len(c))

    def of(self, residues):
        return self.R(self.PX([self.F.base(c) for c in residues]))

    def embed(self, f):
        c = self.F.coeffs(f)
        out = [self.F.base(0)] * self.d
        for i, v in enumerate(c):
            out[i * self.step] = v
        return self.R(self.PX(out))

    def to_field(self, a):
        c = self.cf(a)
        if any(c[i] != 0 for i in range(self.d) if i % self.step):
            return None
        return self.F.S(self.F.P([c[i * self.step] for i in range(2 * self.k)]))

    def sigma(self, a, t):
        c = self.cf(a)
        out = [self.F.base(0)] * self.d
        for i, v in enumerate(c):
            e = (i * t) % (2 * self.d)
            if e >= self.d:
                out[e - self.d] -= v
            else:
                out[e] += v
        return self.R(self.PX(out))

    def tr(self, a):
        return sum((self.sigma(a, t) for t in self.H), self.R(0))

    def E(self, Y, v):
        """(k/d) Tr_H(Y sigma_{-1}(v)), as a field element (Theorem 2)."""
        ratio = self.F.base(self.k) / self.F.base(self.d)
        return self.to_field(ratio * self.tr(Y * self.sigma(v, 2 * self.d - 1)))

    def psi(self, vec):
        """psi: F^{d/k} -> R_q, the low half then the high half."""
        h = self.d // (2 * self.k)
        assert len(vec) == self.d // self.k
        lo = sum((vec[i] * self.x ** i for i in range(h)), self.R(0))
        hi = sum((vec[h + i] * self.x ** i for i in range(h)), self.R(0))
        return lo + self.x ** (self.d // 2) * hi

    def centred(self, a):
        q = self.q
        return [int(c) if int(c) <= (q - 1) // 2 else int(c) - q for c in self.cf(a)]

    def lift(self, a):
        """A ring element with small centred coefficients, in Z[X]/(X^d + 1) (as a list)."""
        return self.centred(a)

    def mul_int(self, a, b):
        """The product of two integer coefficient lists in Z[X]/(X^d + 1)."""
        d = self.d
        out = [0] * d
        for i, x in enumerate(a):
            if x == 0:
                continue
            for j, y in enumerate(b):
                e = i + j
                if e >= d:
                    out[e - d] -= x * y
                else:
                    out[e] += x * y
        return out


def gadget(R, digits, b, delta):
    """G: delta digits per element, least significant first, element after element."""
    return [sum((digits[i + e] * b ** e for e in range(delta)), R.R(0))
            for i in range(0, len(digits), delta)]


def digits_in(R, polys, lo, hi):
    return all(lo <= c <= hi for p in polys for c in R.centred(p))


def balanced(v, b, n):
    """The appendix's G^{-1} on an integer: digits in [-floor(b/2), ceil(b/2) - 1]."""
    lo = -(b // 2)
    out = []
    for _ in range(n):
        x = ((v - lo) % b) + lo
        out.append(x)
        v = (v - x) // b
    return out if v == 0 else None


def g_inv_spec(R, polys, b, delta, q):
    """The appendix's G^{-1}: per coefficient the representative of v or v - q that has digits."""
    out = []
    for p in polys:
        per = []
        for c in R.cf(p):
            ds = balanced(int(c), b, delta) or balanced(int(c) - q, b, delta)
            if ds is None:
                return None
            per.append(ds)
        for e in range(delta):
            out.append(R.of([per[i][e] % q for i in range(R.d)]))
    return out


def inner(a, v, zero):
    return sum((x * y for x, y in zip(a, v)), zero)


def mat_vec(M, v, zero):
    return [inner(row, v, zero) for row in M]


def monomial_list(F, xs, msb_first):
    """The 2^n monomials of xs: entry i is prod_t x_t^{bit_t(i)}, x_1 the most (or least)
    significant bit."""
    n = len(xs)
    out = []
    for i in range(2 ** n):
        m = F.one
        for t in range(n):
            bit = (i >> (n - 1 - t)) & 1 if msb_first else (i >> t) & 1
            if bit:
                m = m * xs[t]
        out.append(m)
    return out


def evaluate(F, coeffs, xs):
    """f(x) for the Rust coefficient order: coefficient i multiplies prod_t X_t^{bit_t(i)}, X_1
    the most significant bit."""
    return inner(coeffs, monomial_list(F, xs, True), F.zero)


def run(inst):
    kind = inst["kind"]
    q, d, k, b, delta = inst["q"], inst["d"], inst["k"], inst["b"], inst["delta"]
    n_a, nu, r, m, extra = inst["n_a"], inst["nu"], inst["r"], inst["m"], inst["extra"]
    F = Field(q, k)
    R = Ring(F, d)
    P = lambda c: R.of(c)
    Ps = lambda cs: [P(c) for c in cs]
    E_ = lambda c: F.of(c)
    Es = lambda cs: [E_(c) for c in cs]
    zero = R.R(0)
    name = lambda s: f"{kind}: {s}"
    lo, hi = -(b // 2), (b + 1) // 2 - 1
    chunks = [Ps(ch) for ch in inst["chunks"]]
    msg = 2 ** m
    chunk_len = msg + extra + 1
    check(len(chunks) == 2 ** r and all(len(ch) == chunk_len for ch in chunks),
          name(f"{2 ** r} chunks of 2^m + extra + 1 = {chunk_len} ring elements"))

    # Commit.
    s = [Ps(x) for x in inst["s"]]
    rho = [Ps(x) for x in inst["rho"]]
    t = [Ps(x) for x in inst["t"]]
    t_hat = Ps(inst["t_hat"])
    u = Ps(inst["u"])
    A, A1p, B, D = ([Ps(row) for row in inst[key]] for key in ("A", "A1p", "B", "D"))
    check(all(gadget(R, s[i], b, delta) == chunks[i] for i in range(2 ** r)),
          name("G s_i is the chunk, its masking row last"))
    check(all(digits_in(R, x, lo, hi) for x in s), name(f"the digits of s_i lie in [{lo}, {hi}]"))
    spec = [g_inv_spec(R, ch, b, delta, q) for ch in chunks]
    check(spec == s, name("s_i is the appendix's G^{-1} of the chunk"))
    check(all(len(x) == nu + n_a for x in rho) and all(digits_in(R, x, lo, hi) for x in rho),
          name("rho_i has nu + n_A ring elements of digits in S_b"))
    Ap = lambda v: [x + y for x, y in zip(mat_vec(A1p, v[:nu], zero), v[nu:])]
    check(all(t[i] == [x + y for x, y in zip(mat_vec(A, s[i], zero), Ap(rho[i]))] for i in range(2 ** r)),
          name("t_i = A s_i + [A'_1 | I] rho_i"))
    check(gadget(R, t_hat, b, delta) == [x for ti in t for x in ti] and digits_in(R, t_hat, lo, hi),
          name("G t_hat = t, chunk after chunk, with digits"))
    check(mat_vec(B, t_hat, zero) == u, name("u = B t_hat"))

    # The value, the layout and the weights.
    ak = 1
    while 2 ** ak < d // k:
        ak += 1
    ratio_one = R.R(1)
    claim_b, claim_a = Ps(inst["claim_b"]), Ps(inst["claim_a"])
    target = P(inst["claim_target"])
    x_star = E_(inst["x_star"])
    check(x_star != 0, name("x* is not zero"))
    if kind in ("multilinear", "merged"):
        vars_ = inst["vars"]
        coeffs = Es(inst["coeffs"])
        xs = Es(inst["point"])
        y = evaluate(F, coeffs, xs)
        value = E_(inst["value"]) if kind == "multilinear" else Es(inst["values"])[0]
        check(value == y, name("the value is f(x)"))
        n_pos = d // k

        def layout_as(order):
            """Chunk i, element j, psi position p of the coefficient with these index fields."""
            ok = True
            for i in range(2 ** r):
                for j in range(msg):
                    cells = []
                    for p in range(n_pos):
                        if order == "rust":
                            ii, jj, pp = i, j, p
                        else:
                            ii, jj, pp = bitrev(i, r), bitrev(j, m), bitrev(p, ak)
                        cells.append(R.embed(coeffs[(ii << (m + ak)) | (jj << ak) | pp]))
                    ok = ok and chunks[i][j] == R.psi(cells)
            return ok

        rust_layout = layout_as("rust")
        check(rust_layout, name("the message is psi of d/k consecutive coefficients (Rust order)"))
        exact = layout_as("spec")
        print(f"INFO {kind}: the layout equals the appendix's (nat, bits) order "
              + ("exactly" if exact else
                 "up to reversing the bits of the chunk, element and psi indices" if rust_layout else "in neither form"))
        bv = [R.embed(mm) for mm in monomial_list(F, xs[:r], True)]
        av = [R.embed(mm) for mm in monomial_list(F, xs[r:r + m], True)]
        v = R.psi([R.embed(mm) for mm in monomial_list(F, xs[r + m:], True)])
        check(P(inst["v_trace"]) == v, name("v = psi of the monomials of the last alpha - kappa coordinates"))
        check(claim_b == bv, name("b: the monomials of the first r coordinates"))
        Y = inner(bv, [inner(av, ch[:msg], zero) for ch in chunks], zero)
        check(R.E(Y, v) == value, name("E(Y) = y for Y = sum_i b_i a^T f_i"))
    if kind == "multilinear":
        Ym = inner(bv, [ch[-1] for ch in chunks], zero)
        y_m = E_(inst["y_mask"])
        check(R.E(Ym, v) == y_m, name("y_m = E(Y_m) for Y_m = sum_i b_i m_i"))
        check(target == Y + R.embed(x_star) * Ym, name("Y* = Y + x* Y_m"))
        check(R.E(target, v) == value + x_star * y_m, name("the verifier's check E(Y*) = y + x* y_m"))
        check(claim_a == av + [R.embed(x_star)], name("a* = (a, x*)"))
    if kind == "vector":
        values, weights = Es(inst["values"]), Es(inst["weights"])
        value = E_(inst["value"])
        check(value == inner(values, weights, F.zero), name("the value is <s, e>"))
        n_pos = d // k
        pad = lambda xs: xs + [F.zero] * (-len(xs) % n_pos)
        blocks = lambda xs: [R.psi([R.embed(x) for x in pad(xs)[i:i + n_pos]])
                             for i in range(0, len(pad(xs)), n_pos)]
        sb = blocks(values)
        check(chunks[0][:len(sb)] == sb and all(e == zero for e in chunks[0][len(sb):msg]),
              name("the chunk is psi of blocks of d/k entries, zero-padded"))
        aw = [R.sigma(e, 2 * d - 1) for e in blocks(weights)]
        aw = aw + [zero] * (msg - len(aw))
        check(claim_b == [ratio_one], name("b = (1)"))
        check(claim_a == aw + [R.embed(x_star)], name("a* = (sigma_{-1}(psi(blocks of e)), x*)"))
        Y = inner(aw, chunks[0][:msg], zero)
        check(R.E(Y, ratio_one) == value, name("E(Y) = <s, e> with v = 1"))
        Ym = chunks[0][-1]
        y_m = E_(inst["y_mask"])
        check(R.E(Ym, ratio_one) == y_m, name("y_m = E(m) with v = 1"))
        check(target == Y + R.embed(x_star) * Ym, name("Y* = Y + x* m"))
        check(R.E(target, ratio_one) == value + x_star * y_m, name("the verifier's check E(Y*) = y + x* y_m"))
    if kind == "merged":
        values = Es(inst["values"])
        g = [Es(inst["g0"]), Es(inst["g1"])]
        e = [Es(inst["e0"]), Es(inst["e1"])]
        lam = E_(inst["lambda"])
        check(lam != 0, name("lambda is not zero"))
        n_pos = d // k
        pad = lambda xs: xs + [F.zero] * (-len(xs) % n_pos)
        blocks = lambda xs: [R.psi([R.embed(x) for x in pad(xs)[i:i + n_pos]])
                             for i in range(0, len(pad(xs)), n_pos)]
        at = msg
        ys, rows, ws = [], [], []
        ok_rows = True
        for gv, ev, val in zip(g, e, values[1:]):
            bl = blocks(gv)
            ok_rows = ok_rows and chunks[0][at:at + len(bl)] == bl
            w = [R.sigma(x, 2 * d - 1) for x in blocks(ev)]
            ys.append(inner(w, chunks[0][at:at + len(bl)], zero))
            rows.append(chunks[0][at + len(bl)])
            ws.append(w)
            check(val == inner(gv, ev, F.zero), name("a vector's value is <g, e>"))
            at += len(bl) + 1
        check(ok_rows and at == msg + extra, name("chunk 0's extra rows: each vector's blocks, then its masking row"))
        check(all(x == zero for ch in chunks[1:] for x in ch[msg:msg + extra]), name("the other chunks hold zeros in the extra rows"))
        Ym = inner(bv, [ch[-1] for ch in chunks], zero)
        y_masks = Es(inst["y_masks"])
        check([R.E(Ym, v), R.E(rows[0], ratio_one), R.E(rows[1], ratio_one)] == y_masks,
              name("the three y_m: E(Y_m) with v, and E of each vector's masking row with v = 1"))
        X = R.embed(x_star)
        stars = Ps(inst["Y_stars"])
        check(stars == [Y + X * Ym, ys[0] + X * rows[0], ys[1] + X * rows[1]], name("the three Y* = Y + x* Y_m"))
        check([R.E(stars[0], v), R.E(stars[1], ratio_one), R.E(stars[2], ratio_one)]
              == [values[i] + x_star * y_masks[i] for i in range(3)], name("the three trace checks"))
        L = R.embed(lam)
        L2 = R.embed(lam * lam)
        want = av + [L * x for x in ws[0]] + [R.embed(lam * x_star)] + [L2 * x for x in ws[1]] \
            + [R.embed(lam * lam * x_star)] + [X]
        check(claim_a == want, name("a* = (a, lambda w_0, lambda x*, lambda^2 w_1, lambda^2 x*, x*)"))
        check(target == stars[0] + L * stars[1] + L2 * stars[2], name("the target Y*_w + lambda Y*_0 + lambda^2 Y*_1"))

    # Level 0.
    w = Ps(inst["w"])
    w_hat = Ps(inst["w_hat"])
    v_commit = Ps(inst["v"])
    c = Ps(inst["c"])
    z, z_rho = Ps(inst["z"]), Ps(inst["z_rho"])
    w0, w0_hat = P(inst["w0"]), Ps(inst["w0_hat"])
    t0, t0_hat = Ps(inst["t0"]), Ps(inst["t0_hat"])
    beta, sigma, omega = inst["beta"], inst["sigma"], inst["omega"]
    check(w == [inner(claim_a, ch, zero) for ch in chunks], name("w_i = a*^T G s_i"))
    check(inner(claim_b, w, zero) == target, name("the first claim holds: sum_i b_i w_i = target"))
    check(gadget(R, w_hat, b, delta) == w and digits_in(R, w_hat, lo, hi), name("G w_hat = w, with digits"))
    check(gadget(R, w0_hat, b, delta) == [w0] and digits_in(R, w0_hat, lo, hi), name("G w0_hat = w_0, with digits"))
    check(gadget(R, t0_hat, b, delta) == t0 and digits_in(R, t0_hat, lo, hi), name("G t0_hat = t_0, with digits"))
    check(mat_vec(D, w_hat + w0_hat + t0_hat, zero) == v_commit, name("v = D (w_hat, w0_hat, t0_hat)"))
    l1 = [sum(abs(x) for x in R.centred(ci)) for ci in c]
    check(len(c) == 2 ** r and all(x <= omega for x in l1),
          name(f"2^r challenges in the l1 ball of radius omega = {omega} (norms {sorted(set(l1))})"))
    # The masks over Z: s_0 = z - sum_i c_i s_i and rho_0 = z_rho - sum_i c_i rho_i.
    ci = [R.lift(x) for x in c]

    def masks(fold, parts):
        out = []
        for pos in range(len(fold)):
            acc = R.lift(fold[pos])
            for i in range(2 ** r):
                prod = R.mul_int(ci[i], R.lift(parts[i][pos]))
                acc = [x - y for x, y in zip(acc, prod)]
            out.append(acc)
        return out

    s0 = masks(z, s)
    rho0 = masks(z_rho, rho)
    s0_ring = [R.of([x % q for x in a]) for a in s0]
    rho0_ring = [R.of([x % q for x in a]) for a in rho0]
    check(w0 == inner(claim_a, gadget(R, s0_ring, b, delta), zero), name("w_0 = a*^T G s_0"))
    check(t0 == [x + y for x, y in zip(mat_vec(A, s0_ring, zero), Ap(rho0_ring))], name("t_0 = A s_0 + [A'_1 | I] rho_0"))
    zmax = max(max(abs(x) for x in R.centred(p)) for p in z + z_rho)
    check(zmax <= beta, name(f"||(z, z_rho)||_inf = {zmax} <= beta = {beta}"))
    samples = [x for a in s0 + rho0 for x in a]
    sd = statistics.pstdev(samples)
    check(abs(sd / sigma - 1) < 5 / (2 * len(samples)) ** 0.5 + 0.02,
          name(f"the masks' {len(samples)} coefficients have standard deviation {sd:.0f}, sigma = {sigma}"))
    check(max(abs(x) for x in samples) < 16.3 * sigma, name("the masks lie within the sampler's support"))
    # The verifier's identities, which follow from the above.
    lhs = [x + y for x, y in zip(mat_vec(A, z, zero), Ap(z_rho))]
    rhs = [sum((c[i] * t[i][row] for i in range(2 ** r)), zero) + t0[row] for row in range(n_a)]
    check(lhs == rhs, name("A z + A' z_rho = sum_i c_i t_i + t_0"))
    check(inner(claim_a, gadget(R, z, b, delta), zero) == inner(c, w, zero) + w0,
          name("a*^T G z = sum_i c_i w_i + w_0"))


def main():
    path = Path(__file__).resolve().parent / "masked_spec.json"
    data = json.loads(path.read_text())
    for inst in data["instances"]:
        run(inst)
    print(f"{len(FAILS)} failed" if FAILS else "all checks pass")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
