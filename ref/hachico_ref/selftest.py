"""Unit self-tests of the oracle's arithmetic against independent references (Sage, brute force,
subfield.py).  Each test returns (name, ok, detail)."""
import random
from itertools import product

import numpy as np

from .context import SageRing
from .fqk import Fqk
from .params import TOY_TABLE, check_toy_table, digit_bounds, toy_preset
from .prg import Prg, l1ball_count, l1ball_sample
from .ring import Ring
from .subfield import Subfield, self_test as subfield_self_test
from .sumcheck import Blinder


def t_params():
    lines = check_toy_table()
    return [("params: toy table and q rule", True, "; ".join(lines))]


def t_subfield():
    out = []
    for name, row in TOY_TABLE.items():
        out.append(("subfield.self_test q=%d" % row[1], bool(subfield_self_test(row[1], 16, 4, trials=3)), ""))
    return out


def t_fqk(seed=1):
    out = []
    rnd = random.Random(seed)
    for name, row in TOY_TABLE.items():
        q = row[1]
        S = Subfield(q, 16, 4)
        F = Fqk(S)
        ok = F.self_check(trials=100, seed=seed)
        # vectorized axioms on random arrays
        X = np.array([[rnd.randrange(q) for _ in range(4)] for _ in range(64)], dtype=np.int64)
        Y = np.array([[rnd.randrange(q) for _ in range(4)] for _ in range(64)], dtype=np.int64)
        Z = np.array([[rnd.randrange(q) for _ in range(4)] for _ in range(64)], dtype=np.int64)
        ok &= np.array_equal(F.mul(F.mul(X, Y), Z), F.mul(X, F.mul(Y, Z)))
        ok &= np.array_equal(F.mul(X, F.add(Y, Z)), F.add(F.mul(X, Y), F.mul(X, Z)))
        ok &= np.array_equal(F.mul(X, Y), F.mul(Y, X))
        x = X[0]
        ok &= F.eq(F.mul(x, F.inv(x)), F.one())
        # Frobenius has order k: x^(q^k) = x and x^(q^2) != x for a generic x (k = 4)
        ok &= F.eq(F.pow(x, q ** 4), x) and not F.eq(F.pow(x, q ** 2), x)
        out.append(("fqk vs subfield.fmul, axioms, Frobenius (q=%d)" % q, bool(ok), "100 products + 64-array axioms"))
    return out


def t_ring(seed=2):
    out = []
    rnd = random.Random(seed)
    for name, row in TOY_TABLE.items():
        q = row[1]
        R = Ring(q, 16)
        SR = SageRing(q, 16)
        ok = True
        for trial in range(60):
            if trial == 0:
                a = [q - 1] * 16
                b = [q - 1] * 16
            elif trial == 1:
                a = [(q - 1) // 2] * 16
                b = [(q + 1) // 2] * 16
            else:
                a = [rnd.randrange(q) for _ in range(16)]
                b = [rnd.randrange(q) for _ in range(16)]
            got = R.mul(np.array(a), np.array(b)).tolist()
            ref = SR.coeffs(SR.poly(a) * SR.poly(b) % SR.mod_d, 16)
            ok &= got == ref
            full = R.polymul_full(np.array(a), np.array(b)).tolist()
            ok &= full == SR.coeffs(SR.poly(a) * SR.poly(b), 31)
        M = np.array([[[rnd.randrange(q) for _ in range(16)] for _ in range(7)] for _ in range(3)])
        v = np.array([[rnd.randrange(q) for _ in range(16)] for _ in range(7)])
        ok &= R.matvec(M, v).tolist() == SR.matvec(M.tolist(), v.tolist())
        out.append(("ring mul/full/matvec vs Sage (q=%d)" % q, bool(ok), "60 products incl. q-1 and centered extremes"))
    return out


def t_gadgets():
    out = []
    for name, row in TOY_TABLE.items():
        P = toy_preset(row[0])
        R = Ring(P.q, P.d)
        q, b, delta = P.q, P.b, P.delta
        lo, hi = digit_bounds(b)
        top = R.window_top(b, delta)
        S = (b ** delta - 1) // (b - 1)
        xs = [0, 1, 2, top - 1, top, top + 1, (q - 1) // 2, (q + 1) // 2, q - 2, q - 1]
        rng = random.Random(3)
        xs += [rng.randrange(q) for _ in range(6)]
        xs = xs[:16]
        t = np.array([xs], dtype=np.int64)
        dg = R.g_inv(t, b, delta)
        ok = np.array_equal(R.g_apply(dg, b, delta), t % q)
        c = R.centered(dg)
        ok &= int(c.min()) >= lo and int(c.max()) <= hi
        # the representative rule: recompose over Z and compare with x if x <= top else x - q
        pw = np.array([b ** e for e in range(delta)], dtype=object)
        recon = [int(sum(int(c[e, j]) * int(pw[e]) for e in range(delta))) for j in range(16)]
        ok &= recon == [x if x <= top else x - q for x in xs]
        ok &= lo * S <= min(recon) and max(recon) <= hi * S
        # J^{-1}: boundary passes, one beyond fails, at tau digits
        tau = P.tau
        St = (b ** tau - 1) // (b - 1)
        good = np.array([[hi * St, lo * St, P.beta, -P.beta, 0, 1, -1] + [0] * 9], dtype=np.int64)
        jd = R.j_inv(good, b, tau)
        ok &= np.array_equal(R.j_apply_int(jd, b, tau), good)
        ok &= int(jd.min()) >= lo and int(jd.max()) <= hi
        fails = 0
        for bad in (hi * St + 1, lo * St - 1):
            try:
                R.j_inv(np.array([[bad] + [0] * 15]), b, tau)
            except ValueError:
                fails += 1
        ok &= fails == 2
        ok &= hi * St >= P.beta and hi * ((b ** (tau - 1) - 1) // (b - 1)) < P.beta   # tau is tight
        out.append(("gadgets G/J round trips and boundaries (%s)" % name, bool(ok),
                    "top=%d, J window [%d, %d], beta=%d" % (top, lo * St, hi * St, P.beta)))
    return out


def t_l1ball(seed=4):
    out = []
    ok = True
    for d, w in [(1, 1), (3, 2), (4, 3), (5, 2), (6, 4), (16, 2)]:
        brute = sum(1 for c in product(range(-w, w + 1), repeat=d) if sum(abs(x) for x in c) <= w) if d <= 6 else None
        if brute is not None:
            ok &= brute == l1ball_count(d, w)
    ok &= l1ball_count(16, 2) == 545
    out.append(("l1ball count vs brute force", bool(ok), "(d, omega) in {(1,1),(3,2),(4,3),(5,2),(6,4)}; |C(16,2)| = 545"))
    # chi-square over the whole ball at (d, omega) = (4, 2)
    d, w = 4, 2
    ball = [c for c in product(range(-w, w + 1), repeat=d) if sum(abs(x) for x in c) <= w]
    idx = {c: i for i, c in enumerate(ball)}
    counts = [0] * len(ball)
    prg = Prg(b"selftest", "l1ball", seed)
    T = 41000
    for _ in range(T):
        counts[idx[tuple(l1ball_sample(prg, d, w))]] += 1
    exp = T / len(ball)
    chi2 = sum((c - exp) ** 2 / exp for c in counts)
    # 99.9% quantile of chi^2 with 40 degrees of freedom is 73.40
    out.append(("l1ball sampler chi^2 over C(4,2) (41 elements, 41000 draws)", bool(chi2 < 73.40),
                "chi2 = %.2f < 73.40 (0.999 quantile, 40 dof), seed %d" % (chi2, seed)))
    prg = Prg(b"selftest", "l1ball16", seed)
    ok = all(sum(abs(x) for x in l1ball_sample(prg, 16, 2)) <= 2 for _ in range(2000))
    out.append(("l1ball sampler stays in C(16,2)", bool(ok), "2000 draws"))
    return out


def t_samplers(seed=7):
    """chi^2 uniformity of the mask sampler [-gamma, gamma] and of S_b, and Rej's hypothesis check."""
    from .r2l import rej
    out = []
    prg = Prg(b"selftest", "boxed", seed)
    g, T = 7, 30000
    xs = prg.boxed(g, T)
    cnt = [0] * (2 * g + 1)
    for x in xs:
        cnt[x + g] += 1
    exp = T / (2 * g + 1)
    chi2 = sum((c - exp) ** 2 / exp for c in cnt)
    # 99.9% quantile of chi^2 with 14 dof is 36.12
    out.append(("mask sampler chi^2 on [-7, 7] (30000 draws)", bool(chi2 < 36.12 and min(xs) == -g and max(xs) == g),
                "chi2 = %.2f < 36.12 (0.999 quantile, 14 dof)" % chi2))
    prg = Prg(b"selftest", "short", seed)
    xs = prg.short(4, 20000)
    cnt = {v: xs.count(v) for v in range(-2, 2)}
    exp = 20000 / 4
    chi2 = sum((c - exp) ** 2 / exp for c in cnt.values())
    # 99.9% quantile of chi^2 with 3 dof is 16.27
    out.append(("S_4 sampler chi^2 on [-2, 1] (20000 draws)", bool(chi2 < 16.27 and set(xs) == {-2, -1, 0, 1}),
                "chi2 = %.2f < 16.27 (0.999 quantile, 3 dof)" % chi2))
    raised = False
    try:
        rej(np.array([3, 0]), np.array([0, 0]), 10, 2)
    except ValueError:
        raised = True
    ok = raised and rej(np.array([2, 0]), np.array([6, 0]), 10, 2) is not None \
        and rej(np.array([2, 0]), np.array([7, 0]), 10, 2) is None
    out.append(("Rej: raises for ||v|| > beta1; accepts iff ||z|| <= gamma - beta1", bool(ok), ""))
    return out


def t_multilinear(seed=5):
    rnd = random.Random(seed)
    q = TOY_TABLE["toy-n1"][1]
    S = Subfield(q, 16, 4)
    F = Fqk(S)
    rf = lambda: np.array([rnd.randrange(q) for _ in range(4)], dtype=np.int64)
    s = 5
    tau = np.array([rf() for _ in range(s)])
    tab = F.eq_table(tau)
    ok = True
    for x in [0, 1, 7, 19, 31]:
        bits = [(x >> (s - 1 - j)) & 1 for j in range(s)]
        ok &= F.eq(tab[x], F.eq_point(tau, bits))
    T = np.array([rf() for _ in range(2 ** s)])
    pt = np.array([rf() for _ in range(s)])
    ok &= F.eq(F.mle_eval(T, pt), F.sum(F.mul(F.eq_table(pt), T), axis=0))
    for x in [0, 13, 31]:
        bits = np.array([F.from_int((x >> (s - 1 - j)) & 1) for j in range(s)])
        ok &= F.eq(F.mle_eval(T, bits), T[x])
    # Lagrange at nodes 1..D+1
    D = 6
    coeffs = [rf() for _ in range(D + 1)]
    ev = lambda X: F.sum(F.mul(np.array(coeffs), F.powers(X, D + 1)), axis=0)
    vals = np.array([ev(F.from_int(t)) for t in range(1, D + 2)])
    x = rf()
    ok &= F.eq(F.lagrange_eval(vals, x), ev(x)) and F.eq(F.lagrange_eval(vals, F.zeros()), ev(F.zeros()))
    return [("eq table, MLE, Lagrange interpolation", bool(ok), "s = 5, D = 6")]


class _MiniCtx:
    pass


def t_blinder(seed=6):
    """Blinder closed forms (hypercube sum, round polynomial, evaluation) against brute force at l'' = 4."""
    rnd = random.Random(seed)
    q = TOY_TABLE["toy-n1"][1]
    S = Subfield(q, 16, 4)
    F = Fqk(S)
    L, D = 4, 3
    g = Blinder.__new__(Blinder)
    g.F, g.D, g.L = F, D, L
    rf = lambda: np.array([rnd.randrange(q) for _ in range(4)], dtype=np.int64)
    g.s0 = rf()
    g.s = np.array([[rf() for _ in range(D)] for _ in range(L)])

    def ev(point):
        acc = g.s0
        for j in range(L):
            for k in range(D):
                acc = F.add(acc, F.mul(g.s[j, k], F.pow(point[j], k + 1)))
        return acc

    ok = True
    tot = F.zeros()
    for v in product([0, 1], repeat=L):
        tot = F.add(tot, ev([F.from_int(b) for b in v]))
    ok &= F.eq(tot, g.hypercube_sum())
    a = [rf() for _ in range(L)]
    for r in range(1, L + 1):
        x = rf()
        brute = F.zeros()
        for v in product([0, 1], repeat=L - r):
            brute = F.add(brute, ev(a[:r - 1] + [x] + [F.from_int(b) for b in v]))
        ok &= F.eq(brute, g.round_poly(r, a[:r - 1], x))
    ok &= F.eq(ev(a), g.evaluate(np.array(a)))
    return [("blinder sum / round polynomial / evaluation vs brute force", bool(ok), "l'' = 4, D = 3")]


def run_all():
    res = []
    for fn in (t_params, t_subfield, t_fqk, t_ring, t_gadgets, t_l1ball, t_samplers, t_multilinear, t_blinder):
        res.extend(fn())
    return res
