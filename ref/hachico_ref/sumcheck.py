"""The collaborative sumcheck (5_BB:195-335): quotient shares, share polynomials, blinders, F_0 / F_1,
rounds through F_AH(I), and the verifier.

Conventions (see ref/README.md for the discussion):
  * Hypercube {0,1}^{l''} with V = (U, L) = (V_1, ..., V_{l''}), U the first ceil(log(mu+n')) variables and
    L the last log(2d) (5_BB:207).  A point x has index nat(x) = sum_j x_j 2^{l''-j} (V_1 most significant,
    m0-spec), so the draft's chi_{bit(u)||bit(l)} (1-indexed u, l) is the table entry (u-1)*2d + (l-1).
  * Round r binds V_r, i.e. the most significant remaining index bit (T'[x] = T[x] + a_r (T[x+h] - T[x])).
  * eq(tau_0, (U, L, bit(i))): tau_0[0:l''] pairs with (U, L), tau_0[l'':] with the ceil(log n) bits of
    prover i - 1 (MSB first).  eq(tau_1, bit(i')): i' in [nu] is the table entry i' - 1 of eq_table(tau_1).
  * Round messages are the values at the nodes 1..D+1 (D = d_0 resp. d_1), as in 5_BB:290-294; the verifier
    gets H(0) by Lagrange interpolation.
  * S_i are the 0-indexed positions held by prover i (r2l.Layout.partition); positions of all-zero blocks
    owned by nobody (n = 3) are in no S_i, so F_{0,i} forces every share there to be zero.
"""
from collections import OrderedDict

import numpy as np

from .prg import Prg


# --------------------------------------------------------------- quotient ----
def quotient_shares(ctx, M, z_shares, y_shares):
    """<r>_p = 1/2 (1 - X^d) (M <z'>_p - {y}_p) mod X^{2d} + 1 (5_BB:198-200), shape (n, n', 2d).

    With P = M<z'>_p - {y}_p of degree < 2d split as P_lo + X^d P_hi:
    1/2 (1 - X^d) P = 1/2 (P_lo + P_hi) + X^d 1/2 (P_hi - P_lo)  (mod X^{2d} + 1)."""
    P, R = ctx.P, ctx.R
    q, d = P.q, P.d
    inv2 = (q + 1) // 2
    out = np.zeros((P.n, P.n_prime, 2 * d), dtype=np.int64)
    for p in range(P.n):
        full = R.matvec_full(M, z_shares[p])                 # (n', 2d), degree <= 2d - 2
        full[:, :d] = (full[:, :d] - y_shares[p]) % q
        lo, hi = full[:, :d], full[:, d:]
        out[p, :, :d] = (lo + hi) * inv2 % q
        out[p, :, d:] = (hi - lo) * inv2 % q
    return out


def check_quotient(ctx, M, z_prime, y_vec, r_sum):
    """Sage check: (X^d + 1) r = M z' - y over Z_q[X] with deg r < d, and the shares' high halves cancel."""
    SR, P = ctx.SR, ctx.P
    d = P.d
    ok_hi = not np.asarray(r_sum)[:, d:].any()
    ok_rel = True
    for i in range(P.n_prime):
        acc = SR.PR(0)
        for j in range(P.mu):
            if z_prime[j].any() and M[i, j].any():
                acc += SR.poly(M[i, j]) * SR.poly(z_prime[j])
        acc -= SR.poly(y_vec[i])
        ok_rel &= (acc == (SR.X ** d + 1) * SR.poly(r_sum[i, :d]))
    return bool(ok_hi), bool(ok_rel)


def check_quotient_formula(ctx, M, z_share, y_share, r_share):
    """Sage check of one share against the literal product 1/2 (1 - X^d) (...) in Z_q[X]/(X^{2d} + 1)."""
    SR, P = ctx.SR, ctx.P
    d = P.d
    inv2 = (P.q + 1) // 2
    ok = True
    for i in range(P.n_prime):
        acc = SR.PR(0)
        for j in range(P.mu):
            if z_share[j].any() and M[i, j].any():
                acc += SR.poly(M[i, j]) * SR.poly(z_share[j])
        acc -= SR.poly(y_share[i])
        val = (inv2 * (1 - SR.X ** d) * acc) % SR.mod_2d
        ok &= (SR.coeffs(val, 2 * d) == [int(c) for c in r_share[i]])
    return bool(ok)


# ----------------------------------------------------------------- tables ----
def w_table(ctx, z_share, r_share):
    """Hypercube values of <w~>_p (5_BB:225-229), shape (2^{l''},) over Z_q:
    entry (u0, l0) = z'_share[u0] coefficient l0 for u0 < mu, l0 < d; <r>_p[u0 - mu] coefficient l0 for
    mu <= u0 < mu + n', l0 < 2d; 0 otherwise."""
    P = ctx.P
    T = np.zeros((2 ** P.ell_U, 2 * P.d), dtype=np.int64)
    T[:P.mu, :P.d] = z_share
    T[P.mu:P.mu + P.n_prime, :] = r_share
    return T.reshape(-1)


def row_indicator(ctx, rows):
    """0/1 table over the hypercube that is 1 on the U-rows `rows` (0-indexed), for every L."""
    P = ctx.P
    T = np.zeros((2 ** P.ell_U, 2 * P.d), dtype=np.int64)
    T[list(rows), :] = 1
    return T.reshape(-1)


def ind_rows(ctx):
    """The draft's sum_{u=mu}^{mu+n'} chi_bit(u)(U): 1-indexed mu..mu+n' = 0-indexed mu-1..mu+n'-1."""
    P = ctx.P
    return range(P.mu - 1, P.mu + P.n_prime)


def bits_of(x, s):
    return [(x >> (s - 1 - j)) & 1 for j in range(s)]


def mstar_table(ctx, M, E_mat, eq1, alpha):
    """Hypercube values of M*(U, L) = sum_{i' in [nu]} eq(tau_1, bit(i')) M~_alpha(bit(i'), U, L)
    (5_BB:254-260), shape (2^{l''}, k).  This is the provers' construction; the verifier uses
    `mstar_eval_factorized` instead."""
    P, F = ctx.P, ctx.F
    d, mu, npr = P.d, P.mu, P.n_prime
    q = P.q
    apow = F.powers(alpha, 2 * d)                                     # alpha^0 .. alpha^{2d-1}
    M_alpha = np.einsum('iuj,jc->iuc', M, apow[:d]) % q               # M_{i',u}(alpha), (n', mu, k)
    c_u = F.sum(F.mul(eq1[:npr][:, None, :], M_alpha), axis=0)        # (mu, k)
    T = F.zeros(2 ** P.ell_U, 2 * d)
    # first term, u <= mu (all l in [2d])
    T[:mu] = F.mul(c_u[:, None, :], apow[None, :, :])
    # first term, u = i' + mu: -(alpha^d + 1) alpha^{l-1}
    ad1 = F.add(F.pow(alpha, d), F.one())
    for i0 in range(npr):
        T[mu + i0] = F.neg(F.mul(F.mul(eq1[i0], ad1)[None, :], apow))
    # second term: rows n'+1..n'+d, u = mu, l in [d]: E_{i'-n', l}
    acc = F.zeros(d)
    for j0 in range(d):
        acc = F.add(acc, F.smul(E_mat[j0], np.broadcast_to(eq1[npr + j0], (d, F.k))))
    T[mu - 1, :d] = F.add(T[mu - 1, :d], acc)
    # third term: u in [mu+n'], l in d+1..2d, i' = n' + d + u + (l - d - 1)(mu + n')
    u0 = np.arange(mu + npr)[:, None]
    l0 = np.arange(d, 2 * d)[None, :]
    idx = npr + d + u0 + (l0 - d) * (mu + npr)
    assert idx.max() == P.nu - 1
    T[:mu + npr, d:] = F.add(T[:mu + npr, d:], eq1[idx])
    return T.reshape(-1, F.k)


def claimed_sum(ctx, eq1, y_vec, y_eval_ring, alpha):
    """a = sum_{i' in [n']} eq(tau_1, bit(i')) y_{i'}(alpha) + sum_{i'=n'+1}^{n'+d} eq(tau_1, bit(i')) y_{i'-n'-1}
    (5_BB:263-264)."""
    P, F = ctx.P, ctx.F
    apow = F.powers(alpha, P.d)
    y_alpha = np.einsum('ij,jc->ic', y_vec, apow) % P.q
    a = F.sum(F.mul(eq1[:P.n_prime], y_alpha), axis=0)
    a = F.add(a, F.sum(F.smul(np.asarray(y_eval_ring), eq1[P.n_prime:P.n_prime + P.d]), axis=0))
    return a


# --------------------------------------------------------------- blinders ----
class Blinder:
    """g(V) = s_0 + sum_{j in [l'']} sum_{k in [D]} s_{j,k} V_j^k with uniform coefficients (5_BB:209-214)."""

    def __init__(self, ctx, domain, D):
        P, F = ctx.P, ctx.F
        self.F, self.D, self.L = F, D, P.ell_pp
        prg = Prg(ctx.seed, domain)
        vals = np.array([prg.mod_q(F.q, F.k) for _ in range(1 + self.L * D)], dtype=np.int64)
        self.s0 = vals[0]
        self.s = vals[1:].reshape(self.L, D, F.k)      # s[j-1, k-1] = s_{j,k}

    def hypercube_sum(self):
        """G = sum_{v in {0,1}^{l''}} g(v) = 2^{l''} s_0 + 2^{l''-1} sum_{j,k} s_{j,k}."""
        F = self.F
        tot = F.sum(self.s.reshape(-1, F.k), axis=0)
        return F.add(F.smul(pow(2, self.L, F.q), self.s0), F.smul(pow(2, self.L - 1, F.q), tot))

    def _upow_sum(self, j, x):
        """sum_k s_{j,k} x^k for x in F (j 1-indexed)."""
        F = self.F
        pw = F.powers(x, self.D + 1)[1:]
        return F.sum(F.mul(self.s[j - 1], pw), axis=0)

    def round_poly(self, r, prefix, x):
        """sum over v_{r+1..l''} in {0,1} of g(a_1..a_{r-1}, x, v_{r+1}, ...), given prefix = (a_1..a_{r-1})."""
        F, L = self.F, self.L
        pre = self.s0
        for j, aj in enumerate(prefix, start=1):
            pre = F.add(pre, self._upow_sum(j, aj))
        cur = self._upow_sum(r, x)
        out = F.smul(pow(2, L - r, F.q), F.add(pre, cur))
        if r < L:
            post = F.sum(self.s[r:].reshape(-1, F.k), axis=0)
            out = F.add(out, F.smul(pow(2, L - r - 1, F.q), post))
        return out

    def evaluate(self, point):
        F = self.F
        acc = self.s0
        for j, aj in enumerate(point, start=1):
            acc = F.add(acc, self._upow_sum(j, aj))
        return acc

    def coefficients(self):
        return dict(s0=self.s0, s=self.s)


# ------------------------------------------------------------------ prover ----
def range_product(F, W, b):
    """prod_{c in {-b+1, ..., b-1}} (W - c)."""
    q = F.q
    acc = None
    for c in range(-b + 1, b):
        fac = F.sub(W, F.from_int(c % q))
        acc = fac if acc is None else F.mul(acc, fac)
    return acc


def prove(ctx, r2l_data, conv, E_mat, coins, z_shares=None, r_shares=None):
    """All provers' side of the sumcheck.  Returns (transcript, private) where transcript holds only what V
    sees and private holds the F_AH inputs, the polynomials (for the ideal-PCS check) and the tables.

    z_shares / r_shares may be overridden (malicious-witness tests in negative.py)."""
    P, F = ctx.P, ctx.F
    n, Lpp, q, b = P.n, P.ell_pp, P.q, P.b
    M = r2l_data["M"]
    if z_shares is None:
        z_shares = r2l_data["z_shares"]
    if r_shares is None:
        r_shares = quotient_shares(ctx, M, z_shares, r2l_data["y_shares"])
    W_base = [w_table(ctx, z_shares[p], r_shares[p]) for p in range(n)]
    g0 = [Blinder(ctx, "sc/g0/p%d" % (p + 1), P.d0) for p in range(n)]
    g1 = [Blinder(ctx, "sc/g1/p%d" % (p + 1), P.d1) for p in range(n)]
    G0 = np.array([g.hypercube_sum() for g in g0])
    G1 = np.array([g.hypercube_sum() for g in g1])
    # V's coins after the first messages (G, commitments)
    ch = coins.sumcheck()
    alpha, rho, tau0, tau1 = ch["alpha"], ch["rho"], ch["tau0"], ch["tau1"]
    eq1 = F.eq_table(tau1)
    MST = mstar_table(ctx, M, E_mat, eq1, alpha)
    MST_table = MST
    EQ = F.eq_table(tau0[:Lpp])
    eqp = [F.eq_point(tau0[Lpp:], bits_of(p, P.logn)) for p in range(n)]
    part = r2l_data["partition"]
    E_tab = [F.lift(row_indicator(ctx, part[p + 1])) for p in range(n)]
    IND = F.lift(row_indicator(ctx, ind_rows(ctx)))
    W = [F.lift(t) for t in W_base]
    nodes0 = list(range(1, P.d0 + 2))
    nodes1 = list(range(1, P.d1 + 2))
    one = F.one()
    rounds, fah_inputs, a_pts = [], [], []
    for r in range(1, Lpp + 1):
        h = W[0].shape[0] // 2
        lohi = lambda T: (T[:h], F.sub(T[h:], T[:h]))
        EQl, EQd = lohi(EQ)
        INl, INd = lohi(IND)
        MSl, MSd = lohi(MST)
        Wl = [lohi(T) for T in W]
        El = [lohi(T) for T in E_tab]
        H0p = F.zeros(n, len(nodes0))
        H1p = F.zeros(n, len(nodes1))
        F0only = F.zeros(len(nodes0))          # sum_p F_{0,p} round values without blinders (degree audit)
        F1only = F.zeros(len(nodes1))
        for ti, t in enumerate(nodes0):
            EQt = F.add(EQl, F.smul(t, EQd))
            INt = F.add(INl, F.smul(t, INd))
            MSt = F.add(MSl, F.smul(t, MSd)) if t in nodes1 else None
            tF = F.from_int(t)
            for p in range(n):
                Wt = F.add(Wl[p][0], F.smul(t, Wl[p][1]))
                Et = F.add(El[p][0], F.smul(t, El[p][1]))
                term = F.add(F.mul(Et, range_product(F, Wt, b)), F.mul(F.sub(F.sub(one, Et), INt), Wt))
                f0 = F.mul(eqp[p], F.sum(F.mul(EQt, term), axis=0))
                F0only[ti] = F.add(F0only[ti], f0)
                H0p[p, ti] = F.add(f0, F.mul(rho, g0[p].round_poly(r, a_pts, tF)))
                if MSt is not None:
                    f1 = F.sum(F.mul(Wt, MSt), axis=0)
                    F1only[ti] = F.add(F1only[ti], f1)
                    H1p[p, ti] = F.add(f1, F.mul(rho, g1[p].round_poly(r, a_pts, tF)))
        H0 = F.sum(H0p, axis=0)                 # F_AH(I)
        H1 = F.sum(H1p, axis=0)
        fah_inputs.append(dict(H0=H0p, H1=H1p, F0_only=F0only, F1_only=F1only))
        rounds.append(dict(H0=H0, H1=H1))
        a_r = coins.round(r)
        a_pts.append(a_r)
        EQ, IND, MST = F.bind_top(EQ, a_r), F.bind_top(IND, a_r), F.bind_top(MST, a_r)
        W = [F.bind_top(T, a_r) for T in W]
        E_tab = [F.bind_top(T, a_r) for T in E_tab]
    a_vec = np.array(a_pts)
    final = dict(w=np.array([T[0] for T in W]), g0=np.array([g.evaluate(a_vec) for g in g0]),
                 g1=np.array([g.evaluate(a_vec) for g in g1]))
    transcript = dict(G0=G0, G1=G1, rounds=rounds, final=final)
    private = dict(r_shares=r_shares, W_base=W_base, g0=g0, g1=g1, fah_inputs=fah_inputs, eq1=eq1,
                   MST_table=MST_table, coins=ch, a=a_vec)
    return transcript, private


def degree_at_nodes(F, vals):
    """Degree of the polynomial through vals at the equally spaced nodes 1..D+1 (forward differences):
    the largest j with Delta^j != 0, or -1 for the zero polynomial."""
    cur = np.asarray(vals, dtype=np.int64) % F.q
    deg = -1
    for j in range(cur.shape[0]):
        if cur.any():
            deg = j
        cur = (cur[1:] - cur[:-1]) % F.q
    return deg


# ---------------------------------------------------------- hypercube checks ----
def hypercube_checks(ctx, r2l_data, r_shares, E_mat, y_eval_ring, coins, z_shares=None):
    """Oracle-level checks of the two claims before any sumcheck round: P_p(x) = 0 at every hypercube point
    for every prover (so sum F_0 = 0 for any tau_0), and sum_x sum_p <w~>_p(x) M*(x) = a."""
    P, F = ctx.P, ctx.F
    q, b = P.q, P.b
    if z_shares is None:
        z_shares = r2l_data["z_shares"]
    part = r2l_data["partition"]
    IND = row_indicator(ctx, ind_rows(ctx))
    zero_ok = True
    bad_points = 0
    for p in range(P.n):
        W = w_table(ctx, z_shares[p], r_shares[p])
        E = row_indicator(ctx, part[p + 1])
        prod = np.ones_like(W)
        for c in range(-b + 1, b):
            prod = prod * ((W - c) % q) % q
        val = (E * prod + (1 - E - IND) * W) % q
        bad_points += int(np.count_nonzero(val))
        zero_ok &= not val.any()
    ch = coins.sumcheck()
    eq1 = F.eq_table(ch["tau1"])
    MST = mstar_table(ctx, r2l_data["M"], E_mat, eq1, ch["alpha"])
    tot = F.zeros()
    for p in range(P.n):
        W = F.lift(w_table(ctx, z_shares[p], r_shares[p]))
        tot = F.add(tot, F.sum(F.mul(W, MST), axis=0))
    a = claimed_sum(ctx, eq1, r2l_data["y_vec"], y_eval_ring, ch["alpha"])
    return dict(zero_check_all_points=bool(zero_ok), nonzero_points=bad_points,
                linear_sum_equals_a=F.eq(tot, a), a=a, linear_sum=tot)


# ---------------------------------------------------------------- verifier ----
def mstar_eval_factorized(ctx, M, E_mat, eq1, alpha, a_pt):
    """M~*(a) evaluated from the definition (5_BB:254-260) term by term with eq(a_U, .) and eq(a_L, .)
    factored, without building the 2^{l''} table (independent of `mstar_table`)."""
    P, F = ctx.P, ctx.F
    d, mu, npr, q = P.d, P.mu, P.n_prime, P.q
    aU, aL = a_pt[:P.ell_U], a_pt[P.ell_U:]
    eqU = F.eq_table(aU)                                   # eq(a_U, bit(u0))
    eqL = F.eq_table(aL)                                   # eq(a_L, bit(l0))
    # A_L = sum_{l0 < 2d} eq(a_L, bit(l0)) alpha^{l0} = prod_j ((1 - a_Lj) + a_Lj alpha^{2^{(alpha+1) - j}})
    A_L = F.one()
    for j, aj in enumerate(aL, start=1):
        e = 2 ** (P.ell_L - j)
        A_L = F.mul(A_L, F.add(F.sub(F.one(), aj), F.mul(aj, F.pow(alpha, e))))
    apow = F.powers(alpha, d)
    M_alpha = np.einsum('iuj,jc->iuc', M, apow) % q
    c_u = F.sum(F.mul(eq1[:npr][:, None, :], M_alpha), axis=0)
    t1a = F.mul(F.sum(F.mul(eqU[:mu], c_u), axis=0), A_L)
    ad1 = F.add(F.pow(alpha, d), F.one())
    t1b = F.neg(F.mul(F.mul(ad1, A_L), F.sum(F.mul(eq1[:npr], eqU[mu:mu + npr]), axis=0)))
    # second term
    inner = F.zeros()
    for j0 in range(d):
        rowsum = F.sum(F.smul(E_mat[j0], eqL[:d]), axis=0)          # sum_l0 E[j0][l0] eq(a_L, bit(l0))
        inner = F.add(inner, F.mul(eq1[npr + j0], rowsum))
    t2 = F.mul(eqU[mu - 1], inner)
    # third term
    t3 = F.zeros()
    for l0 in range(d, 2 * d):
        idx = npr + d + np.arange(mu + npr) + (l0 - d) * (mu + npr)
        t3 = F.add(t3, F.mul(eqL[l0], F.sum(F.mul(eqU[:mu + npr], eq1[idx]), axis=0)))
    return F.add(F.add(t1a, t1b), F.add(t2, t3))


def verify(ctx, public, transcript, coins, pcs):
    """V's side of the sumcheck.  public = dict(M, y_vec, E_mat, y_eval_ring, partition) recomputed by V from
    public data; pcs = ideal stand-in for PCS.Verify (evaluates the committed polynomials directly).
    Returns an ordered dict of named boolean checks (all must hold)."""
    P, F = ctx.P, ctx.F
    n, Lpp = P.n, P.ell_pp
    checks = OrderedDict()
    ch = coins.sumcheck()
    alpha, rho, tau0, tau1 = ch["alpha"], ch["rho"], ch["tau0"], ch["tau1"]
    eq1 = F.eq_table(tau1)
    a_claim = claimed_sum(ctx, eq1, public["y_vec"], public["y_eval_ring"], alpha)
    G0, G1 = transcript["G0"], transcript["G1"]
    prev0 = F.mul(rho, F.sum(G0, axis=0))                    # H_{0,0}
    prev1 = F.add(a_claim, F.mul(rho, F.sum(G1, axis=0)))    # H_{1,0}
    a_pts = []
    for r, msg in enumerate(transcript["rounds"], start=1):
        H0, H1 = msg["H0"], msg["H1"]
        assert H0.shape[0] == P.d0 + 1 and H1.shape[0] == P.d1 + 1
        z0, z1 = F.lagrange_eval(H0, F.zeros()), F.lagrange_eval(H1, F.zeros())
        checks["round_%02d_H0" % r] = F.eq(prev0, F.add(z0, H0[0]))
        checks["round_%02d_H1" % r] = F.eq(prev1, F.add(z1, H1[0]))
        a_r = coins.round(r)
        a_pts.append(a_r)
        prev0, prev1 = F.lagrange_eval(H0, a_r), F.lagrange_eval(H1, a_r)
    assert len(a_pts) == Lpp
    a_vec = np.array(a_pts)
    fin = transcript["final"]
    for p in range(n):
        checks["pcs_open_g0_p%d" % (p + 1)] = F.eq(pcs.g0(p, a_vec), fin["g0"][p])
        checks["pcs_open_g1_p%d" % (p + 1)] = F.eq(pcs.g1(p, a_vec), fin["g1"][p])
        checks["pcs_open_w_p%d" % (p + 1)] = F.eq(pcs.w(p, a_vec), fin["w"][p])
    # V's own F_{0,p}(a), F_{1,p}(a)
    aU = a_vec[:P.ell_U]
    eqU = F.eq_table(aU)
    ind = F.sum(eqU[list(ind_rows(ctx))], axis=0)
    eq_tau_a = F.one()
    for t, x in zip(tau0[:Lpp], a_vec):
        eq_tau_a = F.mul(eq_tau_a, F.add(F.mul(t, x), F.mul(F.sub(F.one(), t), F.sub(F.one(), x))))
    mstar = mstar_eval_factorized(ctx, public["M"], public["E_mat"], eq1, alpha, a_vec)
    tot0, tot1 = F.zeros(), F.zeros()
    for p in range(n):
        e_p = F.sum(eqU[public["partition"][p + 1]], axis=0)
        wv = fin["w"][p]
        eqf = F.mul(eq_tau_a, F.eq_point(tau0[Lpp:], bits_of(p, P.logn)))
        term = F.add(F.mul(e_p, range_product(F, wv, P.b)), F.mul(F.sub(F.sub(F.one(), e_p), ind), wv))
        tot0 = F.add(tot0, F.add(F.mul(eqf, term), F.mul(rho, fin["g0"][p])))
        tot1 = F.add(tot1, F.add(F.mul(wv, mstar), F.mul(rho, fin["g1"][p])))
    checks["final_H0"] = F.eq(prev0, tot0)
    checks["final_H1"] = F.eq(prev1, tot1)
    return checks, dict(a_claim=a_claim, a=a_vec, mstar_at_a=mstar)


class IdealPCS:
    """Stand-in for the deferred base PCS: 'opening' a commitment means evaluating the committed polynomial
    directly (the commitments themselves are not modelled)."""

    def __init__(self, ctx, private):
        self.ctx = ctx
        self.g0_polys, self.g1_polys = private["g0"], private["g1"]
        self.W = [ctx.F.lift(t) for t in private["W_base"]]

    def g0(self, p, a):
        return self.g0_polys[p].evaluate(a)

    def g1(self, p, a):
        return self.g1_polys[p].evaluate(a)

    def w(self, p, a):
        """<w~>_p(a) as sum_x eq(a, x) W_p[x] (a different evaluation path from the provers' folding)."""
        F = self.ctx.F
        return F.sum(F.mul(F.eq_table(a), self.W[p]), axis=0)
