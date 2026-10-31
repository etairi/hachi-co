"""Numerical checks of the formulas of the t-hiding proof (t-hiding.tex, tH) on oracle runs.

The proof computes what the honest provers contribute from public data and the corrupted parties' actual
inputs.  These checks compare each formula with the same quantity computed directly from the honest provers'
private data, for every corruption set tried, with corrupted parties that deviate: they submit arbitrary images
V_C and U_C to F_AH(D) and F_AH(B) and a short vector Z_C to F_AH(1^n).  A check is evidence for the identity at
these parameters and seeds, not a proof of it.  Base PCS openings are not involved.
"""
from collections import OrderedDict
from itertools import combinations

import numpy as np

from . import sumcheck as sc
from .prg import Prg
from .r2l import a_times_G, build_y


def corruption_sets(n):
    """Every proper subset for n <= 3; otherwise a selection, with and without prover 1."""
    if n <= 3:
        return [c for size in range(n) for c in combinations(range(1, n + 1), size)]
    return [(), (1,), (2,), (1, 2), (2, 3), (n,), tuple(range(2, n + 1))]


def refresh_checks(res):
    """tH Eq. hide-refresh: sum_{j in H} f~_j = f - sum_{c in C} f_c - sum_{i in H, c in C} sigma_{i,c}
    + sum_{c in C, j in H} sigma_{c,j}, and Y_H of Eq. hide-YH from the permitted value f(x)."""
    ctx, inp, cd = res["ctx"], res["inp"], res["cdata"]
    P, F, S = ctx.P, ctx.F, ctx.S
    q = P.q
    sigma = cd["reshare"]
    out = OrderedDict()
    mono = S.monomials([F.to_sub(xi) for xi in inp["x"]])
    for C in corruption_sets(P.n):
        H = [p for p in range(1, P.n + 1) if p not in C]
        lhs = cd["f_shares"][[p - 1 for p in H]].sum(axis=0) % q
        known = np.zeros_like(lhs)                  # everything on the right except f
        for c in C:
            known = known - inp["f_init"][c - 1]
            for i in H:
                known = known - sigma[i - 1, c - 1] + sigma[c - 1, i - 1]
        known %= q
        name = "C=%s" % (list(C),)
        out["hide-refresh " + name] = bool(np.array_equal(lhs, (inp["f"] + known) % q))
        # Y_H = sum_{j in H} f~_j(x) = f(x) + (the known polynomial)(x)
        Y_H = F.from_sub(S.inner([F.to_sub(t) for t in lhs], mono))
        rhs = F.add(inp["y"], F.from_sub(S.inner([F.to_sub(t) for t in known], mono)))
        out["hide-YH " + name] = F.eq(Y_H, rhs)
    return out


def trace_checks(res):
    """tH:145-151: E(u_i) = f~_i(x) for every prover, so E(u_H) = Y_H for every H."""
    ctx, conv = res["ctx"], res["conv"]
    P, F, S = ctx.P, ctx.F, ctx.S
    mono = S.monomials([F.to_sub(xi) for xi in res["inp"]["x"]])
    ok = True
    for p in range(P.n):
        f_i = F.from_sub(S.inner([F.to_sub(t) for t in res["cdata"]["f_shares"][p]], mono))
        ok &= bool(np.array_equal(res["E_mat"] @ conv["u_shares"][p] % P.q, ctx.embed(f_i)))
    return OrderedDict([("E(u_i) = f~_i(x) for every prover", ok)])


def honest_aggregates(res, C, seed):
    """The honest provers' shares against corrupted inputs V_C, U_C, Z_C (all zero for C empty)."""
    ctx = res["ctx"]
    P, R = ctx.P, ctx.R
    q, d = P.q, P.d
    r2l, acc = res["r2l"], res["r2l"]["attempts"][-1]
    L = r2l["layout"]
    H = [p for p in range(1, P.n + 1) if p not in C]
    prg = Prg(ctx.seed, "hide/deviation/" + seed)
    if C:
        V_C = np.array(prg.mod_q(q, P.n_D * d), dtype=np.int64).reshape(P.n_D, d)
        U_C = np.array(prg.mod_q(q, P.n_B * d), dtype=np.int64).reshape(P.n_B, d)
        Z_C = np.array([prg.range(-3, 3) for _ in range(P.width_A * d)], dtype=np.int64).reshape(P.width_A, d)
    else:
        V_C, U_C = np.zeros((P.n_D, d), dtype=np.int64), np.zeros((P.n_B, d), dtype=np.int64)
        Z_C = np.zeros((P.width_A, d), dtype=np.int64)
    # the honest images and responses
    D_slices = acc["D_input_slices"]
    V_H = R.matvec(res["pp"]["D"], D_slices[[p - 1 for p in H]].sum(axis=0) % q)
    w = P.n_A * P.delta
    t_rho = np.zeros((P.width_B, d), dtype=np.int64)
    for p in H:
        for i, st in res["states"][p].items():
            t_rho[i * w:(i + 1) * w] = st["t_hat"]
        first = P.n_blocks * w + (p - 1) * P.m_u
        t_rho[first:first + P.m_u] = R.reduce(res["cdata"]["rho_u"][p - 1])
    U_H = R.matvec(res["pp"]["B"], t_rho)
    Z_H = sum(acc["provers"][p]["z"] for p in H)
    # the public outputs of the functionalities
    v = (V_H + V_C) % q
    u = (U_H + U_C) % q
    z = Z_H + Z_C
    e1 = 1 if 1 in H else 0
    # the honest shares of z': only prover 1's z_hat depends on the output z
    zs = r2l["z_shares"].copy()
    if e1:
        zs[0, L.z:L.u] = R.reduce(R.j_inv(z, P.b, P.tau))
    y = build_y(ctx, v, u)
    y_shares = np.zeros((P.n, P.n_prime, d), dtype=np.int64)
    y_shares[0] = y
    rs = sc.quotient_shares(ctx, r2l["M"], zs, y_shares)
    return dict(H=H, e1=e1, V_C=V_C, U_C=U_C, Z_H=Z_H, z=z, v=v, u=u, y=y, zs=zs, rs=rs, y_shares=y_shares)


def residual_checks(res, C, ag):
    """tH Eqs. hide-residual and hide-quotient, then hide-S1 and hide-S0."""
    ctx = res["ctx"]
    P, R, F = ctx.P, ctx.R, ctx.F
    q, d, npr, mu = P.q, P.d, P.n_prime, P.mu
    r2l = res["r2l"]
    L = r2l["layout"]
    M = r2l["M"]
    H, e1 = ag["H"], ag["e1"]
    idx = [p - 1 for p in H]
    inv2 = (q + 1) // 2
    out = OrderedDict()

    # R_H = (M z'_H - y_H) mod X^d + 1, from the honest shares
    z_H = ag["zs"][idx].sum(axis=0) % q
    y_H = (e1 * ag["y"]) % q
    Pf = R.matvec_full(M, z_H)                                  # (n', 2d), before ring reduction
    Pf[:, :d] = (Pf[:, :d] - y_H) % q
    P0, P1 = Pf[:, :d], Pf[:, d:]
    R_H = (P0 - P1) % q

    # the five blocks of Eq. hide-residual
    want = np.zeros((npr, d), dtype=np.int64)
    want[L.row_D:L.row_D + P.n_D] = ((1 - e1) * ag["v"] - ag["V_C"]) % q
    want[L.row_B:L.row_B + P.n_B] = ((1 - e1) * ag["u"] - ag["U_C"]) % q
    diff = R.reduce(ag["Z_H"] - e1 * ag["z"])
    want[L.row_c] = a_times_G(ctx, r2l["a_ring"], diff)
    want[L.row_A:L.row_A + P.n_A] = R.matvec(res["pp"]["A"], diff)
    out["hide-residual"] = bool(np.array_equal(R_H, want))
    if C:
        # Negative control: the formula must depend on the corrupted inputs, or the check above would be vacuous.
        ignoring = want.copy()
        ignoring[L.row_D:L.row_D + P.n_D] = (1 - e1) * ag["v"] % q
        out["control: dropping V_C from the formula is detected"] = not np.array_equal(R_H, ignoring)

    # Eq. hide-quotient
    r_H = ag["rs"][idx].sum(axis=0) % q
    lhs = np.zeros((npr, 3 * d), dtype=np.int64)
    lhs[:, :2 * d] = Pf
    lhs[:, :2 * d] = (lhs[:, :2 * d] - r_H) % q                 # - r_H
    lhs[:, d:3 * d] = (lhs[:, d:3 * d] - r_H) % q               # - X^d r_H
    rhs = np.zeros((npr, 3 * d), dtype=np.int64)
    rhs[:, :d] = R_H * inv2 % q
    rhs[:, 2 * d:3 * d] = R_H * inv2 % q
    out["hide-quotient"] = bool(np.array_equal(lhs, rhs))
    out["high coefficients of r_H = -R_H/2"] = bool(np.array_equal(r_H[:, d:], (-R_H * inv2) % q))

    # the two sums, directly from the honest tables (the pads included)
    ch = res["private"]["coins"]
    alpha = ch["alpha"]
    eq1 = F.eq_table(ch["tau1"])
    MST = sc.mstar_table(ctx, M, res["E_mat"], eq1, alpha)
    S1_direct = F.zeros()
    pads = res["private"]["pads"]
    part = r2l["partition"]
    IND = sc.row_indicator(ctx, sc.ind_rows(ctx))
    S0_points = 0
    for p in H:
        base = sc.w_table(ctx, ag["zs"][p - 1], ag["rs"][p - 1], None if P.pad == "field" else pads[p - 1])
        W = sc.w_lifted(ctx, base, pads[p - 1])
        S1_direct = F.add(S1_direct, F.sum(F.mul(W, MST), axis=0))
        # Eq. hide-S0: F_{0,p} vanishes at every Boolean point.  The pad's row has both coefficients zero, so the
        # base table with the pad's entry at zero gives the same values.
        E = sc.row_indicator(ctx, part[p])
        prod = np.ones_like(base)
        for c in range(-P.b + 1, P.b):
            prod = prod * ((base - c) % q) % q
        S0_points += int(np.count_nonzero((E * prod + (1 - E - IND) * base) % q))
    out["hide-S0 (F_0 vanishes at every Boolean point)"] = S0_points == 0

    # Eq. hide-S1, from public data, the corrupted inputs and E(u_H)
    apow = F.powers(alpha, 2 * d)
    y_at = np.einsum('ij,jc->ic', y_H, apow[:d]) % q
    R_at = np.einsum('ij,jc->ic', R_H, apow[:d]) % q
    factor = F.smul(inv2, F.add(F.one(), F.pow(alpha, 2 * d)))
    S1 = F.zeros()
    for j0 in range(npr):
        S1 = F.add(S1, F.mul(eq1[j0], F.add(y_at[j0], F.mul(factor, R_at[j0]))))
    u_H = res["conv"]["u_shares"][idx].sum(axis=0) % q
    E_u = res["E_mat"] @ u_H % q
    for l0 in range(d):
        S1 = F.add(S1, F.smul(int(E_u[l0]), eq1[npr + l0]))
    for j0 in range(npr):
        for l0 in range(d):
            p_idx = npr + d + (mu + j0) + l0 * (mu + npr)       # p(j, l) - 1, with j = j0 + 1
            S1 = F.sub(S1, F.smul(int(R_H[j0, l0]) * inv2 % q, eq1[p_idx]))
    out["hide-S1"] = F.eq(S1, S1_direct)
    return out


def pad_checks(res):
    """tH Eq. hide-pad, W_p(a) = W_p^0(a) + c(a) rho^w_p, at the sumcheck's point (pad "field")."""
    ctx = res["ctx"]
    P, F = ctx.P, ctx.F
    if P.pad != "field":
        return OrderedDict()
    a = res["private"]["a"]
    eq_a = F.eq_table(a)
    c_a = eq_a[sc.pad_index(ctx)]
    ok = True
    for p in range(P.n):
        base = F.sum(F.mul(eq_a, F.lift(res["private"]["W_base"][p])), axis=0)
        ok &= F.eq(F.add(base, F.mul(c_a, res["private"]["pads"][p])), res["transcript"]["final"]["w"][p])
    return OrderedDict([("hide-pad at the final point", ok)])


def run(res):
    """Every check on one run; returns an ordered dict name -> bool."""
    out = OrderedDict()
    out.update(refresh_checks(res))
    out.update(trace_checks(res))
    out.update(pad_checks(res))
    for C in corruption_sets(res["ctx"].P.n):
        ag = honest_aggregates(res, C, "C" + "-".join(map(str, C)))
        for name, ok in residual_checks(res, C, ag).items():
            out["%s C=%s" % (name, list(C))] = ok
    return out
