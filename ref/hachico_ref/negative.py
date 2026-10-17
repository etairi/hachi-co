"""Negative tests: tampered messages and witnesses must be rejected, at the expected check.

Two kinds:
  * transcript tampering: the verifier is re-run on a modified transcript with the same public coins
    (the coins never depend on messages, so the later honest messages are unchanged);
  * malicious witnesses: the provers re-run the sumcheck on modified shares of z' (same coins).
Each test states which checks must fail and which must still pass; a test passes iff exactly that happens.
"""
import copy
from collections import OrderedDict
from fractions import Fraction
from itertools import product

import numpy as np

from . import sumcheck as sc
from .commit import open_check
from .conv import verify_conv
from .r2l import rej


def _failing(checks):
    return sorted(k for k, v in checks.items() if not v)


def _verify(res, tr, public=None):
    ctx = res["ctx"]
    public = res["public"] if public is None else public
    checks, _ = sc.verify(ctx, public, tr, res["coins"], sc.IdealPCS(ctx, res["private"]))
    return checks


def _record(out, name, checks, must_fail, must_pass=None, note=""):
    failing = _failing(checks)
    ok = set(must_fail) <= set(failing)
    if must_pass is not None:
        ok &= not (set(must_pass) & set(failing))
    out[name] = dict(passed=bool(ok), failing_checks=failing, expected_to_fail=sorted(must_fail),
                     expected_to_pass=sorted(must_pass) if must_pass is not None else "not constrained",
                     note=note)


def transcript_tests(res):
    ctx = res["ctx"]
    P, F = ctx.P, ctx.F
    Lpp = P.ell_pp
    out = OrderedDict()
    tr0 = res["transcript"]
    for r in sorted({1, (Lpp + 1) // 2, Lpp}):
        for which in ("H0", "H1"):
            tr = copy.deepcopy(tr0)
            tr["rounds"][r - 1][which][0] = F.add(tr["rounds"][r - 1][which][0], F.one())
            name = "round_%02d_%s" % (r, which)
            _record(out, "tamper %s(1) += 1 at round %d" % (which, r), _verify(res, tr), [name],
                    note="the round check of the tampered round fails")
            # consistent tamper: add delta X (X - 1); passes its own round, caught one step later
            tr = copy.deepcopy(tr0)
            D = tr["rounds"][r - 1][which].shape[0]
            for ti in range(D):
                t = ti + 1
                tr["rounds"][r - 1][which][ti] = F.add(tr["rounds"][r - 1][which][ti], F.from_int(5 * t * (t - 1)))
            nxt = ("round_%02d_%s" % (r + 1, which)) if r < Lpp else ("final_%s" % which)
            _record(out, "tamper %s += 5 X(X-1) at round %d" % (which, r), _verify(res, tr), [nxt],
                    must_pass=[name],
                    note="sum at 0 and 1 unchanged, so round %d passes; the next check fails" % r)
    for p in range(P.n):
        tr = copy.deepcopy(tr0)
        tr["final"]["w"][p] = F.add(tr["final"]["w"][p], F.one())
        _record(out, "tamper final <w~>_%d(a) += 1" % (p + 1), _verify(res, tr),
                ["pcs_open_w_p%d" % (p + 1), "final_H0", "final_H1"],
                note="caught by the (ideal) PCS opening and, independently, by both final checks")
        tr = copy.deepcopy(tr0)
        tr["final"]["g0"][p] = F.add(tr["final"]["g0"][p], F.one())
        _record(out, "tamper final g_0,%d(a) += 1" % (p + 1), _verify(res, tr),
                ["pcs_open_g0_p%d" % (p + 1), "final_H0"], must_pass=["final_H1"])
        tr = copy.deepcopy(tr0)
        tr["final"]["g1"][p] = F.add(tr["final"]["g1"][p], F.one())
        _record(out, "tamper final g_1,%d(a) += 1" % (p + 1), _verify(res, tr),
                ["pcs_open_g1_p%d" % (p + 1), "final_H1"], must_pass=["final_H0"])
        tr = copy.deepcopy(tr0)
        tr["G0"][p] = F.add(tr["G0"][p], F.one())
        _record(out, "tamper G_0,%d += 1" % (p + 1), _verify(res, tr), ["round_01_H0"],
                must_pass=["round_01_H1"])
    # public-input / earlier-message tampering
    y_bad = F.add(res["inp"]["y"], F.one())
    out["tamper claimed y += 1: Pi_conv check"] = dict(
        passed=not verify_conv(ctx, y_bad, res["conv"]["E_u"]), note="V's check y = E(u) fails")
    pub = dict(res["public"])
    pub["y_eval_ring"] = ctx.embed(y_bad)
    _record(out, "tamper claimed y += 1: sumcheck", _verify(res, tr0, pub), ["round_01_H1"],
            must_pass=["round_01_H0"], note="the E rows bind <u> in z' to y")
    Eu_bad = res["conv"]["E_u"].copy()
    Eu_bad[0] = (Eu_bad[0] + 1) % P.q
    out["tamper E(u) message: Pi_conv check"] = dict(
        passed=not verify_conv(ctx, res["inp"]["y"], Eu_bad), note="V's check y = E(u) fails")
    pub = dict(res["public"])
    yv = pub["y_vec"].copy()
    yv[0, 0] = (yv[0, 0] + 1) % P.q
    pub["y_vec"] = yv
    _record(out, "tamper v (accepted attempt) coefficient 0 += 1", _verify(res, tr0, pub), ["round_01_H1"],
            must_pass=["round_01_H0"], note="v enters y, hence the claimed sum a")
    return out


def _prove_verify(res, z_shares=None, r_shares=None):
    ctx = res["ctx"]
    tr, priv = sc.prove(ctx, res["r2l"], res["conv"], res["E_mat"], res["coins"], z_shares=z_shares,
                        r_shares=r_shares)
    checks, _ = sc.verify(ctx, res["public"], tr, res["coins"], sc.IdealPCS(ctx, priv))
    return checks


def witness_tests(res):
    ctx = res["ctx"]
    P, R = ctx.P, ctx.R
    q, b = P.q, P.b
    L = res["r2l"]["layout"]
    zs0 = res["r2l"]["z_shares"]
    out = OrderedDict()
    H0_names = [k for k in res["verifier_checks"] if k.endswith("H0")]
    H1_names = [k for k in res["verifier_checks"] if k.endswith("H1")]
    # (a) non-canonical z_hat digits in prover 1's share: digit t -> x + b, digit t+1 -> y - 1.
    #     J z_hat (hence M z') is unchanged, but x + b is outside {-b+1, ..., b-1}.
    zh = R.centered(zs0[0, L.z:L.u])
    found = None
    for idx in range(zh.shape[0]):
        t = idx % P.tau
        if t == P.tau - 1:
            continue
        for l0 in range(P.d):
            if zh[idx, l0] >= 0:
                found = (idx, l0)
                break
        if found:
            break
    idx, l0 = found
    zs = zs0.copy()
    zs[0, L.z + idx, l0] = (zh[idx, l0] + b) % q
    zs[0, L.z + idx + 1, l0] = (zh[idx + 1, l0] - 1) % q
    zsum = zs.sum(axis=0) % q
    lin = ctx.SR.matvec(res["r2l"]["M"].tolist(), zsum.tolist()) == res["r2l"]["y_vec"].tolist()
    _record(out, "witness: out-of-range z_hat digit (J z_hat unchanged)", _prove_verify(res, zs),
            ["round_01_H0"], must_pass=H1_names,
            note="M z' = y still holds (%s); only the range check of F_0 can catch it" % lin)
    out["witness: out-of-range z_hat digit (J z_hat unchanged)"]["linear_relation_still_holds"] = bool(lin)
    # (b) in-range change of one w_hat digit of a block of prover 1: breaks M z' = y only
    i0 = res["ctx"].P.blocks_of(1)[0]
    pos = L.w + i0 * P.delta
    x = int(R.centered(zs0[0, pos, 0]))
    newx = x + 1 if x + 1 <= (b + 1) // 2 - 1 else x - 1
    zs = zs0.copy()
    zs[0, pos, 0] = newx % q
    _record(out, "witness: in-range w_hat digit changed (M z' != y)", _prove_verify(res, zs),
            ["round_01_H1"], must_pass=H0_names, note="only the linear sumcheck can catch it")
    if P.n >= 2:
        # (c) prover 2 writes into prover 1's slot (z_hat), prover 1 compensates: aggregate unchanged
        zs = zs0.copy()
        zs[1, L.z, 0] = (zs[1, L.z, 0] + 1) % q
        zs[0, L.z, 0] = (zs[0, L.z, 0] - 1) % q
        _record(out, "witness: prover 2 nonzero outside its slice (aggregate unchanged)",
                _prove_verify(res, zs), ["round_01_H0"], must_pass=H1_names,
                note="the zero-check part of F_0 catches the slice violation")
        # (d) <u> and quotient rows are ordinary additive shares: compensated changes must be accepted
        zs = zs0.copy()
        zs[0, L.u, 3] = (zs[0, L.u, 3] + 7) % q
        zs[1, L.u, 3] = (zs[1, L.u, 3] - 7) % q
        rs = sc.quotient_shares(ctx, res["r2l"]["M"], zs, res["r2l"]["y_shares"])
        checks = _prove_verify(res, zs, rs)
        out["witness: re-randomized <u> shares are accepted"] = dict(
            passed=all(checks.values()), failing_checks=_failing(checks),
            note="<u> (position mu) is not range- or zero-checked per prover (S_i is inside [mu-1])")
        rs = sc.quotient_shares(ctx, res["r2l"]["M"], zs0, res["r2l"]["y_shares"]).copy()
        rs[0, 0, P.d + 2] = (rs[0, 0, P.d + 2] + 11) % q
        rs[1, 0, P.d + 2] = (rs[1, 0, P.d + 2] - 11) % q
        checks = _prove_verify(res, zs0, rs)
        out["witness: re-randomized quotient shares are accepted"] = dict(
            passed=all(checks.values()), failing_checks=_failing(checks),
            note="the quotient rows mu+1..mu+n' are not checked per prover; only their sum")
    # (e) quotient with a nonzero high half in the sum (a 'quotient' that is not one)
    rs = sc.quotient_shares(ctx, res["r2l"]["M"], zs0, res["r2l"]["y_shares"]).copy()
    rs[0, 1, P.d + 1] = (rs[0, 1, P.d + 1] + 1) % q
    _record(out, "witness: quotient sum with a nonzero coefficient of X^{d+1}", _prove_verify(res, zs0, rs),
            ["round_01_H1"], must_pass=H0_names, note="the third term of M~ (zero high coefficients)")
    return out


def open_tests(res):
    ctx = res["ctx"]
    P = ctx.P
    out = OrderedDict()
    st = copy.deepcopy(res["states"])
    p0 = 1
    i0 = sorted(st[p0].keys())[0]
    s = st[p0][i0]["s"].copy()
    s[0, 0] = (s[0, 0] + 1) % P.q
    st[p0][i0]["s"] = s
    bit, info = open_check(ctx, res["pp"], st, res["u_com"], res["inp"]["f"])
    out["open: one digit of s_i changed"] = dict(passed=not bit, failing=[k for k, v in info["checks"].items() if not v])
    f_bad = res["inp"]["f"].copy()
    f_bad[5, 1] = (f_bad[5, 1] + 1) % P.q
    bit, info = open_check(ctx, res["pp"], res["states"], res["u_com"], f_bad)
    out["open: f changed in one coordinate"] = dict(passed=not bit,
                                                   failing=[k for k, v in info["checks"].items() if not v])
    c = np.zeros((P.n_blocks, P.d), dtype=np.int64)
    c[:, 0] = 1
    c[0, 1] = P.omega_bar          # ||c_0||_1 = omega_bar + 1
    bit, info = open_check(ctx, res["pp"], res["states"], res["u_com"], res["inp"]["f"], c=c)
    out["open: c outside the slack space"] = dict(passed=not bit,
                                                 failing=[k for k, v in info["checks"].items() if not v])
    return out


# ------------------------------------------------------------ rejection sampling ----
def rej_exhaustive(N=2, gamma=4, beta1=2):
    """Exhaustive Rej over all masks for every v in [-(beta1+1), beta1+1]^N (draft 4_P:117-127):
    ||v|| <= beta1 -> accepted outputs are each z in the box exactly once; ||v|| = beta1+1 -> not."""
    box = list(product(range(-gamma + beta1, gamma - beta1 + 1), repeat=N))
    masks = list(product(range(-gamma, gamma + 1), repeat=N))
    good, bad_detected, total_bad = True, True, 0
    for v in product(range(-beta1 - 1, beta1 + 2), repeat=N):
        acc = []
        for rr in masks:
            z = rej(np.array(v), np.array(rr), gamma, beta1, check=False)   # the oracle's own Rej
            if z is not None:
                acc.append(tuple(int(c) for c in z))
        vn = max(abs(c) for c in v)
        if vn <= beta1:
            good &= sorted(acc) == sorted(box)
        else:
            total_bad += 1
            bad_detected &= sorted(acc) != sorted(box)
    return dict(N=N, gamma=gamma, beta1=beta1, accepted_masks_per_v=(2 * gamma - 2 * beta1 + 1) ** N,
                within_bound_uniform=bool(good), boundary_v_count=total_bad, boundary_breaks_uniformity=bool(bad_detected))


def rej_boundary_toy(P):
    """At the toy parameters, exactly: for v = (beta1 + 1, 0, ..., 0) the output z with z_1 = -gamma + beta1
    needs r_1 = -gamma - 1, outside the mask range, so P[Rej(v) = z] = 0 while SimRej gives it
    (2 gamma + 1)^{-N} > 0.  The accepted set has (2g - 2b1)(2g - 2b1 + 1)^{N-1} < (2g - 2b1 + 1)^N elements."""
    g, b1, N = P.gamma, P.beta1, P.N
    v1 = b1 + 1
    lo, hi = -g + b1, g - b1
    reach = sum(1 for z in range(lo, hi + 1) if -g <= z - v1 <= g)
    size_v = reach * (2 * g - 2 * b1 + 1) ** (N - 1)
    size_box = (2 * g - 2 * b1 + 1) ** N
    v1b = b1
    reach_b = sum(1 for z in range(lo, hi + 1) if -g <= z - v1b <= g)
    return dict(first_coordinate_reachable_at_beta1_plus_1=reach, box_side=2 * g - 2 * b1 + 1,
                first_coordinate_reachable_at_beta1=reach_b,
                accepted_set_smaller=bool(size_v < size_box), within_bound_full=bool(reach_b == 2 * g - 2 * b1 + 1),
                acceptance_ratio_boundary_over_box=str(Fraction(reach, 2 * g - 2 * b1 + 1)))


def abort_rate(P, trials=10000, seed=12345):
    """Empirical abort rate of the n-prover Rej step versus the exact 1 - M^{-n} (numpy PCG64, fixed seed;
    folds at their worst-case norm beta1 in one coordinate — the rate does not depend on the fold)."""
    import math
    rng = np.random.default_rng(seed)
    aborts = 0
    fold = np.zeros(P.N, dtype=np.int64)
    fold[0] = P.beta1
    for _ in range(trials):
        acc_all = True
        for _p in range(P.n):
            r = rng.integers(-P.gamma, P.gamma + 1, size=P.N)
            if rej(fold, r, P.gamma, P.beta1) is None:                     # the oracle's own Rej
                acc_all = False
                break
        aborts += not acc_all
    p_exact = 1 - math.exp(-P.n * P.ln_M)
    sigma = math.sqrt(p_exact * (1 - p_exact) / trials)
    rate = aborts / trials
    return dict(trials=trials, seed=seed, rng="numpy PCG64", aborts=aborts, rate=rate, exact=p_exact,
                sigma=sigma, within_3_sigma=bool(abs(rate - p_exact) <= 3 * sigma))
