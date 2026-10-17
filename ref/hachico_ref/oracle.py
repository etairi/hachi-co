"""One honest run of Commit, Open, Pi_conv, ring-to-linear and the collaborative sumcheck, with every
relation checked, plus the verifier run from public data only."""
import time
from collections import OrderedDict

import numpy as np

from . import sumcheck as sc
from .commit import commit, open_check, sample_input, setup
from .context import Ctx
from .conv import e_matrix, pi_conv, public_vectors, verify_conv
from .r2l import Coins, build_M, build_y, check_linear_relation, ring_to_linear


def verifier_public(ctx, pp, u_com, x, y, r2l_transcript):
    """Everything V needs, recomputed from public data: pp, comm = u, x, y, and the accepted attempt's
    (v, c).  Returns dict(M, y_vec, E_mat, y_eval_ring, partition)."""
    pub = public_vectors(ctx, x)
    a_ring = np.array([ctx.embed(t) for t in pub["a"]])
    b_ring = np.array([ctx.embed(t) for t in pub["b"]])
    acc = r2l_transcript[-1]
    assert not acc["abort"]
    M = build_M(ctx, pp, b_ring, a_ring, acc["c"])
    from .r2l import Layout
    S, _ = Layout(ctx.P).partition()
    return dict(M=M, y_vec=build_y(ctx, acc["v"], u_com), E_mat=e_matrix(ctx, pub["v_point"]),
                y_eval_ring=ctx.embed(y), partition=S)


def run_instance(P, seed, setup_seed=None, log=print):
    t_start = time.time()
    T = OrderedDict()
    ctx = Ctx(P, seed, setup_seed)
    rel = OrderedDict()                     # oracle-level relation checks

    t = time.time()
    pp = setup(ctx)
    inp = sample_input(ctx)
    T["setup+input"] = time.time() - t

    t = time.time()
    u_com, states, cdata = commit(ctx, pp, inp["f_init"])
    open_bit, open_info = open_check(ctx, pp, states, u_com, inp["f"])
    rel["open(pp, st, u, f, c*) = 1"] = bool(open_bit)
    for k_, v_ in open_info["checks"].items():
        rel["open." + k_] = v_
    T["commit+open"] = time.time() - t

    t = time.time()
    conv = pi_conv(ctx, states, inp["x"])
    E_mat = e_matrix(ctx, conv["v_point"])
    rel["conv.E_matrix_times_u_eq_E(u)"] = bool(np.array_equal(E_mat @ conv["u"] % P.q, conv["E_u"]))
    # u = f'(x') over all blocks (aggregate), and E(u) = f(x) computed independently with subfield.py
    S_ = ctx.S
    fx = inp["y"]
    rel["conv.E(u)_in_R_q^H"] = bool(ctx.in_subfield(conv["E_u"]))
    rel["conv.verifier_y_eq_E(u)"] = verify_conv(ctx, fx, conv["E_u"])
    T["conv"] = time.time() - t

    t = time.time()
    coins = Coins(ctx)
    r2l = ring_to_linear(ctx, pp, states, conv, u_com, coins)
    rel["r2l.challenges_in_C"] = all(a["c_in_C"] for a in r2l["attempts"])
    rel["r2l.prover1_norm_check_z_le_beta"] = r2l["z_norm_ok"]
    rel["r2l.J_z_hat_eq_z"] = bool(np.array_equal(ctx.R.j_apply_int(r2l["z_hat"], P.b, P.tau), r2l["z"]))
    rel["r2l.z_eq_sum_c_s_plus_sum_masks"] = check_z_decomposition(ctx, states, r2l)
    rel["r2l.M_z'_eq_y (Sage)"] = check_linear_relation(ctx, r2l["M"], r2l["z_prime"], r2l["y_vec"])
    rel.update(row_block_checks(ctx, pp, states, conv, u_com, r2l))
    T["ring_to_linear"] = time.time() - t

    t = time.time()
    tr, priv = sc.prove(ctx, r2l, conv, E_mat, coins)
    T["sumcheck_prove"] = time.time() - t
    t = time.time()
    r_shares = priv["r_shares"]
    r_sum = r_shares.sum(axis=0) % P.q
    ok_hi, ok_rel = sc.check_quotient(ctx, r2l["M"], r2l["z_prime"], r2l["y_vec"], r_sum)
    rel["sc.quotient_high_half_of_sum_is_0"] = ok_hi
    rel["sc.(X^d+1) r = M z' - y over Z_q[X] (Sage)"] = ok_rel
    rel["sc.quotient_share_1_matches_literal_formula (Sage)"] = sc.check_quotient_formula(
        ctx, r2l["M"], r2l["z_shares"][0], r2l["y_shares"][0], r_shares[0])
    hc = sc.hypercube_checks(ctx, r2l, r_shares, E_mat, ctx.embed(inp["y"]), coins)
    rel["sc.zero_check_P_p(x)=0_on_hypercube"] = hc["zero_check_all_points"]
    rel["sc.sum_x sum_p w~_p(x) M*(x) = a"] = hc["linear_sum_equals_a"]
    # degree audit of the unblinded round polynomials (the draft sizes messages by d0, d1)
    degs0 = [sc.degree_at_nodes(ctx.F, fi["F0_only"]) for fi in priv["fah_inputs"]]
    degs1 = [sc.degree_at_nodes(ctx.F, fi["F1_only"]) for fi in priv["fah_inputs"]]
    rel["sc.deg(sum_p F_0 round poly) <= d0"] = max(degs0) <= P.d0
    rel["sc.deg(sum_p F_1 round poly) <= d1"] = max(degs1) <= P.d1
    hc["round_degrees_F0"] = degs0
    hc["round_degrees_F1"] = degs1
    T["relation_checks"] = time.time() - t

    t = time.time()
    public = verifier_public(ctx, pp, u_com, inp["x"], inp["y"], r2l["attempts"])
    rel["verifier.M_equals_provers_M"] = bool(np.array_equal(public["M"], r2l["M"]))
    pcs = sc.IdealPCS(ctx, priv)
    vchecks, vinfo = sc.verify(ctx, public, tr, coins, pcs)
    conv_ok = verify_conv(ctx, inp["y"], conv["E_u"])
    vall = OrderedDict([("conv_y_eq_E(u)", conv_ok)])
    vall.update(vchecks)
    rel["sc.mstar_table_mle(a) = factorized M~*(a)"] = ctx.F.eq(
        ctx.F.mle_eval(priv["MST_table"], vinfo["a"]), vinfo["mstar_at_a"])
    rel["sc.a_claim(V) = a(provers)"] = ctx.F.eq(vinfo["a_claim"], hc["a"])
    T["verify"] = time.time() - t
    T["total"] = time.time() - t_start
    return dict(ctx=ctx, pp=pp, inp=inp, u_com=u_com, states=states, cdata=cdata, open_info=open_info,
                conv=conv, E_mat=E_mat, r2l=r2l, coins=coins, transcript=tr, private=priv, public=public,
                hypercube=hc, relations=rel, verifier_checks=vall, verifier_accept=all(vall.values()),
                verifier_info=vinfo, timings=T)


def check_z_decomposition(ctx, states, r2l):
    """z = sum_{i'} c_{i'} s_{i'} + sum_p r_p over the integers (5_BB:130), recomputed block by block."""
    from .r2l import int_ring_mul_rows
    P, R = ctx.P, ctx.R
    acc = r2l["attempts"][-1]
    tot = np.zeros((P.width_A, P.d), dtype=np.int64)
    for p in states:
        for i, st in states[p].items():
            tot += int_ring_mul_rows(acc["c"][i], R.centered(st["s"]), P.d)
    for p in range(1, P.n + 1):
        tot += acc["provers"][p]["mask"]
    return bool(np.array_equal(tot, r2l["z"]))


def row_block_checks(ctx, pp, states, conv, u_com, r2l):
    """The meaning of each row block of M (5_BB:182), checked directly on the unpacked values."""
    P, R = ctx.P, ctx.R
    out = OrderedDict()
    acc = r2l["attempts"][-1]
    q = P.q
    # row 3: b^T G w_hat = u
    lhs = R.zero()
    for i in range(P.n_blocks):
        wi = R.g_apply(r2l["w_hat"][i * P.delta:(i + 1) * P.delta], P.b, P.delta)[0]
        lhs = (lhs + R.mul(r2l["b_ring"][i], wi)) % q
    out["r2l.row_b: sum_i b_i w_i = u"] = bool(np.array_equal(lhs, conv["u"]))
    # row 4: sum c_i w_i + sum_p w0_p = a^T G z
    lhs = R.zero()
    for i in range(P.n_blocks):
        lhs = (lhs + R.mul(R.reduce(acc["c"][i]), r2l["w"][i])) % q
    for p in range(1, P.n + 1):
        lhs = (lhs + acc["provers"][p]["w0"]) % q
    g = R.g_apply(R.reduce(r2l["z"]), P.b, P.delta)
    rhs = R.zero()
    for j in range(P.block_len):
        rhs = (rhs + R.mul(r2l["a_ring"][j], g[j])) % q
    out["r2l.row_c: sum c_i w_i + sum w0_p = a^T G z"] = bool(np.array_equal(lhs, rhs))
    # row 5: sum c_i t_i + sum t0_p = A z
    lhs = R.zero(P.n_A)
    for p in states:
        for i, st in states[p].items():
            for a_ in range(P.n_A):
                lhs[a_] = (lhs[a_] + R.mul(R.reduce(acc["c"][i]), st["t"][a_])) % q
    for p in range(1, P.n + 1):
        lhs = (lhs + acc["provers"][p]["t0"]) % q
    out["r2l.row_A: sum c_i t_i + sum t0_p = A z"] = bool(np.array_equal(lhs, R.matvec(pp["A"], R.reduce(r2l["z"]))))
    return out
