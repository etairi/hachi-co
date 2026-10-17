"""JSON fixtures for the Rust milestone-1 code.

Encoding (also in ref/README.md):
  * residue: integer in [0, q); short integer: signed integer;
  * R_q element: list of d residues, index j <-> X^j;  short ring element (masks, c, z, rho): d signed ints;
  * Z_q[X] polynomial of degree < 2d (quotient shares): list of 2d residues;
  * F_{q^k} element: list of k residues, the coordinates (a_0, ..., a_{k-1}) in Hachi's basis
    a_0 + sum_m a_m (t^m - t^{2k-m}), t = X^{d/(2k)};
  * vectors/matrices: nested lists, outermost index first;
  * large derived objects are given as SHA-256 digests of the concatenation of their residues as
    little-endian u64, in row-major order of the stated shape.
"""
import hashlib
import json
import platform
from collections import OrderedDict

import numpy as np

from .prg import l1ball_count


def L(a):
    return np.asarray(a, dtype=np.int64).tolist()


def digest(a, q):
    arr = (np.asarray(a, dtype=np.int64) % q).astype("<u8")
    return hashlib.sha256(arr.tobytes()).hexdigest()


def ranges(pos):
    out = []
    for p in pos:
        if out and out[-1][1] == p:
            out[-1][1] = p + 1
        else:
            out.append([p, p + 1])
    return out


CONVENTIONS = OrderedDict([
    ("indexing", "All positions are 0-indexed. The draft's 1-indexed position u (z', y, rows i' of M~, the "
                 "hypercube coordinates bit(u), bit(l)) is our u - 1. Provers are numbered 1..n."),
    ("bit_strings", "nat(b_1..b_s) = sum_j b_j 2^{s-j} (b_1 most significant; m0-spec). Coefficient vectors of "
                    "multilinear polynomials are ordered by nat: f[nat(q)] multiplies prod_j X_j^{q_j}, X_1 <-> the "
                    "most significant bit. monomials(x)[nat(b)] = prod_j x_j^{b_j} (subfield.py)."),
    ("field", "F_{q^k} = R_q^H in Hachi's basis (Hachi Eq. (7)); an element is (a_0..a_{k-1}); "
              "multiplication is that of R_q (subfield.py)."),
    ("psi_E", "psi, psi^{-1}, E exactly as ref/hachico_ref/subfield.py; E_matrix[i][j] = coefficient i of E(X^j)."),
    ("gadgets", "Digits balanced in [ceil(-b/2), ceil(b/2)-1]. G_{b,n} = I_n (x) (1, b, .., b^{delta-1}), "
                "element-major: entry i's digits at i*delta .. i*delta+delta-1 (digit e has weight b^e); J the same "
                "with tau digits. G^{-1}(x) decomposes rep(x) = x if x <= top else x - q, top = (ceil(b/2)-1)"
                "(b^delta-1)/(b-1) (legacy crates/hachi to_window rule). J^{-1}(z) decomposes the integer z."),
    ("blocks", "f' has 2^r blocks of 2^m ring elements, f'[i*2^m + j]; block i is owned by prover "
               "(i >> (r - ceil(log n))) + 1 if that is <= n, else it is all-zero and owned by nobody."),
    ("pconv", "f_hat[nat(p) 2^l + nat(q)] = <f>_{nat(p)+1}[nat(q)] for nat(p) < n, else 0; "
              "f'[t] = psi(f_hat[t*(d/k) .. t*(d/k) + d/k - 1])."),
    ("commit_state", "s_i = G^{-1}_{2^m}(f'_i) (shape 2^m*delta x d), t_i = A s_i, t_hat_i = G^{-1}_{n_A}(t_i); "
                     "t_hat = (t_hat_0, .., t_hat_{2^r-1}) with zero blocks for unowned i."),
    ("conv", "x' = 1^{ceil(log n)} || x[0 : l-alpha+kappa]; b = monomials(x'[0:r]); a = monomials(x'[r:l']); "
             "v_point = psi(monomials(x[l-alpha+kappa : l])); d[nat(i) 2^m + nat(j)] = b_i a_j; "
             "<u>_p = sum over p's blocks of d_t f'_t; F_AH(E) outputs E(u); u itself stays shared."),
    ("z_prime_layout", "z' = [w_hat (2^r delta) | w0_hat (n delta) | t0_hat (n n_A delta) | rho (n m_rho) | "
                       "t_hat (2^r n_A delta) | z_hat (2^m delta tau) | <u> (1)]; z_hat entry (j*delta+e)*tau+t is "
                       "digit t of z[j*delta+e]."),
    ("M_rows", "rows of M: D (n_D), B (n_B), b^T G (1), c^T(x)G_1 | 1_n^T(x)G_1 | -a^T G J (1), "
               "1_n^T(x)G_{n_A} | c^T(x)G_{n_A} | -A J (n_A); y = (v, u_com, 0, 0, 0)."),
    ("shares", "<z'>_p equals z' on the positions S_p and 0 elsewhere, except position mu-1 (<u>), which holds "
               "<u>_p. Prover 1 holds y; the other provers hold 0. S_p: p's blocks of w_hat and t_hat, its own "
               "w0_hat, t0_hat, rho, and (p = 1) z_hat. Positions of unowned zero blocks are in no S_p."),
    ("quotient", "<r>_p = 1/2 (1 - X^d)(M <z'>_p - {y}_p) mod X^{2d}+1 with entries read as degree < d "
                 "polynomials; stored as 2d residues."),
    ("hypercube", "V = (U, L), |U| = ell_U = ceil(log(mu+n')), |L| = ell_L = log 2d; point index "
                  "nat(V) = u0 * 2d + l0 with V_1 the most significant bit. <w~>_p table: (u0 < mu, l0 < d) -> "
                  "<z'>_p[u0] coefficient l0; (mu <= u0 < mu+n', any l0) -> <r>_p[u0-mu] coefficient l0; else 0."),
    ("eq", "eq_table(tau)[nat(x)] = prod_j (tau_j x_j + (1-tau_j)(1-x_j)), tau_1 pairs with the most significant "
           "bit. eq(tau_0, (U, L, bit(i))) uses tau_0[0:ell_pp] for (U, L) and tau_0[ell_pp:] for the "
           "ceil(log n) bits of i-1. eq(tau_1, bit(i')) is eq_table(tau_1)[i'-1]."),
    ("M_tilde", "M*(u0, l0) = sum_{i'} eq(tau_1, bit(i')) M~_alpha(bit(i'), u0, l0): (u0 < mu) c_u alpha^{l0} for all "
                "l0 < 2d with c_u = sum_{i0 < n'} eq1[i0] M[i0][u0](alpha); (mu <= u0 < mu+n') -eq1[u0-mu] "
                "(alpha^d + 1) alpha^{l0}; plus at u0 = mu-1, l0 < d: sum_{j0<d} eq1[n'+j0] E[j0][l0]; plus for "
                "u0 < mu+n', l0 >= d: eq1[n' + d + u0 + (l0-d)(mu+n')]."),
    ("F0", "F_{0,p} = eq(tau_0,(U,L,bit(p))) [ e_p prod_{c=-b+1}^{b-1} (<w~>_p - c) + (1 - e_p - ind) <w~>_p ], "
           "e_p = indicator of S_p (rows), ind = indicator of rows mu-1 .. mu+n'-1 (the draft's mu .. mu+n')."),
    ("rounds", "round r binds V_r; messages are H_{0,r}(t) for t = 1..d0+1 and H_{1,r}(t) for t = 1..d1+1 "
               "(the F_AH(I) outputs); V interpolates H(0) and H(a_r)."),
    ("blinders", "g(V) = s0 + sum_{j=1}^{ell_pp} sum_{k=1}^{D} s[j-1][k-1] V_j^k, D = d0 for g0 and d1 for g1."),
    ("challenges", "c_i uniform in {c : ||c||_1 <= omega} (exact sampler, prg.py); alpha, rho, tau0, tau1, a_r "
                   "uniform in F_{q^k}; all verifier coins come from the instance seed (public coins)."),
    ("retries", "An abort ends Eval with (bot, bot); the caller restarts it. Pi_conv is deterministic, so the "
                "oracle re-runs only the ring-to-linear step with fresh masks, rho and c (attempt index)."),
    ("base_pcs", "Deferred: commitments are not modelled; openings are checked by evaluating the committed "
                 "polynomials directly (IdealPCS)."),
])


def build_fixture(res, neg, command, selection=None):
    ctx = res["ctx"]
    P, q = ctx.P, ctx.P.q
    r2l, tr, priv, conv = res["r2l"], res["transcript"], res["private"], res["conv"]
    fx = OrderedDict()
    fx["format"] = "hachi-co-ref-fixture/v1"
    fx["preset"] = P.name
    import sage.version
    fx["generated_by"] = OrderedDict([
        ("command", command), ("sage", sage.version.version), ("python", platform.python_version()),
        ("numpy", np.__version__), ("draft", "collaborative-hatchi commit 625ced1"),
    ])
    fx["conventions"] = CONVENTIONS
    pd = P.as_dict()
    pd["acceptance_per_prover"] = P.acceptance_per_prover()
    pd["abort_probability"] = 1 - P.acceptance_per_prover() ** P.n
    pd["l1ball_size"] = int(l1ball_count(P.d, P.omega))
    fx["params"] = pd
    fx["seeds"] = OrderedDict([("instance", ctx.seed.decode()), ("setup", ctx.setup_seed.decode()),
                               ("prg", "SHAKE128 counter mode with domain separation, ref/hachico_ref/prg.py")])
    if selection is not None:
        fx["seeds"]["selection"] = selection
    fx["setup"] = OrderedDict([("A", L(res["pp"]["A"])), ("B", L(res["pp"]["B"])), ("D", L(res["pp"]["D"]))])
    inp = res["inp"]
    fx["input"] = OrderedDict([("f", L(inp["f"])), ("f_init_shares", L(inp["f_init"])), ("x", L(inp["x"])),
                               ("y", L(inp["y"])), ("y_ring", L(ctx.embed(inp["y"])))])
    cd = res["cdata"]
    owner = [P.owner_of_block(i) or 0 for i in range(P.n_blocks)]
    blocks = []
    for i in range(P.n_blocks):
        p = owner[i]
        if p:
            st = res["states"][p][i]
            blocks.append(OrderedDict([("index", i), ("owner", p), ("s", L(st["s"])), ("t", L(st["t"])),
                                       ("t_hat", L(st["t_hat"]))]))
    fx["commit"] = OrderedDict([
        ("reshare", L(cd["reshare"])), ("f_shares", L(cd["f_shares"])), ("f_hat", L(cd["f_hat"])),
        ("f_prime", L(cd["f_prime"])), ("block_owner", owner), ("blocks", blocks), ("t_hat", L(cd["t_hat"])),
        ("u", L(res["u_com"])),
    ])
    oi = res["open_info"]
    fx["open"] = OrderedDict([("c", "c* = (1, ..., 1)"), ("bit", all(oi["checks"].values())),
                              ("checks", oi["checks"]), ("max_norm_c_s", oi["max_norm_c_s"]),
                              ("max_norm_t_hat", oi["max_norm_t_hat"])])
    fx["conv"] = OrderedDict([
        ("x_prime", L(conv["x_prime"])), ("b", L(conv["b"])), ("a", L(conv["a"])), ("v_point", L(conv["v_point"])),
        ("d", L(conv["d"])), ("u_shares", L(conv["u_shares"])), ("u_inside_F_AH", L(conv["u"])),
        ("E_u", L(conv["E_u"])), ("E_matrix", L(res["E_mat"])),
        ("verifier_y_eq_E_u", res["verifier_checks"]["conv_y_eq_E(u)"]),
    ])
    atts = []
    for a in r2l["attempts"]:
        provs = []
        for p in range(1, P.n + 1):
            pr = a["provers"][p]
            provs.append(OrderedDict([
                ("prover", p), ("mask", L(pr["mask"])), ("rho", L(pr["rho"])), ("t0", L(pr["t0"])),
                ("t0_hat", L(pr["t0_hat"])), ("w0", L(pr["w0"])), ("w0_hat", L(pr["w0_hat"])),
                ("fold_norm", pr["fold_norm"]), ("z_norm", pr["z_norm"]), ("accept", pr["accept"])]))
        atts.append(OrderedDict([("attempt", a["attempt"]), ("provers", provs), ("v", L(a["v"])), ("c", L(a["c"])),
                                 ("c_in_C", a["c_in_C"]), ("abort", a["abort"]),
                                 ("aborting_provers", a["aborting_provers"])]))
    acc = r2l["attempts"][-1]
    lay = r2l["layout"]
    fx["ring_to_linear"] = OrderedDict([
        ("w", L(r2l["w"])), ("w_hat", L(r2l["w_hat"])), ("attempts", atts),
        ("accepted_attempt", r2l["accepted_attempt"]),
        ("z_i", [L(acc["provers"][p]["z"]) for p in range(1, P.n + 1)]),
        ("z", L(r2l["z"])), ("z_norm", r2l["z_norm"]), ("z_norm_le_beta", r2l["z_norm_ok"]),
        ("layout", OrderedDict([("blocks", [list(t) for t in lay.blocks()]),
                                ("rows", OrderedDict([("D", lay.row_D), ("B", lay.row_B), ("b", lay.row_b),
                                                      ("c", lay.row_c), ("A", lay.row_A)]))])),
        ("partition", OrderedDict((str(p), ranges(v)) for p, v in r2l["partition"].items())),
        ("unheld", ranges(r2l["unheld"])),
        ("z_prime", L(r2l["z_prime"])),
        ("y", L(r2l["y_vec"])),
        ("M_shape", list(r2l["M"].shape)), ("M_sha256", digest(r2l["M"], q)),
        ("M_nonzero_ring_entries", int(np.count_nonzero(r2l["M"].any(axis=2)))),
    ])
    ch = priv["coins"]
    rounds = []
    for r, (m, fi) in enumerate(zip(tr["rounds"], priv["fah_inputs"]), start=1):
        rounds.append(OrderedDict([("round", r), ("H0", L(m["H0"])), ("H1", L(m["H1"])),
                                   ("F_AH_inputs_H0", L(fi["H0"])), ("F_AH_inputs_H1", L(fi["H1"]))]))
    rs = priv["r_shares"]
    fx["sumcheck"] = OrderedDict([
        ("quotient_shares", L(rs)), ("quotient", L((rs.sum(axis=0) % q)[:, :P.d])),
        ("blinders", [OrderedDict([("g0", OrderedDict([("s0", L(priv["g0"][p].s0)), ("s", L(priv["g0"][p].s))])),
                                   ("g1", OrderedDict([("s0", L(priv["g1"][p].s0)), ("s", L(priv["g1"][p].s))]))])
                      for p in range(P.n)]),
        ("G0", L(tr["G0"])), ("G1", L(tr["G1"])),
        ("coins", OrderedDict([("alpha", L(ch["alpha"])), ("rho", L(ch["rho"])), ("tau0", L(ch["tau0"])),
                               ("tau1", L(ch["tau1"])), ("a", L(priv["a"]))])),
        ("a_claim", L(res["verifier_info"]["a_claim"])),
        ("rounds", rounds),
        ("final", OrderedDict([("w", L(tr["final"]["w"])), ("g0", L(tr["final"]["g0"])),
                               ("g1", L(tr["final"]["g1"]))])),
        ("mstar_at_a", L(res["verifier_info"]["mstar_at_a"])),
        ("round_degrees_unblinded", OrderedDict([("F0", res["hypercube"]["round_degrees_F0"]),
                                                 ("F1", res["hypercube"]["round_degrees_F1"])])),
        ("digests", OrderedDict(
            [("w_table_p%d" % (p + 1), digest(priv["W_base"][p], q)) for p in range(P.n)]
            + [("mstar_table", digest(priv["MST_table"], q)), ("eq_tau1_table", digest(priv["eq1"], q))])),
        ("digest_shapes", OrderedDict([("w_table", [2 ** P.ell_pp]), ("mstar_table", [2 ** P.ell_pp, P.k]),
                                       ("eq_tau1_table", [2 ** P.log_nu, P.k])])),
    ])
    fx["verifier"] = OrderedDict([("checks", res["verifier_checks"]), ("accept", res["verifier_accept"])])
    fx["relations"] = res["relations"]
    fx["negative_tests"] = neg
    return fx


def _dump(obj, indent=0):
    """Dicts indented one key per line; lists (the data) compact on one line."""
    pad = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = ['%s  %s: %s' % (pad, json.dumps(str(k)), _dump(v, indent + 1)) for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        items = ["%s  %s" % (pad, _dump(v, indent + 1)) for v in obj]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(obj, separators=(",", ":"))


def write_fixture(fx, path):
    text = _dump(fx) + "\n"
    json.loads(text)        # round-trip check
    with open(path, "w") as fh:
        fh.write(text)
    return len(text)
