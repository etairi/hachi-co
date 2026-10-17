"""Setup (matrices), the instance input, AS.Share, PConv, Commit and Open.

Draft references: Setup 6_CH:263-298; Commit 6_CH:300-391; Open 6_CH:393-406; PConv F2R:36-87;
AS.Share 4_P:150-164; Slice 4_P:167-183; F_AH 4_P:185-201.

Indexing (m0-spec "Bit strings"): nat(b_1..b_s) = sum_j b_j 2^{s-j}, b_1 most significant, values in
[0, 2^s).  The draft's nat (4_P:7) is 1-indexed and LSB-first; we use the m0-spec convention throughout,
so the draft's 1-indexed position i corresponds to our 0-indexed i - 1, and prover p (1-indexed) owns the
blocks whose top ceil(log n) bits have nat = p - 1.
"""
import numpy as np

from .prg import Prg


# ---------------------------------------------------------------- Setup ----
def expand_matrix(seed, domain, rows, cols, d, q):
    """Entry (i, j) is d residues uniform mod q from its own stream Prg(seed, domain, i*cols + j)."""
    out = np.zeros((rows, cols, d), dtype=np.int64)
    for i in range(rows):
        for j in range(cols):
            out[i, j] = Prg(seed, domain, i * cols + j).mod_q(q, d)
    return out


def setup(ctx):
    """A in R_q^{n_A x delta 2^m}, B in R_q^{n_B x n_A delta 2^r}, D in R_q^{n_D x width_D} (6_CH:283-288).
    The base-PCS parameters (6_CH:289-292) are omitted: Hachi as the base PCS is deferred."""
    P = ctx.P
    return dict(
        A=expand_matrix(ctx.setup_seed, "setup/A", P.n_A, P.width_A, P.d, P.q),
        B=expand_matrix(ctx.setup_seed, "setup/B", P.n_B, P.width_B, P.d, P.q),
        D=expand_matrix(ctx.setup_seed, "setup/D", P.n_D, P.width_D, P.d, P.q),
    )


# ---------------------------------------------------------------- input ----
def random_field_vec(F, prg, count):
    return np.array([prg.mod_q(F.q, F.k) for _ in range(count)], dtype=np.int64).reshape(count, F.k)


def as_share(F, x, n, prg):
    """AS.Share(x, n) (4_P:154): shares 1..n-1 uniform, share n = x - sum.  x has shape (len, k)."""
    shares = np.zeros((n,) + x.shape, dtype=np.int64)
    for i in range(n - 1):
        shares[i] = random_field_vec(F, prg, x.shape[0])
    shares[n - 1] = (x - shares[:n - 1].sum(axis=0)) % F.q
    return shares


def sample_input(ctx):
    """f (coefficients, nat(q) order, X_1 <-> most significant bit), the initial additive shares f_i of f
    held by the provers (6_CH:301), the evaluation point x, and y = f(x) computed with subfield.py."""
    P, F, S = ctx.P, ctx.F, ctx.S
    f = random_field_vec(F, Prg(ctx.seed, "input/f"), 2 ** P.ell)
    f_init = as_share(F, f, P.n, Prg(ctx.seed, "input/f_init"))
    x = random_field_vec(F, Prg(ctx.seed, "input/x"), P.ell)
    mono = S.monomials([F.to_sub(xi) for xi in x])
    y = F.from_sub(S.inner([F.to_sub(fi) for fi in f], mono))
    return dict(f=f, f_init=f_init, x=x, y=y)


# ---------------------------------------------------------------- PConv ----
def pconv(ctx, f_shares):
    """PConv (F2R:36-87).  f_shares (n, 2^l, k) -> f_hat (2^{l-hat}, k) and f' (2^{l'}, d).

    f_hat[nat(p || q)] = <f_q>_{nat(p)+1} if nat(p) < n else 0, nat(p||q) = nat(p) 2^l + nat(q);
    f'[nat(i||j)] = psi((f_hat[nat(i||j||k)])_{k in {0,1}^{alpha-kappa}}).
    Each prover can compute exactly the blocks it owns from its own share; we compute all blocks at once and
    the per-block owner is recorded separately (block i is all-zero iff it has no owner)."""
    P, F, S = ctx.P, ctx.F, ctx.S
    lhat = P.logn + P.ell
    f_hat = F.zeros(2 ** lhat)
    for p in range(1, P.n + 1):
        f_hat[(p - 1) * 2 ** P.ell: p * 2 ** P.ell] = f_shares[p - 1]
    g = 2 ** (P.alpha - P.kappa)       # = d/k
    assert lhat - P.ell_prime == P.alpha - P.kappa
    f_prime = np.zeros((2 ** P.ell_prime, P.d), dtype=np.int64)
    for t in range(2 ** P.ell_prime):
        chunk = [F.to_sub(f_hat[t * g + kk]) for kk in range(g)]
        f_prime[t] = [int(c) for c in S.psi(chunk)]
    return f_hat, f_prime


# --------------------------------------------------------------- Commit ----
def commit(ctx, pp, f_init):
    """Commit<P_1(f_1), ..., P_n(f_n)>(pp) (6_CH:300-391).  Returns (u, states, data).

    states[p] = {block i: dict(f_prime, s, t, t_hat)} for the blocks prover p owns (6_CH:390)."""
    P, R, F = ctx.P, ctx.R, ctx.F
    n = P.n
    # step 1-2: resharing by AS.Share, then local sums
    resh = np.zeros((n, n) + f_init.shape[1:], dtype=np.int64)       # resh[i][j] = <f_i>_j
    for i in range(n):
        resh[i] = as_share(F, f_init[i], n, Prg(ctx.seed, "commit/reshare", i))
    f_shares = resh.sum(axis=0) % F.q                                 # <f>_j = sum_i <f_i>_j
    # step 3: PConv
    f_hat, f_prime = pconv(ctx, f_shares)
    # step 4: local inner commitments of the owned blocks
    A, B = pp["A"], pp["B"]
    states = {p: {} for p in range(1, n + 1)}
    bl = P.block_len
    for i in range(P.n_blocks):
        p = P.owner_of_block(i)
        fpi = f_prime[i * bl:(i + 1) * bl]
        if p is None:
            assert not fpi.any(), "an unowned block must be zero"
            continue
        s = R.g_inv(fpi, P.b, P.delta)                    # G^{-1}_{2^m}(f'_i), (2^m delta, d)
        t = R.matvec(A, s)                                # A s_i, (n_A, d)
        t_hat = R.g_inv(t, P.b, P.delta)                  # G^{-1}_{n_A}(t_i), (n_A delta, d)
        states[p][i] = dict(f_prime=fpi, s=s, t=t, t_hat=t_hat)
    # step 5: slices of t_hat and F_AH(B)
    w = P.n_A * P.delta
    slices = np.zeros((n, P.n_blocks * w, P.d), dtype=np.int64)
    for p in states:
        for i, st in states[p].items():
            slices[p - 1, i * w:(i + 1) * w] = st["t_hat"]
    t_hat_all = slices.sum(axis=0) % P.q
    u = R.matvec(B, t_hat_all)
    data = dict(reshare=resh, f_shares=f_shares, f_hat=f_hat, f_prime=f_prime, t_hat_slices=slices,
                t_hat=t_hat_all, u=u)
    return u, states, data


# ----------------------------------------------------------------- Open ----
def open_check(ctx, pp, states, u, f, c=None):
    """Open(pp, st, comm, f, c) (6_CH:393-406), with every relation checked in Sage arithmetic.

    c defaults to c* = (1, ..., 1).  Returns (bit, checks) where checks names every sub-condition."""
    P, R, F, S, SR = ctx.P, ctx.R, ctx.F, ctx.S, ctx.SR
    nb, bl = P.n_blocks, P.block_len
    if c is None:
        c = np.zeros((nb, P.d), dtype=np.int64)
        c[:, 0] = 1
    checks = {}
    # slack space Z: ||c_i||_1 <= omega_bar and c_i invertible
    cc = R.centered(c)
    checks["c_in_slack_space"] = bool(all(int(np.abs(cc[i]).sum()) <= P.omega_bar and SR.is_unit(c[i])
                                          for i in range(nb)))
    # 1. assemble
    s_list, th_list, several = [], [], False
    zero_s = np.zeros((bl * P.delta, P.d), dtype=np.int64)
    zero_t = np.zeros((P.n_A * P.delta, P.d), dtype=np.int64)
    for i in range(nb):
        holders = [p for p in states if i in states[p]]
        if len(holders) > 1:
            several = True
        if holders:
            s_list.append(states[holders[0]][i]["s"])
            th_list.append(states[holders[0]][i]["t_hat"])
        else:
            s_list.append(zero_s)
            th_list.append(zero_t)
    checks["no_block_in_several_states"] = not several
    t_hat = np.concatenate(th_list)
    # 2. decode the message: f'_{i||j} = G s_i, f_hat = psi^{-1}, f_bar_q = sum_p f_hat_{p||q}
    g = 2 ** (P.alpha - P.kappa)
    f_hat = F.zeros(2 ** (P.logn + P.ell))
    for i in range(nb):
        fpi = R.g_apply(s_list[i], P.b, P.delta)
        for j in range(bl):
            dec = S.psi_inv(S.ring([int(x) for x in fpi[j]]))
            for kk in range(g):
                f_hat[(i * bl + j) * g + kk] = F.from_sub(dec[kk])
    f_bar = F.zeros(2 ** P.ell)
    for p in range(2 ** P.logn):
        f_bar = F.add(f_bar, f_hat[p * 2 ** P.ell:(p + 1) * 2 ** P.ell])
    # 3. conditions
    A, B = pp["A"], pp["B"]
    ok_a = True
    norm_cs = 0
    for i in range(nb):
        lhs = SR.matvec(A.tolist(), s_list[i].tolist())
        rhs = R.g_apply(th_list[i], P.b, P.delta).tolist()
        ok_a &= (lhs == rhs)
        # ||c_i s_i||_inf of the R_q product, centered (4_P:86)
        cs = np.array([R.mul(c[i], s_list[i][e]) for e in range(s_list[i].shape[0])])
        norm_cs = max(norm_cs, R.inf_norm(cs))
    checks["A_s_eq_G_that"] = bool(ok_a)
    checks["norm_c_s_le_beta_bar"] = bool(norm_cs <= P.beta_bar)
    checks["B_that_eq_u"] = SR.matvec(B.tolist(), t_hat.tolist()) == np.asarray(u).tolist()
    checks["norm_that_le_gamma_bar"] = bool(R.inf_norm(t_hat) <= P.gamma_bar)
    checks["f_eq_f_bar"] = bool(np.array_equal(f_bar % P.q, np.asarray(f) % P.q))
    bit = all(checks.values())
    return bit, dict(checks=checks, max_norm_c_s=int(norm_cs), max_norm_t_hat=R.inf_norm(t_hat))
