"""Pi_conv, the collaborative reduction from R_eval,F_{q^k} to R_eval,R_q (F2R:89-169).

Choices (see ref/README.md):
  * PConv already ran in Commit; Pi_conv reuses f' from the provers' states instead of running PConv again
    (F2R:124-126 runs it again; the output is identical because PConv is deterministic).
  * u stays secret-shared: the provers hold <u>_p = d^T {f'}_p and only E(u) is opened through F_AH(E)
    (the author comment at F2R:100; 5_BB:171 keeps <u> in the witness).  The "public" u of F2R:100/166
    is therefore not output.
"""
import numpy as np


def public_vectors(ctx, x):
    """x' = 1^{ceil(log n)} || (x_1..x_{l-alpha+kappa}), b, a, v, d (F2R:111-122), all from public data.

    b[nat(i)] = prod_j x'_j^{i_j} over the first r coordinates, a[nat(j)] over the next m (monomials in the
    m0-spec bit order), v = psi(monomials(x_{l-alpha+kappa+1}, ..., x_l)), d[nat(i) 2^m + nat(j)] = b_i a_j."""
    P, F, S = ctx.P, ctx.F, ctx.S
    lt = P.ell - P.alpha + P.kappa
    xs = [F.to_sub(xi) for xi in x]
    one = S.fone()
    x_prime = [one] * P.logn + xs[:lt]
    assert len(x_prime) == P.ell_prime
    b_vec = S.monomials(x_prime[:P.r])
    a_vec = S.monomials(x_prime[P.r:])
    v_point = S.psi(S.monomials(xs[lt:]))
    d_vec = [S.fmul(b_vec[i], a_vec[j]) for i in range(2 ** P.r) for j in range(2 ** P.m)]
    arr = lambda L: np.array([[int(c) for c in e] for e in L], dtype=np.int64)
    return dict(x_prime=arr(x_prime), b=arr(b_vec), a=arr(a_vec), v_point=np.array([int(c) for c in v_point]),
                d=arr(d_vec))


def e_matrix(ctx, v_point):
    """The d x d matrix over Z_q with cf(E(u)) = E cf(u) (subfield.py E_matrix; E's row/column j <-> X^j)."""
    S = ctx.S
    M = S.E_matrix(S.ring([int(c) for c in v_point]))
    return np.array([[int(c) for c in row] for row in M], dtype=np.int64)


def pi_conv(ctx, states, x):
    """Provers' side: <u>_p = d^T {f'}_p; F_AH(E) on <u> gives E(u) (sent to V)."""
    P, R = ctx.P, ctx.R
    pub = public_vectors(ctx, x)
    d_ring = np.array([ctx.embed(dv) for dv in pub["d"]])        # d_t embedded in R_q (in R_q^H)
    bl = P.block_len
    u_sh = np.zeros((P.n, P.d), dtype=np.int64)
    for p in states:
        acc = R.zero()
        for i, st in states[p].items():
            for j in range(bl):
                acc = (acc + R.mul(d_ring[i * bl + j], st["f_prime"][j])) % P.q
        u_sh[p - 1] = acc
    u = u_sh.sum(axis=0) % P.q                                   # inside F_AH(E)
    S = ctx.S
    Eu = np.array([int(c) for c in S.E(S.ring([int(c) for c in u]), S.ring([int(c) for c in pub["v_point"]]))],
                  dtype=np.int64)
    return dict(pub, u_shares=u_sh, u=u, E_u=Eu)


def verify_conv(ctx, y, E_u):
    """V's check y = E(u) (F2R:164), with E(u) interpreted in F_{q^k} through R_q^H."""
    if not ctx.in_subfield(E_u):
        return False
    return bool(np.array_equal(ctx.extract(E_u), np.asarray(y) % ctx.P.q))
