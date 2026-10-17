"""Ring evaluation claims to linear relations over R_q, with masks, rho and rejection sampling
(5_BB:98-193), plus the verifier's public coins.

Layout of z' (5_BB:164-172), 0-indexed positions (the draft's 1-indexed position u is our u - 1):
    [ w_hat (2^r delta) | w0_hat (n delta) | t0_hat (n n_A delta) | rho (n m_rho) | t_hat (2^r n_A delta)
      | z_hat (2^m delta tau) | <u> (1) ],   mu = len(z')  (6_CH:278)
Inside each block the gadget layout is element-major (m0-spec): w_hat_i = G^{-1}(w_i) at i*delta + e;
t_hat_i at i*n_A*delta + a*delta + e; z_hat at (j*delta + e)*tau + t for the entry z[j*delta + e].

Rows of M (5_BB:157-163): [ D (n_D) | B (n_B) | b^T G (1) | c^T (x) G_1 ... (1) | ... A J (n_A) ],
n' = n_A + n_B + n_D + 2.  y = ({v}, {u_com}, 0, 0, 0).
"""
import numpy as np

from .prg import Prg, l1ball_sample


class Layout:
    def __init__(self, P):
        self.P = P
        d, r, m = P.delta, P.r, P.m
        self.w = 0
        self.w0 = self.w + 2 ** r * d
        self.t0 = self.w0 + P.n * d
        self.rho = self.t0 + P.n * P.n_A * d
        self.t = self.rho + P.n * P.m_rho
        self.z = self.t + 2 ** r * P.n_A * d
        self.u = self.z + 2 ** m * d * P.tau
        assert self.u + 1 == P.mu, "layout must match mu of 6_CH:278"
        assert self.t == P.width_D, "D acts on (w_hat, w0_hat, t0_hat, rho)"
        # rows of M
        self.row_D = 0
        self.row_B = P.n_D
        self.row_b = P.n_D + P.n_B
        self.row_c = P.n_D + P.n_B + 1
        self.row_A = P.n_D + P.n_B + 2
        assert self.row_A + P.n_A == P.n_prime

    def blocks(self):
        """Named blocks (name, start, length) for documentation and fixtures."""
        P = self.P
        return [("w_hat", self.w, self.w0 - self.w), ("w0_hat", self.w0, self.t0 - self.w0),
                ("t0_hat", self.t0, self.rho - self.t0), ("rho", self.rho, self.t - self.rho),
                ("t_hat", self.t, self.z - self.t), ("z_hat", self.z, self.u - self.z), ("u", self.u, 1)]

    def partition(self):
        """S_p (5_BB:197) as sorted lists of 0-indexed positions in [0, mu-1), and the positions of all-zero
        blocks owned by nobody (non-empty only when n is not a power of two)."""
        P = self.P
        d = P.delta
        S = {p: [] for p in range(1, P.n + 1)}
        unheld = []
        for i in range(P.n_blocks):
            p = P.owner_of_block(i)
            pos_w = list(range(self.w + i * d, self.w + (i + 1) * d))
            pos_t = list(range(self.t + i * P.n_A * d, self.t + (i + 1) * P.n_A * d))
            (S[p] if p else unheld).extend(pos_w + pos_t)
        for p in range(1, P.n + 1):
            S[p].extend(range(self.w0 + (p - 1) * d, self.w0 + p * d))
            S[p].extend(range(self.t0 + (p - 1) * P.n_A * d, self.t0 + p * P.n_A * d))
            S[p].extend(range(self.rho + (p - 1) * P.m_rho, self.rho + p * P.m_rho))
        S[1].extend(range(self.z, self.u))
        for p in S:
            S[p] = sorted(S[p])
        allpos = sorted(sum(S.values(), []) + unheld)
        assert allpos == list(range(P.mu - 1)), "S_1..S_n and the unheld positions partition [mu-1]"
        return S, sorted(unheld)


# ------------------------------------------------------------ verifier coins ----
class Coins:
    """All verifier randomness (public coins), derived from the instance seed with domain separation.
    Public-coin: none of it depends on the provers' messages."""

    def __init__(self, ctx, tag="verifier"):
        self.ctx = ctx
        self.tag = tag

    def _prg(self, name, index=0):
        return Prg(self.ctx.seed, self.tag + "/" + name, index)

    def challenge(self, attempt):
        """c in C^{2^r}, C = {c in R_q : ||c||_1 <= omega} (5_BB:122-126), uniform; signed coefficients."""
        P = self.ctx.P
        prg = self._prg("r2l/c", attempt)
        return np.array([l1ball_sample(prg, P.d, P.omega) for _ in range(P.n_blocks)], dtype=np.int64)

    def field(self, name, count, index=0):
        F = self.ctx.F
        prg = self._prg(name, index)
        return np.array([prg.mod_q(F.q, F.k) for _ in range(count)], dtype=np.int64).reshape(count, F.k)

    def sumcheck(self):
        P = self.ctx.P
        return dict(alpha=self.field("sc/alpha", 1)[0], rho=self.field("sc/rho", 1)[0],
                    tau0=self.field("sc/tau0", P.ell_pp + P.logn), tau1=self.field("sc/tau1", P.log_nu))

    def round(self, r):
        return self.field("sc/a", 1, r)[0]


# ----------------------------------------------------------------- helpers ----
def rej(v, r, gamma, beta1, check=True):
    """Rej(v, chi_z, chi_r, M; r) for the uniform distributions of 4_P:117-121: z = v + r is output iff
    ||z||_inf <= gamma - beta1 (probability min{chi_z(z)/(M chi_r(r)), 1} is 1 or 0), else None (bot).
    With check=True, ||v||_inf > beta1 raises ValueError (outside the lemma's hypothesis, 4_P:123)."""
    v = np.asarray(v, dtype=np.int64)
    if check and v.size and int(np.abs(v).max()) > beta1:
        raise ValueError("Rej: ||v||_inf > beta1")
    z = v + np.asarray(r, dtype=np.int64)
    return z if (not z.size or int(np.abs(z).max()) <= gamma - beta1) else None


def shift_int(Sm, j, d):
    """X^j * s for integer polynomials s (rows, d), negacyclic, j in [0, d)."""
    out = np.roll(Sm, j, axis=-1)
    out[..., :j] *= -1
    return out


def int_ring_mul_rows(c, Sm, d):
    """c * s_e for every row s_e of Sm (exact integers in Z[X]/(X^d+1)); c is a short integer polynomial."""
    out = np.zeros_like(Sm)
    for j in range(d):
        if c[j]:
            out += int(c[j]) * shift_int(Sm, j, d)
    return out


def a_times_G(ctx, a_ring, vec):
    """a^T G_{2^m} vec for vec in R_q^{delta 2^m} -> R_q."""
    P, R = ctx.P, ctx.R
    g = R.g_apply(vec, P.b, P.delta)             # (2^m, d)
    acc = R.zero()
    for j in range(P.block_len):
        acc = (acc + R.mul(a_ring[j], g[j])) % P.q
    return acc


def build_M(ctx, pp, b_ring, a_ring, c):
    """The public matrix M (5_BB:157-163) as an (n', mu, d) array of ring elements."""
    P, R = ctx.P, ctx.R
    L = Layout(P)
    q, d, b, delta, tau = P.q, P.d, P.b, P.delta, P.tau
    M = np.zeros((P.n_prime, P.mu, d), dtype=np.int64)
    bp = [pow(b, e, q) for e in range(delta + tau)]
    M[L.row_D:L.row_D + P.n_D, 0:P.width_D] = pp["D"]
    M[L.row_B:L.row_B + P.n_B, L.t:L.t + P.width_B] = pp["B"]
    cq = R.reduce(c)
    for i in range(P.n_blocks):
        for e in range(delta):
            M[L.row_b, L.w + i * delta + e] = b_ring[i] * bp[e] % q          # b^T G_{2^r}
            M[L.row_c, L.w + i * delta + e] = cq[i] * bp[e] % q              # c^T (x) G_1
    M[L.row_b, L.u] = R.scalar(-1)
    for p in range(P.n):
        for e in range(delta):
            M[L.row_c, L.w0 + p * delta + e] = R.scalar(bp[e])               # 1_n^T (x) G_1
    for j in range(P.block_len):
        for e in range(delta):
            for t in range(tau):
                M[L.row_c, L.z + (j * delta + e) * tau + t] = (-a_ring[j] * bp[e + t]) % q   # -a^T G J
    for a_ in range(P.n_A):
        row = L.row_A + a_
        for p in range(P.n):
            for e in range(delta):
                M[row, L.t0 + p * P.n_A * delta + a_ * delta + e] = R.scalar(bp[e])        # 1_n^T (x) G_{n_A}
        for i in range(P.n_blocks):
            for e in range(delta):
                M[row, L.t + i * P.n_A * delta + a_ * delta + e] = cq[i] * bp[e] % q       # c^T (x) G_{n_A}
        for e2 in range(P.width_A):
            for t in range(tau):
                M[row, L.z + e2 * tau + t] = (-pp["A"][a_, e2] * bp[t]) % q                # -A J
    return M


def build_y(ctx, v_msg, u_com):
    P = ctx.P
    y = np.zeros((P.n_prime, P.d), dtype=np.int64)
    y[0:P.n_D] = v_msg
    y[P.n_D:P.n_D + P.n_B] = u_com
    return y


# ------------------------------------------------------------------ protocol ----
def ring_to_linear(ctx, pp, states, conv, u_com, coins, max_attempts=64):
    """The provers' side of 5_BB:102-182, looping over attempts (fresh r_i, rho_i, hence fresh v and c) until
    no prover aborts.  Returns a dict with every intermediate value and the attempt log."""
    P, R = ctx.P, ctx.R
    L = Layout(P)
    q, d, delta = P.q, P.d, P.delta
    a_ring = np.array([ctx.embed(x) for x in conv["a"]])
    b_ring = np.array([ctx.embed(x) for x in conv["b"]])
    # step 1: w_i = a^T G_{2^m} s_i, w_hat_i = G_1^{-1}(w_i) for owned blocks
    w = np.zeros((P.n_blocks, d), dtype=np.int64)
    w_hat = np.zeros((P.n_blocks * delta, d), dtype=np.int64)
    for p in states:
        for i, st in states[p].items():
            w[i] = a_times_G(ctx, a_ring, st["s"])
            w_hat[i * delta:(i + 1) * delta] = R.g_inv(w[i][None, :], P.b, delta)
    attempts = []
    for att in range(max_attempts):
        rec = dict(attempt=att, provers={})
        # step 2: masks r_i, rho_i, and their images
        D_in = np.zeros((P.n, P.width_D, d), dtype=np.int64)     # slices of (w_hat, w0_hat, t0_hat, rho)
        for p in range(1, P.n + 1):
            mprg = Prg(ctx.seed, "r2l/mask/p%d" % p, att)
            mask = np.array(mprg.boxed(P.gamma, P.width_A * d), dtype=np.int64).reshape(P.width_A, d)
            rprg = Prg(ctx.seed, "r2l/rho/p%d" % p, att)
            rho = np.array(rprg.short(P.b, P.m_rho * d), dtype=np.int64).reshape(P.m_rho, d)
            mq = R.reduce(mask)
            t0 = R.matvec(pp["A"], mq)
            t0_hat = R.g_inv(t0, P.b, delta)
            w0 = a_times_G(ctx, a_ring, mq)
            w0_hat = R.g_inv(w0[None, :], P.b, delta)
            for i in P.blocks_of(p):
                D_in[p - 1, L.w + i * delta:L.w + (i + 1) * delta] = w_hat[i * delta:(i + 1) * delta]
            D_in[p - 1, L.w0 + (p - 1) * delta:L.w0 + p * delta] = w0_hat
            D_in[p - 1, L.t0 + (p - 1) * P.n_A * delta:L.t0 + p * P.n_A * delta] = t0_hat
            D_in[p - 1, L.rho + (p - 1) * P.m_rho:L.rho + p * P.m_rho] = R.reduce(rho)
            rec["provers"][p] = dict(mask=mask, rho=rho, t0=t0, t0_hat=t0_hat, w0=w0, w0_hat=w0_hat)
        # step 3: F_AH(D) on the slices
        v_msg = R.matvec(pp["D"], D_in.sum(axis=0) % q)
        rec["D_input_slices"] = D_in
        rec["v"] = v_msg
        # step 4: challenge
        c = coins.challenge(att)
        rec["c"] = c
        rec["c_in_C"] = bool(all(int(np.abs(ci).sum()) <= P.omega for ci in c))
        assert rec["c_in_C"]
        # step 5: per-prover fold, Rej
        accepted = {}
        for p in range(1, P.n + 1):
            fold = np.zeros((P.width_A, d), dtype=np.int64)
            for i in P.blocks_of(p):
                fold += int_ring_mul_rows(c[i], R.centered(states[p][i]["s"]), d)
            fnorm = int(np.abs(fold).max())
            bound_text = len(P.blocks_of(p)) * P.omega * (P.b // 2)
            assert fnorm <= bound_text <= P.beta1, "fold norm bound of 5_BB:129"
            mask = rec["provers"][p]["mask"]
            zp = rej(fold, mask, P.gamma, P.beta1)
            ok = zp is not None
            rec["provers"][p].update(fold=fold, fold_norm=fnorm, z=(fold + mask) if ok else None,
                                     accept=bool(ok), z_norm=int(np.abs(fold + mask).max()))
            if ok:
                accepted[p] = zp
        rec["aborting_provers"] = [p for p in range(1, P.n + 1) if not rec["provers"][p]["accept"]]
        rec["abort"] = bool(rec["aborting_provers"])
        attempts.append(rec)
        if rec["abort"]:
            continue
        # steps 5-6: gather at prover 1, norm check, z_hat
        z = sum(accepted.values())
        z_norm = int(np.abs(z).max())
        norm_ok = z_norm <= P.beta
        assert norm_ok
        z_hat = R.j_inv(z, P.b, P.tau)
        assert np.array_equal(R.j_apply_int(z_hat, P.b, P.tau), z)
        break
    else:
        raise RuntimeError("no accepting attempt within %d attempts" % max_attempts)
    acc = attempts[-1]
    # steps 7-9: slices of z', y and the matrix M
    t_hat = np.zeros((P.n_blocks * P.n_A * delta, d), dtype=np.int64)
    for p in states:
        for i, st in states[p].items():
            t_hat[i * P.n_A * delta:(i + 1) * P.n_A * delta] = st["t_hat"]
    zp_all = np.zeros((P.mu, d), dtype=np.int64)
    zp_all[L.w:L.w0] = w_hat
    zp_all[L.w0:L.t0] = np.concatenate([acc["provers"][p]["w0_hat"] for p in range(1, P.n + 1)])
    zp_all[L.t0:L.rho] = np.concatenate([acc["provers"][p]["t0_hat"] for p in range(1, P.n + 1)])
    zp_all[L.rho:L.t] = R.reduce(np.concatenate([acc["provers"][p]["rho"] for p in range(1, P.n + 1)]))
    zp_all[L.t:L.z] = t_hat
    zp_all[L.z:L.u] = R.reduce(z_hat)
    zp_all[L.u] = conv["u"]
    S, unheld = L.partition()
    z_shares = np.zeros((P.n, P.mu, d), dtype=np.int64)
    for p in range(1, P.n + 1):
        z_shares[p - 1, S[p]] = zp_all[S[p]]
        z_shares[p - 1, L.u] = conv["u_shares"][p - 1]
    assert np.array_equal(z_shares.sum(axis=0) % q, zp_all), "shares of z' must add up to z'"
    assert not zp_all[unheld].any()
    y_vec = build_y(ctx, acc["v"], u_com)
    y_shares = np.zeros((P.n, P.n_prime, d), dtype=np.int64)
    y_shares[0] = y_vec                                 # prover 1 holds y; the others hold 0 (5_BB:134)
    M = build_M(ctx, pp, b_ring, a_ring, acc["c"])
    return dict(w=w, w_hat=w_hat, attempts=attempts, accepted_attempt=acc["attempt"], z=z, z_norm=z_norm,
                z_norm_ok=bool(norm_ok), z_hat=z_hat, t_hat=t_hat, z_prime=zp_all, z_shares=z_shares,
                partition=S, unheld=unheld, y_vec=y_vec, y_shares=y_shares, M=M, layout=L,
                a_ring=a_ring, b_ring=b_ring)


def check_linear_relation(ctx, M, z_prime, y_vec):
    """M z' = y over R_q, in Sage arithmetic (independent of the numpy ring code)."""
    return ctx.SR.matvec(M.tolist(), z_prime.tolist()) == np.asarray(y_vec).tolist()
