"""Toy presets and the Setup parameter cascade of the draft.

Source: draft 6_Collaborative_Hachi.tex (6_CH) lines 263-298, with the M0 conventions of docs/m0-spec.md:

  * tau rule "tight" (user decision 2026-10-17): the least tau with
    (ceil(b/2) - 1) * (b^tau - 1)/(b - 1) >= beta.  The draft's tau = ceil(log_b beta) + 1 (6_CH:276) is
    available as tau_rule="draft".
  * q is the smallest prime p = 5 (mod 8) with p > max(16 omega b^tau, 1 + 2^(ell+3) n omega^2 b, 2 beta + 1),
    iterated from q = 2^20 until delta = ceil(log_b q) is stable (m0-spec "Choosing q").

All quantities are exact integers; the rejection constant M is reported through log(M) only.
"""
from dataclasses import dataclass, asdict
import math

from sage.all import Integer, RealField


def ceil_log2(x):
    """Smallest k >= 0 with 2^k >= x (x >= 1).  ceil(log n) of the draft, with ceil(log 1) = 0."""
    assert x >= 1
    k = 0
    while (1 << k) < x:
        k += 1
    return k


def num_digits(q, b):
    """delta = ceil(log_b q): the smallest delta with b^delta >= q."""
    delta, p = 0, 1
    while p < q:
        p *= b
        delta += 1
    return delta


def digit_bounds(b):
    """Balanced digit range [ceil(-b/2), ceil(b/2) - 1] (draft 4_P:88, 96, 100)."""
    lo = -(b // 2)            # ceil(-b/2)
    hi = (b + 1) // 2 - 1     # ceil(b/2) - 1
    return lo, hi


def tight_tau(beta, b):
    """Least tau with (ceil(b/2) - 1) * (b^tau - 1)/(b - 1) >= beta (m0-spec tau rule)."""
    _, hi = digit_bounds(b)
    assert hi >= 1, "b >= 3 needed"
    tau = 0
    while hi * ((b ** tau - 1) // (b - 1)) < beta:
        tau += 1
    return tau


def draft_tau(beta, b):
    """tau = ceil(log_b beta) + 1 (6_CH:276)."""
    t, p = 0, 1
    while p < beta:
        p *= b
        t += 1
    return t + 1


@dataclass(frozen=True)
class Params:
    name: str
    # choices
    n: int
    ell: int
    alpha: int          # log2 d
    kappa: int          # log2 k
    b: int
    omega: int
    n_A: int
    n_B: int
    n_D: int
    m_rho: int
    q: int
    tau_rule: str
    # derived (6_CH:265-281)
    d: int
    k: int
    delta: int
    logn: int           # ceil(log n)
    ell_prime: int      # l' = ceil(log n) + l - alpha + kappa
    r: int
    m: int
    beta1: int
    N: int
    gamma: int
    beta: int
    tau: int
    beta_bar: int
    omega_bar: int
    gamma_bar: int
    mu: int
    n_prime: int
    ell_U: int          # ceil(log(mu + n'))
    ell_L: int          # log(2d)
    ell_pp: int         # l'' = ceil(log(mu + n')) + log(2d)
    d0: int
    d1: int
    nu: int
    log_nu: int         # ceil(log nu), the length of tau_1
    width_A: int        # delta 2^m
    width_B: int        # n_A delta 2^r
    width_D: int        # delta (2^r + n + n n_A) + n m_rho
    ln_M: float         # ln of M = ((2 gamma + 1)/(2 gamma - 2 beta1 + 1))^N

    def as_dict(self):
        return asdict(self)

    @property
    def n_blocks(self):
        return 2 ** self.r

    @property
    def block_len(self):
        return 2 ** self.m

    @property
    def blocks_per_prover(self):
        return 2 ** (self.r - self.logn)

    def owner_of_block(self, i):
        """Prover (1-indexed) owning block i in [0, 2^r), or None for an all-zero block.

        Block i (= nat of its r bits, MSB first) belongs to prover p iff its top ceil(log n) bits have
        nat = p - 1 (draft 6_CH:406, F2R:46-52 with the 0-indexed nat of m0-spec)."""
        p = (i >> (self.r - self.logn)) + 1
        return p if p <= self.n else None

    def blocks_of(self, p):
        return [i for i in range(self.n_blocks) if self.owner_of_block(i) == p]

    def acceptance_per_prover(self):
        """1/M as a float."""
        return math.exp(-self.ln_M)


def derive(name, n, ell, alpha, kappa, b, omega, n_A, n_B, n_D, m_rho, q, tau_rule="tight"):
    q = int(q)
    assert Integer(q).is_prime() and q % 8 == 5, "q must be a prime = 5 mod 8"
    assert b >= 3 and m_rho > n_D
    d, k = 2 ** alpha, 2 ** kappa
    assert 1 <= alpha - kappa <= ell, "1 <= alpha - kappa <= ell (6_CH:273)"
    assert (d // 2) % k == 0
    delta = num_digits(q, b)
    logn = ceil_log2(n)
    ell_prime = logn + ell - alpha + kappa
    r = max((ell_prime + 1) // 2, logn)
    m = ell_prime - r
    assert m >= 0
    beta1 = 2 ** (r - logn) * omega * b
    N = d * delta * 2 ** m
    gamma = 2 * n * N * beta1
    beta = n * gamma
    if tau_rule == "tight":
        tau = tight_tau(beta, b)
    elif tau_rule == "draft":
        tau = draft_tau(beta, b)
    else:
        raise ValueError(tau_rule)
    beta_bar, omega_bar, gamma_bar = 2 * b ** tau, 2 * omega, b
    mu = (2 ** r * delta + n * delta + 2 ** r * n_A * delta + n * n_A * delta + n * m_rho
          + 2 ** m * delta * tau + 1)
    n_prime = n_A + n_B + n_D + 2
    ell_U = ceil_log2(mu + n_prime)
    ell_L = ceil_log2(2 * d)
    ell_pp = ell_U + ell_L
    d0, d1 = 2 * b + n_prime + 1, 3
    nu = n_prime + d + (mu + n_prime) * d
    RR = RealField(200)
    ln_M = float(N * (RR(2 * gamma + 1) / RR(2 * gamma - 2 * beta1 + 1)).log())
    return Params(
        name=name, n=n, ell=ell, alpha=alpha, kappa=kappa, b=b, omega=omega, n_A=n_A, n_B=n_B, n_D=n_D,
        m_rho=m_rho, q=q, tau_rule=tau_rule, d=d, k=k, delta=delta, logn=logn, ell_prime=ell_prime, r=r, m=m,
        beta1=beta1, N=N, gamma=gamma, beta=beta, tau=tau, beta_bar=beta_bar, omega_bar=omega_bar,
        gamma_bar=gamma_bar, mu=mu, n_prime=n_prime, ell_U=ell_U, ell_L=ell_L, ell_pp=ell_pp, d0=d0, d1=d1,
        nu=nu, log_nu=ceil_log2(nu), width_A=delta * 2 ** m, width_B=n_A * delta * 2 ** r,
        width_D=delta * (2 ** r + n + n * n_A) + n * m_rho, ln_M=ln_M)


def choose_q(n, ell, alpha, kappa, b, omega, n_A, n_B, n_D, m_rho, tau_rule="tight", start=2 ** 20):
    """The m0-spec rule for q (smallest admissible prime = 5 mod 8, delta iterated to a fixed point)."""
    q = start
    while True:
        delta = num_digits(q, b)
        d = 2 ** alpha
        logn = ceil_log2(n)
        ell_prime = logn + ell - alpha + kappa
        r = max((ell_prime + 1) // 2, logn)
        m = ell_prime - r
        beta1 = 2 ** (r - logn) * omega * b
        N = d * delta * 2 ** m
        beta = n * 2 * n * N * beta1
        tau = tight_tau(beta, b) if tau_rule == "tight" else draft_tau(beta, b)
        bound = max(16 * omega * b ** tau, 1 + 2 ** (ell + 3) * n * omega ** 2 * b, 2 * beta + 1)
        p = bound + 1
        while not (p % 8 == 5 and Integer(p).is_prime()):
            p += 1
        if num_digits(p, b) == delta:
            return p
        q = p


TOY_COMMON = dict(ell=6, alpha=4, kappa=2, b=4, omega=2, n_A=1, n_B=1, n_D=1, m_rho=2)

# name -> (n, q, delta, r, m, beta1, N, gamma, beta, tau, mu, l'', d0).
# toy-n1..n3: the m0-spec "Toy presets" table.  toy-n4 and toy-n8: the same
# choices with n = 4 and 8, every number produced by choose_q and derive below.  check_toy_table re-derives
# every row.  toy-n4 has the modulus of toy-n3: both have tau = 11, so the binding guard 16 omega b^tau = 2^27
# is the largest bound for both.
TOY_TABLE = {
    "toy-n1": (1, 8388637, 12, 2, 2, 32, 768, 49152, 49152, 9, 555, 15, 14),
    "toy-n2": (2, 33554501, 13, 3, 2, 32, 832, 106496, 212992, 10, 785, 15, 14),
    "toy-n3": (3, 134217757, 14, 3, 3, 16, 1792, 172032, 516096, 11, 1547, 16, 14),
    "toy-n4": (4, 134217757, 14, 3, 3, 16, 1792, 229376, 917504, 11, 1577, 16, 14),
    "toy-n8": (8, 536871029, 15, 4, 3, 16, 1920, 491520, 3932160, 12, 2177, 17, 14),
}


def toy_preset(n):
    name = "toy-n%d" % n
    row = TOY_TABLE[name]
    return derive(name, n=n, q=row[1], **TOY_COMMON)


def check_toy_table():
    """Re-derives every row of TOY_TABLE (including q by the selection rule); returns a list of lines."""
    out = []
    for name, row in TOY_TABLE.items():
        n = row[0]
        q_rule = choose_q(n, **TOY_COMMON)
        P = derive(name, n=n, q=q_rule, **TOY_COMMON)
        got = (P.n, P.q, P.delta, P.r, P.m, P.beta1, P.N, P.gamma, P.beta, P.tau, P.mu, P.ell_pp, P.d0)
        assert got == row, (name, got, row)
        assert P.ln_M <= 1.0 / n, "M <= e^(1/n) (6_CH:275)"
        out.append("%s: table reproduced (q by rule = %d), ln M = %.6f <= 1/n = %.6f"
                   % (name, q_rule, P.ln_M, 1.0 / n))
    return out
