# An independent check of `src/gauss_tables.rs` and of the sign-bit construction of
# `hachi_co_sampling::gauss` (run: `sage crates/sampling/tests/fixtures/gauss_check.sage`).
#
# 1. Exactness. For each table, the probability that the construction (a half-Gaussian K from
#    the table and a sign bit; -K for sign 0, K for sign 1 and K != 0, a new draw for sign 1 and
#    K = 0) returns z, against Pr[z] = rho(z) / rho(Z) of D_{Z,sigma} with rho(z) =
#    exp(-z^2 / (2 sigma^2)), computed independently of the table generator: in RealField(512),
#    with the sum over |z| <= 40 sigma (the rest is below 2^-1000). Prints the statistical
#    distance and the largest relative error.
# 2. Convention. Draws from Sage's DiscreteGaussianDistributionIntegerSampler(sigma) (sigma is the
#    standard deviation there too) and a chi-square test of those draws against the table's
#    probabilities, bins of expected count >= 5, tail bins merged.
import re
from pathlib import Path

from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler

R = RealField(512)
src = Path(__file__).resolve().parents[2] / "src" / "gauss_tables.rs"
text = src.read_text()
tables = {}
for name, body in re.findall(r"pub const (GAUSS_SIGMA_\d+): \[u128; \d+\] = \[(.*?)\];", text, re.S):
    tables[int(name.split("_")[-1])] = [Integer(h, 16) for h in re.findall(r"0x([0-9a-f]{32})", body)]

set_random_seed(20261021)
for sigma, t in sorted(tables.items()):
    two = R(2) ** 128
    # Pr[K = i] = (T_{i-1} - T_i) / 2^128, T_{-1} = 2^128, T_len = 0.
    full = [Integer(2) ** 128] + t + [0]
    pk = [R(full[i] - full[i + 1]) / two for i in range(len(t) + 1)]
    accept = 1 - pk[0] / 2
    table_pmf = {0: (pk[0] / 2) / accept}
    for i in range(1, len(pk)):
        table_pmf[i] = table_pmf[-i] = (pk[i] / 2) / accept
    s2 = R(sigma) ** 2
    rho = lambda z: exp(-R(z) ** 2 / (2 * s2))
    span = 40 * sigma
    total = sum(rho(z) for z in range(-span, span + 1))
    sd = R(0)
    rel = R(0)
    for z in range(-span, span + 1):
        exact = rho(z) / total
        mine = table_pmf.get(z, R(0))
        sd += abs(mine - exact)
        if z in table_pmf and exact > 0:
            rel = max(rel, abs(mine / exact - 1))
    sd /= 2
    bound = len(t)
    print(f"sigma {sigma:2d}: support |z| <= {bound}, statistical distance 2^{float(log(sd, 2)):.1f}, "
          f"largest relative error of a supported value 2^{float(log(rel, 2)):.1f}")
    # 2. Sage's sampler against the table's probabilities.
    D = DiscreteGaussianDistributionIntegerSampler(sigma=sigma)
    n = 200000
    counts = {}
    for _ in range(n):
        z = ZZ(D())
        counts[z] = counts.get(z, 0) + 1
    outside = sum(c for z, c in counts.items() if abs(z) > bound)
    # Bins: every z with expected count >= 5, and the two tails merged into one bin.
    zs = [z for z in range(-bound, bound + 1) if n * table_pmf[z] >= 5]
    lo, hi = min(zs), max(zs)
    chi2 = R(0)
    for z in zs:
        e = n * table_pmf[z]
        chi2 += (counts.get(z, 0) - e) ** 2 / e
    tail_e = n * (1 - sum(table_pmf[z] for z in zs))
    tail_o = n - sum(counts.get(z, 0) for z in zs)
    if tail_e >= 5:
        chi2 += (tail_o - tail_e) ** 2 / tail_e
        dof = len(zs)
    else:
        dof = len(zs) - 1
    from scipy.stats import chi2 as chi2dist
    p = chi2dist.sf(float(chi2), dof)
    print(f"          Sage's sampler, {n} draws: {outside} outside the support; chi-square {float(chi2):.1f} "
          f"on {dof} degrees of freedom, p = {p:.3f}")
