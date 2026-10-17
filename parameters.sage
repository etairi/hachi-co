import sys
import os
import random
import shutil
import signal
import subprocess
import tempfile
import atexit


# Importing the lattice estimator (https://github.com/malb/lattice-estimator) for SIS hardness.
# The estimator is cloned into a fresh temporary directory, imported from there, and the clone is
# removed again when this script exits (normally, on an error, on Ctrl-C or on SIGTERM).
# Set LATTICE_ESTIMATOR_DIR to the path of an existing checkout to use it instead (it is then left untouched).
ESTIMATOR_REPO = "https://github.com/malb/lattice-estimator"

_estimator_dir = os.environ.get("LATTICE_ESTIMATOR_DIR")
_estimator_tmp = None

if _estimator_dir is None:
    _estimator_tmp = tempfile.mkdtemp(prefix="lattice-estimator-")
    _estimator_dir = os.path.join(_estimator_tmp, "lattice-estimator")
    print("Cloning", ESTIMATOR_REPO, "into", _estimator_dir)
    subprocess.run(["git", "clone", "--depth", "1", "--quiet", ESTIMATOR_REPO, _estimator_dir], check=True)

# Record the estimator revision the results were produced with.
_estimator_rev = subprocess.run(["git", "-C", _estimator_dir, "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
print("Lattice estimator revision:", _estimator_rev or "unknown")


# Sage writes the preparsed copy next to this script and runs it as __file__ (= sys.argv[0]); the
# module globals are already cleared when the atexit handler runs, so the path is captured here.
_preparsed_path = os.path.abspath(globals().get("__file__") or sys.argv[0])


def _cleanup():
    # Remove the temporary clone (never a user-supplied checkout).
    if _estimator_tmp is not None and os.path.isdir(_estimator_tmp):
        shutil.rmtree(_estimator_tmp, ignore_errors=True)
        print("Removed temporary clone", _estimator_tmp)

    # Remove the preparsed copy.
    if _preparsed_path.endswith(".sage.py") and os.path.isfile(_preparsed_path):
        os.remove(_preparsed_path)


atexit.register(_cleanup)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(int(1)))   # int(): a Sage Integer would be printed to stderr

sys.path.insert(1, _estimator_dir)
from estimator import *


# Global Parameters
secpar = 120            # Security parameter
logq = 32               # Log of the prime modulus  
kappa = 2               # Log of the degree of the field extension
poly_length = 30        # Length of the initial polynomial
repetitions = 1         # Ignore it if we don't do fully-splitting rings
JL = 0                  # Boolean to check if we use JL or not
TESTS = 50              # Number of tests we are willing to spend on

# Which parts of the script to run
RUN_SINGLE_PROVER = "--hachi-co-only" not in sys.argv   # the single-prover Hachi search below (random sampling, estimator calls)
RUN_COLLABORATIVE = True    # the Hachi-Co parameter cascade at the end of this file (`sage parameters.sage --hachi-co-only` runs only that)

# Powers of two
q = 2^logq
k = 2^kappa

# Array containing information about the challenge space. We consider power-of-two d between 2^5 and 2^9. 
# That is, ChSpaceL1norm[i]/ChSpaceSize[i] provides the bound on the L1 norm/size of challenges in the ring Rq with d = 2^{5+i}
ChSpaceL1norm = [256, 128, 128, 25, 22]
ChSpaceSize = [16^32, 5^64, 3^128, binomial(256, ChSpaceL1norm[3]) * 2^(ChSpaceL1norm[3]), binomial(512, ChSpaceL1norm[4]) * 2^(ChSpaceL1norm[4])]

# Algorithm to check whether SIS is hard against known lattice attacks
def is_SIS_secure(n, q, beta, target=None):
    # target: required security level in bits (default: the global secpar)
    target = secpar if target is None else target

    # Checking the trivial condition
    if beta >= (q-1)/2:
        return 0

    # Trivial linear-algebra attack, which the estimator does not model for bounds just below (q-1)/2:
    # pick the free coordinates of x in [-beta, beta] and solve for the n dependent ones mod q; each of
    # those lands in [-beta, beta] with probability (2 beta + 1)/q, so one linear solve succeeds with
    # probability ((2 beta + 1)/q)^n. Require this to be at most 2^-target (a necessary condition).
    if n * log(RR(q) / (2 * beta + 1), 2) < target:
        return 0

    # Running the lattice estimator. It raises OverflowError (in reduction.short_vectors, computing
    # 2^(0.2075 * sieve_dim) as a float) once the sieving dimension exceeds about 4900, i.e. when the
    # attack costs more than 2^1000 operations; such an instance is secure at every target used here.
    params = SIS.Parameters(n, q, beta, norm=oo)
    try:
        params = SIS.lattice(params)
    except OverflowError:
        return 1

    # Considering the best lattice attack
    if min(log(params["rop"],2), log(params["red"],2) , log(params["sieve"], 2)) < target:
        return 0
    else:
        return 1


# Binary search algorithm to find a SIS rank (e.g. nA, nB, nD)
def find_SIS_rank(lower_bound, upper_bound, d, q, beta, target=None):
    if lower_bound == upper_bound:
        return lower_bound
    else:
        n = int((lower_bound + upper_bound)/2)
        if is_SIS_secure(n*d, q , beta, target):
            return find_SIS_rank(lower_bound, n, d,  q, beta, target)
        else:
            return find_SIS_rank(n + 1, upper_bound, d, q, beta, target)

######################## Parameter script #########################################
if RUN_SINGLE_PROVER:

    round = 0                               # Variable to keep track of the number of rounds
    single_round_proof_size = 1             # The best possible communication in a single round
    proof_size = 0                          # Total proof size. Currently not ouput


    while single_round_proof_size != 0:

        print("ROUND ", round+1, ".  Current witness log length: ", poly_length)

        min_folded_witness_length = poly_length
        single_round_proof_size = 0

        # The following Boolean checks whether we commit to the polynomial (in the first round), or its MLE (second round onwards)
        if_witness_already_decomposed = min(round, 1)

        # Initializing variables for the best possible parameters for r and d
        final_logr = 0
        finald_d = 0

        # Probabilistic sampling
        for _ in range(TESTS):

            # Sampling random parameters
            alpha = random.randint(7, 9)
            delta = random.randint(2, 16)
            logr = random.randint(1, ceil((poly_length - alpha) / 2))

            # Setting main parameters
            r = 2^logr
            m = 2^(poly_length - alpha) / r
            d = 2^alpha

            # L1 norm of the challenge
            omega = ChSpaceL1norm[alpha-5]

            # Bounding infinity norm of z
            expected_inf_bound = (q^(1/delta) - 1)/2 * omega * sqrt(r)

            # Parameters to check relaxed binding of the inner-outer commitment scheme
            com_beta = expected_inf_bound
            com_omega = 2 * omega
            binding_length_bound = 2 * com_omega * com_beta

            # The following is the length of hat(z), i.e. decomposed z.
            # We note that from second round, when we commit to MLE, the ``delta'' factor doesn't appear for z.
            length_of_z = repetitions * m * d * log(2 * expected_inf_bound, q^(1/delta)) * (if_witness_already_decomposed  + (1 - if_witness_already_decomposed) * delta)

            # The following "if" checks:
            # (i) if binding is trivially broken by MSIS, 
            # (ii) the CWSS knowledge error is large, and 
            # (iii) if the length of z is already larger than the initial vector

            if binding_length_bound < (q-1)/2 and r/ChSpaceSize[alpha-5] < 2^(-secpar) and log(length_of_z,2) < poly_length:

                # Finding SIS dimensions for matrix A
                nA = find_SIS_rank(1, 2^11/d, d, q, binding_length_bound)

                # Finding SIS dimension for matrix B. Actually, we should run the script, but it gives an error because SIS is too hard.
                # So, let's just set nB = 1.
                # nB = find_SIS_rank(1, 2^11/d, d, q, q^(1/delta))
                nB = 1

                # We can fairly confidently assume the dimension for D is the same as for B
                nD = nB

                # Computing the rest of folded witnesses, e.g. hat(t) and hat(w)
                length_of_t_hat = r * nA * d * log(q, q^(1/delta))

                # For hat(w), remember that it is over field extension (for second round onwards, not first). Hence, the k factor.
                length_of_w_hat = r * d * log(q, q^(1/delta)) * (if_witness_already_decomposed * k + (1 - if_witness_already_decomposed))

                # Don't forget the carry vector
                length_of_carry_vector = (repetitions * nA + nB + nD + 2) * d

                # Total witness length
                log_witness_length = ceil(log(length_of_z + length_of_t_hat + length_of_w_hat + length_of_carry_vector, 2))
                witness_length = 2^log_witness_length

                # Computing the partial proof sizes. There are two choices, via sumcheck or Johnson-Lindnestrauss
                rangeproof = 0
                if JL == 1:
                    rangeproof = 256 * logq
                else:
                    rangeproof = ceil(q^(1/delta)) * logq * k * log_witness_length

                # Sumcheck for linear equations
                linear_sumcheck = 2 * logq * k * log_witness_length

                total_sumcheck = rangeproof + linear_sumcheck
                outer_commitment = nB * d * logq
                Zq_to_Rq_transformation_cost = d * logq
            
                # "If" to check whether we have founded even smaller folded witness
                if log_witness_length <= min_folded_witness_length:
                    # We update the witness length
                    min_folded_witness_length = log_witness_length

                    # Updating the best possible single-round proof size
                    if single_round_proof_size == 0 or single_round_proof_size > total_sumcheck + outer_commitment + Zq_to_Rq_transformation_cost:
                        single_round_proof_size =  total_sumcheck + outer_commitment + Zq_to_Rq_transformation_cost
                        final_logr = logr
                        final_d = d

        # Checking if we managed to actually fold anything. If not, break the loop.
        if min_folded_witness_length >= poly_length:
            single_round_proof_size = 0

        # Increase the round number. Update the witness size for the next round.
        round += 1
        poly_length = min_folded_witness_length

        # Update the proof size
        proof_size += single_round_proof_size

        print("Round completed. Folding parameters: ", d, delta, logr, poly_length - alpha - logr, (single_round_proof_size/2^13).n(), " KB")


    print("Final witness length in bits: ", poly_length)


######################## Hachi-Co (collaborative) parameter script ####################################
#
# The Setup cascade of the Hachi-Co draft (collaborative-hatchi, commit 625ced1; 6_CH below is
# 6_Collaborative_Hachi.tex, Setup at lines 263-298, parameters at 265-281), evaluated with exact
# integers exactly as the Rust crate hachi-co-params (crates/params) does, and the hardness estimates
# that choose the secure presets. Outputs (all under crates/params/):
#   presets/secure.json             the secure presets: choices, derived parameters, estimates (read by Rust)
#   presets/README.md               the same as a table, with the estimator revision and the command
#   tests/fixtures/sage_cascade.json  a grid of choices with Sage's cascade and report, for the Rust tests
#
# Run (this section only):   sage parameters.sage --hachi-co-only
# Development options:       --hachi-co-grid=l:n:alpha,...   only these (l, n, alpha)
#                            --hachi-co-out=DIR               write the three files under DIR instead
#
# CASCADE (6_CH:270-287; log is base 2):
#   d = 2^alpha, k = 2^kappa, delta = ceil(log_b q), l' = ceil(log n) + l - alpha + kappa,
#   r = max(ceil(l'/2), ceil(log n)), m = l' - r,
#   beta_1 = 2^(r - ceil(log n)) omega b, N = d delta 2^m, gamma = 2 n N beta_1,
#   M = ((2 gamma + 1)/(2 gamma - 2 beta_1 + 1))^N, beta = n gamma,
#   tau (rule below), (beta_bar, omega_bar, gamma_bar) = (2 b^tau, 2 omega, b),
#   mu = 2^r delta + n delta + 2^r n_A delta + n n_A delta + n m_rho + 2^m delta tau + 1,
#   n' = n_A + n_B + n_D + 2, l'' = ceil(log(mu + n')) + log 2d, d_0 = 2b + n' + 1, d_1 = 3,
#   nu = n' + d + (mu + n') d; widths: A delta 2^m, B n_A delta 2^r, D delta (2^r + n + n n_A) + n m_rho.
# RULES (docs/m0-spec.md):
#   tau ("Tight", the user's decision of 2026-10-17): the least tau with
#       (ceil(b/2) - 1) (b^tau - 1)/(b - 1) >= beta, so that balanced digits represent every |z| <= beta;
#       "Draft" is the draft's ceil(log_b beta) + 1 (6_CH:276), "Ceil" is ceil(log_b beta).
#   q ("Smallest"): the smallest prime p = 5 mod 8 with p > max(16 omega b^tau, 1 + 2^(l+3) n omega^2 b,
#       2 beta + 1), tau and beta computed with delta = ceil(log_b p). 16 omega b^tau = 4 omega_bar beta_bar
#       is the binding guard of 6_CH:272. Since the bounds depend on p only through delta, the candidates
#       with a given delta form the band b^(delta-1) < p <= b^delta with fixed bounds; scanning the bands
#       upwards gives the smallest admissible prime exactly.
#   omega ("MinForTarget"): the smallest radius with 2^(r+1)/|C| <= 2^-lambda, C the l1 ball
#       {c : ||c||_1 <= omega} of 5_Building_Blocks.tex:122, |C| = sum_i 2^i binom(d, i) binom(omega, i).
#
# SECURITY (lambda = 115 bits, k = 4, b = 16, tight tau, l1 ball, omega minimal):
#   MSIS for A (relaxed binding, 6_CH:563): rank n_A d, width delta 2^m, infinity bound
#       2 omega_bar beta_bar = 8 omega b^tau. This bound is just below q/2 at the smallest admissible q.
#       First the trivial linear-algebra attack that the estimator does not model: fix the free
#       coordinates in [-B, B] and solve for the rank dependent ones, each of which lands in [-B, B] with
#       probability (2B + 1)/q; require rank log2(q/(2B + 1)) >= lambda (the same check as is_SIS_secure
#       above). Then SIS.lattice (infinity norm).
#   MSIS for B (6_CH:563): rank n_B d, width n_A delta 2^r, infinity bound 2 gamma_bar = 2b.
#   MLWE for the hiding of the first message v (5_Building_Blocks.tex:193): MLWE_{q, d, n_D, m_rho - n_D, b},
#       i.e. LWE with secret dimension (m_rho - n_D) d, n_D d samples, secret and error uniform on the
#       coefficient range of S_b, [ceil(-b/2), ceil(b/2) - 1] = [-8, 7] (4_preliminaries.tex:88, 111-114).
#       LWE.estimate without arora-gb and bkw: the adversary sees only n_D d samples per secret (every
#       execution samples a fresh rho), and both attacks need far more; coded-BKW also refuses the
#       non-centred distribution. When n_D d < (m_rho - n_D) d (fewer samples than secret coordinates),
#       also without dual_hybrid: its cost model (MATZOV in lwe_dual.py) assumes n samples
#       (MATZOV.cost sets m = params.n, and MATZOV.__call__ passes no m) and uses params.m only to cap
#       the block size. Such an instance therefore gets the cost of an easier one: e.g. n = 1280,
#       m = 256, q ~ 2^45 gives 2^103.8 with m = 256, 1280 or oo alike, and n = 1536 gives oo once the
#       cap beta <= 256 binds. The primal attacks and the plain dual use at most m samples.
#   Estimator details. An attacker may use any m' of the matrix's columns. The estimator's own search
#       over the number of ignored columns (zeta in SIS.lattice) is unreliable here: it misses interior
#       optima (rank 1024, q ~ 2^63: 2^105.9 reported at the boundary, 2^95.2 on 3072 columns) and
#       returns an infinite cost when given many columns. So SIS.lattice is evaluated with zeta = 0 on
#       m' columns for m' on a geometric grid from 1.25 to 64 times the rank (capped by the true
#       width), the scan stops once the costs leave their finite window, and the minimum is refined by
#       a ternary search between its grid neighbours. An OverflowError (cost beyond 2^1000) counts as an
#       infinite cost at that width, as in is_SIS_secure. When the Euclidean baseline that SIS.lattice
#       starts from needs block size <= 40 the estimator raises a TypeError (its block-size search range
#       is empty); then the Euclidean cost with the same bound is used, an upper bound on the
#       infinity-norm cost (||x||_2 <= B implies ||x||_inf <= B), which can only show insecurity.
#   CHOICES where the paper is silent: b = 16 (the prototype's base), n_D = 1 (it minimises m_rho; the
#       draft states no MSIS condition for D); n_A and q: the lowest delta-band of admissible q that
#       admits a secure n_A, the smallest secure n_A in it, and the smallest prime = 5 mod 8 of that band
#       above the Setup bounds that leaves the MSIS margin for that n_A (bisection on log2 q to 1/256 bit,
#       the final prime checked again). Higher bands can need fewer rows of A, but each band adds a digit
#       to delta, which costs more in mu and N; the README lists that frontier. n_B and m_rho: the
#       smallest secure values at the chosen q.
#
# The threshold t of Setup is not used by the construction; presets record t = floor((n - 1)/2).

import json
import datetime

HCO_LAMBDA = 115                      # security target (bits)
HCO_KAPPA = 2                         # k = 4
HCO_B = 16                            # decomposition base
HCO_ND = 1                            # height of D
HCO_TAU_RULE = "Tight"
HCO_CHALLENGE_SET = "L1Ball"
HCO_Q_LIMIT = 2^63                    # residues are u64 with 128-bit products: q < 2^63
HCO_MAX_RANK = 2^13                   # largest MSIS rank n_A d (and n_B d) tried
HCO_Q_STEPS = 256                     # bisection on log2 q to 1/HCO_Q_STEPS bit
# The configurations (l, n, alpha): the brief's grid, then larger l and n that may still fit q < 2^63.
HCO_GRID = ([(l, n, a) for l in (16, 20, 24) for n in (2, 4) for a in (8, 10)]
            + [(l, n, a) for l in (28, 30) for n in (2, 4) for a in (8, 10)]
            + [(l, 8, a) for l in (16, 20, 24) for a in (8, 10)])

_hco_root = os.path.dirname(_preparsed_path)
_hco_out = os.path.join(_hco_root, "crates", "params")
for _arg in sys.argv[1:]:
    if _arg.startswith("--hachi-co-grid="):
        HCO_GRID = [tuple(int(v) for v in item.split(":")) for item in _arg.split("=", 1)[1].split(",")]
    elif _arg.startswith("--hachi-co-out="):
        _hco_out = os.path.abspath(_arg.split("=", 1)[1])


# ---------------------------------------------------------------------------------------------
# Exact cascade (mirrors crates/params/src/{rules,cascade,report}.rs)
# ---------------------------------------------------------------------------------------------

def hco_ceil_log2(x):
    """ceil(log2 x) for an integer x >= 1 (0 for x <= 1)."""
    x = int(x)
    return 0 if x <= 1 else (x - 1).bit_length()


def hco_ceil_log(b, x):
    """ceil(log_b x): the least t >= 0 with b^t >= x."""
    b = int(b); x = int(x); t = 0; p = 1
    while p < x:
        p *= b; t += 1
    return t


def hco_j_capacity(b, tau):
    """(ceil(b/2) - 1)(b^tau - 1)/(b - 1): the largest positive value of tau balanced base-b digits."""
    b = int(b)
    return ((b + 1) // 2 - 1) * (b^tau - 1) // (b - 1)


def hco_tau(rule, b, beta):
    if rule == "Tight":
        assert b >= 3, "the tight rule needs b >= 3"
        t = 0
        while hco_j_capacity(b, t) < beta:
            t += 1
        return t
    if rule == "Draft":
        return hco_ceil_log(b, beta) + 1
    assert rule == "Ceil"
    return hco_ceil_log(b, beta)


def hco_challenge_set_size(cs, d, omega):
    if cs == "L1Ball":
        return sum(2^i * binomial(d, i) * binomial(omega, i) for i in range(0, min(d, omega) + 1))
    assert cs == "Sparse"
    return binomial(d, omega) * 2^omega if omega <= d else 0


def hco_min_omega(cs, d, r, lam):
    cap = 2^16 if cs == "L1Ball" else d
    for omega in range(1, cap + 1):
        if hco_challenge_set_size(cs, d, omega) >= 2^(r + 1 + lam):
            return omega
    raise ValueError("no omega")


def hco_next_prime_5mod8(x):
    """The smallest prime p = 5 mod 8 with p > x."""
    p = int(x) + 1
    p += (5 - p) % 8
    while not is_prime(p):
        p += 8
    return p


def hco_structure(ch):
    d = 2^ch["alpha"]; k = 2^ch["kappa"]
    log_n = hco_ceil_log2(ch["n"])
    ell_prime = log_n + ch["ell"] - ch["alpha"] + ch["kappa"]
    assert ell_prime >= 0
    r = max((ell_prime + 1) // 2, log_n)
    m = ell_prime - r
    assert m >= 0
    return {"d": d, "k": k, "log_n": log_n, "ell_prime": ell_prime, "r": r, "m": m}


def hco_delta_dependent(ch, st, omega, delta):
    """6_CH:275-276 for a given delta, and the binding guard 16 omega b^tau."""
    n = ch["n"]; b = ch["b"]
    beta1 = 2^(st["r"] - st["log_n"]) * omega * b
    big_n = st["d"] * delta * 2^st["m"]
    gamma = 2 * n * big_n * beta1
    beta = n * gamma
    tau = hco_tau(ch["tau_rule"], b, beta)
    return {"beta1": beta1, "big_n": big_n, "gamma": gamma, "beta": beta, "tau": tau,
            "binding_guard": 16 * omega * b^tau}


def hco_range_guard(ch, omega):
    return 1 + 2^(ch["ell"] + 3) * ch["n"] * omega^2 * ch["b"]


def hco_q_lower_bound(dd, range_guard):
    return max(dd["binding_guard"], range_guard, 2 * dd["beta"] + 1)


def hco_smallest_q(ch, st, omega, range_guard):
    """The smallest prime q = 5 mod 8 above the bounds, scanning the bands b^(delta-1) < q <= b^delta."""
    b = ch["b"]; delta = 1
    while b^(delta - 1) < 2^64:
        dd = hco_delta_dependent(ch, st, omega, delta)
        p = hco_next_prime_5mod8(max(hco_q_lower_bound(dd, range_guard), b^(delta - 1)))
        if p <= b^delta:
            return p, delta
        delta += 1
    raise ValueError("no modulus below 2^64")


def hco_omega_of(ch, st):
    return ch["omega"]["Fixed"] if isinstance(ch["omega"], dict) else hco_min_omega(ch["challenge_set"], st["d"], st["r"], ch["lambda"])


def hco_derive(ch):
    """The cascade on a choices dict in the layout of the Rust Choices; returns (params dict, report dict)."""
    st = hco_structure(ch)
    omega = hco_omega_of(ch, st)
    rg = hco_range_guard(ch, omega)
    if isinstance(ch["q"], dict):
        q = ch["q"]["Fixed"]; delta = hco_ceil_log(ch["b"], q)
    else:
        q, delta = hco_smallest_q(ch, st, omega, rg)
    dd = hco_delta_dependent(ch, st, omega, delta)
    n = ch["n"]; b = ch["b"]; n_a = ch["n_a"]; r = st["r"]; m = st["m"]; tau = dd["tau"]; d = st["d"]
    mu = 2^r * delta + n * delta + 2^r * n_a * delta + n * n_a * delta + n * ch["m_rho"] + 2^m * delta * tau + 1
    n_prime = n_a + ch["n_b"] + ch["n_d"] + 2
    R = RealField(256)
    ln_m = dd["big_n"] * log(R(2 * dd["gamma"] + 1) / R(2 * dd["gamma"] - 2 * dd["beta1"] + 1))
    P = {
        "choices": ch, "omega": omega, "q": q, "d": d, "k": st["k"], "delta": delta, "log_n": st["log_n"],
        "ell_prime": st["ell_prime"], "r": r, "m": m, "beta1": dd["beta1"], "big_n": dd["big_n"],
        "gamma": dd["gamma"], "rejection_m": float(exp(ln_m)), "log2_rejection_m": float(ln_m / log(R(2))),
        "beta": dd["beta"], "tau": tau, "beta_bar": 2 * b^tau, "omega_bar": 2 * omega, "gamma_bar": b,
        "mu": mu, "n_prime": n_prime, "ell_dprime": hco_ceil_log2(mu + n_prime) + ch["alpha"] + 1,
        "d0": 2 * b + n_prime + 1, "d1": 3, "nu": n_prime + d + (mu + n_prime) * d,
        "width_a": delta * 2^m, "width_b": n_a * delta * 2^r,
        "width_d": delta * (2^r + n + n * n_a) + n * ch["m_rho"],
        "binding_guard": dd["binding_guard"], "range_guard": rg, "msis_bound_a": 8 * omega * b^tau,
        "msis_bound_b": 2 * b, "j_inv_capacity": hco_j_capacity(b, tau),
    }
    size = hco_challenge_set_size(ch["challenge_set"], d, omega)
    amk = ch["alpha"] - ch["kappa"]
    rep = {
        "QPrime": bool(is_prime(q)),
        "QFiveMod8": q % 8 == 5,
        "QAboveRangeGuard": q > rg,
        "QAboveBindingGuard": q > dd["binding_guard"],
        "CenteredResponses": 2 * dd["beta"] < q,
        "AlphaMinusKappa": 1 <= amk <= ch["ell"],
        "KDividesHalfD": d >= 2 and (d // 2) % st["k"] == 0,
        "BAtLeast3": b >= 3,
        "MRhoAboveND": ch["m_rho"] > ch["n_d"],
        "QBelow2Pow63": q < 2^63,
        "RejectionConstant": bool(ln_m <= R(1) / n),
        "JInvRepresentable": P["j_inv_capacity"] >= dd["beta"],
        "KnowledgeError": size >= 2^(r + 1 + ch["lambda"]),
    }
    return P, rep


def hco_choices(lam, n, ell, alpha, kappa, b, omega, cs, tau_rule, n_a, n_b, n_d, m_rho, q):
    """A choices dict in the serde layout of the Rust Choices (omega/q: an int for Fixed, None otherwise)."""
    return {"lambda": lam, "n": n, "t": (n - 1) // 2, "ell": ell, "alpha": alpha, "kappa": kappa, "b": b,
            "omega": "MinForTarget" if omega is None else {"Fixed": omega},
            "challenge_set": cs, "tau_rule": tau_rule, "n_a": n_a, "n_b": n_b, "n_d": n_d, "m_rho": m_rho,
            "q": "Smallest" if q is None else {"Fixed": q}}


def hco_plain(x):
    """Sage numbers to Python numbers, recursively (for json)."""
    if isinstance(x, dict):
        return {k: hco_plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [hco_plain(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, float):
        return x
    if x in ZZ:
        return int(x)
    return float(x)


# ---------------------------------------------------------------------------------------------
# Hardness estimates
# ---------------------------------------------------------------------------------------------

HCO_COLUMN_RATIOS = [1.25, 1.35, 1.5, 1.7, 2, 2.4, 3, 3.8, 5, 6.7, 9, 12.3, 17, 24, 32, 48, 64]


_hco_sis_cost_cache = {}

def hco_sis_cost(rank, q, bound, cols):
    """SIS.lattice (infinity norm) on exactly `cols` columns (zeta = 0). Returns (log2 cost or None, how, beta);
    None means an infinite cost at this width or an OverflowError (cost beyond 2^1000)."""
    key = (int(rank), int(q), int(bound), int(cols))
    if key not in _hco_sis_cost_cache:
        _hco_sis_cost_cache[key] = _hco_sis_cost(*key)
    return _hco_sis_cost_cache[key]


def _hco_sis_cost(rank, q, bound, cols):
    try:
        c = SIS.lattice(SIS.Parameters(rank, q, bound, m=cols, norm=oo), zeta=0)
    except OverflowError:
        return None, "overflow", None
    except TypeError:
        # Empty block-size search in SISLattice.cost_zeta (the Euclidean baseline needs block size <= 40):
        # the Euclidean cost with the same bound is an upper bound on the infinity-norm cost.
        c = SIS.lattice(SIS.Parameters(rank, q, bound, m=cols, norm=2))
        return (None if c["rop"] == oo else float(log(c["rop"], 2))), "euclidean upper bound", c.get("beta")
    return (None if c["rop"] == oo else float(log(c["rop"], 2))), "infinity norm", c.get("beta")


_hco_msis_cache = {}

def hco_msis(rank, q, bound, columns, target=HCO_LAMBDA, decide=False):
    """Infinity-norm MSIS with `rank` equations over Z_q, `columns` columns and bound `bound`.

    After the trivial checks, SIS.lattice is evaluated on m' columns with zeta = 0 for m' on a geometric grid
    (HCO_COLUMN_RATIOS times the rank, capped by `columns`), stopping once the costs have left their finite window,
    and the minimum is refined by a ternary search between its grid neighbours: the estimator's own search over
    the number of ignored columns (zeta) misses interior optima and returns an infinite cost for wide matrices.
    With decide=True the scan stops at the first cost below the target (the instance is then insecure).
    Returns a dict: trivial_bits, lattice_bits (None: no finite cost in the scanned widths), estimator_columns
    (the width of the cheapest attack), method, secure."""
    key = (int(rank), int(q), int(bound), int(columns), int(target), bool(decide))
    if key in _hco_msis_cache:
        return _hco_msis_cache[key]
    full_key = key[:5] + (False,)
    if decide and full_key in _hco_msis_cache:
        return _hco_msis_cache[full_key]
    rank, q, bound, columns = key[:4]
    out = {"rank": rank, "columns": columns, "bound": bound, "estimator_columns": 0,
           "trivial_bits": None, "lattice_bits": None, "method": "", "secure": False}
    if 2 * bound + 1 >= q:
        out["method"] = "bound >= (q - 1)/2: trivially broken"
    elif rank * log(RR(q) / (2 * bound + 1), 2) < target:
        out["trivial_bits"] = float(rank * log(RR(q) / (2 * bound + 1), 2))
        out["method"] = "trivial linear-algebra attack"
    else:
        out["trivial_bits"] = float(rank * log(RR(q) / (2 * bound + 1), 2))
        evaluated = {}
        def cost(m_):
            if m_ not in evaluated:
                evaluated[m_] = hco_sis_cost(rank, q, bound, m_)
            return evaluated[m_]
        grid = sorted(set(min(columns, int(ceil(f * rank))) for f in HCO_COLUMN_RATIOS))
        grid = [m_ for m_ in grid if m_ > rank]
        best = None; seen = False; rising = 0
        for m_ in grid:
            bits = cost(m_)[0]
            if bits is None:
                if seen:
                    break                       # past the window of finite costs
                continue
            seen = True
            if best is None or bits < best:
                best = bits; rising = 0
            else:
                rising += 1
            if (decide and bits < target) or rising >= 2:
                break
        finite = [(cost(m_)[0], m_) for m_ in evaluated if cost(m_)[0] is not None]
        if finite and not (decide and min(finite)[0] < target):
            # Ternary search on the integers between the grid neighbours of the best width.
            _, m_best = min(finite)
            i = grid.index(m_best)
            lo = grid[i - 1] if i > 0 else rank + 1
            hi = grid[i + 1] if i + 1 < len(grid) else m_best
            while hi - lo > 2:
                a = lo + (hi - lo) // 3; b = hi - (hi - lo) // 3
                ca = cost(a)[0]; cb = cost(b)[0]
                ca = oo if ca is None else ca; cb = oo if cb is None else cb
                if ca <= cb:
                    hi = b
                else:
                    lo = a
                if decide and min(ca, cb) < target:
                    break
            for m_ in range(lo, hi + 1):
                cost(m_)
            finite = [(cost(m_)[0], m_) for m_ in evaluated if cost(m_)[0] is not None]
        if finite:
            bits, m_best = min(finite)
            _, how, beta = cost(m_best)
            out["lattice_bits"] = bits
            out["estimator_columns"] = m_best
            out["method"] = "SIS.lattice, %s, zeta = 0, cheapest on %d of %d columns (beta = %s; %d widths evaluated)" % (
                how, m_best, columns, beta, len(evaluated))
            out["secure"] = bits >= target and how == "infinity norm"
        else:
            out["method"] = "SIS.lattice: no finite cost on up to %d columns (overflow or infinite)" % max(grid)
            out["estimator_columns"] = max(grid)
            out["secure"] = True
    out["secure"] = bool(out["secure"])
    _hco_msis_cache[key] = out
    return out


def hco_msis_secure(rank, q, bound, columns):
    return hco_msis(rank, q, bound, columns, decide=True)["secure"]


_hco_mlwe_cache = {}

def hco_mlwe(q, d, n_d, m_rho, b, rough=False):
    """MLWE_{q, d, n_D, m_rho - n_D, b}: LWE with (m_rho - n_D) d secret coefficients, n_D d samples,
    secret and error uniform on [ceil(-b/2), ceil(b/2) - 1]. Returns (bits or None, attack)."""
    key = (int(q), int(d), int(n_d), int(m_rho), int(b), bool(rough))
    if key in _hco_mlwe_cache:
        return _hco_mlwe_cache[key]
    lo = -(b // 2); hi = (b + 1) // 2 - 1
    X = ND.Uniform(lo, hi)
    params = LWE.Parameters(n=(m_rho - n_d) * d, q=q, Xs=X, Xe=X, m=n_d * d, tag="hiding")
    # dual_hybrid assumes n samples (see the header), so it is left out when fewer are available. The
    # rough estimate only brackets the search in hco_find_mrho, which then decides with this one.
    deny = ("arora-gb", "bkw") + (("dual_hybrid",) if params.m < params.n else ())
    if rough:
        res = LWE.estimate.rough(params, quiet=True)
    else:
        res = LWE.estimate(params, deny_list=deny, quiet=True)
    finite = [(name, c["rop"]) for name, c in res.items() if c["rop"] != oo]
    if finite:
        name, rop = min(finite, key=lambda kv: kv[1])
        val = (float(log(rop, 2)), name)
    else:
        val = (None, "none finite")
    _hco_mlwe_cache[key] = val
    return val


def hco_mlwe_secure(bits, target=HCO_LAMBDA):
    return bits is None or bits >= target


# ---------------------------------------------------------------------------------------------
# Searches
# ---------------------------------------------------------------------------------------------

def hco_bands(ch, st, omega):
    """The delta-bands that contain admissible q < 2^63: (delta, smallest admissible prime, top q, B_A)."""
    b = ch["b"]; rg = hco_range_guard(ch, omega)
    bands = []
    delta = 1
    while b^(delta - 1) < HCO_Q_LIMIT:
        dd = hco_delta_dependent(ch, st, omega, delta)
        top = min(b^delta, HCO_Q_LIMIT - 1)
        lo = max(hco_q_lower_bound(dd, rg), b^(delta - 1))
        if lo < top:
            low = hco_next_prime_5mod8(lo)
            if low <= top:
                bands.append({"delta": delta, "low": low, "top": top, "bound": dd["binding_guard"] // 2})
        delta += 1
    return bands


def hco_find_na_q(ch, st, omega, say):
    """MSIS for A: in the lowest delta-band of admissible q < 2^63 where some n_A <= HCO_MAX_RANK/d is secure,
    the smallest such n_A and the smallest such q (MSIS security grows with the rank and, inside a band, with q).
    Also returns the frontier [(delta, n_A)]: the smallest n_A at the top of each band where it improves on
    every lower band. A higher band would save rows of A at the price of a digit of delta, which costs more:
    mu and N grow with delta (secure-l16-n2-a8: n_A = 7 at delta = 12, n_A = 5 at delta = 16 with a larger mu)."""
    d = st["d"]
    bands = hco_bands(ch, st, omega)
    max_na = max(1, HCO_MAX_RANK // d)
    def secure(n_a, band, q):
        return hco_msis_secure(n_a * d, q, band["bound"], band["delta"] * 2^st["m"] * d)
    def smallest(band, lo, hi):
        """The smallest n_A in (lo, hi] secure at the band top, given insecure at lo and secure at hi."""
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if secure(mid, band, band["top"]):
                hi = mid
            else:
                lo = mid
        return hi
    chosen = None; best = None; frontier = []
    for band in bands:
        if best is None:
            hi = 1
            while hi < max_na and not secure(hi, band, band["top"]):
                hi = min(2 * hi, max_na)
            if not secure(hi, band, band["top"]):
                say("    band delta = %d: insecure up to n_A = %d" % (band["delta"], max_na))
                continue
            best = smallest(band, hi // 2 if hi > 1 else 0, hi)
            chosen = (band, best)
        elif best > 1 and secure(best - 1, band, band["top"]):
            best = smallest(band, 0, best - 1)
        else:
            continue
        frontier.append((band["delta"], best))
        say("    band delta = %d (q <= 2^%.2f, B_A = 2^%.2f): n_A = %d suffices at the top of the band" % (
            band["delta"], float(log(RR(band["top"]), 2)), float(log(RR(band["bound"]), 2)), best))
    if chosen is None:
        return None
    band, n_a = chosen
    if secure(n_a, band, band["low"]):
        return n_a, band["low"], band, frontier
    # Bisection on log2 q: insecure at 2^lo_x, secure at 2^hi_x.
    lo_x = RR(log(band["low"], 2)); hi_x = RR(log(band["top"], 2))
    while hi_x - lo_x > RR(1) / HCO_Q_STEPS:
        mid = (lo_x + hi_x) / 2
        if secure(n_a, band, floor(RR(2)^mid)):
            hi_x = mid
        else:
            lo_x = mid
    for _ in range(64):
        q = hco_next_prime_5mod8(max(band["low"], floor(RR(2)^hi_x)) - 1)
        if q > band["top"]:
            break
        if secure(n_a, band, q):
            return n_a, q, band, frontier
        hi_x += RR(1) / HCO_Q_STEPS   # the estimator is not monotone at this scale: step up
    say("    no prime above the bisection point is secure in band delta = %d" % band["delta"])
    return None


def hco_find_nb(q, d, width_b, b):
    for n_b in range(1, max(1, HCO_MAX_RANK // d) + 1):
        if hco_msis_secure(n_b * d, q, 2 * b, width_b * d):
            return n_b, hco_msis(n_b * d, q, 2 * b, width_b * d)
    return None, None


def hco_find_mrho(q, d, n_d, b, say):
    """The smallest m_rho > n_D with MLWE_{q, d, n_D, m_rho - n_D, b} secure by the full LWE.estimate.
    The rough estimate brackets the search; the answer is confirmed with full estimates downwards."""
    m_hi = None
    for m_rho in range(n_d + 1, n_d + 65):
        bits, _ = hco_mlwe(q, d, n_d, m_rho, b, rough=True)
        if hco_mlwe_secure(bits):
            m_hi = m_rho
            break
    assert m_hi is not None, "no m_rho reaches the target even by the rough estimate"
    m_rho = m_hi
    full = hco_mlwe(q, d, n_d, m_rho, b)
    while not hco_mlwe_secure(full[0]):
        m_rho += 1
        full = hco_mlwe(q, d, n_d, m_rho, b)
    while m_rho - 1 > n_d:
        below = hco_mlwe(q, d, n_d, m_rho - 1, b)
        say("    m_rho = %d: 2^%s (%s)" % (m_rho - 1, "%.1f" % below[0] if below[0] is not None else "inf", below[1]))
        if not hco_mlwe_secure(below[0]):
            break
        m_rho -= 1
        full = below
    return m_rho, full


# ---------------------------------------------------------------------------------------------
# Fixture for the Rust tests (Sage's cascade and report on a grid of choices; no estimator)
# ---------------------------------------------------------------------------------------------

def hco_fixture():
    entries = []
    def add(ch):
        P, rep = hco_derive(ch)
        entries.append({"params": hco_plain(P), "report": rep})
    # The toy presets of docs/m0-spec.md and variations of every choice around them.
    for n in (1, 2, 3):
        add(hco_choices(115, n, 6, 4, 2, 4, 2, "L1Ball", "Tight", 1, 1, 1, 2, None))
    rng = random.Random(int(20261017))                 # Python 3.14 rejects a Sage Integer seed
    for _ in range(100):
        b = rng.choice([2, 3, 4, 5, 7, 8, 16, 32])
        tau_rule = rng.choice(["Tight", "Draft", "Ceil"])
        if b == 2 and tau_rule == "Tight":
            tau_rule = "Draft"                              # the tight rule needs b >= 3
        cs = rng.choice(["L1Ball", "Sparse"])
        n = rng.choice([1, 2, 3, 4, 5, 8, 9])
        alpha = rng.choice([3, 4, 5, 6, 8])
        kappa = rng.randint(0, min(alpha, 4))              # kappa = alpha (alpha - kappa = 0) also occurs
        ell = rng.randint(max(1, alpha - kappa), alpha - kappa + 10)
        lam = rng.choice([4, 10, 40, 115])
        omega = None if rng.random() < 0.4 else rng.randint(1, 6)
        n_a, n_b, n_d = rng.randint(1, 4), rng.randint(1, 2), rng.randint(1, 3)
        m_rho = rng.randint(n_d, n_d + 4)                  # m_rho = n_D also occurs (a violation)
        ch = hco_choices(lam, n, ell, alpha, kappa, b, omega, cs, tau_rule, n_a, n_b, n_d, m_rho, None)
        try:
            P, _ = hco_derive(ch)
        except (ValueError, AssertionError):
            continue                                        # no omega, or no modulus below 2^64
        if P["q"] >= 2^64:
            continue
        add(ch)
        # Fixed moduli around the boundaries of the q conditions.
        q0 = P["q"]
        choice = rng.randint(0, 6)
        fixed = [q0 + 8, next_prime(q0 + 1), P["binding_guard"], P["binding_guard"] + 1,
                 P["range_guard"], 2 * P["beta"], 2 * P["beta"] + 1][choice]
        if 2 <= fixed < 2^64:
            add(dict(ch, q={"Fixed": int(fixed)}))
    # The configurations of the secure grid at the smallest admissible q (n_A = 8, m_rho = 7), for alpha 8, 9, 10.
    for l in (16, 20, 24, 28, 30):
        for n in (2, 4, 8):
            for alpha in (8, 9, 10):
                ch = hco_choices(HCO_LAMBDA, n, l, alpha, HCO_KAPPA, HCO_B, None, HCO_CHALLENGE_SET, HCO_TAU_RULE, 8, 1, 1, 7, None)
                try:
                    if hco_derive(ch)[0]["q"] < 2^64:
                        add(ch)
                except ValueError:
                    pass                                    # no modulus below 2^64
    # Moduli at the top of u64 (q >= 2^63 fails QBelow2Pow63).
    toy = hco_choices(115, 2, 6, 4, 2, 4, 2, "L1Ball", "Tight", 1, 1, 1, 2, None)
    for q in (2^63 - 25, 2^63 + 29, 2^64 - 59):
        add(dict(toy, q={"Fixed": int(q)}))
    # Exact boundaries of M <= e^(1/n): the least gamma at which it holds, for (n, N, beta_1).
    R = RealField(4000)
    boundaries = []
    for (n, big_n, beta1) in [(1, 768, 32), (2, 832, 32), (3, 1792, 16), (4, 1000, 7), (16, 3, 1), (1, 1, 1), (5, 2^20, 45056)]:
        holds = lambda g: big_n * log(R(2 * g + 1) / R(2 * g - 2 * beta1 + 1)) <= R(1) / n
        lo, hi = beta1, 4 * n * big_n * beta1 + 2          # fails at lo (gamma = beta_1: M large), holds at hi
        assert not holds(lo) and holds(hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if holds(mid):
                hi = mid
            else:
                lo = mid
        boundaries.append({"n": n, "big_n": big_n, "beta1": beta1, "least_gamma": hi})
    # Primes = 5 mod 8 just above powers of two (cross-check of the Rust prime search near 2^64).
    primes = [{"above": int(2^e), "p": hco_next_prime_5mod8(2^e)} for e in range(20, 64)]
    return {"generator": "parameters.sage (Hachi-Co section), function hco_fixture; seed 20261017",
            "entries": entries, "rejection_boundaries": hco_plain(boundaries), "primes_5_mod_8": primes}


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------

def hco_json_lines(doc):
    """JSON text of a dict whose list values are written one compact element per line (readable diffs)."""
    parts = []
    for key, value in doc.items():
        if isinstance(value, list):
            items = ",\n  ".join(json.dumps(e, separators=(",", ":")) for e in value)
            parts.append("%s: [\n  %s\n ]" % (json.dumps(key), items))
        else:
            parts.append("%s: %s" % (json.dumps(key), json.dumps(value)))
    return "{\n " + ",\n ".join(parts) + "\n}\n"


def hco_log(msg):
    print(msg)
    sys.stdout.flush()


def hco_fmt_bits(x):
    return "inf" if x is None else "%.1f" % x


if RUN_COLLABORATIVE:
    hco_log("\n==================== Hachi-Co parameters (6_Collaborative_Hachi.tex:263-298) ====================")
    hco_log("lambda = %d, k = %d, b = %d, tau = %s, challenge set = %s (omega minimal), n_D = %d, q < 2^63" % (
        HCO_LAMBDA, 2^HCO_KAPPA, HCO_B, HCO_TAU_RULE, HCO_CHALLENGE_SET, HCO_ND))
    _hco_started = datetime.datetime.now()

    # The toy presets of docs/m0-spec.md (a check of this implementation against the brief's table).
    for _n, _row in [(1, (8388637, 12, 49152, 9, 555, 15, 14)), (2, (33554501, 13, 212992, 10, 785, 15, 14)),
                     (3, (134217757, 14, 516096, 11, 1547, 16, 14))]:
        _P, _ = hco_derive(hco_choices(115, _n, 6, 4, 2, 4, 2, "L1Ball", "Tight", 1, 1, 1, 2, None))
        assert (_P["q"], _P["delta"], _P["beta"], _P["tau"], _P["mu"], _P["ell_dprime"], _P["d0"]) == _row, _n
    hco_log("toy presets: the brief's table is reproduced")

    _fixture = hco_fixture()
    _fixture_path = os.path.join(_hco_out, "tests", "fixtures", "sage_cascade.json")
    os.makedirs(os.path.dirname(_fixture_path), exist_ok=True)
    with open(_fixture_path, "w") as _f:
        _f.write(hco_json_lines(hco_plain(_fixture)))
    hco_log("wrote %s (%d cascade entries)" % (_fixture_path, len(_fixture["entries"])))

    _presets = []
    _rejected = []
    _frontiers = {}
    for (_l, _n, _alpha) in HCO_GRID:
        _name = "secure-l%d-n%d-a%d" % (_l, _n, _alpha)
        hco_log("\n%s: l = %d, n = %d, alpha = %d (d = %d)" % (_name, _l, _n, _alpha, 2^_alpha))
        _ch = hco_choices(HCO_LAMBDA, _n, _l, _alpha, HCO_KAPPA, HCO_B, None, HCO_CHALLENGE_SET, HCO_TAU_RULE,
                          1, 1, HCO_ND, HCO_ND + 1, None)
        _st = hco_structure(_ch)
        _omega = hco_omega_of(_ch, _st)
        try:
            _qmin = hco_derive(_ch)[0]["q"]
        except ValueError:
            _qmin = 2^64                                    # no admissible modulus below 2^64
        hco_log("  omega = %d, r = %d, m = %d, smallest admissible q = %d (2^%.2f)" % (
            _omega, _st["r"], _st["m"], _qmin, float(log(RR(_qmin), 2))))
        if _qmin >= HCO_Q_LIMIT:
            _rejected.append((_name, "the smallest admissible q is 2^%.2f >= 2^63" % float(log(RR(_qmin), 2))))
            hco_log("  rejected: " + _rejected[-1][1])
            continue
        _found = hco_find_na_q(_ch, _st, _omega, hco_log)
        if _found is None:
            _rejected.append((_name, "MSIS for A is insecure for every n_A <= %d and admissible q < 2^63" % max(1, HCO_MAX_RANK // 2^_alpha)))
            hco_log("  rejected: " + _rejected[-1][1])
            continue
        _n_a, _q, _band, _frontier = _found
        _P0, _ = hco_derive(dict(_ch, n_a=_n_a, q={"Fixed": _q}))
        hco_log("  n_A = %d, q = %d (2^%.4f, delta = %d, tau = %d)" % (_n_a, _q, float(log(RR(_q), 2)), _P0["delta"], _P0["tau"]))
        _n_b, _est_b = hco_find_nb(_q, _P0["d"], _P0["width_b"], HCO_B)
        _m_rho, (_mlwe_bits, _mlwe_attack) = hco_find_mrho(_q, _P0["d"], HCO_ND, HCO_B, hco_log)
        _final = dict(_ch, n_a=_n_a, n_b=_n_b, m_rho=_m_rho, q={"Fixed": int(_q)})
        _P, _rep = hco_derive(_final)
        assert all(_rep.values()), (_name, _rep)
        _est_a = hco_msis(_n_a * _P["d"], _q, _P["msis_bound_a"], _P["width_a"] * _P["d"])
        assert _est_a["secure"] and _est_b["secure"] and hco_mlwe_secure(_mlwe_bits)
        _security = {
            "lambda": HCO_LAMBDA,
            "msis_a": {k: _est_a[k] for k in ("rank", "columns", "estimator_columns", "bound", "trivial_bits", "lattice_bits", "method")},
            "msis_b": {k: _est_b[k] for k in ("rank", "columns", "estimator_columns", "bound", "trivial_bits", "lattice_bits", "method")},
            "mlwe": {"secret_dim": (_m_rho - HCO_ND) * _P["d"], "samples": HCO_ND * _P["d"],
                     "digit_range": [-(HCO_B // 2), (HCO_B + 1) // 2 - 1], "bits": _mlwe_bits, "attack": _mlwe_attack},
        }
        hco_log("  n_B = %d, m_rho = %d; MSIS A: trivial 2^%s, lattice 2^%s; MSIS B: 2^%s; MLWE: 2^%s (%s); mu = %d, l'' = %d" % (
            _n_b, _m_rho, hco_fmt_bits(_est_a["trivial_bits"]), hco_fmt_bits(_est_a["lattice_bits"]),
            hco_fmt_bits(_est_b["lattice_bits"]), hco_fmt_bits(_mlwe_bits), _mlwe_attack, _P["mu"], _P["ell_dprime"]))
        _presets.append({"name": _name, "params": hco_plain(_P), "security": hco_plain(_security)})
        _frontiers[_name] = _frontier

    _cmd = "sage parameters.sage " + " ".join(sys.argv[1:])
    _sage_version = version()
    _doc = {
        "format": 1,
        "generator": "parameters.sage (Hachi-Co section)",
        "command": _cmd.strip(),
        "sage_version": _sage_version,
        "estimator": {"repository": ESTIMATOR_REPO, "revision": _estimator_rev},
        "generated": _hco_started.strftime("%Y-%m-%d"),
        "presets": _presets,
    }
    _json_path = os.path.join(_hco_out, "presets", "secure.json")
    os.makedirs(os.path.dirname(_json_path), exist_ok=True)
    with open(_json_path, "w") as _f:
        json.dump(hco_plain(_doc), _f, indent=1)
        _f.write("\n")
    hco_log("\nwrote %s (%d presets)" % (_json_path, len(_presets)))

    # The human-readable table.
    _lines = [
        "# Secure Hachi-Co presets",
        "",
        "Written by `parameters.sage` (Hachi-Co section); do not edit by hand. The same data, with every derived",
        "quantity, is in `secure.json`, which `hachi-co-params` embeds and re-derives in its tests.",
        "",
        "- Command: `%s` (run from the repository root; %s, started %s)." % (_cmd.strip(), _sage_version, _hco_started.strftime("%Y-%m-%d %H:%M")),
        "- Lattice estimator: %s, revision `%s` (%s)." % (ESTIMATOR_REPO, _estimator_rev, "cloned into a temporary directory and removed at exit" if _estimator_tmp is not None else "an existing checkout given by LATTICE_ESTIMATOR_DIR"),
        "- Cascade: `6_Collaborative_Hachi.tex:265-281` of the draft at commit `625ced1`, with the tight tau and the rules of `docs/m0-spec.md`.",
        "- Fixed choices: lambda = %d, k = %d (kappa = %d), b = %d, tight tau, the l1-ball challenge set with the smallest omega such that 2^(r+1)/|C| <= 2^-lambda, n_D = %d, t = floor((n - 1)/2)." % (HCO_LAMBDA, 2^HCO_KAPPA, HCO_KAPPA, HCO_B, HCO_ND),
        "- n_A and q: MSIS for A (rank n_A d, width delta 2^m, infinity bound 2 omega_bar beta_bar = 8 omega b^tau, 6_Collaborative_Hachi.tex:563) in the lowest delta-band (b^(delta-1) < q <= b^delta) of admissible q < 2^63 that admits a secure n_A: the smallest secure n_A in that band, and the smallest prime = 5 mod 8 of the band above the Setup bounds with that MSIS margin (bisection on log2 q to 1/%d bit, the final prime checked again). The last column lists the higher bands where fewer rows of A would do; each band adds a digit to delta, which costs more in mu and N. n_B: the smallest height with MSIS for B secure (bound 2 gamma_bar = 2b); m_rho: the smallest value with MLWE_{q, d, n_D, m_rho - n_D, b} secure (5_Building_Blocks.tex:193)." % HCO_Q_STEPS,
        "- MSIS: first the trivial linear-algebra attack, rank log2(q/(2B + 1)) >= lambda, then `SIS.lattice` (infinity norm) with zeta = 0 on m' columns, minimised over m' (a grid of 1.25 to 64 times the rank, refined by ternary search), because the estimator's own search over ignored columns misses interior optima and returns infinite costs for wide matrices. MLWE: `LWE.estimate` without arora-gb and bkw (only n_D d samples per secret), and also without dual_hybrid when n_D d < (m_rho - n_D) d, because its MATZOV cost model assumes n samples and uses m only to cap the block size; secret and error uniform on [-8, 7].",
        "- Bits are log2 of the estimated cost; 'inf' means that no number of columns gave a finite cost (every evaluation infinite or beyond 2^1000). Estimates, not proofs; see the notes in `parameters.sage`.",
        "",
        "| preset | l | n | d | omega | q | log2 q | delta | tau | r | m | n_A | n_B | m_rho | mu | l'' | N | log2 beta | MSIS A trivial / lattice | MSIS B | MLWE (attack) | M^-n | fewer rows of A in higher bands (delta: n_A) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _e in _presets:
        _p = _e["params"]; _s = _e["security"]; _c = _p["choices"]
        _alt = "; ".join("%d: %d" % (_dl, _na) for (_dl, _na) in _frontiers[_e["name"]][1:]) or "none"
        _lines.append("| %s | %d | %d | %d | %d | %d | %.3f | %d | %d | %d | %d | %d | %d | %d | %d | %d | %d | %.2f | %s / %s | %s | %s (%s) | %.3f | %s |" % (
            _e["name"], _c["ell"], _c["n"], _p["d"], _p["omega"], _p["q"], float(log(RR(_p["q"]), 2)), _p["delta"], _p["tau"],
            _p["r"], _p["m"], _c["n_a"], _c["n_b"], _c["m_rho"], _p["mu"], _p["ell_dprime"], _p["big_n"],
            float(log(RR(_p["beta"]), 2)), hco_fmt_bits(_s["msis_a"]["trivial_bits"]), hco_fmt_bits(_s["msis_a"]["lattice_bits"]),
            hco_fmt_bits(_s["msis_b"]["lattice_bits"]), hco_fmt_bits(_s["mlwe"]["bits"]), _s["mlwe"]["attack"],
            float(_p["rejection_m"]^(-_c["n"])), _alt))
    if _rejected:
        _lines += ["", "Configurations without a secure preset below 2^63:", ""]
        _lines += ["- %s: %s." % (_name, _why) for (_name, _why) in _rejected]
    _readme_path = os.path.join(_hco_out, "presets", "README.md")
    with open(_readme_path, "w") as _f:
        _f.write("\n".join(_lines) + "\n")
    hco_log("wrote %s" % _readme_path)
    hco_log("Hachi-Co section finished in %s" % (datetime.datetime.now() - _hco_started))
