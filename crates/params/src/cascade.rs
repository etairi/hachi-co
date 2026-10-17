//! The Setup cascade of the draft, 6_Collaborative_Hachi.tex:265-281, evaluated exactly.

use crate::choices::{Choices, Omega, QChoice, TauRule};
use crate::error::{Error, Result};
use crate::report::Report;
use crate::rules;
use serde::{Deserialize, Serialize};

/// Every quantity of $\mathsf{Setup}$ (6_Collaborative_Hachi.tex:265-288), computed by
/// [`derive`](fn@crate::derive) from a [`Choices`].
///
/// Integers are exact (`u64`/`u128`, every operation checked); only the rejection constant $M$
/// is a float. The draft's $\log$ is base 2. Line numbers refer to 6_Collaborative_Hachi.tex
/// unless stated otherwise.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    /// The choices these parameters were derived from (with [`Omega::MinForTarget`] and
    /// [`QChoice::Smallest`] unresolved; the resolved values are [`Params::omega`] and
    /// [`Params::q`]).
    pub choices: Choices,
    /// The challenge radius $\omega$ (line 271).
    pub omega: u64,
    /// The modulus $q$ (line 272).
    pub q: u64,
    /// The ring degree $d = 2^\alpha$ (line 270).
    pub d: u64,
    /// The extension degree $k = 2^\kappa$ (line 274).
    pub k: u64,
    /// $\delta = \lceil \log_b q \rceil$, the number of digits of $\mathbf G^{-1}$ (line 274).
    pub delta: u32,
    /// $\lceil \log n \rceil$ (line 274).
    pub log_n: u32,
    /// $\ell' = \lceil \log n \rceil + \ell - \alpha + \kappa$ (line 274).
    pub ell_prime: u32,
    /// $r = \max(\lceil \ell'/2 \rceil, \lceil \log n \rceil)$: the witness has $2^r$ blocks
    /// (line 274).
    pub r: u32,
    /// $m = \ell' - r$: each block has $2^m$ ring elements (line 274).
    pub m: u32,
    /// $\beta_1 = 2^{r - \lceil \log n \rceil} \omega b$, the bound on each prover's share of the
    /// folded witness (line 275).
    pub beta1: u128,
    /// $N = d \delta 2^m$, the number of integer coefficients of a mask (line 275).
    pub big_n: u128,
    /// $\gamma = 2nN\beta_1$, the width of the masks (line 275).
    pub gamma: u128,
    /// The rejection constant $M = \left((2\gamma + 1)/(2\gamma - 2\beta_1 + 1)\right)^N$
    /// (line 275; 4_preliminaries.tex:117), as `f64`.
    pub rejection_m: f64,
    /// $\log_2 M$, as `f64`.
    pub log2_rejection_m: f64,
    /// $\beta = n\gamma$, the bound on the folded witness $z$ (line 275).
    pub beta: u128,
    /// $\tau$, the number of digits of $\mathbf J^{-1}$, by [`Choices::tau_rule`] (line 276 for
    /// the draft's rule).
    pub tau: u32,
    /// $\bar\beta = 2b^\tau$ (line 277).
    pub beta_bar: u128,
    /// $\bar\omega = 2\omega$ (line 277).
    pub omega_bar: u64,
    /// $\bar\gamma = b$ (line 277).
    pub gamma_bar: u64,
    /// $\mu = 2^r\delta + n\delta + 2^r n_A\delta + n n_A\delta + n m_\rho + 2^m\delta\tau + 1$,
    /// the length of $\mathbf z'$ (line 278; 5_Building_Blocks.tex:164-172).
    pub mu: u128,
    /// $n' = n_A + n_B + n_D + 2$ (line 279).
    pub n_prime: u64,
    /// $\ell'' = \lceil \log(\mu + n') \rceil + \log 2d$ (line 280).
    pub ell_dprime: u32,
    /// $d_0 = 2b + n' + 1$ (line 281).
    pub d0: u64,
    /// $d_1 = 3$ (line 281).
    pub d1: u64,
    /// $\nu = n' + d + (\mu + n') d$ (line 281).
    pub nu: u128,
    /// The width of $\mathbf A \in \mathcal R_q^{n_A \times \delta 2^m}$ (line 285).
    pub width_a: u128,
    /// The width of $\mathbf B \in \mathcal R_q^{n_B \times n_A \delta 2^r}$ (line 286).
    pub width_b: u128,
    /// The width of $\mathbf D \in \mathcal R_q^{n_D \times (\delta(2^r + n + n n_A) + n
    /// m_\rho)}$ (line 287).
    pub width_d: u128,
    /// $4\bar\omega\bar\beta = 16\omega b^\tau$, the binding guard of line 272 ($q$ must exceed
    /// it).
    pub binding_guard: u128,
    /// $1 + 2^{\ell+3} n \omega^2 b$, the other lower bound on $q$ of line 272.
    pub range_guard: u128,
    /// $2\bar\omega\bar\beta = 8\omega b^\tau$, the $\infty$-norm bound of the
    /// $\mathsf{MSIS}$ instance for $\mathbf A$ (line 563).
    pub msis_bound_a: u128,
    /// $2\bar\gamma = 2b$, the $\infty$-norm bound of the $\mathsf{MSIS}$ instance for
    /// $\mathbf B$ (line 563).
    pub msis_bound_b: u128,
    /// $(\lceil b/2 \rceil - 1)(b^\tau - 1)/(b - 1)$, the largest value $\mathbf J^{-1}$
    /// represents with $\tau$ digits ([`rules::j_inv_capacity`]).
    pub j_inv_capacity: u128,
}

/// The part of the cascade that depends on $\delta$ (lines 275-276) and the bounds on $q$ it
/// implies (line 272 and `docs/m0-spec.md`).
struct DeltaDependent {
    beta1: u128,
    big_n: u128,
    gamma: u128,
    beta: u128,
    tau: u32,
    binding_guard: u128,
}

impl DeltaDependent {
    /// Lines 275-277 for a given $\delta$.
    fn new(c: &Choices, s: &Structure, omega: u64, delta: u32) -> Result<Self> {
        let n = u128::from(c.n);
        let b = u128::from(c.b);
        let omega = u128::from(omega);
        let pow2 = |e: u32, what| rules::checked_pow(2, e, what);
        let beta1 = pow2(s.r - s.log_n, "beta_1")?
            .checked_mul(omega)
            .and_then(|x| x.checked_mul(b))
            .ok_or(Error::Overflow("beta_1"))?;
        let big_n = u128::from(s.d)
            .checked_mul(u128::from(delta))
            .and_then(|x| x.checked_mul(pow2(s.m, "N").ok()?))
            .ok_or(Error::Overflow("N"))?;
        let gamma = (2 * n)
            .checked_mul(big_n)
            .and_then(|x| x.checked_mul(beta1))
            .ok_or(Error::Overflow("gamma"))?;
        let beta = n.checked_mul(gamma).ok_or(Error::Overflow("beta"))?;
        let tau = rules::tau(c.tau_rule, c.b, beta)?;
        let binding_guard = rules::checked_pow(b, tau, "b^tau")?
            .checked_mul(16 * omega)
            .ok_or(Error::Overflow("4 omega_bar beta_bar"))?;
        Ok(Self {
            beta1,
            big_n,
            gamma,
            beta,
            tau,
            binding_guard,
        })
    }

    /// $\max(16\omega b^\tau,\ 1 + 2^{\ell+3} n\omega^2 b,\ 2\beta + 1)$: [`QChoice::Smallest`]
    /// picks the smallest prime $\equiv 5 \pmod 8$ above it.
    fn q_lower_bound(&self, range_guard: u128) -> Result<u128> {
        let centered = self
            .beta
            .checked_mul(2)
            .and_then(|x| x.checked_add(1))
            .ok_or(Error::Overflow("2 beta + 1"))?;
        Ok(self.binding_guard.max(range_guard).max(centered))
    }
}

/// The part of the cascade that depends neither on $\omega$ nor on $q$ (lines 270, 274).
struct Structure {
    d: u64,
    k: u64,
    log_n: u32,
    ell_prime: u32,
    r: u32,
    m: u32,
}

impl Structure {
    fn new(c: &Choices) -> Result<Self> {
        let d = 1u64
            .checked_shl(c.alpha)
            .ok_or(Error::Overflow("d = 2^alpha"))?;
        let k = 1u64
            .checked_shl(c.kappa)
            .ok_or(Error::Overflow("k = 2^kappa"))?;
        let log_n = rules::ceil_log2(u128::from(c.n));
        let ell_prime =
            i64::from(log_n) + i64::from(c.ell) - i64::from(c.alpha) + i64::from(c.kappa);
        if ell_prime < 0 {
            return Err(Error::NegativeExponent(format!(
                "l' = ceil(log n) + l - alpha + kappa = {ell_prime} < 0, which needs alpha - kappa > l \
                 (6_Collaborative_Hachi.tex:273 requires 1 <= alpha - kappa <= l)"
            )));
        }
        let ell_prime = u32::try_from(ell_prime).map_err(|_| Error::Overflow("l'"))?;
        let r = ell_prime.div_ceil(2).max(log_n);
        let m = ell_prime.checked_sub(r).ok_or_else(|| {
            Error::NegativeExponent(format!(
                "m = l' - r = {ell_prime} - {r} < 0, which needs alpha - kappa > l \
                 (6_Collaborative_Hachi.tex:273 requires 1 <= alpha - kappa <= l)"
            ))
        })?;
        Ok(Self {
            d,
            k,
            log_n,
            ell_prime,
            r,
            m,
        })
    }
}

/// $1 + 2^{\ell+3} n \omega^2 b$ (line 272).
fn range_guard(c: &Choices, omega: u64) -> Result<u128> {
    let omega = u128::from(omega);
    let ell3 = c.ell.checked_add(3).ok_or(Error::Overflow("l + 3"))?;
    rules::checked_pow(2, ell3, "2^(l+3)")?
        .checked_mul(u128::from(c.n))
        .and_then(|x| x.checked_mul(omega.checked_mul(omega)?))
        .and_then(|x| x.checked_mul(u128::from(c.b)))
        .and_then(|x| x.checked_add(1))
        .ok_or(Error::Overflow("1 + 2^(l+3) n omega^2 b"))
}

/// The smallest prime $q \equiv 5 \pmod 8$ with $q > \max(16\omega b^\tau,\ 1 + 2^{\ell+3} n
/// \omega^2 b,\ 2\beta + 1)$, where $\tau$ and $\beta$ use $\delta = \lceil \log_b q \rceil$
/// ([`QChoice::Smallest`]).
///
/// The bounds depend on $q$ only through $\delta$, so the candidates with a given $\delta$ form
/// the band $b^{\delta-1} < q \le b^\delta$ with fixed bounds. The function scans
/// $\delta = 1, 2, \ldots$ and returns, for the first band that contains one, the smallest
/// admissible prime of that band; since the bands are ordered, it is the smallest admissible
/// prime overall. (`docs/m0-spec.md` describes the equivalent fixed-point iteration from
/// $q_0 = 2^{20}$; started below the answer that iteration never overshoots it, so the two agree
/// whenever the answer is at least $2^{20}$, as the tests check.)
fn smallest_q(c: &Choices, s: &Structure, omega: u64, range_guard: u128) -> Result<(u64, u32)> {
    let b = u128::from(c.b);
    let mut band_low: u128 = 1; // b^(delta - 1)
    for delta in 1u32.. {
        if band_low > u128::from(u64::MAX) {
            break;
        }
        let band_high = band_low.saturating_mul(b); // b^delta
        // Bounds beyond u128 mean that no u64 modulus exists; other errors are passed on.
        let bound = DeltaDependent::new(c, s, omega, delta)
            .and_then(|dd| dd.q_lower_bound(range_guard))
            .map_err(|e| match e {
                Error::Overflow(what) => Error::NoModulus(format!(
                    "at delta = {delta} the bound overflows while computing {what}"
                )),
                other => other,
            })?;
        if bound >= u128::from(u64::MAX) {
            // The bounds only grow with delta.
            return Err(Error::NoModulus(format!(
                "the lower bound {bound} at delta = {delta} is not below 2^64"
            )));
        }
        if let Some(p) = rules::next_prime_5_mod_8(bound.max(band_low))
            && u128::from(p) <= band_high
        {
            return Ok((p, delta));
        }
        band_low = band_high;
    }
    Err(Error::NoModulus(
        "every band below 2^64 was searched".into(),
    ))
}

/// Evaluates $\mathsf{Setup}$ (6_Collaborative_Hachi.tex:265-288) on `choices`.
///
/// The cascade is followed verbatim, in the draft's order: $d, k$ (lines 270, 274); $\omega$
/// (line 271, or [`Omega::MinForTarget`]); $\delta, \ell', r, m$ (line 274); $\beta_1, N,
/// \gamma, M, \beta$ (line 275); $\tau$ (line 276, or the chosen [`TauRule`]);
/// $(\bar\beta, \bar\omega, \bar\gamma)$ (line 277); $\mu, n'$ (lines 278-279); $\ell''$
/// (line 280); $d_0, d_1, \nu$ (line 281); the matrix widths (lines 285-287). $q$ is either
/// fixed or chosen by the rule of `docs/m0-spec.md` ([`QChoice::Smallest`]).
///
/// The [`Report`] lists every condition of the draft and of `docs/m0-spec.md` with whether it
/// holds. Violated conditions are *not* errors.
///
/// # Errors
///
/// Only when the cascade cannot be evaluated: $n = 0$, $b < 2$, $n_A$, $n_B$ or $n_D$ zero,
/// [`Omega::Fixed`]`(0)`, [`QChoice::Fixed`] below 2 ([`Error::InvalidChoice`]); the tight rule
/// with $b = 2$ ([`Error::TightTauUndefined`]); $\alpha - \kappa > \ell$
/// ([`Error::NegativeExponent`]); a quantity beyond `u128` ([`Error::Overflow`]); no radius or
/// no modulus found ([`Error::NoOmega`], [`Error::NoModulus`]).
pub fn derive(choices: &Choices) -> Result<(Params, Report)> {
    let c = choices;
    validate(c)?;
    let s = Structure::new(c)?;
    let omega = match c.omega {
        Omega::Fixed(w) => w,
        Omega::MinForTarget => rules::min_omega(c.challenge_set, s.d, s.r, c.lambda)?,
    };
    let range_guard = range_guard(c, omega)?;
    let (q, delta) = match c.q {
        QChoice::Fixed(q) => (q, rules::ceil_log(u128::from(c.b), u128::from(q))?),
        QChoice::Smallest => smallest_q(c, &s, omega, range_guard)?,
    };
    let dd = DeltaDependent::new(c, &s, omega, delta)?;

    let b = u128::from(c.b);
    let n = u128::from(c.n);
    let n_a = u128::from(c.n_a);
    let delta128 = u128::from(delta);
    let overflow = |what| move || Error::Overflow(what);

    let b_tau = rules::checked_pow(b, dd.tau, "b^tau")?;
    let beta_bar = b_tau.checked_mul(2).ok_or_else(overflow("beta_bar"))?;
    let omega_bar = omega.checked_mul(2).ok_or_else(overflow("omega_bar"))?;
    let gamma_bar = c.b;

    let two_r = rules::checked_pow(2, s.r, "2^r")?;
    let two_m = rules::checked_pow(2, s.m, "2^m")?;
    // mu = 2^r delta + n delta + 2^r n_A delta + n n_A delta + n m_rho + 2^m delta tau + 1.
    let mu_terms = [
        two_r.checked_mul(delta128),
        n.checked_mul(delta128),
        two_r.checked_mul(n_a).and_then(|x| x.checked_mul(delta128)),
        n.checked_mul(n_a).and_then(|x| x.checked_mul(delta128)),
        n.checked_mul(u128::from(c.m_rho)),
        two_m
            .checked_mul(delta128)
            .and_then(|x| x.checked_mul(u128::from(dd.tau))),
        Some(1),
    ];
    let mu = mu_terms
        .into_iter()
        .try_fold(0u128, |acc, t| acc.checked_add(t?))
        .ok_or_else(overflow("mu"))?;
    let n_prime = [c.n_a, c.n_b, c.n_d, 2]
        .into_iter()
        .try_fold(0u64, |acc, t| acc.checked_add(t))
        .ok_or_else(overflow("n'"))?;
    let mu_plus_n_prime = mu
        .checked_add(u128::from(n_prime))
        .ok_or_else(overflow("mu + n'"))?;
    // log 2d = alpha + 1.
    let ell_dprime = rules::ceil_log2(mu_plus_n_prime) + c.alpha + 1;
    let d0 =
        c.b.checked_mul(2)
            .and_then(|x| x.checked_add(n_prime))
            .and_then(|x| x.checked_add(1))
            .ok_or_else(overflow("d_0"))?;
    let d1 = 3;
    let d128 = u128::from(s.d);
    let nu = mu_plus_n_prime
        .checked_mul(d128)
        .and_then(|x| x.checked_add(d128))
        .and_then(|x| x.checked_add(u128::from(n_prime)))
        .ok_or_else(overflow("nu"))?;

    let width_a = delta128
        .checked_mul(two_m)
        .ok_or_else(overflow("width of A"))?;
    let width_b = n_a
        .checked_mul(delta128)
        .and_then(|x| x.checked_mul(two_r))
        .ok_or_else(overflow("width of B"))?;
    // delta (2^r + n + n n_A) + n m_rho.
    let width_d = n
        .checked_mul(n_a)
        .and_then(|x| x.checked_add(two_r))
        .and_then(|x| x.checked_add(n))
        .and_then(|x| x.checked_mul(delta128))
        .and_then(|x| x.checked_add(n.checked_mul(u128::from(c.m_rho))?))
        .ok_or_else(overflow("width of D"))?;

    let msis_bound_a = dd.binding_guard / 2;
    let msis_bound_b = b.checked_mul(2).ok_or_else(overflow("2 gamma_bar"))?;
    let j_inv_capacity = rules::j_inv_capacity(c.b, dd.tau)?;

    let log2_rejection_m = rules::log2_rejection_constant(dd.big_n, dd.gamma, dd.beta1);
    let rejection_m = log2_rejection_m.exp2();

    let params = Params {
        choices: *c,
        omega,
        q,
        d: s.d,
        k: s.k,
        delta,
        log_n: s.log_n,
        ell_prime: s.ell_prime,
        r: s.r,
        m: s.m,
        beta1: dd.beta1,
        big_n: dd.big_n,
        gamma: dd.gamma,
        rejection_m,
        log2_rejection_m,
        beta: dd.beta,
        tau: dd.tau,
        beta_bar,
        omega_bar,
        gamma_bar,
        mu,
        n_prime,
        ell_dprime,
        d0,
        d1,
        nu,
        width_a,
        width_b,
        width_d,
        binding_guard: dd.binding_guard,
        range_guard,
        msis_bound_a,
        msis_bound_b,
        j_inv_capacity,
    };
    let report = Report::new(&params)?;
    Ok((params, report))
}

/// The domain checks of [`derive`](fn@crate::derive).
fn validate(c: &Choices) -> Result<()> {
    let invalid = |msg: &str| Err(Error::InvalidChoice(msg.into()));
    if c.n == 0 {
        return invalid("the number of provers n must be at least 1");
    }
    if c.b < 2 {
        return invalid("the decomposition base b must be at least 2 (the draft requires b >= 3)");
    }
    if c.tau_rule == TauRule::Tight && c.b < 3 {
        return Err(Error::TightTauUndefined { b: c.b });
    }
    if c.n_a == 0 || c.n_b == 0 || c.n_d == 0 {
        return invalid("the heights n_A, n_B and n_D of A, B and D must be at least 1");
    }
    if c.omega == Omega::Fixed(0) {
        return invalid("the challenge radius omega must be at least 1");
    }
    if let QChoice::Fixed(q) = c.q
        && q < 2
    {
        return invalid(
            "a fixed modulus must be at least 2 (delta = ceil(log_b q) is undefined below)",
        );
    }
    Ok(())
}
