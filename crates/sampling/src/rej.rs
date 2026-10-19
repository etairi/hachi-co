//! Rejection sampling, uniform variant (4_preliminaries.tex:117-127).
//!
//! Fix $N, \gamma, \beta_1$ with $\beta_1 \le \gamma$. The masks follow $\chi_r$, uniform on
//! $[-\gamma, \gamma]^N$, and accepted outputs follow $\chi_z$, uniform on
//! $[-\gamma + \beta_1, \gamma - \beta_1]^N$, with rejection constant
//! $$M = \left(\frac{2\gamma + 1}{2\gamma - 2\beta_1 + 1}\right)^N .$$
//! [`rej`] outputs $z = v + r$ iff $\lVert z \rVert_\infty \le \gamma - \beta_1$ (for these
//! distributions the acceptance probability $\min\lbrace \chi_z(z) / (M \chi_r(r)), 1 \rbrace$ is
//! $1$ on that box and $0$ off it, 4_preliminaries.tex:119), and [`sim_rej`] outputs a fresh
//! $z \leftarrow \chi_z$ with probability $1/M$. By the lemma (4_preliminaries.tex:122-127, which
//! cites Lemma 4 of Del Pino and Niot, "Finally! A Compact Lattice-Based Threshold Signature",
//! PKC 2025), for every $v$ with $\lVert v \rVert_\infty \le \beta_1$ the two have the same
//! output distribution. In Hachi-Co each prover runs [`rej`] on its own fold
//! (5_Building_Blocks.tex:127-130) with $N = d \delta 2^m$, $\beta_1 = 2^{r - \lceil \log n \rceil} \omega b$
//! and $\gamma = 2 n N \beta_1$ (6_Collaborative_Hachi.tex:275).
//!
//! Vectors are flat coefficient vectors of length $N$ (the $N/d$ ring elements' coefficients,
//! as $\chi_r$ and $\chi_z$ are defined, 4_preliminaries.tex:117).

use crate::draw::{Draw, FromRng};
use crate::uniform::boxed_with;
use crate::{Error, Result};
use core::f64::consts::{FRAC_1_SQRT_2, SQRT_2};
use num_bigint::BigUint;
use num_traits::Zero;
use rand::{CryptoRng, Rng};

/// An exact rational number $\mathit{numerator} / \mathit{denominator}$ in lowest terms, with a
/// positive denominator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ratio {
    /// The numerator.
    pub numerator: BigUint,
    /// The denominator, positive and coprime to the numerator.
    pub denominator: BigUint,
}

impl Ratio {
    /// $\log_2(\mathit{numerator} / \mathit{denominator})$ in floating point, computed from the
    /// exact integers: $-\infty$ for a zero numerator; outside the type's invariant, $+\infty$
    /// for a zero denominator and NaN for $0/0$. The ratio need not be in lowest terms.
    ///
    /// **Accuracy.** The error is below $0.52$ ulp of the exact value for a normal result (so the
    /// result is the correctly rounded value unless the exact one lies within $0.02$ ulp of a
    /// rounding boundary) and below one ulp ($2^{-1074}$) for a subnormal one; exact powers of
    /// two give their exponent exactly. Nothing cancels, however long the integers and however
    /// close the ratio is to $1$ or to a power of two:
    /// 1. $e$ is the integer nearest to the result, from the bit lengths and the leading 128 bits
    ///    of both integers, so that $y = \mathit{numerator} / (2^e \mathit{denominator})$ lies in
    ///    $[2^{-1/2}, 2^{1/2}]$ up to a relative $2^{-50}$;
    /// 2. $t = (y - 1)/(y + 1)$, of absolute value below $0.1716$ ($3 - 2\sqrt 2$ up to that
    ///    slack), is the quotient of the exact difference by the exact sum of
    ///    $\mathit{numerator}$ and $2^e \mathit{denominator}$ (one of them shifted); it is taken
    ///    to 64 bits from the leading 128 bits of the difference and the leading 64 bits of the
    ///    sum, with a relative error below $2^{-61}$;
    /// 3. $\log_2 y = \frac{2}{\ln 2} \operatorname{artanh} t = \frac{2}{\ln 2}\, t \sum_{j \ge 0}
    ///    t^{2j} / (2j + 1)$ is summed in fixed point with 64 fractional bits (`u128` products,
    ///    at most 13 terms as $t^2 \lt 0.03$), with $2/\ln 2$ rounded to 62 fractional bits; its
    ///    relative error before the final rounding is below $2^{-59}$;
    /// 4. the result is $e + \log_2 y$, rounded once to `f64`; for $e \ne 0$, where its absolute
    ///    value is at least $1/2$ (up to the slack of step 1), from a fixed-point sum with 62
    ///    fractional bits.
    ///
    /// The cost is linear in the length of the integers (two shifts, a comparison, a subtraction
    /// and an addition); everything else takes constant time.
    pub fn log2(&self) -> f64 {
        log2_ratio(&self.numerator, &self.denominator)
    }
}

/// $\mathrm{round}(2^{63} / \ln 2)$, that is $2/\ln 2$ with 62 fractional bits, rounded to
/// nearest. SageMath 10.9: `round(RealField(4000)(2)^63 / log(RealField(4000)(2)))`; the unit
/// tests check it against $\ln 2 = \sum_{j \ge 1} 1/(j 2^j)$.
const TWO_OVER_LN2_Q62: u128 = 13_306_513_097_844_322_492;

/// $2^{-62}$.
const TWO_POW_MINUS_62: f64 = 1.0 / (1u64 << 62) as f64;

/// [`Ratio::log2`] of $a/b$ (the algorithm and its error bound are documented there).
fn log2_ratio(a: &BigUint, b: &BigUint) -> f64 {
    match (a.is_zero(), b.is_zero()) {
        (true, true) => return f64::NAN,
        (true, false) => return f64::NEG_INFINITY,
        (false, true) => return f64::INFINITY,
        (false, false) => {}
    }
    let e = nearest_exponent(a, b);
    // a / b = 2^e a' / b', with y = a' / b' in [2^-1/2, 2^1/2] (up to a relative 2^-50).
    let shifted;
    let (a, b) = if e >= 0 {
        shifted = b << e.unsigned_abs();
        (a, &shifted)
    } else {
        shifted = a << e.unsigned_abs();
        (&shifted, b)
    };
    let (negative, diff) = if a >= b {
        (false, a - b)
    } else {
        (true, b - a)
    };
    if diff.is_zero() {
        // a / b = 2^e exactly.
        return e as f64;
    }
    let sum = a + b;
    // |t| = diff / sum = T 2^-k with 2^63 <= T < 2^64, from the quotient in [2^63, 2^65) of the
    // leading 128 bits of diff by the leading 64 bits of sum: three truncations, each by a
    // relative 2^-63 at most. |t| < 1/4 makes bits(sum) - bits(diff) >= 2, so k >= 65.
    let quotient = leading_u128(&diff) / (leading_u128(&sum) >> 64);
    let base = sum.bits() - diff.bits();
    let (t, k) = if quotient >> 64 == 0 {
        (quotient, 64 + base)
    } else {
        (quotient >> 1, 63 + base)
    };
    // t^2 with 64 fractional bits (zero when t^2 < 2^-64), below 0.03 * 2^64 < 2^59.
    let tau = (t * t)
        .checked_shr(u32::try_from(2 * k - 64).unwrap_or(u32::MAX))
        .unwrap_or(0);
    // s = sum_{j >= 1} t^{2j} / (2j + 1), 64 fractional bits, below 0.0102 * 2^64.
    let (mut s, mut power, mut j) = (0u128, tau, 1u128);
    while power != 0 {
        s += power / (2 * j + 1);
        power = (power * tau) >> 64;
        j += 1;
    }
    // (2 / ln 2) |t| (1 + s) = m 2^-(62 + k) = |log2 y| <= 1/2, with m < 2^128.
    let m = TWO_OVER_LN2_Q62 * (t + ((t * s) >> 64));
    if e == 0 {
        let magnitude = scale_down(m as f64, 62 + k);
        return if negative { -magnitude } else { magnitude };
    }
    // |log2 y| with 62 fractional bits, at most 2^61 (plus the slack of e).
    let fraction = m
        .checked_shr(u32::try_from(k).unwrap_or(u32::MAX))
        .unwrap_or(0) as i128;
    let fixed = (i128::from(e) << 62) + if negative { -fraction } else { fraction };
    fixed as f64 * TWO_POW_MINUS_62
}

/// The integer $e$ nearest to $\log_2(a/b)$ for $a, b \gt 0$, up to a relative $2^{-50}$ in the
/// ratio: $e_0$ is the difference of the bit lengths, so that $a/(2^{e_0} b) \in (1/2, 2)$, and
/// $e = e_0 \pm 1$ if the ratio of the leading 128 bits is at least $\sqrt 2$ or below
/// $1/\sqrt 2$.
fn nearest_exponent(a: &BigUint, b: &BigUint) -> i64 {
    let e0 = a.bits() as i64 - b.bits() as i64;
    let ratio = leading_u128(a) as f64 / leading_u128(b) as f64;
    if ratio >= SQRT_2 {
        e0 + 1
    } else if ratio < FRAC_1_SQRT_2 {
        e0 - 1
    } else {
        e0
    }
}

/// The leading 128 bits of $x \gt 0$, truncated, as an integer in $[2^{127}, 2^{128})$ (shifted
/// left when $x$ is shorter); 0 for $x = 0$. Only the top digits of $x$ are read.
fn leading_u128(x: &BigUint) -> u128 {
    let bits = x.bits();
    let top = if bits > 128 {
        x >> (bits - 128)
    } else {
        x << (128 - bits)
    };
    let mut digits = top.iter_u64_digits();
    let low = digits.next().unwrap_or(0);
    let high = digits.next().unwrap_or(0);
    (u128::from(high) << 64) | u128::from(low)
}

/// $x \cdot 2^{-n}$ for $2^{120} \le x \lt 2^{128}$, with a single rounding: the first factor
/// keeps the value normal, and a product below $2^{-1075}$ is $0$.
fn scale_down(x: f64, n: u64) -> f64 {
    if n > 1400 {
        return 0.0;
    }
    let half = n / 2;
    x * pow2_neg(half) * pow2_neg(n - half)
}

/// $2^{-n}$ for $n \le 1022$, exactly.
fn pow2_neg(n: u64) -> f64 {
    f64::from_bits((1023 - n) << 52)
}

/// The largest bit length of the numerator or denominator that [`acceptance_probability`]
/// computes exactly: $2^{26}$ bits (8 MiB per integer). Larger requests are refused; use
/// [`log2_acceptance_probability`] there.
pub const MAX_EXACT_BITS: u64 = 1 << 26;

/// Check $\beta_1 \le \gamma \le 2^{63} - 1$.
fn check_parameters(gamma: u64, beta1: u64) -> Result<()> {
    if gamma > i64::MAX as u64 {
        return Err(Error::InvalidParameter("gamma must be at most 2^63 - 1"));
    }
    if beta1 > gamma {
        return Err(Error::InvalidParameter(
            "beta1 must be at most gamma (4_preliminaries.tex:117)",
        ));
    }
    Ok(())
}

/// A mask $r \leftarrow \chi_r$, uniform on $[-\gamma, \gamma]^N$ (4_preliminaries.tex:117;
/// each prover's $r_i$, 5_Building_Blocks.tex:110).
///
/// This is [`crate::uniform::boxed`] with a [`CryptoRng`] bound: a mask hides the secret fold,
/// so it must come from a cryptographic generator, and it must never be reused or revealed
/// (plan, finding F5).
///
/// # Errors
/// [`Error::InvalidParameter`] if $\gamma \gt 2^{63} - 1$; [`Error::RejectionLimit`] if the
/// RNG is broken.
pub fn mask<R: CryptoRng + ?Sized>(rng: &mut R, gamma: u64, n: usize) -> Result<Vec<i64>> {
    boxed_with(&mut FromRng(rng), gamma, n)
}

/// $\mathsf{Rej}(v, \chi_z, \chi_r, M; r)$ of 4_preliminaries.tex:119: with $z = v + r$, returns
/// `Some(z)` if $\lVert z \rVert_\infty \le \gamma - \beta_1$ and `None` (that is, $\bot$)
/// otherwise.
///
/// The lemma (4_preliminaries.tex:122-127) needs $\lVert v \rVert_\infty \le \beta_1$ and a mask
/// in the support of $\chi_r$, so both are checked and their violation is an error rather than
/// an output.
///
/// The checks run over all coordinates without secret-dependent branches or early exit, and
/// branch once on the combined result (the plan's rule that the rejection check runs in
/// constant time). This is best effort: Rust does not guarantee constant-time code generation.
/// On rejection or error, the buffer holding $z$ is overwritten with zeros before it is
/// freed, also best effort.
///
/// # Errors
/// [`Error::InvalidParameter`] unless $\beta_1 \le \gamma \le 2^{63} - 1$;
/// [`Error::LengthMismatch`] if $v$ and $r$ differ in length; [`Error::MaskOutOfRange`] if
/// $\lVert r \rVert_\infty \gt \gamma$; [`Error::SecretTooLarge`] if
/// $\lVert v \rVert_\infty \gt \beta_1$.
pub fn rej(v: &[i64], gamma: u64, beta1: u64, r: &[i64]) -> Result<Option<Vec<i64>>> {
    check_parameters(gamma, beta1)?;
    if v.len() != r.len() {
        return Err(Error::LengthMismatch {
            left: v.len(),
            right: r.len(),
        });
    }
    let mut check = check(v, r, gamma, beta1);
    let outcome = if check.mask_too_large {
        Err(Error::MaskOutOfRange)
    } else if check.secret_too_large {
        Err(Error::SecretTooLarge)
    } else if check.reject {
        Ok(None)
    } else {
        return Ok(Some(check.z));
    };
    check.z.fill(0);
    core::hint::black_box(&check.z);
    outcome
}

/// The outcome of [`check`].
pub(crate) struct Check {
    /// $z = v + r$, each entry wrapped to `i64` (exact whenever $z$ is accepted).
    pub(crate) z: Vec<i64>,
    /// $\lVert r \rVert_\infty \gt \gamma$.
    pub(crate) mask_too_large: bool,
    /// $\lVert v \rVert_\infty \gt \beta_1$.
    pub(crate) secret_too_large: bool,
    /// $\lVert z \rVert_\infty \gt \gamma - \beta_1$.
    pub(crate) reject: bool,
}

/// The acceptance rule of [`rej`] without its input validation, for $v$, $r$ of equal length
/// and $\beta_1 \le \gamma \le 2^{63} - 1$. The unit tests also apply it at
/// $\lVert v \rVert_\infty = \beta_1 + 1$, where [`rej`] refuses to run.
pub(crate) fn check(v: &[i64], r: &[i64], gamma: u64, beta1: u64) -> Check {
    debug_assert_eq!(v.len(), r.len());
    debug_assert!(beta1 <= gamma && gamma <= i64::MAX as u64);
    let (gamma, beta1) = (i128::from(gamma), i128::from(beta1));
    let bound = gamma - beta1;
    let (mut mask_flag, mut secret_flag, mut reject_flag) = (0u8, 0u8, 0u8);
    let mut z = Vec::with_capacity(v.len());
    for (&vi, &ri) in v.iter().zip(r) {
        let (vi, ri) = (i128::from(vi), i128::from(ri));
        let zi = vi + ri;
        mask_flag |= exceeds(ri, gamma);
        secret_flag |= exceeds(vi, beta1);
        reject_flag |= exceeds(zi, bound);
        z.push(zi as i64);
    }
    Check {
        z,
        mask_too_large: mask_flag != 0,
        secret_too_large: secret_flag != 0,
        reject: reject_flag != 0,
    }
}

/// $1$ if $|x| \gt b$, else $0$, by arithmetic only. Requires $|x| \le 2^{64}$ and
/// $0 \le b \lt 2^{64}$, so that $b - |x|$ fits in `i128` and its sign bit is the answer.
#[inline]
fn exceeds(x: i128, b: i128) -> u8 {
    let sign = x >> 127; // 0 or -1
    let abs = (x ^ sign) - sign;
    ((b - abs) >> 127) as u8 & 1
}

/// $\mathsf{SimRej}(\chi_z, M)$ of 4_preliminaries.tex:120: outputs $z \leftarrow \chi_z$ (uniform
/// on $[-\gamma + \beta_1, \gamma - \beta_1]^N$) with probability exactly $1/M$, and `None`
/// ($\bot$) otherwise.
///
/// The coin of probability $1/M = \left(\frac{2\gamma - 2\beta_1 + 1}{2\gamma + 1}\right)^N$ is
/// realized as $N$ independent draws $u_k \leftarrow$ uniform in $[0, 2\gamma + 1)$, all of which
/// must be below $2\gamma - 2\beta_1 + 1$ (stopping at the first that is not); then $z$ is a fresh
/// draw of [`crate::uniform::boxed`]$(\gamma - \beta_1)$. This does not reuse [`rej`], so the
/// tests that compare the two check the lemma rather than assume it. It is a simulator: it
/// handles no secret and need not run in constant time.
///
/// # Errors
/// [`Error::InvalidParameter`] unless $\beta_1 \le \gamma \le 2^{63} - 1$;
/// [`Error::RejectionLimit`] if the RNG is broken.
pub fn sim_rej<R: Rng + ?Sized>(
    rng: &mut R,
    gamma: u64,
    beta1: u64,
    n: usize,
) -> Result<Option<Vec<i64>>> {
    check_parameters(gamma, beta1)?;
    sim_rej_with(&mut FromRng(rng), gamma, beta1, n)
}

/// [`sim_rej`] on any [`Draw`], for valid parameters.
pub(crate) fn sim_rej_with<D: Draw>(
    draw: &mut D,
    gamma: u64,
    beta1: u64,
    n: usize,
) -> Result<Option<Vec<i64>>> {
    let outer = 2 * gamma + 1;
    let inner = outer - 2 * beta1;
    for _ in 0..n {
        if draw.below(outer)? >= inner {
            return Ok(None);
        }
    }
    boxed_with(draw, gamma - beta1, n).map(Some)
}

/// The acceptance probability of [`rej`] (and of [`sim_rej`]) as an exact rational in lowest
/// terms:
/// $$\frac{1}{M} = \left(\frac{2\gamma - 2\beta_1 + 1}{2\gamma + 1}\right)^N .$$
///
/// With $a = 2\gamma - 2\beta_1 + 1$, $b = 2\gamma + 1$ and $g = \gcd(a, b)$, the result is
/// $(a/g)^N / (b/g)^N$, which is in lowest terms because coprime bases have coprime powers. The integers have about
/// $N \log_2(2\gamma + 1)$ bits; requests above [`MAX_EXACT_BITS`] are refused.
///
/// # Errors
/// [`Error::InvalidParameter`] unless $\beta_1 \le \gamma \le 2^{63} - 1$, or if $N \gt 2^{32} - 1$
/// or the result would exceed [`MAX_EXACT_BITS`] bits.
pub fn acceptance_probability(gamma: u64, beta1: u64, n: usize) -> Result<Ratio> {
    check_parameters(gamma, beta1)?;
    let exponent = u32::try_from(n)
        .map_err(|_| Error::InvalidParameter("N too large for an exact acceptance probability"))?;
    let outer = 2 * gamma + 1;
    let inner = outer - 2 * beta1;
    let g = gcd(inner, outer);
    let (a, b) = (inner / g, outer / g);
    if u64::from(64 - b.leading_zeros()) * n as u64 > MAX_EXACT_BITS {
        return Err(Error::InvalidParameter(
            "exact acceptance probability too large; use log2_acceptance_probability",
        ));
    }
    Ok(Ratio {
        numerator: BigUint::from(a).pow(exponent),
        denominator: BigUint::from(b).pow(exponent),
    })
}

/// $\log_2(1/M) = N \log_2\left(1 - \frac{2\beta_1}{2\gamma + 1}\right)$ in floating point,
/// computed with `ln_1p` so that it stays accurate for $\beta_1 \ll \gamma$; $\log_2 M$ is its
/// negation. The Setup constraint $M \le e^{1/n}$ (6_Collaborative_Hachi.tex:275) reads
/// $\log_2(1/M) \ge -\log_2(e)/n$.
///
/// # Errors
/// [`Error::InvalidParameter`] unless $\beta_1 \le \gamma \le 2^{63} - 1$.
pub fn log2_acceptance_probability(gamma: u64, beta1: u64, n: usize) -> Result<f64> {
    check_parameters(gamma, beta1)?;
    let x = -(2.0 * beta1 as f64) / (2.0 * gamma as f64 + 1.0);
    Ok(n as f64 * x.ln_1p() / core::f64::consts::LN_2)
}

/// Euclid's algorithm.
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod log2_tests;
#[cfg(test)]
mod tests;
