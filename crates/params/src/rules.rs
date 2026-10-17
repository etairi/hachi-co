//! Exact arithmetic used by the Setup cascade: integer logarithms, the $\tau$ rules, primality,
//! the size of the challenge set, and the exact test of $M \le e^{1/n}$.
//!
//! Every function here works on exact integers (`u64`, `u128` or [`BigUint`]); floating point is
//! used only to *display* logarithms, never to decide a condition.

use crate::choices::{ChallengeSet, TauRule};
use crate::error::{Error, Result};
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};

/// $\lceil \log_2 x \rceil$ for $x \ge 1$ (and $0$ for $x \le 1$).
///
/// This is the draft's $\lceil \log n \rceil$ and $\lceil \log(\mu + n') \rceil$
/// (6_Collaborative_Hachi.tex:274, 280); the draft's $\log$ is base 2.
pub fn ceil_log2(x: u128) -> u32 {
    if x <= 1 {
        0
    } else {
        u128::BITS - (x - 1).leading_zeros()
    }
}

/// $\lceil \log_b x \rceil$, the least $t \ge 0$ with $b^t \ge x$.
///
/// With $x = q$ this is $\delta = \lceil \log_b q \rceil$ (6_Collaborative_Hachi.tex:274). The
/// power is computed with saturation: once $b^t$ exceeds `u128::MAX` it is at least $x$, so the
/// result is exact for every `u128` input.
///
/// # Errors
///
/// [`Error::InvalidChoice`] if $b < 2$.
pub fn ceil_log(b: u128, x: u128) -> Result<u32> {
    if b < 2 {
        return Err(Error::InvalidChoice(format!("logarithm to base {b} < 2")));
    }
    let mut t = 0;
    let mut p: u128 = 1;
    while p < x {
        p = p.saturating_mul(b);
        t += 1;
    }
    Ok(t)
}

/// $b^e$ in `u128`, or [`Error::Overflow`] naming `what`.
pub fn checked_pow(b: u128, e: u32, what: &'static str) -> Result<u128> {
    b.checked_pow(e).ok_or(Error::Overflow(what))
}

/// The largest *positive* integer that $\tau$ balanced base-$b$ digits represent:
/// $(\lceil b/2 \rceil - 1) \cdot (b^\tau - 1)/(b - 1)$.
///
/// The digits of $\mathbf J^{-1}$ lie in $[\lceil -b/2 \rceil, \lceil b/2 \rceil - 1]$
/// (4_preliminaries.tex:100), so the positive side of the representable window has this many
/// elements (`docs/m0-spec.md`, "Digit sets" and "τ rule").
///
/// # Errors
///
/// [`Error::InvalidChoice`] if $b < 2$; [`Error::Overflow`] beyond `u128`.
pub fn j_inv_capacity(b: u64, tau: u32) -> Result<u128> {
    if b < 2 {
        return Err(Error::InvalidChoice(format!("digit base {b} < 2")));
    }
    let b = u128::from(b);
    let top_digit = b.div_ceil(2) - 1;
    // (b^tau - 1)/(b - 1) = 1 + b + ... + b^(tau - 1), accumulated exactly.
    let mut geometric: u128 = 0;
    for _ in 0..tau {
        geometric = geometric
            .checked_mul(b)
            .and_then(|g| g.checked_add(1))
            .ok_or(Error::Overflow("(b^tau - 1)/(b - 1)"))?;
    }
    top_digit
        .checked_mul(geometric)
        .ok_or(Error::Overflow("the J^-1 capacity"))
}

/// The number of digits $\tau$ of $\mathbf J^{-1}$ under `rule`, for the bound $\beta$ on the
/// folded witness.
///
/// - [`TauRule::Tight`]: the least $\tau$ with
///   $(\lceil b/2 \rceil - 1)(b^\tau - 1)/(b - 1) \ge \beta$ (user decision of 2026-10-17,
///   `docs/m0-spec.md`). Needs $b \ge 3$, otherwise [`Error::TightTauUndefined`].
/// - [`TauRule::Draft`]: $\tau = \lceil \log_b \beta \rceil + 1$ (6_Collaborative_Hachi.tex:276).
/// - [`TauRule::Ceil`]: $\tau = \lceil \log_b \beta \rceil$, which in general does *not* make
///   $\beta$ representable (balanced digits cover only about $b^\tau/2$); the report flags it.
///
/// # Errors
///
/// [`Error::TightTauUndefined`] for the tight rule with $b < 3$, [`Error::InvalidChoice`] for
/// $b < 2$, [`Error::Overflow`] beyond `u128`.
pub fn tau(rule: TauRule, b: u64, beta: u128) -> Result<u32> {
    match rule {
        TauRule::Tight => {
            if b < 3 {
                return Err(Error::TightTauUndefined { b });
            }
            let mut t: u32 = 0;
            while j_inv_capacity(b, t)? < beta {
                t = t.checked_add(1).ok_or(Error::Overflow("tau"))?;
            }
            Ok(t)
        }
        TauRule::Draft => Ok(ceil_log(u128::from(b), beta)? + 1),
        TauRule::Ceil => ceil_log(u128::from(b), beta),
    }
}

/// $x \cdot y \bmod n$ through a `u128` product.
fn mul_mod(x: u64, y: u64, n: u64) -> u64 {
    // The remainder is below n <= u64::MAX, so the narrowing cast is exact.
    ((u128::from(x) * u128::from(y)) % u128::from(n)) as u64
}

/// $x^e \bmod n$ by square and multiply.
fn pow_mod(mut x: u64, mut e: u64, n: u64) -> u64 {
    let mut acc = 1 % n;
    x %= n;
    while e > 0 {
        if e & 1 == 1 {
            acc = mul_mod(acc, x, n);
        }
        x = mul_mod(x, x, n);
        e >>= 1;
    }
    acc
}

/// Deterministic primality test for every `u64`.
///
/// Trial division by the first twelve primes, then strong-probable-prime tests to the same
/// twelve bases $2, 3, \ldots, 37$. No composite below $3.18 \cdot 10^{23} > 2^{64}$ is a strong
/// pseudoprime to all twelve bases (Sorenson and Webster, *Strong pseudoprimes to twelve prime
/// bases*, Math. Comp. 86 (2017)), so the test is exact on `u64`. The unit tests compare it with
/// trial division on every integer below $2^{20}$.
pub fn is_prime_u64(n: u64) -> bool {
    const BASES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    if n < 2 {
        return false;
    }
    for p in BASES {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    'bases: for a in BASES {
        let mut x = pow_mod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'bases;
            }
        }
        return false;
    }
    true
}

/// The smallest prime $p \equiv 5 \pmod 8$ with $p > x$, or `None` if there is none below
/// $2^{64}$.
pub fn next_prime_5_mod_8(x: u128) -> Option<u64> {
    let start = x.checked_add(1)?;
    // Round up to the residue class 5 mod 8.
    let mut p = start + (5 + 8 - (start % 8)) % 8;
    while p <= u128::from(u64::MAX) {
        // p <= u64::MAX was just checked.
        let candidate = p as u64;
        if is_prime_u64(candidate) {
            return Some(candidate);
        }
        p += 8;
    }
    None
}

/// $\binom{n}{k}$ as a [`BigUint`] (zero for $k > n$).
fn binomial(n: u64, k: u64) -> BigUint {
    if k > n {
        return BigUint::zero();
    }
    let k = k.min(n - k);
    let mut acc = BigUint::one();
    for i in 0..k {
        // acc = C(n, i) exactly; C(n, i + 1) = C(n, i) * (n - i) / (i + 1) is exact.
        acc *= n - i;
        acc /= i + 1;
    }
    acc
}

/// $\lvert C \rvert$ for the challenge set of radius $\omega$ in $\mathcal R$ of degree $d$.
///
/// - [`ChallengeSet::L1Ball`]: the draft's $C = \lbrace c : \lVert c \rVert_1 \le \omega \rbrace$
///   (5_Building_Blocks.tex:122), with
///   $\lvert C \rvert = \sum_{i=0}^{\min(d, \omega)} 2^i \binom{d}{i} \binom{\omega}{i}$
///   (choose the support of size $i$, the signs, and a composition of a total $\le \omega$
///   into $i$ positive parts). This counts integer vectors; it is the size in $\mathcal R_q$ too
///   as long as $\omega < q/2$, which the cascade guarantees ($q > 16 \omega b^\tau$).
/// - [`ChallengeSet::Sparse`]: exactly $\omega$ nonzero coefficients in $\lbrace \pm 1 \rbrace$,
///   the prototype's set, with $\lvert C \rvert = \binom{d}{\omega} 2^\omega$ (zero for
///   $\omega > d$).
pub fn challenge_set_size(set: ChallengeSet, d: u64, omega: u64) -> BigUint {
    match set {
        ChallengeSet::L1Ball => {
            let mut total = BigUint::zero();
            for i in 0..=d.min(omega) {
                total += (binomial(d, i) * binomial(omega, i)) << i;
            }
            total
        }
        ChallengeSet::Sparse => binomial(d, omega) << omega,
    }
}

/// Whether the knowledge error of the folding challenge meets the target:
/// $2^{r+1}/\lvert C \rvert \le 2^{-\lambda}$, i.e. $\lvert C \rvert \ge 2^{r + 1 + \lambda}$.
///
/// The error is that of coordinate-wise special soundness with $2^r$ coordinates and two
/// transcripts per coordinate (Hachi, Lemma 4); the draft does not state the condition in its
/// Setup, so it is the assumption (A3) of the earlier analysis, adopted by `docs/m0-spec.md`.
pub fn knowledge_error_holds(size: &BigUint, r: u32, lambda: u32) -> bool {
    let exponent = u64::from(r) + 1 + u64::from(lambda);
    *size >= (BigUint::one() << exponent)
}

/// $\log_2 x$ of a positive [`BigUint`], for display (`-inf` for zero).
pub fn log2_biguint(x: &BigUint) -> f64 {
    if x.is_zero() {
        return f64::NEG_INFINITY;
    }
    let bits = x.bits();
    if bits <= 64 {
        // x < 2^64, so to_u64 succeeds.
        return x.to_u64().map_or(f64::NAN, |v| (v as f64).log2());
    }
    let shift = bits - 64;
    let top = (x >> shift).to_u64().map_or(f64::NAN, |v| v as f64);
    top.log2() + shift as f64
}

/// The largest radius [`min_omega`] tries for the $\ell_1$ ball.
pub const OMEGA_SEARCH_CAP: u64 = 1 << 16;

/// The smallest radius $\omega \ge 1$ with $2^{r+1}/\lvert C \rvert \le 2^{-\lambda}$
/// ([`Omega::MinForTarget`](crate::Omega::MinForTarget)).
///
/// For the $\ell_1$ ball $\lvert C \rvert$ grows strictly with $\omega$; the search stops at
/// [`OMEGA_SEARCH_CAP`]. For the sparse set it tries $\omega = 1, \ldots, d$. Either way the
/// failure is [`Error::NoOmega`].
pub fn min_omega(set: ChallengeSet, d: u64, r: u32, lambda: u32) -> Result<u64> {
    let cap = match set {
        ChallengeSet::L1Ball => OMEGA_SEARCH_CAP,
        ChallengeSet::Sparse => d,
    };
    (1..=cap)
        .find(|&omega| knowledge_error_holds(&challenge_set_size(set, d, omega), r, lambda))
        .ok_or(Error::NoOmega { lambda, cap })
}

/// $\log_2 M$ for the rejection constant
/// $M = \left((2\gamma + 1)/(2\gamma - 2\beta_1 + 1)\right)^N$ (6_Collaborative_Hachi.tex:275,
/// 4_preliminaries.tex:117), computed as $N \cdot \mathrm{ln1p}(2\beta_1/(2\gamma - 2\beta_1 + 1))
/// / \ln 2$ in `f64`. Returns $+\infty$ if $\beta_1 > \gamma$ (the accepting box is empty).
pub fn log2_rejection_constant(big_n: u128, gamma: u128, beta1: u128) -> f64 {
    if beta1 > gamma {
        return f64::INFINITY;
    }
    // 2 beta_1 / (2 gamma - 2 beta_1 + 1) = beta_1 / ((gamma - beta_1) + 1/2), without overflow.
    let x = beta1 as f64 / ((gamma - beta1) as f64 + 0.5);
    big_n as f64 * x.ln_1p() / core::f64::consts::LN_2
}

/// A positive rational number $p/s$ with [`BigUint`] parts (no reduction).
#[derive(Clone)]
struct Ratio {
    num: BigUint,
    den: BigUint,
}

impl Ratio {
    fn add(&self, other: &Ratio) -> Ratio {
        Ratio {
            num: &self.num * &other.den + &other.num * &self.den,
            den: &self.den * &other.den,
        }
    }

    /// `self <= other`, by cross-multiplication.
    fn le(&self, other: &Ratio) -> bool {
        &self.num * &other.den <= &other.num * &self.den
    }
}

/// The largest number of series terms [`rejection_constant_within`] uses.
pub const REJECTION_SERIES_CAP: u32 = 256;

/// Whether $M = \left((2\gamma + 1)/(2\gamma - 2\beta_1 + 1)\right)^N \le e^{1/n}$, decided
/// exactly (6_Collaborative_Hachi.tex:275).
///
/// With $y = \beta_1/(2\gamma - \beta_1 + 1)$ one has $(1 + y)/(1 - y) = (2\gamma + 1)/(2\gamma -
/// 2\beta_1 + 1)$, so
/// $$\ln M = 2N \operatorname{artanh} y = 2N \sum_{j \ge 0} \frac{y^{2j+1}}{2j+1}.$$
/// All terms are positive, so the partial sum $L_K$ of the terms $j \le K$ is a lower bound and
/// $L_K + 2N y^{2K+3}/((2K+3)(1 - y^2))$ an upper bound (the tail is at most a geometric series
/// with ratio $y^2$). Both are rationals, compared exactly with $1/n$; $K$ grows until they
/// decide. Equality $\ln M = 1/n$ is impossible because $e^{1/n}$ is irrational, so the loop
/// ends; [`REJECTION_SERIES_CAP`] bounds it anyway ([`Error::Undecided`]).
///
/// For the cascade's $\gamma = 2nN\beta_1$ the first pair of bounds ($K = 0$) already decides,
/// and the condition always holds (the draft asserts it): $y = \beta_1/(4nN\beta_1 - \beta_1 + 1)
/// \le 1/(4nN - 1) \le 1/3$ and $2Ny \le 2N/(4nN - 1) \le 2/(3n)$, so the upper bound is at most
/// $\frac{2}{3n}\left(1 + \frac{y^2}{3(1 - y^2)}\right) \le \frac{25}{36n} < \frac1n$. In fact
/// $\ln M$ is just above $1/(2n)$. The function is general so that its failure boundary can be
/// tested.
///
/// Returns [`Error::InvalidChoice`] if $n = 0$, and `Ok(false)` if $\beta_1 > \gamma$ (then
/// $2\gamma - 2\beta_1 + 1 \le 0$, the accepting box is empty and $M = \infty$).
pub fn rejection_constant_within(n: u64, big_n: u128, gamma: u128, beta1: u128) -> Result<bool> {
    if n == 0 {
        return Err(Error::InvalidChoice(
            "the number of provers n must be at least 1".into(),
        ));
    }
    if beta1 > gamma {
        return Ok(false);
    }
    if beta1 == 0 || big_n == 0 {
        // M = 1 <= e^(1/n).
        return Ok(true);
    }
    // y = p / s with p < s.
    let p = BigUint::from(beta1);
    let s = BigUint::from(gamma) * 2u32 - BigUint::from(beta1) + 1u32;
    let two_n = BigUint::from(big_n) * 2u32;
    let target = Ratio {
        num: BigUint::one(),
        den: BigUint::from(n),
    };
    let p2 = &p * &p;
    let s2 = &s * &s;
    // term = y^(2j+1) as p^(2j+1) / s^(2j+1).
    let mut p_pow = p.clone();
    let mut s_pow = s.clone();
    let mut lower = Ratio {
        num: BigUint::zero(),
        den: BigUint::one(),
    };
    for j in 0..REJECTION_SERIES_CAP {
        let odd = u64::from(2 * j + 1);
        lower = lower.add(&Ratio {
            num: &two_n * &p_pow,
            den: &s_pow * odd,
        });
        // Tail after term j: 2N y^(2j+3) / ((2j+3)(1 - y^2)) = 2N p^(2j+3) / ((2j+3) s^(2j+1) (s^2 - p^2)).
        let tail = Ratio {
            num: &two_n * &p_pow * &p2,
            den: &s_pow * (&s2 - &p2) * (odd + 2),
        };
        let upper = lower.add(&tail);
        if upper.le(&target) {
            return Ok(true);
        }
        if !lower.le(&target) {
            return Ok(false);
        }
        p_pow *= &p2;
        s_pow *= &s2;
    }
    Err(Error::Undecided(format!(
        "n = {n}, N = {big_n}, gamma = {gamma}, beta_1 = {beta1}"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial_division(n: u64) -> bool {
        if n < 2 {
            return false;
        }
        let mut p = 2;
        while p * p <= n {
            if n.is_multiple_of(p) {
                return false;
            }
            p += 1;
        }
        true
    }

    #[test]
    fn primality_matches_trial_division_below_2_pow_20() {
        for n in 0..(1u64 << 20) {
            assert_eq!(is_prime_u64(n), trial_division(n), "n = {n}");
        }
    }

    #[test]
    fn primality_at_the_top_of_u64_and_on_strong_pseudoprimes() {
        // Primes: Mersenne 2^61 - 1, the largest primes below 2^63 and 2^64.
        for p in [(1u64 << 61) - 1, (1u64 << 63) - 25, u64::MAX - 58] {
            assert!(is_prime_u64(p), "{p} is prime");
        }
        // Composites: Carmichael numbers, strong pseudoprimes to many small bases,
        // a product of two primes near 2^32, and 2^64 - 1.
        let composites = [
            561u64,
            1105,
            3_215_031_751,             // spsp(2, 3, 5, 7)
            2_152_302_898_747,         // spsp(2, ..., 11)
            3_474_749_660_383,         // spsp(2, ..., 13)
            341_550_071_728_321,       // spsp(2, ..., 17)
            3_825_123_056_546_413_051, // spsp(2, ..., 23)
            4_294_967_291 * 4_294_967_279,
            u64::MAX,
        ];
        for c in composites {
            assert!(!is_prime_u64(c), "{c} is composite");
        }
    }

    #[test]
    fn next_prime_5_mod_8_is_the_first_one_by_exhaustion() {
        for x in 0u128..20_000 {
            let p = next_prime_5_mod_8(x).expect("exists");
            assert!(u128::from(p) > x && p % 8 == 5 && trial_division(p));
            for y in (x + 1)..u128::from(p) {
                let y = y as u64;
                assert!(
                    !(y % 8 == 5 && trial_division(y)),
                    "skipped {y} for x = {x}"
                );
            }
        }
        // Near the top of u64: no prime = 5 mod 8 lies above u64::MAX - 4.
        assert_eq!(next_prime_5_mod_8(u128::from(u64::MAX)), None);
        assert_eq!(next_prime_5_mod_8(u128::MAX), None);
    }

    #[test]
    fn integer_logarithms_match_naive_powers() {
        for x in 1u128..5000 {
            let naive2 = (0..).find(|&t| 1u128 << t >= x).unwrap();
            assert_eq!(ceil_log2(x), naive2, "x = {x}");
            for b in 2u128..20 {
                let naive = (0..).find(|&t| b.pow(t) >= x).unwrap();
                assert_eq!(ceil_log(b, x).unwrap(), naive, "b = {b}, x = {x}");
            }
        }
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(u128::MAX), 128);
        assert_eq!(ceil_log(16, u128::MAX).unwrap(), 32);
        assert_eq!(ceil_log(3, u128::MAX).unwrap(), 81); // 3^80 < 2^128 - 1 < 3^81
        assert!(ceil_log(1, 5).is_err() && ceil_log(0, 5).is_err());
        assert!(j_inv_capacity(1, 3).is_err() && j_inv_capacity(0, 3).is_err());
    }

    /// Brute force: the largest positive value representable by tau balanced base-b digits.
    fn capacity_by_enumeration(b: i64, tau: u32) -> i64 {
        let lo = -(b / 2); // ceil(-b/2)
        let hi = (b + 1) / 2 - 1; // ceil(b/2) - 1
        let mut values = vec![0i64];
        for i in 0..tau {
            let w = b.pow(i);
            values = values
                .iter()
                .flat_map(|v| (lo..=hi).map(move |digit| v + digit * w))
                .collect();
        }
        *values.iter().max().unwrap()
    }

    #[test]
    fn j_inv_capacity_matches_enumeration() {
        for b in 2u64..9 {
            for tau in 0..6 {
                assert_eq!(
                    j_inv_capacity(b, tau).unwrap(),
                    capacity_by_enumeration(b as i64, tau) as u128,
                    "b = {b}, tau = {tau}"
                );
            }
        }
    }

    #[test]
    fn tight_tau_is_least_representable_and_flips_at_capacity() {
        for b in 3u64..20 {
            for t in 1..8 {
                let cap = j_inv_capacity(b, t).unwrap();
                // At the capacity tau digits suffice; one beyond needs one more digit.
                assert_eq!(tau(TauRule::Tight, b, cap).unwrap(), t, "b = {b}");
                assert_eq!(tau(TauRule::Tight, b, cap + 1).unwrap(), t + 1, "b = {b}");
            }
        }
        assert_eq!(tau(TauRule::Tight, 16, 0).unwrap(), 0);
        assert!(matches!(
            tau(TauRule::Tight, 2, 5),
            Err(Error::TightTauUndefined { b: 2 })
        ));
    }

    #[test]
    fn draft_and_ceil_tau() {
        assert_eq!(tau(TauRule::Ceil, 16, 4096).unwrap(), 3);
        assert_eq!(tau(TauRule::Ceil, 16, 4097).unwrap(), 4);
        assert_eq!(tau(TauRule::Draft, 16, 4096).unwrap(), 4);
        assert_eq!(tau(TauRule::Draft, 16, 4097).unwrap(), 5);
        // The draft's rule always represents beta for b >= 3; the plain ceiling does not.
        for b in 3u64..12 {
            for beta in 1u128..3000 {
                let draft = tau(TauRule::Draft, b, beta).unwrap();
                assert!(j_inv_capacity(b, draft).unwrap() >= beta);
            }
        }
        let t = tau(TauRule::Ceil, 4, 49152).unwrap();
        assert!(j_inv_capacity(4, t).unwrap() < 49152);
    }

    /// Brute force |C| by enumerating Z^d vectors with the given constraint.
    fn count_by_enumeration(set: ChallengeSet, d: usize, omega: i64) -> u64 {
        fn rec(set: ChallengeSet, left: usize, omega: i64, l1: i64, nonzero: i64) -> u64 {
            if left == 0 {
                return match set {
                    ChallengeSet::L1Ball => u64::from(l1 <= omega),
                    ChallengeSet::Sparse => u64::from(nonzero == omega && l1 == omega),
                };
            }
            let range = match set {
                ChallengeSet::L1Ball => -omega..=omega,
                ChallengeSet::Sparse => -1..=1,
            };
            range
                .filter(|c| l1 + c.abs() <= omega)
                .map(|c| {
                    rec(
                        set,
                        left - 1,
                        omega,
                        l1 + c.abs(),
                        nonzero + i64::from(c != 0),
                    )
                })
                .sum()
        }
        rec(set, d, omega, 0, 0)
    }

    #[test]
    fn challenge_set_size_matches_enumeration() {
        for set in [ChallengeSet::L1Ball, ChallengeSet::Sparse] {
            for d in 1..=6usize {
                for omega in 0..=7i64 {
                    assert_eq!(
                        challenge_set_size(set, d as u64, omega as u64),
                        BigUint::from(count_by_enumeration(set, d, omega)),
                        "{set:?} d = {d} omega = {omega}"
                    );
                }
            }
        }
    }

    #[test]
    fn knowledge_error_flips_at_the_exact_power_of_two() {
        // |C| = 2^(r + 1 + lambda) holds; one less fails.
        let r = 3;
        let lambda = 10;
        let exact = BigUint::one() << (r + 1 + lambda);
        assert!(knowledge_error_holds(&exact, r, lambda));
        assert!(!knowledge_error_holds(&(exact - 1u32), r, lambda));
        // Toy preset: d = 16, omega = 2, |C| = 545, r = 2: 2^9 <= 545 < 2^10.
        let c = challenge_set_size(ChallengeSet::L1Ball, 16, 2);
        assert_eq!(c, BigUint::from(545u32));
        assert!(knowledge_error_holds(&c, 2, 6));
        assert!(!knowledge_error_holds(&c, 2, 7));
    }

    #[test]
    fn min_omega_is_minimal() {
        for (d, r, lambda) in [
            (16u64, 2u32, 6u32),
            (256, 8, 115),
            (1024, 7, 115),
            (64, 5, 40),
        ] {
            for set in [ChallengeSet::L1Ball, ChallengeSet::Sparse] {
                let w = min_omega(set, d, r, lambda).unwrap();
                assert!(knowledge_error_holds(
                    &challenge_set_size(set, d, w),
                    r,
                    lambda
                ));
                assert!(
                    w == 1 || !knowledge_error_holds(&challenge_set_size(set, d, w - 1), r, lambda)
                );
            }
        }
        // The sparse set of degree 16 has at most C(16, 11) 2^11 < 2^23 elements.
        assert!(matches!(
            min_omega(ChallengeSet::Sparse, 16, 2, 30),
            Err(Error::NoOmega { cap: 16, .. })
        ));
    }

    #[test]
    fn log2_of_big_integers() {
        assert_eq!(log2_biguint(&BigUint::from(1u32)), 0.0);
        assert_eq!(log2_biguint(&(BigUint::one() << 200u32)), 200.0);
        let x = BigUint::from(3u32) << 300u32;
        assert!((log2_biguint(&x) - (300.0 + 3f64.log2())).abs() < 1e-12);
        assert_eq!(log2_biguint(&BigUint::zero()), f64::NEG_INFINITY);
    }

    #[test]
    fn rejection_constant_matches_the_cascade_case() {
        // gamma = 2 n N beta_1 always passes.
        for (n, big_n, beta1) in [(1u64, 768u128, 32u128), (4, 1 << 20, 45056), (16, 3, 1)] {
            let gamma = 2 * u128::from(n) * big_n * beta1;
            assert!(rejection_constant_within(n, big_n, gamma, beta1).unwrap());
            // ln M lies just above 1/(2n) and well below 1/n.
            let ln_m = log2_rejection_constant(big_n, gamma, beta1) * core::f64::consts::LN_2;
            let half = 1.0 / (2 * n) as f64;
            assert!(ln_m >= half && ln_m <= half * (1.0 + 1e-3), "ln M = {ln_m}");
        }
        assert!(!rejection_constant_within(1, 10, 3, 4).unwrap());
        assert!(rejection_constant_within(1, 10, 3, 0).unwrap());
        assert!(rejection_constant_within(0, 10, 30, 1).is_err());
    }
}
