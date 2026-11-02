//! Discrete Gaussians $D_{\mathbb Z,\sigma}$ centred at 0: for $\sigma = 2^j$ with
//! $0 \le j \le$ [`MAX_LOG_SIGMA`] from tables, and for the multiples of 16 from the last table
//! ([`WideGaussian`]), the masks of hiding Hachi.
//!
//! $\sigma$ is the standard deviation: $\Pr\[x\] \propto \exp(-x^2/(2\sigma^2))$. The width of GPV
//! and of Hwang-Seo-Song is $s = \sigma\sqrt{2\pi}$.
//!
//! **Construction** (ntru-trapdoor's `src/keygen/gauss.rs`, same author, MIT, with entries of 192
//! bits instead of 128). Each try reads four 64-bit words, little-endian, from the source:
//! - $V = w_0 + 2^{64} w_1 + 2^{128} w_2$, uniform in $\[0, 2^{192})$
//! - a sign bit $b$, the low bit of $w_3$.
//!
//! Then $K = \\#\lbrace j : V < T_j \rbrace$ for the table $T$ of [`crate::gauss_tables`], so
//! $\Pr\[K > j\] = T_j / 2^{192}$. A try returns $-K$ if $b = 0$, and $K$ if $b = 1$ and $K \ne 0$.
//! It is repeated if $b = 1$ and $K = 0$. So a try returns $x$ with probability
//! $\frac12 \Pr\[K = \lvert x \rvert\]$ for every $x$, 0 included (0 only with $b = 0$), and since
//! $\Pr\[K = i\] \propto \rho(i)$, a sample has $\Pr\[x\] \propto \rho(x)$, up to the rounding of
//! the table.
//!
//! **Support and distance.**
//! - Every sample has absolute value at most the table's length, [`bound`] ($16\sigma$ to
//!   $16.2\sigma$).
//! - The construction is within statistical distance about $2^{-190.6}$ ($\sigma = 1$) to
//!   $2^{-186.6}$ ($\sigma = 16$) of $D_{\mathbb Z,\sigma}$. This was computed in Sage against the
//!   exact distribution, independently of the table generator (`tests/fixtures/gauss_check.sage`).
//!   So $N$ samples are within $N \cdot 2^{-186.5}$ of $N$ exact ones, below $2^{-128}$ for $N$ up
//!   to $2^{58}$. With the 128-bit tables of earlier versions the distance was about $2^{-122.9}$
//!   per sample.
//! - The far tail rows have relative errors of up to $2^{0.7}$, so the tables support
//!   statistical-distance arguments, not Rényi-divergence ones.
//!
//! **Time.** The count of $K$, the repetition and the sign make the time depend on the sample.
//! As everywhere in this crate, constant time is not a goal.
//!
//! **Wide Gaussians** ([`WideGaussian`]). The masks of hiding Hachi need $\sigma$ far above the
//! tables, so $\sigma = 16k$ is reached from the table of $\sigma_0 = 16$ as in BLISS (Ducas,
//! Durmus, Lepoint, Lyubashevsky, CRYPTO 2013, Section 6):
//! 1. draw $K \ge 0$ from the table, so $\Pr\[K = i\] \propto \rho_{\sigma_0}(i)$, and $y$ uniform
//!    in $\[0, k)$, and set $z = kK + y$. Every integer $z \ge 0$ has one such form, and it is
//!    proposed with probability proportional to $\exp(-(kK)^2 / (2\sigma^2))$.
//! 2. keep $z$ with probability $\exp(-(z^2 - (kK)^2) / (2\sigma^2)) = \exp(-y(y + 2kK) /
//!    (512 k^2))$, from [`bernoulli_exp`]. A kept $z$ then has probability proportional to
//!    $\exp(-z^2 / (2\sigma^2))$.
//! 3. give it a sign by the rule of the tables: a bit $b$, $-z$ if $b = 0$, $z$ if $b = 1$ and
//!    $z \ne 0$, and start again if $b = 1$ and $z = 0$.
//!
//! Step 2 is exact: the exponent is a ratio of integers and [`bernoulli_exp`] uses no floating
//! point. So the only distance from $D_{\mathbb Z,\sigma}$ comes from the table of step 1. With
//! $r(K)$ the ratio of the table's $\Pr\[K\]$ to the exact one, a sample in block
//! $K = \lfloor \lvert x \rvert / k \rfloor$ has probability $D_{\mathbb Z,\sigma}(x) r(K) /
//! \mathbb E\[r\]$. So the distance can be computed, and `tests/fixtures/gauss_check.sage` does
//! that with the real table: $2^{-186.58}$ per sample at $k = 1$, falling with $k$ at the values
//! computed, to $2^{-186.91}$ in the limit. Samples have absolute value below
//! `(table length + 1) k`, about $16.3\sigma$, and a proposal is kept with probability about
//! $0.976$ at large $k$.

use crate::gauss_tables::{
    Entry, GAUSS_SIGMA_1, GAUSS_SIGMA_2, GAUSS_SIGMA_4, GAUSS_SIGMA_8, GAUSS_SIGMA_16,
};
use crate::uniform::MAX_TRIES;
use crate::{Error, Result};
use rand::Rng;

/// The largest $j$ with a table: $\sigma = 2^j \le 16$.
pub const MAX_LOG_SIGMA: u32 = 4;

/// The table of $\sigma = 2^j$, $j$ = `log_sigma`.
///
/// # Errors
/// [`Error::InvalidParameter`] for $j \gt$ [`MAX_LOG_SIGMA`].
pub fn table(log_sigma: u32) -> Result<&'static [Entry]> {
    Ok(match log_sigma {
        0 => &GAUSS_SIGMA_1,
        1 => &GAUSS_SIGMA_2,
        2 => &GAUSS_SIGMA_4,
        3 => &GAUSS_SIGMA_8,
        4 => &GAUSS_SIGMA_16,
        _ => return Err(Error::InvalidParameter("log_sigma above MAX_LOG_SIGMA")),
    })
}

/// The bound on the absolute value of a sample of $\sigma = 2^j$: the length of its table.
///
/// # Errors
/// As [`table`].
pub fn bound(log_sigma: u32) -> Result<u64> {
    Ok(table(log_sigma)?.len() as u64)
}

/// `len` samples of $D_{\mathbb Z,\sigma}$ with $\sigma = 2^j$, $j$ = `log_sigma` (module
/// documentation), sample after sample.
///
/// # Errors
/// - As [`table`].
/// - [`Error::RejectionLimit`] if a sample needs [`MAX_TRIES`] tries. A try is repeated with
///   probability $\frac12 \Pr\[K = 0\] \lt \frac12$, so a uniform source does this with probability
///   below $2^{-256}$.
pub fn gaussian<R: Rng + ?Sized>(rng: &mut R, log_sigma: u32, len: usize) -> Result<Vec<i64>> {
    let table = table(log_sigma)?;
    (0..len).map(|_| sample(rng, table)).collect()
}

/// One sample from `table` (module documentation).
fn sample<R: Rng + ?Sized>(rng: &mut R, table: &[Entry]) -> Result<i64> {
    for _ in 0..MAX_TRIES {
        let k = magnitude(rng, table) as i64;
        let sign = rng.next_u64() & 1;
        if sign == 0 {
            return Ok(-k);
        }
        if k != 0 {
            return Ok(k);
        }
    }
    Err(Error::RejectionLimit(MAX_TRIES))
}

/// $K = \\#\lbrace j : V < T_j \rbrace$ for $V$ read from three words (module documentation).
fn magnitude<R: Rng + ?Sized>(rng: &mut R, table: &[Entry]) -> u64 {
    let (w0, w1, w2) = (rng.next_u64(), rng.next_u64(), rng.next_u64());
    // V = w0 + 2^64 w1 + 2^128 w2, as an entry: most significant limb first.
    let v: Entry = [w2, w1, w0];
    // The entries are non-increasing, so {j : V < T_j} is a prefix.
    table.partition_point(|t| v < *t) as u64
}

/// A value uniform in $\[0, n)$, for $n \ge 1$: masked words, repeated until below $n$.
fn below<R: Rng + ?Sized>(rng: &mut R, n: u128) -> Result<u128> {
    if n == 1 {
        return Ok(0);
    }
    let bits = u128::BITS - (n - 1).leading_zeros();
    let mask = u128::MAX >> (u128::BITS - bits);
    for _ in 0..MAX_TRIES {
        let low = u128::from(rng.next_u64());
        let v = if bits > 64 {
            low | (u128::from(rng.next_u64()) << 64)
        } else {
            low
        };
        if v & mask < n {
            return Ok(v & mask);
        }
    }
    Err(Error::RejectionLimit(MAX_TRIES))
}

/// A bit that is true with probability exactly $\exp(-\gamma)$, for $\gamma \in \[0, 1\]$ given
/// as `num` / `den` with `num` $\le$ `den`, by Algorithm 1 of Canonne, Kamath and Steinke (NeurIPS
/// 2020): count the Bernoulli($\gamma / i$) successes from $i = 1$ up to the first failure, and
/// return whether that failure came at an odd $i$.
fn bernoulli_exp_unit<R: Rng + ?Sized>(rng: &mut R, num: u128, den: u128) -> Result<bool> {
    let mut i: u128 = 1;
    loop {
        let bound = den
            .checked_mul(i)
            .ok_or(Error::InvalidParameter("the denominator is too large"))?;
        if below(rng, bound)? >= num {
            return Ok(i % 2 == 1);
        }
        i += 1;
    }
}

/// A bit that is true with probability exactly $\exp(-\mathit{num}/\mathit{den})$, given a uniform
/// source. For $\gamma \ge 1$ it multiplies $\lfloor \gamma \rfloor$ draws of $\exp(-1)$ with one
/// of the remainder, stopping at the first false draw.
///
/// # Errors
/// - [`Error::InvalidParameter`] if `den` is 0, or too large for the small multiples of it that
///   the draws need (a denominator below $2^{120}$ overflows with probability below $2^{-200}$).
/// - [`Error::RejectionLimit`] as for the uniform samplers.
pub fn bernoulli_exp<R: Rng + ?Sized>(rng: &mut R, num: u128, den: u128) -> Result<bool> {
    if den == 0 {
        return Err(Error::InvalidParameter("the denominator is zero"));
    }
    let mut rest = num;
    while rest >= den {
        if !bernoulli_exp_unit(rng, den, den)? {
            return Ok(false);
        }
        rest -= den;
    }
    bernoulli_exp_unit(rng, rest, den)
}

/// The standard deviation of the table that [`WideGaussian`] starts from.
pub const WIDE_BASE: u64 = 16;

/// The largest $\sigma$ of a [`WideGaussian`]: then $\lvert z \rvert \lt 2^{63}$, and every
/// product of step 2 fits 128 bits.
pub const MAX_WIDE_SIGMA: u64 = 1 << 58;

/// $D_{\mathbb Z,\sigma}$ for a multiple $\sigma = 16k$ of [`WIDE_BASE`] (module documentation,
/// "Wide Gaussians").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WideGaussian {
    /// $k = \sigma / 16$.
    k: u64,
    /// The denominator $512 k^2$ of step 2.
    den: u128,
    /// The table of $\sigma_0 = 16$.
    table: &'static [Entry],
}

impl WideGaussian {
    /// The sampler of standard deviation `sigma`.
    ///
    /// # Errors
    /// [`Error::InvalidParameter`] unless `sigma` is a positive multiple of [`WIDE_BASE`] of at
    /// most [`MAX_WIDE_SIGMA`].
    pub fn new(sigma: u64) -> Result<Self> {
        if sigma == 0 || !sigma.is_multiple_of(WIDE_BASE) || sigma > MAX_WIDE_SIGMA {
            return Err(Error::InvalidParameter(
                "sigma must be a positive multiple of 16 of at most 2^58",
            ));
        }
        let k = sigma / WIDE_BASE;
        Ok(Self {
            k,
            den: 512 * u128::from(k) * u128::from(k),
            table: &GAUSS_SIGMA_16,
        })
    }

    /// $\sigma = 16k$.
    pub fn sigma(&self) -> u64 {
        self.k * WIDE_BASE
    }

    /// A strict bound on the absolute value of a sample: `(table length + 1)` $k$.
    pub fn bound(&self) -> u64 {
        (self.table.len() as u64 + 1) * self.k
    }

    /// One sample. A try reads three words for $K$, then one or more for $y$ and the sign, then
    /// what [`bernoulli_exp`] needs.
    ///
    /// # Errors
    /// As [`bernoulli_exp`]. [`Error::RejectionLimit`] if a sample needs [`MAX_TRIES`] tries, each
    /// of which fails with probability below $3/5$.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<i64> {
        let k = u128::from(self.k);
        // The bits of y, below the sign bit of the same word.
        let bits = 64 - (self.k - 1).leading_zeros();
        let mask = (1u64 << bits) - 1;
        for _ in 0..MAX_TRIES {
            let big = magnitude(rng, self.table);
            let (y, sign) = 'draw: {
                for _ in 0..MAX_TRIES {
                    let word = rng.next_u64();
                    let y = (word >> 1) & mask;
                    if y < self.k {
                        break 'draw (y, word & 1);
                    }
                }
                return Err(Error::RejectionLimit(MAX_TRIES));
            };
            let y = u128::from(y);
            let exponent = y * (y + 2 * k * u128::from(big));
            if !bernoulli_exp(rng, exponent, self.den)? {
                continue;
            }
            let z = (big * self.k + y as u64) as i64;
            if sign == 0 {
                return Ok(-z);
            }
            if z != 0 {
                return Ok(z);
            }
        }
        Err(Error::RejectionLimit(MAX_TRIES))
    }

    /// Fills `out` with samples, in order.
    ///
    /// # Errors
    /// As [`WideGaussian::sample`].
    pub fn fill<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [i64]) -> Result<()> {
        for x in out {
            *x = self.sample(rng)?;
        }
        Ok(())
    }
}
