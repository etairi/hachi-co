//! Discrete Gaussians $D_{\mathbb Z,\sigma}$ centred at 0, for $\sigma = 2^j$ with
//! $0 \le j \le$ [`MAX_LOG_SIGMA`]: the commitment randomness of hiding Hachi
//! (`docs/hachi-hiding.md`, 6.4).
//!
//! $\sigma$ is the standard deviation: $\Pr\[x\] \propto \exp(-x^2/(2\sigma^2))$. The width of GPV
//! and of Hwang-Seo-Song is $s = \sigma\sqrt{2\pi}$.
//!
//! **Construction** (ntru-trapdoor's `src/keygen/gauss.rs`, same author, MIT). Each try reads three
//! 64-bit words, little-endian, from the source:
//! - $V = w_0 + 2^{64} w_1$, uniform in $\[0, 2^{128})$;
//! - a sign bit $b$, the low bit of $w_2$.
//!
//! Then $K = \\#\lbrace j : V < T_j \rbrace$ for the table $T$ of [`crate::gauss_tables`], so
//! $\Pr\[K > j\] = T_j / 2^{128}$. A try returns $-K$ if $b = 0$, and $K$ if $b = 1$ and $K \ne 0$.
//! It is repeated if $b = 1$ and $K = 0$. So a try returns $x$ with probability
//! $\frac12 \Pr\[K = \lvert x \rvert\]$ for every $x$, 0 included (0 only with $b = 0$), and since
//! $\Pr\[K = i\] \propto \rho(i)$, a sample has $\Pr\[x\] \propto \rho(x)$, up to the rounding of
//! the table.
//!
//! **Support and distance.**
//! - Every sample has absolute value at most the table's length, [`bound`] ($13\sigma$ to
//!   $13.2\sigma$).
//! - The construction is within statistical distance about $2^{-126.2}$ ($\sigma = 1$) to
//!   $2^{-122.9}$ ($\sigma = 16$; $2^{-122.87}$ unrounded) of $D_{\mathbb Z,\sigma}$. This was
//!   computed in Sage against the exact distribution, independently of the table generator
//!   (`tests/fixtures/gauss_check.sage`).
//! - The far tail rows have relative errors of up to $2^{1.3}$, so the tables support
//!   statistical-distance arguments, not Rényi-divergence ones.
//!
//! **Time.** The count of $K$, the repetition and the sign make the time depend on the sample.
//! As everywhere in this crate, constant time is not a goal.

use crate::gauss_tables::{
    GAUSS_SIGMA_1, GAUSS_SIGMA_2, GAUSS_SIGMA_4, GAUSS_SIGMA_8, GAUSS_SIGMA_16,
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
pub fn table(log_sigma: u32) -> Result<&'static [u128]> {
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
fn sample<R: Rng + ?Sized>(rng: &mut R, table: &[u128]) -> Result<i64> {
    for _ in 0..MAX_TRIES {
        let low = u128::from(rng.next_u64());
        let v = low | (u128::from(rng.next_u64()) << 64);
        let sign = rng.next_u64() & 1;
        // The entries are non-increasing, so {j : V < T_j} is a prefix.
        let k = table.partition_point(|t| v < *t) as i64;
        if sign == 0 {
            return Ok(-k);
        }
        if k != 0 {
            return Ok(k);
        }
    }
    Err(Error::RejectionLimit(MAX_TRIES))
}
