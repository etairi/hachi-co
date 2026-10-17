//! Exact uniform samplers by rejection.
//!
//! Every sampler here applies one rejection rule, that of [`uniform_below`] ([`uniform_below_big`]
//! applies it to several words at once): draw a 64-bit word, keep its low
//! $b = \lceil \log_2 n \rceil$ bits, and accept the value if it is below $n$. The kept bits are
//! uniform on $[0, 2^b)$, so an accepted value is exactly uniform on $[0, n)$, and each try
//! succeeds with probability $n / 2^b \gt 1/2$.
//!
//! Every vector sampler draws its entries in order, each from its own words, so the first $k$
//! entries of a length-$n$ draw equal a length-$k$ draw from the same stream. (jali packs
//! $b$-bit words; one word per try is simpler to reproduce from Python or Sage, which the
//! known-answer tests do, at the cost of more stream bytes.)

// Ported from jali (MIT, (c) 2026 Erkan Tairi): src/rand/uniform.rs (`uniform`, `bounded`), with one
// 64-bit word per try instead of packed words.

use crate::draw::{Draw, FromRng};
use crate::{Error, Result};
use num_bigint::BigUint;
use num_traits::{One, Zero};
use rand::Rng;

/// The cap on the tries of one rejection loop. Each try succeeds with probability above $1/2$,
/// so a uniform source reaches the cap with probability below $2^{-256}$; reaching it returns
/// [`Error::RejectionLimit`].
pub const MAX_TRIES: u32 = 256;

/// A uniform integer in $[0, n)$, exactly.
///
/// For $n = 1$ the result is $0$ and nothing is read. Otherwise, with
/// $b = \lceil \log_2 n \rceil$ (the bit length of $n - 1$), each try reads one word
/// $w$ = `rng.next_u64()` and returns $w \bmod 2^b$ if that is below $n$.
///
/// # Errors
/// [`Error::EmptyRange`] if $n = 0$; [`Error::RejectionLimit`] after [`MAX_TRIES`] failed tries.
pub fn uniform_below<R: Rng + ?Sized>(rng: &mut R, n: u64) -> Result<u64> {
    match n {
        0 => Err(Error::EmptyRange),
        1 => Ok(0),
        _ => {
            let mask = u64::MAX >> (n - 1).leading_zeros();
            for _ in 0..MAX_TRIES {
                let w = rng.next_u64() & mask;
                if w < n {
                    return Ok(w);
                }
            }
            Err(Error::RejectionLimit(MAX_TRIES))
        }
    }
}

/// A uniform integer in $[0, n)$ for a big-integer bound, exactly.
///
/// For $n = 1$ the result is $0$ and nothing is read. Otherwise, with $b$ the bit length of
/// $n - 1$ and $L = \lceil b / 64 \rceil$, each try reads $L$ words $w_0, \dots, w_{L-1}$ and
/// returns $x = \left(\sum_k w_k 2^{64k}\right) \bmod 2^b$ if $x \lt n$. For $n \le 2^{64}$ this
/// reads the same words and returns the same value as [`uniform_below`].
///
/// # Errors
/// [`Error::EmptyRange`] if $n = 0$; [`Error::RejectionLimit`] after [`MAX_TRIES`] failed tries.
pub fn uniform_below_big<R: Rng + ?Sized>(rng: &mut R, n: &BigUint) -> Result<BigUint> {
    if n.is_zero() {
        return Err(Error::EmptyRange);
    }
    if n.is_one() {
        return Ok(BigUint::zero());
    }
    let bits = (n - 1u32).bits();
    let words = bits.div_ceil(64) as usize;
    let top_mask = u64::MAX >> (64 * words as u64 - bits);
    let mut limbs = vec![0u64; words];
    for _ in 0..MAX_TRIES {
        for limb in limbs.iter_mut() {
            *limb = rng.next_u64();
        }
        limbs[words - 1] &= top_mask;
        let bytes: Vec<u8> = limbs.iter().flat_map(|limb| limb.to_le_bytes()).collect();
        let x = BigUint::from_bytes_le(&bytes);
        if &x < n {
            return Ok(x);
        }
    }
    Err(Error::RejectionLimit(MAX_TRIES))
}

/// A uniform integer in the closed interval $[lo, hi]$, exactly.
///
/// Returns $lo + u$ with $u$ = [`uniform_below`]$(hi - lo + 1)$. For the full range
/// $hi - lo + 1 = 2^{64}$, every word is accepted and $u$ is one word.
///
/// # Errors
/// [`Error::EmptyRange`] if $lo \gt hi$; [`Error::RejectionLimit`] as in [`uniform_below`].
pub fn uniform_range_i64<R: Rng + ?Sized>(rng: &mut R, lo: i64, hi: i64) -> Result<i64> {
    if lo > hi {
        return Err(Error::EmptyRange);
    }
    let width = i128::from(hi) - i128::from(lo) + 1;
    let offset = match u64::try_from(width) {
        Ok(width) => uniform_below(rng, width)?,
        // width = 2^64: the mask keeps all 64 bits and every word is below the bound.
        Err(_) => rng.next_u64(),
    };
    Ok((i128::from(lo) + i128::from(offset)) as i64)
}

/// `len` residues uniform in $[0, q)$ (an element of $\mathbb Z_q^{len}$, or of $R_q$ for
/// `len` $= d$, coefficient $j$ for $X^j$). Each entry is [`uniform_below`]$(q)$.
///
/// This is the distribution of the public matrices $A, B, D \leftarrow R_q^{\cdot \times \cdot}$
/// of Setup (6_Collaborative_Hachi.tex:283-287); see [`crate::matrix`].
///
/// # Errors
/// [`Error::EmptyRange`] if $q = 0$; [`Error::RejectionLimit`] as in [`uniform_below`].
pub fn uniform_mod_q<R: Rng + ?Sized>(rng: &mut R, q: u64, len: usize) -> Result<Vec<u64>> {
    if q == 0 {
        return Err(Error::EmptyRange);
    }
    (0..len).map(|_| uniform_below(rng, q)).collect()
}

/// `len` integers uniform in $[-\gamma, \gamma]$: a draw from the uniform distribution on the box
/// $[-\gamma, \gamma]^{len}$, which is $\chi_r$ of the rejection-sampling paragraph
/// (4_preliminaries.tex:117) for `len` $= N$. Each entry is
/// [`uniform_below`]$(2\gamma + 1) - \gamma$. For secret masks use [`crate::rej::mask`], which
/// demands a cryptographic RNG.
///
/// # Errors
/// [`Error::InvalidParameter`] if $\gamma \gt 2^{63} - 1$ (entries are `i64`);
/// [`Error::RejectionLimit`] as in [`uniform_below`].
pub fn boxed<R: Rng + ?Sized>(rng: &mut R, gamma: u64, len: usize) -> Result<Vec<i64>> {
    boxed_with(&mut FromRng(rng), gamma, len)
}

/// [`boxed`] on any [`Draw`].
pub(crate) fn boxed_with<D: Draw>(draw: &mut D, gamma: u64, len: usize) -> Result<Vec<i64>> {
    if gamma > i64::MAX as u64 {
        return Err(Error::InvalidParameter("gamma must be at most 2^63 - 1"));
    }
    let width = 2 * gamma + 1;
    (0..len)
        .map(|_| Ok((i128::from(draw.below(width)?) - i128::from(gamma)) as i64))
        .collect()
}

/// `len` integers uniform in $[\lceil -\beta/2 \rceil, \lceil \beta/2 \rceil - 1]$, the coefficient
/// range of $S_\beta$ (4_preliminaries.tex:88); for $\beta = b$ these are the uniform secrets
/// $S_b$ of Module-LWE (4_preliminaries.tex:114) and the commitment randomness
/// $\rho_i \leftarrow S_b^{m_\rho}$ (5_Building_Blocks.tex:110).
///
/// The range has exactly $\beta$ integers, from $-\lfloor \beta/2 \rfloor$ to
/// $\lceil \beta/2 \rceil - 1$; each entry is [`uniform_below`]$(\beta) - \lfloor \beta/2 \rfloor$.
///
/// # Errors
/// [`Error::EmptyRange`] if $\beta = 0$; [`Error::RejectionLimit`] as in [`uniform_below`].
pub fn short<R: Rng + ?Sized>(rng: &mut R, beta: u64, len: usize) -> Result<Vec<i64>> {
    if beta == 0 {
        return Err(Error::EmptyRange);
    }
    let lo = -i128::from(beta / 2);
    (0..len)
        .map(|_| Ok((lo + i128::from(uniform_below(rng, beta)?)) as i64))
        .collect()
}
