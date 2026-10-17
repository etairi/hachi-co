//! Challenges: the $\ell_1$ ball $C = \lbrace c \in R_q : \lVert c \rVert_1 \le \omega \rbrace$ of
//! 5_Building_Blocks.tex:122, sampled exactly uniformly, and the prototype's sparse set.
//!
//! The soundness analysis assumes that the verifier samples uniformly from $C$
//! (6_Collaborative_Hachi.tex:977-979). A challenge is returned as its $d$ coefficients, entry $j$
//! for $X^j$, as signed integers. Their $\ell_1$ norm is at most $\omega$, so for
//! $\omega \le (q-1)/2$ these are the centered representatives and the map to $R_q$ is a
//! bijection from $\lbrace c \in \mathbb Z^d : \lVert c \rVert_1 \le \omega \rbrace$ onto $C$
//! (the norm of 4_preliminaries.tex:86 uses centered representatives). The Setup bounds on $q$
//! (6_Collaborative_Hachi.tex:272) give $q \gt 2\omega$.
//!
//! **Size of the ball.** A vector with $i$ nonzero coefficients is given by its support
//! ($\binom{d}{i}$ choices), its signs ($2^i$), and the absolute values, a composition
//! $(a_1, \dots, a_i)$ of positive integers with sum at most $\omega$. These compositions are in
//! bijection with the $i$-subsets $s_1 \lt \dots \lt s_i$ of $[1, \omega]$ (partial sums
//! $s_j = a_1 + \dots + a_j$), so there are $\binom{\omega}{i}$ of them and
//! $$|C| = \sum_{i=0}^{\min(d, \omega)} 2^i \binom{d}{i} \binom{\omega}{i},$$
//! the Delannoy number $D(d, \omega)$.
//!
//! **Sampler.** [`sample`] follows the same bijection with exact draws: it picks the support size
//! $i$ with probability $2^i \binom{d}{i} \binom{\omega}{i} / |C|$ by one big-integer draw, then a
//! uniform $i$-subset of positions, a uniform $i$-subset of $[1, \omega]$ (the partial sums), and
//! $i$ uniform signs. Every vector of $C$ is reached by exactly one choice of
//! (support, partial sums, signs), and each such choice with support size $i$ has probability
//! $\frac{2^i \binom{d}{i} \binom{\omega}{i}}{|C|} \cdot \frac{1}{\binom{d}{i}} \cdot \frac{1}{\binom{\omega}{i}} \cdot \frac{1}{2^i} = \frac{1}{|C|}$.
//! The unit tests confirm this by enumerating every sequence of draws for small $(d, \omega)$.

use crate::draw::{Draw, FromRng};
use crate::{Error, Result};
use num_bigint::BigUint;
use num_traits::One;
use rand::Rng;
use std::collections::BTreeSet;

/// The weights $W_i = 2^i \binom{d}{i} \binom{\omega}{i}$ for $i = 0, \dots, \min(d, \omega)$.
fn weights(d: usize, omega: u64) -> Vec<BigUint> {
    let top = (d as u64).min(omega);
    let mut out = Vec::new();
    // C(d, i) and C(omega, i), updated by C(n, i + 1) = C(n, i) (n - i) / (i + 1), an exact division.
    let mut binom_d = BigUint::one();
    let mut binom_omega = BigUint::one();
    for i in 0..=top {
        out.push((&binom_d * &binom_omega) << i);
        if i < top {
            binom_d = binom_d * (d as u64 - i) / (i + 1);
            binom_omega = binom_omega * (omega - i) / (i + 1);
        }
    }
    out
}

/// $|C| = \sum_{i=0}^{\min(d, \omega)} 2^i \binom{d}{i} \binom{\omega}{i}$, the number of
/// $c \in \mathbb Z^d$ with $\lVert c \rVert_1 \le \omega$ (the Delannoy number $D(d, \omega)$;
/// derivation in the [module documentation](self)).
///
/// It takes $O(\min(d, \omega))$ big-integer operations. It determines the knowledge error
/// $2^{r+1}/|C|$ of the challenge set, which the parameters must keep below $2^{-\lambda}$
/// (docs/m0-spec.md, params constraints).
pub fn count(d: usize, omega: u64) -> BigUint {
    weights(d, omega).into_iter().sum()
}

/// The size $\binom{d}{\omega} 2^\omega$ of the sparse set of [`sample_sparse`] (zero if
/// $\omega \gt d$).
pub fn count_sparse(d: usize, omega: u64) -> BigUint {
    if omega > d as u64 {
        return BigUint::ZERO;
    }
    let mut binom = BigUint::one();
    for i in 0..omega {
        binom = binom * (d as u64 - i) / (i + 1);
    }
    binom << omega
}

/// Whether $\lVert c \rVert_1 \le \omega$, computed exactly (no overflow).
///
/// Every prover checks that the challenge lies in the challenge set before using it
/// (implementation rule of the plan, finding F5). For a challenge held as residues mod $q$,
/// pass its centered representatives.
pub fn contains(c: &[i64], omega: u64) -> bool {
    let norm: u128 = c.iter().map(|x| u128::from(x.unsigned_abs())).sum();
    norm <= u128::from(omega)
}

/// Whether $c$ has exactly $\omega$ nonzero coefficients, each $\pm 1$ (the sparse set of
/// [`sample_sparse`]).
pub fn contains_sparse(c: &[i64], omega: u64) -> bool {
    c.iter().all(|x| x.unsigned_abs() <= 1) && c.iter().filter(|x| **x != 0).count() as u64 == omega
}

/// The $\ell_1$ ball of radius $\omega$ in dimension $d$, with the support-size weights
/// precomputed for repeated sampling (a proof draws $2^r$ challenges,
/// 5_Building_Blocks.tex:122-126).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct L1Ball {
    d: usize,
    omega: u64,
    /// `cumulative[i]` $= W_0 + \dots + W_i$; the last entry is $|C|$.
    cumulative: Vec<BigUint>,
}

impl L1Ball {
    /// The ball $\lbrace c \in \mathbb Z^d : \lVert c \rVert_1 \le \omega \rbrace$. This computes the
    /// $\min(d, \omega) + 1$ weights $W_i$ (see [`sample`]) as big integers.
    ///
    /// # Errors
    /// [`Error::InvalidParameter`] if $\omega \gt 2^{63} - 1$ (coefficients are `i64`).
    pub fn new(d: usize, omega: u64) -> Result<Self> {
        if omega > i64::MAX as u64 {
            return Err(Error::InvalidParameter("omega must be at most 2^63 - 1"));
        }
        let mut cumulative = weights(d, omega);
        for i in 1..cumulative.len() {
            let previous = cumulative[i - 1].clone();
            cumulative[i] += previous;
        }
        Ok(Self {
            d,
            omega,
            cumulative,
        })
    }

    /// The dimension $d$.
    pub fn degree(&self) -> usize {
        self.d
    }

    /// The radius $\omega$.
    pub fn omega(&self) -> u64 {
        self.omega
    }

    /// $|C|$, as [`count`].
    pub fn count(&self) -> &BigUint {
        self.cumulative
            .last()
            .expect("the weight W_0 = 1 is always present")
    }

    /// A uniform element of the ball; see [`sample`] for the exact procedure.
    ///
    /// # Errors
    /// [`Error::RejectionLimit`] if the RNG is broken (see [`crate::uniform::MAX_TRIES`]).
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<i64>> {
        self.sample_with(&mut FromRng(rng))
    }

    /// [`L1Ball::sample`] on any [`Draw`].
    pub(crate) fn sample_with<D: Draw>(&self, draw: &mut D) -> Result<Vec<i64>> {
        let u = draw.below_big(self.count())?;
        // The first i with u < W_0 + ... + W_i; it exists because u < |C|.
        let i = self.cumulative.partition_point(|c| c <= &u) as u64;
        let positions = floyd(draw, self.d as u64, i)?;
        let partial_sums = floyd(draw, self.omega, i)?;
        let mut c = vec![0i64; self.d];
        let mut previous = 0u64;
        for (&position, &s) in positions.iter().zip(&partial_sums) {
            // s + 1 is the j-th smallest element of the subset of [1, omega].
            let a = (s + 1 - previous) as i64;
            previous = s + 1;
            c[position as usize] = if draw.below(2)? == 0 { a } else { -a };
        }
        Ok(c)
    }
}

/// A uniform element of $C = \lbrace c \in \mathbb Z^d : \lVert c \rVert_1 \le \omega \rbrace$
/// (5_Building_Blocks.tex:122), exactly.
///
/// Procedure (all draws exact, in this order; the draft does not fix a sampler):
/// 1. $u \leftarrow$ [`uniform_below_big`](crate::uniform::uniform_below_big)$(|C|)$, and $i$
///    the least index with $u \lt W_0 + \dots + W_i$, where
///    $W_i = 2^i \binom{d}{i} \binom{\omega}{i}$;
/// 2. the positions $p_1 \lt \dots \lt p_i$: a uniform $i$-subset of $[0, d)$ by Floyd's
///    algorithm (for $j = d - i, \dots, d - 1$: $t \leftarrow$ uniform in $[0, j]$; add $j$ if
///    $t$ is already chosen, else add $t$);
/// 3. the partial sums $s_1 \lt \dots \lt s_i$: a uniform $i$-subset of $[1, \omega]$, by
///    Floyd's algorithm on $[0, \omega)$ plus one;
/// 4. for $j = 1, \dots, i$: a sign bit $\sigma_j \leftarrow$ uniform in $\lbrace 0, 1 \rbrace$, and
///    $c_{p_j} = (-1)^{\sigma_j} (s_j - s_{j-1})$ with $s_0 = 0$.
///
/// The [module documentation](self) shows that the output is exactly uniform. This recomputes
/// the weights on every call; use [`L1Ball`] to draw many challenges.
///
/// # Errors
/// [`Error::InvalidParameter`] if $\omega \gt 2^{63} - 1$; [`Error::RejectionLimit`] if the
/// RNG is broken.
pub fn sample<R: Rng + ?Sized>(rng: &mut R, d: usize, omega: u64) -> Result<Vec<i64>> {
    L1Ball::new(d, omega)?.sample(rng)
}

/// A uniform element of the prototype's challenge set: exactly $\omega$ nonzero coefficients,
/// each $\pm 1$ (`crates/hachi/src/arithmetic/poly_chal.rs`). The draft's set is the $\ell_1$
/// ball of [`sample`]; this one is kept for comparison (plan, question A8).
///
/// Procedure: the positions are a uniform $\omega$-subset of $[0, d)$ by Floyd's algorithm (as in
/// step 2 of [`sample`]); then, in increasing order of position, a sign bit
/// $\sigma \leftarrow$ uniform in $\lbrace 0, 1 \rbrace$ gives the coefficient $(-1)^\sigma$. The
/// prototype's own sampler (a partial Fisher-Yates shuffle) draws the same distribution from a
/// different stream.
///
/// # Errors
/// [`Error::InvalidParameter`] if $\omega \gt d$ (the set is empty); [`Error::RejectionLimit`]
/// if the RNG is broken.
pub fn sample_sparse<R: Rng + ?Sized>(rng: &mut R, d: usize, omega: u64) -> Result<Vec<i64>> {
    if omega > d as u64 {
        return Err(Error::InvalidParameter("sparse challenges need omega <= d"));
    }
    sparse_with(&mut FromRng(rng), d, omega)
}

/// [`sample_sparse`] on any [`Draw`], for $\omega \le d$.
pub(crate) fn sparse_with<D: Draw>(draw: &mut D, d: usize, omega: u64) -> Result<Vec<i64>> {
    let mut c = vec![0i64; d];
    for position in floyd(draw, d as u64, omega)? {
        c[position as usize] = if draw.below(2)? == 0 { 1 } else { -1 };
    }
    Ok(c)
}

/// A uniform $k$-subset of $[0, n)$, in increasing order, by Floyd's algorithm (Bentley and
/// Floyd, "A sample of brilliance", CACM 30(9), 1987): for $j = n - k, \dots, n - 1$, draw $t$
/// uniform in $[0, j]$ and add $j$ if $t$ is already in the set, else $t$. It makes exactly $k$
/// draws, of ranges $n - k + 1, \dots, n$; a range of $1$ reads nothing.
pub(crate) fn floyd<D: Draw>(draw: &mut D, n: u64, k: u64) -> Result<Vec<u64>> {
    debug_assert!(k <= n, "subset larger than the set");
    let mut set = BTreeSet::new();
    for j in n - k..n {
        let t = draw.below(j + 1)?;
        if !set.insert(t) {
            // Before this step the set lies in [0, j), so j is new.
            set.insert(j);
        }
    }
    Ok(set.into_iter().collect())
}

#[cfg(test)]
mod tests;
