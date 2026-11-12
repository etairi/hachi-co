//! The sets $R\_p \subseteq \[L\]$ of the iterations of the rejection sampling that prover $p$
//! accepted, and the selected iteration $j^\ast$ (`fig:ring-to-linear`, steps 5-6, since the
//! draft's commit `1096ea2`): "Let $R\_i \subseteq \[L\]$ be the set of all iterations $j \in \[L\]$
//! for which $\mathbf z\_i^j \neq \bot$. Each prover $i \in \[n\]$ broadcasts $R\_i$ to the provers
//! and verifier, who each locally compute $j^\ast \in \[L\]$ to be the smallest value in
//! $\bigcap\_{i \in \[n\]} R\_i$."
//!
//! A set is $\lceil L/64 \rceil$ little-endian words ([`hachi_co_params::sizes::accept_set_words`]):
//! iteration $j$ (0-indexed, the draft's $j + 1$) is bit $j \bmod 64$ of word $\lfloor j/64 \rfloor$.
//! These functions take proof data and never panic on it.

use crate::{Error, Result};
use hachi_co_core::Context;
use hachi_co_mpc::PartyId;

/// The number of words of a set of accepted iterations, $\lceil L/64 \rceil$.
pub fn accept_set_len<const K: usize>(ctx: &Context<K>) -> usize {
    ctx.layout.rej_iterations.div_ceil(64)
}

/// The set of the iterations $j$ with `accepted[j]`, as [`accept_set_len`] words.
///
/// # Errors
/// [`Error::Shape`] unless there are $L$ flags.
pub fn encode_accept_set<const K: usize>(ctx: &Context<K>, accepted: &[bool]) -> Result<Vec<u64>> {
    let iterations = ctx.layout.rej_iterations;
    crate::local::check_len(
        "accept flags (one per iteration)",
        iterations,
        accepted.len(),
    )?;
    let mut words = vec![0u64; accept_set_len(ctx)];
    for (j, _) in accepted.iter().enumerate().filter(|(_, a)| **a) {
        words[j / 64] |= 1 << (j % 64);
    }
    Ok(words)
}

/// Checks that `words`, the set of party `party`, is a set of iterations below $L$: it has
/// [`accept_set_len`] words and no bit at or beyond $L$ (so that every set has one encoding).
///
/// # Errors
/// [`Error::AcceptSet`] otherwise.
pub fn check_accept_set<const K: usize>(
    ctx: &Context<K>,
    party: PartyId,
    words: &[u64],
) -> Result<()> {
    if words.len() != accept_set_len(ctx) {
        return Err(Error::AcceptSet {
            party,
            what: "it does not have ceil(L/64) words",
        });
    }
    let iterations = ctx.layout.rej_iterations;
    let beyond = match words.last() {
        Some(last) if !iterations.is_multiple_of(64) => last >> (iterations % 64) != 0,
        _ => false,
    };
    if beyond {
        return Err(Error::AcceptSet {
            party,
            what: "it names an iteration at or beyond L",
        });
    }
    Ok(())
}

/// Whether iteration $j$ is in the set `words`.
pub fn accepts(words: &[u64], j: usize) -> bool {
    words.get(j / 64).is_some_and(|w| w >> (j % 64) & 1 == 1)
}

/// $j^\ast$, the smallest iteration in every set of `sets` (`sets[p - 1]` of prover $p$), each
/// checked by [`check_accept_set`] first; `None` if the intersection is empty.
///
/// # Errors
/// [`Error::Shape`] unless there are $n$ sets; [`Error::AcceptSet`] for a malformed set.
pub fn select_iteration<const K: usize>(
    ctx: &Context<K>,
    sets: &[Vec<u64>],
) -> Result<Option<usize>> {
    crate::local::check_len("accept sets (one per prover)", ctx.layout.n, sets.len())?;
    let mut common = vec![u64::MAX; accept_set_len(ctx)];
    for (p0, set) in sets.iter().enumerate() {
        check_accept_set(ctx, p0 + 1, set)?;
        for (c, w) in common.iter_mut().zip(set) {
            *c &= w;
        }
    }
    Ok(common
        .iter()
        .enumerate()
        .find(|(_, w)| **w != 0)
        .map(|(i, w)| i * 64 + w.trailing_zeros() as usize))
}
