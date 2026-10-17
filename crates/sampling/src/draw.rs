//! The source of exact uniform draws that the composite samplers are written against.
//!
//! [`l1ball`](crate::l1ball), [`rej::sim_rej`](crate::rej::sim_rej) and the box samplers call
//! only [`Draw::below`] and [`Draw::below_big`]. In production the draws come from an RNG
//! through [`FromRng`]; the unit tests replace it by an enumerator of every sequence of draws
//! (`crate::enumerate`), which yields the exact output distribution of a sampler.

use crate::Result;
use crate::uniform::{uniform_below, uniform_below_big};
use num_bigint::BigUint;
use rand::Rng;

/// Exact uniform draws from $[0, n)$.
pub(crate) trait Draw {
    /// A uniform integer in $[0, n)$, for $n \ge 1$.
    fn below(&mut self, n: u64) -> Result<u64>;
    /// A uniform integer in $[0, n)$, for $n \ge 1$.
    fn below_big(&mut self, n: &BigUint) -> Result<BigUint>;
}

/// The draws of [`uniform_below`] and [`uniform_below_big`] on an RNG.
pub(crate) struct FromRng<'a, R: ?Sized>(pub(crate) &'a mut R);

impl<R: Rng + ?Sized> Draw for FromRng<'_, R> {
    fn below(&mut self, n: u64) -> Result<u64> {
        uniform_below(self.0, n)
    }

    fn below_big(&mut self, n: &BigUint) -> Result<BigUint> {
        uniform_below_big(self.0, n)
    }
}
