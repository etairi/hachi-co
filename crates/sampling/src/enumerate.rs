//! Test only: the exact output distribution of a sampler written against [`Draw`], obtained by
//! running it on every sequence of draws.
//!
//! A run is a path in the sampler's tree of draws; a path whose draws have ranges
//! $n_1, \dots, n_L$ has probability $1 / (n_1 \cdots n_L)$ under exact uniform draws. The paths
//! are visited in lexicographic order (depth first): after each run, the deepest draw that can
//! still be incremented is incremented and everything after it is dropped, and the next run
//! replays that prefix and extends it with zeros.

use crate::Result;
use crate::draw::Draw;
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use std::collections::{BTreeMap, BTreeSet};

/// A [`Draw`] that replays a prefix of choices, then chooses $0$, recording every range.
pub(crate) struct Replay {
    path: Vec<BigUint>,
    ranges: Vec<BigUint>,
}

impl Replay {
    fn take(&mut self, n: BigUint) -> BigUint {
        assert!(!n.is_zero(), "a draw from an empty range");
        let k = self.ranges.len();
        if k == self.path.len() {
            self.path.push(BigUint::zero());
        }
        let choice = self.path[k].clone();
        assert!(
            choice < n,
            "the replayed prefix does not fit the draw ranges"
        );
        self.ranges.push(n);
        choice
    }
}

impl Draw for Replay {
    fn below(&mut self, n: u64) -> Result<u64> {
        Ok(self
            .take(BigUint::from(n))
            .to_u64()
            .expect("below n <= u64::MAX"))
    }

    fn below_big(&mut self, n: &BigUint) -> Result<BigUint> {
        Ok(self.take(n.clone()))
    }
}

/// An exact distribution: outcome $x$ has probability `weights[x] / denominator`.
pub(crate) struct Exact<T> {
    pub(crate) weights: BTreeMap<T, BigUint>,
    pub(crate) denominator: BigUint,
    /// The number of paths visited.
    pub(crate) paths: usize,
}

fn gcd(a: &BigUint, b: &BigUint) -> BigUint {
    let (mut a, mut b) = (a.clone(), b.clone());
    while !b.is_zero() {
        let r = &a % &b;
        a = b;
        b = r;
    }
    a
}

/// Run `sampler` on every path of its draw tree and return its exact output distribution.
pub(crate) fn exact<T: Ord>(mut sampler: impl FnMut(&mut Replay) -> T) -> Exact<T> {
    let mut leaves: Vec<(T, BigUint)> = Vec::new();
    let mut prefix: Vec<BigUint> = Vec::new();
    loop {
        let mut replay = Replay {
            path: prefix,
            ranges: Vec::new(),
        };
        let outcome = sampler(&mut replay);
        assert_eq!(
            replay.ranges.len(),
            replay.path.len(),
            "the sampler did not consume its whole prefix"
        );
        leaves.push((outcome, replay.ranges.iter().product()));
        let Replay { mut path, ranges } = replay;
        // The next path: increment the deepest draw that has room, dropping the rest.
        loop {
            let Some(last) = path.len().checked_sub(1) else {
                return finish(leaves);
            };
            path[last] += 1u32;
            if path[last] < ranges[last] {
                break;
            }
            path.pop();
        }
        prefix = path;
    }
}

fn finish<T: Ord>(leaves: Vec<(T, BigUint)>) -> Exact<T> {
    let distinct: BTreeSet<&BigUint> = leaves.iter().map(|(_, p)| p).collect();
    let mut lcm = BigUint::from(1u32);
    for p in distinct {
        lcm = &lcm / gcd(&lcm, p) * p;
    }
    let paths = leaves.len();
    let mut weights = BTreeMap::new();
    for (outcome, p) in leaves {
        *weights.entry(outcome).or_insert_with(BigUint::zero) += &lcm / &p;
    }
    let total: BigUint = weights.values().sum();
    assert_eq!(total, lcm, "the path probabilities do not sum to one");
    Exact {
        weights,
        denominator: lcm,
        paths,
    }
}
