//! Shared helpers of the integration tests: the test moduli, a seeded generator, and exact
//! big-integer references.
#![allow(dead_code)]

use num_bigint::BigInt;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// $2^{32} - 99$, the legacy Hachi modulus; $\equiv 5 \pmod 8$.
pub const Q32: u64 = 4294967197;
/// $2^{52} - 395$, the largest prime $\equiv 5 \pmod 8$ below $2^{52}$ (SageMath 10.9).
pub const Q52: u64 = 4503599627370101;
/// $2^{61} - 259$, the largest prime $\equiv 5 \pmod 8$ below $2^{61}$ (SageMath 10.9).
pub const Q61: u64 = 2305843009213693693;
/// $2^{63} - 259$, the largest prime $\equiv 5 \pmod 8$ below $2^{63}$ (SageMath 10.9): the top
/// of the supported range, beyond the exit criteria.
pub const Q63: u64 = 9223372036854775549;

/// The three moduli of the exit criteria.
pub const MODULI: [u64; 3] = [Q32, Q52, Q61];
/// The exit-criteria moduli and the top of the range.
pub const ALL_MODULI: [u64; 4] = [Q32, Q52, Q61, Q63];

/// The seeded generator of a test (ChaCha8 from `rand_chacha`; the seed is in each test).
pub fn rng(seed: u64) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed)
}

/// `n` uniform residues modulo `q`.
pub fn residues(rng: &mut ChaCha8Rng, q: u64, n: usize) -> Vec<u64> {
    (0..n).map(|_| rng.random_range(0..q)).collect()
}

/// The centered representative of a residue, as `i128`.
pub fn centered(x: u64, q: u64) -> i128 {
    let x = i128::from(x % q);
    if x > i128::from(q / 2) {
        x - i128::from(q)
    } else {
        x
    }
}

/// The residue modulo `q` of a big integer.
pub fn big_mod(x: &BigInt, q: u64) -> u64 {
    let r = ((x % BigInt::from(q)) + BigInt::from(q)) % BigInt::from(q);
    u64::try_from(r).unwrap()
}

/// The exact negacyclic product over $\mathbb Z[X]/(X^d + 1)$ in num-bigint, term by term: the
/// slow reference.
pub fn negacyclic_bigint(a: &[i128], b: &[i128]) -> Vec<BigInt> {
    let d = a.len();
    let mut out = vec![BigInt::from(0); d];
    for (i, x) in a.iter().enumerate() {
        let x = BigInt::from(*x);
        for (j, y) in b.iter().enumerate() {
            let term = &x * BigInt::from(*y);
            if i + j < d {
                out[i + j] += term;
            } else {
                out[i + j - d] -= term;
            }
        }
    }
    out
}

/// A 256-bit two's-complement accumulator `hi * 2^128 + lo` of `i128` terms: exact as long as
/// `hi` does not overflow, which needs more than $2^{127}$ terms of absolute value below
/// $2^{127}$.
#[derive(Clone, Copy, Default)]
pub struct Wide {
    hi: i128,
    lo: u128,
}
impl Wide {
    pub fn add(&mut self, v: i128) {
        let (sum, carry) = self.lo.overflowing_add(v as u128);
        self.hi += (v >> 127) + i128::from(carry);
        self.lo = sum;
    }
    pub fn to_bigint(self) -> BigInt {
        (BigInt::from(self.hi) << 128) + BigInt::from(self.lo)
    }
}

/// Exact sums $\sum_t a_t b_t$ of negacyclic products over $\mathbb Z[X]/(X^d + 1)$ of operands
/// below $2^{63}$ in absolute value, with [`Wide`] accumulators (each product fits `i128`); the
/// fast reference, checked against [`negacyclic_bigint`] in `rns.rs`.
pub fn negacyclic_sum_wide(a: &[Vec<i128>], b: &[Vec<i128>]) -> Vec<BigInt> {
    let d = a[0].len();
    let mut acc = vec![Wide::default(); d];
    for (x, y) in a.iter().zip(b) {
        add_negacyclic_product(&mut acc, x, y);
    }
    acc.into_iter().map(Wide::to_bigint).collect()
}

/// `acc += a * b` exactly, over $\mathbb Z[X]/(X^d + 1)$, for operands below $2^{63}$ in absolute
/// value.
pub fn add_negacyclic_product(acc: &mut [Wide], a: &[i128], b: &[i128]) {
    let d = acc.len();
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            let term = x * y;
            if i + j < d {
                acc[i + j].add(term);
            } else {
                acc[i + j - d].add(-term);
            }
        }
    }
}
