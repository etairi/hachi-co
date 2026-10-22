//! Helpers shared by the integration tests: scripted and counting RNGs, the chi-square
//! statistic with its thresholds, and hex decoding.
#![allow(dead_code)]

use core::convert::Infallible;
use rand::{Rng, TryCryptoRng, TryRng};

/// An RNG that returns a fixed sequence of 64-bit words and panics when it runs out; only
/// `next_u64` is supported, which is all the samplers use.
pub struct Scripted {
    words: Vec<u64>,
    pub read: usize,
}

impl Scripted {
    pub fn new(words: &[u64]) -> Self {
        Self {
            words: words.to_vec(),
            read: 0,
        }
    }
}

impl TryRng for Scripted {
    type Error = Infallible;
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        panic!("the samplers read whole 64-bit words")
    }
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        let w = *self.words.get(self.read).expect("script exhausted");
        self.read += 1;
        Ok(w)
    }
    fn try_fill_bytes(&mut self, _: &mut [u8]) -> Result<(), Infallible> {
        panic!("the samplers read whole 64-bit words")
    }
}

/// A constant RNG, to drive rejection loops into their cap.
pub struct Constant(pub u64, pub usize);

impl TryRng for Constant {
    type Error = Infallible;
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        panic!("the samplers read whole 64-bit words")
    }
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        self.1 += 1;
        Ok(self.0)
    }
    fn try_fill_bytes(&mut self, _: &mut [u8]) -> Result<(), Infallible> {
        panic!("the samplers read whole 64-bit words")
    }
}

/// Counts the 64-bit words drawn from the inner RNG; any other kind of read panics.
pub struct Counting<R> {
    pub inner: R,
    pub words: usize,
}

impl<R: Rng> Counting<R> {
    pub fn new(inner: R) -> Self {
        Self { inner, words: 0 }
    }
}

impl<R: Rng> TryRng for Counting<R> {
    type Error = Infallible;
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        panic!("the samplers read whole 64-bit words")
    }
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        self.words += 1;
        Ok(self.inner.next_u64())
    }
    fn try_fill_bytes(&mut self, _: &mut [u8]) -> Result<(), Infallible> {
        panic!("the samplers read whole 64-bit words")
    }
}

// The inner RNGs used here (ShakePrg, ChaCha20Rng) are cryptographic, so the wrapper may be
// passed where the API demands a CryptoRng.
impl<R: Rng + TryCryptoRng> TryCryptoRng for Counting<R> {}

/// Pearson's statistic $\sum_k (O_k - E_k)^2 / E_k$.
pub fn chi_square(observed: &[u64], expected: &[f64]) -> f64 {
    assert_eq!(observed.len(), expected.len());
    observed
        .iter()
        .zip(expected)
        .map(|(o, e)| (*o as f64 - e).powi(2) / e)
        .sum()
}

/// The $1 - 10^{-4}$ quantile of the chi-square distribution with `df` degrees of freedom:
/// a correct sampler exceeds it with probability $10^{-4}$ per test. Values from
/// `sage -python -c "from scipy.stats import chi2; print(chi2.ppf(1 - 1e-4, df))"`
/// (scipy 1.17.1), rounded to 4 decimals.
pub fn chi_square_threshold(df: usize) -> f64 {
    match df {
        1 => 15.1367,
        2 => 18.4207,
        3 => 21.1075,
        4 => 23.5127,
        5 => 25.7448,
        6 => 27.8563,
        8 => 31.8276,
        9 => 33.7199,
        10 => 35.564,
        12 => 39.1344,
        14 => 42.5793,
        15 => 44.2632,
        16 => 45.9249,
        24 => 58.613,
        26 => 61.6573,
        39 => 80.6462,
        40 => 82.0623,
        50 => 95.9687,
        60 => 109.5029,
        62 => 112.1737,
        98 => 158.7915,
        128 => 196.2112,
        144 => 215.8132,
        159 => 234.0128,
        _ => panic!("no chi-square threshold tabulated for df = {df}"),
    }
}

/// Assert that the counts pass the chi-square test against `expected`, printing the statistic.
pub fn assert_chi_square(what: &str, observed: &[u64], expected: &[f64]) {
    let df = observed.len() - 1;
    let stat = chi_square(observed, expected);
    let threshold = chi_square_threshold(df);
    println!("chi2 {what}: {stat:.2} (df {df}, threshold {threshold})");
    assert!(
        stat < threshold,
        "{what}: chi2 = {stat:.2} >= {threshold} (df {df})"
    );
}

pub fn hex(s: &str) -> Vec<u8> {
    assert_eq!(s.len() % 2, 0, "odd hex length");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex digit"))
        .collect()
}

pub fn seed(s: &str) -> [u8; 32] {
    hex(s).try_into().expect("32-byte seed")
}
