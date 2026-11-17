//! The challenges $c_i$ of the amortized opening, distributed as the reference's
//! (`polyvec_challenge`, `challenge` in `poly.c`).
//!
//! **Distribution.** Uniform signs and a uniform arrangement of $\tau_1 = 32$ coefficients
//! $\pm 1$, $\tau_2 = 8$ coefficients $\pm 2$ and 24 zeros (the reference's shuffle: for $k = 24,
//! \dots, 63$ a byte $b \bmod 64$ is drawn until $b \le k$, $c_k \gets c_b$ and $c_b \gets \pm 1$
//! for $k < 56$, $\pm 2$ otherwise, the sign the next bit of a 40-bit word), kept if the operator
//! norm $\max_\zeta |c(\zeta)|$ over the primitive 128th roots of unity is at most $T = 14$.
//! Each challenge $i$ has its own SHAKE128 stream ([`ShakePrg`] with the seed squeezed from the
//! transcript, [`CHALLENGE_DOMAIN`] and the index $i$), where the reference draws them all from
//! one stream: the distribution is the same.
//!
//! **Operator norm.** $|c(\zeta)|^2$ for $\zeta = e^{i\pi(2k+1)/64}$, $k < 32$ (the other 32 roots
//! are their conjugates), is the sum over the nonzero coefficients of $c_j \zeta^j$, with the
//! cosines and sines of multiples of $\pi/64$ from a table of correctly rounded constants, in a
//! fixed order and with no fused operations, so that every platform computes the same bits and
//! keeps the same challenges. **Deviation:** the reference's `poly_fft` reads $c_j + c_j i$ where
//! it should read $c_j + c_{j+32} i$, so its check bounds $\sqrt 2 \max |c_{\mathrm{lo}}(\zeta)|$
//! over the roots of $X^{32} = i$ ($c_{\mathrm{lo}}$ the lower half of $c$) instead of the
//! operator norm; this module bounds the operator norm itself (the tests compare the two). Of
//! 20000 challenges of the reference's `polyvec_challenge`, 16637 (83.2%) had an operator norm
//! above 14, the largest 25.38, and its `poly_opnorm` agreed with $\sqrt 2 \max
//! |c_{\mathrm{lo}}(\zeta)|$ to $10^{-13}$ (measured on an AMD EPYC 9B14, 17 November 2026). Here
//! a draw passes with probability about 0.136, 7.35 draws per challenge on average.
use crate::params::{N, T, TAU1, TAU2};
use hachi_co_sampling::ShakePrg;
use rand::Rng;

/// The domain of the challenges' streams.
pub const CHALLENGE_DOMAIN: &[u8] = b"hachi-co-greyhound/v1/challenge";

/// $\cos(\pi t / 64)$ for $t = 0, \dots, 32$, correctly rounded to `f64` (from 60-digit decimal
/// arithmetic: $\pi$ by Machin's formula, the cosine by its series).
const COS: [f64; 33] = [
    1.0,
    0.9987954562051724,
    0.9951847266721969,
    0.989176509964781,
    0.9807852804032304,
    0.970031253194544,
    0.9569403357322088,
    0.9415440651830208,
    0.9238795325112867,
    0.9039892931234433,
    0.881921264348355,
    0.8577286100002721,
    0.8314696123025452,
    0.8032075314806449,
    0.773010453362737,
    0.7409511253549591,
    std::f64::consts::FRAC_1_SQRT_2,
    0.6715589548470184,
    0.6343932841636455,
    0.5956993044924334,
    0.5555702330196022,
    0.5141027441932218,
    0.47139673682599764,
    0.4275550934302821,
    0.3826834323650898,
    0.33688985339222005,
    0.2902846772544624,
    0.2429801799032639,
    0.19509032201612828,
    0.14673047445536175,
    0.0980171403295606,
    0.049067674327418015,
    0.0,
];

/// $\cos(\pi t / 64)$ for any $t$, by the symmetries of the cosine (exact sign changes).
fn cos128(t: usize) -> f64 {
    match t % 128 {
        t @ 0..=32 => COS[t],
        t @ 33..=64 => -COS[64 - t],
        t @ 65..=96 => -COS[t - 64],
        t => COS[128 - t],
    }
}

/// $\sin(\pi t / 64) = \cos(\pi (t - 32) / 64)$.
fn sin128(t: usize) -> f64 {
    cos128((t + 96) % 128)
}

/// $\max_\zeta |c(\zeta)|^2$ over the primitive 128th roots of unity: the square of the operator
/// norm of multiplication by $c$ in $\mathbb R\[X\]/(X^{64} + 1)$.
pub fn opnorm_sq(c: &[i8; N]) -> f64 {
    let terms: Vec<(usize, f64)> = c
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != 0)
        .map(|(j, v)| (j, f64::from(*v)))
        .collect();
    let mut best = 0f64;
    for k in 0..N / 2 {
        let (mut re, mut im) = (0f64, 0f64);
        for &(j, v) in &terms {
            let t = ((2 * k + 1) * j) % 128;
            re += v * cos128(t);
            im += v * sin128(t);
        }
        best = best.max(re * re + im * im);
    }
    best
}

/// A challenge: 64 coefficients, 32 of them $\pm 1$, 8 of them $\pm 2$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Challenge {
    /// The coefficients, index $j$ for $X^j$.
    pub coeffs: [i8; N],
}

/// The bytes of a stream, read in blocks of the SHAKE128 rate.
struct Bytes {
    prg: ShakePrg,
    buf: [u8; 168],
    pos: usize,
}

impl Bytes {
    fn next(&mut self) -> u8 {
        if self.pos == self.buf.len() {
            self.prg.fill_bytes(&mut self.buf);
            self.pos = 0;
        }
        self.pos += 1;
        self.buf[self.pos - 1]
    }
}

impl Challenge {
    /// Challenge `index` of the seed `seed`, with the number of draws it took.
    pub fn sample_counted(seed: &[u8; 32], index: u64) -> (Self, u32) {
        let mut bytes = Bytes {
            prg: ShakePrg::new(seed, CHALLENGE_DOMAIN, index),
            buf: [0; 168],
            pos: 168,
        };
        let mut draws = 0;
        loop {
            draws += 1;
            let c = Self::draw(&mut bytes);
            if opnorm_sq(&c.coeffs) <= T * T {
                return (c, draws);
            }
        }
    }

    /// Challenge `index` of the seed `seed`.
    pub fn sample(seed: &[u8; 32], index: u64) -> Self {
        Self::sample_counted(seed, index).0
    }

    /// One draw of the reference's shuffle, before the operator-norm check.
    fn draw(bytes: &mut Bytes) -> Self {
        let sign_bytes = (TAU1 + TAU2).div_ceil(8);
        let mut signs = 0u64;
        for i in 0..sign_bytes {
            signs |= u64::from(bytes.next()) << (8 * i);
        }
        let mut c = [0i8; N];
        let mut k = N - TAU1 - TAU2;
        while k < N {
            let b = usize::from(bytes.next()) & (N - 1);
            if b <= k {
                c[k] = c[b];
                let v = if k < N - TAU2 { 1 } else { 2 };
                c[b] = if signs & 1 == 1 { -v } else { v };
                signs >>= 1;
                k += 1;
            }
        }
        Self { coeffs: c }
    }

    /// The coefficients as `i64`.
    pub fn to_i64(&self) -> [i64; N] {
        self.coeffs.map(i64::from)
    }

    /// $\Vert c \Vert_1$.
    pub fn l1_norm(&self) -> u64 {
        self.coeffs
            .iter()
            .map(|x| u64::from(x.unsigned_abs()))
            .sum()
    }
}

/// The challenges $c_0, \dots, c_{n-1}$ of a squeezed seed, in parallel.
pub fn challenges(seed: &[u8; 32], n: usize) -> Vec<Challenge> {
    use rayon::prelude::*;
    (0..n as u64)
        .into_par_iter()
        .map(|i| Challenge::sample(seed, i))
        .collect()
}

#[cfg(test)]
mod tests;
