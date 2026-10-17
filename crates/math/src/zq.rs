//! $\mathbb Z_q$ for a runtime prime $q < 2^{63}$.
//!
//! Residues are `u64` values in $[0, q)$; short signed values are `i64`, and the centered
//! representative of $x$ is the one in $[-(q-1)/2, (q-1)/2]$ (draft `4_preliminaries.tex`:86,
//! $w \bmod^{\pm} q$). Every method reads a `u64` argument modulo $q$, so an unreduced argument
//! stands for its residue class, and every method returns a reduced residue.
//!
//! Products are computed exactly as `u128` and reduced by the `u128` remainder; this is the
//! reference arithmetic of the crate, which the faster paths elsewhere (Shoup products in the
//! transforms and the CRT) are tested against.

use crate::{
    arith::{self, mul_mod, reduce},
    error::{Error, Result},
};

/// Deterministic primality test for every `u64`: trial division by the primes up to 37, then
/// the strong-probable-prime test to the seven bases of Jim Sinclair's set, which no composite
/// below $2^{64}$ passes. Ported from jali (`src/math/int.rs`).
pub fn is_prime(n: u64) -> bool {
    arith::is_prime(n)
}

/// The ring $\mathbb Z_q$ for an odd prime $q < 2^{63}$ (`docs/m0-spec.md`, "Conventions": a
/// runtime prime with 64-bit residues; the draft's modulus, `4_preliminaries.tex`:61).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Zq {
    q: u64,
    /// $(q-1)/2$: residues above it are negative when centered.
    half: u64,
}

impl Zq {
    /// $\mathbb Z_q$ for an odd prime $3 \le q < 2^{63}$ (the modulus of $\mathcal R_q$, draft
    /// `4_preliminaries.tex`:61); primality is checked by the deterministic Miller–Rabin test
    /// [`is_prime`].
    ///
    /// # Errors
    /// [`Error::Modulus`] if $q$ is even, not prime, or at least $2^{63}$.
    pub fn new(q: u64) -> Result<Self> {
        if q < 3 || q.is_multiple_of(2) || q >= 1 << 63 || !is_prime(q) {
            return Err(Error::Modulus(q));
        }
        Ok(Self { q, half: q / 2 })
    }

    /// The modulus $q$ (draft `4_preliminaries.tex`:61).
    pub fn modulus(&self) -> u64 {
        self.q
    }

    /// $(q-1)/2$, the largest centered representative (draft `4_preliminaries.tex`:86).
    pub fn half(&self) -> u64 {
        self.half
    }

    /// $x \bmod^+ q \in [0, q)$, the reduced residue of any `u64` (draft `4_preliminaries.tex`:86).
    #[inline]
    pub fn reduce(&self, x: u64) -> u64 {
        reduce(x, self.q)
    }

    /// $(a + b) \bmod q$, addition in $\mathbb Z_q$ (draft `4_preliminaries.tex`:61).
    #[inline]
    pub fn add(&self, a: u64, b: u64) -> u64 {
        arith::add_mod(self.reduce(a), self.reduce(b), self.q)
    }

    /// $(a - b) \bmod q$, subtraction in $\mathbb Z_q$ (draft `4_preliminaries.tex`:61).
    #[inline]
    pub fn sub(&self, a: u64, b: u64) -> u64 {
        arith::sub_mod(self.reduce(a), self.reduce(b), self.q)
    }

    /// $-a \bmod q$, negation in $\mathbb Z_q$ (draft `4_preliminaries.tex`:61).
    #[inline]
    pub fn neg(&self, a: u64) -> u64 {
        let a = self.reduce(a);
        if a == 0 { 0 } else { self.q - a }
    }

    /// $ab \bmod q$, multiplication in $\mathbb Z_q$ (draft `4_preliminaries.tex`:61), from the
    /// exact `u128` product (valid for any `u64` operands).
    #[inline]
    pub fn mul(&self, a: u64, b: u64) -> u64 {
        mul_mod(a, b, self.q)
    }

    /// $a^e \bmod q$ in $\mathbb Z_q$ (draft `4_preliminaries.tex`:61) by square and multiply, with
    /// $a^0 = 1$ (also for $a = 0$).
    pub fn pow(&self, a: u64, e: u64) -> u64 {
        arith::pow_mod(a, e, self.q)
    }

    /// $a^{-1} \bmod q$ in the field $\mathbb Z_q$ (draft `4_preliminaries.tex`:61: $q$ is
    /// prime), or `None` for $a \equiv 0$. Computed by the extended Euclidean algorithm on
    /// $(q, a)$.
    pub fn inv(&self, a: u64) -> Option<u64> {
        let a = self.reduce(a);
        if a == 0 {
            return None;
        }
        // Invariant: r_i = t_i * a (mod q); the coefficients satisfy |t_i| <= q.
        let (mut r0, mut r1) = (i128::from(self.q), i128::from(a));
        let (mut t0, mut t1) = (0i128, 1i128);
        while r1 != 0 {
            let quotient = r0 / r1;
            (r0, r1) = (r1, r0 - quotient * r1);
            (t0, t1) = (t1, t0 - quotient * t1);
        }
        debug_assert_eq!(r0, 1, "q is prime and a is nonzero");
        Some(t0.rem_euclid(i128::from(self.q)) as u64)
    }

    /// $x \bmod^+ q$ for a `u128` (draft `4_preliminaries.tex`:86).
    #[inline]
    pub fn reduce_u128(&self, x: u128) -> u64 {
        (x % u128::from(self.q)) as u64
    }

    /// $x \bmod^+ q$ for a signed integer (draft `4_preliminaries.tex`:86).
    #[inline]
    pub fn from_i64(&self, x: i64) -> u64 {
        i128::from(x).rem_euclid(i128::from(self.q)) as u64
    }

    /// $x \bmod^+ q$ for a signed 128-bit integer (draft `4_preliminaries.tex`:86).
    #[inline]
    pub fn from_i128(&self, x: i128) -> u64 {
        x.rem_euclid(i128::from(self.q)) as u64
    }

    /// The centered representative $x \bmod^{\pm} q \in [-(q-1)/2, (q-1)/2]$ (draft
    /// `4_preliminaries.tex`:86).
    #[inline]
    pub fn to_centered(&self, x: u64) -> i64 {
        let x = self.reduce(x);
        if x > self.half {
            x as i64 - self.q as i64
        } else {
            x as i64
        }
    }
}
