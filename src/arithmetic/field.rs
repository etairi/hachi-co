//! The prime field $\mathbb F_q$ and the tower $\mathbb F_{q^2} \subset \mathbb F_{q^4}$ over it,
//! as arkworks (`ark-ff`) types.
//!
//! The sumchecks of the paper's Section 4.3 run over $\mathbb F_{q^4}$, the degree-4 extension
//! chosen in Section 5.4 for approximately 128 bits of soundness (the paper's extension degree
//! $k = 4$, not to be confused with `params.k`, the challenge sparsity). This is the only place
//! where arkworks arithmetic is used: products of residues of $\mathbb Z_q$ are done by hand on
//! `u64` (see [`crate::arithmetic::ring`] and [`crate::arithmetic::utils`]), and integers enter
//! the field through `Fq::from`, which reduces a `u64` residue, or a signed `i64`, modulo $q$
//! (paper, Section 5.4, "Field Arithmetic").
//!
//! ## The tower as the code builds it
//!
//! * $\mathbb F_q = \mathbb Z/q\mathbb Z$ with $q = 4294967197 = 2^{32} - 99$, a prime with
//!   $q \equiv 5 \pmod 8$, in Montgomery form on one 64-bit limb ($R = 2^{64}$).
//! * $\mathbb F_{q^2} = \mathbb F_q\[u\]/(u^2 - \nu)$. The source writes the non-residue as
//!   `Fq::new_unchecked(BigInt!("5"))`, but `new_unchecked` takes the *Montgomery representation*
//!   of an element, so the constant is the field element
//!   $\nu = 5 \cdot 2^{-64} \bmod q = 3546053929$ rather than $5$ (the test vectors of this
//!   module confirm $u^2 = \nu$). It is still a quadratic non-residue, as $2^{-64}$ is a square
//!   and $5$ is a non-residue modulo this $q$ ($q \equiv 2 \pmod 5$, so by quadratic reciprocity
//!   $(5/q) = (q/5) = (2/5) = -1$; checked directly, $5^{(q-1)/2} \equiv -1 \pmod q$), so
//!   $\mathbb F_{q^2}$ is a field. The protocol does not depend on which non-residue is used (all
//!   choices give isomorphic fields), but transcripts and test vectors do. That $5$ is a
//!   non-residue is a property of this particular modulus, not of every $q \equiv 5 \pmod 8$.
//! * $\mathbb F_{q^4} = \mathbb F_{q^2}\[v\]/(v^2 - u)$. arkworks' `Fp4Config` hard-codes the
//!   multiplication by the non-residue as multiplication by $u = (0, 1)$, whatever the declared
//!   `NONRESIDUE` constant is, so $v^2 = u$ holds (again confirmed by the test vectors). $u$ is a
//!   non-square in $\mathbb F_{q^2}$ because its norm to $\mathbb F_q$, $u \cdot \bar u = -u^2 = -\nu$,
//!   is a non-square there ($-1$ is a square as $q \equiv 1 \pmod 4$, and $\nu$ is not).
//!
//! An element of $\mathbb F_{q^4}$ is $(a_0 + a_1 u) + (a_2 + a_3 u) v$ with
//! $a_0, \dots, a_3 \in \mathbb F_q$, built as `Fq4::new(Fq2::new(a0, a1), Fq2::new(a2, a3))`;
//! the fields `c0`, `c1` hold the two $\mathbb F_{q^2}$ coordinates and `c0.c0`, `c0.c1`,
//! `c1.c0`, `c1.c1` the four $\mathbb F_q$ coordinates, in the order used by the transcript
//! encoding of [`crate::arithmetic::sumcheck`].
//!
//! ## Constants the scheme never exercises
//!
//! The scheme uses only addition, subtraction, multiplication, inversion (Lagrange interpolation
//! in the sumcheck), exponentiation, sampling and conversion from integers. The remaining
//! constants are not consistent with the tower, but no code path reads them:
//!
//! * `FqConfig` declares `generator = 3`, but $3$ is a quadratic residue modulo $q$ with
//!   multiplicative order $(q - 1)/4$, not a generator of $\mathbb F_q^\times$ (the `MontConfig`
//!   derive does not check this). arkworks uses the generator only to derive the two-adic root
//!   of unity for square roots and FFT domains, neither of which is used here.
//! * The Frobenius tables are written with `new_unchecked` too, so they do not hold $\pm 1$, and
//!   `Fq4Config::FROBENIUS_COEFF_FP4_C1` has two entries where arkworks indexes by the Frobenius
//!   power modulo 4 and expects the four values $\nu^{(q^i - 1)/4}$, $i = 0, 1, 2, 3$. A call to
//!   `frobenius_map` on $\mathbb F_{q^4}$ would therefore be wrong or panic; the scheme never
//!   makes one.

use ark_ff::fields::{Fp, Fp2, Fp2Config, Fp4, Fp4Config, MontBackend, MontConfig};
use ark_ff::BigInt;

// ---- The base prime field Zq ---- //
/// arkworks configuration of the prime field $\mathbb F_q$, $q = 4294967197$: Montgomery
/// arithmetic on a single 64-bit limb. See the module documentation for the `generator` caveat.
#[derive(MontConfig)]
#[modulus = "4294967197"]
#[generator = "3"]
pub struct FqConfig;

/// The prime field $\mathbb F_q = \mathbb Z_q$, $q = 2^{32} - 99$. `Fq::from(x)` reduces a `u64`
/// residue, or a signed `i64` short integer, modulo $q$; this is how integers enter the field
/// ([`crate::arithmetic::utils::lift_int`], [`crate::arithmetic::utils::mul_int_field`]).
pub type Fq = Fp<MontBackend<FqConfig, 1>, 1>;

// ---- Degree 2 field extension. ---- //
/// arkworks configuration of $\mathbb F_{q^2} = \mathbb F_q\[u\]/(u^2 - \nu)$ with
/// $\nu = 5 \cdot 2^{-64} \bmod q$ (see the module documentation).
pub struct Fq2Config;

impl Fp2Config for Fq2Config {
    type Fp = Fq;

    /// $u^2 = \nu$: the element whose Montgomery representation is $5$, i.e.
    /// $\nu = 5 \cdot 2^{-64} \bmod q = 3546053929$, a quadratic non-residue.
    const NONRESIDUE: Self::Fp = Fq::new_unchecked(BigInt!("5"));
    /// Frobenius coefficients for $u^{q^i}$, $i = 0, 1$; written with `new_unchecked`, so not
    /// the intended $(1, -1)$. Unused by the scheme (see the module documentation).
    const FROBENIUS_COEFF_FP2_C1: &'static [Self::Fp] = &[
        Fq::new_unchecked(BigInt!("1")),
        Fq::new_unchecked(BigInt!("-1"))
    ];
}

/// The quadratic extension $\mathbb F_{q^2}$; an element $a_0 + a_1 u$ is `Fq2::new(a0, a1)`.
pub type Fq2 = Fp2<Fq2Config>;

// ---- Degree 4 field extension. ---- //
/// arkworks configuration of $\mathbb F_{q^4} = \mathbb F_{q^2}\[v\]/(v^2 - u)$; the
/// multiplication by the non-residue $u$ is hard-coded by arkworks (see the module documentation).
pub struct Fq4Config;

impl Fp4Config for Fq4Config {
    type Fp2Config = Fq2Config;

    /// Declared non-residue; arkworks requires $(0, 1) = u$ and multiplies by $u$ regardless of
    /// this constant. Written with `new_unchecked`, the constant itself is $(0, 2^{-64} \bmod q)$.
    const NONRESIDUE: Fp2<Self::Fp2Config> = Fq2::new(
        Fp::new_unchecked(BigInt!("0")), 
        Fp::new_unchecked(BigInt!("1"))
    );

    /// Frobenius table with two entries where arkworks expects four ($\nu^{(q^i - 1)/4}$ for
    /// $i = 0, \dots, 3$), written with `new_unchecked`; unused by the scheme (see the module
    /// documentation).
    const FROBENIUS_COEFF_FP4_C1: &'static [<Self::Fp2Config as Fp2Config>::Fp] = &[
        Fq::new_unchecked(BigInt!("1")),
        Fq::new_unchecked(BigInt!("-1"))
    ];
}

/// The sumcheck field $\mathbb F_{q^4}$ ([`crate::arithmetic::ExtField`]); an element
/// $c_0 + c_1 v$ with $c_0, c_1 \in \mathbb F_{q^2}$ is `Fq4::new(c0, c1)`.
pub type Fq4 = Fp4<Fq4Config>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fq() {
        let a = Fq::from(10);
        let b = Fq::from(30);

        assert_eq!(Fq::from(40), a+b);
        assert_eq!(Fq::from(300), a*b);
    }

    #[test]
    fn test_fq2() {
        let a = Fq2::new(Fq::from(10), Fq::from(50));
        let b = Fq2::new(Fq::from(30), Fq::from(-8));

        assert_eq!(Fq2::new(Fq::from(40), Fq::from(42)), a+b);
        assert_eq!(Fq2::new(Fq::from(3212570907i64), Fq::from(1420)), a*b);
    }

    #[test]
    fn test_fq4() {
        let a = Fq2::new(Fq::from(10), Fq::from(50));
        let b = Fq2::new(Fq::from(30), Fq::from(-8));

        let c = Fq4::new(a, b);
        let d = Fq4::new(b, a);

        let sum = Fq4::new(
            Fq2::new(Fq::from(40), Fq::from(42)), Fq2::new(Fq::from(40), Fq::from(42)));

        let prod = Fq4::new(
            Fq2::new(Fq::from(612628006), Fq::from(3212572327i64)), Fq2::new(Fq::from(3931686104i64), Fq::from(520)));
        
        assert_eq!(sum, c+d);
        assert_eq!(prod, c*d);
    }
}