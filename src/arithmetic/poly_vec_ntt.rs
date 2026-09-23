//! Vectors of polynomials in multi-prime NTT form: the operand and accumulator format of the
//! NTT products of [`crate::arithmetic::ring::Ring`] (paper, Section 5.4, "Polynomial
//! Arithmetic").
//!
//! ## Layout
//!
//! The native NTT of tfhe-ntt works modulo several primes $p_i$ at once (see the documentation
//! of [`crate::arithmetic::ring`]), so a polynomial in NTT form is one residue vector per prime.
//! A [`PVecNtt`] keeps one flat `Vec<NttType>` per prime: `vec0..vec4` of `u32` on the default
//! `Plan32` path (five primes) and `vec0..vec2` of `u64` with the `nightly` feature (`Plan52`,
//! three primes), each of length `len * d`. Within each of them the layout is that of
//! [`crate::arithmetic::poly_vec::PVec`]: NTT coefficient $j$ of element $i$ is at index
//! `i * d + j`. Here `d` is the *transform length* $N$ of the ring that produced the vector, the
//! ring dimension $d$ in cyclotomic mode and $2d$ in full mode, so it is not necessarily the
//! ring dimension. The coefficients are in tfhe-ntt's internal (bit-reversed) order and are only
//! ever consumed by the plan that produced them (pointwise products and the inverse transform),
//! so their order is opaque to this type. The `element` accessors return one slice per prime.

use crate::arithmetic::{NttType, utils::Logarithm};

/// A vector of polynomials in multi-prime NTT form, one flat residue vector per prime (see the
/// module documentation for the layout).
pub struct PVecNtt {
    /// Number of elements.
    len: usize,
    /// Transform length $N$ of each element, a power of two.
    d: usize,
    /// $\log_2$ of `d`, used to slice elements by shifting.
    logd: usize,
    /// Residues modulo the first prime, `len * d` entries.
    vec0: Vec<NttType>,
    /// Residues modulo the second prime.
    vec1: Vec<NttType>,
    /// Residues modulo the third prime.
    vec2: Vec<NttType>,
    /// Residues modulo the fourth prime (`Plan32` path only).
    #[cfg(not(feature = "nightly"))]
    vec3: Vec<NttType>,
    /// Residues modulo the fifth prime (`Plan32` path only).
    #[cfg(not(feature = "nightly"))]
    vec4: Vec<NttType>,
}

impl PVecNtt {
    /// The zero vector of `len` elements of transform length `d` (a power of two, asserted).
    /// Zero is also the zero of the NTT domain, so a fresh vector is a valid accumulator for
    /// pointwise multiply-accumulate.
    pub fn zero(len: usize, d: usize) -> Self {
        assert!(d.is_power_of_two());
        let logd = d.log();

        let vec0 = vec![0; len * d];
        let vec1 = vec![0; len * d];
        let vec2 = vec![0; len * d];
        #[cfg(not(feature = "nightly"))]
        let vec3 = vec![0; len * d];
        #[cfg(not(feature = "nightly"))]
        let vec4 = vec![0; len * d];

        #[cfg(not(feature = "nightly"))]
        return Self { len, d, logd, vec0, vec1, vec2, vec3, vec4 };

        #[cfg(feature = "nightly")]
        return Self { len, d, logd, vec0, vec1, vec2 }
    }

    /// Number of elements.
    pub fn length(&self) -> usize {
        self.len
    }

    #[cfg(not(feature = "nightly"))]
    /// Element $i$ as five slices of length `d`, one per prime in the order `vec0..vec4`
    /// (`Plan32` path); panics if `i >= len`.
    pub fn element(&self, i: usize) -> (&[NttType], &[NttType], &[NttType], &[NttType], &[NttType]) {
        let s0 = &self.vec0[(i << self.logd)..((i + 1) << self.logd)];
        let s1 = &self.vec1[(i << self.logd)..((i + 1) << self.logd)];
        let s2 = &self.vec2[(i << self.logd)..((i + 1) << self.logd)];
        let s3 = &self.vec3[(i << self.logd)..((i + 1) << self.logd)];
        let s4 = &self.vec4[(i << self.logd)..((i + 1) << self.logd)];

        (s0, s1, s2, s3, s4)
    }

    #[cfg(not(feature = "nightly"))]
    /// Element $i$ as five mutable slices of length `d`, one per prime in the order `vec0..vec4`
    /// (`Plan32` path); panics if `i >= len`.
    pub fn mut_element(&mut self, i: usize) -> (&mut [NttType], &mut [NttType], &mut [NttType], &mut [NttType], &mut [NttType]) {
        let s0 = &mut self.vec0[(i << self.logd)..((i + 1) << self.logd)];
        let s1 = &mut self.vec1[(i << self.logd)..((i + 1) << self.logd)];
        let s2 = &mut self.vec2[(i << self.logd)..((i + 1) << self.logd)];
        let s3 = &mut self.vec3[(i << self.logd)..((i + 1) << self.logd)];
        let s4 = &mut self.vec4[(i << self.logd)..((i + 1) << self.logd)];

        (s0, s1, s2, s3, s4)
    }

    #[cfg(feature = "nightly")]
    /// Element $i$ as three slices of length `d`, one per prime in the order `vec0..vec2`
    /// (`Plan52` path); panics if `i >= len`.
    pub fn element(&self, i: usize) -> (&[NttType], &[NttType], &[NttType]) {
        let s0 = &self.vec0[(i << self.logd)..((i + 1) << self.logd)];
        let s1 = &self.vec1[(i << self.logd)..((i + 1) << self.logd)];
        let s2 = &self.vec2[(i << self.logd)..((i + 1) << self.logd)];

        (s0, s1, s2)
    }

    #[cfg(feature = "nightly")]
    /// Element $i$ as three mutable slices of length `d`, one per prime in the order `vec0..vec2`
    /// (`Plan52` path); panics if `i >= len`.
    pub fn mut_element(&mut self, i: usize) -> (&mut [NttType], &mut [NttType], &mut [NttType]) {
        let s0 = &mut self.vec0[(i << self.logd)..((i + 1) << self.logd)];
        let s1 = &mut self.vec1[(i << self.logd)..((i + 1) << self.logd)];
        let s2 = &mut self.vec2[(i << self.logd)..((i + 1) << self.logd)];

        (s0, s1, s2)
    }
}


/// Deep copy (`Plan32` layout, five residue vectors).
#[cfg(not(feature = "nightly"))]
impl Clone for PVecNtt {
    fn clone(&self) -> Self {
        Self { len: self.len, d: self.d, logd: self.logd, 
            vec0: self.vec0.clone(), vec1: self.vec1.clone(), vec2: self.vec2.clone(), vec3: self.vec3.clone(), vec4: self.vec4.clone() }
    }
}

/// Deep copy (`Plan52` layout, three residue vectors).
#[cfg(feature = "nightly")]
impl Clone for PVecNtt {
    fn clone(&self) -> Self {
        Self { len: self.len, d: self.d, logd: self.logd, 
            vec0: self.vec0.clone(), vec1: self.vec1.clone(), vec2: self.vec2.clone() }
    }
}
