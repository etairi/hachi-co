//! Matrices of polynomials in multi-prime NTT form: the public matrices $\mathbf A$, $\mathbf B$,
//! $\mathbf D$ after [`crate::arithmetic::ring::Ring::mat_fwd_ntt`], kept in this form so that
//! every matrix-vector product ([`crate::arithmetic::ring::Ring::mat_mul_vec`]) transforms only
//! the vector (paper, Section 5.4). A [`PMatNtt`] is a [`PVecNtt`] of `height * width` elements
//! in row-major order: entry `(row, col)` is element `row * width + col`, and each entry is one
//! residue vector per NTT prime of the transform length $N$ ($d$ for a ring in cyclotomic mode,
//! $2d$ in full mode; see [`crate::arithmetic::poly_vec_ntt`]).

use crate::arithmetic::{NttType, poly_vec_ntt::PVecNtt};

/// A `height` by `width` matrix of polynomials in multi-prime NTT form, stored row-major in a
/// [`PVecNtt`] (see the module documentation).
pub struct PMatNtt {
    /// Number of rows.
    height: usize,
    /// Number of columns.
    width: usize,
    /// The `height * width` entries, row after row.
    vec: PVecNtt
}

impl PMatNtt {
    /// The zero matrix of `height * width` entries of transform length `d` (a power of two);
    /// filled by [`crate::arithmetic::ring::Ring::mat_fwd_ntt`].
    pub fn zero(height: usize, width: usize, d: usize) -> Self {
        let vec = PVecNtt::zero(height * width, d);
        Self { height, width, vec }
    }

    /// Number of rows.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Number of columns.
    pub fn width(&self) -> usize {
        self.width
    }

    #[cfg(not(feature = "nightly"))]
    /// Entry `(row, col)` as five slices of length `d`, one per prime (`Plan32` path).
    pub fn element(&self, row: usize, col: usize) -> (&[NttType], &[NttType], &[NttType], &[NttType], &[NttType]) {
        self.vec.element(row * self.width + col)
    }

    #[cfg(not(feature = "nightly"))]
    /// Entry `(row, col)` as five mutable slices of length `d`, one per prime (`Plan32` path).
    pub fn mut_element(&mut self, row: usize, col: usize) -> (&mut [NttType], &mut [NttType], &mut [NttType], &mut [NttType], &mut [NttType]) {
        self.vec.mut_element(row * self.width + col)
    }

    #[cfg(feature = "nightly")]
    /// Entry `(row, col)` as three slices of length `d`, one per prime (`Plan52` path).
    pub fn element(&self, row: usize, col: usize) -> (&[NttType], &[NttType], &[NttType]) {
        self.vec.element(row * self.width + col)
    }

    #[cfg(feature = "nightly")]
    /// Entry `(row, col)` as three mutable slices of length `d`, one per prime (`Plan52` path).
    pub fn mut_element(&mut self, row: usize, col: usize) -> (&mut [NttType], &mut [NttType], &mut [NttType]) {
        self.vec.mut_element(row * self.width + col)
    }
}

/// Deep copy.
impl Clone for PMatNtt {
    fn clone(&self) -> Self {
        Self { height: self.height, width: self.width, vec: self.vec.clone() }
    }
}
