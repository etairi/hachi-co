//! Matrices of polynomials over $\mathbb Z_q$ in coefficient form: the public commitment
//! matrices $\mathbf A$, $\mathbf B$ and $\mathbf D$ of the paper (Section 4.1-4.2, Eq. (13),
//! (14) and (16)), which the prototype expands from seeds. A [`PMat`] is a [`PVec`] of
//! `height * width` elements in row-major order: entry `(row, col)` is element
//! `row * width + col`, and each entry is a polynomial of `d` residues in $\[0, q)$.

use crate::arithmetic::{CoeffType, poly_vec::PVec};

/// A `height` by `width` matrix of polynomials of dimension `d` over $\mathbb Z_q$, stored
/// row-major in a [`PVec`] (see the module documentation).
pub struct PMat {
    /// Number of rows.
    height: usize,
    /// Number of columns.
    width: usize,
    /// The `height * width` entries, row after row.
    vec: PVec
}

impl PMat {
    /// A uniformly random matrix: `height * width` polynomials of `d` independent uniform residues
    /// in $\[0, q)$, expanded deterministically from `seed` by [`PVec::rand`]. This is how every
    /// party derives the public matrices $\mathbf A$, $\mathbf B$, $\mathbf D$ from
    /// `params.a_seed`, `params.b_seed`, `params.d_seed`.
    pub fn rand(height: usize, width: usize, d: usize, q: CoeffType, seed: [u8; 32]) ->  Self {
        let vec = PVec::rand(height * width, d, q, seed);
        Self { height, width, vec }
    }

    /// Number of rows ($n_A$, $n_B$ or $n_D$ of the paper, `params.n`).
    pub fn height(&self) -> usize {
        self.height
    }

    /// Number of columns (`params.width_a`, `params.width_b` or `params.width_d`).
    pub fn width(&self) -> usize {
        self.width
    }

    /// Entry `(row, col)` as the slice of its `d` coefficients; panics if out of range.
    pub fn element(&self, row: usize, col: usize) -> &[CoeffType] {
        self.vec.element(row * self.width + col)
    }
}

/// Deep copy.
impl Clone for PMat {
    fn clone(&self) -> Self {
        Self { height: self.height, width: self.width, vec: self.vec.clone() }
    }
}
