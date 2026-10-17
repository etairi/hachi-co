//! Public matrices from a seed: $A \leftarrow R_q^{n_A \times \delta 2^m}$,
//! $B \leftarrow R_q^{n_B \times n_A \delta 2^r}$ and $D \leftarrow R_q^{n_D \times (\delta(2^r + n + n n_A) + n m_\rho)}$
//! of Setup (6_Collaborative_Hachi.tex:283-287), each entry uniform in $R_q$.
//!
//! Entry $(i, j)$ of a `rows` $\times$ `cols` matrix is read from its own stream,
//! [`ShakePrg::new`]`(seed, domain, i * cols + j)` (row-major index), as $d$ residues of
//! [`uniform_mod_q`], coefficient $k$ for $X^k$. Entries can thus be generated independently and
//! in any order (as jali expands its public matrices, `src/abdlop.rs`).

use crate::prg::ShakePrg;
use crate::uniform::uniform_mod_q;
use crate::{Error, Result};

/// The $d$ coefficients of the entry with row-major index `index`: [`uniform_mod_q`] on the
/// stream [`ShakePrg::new`]`(seed, domain, index)`.
///
/// # Errors
/// [`Error::EmptyRange`] if $q = 0$; [`Error::RejectionLimit`] as in
/// [`crate::uniform::uniform_below`] (probability below $2^{-256}$ per coefficient).
pub fn expand_entry(
    seed: &[u8; 32],
    domain: &[u8],
    index: u64,
    d: usize,
    q: u64,
) -> Result<Vec<u64>> {
    uniform_mod_q(&mut ShakePrg::new(seed, domain, index), q, d)
}

/// The `rows` $\times$ `cols` matrix over $R_q = \mathbb Z_q\[X\]/(X^d + 1)$ whose entry $(i, j)$ is
/// [`expand_entry`]`(seed, domain, i * cols + j, d, q)`; indexed as `m[i][j][k]`, the coefficient
/// of $X^k$ in entry $(i, j)$.
///
/// # Errors
/// [`Error::EmptyRange`] if $q = 0$; [`Error::InvalidParameter`] if `rows * cols` exceeds
/// $2^{64}$; [`Error::RejectionLimit`] as in [`expand_entry`].
pub fn expand(
    seed: &[u8; 32],
    domain: &[u8],
    rows: usize,
    cols: usize,
    d: usize,
    q: u64,
) -> Result<Vec<Vec<Vec<u64>>>> {
    if q == 0 {
        return Err(Error::EmptyRange);
    }
    let entries = u128::from(rows as u64) * u128::from(cols as u64);
    if entries > 1u128 << 64 {
        return Err(Error::InvalidParameter("matrix has more than 2^64 entries"));
    }
    let cols64 = cols as u64;
    (0..rows as u64)
        .map(|i| {
            (0..cols64)
                .map(|j| expand_entry(seed, domain, i * cols64 + j, d, q))
                .collect()
        })
        .collect()
}
