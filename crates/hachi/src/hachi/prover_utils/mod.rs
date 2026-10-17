// Doc comments in this file use `\_` for underscores and `\[` `\]` for brackets inside the
// $...$ KaTeX formulas: rustdoc's Markdown would otherwise pair underscores into emphasis (and
// brackets into links) and split the formula; the escapes reach KaTeX as plain `_`, `[`, `]`.

//! Prover-side utilities for one round of the Hachi evaluation proof.
//!
//! [`crate::hachi::prove`] is the driver; the work is split by the algebraic domain in which it
//! happens:
//!
//! * [`zq_zq`]: the parts that touch the $\mathbb{Z}\_q$ witness stream directly, for a
//!   polynomial over $\mathbb{Z}\_q$ evaluated at a point over $\mathbb{Z}\_q$ (the base-field
//!   case of the paper's Section 3): the reduction of the evaluation claim to the ring element
//!   $Y$ together with the first prover message $\mathbf{w}$, and the assembly of the next witness
//!   $(\mathbf z^{\prime}, \mathbf{r})$ of the paper's Section 4.3.
//! * [`rq`]: the parts that are generic over $\mathbf{R}\_q = \mathbb{Z}\_q\[X\]/(X^d+1)$:
//!   sampling the commitment matrices, the commitment $\mathbf{v} = \mathbf{D}\hat{\mathbf{w}}$,
//!   the folded response $\mathbf{z} = \sum\_i c\_i \mathbf{s}\_i$, and the lifting of ring
//!   products to $\mathbb{Z}\_q\[X\]$ (quotient and remainder modulo $X^d + 1$) that ring
//!   switching needs.
//! * [`sumcheck`]: the two sumcheck polynomials $F\_{0,\tau\_0}$ (range check) and
//!   $F\_{\alpha,\tau\_1}$ (linear relation) over $\mathbb{F}\_{q^4}$, and the prover half of
//!   the sumcheck protocol of the paper's Fig. 6 and 7.
//!
//! Notation follows the crate-level table: `params.r`, `params.m` are the folding parameters
//! ($2^r$ chunks of $2^m$ ring elements), `params.delta` is $\delta$, `params.delta_z` is
//! $\tau$, `params.n` is the height $n$ of the commitment matrices and `params.k` is the
//! challenge sparsity $\omega$. Vectors of ring elements are
//! [`crate::arithmetic::poly_vec::PVec`]s: coefficient $j$ of element $i$ of a vector of
//! dimension-$d$ elements sits at index `i * d + j`. Residues of $\mathbb{Z}\_q$ are `u64` in
//! $\[0, q)$; digits and the response $\mathbf{z}$ are wrapping two's-complement `u64`.

pub mod zq_zq;
pub mod rq;
pub mod sumcheck;