//! Modulus-generic arithmetic for Hachi-Co (milestone 0, `docs/m0-spec.md`).
//!
//! - [`zq`]: $\mathbb Z_q$ for a runtime prime $q < 2^{63}$, with `u128` reference products.
//! - [`ntt`] and [`crt`]: negacyclic transforms over NTT primes below $2^{62}$ (sizes up to 4096)
//!   and the exact CRT straight to $\bmod q$, ported from jali.
//! - [`rns`]: exact products, inner products and matrix-vector products in
//!   $\mathbb Z_q\[X\]/(X^n + 1)$.
//! - [`ring`]: $\mathcal R_q = \mathbb Z_q\[X\]/(X^d + 1)$ with automorphisms and norms.
//! - [`decompose`]: the balanced decompositions $\mathbf G^{-1}$, $\mathbf J^{-1}$ and their
//!   gadgets.
//! - [`fqk`]: $\mathbb F_{q^k} \cong \mathcal R_q^H$ in Hachi's basis (Eq. (7)).
//! - [`subfield`]: $\mathcal R_q^H$, $\mathrm{Tr}\_H$, $\psi$, $\psi^{-1}$ and $E$ (Hachi,
//!   Lemma 5 and Theorem 2; draft `field-to-ring-rok.tex`).
//!
//! **Conventions** (`docs/m0-spec.md`). $q$ is a prime with $3 \le q < 2^{63}$ (and
//! $q \equiv 5 \pmod 8$ for the subfield); residues are `u64` values in $[0, q)$ and short signed
//! values are `i64`. Every function reads a `u64` residue argument modulo its modulus, so an
//! unreduced argument stands for its residue class, and every function returns reduced residues.
//! Errors are values ([`Error`]); they report invalid parameters and inputs of the wrong shape.
//!
//! References: "the draft" is the Hachi-Co paper draft (`collaborative-hatchi`, commit
//! `625ced1`), cited by file and line; "Hachi" is ePrint 2026/156 (NOZ26), cited by lemma,
//! theorem and equation number.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod arith;
pub mod crt;
pub mod decompose;
pub mod error;
pub mod fqk;
pub mod ntt;
pub mod ring;
pub mod rns;
pub mod subfield;
pub mod zq;

pub use error::{Error, Result};
pub use fqk::{Ext, Fqk};
pub use ring::{Poly, Ring};
pub use subfield::Subfield;
pub use zq::Zq;
