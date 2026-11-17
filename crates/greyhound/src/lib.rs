//! # Greyhound
//!
//! The polynomial commitment scheme Greyhound of Nguyen and Seiler (CRYPTO 2024, Cryptology
//! ePrint Archive 2024/1293), ported from the front end of the authors' C implementation
//! (`greyhound.c` of `lattice-dogs/labrador` at commit 8b6626b, "the reference"), with the
//! LaBRADOR proof of the reduced relation by jali 0.3.1 (`jali::labrador`, BS23).
//!
//! A polynomial $f \in \mathbb Z_q\[X\]$ of degree below $64\ell$ is given by its coefficients,
//! read as $\ell$ ring elements of $\mathcal R_q = \mathbb Z_q\[X\]/(X^{64} + 1)$ (coefficient
//! $64i + j$ of $f$ is coefficient $j$ of ring element $i$). Then:
//!
//! - [`PublicParams::new`] chooses the parameters by the reference's rule ([`params`]) and
//!   expands the public matrices from a seed ([`key`]); [`PublicParams::prover_key`] prepares
//!   the prover's transforms.
//! - [`commit()`] commits to $f$ ([`mod@commit`]): the commitment is $u_1 \in
//!   \mathcal R_q^{\kappa_1}$.
//! - [`prove`] proves $f(x) = y$: the front end of [`eval`] (the reference's `polcom_eval`)
//!   produces $u_2$, the challenges and a witness of the reduced relation of [`relation`] (the
//!   reference's `polcom_reduce`), which jali's LaBRADOR proves.
//! - [`verify`] checks the reference's conditions on the stated norm, rebuilds the relation and
//!   verifies the LaBRADOR proof.
//!
//! The modulus is a runtime parameter: any prime $q \equiv 5 \pmod 8$ with $2^{16} < q < 2^{62}$.
//! jali 0.3.0 takes only $q < 2^{48}$, and the workspace patches it to its vendored copy, 0.3.1
//! (`vendor/jali`, with `vendor/jali.patch`), which takes moduli up to $2^{62}$.
//!
//! [`tail`] is the tail of Hachi-Co's base PCS: commitments to Hachi's final tables of digits
//! and one LaBRADOR proof of several multilinear evaluation claims over $\mathbb F_{q^K}$.
//!
//! **Not zero knowledge**, as the reference and jali's LaBRADOR. **Variable time.**
//!
//! Deviations from the reference are documented where they occur: the public matrices
//! ([`key`]), the order of $\hat t$ ([`mod@commit`]), the decomposition of $w_i$, the hashing of
//! $x$ and $y$ and the computation of $w_i$ and $z$ ([`eval`]), the challenges' operator norm
//! ([`challenge`]), the slack of the security rule ([`params`]) and the layout of jali's
//! statement ([`relation`]).
#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod challenge;
pub mod commit;
pub mod digits;
pub mod encode;
pub mod error;
pub mod eval;
pub mod key;
pub mod params;
pub mod pcs;
pub mod relation;
pub mod tail;

pub use commit::{Commitment, Opening, commit};
pub use error::{Error, Result};
pub use eval::{FrontEnd, FrontProof, FrontWitness, evaluate, prove_front};
pub use params::{Params, Security};
pub use pcs::{Proof, ProverKey, PublicParams, prove, prove_labrador, verify};
pub use relation::{Layout, Relation};
pub use tail::{Claim, ProverClaim, TailOpening, TailParams};
