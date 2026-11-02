//! # Hachi
//!
//! **Hachi** is the lattice-based multilinear polynomial commitment scheme of Nguyen, O'Rourke and
//! Zhang (Cryptology ePrint Archive 2026/156). This crate implements it on the modulus-agnostic
//! arithmetic of the Hachi-Co workspace, as the base PCS of Hachi-Co. It started from the
//! authors' public prototype (github.com/georgeorourke/hachi-pcs; see the README's "Attribution"
//! section). That prototype ran one iteration over $q = 2^{32} - 99$ with base-field points; it
//! was ported here, generalised, and completed.
//!
//! ## What is implemented
//!
//! - **Commitments** ([`HachiPcs::commit_multilinear`], [`HachiPcs::commit_vector`]):
//!   - the inner-outer Ajtai commitment of Section 4.1;
//!   - inputs are multilinear polynomials with coefficients in $\mathbb Z_q$, or vectors over
//!     $\mathbb F_{q^K}$ (Hachi-Co's blinders);
//!   - $q$ is any prime $\equiv 5 \pmod 8$ below $2^{63}$, and $\mathbb F_{q^K} =
//!     \mathcal R_q^H$ (Lemma 5).
//! - **Openings at points in $\mathbb F_{q^K}$:**
//!   - multilinear evaluations, reduced to $\mathcal R_q$ by Sections 3.2 and 3.1. A
//!     ring-switching sumcheck binds the partial evaluations of Section 3.2 to the committed
//!     polynomial ([`partials`]). This deviates from the paper, whose check leaves them free,
//!     so that a prover could claim any value;
//!   - inner products with arbitrary public weights, by Theorem 2.
//! - **The complete protocol** (Section 4.4):
//!   - iterations of Sections 4.2-4.3 (fold, ring switching, the range and linear sumchecks),
//!     each followed by the reduction of Section 4.5. That reduction also binds its partial
//!     evaluations by a ring-switching sumcheck ([`partials`], [`iteration`]), a deviation from
//!     Section 4.5 for the same reason as the input reduction;
//!   - every later witness is committed without re-decomposition;
//!   - the last witness is sent in the clear.
//! - **Fiat-Shamir** on a framed transcript that absorbs the parameters, the commitment, the
//!   point and the value, starting from a caller-supplied context.
//! - **The number of iterations** that gives the smallest proof, unless fixed by the choices
//!   (`pcs`, "Schedule").
//! - **Packed encodings**: $\lceil \log_2 q \rceil$ bits per residue and $\lceil \log_2 b
//!   \rceil$ bits per digit of the final witness, with exact word counts (`pcs`,
//!   "Encodings").
//! - **Masks**, optional ([`HachiChoices::hiding`], `hachi-pcs.tex`, Appendix A.2, after Jindo
//!   Section 2.4) for multilinear polynomials over the field and for vectors (the vector chains of
//!   that appendix): randomized commitments with a masking row per chunk, a first claim mixed
//!   with the masks by a challenge, and a first iteration masked by discrete Gaussians with
//!   rejection sampling (`pcs`, "Hiding").
//!
//! **Not implemented:**
//! - **Secure parameters.** [`HachiChoices::toy`] and [`HachiChoices::toy_hiding`] are test
//!   parameters.
//!
//! ## Modules
//!
//! - [`params`]: the choices, layouts and the schedule of iterations.
//! - [`commit`]: the context and the inner-outer commitment.
//! - [`relation`]: the linear relation of an iteration and its ring switching.
//! - [`sumcheck`]: the range and linear sumchecks.
//! - [`iteration`]: one iteration and the reduction to the next claim.
//! - [`partials`]: the sumcheck that binds the partial evaluations of a multilinear opening.
//! - [`pcs`]: the scheme and its encodings.
//! - [`mle`]: multilinear helpers.
//! - [`fs`]: the transcript.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod commit;
pub mod error;
pub mod fs;
pub mod iteration;
pub mod mle;
pub mod params;
pub mod partials;
pub mod pcs;
pub mod relation;
pub mod sumcheck;

pub use error::{Error, Result, VerifyError};
pub use params::{HachiChoices, Hiding, Layout, Schedule};
pub use pcs::{Commitment, HachiPcs, Proof, State, VERSION};
