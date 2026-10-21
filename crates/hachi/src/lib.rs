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
//!   - multilinear evaluations, reduced to $\mathcal R_q$ by Sections 3.2 and 3.1;
//!   - inner products with arbitrary public weights, by Theorem 2.
//! - **The complete protocol** (Section 4.4):
//!   - iterations of Sections 4.2-4.3 (fold, ring switching, the range and linear sumchecks),
//!     each followed by the reduction of Section 4.5;
//!   - every later witness is committed without re-decomposition;
//!   - the last witness is sent in the clear.
//! - **Fiat-Shamir** on a framed transcript that absorbs the parameters, the commitment, the
//!   point and the value, starting from a caller-supplied context.
//! - **The number of iterations** that gives the smallest proof, unless fixed by the choices
//!   (`pcs`, "Schedule").
//! - **Packed encodings**: $\lceil \log_2 q \rceil$ bits per residue and $\lceil \log_2 b
//!   \rceil$ bits per digit of the final witness, with exact word counts (`pcs`,
//!   "Encodings").
//!
//! **Not implemented:**
//! - **Hiding.** The scheme is not hiding (paper, 4_P:333 of the Hachi-Co draft); the
//!   Hwang-Seo-Song changes come later.
//! - **Secure parameters.** [`HachiChoices::toy`] are test parameters.
//!
//! ## Modules
//!
//! - [`params`]: the choices, layouts and the schedule of iterations.
//! - [`commit`]: the context and the inner-outer commitment.
//! - [`relation`]: the linear relation of an iteration and its ring switching.
//! - [`sumcheck`]: the range and linear sumchecks.
//! - [`iteration`]: one iteration and the reduction to the next claim.
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
pub mod pcs;
pub mod relation;
pub mod sumcheck;

pub use error::{Error, Result, VerifyError};
pub use params::{HachiChoices, Layout, Schedule};
pub use pcs::{Commitment, HachiPcs, Proof, State, VERSION};
