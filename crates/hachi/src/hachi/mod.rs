//! # The Hachi polynomial commitment scheme
//!
//! Entry points of the single-prover scheme of the paper, Section 4, for a multilinear polynomial
//! $f$ with coefficients in $\mathbb Z_q$ (the base-field case of Section 3; the conventions and the
//! notation table are in the crate-level overview):
//!
//! * [`setup`]: the public parameters [`setup::Parameters`] (ring, folding split, decomposition
//!   base, dimensions and seeds of the commitment matrices).
//! * [`commit`]: the inner Ajtai commitments $\mathbf t_i = \mathbf A \mathbf s_i$ and the outer
//!   commitment $\mathbf u = \mathbf B \hat{\mathbf t}$ (Section 4.1, Eq. (13)-(14)).
//! * [`prove`]: one round of the evaluation proof: reduction to $\mathbf R_q$, the folded response
//!   $\mathbf z$, ring switching to $\mathbb Z_q\[X\]$ and the two sumchecks over $\mathbb F_{q^4}$
//!   (Sections 4.2 and 4.3, Fig. 3 and Fig. 7).
//! * [`verify`]: the verifier of that round.
//! * [`common`]: the evaluation table of the verification matrix $\mathbf M_\alpha$, built by both
//!   the prover and the verifier (Section 4.3).
//! * [`prover_utils`]: prover-only helpers (first message and reduction, response, lifting to
//!   $\mathbb Z_q\[X\]$, next witness, the sumcheck polynomials $F_0$ and $F_\alpha$).
//!
//! The four operations are the traits [`setup::Setup`], [`commit::Commit`], [`prove::Prove`] and
//! [`verify::Verify`], implemented for the stateless marker type [`Hachi`], so that a call reads
//! `Hachi::commit(&mut witness, &params)`. Ring, field, sumcheck and transcript arithmetic is in
//! [`crate::arithmetic`].

// Documentation convention for equations in this crate: inside $...$ keep an alphanumeric character
// immediately before every `_` (write `\mathbf s_i`, `\mathbb F_{q^4}`, `\hat{\mathbf t}\relax_i`,
// never `\mathbf{s}_i` or `\hat{\mathbf t}_i`). Markdown treats `}_i ... x_{` as emphasis and
// inserts <em> tags into the equation before KaTeX sees it.

pub mod setup;
pub mod commit;
pub mod prove;
pub mod verify;
pub mod common;
mod prover_utils;

/// Marker type on which the scheme is implemented: [`Setup`](setup::Setup),
/// [`Commit`](commit::Commit), [`Prove`](prove::Prove) and [`Verify`](verify::Verify) are
/// implemented for it. It carries no state; all state lives in [`setup::Parameters`],
/// [`commit::CommitmentWithState`] and [`prove::ProofRound`].
pub struct Hachi {}
