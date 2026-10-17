//! The communication layer of Hachi-Co (milestone 1, `docs/m1-spec.md`, section `hachi-co-mpc`).
//!
//! The party code of every building block is written once against the [`Comm`] trait: the
//! additively homomorphic functionality $\mathcal F\_{\mathsf{AH}}$ (draft
//! `4_preliminaries.tex`:185-200), the gather of the masked responses $\mathbf z\_i$ at prover 1
//! (`5_Building_Blocks.tex`:129-130), the agreement on an abort (`5_Building_Blocks.tex`:129, 131)
//! and the pairwise exchange of the resharing in Commit (`6_Collaborative_Hachi.tex`:303-305).
//! Milestone 1 runs it with $n = 1$ on [`Loopback`], where every call is the identity; milestone 2
//! adds an in-memory $n$-party network with byte and round accounting behind the same trait
//! (plan `docs/hachi-co-plan.md`, design rule 5: the MPC primitives know nothing about the
//! protocol).
//!
//! **Conventions.** Parties are numbered $1, \dots, n$ ([`PartyId`]). Values in $\mathbb Z_q$ are
//! `u64` residues; a call reads them modulo [`Comm::modulus`] where it adds them, and returns
//! reduced residues. Short signed vectors (the responses $\mathbf z\_i$) are `i64`. Every call
//! carries a `&'static str` label: all parties make the same calls, with the same labels, in the
//! same order, so that a network implementation can match messages and detect a desynchronised
//! party. Errors are values ([`Error`]); a [`Comm`] never panics on what other parties send.
//!
//! References: "the draft" is the Hachi-Co paper draft (`collaborative-hatchi`, commit
//! `625ced1`), cited by file and line.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod error;
mod loopback;

pub use error::{Error, Result};
pub use loopback::Loopback;

/// A party (prover) index, in $1, \dots, n$ (`docs/m1-spec.md`, "Conventions"; the draft numbers
/// the provers $i \in \[n\]$, `4_preliminaries.tex`:154).
pub type PartyId = usize;

/// The communication primitives of the collaborative provers.
///
/// Every method is called by all $n$ parties with the same `label`, in the same order (a
/// collective call). An implementation returns an [`Error`] for malformed calls (a party index
/// outside $1, \dots, n$, a wrong number of outgoing vectors, inputs of different lengths) and for
/// transport failures; it never panics on data received from other parties.
pub trait Comm {
    /// The number of parties $n \ge 1$.
    fn n(&self) -> usize;

    /// This party's index $p \in \lbrace 1, \dots, n \rbrace$.
    fn id(&self) -> PartyId;

    /// The modulus $q$ of the additions in [`Comm::fah`].
    fn modulus(&self) -> u64;

    /// $\mathcal F\_{\mathsf{AH}}$ (draft `4_preliminaries.tex`:185-200): every party passes the
    /// image $L(\mathbf x\_p)$ of its additive share $\mathbf x\_p$ under the public linear map
    /// $L$ (each party applies $L$ locally, so the functionality only adds), and every party
    /// receives $\sum\_{p=1}^{n} L(\mathbf x\_p) \bmod q$, entry by entry, as reduced residues.
    ///
    /// All parties pass vectors of the same length; the output has that length. Entries are read
    /// modulo $q$, so an unreduced entry stands for its residue class.
    ///
    /// # Errors
    /// [`Error::LengthMismatch`] if the parties' inputs differ in length; [`Error::Transport`] for
    /// a network failure.
    fn fah(&mut self, label: &'static str, local: &[u64]) -> Result<Vec<u64>>;

    /// Gathers one signed vector from every party at party `to` (draft
    /// `5_Building_Blocks.tex`:129: every prover sends $\mathbf z\_i$ to prover 1). Party `to`
    /// receives `Some(v)` with `v[p - 1]` the vector of party $p$ (its own included); every other
    /// party receives `None`. The vectors are delivered unchanged.
    ///
    /// # Errors
    /// [`Error::InvalidParty`] if `to` is not in $1, \dots, n$; [`Error::Transport`] for a network
    /// failure.
    fn gather(
        &mut self,
        label: &'static str,
        to: PartyId,
        local: &[i64],
    ) -> Result<Option<Vec<Vec<i64>>>>;

    /// Agreement on an abort (draft `5_Building_Blocks.tex`:129, 131: a prover that aborts tells
    /// the other provers and the verifier): every party passes a flag and every party receives
    /// `true` iff some party passed `true`.
    ///
    /// # Errors
    /// [`Error::Transport`] for a network failure.
    fn any(&mut self, label: &'static str, local: bool) -> Result<bool>;

    /// Pairwise exchange (draft `6_Collaborative_Hachi.tex`:303-305: every party $i$ sends
    /// $\langle \mathbf f\_i \rangle\_j$ to party $j$): `outgoing[j - 1]` goes to party $j$, this
    /// party included, and the result `incoming` has `incoming[j - 1]` = the vector that party $j$
    /// sent to this party. The vectors are delivered unchanged (not reduced).
    ///
    /// # Errors
    /// [`Error::WrongMessageCount`] unless `outgoing` has $n$ vectors; [`Error::Transport`] for a
    /// network failure.
    fn exchange(&mut self, label: &'static str, outgoing: Vec<Vec<u64>>) -> Result<Vec<Vec<u64>>>;
}
