//! The error type of the communication layer.

use crate::PartyId;

/// A malformed collective call or a transport failure: errors are values at API boundaries
/// (`docs/m1-spec.md`, rule 5). The enum is non-exhaustive so that the network of milestone 2 can
/// add failure kinds.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// A party index outside $1, \dots, n$ (for example the destination of
    /// [`Comm::gather`](crate::Comm::gather)).
    #[error("party {party} is not in 1..={n}")]
    InvalidParty {
        /// The index that was passed.
        party: PartyId,
        /// The number of parties.
        n: usize,
    },
    /// [`Comm::exchange`](crate::Comm::exchange) needs exactly one outgoing vector per party.
    #[error("{label}: expected {expected} outgoing vectors (one per party), got {actual}")]
    WrongMessageCount {
        /// The label of the call.
        label: &'static str,
        /// The number of parties $n$.
        expected: usize,
        /// The number of vectors that was passed.
        actual: usize,
    },
    /// The parties passed inputs of different lengths to the same collective call.
    #[error("{label}: inputs of different lengths ({expected} and {actual})")]
    LengthMismatch {
        /// The label of the call.
        label: &'static str,
        /// The length of this party's input.
        expected: usize,
        /// The length of another party's input.
        actual: usize,
    },
    /// The modulus of $\mathcal F\_{\mathsf{AH}}$ must satisfy $2 \le q \lt 2^{63}$ (the residue
    /// convention of `docs/m0-spec.md`, "Conventions").
    #[error("invalid modulus {0}: F_AH needs 2 <= q < 2^63")]
    InvalidModulus(u64),
    /// A transport failure (a closed channel, a peer that left, a desynchronised call); the
    /// message names it.
    #[error("{label}: {message}")]
    Transport {
        /// The label of the call.
        label: &'static str,
        /// What failed.
        message: String,
    },
}

/// The result type of the crate.
pub type Result<T> = core::result::Result<T, Error>;
