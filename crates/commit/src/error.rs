//! The error type of the crate.

use hachi_co_mpc::PartyId;

/// Everything that can go wrong in Commit and Open: inputs of the wrong shape, a network that does
/// not match the context, malformed messages from other parties, and the errors of the layers
/// below (`hachi-co-core`, `hachi-co-math`, `hachi-co-mpc`). Errors are values
/// (`docs/m1-spec.md`, rule 5): nothing in this crate panics on malformed input.
///
/// The enum is non-exhaustive.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An error of the foundation (`hachi-co-core`): a state or block of the wrong shape, a
    /// randomness source that failed, a replay request that does not match the recording.
    #[error(transparent)]
    Core(#[from] hachi_co_core::Error),
    /// An error of the arithmetic layer (`hachi-co-math`), e.g. a polynomial without $d$
    /// coefficients.
    #[error(transparent)]
    Math(#[from] hachi_co_math::Error),
    /// An error of the communication layer (`hachi-co-mpc`).
    #[error(transparent)]
    Mpc(#[from] hachi_co_mpc::Error),
    /// An input or a received vector has the wrong length.
    #[error("wrong shape of {what}: expected {expected}, got {actual}")]
    Shape {
        /// What has the wrong shape.
        what: &'static str,
        /// The required length.
        expected: usize,
        /// The length that was passed or received.
        actual: usize,
    },
    /// A party index outside $1, \dots, n$.
    #[error("party {party} is not in 1..={n}")]
    InvalidParty {
        /// The index that was passed.
        party: PartyId,
        /// The number of parties.
        n: usize,
    },
    /// The [`hachi_co_mpc::Comm`] does not match the context: another number of parties or
    /// another modulus.
    #[error("the network does not match the context: {what} is {actual}, expected {expected}")]
    Network {
        /// `"n"` or `"q"`.
        what: &'static str,
        /// The value of the context.
        expected: u64,
        /// The value of the network.
        actual: u64,
    },
    /// A party sent a residue that is not below $q$ (every honest party sends reduced residues).
    #[error("party {from} sent {what} with residue {value}, which is not below q = {q}")]
    NonCanonical {
        /// The sender.
        from: PartyId,
        /// What was received.
        what: &'static str,
        /// The residue.
        value: u64,
        /// The modulus.
        q: u64,
    },
    /// The same block of $\mathbf f'$ appears twice in a list of blocks that must hold each block
    /// at most once (a party's state, or the blocks whose slice of $\hat{\mathbf t}$ is formed).
    #[error("block {index} appears more than once")]
    DuplicateBlock {
        /// The block index.
        index: usize,
    },
}

/// The result type of the crate.
pub type Result<T> = core::result::Result<T, Error>;

/// `Err(Shape)` unless `actual == expected`.
pub(crate) fn check_len(what: &'static str, expected: usize, actual: usize) -> Result<()> {
    if actual != expected {
        return Err(Error::Shape {
            what,
            expected,
            actual,
        });
    }
    Ok(())
}

/// $2^e$ as a `usize`, or a shape error naming `what`.
pub(crate) fn pow2(e: usize, what: &'static str) -> Result<usize> {
    u32::try_from(e)
        .ok()
        .and_then(|e| 1usize.checked_shl(e))
        .ok_or(Error::Core(hachi_co_core::Error::Overflow(what)))
}
