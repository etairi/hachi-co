//! The error types of the crate.

use hachi_co_mpc::PartyId;

/// Everything that can go wrong on the provers' side of $\Pi\_{\mathsf{conv}}$ and in the
/// relation checker: inputs of the wrong shape, a network that does not match the context, and
/// the errors of the layers below. Errors are values (`docs/m1-spec.md`, rule 5): nothing in this
/// crate panics on malformed input. The verifier's check has its own type, [`VerifyError`].
///
/// The enum is non-exhaustive.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An error of the foundation (`hachi-co-core`): a state or public vectors of the wrong shape.
    #[error(transparent)]
    Core(#[from] hachi_co_core::Error),
    /// An error of the arithmetic layer (`hachi-co-math`).
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
    /// A party index outside $1, \dots, n$, or a state of another party than the caller.
    #[error("party {party} is not valid here (n = {n})")]
    InvalidParty {
        /// The index.
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
}

/// The result type of the provers' side.
pub type Result<T> = core::result::Result<T, Error>;

/// Why the verifier's check $y = E(u)$ of $\Pi\_{\mathsf{conv}}$ (`field-to-ring-rok.tex`:160-164)
/// rejects. Every malformed or tampered message gives one of these, never a panic.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum VerifyError {
    /// The message $E(u)$ does not have $d$ coefficients.
    #[error("E(u) has {actual} coefficients, expected d = {expected}")]
    Shape {
        /// $d$.
        expected: usize,
        /// The number of coefficients received.
        actual: usize,
    },
    /// A residue of the message $E(u)$ or of the claim $y$ is not below $q$ (the encodings are
    /// canonical).
    #[error("{what}: residue {value} is not below q = {q}")]
    NonCanonical {
        /// `"E(u)"` or `"y"`.
        what: &'static str,
        /// The residue.
        value: u64,
        /// The modulus.
        q: u64,
    },
    /// $E(u) \notin \mathcal R\_q^H$: it is not of the form of Hachi's Eq. (7).
    #[error("E(u) is not in the subfield R_q^H")]
    NotInSubfield,
    /// $E(u) \in \mathcal R\_q^H$ but, read in $\mathbb F\_{q^k}$, it differs from $y$.
    #[error("E(u) differs from the claimed evaluation y")]
    Mismatch,
}

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

/// $2^e$ as a `usize`, or an overflow error naming `what`.
pub(crate) fn pow2(e: usize, what: &'static str) -> Result<usize> {
    u32::try_from(e)
        .ok()
        .and_then(|e| 1usize.checked_shl(e))
        .ok_or(Error::Core(hachi_co_core::Error::Overflow(what)))
}
