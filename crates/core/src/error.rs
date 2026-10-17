//! The error type of the crate.

use hachi_co_mpc::PartyId;

/// Everything that can go wrong at the API boundary of this crate: rejected parameters, inputs of
/// the wrong shape, failures of the arithmetic, sampling and communication layers, misuse of a
/// randomness source, and (feature `fixtures`) fixture and replay errors. Errors are values
/// (`docs/m1-spec.md`, rule 5); nothing in this crate panics on malformed input.
///
/// The enum is non-exhaustive; the block crates may wrap it in their own error types.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An error of the arithmetic layer (`hachi-co-math`): an invalid modulus or degree, a
    /// polynomial of the wrong length, a value that $\mathbf J^{-1}$ cannot represent, an element
    /// outside $\mathcal R_q^H$.
    #[error(transparent)]
    Math(#[from] hachi_co_math::Error),
    /// An error of a sampler (`hachi-co-sampling`), e.g. a mask width above $2^{63} - 1$.
    #[error(transparent)]
    Sampling(#[from] hachi_co_sampling::Error),
    /// An error of the communication layer (`hachi-co-mpc`).
    #[error(transparent)]
    Mpc(#[from] hachi_co_mpc::Error),
    /// The parameter layer (`hachi-co-params`) refused to derive the parameters; the message is
    /// its error.
    #[error("parameters: {0}")]
    Params(String),
    /// A parameter set that this crate cannot work with (for example $K \ne k$); the message
    /// names it.
    #[error("invalid parameter: {0}")]
    Parameter(String),
    /// The layout derived from the parameters is inconsistent (`docs/m1-spec.md`: $u + 1 = \mu$,
    /// $t = $ the width of $\mathbf D$, $\mathrm{row}\_A + n\_A = n'$, and the $S\_p$ with the
    /// unheld positions partition $\[0, \mu - 1)$), or a dimension does not fit in `usize`.
    #[error("layout: {0}")]
    Layout(String),
    /// An input has the wrong length.
    #[error("wrong shape of {what}: expected {expected}, got {actual}")]
    Shape {
        /// What has the wrong shape.
        what: &'static str,
        /// The required length.
        expected: usize,
        /// The length that was passed.
        actual: usize,
    },
    /// A residue that must be canonical (in $\[0, q)$) is not.
    #[error("{what}: residue {value} is not below q = {q}")]
    NonCanonical {
        /// Where the residue was found.
        what: &'static str,
        /// The residue.
        value: u64,
        /// The modulus.
        q: u64,
    },
    /// A party index outside $1, \dots, n$.
    #[error("party {party} is not in 1..={n}")]
    InvalidParty {
        /// The index that was passed.
        party: PartyId,
        /// The number of parties.
        n: usize,
    },
    /// An exact integer computation left the range of `i64` (only for inputs far outside the
    /// protocol's bounds).
    #[error("integer overflow in {0}")]
    Overflow(&'static str),
    /// A [`PrgRandomness`](crate::PrgRandomness) stream was requested a second time. Reusing a
    /// mask or a share leaks the secret it hides (plan `docs/hachi-co-plan.md`, finding F5), so
    /// each stream is served once.
    #[error("randomness stream {kind} with index {index} was already used")]
    RandomnessReuse {
        /// The kind of randomness (the PRG domain).
        kind: &'static str,
        /// The stream index (the attempt for masks and $\boldsymbol\rho$, otherwise 0).
        index: u64,
    },
    /// A challenger was used out of order (for example
    /// [`Challenger::begin_attempt`](crate::Challenger::begin_attempt)`(a)` with $a \gt 0$ before
    /// attempt 0); the message names the misuse.
    #[error("challenger: {0}")]
    Challenger(String),
    /// A replay provider has no recorded value for the request, or the request does not match the
    /// recording (feature `fixtures`).
    #[error("replay: {0}")]
    Replay(String),
    /// A fixture could not be read or decoded, or disagrees with the Rust parameters (feature
    /// `fixtures`).
    #[error("fixture: {0}")]
    Fixture(String),
}

/// The result type of the crate.
pub type Result<T> = core::result::Result<T, Error>;
