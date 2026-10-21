//! Errors: [`Error`] for the prover and the parameter derivation, [`VerifyError`] for the
//! verifier, which never panics on proof data.

/// A failure of parameter derivation, of committing or of proving.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An arithmetic error (a length, a modulus, a decomposition that does not fit).
    #[error(transparent)]
    Math(#[from] hachi_co_math::Error),
    /// A sampling error (an empty range, a challenge set that cannot be sampled).
    #[error(transparent)]
    Sampling(#[from] hachi_co_sampling::Error),
    /// Parameters that the scheme cannot use (a ring degree that $k$ does not divide, a base below
    /// 2, a polynomial with fewer variables than the ring has coefficients, ...).
    #[error("parameter: {0}")]
    Parameter(String),
    /// An input of the wrong length or shape.
    #[error("shape: {0}")]
    Shape(String),
    /// An invariant of the honest prover failed (for example a bound on the folded witness); it
    /// signals a bug or parameters that do not cover the input, never a dishonest verifier.
    #[error("protocol: {0}")]
    Protocol(String),
}

/// Why the verifier rejects.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum VerifyError {
    /// The proof or the statement does not have the shape the parameters prescribe.
    #[error("malformed: {0}")]
    Malformed(String),
    /// A check of the protocol failed; the message names it.
    #[error("check failed: {0}")]
    Check(String),
    /// The verifier could not evaluate a public quantity (parameters it cannot use).
    #[error(transparent)]
    Error(#[from] Error),
}

impl From<hachi_co_math::Error> for VerifyError {
    fn from(e: hachi_co_math::Error) -> Self {
        Self::Error(Error::Math(e))
    }
}

impl From<hachi_co_sampling::Error> for VerifyError {
    fn from(e: hachi_co_sampling::Error) -> Self {
        Self::Error(Error::Sampling(e))
    }
}

/// The result type of the crate.
pub type Result<T> = core::result::Result<T, Error>;
