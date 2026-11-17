//! Errors.

/// A failure of the parameter derivation, of committing, of proving or of verifying.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Parameters that the scheme cannot use, or no parameters that meet the security rule.
    #[error("parameter: {0}")]
    Parameter(&'static str),
    /// An input of the wrong length or with a coefficient outside $[0, q)$.
    #[error("shape: {0}")]
    Shape(&'static str),
    /// A proof or commitment that does not decode.
    #[error("encoding")]
    Encoding,
    /// A check of the verifier failed; the message names it.
    #[error("invalid proof: {0}")]
    InvalidProof(&'static str),
    /// The witness does not satisfy the reduced relation (a bug, or a wrong input to the
    /// prover); the message names the constraint family.
    #[error("witness: {0}")]
    Witness(&'static str),
    /// An error of the workspace arithmetic.
    #[error(transparent)]
    Math(#[from] hachi_co_math::Error),
    /// An error of the samplers.
    #[error(transparent)]
    Sampling(#[from] hachi_co_sampling::Error),
    /// An error of jali's LaBRADOR, among them [`jali::Error::Parameter`] for a modulus of
    /// $2^{62}$ or more, which jali 0.3.1 does not support, and [`jali::Error::InvalidProof`].
    #[error("LaBRADOR: {0}")]
    Labrador(#[from] jali::Error),
}

/// A result with [`Error`].
pub type Result<T> = core::result::Result<T, Error>;
