//! The error type of the crate.
//!
//! Every fallible public function returns [`Result`]. Errors report rejected parameters and
//! inputs of the wrong shape; arithmetic on well-formed inputs never fails. A `u64` residue that
//! is not reduced is not an error: every function reads it modulo its modulus (see the crate
//! documentation).

/// A rejected parameter or input: errors are values at API boundaries (`docs/m0-spec.md`, rule 5).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// The modulus is not an odd prime below $2^{63}$ (the residue convention of
    /// `docs/m0-spec.md`, "Conventions").
    #[error("modulus {0} is not an odd prime below 2^63")]
    Modulus(u64),
    /// The modulus is not congruent to $5 \bmod 8$, which Hachi's Lemma 5 needs for
    /// $\mathcal R_q^H \cong \mathbb F_{q^k}$ (draft `4_preliminaries.tex`:61).
    #[error("modulus {0} is not congruent to 5 mod 8")]
    ModulusClass(u64),
    /// Any other invalid parameter, named by the message.
    #[error("invalid parameter: {0}")]
    Parameter(&'static str),
    /// An input has the wrong length.
    #[error("wrong length: expected {expected}, got {actual}")]
    Length {
        /// The required length.
        expected: usize,
        /// The length that was passed.
        actual: usize,
    },
    /// An integer lies outside the range that the requested number of balanced digits represents
    /// (the failure case of $\mathbf J^{-1}$, draft `4_preliminaries.tex`:100).
    #[error("{value} is outside the representable range [{min}, {max}]")]
    NotRepresentable {
        /// The integer to decompose.
        value: i128,
        /// The smallest representable integer.
        min: i128,
        /// The largest representable integer.
        max: i128,
    },
    /// A ring element is not in the subfield $\mathcal R_q^H$ (Hachi, Lemma 5 and Eq. (7)).
    #[error("ring element is not in the subfield R_q^H")]
    NotInSubfield,
    /// A bound exceeds what the arithmetic supports (for example, more NTT primes than the table
    /// holds).
    #[error("arithmetic capacity exceeded")]
    Overflow,
}

/// The result type of the crate (`docs/m0-spec.md`, rule 5).
pub type Result<T> = core::result::Result<T, Error>;
