//! Errors of the parameter layer.
//!
//! [`derive`](fn@crate::derive) returns an error only when the Setup cascade cannot be evaluated at
//! all (an exponent would be negative, a quantity would overflow `u128`, no modulus fits in a
//! `u64`, and so on). Choices for which the cascade *can* be evaluated but which violate a
//! condition of the draft are not errors: they produce a [`Report`](crate::Report) whose
//! constraints say which conditions fail.

use thiserror::Error;

/// The error type of this crate.
#[derive(Debug, Error)]
pub enum Error {
    /// A choice is outside the domain on which the cascade is defined (for example $n = 0$ or
    /// $b < 2$); the message names the choice and the reason.
    #[error("invalid choice: {0}")]
    InvalidChoice(String),

    /// The tight $\tau$ rule needs $\lceil b/2 \rceil - 1 \ge 1$, i.e. $b \ge 3$: with $b = 2$
    /// the balanced digits $\lbrace -1, 0 \rbrace$ represent no positive value.
    #[error(
        "the tight tau rule is undefined for b = {b}: balanced base-{b} digits represent no positive value"
    )]
    TightTauUndefined {
        /// The decomposition base.
        b: u64,
    },

    /// A quantity of the cascade does not fit in `u128` (or an exponent in `u32`).
    #[error("overflow while computing {0}")]
    Overflow(&'static str),

    /// An exponent of the cascade is negative, so the cascade is undefined. This happens exactly
    /// when $\alpha - \kappa > \ell$ (6_Collaborative_Hachi.tex:273), because then
    /// $m = \ell' - r < 0$.
    #[error("{0}")]
    NegativeExponent(String),

    /// [`Omega::MinForTarget`](crate::Omega::MinForTarget) found no radius up to the search cap
    /// (for the sparse set, none up to $d$).
    #[error("no challenge radius omega <= {cap} reaches 2^(r+1)/|C| <= 2^-{lambda}")]
    NoOmega {
        /// The security target.
        lambda: u32,
        /// The largest radius that was tried.
        cap: u64,
    },

    /// [`QChoice::Smallest`](crate::QChoice::Smallest) found no admissible prime below $2^{64}$.
    #[error("no prime q = 5 (mod 8) below 2^64 satisfies the modulus rule: {0}")]
    NoModulus(String),

    /// The rejection-constant comparison $M \le e^{1/n}$ was not decided within the iteration
    /// cap of its exact series bounds (unreachable for the cascade, see
    /// [`rules::rejection_constant_within`](crate::rules::rejection_constant_within)).
    #[error("could not decide M <= e^(1/n) exactly: {0}")]
    Undecided(String),

    /// The embedded `presets/secure.json` could not be parsed.
    #[error("secure.json: {0}")]
    Json(#[from] serde_json::Error),

    /// No preset has the requested name.
    #[error("unknown preset {0:?}")]
    UnknownPreset(String),
}

/// Shorthand for `Result<T, Error>`.
pub type Result<T> = core::result::Result<T, Error>;
