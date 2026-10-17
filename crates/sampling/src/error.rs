//! The error type of the samplers.

/// Errors returned at the API boundary of the samplers.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// The set to sample from is empty: $n = 0$ in
    /// [`uniform_below`](crate::uniform::uniform_below), $q = 0$, $\beta = 0$ (the set $S_0$ of
    /// 4_preliminaries.tex:88 is empty), or $lo \gt hi$.
    #[error("empty sampling range")]
    EmptyRange,
    /// A parameter lies outside the supported domain; the message names it.
    #[error("invalid parameter: {0}")]
    InvalidParameter(&'static str),
    /// Two vectors that must have the same length do not.
    #[error("length mismatch: {left} != {right}")]
    LengthMismatch {
        /// Length of the first vector.
        left: usize,
        /// Length of the second vector.
        right: usize,
    },
    /// The precondition $\lVert v \rVert_\infty \le \beta_1$ of the rejection-sampling lemma
    /// (4_preliminaries.tex:122-127) is violated. Rej must not run on such a $v$: its output
    /// would no longer be independent of $v$.
    #[error(
        "the secret exceeds beta1 in infinity norm (precondition of Rej, 4_preliminaries.tex:123)"
    )]
    SecretTooLarge,
    /// The mask is not in $[-\gamma, \gamma]^N$, the support of $\chi_r$
    /// (4_preliminaries.tex:117).
    #[error("the mask lies outside [-gamma, gamma]^N, the support of chi_r")]
    MaskOutOfRange,
    /// A rejection loop reached its cap of [`MAX_TRIES`](crate::uniform::MAX_TRIES) tries.
    /// Every try succeeds with probability above $1/2$, so for a uniform source this happens
    /// with probability below $2^{-256}$: the random source is broken.
    #[error("rejection sampling gave up after {0} tries; the random source is not uniform")]
    RejectionLimit(u32),
}

/// `Result` with the sampler [`Error`].
pub type Result<T> = core::result::Result<T, Error>;
