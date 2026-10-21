//! The error types of the crate: [`Error`] for the provers (Setup, Commit, Eval),
//! [`VerifyError`] for the verifier, [`DecodeError`] for [`EvalProof::from_bytes`], and the two
//! kinds of malformed input they share, [`StatementError`] and [`ProofError`].
//!
//! Errors are values (`docs/m1-spec.md`, rule 5): nothing in this crate panics on malformed
//! statements, proofs or encodings.
//!
//! [`EvalProof::from_bytes`]: crate::EvalProof::from_bytes

/// A statement $(\mathbf u, \mathbf x, y)$ that does not fit the context: a commitment
/// $\mathbf u$ without $n\_B$ ring elements of $d$ coefficients, a point $\mathbf x$ without
/// $\ell$ coordinates, or a residue that is not below $q$. The statement is absorbed into the
/// Fiat-Shamir transcript, so its encoding must be canonical ([`crate::check_statement`]).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum StatementError {
    /// A part of the statement has the wrong length.
    #[error("wrong shape of {what}: expected {expected}, got {actual}")]
    Shape {
        /// What has the wrong shape.
        what: &'static str,
        /// The length the context requires.
        expected: usize,
        /// The length that was passed.
        actual: usize,
    },
    /// A residue of the statement is not below $q$.
    #[error("{what}: residue {value} is not below q = {q}")]
    NonCanonical {
        /// Where the residue was found.
        what: &'static str,
        /// The residue.
        value: u64,
        /// The modulus.
        q: u64,
    },
}

/// An [`EvalProof`](crate::EvalProof) that does not fit the context: a component of the wrong
/// length (including a missing or extra base-PCS commitment or proof), a residue that is not
/// below $q$, or an attempt index at or beyond the cap [`hachi_co_r2l::MAX_ATTEMPTS`]
/// ([`EvalProof::validate`](crate::EvalProof::validate)). The words of the base PCS are judged
/// by the sumcheck verifier, with the base PCS.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ProofError {
    /// A component of the proof has the wrong length.
    #[error("malformed proof: {what} has {actual} entries, expected {expected}")]
    Shape {
        /// The component.
        what: &'static str,
        /// The length the context requires.
        expected: usize,
        /// The length that was found.
        actual: usize,
    },
    /// A residue of the proof is not below $q$.
    #[error("malformed proof: {what} has the residue {value}, which is not below q = {q}")]
    NonCanonical {
        /// The component.
        what: &'static str,
        /// The residue.
        value: u64,
        /// The modulus.
        q: u64,
    },
    /// The index $a^\ast$ of the accepted attempt is at or beyond the cap on the attempts of the
    /// ring-to-linear step ([`hachi_co_r2l::MAX_ATTEMPTS`]); honest provers never produce it.
    #[error("malformed proof: attempt index {attempt} is not below the cap {cap}")]
    AttemptOutOfRange {
        /// The index in the proof.
        attempt: u64,
        /// The cap.
        cap: u64,
    },
}

/// Why [`EvalProof::from_bytes`](crate::EvalProof::from_bytes) refused an encoding. The length
/// of a valid encoding is fixed by the context and the lengths of the base-PCS items, which come
/// first ([`crate::encoded_len`]), so every length mismatch is caught before the rest of the
/// body is read.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum DecodeError {
    /// Fewer bytes than the encoding of a proof for this context has.
    #[error("truncated encoding: {actual} bytes, expected {expected}")]
    Truncated {
        /// The length of the encoding for the context.
        expected: usize,
        /// The number of bytes given.
        actual: usize,
    },
    /// More bytes than the encoding of a proof for this context has.
    #[error("trailing bytes: {actual} bytes, expected {expected}")]
    TrailingBytes {
        /// The length of the encoding for the context.
        expected: usize,
        /// The number of bytes given.
        actual: usize,
    },
    /// The first 8 bytes are not [`MAGIC`](crate::MAGIC).
    #[error("the encoding does not start with the magic bytes of an Eval proof")]
    BadMagic,
    /// The version word is not [`VERSION`](crate::VERSION).
    #[error("unsupported encoding version {0}")]
    UnsupportedVersion(u64),
    /// The body decodes to a proof that does not fit the context (a residue not below $q$, or
    /// an attempt index at or beyond the cap).
    #[error(transparent)]
    Invalid(#[from] ProofError),
    /// The length of the encoding does not fit in `usize`: for layouts far beyond any parameter
    /// set, or for lengths of base-PCS items in the encoding that add up to more than that.
    #[error("the length of the encoding overflows usize")]
    Overflow,
}

/// What can go wrong on the provers' side: a statement that does not fit the context, a claimed
/// evaluation that differs from the one the provers computed, and the errors of the building
/// blocks (randomness, challenger and network included).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An error of the foundation (`hachi-co-core`), e.g. of [`hachi_co_core::Context::setup`].
    #[error(transparent)]
    Core(#[from] hachi_co_core::Error),
    /// An error of Commit (`hachi-co-commit`).
    #[error(transparent)]
    Commit(#[from] hachi_co_commit::Error),
    /// An error of $\Pi\_{\mathsf{conv}}$ (`hachi-co-conv`).
    #[error(transparent)]
    Conv(#[from] hachi_co_conv::Error),
    /// An error of the ring-to-linear step (`hachi-co-r2l`), including
    /// [`TooManyAttempts`](hachi_co_r2l::Error::TooManyAttempts).
    #[error(transparent)]
    R2l(#[from] hachi_co_r2l::Error),
    /// An error of the sumcheck provers (`hachi-co-sumcheck`).
    #[error(transparent)]
    Sumcheck(#[from] hachi_co_sumcheck::Error),
    /// Hachi refused its parameters ([`HachiBase::new`](crate::HachiBase::new)).
    #[error("Hachi: {0}")]
    Hachi(#[from] hachi::Error),
    /// The statement does not fit the context.
    #[error("statement: {0}")]
    Statement(#[from] StatementError),
    /// The claimed evaluation $y$ fails the check $y = E(u)$ of $\Pi\_{\mathsf{conv}}$ with the
    /// $E(u)$ the provers computed (`field-to-ring-rok.tex`:160-164), so the verifier would
    /// reject; the provers stop before the ring-to-linear step.
    #[error("the claimed evaluation fails y = E(u): {0}")]
    ClaimMismatch(hachi_co_conv::VerifyError),
}

/// Why the verifier rejected, by step. Every malformed or tampered statement or proof gives one
/// of these values, never a panic.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum VerifyError {
    /// The statement does not fit the context.
    #[error("statement: {0}")]
    Statement(#[from] StatementError),
    /// The proof does not fit the context.
    #[error(transparent)]
    Proof(#[from] ProofError),
    /// $\Pi\_{\mathsf{conv}}$'s check failed: $E(u) \notin \mathcal R\_q^H$ or $y \ne E(u)$
    /// (`field-to-ring-rok.tex`:163-164).
    #[error("Pi_conv: {0}")]
    Conv(hachi_co_conv::VerifyError),
    /// The public vectors of $\Pi\_{\mathsf{conv}}$ or the matrix of $E$ could not be computed (not
    /// expected after the statement checks; kept as a value rather than a panic).
    #[error("public vectors: {0}")]
    PublicVectors(hachi_co_conv::Error),
    /// The ring-to-linear step: the challenges of the accepted attempt, $\mathbf c \in
    /// \mathcal C^{2^r}$, or $\mathbf M$ and $\mathbf y$ (`5_Building_Blocks.tex`:122-126,
    /// 157-179).
    #[error("ring-to-linear: {0}")]
    RingToLinear(hachi_co_r2l::Error),
    /// The sumcheck verifier rejected (`5_Building_Blocks.tex`:276-334): a failing round or final
    /// check ([`Rejected`](hachi_co_sumcheck::VerifyError::Rejected)), a base-PCS opening that
    /// the base PCS rejects ([`Opening`](hachi_co_sumcheck::VerifyError::Opening), with the
    /// party and the polynomial), or a failing challenger.
    #[error("sumcheck: {0}")]
    Sumcheck(hachi_co_sumcheck::VerifyError),
}
