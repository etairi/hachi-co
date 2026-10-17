//! The error type of the crate.

use hachi_co_mpc::PartyId;

/// Everything that can go wrong in the ring-to-linear step: errors of the layers below, inputs of
/// the wrong shape, challenges outside $\mathcal C$, a failed check of prover 1, a misbehaving
/// communication layer, and the attempt cap. Errors are values (`docs/m1-spec.md`, rule 5);
/// nothing in this crate panics on malformed input, and the verifier-side functions
/// ([`linear_relation`](crate::linear_relation), [`verifier_challenges`](crate::verifier_challenges))
/// return an error for every malformed or out-of-range proof value.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An error of `hachi-co-core` (shapes of the inter-block types, $\mathbf M$ and $\mathbf y$,
    /// the randomness and the challenger, including
    /// [`RandomnessReuse`](hachi_co_core::Error::RandomnessReuse)).
    #[error(transparent)]
    Core(#[from] hachi_co_core::Error),
    /// An error of the arithmetic layer (`hachi-co-math`), for example a value that
    /// $\mathbf J^{-1}$ cannot represent.
    #[error(transparent)]
    Math(#[from] hachi_co_math::Error),
    /// An error of `hachi-co-sampling`; in particular the refusals of $\mathsf{Rej}$
    /// ([`SecretTooLarge`](hachi_co_sampling::Error::SecretTooLarge) for a fold with
    /// $\Vert \cdot \Vert\_\infty \gt \beta\_1$, outside the hypothesis of the lemma,
    /// `4_preliminaries.tex`:122-127, and
    /// [`MaskOutOfRange`](hachi_co_sampling::Error::MaskOutOfRange) for a mask outside
    /// $\[-\gamma, \gamma\]^N$).
    #[error(transparent)]
    Sampling(#[from] hachi_co_sampling::Error),
    /// An error of the communication layer (`hachi-co-mpc`).
    #[error(transparent)]
    Mpc(#[from] hachi_co_mpc::Error),
    /// An input or a received message has the wrong length.
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
    /// A coefficient of a short vector is outside its range (for example $\boldsymbol\rho\_p$
    /// outside $S\_b$, `4_preliminaries.tex`:88).
    #[error("{what}: a coefficient is outside [{min}, {max}]")]
    OutOfRange {
        /// The vector.
        what: &'static str,
        /// The smallest allowed coefficient.
        min: i64,
        /// The largest allowed coefficient.
        max: i64,
    },
    /// An exact integer computation left the range of `i64` (only for inputs far outside the
    /// protocol's bounds).
    #[error("integer overflow in {0}")]
    Overflow(&'static str),
    /// A parameter this crate cannot work with; the message names it.
    #[error("invalid parameter: {0}")]
    Parameter(String),
    /// A challenge $c\_i$ is not in $\mathcal C = \lbrace c \in \mathcal R\_q : \Vert c \Vert\_1
    /// \le \omega \rbrace$ (`5_Building_Blocks.tex`:122). Every prover checks the challenges
    /// before using them (plan `docs/hachi-co-plan.md`, finding F5), and so does the verifier.
    #[error("challenge c_{index} has l1 norm {l1_norm} > omega = {omega}")]
    ChallengeNotInC {
        /// The index $i \lt 2^r$ of the challenge.
        index: usize,
        /// $\Vert c\_i \Vert\_1$.
        l1_norm: u128,
        /// $\omega$.
        omega: u64,
    },
    /// The communication layer and the party's data disagree (number of parties, party index or
    /// modulus).
    #[error("{what}: expected {expected}, got {actual}")]
    PartyMismatch {
        /// What disagrees.
        what: &'static str,
        /// The value of the context or the state.
        expected: usize,
        /// The value of the communication layer.
        actual: usize,
    },
    /// The communication layer broke its contract (for example a gather that delivers the
    /// responses to a party other than prover 1, or an abort agreement that contradicts the
    /// party's own flag); the message names it.
    #[error("protocol: {0}")]
    Protocol(String),
    /// Prover 1's check $\Vert \mathbf z \Vert\_\infty \le \beta$ failed
    /// (`5_Building_Blocks.tex`:131), returned at prover 1. It never fails for honest provers
    /// (`5_Building_Blocks.tex`:193), so it is not retried: the step stops with this error.
    #[error("prover 1's norm check failed: ||z||_inf = {norm} > beta = {beta}")]
    NormCheck {
        /// $\Vert \mathbf z \Vert\_\infty$.
        norm: u64,
        /// $\beta = n\gamma$.
        beta: u64,
    },
    /// Returned at the provers $p \gt 1$ when prover 1 announces that its steps after the gather
    /// failed ([`Error::NormCheck`], a malformed response, or $\mathbf J^{-1}$).
    #[error("prover 1 aborted the protocol after the gather")]
    Prover1Abort,
    /// Every one of the `attempts` attempts aborted (`5_Building_Blocks.tex`:129). For honest
    /// provers each attempt aborts with probability $1 - M^{-n} \le 1 - 1/e$
    /// (`6_Collaborative_Hachi.tex`:275), so [`MAX_ATTEMPTS`](crate::MAX_ATTEMPTS) attempts all
    /// abort with probability below $2^{-42}$.
    #[error("all {attempts} attempts of the ring-to-linear step aborted")]
    TooManyAttempts {
        /// The cap that was reached.
        attempts: u64,
    },
    /// A party index outside $1, \dots, n$.
    #[error("party {party} is not in 1..={n}")]
    InvalidParty {
        /// The index that was passed.
        party: PartyId,
        /// The number of parties.
        n: usize,
    },
}

/// The result type of the crate.
pub type Result<T> = core::result::Result<T, Error>;
