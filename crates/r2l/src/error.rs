//! The error type of the crate.

use hachi_co_mpc::PartyId;

/// Everything that can go wrong in the ring-to-linear step: errors of the layers below, inputs of
/// the wrong shape, challenges outside $\mathcal C$, malformed sets of accepted iterations, no
/// iteration accepted by every prover, a failed check of prover 1 and a misbehaving
/// communication layer. Errors are values (`docs/m1-spec.md`, rule 5);
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
    /// A coefficient of a short vector is outside its range (for example $\boldsymbol\rho^v\_p$
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
    /// \le \omega \rbrace$ (`5_Building_Blocks.tex`:134). Every prover checks the challenges
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
    /// The communication layer broke its contract (for example a broadcast that returns the
    /// party's own set of accepted iterations changed); the message names it.
    #[error("protocol: {0}")]
    Protocol(String),
    /// The check $\Vert \mathbf z \Vert\_\infty \le \beta$ failed, so the protocol aborts
    /// (`fig:ring-to-linear-2`, step 9: "The provers check that $\Vert \mathbf z \Vert\_\infty
    /// \le \beta$. \[...\] If not, they abort the protocol", since the draft's commit `0337dfb`,
    /// before which only prover 1 checked, `5_Building_Blocks.tex`:144). Every prover runs the check
    /// on the same $\mathbf z$, so every prover returns this error. It never fails for honest
    /// provers (`5_Building_Blocks.tex`:207).
    #[error("the norm check failed: ||z||_inf = {norm} > beta = {beta}")]
    NormCheck {
        /// $\Vert \mathbf z \Vert\_\infty$.
        norm: u64,
        /// $\beta = n\gamma$.
        beta: u64,
    },
    /// No iteration is in every prover's set $R\_p$ of accepted iterations, so the protocol
    /// aborts (`fig:ring-to-linear`, step 6, since the draft's commit `1096ea2`: "If
    /// $\bigcap\_{i \in \[n\]} R\_i$ is empty, then $\mathcal V$ outputs $\bot$"). For honest
    /// provers every iteration is accepted by all of them with probability $M^{-n} \ge 1/e$,
    /// independently, so this happens with probability $(1 - M^{-n})^L \le (1 - 1/e)^L$.
    #[error("no iteration of the {iterations} is accepted by every prover")]
    NoCommonIteration {
        /// $L$.
        iterations: usize,
    },
    /// A set of accepted iterations (a broadcast message, or a proof value) is malformed: it does
    /// not have $\lceil L/64 \rceil$ words, or it names an iteration at or beyond $L$.
    #[error("the set of accepted iterations of party {party} is malformed: {what}")]
    AcceptSet {
        /// The party whose set it is.
        party: PartyId,
        /// What is wrong.
        what: &'static str,
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
