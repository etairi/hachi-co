//! The error types of the crate: [`Error`] on the provers' side and in the table builders, and
//! [`VerifyError`] for the verifier.

use crate::transcript::ClaimedPoly;
use hachi_co_mpc::PartyId;

/// What can go wrong on the provers' side and in the public table builders: inputs of the wrong
/// shape, parameters the sumcheck cannot run with, a [`PartyProver`](crate::PartyProver) used out
/// of order, and failures of the layers below (`hachi-co-core`: randomness and challengers;
/// `hachi-co-math`; `hachi-co-mpc`). Errors are values (`docs/m1-spec.md`, rule 5).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An error of `hachi-co-core`: a shape check of an inter-block type, the randomness source
    /// or the challenger.
    #[error(transparent)]
    Core(#[from] hachi_co_core::Error),
    /// An error of the arithmetic layer (`hachi-co-math`), e.g. a polynomial of the wrong length
    /// passed to an exact product.
    #[error(transparent)]
    Math(#[from] hachi_co_math::Error),
    /// An error of the communication layer (`hachi-co-mpc`).
    #[error(transparent)]
    Mpc(#[from] hachi_co_mpc::Error),
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
    /// A parameter set the sumcheck cannot run with (for example $d\_0 \lt 2b + 1$, a modulus
    /// $q \le d\_0 + 1$, or a table of more than `usize::MAX` entries); the message names it.
    #[error("invalid parameter: {0}")]
    Parameter(String),
    /// A [`PartyProver`](crate::PartyProver) method was called out of order (for example a round
    /// before [`PartyProver::start`](crate::PartyProver::start)); the message names it.
    #[error("prover state: {0}")]
    State(String),
    /// A party index outside $1, \dots, n$.
    #[error("party {party} is not in 1..={n}")]
    InvalidParty {
        /// The index that was passed.
        party: PartyId,
        /// The number of parties.
        n: usize,
    },
    /// The value that the base PCS opened differs from the party's final value of the sumcheck
    /// (`5_Building_Blocks.tex`:314-318): a fault of the base PCS or of the conversion of the
    /// polynomial before committing. An honest run never gives it; the provers stop rather
    /// than send an opening the verifier would reject.
    #[error(
        "the base PCS opened {} of party {party} to another value than the sumcheck's",
        poly.name()
    )]
    OpeningMismatch {
        /// The party.
        party: PartyId,
        /// The polynomial.
        poly: ClaimedPoly,
    },
}

/// The result type of the provers' side.
pub type Result<T> = core::result::Result<T, Error>;

/// Why [`verify`](crate::verify) rejected: a malformed transcript or public input (wrong
/// lengths, non-canonical residues), parameters it cannot verify with, a failing challenger,
/// failing checks of the sumcheck, or a base-PCS opening that does not verify. The verifier
/// never panics on proof data (`docs/m1-spec.md`, rule 5): every malformed message is one of
/// these values.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum VerifyError {
    /// A message of the transcript or a public input has the wrong length.
    #[error("malformed {what}: expected {expected} entries, got {actual}")]
    Shape {
        /// What has the wrong shape.
        what: &'static str,
        /// The required length.
        expected: usize,
        /// The length that was found.
        actual: usize,
    },
    /// A residue of the transcript or of a public input is not canonical (not in $\[0, q)$).
    #[error("{what}: residue {value} is not below q = {q}")]
    NonCanonical {
        /// Where the residue was found.
        what: &'static str,
        /// The residue.
        value: u64,
        /// The modulus.
        q: u64,
    },
    /// Parameters the verifier cannot work with (e.g. $q \le d\_0 + 1$, so that the round
    /// messages cannot be interpolated); the message names it.
    #[error("invalid parameter: {0}")]
    Parameter(String),
    /// The challenger failed to produce the coins.
    #[error("challenger: {0}")]
    Challenger(hachi_co_core::Error),
    /// The transcript is well formed but some round or final checks of the sumcheck failed
    /// (`5_Building_Blocks.tex`:301-311, 327-333); their names are those of the oracle
    /// (`round_01_H0`, …, `final_H1`; see [`check_names`](crate::check_names)). The verifier
    /// reports these before the openings ([`VerifyError::Opening`]).
    #[error("rejected: the checks {failing:?} failed")]
    Rejected {
        /// The names of the failing round and final checks, in the order of the oracle.
        failing: Vec<String>,
    },
    /// The round and final checks hold, but the base PCS rejected the opening of the polynomial
    /// `poly` of party `party` (`5_Building_Blocks.tex`:321-326; the oracle's check
    /// `pcs_open_<poly>_p<party>`): its commitment or its proof does not decode, or the proof
    /// does not open the commitment to the final value at the point $\mathbf a$. The first
    /// failing opening is reported, in the order of [`check_names`](crate::check_names).
    #[error(
        "the base-PCS opening of {} of party {party} was rejected: {reason}",
        poly.name()
    )]
    Opening {
        /// The party $p \in \lbrace 1, \dots, n \rbrace$ whose opening failed.
        party: PartyId,
        /// The polynomial.
        poly: ClaimedPoly,
        /// The reason the base PCS gave.
        reason: String,
    },
}

impl From<Error> for VerifyError {
    /// The verifier reuses the table builders of the crate; their shape and parameter errors
    /// keep their meaning, every other error (which the builders do not produce on validated
    /// inputs) is reported as a parameter error with its message.
    fn from(e: Error) -> Self {
        match e {
            Error::Shape {
                what,
                expected,
                actual,
            } => Self::Shape {
                what,
                expected,
                actual,
            },
            Error::Core(c) => Self::Challenger(c),
            other => Self::Parameter(other.to_string()),
        }
    }
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

/// $2^e$ as a `usize`, or a parameter error naming `what`.
pub(crate) fn pow2(e: usize, what: &str) -> Result<usize> {
    u32::try_from(e)
        .ok()
        .and_then(|e| 1usize.checked_shl(e))
        .ok_or_else(|| Error::Parameter(format!("2^{e} ({what}) does not fit in usize")))
}
