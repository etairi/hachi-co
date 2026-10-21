//! The collaborative polynomial commitment scheme of the Hachi-Co draft
//! (`6_Collaborative_Hachi.tex`:262-419), composed from the building blocks with Fiat-Shamir:
//! $\mathsf{Setup}$, $\mathsf{Commit}$, $\mathsf{Eval}$ ($\Pi\_{\mathsf{conv}}$, then the
//! ring-to-linear step with retries, then the collaborative sumcheck with the base PCS's
//! commitments and openings) and $\mathsf{Verify}$, with Hachi as the base PCS
//! ([`HachiBase`]), the proof format with its canonical encoding, and its size.
//!
//! | Item | Contents | Draft |
//! |---|---|---|
//! | [`setup`] | $\mathsf{Setup}$: [`Context::setup`] ($\mathbf A, \mathbf B, \mathbf D$ from a seed) | `6_Collaborative_Hachi.tex`:263-298 |
//! | [`commit`] | $\mathsf{Commit}$ of one party: [`hachi_co_commit::commit`] | 300-391 |
//! | [`eval`], [`eval_with`], [`EvalOutput`] | $\mathsf{Eval}$ of one party with Fiat-Shamir and a base PCS ([`BasePcs`]) | 408-419 |
//! | [`verify`], [`verify_with`], [`verify_claims_with`] | the verifier, with the $3n$ openings of the base PCS | 408-419; `5_Building_Blocks.tex`:276-334 |
//! | [`EvalProof`], [`encoded_len`], [`EvalProof::encoded_len`] | the proof, its canonical encoding and its length | — |
//! | [`absorb_statement`], [`check_statement`], [`EVAL_DOMAIN`] | the statement in the transcript | — |
//! | [`hachi_co_params::sizes_with_base`], [`hachi_co_params::sizes`](fn@hachi_co_params::sizes) | the size of the encoding, per component, with and without the base PCS's items | — |
//! | [`HachiBase`], [`HachiBase::setup`] | Hachi as the base PCS, and its $\mathsf{pp}\_{\mathsf{PCS}}$ from the setup seed | 289-292 |
//!
//! The protocol code is that of the building blocks (`hachi-co-commit`, `-conv`, `-r2l`,
//! `-sumcheck`); this crate chains them, owns the Fiat-Shamir transcript of a proof, and
//! defines what the provers send. With [`hachi_co_mpc::Loopback`] it runs one prover; the party
//! code is that of $n$ provers, each calling [`commit`] and [`eval`] with its own network end.
//! The base PCS is a parameter of [`eval`] and [`verify`] (`P: BasePcs<K>`), the same at every
//! party and at the verifier: Hachi ([`HachiBase`]), or `ClearPcs` of `hachi-co-core` (feature
//! `fixtures`), a reference stand-in that the fixture tests use.
//!
//! # What an accepted proof means
//!
//! **What the base PCS adds.** Each prover commits with the base PCS to its share
//! polynomial $\langle \widetilde w \rangle\_p$ and its blinders $g\_{0,p}, g\_{1,p}$ before the
//! sumcheck's coins: the commitments are absorbed into the Fiat-Shamir transcript after
//! $G\_{0,\cdot}, G\_{1,\cdot}$ and before $\alpha, \rho, \tau\_0, \tau\_1$ are drawn
//! (`5_Building_Blocks.tex`:209-241). After the last round, the verifier checks the $3n$
//! openings of these commitments at the point $\mathbf a$ against the final values that its
//! final checks use (lines 314-333); every check must hold. The final values are therefore
//! evaluations of polynomials fixed before the coins, which is what the sumcheck's soundness
//! argument needs; the forgery that the protocol without a base PCS allowed (choose
//! $\langle \widetilde w \rangle\_p(\mathbf a)$ freely after the last coin and solve the two
//! final checks for $g\_{0,p}(\mathbf a)$, $g\_{1,p}(\mathbf a)$) is rejected, and
//! `tests/forgery.rs` checks it in each of its variants, with `ClearPcs` and with Hachi.
//!
//! What this does not say:
//! - **Proof status.** Soundness of the composition is the draft's claim, not a theorem here:
//!   the draft's knowledge-soundness proof is unwritten (`6_Collaborative_Hachi.tex`:588), and
//!   that Hachi's knowledge soundness with relaxed openings suffices for the extractor is the
//!   authors' intuition (answer 4 of 2026-10-21, plan `docs/hachi-co-plan.md`, §0). The tests
//!   check completeness, every check of the verifier, the rejection of every tampered
//!   component and of the known forgery; they do not prove soundness.
//! - **The base PCS.** The guarantee is only as good as the base PCS passed to [`verify`]:
//!   `ClearPcs` is binding (SHA-256 collision resistance) and sound, but neither succinct nor
//!   hiding. [`HachiBase`] is not hiding either (each opening sends its last witness in the
//!   clear), and [`HachiBase::setup`] uses insecure test parameters.
//! - **Fiat-Shamir** is in the random-oracle model (a SHAKE128 transcript,
//!   `hachi-co-transcript`), as accepted for the project (user, 2026-10-17). The digest of the
//!   base PCS's parameters ([`BasePcs::params_digest`]) is part of the statement
//!   ([`absorb_statement`]), and the base PCS's commitments are absorbed before the sumcheck's
//!   coins.
//! - **Not hiding.** The commitment $\mathbf u$ is not hiding (plan, finding F2); the final
//!   values reveal one evaluation of each party's share polynomial (finding F4; revealed for now
//!   by the authors' answer 7 of 2026-10-21); the base PCS's commitments and openings reveal
//!   whatever the base PCS reveals (`ClearPcs`: the polynomials themselves).
//!
//! # Fiat-Shamir
//!
//! One [`hachi_co_core::FsChallenger`] per proof, on a fresh transcript with the domain
//! [`EVAL_DOMAIN`] = `hachi-co/eval/v1`. The provers ([`eval_with`]) and the verifier
//! ([`verify_with`]) make exactly the same challenger calls in the same order, except that the
//! verifier replays only the accepted attempt of the ring-to-linear step and derives the
//! contexts of every party's openings, each prover only its own:
//!
//! | # | Call | Label | Data |
//! |---|---|---|---|
//! | 1 | `absorb_u64` × 15 | `hachi-co/eval/params/<name>` ([`LABELS_PARAMS`]) | $\lambda, n, t, \ell, \alpha, \kappa, b, \omega, $ challenge set, $\tau$ rule, $n\_A, n\_B, n\_D, m\_\rho, q$ ([`statement_words`]) |
//! | 2 | `absorb_polys` × 3 | [`LABEL_PP_A`], [`LABEL_PP_B`], [`LABEL_PP_D`] | $\mathbf A$, $\mathbf B$, $\mathbf D$, row-major |
//! | 3 | `absorb_words` | [`LABEL_PP_BASE`] | the digest of the base PCS's parameters ([`BasePcs::params_digest`]), as four little-endian words |
//! | 4 | `absorb_polys` | [`hachi_co_commit::LABEL_U`] | $\mathbf u$ |
//! | 5 | `absorb_exts` × 2 | [`LABEL_X`], [`LABEL_Y`] | $\mathbf x$; $y$ |
//! | 6 | `absorb_polys` | [`hachi_co_conv::LABEL_E_U`] | $E(u)$ |
//! | 7 | per attempt $a$ (provers: $0, \dots, a^\ast$; verifier: $a^\ast$): `begin_attempt(a)`, `absorb_polys`, `r2l_challenges(a, 2^r, d, ω)` | [`hachi_co_r2l::LABEL_V`] | $\mathbf v$ of attempt $a$ |
//! | 8 | `absorb_exts` × 2, `absorb_words` × 3, `sumcheck_coins` | [`hachi_co_sumcheck::LABEL_G0`], [`hachi_co_sumcheck::LABEL_G1`], [`hachi_co_sumcheck::LABEL_COMMIT_W`], `..._G0`, `..._G1` | $G\_{0,1..n}$; $G\_{1,1..n}$; the base-PCS commitments to $\langle \widetilde w \rangle\_p$, $g\_{0,p}$, $g\_{1,p}$, one word vector per party each |
//! | 9 | per round $r = 1, \dots, \ell''$: `absorb_exts` × 2, `round_coin(r)` | [`hachi_co_sumcheck::LABEL_H0`], [`hachi_co_sumcheck::LABEL_H1`] | $H\_{0,r}(1..d\_0+1)$; $H\_{1,r}(1..d\_1+1)$ |
//! | 10 | `absorb_exts` × 3 | [`hachi_co_sumcheck::LABEL_FINAL_W`], `..._G0`, `..._G1` | the final values |
//! | 11 | `context` per opening (each prover: its 3; the verifier: all $3n$), party by party in the order $g\_0, g\_1, \widetilde w$ | [`hachi_co_sumcheck::opening_label`] | — |
//!
//! Rows 1-5 are [`absorb_statement`]; row 7 is made by [`hachi_co_r2l::run`] and
//! [`hachi_co_r2l::verifier_challenges`] (with [`hachi_co_core::Challenger::begin_attempt`], the
//! challenges of attempt $a$ depend on the statement, $E(u)$, $a$ and that attempt's $\mathbf v$
//! only); rows 8-11 by [`hachi_co_sumcheck::prove`] and [`hachi_co_sumcheck::verify`]. A context
//! is squeezed from a fork of the transcript ([`hachi_co_core::FsChallenger`]) and leaves the
//! transcript unchanged, so after an honest run the provers' and the verifier's transcripts end
//! in the same state (tested). The opening proofs are not absorbed: no coin follows them.
//!
//! # The proof and its size
//!
//! [`EvalProof`] holds $E(u)$, $a^\ast$, $\mathbf v$ and the sumcheck transcript ($G\_{0,p},
//! G\_{1,p}$, the base-PCS commitments, the round messages, the final values, the base-PCS
//! opening proofs), not the challenges. Its encoding ([`EvalProof::to_bytes`], format
//! [`VERSION`] 2) is a 16-byte header ([`MAGIC`], [`VERSION`]) and little-endian `u64` words,
//! with the lengths of the $6n$ base-PCS items first, so that [`EvalProof::from_bytes`] knows
//! the exact length before reading the rest; it accepts exactly the canonical encodings (exact
//! length, residues below $q$, $a^\ast$ below [`hachi_co_r2l::MAX_ATTEMPTS`]; the base PCS's
//! words are judged by the verifier). The size is [`hachi_co_params::sizes_with_base`] for the
//! word counts of the base PCS's items ([`encoded_len`] computes it from the layout);
//! [`hachi_co_params::sizes`](fn@hachi_co_params::sizes) is the Hachi-Co part alone.
//!
//! # Errors
//!
//! Errors are values ([`Error`], [`VerifyError`], [`DecodeError`]); the verifier and the decoder
//! never panic on statements, proofs or encodings (tested with tampered proofs and a seeded fuzz
//! loop).
//!
//! # Choices where the draft and the oracle are silent
//!
//! Each is documented at its item.
//! - The domain [`EVAL_DOMAIN`] and the labels of this crate ([`LABELS_PARAMS`], [`LABEL_PP_A`],
//!   [`LABEL_PP_B`], [`LABEL_PP_D`], [`LABEL_PP_BASE`], [`LABEL_X`], [`LABEL_Y`]); $\mathbf u$
//!   uses the label of `hachi-co-commit`.
//! - The base PCS enters the statement through its parameter digest, not its parameters
//!   ([`BasePcs::params_digest`]); Hachi's digest is a transcript hash of its choices
//!   ([`HACHI_BASE_DIGEST_DOMAIN`]).
//! - Hachi's public matrices come from a seed derived from the setup seed
//!   ([`base_pcs_seed`], domain [`SETUP_DOMAIN_BASE_PCS`]), and its default parameters are its
//!   toy choices at Hachi-Co's ring degree ([`HachiBase::default_choices`]).
//! - The statement's parameter words: the choices with $\omega$ and $q$ resolved and the two
//!   enumerations as small integers ([`statement_words`]); $\mathbf A, \mathbf B, \mathbf D$ are
//!   absorbed in full rather than as a digest.
//! - Statements and proofs with non-canonical residues are rejected, on both sides
//!   ([`check_statement`], [`EvalProof::validate`]).
//! - The provers check $y = E(u)$ after $\Pi\_{\mathsf{conv}}$ and refuse to prove a false claim
//!   ([`Error::ClaimMismatch`]).
//! - The proof format: the order of the components, the 64-bit words, the header and the table
//!   of the base-PCS lengths ([`EvalProof::to_bytes`]); an index $a^\ast$ at or beyond the
//!   attempt cap of `hachi-co-r2l` is malformed.
//! - [`eval_with`] and [`verify_with`] take the challenger, so that the fixture tests can replay
//!   the oracle's coins; [`eval_with`] also returns the party's private data ([`EvalOutput`]);
//!   [`verify_claims_with`] also returns the checked claims.
//!
//! References: "the draft" is the Hachi-Co paper draft (`collaborative-hatchi`, commit
//! `625ced1`), cited by file and line; "the oracle" is the Sage reference in `ref/`.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod base;
mod error;
mod proof;
mod prover;
mod statement;
mod verifier;

pub use base::{HACHI_BASE_DIGEST_DOMAIN, HachiBase, SETUP_DOMAIN_BASE_PCS, base_pcs_seed};
pub use error::{DecodeError, Error, ProofError, StatementError, VerifyError};
pub use proof::{EvalProof, HEADER_LEN, MAGIC, VERSION, encoded_len};
pub use prover::{EvalOutput, commit, eval, eval_with, setup};
pub use statement::{
    EVAL_DOMAIN, LABEL_PP_A, LABEL_PP_B, LABEL_PP_BASE, LABEL_PP_D, LABEL_X, LABEL_Y,
    LABELS_PARAMS, absorb_statement, check_statement, statement_words,
};
pub use verifier::{verify, verify_claims_with, verify_with};

pub use hachi::HachiChoices;
pub use hachi_co_core::{BasePcs, Context, PartyState};
pub use hachi_co_params::BasePcsWords;
pub use hachi_co_sumcheck::{Claim, ClaimedPoly, FinalClaims, PcsWords, SumcheckTranscript};
