//! Work package WP11 of Hachi-Co (`docs/m1-spec.md`, "WP11 `hachi-co`"): the collaborative
//! polynomial commitment scheme of the draft (`6_Collaborative_Hachi.tex`:262-419) composed from
//! WP7-WP10 with Fiat-Shamir: $\mathsf{Setup}$, $\mathsf{Commit}$, $\mathsf{Eval}$
//! ($\Pi\_{\mathsf{conv}}$, then the ring-to-linear step with retries, then the collaborative
//! sumcheck) and $\mathsf{Verify}$, the proof format with its canonical encoding, and its size.
//!
//! | Item | Contents | Draft |
//! |---|---|---|
//! | [`setup`] | $\mathsf{Setup}$: [`Context::setup`] ($\mathbf A, \mathbf B, \mathbf D$ from a seed) | `6_Collaborative_Hachi.tex`:263-298 |
//! | [`commit`] | $\mathsf{Commit}$ of one party: [`hachi_co_commit::commit`] | 300-391 |
//! | [`eval`], [`eval_with`], [`EvalOutput`] | $\mathsf{Eval}$ of one party with Fiat-Shamir | 408-419 |
//! | [`verify`], [`verify_with`] | the verifier, returning the $3n$ final claims ([`FinalClaims`]) | 408-419; `5_Building_Blocks.tex`:276-334 |
//! | [`EvalProof`], [`encoded_len`] | the proof and its canonical encoding | — |
//! | [`absorb_statement`], [`check_statement`], [`EVAL_DOMAIN`] | the statement in the transcript | — |
//! | [`hachi_co_params::sizes`](fn@hachi_co_params::sizes) | the size of the encoding, per component | — |
//!
//! The protocol code is that of the building blocks (`hachi-co-commit`, `-conv`, `-r2l`,
//! `-sumcheck`); this crate chains them, owns the Fiat-Shamir transcript of a proof, and
//! defines what the provers send. With [`hachi_co_mpc::Loopback`] it runs one prover; the party
//! code is that of $n$ provers, each calling [`commit`] and [`eval`] with its own network end.
//!
//! # What an accepted proof means in milestone 1
//!
//! **The base PCS is absent.** The draft commits to the blinders $g\_{0,p}, g\_{1,p}$ and to the
//! share polynomials $\langle \widetilde w \rangle\_p$ with a base PCS before the sumcheck's coins,
//! and opens them at the point $\mathbf a$ at the end (`6_Collaborative_Hachi.tex`:289-292;
//! `5_Building_Blocks.tex`:220-230, 314-326). Milestone 1 has neither the commitments nor the
//! openings (`ref/README.md`, choice 13; milestone 3 adds Hachi as the base PCS). [`verify`]
//! therefore returns the $3n$ final claims $g\_{0,p}(\mathbf a), g\_{1,p}(\mathbf a),
//! \langle \widetilde w \rangle\_p(\mathbf a)$ **unproven**, and an accepted M1 proof is sound only
//! together with them. On its own it proves nothing: a prover can make one for any statement.
//! It sets $E(u)$ to $y$ embedded in $\mathcal R\_q^H$ (which passes $\Pi\_{\mathsf{conv}}$'s
//! check), sends any round polynomials whose values at 0 and 1 add up as the round checks
//! require, and, since the final values are absorbed after the last coin and constrained only
//! by the two final checks, chooses $\langle \widetilde w \rangle\_p(\mathbf a)$ freely and solves
//! these checks for $g\_{0,p}(\mathbf a)$ and $g\_{1,p}(\mathbf a)$, whose coefficient is the coin
//! $\rho$ (nonzero except with probability $q^{-k}$); `tests/forgery.rs` does exactly this for a
//! false evaluation at toy-n1 and the verifier accepts. The commitments to these polynomials
//! before the coins and the openings of the claims are what rule this out. The tests check the
//! claims of honest proofs against the provers' polynomials, as an ideal PCS would. The commitment $\mathbf u$ is not
//! hiding either (plan `docs/hachi-co-plan.md`, finding F2), and the draft's security proofs are
//! unfinished.
//!
//! # Fiat-Shamir
//!
//! One [`hachi_co_core::FsChallenger`] per proof, on a fresh transcript with the domain
//! [`EVAL_DOMAIN`] = `hachi-co/eval/v1`. The provers ([`eval_with`]) and the verifier
//! ([`verify_with`]) make exactly the same challenger calls in the same order, except that the
//! verifier replays only the accepted attempt of the ring-to-linear step:
//!
//! | # | Call | Label | Data |
//! |---|---|---|---|
//! | 1 | `absorb_u64` × 15 | `hachi-co/eval/params/<name>` ([`LABELS_PARAMS`]) | $\lambda, n, t, \ell, \alpha, \kappa, b, \omega, $ challenge set, $\tau$ rule, $n\_A, n\_B, n\_D, m\_\rho, q$ ([`statement_words`]) |
//! | 2 | `absorb_polys` × 3 | [`LABEL_PP_A`], [`LABEL_PP_B`], [`LABEL_PP_D`] | $\mathbf A$, $\mathbf B$, $\mathbf D$, row-major |
//! | 3 | `absorb_polys` | [`hachi_co_commit::LABEL_U`] | $\mathbf u$ |
//! | 4 | `absorb_exts` × 2 | [`LABEL_X`], [`LABEL_Y`] | $\mathbf x$; $y$ |
//! | 5 | `absorb_polys` | [`hachi_co_conv::LABEL_E_U`] | $E(u)$ |
//! | 6 | per attempt $a$ (provers: $0, \dots, a^\ast$; verifier: $a^\ast$): `begin_attempt(a)`, `absorb_polys`, `r2l_challenges(a, 2^r, d, ω)` | [`hachi_co_r2l::LABEL_V`] | $\mathbf v$ of attempt $a$ |
//! | 7 | `absorb_exts` × 2, `sumcheck_coins` | [`hachi_co_sumcheck::LABEL_G0`], [`hachi_co_sumcheck::LABEL_G1`] | $G\_{0,1..n}$; $G\_{1,1..n}$ |
//! | 8 | per round $r = 1, \dots, \ell''$: `absorb_exts` × 2, `round_coin(r)` | [`hachi_co_sumcheck::LABEL_H0`], [`hachi_co_sumcheck::LABEL_H1`] | $H\_{0,r}(1..d\_0+1)$; $H\_{1,r}(1..d\_1+1)$ |
//! | 9 | `absorb_exts` × 3 | [`hachi_co_sumcheck::LABEL_FINAL_W`], `..._G0`, `..._G1` | the final values |
//!
//! Rows 1-4 are [`absorb_statement`]; row 6 is made by [`hachi_co_r2l::run`] and
//! [`hachi_co_r2l::verifier_challenges`] (with [`hachi_co_core::Challenger::begin_attempt`], the
//! challenges of attempt $a$ depend on the statement, $E(u)$, $a$ and that attempt's $\mathbf v$
//! only); rows 7-9 by [`hachi_co_sumcheck::prove`] and [`hachi_co_sumcheck::verify`]. After an
//! honest run the provers' and the verifier's transcripts end in the same state (tested).
//!
//! # The proof and its size
//!
//! [`EvalProof`] holds $E(u)$, $a^\ast$, $\mathbf v$ and the sumcheck transcript ($G\_{0,p},
//! G\_{1,p}$, the round messages, the final values), not the challenges. Its encoding
//! ([`EvalProof::to_bytes`]) is a 16-byte header ([`MAGIC`], [`VERSION`]) and little-endian
//! `u64` words; [`EvalProof::from_bytes`] accepts exactly the canonical encodings (fixed length,
//! residues below $q$, $a^\ast$ below [`hachi_co_r2l::MAX_ATTEMPTS`]). The size is
//! [`hachi_co_params::sizes`](fn@hachi_co_params::sizes); it excludes the base-PCS parts.
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
//!   [`LABEL_PP_B`], [`LABEL_PP_D`], [`LABEL_X`], [`LABEL_Y`]); $\mathbf u$ uses the label of
//!   `hachi-co-commit`.
//! - The statement's parameter words: the choices with $\omega$ and $q$ resolved and the two
//!   enumerations as small integers ([`statement_words`]); $\mathbf A, \mathbf B, \mathbf D$ are
//!   absorbed in full rather than as a digest.
//! - Statements and proofs with non-canonical residues are rejected, on both sides
//!   ([`check_statement`], [`EvalProof::validate`]).
//! - The provers check $y = E(u)$ after $\Pi\_{\mathsf{conv}}$ and refuse to prove a false claim
//!   ([`Error::ClaimMismatch`]).
//! - The proof format: the order of the components, the 64-bit words and the header
//!   ([`EvalProof::to_bytes`]); an index $a^\ast$ at or beyond the attempt cap of
//!   `hachi-co-r2l` is malformed.
//! - [`eval_with`] and [`verify_with`] take the challenger, so that the fixture tests can replay
//!   the oracle's coins; [`eval_with`] also returns the party's private data ([`EvalOutput`]).
//!
//! References: "the draft" is the Hachi-Co paper draft (`collaborative-hatchi`, commit
//! `625ced1`), cited by file and line; "the oracle" is the Sage reference in `ref/`.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod error;
mod proof;
mod prover;
mod statement;
mod verifier;

pub use error::{DecodeError, Error, ProofError, StatementError, VerifyError};
pub use proof::{EvalProof, HEADER_LEN, MAGIC, VERSION, encoded_len};
pub use prover::{EvalOutput, commit, eval, eval_with, setup};
pub use statement::{
    EVAL_DOMAIN, LABEL_PP_A, LABEL_PP_B, LABEL_PP_D, LABEL_X, LABEL_Y, LABELS_PARAMS,
    absorb_statement, check_statement, statement_words,
};
pub use verifier::{verify, verify_with};

pub use hachi_co_core::{Context, PartyState};
pub use hachi_co_sumcheck::{Claim, ClaimedPoly, FinalClaims, SumcheckTranscript};
