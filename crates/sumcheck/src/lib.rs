//! Work package WP10 of Hachi-Co (`docs/m1-spec.md`, "WP10 `hachi-co-sumcheck`"): the
//! collaborative sumcheck of `5_Building_Blocks.tex`:195-334 and its verifier, which reduce the
//! provers' shares of the linear relation $\mathbf M \mathbf z' = \mathbf y$ over $\mathcal R\_q$
//! (with the range conditions on $\mathbf z'$ and $y = E(u)$) to $3n$ evaluation claims, which a
//! base PCS ([`BasePcs`](hachi_co_core::BasePcs)) proves: each prover commits to its share
//! polynomial and its two blinders before the coins and opens them at the end. The Sage oracle is
//! `ref/hachico_ref/sumcheck.py`; every value the oracle records in the section `sumcheck` of
//! `ref/fixtures/toy-n{1,2,3,4,8}.json` is reproduced by this crate (the oracle does not model
//! the base PCS: its ideal PCS accepts an opening exactly when the value is right).
//!
//! | Item | Contents | Draft |
//! |---|---|---|
//! | [`quotient_share`] | $\langle \mathbf r \rangle\_p = \tfrac12 (1 - X^d)(\mathbf M \langle \mathbf z' \rangle\_p - \lbrace \mathbf y \rbrace\_p) \bmod X^{2d} + 1$ | 198-200 |
//! | [`w_table`], [`row_indicator`], [`ind_rows`] | the table of $\langle \widetilde w \rangle\_p$; $e\_p$ and $\mathrm{ind}$ | 225-229, 262, 270 |
//! | [`eq_table`], [`eq_eval`], [`eq_point`], [`bind_top`], [`mle_eval`] | $\widetilde{eq}$ and multilinear extensions | 261; `4_preliminaries.tex`:53-56 |
//! | [`hypercube_sum`], [`round_poly`], [`evaluate`] | the blinders $g\_{0,p}, g\_{1,p}$ | 209-219 |
//! | [`blinder_coefficients`], [`blinder_weights`] | a blinder as a committed vector, opened as an inner product | 220-221, 316-318 |
//! | [`mstar_table`], [`mstar_eval_factorized`], [`claimed_sum`] | $M^\ast = \sum\_{i'} \widetilde{eq}(\tau\_1, \cdot) \widetilde M\_\alpha$ and $a$ | 254-264 |
//! | [`RangeProduct`], [`range_product`] | $\prod\_{c=-b+1}^{b-1} (W - c)$ | 269 |
//! | [`PartyProver`] | party $p$'s rounds: $H\_{0,p,r}(t)$, $H\_{1,p,r}(t)$, the $\mathcal F\_{\mathsf{AH}}(\mathbb I)$ inputs | 267-300 |
//! | [`prove`] | the driver on [`Comm`](hachi_co_mpc::Comm), [`PartyRandomness`](hachi_co_core::PartyRandomness), [`Challenger`](hachi_co_core::Challenger) and a [`BasePcs`](hachi_co_core::BasePcs) | 195-326 |
//! | [`verify`], [`verify_claims`], [`verify_checks`], [`check_names`] | the verifier, with the base-PCS openings, and the oracle's named checks | 276-334 |
//! | [`SumcheckTranscript`], [`PcsWords`], [`SumcheckPrivate`], [`FinalClaims`] | the public messages (base-PCS commitments and proofs as words), a party's private data, the checked claims | 220-241, 300, 314-326 |
//! | [`opening_label`] | the labels of the contexts of the openings | 314-326 |
//! | [`Nodes`], [`degree_at_nodes`] | interpolation at $1, \dots, D + 1$; the degree audit | 290-301, 275 |
//!
//! Line numbers without a file are those of `5_Building_Blocks.tex` in the draft
//! (`collaborative-hatchi`, commit `625ced1`).
//!
//! **Conventions** (`ref/README.md`, "Conventions", choices 7-14, and the fixture's
//! `conventions`): positions are 0-indexed and parties numbered $1, \dots, n$; the hypercube is
//! $V = (U, L)$ with $\lvert U \rvert = \ell\_U = \lceil \log(\mu + n') \rceil$, $\lvert L \rvert =
//! \log 2d$, and the point $(u\_0, l\_0)$ has index $u\_0 \cdot 2d + l\_0$ with $V\_1$ the most
//! significant bit; round $r$ binds $V\_r$; $\widetilde{eq}(\tau\_0, (U, L, \mathsf{bit}(p)))$ pairs
//! $\tau\_0\[0..\ell'')$ with $(U, L)$ and $\tau\_0\[\ell''..)$ with the bits of $p - 1$; the round
//! messages are the values at $t = 1, \dots, D + 1$; $\mathbb F\_{q^k} = \mathcal R\_q^H$ in Hachi's
//! basis (`hachi_co_math::Fqk`).
//!
//! **Choices where the draft and the oracle are silent** (each is documented at its item):
//! - the transcript holds the per-party $G\_{0,p}, G\_{1,p}$, base-PCS commitments, final values
//!   and opening proofs, which every party learns through one
//!   [`Comm::all_gather`](hachi_co_mpc::Comm::all_gather) call each ([`prove`]);
//! - the absorb labels ([`LABEL_G0`], [`LABEL_COMMIT_W`], …), the absorption of the
//!   commitments after $G\_{0,\cdot}, G\_{1,\cdot}$ and of the final values after the last coin,
//!   and the contexts of the openings, derived after the final values with [`opening_label`]
//!   ([`prove`], [`verify_checks`]);
//! - the order of the openings ($g\_0$, $g\_1$, $\widetilde w$ per party, as the oracle's checks)
//!   and of the verdict: a failing round or final check first ([`VerifyError::Rejected`]), then
//!   the first failing opening ([`VerifyError::Opening`]) ([`VerifierReport::verdict`]);
//! - the provers compute the round polynomials of $F\_{0,p}$ at $2b + 2$ nodes and of $F\_{1,p}$
//!   at 3 nodes and extend them exactly to $d\_0 + 1$ resp. $d\_1 + 1$ nodes ([`PartyProver`]);
//! - the provers' inner loops run in a Montgomery representation when $q$ admits it (an
//!   implementation detail with identical values, [`PartyProver`]);
//! - [`PartyProver::new`] refuses $d\_0 \lt 2b + 1$ or $d\_1 \lt 2$, for which the round messages
//!   could not carry the round polynomials;
//! - the verifier is split into [`verify_checks`] (every named check, openings included),
//!   [`verify_claims`] (the checked claims, or the error) and [`verify`] (accept or the error).
//!
//! References: "the draft" is the Hachi-Co paper draft, cited by file and line; "the oracle" is
//! the Sage reference in `ref/`.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod arith;
mod blinder;
mod driver;
mod eq;
mod error;
mod interp;
mod prover;
mod quotient;
mod tables;
mod transcript;
mod verifier;

pub use blinder::{blinder_coefficients, blinder_weights, evaluate, hypercube_sum, round_poly};
pub use driver::{
    ALL_GATHER_COMMITMENTS, ALL_GATHER_FINAL, ALL_GATHER_OPENINGS, ALL_GATHER_SUMS, FAH_ROUND,
    LABEL_COMMIT_G0, LABEL_COMMIT_G1, LABEL_COMMIT_W, LABEL_FINAL_G0, LABEL_FINAL_G1,
    LABEL_FINAL_W, LABEL_G0, LABEL_G1, LABEL_H0, LABEL_H1, opening_label, prove,
};
pub use eq::{bind_top, eq_eval, eq_point, eq_table, mle_eval, msb_bits};
pub use error::{Error, Result, VerifyError};
pub use interp::{Nodes, degree_at_nodes};
pub use prover::PartyProver;
pub use quotient::quotient_share;
pub use tables::{
    RangeProduct, claimed_sum, ind_rows, mstar_table, powers, range_product, row_indicator,
    table_size, w_table,
};
pub use transcript::{
    Claim, ClaimedPoly, FinalClaims, FinalValues, PcsWords, RoundMessage, RoundValues,
    SumcheckPrivate, SumcheckTranscript,
};
pub use verifier::{
    Check, FINAL_H0, FINAL_H1, OpeningCheck, PublicInput, VerifierReport, check_names,
    mstar_eval_factorized, round_check_name, verify, verify_checks, verify_claims,
};
