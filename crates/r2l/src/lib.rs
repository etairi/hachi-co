//! Work package WP9 of Hachi-Co (`docs/m1-spec.md`, "WP9 `hachi-co-r2l`"): the conversion of ring
//! evaluation claims to linear relations over $\mathcal R\_q$ (`5_Building_Blocks.tex`:98-193;
//! Sage oracle `ref/hachico_ref/r2l.py`). After $\Pi\_{\mathsf{conv}}$, the provers hold the
//! blocks $\mathbf s\_i$ of $\mathbf f'$ and shares $\langle u \rangle\_p$; this step ends with a
//! public matrix $\mathbf M$ and shares $\langle \mathbf z' \rangle\_p$, $\lbrace \mathbf y
//! \rbrace\_p$ with $\mathbf M \sum\_p \langle \mathbf z' \rangle\_p = \sum\_p \lbrace \mathbf y
//! \rbrace\_p$, which the collaborative sumcheck (WP10) proves.
//!
//! | Item | Step | Draft (`5_Building_Blocks.tex`) |
//! |---|---|---|
//! | [`owned_w`], [`block_w`], [`a_times_g`] | $w\_i = \mathbf a^T \mathbf G\_{2^m} \mathbf s\_i$, $\hat{\mathbf w}\_i = \mathbf G\_1^{-1}(w\_i)$ for the owned blocks | 103-108 |
//! | [`sample_attempt`], [`mask_images`] | per attempt: $\mathbf r\_p \leftarrow \chi\_r$, $\boldsymbol\rho\_p \leftarrow S\_b^{m\_\rho}$, $\mathbf t\_{0,p}$, $\hat{\mathbf t}\_{0,p}$, $w\_{0,p}$, $\hat{\mathbf w}\_{0,p}$ | 110-114 |
//! | [`d_slice`], [`fah_d_input`] | $p$'s input $\mathbf D \cdot \lbrace (\hat{\mathbf w}, \hat{\mathbf w}\_0, \hat{\mathbf t}\_0, \boldsymbol\rho) \rbrace\_p$ to $\mathcal F\_{\mathsf{AH}}(\mathbf D)$ | 115-121 |
//! | [`check_challenges`] | $\mathbf c \in \mathcal C^{2^r}$ | 122-126 |
//! | [`fold`], [`respond`] | $\sum\_{i \in \mathcal I\_p} c\_i \mathbf s\_i$ over $\mathbb Z\[X\]/(X^d+1)$ and $\mathsf{Rej}$ | 127-129 |
//! | [`aggregate`], [`norm_check`], [`z_hat`] | prover 1: $\mathbf z = \sum\_p \mathbf z\_p$, $\Vert \mathbf z \Vert\_\infty \le \beta$, $\hat{\mathbf z} = \mathbf J^{-1}(\mathbf z)$ | 129-133 |
//! | [`z_prime_share`], [`y_share`] | $\langle \mathbf z' \rangle\_p$ and $\lbrace \mathbf y \rbrace\_p$ ([`LinearShare`](hachi_co_core::LinearShare)) | 134, 154-179 |
//! | [`run`], [`run_with_cap`], [`R2lOutput`] | the driver: attempts until no party aborts, with [`Comm`](hachi_co_mpc::Comm), [`PartyRandomness`](hachi_co_core::PartyRandomness) and [`Challenger`](hachi_co_core::Challenger) | 102-182 |
//! | [`verifier_challenges`], [`linear_relation`] | the verifier: $\mathbf c$ by Fiat-Shamir, $\mathbf c \in \mathcal C^{2^r}$, $\mathbf M$ and $\mathbf y$ | 122-126, 157-179 |
//! | [`sum_shares`], [`residual`], [`relation_holds`], [`row_block_checks`] | test oracles: $\mathbf M \mathbf z' = \mathbf y$ and the meaning of the rows | 154, 182 |
//!
//! **Conventions** (`docs/m1-spec.md`, "Conventions"; `ref/README.md`, "Conventions"): parties
//! are $1, \dots, n$; positions are those of [`hachi_co_core::Layout`]; a ring element is a
//! [`Poly`](hachi_co_core::Poly) of $d$ residues; short vectors (masks, $\boldsymbol\rho$,
//! challenges, folds, responses) are `Vec<i64>`, flat with coefficient $j$ of element $i$ at
//! $id + j$ ($N = d\delta 2^m$ entries for masks, folds and responses, [`response_len`]).
//!
//! **Communication and coins.** Per attempt $a$ the driver calls
//! [`Challenger::begin_attempt`](hachi_co_core::Challenger::begin_attempt)`(a)`, absorbs
//! $\mathbf v$ under [`LABEL_V`] and draws
//! [`r2l_challenges`](hachi_co_core::Challenger::r2l_challenges)`(a, 2^r, d, ω)`; a Fiat-Shamir
//! verifier repeats these calls for the accepted attempt only ([`verifier_challenges`]). The
//! collective calls carry the labels [`COMM_FAH_D`], [`COMM_ABORT`], [`COMM_GATHER_Z`] and
//! [`COMM_PROVER1`], in this order within an attempt (the last two only in an accepted one).
//!
//! **Secrets and constant time.** The masks $\mathbf r\_p$ and their images, the commitment
//! randomness $\boldsymbol\rho\_p$, the digits of $\mathbf s\_i$, the folds and $\mathbf z$ are
//! secret. What aims at constant time (best effort, as Rust does not guarantee constant-time code
//! generation):
//! - $\mathsf{Rej}$ ([`respond`], through `hachi_co_sampling::rej::rej`): the checks of the fold,
//!   the mask and $\mathbf z\_p$ run over all coordinates without branches or early exit, with one
//!   branch on the combined (public) decision;
//! - this crate's own conversions of secrets: the range check and lift of the mask and of
//!   $\boldsymbol\rho\_p$ to residues and the centering of the digits of $\mathbf s\_i$ use
//!   arithmetic only; the exact products of the fold ([`hachi_co_core::short::negacyclic_mul`])
//!   branch only on the public challenge and on overflow, which cannot occur within the bounds;
//!   validity checks branch only on validity, whose outcome is fixed for valid inputs.
//!
//! What is **not** constant time: the images $\mathbf A \mathbf r\_p$ and $\mathbf a^T
//! \mathbf G\_{2^m} \mathbf r\_p$ and their decompositions $\mathbf G^{-1}$, and $w\_i$,
//! $\hat{\mathbf w}\_i$ (they run in `hachi-co-math`, whose $\mathbb Z\_q$ products use `u128`
//! remainders and conditional subtractions and whose $\mathbf G^{-1}$ chooses representatives and
//! digits by comparisons); prover 1's sum, norm check and $\mathbf J^{-1}$ on its private
//! $\mathbf z$; the samplers' rejection loops (whose running time depends on the number of
//! rejected words, which is independent of the values returned). The driver wipes each attempt's
//! mask and fold, and the responses of aborted attempts, as soon as they are no longer needed, and
//! [`AttemptSecrets`] wipes its contents when dropped, also on early returns (best effort: the
//! compiler is only discouraged from eliding the writes, and earlier copies are not reached).
//! [`AttemptSecrets`] and [`MaskImages`] hide their contents from `Debug`.
//!
//! **Choices made here where the draft and the oracle are silent** (each is documented at its
//! item):
//! - the accept bits are agreed ([`COMM_ABORT`]) before any response is sent, and the responses
//!   go to prover 1 only in an accepted attempt (plan `docs/hachi-co-plan.md`, finding F5);
//! - prover 1 announces the outcome of its steps after the gather ([`COMM_PROVER1`]); a failed
//!   norm check is not retried but ends the step with [`Error::NormCheck`] at prover 1 and
//!   [`Error::Prover1Abort`] at the others (it never happens for honest provers,
//!   `5_Building_Blocks.tex`:193);
//! - the attempts are capped at [`MAX_ATTEMPTS`] (the oracle's default);
//! - the labels [`LABEL_V`] and `COMM_*`;
//! - every prover and the verifier check $\mathbf c \in \mathcal C^{2^r}$ with the draft's
//!   $\ell\_1$ ball, and the shapes, canonical residues and ranges of everything they receive
//!   from the randomness source, the challenger and the network.
//!
//! References: "the draft" is the Hachi-Co paper draft (`collaborative-hatchi`, commit
//! `625ced1`), cited by file and line; "the oracle" is the Sage reference in `ref/`.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod ct;
mod driver;
mod error;
mod local;
mod prover1;
mod relation;
mod share;
mod verifier;

pub use driver::{AttemptRecord, R2lOutput, run, run_with_cap};
pub use error::{Error, Result};
pub use local::{
    AttemptSecrets, BlockW, MaskImages, a_times_g, block_w, check_challenges, d_slice, fah_d_input,
    fold, mask_images, owned_w, respond, response_len, sample_attempt,
};
pub use prover1::{aggregate, norm_check, z_hat};
pub use relation::{
    RowBlockChecks, UnpackedWitness, relation_holds, residual, row_block_checks, sum_shares,
};
pub use share::{y_share, z_prime_share};
pub use verifier::{LinearRelation, linear_relation, verifier_challenges};

/// The label under which provers and verifier absorb the first message $\mathbf v$ of each
/// attempt before drawing $\mathbf c$ (`5_Building_Blocks.tex`:121-126).
pub const LABEL_V: &str = "hachi-co/r2l/v";

/// The label of the $\mathcal F\_{\mathsf{AH}}(\mathbf D)$ call that outputs $\mathbf v$
/// (`5_Building_Blocks.tex`:119-121).
pub const COMM_FAH_D: &str = "hachi-co/r2l/F_AH(D)";

/// The label of the agreement on an abort after $\mathsf{Rej}$ (`5_Building_Blocks.tex`:129).
pub const COMM_ABORT: &str = "hachi-co/r2l/abort";

/// The label of the gather of the responses $\mathbf z\_p$ at prover 1
/// (`5_Building_Blocks.tex`:129).
pub const COMM_GATHER_Z: &str = "hachi-co/r2l/z_p";

/// The label of prover 1's announcement whether its steps after the gather failed
/// (`5_Building_Blocks.tex`:131: "If not, it aborts the protocol").
pub const COMM_PROVER1: &str = "hachi-co/r2l/prover-1-check";

/// The default cap on the attempts of [`run`] (the oracle's `max_attempts`). For honest provers
/// an attempt aborts with probability $1 - M^{-n} \le 1 - 1/e$ (`6_Collaborative_Hachi.tex`:275:
/// $M \le e^{1/n}$), so all of them abort with probability at most
/// $(1 - 1/e)^{64} \lt 2^{-42}$.
pub const MAX_ATTEMPTS: u64 = 64;
