//! Work package WP9 of Hachi-Co (`docs/m1-spec.md`, "WP9 `hachi-co-r2l`"): the conversion of ring
//! evaluation claims to linear relations over $\mathcal R\_q$ (`5_Building_Blocks.tex`:98-207,
//! with the $L$ parallel iterations of the rejection sampling of `fig:ring-to-linear` and
//! `fig:ring-to-linear-2` since the draft's commit `1096ea2`; Sage oracle
//! `ref/hachico_ref/r2l.py`). After $\Pi\_{\mathsf{conv}}$, the provers hold the
//! blocks $\mathbf s\_i$ of $\mathbf f'$ and shares $\langle u \rangle\_p$; this step ends with a
//! public matrix $\mathbf M$ and shares $\langle \mathbf z' \rangle\_p$, $\lbrace \mathbf y
//! \rbrace\_p$ with $\mathbf M \sum\_p \langle \mathbf z' \rangle\_p = \sum\_p \lbrace \mathbf y
//! \rbrace\_p$, which the collaborative sumcheck (WP10) proves.
//!
//! | Item | Step | Draft (`5_Building_Blocks.tex`) |
//! |---|---|---|
//! | [`owned_w`], [`block_w`], [`a_times_g`] | $w\_i = \mathbf a^T \mathbf G\_{2^m} \mathbf s\_i$, $\hat{\mathbf w}\_i = \mathbf G\_1^{-1}(w\_i)$ for the owned blocks | 103-108 |
//! | [`sample_masks`], [`mask_images`] | $\mathbf r\_p^j \leftarrow \chi\_r$ for $j \lt L$, $\boldsymbol\rho^v\_p \leftarrow S\_b^{m\_v}$, and per mask $\mathbf t\_{0,p}^j$, $\hat{\mathbf t}\_{0,p}^j$, $w\_{0,p}^j$, $\hat{\mathbf w}\_{0,p}^j$ | `fig:ring-to-linear`, step 3 |
//! | [`d_slice`], [`fah_d_input`] | $p$'s input $\mathbf D \cdot \lbrace (\hat{\mathbf w}, \hat{\mathbf w}\_0, \hat{\mathbf t}\_0, \boldsymbol\rho) \rbrace\_p$ to $\mathcal F\_{\mathsf{AH}}(\mathbf D)$ | 115-121 |
//! | [`check_challenges`] | $\mathbf c \in \mathcal C^{2^r}$ | 122-126 |
//! | [`fold`], [`respond`] | $\sum\_{i \in \mathcal I\_p} c\_i \mathbf s\_i$ over $\mathbb Z\[X\]/(X^d+1)$ and $\mathsf{Rej}$, once per mask | `fig:ring-to-linear`, step 5 |
//! | [`accept`]: [`encode_accept_set`], [`check_accept_set`], [`select_iteration`] | the sets $R\_p$ of accepted iterations and $j^\ast = \min \bigcap\_p R\_p$ | `fig:ring-to-linear`, step 6 |
//! | [`z_of_sum`], [`norm_check`], [`z_hat`] | prover 1: $\mathbf z = \sum\_p \mathbf z\_p^{j^\ast}$, $\Vert \mathbf z \Vert\_\infty \le \beta$, $\hat{\mathbf z} = \mathbf J^{-1}(\mathbf z)$ | 129-133 |
//! | [`z_prime_share`], [`y_share`] | $\langle \mathbf z' \rangle\_p$ and $\lbrace \mathbf y \rbrace\_p$ ([`LinearShare`](hachi_co_core::LinearShare)) | 134, 154-179 |
//! | [`run`], [`R2lOutput`] | the driver, one round with $L$ masks, with [`Comm`](hachi_co_mpc::Comm), [`PartyRandomness`](hachi_co_core::PartyRandomness) and [`Challenger`](hachi_co_core::Challenger) | 102-182 |
//! | [`verifier_challenges`], [`verifier_iteration`], [`linear_relation`] | the verifier: $\mathbf c$ by Fiat-Shamir, $\mathbf c \in \mathcal C^{2^r}$, $j^\ast$ from the sets, $\mathbf M$ and $\mathbf y$ | 122-126, 157-179 |
//! | [`sum_shares`], [`residual`], [`relation_holds`], [`row_block_checks`] | test oracles: $\mathbf M \mathbf z' = \mathbf y$ and the meaning of the rows | 154, 182 |
//!
//! **Conventions** (`docs/m1-spec.md`, "Conventions"; `ref/README.md`, "Conventions"): parties
//! are $1, \dots, n$; positions are those of [`hachi_co_core::Layout`]; a ring element is a
//! [`Poly`](hachi_co_core::Poly) of $d$ residues; short vectors (masks, $\boldsymbol\rho^v$,
//! challenges, folds, responses) are `Vec<i64>`, flat with coefficient $j$ of element $i$ at
//! $id + j$ ($N = d\delta 2^m$ entries for masks, folds and responses, [`response_len`]).
//!
//! **Communication and coins.** The driver absorbs $\mathbf v$ under [`LABEL_V`], draws
//! [`r2l_challenges`](hachi_co_core::Challenger::r2l_challenges)`(2^r, d, ω)`, and after the
//! broadcast of the sets of accepted iterations absorbs them under [`LABEL_ACCEPT`]; a
//! Fiat-Shamir verifier makes the same calls ([`verifier_challenges`], [`verifier_iteration`]).
//! The collective calls carry the labels [`COMM_FAH_D`], [`COMM_ACCEPT`] and [`COMM_FAH_Z`], in
//! this order (the last only when some iteration is accepted by every prover). After the last,
//! every prover checks $\Vert \mathbf z \Vert\_\infty \le \beta$ itself (`fig:ring-to-linear-2`,
//! step 9, since the draft's commit `0337dfb`), so no announcement follows.
//!
//! **Secrets and constant time.** The masks $\mathbf r\_p^j$ and their images, the commitment
//! randomness $\boldsymbol\rho^v\_p$, the digits of $\mathbf s\_i$, the folds and $\mathbf z$ are
//! secret. What aims at constant time (best effort, as Rust does not guarantee constant-time code
//! generation):
//! - $\mathsf{Rej}$ ([`respond`], through `hachi_co_sampling::rej::rej`): the checks of the fold,
//!   the mask and $\mathbf z\_p$ run over all coordinates without branches or early exit, with one
//!   branch on the combined (public) decision;
//! - the conversions of secrets: the range check and lift of the mask and of
//!   $\boldsymbol\rho^v\_p$ to residues (this crate's own) and the centering of the digits of
//!   $\mathbf s\_i$ ([`hachi_co_core::short::center_poly`]) use arithmetic only; the exact
//!   products of the fold ([`hachi_co_core::short::negacyclic_mul`]) branch only on the public
//!   challenge and on overflow, which cannot occur within the bounds; validity checks branch only
//!   on validity, whose outcome is fixed for valid inputs.
//!
//! What is **not** constant time: the images $\mathbf A \mathbf r\_p$ and $\mathbf a^T
//! \mathbf G\_{2^m} \mathbf r\_p$ and their decompositions $\mathbf G^{-1}$, and $w\_i$,
//! $\hat{\mathbf w}\_i$ (they run in `hachi-co-math`, whose $\mathbb Z\_q$ products use `u128`
//! remainders and conditional subtractions and whose $\mathbf G^{-1}$ chooses representatives and
//! digits by comparisons); the norm check on $\mathbf z$ and prover 1's $\mathbf J^{-1}$; the
//! samplers' rejection loops (whose running time depends on the number of
//! rejected words, which is independent of the values returned). The driver wipes the masks and
//! the fold right after $\mathsf{Rej}$ and every response but that of $j^\ast$ before any is
//! sent, and [`MaskSecrets`] wipes its contents when dropped, also on early returns (best effort:
//! the compiler is only discouraged from eliding the writes, and earlier copies are not reached).
//! [`MaskSecrets`] and [`MaskImages`] hide their contents from `Debug`.
//!
//! **Choices made here where the draft and the oracle are silent** (each is documented at its
//! item):
//! - the sets of accepted iterations are broadcast ([`COMM_ACCEPT`]) before any response is
//!   sent, as the draft has it, and only the responses of $j^\ast$ enter
//!   $\mathcal F\_{\mathsf{AH}}(1^n)$ (plan `docs/hachi-co-plan.md`, finding F5); a set is
//!   $\lceil L/64 \rceil$ words ([`accept`]), and iterations are 0-indexed (the draft's $j + 1$);
//! - the labels [`LABEL_V`], [`LABEL_ACCEPT`] and `COMM_*`;
//! - every prover and the verifier check $\mathbf c \in \mathcal C^{2^r}$ with the draft's
//!   $\ell\_1$ ball, and the shapes, canonical residues and ranges of everything they receive
//!   from the randomness source, the challenger and the network.
//!
//! References: "the draft" is the Hachi-Co paper draft (`collaborative-hatchi`, commit
//! `8834c9f`), cited by file and line, and since its commit `1096ea2` by figure and step for the
//! $L$ iterations (and since `0337dfb` for the norm check of every prover); "the oracle" is the
//! Sage reference in `ref/`.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod accept;
mod ct;
mod driver;
mod error;
mod local;
mod prover1;
mod relation;
mod share;
mod verifier;

pub use accept::{accept_set_len, accepts, check_accept_set, encode_accept_set, select_iteration};
pub use driver::{R2lOutput, run, run_sparse};
pub use error::{Error, Result};
pub use local::{
    BlockW, MaskImages, MaskSecrets, a_times_g, block_w, check_challenges, d_slice, fah_d_input,
    fold, mask_images, owned_w, respond, response_len, sample_mask_images, sample_masks,
};
pub use prover1::{norm_check, z_hat, z_of_sum};
pub use relation::{
    R3Conditions, RowBlockChecks, UnpackedWitness, r3_conditions, relation_holds, residual,
    row_block_checks, sum_shares,
};
pub use share::{y_share, z_prime_share, z_prime_share_sparse};
pub use verifier::{
    LinearRelation, check_relation, linear_relation, verifier_challenges, verifier_iteration,
};

/// The label under which provers and verifier absorb the first message $\mathbf v$ before
/// drawing $\mathbf c$ (`5_Building_Blocks.tex`:133-138).
pub const LABEL_V: &str = "hachi-co/r2l/v";

/// The label under which provers and verifier absorb the $n$ sets of accepted iterations
/// (`fig:ring-to-linear`, step 6), one vector of words per prover.
pub const LABEL_ACCEPT: &str = "hachi-co/r2l/accept-sets";

/// The label of the $\mathcal F\_{\mathsf{AH}}(\mathbf D)$ call that outputs $\mathbf v$
/// (`5_Building_Blocks.tex`:131-133).
pub const COMM_FAH_D: &str = "hachi-co/r2l/F_AH(D)";

/// The label of the broadcast of the sets of accepted iterations (`fig:ring-to-linear`, step 6).
pub const COMM_ACCEPT: &str = "hachi-co/r2l/accept-sets";

/// The label of the $\mathcal F\_{\mathsf{AH}}(1^n)$ call that sums the responses
/// $\mathbf z\_p^{j^\ast}$ (`5_Building_Blocks.tex`:142-143).
pub const COMM_FAH_Z: &str = "hachi-co/r2l/F_AH(1^n)";
