//! The shared foundation of the Hachi-Co building blocks (milestone 1, `docs/m1-spec.md`, section
//! `hachi-co-core`).
//!
//! The block crates (`hachi-co-commit`, `-conv`, `-r2l`, `-sumcheck`) depend on this crate and on
//! `hachi-co-mpc`, never on each other: everything they exchange is a type defined here.
//!
//! | Item | Contents | Draft |
//! |---|---|---|
//! | [`Context`], [`PublicParams`] | parameters, $\mathcal R\_q$, $\mathbb Z\_q\[X\]/(X^{2d}+1)$, $\mathcal R\_q^H \cong \mathbb F\_{q^k}$, $\mathbf A, \mathbf B, \mathbf D$ | `6_Collaborative_Hachi.tex`:263-298 |
//! | [`Layout`] | positions in $\mathbf z'$, rows of $\mathbf M$, block owners, $S\_p$, hypercube sizes | `5_Building_Blocks.tex`:157-172, 197; `6_Collaborative_Hachi.tex`:278-281, 406 |
//! | [`build_m`], [`build_y`] | the linear relation $\mathbf M \mathbf z' = \mathbf y$ | `5_Building_Blocks.tex`:154-182 |
//! | [`BlockState`], [`PartyState`], [`PublicVectors`], [`LinearShare`], [`Blinder`], [`SumcheckCoins`] | the values passed between blocks | `6_Collaborative_Hachi.tex`:390; `field-to-ring-rok.tex`:111-122; `5_Building_Blocks.tex`:164-179, 209-214, 243-250 |
//! | [`PartyRandomness`], [`PrgRandomness`] | a party's private randomness | `4_preliminaries.tex`:154; `5_Building_Blocks.tex`:110, 209-214 |
//! | [`Challenger`], [`FsChallenger`] | the verifier's public coins, by Fiat-Shamir, and the contexts of the base-PCS openings | `5_Building_Blocks.tex`:122-126, 243-250, 312, 314-326 |
//! | [`BasePcs`], [`base_pcs`] | the base PCS of the sumcheck (commitments and openings of $\langle \widetilde w \rangle\_p$, $g\_{0,p}$, $g\_{1,p}$), the Möbius transform, evaluations | `5_Building_Blocks.tex`:209-241, 314-333 |
//! | [`short`], [`wire`] | exact short-polynomial products, norms and centered representatives; flat encodings for [`hachi_co_mpc::Comm`] | `4_preliminaries.tex`:86; `5_Building_Blocks.tex`:127-133 |
//!
//! The inter-block types are plain structs with public fields; each has a `validate` method that
//! checks its shape (lengths, and $d$ coefficients per polynomial) against a [`Layout`], not its
//! values. The call order of the coins, including Fiat-Shamir with aborts, is in the
//! documentation of [`Challenger`].
//!
//! With the feature `fixtures` the crate also provides `fixture::Fixture` (the Sage oracle's
//! fixtures `ref/fixtures/toy-n{1,2,3,4,8}.json` with typed decoders and `sha256_residues`) and
//! `replay::{ReplayRandomness, ReplayChallenger}`, which serve the recorded randomness and coins
//! so that the drivers reproduce the fixtures, and the recorded inputs of the blocks
//! (`Fixture::party_state`, `Fixture::public_vectors`, …); the block crates enable it in their
//! dev-dependencies. It also provides `clear_pcs::ClearPcs`, a reference base PCS that commits
//! to the SHA-256 hash of a vector and opens by revealing it (binding and sound, neither
//! succinct nor hiding), for the fixture tests: the oracle does not model the base PCS, which
//! is Hachi otherwise (`hachi-co`'s `HachiBase`).
//!
//! **Conventions** (`docs/m1-spec.md`, "Conventions"; `ref/README.md`, "Conventions"): positions
//! are 0-indexed and provers are numbered $1, \dots, n$ ([`PartyId`]); a ring element is a
//! [`Poly`] with $d$ residues in $\[0, q)$ and a field element an [`Ext`] in Hachi's basis; short
//! signed vectors (masks, $\boldsymbol\rho$, challenges, responses) are `Vec<i64>`, one ring
//! element after the other; [`Matrix`] is row-major. Errors are values ([`Error`]).
//!
//! **Choices made here where the brief and the oracle are silent** (each is documented at its
//! item): the extra arguments `field` of [`PartyRandomness::reshare`],
//! [`PartyRandomness::blinder`], [`Challenger::sumcheck_coins`] and [`Challenger::round_coin`];
//! [`BlinderKind`] for the brief's `which: 0 | 1`; [`Challenger::begin_attempt`] for Fiat-Shamir
//! with aborts; the PRG domains and the reuse check of [`PrgRandomness`]; the absorb encodings,
//! coin and context derivation of [`FsChallenger`] ([`Challenger::absorb_words`] and the forked
//! squeeze of [`Challenger::context`]); the extra dimensions and position ranges of [`Layout`];
//! the canonical-residue check of [`PublicParams::validate`]; the hash, word formats and context
//! check of `ClearPcs`.
//!
//! References: "the draft" is the Hachi-Co paper draft (`collaborative-hatchi`, commit `625ced1`),
//! cited by file and line; "the oracle" is the Sage reference in `ref/`.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod base_pcs;
mod challenger;
mod context;
mod data;
mod error;
mod layout;
mod linear;
mod randomness;
pub mod short;
pub mod wire;

#[cfg(feature = "fixtures")]
pub mod clear_pcs;
#[cfg(feature = "fixtures")]
pub mod fixture;
#[cfg(feature = "fixtures")]
mod fixture_inputs;
#[cfg(feature = "fixtures")]
pub mod replay;

pub use base_pcs::BasePcs;
pub use challenger::{
    Challenger, FS_DOMAIN_R2L_C, FS_DOMAIN_SC_COINS, FS_DOMAIN_SC_ROUND, FS_LABEL_CONTEXT,
    FsChallenger,
};
pub use context::{Context, PublicParams, SETUP_DOMAIN_A, SETUP_DOMAIN_B, SETUP_DOMAIN_D};
pub use data::{
    Blinder, BlinderKind, BlockState, LinearShare, Matrix, PartyState, PublicVectors, SumcheckCoins,
};
pub use error::{Error, Result};
pub use layout::{Layout, NamedBlock};
pub use linear::{build_m, build_y};
pub use randomness::{
    DOMAIN_BASE_PCS, DOMAIN_BLINDER_G0, DOMAIN_BLINDER_G1, DOMAIN_MASK, DOMAIN_RESHARE, DOMAIN_RHO,
    PartyRandomness, PrgRandomness,
};

#[cfg(feature = "fixtures")]
pub use clear_pcs::{ClearCommitment, ClearOpening, ClearPcs, ClearPolynomial};
#[cfg(feature = "fixtures")]
pub use fixture::{Fixture, escape_key, sha256_residues};
#[cfg(feature = "fixtures")]
pub use replay::{REPLAY_BASE_PCS_TAG, REPLAY_CONTEXT_TAG, ReplayChallenger, ReplayRandomness};

pub use hachi_co_math::{Ext, Fqk, Poly, Ring, Subfield};
pub use hachi_co_mpc::PartyId;
