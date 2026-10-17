//! The Setup cascade of the Hachi-Co draft, a constraint report, and presets (milestone 0,
//! `docs/m0-spec.md`, section `hachi-co-params`).
//!
//! | Item | Contents | Draft |
//! |---|---|---|
//! | [`Choices`] | $\lambda, n, t, \ell, \alpha, \kappa, b, \omega$, the challenge set, the $\tau$ rule, $n_A, n_B, n_D, m_\rho$, $q$ | 6_Collaborative_Hachi.tex:263-273 |
//! | [`derive`](fn@crate::derive) | every derived quantity, exactly ([`Params`]) | 6_Collaborative_Hachi.tex:265-288 |
//! | [`Report`] | each condition, whether it holds, and its [`Severity`] | 6_Collaborative_Hachi.tex:269-275, 4_preliminaries.tex:73, 100 |
//! | [`presets`](mod@presets) | the toy presets, the development presets and the secure presets of `presets/secure.json` | `docs/m0-spec.md`; `docs/m1-spec.md`, WP11 |
//! | [`rules`] | integer logarithms, the $\tau$ rules, primality, $\lvert C \rvert$, $M \le e^{1/n}$ | — |
//! | [`sizes`](fn@crate::sizes) | the byte size of the Eval proof's encoding, per component ([`ProofSizes`]) | `docs/m1-spec.md`, WP11 |
//!
//! **Example.**
//!
//! ```
//! use hachi_co_params::{derive, presets::toy_choices};
//!
//! let (params, report) = derive(&toy_choices(2))?;
//! assert_eq!((params.q, params.delta, params.tau, params.mu), (33554501, 13, 10, 785));
//! assert!(report.is_admissible()); // no fatal violation
//! assert!(!report.all_hold()); // the toy misses the knowledge-error target (a warning)
//! # Ok::<(), hachi_co_params::Error>(())
//! ```
//!
//! **Choices this crate makes where the draft is silent** (each is documented at its item):
//! $q$ is either fixed or the smallest admissible prime ([`QChoice::Smallest`]); $\tau$ is tight
//! by default ([`TauRule`]); $\omega$ may be the smallest radius meeting the knowledge-error
//! target ([`Omega::MinForTarget`]); the severities follow the rule in [`report`]; the
//! development presets are a kind of their own ([`PresetKind::Dev`]); the proof encoding that
//! [`sizes`](fn@crate::sizes) counts uses 64-bit words ([`sizes::WORD_BYTES`]).
//!
//! References: "the draft" is the Hachi-Co paper draft (`collaborative-hatchi`, commit
//! `625ced1`), cited by file and line; "Hachi" is ePrint 2026/156 (NOZ26), cited by lemma.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod cascade;
pub mod choices;
pub mod error;
pub mod presets;
pub mod report;
pub mod rules;
pub mod sizes;

pub use cascade::{Params, derive};
pub use choices::{ChallengeSet, Choices, Omega, QChoice, TauRule};
pub use error::{Error, Result};
pub use presets::{Preset, PresetKind, preset, presets};
pub use report::{Constraint, ConstraintId, Report, Severity};
pub use sizes::{ProofSizes, sizes};
