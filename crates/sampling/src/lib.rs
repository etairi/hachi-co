//! Samplers for Hachi-Co: a SHAKE128 PRG, exact uniform and short samplers, uniform challenges
//! from the $\ell_1$ ball, public matrices from a seed, and uniform rejection sampling.
//!
//! | Module | Contents | Draft |
//! |---|---|---|
//! | [`prg`] | [`ShakePrg`], a framed SHAKE128 stream implementing [`rand::Rng`] | — |
//! | [`uniform`] | exact $[0, n)$, $[lo, hi]$, $\mathbb Z_q$, $[-\gamma, \gamma]^N$ and $S_\beta$ | 4_preliminaries.tex:88, 117 |
//! | [`l1ball`] | $\lvert C \rvert$ and exactly uniform $c \in C = \lbrace c : \lVert c \rVert_1 \le \omega \rbrace$ | 5_Building_Blocks.tex:122 |
//! | [`matrix`] | $A$, $B$, $D$ entry by entry from a seed | 6_Collaborative_Hachi.tex:283-287 |
//! | [`rej`] | $\mathsf{Rej}$, $\mathsf{SimRej}$, masks, the exact acceptance probability $1/M$ | 4_preliminaries.tex:117-127 |
//!
//! **Exactness.** Every sampler is exactly uniform on its set given a uniform source, by
//! rejection on whole words (no modular reduction, no floating point). A rejection loop that
//! fails [`uniform::MAX_TRIES`] times in a row returns [`Error::RejectionLimit`], which a
//! uniform source triggers with probability below $2^{-256}$.
//!
//! **Reproducibility.** The byte-level rules (framing of the PRG input, word order, the order of
//! the draws inside each sampler) are part of each item's documentation, and
//! `tests/fixtures/gen_kat.py` reimplements them in Python on `hashlib.shake_128` to produce the
//! known-answer tests in `tests/fixtures/kat.json`.
//!
//! **Secrets.** Masks and commitment randomness must come from a cryptographic generator, such as
//! a [`ShakePrg`] with a secret uniform seed; [`rej::mask`] enforces this with a
//! [`rand::CryptoRng`] bound. In [`uniform::uniform_below`] and the samplers built on it, the
//! running time depends on the number of rejected words, which is independent of the values
//! returned. Beyond that and [`rej::rej`], whose checks avoid secret-dependent branches, nothing
//! here aims at constant time (the big-integer comparison of [`uniform::uniform_below_big`],
//! used for public challenges, is variable time).

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod draw;
#[cfg(test)]
mod enumerate;
pub mod error;
pub mod l1ball;
pub mod matrix;
pub mod prg;
pub mod rej;
pub mod uniform;

pub use error::{Error, Result};
pub use prg::ShakePrg;
