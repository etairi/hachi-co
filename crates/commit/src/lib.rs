//! Work package WP7 of Hachi-Co (`docs/m1-spec.md`, "WP7 `hachi-co-commit`"): the collaborative
//! $\mathsf{Commit}$ and the test oracle $\mathsf{Open}$ of the draft
//! (`6_Collaborative_Hachi.tex`:305-421), without the resharing by $\mathsf{AS.Share}$ since the
//! draft's commit `1096ea2`: $\mathsf{PConv}$ (`field-to-ring-rok.tex`:65-116) of each party's
//! input share, the inner
//! commitments $\mathbf s\_i, \mathbf t\_i, \hat{\mathbf t}\_i$ of the owned blocks and
//! $\mathcal F\_{\mathsf{AH}}(\mathbf B)$; Sage oracle `ref/hachico_ref/commit.py`.
//!
//! Every step is a party-local pure function, and [`commit`] chains them for one party against a
//! [`hachi_co_mpc::Comm`]:
//!
//! | Step (`6_Collaborative_Hachi.tex`) | Function | Collective call |
//! |---|---|---|
//! | 1. the input share $\langle \mathbf f \rangle\_p$ (the resharing of 308-313 is gone since `1096ea2`) | [`input_share`] | |
//! | 2. $\mathsf{PConv}$ of $p$'s blocks (315-318) | [`pconv_blocks`] | |
//! | 3. $\mathbf s\_i, \mathbf t\_i, \hat{\mathbf t}\_i$ (375-381; zero for the blocks of $T\_0$) | [`block_state`], [`party_state`] | |
//! | 4. $\boldsymbol\rho^u\_p \leftarrow S\_b^{m\_u}$ (382) | [`sample_rho_u`] | |
//! | 5. $\mathbf u = \mathbf B (\hat{\mathbf t}, \boldsymbol\rho^u)$ (383-397) | [`t_hat_slice`], [`fah_b_input`] | `fah` ([`LABEL_FAH_B`]) |
//! | 6. $\mathbf u$ and $\mathsf{st}\_p$ (400) | [`commit`] | |
//!
//! [`commit_streaming`] runs the same steps block by block, for a polynomial too large for memory,
//! with the blocks of $\mathbf f'$ on disk.
//!
//! [`open`] is $\mathsf{Open}$ (lines 403-419), reporting each condition ([`OpenReport`]); its
//! slack space needs the invertibility test of [`units`]. The global view of $\mathsf{PConv}$
//! ([`pconv`], [`f_hat`], [`f_prime_of_f_hat`]) and its inverse ([`f_hat_of_f_prime`],
//! [`fold_slots`]) serve Open and the tests.
//!
//! **Conventions** (`ref/README.md`): 0-indexed positions, provers $1, \dots, n$, MSB-first
//! $\mathrm{nat}$, $\mathbb F\_{q^k} = \mathcal R\_q^H$ in Hachi's basis, balanced digits with the
//! representative rule of choice 2 in the element-major layout, block $i$ held by prover
//! $\lfloor i / 2^{r - \lceil \log n \rceil} \rfloor + 1$ when that is at most $n$.
//!
//! **Choices where the draft and the oracle are silent** (each documented at its item):
//! - the coordinates of the party's input share are read modulo $q$ ([`input_share`]);
//! - the labels [`LABEL_FAH_B`] and [`LABEL_U`];
//! - in [`open`]: two entries for the same block in one state count as "several states"; when
//!   several states hold a block, the first entry is used (as the oracle does); $\mathbf u$ is
//!   compared as given (a non-canonical residue fails `B_that_eq_u`, as in the oracle), $f$
//!   modulo $q$ (as in the oracle); every condition is evaluated even when $c \notin \mathcal Z$
//!   (as in the oracle); inputs of the wrong shape are errors, not failed conditions;
//! - [`units::is_unit`] uses the factorization of $X^d + 1$ for $q \equiv 5 \pmod 8$ and
//!   Euclid's algorithm otherwise.
//!
//! References: "the draft" is the Hachi-Co paper draft (`collaborative-hatchi`, commit
//! `8834c9f`), cited by file and line; "the oracle" is the Sage reference in `ref/`.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod driver;
mod error;
mod inner;
mod opening;
mod polyconv;
mod share;
pub mod units;

pub use driver::{LABEL_FAH_B, LABEL_U, commit, commit_streaming};
pub use error::{Error, Result};
pub use inner::{block_state, fah_b_input, party_state, sample_rho_u, t_hat_slice};
pub use opening::{OpenChecks, OpenReport, c_star, open};
pub use polyconv::{f_hat, f_hat_of_f_prime, f_prime_of_f_hat, fold_slots, pconv, pconv_blocks};
pub use share::input_share;
