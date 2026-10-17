//! Work package WP8 of Hachi-Co (`docs/m1-spec.md`, "WP8 `hachi-co-conv`"): the collaborative
//! reduction $\Pi\_{\mathsf{conv}}$ from $\mathcal R\_{\mathsf{eval}, \mathbb F\_{q^k}}$ to
//! $\mathcal R\_{\mathsf{eval}, \mathcal R\_q}$ (`field-to-ring-rok.tex`:89-169): the public
//! vectors $\mathbf x', \mathbf b, \mathbf a, v, \mathbf d$, the shares $\langle u \rangle\_p$,
//! $\mathcal F\_{\mathsf{AH}}(E)$ and the verifier's check $y = E(u)$; Sage oracle
//! `ref/hachico_ref/conv.py`.
//!
//! | Step (`field-to-ring-rok.tex`) | Function | Collective call |
//! |---|---|---|
//! | 1-2. $\mathbf x', \mathbf b, \mathbf a, v, \mathbf d$ (105-123) | [`public_vectors`] | |
//! | 3-4. $\langle u \rangle\_p = \mathbf d^T \lbrace \mathbf f' \rbrace\_p$ (124-131) | [`u_share`] | |
//! | 5. $E$ and its matrix $\mathbf E$ (132-158) | [`e_map`], [`e_matrix`] | |
//! | 6. $E(u)$ by $\mathcal F\_{\mathsf{AH}}(E)$ (160-162) | [`fah_e_input`], [`pi_conv`] | `fah` ([`LABEL_FAH_E`]) |
//! | 6. V checks $y = E(u)$ (164) | [`verify`], [`check`] | |
//!
//! [`check_relation`] checks the reduction itself on reconstructed witnesses: the input relation
//! $f(\mathbf x) = y$, the output relation $u = f'(\mathbf x')$ and the link $E(u) = f(\mathbf x)$.
//!
//! **Conventions** (`ref/README.md`, "Πconv"): $\mathbf x' = \mathbf 1^{\lceil \log n \rceil}
//! \Vert \mathbf x\[0 : \ell - \alpha + \kappa\]$, monomials ordered by the MSB-first
//! $\mathrm{nat}$, $d\[\mathrm{nat}(\mathbf i) 2^m + \mathrm{nat}(\mathbf j)\] = b\_{\mathbf i}
//! a\_{\mathbf j}$, $\mathbb F\_{q^k} = \mathcal R\_q^H$ in Hachi's basis.
//!
//! **Choices** (each documented at its item): $\mathsf{PConv}$ ran in Commit and $\mathbf f'$ is
//! read from the states (the oracle's choice 3); $u$ stays secret-shared and only $E(u)$ is
//! opened (choice 4); V's check rejects non-canonical residues of $E(u)$ and $y$ ([`check`]); the
//! labels [`LABEL_FAH_E`] and [`LABEL_E_U`].
//!
//! References: "the draft" is the Hachi-Co paper draft (`collaborative-hatchi`, commit
//! `625ced1`), cited by file and line; "the oracle" is the Sage reference in `ref/`.
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod driver;
mod error;
mod public;
mod relation;
mod share;
mod verifier;

pub use driver::{ConvOutput, LABEL_E_U, LABEL_FAH_E, pi_conv};
pub use error::{Error, Result, VerifyError};
pub use public::public_vectors;
pub use relation::{ConvRelation, check_relation, evaluate, f_prime_of_states};
pub use share::{e_map, e_matrix, fah_e_input, u_share};
pub use verifier::{check, verify};
