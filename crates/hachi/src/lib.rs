//! # Hachi prototype (and the base of Hachi-Co)
//!
//! A prototype implementation of **Hachi**, the lattice-based multilinear polynomial commitment
//! scheme of Nguyen, O'Rourke and Zhang (Cryptology ePrint Archive 2026/156), and the code base
//! on which **Hachi-Co**, its collaborative variant with a secret-shared witness, is being built.
//! The code started from the authors' public prototype (github.com/georgeorourke/hachi-pcs); see the
//! README's "Attribution" section.
//!
//! Equations in this documentation are rendered with KaTeX (see `katex-header.html`); build the
//! documentation with `cargo docs` (an alias defined in `.cargo/config.toml` that documents this crate's
//! private modules with the header); `cargo docs-open` also opens it in the browser.
//!
//! ## What is implemented
//!
//! One round of the single-prover scheme for a multilinear polynomial
//! $f \in \mathbb Z_q^{\le 1}\[X_1, \dots, X_\ell\]$ with coefficients in $\mathbb Z_q$,
//! $q = 4294967197 = 2^{32} - 99 \equiv 5 \pmod 8$, evaluated at a point $x \in \mathbb Z_q^\ell$
//! (the base-field case, $k = 1$, of the paper's Section 3). The coefficient vector is read as
//! $2^r$ chunks of $2^m$ elements of the ring $\mathbf R_q = \mathbb Z_q\[X\]/(X^d + 1)$,
//! $d = 2^\alpha$, so that $\ell = r + m + \alpha$.
//!
//! * **Commit** ([`hachi::commit`]): each chunk $\mathbf f_i$ is gadget-decomposed into short
//!   digits $\mathbf s_i = \mathbf G^{-1}(\mathbf f_i)$ and committed with the Ajtai
//!   commitment $\mathbf t_i = \mathbf{A}\mathbf s_i$; the outer commitment is
//!   $\mathbf{u} = \mathbf{B}\hat{\mathbf{t}}$ with $\hat{\mathbf{t}} = \mathbf G^{-1}(\mathbf{t})$
//!   (paper, Eq. (13)-(14)).
//! * **Prove** ([`hachi::prove`]): reduces the evaluation claim to the ring element
//!   $Y = \sum_i b_i w_i$ with $w_i = \mathbf a^T \mathbf f_i$, commits to $\hat{\mathbf{w}}$ as
//!   $\mathbf{v} = \mathbf{D}\hat{\mathbf{w}}$, derives the sparse challenges $c_i$ by Fiat-Shamir,
//!   folds the witness into $\mathbf{z} = \sum_i c_i \mathbf s_i$, lifts the five verification
//!   equations of the paper's Eq. (20) from $\mathbf R_q$ to $\mathbb Z_q\[X\]$ (ring switching,
//!   Section 4.3), commits to the next witness $(\mathbf{z}^{\prime}, \mathbf{r})$, and runs two sumchecks
//!   over $\mathbb F_{q^4}$: $F_0$ proves that every entry of the next witness is a short digit,
//!   and $F_\alpha$ proves the linear relation $\mathbf{M}\mathbf{z}^{\prime} = \mathbf{y} + (X^d+1)\mathbf{r}$
//!   evaluated at a random $\alpha \in \mathbb F_{q^4}$.
//! * **Verify** ([`hachi::verify`]): checks the reduction $y = \langle \mathrm{cf}(Y), \mathbf v_x\rangle$,
//!   re-derives every challenge from the transcript, checks both sumchecks round by round, and
//!   checks their final values against the claimed evaluation $y^{\prime} = \tilde{w}(\mathbf{a})$ of the
//!   next witness.
//!
//! **Status.** This is a first-round benchmarking prototype, as in Fig. 8 of the paper: the claimed
//! evaluation $y^{\prime}$ of the next witness is supplied by the prover and trusted; no opening of the
//! commitment $\mathbf{u}^{\prime}$ and no recursion are implemented. The prover is not zero-knowledge and
//! the commitment is deterministic (not hiding).
//!
//! ## Notation: code names versus the paper
//!
//! | Code | Paper | Meaning |
//! |---|---|---|
//! | `params.l` | $\ell$ | number of variables of the committed polynomial |
//! | `params.d`, `params.d.log()` | $d = 2^\alpha$, $\alpha$ | ring dimension |
//! | `params.q` | $q$ | prime modulus, $q \equiv 5 \pmod 8$ |
//! | `params.r`, `params.m` | $r$, $m$ | folding parameters, $2^r$ chunks of $2^m$ ring elements |
//! | `params.n` | $n_A = n_B = n_D$ | height of the commitment matrices |
//! | `params.b`, `params.delta` | $b$, $\delta = \lceil \log_b q \rceil$ | decomposition base and number of digits |
//! | `params.delta_z`, `params.z_bound` | $\tau$, $\beta$ | digits and (heuristic) bound for the response $\mathbf{z}$ |
//! | `params.k` | $\omega$ | number of nonzero $\pm 1$ coefficients of a challenge (**not** the extension degree) |
//! | `ExtField` | $\mathbb F_{q^4}$ | field of the sumcheck challenges, $k = 4$ |
//! | `PVec`, `PMat`, `PChal` | vectors and matrices over $\mathbf R_q$, sparse challenges $c \in \mathcal{C}$ | see [`arithmetic`] |
//! | `ProofRound::y` | $Y$ (the ring element $u$ of Hachi-Co) | reduction of the evaluation claim to $\mathbf R_q$ |
//! | `ProofRound::u_dash`, `ProofRound::y_dash` | $\mathbf{u}^{\prime}$, $y^{\prime}$ | commitment to, and claimed evaluation of, the next witness |
//!
//! Residues of $\mathbb Z_q$ are stored as `u64` in $\[0, q)$; short signed values (digits, the
//! response $\mathbf{z}$) are stored as wrapping two's-complement `u64`. Gadget decomposition of a
//! residue first maps it to its representative in the balanced-digit window (see
//! [`arithmetic::utils::to_window`]).
//!
//! ## Modules
//!
//! * [`arithmetic`]: fields, the NTT-based ring arithmetic, polynomial vectors and matrices,
//!   sparse challenges, gadget decomposition, the Fiat-Shamir transcript and generic sumcheck helpers.
//! * [`hachi`]: setup, commit, prove and verify, plus the prover-side utilities.
//! * [`stream`]: the witness is streamed from a file so that very large polynomials fit in memory.
//! * [`utils`]: witness-file generation and progress output.
//!
//! ## Command line
//!
//! ```text
//! hachi <witness-file> <l>
//! ```
//! runs commit, prove and verify for a witness of $2^\ell$ coefficients streamed from the file
//! (with feature `gen_file`, a random witness file is generated first) and prints timings.

// The modules became public when the binary was split into this library and a thin binary, so rustdoc
// now flags the original doc comments' links to private items. `cargo docs` documents private items, so
// those links resolve.
#![allow(rustdoc::private_intra_doc_links)]

pub mod arithmetic;
pub mod hachi;
pub mod stream;
pub mod utils;
