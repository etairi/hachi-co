//! Arithmetic layer: fields, the NTT-based ring, polynomial vectors and matrices, sparse
//! challenges, gadget decomposition, the Fiat-Shamir transcript and generic sumcheck helpers.
//!
//! These are the objects of the paper's Section 2.1 on which the scheme ([`crate::hachi`]) is
//! built: the ring $\mathbf R_q = \mathbb Z_q\[X\]/(X^d + 1)$ with $d = 2^\alpha$ and a prime
//! $q \equiv 5 \pmod 8$; the polynomial ring $\mathbb Z_q\[X\]$, to which the verification
//! equations are lifted by the ring switching of Section 4.3; the base-$b$ gadget decomposition
//! $\mathbf G^{-1}$; the sparse challenge set $\mathcal C$ of Section 4.2; and the extension field
//! $\mathbb F_{q^4}$ over which the sumchecks of Section 4.3 run.
//!
//! ## Modules
//!
//! * [`field`]: the tower $\mathbb F_q \subset \mathbb F_{q^2} \subset \mathbb F_{q^4}$ as arkworks types.
//! * [`ring`]: multiplication over $\mathbf R_q$ and over $\mathbb Z_q\[X\]$ through the 64-bit
//!   native NTT of tfhe-ntt, and the direct products by sparse challenges.
//! * [`poly`]: evaluation at $\alpha \in \mathbb F_{q^4}$ and division by $X^d + 1$ of a single
//!   polynomial given as a slice.
//! * [`poly_vec`], [`poly_mat`]: vectors and matrices over $\mathbf R_q$ in coefficient form,
//!   and the gadget decomposition of vectors.
//! * [`poly_vec_ntt`], [`poly_mat_ntt`]: the same objects in multi-prime NTT form.
//! * [`poly_chal`]: sparse challenges $c \in \mathcal C$.
//! * [`sumcheck`]: the sumcheck round message and the folding of evaluation tables.
//! * [`fs`]: the Fiat-Shamir transcript.
//! * [`utils`]: integer helpers (balanced digits and the decomposition window, sampling,
//!   multilinear monomials, lifts into $\mathbb F_{q^4}$).
//!
//! ## Representation conventions
//!
//! * A coefficient of $\mathbb Z_q$ is a [`CoeffType`] (`u64`) holding the residue in $\[0, q)$.
//! * A *short* integer (a gadget digit, a coefficient of the response $\mathbf z$, or the product
//!   of a challenge with a short polynomial) is a `u64` holding its two's-complement value: a
//!   negative $-a$ is stored as $2^{64} - a$ and all arithmetic on it is wrapping. Each function
//!   states which of the two conventions it expects; a vector never mixes them, and the types do
//!   not record which one they hold.
//! * A polynomial is a slice of coefficients with the coefficient of $X^j$ at index $j$; vectors
//!   of polynomials are flat, element after element (see [`poly_vec::PVec`]).
//! * NTT-domain residues are [`NttType`]s: `u32` on the default tfhe-ntt `Plan32` path (five
//!   primes below $2^{30}$) and `u64` with the `nightly` feature (`Plan52`, three primes below
//!   $2^{50}$); see [`ring`].
//! * The sumcheck field is [`ExtField`] $= \mathbb F_{q^4}$.

pub mod field;
pub mod poly;
pub mod poly_vec;
pub mod poly_vec_ntt;
pub mod poly_mat;
pub mod poly_mat_ntt;
pub mod poly_chal;
pub mod ring;
pub mod sumcheck;
pub mod fs;
pub mod utils;

/// Storage type of a polynomial coefficient: a residue of $\mathbb Z_q$ in $\[0, q)$, or a
/// wrapping two's-complement short integer (see the module documentation).
pub type CoeffType = u64;

/// Storage type of an NTT-domain residue on the default tfhe-ntt `Plan32` path (five primes
/// below $2^{30}$), selected when the `nightly` feature is off.
#[cfg(not(feature = "nightly"))]
pub type NttType = u32;
/// Storage type of an NTT-domain residue on the tfhe-ntt `Plan52` path (three primes below
/// $2^{50}$, AVX-512 IFMA), selected by the `nightly` feature.
#[cfg(feature = "nightly")]
pub type NttType = u64;

use crate::arithmetic::field::Fq4;
/// The field of the sumcheck challenges and round messages, $\mathbb F_{q^4}$ (the paper's
/// extension degree $k = 4$, Section 5.4; not `params.k`), see [`field`].
pub type ExtField = Fq4;
