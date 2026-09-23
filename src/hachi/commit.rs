//! # Commitment (paper, Section 4.1, Eq. (13)-(14))
//!
//! The witness is the coefficient vector $(f_p)$, $p \lt 2^\ell$, of the multilinear polynomial,
//! read as $2^r$ chunks $\mathbf f_i \in \mathbf R_q^{2^m}$: chunk $i$ is the block of
//! $2^{m+\alpha}$ consecutive coefficients starting at $i \cdot 2^{m+\alpha}$, and its element $j$
//! is the ring element whose coefficient vector is $(f_p)$ for $p \in \[(i 2^m + j) d, (i 2^m + j + 1) d)$.
//! In the coefficient index $p$ the low $\alpha$ bits are therefore the ring index, the next $m$
//! bits the element $j$ and the top $r$ bits the chunk $i$; the scheme pairs them with the
//! variables $x_{r+m}, \dots, x_{\ell-1}$, $x_r, \dots, x_{r+m-1}$ and $x_0, \dots, x_{r-1}$
//! respectively (0-based; see [`crate::hachi::prove`]).
//!
//! The commitment is the inner-outer commitment of Eq. (13)-(14):
//! $$\mathbf s_i = \mathbf G_{2^m}^{-1}(\mathbf f_i) \in \mathbf R_q^{\delta 2^m}, \qquad
//! \mathbf t_i = \mathbf A \mathbf s_i \in \mathbf R_q^{n}, \qquad
//! \hat{\mathbf t} = \mathbf G_{n 2^r}^{-1}(\mathbf t_1, \dots, \mathbf t_{2^r}) \in \mathbf R_q^{n\delta 2^r}, \qquad
//! \mathbf u = \mathbf B \hat{\mathbf t} \in \mathbf R_q^{n}.$$
//! When [`Parameters::decomp_witness`] is `false` the chunks are committed without decomposition,
//! $\mathbf t_i = \mathbf A \mathbf f_i$ with $\mathbf A \in \mathbf R_q^{n \times 2^m}$; this is
//! used for the next witness, whose entries are already digits (Section 4.5).
//!
//! Layouts (a [`PVec`] of elements of dimension $d$ stores coefficient $j$ of element $i$ at index
//! $i d + j$):
//! * `t` has $n 2^r$ elements, with $\mathbf t_i$ at elements $\[i n, (i+1) n)$;
//! * `t_hat` has $n \delta 2^r$ elements: the $\delta$ digits of element $e$ of `t` are its elements
//!   $e\delta, \dots, e\delta + \delta - 1$, least significant digit first, as produced by
//!   [`PVec::b_decomp_zq`] (which first maps each residue into the balanced window, see
//!   [`crate::arithmetic::utils::to_window`]); recomposition is the gadget $(1, b, \dots, b^{\delta-1})$;
//! * `u` has $n$ elements.
//!
//! Digits are short signed integers stored as wrapping `u64`; residues are stored in $\[0, q)$.
//! The commitment is deterministic (no randomness, not hiding) and the matrices are regenerated
//! from the seeds in [`Parameters`] on every call.

use crate::arithmetic::utils::Logarithm;
use crate::{arithmetic::poly_vec::PVec, stream::file_stream::U64FileStream};
use crate::arithmetic::poly_mat::PMat;
use crate::arithmetic::poly_mat_ntt::PMatNtt;
use crate::arithmetic::ring::Ring;

use crate::stream::Stream;

use crate::hachi::Hachi;
use crate::hachi::setup::Parameters;

#[cfg(feature = "verbose")]
use crate::utils::verbose::{progress_bar, tick_item};

/// A commitment together with the prover state needed to open it.
pub struct CommitmentWithState {
    /// Inner commitments $\mathbf t = (\mathbf t_1, \dots, \mathbf t_{2^r}) \in \mathbf R_q^{n 2^r}$,
    /// $\mathbf t_i = \mathbf A \mathbf s_i$ (prover state; $\mathbf t_i$ at elements $\[in, (i+1)n)$).
    pub t: PVec,
    /// $\hat{\mathbf t} = \mathbf G^{-1}(\mathbf t) \in \mathbf R_q^{n\delta 2^r}$, the decomposed inner
    /// commitments (prover state; it becomes part of the next witness).
    pub t_hat: PVec,
    /// Outer commitment $\mathbf u = \mathbf B \hat{\mathbf t} \in \mathbf R_q^{n}$ (Eq. (14)): the
    /// public commitment.
    pub u: PVec
}

/// Commitment to a witness of type `T`.
pub trait Commit<T> {
    /// Commit to `witness` under `params`, returning $\mathbf u$ together with the prover state
    /// $(\mathbf t, \hat{\mathbf t})$. The witness must provide at least $2^\ell$ coefficients in
    /// $\[0, q)$; only the first $2^\ell$ are used.
    fn commit(witness: T, params: &Parameters) -> CommitmentWithState;
}

/// Witness provided as a file stream of `u64` residues (see [`commit_stream`]).
/// The witness is a multilinear polynomial in $\ell$ variables over $\mathbb Z_q$, or equivalently
/// a multilinear polynomial in $\ell - \alpha$ variables over $\mathbf R_q$, each block of $d$
/// consecutive coefficients defining one ring element (layout in the module documentation).
impl Commit<&mut U64FileStream> for Hachi {
    #[time_graph::instrument]
    fn commit(witness: &mut U64FileStream, params: &Parameters) -> CommitmentWithState
    {
        commit_stream(witness, params)
    }
}

/// Witness provided as a slice of `u64` residues held in memory (see [`commit_slice`]); same
/// layout as the stream version. Used for the next witness $(\mathbf z^{\prime}, \mathbf r)$ inside
/// [`crate::hachi::prove`].
impl Commit<&[u64]> for Hachi {
    #[time_graph::instrument]
    fn commit(witness: &[u64], params: &Parameters) -> CommitmentWithState
    {
        commit_slice(witness, params)
    }
}

/// Witness provided as a vector over $\mathbf R_q$: its flat coefficient vector is committed as in
/// the slice version, so it must hold at least $2^\ell$ coefficients in total.
impl Commit<&PVec> for Hachi {
    #[time_graph::instrument]
    fn commit(witness: &PVec, params: &Parameters) -> CommitmentWithState {
        commit_slice(witness.slice(), params)
    }
}

/// Commitment for a witness read from a generic `u64` stream, one chunk $\mathbf f_i$ at a time.
///
/// For $i = 0, \dots, 2^r - 1$: read $\mathbf f_i$ ($2^{m+\alpha}$ coefficients), decompose it into
/// $\mathbf s_i$ if `params.decomp_witness`, and set $\mathbf t_i = \mathbf A \mathbf s_i$ (else
/// $\mathbf A \mathbf f_i$). Then $\hat{\mathbf t} = \mathbf G^{-1}(\mathbf t)$ and
/// $\mathbf u = \mathbf B \hat{\mathbf t}$, with $\mathbf B = \mathbf A$ when `params.reuse_mats`.
/// Products are computed in the NTT domain of $\mathbf R_q$ ([`Ring`] with `cyclotomic = true`,
/// $\mathbf A$ transformed once). Cost: $2^r$ products of an $n \times \delta 2^m$ matrix with a
/// vector plus one product of an $n \times n\delta 2^r$ matrix with a vector, i.e.
/// $n\delta(2^{r+m} + n 2^r)$ ring multiplications; the stream is read once and only one chunk
/// (and its $\delta$ digit vectors) is resident. Panics if the stream is shorter than $2^\ell$.
fn commit_stream(witness: &mut impl Stream<u64>, params: &Parameters) -> CommitmentWithState {
    #[cfg(feature = "verbose")]
    println!("\n==== Commit ====");

    // ensure the stream has sufficient length
    assert!(witness.length() >= 1 << params.l);

    // create the ring Zq[X]/(X^d+1)
    let ring = Ring::init(params.q, params.d, true);

    // create vectors for the commitment vectors
    let mut t = PVec::zero(params.n * (1 << params.r), params.d);
    let mut t_hat = PVec::zero(t.length() * params.delta, params.d);
    let mut u = PVec::zero(params.n, params.d);

    // --- inner commitment ---
    // sample the inner commitment matrix A and perform forward NTT.
    let mat_a = PMat::rand(params.n, params.width_a, params.d, params.q, params.a_seed);
    let mut mat_a_ntt = PMatNtt::zero(params.n, params.width_a, params.d);
    ring.mat_fwd_ntt(&mat_a, &mut mat_a_ntt);

    // create a vector for f_i and s_i
    let mut f_i = PVec::zero(1 << params.m, params.d);
    let mut s_i = PVec::zero(f_i.length() * params.delta, params.d);
    
    // iterate over 0..2^r
    for i in 0..1 << params.r {
        #[cfg(feature = "verbose")]
        progress_bar("Inner Commitment", i, 1 << params.r);

        // read the next chunk f_i
        witness.read(f_i.mut_slice());

        // if decomposing
        if params.decomp_witness {
            f_i.b_decomp_zq(params.q, params.b, params.delta, &mut s_i);
            ring.mat_mul_vec(&mat_a_ntt, &s_i, &mut t, i * params.n);
        }
        // if not decomposing
        else {
            ring.mat_mul_vec(&mat_a_ntt, &f_i, &mut t, i * params.n);
        };
    }

    // --- outer commitment ---
    // get outer commitment matrix
    let mat_b_ntt = {
        if params.reuse_mats { mat_a_ntt }
        else {
            let mat_b = PMat::rand(params.n, params.width_b, params.d, params.q, params.b_seed);
            let mut mat_b_ntt = PMatNtt::zero(params.n, params.width_b, params.d);
            ring.mat_fwd_ntt(&mat_b, &mut mat_b_ntt);
            mat_b_ntt
        }
    };

    // decompose the inner commitment
    t.b_decomp_zq(params.q, params.b, params.delta, &mut t_hat);

    // outer commitment u = B.t_hat over Rq (the lift to Zq[X] is recomputed in prove)
    ring.mat_mul_vec(&mat_b_ntt, &t_hat, &mut u, 0);

    #[cfg(feature = "verbose")]
    tick_item("Outer Commitment");

    #[cfg(feature = "verbose")]
    println!("==== Complete ====\n");

    CommitmentWithState { t, t_hat, u }
}

/// Commitment for a witness held in a `u64` slice: identical to [`commit_stream`], except that
/// chunk $i$ is copied from `witness[i << (m + alpha) .. (i + 1) << (m + alpha)]`. Panics if the
/// slice is shorter than $2^\ell$.
fn commit_slice(witness: &[u64], params: &Parameters) -> CommitmentWithState {
    #[cfg(feature = "verbose")]
    println!("\n==== Commit ====");

    // ensure the stream has sufficient length
    assert!(witness.len() >= 1 << params.l);

    // create the ring Zq[X]/(X^d+1)
    let ring = Ring::init(params.q, params.d, true);

    // log ring dimension
    let alpha = params.d.log();

    // create vectors for the commitment vectors
    let mut t = PVec::zero(params.n * (1 << params.r), params.d);
    let mut t_hat = PVec::zero(t.length() * params.delta, params.d);
    let mut u = PVec::zero(params.n, params.d);

    // --- inner commitment ---
    // sample the inner commitment matrix A and perform forward NTT.
    let mat_a = PMat::rand(params.n, params.width_a, params.d, params.q, params.a_seed);
    let mut mat_a_ntt = PMatNtt::zero(params.n, params.width_a, params.d);
    ring.mat_fwd_ntt(&mat_a, &mut mat_a_ntt);

    // create a vector for f_i and s_i
    let mut f_i = PVec::zero(1 << params.m, params.d);
    let mut s_i = PVec::zero(f_i.length() * params.delta, params.d);
    
    // iterate over 0..2^r
    for i in 0..1 << params.r {
        #[cfg(feature = "verbose")]
        progress_bar("Inner Commitment", i, 1 << params.r);

        // copy the next chunk f_i
        f_i.mut_slice().copy_from_slice(&witness[i << (params.m + alpha)..(i + 1 << (params.m + alpha))]);

        // if decomposing
        if params.decomp_witness {
            f_i.b_decomp_zq(params.q, params.b, params.delta, &mut s_i);
            ring.mat_mul_vec(&mat_a_ntt, &s_i, &mut t, i * params.n);
        }
        // if not decomposing
        else {
            ring.mat_mul_vec(&mat_a_ntt, &f_i, &mut t, i * params.n);
        };
    }

    // --- outer commitment ---
    // get outer commitment matrix
    let mat_b_ntt = {
        if params.reuse_mats { mat_a_ntt }
        else {
            let mat_b = PMat::rand(params.n, params.width_b, params.d, params.q, params.b_seed);
            let mut mat_b_ntt = PMatNtt::zero(params.n, params.width_b, params.d);
            ring.mat_fwd_ntt(&mat_b, &mut mat_b_ntt);
            mat_b_ntt
        }
    };

    // decompose the inner commitment
    t.b_decomp_zq(params.q, params.b, params.delta, &mut t_hat);

    // outer commitment u = B.t_hat over Rq (the lift to Zq[X] is recomputed in prove)
    ring.mat_mul_vec(&mat_b_ntt, &t_hat, &mut u, 0);

    #[cfg(feature = "verbose")]
    tick_item("Outer Commitment");

    #[cfg(feature = "verbose")]
    println!("==== Complete ====\n");

    CommitmentWithState { t, t_hat, u }
}
