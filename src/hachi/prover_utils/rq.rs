// Doc comments in this file use `\_` for underscores and `\[` `\]` for brackets inside the
// $...$ KaTeX formulas: rustdoc's Markdown would otherwise pair underscores into emphasis (and
// brackets into links) and split the formula; the escapes reach KaTeX as plain `_`, `[`, `]`.

//! Prover utilities that are generic over the ring $\mathbf{R}\_q = \mathbb{Z}\_q\[X\]/(X^d+1)$:
//! sampling of the commitment matrices (paper, Section 4.1 and 4.2), the commitment to the first
//! prover message $\mathbf{w}$ (paper, Section 4.2, Eq. (16)), the folded response $\mathbf{z}$
//! (paper, Fig. 3) and the lifting of ring products from $\mathbf{R}\_q$ to $\mathbb{Z}\_q\[X\]$
//! that the ring switching of the paper's Section 4.3 needs.
//!
//! Two [`Ring`]s are in play. The *cyclotomic* ring (`Ring::init(q, d, true)`) computes in
//! $\mathbf{R}\_q$ with inputs and outputs of dimension $d$. The *full* ring
//! (`Ring::init(q, d, false)`) computes in $\mathbb{Z}\_q\[X\]$: inputs of dimension $d$, outputs
//! of dimension $2d$, on an NTT plan of size $2d$ that the product of two polynomials of degree
//! below $d$ never wraps (paper, Section 5.4). A product $p$ computed in the full ring is then
//! split by [`PVec::cyclotomic_div`] into
//! $$p = p\_{\mathrm{rem}} + (X^d + 1) \cdot p\_{\mathrm{quo}}, \qquad \deg p\_{\mathrm{rem}} \lt d, \quad \deg p\_{\mathrm{quo}} \lt d,$$
//! where $p\_{\mathrm{rem}}$ is the value of the product in $\mathbf{R}\_q$ and
//! $p\_{\mathrm{quo}}$ is the quotient that the next witness carries as (part of) the vector
//! $\mathbf{r}$ of the relation $\mathbf{M}\mathbf z^{\prime} = \mathbf{y} + (X^d + 1)\mathbf{r}$
//! (paper, Section 4.3).
//!
//! Layout: a [`PVec`] of $n$ elements of dimension $d$ stores coefficient $j$ of element $i$ at
//! index `i * d + j`. Coefficients of committed values are residues in $\[0, q)$ as `u64`;
//! digits and the response $\mathbf{z}$ are wrapping two's-complement `u64` (see the crate-level
//! documentation).

#[cfg(feature = "verbose")]
use crate::utils::verbose::{progress_bar, tick_item};

use crate::stream::Stream;

use crate::arithmetic::poly_mat_ntt::PMatNtt;
use crate::arithmetic::poly_mat::PMat;
use crate::arithmetic::poly_vec::PVec;
use crate::arithmetic::poly_chal::PChal;
use crate::arithmetic::ring::Ring;

use crate::hachi::setup::Parameters;

/// Sample the commitment matrices $\mathbf{A}$, $\mathbf{B}$ and $\mathbf{D}$ (paper, Section
/// 4.1 and 4.2) from the seeds in `params` and return each together with its NTT form, in the
/// order `(A, A_ntt, B, B_ntt, D, D_ntt)`.
///
/// All three have `params.n` rows; the widths are `params.width_a` ($2^m\delta$ when the witness
/// is decomposed, $2^m$ otherwise), `params.width_b` ($n 2^r \delta$) and `params.width_d`
/// ($2^r\delta$), matching $\mathbf{A} \in \mathbf{R}\_q^{n\_A \times \delta 2^m}$,
/// $\mathbf{B} \in \mathbf{R}\_q^{n\_B \times n\_A \delta 2^r}$ and
/// $\mathbf{D} \in \mathbf{R}\_q^{n\_D \times \delta 2^r}$ of the paper with
/// $n\_A = n\_B = n\_D = n$. Coefficients are uniform residues in $\[0, q)$ expanded from the
/// seed by ChaCha12 ([`PMat::rand`]), so [`crate::hachi::commit`] and [`crate::hachi::verify`]
/// regenerate the same matrices from the same seeds.
///
/// When `params.reuse_mats` is set (all three widths are equal; setup then also makes the three
/// seeds equal) only $\mathbf{D}$ is sampled and $\mathbf{A}$, $\mathbf{B}$ are clones of it, so
/// one matrix serves all commitments (paper, Section 5.4).
///
/// `ring` must be the full $\mathbb{Z}\_q\[X\]$ ring: the NTT forms are allocated on the plan of
/// size $2d$ (`2 * params.d`), as required by the $\mathbb{Z}\_q\[X\]$ products in
/// [`commit_w_and_lift`] and in [`crate::hachi::prove`]. Cost: one forward NTT per matrix entry.
pub fn sample_matrices(params: &Parameters, ring: &Ring) -> (PMat, PMatNtt, PMat, PMatNtt, PMat, PMatNtt) {
    // Sample D and NTT
    let mat_d = PMat::rand(params.n, params.width_d, params.d, params.q, params.d_seed);
    let mut mat_d_ntt = PMatNtt::zero(mat_d.height(), mat_d.width(), 2 * params.d);
    ring.mat_fwd_ntt(&mat_d, &mut mat_d_ntt);

    // If reusing matrices, copy into B and A
    if params.reuse_mats {
        let mat_b = mat_d.clone();
        let mat_b_ntt = mat_d_ntt.clone();
        let mat_a = mat_d.clone();
        let mat_a_ntt = mat_d_ntt.clone();
        
        (mat_a, mat_a_ntt, mat_b, mat_b_ntt, mat_d, mat_d_ntt)
    }
    // If not reusing matrices, sample again
    else {
        // Sample B and NTT
        let mat_b = PMat::rand(params.n, params.width_b, params.d, params.q, params.b_seed);
        let mut mat_b_ntt = PMatNtt::zero(mat_b.height(), mat_b.width(), 2 * params.d);
        ring.mat_fwd_ntt(&mat_b, &mut mat_b_ntt);

        // Sample A and NTT
        let mat_a = PMat::rand(params.n, params.width_a, params.d, params.q, params.a_seed);
        let mut mat_a_ntt = PMatNtt::zero(mat_a.height(), mat_a.width(), 2 * params.d);
        ring.mat_fwd_ntt(&mat_a, &mut mat_a_ntt);

        (mat_a, mat_a_ntt, mat_b, mat_b_ntt, mat_d, mat_d_ntt)
    }   
}

/// Commit to the first prover message $\mathbf{w} = (w\_1, \dots, w\_{2^r})$ and lift the
/// commitment from $\mathbf{R}\_q$ to $\mathbb{Z}\_q\[X\]$ (paper, Section 4.2, Eq. (16)).
///
/// Computes, in this order:
/// * `w_hat` $= \hat{\mathbf{w}} = \mathbf G^{-1}\_{2^r}(\mathbf{w})$, the balanced base-$b$
///   decomposition of `w` into `params.delta` digits per element ([`PVec::b_decomp_zq`], which
///   first maps each residue into the balanced-digit window). Element $i\delta + k$ of `w_hat`
///   is the $k$-th digit polynomial of $w\_i$; every coefficient is a wrapping-signed digit in
///   $\[-b/2, b/2 - 1\]$.
/// * the product $\mathbf{D}\hat{\mathbf{w}}$ over $\mathbb{Z}\_q\[X\]$ (dimension $2d$, via
///   [`Ring::mat_mul_vec`] on the plan of `mat_d_ntt`), then its division by $X^d + 1$,
///   $$\mathbf{D}\hat{\mathbf{w}} = \mathbf{v} + (X^d + 1) \mathbf{v}\_{\mathrm{quo}},$$
///   so that `v` $= \mathbf{v} = \mathbf{D}\hat{\mathbf{w}} \in \mathbf{R}\_q^{n}$ is the
///   commitment sent to the verifier and `v_quo` $= \mathbf{v}\_{\mathrm{quo}}$ is the quotient,
///   which [`crate::hachi::prover_utils::zq_zq::form_next_witness`] decomposes into the first
///   block of the vector $\mathbf{r}$ of the next witness.
///
/// Preconditions: `ring` is the full $\mathbb{Z}\_q\[X\]$ ring and `mat_d_ntt` was built on it
/// by [`sample_matrices`]; `w` has $2^r$ elements with coefficients in $\[0, q)$; `w_hat` has
/// $2^r\delta$ elements and `v`, `v_quo` have `params.n` elements, all of dimension $d$. The
/// three outputs are overwritten, not accumulated; coefficients of `v` and `v_quo` are residues
/// in $\[0, q)$.
pub fn commit_w_and_lift(
    params: &Parameters, ring: &Ring, mat_d_ntt: &PMatNtt, 
    w: &PVec, w_hat: &mut PVec, v: &mut PVec, v_quo: &mut PVec
) {
    // decompose w
    w.b_decomp_zq(params.q, params.b, params.delta, w_hat);

    // multiply, v = D.w_hat over Zq[X]
    let mut v_full = PVec::zero(params.n, 2 * params.d);
    ring.mat_mul_vec(&mat_d_ntt, &w_hat, &mut v_full, 0);

    // cyclotomic reduction    
    v_full.cyclotomic_div(params.q, v_quo, v);

    #[cfg(feature = "verbose")]
    tick_item("Commit to w and lift");
}

#[cfg_attr(feature = "stats", time_graph::instrument)]
/// Compute the folded response
/// $\mathbf{z} = \sum\_{i=1}^{2^r} c\_i \mathbf{s}\_i \in \mathbf{R}\_q^{2^m\delta}$ (paper,
/// Section 4.2 and Fig. 3), streaming the witness once more.
///
/// For each chunk $i \in \[2^r\]$ the $2^m$ ring elements $\mathbf{f}\_i$ are read from
/// `witness` and, with `params.decomp_witness` set, decomposed into the digits
/// $\mathbf{s}\_i = \mathbf G^{-1}\_{2^m}(\mathbf{f}\_i)$ exactly as [`crate::hachi::commit`]
/// did; then $c\_i \mathbf{s}\_i$ is added to `z` in $\mathbf{R}\_q$
/// ([`Ring::chal_mul_poly_vec`]). The product uses the sparse form of the challenge
/// ($\omega$ = `params.k` nonzero coefficients $\pm 1$) and wrapping arithmetic without
/// reduction modulo $q$, so the coefficients of `z` are the exact integers
/// $\sum\_i c\_i \mathbf{s}\_i$, stored as wrapping-signed `u64`.
///
/// Norm check: the function asserts $\lVert \mathbf{z} \rVert\_\infty \le$ `params.z_bound`
/// (the paper's $\beta$; Fig. 3 aborts in this case). The bound is heuristic: `z_bound` is the
/// largest value representable with `params.delta_z` balanced base-$b$ digits, chosen so that
/// [`crate::hachi::prover_utils::zq_zq::form_next_witness`] can decompose `z` with `delta_z`
/// digits, not a bound derived from the parameters. The worst case is
/// $\lVert \mathbf{z} \rVert\_\infty \le 2^r \omega b/2$ (each term satisfies
/// $\lVert c\_i \mathbf{s}\_i \rVert\_\infty \le \lVert c\_i \rVert\_1 \lVert \mathbf{s}\_i \rVert\_\infty$),
/// which exceeds `z_bound` $= 30583$ as soon as $r \ge 8$ with the default $\omega = 16$,
/// $b = 16$; an honest $\mathbf{z}$ is a sum of $2^r\omega$ signed digits per coefficient and is
/// expected (heuristically, as a random walk) to stay far below the worst case. The code
/// verifies nothing beyond the assertion.
///
/// Preconditions: `ring` is the cyclotomic ring; `z` has $2^m\delta$ elements and is zero on
/// entry (the products are accumulated into it); `challenges` has $2^r$ entries; witness
/// coefficients are residues in $\[0, q)$. With `params.decomp_witness` unset the code folds
/// the chunks themselves, $\mathbf{z} = \sum\_i c\_i \mathbf{f}\_i$; this branch is not reached
/// from [`crate::hachi::prove`], which always decomposes, and it contradicts the length
/// assertion on `z` at the top of the function ([`Ring::chal_mul_poly_vec`] requires `z` to
/// have as many elements as $\mathbf{f}\_i$, i.e. $2^m$ rather than $2^m\delta$), so it panics
/// if taken.
///
/// Cost: $2^r \cdot 2^m\delta \cdot \omega \cdot d$ coefficient additions plus one
/// decomposition of the whole witness.
pub fn compute_z(
        witness: &mut impl Stream<u64>,
        params: &Parameters,
        ring: &Ring,
        challenges: &Vec<PChal>,
        z: &mut PVec
    ) {
        assert_eq!((1 << params.m) * params.delta, z.length());
        assert_eq!(1 << params.r, challenges.len());
        witness.reset();

        // create a vector for f_i and s_i
        let mut f_i = PVec::zero(1 << params.m, params.d);
        let mut s_i = PVec::zero(f_i.length() * params.delta, params.d);
        
        // iterate over 0..2^r
        for i in 0..1 << params.r {
            #[cfg(feature = "verbose")]
            progress_bar("Computing response z", i, 1 << params.r);

            // read the next chunk f_i
            witness.read(f_i.mut_slice());

            // if decomposing
            if params.decomp_witness {
                f_i.b_decomp_zq(params.q, params.b, params.delta, &mut s_i);
                ring.chal_mul_poly_vec(&challenges[i], &s_i, z, 0);
            }
            // if not decomposing
            else {
                ring.chal_mul_poly_vec(&challenges[i], &f_i, z, 0);
            };
        }

        // Check the norm of z
        let mut min = 0;
        let mut max = 0;
        
        for x in z.slice() {
            if (*x as i64) > max { max = *x as i64 };
            if (*x as i64) < min { min = *x as i64 }
        }

        assert!(min >= - (params.z_bound as i64));
        assert!(max <= params.z_bound as i64);

        #[cfg(feature = "verbose")]
        tick_item("z is within heuristic bound");
}

/// Lift the inner-commitment side of the last verification equation of the paper's Eq. (20)
/// from $\mathbf{R}\_q$ to $\mathbb{Z}\_q\[X\]$: compute
/// $(\mathbf c^\top \otimes \mathbf{I}\_n)\mathbf{t} = \sum\_{i=1}^{2^r} c\_i \mathbf{t}\_i$
/// over $\mathbb{Z}\_q\[X\]$ and split it as
/// $$\sum\_i c\_i \mathbf{t}\_i = \mathrm{rem} + (X^d + 1) \mathrm{quo}.$$
///
/// By Eq. (19), $\sum\_i c\_i \mathbf{t}\_i = \sum\_i c\_i \mathbf{A}\mathbf{s}\_i = \mathbf{A}\mathbf{z}$
/// in $\mathbf{R}\_q$, so `rem` equals the remainder of $\mathbf{A}\mathbf{z}$ computed over
/// $\mathbb{Z}\_q\[X\]$ (the prover asserts this), and the difference of the two quotients is
/// the $\mathbf{r}$-block of that row of the next witness (see
/// [`crate::hachi::prover_utils::zq_zq::form_next_witness`]). The paper writes this term as
/// $(\mathbf c^\top \otimes \mathbf{G}\_{n\_A})\hat{\mathbf{t}}$; the code uses the
/// undecomposed $\mathbf{t}$, which is the same value since
/// $\mathbf{G}\hat{\mathbf{t}}\_i = \mathbf{t}\_i$.
///
/// Layout: `t` has $n 2^r$ elements with row $j$ of $\mathbf{t}\_i$ at element $in + j$ (the
/// order in which [`crate::hachi::commit`] fills it); `quo` and `rem` have $n$ elements and are
/// overwritten. Each product $c\_i t\_{i,j}$ is formed with [`Ring::chal_mul_poly`] in the full
/// ring (dimension $2d$, reduced modulo $q$), so all coefficients are residues in $\[0, q)$.
///
/// Preconditions: `ring` is the full $\mathbb{Z}\_q\[X\]$ ring; `challenges` has $2^r$ entries;
/// `t` coefficients are residues in $\[0, q)$. Cost: $2^r n \omega d$ coefficient operations.
pub fn lift_c_i_t_i(
        params: &Parameters,
        ring: &Ring,
        challenges: &Vec<PChal>,
        t: &PVec,
        quo: &mut PVec,
        rem: &mut PVec
    ) {
        assert_eq!(1 << params.r, challenges.len());
        assert_eq!(params.n * (1 << params.r), t.length());
        assert_eq!(params.n, quo.length());
        assert_eq!(params.n, rem.length());

        let mut prod_full = PVec::zero(params.n, 2 * params.d);
        
        // iterate over 0..2^r
        for i in 0..1 << params.r {
            // iterate over height of matrix
            for j in 0..params.n {
                // multiply c_i by t_i,j and store in out[j]
                ring.chal_mul_poly(&challenges[i], t.element(i * params.n + j), prod_full.mut_element(j));
            }
        }

        // cyclotomic reduction
        prod_full.cyclotomic_div(params.q, quo, rem);
}

