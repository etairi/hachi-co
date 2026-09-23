// Doc comments in this file use `\_` for underscores and `\[` `\]` for brackets inside the
// $...$ KaTeX formulas: rustdoc's Markdown would otherwise pair underscores into emphasis (and
// brackets into links) and split the formula; the escapes reach KaTeX as plain `_`, `[`, `]`.

//! Prover utilities for a witness over $\mathbb{Z}\_q$ evaluated at a point over $\mathbb{Z}\_q$
//! (the base-field case of the paper's Section 3): the reduction of the evaluation claim to a
//! single ring element $Y \in \mathbf{R}\_q$ together with the first prover message $\mathbf{w}$
//! (paper, Section 3.1 and Section 4.2, Eq. (12) and (15)), and the assembly of the next witness
//! $(\mathbf z^{\prime}, \mathbf{r})$ for the ring-switched sumcheck (paper, Section 4.3).
//!
//! The committed polynomial $f \in \mathbb{Z}\_q^{\le 1}\[X\_1, \dots, X\_\ell\]$ has $2^\ell$
//! coefficients, streamed from a file as residues in $\[0, q)$. Coefficient position
//! $p \in \[0, 2^\ell)$ is read as $p = i \cdot 2^{m+\alpha} + j \cdot d + c$: chunk
//! $i \in \lbrace 0,1 \rbrace^r$ (high bits), ring element $j \in \lbrace 0,1 \rbrace^m$,
//! coefficient $c \in \[0, d)$ (low bits), so that chunk $i$ is the vector
//! $\mathbf{f}\_i = (f\_{i,j})\_j \in \mathbf{R}\_q^{2^m}$ of the paper's Eq. (12). The
//! variables are paired with these indices as: chunk $i$ with `x[0..r]`, element $j$ with
//! `x[r..r+m]`, ring coefficient $c$ with `x[r+m..l]`, always bit $k$ of the index with the
//! $k$-th variable of its group (the convention of [`crate::hachi::common`] and of the
//! verifier's reduction check; `main.rs` permutes its reference evaluation to match). With
//! $$b\_i = \prod\_{k \lt r} x\_{k}^{i\_k}, \qquad a\_j = \prod\_{k \lt m} x\_{r+k}^{j\_k}, \qquad w\_i = \mathbf a^\top \mathbf{f}\_i = \sum\_j a\_j f\_{i,j},$$
//! the evaluation claim becomes $y = \langle \mathrm{cf}(Y), \mathbf{v}\_x \rangle$ for
//! $Y = \sum\_i b\_i w\_i \in \mathbf{R}\_q$ and
//! $\mathbf{v}\_x = (\prod\_{k \lt \alpha} x\_{r+m+k}^{c\_k})\_{c \lt d}$, which is what
//! [`crate::hachi::verify`] checks.
//!
//! Notation follows the crate-level table; see [`crate::hachi::prover_utils`] for the storage
//! conventions.

#[cfg(feature = "verbose")]
use crate::utils::verbose::progress_bar;

use crate::stream::Stream;

use crate::arithmetic::ring::Ring;
use crate::arithmetic::poly_vec::PVec;
use crate::arithmetic::utils::{Logarithm, multi_lin_coeff_int};

use crate::hachi::setup::Parameters;

#[cfg_attr(feature = "stats", time_graph::instrument)]
/// Reduce the evaluation claim over $\mathbb{Z}\_q$ to the ring element $Y \in \mathbf{R}\_q$
/// and compute the first prover message $\mathbf{w} = (w\_i)\_{i \in \[2^r\]}$ in a single pass
/// over the witness stream (both need every coefficient, so they share one read).
///
/// With the notation of the module documentation, for every chunk $i$:
/// * `w[i]` accumulates $w\_i = \mathbf a^\top \mathbf{f}\_i = \sum\_{j \lt 2^m} a\_j f\_{i,j}$
///   (paper, Section 4.2, $w\_i = \mathbf a^\top \mathbf{G}\_{2^m} \mathbf{s}\_i$, which is the
///   same value since $\mathbf{G}\_{2^m}\mathbf{s}\_i = \mathbf{f}\_i$; the code multiplies the
///   undecomposed chunk by the scalars $a\_j$, see [`Ring::int_vec_mul_poly_vec`]);
/// * `y[0]` accumulates $\sum\_{j \lt 2^m} b\_i a\_j f\_{i,j} = b\_i w\_i$, where the scalar
///   $b\_i a\_j$ is read off `x[0..r+m]` at the index `i | (j << r)` (chunk bits low, element
///   bits high, as [`multi_lin_coeff_int`] pairs bit $k$ of the index with `x[k]`).
///
/// On return `y` holds $Y = \sum\_i b\_i w\_i = \sum\_{i,j} b\_i a\_j f\_{i,j}$, the base-field
/// ($k = 1$) instance of the ring element $Y$ of the paper's Section 3.1, and `w` holds
/// $\mathbf{w}$, which [`crate::hachi::prover_utils::rq::commit_w_and_lift`] then commits to.
///
/// Preconditions: `ring` is the cyclotomic ring $\mathbf{R}\_q$ (only scalar-by-polynomial
/// products are taken, so no reduction modulo $X^d + 1$ actually occurs); `y` has one element
/// and `w` has $2^r$ elements, both zero on entry since the products are accumulated; `x` has
/// at least $r + m$ entries in $\[0, q)$; the stream holds at least $2^\ell$ residues in
/// $\[0, q)$. All coefficients of `y` and `w` are residues in $\[0, q)$.
///
/// Cost: one streaming pass, $2^{r+m} \cdot 2d$ scalar multiplications modulo $q$ (with
/// 128-bit intermediates) plus $2^{r+m}$ evaluations of [`multi_lin_coeff_int`] of $O(r + m)$
/// each; memory $O(2^m d)$ for one chunk.
pub fn compute_y_and_w(
        witness: &mut impl Stream<u64>,
        params: &Parameters,
        x: &[u64],
        ring: &Ring,
        y: &mut PVec,
        w: &mut PVec
) {
    assert_eq!(1, y.length());
    assert_eq!(1 << params.r, w.length());

    let y = y.mut_element(0);
    witness.reset();

    // create a vector for reading in the witness
    let mut f_i = PVec::zero(1 << params.m, params.d);

    // pre-process the 2^m coefficients a^T
    let mut a = vec![0u64; 1 << params.m];

    for i in 0..1 << params.m {
        a[i] = multi_lin_coeff_int(&x[params.r..params.r + params.m], i, params.m, params.q)
    }

    // read in the whole witness in 2^r chunks of length 2^{m + alpha}
    for i in 0..1 << params.r {
        #[cfg(feature = "verbose")]
        progress_bar("Compute y and compute w", i, 1 << params.r);

        // since the polynomial is over integers, do not need to do anything more to map each f_i to Rq elements.
        witness.read(f_i.mut_slice());

        // compute w_i
        ring.int_vec_mul_poly_vec(&a, &f_i, w, i);
        
        // iterate over each f_ij
        for j in 0..1 << params.m {
            // the index in {0,1}^{r+m}: chunk index i <-> x[0..r], element index j <-> x[r..r+m]
            // (the same convention as w_i = a^T f_i above and as the verification matrix M)
            let index = i | (j << params.r);

            // get the coefficient for this index
            let coeff = multi_lin_coeff_int(&x[0..params.r + params.m], index, params.r + params.m, params.q);

            // get the current ring element
            let f_ij = f_i.element(j);

            // add coeff * f_ij to y
            ring.int_mul_poly(coeff, f_ij, y);
        }
    }
}

/// Assemble the next witness $\widetilde{w} = (\mathbf z^{\prime}, \mathbf{r})$ of the
/// ring-switched relation $\mathbf{M}\mathbf z^{\prime} = \mathbf{y} + (X^d + 1)\mathbf{r}$
/// (paper, Section 4.3, Eq. (21)) as one flat vector of short digits, zero-padded to a
/// power-of-two length.
///
/// $\mathbf z^{\prime} = (\hat{\mathbf{w}}, \hat{\mathbf{t}}, \hat{\mathbf{z}})$ is the
/// witness of the paper's Eq. (20) and $\mathbf{r}$ collects, row block by row block of the
/// matrix of Eq. (20), the quotients by $X^d + 1$ of the five verification equations computed
/// over $\mathbb{Z}\_q\[X\]$, each gadget-decomposed into `params.delta` digits (the "hidden
/// gadget decomposition of $\mathbf{r}$" of Section 4.3). The layout, in ring elements of
/// dimension $d$ (coefficient $c$ of element $e$ is at index $ed + c$ of the returned vector):
///
/// | offset (elements) | length (elements) | content |
/// |---|---|---|
/// | $0$ | $2^r\delta$ | `w_hat` $= \hat{\mathbf{w}} = \mathbf G^{-1}(\mathbf{w})$, digits |
/// | $2^r\delta$ | $n 2^r \delta$ | `t_hat` $= \hat{\mathbf{t}} = \mathbf G^{-1}(\mathbf{t})$, digits |
/// | $(n+1) 2^r \delta$ | $2^m\delta\tau$ | $\hat{\mathbf{z}} = \mathbf J^{-1}(\mathbf{z})$: `params.delta_z` $= \tau$ digits per element of `z` |
/// | $\mu$ | $n\delta$ | $\mathbf G^{-1}$(`v_quo`): quotient of $\mathbf{D}\hat{\mathbf{w}}$ (rows of $\mathbf{D}$) |
/// | $\mu + n\delta$ | $n\delta$ | $\mathbf G^{-1}$(`u_quo`): quotient of $\mathbf{B}\hat{\mathbf{t}}$ (rows of $\mathbf{B}$) |
/// | $\mu + 2n\delta$ | $\delta$ | zero: $\mathbf b^\top \mathbf{G}\_{2^r}\hat{\mathbf{w}} = \mathbf b^\top\mathbf{w}$ has no quotient, the $b\_i$ being scalars |
/// | $\mu + (2n+1)\delta$ | $\delta$ | $\mathbf G^{-1}$(`c_i_w_i_quo`): quotient of $\sum\_i c\_i w\_i$ in the row $(\mathbf c^\top \otimes \mathbf{G}\_1)\hat{\mathbf{w}} - \mathbf a^\top \mathbf{G}\_{2^m}\mathbf{J}\_{2^m}\hat{\mathbf{z}}$ ($\mathbf a^\top\mathbf{z}$ has no quotient) |
/// | $\mu + (2n+2)\delta$ | $n\delta$ | $\mathbf G^{-1}$(`c_i_t_i_quo` $-$ `mat_a_z_quo` $\bmod q$): quotient of $\sum\_i c\_i \mathbf{t}\_i - \mathbf{A}\mathbf{z}$ (rows of $\mathbf{A}$) |
/// | $\mu + (3n+2)\delta$ | up to the next power of two | zero padding |
///
/// Here $\mu = 2^r\delta + n2^r\delta + 2^m\delta\tau$ is the width of the matrix of Eq. (20)
/// and $3n + 2$ is its number of rows, so $\mathbf{r}$ occupies $(3n+2)\delta$ elements. The
/// same layout, with $-(X^d+1)\mathbf{G}$ in the $\mathbf{r}$ columns, is used by
/// [`crate::hachi::common::form_m_alpha`] to build $\mathbf{M}$ evaluated at $\alpha$, and the
/// same offsets are recomputed by [`crate::hachi::verify`].
///
/// Every entry is a digit: the blocks of $\mathbf z^{\prime}$ and $\mathbf{r}$ are balanced
/// base-$b$ digits in $\[-b/2, b/2 - 1\]$ as wrapping-signed `u64`, and the zero row and the
/// padding are $0$, so the range check $F\_0$ of [`crate::hachi::prover_utils::sumcheck`] runs
/// over the whole vector. `z` is decomposed with [`PVec::b_decomp`] (no residue-to-window
/// mapping: its coefficients are already wrapping-signed and bounded by `params.z_bound`, which
/// lies inside the $\tau$-digit window); all quotients are residues in $\[0, q)$ and are
/// decomposed with [`PVec::b_decomp_zq`].
///
/// Returns `(num_vars, mu, rows, witness)`: the number $v$ of variables of the padded witness
/// ($2^{v}$ is the smallest power of two $\ge (\mu + (3n+2)\delta) d$), $\mu$, the number of
/// rows $3n + 2$ of $\mathbf{M}$ and the witness itself. The caller uses
/// `(mu + rows * params.delta) * params.d` as the length of the unpadded part.
///
/// Preconditions: the lengths asserted at the top of the function (`w_hat`: $2^r\delta$,
/// `t_hat`: $n2^r\delta$, `z`: $2^m\delta$, `c_i_w_i_quo`: $1$, every other quotient: $n$
/// elements).
pub fn form_next_witness(
        params: &Parameters,
        w_hat: &PVec,
        t_hat: &PVec,
        z: &PVec,
        v_quo: &PVec,
        u_quo: &PVec,
        c_i_w_i_quo: &PVec,
        c_i_t_i_quo: &PVec,
        mat_a_z_quo: &PVec

    ) -> (usize, usize, usize, Vec<u64>) {
        assert_eq!((1 << params.r) * params.delta, w_hat.length());
        assert_eq!(params.n * (1 << params.r) * params.delta, t_hat.length());
        assert_eq!((1 << params.m) * params.delta, z.length());
        assert_eq!(params.n, v_quo.length());
        assert_eq!(params.n, u_quo.length());
        assert_eq!(1, c_i_w_i_quo.length());
        assert_eq!(params.n, c_i_t_i_quo.length());
        assert_eq!(params.n, mat_a_z_quo.length());

        // length of the z part of new witness (width of M)
        let mu = w_hat.length() + t_hat.length() + z.length() * params.delta_z;
        
        // length of r part of new witness (height of M * decomposition expansion)
        let n = (v_quo.length() + u_quo.length() + 1 + c_i_w_i_quo.length() + c_i_t_i_quo.length()) * params.delta;

        // new witness must have power of 2 length, so we pad (mu + n) to next power of 2
        let num_vars = ((mu + n) * params.d).log();
        
        // create a vector to store the next witness
        let mut witness = vec![0u64; 1 << num_vars];

        // copy in w_hat
        let mut cur = 0;
        witness[cur..cur + w_hat.length() * params.d].copy_from_slice(w_hat.slice());
        cur += w_hat.length() * params.d;

        // copy in t_hat
        witness[cur..cur + t_hat.length() * params.d].copy_from_slice(t_hat.slice());
        cur += t_hat.length() * params.d;

        // decompose z into z_hat and copy in: z is a wrapping-signed vector with ||z||_inf <= z_bound,
        // which lies inside the delta_z-digit balanced window, so no residue-to-window mapping is applied
        let mut z_hat = PVec::zero(z.length() * params.delta_z, params.d);
        z.b_decomp(params.b, params.delta_z, &mut z_hat);
        witness[cur..cur + z_hat.length() * params.d].copy_from_slice(z_hat.slice());
        cur += z_hat.length() * params.d;

        // sense check
        assert_eq!(cur, mu * params.d);

        // decompose and copy in quotient for commitment v=D.w_hat
        let mut v_quo_hat = PVec::zero(v_quo.length() * params.delta, params.d);
        v_quo.b_decomp_zq(params.q, params.b, params.delta, &mut v_quo_hat);
        witness[cur..cur + v_quo_hat.length() * params.d].copy_from_slice(v_quo_hat.slice());
        cur += v_quo_hat.length() * params.d;

        // decompose and copy in quotient for commitment u=B.t_hat
        let mut u_quo_hat = PVec::zero(u_quo.length() * params.delta, params.d);
        u_quo.b_decomp_zq(params.q, params.b, params.delta, &mut u_quo_hat);
        witness[cur..cur + u_quo_hat.length() * params.d].copy_from_slice(u_quo_hat.slice());
        cur += u_quo_hat.length() * params.d;

        // third row has 0 quotient 
        cur += params.d * params.delta;

        // decompose and copy in quotient for sum_i c_i_w_i
        let mut c_i_w_i_quo_hat = PVec::zero(c_i_w_i_quo.length() * params.delta, params.d);
        c_i_w_i_quo.b_decomp_zq(params.q, params.b, params.delta, &mut c_i_w_i_quo_hat);
        witness[cur..cur + c_i_w_i_quo_hat.length() * params.d].copy_from_slice(c_i_w_i_quo_hat.slice());
        cur += c_i_w_i_quo_hat.length() * params.d;

        // decompose and copy in sum c_i_i_t_i - A.z
        assert_eq!(c_i_t_i_quo.length(), mat_a_z_quo.length());
        let mut quo = PVec::zero(c_i_t_i_quo.length(), params.d);
        let coeffs = quo.mut_slice();

        for i in 0..coeffs.len() {
            let a = c_i_t_i_quo.slice()[i];
            let b = mat_a_z_quo.slice()[i];
            let mut quo_coeff = a as i64 - b as i64;

            if quo_coeff < 0 {
                quo_coeff = params.q as i64 + quo_coeff;
            }

            coeffs[i] = quo_coeff as u64;
        }

        let mut quo_hat = PVec::zero(quo.length() * params.delta, params.d);
        quo.b_decomp_zq(params.q, params.b, params.delta, &mut quo_hat);
        witness[cur..cur + quo_hat.length() * params.d].copy_from_slice(quo_hat.slice());
        cur += quo_hat.length() * params.d;

        // sense check
        assert_eq!((mu + n) * params.d, cur);

        #[cfg(feature = "verbose")]
        println!("Formed next witness ({} variables)", num_vars);

        (num_vars, mu, n/params.delta, witness)
}