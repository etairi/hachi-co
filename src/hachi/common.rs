//! # The verification matrix $\mathbf M_\alpha$ (paper, Section 4.3)
//!
//! After ring switching, the next witness $(\mathbf z^{\prime}, \mathbf r)$ produced by the prover
//! satisfies $\mathbf M^{\prime} \mathbf z^{\prime} = \mathbf y + (X^d+1)\mathbf G_{3n+2}\mathbf r$ over
//! $\mathbb Z_q^{\lt d}\[X\]$, where $\mathbf M^{\prime}$ is the matrix of Eq. (20) (with the response
//! decomposed, $\mathbf z^{\prime} = (\hat{\mathbf w}, \hat{\mathbf t}, \hat{\mathbf z})$) and $\mathbf r$
//! holds the gadget-decomposed quotients. Equivalently
//! $$\mathbf M \begin{bmatrix} \mathbf z^{\prime} \cr \mathbf r \end{bmatrix} = \mathbf y, \qquad
//! \mathbf M = \[\mathbf M^{\prime} \mid -(X^d+1)\mathbf G_{3n+2}\].$$
//! This module builds the table of $\tilde M_\alpha$: every entry of $\mathbf M$ (a polynomial of
//! degree $\lt d$) evaluated at the verifier's $\alpha \in \mathbb F_{q^4}$ with the powers
//! `alpha_pows` $= (1, \alpha, \dots, \alpha^{d-1})$, stored row-major as
//! `evals[row * width + col]` and zero-padded so that both dimensions are powers of two:
//! `height` $= 2^{\lceil \log_2(3n+2) \rceil}$ and
//! `width` $= 2^{\lceil \log_2(\mu + (3n+2)\delta) \rceil}$ with
//! $\mu = \delta 2^r + n\delta 2^r + \delta\tau 2^m$ (the length of $\mathbf z^{\prime}$, in ring elements;
//! `width` also equals the number of entries of the padded next witness divided by $d$). Read as a
//! function on $\lbrace 0,1 \rbrace^{\log \mathrm{width} + \log \mathrm{height}}$ with the column
//! bits low, its multilinear extension at $(\mathbf r_x, \tau_1)$ is
//! $\sum_i \mathrm{eq}(\tau_1, i)\tilde M_\alpha(i, \mathbf r_x)$, the public factor of
//! $F_{\alpha,\tau_1}$ used by the prover ([`crate::hachi::prover_utils::sumcheck::FAlpha`]) and
//! the verifier ([`crate::hachi::verify`]).
//!
//! Block structure (0-based rows and columns; $\mathbf X(\alpha)$ denotes entry-wise evaluation;
//! $b^j$ are the gadget weights of $\mathbf G$, $b^k$ with $k \lt \tau$ those of $\mathbf J$):
//!
//! * Columns $\[0, \delta 2^r)$ belong to $\hat{\mathbf w}$: column $i\delta + j$ is digit $j$ of
//!   $w_i$. Columns $\[\delta 2^r, \delta 2^r + n\delta 2^r)$ belong to $\hat{\mathbf t}$: column
//!   $\delta 2^r + (in + k)\delta + j$ is digit $j$ of row $k$ of $\mathbf t_i$. Columns
//!   $\[\delta 2^r + n\delta 2^r, \mu)$ belong to $\hat{\mathbf z}$: column
//!   $\delta 2^r + n\delta 2^r + c\tau + k$ is digit $k$ of entry $c$ of $\mathbf z$, where entry
//!   $c = i\delta + j$ is digit $j$ of element $i$ of the folded chunk. Columns
//!   $\[\mu, \mu + (3n+2)\delta)$ belong to $\mathbf r$: column $\mu + i\delta + j$ is digit $j$ of
//!   the quotient of row $i$. These are exactly the layouts of `form_next_witness`.
//! * Rows $\[0, n)$ (equation $\mathbf v = \mathbf D\hat{\mathbf w}$): $\mathbf D(\alpha)$ in the
//!   $\hat{\mathbf w}$ columns.
//! * Rows $\[n, 2n)$ ($\mathbf u = \mathbf B\hat{\mathbf t}$): $\mathbf B(\alpha)$ in the
//!   $\hat{\mathbf t}$ columns.
//! * Row $2n$ ($Y = \mathbf b^{\mathsf T}\mathbf G_{2^r}\hat{\mathbf w}$): $b_i b^j$ at
//!   $\hat{\mathbf w}$ column $i\delta + j$, with $b_i = \prod_{t \lt r} x_t^{i_t}$ lifted to
//!   $\mathbb F_{q^4}$.
//! * Row $2n+1$ ($(\mathbf c^{\mathsf T} \otimes \mathbf G_1)\hat{\mathbf w} - \mathbf a^{\mathsf T}\mathbf G_{2^m}\mathbf J_{2^m}\hat{\mathbf z} = 0$):
//!   $c_i(\alpha) b^j$ at $\hat{\mathbf w}$ column $i\delta + j$, and $-a_i b^{j} b^{k}$ at
//!   $\hat{\mathbf z}$ column $(i\delta + j)\tau + k$ (offset by the $\hat{\mathbf z}$ block), with
//!   $a_i = \prod_{t \lt m} x_{r+t}^{i_t}$ and $c_i(\alpha)$ the sparse challenge evaluated at
//!   $\alpha$ ([`PChal::eval`]).
//! * Rows $\[2n+2, 3n+2)$ ($(\mathbf c^{\mathsf T} \otimes \mathbf G_n)\hat{\mathbf t} - \mathbf A\mathbf J_{2^m}\hat{\mathbf z} = 0$):
//!   in row $2n+2+k$, $c_i(\alpha) b^j$ at the $\hat{\mathbf t}$ column of digit $j$ of row $k$ of
//!   $\mathbf t_i$, and $-b^{k^{\prime}} \mathbf A_{k,c}(\alpha)$ at the $\hat{\mathbf z}$ column
//!   $c\tau + k^{\prime}$.
//! * Quotient block: in every row $i \lt 3n+2$, $-(\alpha^d+1) b^j$ at column $\mu + i\delta + j$.
//!   Row $2n$ keeps this block although its quotient is always zero (the $b_i$ are integers).
//!
//! All other entries, including the padding rows and columns, are zero. Building the table costs
//! one evaluation at $\alpha$ per matrix entry ($n(\delta 2^r + n\delta 2^r + \delta 2^m)$
//! polynomials of degree $\lt d$, each $d$ base-field-times-extension-field products) plus
//! `height * width` elements of $\mathbb F_{q^4}$ of storage.

use ark_ff::AdditiveGroup;

use crate::arithmetic::utils::{Logarithm, gadget, lift_int, mul_int_field, multi_lin_coeff_int};
use crate::arithmetic::poly_mat::PMat;
use crate::arithmetic::poly_chal::PChal;
use crate::arithmetic::ExtField;
use crate::arithmetic::poly::Poly;

use crate::hachi::setup::Parameters;

/// Verifier-side entry point: regenerates the commitment matrices from the seeds in `params`
/// (one matrix from `d_seed` when `params.reuse_mats`, else $\mathbf A$, $\mathbf B$, $\mathbf D$
/// from their own seeds) and returns the row-major table of $\mathbf M_\alpha$ described in the
/// module documentation. `x` is the evaluation point (its first $r + m$ coordinates are used),
/// `challenges` the $2^r$ sparse challenges $c_i$, `alpha_pows` $= (1, \alpha, \dots, \alpha^{d-1})$.
#[cfg_attr(feature = "stats", time_graph::instrument)]
pub fn form_m_alpha(
    params: &Parameters,
    x: &[u64],
    challenges: &[PChal],
    alpha_pows: &[ExtField]   
) -> Vec<ExtField>
{
    if params.reuse_mats {
        let mat = PMat::rand(params.n, params.width_d, params.d, params.q, params.d_seed);
        form_m_alpha_same_matrix(params, x, challenges, alpha_pows, &mat)
    }
    else {
        let mat_a = PMat::rand(params.n, params.width_a, params.d, params.q, params.a_seed);
        let mat_b = PMat::rand(params.n, params.width_b, params.d, params.q, params.b_seed);
        let mat_d = PMat::rand(params.n, params.width_d, params.d, params.q, params.d_seed);
        form_m_alpha_different_matrices(params, x, challenges, alpha_pows, &mat_a, &mat_b, &mat_d)
    }
}

/// The table of $\mathbf M_\alpha$ (module documentation) for given commitment matrices
/// $\mathbf A$, $\mathbf B$, $\mathbf D$ in coefficient form. Returns `height * width` elements of
/// $\mathbb F_{q^4}$, entry $(i, j)$ at index `i * width + j`. Requires `params.decomp_witness`
/// (the $\hat{\mathbf z}$ block assumes `width_a` $= \delta 2^m$).
#[cfg_attr(feature = "stats", time_graph::instrument)]
pub fn form_m_alpha_different_matrices(
    params: &Parameters,
    x: &[u64],
    challenges: &[PChal],
    alpha_pows: &[ExtField],
    mat_a: &PMat,
    mat_b: &PMat,
    mat_d: &PMat
) -> Vec<ExtField>
{
    // We are constructing M (evaluated at alpha) such that M.[z|r] = y.
    // M'.z = y + (X^d+1).G_n.r where M' is the verification matrix for equations from previous part of the protocol.
    // So M = [M' | -(X^d+1).G_n], and padded with zeros so its height and width are powers of two.

    // length of z (width of M')
    let mu = params.width_d + params.width_b + params.width_a * params.delta_z;

    // length of r (height of M')
    let n = params.n + params.n + 1 + 1 + params.n;

    // pad n and (mu + n*delta) to the next power of two
    let height = 1 << (n.log());
    let width = 1 << ((mu + n * params.delta).log());

    // create the vector of evaluations of M_alpha_mle
    let mut evals = vec![ExtField::ZERO; height * width];

    // set a particular element of the evaluations
    fn set(evals: &mut Vec<ExtField>, f: ExtField, width: usize, row: usize, col: usize) {
        evals[row * width + col] = f;
    }

    // copy in D
    for row in 0..params.n {
        for col in 0..params.width_d {
            let cur = mat_d.element(row, col);
            let f = cur.eval(alpha_pows);
            set(&mut evals, f, width, row, col);
        }
    }

    // copy in B
    let row_offset = params.n;
    let col_offset = params.width_d;

    for row in 0..params.n {
        for col in 0..params.width_b {
            let cur = mat_b.element(row, col);
            let f = cur.eval(alpha_pows);
            set(&mut evals, f, width, row + row_offset, col + col_offset);
        }
    }

    // copy in b^T . G_{2^r}
    let row = params.n + params.n;
    let gadget_vec = gadget(params.b, params.delta);

    for i in 0..1 << params.r {
        // get the current coefficient
        let b_i = multi_lin_coeff_int(&x[0..params.r], i, params.r, params.q);

        // expand with gadget vector
        for j in 0..params.delta {
            let f = lift_int(b_i * gadget_vec[j]);
            set(&mut evals, f, width, row, i * params.delta + j);
        }
    }

    // copy in c^T o G_1 = c^T . G_{2^r} (fourth row section) and c^T o G_n (fifth row section)
    let row = params.n + params.n + 1;              // for c^T o G_1 (fourth row section)
    let row_offset = params.n + params.n + 1 + 1;   // for c^T o G_n (fifth row section)
    let col_offset = params.width_d;                // for c^T o G_n (fifth row section)

    for i in 0..1 << params.r {
        // get the current challenge and evaluate at alpha
        let c_i = challenges[i].eval(alpha_pows);

        // expand with gadget vector
        for j in 0..params.delta {
            let f = mul_int_field(gadget_vec[j], c_i);

            // set in fourth row section
            set(&mut evals, f, width, row, i * params.delta + j);

            // set in fifth row section
            for k in 0..params.n {
                set(&mut evals, f, width, row_offset + k, col_offset + i * params.delta * params.n + k * params.delta + j);
            }
        }
    }

    // copy in -[a^T . G_{2^m} . G_z]
    let row = params.n + params.n + 1;
    let col_offset = params.width_d + params.width_b;
    let gadget_vec_z = gadget(params.b, params.delta_z);

    for i in 0..1 << params.m {
        // get the current coefficient
        let a_i = multi_lin_coeff_int(&x[params.r..params.r + params.m], i, params.m, params.q);

        // expand with normal gadget vector
        for j in 0..params.delta {
            // expand with gadget matrix for composition of z_hat into z
            for k in 0..params.delta_z {
                let f = - lift_int((a_i * gadget_vec[j]) % params.q * gadget_vec_z[k]);
                set(&mut evals, f, width, row, col_offset + i * params.delta * params.delta_z + j * params.delta_z + k);
            }
        }
    }

    // copy in -A.G_z
    let row_offset = params.n + params.n + 1 + 1;
    let col_offset = params.width_d + params.width_b;

    for row in 0..params.n {
        for col in 0..params.width_a {
            // evaluate the current element of A
            let cur = mat_a.element(row, col).eval(alpha_pows);

            // expand with gadget vector
            for j in 0..params.delta_z {
                let f = - mul_int_field(gadget_vec_z[j], cur);
                set(&mut evals, f, width, row_offset + row, col_offset + col * params.delta_z + j);
            }
        }
    }

    // Calculate -(alpha^d + 1)
    let minus_alpha_d_plus_one = -(alpha_pows[params.d-1] * alpha_pows[1] + alpha_pows[0]);

    // Gadget expand
    let mut minus_alpha_d_plus_one_expanded = Vec::<ExtField>::new();

    for i in 0..params.delta {
        minus_alpha_d_plus_one_expanded.push(mul_int_field(gadget_vec[i], minus_alpha_d_plus_one));
    }
    
    // Set the quotient block: row i, columns mu + i*delta .. mu + (i+1)*delta
    for i in 0..n {
        for j in 0..params.delta {
            set(&mut evals, minus_alpha_d_plus_one_expanded[j], width, i, mu + i * params.delta + j);
        }
    }

    evals
}

/// The table of $\mathbf M_\alpha$ when a single matrix serves as $\mathbf A$, $\mathbf B$ and
/// $\mathbf D$ (`params.reuse_mats`, asserted): identical output to
/// [`form_m_alpha_different_matrices`] with `mat` passed three times, but each matrix entry is
/// evaluated at $\alpha$ once and written to the $\mathbf D(\alpha)$, $\mathbf B(\alpha)$ and
/// $-\mathbf A(\alpha)\mathbf J_{2^m}$ blocks in a single loop.
#[cfg_attr(feature = "stats", time_graph::instrument)]
pub fn form_m_alpha_same_matrix(
    params: &Parameters,
    x: &[u64],
    challenges: &[PChal],
    alpha_pows: &[ExtField],
    mat: &PMat
) -> Vec<ExtField>
{
    // We are constructing M (evaluated at alpha) such that M.[z|r] = y.
    // M'.z = y + (X^d+1).G_n.r where M' is the verification matrix for equations from previous part of the protocol.
    // So M = [M' | -(X^d+1).G_n], and padded with zeros so its height and width are powers of two.
    // Optimise for the case when the commitment matrices are the same.

    assert!(params.reuse_mats);

    // length of z (width of M')
    let mu = params.width_d + params.width_b + params.width_a * params.delta_z;

    // length of r (height of M')
    let n = params.n + params.n + 1 + 1 + params.n;

    // pad n and (mu + n*delta) to the next power of two
    let height = 1 << n.log();
    let width = 1 << (mu + n * params.delta).log();

    // create the vector of evaluations of M_alpha_mle
    let mut evals = vec![ExtField::ZERO; height * width];

    // set a particular element of the evaluations
    fn set(evals: &mut Vec<ExtField>, f: ExtField, width: usize, row: usize, col: usize) {
        evals[row * width + col] = f;
    }

    // gadget vectors
    let gadget_vec = gadget(params.b, params.delta);
    let gadget_vec_z = gadget(params.b, params.delta_z);

    // copy in D, B and -A.G_z
    for row in 0..params.n {
        for col in 0..params.width_d {
            let f = mat.element(row, col).eval(alpha_pows);

            // set D
            set(&mut evals, f, width, row, col);

            // set B
            set(&mut evals, f, width, params.n + row, params.width_d + col);

            // gadget expand and set A.G_z
            let row_offset = params.n + params.n + 1 + 1;
            let col_offset = params.width_d + params.width_b;

            for j in 0..params.delta_z {
                let f = - mul_int_field(gadget_vec_z[j], f);
                set(&mut evals, f, width, row_offset + row, col_offset + col * params.delta_z + j);
            }
        }
    }

    // copy in b^T . G_{2^r}
    let row = params.n + params.n;

    for i in 0..1 << params.r {
        // get the current coefficient
        let b_i = multi_lin_coeff_int(&x[0..params.r], i, params.r, params.q);

        // expand with gadget vector
        for j in 0..params.delta {
            let f = lift_int(b_i * gadget_vec[j]);
            set(&mut evals, f, width, row, i * params.delta + j);
        }
    }

    // copy in c^T o G_1 = c^T . G_{2^r} (fourth row section) and c^T o G_n (fifth row section)
    let row = params.n + params.n + 1;              // for c^T o G_1 (fourth row section)
    let row_offset = params.n + params.n + 1 + 1;   // for c^T o G_n (fifth row section)
    let col_offset = params.width_d;                // for c^T o G_n (fifth row section)

    for i in 0..1 << params.r {
        // get the current challenge and evaluate at alpha
        let c_i = challenges[i].eval(alpha_pows);

        // expand with gadget vector
        for j in 0..params.delta {
            let f = mul_int_field(gadget_vec[j], c_i);

            // set in fourth row section
            set(&mut evals, f, width, row, i * params.delta + j);

            // set in fifth row section
            for k in 0..params.n {
                set(&mut evals, f, width, row_offset + k, col_offset + i * params.delta * params.n + k * params.delta + j);
            }
        }
    }

    // copy in -[a^T . G_{2^m} . G_z]
    let row = params.n + params.n + 1;
    let col_offset = params.width_d + params.width_b;

    for i in 0..1 << params.m {
        // get the current coefficient
        let a_i = multi_lin_coeff_int(&x[params.r..params.r + params.m], i, params.m, params.q);

        // expand with normal gadget vector
        for j in 0..params.delta {
            // expand with gadget matrix for composition of z_hat into z
            for k in 0..params.delta_z {
                let f = - lift_int((a_i * gadget_vec[j]) % params.q * gadget_vec_z[k]);
                set(&mut evals, f, width, row, col_offset + i * params.delta * params.delta_z + j * params.delta_z + k);
            }
        }
    }

    // Calculate -(alpha^d + 1)
    let minus_alpha_d_plus_one = -(alpha_pows[params.d-1] * alpha_pows[1] + alpha_pows[0]);

    // Gadget expand
    let mut minus_alpha_d_plus_one_expanded = Vec::<ExtField>::new();

    for i in 0..params.delta {
        minus_alpha_d_plus_one_expanded.push(mul_int_field(gadget_vec[i], minus_alpha_d_plus_one));
    }
    
    // Set the quotient block: row i, columns mu + i*delta .. mu + (i+1)*delta
    for i in 0..n {
        for j in 0..params.delta {
            set(&mut evals, minus_alpha_d_plus_one_expanded[j], width, i, mu + i * params.delta + j);
        }
    }

    evals
}
