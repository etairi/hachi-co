//! The public matrix $\mathbf M$ and vector $\mathbf y$ of the linear relation
//! $\mathbf M \mathbf z' = \mathbf y$ over $\mathcal R\_q$ produced by the ring-to-linear step
//! (`5_Building_Blocks.tex`:154-182; oracle `ref/hachico_ref/r2l.py`, `build_M` and `build_y`).
//! The provers and the verifier compute both from public data.

use crate::data::{Matrix, check_len, check_polys};
use crate::{Context, Result};
use hachi_co_math::Poly;

/// The matrix $\mathbf M \in \mathcal R\_q^{n' \times \mu}$ of `5_Building_Blocks.tex`:157-163,
/// dense and row-major, for the public vectors $\mathbf b$ and $\mathbf a$ embedded in
/// $\mathcal R\_q$ ([`PublicVectors::b_ring`](crate::PublicVectors::b_ring) and
/// [`PublicVectors::a_ring`](crate::PublicVectors::a_ring)) and the challenges
/// $\mathbf c \in \mathcal C^{2^r}$ (signed coefficients, `5_Building_Blocks.tex`:122-126).
///
/// With the positions of [`Layout`](crate::Layout) (0-indexed; $b^e$ is the constant polynomial
/// $b^e \bmod q$, $i \lt 2^r$, $p \in \lbrace 1, \dots, n \rbrace$, $j \lt 2^m$, $e \lt \delta$,
/// $\theta \lt \tau$, $a \lt n\_A$), and zero everywhere else:
/// - rows $\mathrm{row}\_D + \cdot$: $\mathbf D$ on the columns $\[0, t)$, i.e.
///   $\mathbf D\_{\hat{\mathbf w}} \mid \mathbf D\_0 \mid \mathbf D\_1 \mid \mathbf D\_\rho$;
/// - rows $\mathrm{row}\_B + \cdot$: $\mathbf B$ on the columns of $\hat{\mathbf t}$;
/// - row $\mathrm{row}\_{bG}$ ($\mathbf b^T \mathbf G\_{2^r} \mid -1$): $b\_i b^e$ at
///   $w + i\delta + e$, and $-1$ at $u = \mu - 1$;
/// - row $\mathrm{row}\_c$ ($\mathbf c^T \otimes \mathbf G\_1 \mid \mathbf 1\_n^T \otimes
///   \mathbf G\_1 \mid -\mathbf a^T \mathbf G\_{2^m} \mathbf J\_{\delta 2^m}$): $c\_i b^e$ at
///   $w + i\delta + e$, $b^e$ at $w\_0 + (p-1)\delta + e$, and $-a\_j b^{e + \theta}$ at
///   $z + (j\delta + e)\tau + \theta$;
/// - rows $\mathrm{row}\_A + a$ ($\mathbf 1\_n^T \otimes \mathbf G\_{n\_A} \mid \mathbf c^T
///   \otimes \mathbf G\_{n\_A} \mid -\mathbf A \mathbf J\_{\delta 2^m}$): $b^e$ at
///   $t\_0 + (p-1) n\_A \delta + a\delta + e$, $c\_i b^e$ at $t + i n\_A \delta + a\delta + e$,
///   and $-A\_{a,j'} b^\theta$ at $z + j'\tau + \theta$ for $j' \lt \delta 2^m$.
///
/// The fourth and fifth row blocks state $\sum\_i c\_i w\_i + \sum\_p w\_{0,p} = \mathbf a^T
/// \mathbf G\_{2^m} \mathbf z$ and $\sum\_i c\_i \mathbf t\_i + \sum\_p \mathbf t\_{0,p} = \mathbf A
/// \mathbf z$ (`5_Building_Blocks.tex`:182). The challenges are reduced modulo $q$; checking
/// $\mathbf c \in \mathcal C^{2^r}$ is the caller's job (the ring-to-linear step checks it).
/// Every residue of the result is canonical.
///
/// # Errors
/// [`Error::Shape`](crate::Error::Shape) unless `b_ring` has $2^r$ and `a_ring` $2^m$
/// polynomials of $d$ coefficients and `c` has $2^r$ vectors of $d$ coefficients; the errors of
/// [`PublicParams::validate`](crate::PublicParams::validate) if `ctx.pp` was changed into an
/// invalid shape after the context was built.
pub fn build_m<const K: usize>(
    ctx: &Context<K>,
    b_ring: &[Poly],
    a_ring: &[Poly],
    c: &[Vec<i64>],
) -> Result<Matrix> {
    let l = &ctx.layout;
    let ring = &ctx.ring;
    let zq = ring.zq();
    check_polys("b (ring)", b_ring, l.n_blocks, l.d)?;
    check_polys("a (ring)", a_ring, l.block_len, l.d)?;
    check_len("c (one challenge per block)", l.n_blocks, c.len())?;
    for ci in c {
        check_len("c_i (coefficients)", l.d, ci.len())?;
    }
    // The public fields of the context may have been changed since it was built.
    ctx.pp.validate(l, ctx.q())?;
    let b = zq.reduce(ctx.b());
    let bp: Vec<u64> = (0..l.delta + l.tau).map(|e| zq.pow(b, e as u64)).collect();
    let constant = |x: u64| {
        let mut p = ring.zero();
        p.0[0] = x;
        p
    };
    let cq = c
        .iter()
        .map(|ci| ring.from_i64(ci))
        .collect::<hachi_co_math::Result<Vec<Poly>>>()?;

    let mut m = vec![vec![ring.zero(); l.mu]; l.n_prime];
    // D (n_D rows) on (w_hat, w0_hat, t0_hat, rho) = z'[0..t).
    for (i, row) in ctx.pp.d.iter().enumerate() {
        m[l.row_d + i][..l.width_d].clone_from_slice(row);
    }
    // B (n_B rows) on t_hat.
    for (i, row) in ctx.pp.b.iter().enumerate() {
        m[l.row_b + i][l.t..l.t + l.width_b].clone_from_slice(row);
    }
    // b^T G_{2^r} | ... | -1, and the c^T (x) G_1 part of the c row.
    for i in 0..l.n_blocks {
        for (e, pe) in bp.iter().take(l.delta).enumerate() {
            let col = l.w + i * l.delta + e;
            m[l.row_bg][col] = ring.scalar_mul(*pe, &b_ring[i])?;
            m[l.row_c][col] = ring.scalar_mul(*pe, &cq[i])?;
        }
    }
    m[l.row_bg][l.u] = constant(zq.neg(1));
    // 1_n^T (x) G_1 on w0_hat.
    for p in 0..l.n {
        for (e, pe) in bp.iter().take(l.delta).enumerate() {
            m[l.row_c][l.w0 + p * l.delta + e] = constant(*pe);
        }
    }
    // -a^T G_{2^m} J_{delta 2^m} on z_hat: z_hat entry (j delta + e) tau + theta has weight
    // a_j b^e b^theta.
    for (j, aj) in a_ring.iter().enumerate() {
        for e in 0..l.delta {
            for theta in 0..l.tau {
                let col = l.z + (j * l.delta + e) * l.tau + theta;
                m[l.row_c][col] = ring.neg(&ring.scalar_mul(bp[e + theta], aj)?)?;
            }
        }
    }
    // The n_A rows: 1_n^T (x) G_{n_A} on t0_hat, c^T (x) G_{n_A} on t_hat, -A J on z_hat.
    for a in 0..l.n_a {
        let row = &mut m[l.row_a + a];
        for p in 0..l.n {
            for (e, pe) in bp.iter().take(l.delta).enumerate() {
                row[l.t0 + p * l.n_a * l.delta + a * l.delta + e] = constant(*pe);
            }
        }
        for (i, ci) in cq.iter().enumerate() {
            for (e, pe) in bp.iter().take(l.delta).enumerate() {
                row[l.t + i * l.n_a * l.delta + a * l.delta + e] = ring.scalar_mul(*pe, ci)?;
            }
        }
        for (j, a_entry) in ctx.pp.a[a].iter().enumerate() {
            for (theta, pt) in bp.iter().take(l.tau).enumerate() {
                row[l.z + j * l.tau + theta] = ring.neg(&ring.scalar_mul(*pt, a_entry)?)?;
            }
        }
    }
    Ok(m)
}

/// The vector $\mathbf y = (\mathbf v, \mathbf u, 0, 0, \mathbf 0) \in \mathcal R\_q^{n'}$ of
/// `5_Building_Blocks.tex`:173-179: the first message $\mathbf v \in \mathcal R\_q^{n\_D}$ of the
/// accepted attempt at rows $\mathrm{row}\_D + \cdot$, the commitment
/// $\mathbf u \in \mathcal R\_q^{n\_B}$ at rows $\mathrm{row}\_B + \cdot$, and zeros in the
/// $\mathbf b^T \mathbf G$, $\mathbf c$ and $\mathbf A$ rows. Residues are returned reduced.
///
/// # Errors
/// [`Error::Shape`](crate::Error::Shape) unless `v` has $n\_D$ and `u_com` $n\_B$ polynomials, or
/// [`Error::Math`](crate::Error::Math) for a polynomial without $d$ coefficients.
pub fn build_y<const K: usize>(ctx: &Context<K>, v: &[Poly], u_com: &[Poly]) -> Result<Vec<Poly>> {
    let l = &ctx.layout;
    check_len("v (one entry per row of D)", l.n_d, v.len())?;
    check_len("u (one entry per row of B)", l.n_b, u_com.len())?;
    let mut y = vec![ctx.ring.zero(); l.n_prime];
    for (i, vi) in v.iter().enumerate() {
        y[l.row_d + i] = ctx.ring.from_coeffs(&vi.0)?;
    }
    for (i, ui) in u_com.iter().enumerate() {
        y[l.row_b + i] = ctx.ring.from_coeffs(&ui.0)?;
    }
    Ok(y)
}
