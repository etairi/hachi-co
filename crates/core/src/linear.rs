//! The public matrix $\mathbf M$ and vector $\mathbf y$ of the linear relation
//! $\mathbf M \mathbf z' = \mathbf y$ over $\mathcal R\_q$ produced by the ring-to-linear step
//! (`5_Building_Blocks.tex`:167-196; oracle `ref/hachico_ref/r2l.py`, `build_M` and `build_y`).
//! The provers and the verifier compute both from public data.

use crate::data::{Matrix, check_len, check_polys};
use crate::{Context, Result};
use hachi_co_math::Poly;
use rayon::prelude::*;

/// The matrix $\mathbf M \in \mathcal R\_q^{n' \times (\mu - 1)}$ of `5_Building_Blocks.tex`:170-176
/// (`fig:ring-to-linear-2` since the draft's commit `1096ea2`), dense and row-major, for the
/// public vectors $\mathbf b$ and $\mathbf a$ embedded in $\mathcal R\_q$
/// ([`PublicVectors::b_ring`](crate::PublicVectors::b_ring) and
/// [`PublicVectors::a_ring`](crate::PublicVectors::a_ring)), the challenges
/// $\mathbf c \in \mathcal C^{2^r}$ (signed coefficients, `5_Building_Blocks.tex`:134-138) and
/// the selected iteration $j^\ast \lt L$ of the rejection sampling (0-indexed, the draft's
/// $j^\ast + 1$).
///
/// With the positions of [`Layout`](crate::Layout) (0-indexed; $b^e$ is the constant polynomial
/// $b^e \bmod q$, $i \lt 2^r$, $p \in \lbrace 1, \dots, n \rbrace$, $j \lt 2^m$, $e \lt \delta$,
/// $\theta \lt \tau$, $a \lt n\_A$), and zero everywhere else:
/// - rows $\mathrm{row}\_D + \cdot$: $\mathbf D$ on the columns $\[0, t)$, i.e.
///   $\mathbf D\_{\hat{\mathbf w}} \mid \mathbf D\_0 \mid \mathbf D\_1 \mid \mathbf D\_\rho$;
/// - rows $\mathrm{row}\_B + \cdot$: $\mathbf B$ on the columns $\[t, z)$ of
///   $(\hat{\mathbf t}, \boldsymbol\rho^u)$, i.e. $\mathbf B\_{\hat{\mathbf t}} \mid
///   \mathbf B\_{\boldsymbol\rho}$;
/// - row $\mathrm{row}\_{bG}$ ($\mathbf b^T \mathbf G\_{2^r} \mid -1$): $b\_i b^e$ at
///   $w + i\delta + e$, and $-1$ at $u = \mu - 2$;
/// - row $\mathrm{row}\_c$ ($\mathbf c^T \otimes \mathbf G\_1 \mid \mathbf 1\_n^T \otimes
///   \hat{\mathbf e}\_{j^\ast}^T \otimes \mathbf G\_1 \mid -\mathbf a^T \mathbf G\_{2^m}
///   \mathbf J\_{\delta 2^m}$): $c\_i b^e$ at $w + i\delta + e$, $b^e$ at the positions of
///   $\hat{\mathbf w}\_{0,p}^{j^\ast}$, $w\_0 + ((p-1)L + j^\ast)\delta + e$, and
///   $-a\_j b^{e + \theta}$ at $z + (j\delta + e)\tau + \theta$;
/// - rows $\mathrm{row}\_A + a$ ($\mathbf 1\_n^T \otimes \hat{\mathbf e}\_{j^\ast}^T \otimes
///   \mathbf G\_{n\_A} \mid \mathbf c^T \otimes \mathbf G\_{n\_A} \mid -\mathbf A
///   \mathbf J\_{\delta 2^m}$): $b^e$ at $t\_0 + ((p-1)L + j^\ast) n\_A \delta + a\delta + e$,
///   $c\_i b^e$ at $t + i n\_A \delta + a\delta + e$, and $-A\_{a,j'} b^\theta$ at
///   $z + j'\tau + \theta$ for $j' \lt \delta 2^m$.
///
/// The fourth and fifth row blocks state $\sum\_i c\_i w\_i + \sum\_p w\_{0,p}^{j^\ast} =
/// \mathbf a^T \mathbf G\_{2^m} \mathbf z$ and $\sum\_i c\_i \mathbf t\_i + \sum\_p
/// \mathbf t\_{0,p}^{j^\ast} = \mathbf A \mathbf z$ (`5_Building_Blocks.tex`:196, and the text
/// after `fig:ring-to-linear-2`); the masks of the other iterations enter only the $\mathbf D$
/// rows. The challenges are reduced modulo $q$; checking
/// $\mathbf c \in \mathcal C^{2^r}$ is the caller's job (the ring-to-linear step checks it).
/// Every residue of the result is canonical.
///
/// # Errors
/// [`Error::Shape`](crate::Error::Shape) unless `b_ring` has $2^r$ and `a_ring` $2^m$
/// polynomials of $d$ coefficients and `c` has $2^r$ vectors of $d$ coefficients;
/// [`Error::Layout`](crate::Error::Layout) unless $j^\ast \lt L$; the errors of
/// [`PublicParams::validate`](crate::PublicParams::validate) if `ctx.pp()` was changed into an
/// invalid shape after the context was built.
pub fn build_m<const K: usize>(
    ctx: &Context<K>,
    b_ring: &[Poly],
    a_ring: &[Poly],
    c: &[Vec<i64>],
    j_star: usize,
) -> Result<Matrix> {
    let rows = MRows::new(ctx, b_ring, a_ring, c, j_star)?;
    (0..rows.len()).map(|i| rows.row(i)).collect()
}

/// The rows of [`build_m`] one at a time, for readers that go row by row (the provers' quotient
/// and the column weights of $M^\ast$ in `hachi-co-sumcheck`), so that none of them holds all of
/// $\mathbf M$: [`MRows::new`] checks the inputs as [`build_m`] does, and [`MRows::row`] forms
/// row $i$ with the entries that [`build_m`] gives it (which is [`MRows::row`] for every $i$).
#[derive(Clone, Debug)]
pub struct MRows<'a, const K: usize> {
    ctx: &'a Context<K>,
    b_ring: &'a [Poly],
    a_ring: &'a [Poly],
    /// The challenges reduced modulo $q$.
    cq: Vec<Poly>,
    j_star: usize,
    /// $b^e \bmod q$ for $e \lt \delta + \tau$.
    bp: Vec<u64>,
}

impl<'a, const K: usize> MRows<'a, K> {
    /// The rows for the inputs of [`build_m`].
    ///
    /// # Errors
    /// As [`build_m`].
    pub fn new(
        ctx: &'a Context<K>,
        b_ring: &'a [Poly],
        a_ring: &'a [Poly],
        c: &[Vec<i64>],
        j_star: usize,
    ) -> Result<Self> {
        let l = &ctx.layout;
        let ring = &ctx.ring;
        let zq = ring.zq();
        if j_star >= l.rej_iterations {
            return Err(crate::Error::Layout(format!(
                "the selected iteration j* = {j_star} is not below L = {}",
                l.rej_iterations
            )));
        }
        check_polys("b (ring)", b_ring, l.n_blocks, l.d)?;
        check_polys("a (ring)", a_ring, l.block_len, l.d)?;
        check_len("c (one challenge per block)", l.n_blocks, c.len())?;
        for ci in c {
            check_len("c_i (coefficients)", l.d, ci.len())?;
        }
        // The public fields of the context may have been changed since it was built.
        ctx.pp().validate(l, ctx.q())?;
        let b = zq.reduce(ctx.b());
        let bp: Vec<u64> = (0..l.delta + l.tau).map(|e| zq.pow(b, e as u64)).collect();
        let cq = c
            .iter()
            .map(|ci| ring.from_i64(ci))
            .collect::<hachi_co_math::Result<Vec<Poly>>>()?;
        Ok(Self {
            ctx,
            b_ring,
            a_ring,
            cq,
            j_star,
            bp,
        })
    }

    /// The number of rows, $n'$.
    pub fn len(&self) -> usize {
        self.ctx.layout.n_prime
    }

    /// Whether $\mathbf M$ has no rows (never, for a valid layout).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Row $i$ of $\mathbf M$: $\mu - 1$ polynomials, the entries of the documentation of
    /// [`build_m`] in that row, written in the order of that list. The large blocks are written
    /// in parallel into the zeros of the row, in place ([`Ring::scalar_mul_into`] and
    /// [`Ring::neg_scalar_mul_into`], the residues of `scalar_mul` and `neg`), so that no ring
    /// element is allocated or freed past the row's zeros.
    ///
    /// # Errors
    /// [`Error::Layout`](crate::Error::Layout) for $i \ge n'$.
    pub fn row(&self, i: usize) -> Result<Vec<Poly>> {
        let (ctx, l) = (self.ctx, &self.ctx.layout);
        let ring = &ctx.ring;
        let zq = ring.zq();
        let bp = &self.bp;
        if i >= l.n_prime {
            return Err(crate::Error::Layout(format!(
                "row {i} of M, which has {} rows",
                l.n_prime
            )));
        }
        // The zero row's many ring elements are allocated in parallel.
        let mut row: Vec<Poly> = (0..l.len_z).into_par_iter().map(|_| ring.zero()).collect();
        // `values[k]` times `x[j]` into the k-th of the `width` entries of chunk j, from `start`.
        let blocks = |row: &mut [Poly],
                      start: usize,
                      width: usize,
                      x: &[Poly],
                      at: &(dyn Fn(usize) -> usize + Sync),
                      values: &[u64],
                      negate: bool| {
            row[start..start + x.len() * width]
                .par_chunks_mut(width)
                .zip(x.par_iter())
                .try_for_each(|(entries, xj)| {
                    for (k, value) in values.iter().enumerate() {
                        let out = &mut entries[at(k)];
                        if negate {
                            ring.neg_scalar_mul_into(*value, xj, out)?;
                        } else {
                            ring.scalar_mul_into(*value, xj, out)?;
                        }
                    }
                    Ok::<(), hachi_co_math::Error>(())
                })
        };
        if i < l.row_b {
            // D (n_D rows) on (w_hat, w0_hat, t0_hat, rho_v) = z'[0..t).
            row[..l.width_d]
                .par_iter_mut()
                .zip(ctx.pp().d[i - l.row_d].par_iter())
                .for_each(|(out, x)| out.clone_from(x));
        } else if i < l.row_bg {
            // B (n_B rows) on (t_hat, rho_u) = z'[t..z).
            row[l.t..l.t + l.width_b]
                .par_iter_mut()
                .zip(ctx.pp().b[i - l.row_b].par_iter())
                .for_each(|(out, x)| out.clone_from(x));
        } else if i == l.row_bg {
            // b^T G_{2^r} | ... | -1.
            blocks(
                &mut row,
                l.w,
                l.delta,
                self.b_ring,
                &|e| e,
                &bp[..l.delta],
                false,
            )?;
            row[l.u].0[0] = zq.neg(1);
        } else if i == l.row_c {
            // c^T (x) G_1.
            blocks(
                &mut row,
                l.w,
                l.delta,
                &self.cq,
                &|e| e,
                &bp[..l.delta],
                false,
            )?;
            // 1_n^T (x) e_{j*}^T (x) G_1 on w0_hat: the digits of w0_{p}^{j*} of every prover.
            for p in 1..=l.n {
                let first = l.w0_hat_range(p, self.j_star).start;
                for (e, pe) in bp.iter().take(l.delta).enumerate() {
                    row[first + e].0[0] = *pe;
                }
            }
            // -a^T G_{2^m} J_{delta 2^m} on z_hat: z_hat entry (j delta + e) tau + theta has
            // weight a_j b^e b^theta.
            let weights: Vec<u64> = (0..l.delta * l.tau)
                .map(|k| bp[k / l.tau + k % l.tau])
                .collect();
            let width = l.delta * l.tau;
            blocks(&mut row, l.z, width, self.a_ring, &|k| k, &weights, true)?;
        } else {
            // The n_A rows: 1_n^T (x) e_{j*}^T (x) G_{n_A} on t0_hat, c^T (x) G_{n_A} on t_hat,
            // -A J on z_hat.
            let a = i - l.row_a;
            for p in 1..=l.n {
                let first = l.t0_hat_range(p, self.j_star).start;
                for (e, pe) in bp.iter().take(l.delta).enumerate() {
                    row[first + a * l.delta + e].0[0] = *pe;
                }
            }
            let stride = l.n_a * l.delta;
            let offset = a * l.delta;
            blocks(
                &mut row,
                l.t,
                stride,
                &self.cq,
                &|e| offset + e,
                &bp[..l.delta],
                false,
            )?;
            blocks(
                &mut row,
                l.z,
                l.tau,
                &ctx.pp().a[a],
                &|theta| theta,
                &bp[..l.tau],
                true,
            )?;
        }
        Ok(row)
    }
}

/// The vector $\mathbf y = (\mathbf v, \mathbf u, 0, 0, \mathbf 0) \in \mathcal R\_q^{n'}$ of
/// `5_Building_Blocks.tex`:187-193: the first message $\mathbf v \in \mathcal R\_q^{n\_D}$ at
/// rows $\mathrm{row}\_D + \cdot$, the commitment
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

#[cfg(test)]
mod tests {
    //! The rows of [`MRows`] against the matrix that [`build_m`] formed before it was built on
    //! them (a copy of that code), at the toy presets of one, two and four parties, for the first
    //! and the last iteration of the rejection sampling.
    use super::*;
    use crate::data::{Matrix, check_len, check_polys};
    use hachi_co_math::Ext;
    use hachi_co_params::{derive, presets::toy_choices};

    /// `build_m` as it was before [`MRows`] (2026-11-13), the reference of the rows.
    fn reference_build_m<const K: usize>(
        ctx: &Context<K>,
        b_ring: &[Poly],
        a_ring: &[Poly],
        c: &[Vec<i64>],
        j_star: usize,
    ) -> Result<Matrix> {
        let l = &ctx.layout;
        let ring = &ctx.ring;
        let zq = ring.zq();
        if j_star >= l.rej_iterations {
            return Err(crate::Error::Layout(format!(
                "the selected iteration j* = {j_star} is not below L = {}",
                l.rej_iterations
            )));
        }
        check_polys("b (ring)", b_ring, l.n_blocks, l.d)?;
        check_polys("a (ring)", a_ring, l.block_len, l.d)?;
        check_len("c (one challenge per block)", l.n_blocks, c.len())?;
        for ci in c {
            check_len("c_i (coefficients)", l.d, ci.len())?;
        }
        // The public fields of the context may have been changed since it was built.
        ctx.pp().validate(l, ctx.q())?;
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

        let mut m = vec![vec![ring.zero(); l.len_z]; l.n_prime];
        // D (n_D rows) on (w_hat, w0_hat, t0_hat, rho_v) = z'[0..t).
        for (i, row) in ctx.pp().d.iter().enumerate() {
            m[l.row_d + i][..l.width_d].clone_from_slice(row);
        }
        // B (n_B rows) on (t_hat, rho_u) = z'[t..z).
        for (i, row) in ctx.pp().b.iter().enumerate() {
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
        // 1_n^T (x) e_{j*}^T (x) G_1 on w0_hat: the digits of w0_{p}^{j*} of every prover.
        for p in 1..=l.n {
            let first = l.w0_hat_range(p, j_star).start;
            for (e, pe) in bp.iter().take(l.delta).enumerate() {
                m[l.row_c][first + e] = constant(*pe);
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
        // The n_A rows: 1_n^T (x) e_{j*}^T (x) G_{n_A} on t0_hat, c^T (x) G_{n_A} on t_hat, -A J on
        // z_hat.
        for a in 0..l.n_a {
            let row = &mut m[l.row_a + a];
            for p in 1..=l.n {
                let first = l.t0_hat_range(p, j_star).start;
                for (e, pe) in bp.iter().take(l.delta).enumerate() {
                    row[first + a * l.delta + e] = constant(*pe);
                }
            }
            for (i, ci) in cq.iter().enumerate() {
                for (e, pe) in bp.iter().take(l.delta).enumerate() {
                    row[l.t + i * l.n_a * l.delta + a * l.delta + e] = ring.scalar_mul(*pe, ci)?;
                }
            }
            for (j, a_entry) in ctx.pp().a[a].iter().enumerate() {
                for (theta, pt) in bp.iter().take(l.tau).enumerate() {
                    row[l.z + j * l.tau + theta] = ring.neg(&ring.scalar_mul(*pt, a_entry)?)?;
                }
            }
        }
        Ok(m)
    }

    /// splitmix64.
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    #[test]
    fn the_rows_are_the_rows_of_the_matrix_before() {
        const K: usize = 4;
        for (n, seed) in [(1u64, 41u64), (2, 42), (4, 43)] {
            let (params, _) = derive(&toy_choices(n)).unwrap();
            let ctx = Context::<K>::setup(params, &[seed as u8; 32]).unwrap();
            let l = &ctx.layout;
            let (q, d) = (ctx.q(), l.d);
            let mut state = seed;
            let mut ext = || Ext::<K>(core::array::from_fn(|_| next(&mut state) % q));
            let b_ring: Vec<Poly> = (0..l.n_blocks).map(|_| ctx.sub.embed(&ext())).collect();
            let a_ring: Vec<Poly> = (0..l.block_len).map(|_| ctx.sub.embed(&ext())).collect();
            let c: Vec<Vec<i64>> = (0..l.n_blocks)
                .map(|_| (0..d).map(|_| (next(&mut state) % 3) as i64 - 1).collect())
                .collect();
            for j_star in [0, l.rej_iterations - 1] {
                let want: Matrix = reference_build_m(&ctx, &b_ring, &a_ring, &c, j_star).unwrap();
                assert_eq!(
                    build_m(&ctx, &b_ring, &a_ring, &c, j_star).unwrap(),
                    want,
                    "n = {n}, j* = {j_star}"
                );
                let rows = MRows::new(&ctx, &b_ring, &a_ring, &c, j_star).unwrap();
                assert_eq!(rows.len(), want.len());
                for (i, row) in want.iter().enumerate() {
                    assert_eq!(
                        &rows.row(i).unwrap(),
                        row,
                        "n = {n}, j* = {j_star}, row {i}"
                    );
                }
                assert!(rows.row(l.n_prime).is_err());
            }
            assert!(MRows::new(&ctx, &b_ring, &a_ring, &c, l.rej_iterations).is_err());
        }
    }
}
