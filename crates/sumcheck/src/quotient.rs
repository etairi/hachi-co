//! The provers' quotient shares (`5_Building_Blocks.tex`:227-229; oracle
//! `sumcheck.quotient_shares`).

use crate::error::{Error, Result, check_len};
use hachi_co_core::{Context, LinearShare, Poly};
use rayon::prelude::*;
use std::borrow::Cow;

/// How many products [`quotient_share`] hands to one exact inner product of
/// $\mathbb Z\_q\[X\]/(X^{2d} + 1)$ (bounds the memory of the zero-padded operands).
const CHUNK: usize = 1024;

/// Whether every residue of `p` is $0 \bmod q$.
fn is_zero(p: &Poly, q: u64) -> bool {
    p.0.iter().all(|x| x % q == 0)
}

/// `p` (with $d$ coefficients) as an element of $\mathbb Z\_q\[X\]/(X^{2d} + 1)$: its $d$ residues
/// followed by $d$ zeros.
fn pad(p: &Poly, two_d: usize) -> Poly {
    let mut out = p.0.clone();
    out.resize(two_d, 0);
    Poly(out)
}

/// Party $p$'s share of the quotient,
/// $$\langle \mathbf r \rangle\_p = \tfrac12 (1 - X^d) \bigl(\mathbf M \langle \mathbf z' \rangle\_p -
/// \lbrace \mathbf y \rbrace\_p\bigr) \bmod X^{2d} + 1$$
/// (`5_Building_Blocks.tex`:227-229, with $\tfrac12 (1 - X^d) = (X^d + 1)^{-1}$ in
/// $\mathbb Z\_q\[X\]/(X^{2d} + 1)$, footnote at line 227), where the entries of $\mathbf M$,
/// $\langle \mathbf z' \rangle\_p$ and $\lbrace \mathbf y \rbrace\_p$ are read as polynomials of
/// degree below $d$ (fixture convention `quotient`). Returns $n'$ polynomials of $2d$ residues
/// (`ref/README.md`, "Encodings": a quotient share is a list of $2d$ residues).
///
/// **Computation.** Row $i$ of $P = \mathbf M \langle \mathbf z' \rangle\_p - \lbrace \mathbf y
/// \rbrace\_p$ is a polynomial of degree at most $2d - 2$; it is formed exactly in
/// $\mathbb Z\_q\[X\]/(X^{2d} + 1)$ (`ctx.ring2`, where a product of two zero-padded polynomials
/// of degree below $d$ does not wrap), skipping the pairs $(i, j)$ where $M\_{ij}$ or
/// $\langle z'\_j \rangle\_p$ is zero. With $P = P\_{\mathrm{lo}} + X^d P\_{\mathrm{hi}}$, the product
/// is $\tfrac12 (1 - X^d) P = \tfrac12 (P\_{\mathrm{lo}} + P\_{\mathrm{hi}}) + X^d \cdot \tfrac12
/// (P\_{\mathrm{hi}} - P\_{\mathrm{lo}}) \bmod X^{2d} + 1$, as in the oracle. The tests compare this
/// closed form with the literal product in `ctx.ring2`.
///
/// Summed over the parties, $\mathbf r = \sum\_p \langle \mathbf r \rangle\_p$ satisfies
/// $(X^d + 1) \mathbf r = \mathbf M \mathbf z' - \mathbf y$ over $\mathbb Z\_q\[X\]$, and its high
/// halves vanish iff $\mathbf M \mathbf z' = \mathbf y$ in $\mathcal R\_q$.
///
/// # Errors
/// [`Error::Shape`] unless $\mathbf M$ has $n'$ rows of $\mu - 1$ polynomials and the share has the
/// shape of [`LinearShare::validate`], every polynomial with $d$ coefficients.
pub fn quotient_share<const K: usize>(
    ctx: &Context<K>,
    m: &[Vec<Poly>],
    share: &LinearShare,
) -> Result<Vec<Poly>> {
    let l = &ctx.layout;
    check_matrix(m, l.n_prime, l.len_z, l.d)?;
    quotient_share_rows(ctx, |i| Ok(Cow::Borrowed(&m[i][..])), share)
}

/// [`quotient_share`] with the rows of $\mathbf M$ read one at a time from `rows` (row $i$ of the
/// $n'$, each read once, in order), so that $\mathbf M$ need not be held whole
/// ([`hachi_co_core::MRows`]). Each row of the result depends on its row of $\mathbf M$ only, so
/// it is the result of [`quotient_share`].
///
/// # Errors
/// As [`quotient_share`], for a row of the wrong shape when it is read, and those of `rows`.
pub fn quotient_share_rows<'m, const K: usize>(
    ctx: &Context<K>,
    mut rows: impl FnMut(usize) -> Result<Cow<'m, [Poly]>>,
    share: &LinearShare,
) -> Result<Vec<Poly>> {
    let l = &ctx.layout;
    let (d, q) = (l.d, ctx.q());
    let two_d = 2 * d;
    share.validate(l)?;
    let zq = ctx.ring.zq();
    let inv2 = zq
        .inv(2)
        .ok_or_else(|| Error::Parameter("q must be odd".into()))?;
    let nonzero: Vec<usize> = (0..l.len_z)
        .filter(|j| !is_zero(&share.z_prime[*j], q))
        .collect();
    let mut out = Vec::with_capacity(l.n_prime);
    for (i, y) in share.y.iter().enumerate() {
        let row = rows(i)?;
        check_row(&row, l.len_z, d)?;
        // The columns in order, filtered in parallel.
        let columns: Vec<usize> = nonzero
            .par_iter()
            .copied()
            .filter(|j| !is_zero(&row[*j], q))
            .collect();
        // P = M_i <z'>_p over Z_q[X], degree <= 2d - 2: the chunks' products in parallel, added
        // in order (exact modulo q).
        let parts = columns
            .par_chunks(CHUNK)
            .map(|chunk| {
                let a: Vec<Poly> = chunk.iter().map(|j| pad(&row[*j], two_d)).collect();
                let b: Vec<Poly> = chunk
                    .iter()
                    .map(|j| pad(&share.z_prime[*j], two_d))
                    .collect();
                Ok(ctx.ring2.inner(&a, &b)?)
            })
            .collect::<Result<Vec<Poly>>>()?;
        let mut full = vec![0u64; two_d];
        for part in &parts {
            for (acc, x) in full.iter_mut().zip(&part.0) {
                *acc = zq.add(*acc, *x);
            }
        }
        // A row built for this call is dropped in parallel (its ring elements are many).
        if let Cow::Owned(row) = row {
            row.into_par_iter().for_each(drop);
        }
        for (acc, yj) in full.iter_mut().zip(&y.0) {
            *acc = zq.sub(*acc, *yj);
        }
        let (lo, hi) = full.split_at(d);
        let mut r = vec![0u64; two_d];
        for j in 0..d {
            r[j] = zq.mul(zq.add(lo[j], hi[j]), inv2);
            r[d + j] = zq.mul(zq.sub(hi[j], lo[j]), inv2);
        }
        out.push(Poly(r));
    }
    Ok(out)
}

/// Checks that `m` has `rows` rows of `cols` polynomials with `d` coefficients each.
pub(crate) fn check_matrix(m: &[Vec<Poly>], rows: usize, cols: usize, d: usize) -> Result<()> {
    check_len("M (rows)", rows, m.len())?;
    for row in m {
        check_row(row, cols, d)?;
    }
    Ok(())
}

/// Checks that a row of $\mathbf M$ has `cols` polynomials with `d` coefficients each.
pub(crate) fn check_row(row: &[Poly], cols: usize, d: usize) -> Result<()> {
    check_len("M (columns)", cols, row.len())?;
    for p in row {
        check_len("M (coefficients)", d, p.0.len())?;
    }
    Ok(())
}
