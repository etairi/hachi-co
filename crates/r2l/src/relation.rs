//! Relation checkers (test oracles): $\mathbf M \mathbf z' = \mathbf y$ over $\mathcal R\_q$ for
//! $\mathbf z' = \sum\_p \langle \mathbf z' \rangle\_p$ (`5_Building_Blocks.tex`:154-182), and the
//! meaning of each row block of $\mathbf M$ checked on the unpacked values (line 182; the oracle's
//! `row_block_checks` in `ref/hachico_ref/oracle.py`).

use crate::local::{a_times_g, check_len, response_len};
use crate::{LinearRelation, Result};
use hachi_co_core::{Context, LinearShare, Poly, PublicVectors};
use hachi_co_math::decompose;

/// $\sum\_p \langle \mathbf z' \rangle\_p$ and $\sum\_p \lbrace \mathbf y \rbrace\_p$ over
/// $\mathcal R\_q$ ($\mathsf{AS}.\mathsf{Open}$, `4_preliminaries.tex`:157-158): the whole witness and right-hand
/// side from the parties' shares.
///
/// # Errors
/// [`Error::Shape`](crate::Error::Shape) for no shares; [`Error::Core`](crate::Error::Core) for a
/// share of the wrong shape.
pub fn sum_shares<const K: usize>(ctx: &Context<K>, shares: &[LinearShare]) -> Result<LinearShare> {
    let l = &ctx.layout;
    if shares.is_empty() {
        return Err(crate::Error::Shape {
            what: "shares (at least one)",
            expected: l.n,
            actual: 0,
        });
    }
    let mut sum = LinearShare::zero(l);
    for share in shares {
        share.validate(l)?;
        for (s, x) in sum.z_prime.iter_mut().zip(&share.z_prime) {
            *s = ctx.ring.add(s, x)?;
        }
        for (s, x) in sum.y.iter_mut().zip(&share.y) {
            *s = ctx.ring.add(s, x)?;
        }
    }
    Ok(sum)
}

/// The residual $\mathbf M \mathbf z' - \mathbf y \in \mathcal R\_q^{n'}$ (exact products,
/// [`hachi_co_math::Ring::mat_vec`]).
///
/// # Errors
/// [`Error::Shape`](crate::Error::Shape) unless $\mathbf z'$ has $\mu$ entries and
/// $\mathbf M$, $\mathbf y$ have $n'$ rows; [`Error::Math`](crate::Error::Math) for entries
/// without $d$ coefficients.
pub fn residual<const K: usize>(
    ctx: &Context<K>,
    rel: &LinearRelation,
    z_prime: &[Poly],
) -> Result<Vec<Poly>> {
    let l = &ctx.layout;
    check_len("z'", l.mu, z_prime.len())?;
    check_len("rows of M", l.n_prime, rel.m.len())?;
    check_len("y", l.n_prime, rel.y.len())?;
    for row in &rel.m {
        check_len("row of M", l.mu, row.len())?;
    }
    let mz = ctx.ring.mat_vec(&rel.m, z_prime)?;
    mz.iter()
        .zip(&rel.y)
        .map(|(a, b)| Ok(ctx.ring.sub(a, b)?))
        .collect()
}

/// Whether $\mathbf M \mathbf z' = \mathbf y$ over $\mathcal R\_q$ (`5_Building_Blocks.tex`:154;
/// the oracle's `check_linear_relation`): [`residual`] is zero.
///
/// # Errors
/// As [`residual`].
pub fn relation_holds<const K: usize>(
    ctx: &Context<K>,
    rel: &LinearRelation,
    z_prime: &[Poly],
) -> Result<bool> {
    Ok(residual(ctx, rel, z_prime)?
        .iter()
        .all(|p| p.0.iter().all(|x| *x == 0)))
}

/// The values the row blocks of $\mathbf M$ speak about, before decomposition
/// (`5_Building_Blocks.tex`:103-133), for [`row_block_checks`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnpackedWitness {
    /// $w\_i$ for every block $i \lt 2^r$ ($0$ for an unowned block), line 106.
    pub w: Vec<Poly>,
    /// $\hat{\mathbf w} = (\hat{\mathbf w}\_i)\_{i \lt 2^r}$, $2^r\delta$ entries, line 107.
    pub w_hat: Vec<Poly>,
    /// $w\_{0,p}$ of the accepted attempt for $p = 1, \dots, n$ (entry $p - 1$), line 113.
    pub w0: Vec<Poly>,
    /// $\mathbf t\_{0,p}$ of the accepted attempt for $p = 1, \dots, n$, $n\_A$ entries each,
    /// line 112.
    pub t0: Vec<Vec<Poly>>,
    /// $\mathbf t\_i = \mathbf A \mathbf s\_i$ for every block $i \lt 2^r$ ($\mathbf 0$ for an
    /// unowned block), $n\_A$ entries each (`6_Collaborative_Hachi.tex`:374).
    pub t: Vec<Vec<Poly>>,
    /// $\mathbf z = \sum\_p \mathbf z\_p$ over the integers, flat, $N$ entries (line 130).
    pub z: Vec<i64>,
    /// $u = \sum\_p \langle u \rangle\_p$ ($\Pi\_{\mathsf{conv}}$).
    pub u: Poly,
}

/// The outcome of [`row_block_checks`], one flag per row block of $\mathbf M$ with a meaning
/// beyond the commitments (the oracle's names in parentheses).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowBlockChecks {
    /// Row $\mathbf b^T \mathbf G\_{2^r} \mid -1$: $\sum\_i b\_i \mathbf G\_1 \hat{\mathbf w}\_i =
    /// u$ (`r2l.row_b: sum_i b_i w_i = u`; `5_Building_Blocks.tex`:160).
    pub row_b: bool,
    /// Row $\mathbf c$: $\sum\_i c\_i w\_i + \sum\_p w\_{0,p} = \mathbf a^T \mathbf G\_{2^m}
    /// \mathbf z$ (`r2l.row_c: ...`; line 161 and 182).
    pub row_c: bool,
    /// Rows $\mathbf A$: $\sum\_i c\_i \mathbf t\_i + \sum\_p \mathbf t\_{0,p} = \mathbf A
    /// \mathbf z$ (`r2l.row_A: ...`; line 162 and 182).
    pub row_a: bool,
}

impl RowBlockChecks {
    /// Whether all three hold.
    pub fn all(&self) -> bool {
        self.row_b && self.row_c && self.row_a
    }
}

/// The meaning of the row blocks of $\mathbf M$ (`5_Building_Blocks.tex`:160-162, 182), checked
/// directly on the unpacked values with the challenges $\mathbf c$, as the oracle's
/// `row_block_checks`: $\sum\_i b\_i \mathbf G\_1 \hat{\mathbf w}\_i = u$,
/// $\sum\_i c\_i w\_i + \sum\_p w\_{0,p} = \mathbf a^T \mathbf G\_{2^m} \mathbf z$ and
/// $\sum\_i c\_i \mathbf t\_i + \sum\_p \mathbf t\_{0,p} = \mathbf A \mathbf z$ over
/// $\mathcal R\_q$, with $\mathbf z$ reduced modulo $q$. Unlike $\mathbf M \mathbf z' =
/// \mathbf y$, these use $w\_i$, $\mathbf t\_i$, $w\_{0,p}$, $\mathbf t\_{0,p}$ and $\mathbf z$
/// before their decompositions, so they also test that $\hat{\mathbf w}$, $\hat{\mathbf t}\_0$,
/// $\hat{\mathbf w}\_0$ and $\hat{\mathbf z}$ decompose the right values.
///
/// # Errors
/// [`Error::Core`](crate::Error::Core) if the public vectors have the wrong shape;
/// [`Error::Shape`](crate::Error::Shape) for values or challenges of the wrong shape;
/// [`Error::Math`](crate::Error::Math) for polynomials without $d$ coefficients.
pub fn row_block_checks<const K: usize>(
    ctx: &Context<K>,
    pv: &PublicVectors<K>,
    c: &[Vec<i64>],
    wit: &UnpackedWitness,
) -> Result<RowBlockChecks> {
    let l = &ctx.layout;
    let ring = &ctx.ring;
    pv.validate(l)?;
    check_len("c (one challenge per block)", l.n_blocks, c.len())?;
    check_len("w", l.n_blocks, wit.w.len())?;
    check_len("w_hat", l.n_blocks * l.delta, wit.w_hat.len())?;
    check_len("w_0 (one per prover)", l.n, wit.w0.len())?;
    check_len("t_0 (one per prover)", l.n, wit.t0.len())?;
    check_len("t (one per block)", l.n_blocks, wit.t.len())?;
    for t in wit.t0.iter().chain(&wit.t) {
        check_len("t_i or t_0p (n_A entries)", l.n_a, t.len())?;
    }
    check_len("z (N = d delta 2^m)", response_len(ctx), wit.z.len())?;
    let cq = c
        .iter()
        .map(|ci| ring.from_i64(ci))
        .collect::<hachi_co_math::Result<Vec<_>>>()?;
    let z = wit
        .z
        .chunks(l.d)
        .map(|chunk| ring.from_i64(chunk))
        .collect::<hachi_co_math::Result<Vec<_>>>()?;

    // b^T G_{2^r} w_hat = u.
    let w_from_hat = decompose::g_apply(ring, ctx.b(), l.delta, &wit.w_hat)?;
    let row_b = ring.inner(&pv.b_ring, &w_from_hat)? == ring.from_coeffs(&wit.u.0)?;

    // sum_i c_i w_i + sum_p w_0p = a^T G_{2^m} z.
    let mut lhs = ring.inner(&cq, &wit.w)?;
    for w0 in &wit.w0 {
        lhs = ring.add(&lhs, w0)?;
    }
    let row_c = lhs == a_times_g(ctx, &pv.a_ring, &z)?;

    // sum_i c_i t_i + sum_p t_0p = A z.
    let az = ring.mat_vec(&ctx.pp.a, &z)?;
    let mut row_a = true;
    for (a, az_a) in az.iter().enumerate() {
        let t_col: Vec<Poly> = wit.t.iter().map(|t| t[a].clone()).collect();
        let mut lhs = ring.inner(&cq, &t_col)?;
        for t0 in &wit.t0 {
            lhs = ring.add(&lhs, &t0[a])?;
        }
        row_a &= lhs == *az_a;
    }
    Ok(RowBlockChecks {
        row_b,
        row_c,
        row_a,
    })
}
