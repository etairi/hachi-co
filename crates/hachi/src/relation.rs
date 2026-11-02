//! The linear relation of one iteration (paper, Eq. (20)) and its ring switching (Section 4.3).
//!
//! The witness $\mathbf z' = (\hat{\mathbf w}, \hat{\mathbf t}, \hat{\mathbf z})$ satisfies, over
//! $\mathcal R_q$, the rows
//! 1. $\mathbf D \hat{\mathbf w} = \mathbf v$ ($n_D$ rows).
//! 2. $\mathbf B \hat{\mathbf t} = \mathbf u$ ($n_B$ rows, the commitment);
//! 3. $\mathbf b^\top \mathbf G \hat{\mathbf w} = T$ (the claim $\sum_i b_i \mathbf a^\top
//!    \mathbf f_i = T$, since $w_i = \mathbf a^\top \mathbf f_i$);
//! 4. $(\mathbf c^\top \otimes \mathbf G_1) \hat{\mathbf w} - \mathbf a^\top \mathbf G \mathbf J
//!    \hat{\mathbf z} = 0$ ($\mathbf G$ only for a decomposed witness);
//! 5. $(\mathbf c^\top \otimes \mathbf G_{n_A}) \hat{\mathbf t} - \mathbf A \mathbf J \hat{\mathbf
//!    z} = 0$ ($n_A$ rows).
//!
//! At a masked level ([`Hiding`](crate::params::Hiding), `hachi-pcs.tex`, step 2(c) of `Eval`)
//! the witness is $\mathbf z' = (\hat{\mathbf w}, \hat w_0, \hat{\mathbf t}_0, \hat{\mathbf t},
//! \hat{\mathbf z}, \hat{\mathbf z}_\varrho)$, row 1 applies $\mathbf D$ to $(\hat{\mathbf w},
//! \hat w_0, \hat{\mathbf t}_0)$, row 4 gains $\mathbf G_1 \hat w_0$, and row 5 gains
//! $\mathbf G_{n_A} \hat{\mathbf t}_0 - \mathbf A' \mathbf J \hat{\mathbf z}_\varrho$, since
//! $\sum_i c_i w_i + w_0 = \mathbf a^\top \mathbf G \mathbf z$ and $\sum_i c_i \mathbf t_i +
//! \mathbf t_0 = \mathbf A \mathbf z + \mathbf A' \mathbf z_\varrho$.
//!
//! Over $\mathbb Z_q\[X\]$ each row holds up to $(X^d + 1) q_i$; the quotients are decomposed,
//! $\hat q_i = \mathbf G^{-1}(q_i)$, so that every entry of the next witness is a digit. The next
//! witness is the table $\tilde w$ of the coefficients of $(\mathbf z', \hat{\mathbf q})$, ring
//! element $u$ and coefficient $\ell$ at index $u d + \ell$, padded with zeros to $2^L$ entries.
//!
//! **Columns** (ring elements), in order:
//! - $\hat{\mathbf w}$ at $[0, 2^r\delta)$, chunk $i$ and digit $k$ at $i\delta + k$;
//! - at a masked level, $\hat w_0$, digit $k$ at offset $k$, then $\hat{\mathbf t}_0$, row $i$ and
//!   digit $k$ at offset $i\delta + k$.
//! - $\hat{\mathbf t}$, chunk $j$, row $i$ and digit $k$ at offset $j n_A\delta + i\delta + k$;
//! - $\hat{\mathbf z}$, element $x$ and digit $t$ at offset $x\tau + t$;
//! - at a masked level, $\hat{\mathbf z}_\varrho$, entry $x$ and digit $t$ at offset $x\tau + t$.
//! - $\hat{\mathbf q}$, row $i$ and digit $k$ at $\mu + i\delta + k$.
//!
//! In $\mathbf A' = [\mathbf A_1' \mid \mathbf I_{n_A}]$, entry $x < \nu$ of the randomness meets
//! column $x$ of $\mathbf A_1'$, and entry $\nu + i$ meets row $i$ alone.

use crate::commit::Ctx;
use crate::mle::eval_poly;
use crate::params::IterParams;
use crate::{Error, Result};
use hachi_co_math::decompose::g_inv;
use hachi_co_math::{Ext, Fqk, Poly, Ring};

/// The public data of one iteration's relation.
#[derive(Clone, Copy, Debug)]
pub struct Public<'a> {
    /// The iteration's parameters.
    pub params: &'a IterParams,
    /// $\mathbf b \in \mathcal R_q^{2^r}$ of the claim.
    pub b: &'a [Poly],
    /// $\mathbf a$ of the claim, one entry per ring element of a chunk.
    pub a: &'a [Poly],
    /// The claimed value $T \in \mathcal R_q$.
    pub target: &'a Poly,
    /// The challenges $c_i$ as ring elements.
    pub c: &'a [Poly],
    /// $\mathbf D$ ($n_D \times$ [`IterParams::d_cols`]).
    pub d_mat: &'a [Vec<Poly>],
    /// $\mathbf B$ ($n_B \times 2^r n_A \delta$).
    pub b_mat: &'a [Vec<Poly>],
    /// $\mathbf A$ ($n_A \times W$).
    pub a_mat: &'a [Vec<Poly>],
    /// $\mathbf A_1'$ ($n_A \times \nu$) at a masked level, else empty.
    pub a_rand: &'a [Vec<Poly>],
    /// $\mathbf v$ ($n_D$).
    pub v: &'a [Poly],
    /// The commitment $\mathbf u$ ($n_B$).
    pub u: &'a [Poly],
}

/// The column offsets of the table ([`Public`], module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Columns {
    /// Start of $\hat w_0$ (equal to the start of $\hat{\mathbf t}$ without masks).
    pub w0: usize,
    /// Start of $\hat{\mathbf t}_0$ (equal to the start of $\hat{\mathbf t}$ without masks).
    pub t0: usize,
    /// Start of $\hat{\mathbf t}$.
    pub t: usize,
    /// Start of $\hat{\mathbf z}$.
    pub z: usize,
    /// Start of $\hat{\mathbf z}_\varrho$ (equal to the start of $\hat{\mathbf q}$ without masks).
    pub r: usize,
    /// Start of $\hat{\mathbf q}$ ($= \mu$).
    pub q: usize,
    /// The number of ring-element columns, $2^{L-\alpha}$.
    pub total: usize,
}

impl Columns {
    /// The offsets of an iteration.
    pub fn of(params: &IterParams, alpha: usize) -> Self {
        let (chunks, delta, n_a) = (params.layout.chunks(), params.delta, params.n_a);
        let w0 = chunks * delta;
        let (t0, t) = if params.layout.masked {
            (w0 + delta, w0 + delta + n_a * delta)
        } else {
            (w0, w0)
        };
        let z = t + chunks * n_a * delta;
        Self {
            w0,
            t0,
            t,
            z,
            r: z + params.layout.width(delta) * params.tau,
            q: params.mu,
            total: 1 << (params.table_vars - alpha),
        }
    }
}

/// The masked parts of the witness of a level with masks, undecomposed like the arguments of
/// [`quotients`]: $w_0$, $\hat w_0$, $\mathbf t_0$, $\hat{\mathbf t}_0$ and $\mathbf z_\varrho$.
#[derive(Clone, Copy, Debug)]
pub struct Masked<'a> {
    /// $w_0$.
    pub w0: &'a Poly,
    /// $\hat w_0 = \mathbf G_1^{-1}(w_0)$.
    pub w0_hat: &'a [Poly],
    /// $\mathbf t_0$.
    pub t0: &'a [Poly],
    /// $\hat{\mathbf t}_0 = \mathbf G_{n_A}^{-1}(\mathbf t_0)$.
    pub t0_hat: &'a [Poly],
    /// $\mathbf z_\varrho$.
    pub z_r: &'a [Poly],
}

/// The rows of the relation: $n_D$, $n_B$, the claim row, the fold row and $n_A$ rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rows {
    claim: usize,
    fold: usize,
    a_rows: usize,
}

impl Rows {
    fn of<const K: usize>(ctx: &Ctx<K>) -> Self {
        let claim = ctx.choices.n_d + ctx.choices.n_b;
        Self {
            claim,
            fold: claim + 1,
            a_rows: claim + 2,
        }
    }
}

/// $b^k \bmod q$ for $k < n$.
fn base_powers(ring: &Ring, b: u64, n: usize) -> Vec<u64> {
    let zq = ring.zq();
    let mut out = Vec::with_capacity(n);
    let mut p = 1u64;
    for _ in 0..n {
        out.push(p);
        p = zq.mul(p, zq.reduce(b));
    }
    out
}

/// $p$ read as an element of $\mathbb Z_q\[X\]/(X^{2d} + 1)$ (degree below $d$).
fn lift(ring2: &Ring, p: &Poly) -> Poly {
    let mut out = ring2.zero();
    out.0[..p.0.len()].copy_from_slice(&p.0);
    out
}

/// $\sum_j x_j y_j$ in $\mathbb Z_q\[X\]$, exactly (degrees below $d$), as an element of the ring
/// of degree $2d$.
fn exact_inner(ring2: &Ring, x: &[Poly], y: &[Poly]) -> Result<Poly> {
    Ok(ring2.inner_padded(x, y)?)
}

/// The quotients $q_i$ of the rows: $(\text{row}_i(\mathbf z') - y_i) = (X^d + 1) q_i$ in
/// $\mathbb Z_q\[X\]$, from the undecomposed values $\mathbf w$, the inner commitments
/// $\mathbf t_j$, $\mathbf z$, and at a masked level the parts in `masked` (equal, digit for digit,
/// to the recompositions of the digits).
///
/// # Errors
/// [`Error::Shape`] for masks of the wrong shape. [`Error::Protocol`] if a row does not vanish
/// modulo $X^d + 1$ (a bug). Arithmetic errors.
#[allow(clippy::too_many_arguments)] // the parts of the witness
pub fn quotients<const K: usize>(
    ctx: &Ctx<K>,
    public: &Public<'_>,
    w: &[Poly],
    w_hat: &[Poly],
    t: &[Vec<Poly>],
    t_hat: &[Poly],
    z: &[Poly],
    masked: Option<&Masked<'_>>,
) -> Result<Vec<Poly>> {
    let (ring, ring2, d) = (&ctx.ring, &ctx.ring2, ctx.d());
    let p = public.params;
    let zq = ring.zq();
    let pow = base_powers(ring, ctx.choices.b, p.delta.max(p.tau));
    // The message-level elements of z: G z if decomposed, else z itself.
    let z_msg: Vec<Poly> = if p.layout.decomposed {
        z.chunks(p.delta)
            .map(|digits| {
                let mut acc = ring.zero();
                for (k, x) in digits.iter().enumerate() {
                    acc = ring.add(&acc, &ring.scalar_mul(pow[k], x)?)?;
                }
                Ok(acc)
            })
            .collect::<hachi_co_math::Result<_>>()?
    } else {
        z.to_vec()
    };
    if masked.is_some() != p.layout.masked {
        return Err(Error::Shape(
            "the masks do not fit the iteration's layout".into(),
        ));
    }
    let nu = ctx.hiding().map_or(0, |h| h.nu);
    let mut d_columns: Vec<Poly> = w_hat.to_vec();
    if let Some(m) = masked {
        if public.a_rand.len() != p.n_a
            || public.a_rand.iter().any(|row| row.len() != nu)
            || m.z_r.len() != nu + p.n_a
            || m.t0.len() != p.n_a
        {
            return Err(Error::Shape(
                "the masks do not fit the hiding choices".into(),
            ));
        }
        d_columns.extend_from_slice(m.w0_hat);
        d_columns.extend_from_slice(m.t0_hat);
    }
    // [A1' | I] r for row i, exactly: the matrix's row i against the first nu entries, plus entry
    // nu + i.
    let rand_row = |m1: &[Vec<Poly>], r: &[Poly], i: usize| -> Result<Poly> {
        let prod = exact_inner(ring2, &m1[i], &r[..nu])?;
        Ok(ring2.add(&prod, &lift(ring2, &r[nu + i]))?)
    };
    let mut sums: Vec<(Poly, Poly)> = Vec::new();
    for (i, row) in public.d_mat.iter().enumerate() {
        sums.push((exact_inner(ring2, row, &d_columns)?, public.v[i].clone()));
    }
    for (i, row) in public.b_mat.iter().enumerate() {
        sums.push((exact_inner(ring2, row, t_hat)?, public.u[i].clone()));
    }
    sums.push((exact_inner(ring2, public.b, w)?, public.target.clone()));
    let mut fold = exact_inner(ring2, public.c, w)?;
    if let Some(m) = masked {
        fold = ring2.add(&fold, &lift(ring2, m.w0))?;
    }
    let fold = ring2.sub(&fold, &exact_inner(ring2, public.a, &z_msg)?)?;
    sums.push((fold, ring.zero()));
    for (i, row) in public.a_mat.iter().enumerate() {
        let ti: Vec<Poly> = t.iter().map(|tj| tj[i].clone()).collect();
        let mut s = exact_inner(ring2, public.c, &ti)?;
        if let Some(m) = masked {
            s = ring2.add(&s, &lift(ring2, &m.t0[i]))?;
        }
        let mut s = ring2.sub(&s, &exact_inner(ring2, row, z)?)?;
        if let Some(m) = masked {
            s = ring2.sub(&s, &rand_row(public.a_rand, m.z_r, i)?)?;
        }
        sums.push((s, ring.zero()));
    }
    let mut out = Vec::with_capacity(sums.len());
    for (k, (s, y)) in sums.into_iter().enumerate() {
        let diff = ring2.sub(&s, &lift(ring2, &y))?;
        let (low, high) = diff.0.split_at(d);
        if low
            .iter()
            .zip(high)
            .any(|(l, h)| zq.reduce(*l) != zq.reduce(*h))
        {
            return Err(Error::Protocol(format!(
                "row {k} of the relation does not vanish modulo X^d + 1"
            )));
        }
        out.push(Poly(high.to_vec()));
    }
    Ok(out)
}

/// The next witness table $\tilde w$: the coefficients of $(\mathbf z', \hat{\mathbf q})$ with
/// $\hat{\mathbf q} = \mathbf G^{-1}(\mathbf q)$, padded with zeros to $2^L$ residues.
///
/// # Errors
/// Arithmetic errors; [`Error::Shape`] if the parts do not fit the parameters.
pub fn next_table<const K: usize>(
    ctx: &Ctx<K>,
    params: &IterParams,
    z_prime: &[Poly],
    q: &[Poly],
) -> Result<Vec<u64>> {
    let q_hat = g_inv(&ctx.ring, ctx.choices.b, ctx.delta, q)?;
    let d = ctx.d();
    if z_prime.len() != params.mu || q_hat.len() != params.rows * ctx.delta {
        return Err(Error::Shape(
            "the next witness does not fit its parameters".into(),
        ));
    }
    let mut table = vec![0u64; 1 << params.table_vars];
    for (u, p) in z_prime.iter().chain(&q_hat).enumerate() {
        table[u * d..(u + 1) * d].copy_from_slice(&p.0);
    }
    Ok(table)
}

/// The public weights of the linear sumcheck: $m^\ast(u) = \sum_i \widetilde{eq}(\tau_1, i)
/// M_\alpha(i, u)$ for every column $u$ ($2^{L-\alpha}$ values), and the claimed sum
/// $\sum_i \widetilde{eq}(\tau_1, i) y_i(\alpha)$. `eq1` holds $\widetilde{eq}(\tau_1, i)$ for the
/// rows (padded rows have weight but no entries) and `alpha_pows` holds $\alpha^\ell$,
/// $\ell \le d$.
///
/// # Errors
/// [`Error::Shape`] for public data of the wrong shape.
pub fn mstar<const K: usize>(
    ctx: &Ctx<K>,
    public: &Public<'_>,
    eq1: &[Ext<K>],
    alpha_pows: &[Ext<K>],
) -> Result<(Vec<Ext<K>>, Ext<K>)> {
    let field: &Fqk<K> = ctx.field();
    let p = public.params;
    let (n_a, n_b, n_d, delta, tau) = (p.n_a, ctx.choices.n_b, ctx.choices.n_d, p.delta, p.tau);
    let rows = Rows::of(ctx);
    let nu = ctx.hiding().map_or(0, |h| h.nu);
    let cols = Columns::of(p, ctx.alpha());
    let chunks = p.layout.chunks();
    let width = p.layout.width(delta);
    let d = ctx.d();
    let rand_rows = if p.layout.masked { n_a } else { 0 };
    if eq1.len() < p.rows
        || alpha_pows.len() < d + 1
        || public.b.len() != chunks
        || public.c.len() != chunks
        || public.a.len() != p.layout.chunk_len()
        || public.d_mat.len() != n_d
        || public.b_mat.len() != n_b
        || public.a_mat.len() != n_a
        || public.v.len() != n_d
        || public.u.len() != n_b
        || public.a_rand.len() != rand_rows
        || public.a_rand.iter().any(|row| row.len() != nu)
    {
        return Err(Error::Shape(
            "the relation's public data does not fit its parameters".into(),
        ));
    }
    let ev = |x: &Poly| eval_poly(field, x, &alpha_pows[..d]);
    let pow = base_powers(&ctx.ring, ctx.choices.b, delta.max(tau));
    let mut m = vec![field.zero(); cols.total];
    // w-hat.
    let (b_alpha, c_alpha): (Vec<Ext<K>>, Vec<Ext<K>>) = (
        public.b.iter().map(ev).collect(),
        public.c.iter().map(ev).collect(),
    );
    for (r, row) in public.d_mat.iter().enumerate() {
        for (col, x) in row.iter().enumerate() {
            m[col] = field.add(&m[col], &field.mul(&eq1[r], &ev(x)));
        }
    }
    for (i, (bi, ci)) in b_alpha.iter().zip(&c_alpha).enumerate() {
        let claim = field.mul(&eq1[rows.claim], bi);
        let fold = field.mul(&eq1[rows.fold], ci);
        let both = field.add(&claim, &fold);
        for (k, pk) in pow[..delta].iter().enumerate() {
            let col = i * delta + k;
            m[col] = field.add(&m[col], &field.scale(&both, *pk));
        }
    }
    // w0-hat (row 4) and t0-hat (rows 5), at a masked level.
    if p.layout.masked {
        for (k, pk) in pow[..delta].iter().enumerate() {
            let c = cols.w0 + k;
            m[c] = field.add(&m[c], &field.scale(&eq1[rows.fold], *pk));
            for i in 0..n_a {
                let c = cols.t0 + i * delta + k;
                m[c] = field.add(&m[c], &field.scale(&eq1[rows.a_rows + i], *pk));
            }
        }
    }
    // t-hat.
    for (r, row) in public.b_mat.iter().enumerate() {
        for (col, x) in row.iter().enumerate() {
            let c = cols.t + col;
            m[c] = field.add(&m[c], &field.mul(&eq1[n_d + r], &ev(x)));
        }
    }
    for (j, cj) in c_alpha.iter().enumerate() {
        for i in 0..n_a {
            let weight = field.mul(&eq1[rows.a_rows + i], cj);
            for (k, pk) in pow[..delta].iter().enumerate() {
                let c = cols.t + j * n_a * delta + i * delta + k;
                m[c] = field.add(&m[c], &field.scale(&weight, *pk));
            }
        }
    }
    // z-hat.
    let a_alpha: Vec<Ext<K>> = public.a.iter().map(ev).collect();
    let a_mat_alpha: Vec<Vec<Ext<K>>> = public
        .a_mat
        .iter()
        .map(|row| row.iter().map(ev).collect())
        .collect();
    for x in 0..width {
        let (j, g) = if p.layout.decomposed {
            (x / delta, pow[x % delta])
        } else {
            (x, 1)
        };
        let mut base = field.scale(&field.mul(&eq1[rows.fold], &a_alpha[j]), g);
        for i in 0..n_a {
            base = field.add(&base, &field.mul(&eq1[rows.a_rows + i], &a_mat_alpha[i][x]));
        }
        let neg = field.neg(&base);
        for (t, pt) in pow[..tau].iter().enumerate() {
            m[cols.z + x * tau + t] = field.scale(&neg, *pt);
        }
    }
    // z_varrho-hat (rows 5, -A'), digit t weighted by b^t.
    for (i, row) in public.a_rand.iter().enumerate() {
        let mut entries: Vec<(usize, Ext<K>)> =
            row.iter().enumerate().map(|(x, m)| (x, ev(m))).collect();
        entries.push((nu + i, field.one()));
        for (x, e) in entries {
            let weight = field.neg(&field.mul(&eq1[rows.a_rows + i], &e));
            for (t, pt) in pow[..tau].iter().enumerate() {
                let c = cols.r + x * tau + t;
                m[c] = field.add(&m[c], &field.scale(&weight, *pt));
            }
        }
    }
    // q-hat: -(alpha^d + 1) b^k in the row's own columns.
    let wrap = field.add(&alpha_pows[d], &field.one());
    for (r, e) in eq1[..p.rows].iter().enumerate() {
        let weight = field.neg(&field.mul(e, &wrap));
        for (k, pk) in pow[..delta].iter().enumerate() {
            m[cols.q + r * delta + k] = field.scale(&weight, *pk);
        }
    }
    // The claimed sum.
    let mut sum = field.zero();
    for (r, v) in public.v.iter().enumerate() {
        sum = field.add(&sum, &field.mul(&eq1[r], &ev(v)));
    }
    for (r, u) in public.u.iter().enumerate() {
        sum = field.add(&sum, &field.mul(&eq1[n_d + r], &ev(u)));
    }
    sum = field.add(&sum, &field.mul(&eq1[rows.claim], &ev(public.target)));
    Ok((m, sum))
}
