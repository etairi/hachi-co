//! The linear relation of one iteration (paper, Eq. (20)) and its ring switching (Section 4.3).
//!
//! The witness $\mathbf z' = (\hat{\mathbf w}, \hat{\mathbf t}, \hat{\mathbf z})$ satisfies, over
//! $\mathcal R_q$, the rows
//! 1. $\mathbf D \hat{\mathbf w} = \mathbf v$ ($n_D$ rows);
//! 2. $\mathbf B \hat{\mathbf t} = \mathbf u$ ($n_B$ rows, the commitment);
//! 3. $\mathbf b^\top \mathbf G \hat{\mathbf w} = T$ (the claim $\sum_i b_i \mathbf a^\top
//!    \mathbf f_i = T$, since $w_i = \mathbf a^\top \mathbf f_i$);
//! 4. $(\mathbf c^\top \otimes \mathbf G_1) \hat{\mathbf w} - \mathbf a^\top \mathbf G \mathbf J
//!    \hat{\mathbf z} = 0$ ($\mathbf G$ only for a decomposed witness);
//! 5. $(\mathbf c^\top \otimes \mathbf G_{n_A}) \hat{\mathbf t} - \mathbf A \mathbf J \hat{\mathbf
//!    z} = 0$ ($n_A$ rows).
//!
//! Over $\mathbb Z_q\[X\]$ each row holds up to $(X^d + 1) q_i$; the quotients are decomposed,
//! $\hat q_i = \mathbf G^{-1}(q_i)$, so that every entry of the next witness is a digit. The next
//! witness is the table $\tilde w$ of the coefficients of $(\mathbf z', \hat{\mathbf q})$, ring
//! element $u$ and coefficient $\ell$ at index $u d + \ell$, padded with zeros to $2^L$ entries.
//!
//! **Columns** (ring elements), in order:
//! - $\hat{\mathbf w}$ at $[0, 2^r\delta)$, chunk $i$ and digit $k$ at $i\delta + k$;
//! - $\hat{\mathbf t}$, chunk $j$, row $i$ and digit $k$ at offset $j n_A\delta + i\delta + k$;
//! - $\hat{\mathbf z}$, element $x$ and digit $t$ at offset $x\tau + t$;
//! - $\hat{\mathbf q}$, row $i$ and digit $k$ at $\mu + i\delta + k$.

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
    /// $\mathbf a \in \mathcal R_q^{2^m}$ of the claim.
    pub a: &'a [Poly],
    /// The claimed value $T \in \mathcal R_q$.
    pub target: &'a Poly,
    /// The challenges $c_i$ as ring elements.
    pub c: &'a [Poly],
    /// $\mathbf D$ ($n_D \times 2^r\delta$).
    pub d_mat: &'a [Vec<Poly>],
    /// $\mathbf B$ ($n_B \times 2^r n_A \delta$).
    pub b_mat: &'a [Vec<Poly>],
    /// $\mathbf A$ ($n_A \times W$).
    pub a_mat: &'a [Vec<Poly>],
    /// $\mathbf v$ ($n_D$).
    pub v: &'a [Poly],
    /// The commitment $\mathbf u$ ($n_B$).
    pub u: &'a [Poly],
}

/// The column offsets of the table ([`Public`], module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Columns {
    /// Start of $\hat{\mathbf t}$.
    pub t: usize,
    /// Start of $\hat{\mathbf z}$.
    pub z: usize,
    /// Start of $\hat{\mathbf q}$ ($= \mu$).
    pub q: usize,
    /// The number of ring-element columns, $2^{L-\alpha}$.
    pub total: usize,
}

impl Columns {
    /// The offsets of an iteration.
    pub fn of(params: &IterParams, n_a: usize, alpha: usize) -> Self {
        let chunks = params.layout.chunks();
        let t = chunks * params.delta;
        let z = t + chunks * n_a * params.delta;
        Self {
            t,
            z,
            q: params.mu,
            total: 1 << (params.table_vars - alpha),
        }
    }
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
    let mut acc = ring2.zero();
    for (a, b) in x.iter().zip(y) {
        acc = ring2.add(&acc, &ring2.mul(&lift(ring2, a), &lift(ring2, b))?)?;
    }
    Ok(acc)
}

/// The quotients $q_i$ of the rows: $(\text{row}_i(\mathbf z') - y_i) = (X^d + 1) q_i$ in
/// $\mathbb Z_q\[X\]$, from the undecomposed values $\mathbf w$, the inner commitments
/// $\mathbf t_j$ and $\mathbf z$ (equal, digit for digit, to the recompositions of
/// $\hat{\mathbf w}$, $\hat{\mathbf t}$ and $\hat{\mathbf z}$).
///
/// # Errors
/// [`Error::Protocol`] if a row does not vanish modulo $X^d + 1$ (a bug); arithmetic errors.
pub fn quotients<const K: usize>(
    ctx: &Ctx<K>,
    public: &Public<'_>,
    w: &[Poly],
    w_hat: &[Poly],
    t: &[Vec<Poly>],
    t_hat: &[Poly],
    z: &[Poly],
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
    let mut sums: Vec<(Poly, Poly)> = Vec::new();
    for (i, row) in public.d_mat.iter().enumerate() {
        sums.push((exact_inner(ring2, row, w_hat)?, public.v[i].clone()));
    }
    for (i, row) in public.b_mat.iter().enumerate() {
        sums.push((exact_inner(ring2, row, t_hat)?, public.u[i].clone()));
    }
    sums.push((exact_inner(ring2, public.b, w)?, public.target.clone()));
    let fold = ring2.sub(
        &exact_inner(ring2, public.c, w)?,
        &exact_inner(ring2, public.a, &z_msg)?,
    )?;
    sums.push((fold, ring.zero()));
    for (i, row) in public.a_mat.iter().enumerate() {
        let ti: Vec<Poly> = t.iter().map(|tj| tj[i].clone()).collect();
        let s = ring2.sub(
            &exact_inner(ring2, public.c, &ti)?,
            &exact_inner(ring2, row, z)?,
        )?;
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
    let (n_a, n_b, n_d, delta, tau) = (
        ctx.choices.n_a,
        ctx.choices.n_b,
        ctx.choices.n_d,
        p.delta,
        p.tau,
    );
    let rows = Rows::of(ctx);
    let cols = Columns::of(p, n_a, ctx.alpha());
    let chunks = p.layout.chunks();
    let width = p.layout.width(delta);
    let d = ctx.d();
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
