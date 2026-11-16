//! The hypercube tables and public values of the sumcheck (`5_Building_Blocks.tex`:234-307;
//! oracle `ref/hachico_ref/sumcheck.py`): the witness table $\langle \widetilde w \rangle\_p$, the
//! row indicators $e\_p$ and $\mathrm{ind}$, the range product, the table of
//! $\widetilde M^\ast$, and the claimed sum $a$.
//!
//! **Hypercube** (`ref/README.md`, "Hypercube"; fixture convention `hypercube`):
//! $V = (U, L)$ with $\lvert U \rvert = \ell\_U = \lceil \log(\mu + n') \rceil$ and
//! $\lvert L \rvert = \ell\_L = \log 2d$; the point $(u\_0, l\_0)$ has index $u\_0 \cdot 2d + l\_0$,
//! $V\_1$ the most significant bit. The draft's 1-indexed $\chi\_{\mathsf{bit}(u) \Vert
//! \mathsf{bit}(\ell)}$ is the entry $(u - 1) \cdot 2d + (\ell - 1)$.

use crate::arith::{Arith, Montgomery};
use crate::eq::eq_table;
use crate::error::{Error, Result, check_len, pow2};
use crate::quotient::{check_matrix, check_row};
use core::ops::Range;
use hachi_co_core::{Context, Layout, Poly};
use hachi_co_math::{Ext, Fqk, Zq};
use rayon::prelude::*;
use std::borrow::Cow;

/// $2^{\ell''}$, the number of points of the hypercube.
///
/// # Errors
/// [`Error::Parameter`] if it does not fit in `usize`.
pub fn table_size(layout: &Layout) -> Result<usize> {
    pow2(layout.ell_pp, "the hypercube")
}

/// The hypercube values of party $p$'s witness polynomial $\langle \widetilde w \rangle\_p(U, L)$
/// (`5_Building_Blocks.tex`:254-259; the oracle's `w_table`; fixture convention `hypercube`), over
/// $\mathbb Z\_q$, $2^{\ell''}$ entries, without the pad: the entry $(u\_0, l\_0)$ is coefficient
/// $l\_0$ of $\langle z'\_{u\_0} \rangle\_p$ for $u\_0 \lt \mu - 1$, $l\_0 \lt d$; coefficient
/// $l\_0$ of $\langle r\_{u\_0 - \mu} \rangle\_p$ for $\mu \le u\_0 \lt \mu + n'$,
/// $l\_0 \lt 2d$; and $0$ otherwise, in particular in the pad's row $\mu - 1$. Residues are
/// reduced modulo $q$.
///
/// # Errors
/// [`Error::Shape`] unless `z_share` has $\mu - 1$ polynomials of $d$ coefficients and `r_share` $n'$
/// polynomials of $2d$ coefficients; [`Error::Parameter`] for a hypercube larger than `usize`.
pub fn w_table<const K: usize>(
    ctx: &Context<K>,
    z_share: &[Poly],
    r_share: &[Poly],
) -> Result<Vec<u64>> {
    let l = &ctx.layout;
    let q = ctx.q();
    let two_d = 2 * l.d;
    check_len("<z'>_p", l.len_z, z_share.len())?;
    check_len("<r>_p", l.n_prime, r_share.len())?;
    for p in z_share {
        check_len("<z'>_p (coefficients)", l.d, p.0.len())?;
    }
    for p in r_share {
        check_len("<r>_p (coefficients)", two_d, p.0.len())?;
    }
    let mut table = vec![0u64; table_size(l)?];
    if l.len_z > l.mu || (l.mu + l.n_prime) * two_d > table.len() {
        return Err(Error::Shape {
            what: "rows of the table (z' then r from row mu)",
            expected: table.len() / two_d,
            actual: l.mu + l.n_prime,
        });
    }
    // The rows of z' and of r, each written in parallel (disjoint rows).
    let (z_rows, rest) = table.split_at_mut(l.mu * two_d);
    z_rows
        .par_chunks_mut(two_d)
        .zip(z_share.par_iter())
        .for_each(|(row, p)| {
            for (t, c) in row.iter_mut().zip(&p.0) {
                *t = c % q;
            }
        });
    rest.par_chunks_mut(two_d)
        .zip(r_share.par_iter())
        .for_each(|(row, p)| {
            for (t, c) in row.iter_mut().zip(&p.0) {
                *t = c % q;
            }
        });
    Ok(table)
}

/// The table `base` of [`w_table`] over $\mathbb F\_{q^K}$, with the pad's entry `pad` (its index
/// and value) set when the pad is a field element: the hypercube values of
/// $\langle \widetilde w \rangle\_p$, the pad included.
pub(crate) fn lift_table<const K: usize>(
    field: &Fqk<K>,
    base: &[u64],
    pad: Option<(usize, Ext<K>)>,
) -> Vec<Ext<K>> {
    let mut table: Vec<Ext<K>> = base.iter().map(|x| field.from_base(*x)).collect();
    if let Some((at, value)) = pad {
        table[at] = value;
    }
    table
}

/// The table index of the pad's entry $(\mu, 1)$ (`5_Building_Blocks.tex`:256): row $\mu - 1$,
/// coefficient 0.
pub fn pad_index(layout: &Layout) -> usize {
    layout.pad * 2 * layout.d
}

/// Writes the residues of [`PadRule::Residues`](hachi_co_params::PadRule) into the pad's row, from
/// its first entry.
///
/// # Errors
/// [`Error::Shape`] unless there are $2k$ residues.
pub fn put_pad_residues(layout: &Layout, table: &mut [u64], residues: &[u64]) -> Result<()> {
    check_len("pad residues (2k)", 2 * layout.k, residues.len())?;
    let at = pad_index(layout);
    table[at..at + residues.len()].copy_from_slice(residues);
    Ok(())
}

/// The rows $u$ of the draft's $\sum\_{u = \mu - 1}^{\mu + n'} \chi\_{\mathsf{bit}(u)}(U)$
/// (`5_Building_Blocks.tex`:302): the 1-indexed rows $\mu - 1, \dots, \mu + n'$ are the 0-indexed
/// $\mu - 2, \dots, \mu + n' - 1$, i.e. $\langle u \rangle$, the pad's row and the $n'$ quotient
/// rows (the oracle's `ind_rows`; `ref/README.md`, choice 8).
pub fn ind_rows(layout: &Layout) -> Range<usize> {
    layout.u..layout.mu + layout.n_prime
}

/// The $0/1$ table over $U$ ($2^{\ell\_U}$ entries) that is 1 on the rows `rows`: $e\_p(U) =
/// \sum\_{u \in S\_p} \chi\_{\mathsf{bit}(u)}(U)$ for the ranges [`Layout::s_ranges`]`(p)`, or
/// $\mathrm{ind}(U)$ for [`ind_rows`] (`5_Building_Blocks.tex`:294, 302). The oracle's
/// `row_indicator` broadcasts it over $L$; both factors are constant in $L$, so the provers keep
/// them as tables over $U$.
///
/// # Errors
/// [`Error::Shape`] if a range leaves $\[0, 2^{\ell\_U})$.
pub fn row_indicator(layout: &Layout, rows: &[Range<usize>]) -> Result<Vec<u64>> {
    let size = pow2(layout.ell_u, "the U part of the hypercube")?;
    let mut table = vec![0u64; size];
    for r in rows {
        if r.start > r.end || r.end > size {
            return Err(Error::Shape {
                what: "row range (within the U part of the hypercube)",
                expected: size,
                actual: r.end,
            });
        }
        for x in &mut table[r.clone()] {
            *x = 1;
        }
    }
    Ok(table)
}

/// The range polynomial $\prod\_{c \in \lbrace -b+1, \dots, b-1 \rbrace} (W - c)$ of
/// $F\_{0,p}$ (`5_Building_Blocks.tex`:301; the oracle's `range_product`), evaluated as
/// $W \prod\_{c=1}^{b-1} (W^2 - c^2)$: $b$ multiplications instead of $2b - 2$ (the empty product
/// for $b = 0$ is $1$).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RangeProduct {
    b: u64,
    /// $c^2 \bmod q$ for $c = 1, \dots, b - 1$.
    squares: Vec<u64>,
}

impl RangeProduct {
    /// The range product for the base $b$.
    pub fn new(zq: &Zq, b: u64) -> Self {
        Self {
            b,
            squares: (1..b.max(1)).map(|c| zq.mul(c, c)).collect(),
        }
    }

    /// $c^2 \bmod q$ for $c = 1, \dots, b - 1$.
    pub(crate) fn squares(&self) -> &[u64] {
        &self.squares
    }

    /// $\prod\_{c = -b+1}^{b-1} (w - c)$.
    pub fn eval<const K: usize>(&self, field: &Fqk<K>, w: &Ext<K>) -> Ext<K> {
        if self.b == 0 {
            return field.one();
        }
        let mut acc = *w;
        if let Some((first, rest)) = self.squares.split_first() {
            let w2 = field.square(w);
            let mut prod = sub_base(field, &w2, *first);
            for c2 in rest {
                prod = field.mul(&prod, &sub_base(field, &w2, *c2));
            }
            acc = field.mul(&prod, w);
        }
        acc
    }
}

/// $\prod\_{c = -b+1}^{b-1} (w - c)$ ([`RangeProduct`] for one evaluation).
pub fn range_product<const K: usize>(field: &Fqk<K>, w: &Ext<K>, b: u64) -> Ext<K> {
    RangeProduct::new(field.zq(), b).eval(field, w)
}

/// $x - c$ for a base-field constant $c$ (only coordinate $a\_0$ changes, Hachi Eq. (7)).
pub(crate) fn sub_base<const K: usize>(field: &Fqk<K>, x: &Ext<K>, c: u64) -> Ext<K> {
    let mut out = field.reduce(x);
    out.0[0] = field.zq().sub(out.0[0], c);
    out
}

/// $(x^0, x^1, \dots, x^{\mathrm{count} - 1})$.
pub fn powers<const K: usize>(field: &Fqk<K>, x: &Ext<K>, count: usize) -> Vec<Ext<K>> {
    let mut out = Vec::with_capacity(count);
    let mut current = field.one();
    for _ in 0..count {
        out.push(current);
        current = field.mul(&current, x);
    }
    out
}

/// $\sum\_j c\_j \alpha^j$ for a polynomial with base-field coefficients $c\_j$, given the powers
/// $\alpha^j$ (at least as many as coefficients).
pub(crate) fn eval_at<const K: usize>(field: &Fqk<K>, coeffs: &[u64], powers: &[Ext<K>]) -> Ext<K> {
    let zq = field.zq();
    let mut acc = field.zero();
    for (c, p) in coeffs.iter().zip(powers) {
        let c = zq.reduce(*c);
        if c != 0 {
            acc = field.add(&acc, &field.scale(p, c));
        }
    }
    acc
}

/// Checks the $d \times d$ matrix of $E$ and the length of the $\widetilde{eq}(\tau\_1, \cdot)$
/// table ($2^{\lceil \log \nu \rceil}$).
fn check_e_and_eq1(layout: &Layout, e_matrix: &[Vec<u64>], eq1_len: usize) -> Result<()> {
    check_len("E matrix (rows)", layout.d, e_matrix.len())?;
    for row in e_matrix {
        check_len("E matrix (columns)", layout.d, row.len())?;
    }
    check_len(
        "eq(tau_1) table (2^ceil(log nu) entries)",
        pow2(layout.log_nu, "the eq(tau_1) table")?,
        eq1_len,
    )
}

/// $\widetilde{eq}(\tau\_1, \cdot)$ as two half tables: entry $i$ of
/// [`eq_table`](crate::eq_table)$(\tau\_1)$ is $\mathrm{hi}\[i \gg s\] \cdot \mathrm{lo}\[i \bmod 2^s\]$, with
/// $\mathrm{hi}$ the table of the first $\lvert \tau\_1 \rvert - s$ coordinates and $\mathrm{lo}$
/// that of the last $s = \lfloor \lvert \tau\_1 \rvert / 2 \rfloor$ (the most significant bit
/// goes with the first coordinate, as in [`eq_table`](crate::eq_table)). The product is exact, so each entry is that of
/// the whole table, which [`mstar_table_rows`] then never holds ($2^{\lceil \log \nu \rceil}$
/// values, 2 GiB at secure-l24-n2-a8).
///
/// With Montgomery's form ([`Montgomery`], where the modulus admits it) `hi` is held in it, so
/// that the product $\phi(\mathrm{hi}) \cdot \mathrm{lo}$ is the canonical entry, in about half the
/// time of the canonical product.
#[derive(Clone, Debug)]
pub struct EqSplit<const K: usize> {
    /// In the working representation of `mont`.
    hi: Vec<Ext<K>>,
    lo: Vec<Ext<K>>,
    s: usize,
    mont: Option<Montgomery<K>>,
}

impl<const K: usize> EqSplit<K> {
    /// The two halves of $\widetilde{eq}(\boldsymbol\tau, \cdot)$.
    ///
    /// # Errors
    /// As [`eq_table`](crate::eq_table).
    pub fn new(field: &Fqk<K>, tau: &[Ext<K>]) -> Result<Self> {
        let s = tau.len() / 2;
        let (hi, lo) = tau.split_at(tau.len() - s);
        let mont = Montgomery::<K>::new(field.modulus());
        let mut hi = eq_table(field, hi)?;
        if let Some(m) = &mont {
            hi.iter_mut().for_each(|v| *v = m.enter(v));
        }
        Ok(Self {
            hi,
            lo: eq_table(field, lo)?,
            s,
            mont,
        })
    }

    /// The number of entries, $2^{\lvert \boldsymbol\tau \rvert}$.
    pub fn len(&self) -> usize {
        self.hi.len() << self.s
    }

    /// Whether there are no entries (never: the empty point has the table $(1)$).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Entry $i$ of the whole table, for $i$ below [`EqSplit::len`].
    #[inline]
    pub fn at(&self, field: &Fqk<K>, i: usize) -> Ext<K> {
        let (hi, lo) = (&self.hi[i >> self.s], &self.lo[i & ((1 << self.s) - 1)]);
        match &self.mont {
            Some(m) => m.mul(hi, lo),
            None => field.mul(hi, lo),
        }
    }
}

/// The weights $c\_u = \sum\_{i' \lt n'} \widetilde{eq}(\tau\_1, \mathsf{bit}(i'))\, M\_{i',u}(\alpha)$
/// of the columns $u \lt \mu - 1$ of $\mathbf M$ (fixture convention `M_tilde`), from the powers
/// $\alpha^0, \dots, \alpha^{d-1}$. Zero coefficients of $\mathbf M$ are skipped.
pub(crate) fn column_weights<const K: usize>(
    field: &Fqk<K>,
    m: &[Vec<Poly>],
    eq1: &[Ext<K>],
    apow: &[Ext<K>],
) -> Vec<Ext<K>> {
    let cols = m.first().map_or(0, Vec::len);
    let mut c = vec![field.zero(); cols];
    for (row, e) in m.iter().zip(eq1) {
        add_row_weights(field, &mut c, row, e, apow);
    }
    c
}

/// Adds row $i'$ of $\mathbf M$, of weight $e = \widetilde{eq}(\tau\_1, \mathsf{bit}(i'))$, to the
/// column weights of [`column_weights`]: $c\_u \mathrel{+}= \sum\_j M\_{i',u,j} \beta\_j$ with
/// $\beta\_j = e \alpha^j$. The columns run in parallel, each over its coefficients in order.
fn add_row_weights<const K: usize>(
    field: &Fqk<K>,
    c: &mut [Ext<K>],
    row: &[Poly],
    e: &Ext<K>,
    apow: &[Ext<K>],
) {
    let zq = field.zq();
    let beta: Vec<Ext<K>> = apow.iter().map(|p| field.mul(e, p)).collect();
    c.par_iter_mut().zip(row.par_iter()).for_each(|(cu, poly)| {
        for (coef, b) in poly.0.iter().zip(&beta) {
            let coef = zq.reduce(*coef);
            if coef != 0 {
                *cu = field.add(cu, &field.scale(b, coef));
            }
        }
    });
}

/// The hypercube values of
/// $$M^\ast(U, L) = \sum\_{i' \in \[\nu\]} \widetilde{eq}(\tau\_1, \mathsf{bit}(i'))\,
/// \widetilde M\_\alpha(\mathsf{bit}(i'), U, L)$$
/// (`5_Building_Blocks.tex`:286-292, 303; the oracle's `mstar_table`), $2^{\ell''}$ elements of
/// $\mathbb F\_{q^k}$, by the fixture convention `M_tilde`: at $(u\_0, l\_0)$,
/// - $c\_u \alpha^{l\_0}$ for $u\_0 \lt \mu - 1$ and every $l\_0 \lt 2d$, with
///   $c\_u = \sum\_{i\_0 \lt n'} \mathrm{eq}\_1\[i\_0\] M\_{i\_0, u\_0}(\alpha)$ (first term, $u \le \mu - 1$;
///   `ref/README.md`, choice 10);
/// - $-\mathrm{eq}\_1\[u\_0 - \mu\] (\alpha^d + 1) \alpha^{l\_0}$ for $\mu \le u\_0 \lt \mu + n'$ (first
///   term, $u = i' + \mu$);
/// - plus $\sum\_{j\_0 \lt d} \mathrm{eq}\_1\[n' + j\_0\] E\[j\_0\]\[l\_0\]$ at $u\_0 = \mu - 2$,
///   $l\_0 \lt d$ (second term, the $E$ rows; choice 9);
/// - plus $\mathrm{eq}\_1\[n' + d + u\_0 + (l\_0 - d)(\mu + n')\]$ for $u\_0 \lt \mu + n'$,
///   $l\_0 \ge d$ (third term, the high coefficients);
/// - and $0$ at the pad's entries $(\mu - 1, l\_0)$, $l\_0 \lt d$, and on the padding rows
///   $u\_0 \ge \mu + n'$,
///
/// where $\mathrm{eq}\_1$ is [`eq_table`](crate::eq_table)$(\tau\_1)$ and $E\[j\]\[l\]$ is
/// coefficient $j$ of $E(X^l)$ ([`hachi_co_math::Subfield::e_matrix`]). The cost is
/// $O(\text{nonzero coefficients of } \mathbf M + 2^{\ell''} + d^2)$ operations; the verifier
/// evaluates the multilinear extension at one point without the table
/// ([`mstar_eval_factorized`](crate::mstar_eval_factorized)).
///
/// # Errors
/// [`Error::Shape`] unless $\mathbf M$ is $n' \times (\mu - 1)$ with $d$ coefficients per entry,
/// $E$ is $d \times d$ and $\mathrm{eq}\_1$ has $2^{\lceil \log \nu \rceil}$ entries.
pub fn mstar_table<const K: usize>(
    ctx: &Context<K>,
    m: &[Vec<Poly>],
    e_matrix: &[Vec<u64>],
    eq1: &[Ext<K>],
    alpha: &Ext<K>,
) -> Result<Vec<Ext<K>>> {
    let l = &ctx.layout;
    check_matrix(m, l.n_prime, l.len_z, l.d)?;
    check_e_and_eq1(l, e_matrix, eq1.len())?;
    mstar_table_with(
        ctx,
        |i| Ok(Cow::Borrowed(&m[i][..])),
        e_matrix,
        |i| eq1[i],
        alpha,
    )
}

/// [`mstar_table`] with the rows of $\mathbf M$ read one at a time from `rows` (row $i$ of the
/// $n'$, each read once, in order) and $\mathrm{eq}\_1$ as an [`EqSplit`], so that neither
/// $\mathbf M$ nor the whole $\mathrm{eq}\_1$ is held: the same table, as the column weights add
/// up row by row in the same order and every entry of $\mathrm{eq}\_1$ is the same.
///
/// # Errors
/// As [`mstar_table`], for a row of the wrong shape when it is read, and those of `rows`.
pub fn mstar_table_rows<'m, const K: usize>(
    ctx: &Context<K>,
    rows: impl FnMut(usize) -> Result<Cow<'m, [Poly]>>,
    e_matrix: &[Vec<u64>],
    eq1: &EqSplit<K>,
    alpha: &Ext<K>,
) -> Result<Vec<Ext<K>>> {
    check_e_and_eq1(&ctx.layout, e_matrix, eq1.len())?;
    let field = ctx.field();
    mstar_table_with(ctx, rows, e_matrix, |i| eq1.at(field, i), alpha)
}

/// The table of [`mstar_table`] from the rows of $\mathbf M$ and the entries of
/// $\mathrm{eq}\_1$ (checked by the callers, but the rows' shapes, checked as they are read).
fn mstar_table_with<'m, const K: usize>(
    ctx: &Context<K>,
    mut rows: impl FnMut(usize) -> Result<Cow<'m, [Poly]>>,
    e_matrix: &[Vec<u64>],
    eq1: impl Fn(usize) -> Ext<K> + Sync,
    alpha: &Ext<K>,
) -> Result<Vec<Ext<K>>> {
    let l = &ctx.layout;
    let field = ctx.field();
    let (d, mu, np) = (l.d, l.mu, l.n_prime);
    let two_d = 2 * d;
    let apow = powers(field, alpha, two_d);
    let mut c = vec![field.zero(); l.len_z];
    for i in 0..np {
        let row = rows(i)?;
        check_row(&row, l.len_z, d)?;
        add_row_weights(field, &mut c, &row, &eq1(i), &apow[..d]);
    }
    let mut table = vec![field.zero(); table_size(l)?];
    // First term, u0 < mu - 1: c_u alpha^{l0} for all l0 < 2d (rows in parallel).
    table
        .par_chunks_mut(two_d)
        .zip(c.par_iter())
        .for_each(|(row, cu)| {
            for (t, p) in row.iter_mut().zip(&apow) {
                *t = field.mul(cu, p);
            }
        });
    // First term, u0 = mu + i0: -eq1[i0] (alpha^d + 1) alpha^{l0}.
    let ad1 = field.add(&apow[d], &field.one());
    for i0 in 0..np {
        let coef = field.neg(&field.mul(&eq1(i0), &ad1));
        for (l0, p) in apow.iter().enumerate() {
            table[(mu + i0) * two_d + l0] = field.mul(&coef, p);
        }
    }
    // Second term: rows n' + 1 .. n' + d (1-indexed) at u0 = mu - 2, l0 < d.
    let zq = field.zq();
    for l0 in 0..d {
        let mut acc = field.zero();
        for (j0, row) in e_matrix.iter().enumerate() {
            let coef = zq.reduce(row[l0]);
            if coef != 0 {
                acc = field.add(&acc, &field.scale(&eq1(np + j0), coef));
            }
        }
        let x = l.u * two_d + l0;
        table[x] = field.add(&table[x], &acc);
    }
    // Third term: u0 < mu + n', l0 >= d, i' - 1 = n' + d + u0 + (l0 - d)(mu + n') (rows in
    // parallel; the second term is on the columns l0 < d).
    table
        .par_chunks_mut(two_d)
        .take(mu + np)
        .enumerate()
        .for_each(|(u0, row)| {
            for (l0, t) in row.iter_mut().enumerate().skip(d) {
                let i = np + d + u0 + (l0 - d) * (mu + np);
                *t = field.add(t, &eq1(i));
            }
        });
    Ok(table)
}

/// The parts of [`mstar_table`] from which each entry is formed when it is read
/// ([`MStarParts::at`]), so that the table of $2^{\ell''}$ entries is not held before the first
/// bind (4 GiB at secure-l24-n2-a8): the row factors of the first term ($c\_u$ for
/// $u\_0 \lt \mu - 1$, $-\mathrm{eq}\_1\[u\_0 - \mu\] (\alpha^d + 1)$ for $\mu \le u\_0 \lt \mu + n'$, zero
/// otherwise), the powers $\alpha^{l\_0}$, the second term's row and $\mathrm{eq}\_1$ as an
/// [`EqSplit`]. Each entry is formed by the field operations that [`mstar_table_rows`] uses for
/// it, so it is the same element.
#[derive(Clone, Debug)]
pub struct MStarParts<const K: usize> {
    d: usize,
    mu: usize,
    np: usize,
    /// The row of the second term ($\mathbf E$).
    u: usize,
    /// $\log 2d$: the point $(u\_0, l\_0)$ has index $u\_0 \cdot 2d + l\_0$.
    shift: usize,
    size: usize,
    apow: Vec<Ext<K>>,
    /// The first term's factor of each row $u\_0 \lt \mu + n'$, in the working representation of
    /// `mont`: $\phi(c) \alpha^{l\_0}$ is the canonical product.
    rows: Vec<Ext<K>>,
    /// Montgomery's form, where the modulus admits it.
    mont: Option<Montgomery<K>>,
    /// $\sum\_{j\_0} \mathrm{eq}\_1\[n' + j\_0\] E\[j\_0\]\[l\_0\]$ for $l\_0 \lt d$.
    second: Vec<Ext<K>>,
    eq1: EqSplit<K>,
}

impl<const K: usize> MStarParts<K> {
    /// The parts of [`mstar_table_rows`] with the same inputs: the rows of $\mathbf M$ are read
    /// once, in order, for the column weights.
    ///
    /// # Errors
    /// As [`mstar_table_rows`]; [`Error::Parameter`] if $2d$ is not a power of two.
    pub fn from_rows<'m>(
        ctx: &Context<K>,
        mut rows: impl FnMut(usize) -> Result<Cow<'m, [Poly]>>,
        e_matrix: &[Vec<u64>],
        eq1: EqSplit<K>,
        alpha: &Ext<K>,
    ) -> Result<Self> {
        let l = &ctx.layout;
        check_e_and_eq1(l, e_matrix, eq1.len())?;
        let field = ctx.field();
        let (d, mu, np) = (l.d, l.mu, l.n_prime);
        let two_d = 2 * d;
        if !two_d.is_power_of_two() {
            return Err(Error::Parameter(format!(
                "2d = {two_d} is not a power of two"
            )));
        }
        let size = table_size(l)?;
        if (mu + np) * two_d > size || l.len_z > mu + np || (l.u + 1) * two_d > size {
            return Err(Error::Shape {
                what: "rows of M* (mu + n' rows of 2d points)",
                expected: size / two_d,
                actual: mu + np,
            });
        }
        let apow = powers(field, alpha, two_d);
        let mut c = vec![field.zero(); l.len_z];
        for i in 0..np {
            let row = rows(i)?;
            check_row(&row, l.len_z, d)?;
            add_row_weights(field, &mut c, &row, &eq1.at(field, i), &apow[..d]);
            // A row built for this call is dropped in parallel (its ring elements are many).
            if let Cow::Owned(row) = row {
                row.into_par_iter().for_each(drop);
            }
        }
        let mut factors = vec![field.zero(); mu + np];
        factors[..l.len_z].copy_from_slice(&c);
        let ad1 = field.add(&apow[d], &field.one());
        for i0 in 0..np {
            factors[mu + i0] = field.neg(&field.mul(&eq1.at(field, i0), &ad1));
        }
        let zq = field.zq();
        let second = (0..d)
            .map(|l0| {
                let mut acc = field.zero();
                for (j0, row) in e_matrix.iter().enumerate() {
                    let coef = zq.reduce(row[l0]);
                    if coef != 0 {
                        acc = field.add(&acc, &field.scale(&eq1.at(field, np + j0), coef));
                    }
                }
                acc
            })
            .collect();
        let mont = Montgomery::<K>::new(field.modulus());
        if let Some(m) = &mont {
            factors.iter_mut().for_each(|v| *v = m.enter(v));
        }
        Ok(Self {
            d,
            mu,
            np,
            u: l.u,
            shift: two_d.trailing_zeros() as usize,
            size,
            apow,
            rows: factors,
            mont,
            second,
            eq1,
        })
    }

    /// The number of entries, $2^{\ell''}$.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Whether there are no entries (never: the hypercube has at least $2d$ points).
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Entry $x$ of [`mstar_table`], for $x$ below [`MStarParts::len`].
    #[inline]
    pub fn at(&self, field: &Fqk<K>, x: usize) -> Ext<K> {
        let (u0, l0) = (x >> self.shift, x & ((1 << self.shift) - 1));
        let mut v = match (self.rows.get(u0), &self.mont) {
            (Some(factor), Some(m)) => m.mul(factor, &self.apow[l0]),
            (Some(factor), None) => field.mul(factor, &self.apow[l0]),
            (None, _) => field.zero(),
        };
        if u0 == self.u && l0 < self.d {
            v = field.add(&v, &self.second[l0]);
        }
        if u0 < self.mu + self.np && l0 >= self.d {
            let i = self.np + self.d + u0 + (l0 - self.d) * (self.mu + self.np);
            v = field.add(&v, &self.eq1.at(field, i));
        }
        v
    }

    /// The whole table, [`MStarParts::at`] at every point.
    pub fn table(&self, field: &Fqk<K>) -> Vec<Ext<K>> {
        (0..self.size)
            .into_par_iter()
            .map(|x| self.at(field, x))
            .collect()
    }
}

/// The claimed sum
/// $$a = \sum\_{i' \in \[n'\]} \widetilde{eq}(\tau\_1, \mathsf{bit}(i'))\, y\_{i'}(\alpha) +
/// \sum\_{i' = n'+1}^{n'+d} \widetilde{eq}(\tau\_1, \mathsf{bit}(i'))\, y\_{i' - n' - 1}$$
/// (`5_Building_Blocks.tex`:295-296; the oracle's `claimed_sum`), where $\mathbf y \in
/// \mathcal R\_q^{n'}$ is the right-hand side of $\mathbf M \mathbf z' = \mathbf y$ and
/// $y = \sum\_{j \lt d} y\_j X^j$ is the claimed evaluation embedded in $\mathcal R\_q^H$
/// (`y_ring`; `ref/README.md`, choice 12).
///
/// # Errors
/// [`Error::Shape`] unless $\mathbf y$ has $n'$ polynomials and every polynomial (and `y_ring`)
/// $d$ coefficients, and $\mathrm{eq}\_1$ has $2^{\lceil \log \nu \rceil}$ entries.
pub fn claimed_sum<const K: usize>(
    ctx: &Context<K>,
    eq1: &[Ext<K>],
    y: &[Poly],
    y_ring: &Poly,
    alpha: &Ext<K>,
) -> Result<Ext<K>> {
    let l = &ctx.layout;
    let field = ctx.field();
    check_len("y (one entry per row of M)", l.n_prime, y.len())?;
    check_len(
        "eq(tau_1) table (2^ceil(log nu) entries)",
        pow2(l.log_nu, "the eq(tau_1) table")?,
        eq1.len(),
    )?;
    check_len("y as a ring element", l.d, y_ring.0.len())?;
    let apow = powers(field, alpha, l.d);
    let mut a = field.zero();
    for (yi, e) in y.iter().zip(eq1) {
        check_len("y (coefficients)", l.d, yi.0.len())?;
        a = field.add(&a, &field.mul(e, &eval_at(field, &yi.0, &apow)));
    }
    let zq = field.zq();
    for (yj, e) in y_ring.0.iter().zip(&eq1[l.n_prime..]) {
        a = field.add(&a, &field.scale(e, zq.reduce(*yj)));
    }
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// splitmix64.
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    #[test]
    fn the_split_eq_table_is_the_eq_table() {
        let q = 134217757u64; // toy-n4
        let f = Fqk::<4>::new(q).unwrap();
        let mut state = 51u64;
        for len in 0..=9usize {
            let tau: Vec<Ext<4>> = (0..len)
                .map(|_| Ext(core::array::from_fn(|_| next(&mut state) % q)))
                .collect();
            let whole = eq_table(&f, &tau).unwrap();
            let split = EqSplit::new(&f, &tau).unwrap();
            assert_eq!(split.len(), whole.len(), "{len} coordinates");
            for (i, want) in whole.iter().enumerate() {
                assert_eq!(split.at(&f, i), *want, "{len} coordinates, entry {i}");
            }
        }
    }

    #[test]
    fn m_star_and_the_quotient_by_rows_are_those_of_the_whole_matrix() {
        // Toy contexts of one and two parties, M from random public values, a random tau_1, E
        // and alpha: the table from the whole M and the whole eq table against the table from
        // rows (borrowed, and built by `MRows`) and the split table; the quotient share likewise.
        use hachi_co_core::{LinearShare, MRows, build_m};
        use hachi_co_params::{derive, presets::toy_choices};
        for (n, seed) in [(1u64, 52u64), (2, 53)] {
            let (params, _) = derive(&toy_choices(n)).unwrap();
            let ctx = Context::<4>::setup(params, &[seed as u8; 32]).unwrap();
            let (l, f) = (&ctx.layout, ctx.field());
            let (q, d) = (ctx.q(), l.d);
            let mut state = seed;
            let mut ext = || Ext::<4>(core::array::from_fn(|_| next(&mut state) % q));
            let b_ring: Vec<Poly> = (0..l.n_blocks).map(|_| ctx.sub.embed(&ext())).collect();
            let a_ring: Vec<Poly> = (0..l.block_len).map(|_| ctx.sub.embed(&ext())).collect();
            let tau1: Vec<Ext<4>> = (0..l.log_nu).map(|_| ext()).collect();
            let alpha = ext();
            let mut state = seed + 100;
            let c: Vec<Vec<i64>> = (0..l.n_blocks)
                .map(|_| (0..d).map(|_| (next(&mut state) % 3) as i64 - 1).collect())
                .collect();
            let e_matrix: Vec<Vec<u64>> = (0..d)
                .map(|_| (0..d).map(|_| next(&mut state) % q).collect())
                .collect();
            let m = build_m(&ctx, &b_ring, &a_ring, &c, 0).unwrap();
            let rows = MRows::new(&ctx, &b_ring, &a_ring, &c, 0).unwrap();
            let eq1 = eq_table(f, &tau1).unwrap();
            let split = EqSplit::new(f, &tau1).unwrap();
            let want = mstar_table(&ctx, &m, &e_matrix, &eq1, &alpha).unwrap();
            let borrowed = |i: usize| Ok(Cow::Borrowed(&m[i][..]));
            assert_eq!(
                mstar_table_rows(&ctx, borrowed, &e_matrix, &split, &alpha).unwrap(),
                want,
                "n = {n}, borrowed rows"
            );
            let built = |i: usize| Ok(Cow::Owned(rows.row(i)?));
            assert_eq!(
                mstar_table_rows(&ctx, built, &e_matrix, &split, &alpha).unwrap(),
                want,
                "n = {n}, rows of MRows"
            );
            // M* from its parts, each entry formed when read: the same table.
            let built = |i: usize| Ok(Cow::Owned(rows.row(i)?));
            let parts =
                MStarParts::from_rows(&ctx, built, &e_matrix, split.clone(), &alpha).unwrap();
            assert_eq!(parts.len(), want.len(), "n = {n}");
            assert_eq!(parts.table(f), want, "n = {n}, parts");
            for x in [0, 1, want.len() / 2, want.len() - 1] {
                assert_eq!(parts.at(f, x), want[x], "n = {n}, entry {x}");
            }
            let short = &e_matrix[1..];
            assert!(MStarParts::from_rows(&ctx, borrowed, short, split.clone(), &alpha).is_err());
            let share = LinearShare {
                z_prime: (0..l.len_z)
                    .map(|_| Poly((0..d).map(|_| next(&mut state) % q).collect()))
                    .collect(),
                y: (0..l.n_prime)
                    .map(|_| Poly((0..d).map(|_| next(&mut state) % q).collect()))
                    .collect(),
            };
            let quotient = crate::quotient_share(&ctx, &m, &share).unwrap();
            let built = |i: usize| Ok(Cow::Owned(rows.row(i)?));
            assert_eq!(
                crate::quotient_share_rows(&ctx, built, &share).unwrap(),
                quotient,
                "n = {n}"
            );
        }
    }

    #[test]
    fn range_product_matches_the_literal_product() {
        let q = 134217757u64; // toy-n4
        let f = Fqk::<4>::new(q).unwrap();
        for b in 0..=6u64 {
            let rp = RangeProduct::new(f.zq(), b);
            for w in [
                f.zero(),
                f.one(),
                f.from_base(q - 1),
                f.from_base(b),
                f.from_base(q - b),
                Ext([5, 7, 11, 13]),
            ] {
                let mut lit = f.one();
                for c in -(b as i64) + 1..=(b as i64) - 1 {
                    lit = f.mul(&lit, &sub_base(&f, &w, f.zq().from_i64(c)));
                }
                assert_eq!(rp.eval(&f, &w), lit, "b = {b}");
            }
            // The roots are exactly the integers -b+1..b-1 (the failure boundary is +-b).
            for c in -(b as i64) + 1..=(b as i64) - 1 {
                assert!(f.is_zero(&rp.eval(&f, &f.from_base(f.zq().from_i64(c)))));
            }
            if b > 0 {
                for c in [b as i64, -(b as i64)] {
                    assert!(!f.is_zero(&rp.eval(&f, &f.from_base(f.zq().from_i64(c)))));
                }
            }
        }
    }

    #[test]
    fn powers_and_evaluation() {
        let q = 8388637u64;
        let f = Fqk::<4>::new(q).unwrap();
        let x = Ext([3, 1, 4, 1]);
        let p = powers(&f, &x, 5);
        assert_eq!(p[0], f.one());
        assert_eq!(p[4], f.pow(&x, &[4]));
        // 2 + 3x + 0x^2 + (q-1)x^3
        let v = eval_at(&f, &[2, 3, 0, q - 1], &p);
        let want = f.sub(&f.add(&f.from_base(2), &f.scale(&x, 3)), &p[3]);
        assert_eq!(v, want);
    }
}
