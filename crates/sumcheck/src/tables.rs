//! The hypercube tables and public values of the sumcheck (`5_Building_Blocks.tex`:205-275;
//! oracle `ref/hachico_ref/sumcheck.py`): the witness table $\langle \widetilde w \rangle\_p$, the
//! row indicators $e\_p$ and $\mathrm{ind}$, the range product, the table of
//! $\widetilde M^\ast$, and the claimed sum $a$.
//!
//! **Hypercube** (`ref/README.md`, "Hypercube"; fixture convention `hypercube`):
//! $V = (U, L)$ with $\lvert U \rvert = \ell\_U = \lceil \log(\mu + n') \rceil$ and
//! $\lvert L \rvert = \ell\_L = \log 2d$; the point $(u\_0, l\_0)$ has index $u\_0 \cdot 2d + l\_0$,
//! $V\_1$ the most significant bit. The draft's 1-indexed $\chi\_{\mathsf{bit}(u) \Vert
//! \mathsf{bit}(\ell)}$ is the entry $(u - 1) \cdot 2d + (\ell - 1)$.

use crate::error::{Error, Result, check_len, pow2};
use crate::quotient::check_matrix;
use core::ops::Range;
use hachi_co_core::{Context, Layout, Poly};
use hachi_co_math::{Ext, Fqk, Zq};

/// $2^{\ell''}$, the number of points of the hypercube.
///
/// # Errors
/// [`Error::Parameter`] if it does not fit in `usize`.
pub fn table_size(layout: &Layout) -> Result<usize> {
    pow2(layout.ell_pp, "the hypercube")
}

/// The hypercube values of party $p$'s witness polynomial $\langle \widetilde w \rangle\_p(U, L)$
/// (`5_Building_Blocks.tex`:225-229; the oracle's `w_table`; fixture convention `hypercube`), over
/// $\mathbb Z\_q$, $2^{\ell''}$ entries: the entry $(u\_0, l\_0)$ is coefficient $l\_0$ of
/// $\langle z'\_{u\_0} \rangle\_p$ for $u\_0 \lt \mu$, $l\_0 \lt d$; coefficient $l\_0$ of
/// $\langle r\_{u\_0 - \mu} \rangle\_p$ for $\mu \le u\_0 \lt \mu + n'$, $l\_0 \lt 2d$; and $0$
/// otherwise. Residues are reduced modulo $q$.
///
/// # Errors
/// [`Error::Shape`] unless `z_share` has $\mu$ polynomials of $d$ coefficients and `r_share` $n'$
/// polynomials of $2d$ coefficients; [`Error::Parameter`] for a hypercube larger than `usize`.
pub fn w_table<const K: usize>(
    ctx: &Context<K>,
    z_share: &[Poly],
    r_share: &[Poly],
) -> Result<Vec<u64>> {
    let l = &ctx.layout;
    let q = ctx.q();
    let two_d = 2 * l.d;
    check_len("<z'>_p", l.mu, z_share.len())?;
    check_len("<r>_p", l.n_prime, r_share.len())?;
    let mut table = vec![0u64; table_size(l)?];
    for (u0, p) in z_share.iter().enumerate() {
        check_len("<z'>_p (coefficients)", l.d, p.0.len())?;
        for (l0, c) in p.0.iter().enumerate() {
            table[u0 * two_d + l0] = c % q;
        }
    }
    for (i0, p) in r_share.iter().enumerate() {
        check_len("<r>_p (coefficients)", two_d, p.0.len())?;
        let base = (l.mu + i0) * two_d;
        for (l0, c) in p.0.iter().enumerate() {
            table[base + l0] = c % q;
        }
    }
    Ok(table)
}

/// The rows $u$ of the draft's $\sum\_{u = \mu}^{\mu + n'} \chi\_{\mathsf{bit}(u)}(U)$
/// (`5_Building_Blocks.tex`:270): the 1-indexed rows $\mu, \dots, \mu + n'$ are the 0-indexed
/// $\mu - 1, \dots, \mu + n' - 1$, i.e. $\langle u \rangle$ and the $n'$ quotient rows (the
/// oracle's `ind_rows`; `ref/README.md`, choice 8).
pub fn ind_rows(layout: &Layout) -> Range<usize> {
    layout.mu - 1..layout.mu + layout.n_prime
}

/// The $0/1$ table over $U$ ($2^{\ell\_U}$ entries) that is 1 on the rows `rows`: $e\_p(U) =
/// \sum\_{u \in S\_p} \chi\_{\mathsf{bit}(u)}(U)$ for the ranges [`Layout::s_ranges`]`(p)`, or
/// $\mathrm{ind}(U)$ for [`ind_rows`] (`5_Building_Blocks.tex`:262, 270). The oracle's
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
/// $F\_{0,p}$ (`5_Building_Blocks.tex`:269; the oracle's `range_product`), evaluated as
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
fn check_e_and_eq1<const K: usize>(
    layout: &Layout,
    e_matrix: &[Vec<u64>],
    eq1: &[Ext<K>],
) -> Result<()> {
    check_len("E matrix (rows)", layout.d, e_matrix.len())?;
    for row in e_matrix {
        check_len("E matrix (columns)", layout.d, row.len())?;
    }
    check_len(
        "eq(tau_1) table (2^ceil(log nu) entries)",
        pow2(layout.log_nu, "the eq(tau_1) table")?,
        eq1.len(),
    )
}

/// The weights $c\_u = \sum\_{i' \lt n'} \widetilde{eq}(\tau\_1, \mathsf{bit}(i'))\, M\_{i',u}(\alpha)$
/// of the columns $u \lt \mu$ of $\mathbf M$ (fixture convention `M_tilde`), from the powers
/// $\alpha^0, \dots, \alpha^{d-1}$. Zero coefficients of $\mathbf M$ are skipped.
pub(crate) fn column_weights<const K: usize>(
    field: &Fqk<K>,
    m: &[Vec<Poly>],
    eq1: &[Ext<K>],
    apow: &[Ext<K>],
) -> Vec<Ext<K>> {
    let zq = field.zq();
    let cols = m.first().map_or(0, Vec::len);
    let mut c = vec![field.zero(); cols];
    for (row, e) in m.iter().zip(eq1) {
        // beta_j = eq1[i'] alpha^j, so that c_u += sum_j M[i'][u][j] beta_j.
        let beta: Vec<Ext<K>> = apow.iter().map(|p| field.mul(e, p)).collect();
        for (cu, poly) in c.iter_mut().zip(row) {
            for (coef, b) in poly.0.iter().zip(&beta) {
                let coef = zq.reduce(*coef);
                if coef != 0 {
                    *cu = field.add(cu, &field.scale(b, coef));
                }
            }
        }
    }
    c
}

/// The hypercube values of
/// $$M^\ast(U, L) = \sum\_{i' \in \[\nu\]} \widetilde{eq}(\tau\_1, \mathsf{bit}(i'))\,
/// \widetilde M\_\alpha(\mathsf{bit}(i'), U, L)$$
/// (`5_Building_Blocks.tex`:254-260, 271; the oracle's `mstar_table`), $2^{\ell''}$ elements of
/// $\mathbb F\_{q^k}$, by the fixture convention `M_tilde`: at $(u\_0, l\_0)$,
/// - $c\_u \alpha^{l\_0}$ for $u\_0 \lt \mu$ and every $l\_0 \lt 2d$, with
///   $c\_u = \sum\_{i\_0 \lt n'} \mathrm{eq}\_1\[i\_0\] M\_{i\_0, u\_0}(\alpha)$ (first term, $u \le \mu$;
///   `ref/README.md`, choice 10);
/// - $-\mathrm{eq}\_1\[u\_0 - \mu\] (\alpha^d + 1) \alpha^{l\_0}$ for $\mu \le u\_0 \lt \mu + n'$ (first
///   term, $u = i' + \mu$);
/// - plus $\sum\_{j\_0 \lt d} \mathrm{eq}\_1\[n' + j\_0\] E\[j\_0\]\[l\_0\]$ at $u\_0 = \mu - 1$,
///   $l\_0 \lt d$ (second term, the $E$ rows; choice 9);
/// - plus $\mathrm{eq}\_1\[n' + d + u\_0 + (l\_0 - d)(\mu + n')\]$ for $u\_0 \lt \mu + n'$,
///   $l\_0 \ge d$ (third term, the high coefficients);
/// - and $0$ on the padding rows $u\_0 \ge \mu + n'$,
///
/// where $\mathrm{eq}\_1$ is [`eq_table`](crate::eq_table)$(\tau\_1)$ and $E\[j\]\[l\]$ is
/// coefficient $j$ of $E(X^l)$ ([`hachi_co_math::Subfield::e_matrix`]). The cost is
/// $O(\text{nonzero coefficients of } \mathbf M + 2^{\ell''} + d^2)$ operations; the verifier
/// evaluates the multilinear extension at one point without the table
/// ([`mstar_eval_factorized`](crate::mstar_eval_factorized)).
///
/// # Errors
/// [`Error::Shape`] unless $\mathbf M$ is $n' \times \mu$ with $d$ coefficients per entry,
/// $E$ is $d \times d$ and $\mathrm{eq}\_1$ has $2^{\lceil \log \nu \rceil}$ entries.
pub fn mstar_table<const K: usize>(
    ctx: &Context<K>,
    m: &[Vec<Poly>],
    e_matrix: &[Vec<u64>],
    eq1: &[Ext<K>],
    alpha: &Ext<K>,
) -> Result<Vec<Ext<K>>> {
    let l = &ctx.layout;
    let field = ctx.field();
    let (d, mu, np) = (l.d, l.mu, l.n_prime);
    let two_d = 2 * d;
    check_matrix(m, np, mu, d)?;
    check_e_and_eq1(l, e_matrix, eq1)?;
    let apow = powers(field, alpha, two_d);
    let c = column_weights(field, m, eq1, &apow[..d]);
    let mut table = vec![field.zero(); table_size(l)?];
    // First term, u0 < mu: c_u alpha^{l0} for all l0 < 2d.
    for (u0, cu) in c.iter().enumerate() {
        for (l0, p) in apow.iter().enumerate() {
            table[u0 * two_d + l0] = field.mul(cu, p);
        }
    }
    // First term, u0 = mu + i0: -eq1[i0] (alpha^d + 1) alpha^{l0}.
    let ad1 = field.add(&apow[d], &field.one());
    for (i0, e) in eq1.iter().take(np).enumerate() {
        let coef = field.neg(&field.mul(e, &ad1));
        for (l0, p) in apow.iter().enumerate() {
            table[(mu + i0) * two_d + l0] = field.mul(&coef, p);
        }
    }
    // Second term: rows n' + 1 .. n' + d (1-indexed) at u0 = mu - 1, l0 < d.
    let zq = field.zq();
    for l0 in 0..d {
        let mut acc = field.zero();
        for (j0, row) in e_matrix.iter().enumerate() {
            let coef = zq.reduce(row[l0]);
            if coef != 0 {
                acc = field.add(&acc, &field.scale(&eq1[np + j0], coef));
            }
        }
        let x = (mu - 1) * two_d + l0;
        table[x] = field.add(&table[x], &acc);
    }
    // Third term: u0 < mu + n', l0 >= d, i' - 1 = n' + d + u0 + (l0 - d)(mu + n').
    for u0 in 0..mu + np {
        for l0 in d..two_d {
            let x = u0 * two_d + l0;
            let i = np + d + u0 + (l0 - d) * (mu + np);
            table[x] = field.add(&table[x], &eq1[i]);
        }
    }
    Ok(table)
}

/// The claimed sum
/// $$a = \sum\_{i' \in \[n'\]} \widetilde{eq}(\tau\_1, \mathsf{bit}(i'))\, y\_{i'}(\alpha) +
/// \sum\_{i' = n'+1}^{n'+d} \widetilde{eq}(\tau\_1, \mathsf{bit}(i'))\, y\_{i' - n' - 1}$$
/// (`5_Building_Blocks.tex`:263-264; the oracle's `claimed_sum`), where $\mathbf y \in
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

    #[test]
    fn range_product_matches_the_literal_product() {
        let q = 134217757u64; // toy-n3
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
