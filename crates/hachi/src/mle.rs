//! Multilinear helpers over $\mathbb F_{q^K}$: tables of $\widetilde{eq}$, evaluation of a
//! multilinear extension, of a ring element at a field point, and Lagrange interpolation.
//!
//! **Bit order.** A table of $2^n$ values is indexed by $\mathrm{nat}(\mathbf b) = \sum_j b_j
//! 2^{n-j}$, so the first variable goes with the most significant bit, as everywhere in Hachi-Co
//! (`ref/README.md`, "Hypercube").

use hachi_co_math::{Ext, Fqk, Poly};

/// $(\widetilde{eq}(\mathbf x, \mathbf b))\_{\mathbf b \in \lbrace 0,1 \rbrace^n}$ for
/// $\mathbf x \in \mathbb F_{q^K}^n$, with $\widetilde{eq}(\mathbf x, \mathbf b) = \prod_j (x_j
/// b_j + (1 - x_j)(1 - b_j))$.
pub fn eq_table<const K: usize>(field: &Fqk<K>, x: &[Ext<K>]) -> Vec<Ext<K>> {
    let mut out = vec![field.one()];
    for xj in x {
        let one_minus = field.sub(&field.one(), xj);
        out = out
            .iter()
            .flat_map(|v| [field.mul(v, &one_minus), field.mul(v, xj)])
            .collect();
    }
    out
}

/// $\widetilde{eq}(\mathbf x, \mathbf y) = \prod_j (x_j y_j + (1 - x_j)(1 - y_j))$.
pub fn eq_eval<const K: usize>(field: &Fqk<K>, x: &[Ext<K>], y: &[Ext<K>]) -> Ext<K> {
    x.iter().zip(y).fold(field.one(), |acc, (a, b)| {
        let ab = field.mul(a, b);
        let one = field.one();
        let rest = field.mul(&field.sub(&one, a), &field.sub(&one, b));
        field.mul(&acc, &field.add(&ab, &rest))
    })
}

/// The multilinear extension of `table` ($2^n$ values) at $\mathbf x \in \mathbb F_{q^K}^n$,
/// binding the first (most significant) variable first.
///
/// # Panics
/// Never for a table of $2^{|\mathbf x|}$ values; callers check the lengths.
pub fn mle_eval<const K: usize>(field: &Fqk<K>, table: &[Ext<K>], x: &[Ext<K>]) -> Ext<K> {
    let mut t = table.to_vec();
    for r in x {
        let half = t.len() / 2;
        let (lo, hi) = t.split_at(half);
        t = lo
            .iter()
            .zip(hi)
            .map(|(a, b)| field.add(a, &field.mul(r, &field.sub(b, a))))
            .collect();
    }
    t.first().copied().unwrap_or_else(|| field.zero())
}

/// $(1, x, x^2, \dots, x^{n-1})$.
pub fn powers<const K: usize>(field: &Fqk<K>, x: &Ext<K>, n: usize) -> Vec<Ext<K>> {
    let mut out = Vec::with_capacity(n);
    let mut p = field.one();
    for _ in 0..n {
        out.push(p);
        p = field.mul(&p, x);
    }
    out
}

/// $p(\alpha) = \sum_\ell p_\ell \alpha^\ell$ for a ring element $p$ read in $\mathbb Z_q\[X\]$,
/// given `powers` $= (\alpha^\ell)\_{\ell < d}$.
pub fn eval_poly<const K: usize>(field: &Fqk<K>, p: &Poly, powers: &[Ext<K>]) -> Ext<K> {
    p.0.iter().zip(powers).fold(field.zero(), |acc, (c, a)| {
        field.add(&acc, &field.scale(a, *c))
    })
}

/// The value at $r$ of the polynomial of degree below `evals.len()` that takes the value
/// `evals[i]` at $i = 0, 1, \dots$ (Lagrange interpolation; the nodes are distinct because $q$
/// exceeds their number).
pub fn interpolate<const K: usize>(field: &Fqk<K>, evals: &[Ext<K>], r: &Ext<K>) -> Ext<K> {
    let zq = field.zq();
    let n = evals.len();
    let mut out = field.zero();
    for (i, e) in evals.iter().enumerate() {
        let mut num = field.one();
        let mut den = 1u64;
        for j in 0..n {
            if j != i {
                num = field.mul(&num, &field.sub(r, &field.from_base(j as u64)));
                let diff = zq.sub(zq.reduce(i as u64), zq.reduce(j as u64));
                den = zq.mul(den, diff);
            }
        }
        let inv = zq.inv(den).unwrap_or(0);
        out = field.add(&out, &field.scale(&field.mul(e, &num), inv));
    }
    out
}
