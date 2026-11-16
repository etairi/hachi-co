//! Multilinear helpers over $\mathbb F_{q^K}$: tables of $\widetilde{eq}$, evaluation of a
//! multilinear extension, of a ring element at a field point, and Lagrange interpolation.
//!
//! **Bit order.** A table of $2^n$ values is indexed by $\mathrm{nat}(\mathbf b) = \sum_j b_j
//! 2^{n-j}$, so the first variable goes with the most significant bit, as everywhere in Hachi-Co
//! (`ref/README.md`, "Hypercube").

use hachi_co_math::montgomery::{Arith, Montgomery};
use hachi_co_math::{Ext, Fqk, Poly};
use rayon::prelude::*;

/// Montgomery's form for the products of a table's entries by one constant, where the modulus
/// admits it and the table has at least [`MONTGOMERY_FROM`] entries: with the constant entered,
/// $\phi(c) x = cx$ is canonical ([`hachi_co_math::montgomery`]), the same field element as the
/// canonical product, in about half its time. Smaller tables take the canonical products.
fn montgomery_for<const K: usize>(field: &Fqk<K>, entries: usize) -> Option<Montgomery<K>> {
    (entries >= MONTGOMERY_FROM)
        .then(|| Montgomery::<K>::new(field.modulus()))
        .flatten()
}

/// The size from which [`montgomery_for`] gives Montgomery's form: below it, forming the form
/// costs more than the products save.
const MONTGOMERY_FROM: usize = 64;

/// $(\widetilde{eq}(\mathbf x, \mathbf b))\_{\mathbf b \in \lbrace 0,1 \rbrace^n}$ for
/// $\mathbf x \in \mathbb F_{q^K}^n$, with $\widetilde{eq}(\mathbf x, \mathbf b) = \prod_j (x_j
/// b_j + (1 - x_j)(1 - b_j))$.
pub fn eq_table<const K: usize>(field: &Fqk<K>, x: &[Ext<K>]) -> Vec<Ext<K>> {
    let mut out = vec![field.one()];
    for xj in x {
        let mut next = vec![field.zero(); 2 * out.len()];
        match montgomery_for(field, out.len()) {
            Some(m) => {
                let xm = m.enter(xj);
                next.par_chunks_mut(2)
                    .zip(out.par_iter())
                    .for_each(|(children, v)| {
                        children[1] = m.mul(&xm, v);
                        children[0] = field.sub(v, &children[1]);
                    });
            }
            None => next
                .par_chunks_mut(2)
                .zip(out.par_iter())
                .for_each(|(children, v)| {
                    children[1] = field.mul(v, xj);
                    children[0] = field.sub(v, &children[1]);
                }),
        }
        out = next;
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

/// Binds the first (most significant) variable of `table` to $r$: $T'\[x\] = T\[x\] + r (T\[x +
/// h\] - T\[x\])$ for the half $h$ of its length, in place.
pub fn fold_top<const K: usize>(field: &Fqk<K>, table: &mut Vec<Ext<K>>, r: &Ext<K>) {
    let half = table.len() / 2;
    let (low, high) = table.split_at_mut(half);
    match montgomery_for(field, half) {
        Some(m) => {
            let rm = m.enter(r);
            low.par_iter_mut().zip(high.par_iter()).for_each(|(a, b)| {
                *a = field.add(a, &m.mul(&rm, &field.sub(b, a)));
            });
        }
        None => low.par_iter_mut().zip(high.par_iter()).for_each(|(a, b)| {
            *a = field.add(a, &field.mul(r, &field.sub(b, a)));
        }),
    }
    table.truncate(half);
    table.shrink_to_fit();
}

/// The multilinear extension of `table` ($2^n$ values) at $\mathbf x \in \mathbb F_{q^K}^n$,
/// binding the first (most significant) variable first.
///
/// # Panics
/// Never for a table of $2^{|\mathbf x|}$ values; callers check the lengths.
pub fn mle_eval<const K: usize>(field: &Fqk<K>, table: &[Ext<K>], x: &[Ext<K>]) -> Ext<K> {
    let mut t = table.to_vec();
    for r in x {
        fold_top(field, &mut t, r);
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

#[cfg(test)]
mod tests {
    use super::*;
    use hachi_co_math::zq::is_prime;

    const K: usize = 4;

    fn exts(q: u64, n: usize, seed: u64) -> Vec<Ext<K>> {
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..n)
            .map(|i| {
                Ext(core::array::from_fn(|c| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    // Boundary residues now and then.
                    match (i + c) % 11 {
                        0 => 0,
                        1 => q - 1,
                        _ => state % q,
                    }
                }))
            })
            .collect()
    }

    /// The canonical binding of the first variable, as `fold_top` formed it before Montgomery's
    /// products.
    fn fold_canonical(field: &Fqk<K>, table: &[Ext<K>], r: &Ext<K>) -> Vec<Ext<K>> {
        let half = table.len() / 2;
        (0..half)
            .map(|i| {
                let (a, b) = (&table[i], &table[i + half]);
                field.add(a, &field.mul(r, &field.sub(b, a)))
            })
            .collect()
    }

    /// The canonical eq table, as `eq_table` formed it before Montgomery's products.
    fn eq_canonical(field: &Fqk<K>, x: &[Ext<K>]) -> Vec<Ext<K>> {
        let mut out = vec![field.one()];
        for xj in x {
            out = out
                .iter()
                .flat_map(|v| {
                    let high = field.mul(v, xj);
                    [field.sub(v, &high), high]
                })
                .collect();
        }
        out
    }

    #[test]
    fn the_products_in_montgomery_form_are_the_canonical_ones() {
        // fold_top and eq_table, below and above MONTGOMERY_FROM, at a small modulus, at
        // secure-l24's and at one too large for Montgomery's lazy reduction at K = 4.
        let mut large = (1u64 << 62) + 5;
        while !is_prime(large) {
            large += 8;
        }
        assert!(Montgomery::<K>::new(large).is_none());
        for q in [8_388_637u64, 15395032342003157, large] {
            let field = Fqk::<K>::new(q).unwrap();
            for log in [1usize, 5, 6, 7, 10] {
                let table = exts(q, 1 << log, log as u64);
                let r = exts(q, 1, 77 + log as u64)[0];
                let mut folded = table.clone();
                fold_top(&field, &mut folded, &r);
                assert_eq!(folded, fold_canonical(&field, &table, &r), "q {q}, 2^{log}");
                let x = exts(q, log, 91 + log as u64);
                assert_eq!(
                    eq_table(&field, &x),
                    eq_canonical(&field, &x),
                    "q {q}, {log}"
                );
            }
        }
    }

    /// Item 139's measurement for the bindings: `fold_top` on 2^21 entries and `eq_table` of 21
    /// variables at secure-l24's modulus, Montgomery's products against the canonical ones,
    /// alternating, best of 3. Run with
    /// `RAYON_NUM_THREADS=5 cargo test --release -p hachi --lib timings_bindings -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing"]
    fn timings_bindings() {
        use std::time::Instant;
        let q = 15395032342003157u64;
        let field = Fqk::<K>::new(q).unwrap();
        let table = exts(q, 1 << 21, 5);
        let r = exts(q, 1, 6)[0];
        let x = exts(q, 21, 7);
        let canonical_fold = |table: &mut Vec<Ext<K>>| {
            let half = table.len() / 2;
            let (low, high) = table.split_at_mut(half);
            low.par_iter_mut().zip(high.par_iter()).for_each(|(a, b)| {
                *a = field.add(a, &field.mul(&r, &field.sub(b, a)));
            });
            table.truncate(half);
        };
        let canonical_eq = || {
            let mut out = vec![field.one()];
            for xj in &x {
                let mut next = vec![field.zero(); 2 * out.len()];
                next.par_chunks_mut(2)
                    .zip(out.par_iter())
                    .for_each(|(children, v)| {
                        children[1] = field.mul(v, xj);
                        children[0] = field.sub(v, &children[1]);
                    });
                out = next;
            }
            out
        };
        let (mut fold_m, mut fold_c, mut eq_m, mut eq_c) = (f64::MAX, f64::MAX, f64::MAX, f64::MAX);
        for _ in 0..3 {
            let (mut a, mut b) = (table.clone(), table.clone());
            let start = Instant::now();
            fold_top(&field, &mut a, &r);
            fold_m = fold_m.min(start.elapsed().as_secs_f64());
            let start = Instant::now();
            canonical_fold(&mut b);
            fold_c = fold_c.min(start.elapsed().as_secs_f64());
            assert_eq!(a, b);
            let start = Instant::now();
            let e = eq_table(&field, &x);
            eq_m = eq_m.min(start.elapsed().as_secs_f64());
            let start = Instant::now();
            let f = canonical_eq();
            eq_c = eq_c.min(start.elapsed().as_secs_f64());
            assert_eq!(e, f);
        }
        println!(
            "fold_top 2^21: {fold_m:.4} s Montgomery, {fold_c:.4} s canonical ({:.2}x); \
             eq_table 2^21: {eq_m:.4} s Montgomery, {eq_c:.4} s canonical ({:.2}x)",
            fold_c / fold_m,
            eq_c / eq_m
        );
    }
}
