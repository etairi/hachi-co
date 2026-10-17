//! The blinding polynomials $g\_{0,p}$ and $g\_{1,p}$ (`5_Building_Blocks.tex`:209-219; oracle
//! `sumcheck.Blinder`; `ref/README.md`, "Blinders"):
//! $$g(V) = s\_0 + \sum\_{j=1}^{\ell''} \sum\_{k=1}^{D} s\_{j,k} V\_j^k,$$
//! with $D = d\_0$ for $g\_{0,p}$ and $D = d\_1$ for $g\_{1,p}$. The coefficients live in
//! [`hachi_co_core::Blinder`] (`s[j - 1][k - 1]` $= s\_{j,k}$); this module has the closed forms:
//! the hypercube sum, the round polynomials and the evaluation.
//!
//! Since $\sum\_{v \in \lbrace 0, 1 \rbrace} v^k = 1$ for $k \ge 1$, summing a variable over
//! $\lbrace 0, 1 \rbrace$ doubles the terms without it and keeps each term $s\_{j,k} V\_j^k$ once
//! per remaining point; hence the closed forms below.

use crate::error::{Result, check_len};
use hachi_co_core::Blinder;
use hachi_co_math::{Ext, Fqk};

/// $\sum\_k s\_{j,k}$ of one row.
fn row_sum<const K: usize>(field: &Fqk<K>, row: &[Ext<K>]) -> Ext<K> {
    row.iter().fold(field.zero(), |acc, s| field.add(&acc, s))
}

/// $\sum\_{k=1}^{D} s\_{j,k} x^k$ of one row.
fn row_eval<const K: usize>(field: &Fqk<K>, row: &[Ext<K>], x: &Ext<K>) -> Ext<K> {
    let mut power = *x;
    let mut acc = field.zero();
    for s in row {
        acc = field.add(&acc, &field.mul(s, &power));
        power = field.mul(&power, x);
    }
    acc
}

/// $G = \sum\_{\mathbf v \in \lbrace 0, 1 \rbrace^{\ell''}} g(\mathbf v) = 2^{\ell''} s\_0 +
/// 2^{\ell'' - 1} \sum\_{j,k} s\_{j,k}$ (`5_Building_Blocks.tex`:218-219; the oracle's
/// `Blinder.hypercube_sum`), with $\ell''$ the number of rows of the blinder (for $\ell'' = 0$,
/// $G = s\_0$).
pub fn hypercube_sum<const K: usize>(field: &Fqk<K>, g: &Blinder<K>) -> Ext<K> {
    let zq = field.zq();
    let ell = g.s.len() as u64;
    if ell == 0 {
        return field.reduce(&g.s0);
    }
    let total = g.s.iter().fold(field.zero(), |acc, row| {
        field.add(&acc, &row_sum(field, row))
    });
    field.add(
        &field.scale(&g.s0, zq.pow(2, ell)),
        &field.scale(&total, zq.pow(2, ell - 1)),
    )
}

/// $g(\mathbf a) = s\_0 + \sum\_j \sum\_k s\_{j,k} a\_j^k$ (the oracle's `Blinder.evaluate`),
/// the value the provers open in `5_Building_Blocks.tex`:316-318.
///
/// # Errors
/// [`Error::Shape`](crate::Error::Shape) unless the point has one coordinate per row.
pub fn evaluate<const K: usize>(
    field: &Fqk<K>,
    g: &Blinder<K>,
    point: &[Ext<K>],
) -> Result<Ext<K>> {
    check_len(
        "blinder point (one coordinate per variable)",
        g.s.len(),
        point.len(),
    )?;
    Ok(g.s
        .iter()
        .zip(point)
        .fold(field.reduce(&g.s0), |acc, (row, a)| {
            field.add(&acc, &row_eval(field, row, a))
        }))
}

/// The round polynomial of round $r \in \lbrace 1, \dots, \ell'' \rbrace$ at $x$:
/// $$\sum\_{v\_{r+1}, \dots, v\_{\ell''} \in \lbrace 0, 1 \rbrace} g(a\_1, \dots, a\_{r-1}, x,
/// v\_{r+1}, \dots) = 2^{\ell'' - r} \Bigl(s\_0 + \sum\_{j \lt r} \sum\_k s\_{j,k} a\_j^k + \sum\_k
/// s\_{r,k} x^k\Bigr) + 2^{\ell'' - r - 1} \sum\_{j \gt r} \sum\_k s\_{j,k}$$
/// (the last term only for $r \lt \ell''$; the oracle's `Blinder.round_poly`), given the prefix
/// $(a\_1, \dots, a\_{r-1})$.
///
/// # Errors
/// [`Error::Shape`](crate::Error::Shape) unless $1 \le r \le \ell''$ and the prefix has $r - 1$
/// coordinates.
pub fn round_poly<const K: usize>(
    field: &Fqk<K>,
    g: &Blinder<K>,
    r: usize,
    prefix: &[Ext<K>],
    x: &Ext<K>,
) -> Result<Ext<K>> {
    let ell = g.s.len();
    if r == 0 || r > ell {
        return Err(crate::Error::Shape {
            what: "blinder round (1..=l'')",
            expected: ell,
            actual: r,
        });
    }
    check_len(
        "blinder round prefix (r - 1 coordinates)",
        r - 1,
        prefix.len(),
    )?;
    let zq = field.zq();
    let mut pre = field.reduce(&g.s0);
    for (row, a) in g.s.iter().zip(prefix) {
        pre = field.add(&pre, &row_eval(field, row, a));
    }
    let current = row_eval(field, &g.s[r - 1], x);
    let mut out = field.scale(&field.add(&pre, &current), zq.pow(2, (ell - r) as u64));
    if r < ell {
        let post = g.s[r..].iter().fold(field.zero(), |acc, row| {
            field.add(&acc, &row_sum(field, row))
        });
        out = field.add(&out, &field.scale(&post, zq.pow(2, (ell - r - 1) as u64)));
    }
    Ok(out)
}

/// The provers' incremental form of [`round_poly`]: the prefix sum $s\_0 + \sum\_{j \lt r} \sum\_k
/// s\_{j,k} a\_j^k$ is updated as the coins arrive, and the suffix sums are precomputed, so that a
/// round costs $O(D)$ operations per node.
#[derive(Clone, Debug)]
pub(crate) struct BlinderRounds<const K: usize> {
    /// $s\_0 + \sum\_{j \lt r} \sum\_k s\_{j,k} a\_j^k$ before round $r$.
    prefix: Ext<K>,
    /// `suffix[j]` $= \sum\_{j' \ge j} \sum\_k s\_{j'+1,k}$ (0-based rows), $\ell'' + 1$ entries.
    suffix: Vec<Ext<K>>,
    /// $2^e \bmod q$ for $e = 0, \dots, \ell''$.
    pow2: Vec<u64>,
}

impl<const K: usize> BlinderRounds<K> {
    pub(crate) fn new(field: &Fqk<K>, g: &Blinder<K>) -> Self {
        let ell = g.s.len();
        let mut suffix = vec![field.zero(); ell + 1];
        for j in (0..ell).rev() {
            suffix[j] = field.add(&suffix[j + 1], &row_sum(field, &g.s[j]));
        }
        let zq = field.zq();
        Self {
            prefix: field.reduce(&g.s0),
            suffix,
            pow2: (0..=ell as u64).map(|e| zq.pow(2, e)).collect(),
        }
    }

    /// The values of the round polynomial of round $r$ ($1 \le r \le \ell''$) at the nodes
    /// $t = 1, \dots, \mathrm{count}$.
    pub(crate) fn values(
        &self,
        field: &Fqk<K>,
        g: &Blinder<K>,
        r: usize,
        count: usize,
    ) -> Vec<Ext<K>> {
        let zq = field.zq();
        let ell = g.s.len();
        let row = &g.s[r - 1];
        let tail = if r < ell {
            field.scale(&self.suffix[r], self.pow2[ell - r - 1])
        } else {
            field.zero()
        };
        (1..=count as u64)
            .map(|t| {
                let mut power = zq.reduce(t);
                let mut current = field.zero();
                for s in row {
                    current = field.add(&current, &field.scale(s, power));
                    power = zq.mul(power, t);
                }
                let head = field.scale(&field.add(&self.prefix, &current), self.pow2[ell - r]);
                field.add(&head, &tail)
            })
            .collect()
    }

    /// Adds $\sum\_k s\_{r,k} a\_r^k$ to the prefix after the coin $a\_r$ of round $r$.
    pub(crate) fn bind(&mut self, field: &Fqk<K>, g: &Blinder<K>, r: usize, a: &Ext<K>) {
        self.prefix = field.add(&self.prefix, &row_eval(field, &g.s[r - 1], a));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eq::msb_bits;

    const Q: u64 = 8388637;

    fn blinder(ell: usize, degree: usize, seed: u64) -> Blinder<4> {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            Ext(core::array::from_fn(|c| (s >> (7 * c + 5)) % Q))
        };
        Blinder {
            s0: next(),
            s: (0..ell)
                .map(|_| (0..degree).map(|_| next()).collect())
                .collect(),
        }
    }

    fn bits_point(f: &Fqk<4>, x: usize, ell: usize) -> Vec<Ext<4>> {
        msb_bits(x, ell)
            .iter()
            .map(|b| f.from_base(u64::from(*b)))
            .collect()
    }

    /// The closed forms against brute force over the hypercube at l'' = 4 (the oracle's
    /// self-test), for both degrees used by the protocol.
    #[test]
    fn closed_forms_match_brute_force() {
        let f = Fqk::<4>::new(Q).unwrap();
        for (ell, degree) in [(4usize, 14usize), (4, 3), (1, 2), (3, 1)] {
            let g = blinder(ell, degree, 42 + degree as u64);
            let brute: Ext<4> = (0..1usize << ell).fold(f.zero(), |acc, x| {
                f.add(&acc, &evaluate(&f, &g, &bits_point(&f, x, ell)).unwrap())
            });
            assert_eq!(hypercube_sum(&f, &g), brute);
            let coins: Vec<Ext<4>> = (0..ell as u64).map(|j| Ext([j + 3, 1, j, 2])).collect();
            let mut rounds = BlinderRounds::new(&f, &g);
            for r in 1..=ell {
                let prefix = &coins[..r - 1];
                let fast = rounds.values(&f, &g, r, degree + 1);
                for t in 1..=degree + 1 {
                    let x = f.from_base(t as u64);
                    // Brute force: sum over the suffix of g(prefix, x, suffix).
                    let rest = ell - r;
                    let mut acc = f.zero();
                    for v in 0..1usize << rest {
                        let mut point = prefix.to_vec();
                        point.push(x);
                        point.extend(bits_point(&f, v, rest));
                        acc = f.add(&acc, &evaluate(&f, &g, &point).unwrap());
                    }
                    assert_eq!(round_poly(&f, &g, r, prefix, &x).unwrap(), acc);
                    assert_eq!(
                        fast[t - 1],
                        acc,
                        "l'' = {ell}, D = {degree}, r = {r}, t = {t}"
                    );
                }
                rounds.bind(&f, &g, r, &coins[r - 1]);
            }
            // After the last round the prefix is g(a).
            assert_eq!(rounds.prefix, evaluate(&f, &g, &coins).unwrap());
        }
    }

    #[test]
    fn round_and_point_shapes_are_checked() {
        let f = Fqk::<4>::new(Q).unwrap();
        let g = blinder(3, 2, 1);
        assert!(evaluate(&f, &g, &[f.one(); 2]).is_err());
        assert!(round_poly(&f, &g, 0, &[], &f.one()).is_err());
        assert!(round_poly(&f, &g, 4, &[f.one(); 3], &f.one()).is_err());
        assert!(round_poly(&f, &g, 2, &[], &f.one()).is_err());
    }
}
