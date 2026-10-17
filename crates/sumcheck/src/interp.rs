//! Univariate polynomials through their values at the nodes $1, 2, \dots, m$: the round messages
//! of the sumcheck are $H\_r(1), \dots, H\_r(D + 1)$ (`5_Building_Blocks.tex`:290-294; oracle
//! `ref/README.md`, choice 11), the verifier interpolates $H\_r(0)$ and $H\_r(a\_r)$ from them, and
//! the provers extend the values of $F\_{0,p}$ and $F\_{1,p}$ computed at a few nodes to all of
//! them ([`Extrapolation`]); plus the oracle's degree audit ([`degree_at_nodes`]).

use crate::error::{Error, Result, check_len};
use hachi_co_math::{Ext, Fqk, Zq};

/// The barycentric weights $w\_j = \prod\_{i \ne j} (j - i)^{-1}$, $j = 1, \dots, m$, of the
/// nodes $1, \dots, m$ in $\mathbb Z\_q$.
///
/// # Errors
/// [`Error::Parameter`] for $m = 0$ or $m \gt q$ (two nodes would coincide modulo $q$).
fn weights(zq: &Zq, m: usize) -> Result<Vec<u64>> {
    if m == 0 || m as u128 > u128::from(zq.modulus()) {
        return Err(Error::Parameter(format!(
            "{m} interpolation nodes 1..={m} are not distinct modulo q = {}",
            zq.modulus()
        )));
    }
    (1..=m)
        .map(|j| {
            let mut den = 1u64;
            for i in (1..=m).filter(|i| *i != j) {
                // j - i is in [-(m - 1), m - 1], nonzero modulo q as m <= q.
                let diff = if j > i {
                    zq.reduce((j - i) as u64)
                } else {
                    zq.neg((i - j) as u64)
                };
                den = zq.mul(den, diff);
            }
            zq.inv(den)
                .ok_or_else(|| Error::Parameter(format!("the nodes 1..={m} coincide modulo q")))
        })
        .collect()
}

/// Lagrange interpolation through the values $(H(1), \dots, H(m))$ of a polynomial $H$ of degree
/// below $m$ over $\mathbb F\_{q^k}$ (the oracle's `Fqk.lagrange_eval`; `ref/README.md`, "Round
/// messages"): $H(x) = \sum\_{j=1}^{m} H(j) \prod\_{i \ne j} (x - i)/(j - i)$, valid at every $x$,
/// nodes included.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nodes {
    weights: Vec<u64>,
}

impl Nodes {
    /// The nodes $1, \dots, m$.
    ///
    /// # Errors
    /// [`Error::Parameter`] for $m = 0$ or $m \gt q$.
    pub fn new(zq: &Zq, m: usize) -> Result<Self> {
        Ok(Self {
            weights: weights(zq, m)?,
        })
    }

    /// The number $m$ of nodes.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Never true: there is at least one node.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// $H(x)$ for the polynomial of degree below $m$ with $H(j)$ = `values[j - 1]`, in $O(m)$
    /// operations (prefix and suffix products of the $x - i$).
    ///
    /// # Errors
    /// [`Error::Shape`] unless there are $m$ values.
    pub fn eval<const K: usize>(
        &self,
        field: &Fqk<K>,
        values: &[Ext<K>],
        x: &Ext<K>,
    ) -> Result<Ext<K>> {
        let m = self.weights.len();
        check_len("interpolation values (one per node)", m, values.len())?;
        let diffs: Vec<Ext<K>> = (1..=m as u64)
            .map(|i| field.sub(x, &field.from_base(i)))
            .collect();
        // suffix[j] = prod_{i >= j} (x - nodes[i]) (0-based).
        let mut suffix = vec![field.one(); m + 1];
        for j in (0..m).rev() {
            suffix[j] = field.mul(&suffix[j + 1], &diffs[j]);
        }
        let mut prefix = field.one();
        let mut out = field.zero();
        for j in 0..m {
            let basis = field.scale(&field.mul(&prefix, &suffix[j + 1]), self.weights[j]);
            out = field.add(&out, &field.mul(&basis, &values[j]));
            prefix = field.mul(&prefix, &diffs[j]);
        }
        Ok(out)
    }
}

/// The extension of the values of a polynomial of degree below `from` at the nodes
/// $1, \dots, \mathrm{from}$ to the nodes $\mathrm{from} + 1, \dots, \mathrm{to}$:
/// $H(t) = \sum\_{j} c\_{t,j} H(j)$ with the base-field coefficients
/// $c\_{t,j} = w\_j \prod\_{i \ne j} (t - i)$ of [`Nodes`].
///
/// The provers compute the round polynomial of $F\_{0,p}$, of degree at most $2b + 1$, at the
/// nodes $1, \dots, 2b + 2$ and extend it to $1, \dots, d\_0 + 1$ (and the round polynomial of
/// $F\_{1,p}$, of degree at most 2, from three nodes to $d\_1 + 1$); this is exact because the
/// polynomial is determined by its values at more nodes than its degree
/// (`ref/README.md`, "Observations about the draft": $d\_0$ is larger than needed).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Extrapolation {
    from: usize,
    /// `coeffs[t - from - 1][j - 1]` $= c\_{t,j}$.
    coeffs: Vec<Vec<u64>>,
}

impl Extrapolation {
    /// From the nodes $1, \dots, \mathrm{from}$ to $1, \dots, \mathrm{to}$ (`to` $\ge$ `from`).
    ///
    /// # Errors
    /// [`Error::Parameter`] for `from` $= 0$, `from` $\gt q$ or `to` $\lt$ `from`.
    pub(crate) fn new(zq: &Zq, from: usize, to: usize) -> Result<Self> {
        if to < from {
            return Err(Error::Parameter(format!(
                "cannot extend {from} node values to {to} nodes"
            )));
        }
        let w = weights(zq, from)?;
        let coeffs = (from + 1..=to)
            .map(|t| {
                (1..=from)
                    .map(|j| {
                        let mut c = w[j - 1];
                        for i in (1..=from).filter(|i| *i != j) {
                            // t > from >= i, so t - i > 0.
                            c = zq.mul(c, zq.reduce((t - i) as u64));
                        }
                        c
                    })
                    .collect()
            })
            .collect();
        Ok(Self { from, coeffs })
    }

    /// Appends the values at the nodes $\mathrm{from} + 1, \dots, \mathrm{to}$ to `values`, which
    /// holds the values at $1, \dots, \mathrm{from}$.
    ///
    /// # Errors
    /// [`Error::Shape`] unless `values` has `from` entries.
    pub(crate) fn extend<const K: usize>(
        &self,
        field: &Fqk<K>,
        values: &mut Vec<Ext<K>>,
    ) -> Result<()> {
        check_len("values to extrapolate", self.from, values.len())?;
        for row in &self.coeffs {
            let mut acc = field.zero();
            for (c, v) in row.iter().zip(values.iter()) {
                acc = field.add(&acc, &field.scale(v, *c));
            }
            values.push(acc);
        }
        Ok(())
    }
}

/// The degree of the polynomial through `values` at the equally spaced nodes $1, \dots, m$, by
/// forward differences: the largest $j$ with $\Delta^j \ne 0$, or `None` for the zero polynomial
/// (the oracle's `sumcheck.degree_at_nodes`, the "degree audit" of the fixtures, key
/// `sumcheck/round_degrees_unblinded`).
pub fn degree_at_nodes<const K: usize>(field: &Fqk<K>, values: &[Ext<K>]) -> Option<usize> {
    let mut current: Vec<Ext<K>> = values.iter().map(|v| field.reduce(v)).collect();
    let mut degree = None;
    for j in 0..values.len() {
        if current.iter().any(|v| !field.is_zero(v)) {
            degree = Some(j);
        }
        current = current
            .windows(2)
            .map(|w| field.sub(&w[1], &w[0]))
            .collect();
    }
    degree
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q: u64 = 8388637; // toy-n1

    fn field() -> Fqk<4> {
        Fqk::new(Q).unwrap()
    }

    /// A deterministic element from a counter (splitmix64, seed in the argument).
    fn ext(seed: u64) -> Ext<4> {
        let mut s = seed;
        Ext(core::array::from_fn(|_| {
            s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            (z ^ (z >> 31)) % Q
        }))
    }

    /// Horner evaluation of sum_i c_i x^i.
    fn horner(f: &Fqk<4>, coeffs: &[Ext<4>], x: &Ext<4>) -> Ext<4> {
        coeffs
            .iter()
            .rev()
            .fold(f.zero(), |acc, c| f.add(&f.mul(&acc, x), c))
    }

    #[test]
    fn interpolation_reproduces_random_polynomials() {
        let f = field();
        for m in 1..=16usize {
            let nodes = Nodes::new(f.zq(), m).unwrap();
            assert_eq!(nodes.len(), m);
            let coeffs: Vec<Ext<4>> = (0..m as u64).map(|i| ext(100 * m as u64 + i)).collect();
            let values: Vec<Ext<4>> = (1..=m as u64)
                .map(|t| horner(&f, &coeffs, &f.from_base(t)))
                .collect();
            for x in [
                f.zero(),
                f.from_base(1),
                f.from_base(m as u64),
                ext(7),
                ext(8),
            ] {
                assert_eq!(
                    nodes.eval(&f, &values, &x).unwrap(),
                    horner(&f, &coeffs, &x),
                    "m = {m}"
                );
            }
            assert!(nodes.eval(&f, &values[1..], &f.zero()).is_err());
        }
    }

    #[test]
    fn extrapolation_is_exact_below_the_degree_bound() {
        let f = field();
        for from in 1..=10usize {
            for to in from..=16 {
                let ext_ = Extrapolation::new(f.zq(), from, to).unwrap();
                let coeffs: Vec<Ext<4>> = (0..from as u64).map(|i| ext(31 * i + 5)).collect();
                let all: Vec<Ext<4>> = (1..=to as u64)
                    .map(|t| horner(&f, &coeffs, &f.from_base(t)))
                    .collect();
                let mut values = all[..from].to_vec();
                ext_.extend(&f, &mut values).unwrap();
                assert_eq!(values, all, "{from} -> {to}");
            }
        }
        // A polynomial of degree `from` is not extended correctly (the failure boundary).
        let ext_ = Extrapolation::new(f.zq(), 3, 6).unwrap();
        let coeffs = [ext(1), ext(2), ext(3), ext(4)];
        let all: Vec<Ext<4>> = (1..=6u64)
            .map(|t| horner(&f, &coeffs, &f.from_base(t)))
            .collect();
        let mut values = all[..3].to_vec();
        ext_.extend(&f, &mut values).unwrap();
        assert_ne!(values, all);
        assert!(Extrapolation::new(f.zq(), 4, 3).is_err());
        assert!(Extrapolation::new(f.zq(), 0, 3).is_err());
    }

    #[test]
    fn too_many_nodes_for_the_modulus_are_refused() {
        let zq = Zq::new(5).unwrap();
        assert!(Nodes::new(&zq, 5).is_ok());
        assert!(Nodes::new(&zq, 6).is_err());
        assert!(Nodes::new(&zq, 0).is_err());
    }

    #[test]
    fn degree_audit_by_forward_differences() {
        let f = field();
        for deg in 0..=9usize {
            let coeffs: Vec<Ext<4>> = (0..=deg as u64).map(|i| ext(17 * i + 3)).collect();
            let values: Vec<Ext<4>> = (1..=15u64)
                .map(|t| horner(&f, &coeffs, &f.from_base(t)))
                .collect();
            assert_eq!(degree_at_nodes(&f, &values), Some(deg));
        }
        assert_eq!(degree_at_nodes(&f, &[f.zero(); 15]), None);
        assert_eq!(degree_at_nodes::<4>(&f, &[]), None);
    }
}
