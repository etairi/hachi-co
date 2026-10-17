//! The subfield $\mathcal R_q^H$, the trace $\mathrm{Tr}\_H$, the packing map $\psi$, its inverse
//! and the map $E$ of the field-to-ring reduction.
//!
//! Definitions (draft `4_preliminaries.tex`:69-84; `field-to-ring-rok.tex`:116, 132-164; Hachi,
//! Lemma 5, Eq. (7) and Theorem 2), exactly as in the Sage reference
//! `ref/hachico_ref/subfield.py`:
//! - $\sigma_i : X \mapsto X^i$ for odd $i$; $H = \langle \sigma_{-1}, \sigma_{4k+1} \rangle$,
//!   $|H| = d/k$, for $k = 2^\kappa$ dividing $d/2$; $\mathcal R_q^H$ is the set of elements
//!   fixed by $H$, a field isomorphic to $\mathbb F_{q^k}$ for $q \equiv 5 \pmod 8$ (Lemma 5),
//!   with coordinates as in [`crate::fqk`].
//! - $\mathrm{Tr}\_H(a) = \sum_{\sigma \in H} \sigma(a)$.
//! - $\psi(a_0, \dots, a_{d/k-1}) = \sum_{i<d/(2k)} a_i X^i + X^{d/2} \sum_{i<d/(2k)}
//!   a_{d/(2k)+i} X^i$ (Theorem 2, Eq. (8)); Theorem 2:
//!   $\mathrm{Tr}\_H(\psi(\mathbf a) \cdot \sigma_{-1}(\psi(\mathbf b))) = (d/k) \langle \mathbf a,
//!   \mathbf b \rangle$.
//! - $\psi^{-1}(u)\_i = (k/d) \mathrm{Tr}\_H(u \cdot X^{-e_i})$ with $e_i = i$ for $i < d/(2k)$ and
//!   $e_i = d/2 + (i - d/(2k))$ otherwise ($\psi$ of the $i$-th unit vector is $X^{e_i}$).
//! - $E(u) = (k/d) \mathrm{Tr}\_H(u \cdot \sigma_{-1}(v))$ for $v = \psi$ of a monomial vector
//!   (`field-to-ring-rok.tex`:116, 134).
//!
//! **Closed forms.** With $s = d/(2k)$ (so $t = X^s$):
//! - $\mathrm{Tr}\_H(X^j) = 0$ unless $s \mid j$ (Hachi, proof of Theorem 2, second claim), and
//!   $\mathrm{Tr}\_H(t^r) = (d/(2k))(t^r + t^{-r})$, since $\sigma_h(t^r) = t^r$ for
//!   $h \equiv 1 \pmod{4k}$ (first claim). So the coordinates of $\mathrm{Tr}\_H(a)$ are
//!   $c_0 = (d/k) a_0$ and $c_m = (d/(2k)) (a_{ms} - a_{(2k-m)s})$, read from $2k$ coefficients
//!   ([`Subfield::trace`]).
//! - $\psi(\mathbf a)$ places coordinates directly: with $A = a_i$ and $B = a_{s+i}$, the
//!   coefficient at $rs + i$ is $A_0$ for $r = 0$; $A_r + B_{k-r}$ for $0 < r < k$; $B_0$ for
//!   $r = k$; and $B_{r-k} - A_{2k-r}$ for $k < r < 2k$ (Hachi, Eq. (9) and the proof of
//!   Theorem 2). Solving gives $\psi^{-1}$: $A_m = (c_m - c_{2k-m})/2$ and
//!   $B_{k-m} = (c_m + c_{2k-m})/2$ for $0 < m < k$.
//!
//! The tests compare every closed form with its definition (sums over $H$ of automorphisms,
//! products with monomials) and with fixtures from the Sage reference.

use crate::{
    error::{Error, Result},
    fqk::{Ext, Fqk},
    ring::{Poly, Ring},
};

/// $\mathcal R_q^H \cong \mathbb F_{q^K}$ inside a ring $\mathcal R_q$ (Hachi, Lemma 5), with
/// $\mathrm{Tr}\_H$, $\psi$, $\psi^{-1}$ and $E$ (draft `4_preliminaries.tex`:69-84,
/// `field-to-ring-rok.tex`:132-164).
#[derive(Clone, Debug)]
pub struct Subfield<const K: usize> {
    ring: Ring,
    field: Fqk<K>,
    /// $s = d/(2K)$: $t = X^s$.
    step: usize,
    /// The odd exponents of $H$ modulo $2d$, sorted.
    h: Vec<usize>,
    /// $2^{-1} \bmod q$.
    inv2: u64,
    /// $d/K \bmod q$ and $d/(2K) \bmod q$.
    d_over_k: u64,
    d_over_2k: u64,
}

impl<const K: usize> Subfield<K> {
    /// The subfield of $\mathcal R_q$ for $K \in \lbrace 1, 2, 4, 8 \rbrace$ dividing $d/2$ and
    /// $q \equiv 5 \pmod 8$ (Hachi, Lemma 5).
    ///
    /// # Errors
    /// [`Error::ModulusClass`] unless $q \equiv 5 \pmod 8$; [`Error::Parameter`] unless
    /// $K \in \lbrace 1, 2, 4, 8 \rbrace$ and $K \mid d/2$.
    pub fn new(ring: &Ring) -> Result<Self> {
        let field = Fqk::<K>::new(ring.modulus())?;
        let d = ring.degree();
        if !(d / 2).is_multiple_of(K) || d < 2 * K {
            return Err(Error::Parameter("K must divide d/2"));
        }
        // The closure of {1} under multiplication by the generators, as in subfield.py.
        let two_d = 2 * d;
        let generators = [two_d - 1, (4 * K + 1) % two_d];
        let mut member = vec![false; two_d];
        member[1] = true;
        let mut frontier = vec![1usize];
        while let Some(x) = frontier.pop() {
            for g in generators {
                let y = x * g % two_d;
                if !member[y] {
                    member[y] = true;
                    frontier.push(y);
                }
            }
        }
        let h: Vec<usize> = (0..two_d).filter(|e| member[*e]).collect();
        debug_assert_eq!(h.len(), d / K, "|H| = d/k (Hachi, proof of Theorem 2)");
        let zq = ring.zq();
        Ok(Self {
            ring: ring.clone(),
            field,
            step: d / (2 * K),
            h,
            inv2: zq.inv(2).expect("q is odd"),
            d_over_k: zq.reduce((d / K) as u64),
            d_over_2k: zq.reduce((d / (2 * K)) as u64),
        })
    }

    /// The ring $\mathcal R_q$ that contains the subfield (draft `4_preliminaries.tex`:67).
    pub fn ring(&self) -> &Ring {
        &self.ring
    }

    /// The field $\mathbb F_{q^K} \cong \mathcal R_q^H$ of the coordinates (Hachi, Lemma 5).
    pub fn field(&self) -> &Fqk<K> {
        &self.field
    }

    /// The exponents $h$ of the automorphisms $\sigma_h \in H = \langle \sigma_{-1},
    /// \sigma_{4K+1} \rangle$, odd residues modulo $2d$ in increasing order; there are $d/K$ of
    /// them (draft `4_preliminaries.tex`:73).
    pub fn h_exponents(&self) -> &[usize] {
        &self.h
    }

    fn check(&self, a: &Poly) -> Result<()> {
        let d = self.ring.degree();
        if a.0.len() != d {
            return Err(Error::Length {
                expected: d,
                actual: a.0.len(),
            });
        }
        Ok(())
    }

    /// The ring element $a_0 + \sum_{m=1}^{K-1} a_m (t^m - t^{2K-m})$, $t = X^{d/(2K)}$, of
    /// coordinates $(a_0, \dots, a_{K-1})$ (Hachi, Eq. (7)): $a_m$ at position $m d/(2K)$ and
    /// $-a_m$ at position $(2K - m) d/(2K)$.
    pub fn embed(&self, x: &Ext<K>) -> Poly {
        let zq = self.ring.zq();
        let mut out = self.ring.zero();
        out.0[0] = zq.reduce(x.0[0]);
        for m in 1..K {
            out.0[m * self.step] = zq.reduce(x.0[m]);
            out.0[(2 * K - m) * self.step] = zq.neg(x.0[m]);
        }
        out
    }

    /// The coordinates of an element of $\mathcal R_q^H$, read at positions $m d/(2K)$.
    ///
    /// # Errors
    /// [`Error::NotInSubfield`] unless `a` has the form of Eq. (7), [`Error::Length`] unless it
    /// has $d$ coefficients.
    pub fn extract(&self, a: &Poly) -> Result<Ext<K>> {
        self.check(a)?;
        let zq = self.ring.zq();
        let x = Ext(core::array::from_fn(|m| zq.reduce(a.0[m * self.step])));
        let canonical = a.0.iter().map(|c| zq.reduce(*c));
        if !canonical.eq(self.embed(&x).0.iter().copied()) {
            return Err(Error::NotInSubfield);
        }
        Ok(x)
    }

    /// Whether `a` is in $\mathcal R_q^H$, i.e. has the form of Eq. (7) (by Lemma 5 these are
    /// exactly the elements fixed by $H$). False for a polynomial without $d$ coefficients.
    pub fn is_in_subfield(&self, a: &Poly) -> bool {
        self.extract(a).is_ok()
    }

    /// The coordinates of $\mathrm{Tr}\_H(a)$: $c_0 = (d/K) a_0$ and
    /// $c_m = (d/(2K))(a_{ms} - a_{(2K-m)s})$, $s = d/(2K)$ (module documentation).
    fn trace_coordinates(&self, a: &[u64]) -> Ext<K> {
        let zq = self.ring.zq();
        Ext(core::array::from_fn(|m| {
            if m == 0 {
                zq.mul(self.d_over_k, a[0])
            } else {
                let diff = zq.sub(a[m * self.step], a[(2 * K - m) * self.step]);
                zq.mul(self.d_over_2k, diff)
            }
        }))
    }

    /// $\mathrm{Tr}\_H(a) = \sum_{\sigma \in H} \sigma(a) \in \mathcal R_q^H$ (draft
    /// `4_preliminaries.tex`:69), by the closed form of the module documentation.
    ///
    /// # Errors
    /// [`Error::Length`] unless `a` has $d$ coefficients.
    pub fn trace(&self, a: &Poly) -> Result<Poly> {
        self.check(a)?;
        Ok(self.embed(&self.trace_coordinates(&a.0)))
    }

    /// $\psi(a_0, \dots, a_{d/K-1}) = \sum_{i<d/(2K)} a_i X^i + X^{d/2} \sum_{i<d/(2K)}
    /// a_{d/(2K)+i} X^i$ (Hachi, Theorem 2, Eq. (8); draft `4_preliminaries.tex`:74-75), by
    /// direct placement of the coordinates (module documentation).
    ///
    /// # Errors
    /// [`Error::Length`] unless there are $d/K$ field elements.
    pub fn psi(&self, a: &[Ext<K>]) -> Result<Poly> {
        let d = self.ring.degree();
        if a.len() != d / K {
            return Err(Error::Length {
                expected: d / K,
                actual: a.len(),
            });
        }
        let zq = self.ring.zq();
        let s = self.step;
        let mut out = self.ring.zero();
        for i in 0..s {
            let (x, y) = (&a[i].0, &a[s + i].0);
            out.0[i] = zq.reduce(x[0]);
            out.0[K * s + i] = zq.reduce(y[0]);
            for r in 1..K {
                out.0[r * s + i] = zq.add(x[r], y[K - r]);
                out.0[(K + r) * s + i] = zq.sub(y[r], x[K - r]);
            }
        }
        Ok(out)
    }

    /// $\psi^{-1}(u)$, the unique $\mathbf a$ with $\psi(\mathbf a) = u$ ($\psi$ is a bijection,
    /// Theorem 2), which equals $((K/d) \mathrm{Tr}\_H(u X^{-e_i}))\_i$; computed by the closed
    /// form of the module documentation.
    ///
    /// # Errors
    /// [`Error::Length`] unless `u` has $d$ coefficients.
    pub fn psi_inv(&self, u: &Poly) -> Result<Vec<Ext<K>>> {
        self.check(u)?;
        let zq = self.ring.zq();
        let s = self.step;
        let mut low = vec![Ext([0u64; K]); s];
        let mut high = vec![Ext([0u64; K]); s];
        for i in 0..s {
            let c = |r: usize| zq.reduce(u.0[r * s + i]);
            low[i].0[0] = c(0);
            high[i].0[0] = c(K);
            for m in 1..K {
                let (cm, cc) = (c(m), c(2 * K - m));
                low[i].0[m] = zq.mul(zq.sub(cm, cc), self.inv2);
                high[i].0[K - m] = zq.mul(zq.add(cm, cc), self.inv2);
            }
        }
        low.extend(high);
        Ok(low)
    }

    /// The monomials $(\prod_j x_j^{b_j})\_{\mathbf b \in \lbrace 0, 1 \rbrace^s}$ of a point
    /// $\mathbf x \in \mathbb F_{q^K}^s$, ordered by $\mathrm{nat}(\mathbf b) = \sum_j b_j
    /// 2^{s-j}$ ($b_1$ the most significant bit; `docs/m0-spec.md`, "Bit strings"), as
    /// `Subfield.monomials` in `ref/hachico_ref/subfield.py`. These are the vectors of
    /// `field-to-ring-rok.tex`:113-116.
    ///
    /// # Errors
    /// [`Error::Overflow`] for $s \ge 48$ (the output would have $2^s$ entries).
    pub fn monomials(&self, x: &[Ext<K>]) -> Result<Vec<Ext<K>>> {
        if x.len() >= 48 {
            return Err(Error::Overflow);
        }
        let mut out = Vec::with_capacity(1 << x.len());
        out.push(self.field.one());
        for xj in x {
            // Appending x_j as the new least significant bit: v -> (v, v x_j).
            out = out
                .iter()
                .flat_map(|v| [*v, self.field.mul(v, xj)])
                .collect();
        }
        Ok(out)
    }

    /// $E(u) = (K/d) \mathrm{Tr}\_H(u \cdot \sigma_{-1}(v)) \in \mathcal R_q^H$
    /// (`field-to-ring-rok.tex`:134, 162), from the exact product in $\mathcal R_q$ and the
    /// closed form of the trace: its coordinates are $y_0$ and $(y_{ms} - y_{(2K-m)s})/2$ for
    /// $y = u \sigma_{-1}(v)$.
    ///
    /// # Errors
    /// [`Error::Length`] unless `u` and `v` have $d$ coefficients.
    pub fn e_map(&self, u: &Poly, v: &Poly) -> Result<Poly> {
        self.check(u)?;
        let w = self.ring.automorphism(v, -1)?;
        let y = self.ring.mul(u, &w)?;
        Ok(self.embed(&self.e_coordinates(|p| y.0[p])))
    }

    /// The coordinates of $(K/d) \mathrm{Tr}\_H(y)$, given the coefficients of $y$.
    fn e_coordinates(&self, y: impl Fn(usize) -> u64) -> Ext<K> {
        let zq = self.ring.zq();
        Ext(core::array::from_fn(|m| {
            if m == 0 {
                zq.reduce(y(0))
            } else {
                zq.mul(
                    zq.sub(y(m * self.step), y((2 * K - m) * self.step)),
                    self.inv2,
                )
            }
        }))
    }

    /// The $d \times d$ matrix $\mathbf M$ over $\mathbb Z_q$ with
    /// $\mathrm{cf}(E(u)) = \mathbf M \cdot \mathrm{cf}(u)$ (`field-to-ring-rok.tex`:140-158),
    /// row-major: column $j$ is $\mathrm{cf}(E(X^j))$, computed in $O(K)$ from
    /// $\sigma_{-1}(v)$ shifted by $j$.
    ///
    /// # Errors
    /// [`Error::Length`] unless `v` has $d$ coefficients.
    pub fn e_matrix(&self, v: &Poly) -> Result<Vec<Vec<u64>>> {
        let d = self.ring.degree();
        let w = self.ring.automorphism(v, -1)?;
        let zq = self.ring.zq();
        // Column j: the coordinates of E(X^j), read from the coefficients of X^j w.
        let columns: Vec<Ext<K>> = (0..d)
            .map(|j| {
                self.e_coordinates(|p| {
                    if p >= j {
                        w.0[p - j]
                    } else {
                        zq.neg(w.0[p + d - j])
                    }
                })
            })
            .collect();
        let mut m = vec![vec![0u64; d]; d];
        for (j, c) in columns.iter().enumerate() {
            m[0][j] = c.0[0];
            for r in 1..K {
                m[r * self.step][j] = c.0[r];
                m[(2 * K - r) * self.step][j] = zq.neg(c.0[r]);
            }
        }
        Ok(m)
    }
}
