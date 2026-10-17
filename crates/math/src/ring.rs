//! The ring $\mathcal R_q = \mathbb Z_q\[X\]/(X^d + 1)$ (draft `4_preliminaries.tex`:67) for a
//! runtime prime $q < 2^{63}$ and a power of two $2 \le d \le 4096$.
//!
//! A ring element is the vector of its $d$ coefficients, index $j$ for $X^j$ ([`Poly`]). Products
//! are exact: [`Ring::mul`], [`Ring::inner`] and [`Ring::mat_vec`] run on the RNS plan of
//! [`crate::rns`], and [`Ring::mul_schoolbook`] is the quadratic reference they are tested
//! against. Every method reads coefficients modulo $q$ and returns reduced coefficients; the
//! only errors are polynomials of the wrong length and invalid exponents.

use crate::{
    arith::{reduce, shoup, shoup_mul},
    error::{Error, Result},
    rns::Plan,
    zq::Zq,
};
use std::sync::Arc;

/// A ring element as its coefficients: entry $j$ is the coefficient of $X^j$, a residue in
/// $[0, q)$ in canonical form (`docs/m0-spec.md`, "Conventions").
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Poly(pub Vec<u64>);

impl Poly {
    /// The coefficients, index $j$ for $X^j$ (draft `4_preliminaries.tex`:67).
    pub fn coeffs(&self) -> &[u64] {
        &self.0
    }
}

impl AsRef<[u64]> for Poly {
    fn as_ref(&self) -> &[u64] {
        &self.0
    }
}

/// The ring $\mathcal R_q = \mathbb Z_q\[X\]/(X^d + 1)$ (draft `4_preliminaries.tex`:67) with its
/// exact multiplication plan. Cloning is cheap (the plan is shared).
#[derive(Clone, Debug)]
pub struct Ring {
    zq: Zq,
    d: usize,
    plan: Arc<Plan>,
}

impl PartialEq for Ring {
    fn eq(&self, rhs: &Self) -> bool {
        self.zq == rhs.zq && self.d == rhs.d
    }
}
impl Eq for Ring {}

impl Ring {
    /// $\mathcal R_q$ for an odd prime $q < 2^{63}$ and a power of two $d = 2^\alpha$,
    /// $2 \le d \le 4096$ (draft `4_preliminaries.tex`:61, 67: ring degrees up to 2048 and the
    /// ring of degree $2d$ used for quotients).
    ///
    /// # Errors
    /// [`Error::Modulus`] for an invalid $q$ and [`Error::Parameter`] for an invalid $d$.
    pub fn new(q: u64, d: usize) -> Result<Self> {
        let zq = Zq::new(q)?;
        let plan = Plan::new(q, d)?;
        Ok(Self {
            zq,
            d,
            plan: Arc::new(plan),
        })
    }

    /// The coefficient ring $\mathbb Z_q$ (draft `4_preliminaries.tex`:67).
    pub fn zq(&self) -> &Zq {
        &self.zq
    }

    /// The modulus $q$ (draft `4_preliminaries.tex`:61).
    pub fn modulus(&self) -> u64 {
        self.zq.modulus()
    }

    /// The degree $d = 2^\alpha$ (draft `4_preliminaries.tex`:61).
    pub fn degree(&self) -> usize {
        self.d
    }

    /// The exact multiplication plan ([`crate::rns`]) behind the products in $\mathcal R_q$ (draft
    /// `4_preliminaries.tex`:67).
    pub fn rns(&self) -> &Plan {
        &self.plan
    }

    fn check(&self, a: &Poly) -> Result<()> {
        if a.0.len() != self.d {
            return Err(Error::Length {
                expected: self.d,
                actual: a.0.len(),
            });
        }
        Ok(())
    }

    /// The zero element of $\mathcal R_q$ (draft `4_preliminaries.tex`:67).
    pub fn zero(&self) -> Poly {
        Poly(vec![0; self.d])
    }

    /// The monomial $X^e$, for any integer $e$ (taken modulo $2d$, as $X^d = -1$ and
    /// $X^{2d} = 1$ in $\mathcal R_q$, draft `4_preliminaries.tex`:67).
    pub fn monomial(&self, e: i64) -> Poly {
        let mut out = self.zero();
        let e = e.rem_euclid(2 * self.d as i64) as usize;
        if e < self.d {
            out.0[e] = 1;
        } else {
            out.0[e - self.d] = self.modulus() - 1;
        }
        out
    }

    /// The polynomial with these coefficients, reduced modulo $q$ (index $j$ for $X^j$, draft
    /// `4_preliminaries.tex`:67).
    ///
    /// # Errors
    /// [`Error::Length`] unless there are $d$ coefficients.
    pub fn from_coeffs(&self, coeffs: &[u64]) -> Result<Poly> {
        let a = Poly(coeffs.iter().map(|x| self.zq.reduce(*x)).collect());
        self.check(&a)?;
        Ok(a)
    }

    /// The polynomial with these signed integer coefficients, reduced modulo $q$: the image of an
    /// element of $\mathcal R = \mathbb Z\[X\]/(X^d + 1)$ in $\mathcal R_q$ (draft
    /// `4_preliminaries.tex`:67).
    ///
    /// # Errors
    /// [`Error::Length`] unless there are $d$ coefficients.
    pub fn from_i64(&self, coeffs: &[i64]) -> Result<Poly> {
        let a = Poly(coeffs.iter().map(|x| self.zq.from_i64(*x)).collect());
        self.check(&a)?;
        Ok(a)
    }

    /// The centered representatives $a_j \bmod^{\pm} q$ of the coefficients (draft
    /// `4_preliminaries.tex`:86).
    ///
    /// # Errors
    /// [`Error::Length`] unless `a` has $d$ coefficients.
    pub fn to_centered(&self, a: &Poly) -> Result<Vec<i64>> {
        self.check(a)?;
        Ok(a.0.iter().map(|x| self.zq.to_centered(*x)).collect())
    }

    /// $a + b$ in $\mathcal R_q$ (draft `4_preliminaries.tex`:67).
    ///
    /// # Errors
    /// [`Error::Length`] unless both operands have $d$ coefficients.
    pub fn add(&self, a: &Poly, b: &Poly) -> Result<Poly> {
        self.check(a)?;
        self.check(b)?;
        Ok(Poly(
            a.0.iter()
                .zip(&b.0)
                .map(|(x, y)| self.zq.add(*x, *y))
                .collect(),
        ))
    }

    /// $a - b$ in $\mathcal R_q$ (draft `4_preliminaries.tex`:67).
    ///
    /// # Errors
    /// [`Error::Length`] unless both operands have $d$ coefficients.
    pub fn sub(&self, a: &Poly, b: &Poly) -> Result<Poly> {
        self.check(a)?;
        self.check(b)?;
        Ok(Poly(
            a.0.iter()
                .zip(&b.0)
                .map(|(x, y)| self.zq.sub(*x, *y))
                .collect(),
        ))
    }

    /// $-a$ in $\mathcal R_q$ (draft `4_preliminaries.tex`:67).
    ///
    /// # Errors
    /// [`Error::Length`] unless `a` has $d$ coefficients.
    pub fn neg(&self, a: &Poly) -> Result<Poly> {
        self.check(a)?;
        Ok(Poly(a.0.iter().map(|x| self.zq.neg(*x)).collect()))
    }

    /// The exact product $ab$ in $\mathcal R_q$ (draft `4_preliminaries.tex`:67), by the RNS plan
    /// ([`crate::rns::Plan::mul`]).
    ///
    /// # Errors
    /// [`Error::Length`] unless both operands have $d$ coefficients.
    pub fn mul(&self, a: &Poly, b: &Poly) -> Result<Poly> {
        Ok(Poly(self.plan.mul(&a.0, &b.0)?))
    }

    /// The product $ab$ by the schoolbook rule $X^i X^j = X^{i+j}$, $X^d = -1$ (draft
    /// `4_preliminaries.tex`:67), with one `u128` remainder per term: the quadratic reference for
    /// [`Ring::mul`].
    ///
    /// # Errors
    /// [`Error::Length`] unless both operands have $d$ coefficients.
    pub fn mul_schoolbook(&self, a: &Poly, b: &Poly) -> Result<Poly> {
        self.check(a)?;
        self.check(b)?;
        let d = self.d;
        let mut out = vec![0u64; d];
        for (i, x) in a.0.iter().enumerate() {
            for (j, y) in b.0.iter().enumerate() {
                let term = self.zq.mul(*x, *y);
                if i + j < d {
                    out[i + j] = self.zq.add(out[i + j], term);
                } else {
                    out[i + j - d] = self.zq.sub(out[i + j - d], term);
                }
            }
        }
        Ok(Poly(out))
    }

    /// $ca$ for a scalar $c \in \mathbb Z_q$ (read modulo $q$; draft `4_preliminaries.tex`:67), by
    /// Shoup products with the constant
    /// $\lfloor c 2^{64}/q \rfloor$.
    ///
    /// # Errors
    /// [`Error::Length`] unless `a` has $d$ coefficients.
    pub fn scalar_mul(&self, c: u64, a: &Poly) -> Result<Poly> {
        self.check(a)?;
        let q = self.modulus();
        let w = reduce(c, q);
        let w_shoup = shoup(w, q);
        Ok(Poly(
            a.0.iter().map(|x| shoup_mul(*x, w, w_shoup, q)).collect(),
        ))
    }

    /// $X^e \cdot a$ for any integer $e$ (taken modulo $2d$; $X^d = -1$, draft
    /// `4_preliminaries.tex`:67).
    ///
    /// # Errors
    /// [`Error::Length`] unless `a` has $d$ coefficients.
    pub fn mul_monomial(&self, a: &Poly, e: i64) -> Result<Poly> {
        self.check(a)?;
        let d = self.d;
        let e = e.rem_euclid(2 * d as i64) as usize;
        let mut out = vec![0u64; d];
        for (j, x) in a.0.iter().enumerate() {
            let pos = (j + e) % (2 * d);
            if pos < d {
                out[pos] = self.zq.reduce(*x);
            } else {
                out[pos - d] = self.zq.neg(*x);
            }
        }
        Ok(Poly(out))
    }

    /// The Galois automorphism $\sigma_i(a) = a(X^i)$ for an odd integer $i$ (taken modulo
    /// $2d$), so that $\sigma_{-1}$ is `automorphism(a, -1)` (draft `4_preliminaries.tex`:69).
    ///
    /// # Errors
    /// [`Error::Parameter`] for an even $i$, [`Error::Length`] unless `a` has $d$ coefficients.
    pub fn automorphism(&self, a: &Poly, i: i64) -> Result<Poly> {
        self.check(a)?;
        let d = self.d;
        let i = i.rem_euclid(2 * d as i64) as usize;
        if i.is_multiple_of(2) {
            return Err(Error::Parameter("automorphism index must be odd"));
        }
        // j -> i j mod 2d permutes the monomials up to sign, as i is a unit modulo 2d.
        let mut out = vec![0u64; d];
        for (j, x) in a.0.iter().enumerate() {
            let pos = (i * j) % (2 * d);
            if pos < d {
                out[pos] = self.zq.reduce(*x);
            } else {
                out[pos - d] = self.zq.neg(*x);
            }
        }
        Ok(Poly(out))
    }

    /// The exact inner product $\sum_t a_t b_t$ over $\mathcal R_q$ (draft
    /// `4_preliminaries.tex`:67; [`crate::rns::Plan::inner`]).
    ///
    /// # Errors
    /// [`Error::Length`] for sides of different lengths or polynomials without $d$
    /// coefficients.
    pub fn inner(&self, a: &[Poly], b: &[Poly]) -> Result<Poly> {
        Ok(Poly(self.plan.inner(a, b)?))
    }

    /// The exact matrix-vector product $(\sum_j m_{ij} v_j)\_i$ over $\mathcal R_q$, as for
    /// $\mathbf t = \mathbf A \mathbf s$ (draft `6_Collaborative_Hachi.tex`:370-376;
    /// [`crate::rns::Plan::mat_vec`]).
    ///
    /// # Errors
    /// [`Error::Length`] for rows of the wrong length or polynomials without $d$ coefficients.
    pub fn mat_vec(&self, m: &[Vec<Poly>], v: &[Poly]) -> Result<Vec<Poly>> {
        Ok(self.plan.mat_vec(m, v)?.into_iter().map(Poly).collect())
    }

    /// The infinity norm $\Vert a \Vert_\infty = \max_j |a_j \bmod^{\pm} q|$ of the centered
    /// coefficients (draft `4_preliminaries.tex`:86).
    ///
    /// # Errors
    /// [`Error::Length`] unless `a` has $d$ coefficients.
    pub fn inf_norm(&self, a: &Poly) -> Result<u64> {
        self.check(a)?;
        Ok(a.0
            .iter()
            .map(|x| self.zq.to_centered(*x).unsigned_abs())
            .max()
            .unwrap_or(0))
    }
}
