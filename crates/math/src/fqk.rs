//! $\mathbb F_{q^k} \cong \mathcal R_q^H$ in Hachi's basis, for $k \in \lbrace 1, 2, 4, 8 \rbrace$
//! and a prime $q \equiv 5 \pmod 8$ (Hachi, Lemma 5 and Eq. (7); draft
//! `4_preliminaries.tex`:77-79).
//!
//! **Basis.** Let $t = X^{d/(2k)}$, a primitive $4k$-th root of unity in $\mathcal R_q$
//! ($t^{2k} = X^d = -1$), and $e_m = t^m + t^{-m} = t^m - t^{2k-m}$. An element is
//! $a_0 + \sum_{m=1}^{k-1} a_m e_m$, with coordinates $(a_0, \dots, a_{k-1}) \in \mathbb Z_q^k$
//! (Eq. (7) with $a_m$ the coefficient of $X^{m d/(2k)}$; `docs/m0-spec.md`, "Field"). The
//! structure is that of $\mathcal R_q$ and does not depend on $d$.
//!
//! **Multiplication** (the Chebyshev rule): $e_a e_b = e_{a+b} + e_{|a-b|}$, with $e_0 = 2$,
//! $e_k = 0$ and $e_{2k-m} = -e_m$. Hence, for $x \cdot y = z$,
//! $$z_0 = x_0 y_0 + 2 \sum_{a=1}^{k-1} x_a y_a, \qquad z_m = x_0 y_m + x_m y_0 +
//! \sum_{a + b = m} x_a y_b + \sum_{|a - b| = m} x_a y_b -
//! \sum_{a + b = 2k - m} x_a y_b \quad (0 < m < k),$$
//! with every sum over $a, b \ge 1$. The tests compare it with the product of the embeddings in
//! $\mathcal R_q$ and with the Sage reference `ref/hachico_ref/subfield.py` (`fmul`).
//!
//! **Frobenius.** On $\mathcal R_q$, $x \mapsto x^q$ is the automorphism $\sigma_q$ (as
//! $(\sum_j a_j X^j)^q = \sum_j a_j X^{qj}$ in characteristic $q$), so on the basis it is the
//! signed permutation $e_m \mapsto e_{qm \bmod 4k}$, rewritten with $e_{4k-j} = e_j$ and
//! $e_{2k-j} = -e_j$.
//!
//! **Inverse.** $x^{-1} = N(x)^{-1} \prod_{j=1}^{k-1} \varphi^j(x)$, with $\varphi$ the Frobenius
//! and $N(x) = \prod_{j=0}^{k-1} \varphi^j(x) \in \mathbb F_q^\times$ the norm.
//!
//! Every operation reads the coordinates of its arguments modulo $q$ and returns reduced
//! coordinates, so it never fails.

use crate::{
    arith::{reduce, sub_mod},
    error::{Error, Result},
    zq::Zq,
};

/// An element of $\mathbb F_{q^k}$ by its coordinates $(a_0, \dots, a_{K-1})$ in Hachi's basis
/// (Eq. (7)): $a_0 + \sum_{m=1}^{K-1} a_m (t^m - t^{2K-m})$. Canonical coordinates are in
/// $[0, q)$; every [`Fqk`] operation returns canonical coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Ext<const K: usize>(pub [u64; K]);

/// The field $\mathbb F_{q^K} \cong \mathcal R_q^H$ for $K \in \lbrace 1, 2, 4, 8 \rbrace$ and a
/// prime $q \equiv 5 \pmod 8$, $q < 2^{63}$ (Hachi, Lemma 5).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fqk<const K: usize> {
    zq: Zq,
    /// The Frobenius as a signed permutation: coordinate $m$ of $x$ goes to coordinate
    /// `frobenius[m].0` of $x^q$, negated if `frobenius[m].1`.
    frobenius: [(usize, bool); K],
}

/// A `u128` accumulator of products below $2^{126}$, reduced lazily so that it never overflows.
#[derive(Clone, Copy)]
struct Acc(u128);
impl Acc {
    #[inline(always)]
    fn add(&mut self, product: u128, q: u64) {
        if self.0 >= 1 << 127 {
            self.0 %= u128::from(q);
        }
        self.0 += product;
    }
    #[inline(always)]
    fn reduce(self, q: u64) -> u64 {
        (self.0 % u128::from(q)) as u64
    }
}

impl<const K: usize> Fqk<K> {
    /// The field for a prime $q \equiv 5 \pmod 8$, $q < 2^{63}$ (Hachi, Lemma 5; draft
    /// `4_preliminaries.tex`:61).
    ///
    /// # Errors
    /// [`Error::Parameter`] unless $K \in \lbrace 1, 2, 4, 8 \rbrace$, [`Error::Modulus`] for
    /// an invalid $q$, [`Error::ModulusClass`] unless $q \equiv 5 \pmod 8$.
    pub fn new(q: u64) -> Result<Self> {
        if !matches!(K, 1 | 2 | 4 | 8) {
            return Err(Error::Parameter("extension degree must be 1, 2, 4 or 8"));
        }
        let zq = Zq::new(q)?;
        if q % 8 != 5 {
            return Err(Error::ModulusClass(q));
        }
        let four_k = 4 * K as u64;
        let mut frobenius = [(0usize, false); K];
        for (m, slot) in frobenius.iter_mut().enumerate().skip(1) {
            // e_m -> e_j with j = q m mod 4K; e_{4K-j} = e_j and e_{2K-j} = -e_j.
            let j = ((q % four_k) * m as u64 % four_k) as usize;
            let j = if j > 2 * K { 4 * K - j } else { j };
            debug_assert!(j != 0 && j != K && j != 2 * K, "q m is not a multiple of K");
            *slot = if j < K { (j, false) } else { (2 * K - j, true) };
        }
        Ok(Self { zq, frobenius })
    }

    /// The base field $\mathbb Z_q = \mathbb F_q$, the coordinates of Eq. (7).
    pub fn zq(&self) -> &Zq {
        &self.zq
    }

    /// The modulus $q$ (draft `4_preliminaries.tex`:61).
    pub fn modulus(&self) -> u64 {
        self.zq.modulus()
    }

    /// The element with the coordinates of `x` (Eq. (7)) reduced modulo $q$.
    pub fn reduce(&self, x: &Ext<K>) -> Ext<K> {
        Ext(x.0.map(|c| self.zq.reduce(c)))
    }

    /// $0$, all coordinates zero (Eq. (7)).
    pub fn zero(&self) -> Ext<K> {
        Ext([0; K])
    }

    /// $1$, the coordinates $(1, 0, \dots, 0)$ (Eq. (7): $a_0$ is the constant coefficient).
    pub fn one(&self) -> Ext<K> {
        self.from_base(1)
    }

    /// The base-field element $c \bmod q$ (coordinate $a_0$ of Eq. (7)).
    pub fn from_base(&self, c: u64) -> Ext<K> {
        let mut out = [0; K];
        out[0] = self.zq.reduce(c);
        Ext(out)
    }

    /// $x + y$, coordinatewise (the addition of $\mathcal R_q$, Hachi Lemma 5).
    pub fn add(&self, x: &Ext<K>, y: &Ext<K>) -> Ext<K> {
        Ext(core::array::from_fn(|m| self.zq.add(x.0[m], y.0[m])))
    }

    /// $x - y$, coordinatewise (Hachi, Lemma 5).
    pub fn sub(&self, x: &Ext<K>, y: &Ext<K>) -> Ext<K> {
        Ext(core::array::from_fn(|m| self.zq.sub(x.0[m], y.0[m])))
    }

    /// $-x$, coordinatewise (Hachi, Lemma 5).
    pub fn neg(&self, x: &Ext<K>) -> Ext<K> {
        Ext(x.0.map(|c| self.zq.neg(c)))
    }

    /// $c x$ for a base-field scalar $c$, coordinatewise (Eq. (7)).
    pub fn scale(&self, x: &Ext<K>, c: u64) -> Ext<K> {
        Ext(x.0.map(|a| self.zq.mul(a, c)))
    }

    /// $xy$, the product of $\mathcal R_q$ restricted to $\mathcal R_q^H$ (Hachi, Lemma 5), by the
    /// Chebyshev rule of the module documentation.
    pub fn mul(&self, x: &Ext<K>, y: &Ext<K>) -> Ext<K> {
        let q = self.modulus();
        let x = x.0.map(|c| reduce(c, q));
        let y = y.0.map(|c| reduce(c, q));
        let mut pos = [Acc(0); K];
        let mut neg = [Acc(0); K];
        for a in 0..K {
            for b in 0..K {
                let p = u128::from(x[a]) * u128::from(y[b]);
                if a == 0 || b == 0 {
                    // 1 * e_b = e_b, e_a * 1 = e_a (and 1 * 1 = 1).
                    pos[a + b].add(p, q);
                    continue;
                }
                // e_a e_b = e_{a+b} + e_{|a-b|}.
                let s = a + b;
                if s < K {
                    pos[s].add(p, q);
                } else if s > K {
                    neg[2 * K - s].add(p, q);
                }
                if a == b {
                    pos[0].add(p, q);
                    pos[0].add(p, q);
                } else {
                    pos[a.abs_diff(b)].add(p, q);
                }
            }
        }
        Ext(core::array::from_fn(|m| {
            sub_mod(pos[m].reduce(q), neg[m].reduce(q), q)
        }))
    }

    /// $x^2$ (Hachi, Lemma 5).
    pub fn square(&self, x: &Ext<K>) -> Ext<K> {
        self.mul(x, x)
    }

    /// $x^e$ in $\mathbb F_{q^K}$ (Hachi, Lemma 5) for an exponent given as little-endian `u64`
    /// limbs (empty means $e = 0$), by square and multiply; $x^0 = 1$, also for $x = 0$.
    pub fn pow(&self, x: &Ext<K>, e: &[u64]) -> Ext<K> {
        let mut result = self.one();
        let mut started = false;
        for limb in e.iter().rev() {
            for bit in (0..64).rev() {
                if started {
                    result = self.square(&result);
                }
                if (limb >> bit) & 1 == 1 {
                    result = self.mul(&result, x);
                    started = true;
                }
            }
        }
        result
    }

    /// Whether $x = 0$, i.e. every coordinate of Eq. (7) is $0 \bmod q$.
    pub fn is_zero(&self, x: &Ext<K>) -> bool {
        x.0.iter().all(|c| self.zq.reduce(*c) == 0)
    }

    /// The Frobenius $x \mapsto x^q$, a signed permutation of the coordinates (module
    /// documentation). It generates $\mathrm{Gal}(\mathbb F_{q^K}/\mathbb F_q)$, of order $K$, as
    /// $\mathcal R_q^H \cong \mathbb F_{q^K}$ (Hachi, Lemma 5).
    pub fn frobenius(&self, x: &Ext<K>) -> Ext<K> {
        let mut out = [0; K];
        for (m, (target, negate)) in self.frobenius.iter().enumerate() {
            let c = self.zq.reduce(x.0[m]);
            out[*target] = if *negate { self.zq.neg(c) } else { c };
        }
        Ext(out)
    }

    /// $x^{-1}$, or `None` for $x = 0$, from the norm (module documentation); every nonzero
    /// element is invertible because $\mathcal R_q^H$ is a field (Hachi, Lemma 5).
    pub fn inv(&self, x: &Ext<K>) -> Option<Ext<K>> {
        if self.is_zero(x) {
            return None;
        }
        // r = phi(x) phi^2(x) ... phi^{K-1}(x).
        let mut conjugate = *x;
        let mut r = self.one();
        for _ in 1..K {
            conjugate = self.frobenius(&conjugate);
            r = self.mul(&r, &conjugate);
        }
        let norm = self.mul(x, &r);
        debug_assert!(norm.0[1..].iter().all(|c| *c == 0), "the norm is in F_q");
        let norm_inv = self.zq.inv(norm.0[0])?;
        Some(self.scale(&r, norm_inv))
    }
}
