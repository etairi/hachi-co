//! The arithmetic of the provers' inner loops: [`Arith`] abstracts $\mathbb F\_{q^k}$ in a working
//! representation, with two implementations, the canonical one of `hachi_co_math::Fqk` and a
//! Montgomery form ([`Montgomery`]) with lazy reduction, which avoids the 128-bit divisions of
//! the canonical product (the provers' rounds are 4 to 5 times faster with it at the moduli of
//! the development presets, `tests/scaling.rs`). Both give exactly the same field elements; the
//! tables of the provers stay canonical and only the sums of a round are formed in the working
//! representation.

use hachi_co_math::{Ext, Fqk};

/// $\mathbb F\_{q^K}$ in a working representation $\phi$, an additive bijection of
/// $\mathbb F\_{q^K}$ (the identity, or $x \mapsto xR$ for Montgomery).
///
/// `mul` multiplies two elements in the working representation into the working
/// representation, and an element in the working representation by a canonical one into a
/// canonical one: $\phi(x) \cdot \phi(y) \mapsto \phi(xy)$ and $\phi(x) \cdot y \mapsto xy$ (for
/// the identity both are the plain product). Additions work in either representation.
pub(crate) trait Arith<const K: usize> {
    /// $\phi(x)$ for a canonical $x$.
    fn enter(&self, x: &Ext<K>) -> Ext<K>;
    /// $\phi(c)$ for a base-field constant $c$, as a residue.
    fn enter_base(&self, c: u64) -> u64;
    /// $x$ from $\phi(x)$.
    fn leave(&self, x: &Ext<K>) -> Ext<K>;
    /// The product (see the trait documentation).
    fn mul(&self, x: &Ext<K>, y: &Ext<K>) -> Ext<K>;
    /// $\phi(x) \cdot c \mapsto x c$ for a canonical base-field $c$.
    fn scale_out(&self, x: &Ext<K>, c: u64) -> Ext<K>;
    /// $x + y$.
    fn add(&self, x: &Ext<K>, y: &Ext<K>) -> Ext<K>;
    /// $x - y$.
    fn sub(&self, x: &Ext<K>, y: &Ext<K>) -> Ext<K>;
    /// $x - c$ for a base-field $c$ in the same representation as $x$.
    fn sub_base(&self, x: &Ext<K>, c: u64) -> Ext<K>;
}

impl<const K: usize> Arith<K> for Fqk<K> {
    fn enter(&self, x: &Ext<K>) -> Ext<K> {
        self.reduce(x)
    }

    fn enter_base(&self, c: u64) -> u64 {
        self.zq().reduce(c)
    }

    fn leave(&self, x: &Ext<K>) -> Ext<K> {
        self.reduce(x)
    }

    fn mul(&self, x: &Ext<K>, y: &Ext<K>) -> Ext<K> {
        Fqk::mul(self, x, y)
    }

    fn scale_out(&self, x: &Ext<K>, c: u64) -> Ext<K> {
        self.scale(x, c)
    }

    fn add(&self, x: &Ext<K>, y: &Ext<K>) -> Ext<K> {
        Fqk::add(self, x, y)
    }

    fn sub(&self, x: &Ext<K>, y: &Ext<K>) -> Ext<K> {
        Fqk::sub(self, x, y)
    }

    fn sub_base(&self, x: &Ext<K>, c: u64) -> Ext<K> {
        crate::tables::sub_base(self, x, c)
    }
}

/// $\mathbb F\_{q^K}$ in Montgomery form, $\phi(x) = xR \bmod q$ coordinatewise with $R = 2^{64}$,
/// for $K \in \lbrace 1, 2, 4, 8 \rbrace$ and an odd $q$ with $\max(3K - 3, 1) \cdot q \lt 2^{64}$
/// (all $q \lt 2^{59}$).
///
/// **Product.** In Hachi's basis (`hachi_co_math::fqk`: $e\_a e\_b = e\_{a+b} + e\_{\lvert a-b
/// \rvert}$, $e\_0 = 2$, $e\_K = 0$, $e\_{2K-m} = -e\_m$) coordinate $m$ of $xy$ is a sum of at most
/// $2K - 1$ products $x\_a y\_b$ (counting the doubled ones) minus at most $K - 2$ others. The
/// products are accumulated in `u128` without reduction, the negative part is replaced by
/// $(K - 2) q^2$ minus it, and each coordinate is reduced once by Montgomery's REDC, which maps
/// $T \lt q 2^{64}$ to $T R^{-1} \bmod q$: the bound above keeps $T \lt (3K - 3) q^2 \lt q 2^{64}$.
/// With $\phi(x) = xR$ and $\phi(y) = yR$ the result is $xyR = \phi(xy)$; with a canonical $y$ it
/// is $xy$.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Montgomery<const K: usize> {
    q: u64,
    /// $-q^{-1} \bmod 2^{64}$.
    q_neg_inv: u64,
    /// $R^2 \bmod q$.
    r2: u64,
    /// $(K - 2) q^2$ (zero for $K \le 2$), the offset of the negative terms.
    offset: u128,
}

impl<const K: usize> Montgomery<K> {
    /// The Montgomery form for the modulus $q$, or `None` if $q$ is even, $q \lt 3$,
    /// $q \ge 2^{63}$, $K$ is not a power of two up to 8, or $\max(3K - 3, 1) \cdot q \ge 2^{64}$.
    pub(crate) fn new(q: u64) -> Option<Self> {
        // q < 2^63 keeps t + m q below 2^128 in REDC.
        if !matches!(K, 1 | 2 | 4 | 8) || q < 3 || q.is_multiple_of(2) || q >= 1 << 63 {
            return None;
        }
        let units = (3 * K as u128 - 3).max(1);
        if units * u128::from(q) >= 1u128 << 64 {
            return None;
        }
        // q^{-1} mod 2^64 by Newton's iteration (q odd): each step doubles the correct bits.
        let mut inv = 1u64;
        for _ in 0..7 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(q.wrapping_mul(inv)));
        }
        let r = ((1u128 << 64) % u128::from(q)) as u64;
        let r2 = (u128::from(r) * u128::from(r) % u128::from(q)) as u64;
        let negatives = K.saturating_sub(2) as u128;
        Some(Self {
            q,
            q_neg_inv: inv.wrapping_neg(),
            r2,
            offset: negatives * u128::from(q) * u128::from(q),
        })
    }

    /// REDC: $T R^{-1} \bmod q$ for $T \lt q 2^{64}$.
    #[inline(always)]
    fn redc(&self, t: u128) -> u64 {
        let m = (t as u64).wrapping_mul(self.q_neg_inv);
        // t + m q < q 2^64 + 2^64 q <= 2^128 as q < 2^63, and it is divisible by 2^64.
        let s = t + u128::from(m) * u128::from(self.q);
        let r = (s >> 64) as u64;
        if r >= self.q { r - self.q } else { r }
    }

    #[inline(always)]
    fn reduce(&self, x: u64) -> u64 {
        if x < self.q { x } else { x % self.q }
    }
}

impl<const K: usize> Arith<K> for Montgomery<K> {
    fn enter(&self, x: &Ext<K>) -> Ext<K> {
        Ext(x.0.map(|c| self.enter_base(c)))
    }

    fn enter_base(&self, c: u64) -> u64 {
        self.redc(u128::from(self.reduce(c)) * u128::from(self.r2))
    }

    fn leave(&self, x: &Ext<K>) -> Ext<K> {
        Ext(x.0.map(|c| self.redc(u128::from(self.reduce(c)))))
    }

    #[inline(always)]
    fn mul(&self, x: &Ext<K>, y: &Ext<K>) -> Ext<K> {
        let (x, y) = (&x.0, &y.0);
        let mut pos = [0u128; K];
        let mut neg = [0u128; K];
        for a in 0..K {
            for b in 0..K {
                let p = u128::from(x[a]) * u128::from(y[b]);
                if a == 0 || b == 0 {
                    pos[a + b] += p;
                    continue;
                }
                let s = a + b;
                if s < K {
                    pos[s] += p;
                } else if s > K {
                    neg[2 * K - s] += p;
                }
                if a == b {
                    pos[0] += 2 * p;
                } else {
                    pos[a.abs_diff(b)] += p;
                }
            }
        }
        Ext(core::array::from_fn(|m| {
            self.redc(pos[m] + (self.offset - neg[m]))
        }))
    }

    fn scale_out(&self, x: &Ext<K>, c: u64) -> Ext<K> {
        let c = u128::from(self.reduce(c));
        Ext(x.0.map(|v| self.redc(u128::from(v) * c)))
    }

    #[inline(always)]
    fn add(&self, x: &Ext<K>, y: &Ext<K>) -> Ext<K> {
        Ext(core::array::from_fn(|m| {
            let s = x.0[m] + y.0[m];
            if s >= self.q { s - self.q } else { s }
        }))
    }

    #[inline(always)]
    fn sub(&self, x: &Ext<K>, y: &Ext<K>) -> Ext<K> {
        Ext(core::array::from_fn(|m| {
            let (a, b) = (x.0[m], y.0[m]);
            if a >= b { a - b } else { a + (self.q - b) }
        }))
    }

    #[inline(always)]
    fn sub_base(&self, x: &Ext<K>, c: u64) -> Ext<K> {
        let mut out = *x;
        let a = out.0[0];
        out.0[0] = if a >= c { a - c } else { a + (self.q - c) };
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hachi_co_math::zq::is_prime;

    /// The largest prime p = 5 mod 8 below `x`.
    fn prime_below(x: u64) -> u64 {
        let mut p = x - (x % 8) + 5;
        while p >= x || !is_prime(p) {
            p -= 8;
        }
        p
    }

    struct SplitMix(u64);
    impl SplitMix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
    }

    /// Every operation against `Fqk` on random and boundary operands (0, 1, q - 1, (q - 1)/2),
    /// for every K at the toy moduli, at 2^37 and at the largest admissible prime for K.
    fn against_fqk<const K: usize>() {
        let bound = (1u128 << 64) / (3 * K as u128 - 3).max(1);
        let largest = prime_below(u64::try_from(bound.min(1 << 63)).unwrap());
        assert!(Montgomery::<K>::new(largest).is_some());
        let mut rng = SplitMix(K as u64);
        for q in [8388637u64, 33554501, 134217757, 137438953501, largest] {
            let f = Fqk::<K>::new(q).unwrap();
            let mt = Montgomery::<K>::new(q).unwrap();
            let special = [0, 1, q - 1, (q - 1) / 2, q / 3];
            let sample = |rng: &mut SplitMix, i: usize| -> Ext<K> {
                Ext(core::array::from_fn(|c| {
                    if i.is_multiple_of(3) {
                        special[(i / 3 + c) % special.len()]
                    } else {
                        rng.next() % q
                    }
                }))
            };
            for i in 0..600 {
                let (x, y) = (sample(&mut rng, i), sample(&mut rng, i + 1));
                let (mx, my) = (mt.enter(&x), mt.enter(&y));
                assert_eq!(mt.leave(&mx), x, "round trip, q = {q}");
                assert_eq!(
                    mt.leave(&mt.mul(&mx, &my)),
                    f.mul(&x, &y),
                    "mont x mont, q = {q}"
                );
                assert_eq!(mt.mul(&mx, &y), f.mul(&x, &y), "mont x canonical, q = {q}");
                assert_eq!(mt.leave(&mt.add(&mx, &my)), f.add(&x, &y));
                assert_eq!(mt.leave(&mt.sub(&mx, &my)), f.sub(&x, &y));
                let c = special[i % special.len()];
                assert_eq!(mt.scale_out(&mx, c), f.scale(&x, c));
                assert_eq!(
                    mt.leave(&mt.sub_base(&mx, mt.enter_base(c))),
                    crate::tables::sub_base(&f, &x, c)
                );
                // The identity representation is the plain field.
                assert_eq!(Arith::mul(&f, &x, &y), f.mul(&x, &y));
            }
        }
        // Moduli beyond the lazy-reduction bound, and even moduli, are refused.
        if K > 1 {
            assert!(Montgomery::<K>::new(prime_below(1 << 63)).is_none());
        }
        assert!(Montgomery::<K>::new(1 << 20).is_none());
    }

    #[test]
    fn montgomery_matches_the_canonical_field() {
        against_fqk::<1>();
        against_fqk::<2>();
        against_fqk::<4>();
        against_fqk::<8>();
    }
}
