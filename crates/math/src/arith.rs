//! Word-level modular arithmetic shared by the transforms, the CRT and $\mathbb Z_q$.
//!
//! Ported from jali (`src/math/ntt.rs`, MIT, by the same author). Every function here has a
//! precondition on its operands, stated in its documentation; callers in this crate establish it,
//! and the tests compare each function with the `u128` remainder on boundary and random operands.

/// $ab \bmod p$ for any $a, b < 2^{64}$ and $p \ge 1$, by a `u128` product and remainder: the
/// reference product of the crate.
#[inline]
pub(crate) fn mul_mod(a: u64, b: u64, p: u64) -> u64 {
    (u128::from(a) * u128::from(b) % u128::from(p)) as u64
}

/// $a^e \bmod p$ by square and multiply, with [`mul_mod`]; $a^0 = 1$ (reduced, so $1 \bmod 1 = 0$
/// is not special-cased: callers pass $p \ge 2$).
#[inline]
pub(crate) fn pow_mod(mut a: u64, mut e: u64, p: u64) -> u64 {
    let mut r = 1 % p;
    a %= p;
    while e != 0 {
        if e & 1 != 0 {
            r = mul_mod(r, a, p);
        }
        a = mul_mod(a, a, p);
        e >>= 1;
    }
    r
}

/// The Shoup constant $\lfloor w 2^{64} / p \rfloor$ of a multiplier $w < p$.
#[inline]
pub(crate) fn shoup(w: u64, p: u64) -> u64 {
    ((u128::from(w) << 64) / u128::from(p)) as u64
}

/// $xw \bmod p$ in $[0, p)$ for any $x < 2^{64}$, given $w < p < 2^{63}$ and `w_shoup` $=$
/// [`shoup`]`(w, p)`.
///
/// Write $\tilde w = \lfloor w 2^{64}/p \rfloor = (w 2^{64} - e)/p$ with $0 \le e < p$. The
/// quotient estimate $k = \lfloor x \tilde w / 2^{64} \rfloor$ satisfies
/// $0 \le xw/p - k < xe/(p 2^{64}) + 1 < 2$, so $xw - kp \in [0, 2p)$. As $2p < 2^{64}$, the
/// wrapping computation below is exact, and one conditional subtraction reduces it.
#[inline(always)]
pub(crate) fn shoup_mul(x: u64, w: u64, w_shoup: u64, p: u64) -> u64 {
    let k = ((u128::from(x) * u128::from(w_shoup)) >> 64) as u64;
    let r = x.wrapping_mul(w).wrapping_sub(k.wrapping_mul(p));
    if r >= p { r - p } else { r }
}

/// $(a + b) \bmod p$ for $a, b < p < 2^{63}$.
#[inline(always)]
pub(crate) fn add_mod(a: u64, b: u64, p: u64) -> u64 {
    let s = a + b;
    if s >= p { s - p } else { s }
}

/// $(a - b) \bmod p$ for $a, b < p$.
#[inline(always)]
pub(crate) fn sub_mod(a: u64, b: u64, p: u64) -> u64 {
    if a >= b { a - b } else { a + (p - b) }
}

/// $x \bmod p$, with a single comparison when $x$ is already reduced.
#[inline(always)]
pub(crate) fn reduce(x: u64, p: u64) -> u64 {
    if x < p { x } else { x % p }
}

/// Reduction modulo an invariant $p$, $2 \le p < 2^{63}$, without a 128-bit division: the
/// products of $\mathbb Z_q$ and $\mathbb F_{q^K}$ take this path, which [`mul_mod`], the
/// reference, checks in the tests.
///
/// With $s$ the leading zeros of $p$, $s \ge 1$, the divisor $D = p 2^s$ has its top bit set, and
/// $v = \lfloor (2^{128} - 1)/D \rfloor - 2^{64}$ is its reciprocal. A number $U = u_1 2^{64} +
/// u_0$ with $u_1 < D$ is divided by Algorithm 4 of Möller and Granlund, *Improved division by
/// invariant integers* (IEEE Trans. Computers 60(2), 2011): two 64-bit products and two
/// conditional corrections. Then $x \bmod p = ((x 2^s) \bmod D)/2^s$.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Reducer {
    p: u64,
    /// $D = p 2^s$.
    divisor: u64,
    /// $v$.
    reciprocal: u64,
    /// $s$.
    shift: u32,
}

impl Reducer {
    /// The reducer of $2 \le p < 2^{63}$.
    pub(crate) fn new(p: u64) -> Self {
        debug_assert!((2..1 << 63).contains(&p));
        let shift = p.leading_zeros();
        let divisor = p << shift;
        Self {
            p,
            divisor,
            // The quotient lies in [2^64, 2^65): the cast takes off 2^64.
            reciprocal: (u128::MAX / u128::from(divisor)) as u64,
            shift,
        }
    }

    /// $U \bmod D$ for $U = u_1 2^{64} + u_0$ with $u_1 < D$.
    #[inline(always)]
    fn rem_2by1(&self, u1: u64, u0: u64) -> u64 {
        let d = self.divisor;
        // v u1 + U = u1 (v + 2^64) + u0 < 2^128, as u1 < D and v + 2^64 <= (2^128 - 1)/D.
        let quotient = u128::from(self.reciprocal) * u128::from(u1)
            + ((u128::from(u1) << 64) | u128::from(u0));
        let q1 = ((quotient >> 64) as u64).wrapping_add(1);
        let q0 = quotient as u64;
        let mut r = u0.wrapping_sub(q1.wrapping_mul(d));
        if r > q0 {
            r = r.wrapping_add(d);
        }
        if r >= d { r - d } else { r }
    }

    /// $x \bmod p$ for $x < p 2^{64}$, such as a product of residues.
    #[inline(always)]
    pub(crate) fn reduce_below(&self, x: u128) -> u64 {
        debug_assert!(x >> 64 < u128::from(self.p));
        let shifted = x << self.shift;
        self.rem_2by1((shifted >> 64) as u64, shifted as u64) >> self.shift
    }

    /// $x \bmod p$ for any `u128`.
    #[inline(always)]
    pub(crate) fn reduce(&self, x: u128) -> u64 {
        let high = (x >> 64) as u64;
        let high = if high < self.p {
            high
        } else {
            // shift >= 1, and high >> (64 - shift) < 2^shift <= D.
            let (u1, u0) = (high >> (64 - self.shift), high << self.shift);
            self.rem_2by1(u1, u0) >> self.shift
        };
        self.reduce_below((u128::from(high) << 64) | u128::from(x as u64))
    }

    /// $ab \bmod p$ for $a, b < p$.
    #[inline(always)]
    pub(crate) fn mul(&self, a: u64, b: u64) -> u64 {
        self.reduce_below(u128::from(a) * u128::from(b))
    }
}

/// Deterministic Miller–Rabin primality test, exact for every 64-bit integer.
///
/// Trial division by the primes up to 37, then the strong-probable-prime test to the seven
/// bases $2, 325, 9375, 28178, 450775, 9780504, 1795265022$ (Jim Sinclair's set), which no odd
/// composite below $2^{64}$ passes. Ported from jali (`src/math/int.rs`).
pub(crate) fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n.is_multiple_of(p) {
            return n == p;
        }
    }
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    'witness: for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022] {
        if a % n == 0 {
            continue;
        }
        let mut x = pow_mod(a % n, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 1..s {
            x = mul_mod(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }
    true
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A splitmix64 stream: the deterministic source of the unit tests (seed in each test).
    pub(crate) struct SplitMix(pub u64);
    impl SplitMix {
        pub fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
    }

    fn reference(x: u64, w: u64, p: u64) -> u64 {
        (u128::from(x) * u128::from(w) % u128::from(p)) as u64
    }

    /// Shoup products against remainders at small primes, at the NTT primes and at the three
    /// largest primes below $2^{63}$ (where the bound $2p < 2^{64}$ is tightest), for boundary
    /// and random multipliers and for every kind of $x < 2^{64}$, including $x \ge p$.
    #[test]
    fn shoup_products_equal_remainders() {
        let mut rng = SplitMix(2);
        let wide: Vec<u64> = (1..1000u64)
            .map(|k| (1u64 << 63) - k)
            .filter(|p| is_prime(*p))
            .take(3)
            .collect();
        assert_eq!(wide[0], (1 << 63) - 25);
        let mut moduli = vec![3u64, 5, 13, 4294967197];
        moduli.extend(crate::ntt::NTT_PRIMES.iter().map(|(p, _)| *p));
        moduli.extend(wide);
        for p in moduli {
            let mut values = vec![0, 1, 2, p / 2, p / 2 + 1, p - 2, p - 1];
            values.extend((0..40).map(|_| rng.next() % p));
            for &w in &values {
                let w_shoup = shoup(w, p);
                for x in values
                    .iter()
                    .copied()
                    .chain([p, p + 1, 2 * p - 1, u64::MAX - 1, u64::MAX])
                    .chain((0..8).map(|_| rng.next()))
                {
                    assert_eq!(
                        shoup_mul(x, w, w_shoup, p),
                        reference(x, w, p),
                        "p {p} x {x} w {w}"
                    );
                }
            }
        }
    }

    /// The reducer against the `u128` remainder: for $p$ from 2 up to the largest values below
    /// $2^{63}$, on every side of a power of two and of a multiple of $p$ (the corrections of
    /// the division), and on random operands of every size.
    #[test]
    fn the_reducer_equals_the_remainder() {
        let mut rng = SplitMix(5);
        let mut moduli: Vec<u64> = vec![2, 3, 4, 5, 7, 13, 1 << 31, (1 << 31) + 1, 4294967197];
        for bits in 2..=62u32 {
            moduli.extend([(1 << bits) - 1, 1 << bits, (1 << bits) + 1]);
        }
        moduli.extend([
            (1 << 63) - 1,
            (1 << 63) - 2,
            (1 << 63) - 25,
            (1 << 63) - 259,
        ]);
        moduli.extend(crate::ntt::NTT_PRIMES.iter().map(|(p, _)| *p));
        moduli.extend((0..200).map(|_| (rng.next() >> 1) | 1));
        moduli.extend((0..200).map(|_| (rng.next() >> (2 + rng.next() % 60)) | 2));
        for p in moduli {
            let r = Reducer::new(p);
            let wide = u128::from(p);
            let mut below: Vec<u128> = vec![0, 1, wide - 1, wide, wide + 1, wide * wide - 1];
            below.extend([(wide << 64) - 1, (wide << 64) - wide, (wide - 1) << 64]);
            let mut any: Vec<u128> = below.clone();
            any.extend([
                u128::MAX,
                u128::MAX - 1,
                u128::MAX - wide,
                1 << 127,
                (1 << 64) - 1,
            ]);
            any.extend([
                1 << 64,
                (u128::MAX / wide) * wide,
                (u128::MAX / wide) * wide - 1,
            ]);
            for _ in 0..200 {
                let x = (u128::from(rng.next()) << 64) | u128::from(rng.next());
                any.push(x);
                any.push(x >> (rng.next() % 128));
                below.push(x % (wide << 64));
                below.push((x >> 64) * u128::from(rng.next() % p) % (wide << 64));
            }
            for x in below {
                assert_eq!(
                    u128::from(r.reduce_below(x)),
                    x % wide,
                    "below, p {p} x {x}"
                );
            }
            for x in any {
                assert_eq!(u128::from(r.reduce(x)), x % wide, "any, p {p} x {x}");
            }
            for _ in 0..200 {
                let (a, b) = (rng.next() % p, rng.next() % p);
                assert_eq!(r.mul(a, b), reference(a, b, p), "p {p} a {a} b {b}");
            }
            assert_eq!(r.mul(p - 1, p - 1), reference(p - 1, p - 1, p));
        }
    }

    #[test]
    fn sums_differences_and_powers_equal_the_reference() {
        let mut rng = SplitMix(3);
        for p in [3u64, 5, 4294967197, (1 << 62) - 57, (1 << 63) - 25] {
            let mut values = vec![0, 1, 2, p / 2, p / 2 + 1, p - 2, p - 1];
            values.extend((0..30).map(|_| rng.next() % p));
            for &a in &values {
                for &b in &values {
                    let (a128, b128, p128) = (u128::from(a), u128::from(b), u128::from(p));
                    assert_eq!(u128::from(add_mod(a, b, p)), (a128 + b128) % p128);
                    assert_eq!(u128::from(sub_mod(a, b, p)), (a128 + p128 - b128) % p128);
                }
                // a^e by repeated multiplication for small e, and Fermat for large e.
                let mut acc = 1 % p;
                for e in 0..20u64 {
                    assert_eq!(pow_mod(a, e, p), acc, "p {p} a {a} e {e}");
                    acc = reference(acc, a, p);
                }
                if a != 0 && is_prime(p) {
                    assert_eq!(pow_mod(a, p - 1, p), 1);
                }
            }
            for x in [0, 1, p - 1, p, p + 1, 2 * p, u64::MAX] {
                assert_eq!(reduce(x, p), x % p);
            }
        }
    }

    /// Primality against a sieve of Eratosthenes below $2^{20}$, exhaustively, and against
    /// values whose status SageMath 10.9 `is_prime` confirmed: primes near $2^{32}$, $2^{52}$,
    /// $2^{61}$, $2^{62}$, $2^{63}$, $2^{64}$, and composites that are strong pseudoprimes to
    /// many bases.
    #[test]
    fn primality_equals_a_sieve_and_known_values() {
        let limit = 1usize << 20;
        let mut sieve = vec![true; limit];
        sieve[0] = false;
        sieve[1] = false;
        let mut i = 2;
        while i * i < limit {
            if sieve[i] {
                let mut j = i * i;
                while j < limit {
                    sieve[j] = false;
                    j += i;
                }
            }
            i += 1;
        }
        for (n, &prime) in sieve.iter().enumerate() {
            assert_eq!(is_prime(n as u64), prime, "n = {n}");
        }
        let primes = [
            4294967197u64,        // 2^32 - 99
            4503599627370101,     // 2^52 - 395
            2305843009213693693,  // 2^61 - 259
            2305843009213693951,  // 2^61 - 1
            4611686018427387733,  // 2^62 - 171
            9223372036854775549,  // 2^63 - 259
            9223372036854775783,  // 2^63 - 25
            18446744073709551557, // 2^64 - 59
            4611686018427322369,  // the first NTT prime
        ];
        for p in primes {
            assert!(is_prime(p), "{p} is prime");
        }
        let composites = [
            3215031751u64,           // 151 * 751 * 28351, strong pseudoprime to bases 2, 3, 5, 7
            3825123056546413051,     // 149491 * 747451 * 34233211, spsp to bases 2, 3, ..., 31
            2305843009213693953,     // 2^61 + 1 = 3 * 768614336404564651
            4294967197 * 4294967197, // (2^32 - 99)^2
            18446744073709551615,    // 2^64 - 1 = 3 * 5 * 17 * 257 * 641 * 65537 * 6700417
            9223372036854775807,     // 2^63 - 1 = 7^2 * 73 * 127 * 337 * 92737 * 649657
            561,
            1105,
            1729,
            2047,
            3277,
            4033,
            4681,
            8321,
        ];
        for n in composites {
            assert!(!is_prime(n), "{n} is composite");
        }
    }
}
