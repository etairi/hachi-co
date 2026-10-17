//! Exact CRT from residues modulo NTT primes straight to a residue modulo $q < 2^{63}$.
//!
//! Ported from jali (`src/math/ring.rs`, `Garner` and `SmallCrt`; MIT, by the same author). The
//! reconstruction is jali's; the constants, which jali derives with 576-bit integers, are derived
//! here in native arithmetic (each step is stated in the documentation of `Garner`), and the
//! tests compare both the constants and the reconstruction with a big-integer CRT.
//!
//! The integer $X$ represented by residues $r_i = X \bmod p_i$ is taken centered, in
//! $(-P/2, P/2)$ with $P = \prod_i p_i$, so that sums of products of centered representatives
//! are recovered exactly as long as their absolute value stays below $P/2$
//! ([`crate::rns`]).

use crate::{
    arith::{add_mod, is_prime, mul_mod, pow_mod, reduce, shoup, shoup_mul, sub_mod},
    error::{Error, Result},
};

/// The largest number of primes a CRT accepts, as in jali (`src/math/ring.rs`, `Garner`).
pub const MAX_PRIMES: usize = 9;

/// Garner's mixed-radix digits and their sign, in native arithmetic.
///
/// With $M_j = p_0 \cdots p_{j-1}$ and $P = M_k$, every $X \in [0, P)$ is $\sum_j v_j M_j$ with
/// mixed-radix digits $v_j \in [0, p_j)$, and $v_i = (r_i - \sum_{j<i} v_j M_j) M_i^{-1} \bmod p_i$
/// from its residues $r_i$. Digit vectors compare as the integers do, most significant first,
/// which decides whether $X > \lfloor P/2 \rfloor$, that is whether the centered value $X - P$ is
/// meant. Every product is a Shoup product, so no step divides.
///
/// The digits of $\lfloor P/2 \rfloor$ are found without big integers: $P$ is odd, so
/// $\lfloor P/2 \rfloor = (P-1)/2$, and $P - 1 = \sum_j (p_j - 1) M_j$ has the digits
/// $p_j - 1$; halving digit by digit from the top with carry $c_{j+1} \in \lbrace 0, 1 \rbrace$
/// gives $h_j = \lfloor (c_{j+1} p_j + p_j - 1)/2 \rfloor < p_j$ and
/// $c_j = (c_{j+1} p_j + p_j - 1) \bmod 2$, and $c_0 = 0$ because $P - 1$ is even.
#[derive(Clone, Debug)]
pub(crate) struct Garner {
    primes: Vec<u64>,
    /// For each prime $p_i$: $M_j \bmod p_i$ for $j < i$, with Shoup constants.
    prefixes: Vec<Vec<(u64, u64)>>,
    /// $M_i^{-1} \bmod p_i$, with Shoup constants.
    inverses: Vec<(u64, u64)>,
    /// The mixed-radix digits of $\lfloor P/2 \rfloor$.
    half: Vec<u64>,
}

impl Garner {
    /// The constants for $1$ to [`MAX_PRIMES`] distinct odd primes $p_i < 2^{62}$.
    fn new(primes: &[u64]) -> Result<Self> {
        if primes.is_empty() || primes.len() > MAX_PRIMES {
            return Err(Error::Parameter("number of CRT primes"));
        }
        for (i, p) in primes.iter().enumerate() {
            if *p < 3 || *p >= 1 << 62 || !is_prime(*p) || primes[..i].contains(p) {
                return Err(Error::Parameter(
                    "CRT primes must be distinct odd primes below 2^62",
                ));
            }
        }
        let k = primes.len();
        // M_j mod m for every j < len, by running products.
        let prefixes_mod = |m: u64, len: usize| -> Vec<u64> {
            let mut out = Vec::with_capacity(len);
            let mut acc = 1 % m;
            for p in &primes[..len] {
                out.push(acc);
                acc = mul_mod(acc, *p, m);
            }
            out
        };
        let mut prefixes = Vec::with_capacity(k);
        let mut inverses = Vec::with_capacity(k);
        for (i, &p) in primes.iter().enumerate() {
            let all = prefixes_mod(p, i + 1);
            prefixes.push(all[..i].iter().map(|m| (*m, shoup(*m, p))).collect());
            // M_i mod p_i is a product of primes other than p_i, hence invertible.
            let inverse = pow_mod(all[i], p - 2, p);
            inverses.push((inverse, shoup(inverse, p)));
        }
        let mut half = vec![0u64; k];
        let mut carry = 0u128;
        for j in (0..k).rev() {
            let p = u128::from(primes[j]);
            let current = carry * p + (p - 1);
            half[j] = (current / 2) as u64;
            carry = current % 2;
        }
        debug_assert_eq!(carry, 0, "P - 1 is even");
        Ok(Self {
            primes: primes.to_vec(),
            prefixes,
            inverses,
            half,
        })
    }

    /// The digits of the integer $X \in [0, P)$ whose residue modulo $p_i$ is `residue(i)`, read
    /// modulo $p_i$, and whether $X > \lfloor P/2 \rfloor$.
    #[inline(always)]
    fn digits(&self, residue: impl Fn(usize) -> u64, digits: &mut [u64; MAX_PRIMES]) -> bool {
        for (i, p) in self.primes.iter().enumerate() {
            let mut sum = 0;
            for (v, (w, w_shoup)) in digits[..i].iter().zip(&self.prefixes[i]) {
                sum = add_mod(sum, shoup_mul(*v, *w, *w_shoup, *p), *p);
            }
            let (w, w_shoup) = self.inverses[i];
            digits[i] = shoup_mul(sub_mod(reduce(residue(i), *p), sum, *p), w, w_shoup, *p);
        }
        for (v, h) in digits[..self.primes.len()].iter().zip(&self.half).rev() {
            if v != h {
                return v > h;
            }
        }
        false
    }
}

/// The CRT for one modulus $2 \le q < 2^{63}$ (jali, `src/math/ring.rs`, `SmallCrt`): the
/// mixed-radix digits $v_j$ of Garner's algorithm, then $X \bmod q = \sum_j v_j (M_j \bmod q)
/// \bmod q$ by Shoup products modulo $q$ (which need $q < 2^{63}$), less $P \bmod q$ for a
/// centered value below zero.
#[derive(Clone, Debug)]
pub struct SmallCrt {
    garner: Garner,
    q: u64,
    /// $M_j \bmod q$, with Shoup constants modulo $q$.
    prefixes_q: Vec<(u64, u64)>,
    /// $P \bmod q$.
    product_q: u64,
}

impl SmallCrt {
    /// The CRT constants (jali, `SmallCrt::new`) for a modulus $2 \le q < 2^{63}$ (prime or not)
    /// and $1$ to [`MAX_PRIMES`] distinct odd primes $p_i < 2^{62}$.
    ///
    /// # Errors
    /// [`Error::Parameter`] if $q$ or the primes violate these conditions.
    pub fn new(q: u64, primes: &[u64]) -> Result<Self> {
        if !(2..1 << 63).contains(&q) {
            return Err(Error::Parameter("CRT modulus must be in [2, 2^63)"));
        }
        let garner = Garner::new(primes)?;
        let mut prefixes_q = Vec::with_capacity(primes.len());
        let mut acc = 1 % q;
        for p in primes {
            prefixes_q.push((acc, shoup(acc, q)));
            acc = mul_mod(acc, *p, q);
        }
        Ok(Self {
            garner,
            q,
            prefixes_q,
            product_q: acc,
        })
    }

    /// The modulus $q$ of the result (jali, `SmallCrt::modulus`).
    pub fn modulus(&self) -> u64 {
        self.q
    }

    /// The primes $p_0, \dots, p_{k-1}$ (jali, `src/math/ring.rs`, `Garner`).
    pub fn primes(&self) -> &[u64] {
        &self.garner.primes
    }

    /// $X \bmod q$ for the centered integer $X \in (-P/2, P/2)$ with $X \equiv r_i \pmod{p_i}$,
    /// one residue $r_i$ per prime (read modulo $p_i$); jali, `SmallCrt::reconstruct`.
    ///
    /// # Errors
    /// [`Error::Length`] unless there is one residue per prime.
    pub fn reconstruct(&self, residues: &[u64]) -> Result<u64> {
        if residues.len() != self.garner.primes.len() {
            return Err(Error::Length {
                expected: self.garner.primes.len(),
                actual: residues.len(),
            });
        }
        Ok(self.reconstruct_with(|i| residues[i]))
    }

    /// [`SmallCrt::reconstruct`] of coefficient `j` of residue vectors `residues[i]`, one per
    /// prime.
    #[inline]
    pub(crate) fn reconstruct_column(&self, residues: &[Vec<u64>], j: usize) -> u64 {
        debug_assert_eq!(residues.len(), self.garner.primes.len());
        self.reconstruct_with(|i| residues[i][j])
    }

    #[inline(always)]
    fn reconstruct_with(&self, residue: impl Fn(usize) -> u64) -> u64 {
        let mut digits = [0u64; MAX_PRIMES];
        let negative = self.garner.digits(residue, &mut digits);
        let mut x = 0;
        for (v, (w, w_shoup)) in digits.iter().zip(&self.prefixes_q) {
            x = add_mod(x, shoup_mul(*v, *w, *w_shoup, self.q), self.q);
        }
        if negative {
            sub_mod(x, self.product_q, self.q)
        } else {
            x
        }
    }
}

#[cfg(test)]
mod tests {
    //! The CRT against a num-bigint CRT at $0$, $1$, $\lfloor P/2 \rfloor$ and its neighbours,
    //! $P - 2$, $P - 1$ and random values, for one to nine primes and moduli from 2 to
    //! $2^{63} - 1$ (ported from jali's `rns/crt_tests.rs`); Garner's constants against their
    //! big-integer definitions.
    use super::*;
    use crate::{arith::tests::SplitMix, ntt::NTT_PRIMES};
    use num_bigint::BigUint;

    fn below(rng: &mut SplitMix, n: &BigUint) -> BigUint {
        let words: Vec<u32> = (0..20)
            .flat_map(|_| {
                let w = rng.next();
                [w as u32, (w >> 32) as u32]
            })
            .collect();
        BigUint::from_slice(&words) % n
    }

    const MODULI: [u64; 11] = [
        2,
        3,
        12,
        13,
        4294967197,
        4503599627370101,
        2305843009213693693,
        (1 << 62) - 1,
        9223372036854775549,
        9223372036854775783,
        (1 << 63) - 1,
    ];

    fn integers(product: &BigUint, rng: &mut SplitMix) -> Vec<BigUint> {
        let half = product >> 1u32;
        let one = BigUint::from(1u32);
        let mut out = vec![
            BigUint::from(0u32),
            one.clone(),
            &half - &one,
            half.clone(),
            &half + &one,
            product - 2u32,
            product - &one,
        ];
        out.extend((0..40).map(|_| below(rng, product)));
        out
    }

    /// $X$ up to $\lfloor P/2 \rfloor$, $X - P$ above, reduced modulo $q$.
    fn centered_mod(x: &BigUint, product: &BigUint, q: u64) -> u64 {
        let q = BigUint::from(q);
        let r = if x > &(product >> 1u32) {
            (&q - ((product - x) % &q)) % &q
        } else {
            x % &q
        };
        r.to_u64_digits().first().copied().unwrap_or(0)
    }

    fn to_u64(x: &BigUint) -> u64 {
        x.to_u64_digits().first().copied().unwrap_or(0)
    }

    #[test]
    fn crt_equals_a_big_integer_crt_for_one_to_nine_primes() {
        let mut rng = SplitMix(8);
        for k in 1..=MAX_PRIMES {
            let primes: Vec<u64> = NTT_PRIMES[..k].iter().map(|(p, _)| *p).collect();
            let product: BigUint = primes.iter().map(|p| BigUint::from(*p)).product();
            for q in MODULI {
                let crt = SmallCrt::new(q, &primes).unwrap();
                assert_eq!(crt.modulus(), q);
                assert_eq!(crt.primes(), &primes[..]);
                let xs = integers(&product, &mut rng);
                let columns: Vec<Vec<u64>> = primes
                    .iter()
                    .map(|p| xs.iter().map(|x| to_u64(&(x % p))).collect())
                    .collect();
                for (j, x) in xs.iter().enumerate() {
                    let want = centered_mod(x, &product, q);
                    let residues: Vec<u64> = columns.iter().map(|c| c[j]).collect();
                    assert_eq!(
                        crt.reconstruct(&residues).unwrap(),
                        want,
                        "{k} primes, q {q}, X {x}"
                    );
                    assert_eq!(crt.reconstruct_column(&columns, j), want);
                    // Residues are read modulo p_i: adding p_i where it fits changes nothing.
                    let shifted: Vec<u64> = residues
                        .iter()
                        .zip(&primes)
                        .map(|(r, p)| r.checked_add(*p).unwrap_or(*r))
                        .collect();
                    assert_eq!(crt.reconstruct(&shifted).unwrap(), want);
                }
            }
        }
    }

    #[test]
    fn garner_constants_equal_their_definitions() {
        for k in 1..=MAX_PRIMES {
            let primes: Vec<u64> = NTT_PRIMES[..k].iter().map(|(p, _)| *p).collect();
            let garner = Garner::new(&primes).unwrap();
            let mut prefix = BigUint::from(1u32);
            let mut prefixes = Vec::new();
            for p in &primes {
                prefixes.push(prefix.clone());
                prefix *= *p;
            }
            let product = prefix;
            let half = &product >> 1u32;
            for (i, p) in primes.iter().enumerate() {
                for (j, (w, w_shoup)) in garner.prefixes[i].iter().enumerate() {
                    assert_eq!(*w, to_u64(&(&prefixes[j] % *p)));
                    assert_eq!(*w_shoup, shoup(*w, *p));
                }
                let m = to_u64(&(&prefixes[i] % *p));
                assert_eq!(mul_mod(garner.inverses[i].0, m, *p), 1);
                // Digit i of floor(P/2) = floor(half / M_i) mod p_i.
                assert_eq!(garner.half[i], to_u64(&((&half / &prefixes[i]) % *p)));
            }
        }
    }

    #[test]
    fn invalid_crt_parameters_are_rejected() {
        let p: Vec<u64> = NTT_PRIMES.iter().map(|(p, _)| *p).collect();
        assert!(SmallCrt::new(13, &[]).is_err());
        let mut ten = p.clone();
        ten.push(4294967197);
        assert!(SmallCrt::new(13, &ten).is_err());
        assert!(SmallCrt::new(13, &[p[0], p[0]]).is_err());
        assert!(
            SmallCrt::new(13, &[p[0], p[0] - 8192]).is_err(),
            "composite"
        );
        assert!(SmallCrt::new(13, &[2]).is_err(), "even prime");
        assert!(
            SmallCrt::new(13, &[4611686018427494401]).is_err(),
            "prime at least 2^62"
        );
        for q in [0, 1, 1 << 63, u64::MAX] {
            assert!(SmallCrt::new(q, &p[..2]).is_err(), "q = {q}");
        }
        let crt = SmallCrt::new(13, &p[..2]).unwrap();
        assert_eq!(
            crt.reconstruct(&[1]),
            Err(Error::Length {
                expected: 2,
                actual: 1
            })
        );
    }
}
