//! Sparse challenge polynomials $c \in \mathcal C \subset \mathbf R_q$ (paper, Section 4.2 for the
//! challenge space, Section 5.4 for the sampling and the sparse products).
//!
//! The challenge space of the prototype is the set of ring elements with *exactly* $k$ nonzero
//! coefficients, each equal to $1$ or $-1$, where $k$ is `params.k`, the paper's $\omega$ (not
//! the extension degree):
//! $$\mathcal C = \Bigl\lbrace c = \sum_{i \lt k} \varepsilon_i X^{e_i} : 0 \le e_0 \lt e_1 \lt \dots \lt e_{k-1} \lt d, \quad \varepsilon_i \in \lbrace -1, 1 \rbrace \Bigr\rbrace,$$
//! so that $\lVert c \rVert_1 = k$ and $\lVert c \rVert_\infty = 1$: a subset of the paper's
//! $\lbrace c \in \mathbf R_q : \lVert c \rVert_1 \le \omega \rbrace$ with $\omega = k$
//! (Section 4.2). Its size is $\binom{d}{k} 2^k$, about $2^{131.6}$ for the default $d = 1024$,
//! $k = 16$ (Section 5.4). Two distinct challenges differ by an element of $\ell_\infty$ norm at
//! most $2$, which is invertible in $\mathbf R_q$ by Lemma 3 of Section 2.1 because
//! $q \equiv 5 \pmod 8$; the paper's extraction argument (Lemma 8, Section 4.2) assumes
//! $\omega \lt q^{1/2} / (2\sqrt 2)$ (so that the difference of two challenges, of $\ell_1$ norm at most $2\omega$, is invertible by Lemma 3).
//!
//! ## Encoding
//!
//! A challenge is stored sparsely, as its $k$ nonzero coefficients only: a `Vec<u32>` in which
//! the entry `(e << 1) | s` encodes the coefficient $\varepsilon X^e$, with sign bit `s = 1` for
//! $\varepsilon = 1$ and `s = 0` for $\varepsilon = -1$. Entries are sorted increasingly, hence by
//! exponent (exponents are distinct), which makes the shifted additions of the sparse products
//! ([`crate::arithmetic::ring::Ring::chal_mul_poly`]) sweep memory in order. A challenge is
//! never expanded to dense form: multiplying by it costs $k d$ additions instead of an NTT
//! (paper, Section 5.4).

use ark_ff::AdditiveGroup;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;

use crate::arithmetic::{ExtField, utils::{Logarithm, rand_int}};

/// A sparse challenge $c \in \mathcal C$: exactly $k$ coefficients equal to $\pm 1$, all others
/// zero, stored as the packed `(exponent << 1) | sign` entries described in the module
/// documentation.
pub struct PChal {
    /// The $k$ nonzero coefficients as `(exponent << 1) | sign`, sorted increasingly (sign bit
    /// `1` means $+1$, `0` means $-1$).
    coeffs: Vec<u32>,
}

impl PChal {
    /// Sample a uniformly random challenge of $\mathcal C$ for ring dimension `d` and sparsity
    /// `k` ($k \le d$) from `rng`, by the partial Fisher-Yates shuffle of the paper's
    /// Section 5.4. Starting from `arr = [0, 1, ..., d - 1]`, step $i$ ($i \lt k$) draws a
    /// uniform integer in $\[0, 2(d - i))$ by rejection sampling ([`rand_int`]); its high bits
    /// select a uniformly random not-yet-struck position `i + index` in `arr[i..d]` and its low
    /// bit is the sign. The selected exponent, packed with its sign, is moved to position $i$
    /// and the value it displaces takes its old place (the swap back is skipped at the last step,
    /// where nothing reads it). The first $k$ entries are then sorted. The $k$ exponents thus form
    /// a uniformly random $k$-subset of $\lbrace 0, \dots, d - 1 \rbrace$ and the signs are
    /// independent uniform bits, so the output is uniform over $\mathcal C$. Cost: $O(d)$ to
    /// build the array plus $O(k \log k)$ to sort. Exponents must fit in 31 bits.
    pub fn rand(d: usize, k: usize, rng: &mut impl Rng) -> Self {
        // create array [0, 1, 2, ..., d - 1]
        let mut arr = vec![0u32; d];

        for i in 1..d {
            arr[i] = i as u32;
        }

        // iterate from 0 to k-1
        for i in 0..k {
            // generate a random number between 0 and 2(d-i)-1
            let n = (d as u64 - i as u64) * 2;
            let val = rand_int(n, n.log(), rng) as u32;

            let index = (val >> 1) as usize;
            let sign = val & 1;

            // set the next output as index-th unstruck item
            let tmp = arr[i];
            arr[i] = (arr[i + index] << 1) | sign;

            // move the overwritten element into the unstruck set
            if i < k - 1 && index > 0 {
                arr[i + index] = tmp
            }
        }

        // sort the vector to help memory access when multiplying
        let mut v = arr[0..k].to_vec();
        v.sort();

        Self { coeffs : v }
    }

    /// Sample `num` independent challenges, in sequence, from a `ChaCha12Rng` seeded with `seed`.
    /// This is the Fiat-Shamir derivation of the challenge vector $(c_1, \dots, c_{2^r})$ of the
    /// paper's Fig. 3 from a transcript seed ([`crate::arithmetic::fs::FS::get_seed`]): prover
    /// and verifier obtain identical vectors from identical seeds.
    pub fn rand_vec(num: usize, d: usize, k: usize, seed: [u8; 32]) -> Vec<Self> {
        let mut vec: Vec<Self> = Vec::with_capacity(num);
        let mut rng = ChaCha12Rng::from_seed(seed);

        for _ in 0..num {
            vec.push(Self::rand(d, k, &mut rng));
        }

        vec
    }

    /// Build a challenge from explicit `(exponent, sign)` entries, in the given order and without
    /// sorting or checking distinctness (tests only). Sign: `false` means $-1$, `true` means $+1$.
    #[cfg(test)]
    pub fn from_entries(entries: &[(usize, bool)]) -> Self {
        Self { coeffs: entries.iter().map(|&(exp, sign)| ((exp as u32) << 1) | (sign as u32)).collect() }
    }

    /// The number $k$ of nonzero coefficients, i.e. $\lVert c \rVert_1$.
    pub fn k(&self) -> usize {
        self.coeffs.len()
    }

    /// The `(exponent, sign)` of the $i$-th nonzero coefficient of this challenge in increasing
    /// order of exponent, $0 \le i \lt k$; `sign == true` means $+1$ and `false` means $-1$.
    pub fn get(&self, i: usize) -> (usize, bool) {
        ((self.coeffs[i] >> 1) as usize, (self.coeffs[i] & 1) == 1)
    }

    /// Evaluate at $\alpha \in \mathbb F_{q^4}$: $c(\alpha) = \sum_{i \lt k} \varepsilon_i \alpha^{e_i}$,
    /// given the table `alpha_pows[j]` $= \alpha^j$ for $j \lt d$
    /// ([`crate::arithmetic::utils::powers`]). Costs $k$ additions or subtractions and no
    /// multiplication. Used when the challenges enter the lifted verification equations at the
    /// ring-switching point $\alpha$ (paper, Section 4.3).
    pub fn eval(&self, alpha_pows: &[ExtField]) -> ExtField { 
        let mut out = ExtField::ZERO;

        for i in 0..self.k() {
            let (exp, sign) = self.get(i);

            if sign {
                out += alpha_pows[exp];
            }
            else {
                out -= alpha_pows[exp];
            }
        }

        out
    }
}


#[cfg(test)]
/// Tests for Challenge Polynomial.
mod test_pchal {
    use rand::rng;

    use crate::arithmetic::{poly::Poly, utils::{powers, rand_field}};

    use super::*;

    #[test]
    fn test_rand() {
        // generate a random sparse poly
        let mut rng = rng();
        let p = PChal::rand(1024, 17, &mut rng);

        // check that p has distinct exponents and print the signs for manual inspection
        fn count(p: &PChal, exp: usize) -> usize {
            let mut n = 0;
            
            for i in 0..p.k() {
                let (exp_2,  _) = p.get(i);
                if exp == exp_2 { n += 1 ;}
            }

            n
        }

        let mut total_neg = 0;
        let mut total_pos = 0;

        for i in 0..p.k() {
            let (exp, sign) = p.get(i);
            assert_eq!(1, count(&p, exp));

            if sign {
                total_pos += 1;
            }
            else {
                total_neg += 1;
            }
        }

        println!("Total positive: {}. Total negative: {}", total_pos, total_neg);  
    }

    #[test]
    fn test_rand_vec() {
        // just test it generates the correct number of elements
        let num = 100;
        assert_eq!(num, PChal::rand_vec(num, 64, 40, [1u8; 32]).len());
    }

    #[test]
    fn test_k() {
        let poly = PChal::rand(1024, 17, &mut rng());
        assert_eq!(17, poly.k());

        let poly = PChal::rand(512, 32, &mut rng());
        assert_eq!(32, poly.k());
    }

    #[test]
    fn test_get() {
        let poly = PChal::rand(1024, 17, &mut rng());
        
        for i in 0..poly.k() {
            let (exp, _) = poly.get(i);
            assert!(exp < 1024);
        }
    }

    #[test]
    fn test_eval() {
        let d = 1024;
        let sparse_poly = PChal::rand(d, 17, &mut rng());
        let mut poly = vec![0u64; d];

        // store sparse poly as normal poly
        for i in 0..sparse_poly.k() {
            let (exp, sign) = sparse_poly.get(i);
            
            if sign {
                poly[exp] = 1;
            }
            else {
                poly[exp] = 4294967196; // -1 mod q;
            }
        }

        // get a random evaluation point
        let alpha = rand_field(4294967197, & mut rng());

        // generate the powers [1, alpha, ..., alpha^d-1] and calculate the correct evaluation
        let pows = powers(alpha, d);

        assert_eq!(poly.as_slice().eval(&pows), sparse_poly.eval(&pows));
    }
}