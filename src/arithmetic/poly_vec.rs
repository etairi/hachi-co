//! Vectors of polynomials over $\mathbb Z_q$ in coefficient form: the vectors over $\mathbf R_q$
//! of the paper (witness chunks $\mathbf f_i$, their digits $\mathbf s_i = \mathbf G^{-1}(\mathbf f_i)$,
//! the commitments $\mathbf t$, $\mathbf u$, $\mathbf v$, the response $\mathbf z$;
//! Section 4.1-4.2) and, in the prover's ring switching (Section 4.3), their lifts to
//! $\mathbb Z_q\[X\]$ of degree below $2d$. The gadget decomposition $\mathbf G^{-1}$ of
//! Section 2.1 is implemented here as well.
//!
//! ## Layout
//!
//! A [`PVec`] of `len` elements with `d` coefficients each is one flat `Vec<u64>` of length
//! `len * d`: the coefficient of $X^j$ of element $i$ is at index `i * d + j`. `d` must be a
//! power of two (element slicing shifts by $\log_2 d$); it is the ring dimension $d$ for elements
//! of $\mathbf R_q$ and $2d$ for the full products that [`crate::arithmetic::ring::Ring`]
//! produces in full mode. Two coefficient conventions exist (see [`crate::arithmetic`]):
//! residues in $\[0, q)$, produced by [`PVec::rand`] and by every reduced product, and wrapping
//! two's-complement short integers, produced by [`PVec::b_decomp`] and [`PVec::b_decomp_zq`]
//! and held by the response $\mathbf z$. Which one a vector holds is fixed by its producer; the
//! type does not record it.
//!
//! ## Decomposed layout
//!
//! For a base $b$ and $\delta$ digits, the decomposition of a vector of $n$ elements is a vector
//! of $n \delta$ elements in which digit $k$ of element $i$ is element $i \delta + k$. This is the
//! column order of the gadget matrix $\mathbf G_{b,n} = \mathbf I_n \otimes (1, b, \dots, b^{\delta - 1})$
//! of the paper's Section 2.1, so that $\mathbf G \cdot \mathbf G^{-1}(\mathbf t) = \mathbf t$ reads
//! $\sum_{k \lt \delta} b^k \cdot \mathrm{out}\[i \delta + k\] = \mathbf t\[i\]$ coefficient by
//! coefficient.

use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::arithmetic::{CoeffType, fs::Serialise, poly::Poly, utils::{Logarithm, b_decomp, rand_int, to_window}};

/// A vector of polynomials over $\mathbb Z_q$ in coefficient form, stored flat (see the module
/// documentation for the layout and the coefficient conventions).
pub struct PVec {
    /// Number of elements (polynomials).
    len: usize,
    /// Number of coefficients of each element, a power of two.
    d: usize,
    /// $\log_2$ of `d`, used to slice elements by shifting.
    logd: usize,
    /// The `len * d` coefficients, element after element.
    vec: Vec<CoeffType>
}

impl PVec {
    /// The zero vector of `len` elements of dimension `d` (a power of two, asserted).
    pub fn zero(len: usize, d: usize) -> Self {
        assert!(d.is_power_of_two());
        let logd = d.log();

        Self { len, d, logd, vec: vec![0; len * d] }
    }

    /// A vector of `len` elements whose `len * d` coefficients are independent uniform residues in
    /// $\[0, q)$, drawn from a `ChaCha12Rng` seeded with `seed` by rejection sampling
    /// ([`rand_int`]). The output is a deterministic function of the seed: this is how the public
    /// matrices $\mathbf A$, $\mathbf B$, $\mathbf D$ are expanded from their seeds
    /// ([`crate::arithmetic::poly_mat::PMat::rand`]). With a small bound in place of `q` (as the
    /// tests do) the coefficients are digits in $\[0, q)$, still read as unsigned.
    pub fn rand(len: usize, d: usize, q: CoeffType, seed: [u8; 32]) ->  Self {
        assert!(d.is_power_of_two());
        let logd = d.log();
        let logq = q.log();
        let mut rng = ChaCha12Rng::from_seed(seed);

        let vec: Vec<CoeffType> = (0..len * d).map(|_| rand_int(q, logq, &mut rng)).collect();

        Self { len, d, logd, vec }
    }

    /// Number of elements.
    pub fn length(&self) -> usize {
        self.len
    }

    /// Element $i$ as the slice of its `d` coefficients, the coefficient of $X^j$ at index $j$;
    /// panics if `i >= len`.
    pub fn element(&self, i: usize) -> &[CoeffType] {
        &self.vec[(i << self.logd)..((i + 1) << self.logd)]
    }

    /// Element $i$ as a mutable slice of its `d` coefficients; panics if `i >= len`.
    pub fn mut_element(&mut self, i: usize) -> &mut [CoeffType] {
        &mut self.vec[(i << self.logd)..((i + 1) << self.logd)]
    }

    /// All `len * d` coefficients, element after element (index `i * d + j`).
    pub fn slice(&self) -> &[CoeffType] {
        &self.vec
    }

    /// All `len * d` coefficients as a mutable slice, element after element.
    pub fn mut_slice(&mut self) -> &mut [CoeffType] {
        &mut self.vec
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Balanced gadget decomposition $\mathbf G^{-1}$ of a vector of *short*, wrapping-signed
    /// polynomials. Every coefficient $x$ is split by [`b_decomp`] into `delta` wrapping-signed
    /// digits $a_0, \dots, a_{\delta - 1} \in \[-b/2, b/2 - 1\]$ with $x = \sum_k a_k b^k$, and digit
    /// $k$ of element $i$ is stored as element $i \delta + k$ of `out` (module documentation).
    /// `base` must be a power of two (its logarithm is taken), and `out` must have `len * delta`
    /// elements (asserted) of the same dimension.
    ///
    /// The split is exact only for $x$ inside the balanced window
    /// $\[-\frac{b}{2} \cdot \frac{b^\delta - 1}{b - 1}, (\frac{b}{2} - 1) \cdot \frac{b^\delta - 1}{b - 1}\]$
    /// (see [`crate::arithmetic::utils::window_top`]); outside it the top carry is silently
    /// lost. The prover applies it to the response $\mathbf z$ with `delta = params.delta_z`
    /// ($\tau$ digits), whose coefficients are assumed to lie within `params.z_bound`, the top of
    /// that window for $b = 16$, $\tau = 4$ (paper, Section 4.2, the decomposition
    /// $\hat{\mathbf z}$ of $\mathbf z$ before Eq. (20)). For residues use
    /// [`b_decomp_zq`](PVec::b_decomp_zq). Cost: linear in `len * d * delta`.
    pub fn b_decomp(&self, base: u64, delta: usize, out: &mut Self) {
        self.decomp_mapped(|x| x, base, delta, out);
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Balanced gadget decomposition $\mathbf G^{-1}$ of a vector of residues in $\[0, q)$. Every
    /// coefficient $x$ is first replaced by its representative in the balanced window, $x$ itself
    /// if $x \le \mathrm{top}(b, \delta)$ and $x - q$ otherwise ([`to_window`]), and then split
    /// into `delta` balanced digits exactly as in [`b_decomp`](PVec::b_decomp), with the same
    /// output layout. The representative is congruent to $x$ and lies inside the window, so
    /// $\mathbf G \cdot \mathbf G^{-1}(\mathbf t) \equiv \mathbf t \pmod q$ holds for every
    /// residue, the property required in the paper's Section 2.1. Requires $b^\delta \ge q$,
    /// i.e. $\delta \ge \lceil \log_b q \rceil$ (`params.delta`), and `base` a power of two;
    /// panics if a coefficient is not below $q$.
    ///
    /// This is the decomposition of the witness chunks, $\mathbf s_i = \mathbf G^{-1}(\mathbf f_i)$,
    /// of the inner commitments, $\hat{\mathbf t} = \mathbf G^{-1}(\mathbf t)$, of $\mathbf w$ into
    /// $\hat{\mathbf w}$ (paper, Eq. (13), (14), (16)) and, in the prover's ring switching, of the
    /// quotients by $X^d + 1$ (Section 4.3).
    pub fn b_decomp_zq(&self, q: u64, base: u64, delta: usize, out: &mut Self) {
        self.decomp_mapped(|x| to_window(x, q, base, delta), base, delta, out);
    }

    /// Common body of the two decompositions: applies `map` to every coefficient, splits the
    /// result with [`b_decomp`] into `delta` base-$2^{\log_2 b}$ digits, and scatters digit $k$ of
    /// coefficient $j$ of element $i$ to `out[i * d * delta + k * d + j]`, i.e. to coefficient $j$
    /// of element $i \delta + k$. Asserts `out.length() == len * delta`.
    fn decomp_mapped(&self, map: impl Fn(CoeffType) -> CoeffType, base: u64, delta: usize, out: &mut Self) {
        assert_eq!(self.length() * delta, out.length());

        let logb = base.log();
        let mut decomp_element = vec![0u64; delta];
        let out = out.mut_slice();

        // iterate over each poly
        for i in 0..self.len {
            // iterate over coefficients of this poly
            for j in 0..self.d {
                // decompose and place in correct coefficient
                b_decomp(map(self.vec[i * self.d + j]), logb, delta, &mut decomp_element);

                for k in 0..delta {
                    out[i * self.d * delta + j + k * self.d] = decomp_element[k];
                }
            }
        }
    }

    /// Elementwise division by $X^d + 1$ over $\mathbb Z_q$ ([`Poly::cyclotomic_div`]): `self`
    /// holds full products of dimension $2d$ with residues in $\[0, q)$, and for every element
    /// $i$ the quotient and the remainder (each of dimension $d$, the remainder being the element
    /// reduced into $\mathbf R_q$) are written to element $i$ of `quotient` and `remainder`, which
    /// must have `len` elements each. This is the split of a lifted product into
    /// $\mathbf M \mathbf z = \mathbf y + (X^d + 1) \mathbf r$ in the prover's ring switching
    /// (paper, Section 4.3).
    pub fn cyclotomic_div(&self, q: CoeffType, quotient: &mut Self, remainder: &mut Self) {
        for i in 0..self.length() {
            self.element(i).cyclotomic_div(q, quotient.mut_element(i), remainder.mut_element(i));
        }
    }
}

/// Transcript encoding: every coefficient as 8 big-endian bytes, element after element
/// (`len * d * 8` bytes). Used by [`crate::arithmetic::fs::FS`] to absorb the commitments
/// $\mathbf u$, $\mathbf v$, $\mathbf u^{\prime}$ and the ring element $Y$.
impl Serialise for PVec {
    fn serialise(&self) -> Vec<u8> {
        let mut bytes = Vec::<u8>::new();

        for x in &self.vec {
            bytes.append(&mut x.to_be_bytes().to_vec());
        }

        bytes
    }
}

/// Deep copy.
impl Clone for PVec {
    fn clone(&self) -> Self {
        Self { len: self.len, d: self.d, logd: self.logd, vec: self.vec.clone() }
    }
}

#[cfg(test)]
/// Tests for PolVec.
mod test_poly_vec {
    use super::*;

    #[test]
    fn test_init() {
        let p1 = PVec::zero(256, 32);
        assert_eq!(vec![0u64; 256*32], p1.slice());
    }

    #[test]
    fn test_rand() {
        // just test length of vector and not all zero - rand already tested in Int.
        let q = (1u64 << 32) - 324321;
        let seed = [1u8; 32];

        let p1 = PVec::rand(256, 32, q, seed);
        assert_eq!(256 * 32, p1.slice().len());
        assert_ne!(vec![0u64; 256*32], p1.slice());
    }

    #[test]
    fn test_len() {
        let p1 = PVec::zero(256, 32);
        assert_eq!(256, p1.length());
    }

    #[test]
    fn test_element() {
        let mut p1 = PVec::zero(256, 32);
        let s = p1.mut_slice();

        for i in 0..256 {
            for j in 0..32 {
                s[i * 32 + j] = i as u64;
            }
        }

        for i in 0..256 {
            assert_eq!([i as u64; 32].as_slice(), p1.element(i));
        }
    }

    #[test]
    fn test_mut_element() {
        let mut p1 = PVec::zero(256, 32);
        let s = p1.mut_slice();

        for i in 0..256 {
            for j in 0..32 {
                s[i * 32 + j] = i as u64;
            }
        }

        for i in 0..256 {
            assert_eq!([i as u64; 32].as_mut_slice(), p1.mut_element(i));
        }
    }

    #[test]
    fn test_slice() {
        let p1 = PVec::zero(256, 32);
        assert_eq!(&p1.vec, p1.slice());
    }

    #[test]
    fn test_poly_vec_b_decomp() {
        let base = 16;
        let delta = 8;
        let n = 1024;
        let d = 64;

        // create a random decomposed vector
        let mut decomp_expected = PVec::rand(n * delta, d, base, [1; 32]);
        
        for i in 0..decomp_expected.slice().len() {
            decomp_expected.mut_slice()[i] = decomp_expected.slice()[i].wrapping_sub(base / 2);
        }

        // create the composed vector
        let mut p_vec = PVec::zero(n, d);

        for i in 0..n {
            let v = p_vec.mut_element(i);

            for j in 0..d {
                for k in 0..delta {
                    v[j] = v[j].wrapping_add((decomp_expected.element(i * delta + k)[j] as i64 * base.pow(k as u32) as i64) as u64); 
                }
            }
        }

        // decompose
        let mut decomp_actual = PVec::zero(n * delta, d);
        p_vec.b_decomp(base, delta, &mut decomp_actual);
        assert_eq!(decomp_expected.slice(), decomp_actual.slice());
    }

    #[test]
    fn test_b_decomp_zq_roundtrip() {
        // q = 2^32 - 99: with 8 balanced hex digits the window top is 2004318071 and 53% of
        // residues lie above it; they must be decomposed as x - q (regression test for the dropped carry).
        let q = 4294967197u64;
        let base = 16u64;
        let delta = 8usize;
        let d = 8;
        let top = crate::arithmetic::utils::window_top(base, delta);
        assert_eq!(2004318071, top);

        let mut vals: Vec<u64> = vec![0, 1, 98, 99, top - 1, top, top + 1, top + 2, 2004317973, (q - 1) / 2, (q - 1) / 2 + 1, 1 << 31, 3000000000, q - 2, q - 1];
        let mut rng = ChaCha12Rng::from_seed([7u8; 32]);
        for _ in 0..10000 { vals.push(rand_int(q, q.log(), &mut rng)); }
        while vals.len() % d != 0 { vals.push(0); }

        let mut p_vec = PVec::zero(vals.len() / d, d);
        p_vec.mut_slice().copy_from_slice(&vals);
        let mut decomp = PVec::zero(p_vec.length() * delta, d);
        p_vec.b_decomp_zq(q, base, delta, &mut decomp);

        for i in 0..p_vec.length() {
            for j in 0..d {
                let mut rec: i128 = 0;
                for k in 0..delta {
                    let digit = decomp.element(i * delta + k)[j] as i64;   // wrapping-signed digit
                    assert!(-(base as i64) / 2 <= digit && digit <= base as i64 / 2 - 1, "digit {} out of range", digit);
                    rec += digit as i128 * (base as i128).pow(k as u32);
                }
                let rec_mod_q = rec.rem_euclid(q as i128) as u64;
                assert_eq!(vals[i * d + j], rec_mod_q, "G . G^-1 mismatch for x = {}", vals[i * d + j]);
            }
        }
    }

    #[test]
    fn test_poly_vec_cyclotomic_div() {
        let q = 54351;
        let p = PVec::rand(16, 64, q, [1u8; 32]);
        let mut quo = PVec::zero(16, 32);
        let mut rem = PVec::zero(16, 32);
        p.cyclotomic_div(q, &mut quo, &mut rem);

        let mut expected_quo = vec![0u64; 32];
        let mut expected_rem = vec![0u64; 32];

        for i in 0..p.length() {
            p.element(i).cyclotomic_div(q, &mut expected_quo, &mut expected_rem);
            assert_eq!(expected_quo, quo.element(i));
            assert_eq!(expected_rem, rem.element(i))
        }
    }
}