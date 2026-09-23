use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::arithmetic::{CoeffType, fs::Serialise, poly::Poly, utils::{Logarithm, b_decomp, rand_int, to_window}};

/// Representation of a vectors of polynomials over Zq with maximum degree d-1.
/// Stored in default (coefficient) format one after another in a single vector
/// (Array of Structure).
pub struct PVec {
    len: usize,
    d: usize,
    logd: usize,
    vec: Vec<CoeffType>
}

impl PVec {
    /// Initialize a zero vector.
    pub fn zero(len: usize, d: usize) -> Self {
        assert!(d.is_power_of_two());
        let logd = d.log();

        Self { len, d, logd, vec: vec![0; len * d] }
    }

    /// Sample a vector with elements that have uniform random Zq coefficients.
    pub fn rand(len: usize, d: usize, q: CoeffType, seed: [u8; 32]) ->  Self {
        assert!(d.is_power_of_two());
        let logd = d.log();
        let logq = q.log();
        let mut rng = ChaCha12Rng::from_seed(seed);

        let vec: Vec<CoeffType> = (0..len * d).map(|_| rand_int(q, logq, &mut rng)).collect();

        Self { len, d, logd, vec }
    }

    /// Length of vector.
    pub fn length(&self) -> usize {
        self.len
    }

    /// Get the i-th element of the vector as a slice.
    pub fn element(&self, i: usize) -> &[CoeffType] {
        &self.vec[(i << self.logd)..((i + 1) << self.logd)]
    }

    /// Get the i-th element of the vector as a mutable slice.
    pub fn mut_element(&mut self, i: usize) -> &mut [CoeffType] {
        &mut self.vec[(i << self.logd)..((i + 1) << self.logd)]
    }

    /// Get a reference to the whole internal vector.
    pub fn slice(&self) -> &[CoeffType] {
        &self.vec
    }

    /// Get a mutable reference to the whole internal vector.
    pub fn mut_slice(&mut self) -> &mut [CoeffType] {
        &mut self.vec
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Balanced decomposition of the vector of polynomials, for wrapping-signed coefficients that
    /// already lie inside the `delta`-digit balanced window (e.g. the bounded response z).
    pub fn b_decomp(&self, base: u64, delta: usize, out: &mut Self) {
        self.decomp_mapped(|x| x, base, delta, out);
    }

    #[cfg_attr(feature = "stats", time_graph::instrument)]
    /// Balanced decomposition of a vector of polynomials whose coefficients are residues in [0, q).
    /// Each coefficient is first mapped to its representative in the balanced-digit window
    /// (see `to_window`), so that recomposition G . G^{-1}(x) = x holds modulo q for every residue.
    pub fn b_decomp_zq(&self, q: u64, base: u64, delta: usize, out: &mut Self) {
        self.decomp_mapped(|x| to_window(x, q, base, delta), base, delta, out);
    }

    /// Decompose each coefficient after applying `map`, scattering digit k of coefficient j of
    /// polynomial i to out[i * d * delta + j + k * d].
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

    /// Perform cyclotomic reduction of each element of the vector.
    pub fn cyclotomic_div(&self, q: CoeffType, quotient: &mut Self, remainder: &mut Self) {
        for i in 0..self.length() {
            self.element(i).cyclotomic_div(q, quotient.mut_element(i), remainder.mut_element(i));
        }
    }
}

/// Serialisation of polynomial in coefficient form.
impl Serialise for PVec {
    fn serialise(&self) -> Vec<u8> {
        let mut bytes = Vec::<u8>::new();

        for x in &self.vec {
            bytes.append(&mut x.to_be_bytes().to_vec());
        }

        bytes
    }
}

/// Clone the vector.
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