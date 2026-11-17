//! The public matrices $A \in \mathcal R_q^{\kappa \times mf}$, $B \in \mathcal R_q^{\kappa_1
//! \times \kappa f_u n}$ and $D \in \mathcal R_q^{\kappa_1 \times f_u n}$, uniform, expanded
//! from a 32-byte seed.
//!
//! Entries $j$ of row $i$ come in blocks of [`BLOCK`] from their own SHAKE128 stream,
//! [`ShakePrg`]`(seed, domain, i * blocks + j / BLOCK)` with `blocks` the blocks of a row and the
//! domain naming the matrix, so blocks are expanded independently and in parallel. A coefficient
//! is the first candidate below $q$, a candidate being the next $\lceil L/8 \rceil$ bytes,
//! little-endian, masked to the $L$ bits of $q - 1$.
//!
//! **Deviation.** The reference derives one sequence of uniform ring elements from a fixed seed
//! (`init_comkey`) and reads all three matrices from it, each as a structured matrix over an
//! extension ring (`polxvec_mul_extension`: blocks of $2^{\lceil \log_2 k \rceil}$ entries, for
//! rank $k$, form one element of $\mathbb Z_q\[Y\]/(Y^{64 \cdot 2^{\lceil \log_2 k \rceil}} + 1)$,
//! and the product keeps the first $k$ of its components), with $D$ and the first block of
//! $B$ reading the same elements as $A$. Here the three matrices are independent and uniform,
//! the Module-SIS setting that the hardness rule of [`crate::params`] estimates. The reduced
//! relation lists the matrices' rows in full either way.
use crate::error::Result;
use crate::params::{N, Params};
use hachi_co_math::Poly;
use hachi_co_sampling::ShakePrg;
use rand::Rng;
use rayon::prelude::*;

/// Entries per stream.
pub const BLOCK: usize = 256;

/// The domains of the three matrices.
pub const DOMAIN_A: &[u8] = b"hachi-co-greyhound/v1/A";
/// The domain of $B$.
pub const DOMAIN_B: &[u8] = b"hachi-co-greyhound/v1/B";
/// The domain of $D$.
pub const DOMAIN_D: &[u8] = b"hachi-co-greyhound/v1/D";

/// Uniform residues modulo $q$ from a stream (module documentation).
fn fill_uniform(prg: &mut ShakePrg, q: u64, out: &mut [u64]) {
    let bits = 64 - (q - 1).leading_zeros();
    let width = bits.div_ceil(8) as usize;
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let mut buf = [0u8; 168 * 8];
    let mut pos = buf.len();
    for o in out.iter_mut() {
        loop {
            if pos + width > buf.len() {
                prg.fill_bytes(&mut buf);
                pos = 0;
            }
            let mut v = 0u64;
            for (i, byte) in buf[pos..pos + width].iter().enumerate() {
                v |= u64::from(*byte) << (8 * i);
            }
            pos += width;
            v &= mask;
            if v < q {
                *o = v;
                break;
            }
        }
    }
}

/// The `rows` $\times$ `cols` matrix of `domain` (module documentation).
pub fn expand(seed: &[u8; 32], domain: &[u8], rows: usize, cols: usize, q: u64) -> Vec<Vec<Poly>> {
    let blocks = cols.div_ceil(BLOCK);
    let parts: Vec<Vec<Poly>> = (0..rows * blocks)
        .into_par_iter()
        .map(|index| {
            let (row, block) = (index / blocks, index % blocks);
            let count = BLOCK.min(cols - block * BLOCK);
            let mut prg = ShakePrg::new(seed, domain, (row * blocks + block) as u64);
            let mut coeffs = vec![0u64; count * N];
            fill_uniform(&mut prg, q, &mut coeffs);
            coeffs
                .as_chunks::<N>()
                .0
                .iter()
                .map(|c| Poly(c.to_vec()))
                .collect()
        })
        .collect();
    let mut out: Vec<Vec<Poly>> = (0..rows).map(|_| Vec::with_capacity(cols)).collect();
    for (index, part) in parts.into_iter().enumerate() {
        out[index / blocks].extend(part);
    }
    out
}

/// The three public matrices of a commitment.
#[derive(Clone, Debug)]
pub struct Key {
    /// $A$, $\kappa$ rows of $mf$ entries: the inner commitments $t_i = A \hat s_i$.
    pub a: Vec<Vec<Poly>>,
    /// $B$, $\kappa_1$ rows of $\kappa f_u n$ entries: the commitment $u_1 = B \hat t$.
    pub b: Vec<Vec<Poly>>,
    /// $D$, $\kappa_1$ rows of $f_u n$ entries: the evaluation commitment $u_2 = D \hat w$.
    pub d: Vec<Vec<Poly>>,
}

impl Key {
    /// The matrices of `params` from `seed`.
    ///
    /// # Errors
    /// None in practice: the signature leaves room for samplers that can fail.
    pub fn expand(params: &Params, seed: &[u8; 32]) -> Result<Self> {
        let q = params.q;
        Ok(Self {
            a: expand(seed, DOMAIN_A, params.kappa, params.z_len(), q),
            b: expand(seed, DOMAIN_B, params.kappa1, params.t_len(), q),
            d: expand(seed, DOMAIN_D, params.kappa1, params.w_len(), q),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_are_independent_and_uniform_in_range() {
        let q = 4294967197u64;
        let whole = expand(&[1u8; 32], DOMAIN_A, 3, 600, q);
        assert_eq!(whole.len(), 3);
        assert!(whole.iter().all(|row| row.len() == 600));
        assert!(whole.iter().flatten().flat_map(|p| &p.0).all(|x| *x < q));
        // Each row's blocks come from their own streams, so a narrower matrix is a prefix of
        // each row's first block only when the block count agrees; rows differ.
        assert_ne!(whole[0][0], whole[1][0]);
        assert_ne!(whole[0][0], expand(&[1u8; 32], DOMAIN_B, 3, 600, q)[0][0]);
        // A 57-bit modulus rejects about a third of the candidates and stays in range.
        let q57 = 93266870910952069u64;
        let m = expand(&[2u8; 32], DOMAIN_D, 2, 300, q57);
        assert!(m.iter().flatten().flat_map(|p| &p.0).all(|x| *x < q57));
        let mean = m
            .iter()
            .flatten()
            .flat_map(|p| &p.0)
            .map(|x| *x as f64)
            .sum::<f64>()
            / (2.0 * 300.0 * 64.0);
        assert!((mean / q57 as f64 - 0.5).abs() < 0.01);
    }
}
