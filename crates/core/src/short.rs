//! Exact arithmetic on short integer polynomials and infinity norms, shared by the blocks that
//! handle masks, folds and responses (the ring-to-linear step, `5_Building_Blocks.tex`:127-133,
//! and the test oracle Open, `6_Collaborative_Hachi.tex`:401).
//!
//! A short polynomial is the vector of its $d$ signed coefficients (index $j$ for $X^j$), an
//! element of $\mathcal R = \mathbb Z\[X\]/(X^d + 1)$ (`4_preliminaries.tex`:67).

use crate::data::check_len;
use crate::{Error, Result};
use hachi_co_math::{Poly, Ring};

/// The exact product $c \cdot s$ in $\mathbb Z\[X\]/(X^d + 1)$ (negacyclic: $X^d = -1$), for
/// two integer polynomials of the same length $d$; the oracle's `int_ring_mul_rows` for one row
/// (`ref/hachico_ref/r2l.py`). In the ring-to-linear step $c$ is a challenge and $s$ the centered
/// digits of a block, so that $\sum\_{i' \in \mathcal I\_i} c\_{i'} \mathbf s\_{i'}$ is computed
/// over the integers (`5_Building_Blocks.tex`:128; `ref/README.md`, choice 17).
///
/// The products and sums are formed in `i128` with overflow checks.
///
/// # Errors
/// [`Error::Shape`] if the lengths differ; [`Error::Overflow`] if a coefficient of the result does
/// not fit in `i64` (impossible for $\Vert c \Vert\_1 \Vert s \Vert\_\infty \lt 2^{63}$).
pub fn negacyclic_mul(c: &[i64], s: &[i64]) -> Result<Vec<i64>> {
    check_len("negacyclic product (equal lengths)", c.len(), s.len())?;
    let d = c.len();
    let mut acc = vec![0i128; d];
    for (i, ci) in c.iter().enumerate() {
        if *ci == 0 {
            continue;
        }
        for (j, sj) in s.iter().enumerate() {
            let product = i128::from(*ci) * i128::from(*sj);
            let (k, term) = if i + j < d {
                (i + j, Some(product))
            } else {
                (i + j - d, product.checked_neg())
            };
            acc[k] = term
                .and_then(|t| acc[k].checked_add(t))
                .ok_or(Error::Overflow("negacyclic product"))?;
        }
    }
    acc.into_iter()
        .map(|x| i64::try_from(x).map_err(|_| Error::Overflow("negacyclic product")))
        .collect()
}

/// $c \cdot s\_e$ for every row $s\_e$ of `rows` ([`negacyclic_mul`] row by row; the oracle's
/// `int_ring_mul_rows`).
///
/// # Errors
/// As [`negacyclic_mul`].
pub fn negacyclic_mul_rows(c: &[i64], rows: &[Vec<i64>]) -> Result<Vec<Vec<i64>>> {
    rows.iter().map(|s| negacyclic_mul(c, s)).collect()
}

/// $\Vert v \Vert\_\infty = \max\_j |v\_j|$ of signed integers (`4_preliminaries.tex`:86), and
/// 0 for an empty vector.
pub fn inf_norm(v: &[i64]) -> u64 {
    v.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0)
}

/// $\Vert (v\_1, \dots, v\_L) \Vert\_\infty$, the largest [`inf_norm`] of the rows
/// (`4_preliminaries.tex`:86).
pub fn inf_norm_rows(rows: &[Vec<i64>]) -> u64 {
    rows.iter().map(|r| inf_norm(r)).max().unwrap_or(0)
}

/// The centered representatives $a\_j \bmod^{\pm} q \in \[-(q-1)/2, (q-1)/2\]$ of every
/// polynomial (`4_preliminaries.tex`:86), e.g. the digits of $\mathbf s\_i$ as integers.
///
/// # Errors
/// [`Error::Math`] for a polynomial without $d$ coefficients.
pub fn centered(ring: &Ring, polys: &[Poly]) -> Result<Vec<Vec<i64>>> {
    Ok(polys
        .iter()
        .map(|p| ring.to_centered(p))
        .collect::<hachi_co_math::Result<Vec<_>>>()?)
}

/// $\Vert (a\_1, \dots, a\_L) \Vert\_\infty$ over centered representatives
/// (`4_preliminaries.tex`:86), e.g. $\Vert \hat{\mathbf t} \Vert\_\infty$ in Open
/// (`6_Collaborative_Hachi.tex`:402); 0 for an empty vector.
///
/// # Errors
/// [`Error::Math`] for a polynomial without $d$ coefficients.
pub fn inf_norm_polys(ring: &Ring, polys: &[Poly]) -> Result<u64> {
    let mut norm = 0;
    for p in polys {
        norm = norm.max(ring.inf_norm(p)?);
    }
    Ok(norm)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The schoolbook product over `Z[X]/(X^d + 1)`, accumulated in i128.
    fn reference(c: &[i64], s: &[i64]) -> Vec<i128> {
        let d = c.len();
        let mut out = vec![0i128; d];
        for i in 0..d {
            for j in 0..d {
                let p = i128::from(c[i]) * i128::from(s[j]);
                if i + j < d {
                    out[i + j] += p;
                } else {
                    out[i + j - d] -= p;
                }
            }
        }
        out
    }

    #[test]
    fn monomials_wrap_with_a_sign() {
        let d = 8;
        for a in 0..d {
            for b in 0..d {
                let mut x = vec![0i64; d];
                let mut y = vec![0i64; d];
                x[a] = 1;
                y[b] = 1;
                let mut want = vec![0i64; d];
                if a + b < d {
                    want[a + b] = 1;
                } else {
                    want[a + b - d] = -1;
                }
                assert_eq!(negacyclic_mul(&x, &y).unwrap(), want, "X^{a} X^{b}");
            }
        }
    }

    #[test]
    fn matches_the_schoolbook_reference_and_the_ring_product() {
        // A fixed LCG (seed 0x5eed) for reproducible short inputs.
        let mut state = 0x5eedu64;
        let mut next = |bound: i64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as i64).rem_euclid(2 * bound + 1) - bound
        };
        let q = 8388637u64; // toy-n1
        let ring = Ring::new(q, 16).unwrap();
        for trial in 0..200 {
            let c: Vec<i64> = (0..16).map(|_| next(2)).collect();
            let s: Vec<i64> = (0..16)
                .map(|_| next(if trial % 2 == 0 { 2 } else { 1 << 20 }))
                .collect();
            let got = negacyclic_mul(&c, &s).unwrap();
            let want = reference(&c, &s);
            assert_eq!(got.iter().map(|x| i128::from(*x)).collect::<Vec<_>>(), want);
            // Reduced mod q it is the product in R_q.
            let rq = ring
                .mul(&ring.from_i64(&c).unwrap(), &ring.from_i64(&s).unwrap())
                .unwrap();
            assert_eq!(ring.from_i64(&got).unwrap(), rq);
        }
    }

    #[test]
    fn rejects_mismatched_lengths_and_overflow() {
        assert!(matches!(
            negacyclic_mul(&[1, 2], &[1, 2, 3]),
            Err(Error::Shape { .. })
        ));
        assert_eq!(negacyclic_mul(&[], &[]).unwrap(), Vec::<i64>::new());
        assert_eq!(
            negacyclic_mul(&[i64::MAX, 0], &[2, 0]),
            Err(Error::Overflow("negacyclic product"))
        );
        // i64::MIN * i64::MIN = 2^126 fits i128 but not the result.
        assert_eq!(
            negacyclic_mul(&[i64::MIN], &[i64::MIN]),
            Err(Error::Overflow("negacyclic product"))
        );
        // The largest exact result passes.
        assert_eq!(negacyclic_mul(&[i64::MAX], &[1]).unwrap(), vec![i64::MAX]);
        assert_eq!(
            negacyclic_mul(&[0, 1], &[0, i64::MIN + 1]).unwrap(),
            vec![i64::MAX, 0]
        );
    }

    #[test]
    fn norms() {
        assert_eq!(inf_norm(&[]), 0);
        assert_eq!(inf_norm(&[3, -7, 5]), 7);
        assert_eq!(inf_norm(&[i64::MIN]), 1u64 << 63);
        assert_eq!(inf_norm_rows(&[vec![1, -2], vec![], vec![0, 4]]), 4);
        let ring = Ring::new(8388637, 4).unwrap();
        let q = 8388637u64;
        let polys = [Poly(vec![0, 1, q - 1, 2]), Poly(vec![q - 5, 0, 0, 0])];
        assert_eq!(inf_norm_polys(&ring, &polys).unwrap(), 5);
        assert_eq!(
            centered(&ring, &polys).unwrap(),
            vec![vec![0, 1, -1, 2], vec![-5, 0, 0, 0]]
        );
        assert!(inf_norm_polys(&ring, &[Poly(vec![1])]).is_err());
    }
}
