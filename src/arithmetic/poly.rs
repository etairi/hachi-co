//! Operations on a single polynomial over $\mathbb Z_q$ given as a slice of `u64` coefficients,
//! with the coefficient of $X^j$ at index $j$: evaluation at a point of $\mathbb F_{q^4}$ and
//! division by $X^d + 1$. Both belong to the ring switching of the paper's Section 4.3, where
//! the verification equations are lifted from $\mathbf R_q$ to $\mathbb Z_q\[X\]$, split into a
//! quotient and a remainder modulo $X^d + 1$, and evaluated at a random $\alpha \in \mathbb F_{q^4}$.
//! The elements of a [`crate::arithmetic::poly_vec::PVec`] are such slices, and the vector
//! versions of both operations live there.

use ark_ff::AdditiveGroup;

use crate::arithmetic::{CoeffType, ExtField};
use crate::arithmetic::utils::mul_int_field;

/// A polynomial in coefficient form over $\mathbb Z_q$, viewed through a slice of residues.
pub trait Poly {
    /// Evaluate at $\alpha \in \mathbb F_{q^4}$: returns $\sum_{j \lt n} p_j \alpha^j$ for a slice
    /// of length $n$, given the table `alpha_pows[j]` $= \alpha^j$ (at least $n$ entries, see
    /// [`crate::arithmetic::utils::powers`]). Each coefficient is read as an unsigned integer and
    /// reduced modulo $q$ by [`mul_int_field`], so the slice must hold residues in $\[0, q)$: a
    /// wrapping-signed coefficient would be read as $2^{64} - \lvert x \rvert \bmod q$, not as
    /// $-\lvert x \rvert$ (signed data goes through [`crate::arithmetic::utils::lift_int`]
    /// instead). Cost: $n$ products of an $\mathbb F_{q^4}$ element by an $\mathbb F_q$ element.
    fn eval(&self, alpha_pows: &[ExtField]) -> ExtField;

    /// Divide by $X^d + 1$ over $\mathbb Z_q$. For a slice of length $2d$ holding the residues of
    /// $p = p_{\mathrm{lo}} + X^d p_{\mathrm{hi}}$ with $\deg p_{\mathrm{lo}}, \deg p_{\mathrm{hi}} \lt d$,
    /// writes $\mathrm{quotient} = p_{\mathrm{hi}}$ and
    /// $\mathrm{remainder} = p_{\mathrm{lo}} - p_{\mathrm{hi}} \bmod q$, so that
    /// $p = (X^d + 1) \cdot \mathrm{quotient} + \mathrm{remainder}$ in $\mathbb Z_q\[X\]$ and the
    /// remainder is the image of $p$ in $\mathbf R_q$. $d$ is half the slice length; both outputs
    /// must have length $d$ (asserted) and receive residues in $\[0, q)$. Precondition: the input
    /// holds residues in $\[0, q)$ and $q \lt 2^{63}$ (the sign test on the wrapping difference
    /// relies on both); the quotient coefficients are copied unchanged. Cost: $d$ subtractions.
    fn cyclotomic_div(&self, q: CoeffType, quotient: &mut [CoeffType], remainder: &mut [CoeffType]);
}

/// The slice implementation: a `&[u64]` of length $n$ is the polynomial
/// $\sum_{j \lt n} p_j X^j$; `cyclotomic_div` infers $d = n / 2$.
impl Poly for &[u64] {
    fn eval(&self, alpha_pows: &[ExtField]) -> ExtField {
        let mut out = ExtField::ZERO;

        for i in 0..self.len() {
            out += mul_int_field(self[i], alpha_pows[i]);
        }

        out
    }

    fn cyclotomic_div(&self, q: CoeffType, quotient: &mut [CoeffType], remainder: &mut [CoeffType]) {
        let d = self.len() / 2;
        assert_eq!(d, quotient.len());
        assert_eq!(d, remainder.len());

        for i in 0..d {
            remainder[i] = self[i].wrapping_sub(self[i+d]);
            quotient[i] = self[i+d];

            if (remainder[i] as i64) < 0 {
                remainder[i] = remainder[i].wrapping_add(q);
            }
        }
    }
}

#[cfg(test)]
/// Tests for Poly.
mod test_poly {
    use ark_ff::{Field, UniformRand};

    use crate::arithmetic::field::Fq4;

    use super::*;

    #[test]
    fn test_eval() {
        // 3 + x + 5x^2 + 3432x^3 + 4313x^4 + 8761x^5 + 2x^6 + 535430x^7
        let poly = [3u64, 1, 5, 3432, 4313, 8761, 2, 535430].as_slice();

        // get a random evaluation point
        let mut rng = ark_std::test_rng();
        let alpha = Fq4::rand(&mut rng);

        // generate the powers [1, alpha, ..., alpha^7] and calculate the correct evaluation
        let mut pows: Vec<Fq4> = Vec::new();
        let mut expected = Fq4::ZERO;

        for i in 0..8 {
            pows.push(alpha.pow([i]));
            expected += mul_int_field(poly[i as usize], alpha.pow([i]));
        }

        // evaluate the polynomial
        let actual = poly.eval(&pows);

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_cyclotomic_div() {
        // 3 + x + 5x^2 + 3432x^3 + 4313x^4 + 8761x^5 + 2x^6 + 535430x^7
        let poly = [3u64, 1, 5, 3432, 4313, 8761, 2, 535430].as_slice();

        let q = (1i64 << 32) - 43221;

        // expected division
        let expected_quo = [4313, 8761, 2, 535430].as_slice();
        let expected_rem = [4294919765, 4294915315, 3, 4294392077].as_slice();

        // actual division
        let mut actual_quo = [0u64; 4];
        let mut actual_rem = [0u64; 4];

        poly.cyclotomic_div(q as u64, &mut actual_quo, &mut actual_rem);

        assert_eq!(expected_quo, actual_quo);
        assert_eq!(expected_rem, actual_rem);
    }
}