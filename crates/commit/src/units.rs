//! The units of $\mathcal R\_q = \mathbb Z\_q\[X\]/(X^d + 1)$: the invertibility test of Open's
//! slack space $\mathcal Z = \lbrace (c\_1, \dots, c\_{2^r}) \in (\mathcal R\_q^\times)^{2^r} :
//! \Vert c\_i \Vert\_1 \le \bar\omega \rbrace$ (`6_Collaborative_Hachi.tex`:393-395; the oracle's
//! `SageRing.is_unit`, `ref/README.md`, choice 18). `hachi-co-math` has no such test.
//!
//! An element $c \in \mathcal R\_q$ is a unit iff $\gcd(c, X^d + 1) = 1$ in $\mathbb F\_q\[X\]$
//! ($q$ is prime). Two exact implementations:
//!
//! - [`is_unit_euclid`] computes the gcd by Euclid's algorithm over $\mathbb F\_q\[X\]$, for any
//!   prime $q$, in $O(d^2)$ operations.
//! - [`is_unit`] uses the factorization $X^d + 1 = (X^{d/2} - r)(X^{d/2} + r)$ with
//!   $r^2 = -1$ into two irreducible factors, valid for $q \equiv 5 \pmod 8$ (the moduli of the
//!   draft, `6_Collaborative_Hachi.tex`:272): by the Chinese remainder theorem $c$ is a unit iff
//!   it is nonzero modulo both factors. Writing $c = c\_{\mathrm{lo}} + X^{d/2}
//!   c\_{\mathrm{hi}}$ with $\deg c\_{\mathrm{lo}}, \deg c\_{\mathrm{hi}} \lt d/2$, the residues are
//!   $c\_{\mathrm{lo}} \pm r\, c\_{\mathrm{hi}}$, so the test takes $O(d)$ operations. For other
//!   moduli it falls back to [`is_unit_euclid`].
//!
//! **Why the factors are irreducible** (proved here; the statement is the case $k = 2$ of the
//! partially splitting rings of Lyubashevsky and Seiler, EUROCRYPT 2018, which we recall but did
//! not re-read). $X^d + 1 = \Phi\_{2d}$ for $d = 2^\alpha$, and over $\mathbb F\_q$ ($q$ odd) it
//! splits into $\varphi(2d)/o$ distinct irreducible factors of degree $o$, the multiplicative
//! order of $q$ modulo $2d$. For $q \equiv 5 \pmod 8$: $v\_2(q - 1) = 2$ and $v\_2(q + 1) = 1$, so
//! by lifting the exponent $v\_2(q^{2^i} - 1) = i + 2$ for $i \ge 1$, and $o = 2^{\alpha - 1} =
//! d/2$ for $d \ge 4$ ($o = 1 = d/2$ for $d = 2$, where $q \equiv 1 \pmod 4$). Hence there are
//! exactly $d/(d/2) = 2$ irreducible factors, each of degree $d/2$. Since
//! $(X^{d/2} - r)(X^{d/2} + r) = X^d - r^2 = X^d + 1$, unique factorization makes these the two
//! factors. $r = 2^{(q-1)/4}$ works because $2$ is a non-residue modulo $q \equiv 5 \pmod 8$, so
//! $r^2 = 2^{(q-1)/2} = -1$ by Euler's criterion ([`sqrt_minus_one`] checks $r^2 = -1$).
//! Additionally checked in SageMath 10.9 (factorization of $X^d + 1$ over $\mathbb F\_q$ for
//! eleven primes $q \equiv 5 \pmod 8$, among them the three toy moduli, and $2 \le d \le 2048$:
//! 115 cases), and in the tests of this crate against [`is_unit_euclid`] (exhaustively for small
//! rings, against a brute-force inverse search for the smallest, and on constructed non-units at
//! the toy moduli).

use crate::error::{Result, check_len};
use hachi_co_math::{Poly, Ring, Zq};

/// A square root $r$ of $-1$ modulo $q$ for $q \equiv 5 \pmod 8$: $r = 2^{(q-1)/4} \bmod q$
/// (module documentation). `None` for any other modulus (or if the check $r^2 = -1$ failed,
/// which cannot happen for a prime $q \equiv 5 \pmod 8$).
pub fn sqrt_minus_one(zq: &Zq) -> Option<u64> {
    let q = zq.modulus();
    if q % 8 != 5 {
        return None;
    }
    let r = zq.pow(2, (q - 1) / 4);
    (zq.mul(r, r) == q - 1).then_some(r)
}

/// Whether $c$ is a unit of $\mathcal R\_q$, i.e. $\gcd(c, X^d + 1) = 1$ in $\mathbb F\_q\[X\]$
/// (coefficients read modulo $q$). For $q \equiv 5 \pmod 8$ by the two residues
/// $c\_{\mathrm{lo}} \pm r\, c\_{\mathrm{hi}}$ modulo the irreducible factors
/// $X^{d/2} \mp r$ of $X^d + 1$ (module documentation), otherwise by [`is_unit_euclid`]. The
/// zero polynomial is not a unit.
///
/// # Errors
/// [`Error::Shape`](crate::Error::Shape) unless `c` has $d$ coefficients.
pub fn is_unit(ring: &Ring, c: &Poly) -> Result<bool> {
    check_len("c (d coefficients)", ring.degree(), c.0.len())?;
    let zq = ring.zq();
    match sqrt_minus_one(zq) {
        Some(r) => {
            let (lo, hi) = c.0.split_at(ring.degree() / 2);
            let (mut plus, mut minus) = (false, false);
            for (a, b) in lo.iter().zip(hi) {
                let rb = zq.mul(r, *b);
                plus |= zq.add(*a, rb) != 0;
                minus |= zq.sub(*a, rb) != 0;
            }
            Ok(plus && minus)
        }
        None => is_unit_euclid(ring, c),
    }
}

/// Whether $c$ is a unit of $\mathcal R\_q$, by Euclid's algorithm: the gcd of $c$ (coefficients
/// read modulo $q$) and $X^d + 1$ in $\mathbb F\_q\[X\]$ is computed by repeated division with
/// remainder, and $c$ is a unit iff it is a nonzero constant. Exact for every prime $q$;
/// $O(d^2)$ operations. The reference against which [`is_unit`] is tested.
///
/// # Errors
/// [`Error::Shape`](crate::Error::Shape) unless `c` has $d$ coefficients.
pub fn is_unit_euclid(ring: &Ring, c: &Poly) -> Result<bool> {
    let d = ring.degree();
    check_len("c (d coefficients)", d, c.0.len())?;
    let zq = ring.zq();
    // a = X^d + 1, b = c; both trimmed (no leading zero coefficients).
    let mut a = vec![0u64; d + 1];
    a[0] = 1;
    a[d] = 1;
    let mut b: Vec<u64> = c.0.iter().map(|x| zq.reduce(*x)).collect();
    trim(&mut b);
    while !b.is_empty() {
        let r = remainder(zq, &a, &b);
        a = b;
        b = r;
    }
    // a is the gcd up to a nonzero scalar.
    Ok(a.len() == 1)
}

/// Removes the leading zero coefficients.
fn trim(p: &mut Vec<u64>) {
    while p.last() == Some(&0) {
        p.pop();
    }
}

/// $a \bmod b$ in $\mathbb F\_q\[X\]$ for trimmed $a$ and a trimmed nonzero $b$ with reduced
/// coefficients; the result is trimmed.
fn remainder(zq: &Zq, a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut r = a.to_vec();
    // b is trimmed and reduced, so its leading coefficient lies in [1, q) and is invertible
    // (q is prime): the `else` branch is unreachable.
    let Some(lead_inv) = b.last().and_then(|l| zq.inv(*l)) else {
        return Vec::new();
    };
    while r.len() >= b.len() {
        let Some(top) = r.last() else { break };
        let factor = zq.mul(*top, lead_inv);
        let shift = r.len() - b.len();
        for (j, bj) in b.iter().enumerate() {
            r[shift + j] = zq.sub(r[shift + j], zq.mul(factor, *bj));
        }
        // The leading coefficient is now zero, so the length drops by at least one.
        trim(&mut r);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remainder_by_constants_and_monomials() {
        let zq = Zq::new(13).unwrap();
        // (X^3 + 2X + 5) mod 3 = 0; mod X = 5; mod (X + 1) = -1 - 2 + 5 = 2.
        let a = [5, 2, 0, 1];
        assert_eq!(remainder(&zq, &a, &[3]), Vec::<u64>::new());
        assert_eq!(remainder(&zq, &a, &[0, 1]), vec![5]);
        assert_eq!(remainder(&zq, &a, &[1, 1]), vec![2]);
        // A divisor of higher degree leaves a unchanged.
        assert_eq!(remainder(&zq, &[5, 2], &[1, 0, 1]), vec![5, 2]);
    }

    #[test]
    fn square_roots_of_minus_one() {
        for q in [
            5u64,
            13,
            29,
            37,
            8388637,
            33554501,
            134217757,
            9223372036854775549,
        ] {
            let zq = Zq::new(q).unwrap();
            let r = sqrt_minus_one(&zq).unwrap();
            assert_eq!(zq.mul(r, r), q - 1, "q = {q}");
        }
        for q in [3u64, 7, 11, 17, 41] {
            assert_eq!(sqrt_minus_one(&Zq::new(q).unwrap()), None, "q = {q}");
        }
    }
}
