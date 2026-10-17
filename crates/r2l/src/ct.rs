//! Conversions of secret values between residues and signed integers without branches on the
//! values, and best-effort wiping of secret buffers.
//!
//! "Without branches" is best effort: the code uses only arithmetic, shifts and masks on the
//! secret values, but Rust does not guarantee constant-time code generation.

use crate::{Error, Result};
use hachi_co_math::Poly;

/// $x \bmod^{\pm} q \in \[-(q-1)/2, (q-1)/2\]$ for a residue $x \lt q$, odd $q \lt 2^{63}$
/// (`4_preliminaries.tex`:86), by arithmetic only.
#[inline]
pub(crate) fn center(x: u64, q: u64) -> i64 {
    debug_assert!(x < q && q < 1 << 63 && q % 2 == 1);
    let (x, q) = (x as i64, q as i64);
    // All ones iff x > (q - 1)/2.
    let negative = ((q >> 1) - x) >> 63;
    x - (q & negative)
}

/// $x \bmod^+ q \in \[0, q)$ for $|x| \lt q \lt 2^{63}$, by arithmetic only.
#[inline]
pub(crate) fn lift(x: i64, q: u64) -> u64 {
    debug_assert!(q < 1 << 63 && x.unsigned_abs() < q);
    // x >> 63 is all ones iff x < 0.
    (x + ((q as i64) & (x >> 63))) as u64
}

/// The centered representatives of the coefficients of `p` ([`center`]), after checking that
/// every coefficient is a canonical residue (one branch on the combined result).
///
/// # Errors
/// [`Error::NonCanonical`] naming `what` if a coefficient is $\ge q$ (the reported value is the
/// largest coefficient, which is not secret once the input is malformed).
pub(crate) fn center_poly(p: &Poly, q: u64, what: &'static str) -> Result<Vec<i64>> {
    let mut bad = 0u8;
    for x in &p.0 {
        bad |= u8::from(*x >= q);
    }
    if bad != 0 {
        let value = p.0.iter().copied().max().unwrap_or(0);
        return Err(Error::NonCanonical { what, value, q });
    }
    Ok(p.0.iter().map(|x| center(*x, q)).collect())
}

/// The residues of the signed coefficients `xs` ([`lift`]), $d$ per polynomial, after checking
/// $|x| \le$ `bound` for every coefficient (one branch on the combined result); `bound` $\lt q$
/// is a public bound.
///
/// # Errors
/// `out_of_range` if some $|x| \gt$ `bound`; [`Error::Parameter`] if `bound` $\ge q$ or $d = 0$;
/// [`Error::Shape`] if the length is not a multiple of $d$.
pub(crate) fn lift_polys(
    xs: &[i64],
    d: usize,
    q: u64,
    bound: u64,
    out_of_range: impl FnOnce() -> Error,
) -> Result<Vec<Poly>> {
    if bound >= q || d == 0 {
        return Err(Error::Parameter(format!(
            "short coefficients bounded by {bound} need q > {bound} and d > 0 (q = {q}, d = {d})"
        )));
    }
    if !xs.len().is_multiple_of(d) {
        return Err(Error::Shape {
            what: "short vector (a multiple of d coefficients)",
            expected: xs.len().next_multiple_of(d),
            actual: xs.len(),
        });
    }
    let mut bad = 0u8;
    for x in xs {
        bad |= u8::from(x.unsigned_abs() > bound);
    }
    if bad != 0 {
        return Err(out_of_range());
    }
    Ok(xs
        .chunks(d)
        .map(|chunk| Poly(chunk.iter().map(|x| lift(*x, q)).collect()))
        .collect())
}

/// Overwrites a secret integer buffer with zeros (best effort: copies made earlier, for example
/// by reallocation, are not reached, and the compiler is only discouraged from eliding the
/// writes).
pub(crate) fn wipe(v: &mut [i64]) {
    v.fill(0);
    core::hint::black_box(&*v);
}

/// [`wipe`] for polynomials.
pub(crate) fn wipe_polys(v: &mut [Poly]) {
    for p in v.iter_mut() {
        p.0.fill(0);
    }
    core::hint::black_box(&*v);
}

#[cfg(test)]
mod tests {
    use super::*;
    use hachi_co_math::Zq;

    /// The toy moduli (`docs/m0-spec.md`, "Toy presets") and two extreme odd primes.
    const MODULI: [u64; 5] = [3, 8388637, 33554501, 134217757, 9223372036854775783];

    #[test]
    fn center_and_lift_match_the_reference_at_the_boundaries() {
        for q in MODULI {
            let zq = Zq::new(q).unwrap();
            let half = q / 2;
            let residues = [0, 1, half.saturating_sub(1), half, half + 1, q - 2, q - 1];
            for x in residues.into_iter().filter(|x| *x < q) {
                assert_eq!(center(x, q), zq.to_centered(x), "q = {q}, x = {x}");
                let c = zq.to_centered(x);
                assert_eq!(lift(c, q), x, "q = {q}, x = {c}");
            }
            let big = (q - 1) as i64;
            for x in [-big, -big + 1, -1, 0, 1, big - 1, big] {
                assert_eq!(lift(x, q), zq.from_i64(x), "q = {q}, x = {x}");
            }
        }
    }

    #[test]
    fn checked_conversions_refuse_bad_inputs() {
        let q = 8388637;
        assert_eq!(
            center_poly(&Poly(vec![0, q - 1, 1]), q, "s").unwrap(),
            vec![0, -1, 1]
        );
        assert_eq!(
            center_poly(&Poly(vec![0, q, 1]), q, "s"),
            Err(Error::NonCanonical {
                what: "s",
                value: q,
                q
            })
        );
        let oor = || Error::Overflow("test");
        assert_eq!(
            lift_polys(&[-3, 3, 0, 1], 2, q, 3, oor).unwrap(),
            vec![Poly(vec![q - 3, 3]), Poly(vec![0, 1])]
        );
        assert_eq!(
            lift_polys(&[-4, 3], 2, q, 3, oor),
            Err(Error::Overflow("test"))
        );
        assert_eq!(
            lift_polys(&[i64::MIN, 0], 2, q, 3, oor),
            Err(Error::Overflow("test"))
        );
        assert!(matches!(
            lift_polys(&[0, 0, 0], 2, q, 3, oor),
            Err(Error::Shape { .. })
        ));
        assert!(matches!(
            lift_polys(&[0, 0], 2, q, q, oor),
            Err(Error::Parameter(_))
        ));
        let mut v = vec![5i64, -7];
        wipe(&mut v);
        assert_eq!(v, vec![0, 0]);
        let mut p = vec![Poly(vec![3, 4])];
        wipe_polys(&mut p);
        assert_eq!(p, vec![Poly(vec![0, 0])]);
    }
}
