//! The invertibility test of $\mathcal R\_q$ (`hachi_co_commit::units`): the factorization-based
//! [`is_unit`] against Euclid's algorithm [`is_unit_euclid`], both against a brute-force inverse
//! search and against the number of units $(q^{d/2} - 1)^2$ (CRT with two irreducible factors of
//! degree $d/2$, for $q \equiv 5 \pmod 8$), exhaustively over small rings; constructed non-units
//! and the whole $\ell\_1$ ball of radius $\bar\omega = 4$ at the toy moduli; the fallback for
//! other moduli (seeds 70-79).
use hachi_co_commit::Error;
use hachi_co_commit::units::{is_unit, is_unit_euclid, sqrt_minus_one};
use hachi_co_math::{Poly, Ring};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Every element of $\mathcal R\_q$ for tiny $q^d$, in counting order.
fn all_elements(q: u64, d: usize) -> impl Iterator<Item = Poly> {
    let total = q.pow(d as u32);
    (0..total).map(move |mut x| {
        let mut c = vec![0u64; d];
        for coeff in c.iter_mut() {
            *coeff = x % q;
            x /= q;
        }
        Poly(c)
    })
}

/// Whether $c$ has an inverse, by trying every element (schoolbook products).
fn has_inverse(ring: &Ring, c: &Poly, elements: &[Poly]) -> bool {
    let one = ring.monomial(0);
    elements
        .iter()
        .any(|x| ring.mul_schoolbook(c, x).unwrap() == one)
}

#[test]
fn both_tests_agree_with_brute_force_and_count_on_small_rings() {
    // (q, d, brute-force inverse search?)
    let cases = [
        (5u64, 2usize, true),
        (13, 2, true),
        (29, 2, true),
        (37, 2, true),
        (5, 4, true),
        (13, 4, false),
        (29, 4, false),
        (5, 8, false),
    ];
    for (q, d, brute) in cases {
        let ring = Ring::new(q, d).unwrap();
        let elements: Vec<Poly> = all_elements(q, d).collect();
        let mut units = 0u64;
        for c in &elements {
            let fast = is_unit(&ring, c).unwrap();
            let euclid = is_unit_euclid(&ring, c).unwrap();
            assert_eq!(fast, euclid, "q = {q}, d = {d}, c = {:?}", c.0);
            if brute {
                assert_eq!(fast, has_inverse(&ring, c, &elements), "q = {q}, d = {d}");
            }
            units += u64::from(fast);
        }
        let half = q.pow(d as u32 / 2);
        assert_eq!(
            units,
            (half - 1) * (half - 1),
            "q = {q}, d = {d}: unit count"
        );
    }
}

/// $(X^{d/2} \mp r) g \bmod (X^d + 1)$ for random $g$: never units.
#[test]
fn constructed_non_units_at_the_toy_moduli() {
    let mut rng = ChaCha8Rng::seed_from_u64(70);
    for q in [8388637u64, 33554501, 134217757] {
        for d in [2usize, 4, 16, 64] {
            let ring = Ring::new(q, d).unwrap();
            let r = sqrt_minus_one(ring.zq()).unwrap();
            for sign in [r, q - r] {
                // X^(d/2) - sign.
                let mut factor = ring.zero();
                factor.0[0] = q - sign;
                factor.0[d / 2] = 1;
                assert!(!is_unit(&ring, &factor).unwrap());
                assert!(!is_unit_euclid(&ring, &factor).unwrap());
                for _ in 0..20 {
                    let g = Poly((0..d).map(|_| rng.random_range(0..q)).collect());
                    let c = ring.mul(&factor, &g).unwrap();
                    assert!(!is_unit(&ring, &c).unwrap(), "q = {q}, d = {d}");
                    assert!(!is_unit_euclid(&ring, &c).unwrap(), "q = {q}, d = {d}");
                    // A unit plus a non-unit: almost surely a unit; both tests agree.
                    let shifted = ring.add(&c, &ring.monomial(0)).unwrap();
                    assert_eq!(
                        is_unit(&ring, &shifted).unwrap(),
                        is_unit_euclid(&ring, &shifted).unwrap()
                    );
                }
            }
            // Zero is not a unit; monomials and random elements (almost surely) are.
            assert!(!is_unit(&ring, &ring.zero()).unwrap());
            assert!(!is_unit_euclid(&ring, &ring.zero()).unwrap());
            for e in [
                0i64,
                1,
                d as i64 / 2,
                d as i64 - 1,
                d as i64,
                2 * d as i64 - 1,
            ] {
                let m = ring.monomial(e);
                assert!(is_unit(&ring, &m).unwrap() && is_unit_euclid(&ring, &m).unwrap());
            }
            for _ in 0..20 {
                let c = Poly((0..d).map(|_| rng.random_range(0..q)).collect());
                assert_eq!(
                    is_unit(&ring, &c).unwrap(),
                    is_unit_euclid(&ring, &c).unwrap()
                );
            }
        }
    }
}

/// Every element of the $\ell\_1$ ball of radius 4 in $\mathbb Z^{16}$ (50 049 elements; Open's
/// slack space at the toy presets has $\bar\omega = 2\omega = 4$): both tests agree, and every
/// nonzero element is a unit at the three toy moduli.
#[test]
fn the_toy_slack_ball_consists_of_units() {
    fn ball(d: usize, radius: i64, prefix: &mut Vec<i64>, out: &mut Vec<Vec<i64>>) {
        if prefix.len() == d {
            out.push(prefix.clone());
            return;
        }
        let used: i64 = prefix.iter().map(|x| x.abs()).sum();
        for v in -(radius - used)..=(radius - used) {
            prefix.push(v);
            ball(d, radius, prefix, out);
            prefix.pop();
        }
    }
    let mut elements = Vec::new();
    ball(16, 4, &mut Vec::new(), &mut elements);
    assert_eq!(elements.len(), 50049);
    for q in [8388637u64, 33554501, 134217757] {
        let ring = Ring::new(q, 16).unwrap();
        for c in &elements {
            let p = ring.from_i64(c).unwrap();
            let fast = is_unit(&ring, &p).unwrap();
            assert_eq!(fast, is_unit_euclid(&ring, &p).unwrap(), "{c:?}");
            assert_eq!(fast, c.iter().any(|x| *x != 0), "{c:?}");
        }
    }
}

/// For $q \not\equiv 5 \pmod 8$ there is no factorization shortcut: [`is_unit`] falls back to
/// Euclid's algorithm, which agrees with the brute-force inverse search.
#[test]
fn other_moduli_fall_back_to_euclid() {
    for (q, d) in [
        (3u64, 2usize),
        (7, 2),
        (17, 2),
        (41, 2),
        (3, 4),
        (7, 4),
        (17, 4),
        (3, 8),
    ] {
        assert_eq!(sqrt_minus_one(Ring::new(q, d).unwrap().zq()), None);
        let ring = Ring::new(q, d).unwrap();
        let elements: Vec<Poly> = all_elements(q, d).collect();
        for c in &elements {
            let fast = is_unit(&ring, c).unwrap();
            assert_eq!(fast, is_unit_euclid(&ring, c).unwrap());
            if elements.len() <= 2401 {
                assert_eq!(fast, has_inverse(&ring, c, &elements), "q = {q}, d = {d}");
            }
        }
    }
}

#[test]
fn wrong_lengths_are_errors() {
    let ring = Ring::new(8388637, 16).unwrap();
    for len in [0usize, 15, 17] {
        let c = Poly(vec![1; len]);
        assert!(matches!(is_unit(&ring, &c), Err(Error::Shape { .. })));
        assert!(matches!(
            is_unit_euclid(&ring, &c),
            Err(Error::Shape { .. })
        ));
    }
    // Unreduced coefficients are read modulo q.
    let q = 8388637u64;
    let mut c = ring.zero();
    c.0[0] = q;
    assert!(!is_unit(&ring, &c).unwrap() && !is_unit_euclid(&ring, &c).unwrap());
    c.0[0] = q + 1;
    assert!(is_unit(&ring, &c).unwrap() && is_unit_euclid(&ring, &c).unwrap());
}
