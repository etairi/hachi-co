//! $\mathbb F_{q^K}$ in Hachi's basis: the Chebyshev-rule product against the product of the
//! embeddings in $\mathcal R_q$ (schoolbook, with an embedding written out here independently of
//! the crate's), the field axioms, $x^{q^K - 1} = 1$, inverses against $x^{q^K - 2}$, and the
//! Frobenius against $x^q$ and $\sigma_q$, with order exactly $K$ (seeds 40-49).
mod common;

use common::*;
use hachi_co_math::{Error, Ext, Fqk, Poly, Ring};
use num_bigint::BigUint;
use rand::RngExt;

/// Eq. (7): $a_m$ at position $m d/(2K)$ and $-a_m$ at $(2K - m) d/(2K)$.
fn embed(x: &[u64], q: u64, d: usize) -> Poly {
    let k = x.len();
    let s = d / (2 * k);
    let mut c = vec![0u64; d];
    c[0] = x[0] % q;
    for m in 1..k {
        c[m * s] = x[m] % q;
        c[(2 * k - m) * s] = (q - x[m] % q) % q;
    }
    Poly(c)
}

/// The coordinates of a ring element of the form of Eq. (7), or `None`.
fn extract(a: &Poly, q: u64, k: usize) -> Option<Vec<u64>> {
    let d = a.0.len();
    let s = d / (2 * k);
    let x: Vec<u64> = (0..k).map(|m| a.0[m * s]).collect();
    (embed(&x, q, d) == *a).then_some(x)
}

fn limbs(x: &BigUint) -> Vec<u64> {
    x.to_u64_digits()
}

fn elements<const K: usize>(q: u64, rng: &mut rand_chacha::ChaCha8Rng) -> Vec<Ext<K>> {
    let mut out = vec![Ext([0; K]), Ext([q - 1; K])];
    let mut minus_one = [0u64; K];
    minus_one[0] = q - 1;
    out.push(Ext(minus_one));
    for m in 0..K {
        let mut unit = [0u64; K];
        unit[m] = 1;
        out.push(Ext(unit));
    }
    for _ in 0..6 {
        out.push(Ext(core::array::from_fn(|_| rng.random_range(0..q))));
    }
    out
}

fn check_field<const K: usize>(q: u64, seed: u64) {
    let mut rng = rng(seed);
    let f = Fqk::<K>::new(q).unwrap();
    assert_eq!(f.modulus(), q);
    let xs = elements::<K>(q, &mut rng);
    let zero = f.zero();
    let one = f.one();
    // The product against the product of the embeddings, at several ring degrees.
    for d in [2 * K, 4 * K, 64.max(2 * K)] {
        let ring = Ring::new(q, d).unwrap();
        for x in &xs {
            for y in &xs {
                let product = ring
                    .mul_schoolbook(&embed(&x.0, q, d), &embed(&y.0, q, d))
                    .unwrap();
                let want = extract(&product, q, K).expect("R_q^H is closed under products");
                assert_eq!(
                    f.mul(x, y).0.to_vec(),
                    want,
                    "q {q} K {K} d {d} x {x:?} y {y:?}"
                );
            }
            // The Frobenius is sigma_q on R_q (characteristic q).
            let sigma = ring
                .automorphism(&embed(&x.0, q, d), (q % (2 * d as u64)) as i64)
                .unwrap();
            assert_eq!(f.frobenius(x).0.to_vec(), extract(&sigma, q, K).unwrap());
        }
    }
    // Field axioms.
    for x in &xs {
        assert_eq!(f.add(x, &zero), f.reduce(x));
        assert_eq!(f.mul(x, &one), f.reduce(x));
        assert_eq!(f.add(x, &f.neg(x)), zero);
        assert_eq!(f.sub(x, x), zero);
        assert_eq!(f.square(x), f.mul(x, x));
        assert_eq!(f.mul(x, &zero), zero);
        match f.inv(x) {
            None => assert!(f.is_zero(x)),
            Some(inv) => {
                assert_eq!(f.mul(x, &inv), one, "q {q} K {K} x {x:?}");
                // x^{-1} = x^{q^K - 2}.
                let e = BigUint::from(q).pow(K as u32) - 2u32;
                assert_eq!(inv, f.pow(x, &limbs(&e)));
                // x^{q^K - 1} = 1.
                let e = BigUint::from(q).pow(K as u32) - 1u32;
                assert_eq!(f.pow(x, &limbs(&e)), one);
            }
        }
        for y in &xs {
            assert_eq!(f.mul(x, y), f.mul(y, x));
            assert_eq!(f.add(x, y), f.add(y, x));
            assert_eq!(f.sub(x, y), f.add(x, &f.neg(y)));
            for z in xs.iter().step_by(3) {
                assert_eq!(f.mul(&f.mul(x, y), z), f.mul(x, &f.mul(y, z)));
                assert_eq!(f.add(&f.add(x, y), z), f.add(x, &f.add(y, z)));
                assert_eq!(f.mul(x, &f.add(y, z)), f.add(&f.mul(x, y), &f.mul(x, z)));
            }
            // The Frobenius is a ring homomorphism.
            assert_eq!(
                f.frobenius(&f.mul(x, y)),
                f.mul(&f.frobenius(x), &f.frobenius(y))
            );
            assert_eq!(
                f.frobenius(&f.add(x, y)),
                f.add(&f.frobenius(x), &f.frobenius(y))
            );
        }
        // The Frobenius is x -> x^q and has order dividing K.
        assert_eq!(f.frobenius(x), f.pow(x, &[q]));
        let mut y = *x;
        for _ in 0..K {
            y = f.frobenius(&y);
        }
        assert_eq!(y, f.reduce(x));
    }
    // Order exactly K: for 0 < j < K some basis element moves (e_1 generates the field).
    let mut e1 = [0u64; K];
    if K > 1 {
        e1[1] = 1;
        let mut y = Ext(e1);
        for j in 1..K {
            y = f.frobenius(&y);
            assert_ne!(y, Ext(e1), "phi^{j}(e_1) = e_1 with K = {K}");
        }
    }
    // Base-field elements are fixed and multiply coordinatewise.
    let c = rng.random_range(0..q);
    assert_eq!(f.frobenius(&f.from_base(c)), f.from_base(c));
    for x in &xs {
        assert_eq!(f.mul(&f.from_base(c), x), f.scale(x, c));
    }
    // Powers: x^0 = 1 (also 0^0), x^(a+b) = x^a x^b, leading zero limbs.
    for x in xs.iter().take(8) {
        assert_eq!(f.pow(x, &[]), one);
        assert_eq!(f.pow(x, &[0, 0]), one);
        assert_eq!(f.pow(x, &[1, 0, 0]), f.reduce(x));
        let a: u128 = rng.random();
        let b: u128 = rng.random();
        let split = |e: u128| vec![e as u64, (e >> 64) as u64];
        let sum = BigUint::from(a) + BigUint::from(b);
        assert_eq!(
            f.pow(x, &limbs(&sum)),
            f.mul(&f.pow(x, &split(a)), &f.pow(x, &split(b)))
        );
    }
    assert_eq!(f.inv(&zero), None);
    assert_eq!(f.inv(&Ext([q; K])), None, "unreduced zero");
    assert!(f.is_zero(&Ext([q; K])));
    // Unreduced coordinates are read modulo q.
    let x = xs[xs.len() - 1];
    let y = xs[xs.len() - 2];
    let x_plus_q = Ext(x.0.map(|c| c + q));
    assert_eq!(f.mul(&x_plus_q, &y), f.mul(&x, &y));
    assert_eq!(f.add(&x_plus_q, &y), f.add(&x, &y));
    assert_eq!(f.frobenius(&x_plus_q), f.frobenius(&x));
}

#[test]
fn fields_of_degree_1_2_4_8_over_the_test_moduli() {
    for (i, q) in ALL_MODULI.into_iter().enumerate() {
        let seed = 40 + i as u64;
        check_field::<1>(q, seed);
        check_field::<2>(q, seed);
        check_field::<4>(q, seed);
        check_field::<8>(q, seed);
    }
    // A small field: F_{13^K}, where products are easy to read.
    check_field::<2>(13, 48);
    check_field::<4>(13, 48);
    check_field::<8>(13, 48);
}

/// The degree-2 field is $\mathbb Z_q[e]/(e^2 - 2)$ with $e = t + t^{-1}$, $t^4 = -1$.
#[test]
fn the_degree_2_field_is_the_square_root_of_2() {
    let f = Fqk::<2>::new(Q61).unwrap();
    let e = Ext([0, 1]);
    assert_eq!(f.square(&e), f.from_base(2));
    // 2 is not a square modulo q = 5 mod 8, so e is not in F_q: phi(e) = -e.
    assert_eq!(f.frobenius(&e), f.neg(&e));
}

#[test]
fn invalid_fields_are_rejected() {
    assert!(matches!(Fqk::<3>::new(Q61), Err(Error::Parameter(_))));
    assert!(matches!(Fqk::<16>::new(Q61), Err(Error::Parameter(_))));
    for q in [97u64, 11, 9223372036854775783] {
        // 1, 3 and 7 mod 8.
        assert_eq!(Fqk::<4>::new(q), Err(Error::ModulusClass(q)), "q = {q}");
    }
    assert_eq!(Fqk::<4>::new(21), Err(Error::Modulus(21)));
    assert!(Fqk::<1>::new(13).is_ok());
}
