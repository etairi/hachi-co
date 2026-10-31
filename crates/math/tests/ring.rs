//! $\mathcal R_q$: the exact product against the schoolbook reference for the test moduli and
//! $d \in \lbrace 2, 16, 256, 1024, 2048, 4096 \rbrace$ on worst-case and random operands; the
//! schoolbook reference against num-bigint; the other operations against their definitions
//! (seeds 10-19).
mod common;

use common::*;
use hachi_co_math::{Error, Poly, Ring};
use rand::RngExt;

/// Worst-case operands: all $q-1$ (centered $-1$), all $(q-1)/2$ and all $(q+1)/2$ (the centered
/// extremes $\pm(q-1)/2$), alternating extremes, and the extreme with a sign flip after the first
/// coefficient (maximal coefficient 0 of the product).
fn worst_cases(q: u64, d: usize) -> Vec<Vec<u64>> {
    let (top, bottom) = ((q - 1) / 2, q.div_ceil(2));
    vec![
        vec![q - 1; d],
        vec![top; d],
        vec![bottom; d],
        (0..d)
            .map(|i| if i % 2 == 0 { top } else { bottom })
            .collect(),
        (0..d).map(|i| if i == 0 { top } else { bottom }).collect(),
    ]
}

#[test]
fn products_equal_the_schoolbook_reference() {
    let mut rng = rng(10);
    for q in ALL_MODULI {
        for d in [2usize, 16, 256, 1024, 2048] {
            let ring = Ring::new(q, d).unwrap();
            let mut operands = worst_cases(q, d);
            operands.push(residues(&mut rng, q, d));
            operands.push(residues(&mut rng, q, d));
            // Large d: all pairs of the worst cases cost too much schoolbook time; use the pairs
            // (x, x), (x, next) and (x, random).
            let n = operands.len();
            for i in 0..n {
                for j in [i, (i + 1) % n, n - 1] {
                    let a = Poly(operands[i].clone());
                    let b = Poly(operands[j].clone());
                    let fast = ring.mul(&a, &b).unwrap();
                    let slow = ring.mul_schoolbook(&a, &b).unwrap();
                    assert_eq!(fast, slow, "q {q} d {d} operands {i} {j}");
                }
            }
        }
    }
}

#[test]
fn products_of_degree_4096_equal_the_schoolbook_reference() {
    let mut rng = rng(11);
    for q in [Q61, Q63] {
        let ring = Ring::new(q, 4096).unwrap();
        let a = Poly(vec![(q - 1) / 2; 4096]);
        let b = Poly(residues(&mut rng, q, 4096));
        assert_eq!(
            ring.mul(&a, &a).unwrap(),
            ring.mul_schoolbook(&a, &a).unwrap()
        );
        assert_eq!(
            ring.mul(&a, &b).unwrap(),
            ring.mul_schoolbook(&a, &b).unwrap()
        );
    }
}

/// The schoolbook reference against the exact product over $\mathbb Z$ (num-bigint) reduced
/// modulo $q$, on the worst cases and random operands.
#[test]
fn the_schoolbook_reference_equals_big_integer_products() {
    let mut rng = rng(12);
    for q in ALL_MODULI {
        for d in [2usize, 16, 64] {
            let ring = Ring::new(q, d).unwrap();
            let mut operands = worst_cases(q, d);
            operands.push(residues(&mut rng, q, d));
            for x in &operands {
                for y in &operands {
                    let exact = negacyclic_bigint(
                        &x.iter().map(|c| centered(*c, q)).collect::<Vec<_>>(),
                        &y.iter().map(|c| centered(*c, q)).collect::<Vec<_>>(),
                    );
                    let want: Vec<u64> = exact.iter().map(|c| big_mod(c, q)).collect();
                    let got = ring
                        .mul_schoolbook(&Poly(x.clone()), &Poly(y.clone()))
                        .unwrap();
                    assert_eq!(got.0, want, "q {q} d {d}");
                }
            }
        }
    }
}

#[test]
fn elementwise_operations_equal_their_definitions() {
    let mut rng = rng(13);
    for q in ALL_MODULI {
        let d = 16;
        let ring = Ring::new(q, d).unwrap();
        assert_eq!(ring.modulus(), q);
        assert_eq!(ring.degree(), d);
        assert_eq!(ring.zq().modulus(), q);
        let a = Poly(residues(&mut rng, q, d));
        let b = Poly(residues(&mut rng, q, d));
        let big = |x: u64| num_bigint::BigInt::from(x);
        for j in 0..d {
            assert_eq!(
                ring.add(&a, &b).unwrap().0[j],
                big_mod(&(big(a.0[j]) + big(b.0[j])), q)
            );
            assert_eq!(
                ring.sub(&a, &b).unwrap().0[j],
                big_mod(&(big(a.0[j]) - big(b.0[j])), q)
            );
            assert_eq!(ring.neg(&a).unwrap().0[j], big_mod(&-big(a.0[j]), q));
        }
        // Scalar products (Shoup) against u128 remainders, for boundary and unreduced scalars.
        for c in [0, 1, 2, q / 2, q - 1, q, q + 1, u64::MAX] {
            let got = ring.scalar_mul(c, &a).unwrap();
            for j in 0..d {
                assert_eq!(got.0[j], big_mod(&(big(c) * big(a.0[j])), q), "c {c}");
            }
        }
        // Constructors and centered values.
        let signed: Vec<i64> = (0..d as i64).map(|i| i - 8).collect();
        let p = ring.from_i64(&signed).unwrap();
        assert_eq!(ring.to_centered(&p).unwrap(), signed);
        assert_eq!(ring.inf_norm(&p).unwrap(), 8);
        let unreduced: Vec<u64> = (0..d as u64).map(|i| q + i).collect();
        assert_eq!(
            ring.from_coeffs(&unreduced).unwrap().0,
            (0..d as u64).collect::<Vec<_>>()
        );
        assert_eq!(
            ring.inf_norm(&Poly(vec![q.div_ceil(2); d])).unwrap(),
            (q - 1) / 2
        );
        assert_eq!(ring.inf_norm(&ring.zero()).unwrap(), 0);
        // Unreduced operands are read modulo q.
        let a_plus_q = Poly(a.0.iter().map(|x| x + q).collect());
        assert_eq!(ring.mul(&a_plus_q, &b).unwrap(), ring.mul(&a, &b).unwrap());
        assert_eq!(ring.add(&a_plus_q, &b).unwrap(), ring.add(&a, &b).unwrap());
    }
}

#[test]
fn monomials_and_automorphisms_equal_their_definitions() {
    let mut rng = rng(14);
    for q in [Q32, Q61] {
        for d in [2usize, 16, 64] {
            let ring = Ring::new(q, d).unwrap();
            let a = Poly(residues(&mut rng, q, d));
            let b = Poly(residues(&mut rng, q, d));
            let two_d = 2 * d as i64;
            for e in [
                -two_d - 1,
                -3,
                -1,
                0,
                1,
                d as i64 - 1,
                d as i64,
                two_d - 1,
                two_d,
                5 * two_d + 3,
            ] {
                let x_e = ring.monomial(e);
                assert_eq!(
                    ring.mul_monomial(&a, e).unwrap(),
                    ring.mul_schoolbook(&a, &x_e).unwrap(),
                    "e {e}"
                );
                assert_eq!(ring.mul_monomial(&ring.monomial(0), e).unwrap(), x_e);
            }
            // X^d = -1 and X^{2d} = 1.
            assert_eq!(
                ring.monomial(d as i64),
                ring.neg(&ring.monomial(0)).unwrap()
            );
            assert_eq!(ring.monomial(two_d), ring.monomial(0));
            for i in (-two_d + 1..two_d).step_by(2) {
                let sigma = |x: &Poly| ring.automorphism(x, i).unwrap();
                // sigma_i(a) by its definition: a(X^i) = sum_j a_j X^{ij}.
                let mut want = ring.zero();
                for (j, c) in a.0.iter().enumerate() {
                    let term = ring.scalar_mul(*c, &ring.monomial(i * j as i64)).unwrap();
                    want = ring.add(&want, &term).unwrap();
                }
                assert_eq!(sigma(&a), want, "i {i}");
                // A ring automorphism: sigma_i(ab) = sigma_i(a) sigma_i(b), additive, unital.
                assert_eq!(
                    sigma(&ring.mul(&a, &b).unwrap()),
                    ring.mul(&sigma(&a), &sigma(&b)).unwrap()
                );
                assert_eq!(
                    sigma(&ring.add(&a, &b).unwrap()),
                    ring.add(&sigma(&a), &sigma(&b)).unwrap()
                );
                // sigma_i sigma_j = sigma_{ij}.
                for jj in [-1i64, 3, 5] {
                    let lhs = ring
                        .automorphism(&ring.automorphism(&a, jj).unwrap(), i)
                        .unwrap();
                    assert_eq!(lhs, ring.automorphism(&a, i * jj).unwrap());
                }
            }
            assert_eq!(ring.automorphism(&a, 1).unwrap(), a);
            assert!(matches!(ring.automorphism(&a, 2), Err(Error::Parameter(_))));
            assert!(matches!(
                ring.automorphism(&a, two_d),
                Err(Error::Parameter(_))
            ));
        }
    }
}

#[test]
fn inner_products_and_matrix_products_equal_sums_of_products() {
    let mut rng = rng(15);
    for q in [Q32, Q63] {
        let d = 64;
        let ring = Ring::new(q, d).unwrap();
        let a: Vec<Poly> = (0..5).map(|_| Poly(residues(&mut rng, q, d))).collect();
        let b: Vec<Poly> = (0..5).map(|_| Poly(residues(&mut rng, q, d))).collect();
        let mut want = ring.zero();
        for (x, y) in a.iter().zip(&b) {
            want = ring
                .add(&want, &ring.mul_schoolbook(x, y).unwrap())
                .unwrap();
        }
        assert_eq!(ring.inner(&a, &b).unwrap(), want);
        let m = vec![a.clone(), b.clone(), a.iter().rev().cloned().collect()];
        let got = ring.mat_vec(&m, &b).unwrap();
        for (row, got) in m.iter().zip(&got) {
            assert_eq!(*got, ring.inner(row, &b).unwrap());
        }
        // The same matrix, transformed once, against two vectors.
        let prepared = ring.prepare(&m).unwrap();
        assert_eq!(ring.mat_vec_prepared(&prepared, &b).unwrap(), got);
        assert_eq!(
            ring.mat_vec_prepared(&prepared, &a).unwrap(),
            ring.mat_vec(&m, &a).unwrap()
        );
        assert!(ring.mat_vec_prepared(&prepared, &a[1..]).is_err());
        assert!(ring.prepare(&[vec![Poly(vec![0; d - 1])]]).is_err());
    }
}

#[test]
fn padded_inner_products_are_products_that_do_not_wrap() {
    let mut rng = rng(16);
    for q in [Q32, Q63] {
        let d = 32;
        let ring = Ring::new(q, d).unwrap();
        let ring2 = Ring::new(q, 2 * d).unwrap();
        // 150 pairs: more than one chunk of NADDS, and every length up to d.
        let short = |rng: &mut _, i: usize| Poly(residues(rng, q, i % d + 1));
        let a: Vec<Poly> = (0..150).map(|i| short(&mut rng, i)).collect();
        let b: Vec<Poly> = (0..150).map(|i| short(&mut rng, 2 * i)).collect();
        let lifted = |x: &Poly| {
            let mut c = x.0.clone();
            c.resize(2 * d, 0);
            Poly(c)
        };
        let mut want = ring2.zero();
        for (x, y) in a.iter().zip(&b) {
            let product = ring2.mul(&lifted(x), &lifted(y)).unwrap();
            want = ring2.add(&want, &product).unwrap();
        }
        assert_eq!(ring2.inner_padded(&a, &b).unwrap(), want);
        // Reduced modulo X^d + 1, the padded sum is the inner product of R_q.
        let full = |p: &Poly| {
            let mut c = p.0.clone();
            c.resize(d, 0);
            Poly(c)
        };
        let (fa, fb): (Vec<Poly>, Vec<Poly>) =
            (a.iter().map(full).collect(), b.iter().map(full).collect());
        let (low, high) = want.0.split_at(d);
        let reduced: Vec<u64> = low
            .iter()
            .zip(high)
            .map(|(l, h)| ring.zq().sub(*l, *h))
            .collect();
        assert_eq!(ring.inner(&fa, &fb).unwrap().0, reduced);
    }
    let ring = Ring::new(Q32, 4).unwrap();
    let long = [Poly(vec![0; 5])];
    assert!(ring.inner_padded(&long, &long).is_err());
    assert!(ring.inner_padded(&[ring.zero()], &[]).is_err());
}

#[test]
fn sparse_products_equal_the_exact_product() {
    let mut rng = rng(17);
    for q in [5u64, Q32, Q52, Q61, Q63] {
        for d in [2usize, 16, 256] {
            let ring = Ring::new(q, d).unwrap();
            for round in 0..12 {
                // Every kind of challenge: empty, all of X^j, signs, and larger coefficients.
                let mut c = vec![0i64; d];
                let weight = [0, 1, 2, d / 2, d][round % 5];
                for _ in 0..weight {
                    let j = rng.random_range(0..d);
                    c[j] = [1, -1, 2, -2, 7, -3, i64::MAX, i64::MIN + 1][round % 8];
                }
                if round == 7 {
                    c[d - 1] = 1;
                    c[0] = -1;
                }
                let a = Poly(residues(&mut rng, q, d));
                let start = Poly(residues(&mut rng, q, d));
                let mut acc = start.clone();
                let sparse = ring.sparse(&c).unwrap();
                ring.add_sparse_product(&mut acc, &sparse, &a).unwrap();
                let product = ring.mul(&ring.from_i64(&c).unwrap(), &a).unwrap();
                assert_eq!(
                    acc,
                    ring.add(&start, &product).unwrap(),
                    "q {q} d {d} c {c:?}"
                );
                // Coefficients read modulo q, as everywhere in the ring.
                let unreduced = Poly(a.0.iter().map(|x| x + q).collect());
                if q < 1 << 62 {
                    let mut again = start.clone();
                    ring.add_sparse_product(&mut again, &sparse, &unreduced)
                        .unwrap();
                    assert_eq!(again, acc, "q {q} d {d}");
                }
            }
        }
    }
    let ring = Ring::new(Q32, 16).unwrap();
    assert!(ring.sparse(&[0; 15]).is_err());
    let sparse = ring.sparse(&[0; 16]).unwrap();
    let mut short = Poly(vec![0; 15]);
    assert!(
        ring.add_sparse_product(&mut short, &sparse, &ring.zero())
            .is_err()
    );
    let mut ok = ring.zero();
    assert!(ring.add_sparse_product(&mut ok, &sparse, &short).is_err());
}

#[test]
fn invalid_rings_and_shapes_are_rejected() {
    assert_eq!(Ring::new(15, 16).unwrap_err(), Error::Modulus(15));
    assert_eq!(Ring::new(1 << 63, 16).unwrap_err(), Error::Modulus(1 << 63));
    for d in [0usize, 1, 3, 24, 8192] {
        assert!(
            matches!(Ring::new(Q32, d), Err(Error::Parameter(_))),
            "d = {d}"
        );
    }
    let ring = Ring::new(Q32, 16).unwrap();
    let short = Poly(vec![0; 15]);
    let ok = ring.zero();
    let length = Err(Error::Length {
        expected: 16,
        actual: 15,
    });
    assert_eq!(ring.add(&ok, &short), length);
    assert_eq!(ring.sub(&short, &ok), length);
    assert_eq!(ring.neg(&short), length);
    assert_eq!(ring.mul(&ok, &short), length);
    assert_eq!(ring.mul_schoolbook(&ok, &short), length);
    assert_eq!(ring.scalar_mul(3, &short), length);
    assert_eq!(ring.mul_monomial(&short, 1), length);
    assert_eq!(ring.automorphism(&short, 1), length);
    assert_eq!(ring.inf_norm(&short), length.map(|_: Poly| 0));
    assert!(ring.from_coeffs(&[0; 15]).is_err());
    assert!(ring.from_i64(&[0; 17]).is_err());
    assert!(ring.inner(std::slice::from_ref(&ok), &[]).is_err());
    assert!(
        ring.inner(std::slice::from_ref(&ok), std::slice::from_ref(&short))
            .is_err()
    );
    assert!(
        ring.mat_vec(&[vec![ok.clone()]], &[ok.clone(), ok])
            .is_err()
    );
}
