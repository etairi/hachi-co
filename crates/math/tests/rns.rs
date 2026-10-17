//! Accumulated products: [`Plan::inner`] and [`Plan::mat_vec`] with $1$, NADDS$-1$, NADDS,
//! NADDS$+1$ and $3 \cdot$NADDS$+5$ products against exact big-integer sums reduced modulo $q$,
//! for the test moduli and $d \in \lbrace 16, 256, 1024, 2048 \rbrace$, on worst-case operands
//! (the centered extremes, which give the largest accumulated coefficients the prime count must
//! cover) and on random operands (seeds 20-29).
mod common;

use common::*;
use hachi_co_math::{
    ntt::NTT_PRIMES,
    rns::{NADDS, Plan},
};
use num_bigint::{BigInt, BigUint};

const COUNTS: [usize; 5] = [1, NADDS - 1, NADDS, NADDS + 1, 3 * NADDS + 5];

fn lift(v: &[u64], q: u64) -> Vec<i128> {
    v.iter().map(|x| centered(*x, q)).collect()
}

fn reduce_all(v: &[BigInt], q: u64) -> Vec<u64> {
    v.iter().map(|x| big_mod(x, q)).collect()
}

/// The fast exact reference against the slow one, including the extremes of the lift.
#[test]
fn the_wide_reference_equals_the_bigint_reference() {
    let mut rng = rng(20);
    for q in ALL_MODULI {
        for d in [2usize, 16, 64] {
            let top = (q - 1) / 2;
            let mut ops = vec![vec![top; d], vec![top + 1; d], residues(&mut rng, q, d)];
            ops.push(residues(&mut rng, q, d));
            let a: Vec<Vec<i128>> = ops.iter().map(|v| lift(v, q)).collect();
            let b: Vec<Vec<i128>> = ops.iter().rev().map(|v| lift(v, q)).collect();
            let mut want = vec![BigInt::from(0); d];
            for (x, y) in a.iter().zip(&b) {
                for (w, t) in want.iter_mut().zip(negacyclic_bigint(x, y)) {
                    *w += t;
                }
            }
            assert_eq!(negacyclic_sum_wide(&a, &b), want);
        }
    }
}

/// Worst-case operands repeated `count` times: the exact sum is `count` times one exact product.
/// The pattern (all $(q-1)/2$) $\times$ (all $(q-1)/2$) puts $d ((q-1)/2)^2$ in coefficient
/// $d - 1$ of each product, the largest absolute value a product of centered residues has, so
/// the NADDS-product chunks reach the bound the prime count is chosen for.
#[test]
fn accumulated_worst_case_products_equal_big_integer_sums() {
    for q in ALL_MODULI {
        let (top, bottom) = ((q - 1) / 2, q.div_ceil(2));
        for d in [16usize, 256, 1024, 2048] {
            let plan = Plan::new(q, d).unwrap();
            let patterns: Vec<(Vec<u64>, Vec<u64>)> = vec![
                (vec![top; d], vec![top; d]),
                (vec![top; d], vec![bottom; d]),
                (vec![q - 1; d], vec![q - 1; d]),
                (
                    (0..d).map(|i| if i == 0 { top } else { bottom }).collect(),
                    vec![top; d],
                ),
            ];
            for (x, y) in &patterns {
                let one = negacyclic_sum_wide(&[lift(x, q)], &[lift(y, q)]);
                // The extreme is reached: coefficient d-1 of top*top is d * top^2.
                if x[0] == top && y == &vec![top; d] && x[1] == top {
                    assert_eq!(one[d - 1], BigInt::from(d) * BigInt::from(top).pow(2));
                }
                for count in COUNTS {
                    let a = vec![x.as_slice(); count];
                    let b = vec![y.as_slice(); count];
                    let got = plan.inner(&a, &b).unwrap();
                    let want: Vec<u64> = one
                        .iter()
                        .map(|c| big_mod(&(c * BigInt::from(count)), q))
                        .collect();
                    assert_eq!(got, want, "q {q} d {d} count {count}");
                }
            }
        }
    }
}

/// Random distinct operands with the exact sum of all products. At $d \le 256$ for every modulus;
/// at $d \in \lbrace 1024, 2048 \rbrace$ for the two largest moduli (the exact reference is
/// quadratic in $d$ per product).
#[test]
fn accumulated_random_products_equal_big_integer_sums() {
    let mut rng = rng(21);
    let mut cases: Vec<(u64, usize)> = Vec::new();
    for q in ALL_MODULI {
        cases.push((q, 16));
        cases.push((q, 256));
    }
    for q in [Q61, Q63] {
        cases.push((q, 1024));
        cases.push((q, 2048));
    }
    for (q, d) in cases {
        let plan = Plan::new(q, d).unwrap();
        let max = *COUNTS.iter().max().unwrap();
        // Random operands, with every eighth pair at the centered extremes with random signs.
        let top = (q - 1) / 2;
        let mut a = Vec::with_capacity(max);
        let mut b = Vec::with_capacity(max);
        for t in 0..max {
            if t % 8 == 0 {
                let signs = residues(&mut rng, 2, 2 * d);
                a.push(
                    signs[..d]
                        .iter()
                        .map(|s| if *s == 0 { top } else { top + 1 })
                        .collect::<Vec<u64>>(),
                );
                b.push(
                    signs[d..]
                        .iter()
                        .map(|s| if *s == 0 { top } else { top + 1 })
                        .collect::<Vec<u64>>(),
                );
            } else {
                a.push(residues(&mut rng, q, d));
                b.push(residues(&mut rng, q, d));
            }
        }
        // The exact sums of the first `count` products, as prefix sums (COUNTS is increasing).
        let mut acc = vec![Wide::default(); d];
        let mut added = 0;
        for count in COUNTS {
            while added < count {
                add_negacyclic_product(&mut acc, &lift(&a[added], q), &lift(&b[added], q));
                added += 1;
            }
            let exact: Vec<BigInt> = acc.iter().map(|w| w.to_bigint()).collect();
            let got = plan.inner(&a[..count], &b[..count]).unwrap();
            assert_eq!(got, reduce_all(&exact, q), "q {q} d {d} count {count}");
        }
    }
}

/// Matrix rows of every count, against the inner products and the exact sums.
#[test]
fn matrix_products_equal_row_inner_products() {
    let mut rng = rng(22);
    for q in [Q32, Q61, Q63] {
        let d = 16;
        let plan = Plan::new(q, d).unwrap();
        for count in COUNTS {
            let v: Vec<Vec<u64>> = (0..count).map(|_| residues(&mut rng, q, d)).collect();
            let m: Vec<Vec<Vec<u64>>> = (0..3)
                .map(|r| {
                    (0..count)
                        .map(|_| {
                            if r == 0 {
                                vec![(q - 1) / 2; d]
                            } else {
                                residues(&mut rng, q, d)
                            }
                        })
                        .collect()
                })
                .collect();
            let got = plan.mat_vec(&m, &v).unwrap();
            assert_eq!(got.len(), 3);
            let lv: Vec<Vec<i128>> = v.iter().map(|x| lift(x, q)).collect();
            for (row, got) in m.iter().zip(&got) {
                assert_eq!(*got, plan.inner(row, &v).unwrap());
                let lrow: Vec<Vec<i128>> = row.iter().map(|x| lift(x, q)).collect();
                assert_eq!(
                    *got,
                    reduce_all(&negacyclic_sum_wide(&lrow, &lv), q),
                    "q {q} count {count}"
                );
            }
        }
        // An empty matrix and empty rows.
        let empty: Vec<Vec<Vec<u64>>> = Vec::new();
        assert!(plan.mat_vec(&empty, &[vec![0u64; d]]).unwrap().is_empty());
        let rows: Vec<Vec<Vec<u64>>> = vec![Vec::new(), Vec::new()];
        assert_eq!(
            plan.mat_vec::<_, Vec<u64>, Vec<u64>>(&rows, &[]).unwrap(),
            vec![vec![0; d]; 2]
        );
    }
}

/// Moduli that are not prime and the smallest moduli: the plan is exact for any
/// $2 \le q < 2^{63}$.
#[test]
fn products_are_exact_for_any_modulus_below_2_to_63() {
    let mut rng = rng(23);
    for q in [2u64, 3, 12, 13, (1 << 62) - 1, (1 << 63) - 1] {
        for d in [2usize, 64] {
            let plan = Plan::new(q, d).unwrap();
            let a: Vec<Vec<u64>> = (0..NADDS + 1).map(|_| residues(&mut rng, q, d)).collect();
            let b: Vec<Vec<u64>> = (0..NADDS + 1).map(|_| residues(&mut rng, q, d)).collect();
            let la: Vec<Vec<i128>> = a.iter().map(|v| lift(v, q)).collect();
            let lb: Vec<Vec<i128>> = b.iter().map(|v| lift(v, q)).collect();
            assert_eq!(
                plan.inner(&a, &b).unwrap(),
                reduce_all(&negacyclic_sum_wide(&la, &lb), q),
                "q {q} d {d}"
            );
            assert_eq!(plan.modulus(), q);
            assert_eq!(plan.degree(), d);
        }
    }
}

/// The prime-count boundary: the largest modulus $q$ with $(q-1)^2 n \cdot \mathrm{NADDS} < P_K$
/// (the product of the first $K$ primes) selects $K$ primes with the least slack the rule allows,
/// and $q + 1$ selects $K + 1$. Worst-case accumulations are exact on both sides. At the tight
/// modulus, NADDS products of (all $q - 1$) $\times$ (all $q - 1$) would reach $P_K$ without the
/// centered lift (they are $n (q-1)^2$ per coefficient), so this also tests the lift.
#[test]
fn products_are_exact_at_the_prime_count_boundary() {
    for (k, n) in [(1usize, 16usize), (1, 4096), (2, 16), (2, 4096)] {
        let p_k: BigUint = NTT_PRIMES[..k]
            .iter()
            .map(|(p, _)| BigUint::from(*p))
            .product();
        let scale = BigUint::from(n * NADDS);
        let q = u64::try_from(((&p_k - 1u32) / &scale).sqrt()).unwrap() + 1;
        assert!(BigUint::from(q - 1).pow(2) * &scale < p_k);
        assert!(BigUint::from(q).pow(2) * &scale >= p_k);
        assert_eq!(Plan::new(q, n).unwrap().primes().len(), k, "q {q} n {n}");
        assert_eq!(
            Plan::new(q + 1, n).unwrap().primes().len(),
            k + 1,
            "q {} n {n}",
            q + 1
        );
        for q in [q, q + 1] {
            let plan = Plan::new(q, n).unwrap();
            let (top, bottom) = ((q - 1) / 2, q.div_ceil(2));
            for (x, y) in [
                (vec![q - 1; n], vec![q - 1; n]),
                (vec![top; n], vec![top; n]),
                (vec![bottom; n], vec![top; n]),
            ] {
                let one = negacyclic_sum_wide(&[lift(&x, q)], &[lift(&y, q)]);
                for count in COUNTS {
                    let got = plan
                        .inner(&vec![x.as_slice(); count], &vec![y.as_slice(); count])
                        .unwrap();
                    let want: Vec<u64> = one
                        .iter()
                        .map(|c| big_mod(&(c * BigInt::from(count)), q))
                        .collect();
                    assert_eq!(got, want, "q {q} n {n} count {count}");
                }
            }
        }
    }
}
