//! The challenge sets: $|C|$ against brute force and the Delannoy recurrence, membership, errors,
//! and chi-square tests of the samplers over all of $C$ (and of the sparse set) for tiny
//! $(d, \omega)$, with fixed seeds and the threshold at $1 - 10^{-4}$
//! (`common::chi_square_threshold`). The exact enumeration of the samplers' draws is in the unit
//! tests (`src/l1ball/tests.rs`).

mod common;

use common::assert_chi_square;
use hachi_co_sampling::l1ball::{
    L1Ball, contains, contains_sparse, count, count_sparse, sample, sample_sparse,
};
use hachi_co_sampling::{Error, ShakePrg};
use num_bigint::BigUint;
use std::collections::BTreeMap;

/// Every vector of $[-\omega, \omega]^d$ with $\ell_1$ norm at most $\omega$, in lexicographic order.
fn ball(d: usize, omega: u64) -> Vec<Vec<i64>> {
    let w = omega as i64;
    let mut out = vec![vec![]];
    for _ in 0..d {
        out = out
            .into_iter()
            .flat_map(|prefix: Vec<i64>| {
                let used: i64 = prefix.iter().map(|x| x.abs()).sum();
                (-w..=w).filter(move |x| used + x.abs() <= w).map(move |x| {
                    let mut v = prefix.clone();
                    v.push(x);
                    v
                })
            })
            .collect();
    }
    out
}

/// Brute-force count over the whole box $[-\omega, \omega]^d$ (no pruning), for small sizes.
fn brute_force_count(d: usize, omega: u64) -> u64 {
    let w = omega as i64;
    let side = 2 * w + 1;
    let total = side.pow(d as u32);
    (0..total)
        .filter(|&code| {
            let mut index = code;
            let mut norm = 0;
            for _ in 0..d {
                norm += (index % side - w).abs();
                index /= side;
            }
            norm <= w
        })
        .count() as u64
}

#[test]
fn count_equals_brute_force() {
    for d in 0..=5usize {
        for omega in 0..=6u64 {
            let brute = brute_force_count(d, omega);
            assert_eq!(
                count(d, omega),
                BigUint::from(brute),
                "d = {d}, omega = {omega}"
            );
            assert_eq!(ball(d, omega).len() as u64, brute);
        }
    }
}

#[test]
fn count_satisfies_the_delannoy_identities() {
    // Central Delannoy numbers D(n, n), OEIS A001850 (a standard result; D(0..=6) are also
    // re-derived by `count_equals_brute_force`).
    for (n, expected) in [1u64, 3, 13, 63, 321, 1683, 8989, 48639, 265729]
        .into_iter()
        .enumerate()
    {
        assert_eq!(count(n, n as u64), BigUint::from(expected), "D({n}, {n})");
    }
    // D(d, w) = D(d - 1, w) + D(d, w - 1) + D(d - 1, w - 1), and symmetry, up to large values.
    for d in 1..=40usize {
        for w in 1..=40u64 {
            let lhs = count(d, w);
            assert_eq!(
                lhs,
                count(d - 1, w) + count(d, w - 1) + count(d - 1, w - 1),
                "D({d}, {w})"
            );
            assert_eq!(lhs, count(w as usize, d as u64), "symmetry D({d}, {w})");
        }
    }
    // Edges: D(0, w) = D(d, 0) = 1 and D(1, w) = 2w + 1.
    assert_eq!(count(0, u64::MAX), BigUint::from(1u32));
    assert_eq!(count(1 << 40, 0), BigUint::from(1u32));
    assert_eq!(count(1, u64::MAX), BigUint::from(u64::MAX) * 2u32 + 1u32);
}

#[test]
fn sparse_count_equals_brute_force() {
    for d in 0..=6usize {
        for omega in 0..=7u64 {
            let brute = ball(d, d as u64)
                .into_iter()
                .filter(|c| {
                    c.iter().all(|x| x.abs() <= 1)
                        && c.iter().filter(|x| **x != 0).count() as u64 == omega
                })
                .count();
            assert_eq!(
                count_sparse(d, omega),
                BigUint::from(brute),
                "d = {d}, omega = {omega}"
            );
        }
    }
}

#[test]
fn membership_is_exact() {
    assert!(contains(&[], 0));
    assert!(contains(&[0, 0, 0], 0));
    assert!(contains(&[1, -2, 0], 3));
    assert!(!contains(&[1, -2, 0], 2));
    // No overflow: sums beyond 2^64.
    assert!(!contains(&[i64::MIN, i64::MIN, i64::MIN], u64::MAX));
    assert!(contains(&[i64::MIN, i64::MAX], u64::MAX));
    assert!(!contains(&[i64::MIN, i64::MAX, 1], u64::MAX));
    assert!(contains_sparse(&[0, 1, -1, 0], 2));
    assert!(!contains_sparse(&[0, 1, -1, 0], 1));
    assert!(!contains_sparse(&[0, 2, 0, 0], 1));
    assert!(contains_sparse(&[], 0));
}

#[test]
fn errors_and_degenerate_sets() {
    let mut rng = ShakePrg::new(&[9; 32], b"errors", 0);
    assert!(matches!(
        sample_sparse(&mut rng, 3, 4),
        Err(Error::InvalidParameter(_))
    ));
    assert!(matches!(
        L1Ball::new(4, 1 << 63),
        Err(Error::InvalidParameter(_))
    ));
    assert!(matches!(
        sample(&mut rng, 4, u64::MAX),
        Err(Error::InvalidParameter(_))
    ));
    assert_eq!(sample(&mut rng, 0, 5), Ok(vec![]));
    assert_eq!(sample(&mut rng, 5, 0), Ok(vec![0; 5]));
    assert_eq!(sample_sparse(&mut rng, 4, 0), Ok(vec![0; 4]));
    // The largest omega accepted: a 1-dimensional ball [-(2^63 - 1), 2^63 - 1].
    let x = sample(&mut rng, 1, i64::MAX as u64).unwrap();
    assert!(contains(&x, i64::MAX as u64));
}

#[test]
fn samples_at_protocol_sizes_lie_in_the_set() {
    // d = 1024 with the omega of a secure-sized ball, and the toy presets' (16, 2).
    let mut rng = ShakePrg::new(&[0x22; 32], b"membership", 0);
    for (d, omega) in [(16usize, 2u64), (1024, 40), (1024, 200), (64, 1000)] {
        let ball = L1Ball::new(d, omega).unwrap();
        let mut max_norm = 0;
        for _ in 0..200 {
            let c = ball.sample(&mut rng).unwrap();
            assert_eq!(c.len(), d);
            assert!(contains(&c, omega), "d = {d}, omega = {omega}");
            max_norm = max_norm.max(c.iter().map(|x| x.unsigned_abs()).sum::<u64>());
        }
        // Most of the ball's mass is near the sphere ||c||_1 = omega.
        assert!(
            max_norm + 5 >= omega,
            "d = {d}, omega = {omega}: max norm {max_norm}"
        );
    }
    for (d, omega) in [(16usize, 2u64), (1024, 40), (1024, 1024)] {
        for _ in 0..50 {
            let c = sample_sparse(&mut rng, d, omega).unwrap();
            assert_eq!(c.len(), d);
            assert!(contains_sparse(&c, omega));
        }
    }
}

/// Draw `rounds * |set|` samples and run the chi-square test over every element of `set`.
fn chi_square_over(
    what: &str,
    set: &[Vec<i64>],
    rounds: usize,
    mut draw: impl FnMut() -> Vec<i64>,
) {
    let index: BTreeMap<&Vec<i64>, usize> = set.iter().enumerate().map(|(i, c)| (c, i)).collect();
    let mut counts = vec![0u64; set.len()];
    for _ in 0..rounds * set.len() {
        let c = draw();
        counts[*index
            .get(&c)
            .unwrap_or_else(|| panic!("{what}: {c:?} is not in the set"))] += 1;
    }
    assert!(
        counts.iter().all(|c| *c > 0),
        "{what}: an element was never drawn"
    );
    assert_chi_square(what, &counts, &vec![rounds as f64; set.len()]);
}

#[test]
fn l1_ball_samples_pass_chi_square_over_all_of_c() {
    // ShakePrg with seed [0x5a; 32], domain "chi2/l1ball", index i for the i-th case.
    for (i, (d, omega)) in [
        (4usize, 3u64),
        (3, 3),
        (2, 5),
        (5, 2),
        (1, 6),
        (6, 1),
        (8, 2),
    ]
    .into_iter()
    .enumerate()
    {
        let set = ball(d, omega);
        assert_eq!(BigUint::from(set.len()), count(d, omega));
        let sampler = L1Ball::new(d, omega).unwrap();
        let mut rng = ShakePrg::new(&[0x5a; 32], b"chi2/l1ball", i as u64);
        chi_square_over(
            &format!("l1 ball d = {d}, omega = {omega}, |C| = {}", set.len()),
            &set,
            200,
            || sampler.sample(&mut rng).unwrap(),
        );
    }
}

#[test]
fn sparse_samples_pass_chi_square_over_the_whole_set() {
    for (i, (d, omega)) in [(5usize, 2u64), (4, 4), (6, 3)].into_iter().enumerate() {
        let set: Vec<Vec<i64>> = ball(d, d as u64)
            .into_iter()
            .filter(|c| contains_sparse(c, omega))
            .collect();
        let mut rng = ShakePrg::new(&[0x5b; 32], b"chi2/sparse", i as u64);
        chi_square_over(
            &format!("sparse d = {d}, omega = {omega}, size {}", set.len()),
            &set,
            200,
            || sample_sparse(&mut rng, d, omega).unwrap(),
        );
    }
}
