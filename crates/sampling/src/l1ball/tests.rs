//! Exactness of the challenge samplers: every sequence of draws is enumerated
//! (`crate::enumerate`), and the resulting output distribution must be exactly uniform on the
//! brute-force enumeration of the set. This is a proof of uniformity for the parameters listed,
//! given exact uniform draws (which `tests/uniform.rs` checks for [`uniform_below`]).
//!
//! [`uniform_below`]: crate::uniform::uniform_below

use super::*;
use crate::enumerate::exact;
use num_bigint::BigUint;

/// Every $c \in [-\omega, \omega]^d$ with $\lVert c \rVert_1 \le \omega$, by brute force.
fn brute_force_ball(d: usize, omega: u64) -> BTreeSet<Vec<i64>> {
    let w = omega as i64;
    let mut out = BTreeSet::new();
    let mut c = vec![-w; d];
    loop {
        if c.iter().map(|x| x.unsigned_abs()).sum::<u64>() <= omega {
            out.insert(c.clone());
        }
        // Odometer over [-w, w]^d.
        let mut k = 0;
        loop {
            if k == d {
                return out;
            }
            if c[k] < w {
                c[k] += 1;
                break;
            }
            c[k] = -w;
            k += 1;
        }
    }
}

/// Every $c \in \lbrace -1, 0, 1 \rbrace^d$ with exactly $\omega$ nonzero entries.
fn brute_force_sparse(d: usize, omega: u64) -> BTreeSet<Vec<i64>> {
    brute_force_ball(d, d as u64)
        .into_iter()
        .filter(|c| {
            c.iter().all(|x| x.abs() <= 1) && c.iter().filter(|x| **x != 0).count() as u64 == omega
        })
        .collect()
}

fn binomial(n: u64, k: u64) -> u64 {
    (0..k).fold(1, |acc, i| acc * (n - i) / (i + 1))
}

/// Check that `exact` puts the same weight on every element of `set` and nothing elsewhere.
fn assert_uniform_on(
    exact: &crate::enumerate::Exact<Vec<i64>>,
    set: &BTreeSet<Vec<i64>>,
    what: &str,
) {
    let support: BTreeSet<Vec<i64>> = exact.weights.keys().cloned().collect();
    assert_eq!(
        &support, set,
        "{what}: support differs from the brute-force set"
    );
    let size = BigUint::from(set.len());
    for (c, w) in &exact.weights {
        assert_eq!(
            w * &size,
            exact.denominator,
            "{what}: probability of {c:?} is not 1/{}",
            set.len()
        );
    }
}

#[test]
fn floyd_draws_every_k_subset_with_equal_probability() {
    for n in 0..=7u64 {
        for k in 0..=n {
            let dist = exact(|r| floyd(r, n, k).unwrap());
            let subsets = binomial(n, k);
            assert_eq!(dist.weights.len() as u64, subsets, "n = {n}, k = {k}");
            for (s, w) in &dist.weights {
                assert_eq!(s.len() as u64, k);
                assert!(s.windows(2).all(|p| p[0] < p[1]) && s.iter().all(|x| *x < n));
                assert_eq!(
                    w * BigUint::from(subsets),
                    dist.denominator,
                    "n = {n}, k = {k}, {s:?}"
                );
            }
            // Exactly k draws of ranges n-k+1, ..., n: n!/(n-k)! paths.
            let paths: u64 = (n - k + 1..=n).product();
            assert_eq!(dist.paths as u64, paths, "n = {n}, k = {k}");
        }
    }
}

#[test]
fn l1_ball_sampler_is_exactly_uniform_on_the_ball() {
    let cases: [(usize, u64); 16] = [
        (0, 0),
        (0, 3),
        (3, 0),
        (1, 1),
        (1, 4),
        (4, 1),
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
        (4, 2),
        (2, 4),
        (4, 3),
        (3, 4),
        (2, 5),
        (5, 2),
    ];
    for (d, omega) in cases {
        let ball = L1Ball::new(d, omega).unwrap();
        let set = brute_force_ball(d, omega);
        assert_eq!(
            *ball.count(),
            BigUint::from(set.len()),
            "count, d = {d}, omega = {omega}"
        );
        let dist = exact(|r| ball.sample_with(r).unwrap());
        assert_uniform_on(&dist, &set, &format!("l1 ball d = {d}, omega = {omega}"));
        println!(
            "l1 ball d = {d}, omega = {omega}: |C| = {}, {} paths, exactly uniform",
            set.len(),
            dist.paths
        );
    }
}

#[test]
fn sparse_sampler_is_exactly_uniform_on_the_sparse_set() {
    let cases: [(usize, u64); 10] = [
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 1),
        (3, 1),
        (3, 2),
        (4, 2),
        (4, 4),
        (5, 3),
        (6, 2),
    ];
    for (d, omega) in cases {
        let set = brute_force_sparse(d, omega);
        assert_eq!(
            count_sparse(d, omega),
            BigUint::from(set.len()),
            "d = {d}, omega = {omega}"
        );
        let dist = exact(|r| sparse_with(r, d, omega).unwrap());
        assert_uniform_on(&dist, &set, &format!("sparse d = {d}, omega = {omega}"));
    }
}

#[test]
fn the_support_size_draw_follows_the_weights() {
    // Implied by the uniformity test, but checked directly for one larger case: the probability
    // of support size i is W_i / |C| (the first draw only, so the tree is small).
    let (d, omega) = (6usize, 5u64);
    let ball = L1Ball::new(d, omega).unwrap();
    let dist = exact(|r| {
        let u = r.below_big(ball.count()).unwrap();
        ball.cumulative.partition_point(|c| c <= &u)
    });
    let w = weights(d, omega);
    let total: BigUint = w.iter().sum();
    for (i, wi) in w.iter().enumerate() {
        assert_eq!(
            &dist.weights[&i] * &total,
            wi * &dist.denominator,
            "support size {i}"
        );
    }
}
