//! Rej by exhaustive enumeration of all masks at tiny $(N, \gamma, \beta_1)$, its failure boundary
//! at $\lVert v \rVert_\infty = \beta_1 + 1$, the exact acceptance probability, and SimRej's exact
//! output distribution (every sequence of draws enumerated) against Rej's for every $v$.

use super::*;
use crate::enumerate::exact;
use num_bigint::BigUint;
use std::collections::{BTreeMap, BTreeSet};

/// Tiny $(N, \gamma, \beta_1)$ with $\beta_1 \le \gamma$, including $\beta_1 = 0$, $\beta_1 = \gamma$
/// and $\gamma = 0$.
const PARAMETERS: [(usize, u64, u64); 16] = [
    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1),
    (1, 2, 1),
    (1, 3, 2),
    (1, 5, 2),
    (1, 4, 4),
    (2, 1, 0),
    (2, 1, 1),
    (2, 2, 1),
    (2, 3, 1),
    (2, 3, 2),
    (2, 3, 3),
    (3, 2, 1),
    (3, 3, 2),
    (4, 2, 1),
];

/// All vectors in $[-b, b]^N$, in lexicographic order.
fn cube(n: usize, b: u64) -> Vec<Vec<i64>> {
    let b = b as i64;
    let mut out = vec![vec![]];
    for _ in 0..n {
        out = out
            .into_iter()
            .flat_map(|prefix| {
                (-b..=b).map(move |x| {
                    let mut v = prefix.clone();
                    v.push(x);
                    v
                })
            })
            .collect();
    }
    out
}

fn inf_norm(v: &[i64]) -> u64 {
    v.iter().map(|x| x.unsigned_abs()).max().unwrap_or(0)
}

fn pow(base: u64, n: usize) -> u64 {
    base.pow(n as u32)
}

#[test]
fn rej_accepts_each_inner_point_exactly_once_for_every_admissible_v() {
    for (n, gamma, beta1) in PARAMETERS {
        let masks = cube(n, gamma);
        let inner: BTreeSet<Vec<i64>> = cube(n, gamma - beta1).into_iter().collect();
        assert_eq!(inner.len() as u64, pow(2 * gamma - 2 * beta1 + 1, n));
        for v in cube(n, beta1) {
            let mut hits: BTreeMap<Vec<i64>, usize> = BTreeMap::new();
            for r in &masks {
                if let Some(z) = rej(&v, gamma, beta1, r).unwrap() {
                    let sum: Vec<i64> = v.iter().zip(r).map(|(a, b)| a + b).collect();
                    assert_eq!(z, sum, "output is not v + r");
                    *hits.entry(z).or_default() += 1;
                }
            }
            let accepted: usize = hits.values().sum();
            assert_eq!(
                accepted as u64,
                pow(2 * gamma - 2 * beta1 + 1, n),
                "(N, gamma, beta1) = ({n}, {gamma}, {beta1}), v = {v:?}: accepted count"
            );
            assert!(
                hits.values().all(|h| *h == 1),
                "an output is hit twice, v = {v:?}"
            );
            let outputs: BTreeSet<Vec<i64>> = hits.into_keys().collect();
            assert_eq!(
                outputs, inner,
                "accepted outputs are not the inner box, v = {v:?}"
            );
        }
    }
}

#[test]
fn the_lemma_fails_at_norm_beta1_plus_one() {
    for (n, gamma, beta1) in PARAMETERS {
        let masks = cube(n, gamma);
        let inner: BTreeSet<Vec<i64>> = cube(n, gamma - beta1).into_iter().collect();
        let boundary: Vec<Vec<i64>> = cube(n, beta1 + 1)
            .into_iter()
            .filter(|v| inf_norm(v) == beta1 + 1)
            .collect();
        assert!(!boundary.is_empty());
        for v in boundary {
            let mut outputs = BTreeMap::<Vec<i64>, usize>::new();
            for r in &masks {
                // Rej refuses such a v ...
                assert_eq!(rej(&v, gamma, beta1, r), Err(Error::SecretTooLarge));
                // ... and its acceptance rule, applied anyway, no longer covers the inner box.
                let c = check(&v, r, gamma, beta1);
                assert!(c.secret_too_large && !c.mask_too_large);
                if !c.reject {
                    *outputs.entry(c.z).or_default() += 1;
                }
            }
            // Each coordinate with |v_i| = beta1 + 1 loses exactly the inner value
            // -sign(v_i) (gamma - beta1), which would need |r_i| = gamma + 1.
            let lost = v.iter().filter(|x| x.unsigned_abs() == beta1 + 1).count();
            let width = 2 * gamma - 2 * beta1 + 1;
            let expected = pow(width - 1, lost) * pow(width, n - lost);
            let accepted: usize = outputs.values().sum();
            assert_eq!(
                accepted as u64, expected,
                "({n}, {gamma}, {beta1}), v = {v:?}"
            );
            assert!(
                accepted < inner.len(),
                "the accepted set must shrink, v = {v:?}"
            );
            assert!(outputs.values().all(|h| *h == 1));
            let missing: Vec<&Vec<i64>> =
                inner.iter().filter(|z| !outputs.contains_key(*z)).collect();
            for z in &missing {
                assert!(
                    z.iter()
                        .zip(&v)
                        .any(|(zi, vi)| vi.unsigned_abs() == beta1 + 1
                            && *zi == -vi.signum() * (gamma - beta1) as i64),
                    "unexpected missing output {z:?} for v = {v:?}"
                );
            }
            assert!(outputs.keys().all(|z| inner.contains(z)));
        }
    }
}

/// The accepted count of [`rej`] over all masks, for one admissible $v$, by enumeration.
fn accepted_count(n: usize, gamma: u64, beta1: u64, v: &[i64]) -> u64 {
    cube(n, gamma)
        .iter()
        .filter(|r| rej(v, gamma, beta1, r).unwrap().is_some())
        .count() as u64
}

fn gcd(a: &BigUint, b: &BigUint) -> BigUint {
    let (mut a, mut b) = (a.clone(), b.clone());
    while b != BigUint::ZERO {
        let r = &a % &b;
        a = b;
        b = r;
    }
    a
}

#[test]
fn acceptance_probability_matches_the_enumerated_counts() {
    for (n, gamma, beta1) in PARAMETERS {
        let p = acceptance_probability(gamma, beta1, n).unwrap();
        assert_eq!(
            gcd(&p.numerator, &p.denominator),
            BigUint::from(1u32),
            "lowest terms"
        );
        let total = BigUint::from(pow(2 * gamma + 1, n));
        for v in [vec![0; n], vec![beta1 as i64; n], vec![-(beta1 as i64); n]] {
            let count = BigUint::from(accepted_count(n, gamma, beta1, &v));
            // count / total == numerator / denominator
            assert_eq!(
                &count * &p.denominator,
                &total * &p.numerator,
                "({n}, {gamma}, {beta1})"
            );
        }
        let log2 = log2_acceptance_probability(gamma, beta1, n).unwrap();
        let reference =
            (pow(2 * gamma - 2 * beta1 + 1, n) as f64 / pow(2 * gamma + 1, n) as f64).log2();
        assert!(
            (log2 - reference).abs() <= 1e-12 * reference.abs().max(1.0),
            "log2 helper"
        );
        assert!(
            (p.log2() - reference).abs() <= 1e-12 * reference.abs().max(1.0),
            "Ratio::log2"
        );
    }
}

#[test]
fn sim_rej_has_exactly_the_output_distribution_of_rej() {
    for (n, gamma, beta1) in PARAMETERS {
        let sim = exact(|r| sim_rej_with(r, gamma, beta1, n).unwrap());
        let masks = cube(n, gamma);
        let total = BigUint::from(masks.len());
        // SimRej: bottom with probability 1 - 1/M, each inner point with 1/(2 gamma + 1)^N.
        let inner = cube(n, gamma - beta1);
        assert_eq!(sim.weights.len(), inner.len() + usize::from(beta1 > 0));
        for z in &inner {
            assert_eq!(
                &sim.weights[&Some(z.clone())] * &total,
                sim.denominator,
                "SimRej P[{z:?}]"
            );
        }
        for v in cube(n, beta1) {
            let mut rej_weights: BTreeMap<Option<Vec<i64>>, u64> = BTreeMap::new();
            for r in &masks {
                *rej_weights
                    .entry(rej(&v, gamma, beta1, r).unwrap())
                    .or_default() += 1;
            }
            assert_eq!(
                rej_weights.len(),
                sim.weights.len(),
                "v = {v:?}: supports differ"
            );
            for (outcome, count) in &rej_weights {
                // count / total == sim weight / sim denominator
                assert_eq!(
                    BigUint::from(*count) * &sim.denominator,
                    &sim.weights[outcome] * &total,
                    "({n}, {gamma}, {beta1}), v = {v:?}, outcome {outcome:?}"
                );
            }
        }
    }
}

#[test]
fn rej_rejects_bad_inputs() {
    assert!(matches!(
        rej(&[0], 1, 2, &[0]),
        Err(Error::InvalidParameter(_))
    ));
    assert!(matches!(
        rej(&[0], 1 << 63, 0, &[0]),
        Err(Error::InvalidParameter(_))
    ));
    assert_eq!(
        rej(&[0, 0], 2, 1, &[0]),
        Err(Error::LengthMismatch { left: 2, right: 1 })
    );
    assert_eq!(rej(&[0, 0], 2, 1, &[0, 3]), Err(Error::MaskOutOfRange));
    assert_eq!(rej(&[0, 0], 2, 1, &[-3, 0]), Err(Error::MaskOutOfRange));
    assert_eq!(rej(&[2, 0], 2, 1, &[0, 0]), Err(Error::SecretTooLarge));
    assert_eq!(rej(&[0, -2], 2, 1, &[0, 0]), Err(Error::SecretTooLarge));
    // Both violated: the mask is reported.
    assert_eq!(rej(&[2, 0], 2, 1, &[3, 0]), Err(Error::MaskOutOfRange));
    // Empty vectors: N = 0 always accepts.
    assert_eq!(rej(&[], 2, 1, &[]), Ok(Some(vec![])));
    assert!(matches!(
        sim_rej_with_rng_check(),
        Err(Error::InvalidParameter(_))
    ));
    assert!(matches!(
        acceptance_probability(1, 2, 1),
        Err(Error::InvalidParameter(_))
    ));
    assert!(matches!(
        log2_acceptance_probability(1, 2, 1),
        Err(Error::InvalidParameter(_))
    ));
    assert!(matches!(
        acceptance_probability(1 << 40, 1, 1 << 30),
        Err(Error::InvalidParameter(_))
    ));
}

fn sim_rej_with_rng_check() -> Result<Option<Vec<i64>>> {
    sim_rej(&mut crate::ShakePrg::new(&[0; 32], b"unused", 0), 1, 2, 1)
}

#[test]
fn rej_is_exact_at_the_extremes_of_i64() {
    let max = i64::MAX as u64;
    // gamma = 2^63 - 1: |i64::MIN| = 2^63 exceeds it.
    assert_eq!(rej(&[0], max, 0, &[i64::MIN]), Err(Error::MaskOutOfRange));
    assert_eq!(rej(&[0], max, 0, &[i64::MAX]), Ok(Some(vec![i64::MAX])));
    assert_eq!(rej(&[0], max, 0, &[-i64::MAX]), Ok(Some(vec![-i64::MAX])));
    // z = 2^63 does not fit in i64; it must be rejected, not wrapped.
    assert_eq!(rej(&[1], max, 1, &[i64::MAX]), Ok(None));
    assert_eq!(rej(&[-1], max, 1, &[-i64::MAX]), Ok(None));
    assert_eq!(
        rej(&[1], max, 1, &[i64::MAX - 2]),
        Ok(Some(vec![i64::MAX - 1]))
    );
    assert_eq!(rej(&[i64::MIN], max, max, &[0]), Err(Error::SecretTooLarge));
    assert_eq!(rej(&[-i64::MAX], max, max, &[i64::MAX]), Ok(Some(vec![0])));
    assert_eq!(rej(&[-i64::MAX], max, max, &[i64::MAX - 1]), Ok(None));
}

#[test]
fn exceeds_matches_the_naive_comparison() {
    let edges: Vec<i128> = [
        0i128,
        1,
        2,
        3,
        (1 << 32) - 1,
        1 << 62,
        i64::MAX as i128 - 1,
        i64::MAX as i128,
        1 << 63,
        (1 << 64) - 1,
        1 << 64,
    ]
    .into_iter()
    .flat_map(|x| [x, -x])
    .collect();
    for &x in &edges {
        for &b in edges.iter().filter(|b| (0..1 << 64).contains(*b)) {
            assert_eq!(exceeds(x, b) == 1, x.abs() > b, "x = {x}, b = {b}");
        }
    }
}

#[test]
fn toy_presets_accept_with_probability_near_e_to_the_minus_half() {
    // (n, N, gamma, beta1) of the toy presets (docs/m0-spec.md). With gamma = 2nN beta1, each
    // prover accepts with probability 1/M ~ e^{-1/(2n)}, so all n provers together accept with
    // probability M^{-n} ~ e^{-1/2} >= 1/e (6_Collaborative_Hachi.tex:275, 522).
    for (n, big_n, gamma, beta1) in [
        (1u32, 768usize, 49152u64, 32u64),
        (2, 832, 106496, 32),
        (3, 1792, 172032, 16),
    ] {
        assert_eq!(gamma, 2 * u64::from(n) * big_n as u64 * beta1);
        let exact = acceptance_probability(gamma, beta1, big_n).unwrap();
        let log2 = log2_acceptance_probability(gamma, beta1, big_n).unwrap();
        assert!(
            (exact.log2() - log2).abs() <= 1e-9,
            "n = {n}: {} vs {log2}",
            exact.log2()
        );
        let all = (f64::from(n) * log2).exp2();
        assert!(
            (all - (-0.5f64).exp()).abs() < 1e-3,
            "n = {n}: M^-n = {all}"
        );
        assert!(
            f64::from(n) * log2 >= -core::f64::consts::LOG2_E,
            "M <= e^(1/n) fails"
        );
        println!("toy n = {n}: 1/M = {:.6}, M^-n = {all:.6}", log2.exp2());
    }
}
