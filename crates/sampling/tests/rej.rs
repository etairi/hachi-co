//! Rej and SimRej with real randomness: chi-square tests of their output distributions against
//! the exact law of the lemma (4_preliminaries.tex:122-127), with fixed seeds and the threshold at
//! $1 - 10^{-4}$ (`common::chi_square_threshold`). The exhaustive and exact checks are in the unit
//! tests (`src/rej/tests.rs`).

mod common;

use common::{Counting, assert_chi_square};
use hachi_co_sampling::rej::{acceptance_probability, mask, rej, sim_rej};
use hachi_co_sampling::{Error, ShakePrg};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::collections::BTreeMap;

const N: usize = 2;
const GAMMA: u64 = 2;
const BETA1: u64 = 1;

/// The outcomes of SimRej at (N, gamma, beta1) = (2, 2, 1): bottom, then the 9 points of
/// [-1, 1]^2, with their exact probabilities 16/25 and 1/25 each.
fn outcomes() -> (Vec<Option<Vec<i64>>>, Vec<f64>) {
    let mut keys = vec![None];
    let mut probabilities = vec![16.0 / 25.0];
    for a in -1..=1 {
        for b in -1..=1 {
            keys.push(Some(vec![a, b]));
            probabilities.push(1.0 / 25.0);
        }
    }
    let p = acceptance_probability(GAMMA, BETA1, N).unwrap();
    assert_eq!((p.numerator, p.denominator), (9u32.into(), 25u32.into()));
    (keys, probabilities)
}

fn tally(trials: usize, mut draw: impl FnMut() -> Option<Vec<i64>>) -> Vec<u64> {
    let (keys, _) = outcomes();
    let index: BTreeMap<Option<Vec<i64>>, usize> =
        keys.into_iter().enumerate().map(|(i, k)| (k, i)).collect();
    let mut counts = vec![0u64; index.len()];
    for _ in 0..trials {
        let outcome = draw();
        counts[*index
            .get(&outcome)
            .unwrap_or_else(|| panic!("impossible outcome {outcome:?}"))] += 1;
    }
    counts
}

#[test]
fn rej_with_fresh_masks_matches_the_simrej_law_for_every_v() {
    // ChaCha20 seeded with 0x4a11 + k for the k-th v; 50000 attempts per v.
    let (_, probabilities) = outcomes();
    let trials = 50_000;
    let expected: Vec<f64> = probabilities.iter().map(|p| p * trials as f64).collect();
    let mut k = 0;
    for v0 in -1..=1i64 {
        for v1 in -1..=1i64 {
            let v = [v0, v1];
            let mut rng = Counting::new(ChaCha20Rng::seed_from_u64(0x4a11 + k));
            k += 1;
            let counts = tally(trials, || {
                let r = mask(&mut rng, GAMMA, N).unwrap();
                rej(&v, GAMMA, BETA1, &r).unwrap()
            });
            assert_chi_square(&format!("Rej v = {v:?}"), &counts, &expected);
        }
    }
}

#[test]
fn sim_rej_matches_its_law() {
    let (_, probabilities) = outcomes();
    let trials = 100_000;
    let expected: Vec<f64> = probabilities.iter().map(|p| p * trials as f64).collect();
    let mut rng = ShakePrg::new(&[0x77; 32], b"chi2/sim_rej", 0);
    let counts = tally(trials, || sim_rej(&mut rng, GAMMA, BETA1, N).unwrap());
    assert_chi_square("SimRej", &counts, &expected);
}

#[test]
fn masks_cover_the_box_and_reject_bad_parameters() {
    let mut rng = ShakePrg::new(&[0x78; 32], b"masks", 0);
    let r = mask(&mut rng, 3, 10_000).unwrap();
    assert!(r.iter().all(|x| x.abs() <= 3));
    for x in -3..=3 {
        assert!(r.contains(&x), "{x} never drawn");
    }
    assert!(matches!(
        mask(&mut rng, 1 << 63, 1),
        Err(Error::InvalidParameter(_))
    ));
    assert!(matches!(
        sim_rej(&mut rng, 1, 2, 1),
        Err(Error::InvalidParameter(_))
    ));
    assert_eq!(
        sim_rej(&mut rng, 5, 0, 3).map(|z| z.is_some()),
        Ok(true),
        "beta1 = 0 never rejects"
    );
}
