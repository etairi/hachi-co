//! Rejection sampling at the toy parameters: the boundary cases of
//! `negative_tests.rej_boundary_toy` through this crate's Rej step, and the abort rate of the
//! n-prover step over 10^4 emulated attempts per preset against the exact 1 - M^{-n}.

mod common;

use common::{K, load};
use hachi_co_core::{Fixture, PrgRandomness};
use hachi_co_r2l::{Error, fold, respond, response_len, sample_attempt};
use hachi_co_sampling::l1ball::L1Ball;
use hachi_co_sampling::rej::acceptance_probability;
use hachi_co_sampling::{Error as SamplingError, ShakePrg};
use num_bigint::BigUint;
use sha2::{Digest, Sha256};
use std::time::Instant;

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// At ||v||_inf = beta_1 every first coordinate of the box [-gamma + beta_1, gamma - beta_1] is
/// output by exactly one mask (exhaustively over the first mask coordinate, the others zero, with
/// full-length vectors); at beta_1 + 1 this crate's Rej refuses to run (the lemma's hypothesis,
/// 4_preliminaries.tex:123), and the acceptance rule itself would reach one value fewer, so that
/// Rej(v) would not be distributed as SimRej (the oracle's `rej_boundary_toy`).
#[test]
fn rej_boundary_cases_of_every_fixture() {
    for name in Fixture::TOY {
        let (fx, ctx, _, _) = load(name);
        let rec = |k: &str| format!("/negative_tests/rej_boundary_toy/{k}");
        let (g, b1) = (ctx.gamma() as i64, ctx.beta1() as i64);
        let (lo, hi) = (-g + b1, g - b1);
        let box_side = (hi - lo + 1) as usize;
        assert_eq!(box_side, fx.usize(&rec("box_side")).unwrap(), "{name}");
        let len = response_len(&ctx);

        // ||v|| = beta_1, v = (beta_1, 0, ..., 0): every z_1 of the box, each by one mask.
        let mut v = vec![0i64; len];
        v[0] = b1;
        let mut mask = vec![0i64; len];
        let mut hits = vec![0u8; box_side];
        for r1 in -g..=g {
            mask[0] = r1;
            match respond(&ctx, &v, &mask).unwrap() {
                Some(z) => {
                    assert_eq!(z[0], b1 + r1);
                    assert!(z[1..].iter().all(|x| *x == 0));
                    hits[(z[0] - lo) as usize] += 1;
                }
                None => assert!((b1 + r1).abs() > hi),
            }
        }
        assert!(hits.iter().all(|h| *h <= 1), "{name}: no z_1 twice");
        let reach = hits.iter().filter(|h| **h == 1).count();
        assert_eq!(
            reach,
            fx.usize(&rec("first_coordinate_reachable_at_beta1"))
                .unwrap()
        );
        assert_eq!(
            reach == box_side,
            fx.bool(&rec("within_bound_full")).unwrap()
        );

        // ||v|| = beta_1 + 1 (either sign): refused, whatever the mask.
        for v1 in [b1 + 1, -(b1 + 1)] {
            v[0] = v1;
            for r1 in [-g, 0, g] {
                mask[0] = r1;
                assert_eq!(
                    respond(&ctx, &v, &mask),
                    Err(Error::Sampling(SamplingError::SecretTooLarge)),
                    "{name}: v_1 = {v1}"
                );
            }
        }
        // The acceptance rule at v_1 = beta_1 + 1 would reach 2 gamma - 2 beta_1 values only.
        let reach_bad = (-g..=g).filter(|r1| (b1 + 1 + r1).abs() <= hi).count();
        assert_eq!(
            reach_bad,
            fx.usize(&rec("first_coordinate_reachable_at_beta1_plus_1"))
                .unwrap()
        );
        assert_eq!(reach_bad, box_side - 1);
        assert_eq!(
            reach_bad < box_side,
            fx.bool(&rec("accepted_set_smaller")).unwrap()
        );
        let (num, den) = (reach_bad as u64, box_side as u64);
        let gd = gcd(num, den);
        assert_eq!(
            format!("{}/{}", num / gd, den / gd),
            fx.str(&rec("acceptance_ratio_boundary_over_box")).unwrap()
        );

        // The mask's support is checked too: gamma + 1 is refused, gamma is not.
        v[0] = 0;
        mask[0] = g + 1;
        assert_eq!(
            respond(&ctx, &v, &mask),
            Err(Error::Sampling(SamplingError::MaskOutOfRange))
        );
        mask[0] = -g;
        assert_eq!(respond(&ctx, &v, &mask).unwrap(), None);
        mask[0] = -hi;
        assert!(respond(&ctx, &v, &mask).unwrap().is_some());
        // Wrong lengths.
        assert!(matches!(
            respond(&ctx, &v[1..], &mask),
            Err(Error::Shape { .. })
        ));
        assert!(matches!(
            respond(&ctx, &v, &mask[1..]),
            Err(Error::Shape { .. })
        ));
    }
}

/// The master tag of the abort-rate seeds: party p of run i at preset X uses the PRG seed
/// SHA-256(TAG || X || LE64(i) || LE64(p)), and run i's challenges are drawn from the ShakePrg
/// with seed SHA-256(TAG || X || "c"), domain `C_DOMAIN`, index i.
const TAG: &str = "hachi-co-r2l/abort-rate/v1";
const C_DOMAIN: &[u8] = b"hachi-co-r2l/abort-rate/c";
const TRIALS: u64 = 10_000;

/// `num / den` in floating point for `num <= den`, from the top 64 bits of both (exact shifts),
/// so that the relative error is about 2^-52. (`Ratio::log2` subtracts two logarithms of about
/// 34000 bits at toy-n3 and loses about 4e-12 to cancellation, which the cross-check against the
/// oracle's float at 1e-12 detects.)
fn to_f64(num: &BigUint, den: &BigUint) -> f64 {
    let shift = den.bits().saturating_sub(64);
    let top = |x: &BigUint| (x >> shift).iter_u64_digits().next().unwrap_or(0) as f64;
    top(num) / top(den)
}

fn seed(preset: &str, parts: &[&[u8]]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(TAG.as_bytes());
    h.update(preset.as_bytes());
    for part in parts {
        h.update(part);
    }
    h.finalize().into()
}

/// The abort rate of the n-prover step: 10^4 attempts per preset, each with every party emulated
/// locally (fresh `PrgRandomness` per party and run, the party's real fold for fresh uniform
/// challenges, and this crate's Rej); an attempt aborts iff some party rejects. The observed
/// fraction must lie within 3 sigma of the exact 1 - M^{-n}, M^{-1} from
/// `hachi_co_sampling::rej::acceptance_probability` (5_Building_Blocks.tex:193).
#[test]
fn abort_rate_matches_one_minus_m_to_the_minus_n() {
    for name in Fixture::TOY {
        let start = Instant::now();
        let (fx, ctx, pv, states) = load(name);
        let l = &ctx.layout;
        let n = l.n;
        let ratio = acceptance_probability(ctx.gamma(), ctx.beta1(), response_len(&ctx)).unwrap();
        let accept_one = to_f64(&ratio.numerator, &ratio.denominator);
        let expected = 1.0 - accept_one.powi(n as i32);
        // The oracle's float of the same quantity (its params section), to 1e-12.
        let recorded = fx
            .get("/params/abort_probability")
            .unwrap()
            .as_f64()
            .unwrap();
        assert!(
            (expected - recorded).abs() <= 1e-12,
            "{name}: {expected} vs {recorded}"
        );
        let ball = L1Ball::new(l.d, ctx.omega()).unwrap();
        let c_seed = seed(name, &[b"c"]);
        let mut aborts = 0u64;
        for run in 0..TRIALS {
            let mut cprg = ShakePrg::new(&c_seed, C_DOMAIN, run);
            let c: Vec<Vec<i64>> = (0..l.n_blocks)
                .map(|_| ball.sample(&mut cprg).unwrap())
                .collect();
            let mut abort = false;
            for (p, st) in (1..=n).zip(&states) {
                let mut rng = PrgRandomness::new(seed(
                    name,
                    &[&run.to_le_bytes(), &(p as u64).to_le_bytes()],
                ));
                let secrets = sample_attempt::<K, _>(&ctx, &pv, &mut rng, 0).unwrap();
                let f = fold(&ctx, st, &c).unwrap();
                abort |= respond(&ctx, &f, &secrets.mask).unwrap().is_none();
            }
            aborts += u64::from(abort);
        }
        let rate = aborts as f64 / TRIALS as f64;
        let sigma = (expected * (1.0 - expected) / TRIALS as f64).sqrt();
        let z = (rate - expected) / sigma;
        println!(
            "{name}: {aborts} aborts in {TRIALS} attempts, rate {rate:.4}, exact 1 - M^-n = \
             {expected:.6}, sigma {sigma:.6}, (rate - exact)/sigma = {z:+.2}, seeds {TAG}, \
             {:.1} s",
            start.elapsed().as_secs_f64()
        );
        assert!(
            z.abs() <= 3.0,
            "{name}: {rate} is not within 3 sigma of {expected}"
        );
    }
}
