//! `PrgRandomness`: the documented streams, the distributions' supports, the base PCS's seeds,
//! and the refusal to serve a stream twice.

use hachi_co_core::{
    BlinderKind, DOMAIN_BASE_PCS, DOMAIN_BLINDER_G0, DOMAIN_BLINDER_G1, DOMAIN_MASK, DOMAIN_RHO_U,
    DOMAIN_RHO_V, DOMAIN_RHO_W, Error, Ext, Fqk, PartyRandomness, PrgRandomness,
};
use hachi_co_sampling::{ShakePrg, rej, uniform};
use rand::Rng;

const K: usize = 4;
/// The toy-n1 modulus and mask width (`docs/m0-spec.md`, "Toy presets").
const Q: u64 = 8388637;
const GAMMA: u64 = 49152;
const SEED: [u8; 32] = *b"hachi-co core party seed 0123456";

fn field() -> Fqk<K> {
    Fqk::new(Q).unwrap()
}

fn ext_from(prg: &mut ShakePrg) -> Ext<K> {
    let c = uniform::uniform_mod_q(prg, Q, K).unwrap();
    Ext([c[0], c[1], c[2], c[3]])
}

#[test]
fn masks_and_rho_follow_the_documented_streams() {
    let mut rng = PrgRandomness::new(SEED);
    for iteration in [0u64, 1, 7] {
        let mask = PartyRandomness::<K>::mask(&mut rng, iteration, GAMMA, 768).unwrap();
        let want = rej::mask(
            &mut ShakePrg::new(&SEED, DOMAIN_MASK.as_bytes(), iteration),
            GAMMA,
            768,
        );
        assert_eq!(mask, want.unwrap());
        assert!(mask.iter().all(|x| x.unsigned_abs() <= GAMMA));
    }
    // rho^v has one stream, for all iterations.
    let rho = PartyRandomness::<K>::rho_v(&mut rng, 4, 32).unwrap();
    let want = uniform::short(&mut ShakePrg::new(&SEED, DOMAIN_RHO_V.as_bytes(), 0), 4, 32);
    assert_eq!(rho, want.unwrap());
    assert!(rho.iter().all(|x| (-2..=1).contains(x)));
    // rho^u has one stream per PrgRandomness, drawn in Commit.
    let rho_u = PartyRandomness::<K>::rho_u(&mut rng, 4, 32).unwrap();
    let want = uniform::short(&mut ShakePrg::new(&SEED, DOMAIN_RHO_U.as_bytes(), 0), 4, 32);
    assert_eq!(rho_u, want.unwrap());
    assert!(rho_u.iter().all(|x| (-2..=1).contains(x)));
}

/// The two rules of the pad share a stream: a party draws one pad per Eval, under either rule.
#[test]
fn the_pad_has_one_stream() {
    let f = Fqk::<K>::new(Q).unwrap();
    let mut rng = PrgRandomness::new(SEED);
    let pad = PartyRandomness::<K>::rho_w(&mut rng, &f).unwrap();
    assert!(pad.0.iter().all(|x| *x < Q));
    assert!(matches!(
        PartyRandomness::<K>::rho_w(&mut rng, &f),
        Err(Error::RandomnessReuse {
            kind: DOMAIN_RHO_W,
            index: 0
        })
    ));
    assert!(matches!(
        PartyRandomness::<K>::rho_w_residues(&mut rng, Q, 8),
        Err(Error::RandomnessReuse { .. })
    ));
    let mut rng = PrgRandomness::new(SEED);
    let residues = PartyRandomness::<K>::rho_w_residues(&mut rng, Q, 8).unwrap();
    let want = uniform::uniform_mod_q(&mut ShakePrg::new(&SEED, DOMAIN_RHO_W.as_bytes(), 0), Q, 8);
    assert_eq!(residues, want.unwrap());
    assert!(matches!(
        PartyRandomness::<K>::rho_w(&mut rng, &f),
        Err(Error::RandomnessReuse { .. })
    ));
}

#[test]
fn supports_are_exactly_the_boxes() {
    let mut rng = PrgRandomness::new(SEED);
    let mask = PartyRandomness::<K>::mask(&mut rng, 0, 1, 3000).unwrap();
    for v in -1..=1 {
        assert!(mask.contains(&v), "value {v} appears");
    }
    assert!(mask.iter().all(|x| (-1..=1).contains(x)));
    // S_b = [ceil(-b/2), ceil(b/2) - 1]: b = 4 -> [-2, 1], b = 5 -> [-2, 2].
    for (b, lo, hi) in [(4u64, -2i64, 1i64), (5, -2, 2), (3, -1, 1)] {
        let rho = PartyRandomness::<K>::rho_v(&mut PrgRandomness::new(SEED), b, 3000).unwrap();
        for v in lo..=hi {
            assert!(rho.contains(&v), "b = {b}: value {v} appears");
        }
        assert!(rho.iter().all(|x| (lo..=hi).contains(x)), "b = {b}");
    }
}

#[test]
fn determinism_and_independence() {
    let draw = |seed: [u8; 32], iteration| {
        PartyRandomness::<K>::mask(&mut PrgRandomness::new(seed), iteration, GAMMA, 64).unwrap()
    };
    assert_eq!(draw(SEED, 3), draw(SEED, 3));
    assert_ne!(draw(SEED, 3), draw(SEED, 4));
    let mut other = SEED;
    other[0] ^= 1;
    assert_ne!(draw(SEED, 3), draw(other, 3));
}

#[test]
fn blinders_have_the_requested_shape_and_separate_streams() {
    let f = field();
    let mut rng = PrgRandomness::new(SEED);
    let g0 = rng.blinder(&f, BlinderKind::G0, 15, 14).unwrap();
    let g1 = rng.blinder(&f, BlinderKind::G1, 15, 3).unwrap();
    g0.validate(15, 14).unwrap();
    g1.validate(15, 3).unwrap();
    assert!(g0.validate(15, 3).is_err() && g1.validate(14, 3).is_err());
    assert_ne!(g0.s0, g1.s0);
    for (domain, g) in [(DOMAIN_BLINDER_G0, &g0), (DOMAIN_BLINDER_G1, &g1)] {
        let mut prg = ShakePrg::new(&SEED, domain.as_bytes(), 0);
        assert_eq!(g.s0, ext_from(&mut prg));
        for row in &g.s {
            for x in row {
                assert_eq!(*x, ext_from(&mut prg));
            }
        }
    }
    assert_eq!(
        rng.blinder(&f, BlinderKind::G0, 15, 14),
        Err(Error::RandomnessReuse {
            kind: DOMAIN_BLINDER_G0,
            index: 0
        })
    );
}

#[test]
fn base_pcs_seeds_follow_their_stream_and_are_served_once() {
    let mut rng = PrgRandomness::new(SEED);
    let seeds: Vec<[u8; 32]> = (0..6u64)
        .map(|i| PartyRandomness::<K>::base_pcs(&mut rng, i).unwrap())
        .collect();
    for (i, s) in seeds.iter().enumerate() {
        let mut want = [0u8; 32];
        ShakePrg::new(&SEED, DOMAIN_BASE_PCS.as_bytes(), i as u64).fill_bytes(&mut want);
        assert_eq!(*s, want, "index {i}");
        for t in &seeds[..i] {
            assert_ne!(s, t, "distinct seeds");
        }
        assert_eq!(
            PartyRandomness::<K>::base_pcs(&mut rng, i as u64),
            Err(Error::RandomnessReuse {
                kind: DOMAIN_BASE_PCS,
                index: i as u64
            })
        );
    }
    assert_eq!(DOMAIN_BASE_PCS, "hachi-co/sumcheck/base-pcs");
}

#[test]
fn every_stream_is_served_once_and_failures_do_not_use_it_up() {
    let mut rng = PrgRandomness::new(SEED);
    PartyRandomness::<K>::mask(&mut rng, 0, GAMMA, 8).unwrap();
    assert_eq!(
        PartyRandomness::<K>::mask(&mut rng, 0, GAMMA, 8),
        Err(Error::RandomnessReuse {
            kind: DOMAIN_MASK,
            index: 0
        })
    );
    // Other kinds and other iterations are separate streams.
    PartyRandomness::<K>::mask(&mut rng, 1, GAMMA, 8).unwrap();
    PartyRandomness::<K>::rho_u(&mut rng, 4, 8).unwrap();
    assert!(matches!(
        PartyRandomness::<K>::rho_u(&mut rng, 4, 8),
        Err(Error::RandomnessReuse { .. })
    ));
    // A failed request (gamma above 2^63 - 1, b = 0) does not use up its stream.
    assert!(matches!(
        PartyRandomness::<K>::mask(&mut rng, 5, u64::MAX, 8),
        Err(Error::Sampling(_))
    ));
    PartyRandomness::<K>::mask(&mut rng, 5, GAMMA, 8).unwrap();
    assert!(matches!(
        PartyRandomness::<K>::rho_v(&mut rng, 0, 8),
        Err(Error::Sampling(_))
    ));
    PartyRandomness::<K>::rho_v(&mut rng, 4, 8).unwrap();
    assert!(matches!(
        PartyRandomness::<K>::rho_v(&mut rng, 4, 8),
        Err(Error::RandomnessReuse { .. })
    ));
    // Debug output does not show the seed.
    let debug = format!("{rng:?}");
    assert!(debug.starts_with("PrgRandomness"));
    assert!(!debug.contains("seed") && !debug.contains("48, 97, 99"));
}

/// `masks` against `mask` for every iteration on the same seed (the streams drawn in parallel
/// give the masks drawn one by one), then reuse refused both ways.
#[test]
fn the_masks_drawn_together_are_the_masks_drawn_one_by_one() {
    let (gamma, len) = (1u64 << 20, 1000);
    let mut together = PrgRandomness::new([42; 32]);
    let mut one_by_one = PrgRandomness::new([42; 32]);
    let got = PartyRandomness::<4>::masks(&mut together, 7, gamma, len).unwrap();
    let want: Vec<Vec<i64>> = (0..7)
        .map(|j| PartyRandomness::<4>::mask(&mut one_by_one, j, gamma, len).unwrap())
        .collect();
    assert_eq!(got, want);
    assert!(PartyRandomness::<4>::masks(&mut together, 7, gamma, len).is_err());
    assert!(PartyRandomness::<4>::mask(&mut together, 3, gamma, len).is_err());
    assert!(PartyRandomness::<4>::masks(&mut one_by_one, 1, gamma, len).is_err());
    let mut fresh = PrgRandomness::new([42; 32]);
    PartyRandomness::<4>::mask(&mut fresh, 2, gamma, len).unwrap();
    assert!(
        PartyRandomness::<4>::masks(&mut fresh, 3, gamma, len).is_err(),
        "stream 2 is used"
    );
}

/// A mask is read again (`mask_again`, `masks_again`) only once it was drawn, as the same mask,
/// in groups (`masks_in`); a fresh draw of a mask read again is still refused.
#[test]
fn masks_are_read_again_only_once_drawn() {
    let (gamma, len) = (1u64 << 20, 1000);
    let mut rng = PrgRandomness::new([43; 32]);
    let mut reference = PrgRandomness::new([43; 32]);
    assert!(matches!(
        PartyRandomness::<4>::masks_again(&mut rng, 0..2, gamma, len),
        Err(Error::MaskNotDrawn { index: 0 })
    ));
    let first = PartyRandomness::<4>::masks_in(&mut rng, 0..3, gamma, len).unwrap();
    let second = PartyRandomness::<4>::masks_in(&mut rng, 3..5, gamma, len).unwrap();
    let want = PartyRandomness::<4>::masks(&mut reference, 5, gamma, len).unwrap();
    assert_eq!([first, second].concat(), want);
    assert_eq!(
        PartyRandomness::<4>::masks_again(&mut rng, 1..5, gamma, len).unwrap(),
        want[1..5]
    );
    assert_eq!(
        PartyRandomness::<4>::mask_again(&mut rng, 4, gamma, len).unwrap(),
        want[4]
    );
    assert!(matches!(
        PartyRandomness::<4>::mask_again(&mut rng, 5, gamma, len),
        Err(Error::MaskNotDrawn { index: 5 })
    ));
    assert!(PartyRandomness::<4>::masks_in(&mut rng, 4..6, gamma, len).is_err());
    assert!(PartyRandomness::<4>::mask(&mut rng, 2, gamma, len).is_err());
}
