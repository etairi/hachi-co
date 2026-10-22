//! `PrgRandomness`: the documented streams, the distributions' supports, AS.Share, the base
//! PCS's seeds, and the refusal to serve a stream twice.

use hachi_co_core::{
    BlinderKind, DOMAIN_BASE_PCS, DOMAIN_BLINDER_G0, DOMAIN_BLINDER_G1, DOMAIN_MASK,
    DOMAIN_RESHARE, DOMAIN_RHO, Error, Ext, Fqk, PartyRandomness, PrgRandomness,
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
    for attempt in [0u64, 1, 7] {
        let mask = PartyRandomness::<K>::mask(&mut rng, attempt, GAMMA, 768).unwrap();
        let want = rej::mask(
            &mut ShakePrg::new(&SEED, DOMAIN_MASK.as_bytes(), attempt),
            GAMMA,
            768,
        );
        assert_eq!(mask, want.unwrap());
        assert!(mask.iter().all(|x| x.unsigned_abs() <= GAMMA));
        let rho = PartyRandomness::<K>::rho(&mut rng, attempt, 4, 32).unwrap();
        let want = uniform::short(
            &mut ShakePrg::new(&SEED, DOMAIN_RHO.as_bytes(), attempt),
            4,
            32,
        );
        assert_eq!(rho, want.unwrap());
        assert!(rho.iter().all(|x| (-2..=1).contains(x)));
    }
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
    for (attempt, b, lo, hi) in [(0u64, 4u64, -2i64, 1i64), (1, 5, -2, 2), (2, 3, -1, 1)] {
        let rho = PartyRandomness::<K>::rho(&mut rng, attempt, b, 3000).unwrap();
        for v in lo..=hi {
            assert!(rho.contains(&v), "b = {b}: value {v} appears");
        }
        assert!(rho.iter().all(|x| (lo..=hi).contains(x)), "b = {b}");
    }
}

#[test]
fn determinism_and_independence() {
    let draw = |seed: [u8; 32], attempt| {
        PartyRandomness::<K>::mask(&mut PrgRandomness::new(seed), attempt, GAMMA, 64).unwrap()
    };
    assert_eq!(draw(SEED, 3), draw(SEED, 3));
    assert_ne!(draw(SEED, 3), draw(SEED, 4));
    let mut other = SEED;
    other[0] ^= 1;
    assert_ne!(draw(SEED, 3), draw(other, 3));
}

#[test]
fn reshare_is_additive_sharing() {
    let f = field();
    let input: Vec<Ext<K>> = (0..64u64).map(|i| Ext([i, 2 * i, Q - 1 - i, 5])).collect();
    for n in 1..=5 {
        let mut rng = PrgRandomness::new(SEED);
        let shares = PartyRandomness::<K>::reshare(&mut rng, &f, &input, n).unwrap();
        assert_eq!(shares.len(), n);
        let mut sum = vec![f.zero(); input.len()];
        for s in &shares {
            assert_eq!(s.len(), input.len());
            assert!(s.iter().all(|x| x.0.iter().all(|c| *c < Q)));
            for (acc, x) in sum.iter_mut().zip(s) {
                *acc = f.add(acc, x);
            }
        }
        assert_eq!(sum, input, "n = {n}");
        // Shares 1..n-1 are the documented stream, element by element.
        let mut prg = ShakePrg::new(&SEED, DOMAIN_RESHARE.as_bytes(), 0);
        for s in &shares[..n - 1] {
            for x in s {
                assert_eq!(*x, ext_from(&mut prg));
            }
        }
        // A second resharing from the same source is refused.
        assert_eq!(
            PartyRandomness::<K>::reshare(&mut rng, &f, &input, n),
            Err(Error::RandomnessReuse {
                kind: DOMAIN_RESHARE,
                index: 0
            })
        );
    }
    // Unreduced inputs stand for their residue class.
    let unreduced: Vec<Ext<K>> = input.iter().map(|x| Ext(x.0.map(|c| c + Q))).collect();
    let shares =
        PartyRandomness::<K>::reshare(&mut PrgRandomness::new(SEED), &f, &unreduced, 1).unwrap();
    assert_eq!(shares, vec![input.clone()]);
    assert!(matches!(
        PartyRandomness::<K>::reshare(&mut PrgRandomness::new(SEED), &f, &input, 0),
        Err(Error::Parameter(_))
    ));
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
    // Other kinds and other attempts are separate streams.
    PartyRandomness::<K>::rho(&mut rng, 0, 4, 8).unwrap();
    PartyRandomness::<K>::mask(&mut rng, 1, GAMMA, 8).unwrap();
    assert!(matches!(
        PartyRandomness::<K>::rho(&mut rng, 0, 4, 8),
        Err(Error::RandomnessReuse { .. })
    ));
    // A failed request (gamma above 2^63 - 1, b = 0) does not use up its stream.
    assert!(matches!(
        PartyRandomness::<K>::mask(&mut rng, 5, u64::MAX, 8),
        Err(Error::Sampling(_))
    ));
    PartyRandomness::<K>::mask(&mut rng, 5, GAMMA, 8).unwrap();
    assert!(matches!(
        PartyRandomness::<K>::rho(&mut rng, 9, 0, 8),
        Err(Error::Sampling(_))
    ));
    PartyRandomness::<K>::rho(&mut rng, 9, 4, 8).unwrap();
    // Debug output does not show the seed.
    let debug = format!("{rng:?}");
    assert!(debug.starts_with("PrgRandomness"));
    assert!(!debug.contains("seed") && !debug.contains("48, 97, 99"));
}
