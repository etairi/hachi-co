//! The replay providers return exactly the values the oracle recorded, for every party of every
//! fixture, and refuse requests that do not match the recording.
#![cfg(feature = "fixtures")]

use hachi_co_core::{
    BlinderKind, Challenger, Error, Fixture, Fqk, PartyRandomness, Poly, ReplayChallenger,
    ReplayRandomness,
};
use hachi_co_sampling::l1ball;

const K: usize = 4;

fn replay_err<T: core::fmt::Debug>(r: hachi_co_core::Result<T>) -> bool {
    matches!(r, Err(Error::Replay(_)))
}

#[test]
fn replay_randomness_serves_the_recorded_values() {
    for name in Fixture::TOY {
        let fx = Fixture::load(name).unwrap();
        let ctx = fx.context::<K>().unwrap();
        let l = &ctx.layout;
        let f = ctx.field();
        let attempts = fx
            .get("/ring_to_linear/attempts")
            .unwrap()
            .as_array()
            .unwrap()
            .len();
        for p in 1..=fx.n() {
            let mut rng = ReplayRandomness::new(&fx, p).unwrap();
            assert_eq!((rng.party(), rng.attempts()), (p, attempts));
            // AS.Share of the party's initial share: all n recorded shares, the last recomputed.
            let input = fx
                .exts::<K>(&format!("/input/f_init_shares/{}", p - 1))
                .unwrap();
            let shares = rng.reshare(f, &input, fx.n()).unwrap();
            assert_eq!(
                shares,
                fx.ext_rows::<K>(&format!("/commit/reshare/{}", p - 1))
                    .unwrap(),
                "{name} p = {p}"
            );
            // Another input: shares 1..n-1 stay, share n follows the input.
            let mut other = input.clone();
            other[0] = f.add(&other[0], &f.one());
            let shares2 = rng.reshare(f, &other, fx.n()).unwrap();
            assert_eq!(shares2[..fx.n() - 1], shares[..fx.n() - 1]);
            assert_eq!(
                shares2[fx.n() - 1][0],
                f.add(&shares[fx.n() - 1][0], &f.one())
            );
            // Masks and rho of every attempt, flattened (element i, coefficient j at i d + j).
            for a in 0..attempts {
                let prefix = format!("/ring_to_linear/attempts/{a}/provers/{}", p - 1);
                let mask =
                    PartyRandomness::<K>::mask(&mut rng, a as u64, ctx.gamma(), l.width_a * l.d);
                assert_eq!(mask.unwrap(), fx.ints(&format!("{prefix}/mask")).unwrap());
                let rows = fx.int_rows(&format!("{prefix}/mask")).unwrap();
                assert_eq!((rows.len(), rows[0].len()), (l.width_a, l.d));
                let rho = PartyRandomness::<K>::rho(&mut rng, a as u64, ctx.b(), l.m_rho * l.d);
                assert_eq!(rho.unwrap(), fx.ints(&format!("{prefix}/rho")).unwrap());
            }
            // Blinders.
            for (kind, key) in [(BlinderKind::G0, "g0"), (BlinderKind::G1, "g1")] {
                let g = rng.blinder(f, kind, l.ell_pp, kind.degree(l)).unwrap();
                let prefix = format!("/sumcheck/blinders/{}/{key}", p - 1);
                assert_eq!(g.s0, fx.ext::<K>(&format!("{prefix}/s0")).unwrap());
                assert_eq!(g.s, fx.ext_rows::<K>(&format!("{prefix}/s")).unwrap());
                g.validate(l.ell_pp, kind.degree(l)).unwrap();
            }
        }
    }
}

#[test]
fn replay_randomness_refuses_mismatched_requests() {
    let fx = Fixture::load("toy-n3").unwrap();
    let ctx = fx.context::<K>().unwrap();
    let l = &ctx.layout;
    let f = ctx.field();
    for p in [0, 4] {
        assert_eq!(
            ReplayRandomness::new(&fx, p),
            Err(Error::InvalidParty { party: p, n: 3 })
        );
    }
    let mut rng = ReplayRandomness::new(&fx, 2).unwrap();
    let n_mask = l.width_a * l.d;
    let attempts = rng.attempts() as u64;
    assert!(replay_err(PartyRandomness::<K>::mask(
        &mut rng,
        attempts,
        ctx.gamma(),
        n_mask
    )));
    assert!(replay_err(PartyRandomness::<K>::mask(
        &mut rng,
        u64::MAX,
        ctx.gamma(),
        n_mask
    )));
    assert!(replay_err(PartyRandomness::<K>::mask(
        &mut rng,
        0,
        ctx.gamma() + 1,
        n_mask
    )));
    assert!(replay_err(PartyRandomness::<K>::mask(
        &mut rng,
        0,
        ctx.gamma(),
        n_mask - 1
    )));
    assert!(replay_err(PartyRandomness::<K>::rho(
        &mut rng,
        0,
        ctx.b() + 1,
        l.m_rho * l.d
    )));
    assert!(replay_err(PartyRandomness::<K>::rho(
        &mut rng,
        0,
        ctx.b(),
        l.m_rho * l.d + 1
    )));
    assert!(replay_err(PartyRandomness::<K>::rho(
        &mut rng,
        attempts,
        ctx.b(),
        l.m_rho * l.d
    )));
    let input = fx.exts::<K>("/input/f_init_shares/1").unwrap();
    assert!(replay_err(rng.reshare(f, &input, 2)));
    assert!(replay_err(rng.reshare(f, &input[1..], 3)));
    assert!(replay_err(rng.blinder(f, BlinderKind::G0, l.ell_pp, l.d1)));
    assert!(replay_err(rng.blinder(
        f,
        BlinderKind::G1,
        l.ell_pp + 1,
        l.d1
    )));
    // Another modulus, or another extension degree.
    let other = Fqk::<K>::new(8388637).unwrap();
    assert!(replay_err(rng.reshare(&other, &input, 3)));
    assert!(replay_err(rng.blinder(
        &other,
        BlinderKind::G0,
        l.ell_pp,
        l.d0
    )));
    let k2 = Fqk::<2>::new(fx.q()).unwrap();
    assert!(replay_err(rng.blinder(
        &k2,
        BlinderKind::G0,
        l.ell_pp,
        l.d0
    )));
    // Repeated requests are fine for the replay.
    let first = PartyRandomness::<K>::mask(&mut rng, 0, ctx.gamma(), n_mask).unwrap();
    assert_eq!(
        PartyRandomness::<K>::mask(&mut rng, 0, ctx.gamma(), n_mask).unwrap(),
        first
    );
}

#[test]
fn replay_challenger_serves_the_recorded_coins() {
    for name in Fixture::TOY {
        let fx = Fixture::load(name).unwrap();
        let ctx = fx.context::<K>().unwrap();
        let l = &ctx.layout;
        let f = ctx.field();
        let mut ch = ReplayChallenger::new(&fx).unwrap();
        assert_eq!(ch.rounds(), l.ell_pp);
        for a in 0..ch.attempts() {
            // Absorbs and attempts are ignored.
            Challenger::<K>::begin_attempt(&mut ch, a as u64).unwrap();
            Challenger::<K>::absorb_polys(&mut ch, "v", &[Poly(vec![1; l.d])]);
            Challenger::<K>::absorb_u64(&mut ch, "attempt", a as u64);
            let c =
                Challenger::<K>::r2l_challenges(&mut ch, a as u64, l.n_blocks, l.d, ctx.omega());
            let c = c.unwrap();
            assert_eq!(
                c,
                fx.int_rows(&format!("/ring_to_linear/attempts/{a}/c"))
                    .unwrap()
            );
            assert!(
                c.iter().all(|ci| l1ball::contains(ci, ctx.omega())),
                "{name}: c in C"
            );
            assert!(
                fx.bool(&format!("/ring_to_linear/attempts/{a}/c_in_C"))
                    .unwrap()
            );
        }
        ch.absorb_exts("G", &[f.one()]);
        let coins = ch.sumcheck_coins(f, l.ell_pp, l.log_n, l.log_nu).unwrap();
        coins.validate(l, ctx.q()).unwrap();
        assert_eq!(coins.alpha, fx.ext::<K>("/sumcheck/coins/alpha").unwrap());
        assert_eq!(coins.rho, fx.ext::<K>("/sumcheck/coins/rho").unwrap());
        assert_eq!(coins.tau0, fx.exts::<K>("/sumcheck/coins/tau0").unwrap());
        assert_eq!(coins.tau1, fx.exts::<K>("/sumcheck/coins/tau1").unwrap());
        for r in 1..=l.ell_pp {
            assert_eq!(
                ch.round_coin(f, r).unwrap(),
                fx.ext::<K>(&format!("/sumcheck/coins/a/{}", r - 1))
                    .unwrap(),
                "{name} round {r}"
            );
        }
        assert!(replay_err(ch.round_coin(f, 0)));
        assert!(replay_err(ch.round_coin(f, l.ell_pp + 1)));
    }
}

#[test]
fn replay_challenger_refuses_mismatched_requests() {
    let fx = Fixture::load("toy-n1").unwrap();
    let ctx = fx.context::<K>().unwrap();
    let l = &ctx.layout;
    let f = ctx.field();
    let mut ch = ReplayChallenger::new(&fx).unwrap();
    let attempts = ch.attempts() as u64;
    let (count, d, omega) = (l.n_blocks, l.d, ctx.omega());
    assert!(replay_err(Challenger::<K>::r2l_challenges(
        &mut ch, attempts, count, d, omega
    )));
    assert!(replay_err(Challenger::<K>::r2l_challenges(
        &mut ch,
        0,
        count + 1,
        d,
        omega
    )));
    assert!(replay_err(Challenger::<K>::r2l_challenges(
        &mut ch,
        0,
        count,
        d - 1,
        omega
    )));
    assert!(replay_err(Challenger::<K>::r2l_challenges(
        &mut ch,
        0,
        count,
        d,
        omega + 1
    )));
    assert!(replay_err(ch.sumcheck_coins(
        f,
        l.ell_pp + 1,
        l.log_n,
        l.log_nu
    )));
    assert!(replay_err(ch.sumcheck_coins(
        f,
        l.ell_pp,
        l.log_n,
        l.log_nu - 1
    )));
    let other = Fqk::<K>::new(33554501).unwrap();
    assert!(replay_err(
        ch.sumcheck_coins(&other, l.ell_pp, l.log_n, l.log_nu)
    ));
    assert!(replay_err(ch.round_coin(&other, 1)));
}
