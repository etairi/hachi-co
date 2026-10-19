//! The driver `prove`: with `Loopback`, `ReplayRandomness` and `ReplayChallenger` it reproduces
//! toy-n1; on the in-memory network of `hachi-co-mpc` (n threads, masked-sum F_AH) it reproduces
//! the multi-party fixtures at every party; honest Fiat-Shamir runs (`PrgRandomness`, `FsChallenger`)
//! are accepted by the verifier and leave the provers' and the verifier's transcripts in the same
//! state; tampering under Fiat-Shamir is rejected.

mod common;

use common::{Instance, K, fixture_transcript, honest_runs, ideal_pcs, instances};
use hachi_co_core::{
    Challenger, FsChallenger, PartyRandomness, PrgRandomness, ReplayChallenger, ReplayRandomness,
};
use hachi_co_mpc::{Comm, Loopback, MemoryConfig, PartyId, run_in_memory};
use hachi_co_sumcheck::{
    Error, PublicInput, SumcheckPrivate, SumcheckTranscript, VerifyError, prove, verify,
    verify_checks,
};
use std::time::Instant;

/// The Fiat-Shamir domain of the tests.
const DOMAIN: &[u8] = b"hachi-co/test/sumcheck";

/// Runs `prove` at every party of `inst` on threads over the in-memory network, with the randomness and challenger made by
/// `rng` and `ch` for each party.
fn run_network<R, H>(
    inst: &Instance,
    rng: impl Fn(PartyId) -> R + Sync,
    ch: impl Fn(PartyId) -> H + Sync,
) -> Vec<(SumcheckTranscript<K>, SumcheckPrivate<K>, H)>
where
    R: PartyRandomness<K>,
    H: Challenger<K> + Send,
{
    let (n, q) = (inst.n(), inst.ctx.q());
    let mut seed = [n as u8; 32];
    seed[..8].copy_from_slice(&q.to_le_bytes());
    run_in_memory(&MemoryConfig::new(n, q, seed), |net| {
        let p = net.id();
        let (mut r, mut c) = (rng(p), ch(p));
        let (t, private) = prove(
            &inst.ctx,
            net,
            &mut r,
            &mut c,
            &inst.m,
            &inst.e,
            &inst.shares[p - 1],
        )
        .unwrap();
        (t, private, c)
    })
    .expect("a valid network")
    .into_iter()
    .map(|run| run.expect("every party connects").output)
    .collect()
}

fn public(inst: &Instance) -> PublicInput<'_> {
    PublicInput::new(&inst.ctx, &inst.m, &inst.y, &inst.e, &inst.y_ring)
}

/// The driver with `Loopback`, `ReplayRandomness` and `ReplayChallenger` reproduces toy-n1: the
/// transcript, and the private data equal to the lockstep run checked against the fixture.
#[test]
fn loopback_driver_reproduces_toy_n1() {
    let inst = &instances()[0];
    let fx = &inst.fx;
    let mut comm = Loopback::new(inst.ctx.q()).unwrap();
    let mut rng = ReplayRandomness::new(fx, 1).unwrap();
    let mut ch = ReplayChallenger::new(fx).unwrap();
    let (transcript, private) = prove(
        &inst.ctx,
        &mut comm,
        &mut rng,
        &mut ch,
        &inst.m,
        &inst.e,
        &inst.shares[0],
    )
    .unwrap();
    assert_eq!(transcript, fixture_transcript(fx));
    assert_eq!(private, honest_runs()[0].privates[0]);
    assert_eq!(
        private.quotient,
        fx.polys_of_len("/sumcheck/quotient_shares/0", 2 * inst.ctx.layout.d)
            .unwrap()
    );
    let mut ch = ReplayChallenger::new(fx).unwrap();
    let claims = verify(&inst.ctx, &public(inst), &transcript, &mut ch).unwrap();
    assert!(
        ideal_pcs(inst.field(), &claims, &[private])
            .iter()
            .all(|(_, ok)| *ok)
    );
}

/// The driver on n threads with replayed randomness and coins reproduces toy-n2 and toy-n3 at
/// every party (the per-party G and final values travel through one F_AH call each).
#[test]
fn threaded_driver_reproduces_toy_n2_and_n3() {
    for (inst, run) in instances().iter().zip(honest_runs()).skip(1) {
        let fx = &inst.fx;
        let outputs = run_network(
            inst,
            |p| ReplayRandomness::new(fx, p).unwrap(),
            |_| ReplayChallenger::new(fx).unwrap(),
        );
        let recorded = fixture_transcript(fx);
        for (p0, (transcript, private, _)) in outputs.iter().enumerate() {
            assert_eq!(transcript, &recorded, "{}: party {}", inst.name, p0 + 1);
            assert_eq!(
                private,
                &run.privates[p0],
                "{}: party {}",
                inst.name,
                p0 + 1
            );
        }
    }
}

/// Honest Fiat-Shamir runs (fresh `PrgRandomness` seeds 1, 2, 3 per party, `FsChallenger` on the
/// domain "hachi-co/test/sumcheck") at toy-n1 (Loopback) and at toy-n2, toy-n3 (threads) are
/// accepted; the claims are the provers' polynomials; every party's transcript and the verifier's
/// end in the same state. One run at toy-n1 is timed (printed with `--nocapture`).
#[test]
fn honest_fiat_shamir_runs_are_accepted() {
    for seed in 1u8..=3 {
        let inst = &instances()[0];
        let start = Instant::now();
        let mut comm = Loopback::new(inst.ctx.q()).unwrap();
        let mut rng = PrgRandomness::new([seed; 32]);
        let mut ch = FsChallenger::new(DOMAIN);
        let (transcript, private) = prove(
            &inst.ctx,
            &mut comm,
            &mut rng,
            &mut ch,
            &inst.m,
            &inst.e,
            &inst.shares[0],
        )
        .unwrap();
        let proved = start.elapsed();
        let start = Instant::now();
        let mut vch = FsChallenger::new(DOMAIN);
        let claims = verify(&inst.ctx, &public(inst), &transcript, &mut vch).unwrap();
        let verified = start.elapsed();
        eprintln!(
            "toy-n1, PrgRandomness seed [{seed}; 32]: prove {proved:?}, verify {verified:?} \
             (l'' = {}, 2^l'' = {} points)",
            inst.ctx.layout.ell_pp,
            1usize << inst.ctx.layout.ell_pp
        );
        assert_eq!(ch.transcript().digest(), vch.transcript().digest());
        assert!(
            ideal_pcs(inst.field(), &claims, &[private])
                .iter()
                .all(|(_, ok)| *ok)
        );
        assert_ne!(transcript, fixture_transcript(&inst.fx));
    }
    for inst in &instances()[1..] {
        let outputs = run_network(
            inst,
            |p| PrgRandomness::new([p as u8 + 16; 32]),
            |_| FsChallenger::new(DOMAIN),
        );
        let transcript = &outputs[0].0;
        let mut vch = FsChallenger::new(DOMAIN);
        let claims = verify(&inst.ctx, &public(inst), transcript, &mut vch).unwrap();
        let privates: Vec<SumcheckPrivate<K>> = outputs.iter().map(|o| o.1.clone()).collect();
        assert!(
            ideal_pcs(inst.field(), &claims, &privates)
                .iter()
                .all(|(_, ok)| *ok)
        );
        for (t, _, ch) in &outputs {
            assert_eq!(t, transcript);
            assert_eq!(ch.transcript().digest(), vch.transcript().digest());
        }
    }
}

/// Under Fiat-Shamir a tampered message also changes every later coin; the verifier rejects, and
/// a verifier on another domain (another statement) rejects the honest transcript. A challenger
/// that absorbed something else first, too.
#[test]
fn fiat_shamir_tampering_is_rejected() {
    let inst = &instances()[0];
    let f = inst.field();
    let mut comm = Loopback::new(inst.ctx.q()).unwrap();
    let mut rng = PrgRandomness::new([7; 32]);
    let mut ch = FsChallenger::new(DOMAIN);
    let (transcript, _) = prove(
        &inst.ctx,
        &mut comm,
        &mut rng,
        &mut ch,
        &inst.m,
        &inst.e,
        &inst.shares[0],
    )
    .unwrap();
    let rejected = |t: &SumcheckTranscript<K>, mut vch: FsChallenger| {
        matches!(
            verify(&inst.ctx, &public(inst), t, &mut vch),
            Err(VerifyError::Rejected { .. })
        )
    };
    assert!(!rejected(&transcript, FsChallenger::new(DOMAIN)));
    for r in [0, 7, transcript.rounds.len() - 1] {
        let mut t = transcript.clone();
        t.rounds[r].h1[2] = f.add(&t.rounds[r].h1[2], &f.one());
        assert!(rejected(&t, FsChallenger::new(DOMAIN)), "round {}", r + 1);
        // A consistent change (the sum at 0 and 1 unchanged) changes the next coins.
        let mut t = transcript.clone();
        for (i, v) in t.rounds[r].h0.iter_mut().enumerate() {
            let x = (i + 1) as u64;
            *v = f.add(v, &f.from_base(3 * x * (x - 1)));
        }
        let mut vch = FsChallenger::new(DOMAIN);
        let report = verify_checks(&inst.ctx, &public(inst), &t, &mut vch).unwrap();
        assert!(!report.accepted(), "round {}", r + 1);
    }
    let mut t = transcript.clone();
    t.g1_sums[0] = f.add(&t.g1_sums[0], &f.one());
    assert!(rejected(&t, FsChallenger::new(DOMAIN)));
    let mut t = transcript.clone();
    t.finals.g0[0] = f.add(&t.finals.g0[0], &f.one());
    assert!(rejected(&t, FsChallenger::new(DOMAIN)));
    assert!(rejected(
        &transcript,
        FsChallenger::new(b"another statement")
    ));
    let mut other = FsChallenger::new(DOMAIN);
    Challenger::<K>::absorb_u64(&mut other, "something else", 1);
    assert!(rejected(&transcript, other));
}

/// The driver refuses a network that does not match the context, and a prover cannot draw a
/// blinder stream twice from one `PrgRandomness` (a second Eval needs a fresh seed).
#[test]
fn driver_refuses_mismatched_networks_and_reused_randomness() {
    let inst = &instances()[1]; // n = 2
    let mut comm = Loopback::new(inst.ctx.q()).unwrap();
    let mut rng = PrgRandomness::new([1; 32]);
    let mut ch = FsChallenger::new(DOMAIN);
    let r = prove(
        &inst.ctx,
        &mut comm,
        &mut rng,
        &mut ch,
        &inst.m,
        &inst.e,
        &inst.shares[0],
    );
    assert!(matches!(r, Err(Error::Parameter(_))));
    let inst = &instances()[0];
    let mut comm = Loopback::new(inst.ctx.q() + 2).unwrap();
    let r = prove(
        &inst.ctx,
        &mut comm,
        &mut rng,
        &mut ch,
        &inst.m,
        &inst.e,
        &inst.shares[0],
    );
    assert!(matches!(r, Err(Error::Parameter(_))));
    let mut comm = Loopback::new(inst.ctx.q()).unwrap();
    let mut rng = PrgRandomness::new([1; 32]);
    prove(
        &inst.ctx,
        &mut comm,
        &mut rng,
        &mut ch,
        &inst.m,
        &inst.e,
        &inst.shares[0],
    )
    .unwrap();
    let again = prove(
        &inst.ctx,
        &mut comm,
        &mut rng,
        &mut ch,
        &inst.m,
        &inst.e,
        &inst.shares[0],
    );
    assert!(matches!(
        again,
        Err(Error::Core(hachi_co_core::Error::RandomnessReuse { .. }))
    ));
    // A share of the wrong shape is an error.
    let mut bad = inst.shares[0].clone();
    bad.z_prime.pop();
    let mut rng = PrgRandomness::new([2; 32]);
    let r = prove(
        &inst.ctx, &mut comm, &mut rng, &mut ch, &inst.m, &inst.e, &bad,
    );
    assert!(r.is_err());
}
