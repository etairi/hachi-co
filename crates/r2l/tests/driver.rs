//! The driver `run`: it reproduces the fixtures with replayed randomness and coins (toy-n1 on the
//! loopback; the other fixtures on the in-memory network), honest Fiat-Shamir runs satisfy
//! M z' = y and the verifier re-derives their challenges, the attempt cap holds, and misbehaving
//! components (network, challenger, randomness) give errors, not panics.

mod common;

use common::{K, attempts, load, run_parties, u_com, u_share};
use hachi_co_core::{
    Challenger, Context, Ext, Fixture, Fqk, FsChallenger, PartyRandomness, PartyState, Poly,
    PrgRandomness, PublicVectors, ReplayChallenger, ReplayRandomness, SumcheckCoins,
};
use hachi_co_mpc::{Comm, Loopback, PartyId};
use hachi_co_r2l::{
    Error, MAX_ATTEMPTS, R2lOutput, linear_relation, relation_holds, run, run_with_cap, sum_shares,
    verifier_challenges,
};
use sha2::{Digest, Sha256};
use std::sync::Mutex;

const DOMAIN: &[u8] = b"hachi-co r2l driver test";

/// A 32-byte seed from a tag and indices (SHA-256 of the tag and the indices as LE64).
fn seed(tag: &str, indices: &[u64]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(tag.as_bytes());
    for i in indices {
        h.update(i.to_le_bytes());
    }
    h.finalize().into()
}

/// Checks a run's output against the fixture: per attempt v, c, this party's accept bit and the
/// abort; the accepted attempt, v and c.
fn compare_with_fixture(fx: &Fixture, out: &R2lOutput, p: PartyId) {
    let name = fx.name();
    let accepted = fx.usize("/ring_to_linear/accepted_attempt").unwrap();
    assert_eq!(out.accepted_attempt, accepted as u64, "{name} p={p}");
    assert_eq!(out.attempts.len(), attempts(fx), "{name} p={p}");
    for (a, rec) in out.attempts.iter().enumerate() {
        let at = |k: &str| format!("/ring_to_linear/attempts/{a}/{k}");
        assert_eq!(rec.attempt, a as u64);
        assert_eq!(rec.v, fx.polys(&at("v")).unwrap(), "{name} p={p} a={a}: v");
        assert_eq!(
            rec.c,
            fx.int_rows(&at("c")).unwrap(),
            "{name} p={p} a={a}: c"
        );
        assert_eq!(
            rec.accepted,
            fx.bool(&at(&format!("provers/{}/accept", p - 1))).unwrap(),
            "{name} p={p} a={a}: accept"
        );
        assert_eq!(
            rec.aborted,
            fx.bool(&at("abort")).unwrap(),
            "{name} a={a}: abort"
        );
    }
    let at = |k: &str| format!("/ring_to_linear/attempts/{accepted}/{k}");
    assert_eq!(out.v, fx.polys(&at("v")).unwrap());
    assert_eq!(out.c, fx.int_rows(&at("c")).unwrap());
}

/// Runs every party of the fixture's instance in its own thread over the in-memory network.
fn run_threaded<R, H>(
    ctx: &Context<K>,
    pv: &PublicVectors<K>,
    states: &[PartyState<K>],
    u_shares: &[Poly],
    u: &[Poly],
    providers: Vec<(R, H)>,
) -> Vec<hachi_co_r2l::Result<R2lOutput>>
where
    R: PartyRandomness<K> + Send,
    H: Challenger<K> + Send,
{
    let providers: Vec<Mutex<Option<(R, H)>>> =
        providers.into_iter().map(|p| Mutex::new(Some(p))).collect();
    run_parties(states.len(), ctx.q(), |p, comm| {
        let (mut rng, mut ch) = providers[p - 1]
            .lock()
            .unwrap()
            .take()
            .expect("one provider per party");
        run(
            ctx,
            comm,
            &mut rng,
            &mut ch,
            &states[p - 1],
            pv,
            &u_shares[p - 1],
            u,
        )
    })
}

#[test]
fn loopback_replay_reproduces_toy_n1_with_its_aborted_attempts() {
    let (fx, ctx, pv, states) = load("toy-n1");
    let mut comm = Loopback::new(ctx.q()).unwrap();
    let mut rng = ReplayRandomness::new(&fx, 1).unwrap();
    let mut ch = ReplayChallenger::new(&fx).unwrap();
    let out = run(
        &ctx,
        &mut comm,
        &mut rng,
        &mut ch,
        &states[0],
        &pv,
        &u_share(&fx, 1),
        &u_com(&fx),
    )
    .unwrap();
    compare_with_fixture(&fx, &out, 1);
    assert_eq!(
        out.attempts.iter().map(|r| r.aborted).collect::<Vec<_>>(),
        [true, true, true, true, false],
        "toy-n1 has four aborted attempts (ref/README.md, choice 19)"
    );
    // n = 1: the share is the whole witness, <u>_1 = u included, and the whole y.
    assert_eq!(
        out.share.z_prime,
        fx.polys("/ring_to_linear/z_prime").unwrap()
    );
    assert_eq!(out.share.y, fx.polys("/ring_to_linear/y").unwrap());
    let rel = linear_relation(&ctx, &pv, &out.v, &out.c, &u_com(&fx)).unwrap();
    assert!(relation_holds(&ctx, &rel, &out.share.z_prime).unwrap());
}

#[test]
fn the_attempt_cap_is_enforced() {
    let (fx, ctx, pv, states) = load("toy-n1");
    let accepted = fx.usize("/ring_to_linear/accepted_attempt").unwrap() as u64;
    for cap in [0, 1, accepted] {
        let mut comm = Loopback::new(ctx.q()).unwrap();
        let mut rng = ReplayRandomness::new(&fx, 1).unwrap();
        let mut ch = ReplayChallenger::new(&fx).unwrap();
        let r = run_with_cap(
            &ctx,
            &mut comm,
            &mut rng,
            &mut ch,
            &states[0],
            &pv,
            &u_share(&fx, 1),
            &u_com(&fx),
            cap,
        );
        assert_eq!(r, Err(Error::TooManyAttempts { attempts: cap }));
    }
    let mut comm = Loopback::new(ctx.q()).unwrap();
    let mut rng = ReplayRandomness::new(&fx, 1).unwrap();
    let mut ch = ReplayChallenger::new(&fx).unwrap();
    let out = run_with_cap(
        &ctx,
        &mut comm,
        &mut rng,
        &mut ch,
        &states[0],
        &pv,
        &u_share(&fx, 1),
        &u_com(&fx),
        accepted + 1,
    )
    .unwrap();
    assert_eq!(out.accepted_attempt, accepted);
    assert_eq!(MAX_ATTEMPTS, 64);
}

#[test]
fn threaded_replay_reproduces_every_fixture() {
    for name in Fixture::TOY {
        let (fx, ctx, pv, states) = load(name);
        let n = fx.n();
        let u_shares: Vec<Poly> = (1..=n).map(|p| u_share(&fx, p)).collect();
        let providers = (1..=n)
            .map(|p| {
                (
                    ReplayRandomness::new(&fx, p).unwrap(),
                    ReplayChallenger::new(&fx).unwrap(),
                )
            })
            .collect();
        let outs: Vec<R2lOutput> =
            run_threaded(&ctx, &pv, &states, &u_shares, &u_com(&fx), providers)
                .into_iter()
                .map(Result::unwrap)
                .collect();
        for (p, out) in (1..=n).zip(&outs) {
            compare_with_fixture(&fx, out, p);
            // y sits at prover 1 only (5_Building_Blocks.tex:134).
            assert_eq!(
                out.share.y.iter().any(|y| y.0.iter().any(|x| *x != 0)),
                p == 1
            );
        }
        let shares: Vec<_> = outs.iter().map(|o| o.share.clone()).collect();
        let sum = sum_shares(&ctx, &shares).unwrap();
        assert_eq!(
            sum.z_prime,
            fx.polys("/ring_to_linear/z_prime").unwrap(),
            "{name}"
        );
        assert_eq!(sum.y, fx.polys("/ring_to_linear/y").unwrap(), "{name}");
        let rel = linear_relation(&ctx, &pv, &outs[0].v, &outs[0].c, &u_com(&fx)).unwrap();
        assert!(relation_holds(&ctx, &rel, &sum.z_prime).unwrap(), "{name}");
    }
}

/// Honest runs with Fiat-Shamir at toy-n1: fresh seeds, the relation holds for the verifier's M
/// and y, the verifier re-derives c from (a*, v) and reaches the provers' transcript state, and
/// the aborted attempts had other challenges.
#[test]
fn honest_fiat_shamir_runs_at_toy_n1_satisfy_the_relation() {
    let (fx, ctx, pv, states) = load("toy-n1");
    let u = u_com(&fx);
    let runs = 24;
    let mut aborted = 0;
    for i in 0..runs {
        let mut comm = Loopback::new(ctx.q()).unwrap();
        let mut rng = PrgRandomness::new(seed("hachi-co-r2l/test/fs/toy-n1", &[i]));
        // In odd runs a statement is absorbed before the step, as the PCS will do; the verifier
        // starts from the same state.
        let statement = |ch: &mut FsChallenger| {
            if i % 2 == 1 {
                Challenger::<K>::absorb_u64(ch, "test/statement", i);
            }
        };
        let mut ch = FsChallenger::new(DOMAIN);
        statement(&mut ch);
        let out = run(
            &ctx,
            &mut comm,
            &mut rng,
            &mut ch,
            &states[0],
            &pv,
            &u_share(&fx, 1),
            &u,
        )
        .unwrap();
        let rel = linear_relation(&ctx, &pv, &out.v, &out.c, &u).unwrap();
        assert!(
            relation_holds(&ctx, &rel, &out.share.z_prime).unwrap(),
            "run {i}"
        );
        // The verifier, from the proof values (a*, v) only.
        let mut vch = FsChallenger::new(DOMAIN);
        statement(&mut vch);
        let c = verifier_challenges(&ctx, &mut vch, out.accepted_attempt, &out.v).unwrap();
        assert_eq!(c, out.c, "run {i}");
        assert_eq!(vch.transcript().digest(), ch.transcript().digest());
        // Without the statement (or with another one) the verifier gets other challenges.
        let mut other = FsChallenger::new(DOMAIN);
        Challenger::<K>::absorb_u64(&mut other, "test/statement", i + 1);
        let c_other = verifier_challenges(&ctx, &mut other, out.accepted_attempt, &out.v).unwrap();
        assert_ne!(c_other, out.c);
        // The log: aborted attempts first, the accepted one last, all with distinct challenges.
        let (last, earlier) = out.attempts.split_last().unwrap();
        assert!(!last.aborted && last.accepted && last.c == out.c && last.v == out.v);
        assert!(
            earlier
                .iter()
                .all(|r| r.aborted && !r.accepted && r.c != out.c)
        );
        aborted += earlier.len();
        // A tampered v or another attempt index gives the verifier other challenges.
        let mut v2 = out.v.clone();
        v2[0].0[0] = (v2[0].0[0] + 1) % ctx.q();
        let c2 = verifier_challenges(
            &ctx,
            &mut FsChallenger::new(DOMAIN),
            out.accepted_attempt,
            &v2,
        )
        .unwrap();
        assert_ne!(c2, out.c);
        let c3 = verifier_challenges(
            &ctx,
            &mut FsChallenger::new(DOMAIN),
            out.accepted_attempt + 1,
            &out.v,
        )
        .unwrap();
        assert_ne!(c3, out.c);
        // The randomness of a run is served once: running again with it is refused.
        let mut ch = FsChallenger::new(DOMAIN);
        let again = run(
            &ctx,
            &mut comm,
            &mut rng,
            &mut ch,
            &states[0],
            &pv,
            &u_share(&fx, 1),
            &u,
        );
        assert!(matches!(
            again,
            Err(Error::Core(hachi_co_core::Error::RandomnessReuse { .. }))
        ));
    }
    // Diagnostic (not an assertion): about 0.39 aborts per attempt at toy-n1.
    println!("toy-n1: {runs} honest Fiat-Shamir runs, {aborted} aborted attempts");
}

/// Honest Fiat-Shamir runs of n = 2, 3, 4 and 8 parties on the in-memory network: every party has its
/// own randomness and its own challenger replica, and all reach the same v, c and attempt.
#[test]
fn honest_fiat_shamir_runs_with_threads_agree_and_satisfy_the_relation() {
    for name in ["toy-n2", "toy-n3", "toy-n4", "toy-n8"] {
        let (fx, ctx, pv, states) = load(name);
        let n = fx.n();
        let u = u_com(&fx);
        let u_shares: Vec<Poly> = (1..=n).map(|p| u_share(&fx, p)).collect();
        for i in 0..3u64 {
            let providers = (1..=n)
                .map(|p| {
                    (
                        PrgRandomness::new(seed(
                            &format!("hachi-co-r2l/test/fs/{name}"),
                            &[i, p as u64],
                        )),
                        FsChallenger::new(DOMAIN),
                    )
                })
                .collect();
            let outs: Vec<R2lOutput> = run_threaded(&ctx, &pv, &states, &u_shares, &u, providers)
                .into_iter()
                .map(Result::unwrap)
                .collect();
            for out in &outs[1..] {
                assert_eq!(out.accepted_attempt, outs[0].accepted_attempt);
                assert_eq!(out.v, outs[0].v);
                assert_eq!(out.c, outs[0].c);
                assert_eq!(out.attempts.len(), outs[0].attempts.len());
            }
            let shares: Vec<_> = outs.iter().map(|o| o.share.clone()).collect();
            let sum = sum_shares(&ctx, &shares).unwrap();
            let rel = linear_relation(&ctx, &pv, &outs[0].v, &outs[0].c, &u).unwrap();
            assert!(
                relation_holds(&ctx, &rel, &sum.z_prime).unwrap(),
                "{name} run {i}"
            );
            let c = verifier_challenges(
                &ctx,
                &mut FsChallenger::new(DOMAIN),
                outs[0].accepted_attempt,
                &outs[0].v,
            )
            .unwrap();
            assert_eq!(c, outs[0].c);
        }
    }
}

#[test]
fn verifier_side_rejects_malformed_proof_values() {
    let (fx, ctx, pv, _) = load("toy-n2");
    let l = &ctx.layout;
    let q = ctx.q();
    let a = fx.usize("/ring_to_linear/accepted_attempt").unwrap();
    let v = fx
        .polys(&format!("/ring_to_linear/attempts/{a}/v"))
        .unwrap();
    let c = fx
        .int_rows(&format!("/ring_to_linear/attempts/{a}/c"))
        .unwrap();
    let u = u_com(&fx);
    linear_relation(&ctx, &pv, &v, &c, &u).unwrap();
    let omega = ctx.omega();
    // c outside C: l1 norm omega + 1, an extreme coefficient, one challenge too few or too many,
    // a challenge with a coefficient too few.
    let mut bad = c.clone();
    bad[3] = vec![0; l.d];
    bad[3][0] = omega as i64;
    bad[3][5] = -1;
    assert_eq!(
        linear_relation(&ctx, &pv, &v, &bad, &u),
        Err(Error::ChallengeNotInC {
            index: 3,
            l1_norm: u128::from(omega) + 1,
            omega
        })
    );
    bad[3] = vec![0; l.d];
    bad[3][7] = i64::MIN;
    assert!(matches!(
        linear_relation(&ctx, &pv, &v, &bad, &u),
        Err(Error::ChallengeNotInC { index: 3, .. })
    ));
    // The boundary: l1 norm exactly omega is in C.
    bad[3] = vec![0; l.d];
    bad[3][2] = -(omega as i64);
    linear_relation(&ctx, &pv, &v, &bad, &u).unwrap();
    let shape = |r: hachi_co_r2l::Result<_>| matches!(r, Err(Error::Shape { .. }));
    assert!(shape(linear_relation(&ctx, &pv, &v, &c[1..], &u)));
    let mut more = c.clone();
    more.push(vec![0; l.d]);
    assert!(shape(linear_relation(&ctx, &pv, &v, &more, &u)));
    let mut short_c = c.clone();
    short_c[0].pop();
    assert!(shape(linear_relation(&ctx, &pv, &v, &short_c, &u)));
    // v and u: wrong counts, wrong lengths, non-canonical residues.
    assert!(shape(linear_relation(&ctx, &pv, &[], &c, &u)));
    assert!(shape(linear_relation(&ctx, &pv, &v, &c, &[])));
    let mut v_bad = v.clone();
    v_bad[0].0.pop();
    assert!(shape(linear_relation(&ctx, &pv, &v_bad, &c, &u)));
    let mut v_bad = v.clone();
    v_bad[0].0[3] = q;
    assert_eq!(
        linear_relation(&ctx, &pv, &v_bad, &c, &u),
        Err(Error::NonCanonical {
            what: "v",
            value: q,
            q
        })
    );
    let mut u_bad = u.clone();
    u_bad[0].0[0] = u64::MAX;
    assert!(matches!(
        linear_relation(&ctx, &pv, &v, &c, &u_bad),
        Err(Error::NonCanonical { .. })
    ));
    let mut ch = FsChallenger::new(DOMAIN);
    assert!(matches!(
        verifier_challenges(&ctx, &mut ch, 0, &v_bad),
        Err(Error::NonCanonical { .. })
    ));
    assert!(matches!(
        verifier_challenges(&ctx, &mut ch, 0, &v[..0]),
        Err(Error::Shape { .. })
    ));
    // Public vectors of the wrong shape.
    let mut pv_bad = pv.clone();
    pv_bad.a_ring.pop();
    assert!(matches!(
        linear_relation(&ctx, &pv_bad, &v, &c, &u),
        Err(Error::Core(hachi_co_core::Error::Shape { .. }))
    ));
}

// ------------------------------------------------------------ misbehaving components ----

/// How [`Faulty`] tampers with the loopback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    FahShort,
    FahNonCanonical,
    AbortLies,
    GatherNone,
    GatherShort,
    GatherHuge,
    Prover1Lies,
}

/// The loopback with one fault.
struct Faulty {
    inner: Loopback,
    fault: Fault,
}

impl Comm for Faulty {
    fn n(&self) -> usize {
        1
    }
    fn id(&self) -> PartyId {
        1
    }
    fn modulus(&self) -> u64 {
        self.inner.modulus()
    }
    fn fah(&mut self, label: &'static str, local: &[u64]) -> hachi_co_mpc::Result<Vec<u64>> {
        let mut out = self.inner.fah(label, local)?;
        match self.fault {
            Fault::FahShort => {
                out.pop();
            }
            Fault::FahNonCanonical => out[0] = self.inner.modulus(),
            _ => {}
        }
        Ok(out)
    }
    fn gather(
        &mut self,
        label: &'static str,
        to: PartyId,
        local: &[i64],
    ) -> hachi_co_mpc::Result<Option<Vec<Vec<i64>>>> {
        let mut out = self.inner.gather(label, to, local)?;
        match self.fault {
            Fault::GatherNone => out = None,
            Fault::GatherShort => {
                if let Some(v) = out.as_mut() {
                    v[0].pop();
                }
            }
            Fault::GatherHuge => {
                if let Some(v) = out.as_mut() {
                    v[0][0] = i64::MAX;
                }
            }
            _ => {}
        }
        Ok(out)
    }
    fn any(&mut self, label: &'static str, local: bool) -> hachi_co_mpc::Result<bool> {
        let out = self.inner.any(label, local)?;
        Ok(match (self.fault, label) {
            (Fault::AbortLies, hachi_co_r2l::COMM_ABORT) => false,
            (Fault::Prover1Lies, hachi_co_r2l::COMM_PROVER1) => true,
            _ => out,
        })
    }
    fn exchange(
        &mut self,
        label: &'static str,
        outgoing: Vec<Vec<u64>>,
    ) -> hachi_co_mpc::Result<Vec<Vec<u64>>> {
        self.inner.exchange(label, outgoing)
    }
    fn all_gather(
        &mut self,
        label: &'static str,
        local: &[u64],
    ) -> hachi_co_mpc::Result<Vec<Vec<u64>>> {
        self.inner.all_gather(label, local)
    }
}

/// The replay challenger, with challenge `c_0` replaced in every attempt.
struct BadChallenger {
    inner: ReplayChallenger,
    c0: Vec<i64>,
}

impl Challenger<K> for BadChallenger {
    fn absorb_polys(&mut self, label: &'static str, polys: &[Poly]) {
        Challenger::<K>::absorb_polys(&mut self.inner, label, polys);
    }
    fn absorb_exts(&mut self, label: &'static str, exts: &[Ext<K>]) {
        self.inner.absorb_exts(label, exts);
    }
    fn absorb_u64(&mut self, label: &'static str, value: u64) {
        Challenger::<K>::absorb_u64(&mut self.inner, label, value);
    }
    fn begin_attempt(&mut self, attempt: u64) -> hachi_co_core::Result<()> {
        Challenger::<K>::begin_attempt(&mut self.inner, attempt)
    }
    fn r2l_challenges(
        &mut self,
        attempt: u64,
        count: usize,
        d: usize,
        omega: u64,
    ) -> hachi_co_core::Result<Vec<Vec<i64>>> {
        let mut c = Challenger::<K>::r2l_challenges(&mut self.inner, attempt, count, d, omega)?;
        c[0].clone_from(&self.c0);
        Ok(c)
    }
    fn sumcheck_coins(
        &mut self,
        field: &Fqk<K>,
        ell_pp: usize,
        log_n: usize,
        log_nu: usize,
    ) -> hachi_co_core::Result<SumcheckCoins<K>> {
        self.inner.sumcheck_coins(field, ell_pp, log_n, log_nu)
    }
    fn round_coin(&mut self, field: &Fqk<K>, round: usize) -> hachi_co_core::Result<Ext<K>> {
        self.inner.round_coin(field, round)
    }
}

#[test]
fn misbehaving_components_give_errors() {
    let (fx, ctx, pv, states) = load("toy-n1");
    let q = ctx.q();
    let u = u_com(&fx);
    let us = u_share(&fx, 1);
    let replay = || {
        (
            ReplayRandomness::new(&fx, 1).unwrap(),
            ReplayChallenger::new(&fx).unwrap(),
        )
    };
    let with_fault = |fault| {
        let (mut rng, mut ch) = replay();
        let mut comm = Faulty {
            inner: Loopback::new(q).unwrap(),
            fault,
        };
        run(&ctx, &mut comm, &mut rng, &mut ch, &states[0], &pv, &us, &u)
    };
    assert!(matches!(
        with_fault(Fault::FahShort),
        Err(Error::Core(hachi_co_core::Error::Shape { .. }))
    ));
    assert!(matches!(
        with_fault(Fault::FahNonCanonical),
        Err(Error::NonCanonical { .. })
    ));
    // Attempt 0 of toy-n1 rejects; a network that hides the abort is caught.
    assert!(matches!(
        with_fault(Fault::AbortLies),
        Err(Error::Protocol(_))
    ));
    assert!(matches!(
        with_fault(Fault::GatherNone),
        Err(Error::Protocol(_))
    ));
    assert!(matches!(
        with_fault(Fault::GatherShort),
        Err(Error::Shape { .. })
    ));
    assert!(matches!(
        with_fault(Fault::GatherHuge),
        Err(Error::NormCheck { .. })
    ));
    assert!(matches!(
        with_fault(Fault::Prover1Lies),
        Err(Error::Protocol(_))
    ));

    // A challenger whose challenge leaves C is refused before the challenge is used.
    let (mut rng, ch) = replay();
    let mut c0 = vec![0i64; ctx.layout.d];
    c0[0] = ctx.omega() as i64 + 1;
    let mut bad = BadChallenger { inner: ch, c0 };
    let mut comm = Loopback::new(q).unwrap();
    assert!(matches!(
        run(
            &ctx, &mut comm, &mut rng, &mut bad, &states[0], &pv, &us, &u
        ),
        Err(Error::ChallengeNotInC { index: 0, .. })
    ));

    // The communication layer must match the context and the state.
    let (fx2, ctx2, pv2, states2) = load("toy-n2");
    let (mut rng, mut ch) = (
        ReplayRandomness::new(&fx2, 1).unwrap(),
        ReplayChallenger::new(&fx2).unwrap(),
    );
    let u2 = u_com(&fx2);
    let mut lb = Loopback::new(ctx2.q()).unwrap();
    assert!(matches!(
        run(
            &ctx2,
            &mut lb,
            &mut rng,
            &mut ch,
            &states2[0],
            &pv2,
            &u_share(&fx2, 1),
            &u2
        ),
        Err(Error::PartyMismatch {
            expected: 2,
            actual: 1,
            ..
        })
    ));
    // Party 1 of a two-party network passes party 2's state; party 2 does nothing.
    let mismatched = run_parties(2, ctx2.q(), |p, comm| {
        (p == 1).then(|| {
            let mut rng = ReplayRandomness::new(&fx2, 1).unwrap();
            let mut ch = ReplayChallenger::new(&fx2).unwrap();
            let us = u_share(&fx2, 2);
            run(&ctx2, comm, &mut rng, &mut ch, &states2[1], &pv2, &us, &u2)
        })
    });
    assert!(matches!(
        mismatched[0],
        Some(Err(Error::PartyMismatch {
            expected: 2,
            actual: 1,
            ..
        }))
    ));
    let (mut rng, mut ch) = replay();
    let mut lb = Loopback::new(q + 2).unwrap();
    assert!(matches!(
        run(&ctx, &mut lb, &mut rng, &mut ch, &states[0], &pv, &us, &u),
        Err(Error::Parameter(_))
    ));
    // Malformed inputs.
    let (mut rng, mut ch) = replay();
    let mut lb = Loopback::new(q).unwrap();
    let mut bad_state = states[0].clone();
    bad_state.blocks[0].s.pop();
    assert!(matches!(
        run(&ctx, &mut lb, &mut rng, &mut ch, &bad_state, &pv, &us, &u),
        Err(Error::Core(hachi_co_core::Error::Shape { .. }))
    ));
    assert!(matches!(
        run(
            &ctx,
            &mut lb,
            &mut rng,
            &mut ch,
            &states[0],
            &pv,
            &Poly(vec![0; 3]),
            &u
        ),
        Err(Error::Shape { .. })
    ));
    assert!(matches!(
        run(
            &ctx,
            &mut lb,
            &mut rng,
            &mut ch,
            &states[0],
            &pv,
            &us,
            &u[..0]
        ),
        Err(Error::Shape { .. })
    ));
    let mut non_canonical = states[0].clone();
    non_canonical.blocks[1].s[2].0[0] = q;
    assert!(matches!(
        run(
            &ctx,
            &mut lb,
            &mut rng,
            &mut ch,
            &non_canonical,
            &pv,
            &us,
            &u
        ),
        Err(Error::NonCanonical { .. })
    ));
}

/// A randomness source that returns malformed values is caught (lengths and ranges).
#[test]
fn malformed_randomness_is_refused() {
    struct Bad(u8);
    impl PartyRandomness<K> for Bad {
        fn reshare(
            &mut self,
            _: &Fqk<K>,
            _: &[Ext<K>],
            _: usize,
        ) -> hachi_co_core::Result<Vec<Vec<Ext<K>>>> {
            unreachable!()
        }
        fn mask(&mut self, _: u64, gamma: u64, len: usize) -> hachi_co_core::Result<Vec<i64>> {
            Ok(match self.0 {
                0 => vec![0; len - 1],
                1 => {
                    let mut m = vec![0; len];
                    m[len - 1] = gamma as i64 + 1;
                    m
                }
                _ => vec![0; len],
            })
        }
        fn rho(&mut self, _: u64, b: u64, len: usize) -> hachi_co_core::Result<Vec<i64>> {
            Ok(match self.0 {
                2 => vec![0; len + 1],
                3 => {
                    // ceil(b/2) is one past the top of S_b.
                    let mut r = vec![0; len];
                    r[0] = b.div_ceil(2) as i64;
                    r
                }
                4 => {
                    let mut r = vec![0; len];
                    r[1] = -((b / 2) as i64) - 1;
                    r
                }
                _ => vec![0; len],
            })
        }
        fn blinder(
            &mut self,
            _: &Fqk<K>,
            _: hachi_co_core::BlinderKind,
            _: usize,
            _: usize,
        ) -> hachi_co_core::Result<hachi_co_core::Blinder<K>> {
            unreachable!()
        }
    }
    let (fx, ctx, pv, states) = load("toy-n1");
    let u = u_com(&fx);
    let expected: [fn(&Error) -> bool; 5] = [
        |e| matches!(e, Error::Shape { .. }),
        |e| matches!(e, Error::Sampling(hachi_co_sampling::Error::MaskOutOfRange)),
        |e| matches!(e, Error::Shape { .. }),
        |e| matches!(e, Error::OutOfRange { .. }),
        |e| matches!(e, Error::OutOfRange { .. }),
    ];
    for (mode, check) in expected.iter().enumerate() {
        let mut rng = Bad(mode as u8);
        let mut ch = ReplayChallenger::new(&fx).unwrap();
        let mut lb = Loopback::new(ctx.q()).unwrap();
        let err = run(
            &ctx,
            &mut lb,
            &mut rng,
            &mut ch,
            &states[0],
            &pv,
            &u_share(&fx, 1),
            &u,
        )
        .unwrap_err();
        assert!(check(&err), "mode {mode}: {err:?}");
    }
    // An all-zero mask and rho are in the support (a terrible draw, but a valid one): z_1 is the
    // fold itself, which Rej accepts, and the relation still holds.
    let mut rng = Bad(5);
    let mut ch = ReplayChallenger::new(&fx).unwrap();
    let mut lb = Loopback::new(ctx.q()).unwrap();
    let out = run(
        &ctx,
        &mut lb,
        &mut rng,
        &mut ch,
        &states[0],
        &pv,
        &u_share(&fx, 1),
        &u,
    )
    .unwrap();
    assert_eq!(out.accepted_attempt, 0);
    let rel = linear_relation(&ctx, &pv, &out.v, &out.c, &u).unwrap();
    assert!(relation_holds(&ctx, &rel, &out.share.z_prime).unwrap());
}
