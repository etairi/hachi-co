//! The driver `run`: it reproduces the fixtures with replayed randomness and coins (toy-n1 on the
//! loopback; the other fixtures on the in-memory network), honest Fiat-Shamir runs satisfy
//! M z' = y and the verifier re-derives their challenges and selected iteration, a run in which
//! no iteration is accepted by every prover aborts, and misbehaving components (network,
//! challenger, randomness) give errors, not panics.

mod common;

use common::{K, accept_sets, load, run_parties, u_com, u_share};
use hachi_co_core::{
    Challenger, Context, Ext, Fixture, Fqk, FsChallenger, PartyRandomness, PartyState, Poly,
    PrgRandomness, PublicVectors, ReplayChallenger, ReplayRandomness, SumcheckCoins,
};
use hachi_co_mpc::{Comm, Loopback, PartyId};
use hachi_co_params::{Choices, RejIterations, derive, presets::toy_choices};
use hachi_co_r2l::{
    COMM_ACCEPT, COMM_FAH_D, COMM_FAH_Z, Error, R2lOutput, accepts, linear_relation, r3_conditions,
    relation_holds, run, select_iteration, sum_shares, verifier_challenges, verifier_iteration,
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

/// Checks a run's output against the fixture: v, c, every prover's set of accepted iterations
/// (party `p`'s own flags included) and j*.
fn compare_with_fixture(fx: &Fixture, ctx: &Context<K>, out: &R2lOutput, p: PartyId) {
    let name = fx.name();
    assert_eq!(
        out.v,
        fx.polys("/ring_to_linear/v").unwrap(),
        "{name} p={p}: v"
    );
    assert_eq!(
        out.c,
        fx.int_rows("/ring_to_linear/c").unwrap(),
        "{name} p={p}: c"
    );
    assert_eq!(out.accept_sets, accept_sets(fx, ctx), "{name} p={p}: R");
    for j in 0..ctx.layout.rej_iterations {
        let recorded = fx
            .bool(&format!("/ring_to_linear/provers/{}/accept/{j}", p - 1))
            .unwrap();
        assert_eq!(
            accepts(&out.accept_sets[p - 1], j),
            recorded,
            "{name} p={p} j={j}"
        );
    }
    assert_eq!(out.j_star, common::j_star(fx), "{name} p={p}: j*");
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
fn loopback_replay_reproduces_toy_n1_with_its_rejected_first_iteration() {
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
    compare_with_fixture(&fx, &ctx, &out, 1);
    assert!(
        out.j_star > 0 && !accepts(&out.accept_sets[0], 0),
        "toy-n1 rejects its first iteration (ref/README.md, choice 19)"
    );
    // n = 1: the share is the whole witness, <u>_1 = u included, and the whole y.
    assert_eq!(
        out.share.z_prime,
        fx.polys("/ring_to_linear/z_prime").unwrap()
    );
    assert_eq!(out.share.y, fx.polys("/ring_to_linear/y").unwrap());
    let rel = linear_relation(&ctx, &pv, &out.v, &out.c, out.j_star, &u_com(&fx)).unwrap();
    assert!(relation_holds(&ctx, &rel, &out.share.z_prime).unwrap());
}

/// With $L = 1$ iteration, a run whose only iteration the prover rejects aborts with
/// `NoCommonIteration` (`fig:ring-to-linear`, step 6), and a run that accepts it completes. The
/// toy-n1 states and public vectors fit a context with $L = 1$ (they do not depend on $L$); its
/// matrices come from another setup seed, so the runs say nothing about the relation.
#[test]
fn a_run_without_a_common_iteration_aborts() {
    let (fx, _, pv, states) = load("toy-n1");
    let (params, _) = derive(&Choices {
        rej_iterations: RejIterations::Fixed(1),
        ..toy_choices(1)
    })
    .unwrap();
    let ctx = Context::<K>::setup(params, &[3; 32]).unwrap();
    assert_eq!(ctx.layout.rej_iterations, 1);
    let state = fx.party_state(&ctx, 1).unwrap();
    let (mut aborted, mut completed) = (0, 0);
    for i in 0..32u64 {
        let mut comm = Loopback::new(ctx.q()).unwrap();
        let mut rng = PrgRandomness::new(seed("hachi-co-r2l/test/abort", &[i]));
        let mut ch = FsChallenger::new(DOMAIN);
        match run(
            &ctx,
            &mut comm,
            &mut rng,
            &mut ch,
            &state,
            &pv,
            &u_share(&fx, 1),
            &u_com(&fx),
        ) {
            Err(e) => {
                assert_eq!(e, Error::NoCommonIteration { iterations: 1 }, "run {i}");
                aborted += 1;
            }
            Ok(out) => {
                assert_eq!((out.j_star, out.accept_sets.clone()), (0, vec![vec![1]]));
                completed += 1;
            }
        }
    }
    // About 39% of the iterations are rejected at the toys (M^{-1} close to e^{-1/2}).
    assert!(
        aborted > 0 && completed > 0,
        "{aborted} aborted, {completed} completed"
    );
    let _ = states;
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
            compare_with_fixture(&fx, &ctx, out, p);
            // y sits at prover 1 only (5_Building_Blocks.tex:147).
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
        let rel = linear_relation(
            &ctx,
            &pv,
            &outs[0].v,
            &outs[0].c,
            outs[0].j_star,
            &u_com(&fx),
        )
        .unwrap();
        assert!(relation_holds(&ctx, &rel, &sum.z_prime).unwrap(), "{name}");
        // The relation R_3 of the draft since c071784: M z' = y, short entries, slices of S.
        let r3 = r3_conditions(&ctx, &rel, &shares).unwrap();
        assert!(r3.hold(), "{name}: {r3:?}");
    }
}

/// Honest runs with Fiat-Shamir at toy-n1: fresh seeds, the relation holds for the verifier's M
/// and y, and the verifier re-derives c from v and j* from the sets and reaches the provers'
/// transcript state.
#[test]
fn honest_fiat_shamir_runs_at_toy_n1_satisfy_the_relation() {
    let (fx, ctx, pv, states) = load("toy-n1");
    let u = u_com(&fx);
    let runs = 24;
    let mut rejected = 0;
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
        let rel = linear_relation(&ctx, &pv, &out.v, &out.c, out.j_star, &u).unwrap();
        assert!(
            relation_holds(&ctx, &rel, &out.share.z_prime).unwrap(),
            "run {i}"
        );
        // The verifier, from the proof values (v, the sets) only.
        let mut vch = FsChallenger::new(DOMAIN);
        statement(&mut vch);
        let c = verifier_challenges(&ctx, &mut vch, &out.v).unwrap();
        assert_eq!(c, out.c, "run {i}");
        let j = verifier_iteration(&ctx, &mut vch, &out.accept_sets).unwrap();
        assert_eq!(j, out.j_star, "run {i}");
        assert_eq!(vch.transcript().digest(), ch.transcript().digest());
        assert_eq!(
            select_iteration(&ctx, &out.accept_sets),
            Ok(Some(out.j_star))
        );
        // Every iteration before j* was rejected (n = 1: the only set).
        assert!((0..out.j_star).all(|j| !accepts(&out.accept_sets[0], j)));
        assert!(accepts(&out.accept_sets[0], out.j_star));
        rejected += out.j_star;
        // Without the statement (or with another one) the verifier gets other challenges.
        let mut other = FsChallenger::new(DOMAIN);
        Challenger::<K>::absorb_u64(&mut other, "test/statement", i + 1);
        let c_other = verifier_challenges(&ctx, &mut other, &out.v).unwrap();
        assert_ne!(c_other, out.c);
        // A tampered v gives the verifier other challenges.
        let mut v2 = out.v.clone();
        v2[0].0[0] = (v2[0].0[0] + 1) % ctx.q();
        let c2 = verifier_challenges(&ctx, &mut FsChallenger::new(DOMAIN), &v2).unwrap();
        assert_ne!(c2, out.c);
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
    // Diagnostic (not an assertion): about 0.39 rejected iterations per iteration at toy-n1.
    println!("toy-n1: {runs} honest Fiat-Shamir runs, {rejected} iterations rejected before j*");
}

/// Honest Fiat-Shamir runs of n = 2, 3, 4 and 8 parties on the in-memory network: every party has its
/// own randomness and its own challenger replica, and all reach the same v, c, sets and j*.
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
                assert_eq!(out.j_star, outs[0].j_star);
                assert_eq!(out.accept_sets, outs[0].accept_sets);
                assert_eq!(out.v, outs[0].v);
                assert_eq!(out.c, outs[0].c);
            }
            let shares: Vec<_> = outs.iter().map(|o| o.share.clone()).collect();
            let sum = sum_shares(&ctx, &shares).unwrap();
            let rel =
                linear_relation(&ctx, &pv, &outs[0].v, &outs[0].c, outs[0].j_star, &u).unwrap();
            assert!(
                relation_holds(&ctx, &rel, &sum.z_prime).unwrap(),
                "{name} run {i}"
            );
            let r3 = r3_conditions(&ctx, &rel, &shares).unwrap();
            assert!(r3.hold(), "{name} run {i}: {r3:?}");
            let mut vch = FsChallenger::new(DOMAIN);
            let c = verifier_challenges(&ctx, &mut vch, &outs[0].v).unwrap();
            assert_eq!(c, outs[0].c);
            assert_eq!(
                verifier_iteration(&ctx, &mut vch, &outs[0].accept_sets),
                Ok(outs[0].j_star)
            );
        }
    }
}

#[test]
fn verifier_side_rejects_malformed_proof_values() {
    let (fx, ctx, pv, _) = load("toy-n2");
    let l = &ctx.layout;
    let q = ctx.q();
    let v = fx.polys("/ring_to_linear/v").unwrap();
    let c = fx.int_rows("/ring_to_linear/c").unwrap();
    let j = common::j_star(&fx);
    let u = u_com(&fx);
    linear_relation(&ctx, &pv, &v, &c, j, &u).unwrap();
    let omega = ctx.omega();
    // c outside C: l1 norm omega + 1, an extreme coefficient, one challenge too few or too many,
    // a challenge with a coefficient too few.
    let mut bad = c.clone();
    bad[3] = vec![0; l.d];
    bad[3][0] = omega as i64;
    bad[3][5] = -1;
    assert_eq!(
        linear_relation(&ctx, &pv, &v, &bad, j, &u),
        Err(Error::ChallengeNotInC {
            index: 3,
            l1_norm: u128::from(omega) + 1,
            omega
        })
    );
    bad[3] = vec![0; l.d];
    bad[3][7] = i64::MIN;
    assert!(matches!(
        linear_relation(&ctx, &pv, &v, &bad, j, &u),
        Err(Error::ChallengeNotInC { index: 3, .. })
    ));
    // The boundary: l1 norm exactly omega is in C.
    bad[3] = vec![0; l.d];
    bad[3][2] = -(omega as i64);
    linear_relation(&ctx, &pv, &v, &bad, j, &u).unwrap();
    let shape = |r: hachi_co_r2l::Result<_>| matches!(r, Err(Error::Shape { .. }));
    assert!(shape(linear_relation(&ctx, &pv, &v, &c[1..], j, &u)));
    let mut more = c.clone();
    more.push(vec![0; l.d]);
    assert!(shape(linear_relation(&ctx, &pv, &v, &more, j, &u)));
    let mut short_c = c.clone();
    short_c[0].pop();
    assert!(shape(linear_relation(&ctx, &pv, &v, &short_c, j, &u)));
    // v and u: wrong counts, wrong lengths, non-canonical residues.
    assert!(shape(linear_relation(&ctx, &pv, &[], &c, j, &u)));
    assert!(shape(linear_relation(&ctx, &pv, &v, &c, j, &[])));
    let mut v_bad = v.clone();
    v_bad[0].0.pop();
    assert!(shape(linear_relation(&ctx, &pv, &v_bad, &c, j, &u)));
    let mut v_bad = v.clone();
    v_bad[0].0[3] = q;
    assert_eq!(
        linear_relation(&ctx, &pv, &v_bad, &c, j, &u),
        Err(Error::NonCanonical {
            what: "v",
            value: q,
            q
        })
    );
    let mut u_bad = u.clone();
    u_bad[0].0[0] = u64::MAX;
    assert!(matches!(
        linear_relation(&ctx, &pv, &v, &c, j, &u_bad),
        Err(Error::NonCanonical { .. })
    ));
    let mut ch = FsChallenger::new(DOMAIN);
    assert!(matches!(
        verifier_challenges(&ctx, &mut ch, &v_bad),
        Err(Error::NonCanonical { .. })
    ));
    assert!(matches!(
        verifier_challenges(&ctx, &mut ch, &v[..0]),
        Err(Error::Shape { .. })
    ));
    // Public vectors of the wrong shape.
    let mut pv_bad = pv.clone();
    pv_bad.a_ring.pop();
    assert!(matches!(
        linear_relation(&ctx, &pv_bad, &v, &c, j, &u),
        Err(Error::Core(hachi_co_core::Error::Shape { .. }))
    ));
}

// ------------------------------------------------------------ misbehaving components ----

/// How [`Faulty`] tampers with the loopback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    FahShort,
    FahNonCanonical,
    SetBeyondL,
    SetChanged,
    SetsMissing,
    SumShort,
    SumNonCanonical,
    SumHuge,
}

/// A party's end of the network (the loopback or the in-memory network) with one fault.
struct Faulty<'a, C: Comm> {
    inner: &'a mut C,
    fault: Fault,
}

impl<C: Comm> Comm for Faulty<'_, C> {
    fn n(&self) -> usize {
        self.inner.n()
    }
    fn id(&self) -> PartyId {
        self.inner.id()
    }
    fn modulus(&self) -> u64 {
        self.inner.modulus()
    }
    fn fah(&mut self, label: &'static str, local: &[u64]) -> hachi_co_mpc::Result<Vec<u64>> {
        let mut out = self.inner.fah(label, local)?;
        let q = self.inner.modulus();
        match (self.fault, label) {
            (Fault::FahShort, COMM_FAH_D) | (Fault::SumShort, COMM_FAH_Z) => {
                out.pop();
            }
            (Fault::FahNonCanonical, COMM_FAH_D) | (Fault::SumNonCanonical, COMM_FAH_Z) => {
                out[0] = q;
            }
            (Fault::SumHuge, COMM_FAH_Z) => out[0] = (q - 1) / 2,
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
        self.inner.gather(label, to, local)
    }
    fn any(&mut self, label: &'static str, local: bool) -> hachi_co_mpc::Result<bool> {
        self.inner.any(label, local)
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
        let mut out = self.inner.all_gather(label, local)?;
        if label == COMM_ACCEPT {
            match self.fault {
                Fault::SetBeyondL => out[0][0] |= 1 << 63,
                Fault::SetChanged => out[0][0] ^= 1,
                Fault::SetsMissing => out.clear(),
                _ => {}
            }
        }
        Ok(out)
    }
}

/// Every prover checks $\Vert \mathbf z \Vert\_\infty \le \beta$ itself (`fig:ring-to-linear-2`,
/// step 9, since the draft's commit `0337dfb`): when the output of $\mathcal F\_{\mathsf{AH}}(1^n)$
/// exceeds $\beta$, every prover stops with `NormCheck`, and when it is the honest sum, every
/// prover completes (the same replay without the fault).
#[test]
fn every_prover_runs_the_norm_check() {
    for name in ["toy-n2", "toy-n3"] {
        let (fx, ctx, pv, states) = load(name);
        let n = fx.n();
        let u = u_com(&fx);
        for fault in [Some(Fault::SumHuge), None] {
            let outs = run_parties(n, ctx.q(), |p, comm| {
                let mut rng = ReplayRandomness::new(&fx, p).unwrap();
                let mut ch = ReplayChallenger::new(&fx).unwrap();
                let us = u_share(&fx, p);
                match fault {
                    Some(fault) => {
                        let mut faulty = Faulty { inner: comm, fault };
                        run(
                            &ctx,
                            &mut faulty,
                            &mut rng,
                            &mut ch,
                            &states[p - 1],
                            &pv,
                            &us,
                            &u,
                        )
                    }
                    None => run(&ctx, comm, &mut rng, &mut ch, &states[p - 1], &pv, &us, &u),
                }
            });
            for (p, out) in (1..=n).zip(&outs) {
                match fault {
                    Some(_) => assert!(
                        matches!(out, Err(Error::NormCheck { .. })),
                        "{name} p = {p}: {out:?}"
                    ),
                    None => assert!(out.is_ok(), "{name} p = {p}: {out:?}"),
                }
            }
        }
    }
}

/// The replay challenger, with challenge `c_0` replaced.
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
    fn absorb_words(&mut self, label: &'static str, words: &[Vec<u64>]) {
        Challenger::<K>::absorb_words(&mut self.inner, label, words);
    }
    fn absorb_u64(&mut self, label: &'static str, value: u64) {
        Challenger::<K>::absorb_u64(&mut self.inner, label, value);
    }
    fn context(&mut self, label: &str) -> [u8; 32] {
        Challenger::<K>::context(&mut self.inner, label)
    }
    fn r2l_challenges(
        &mut self,
        count: usize,
        d: usize,
        omega: u64,
    ) -> hachi_co_core::Result<Vec<Vec<i64>>> {
        let mut c = Challenger::<K>::r2l_challenges(&mut self.inner, count, d, omega)?;
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
        let mut lb = Loopback::new(q).unwrap();
        let mut comm = Faulty {
            inner: &mut lb,
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
    // A broadcast set with an iteration beyond L (L = 16 < 64), a party's own set changed, or a
    // missing set are caught.
    assert_eq!(
        with_fault(Fault::SetBeyondL),
        Err(Error::AcceptSet {
            party: 1,
            what: "it names an iteration at or beyond L"
        })
    );
    assert!(matches!(
        with_fault(Fault::SetChanged),
        Err(Error::Protocol(_))
    ));
    assert!(matches!(
        with_fault(Fault::SetsMissing),
        Err(Error::Protocol(_))
    ));
    assert!(matches!(
        with_fault(Fault::SumShort),
        Err(Error::Shape { .. })
    ));
    assert!(matches!(
        with_fault(Fault::SumNonCanonical),
        Err(Error::NonCanonical { .. })
    ));
    assert!(matches!(
        with_fault(Fault::SumHuge),
        Err(Error::NormCheck { .. })
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
        fn base_pcs(&mut self, _: u64) -> hachi_co_core::Result<[u8; 32]> {
            unreachable!()
        }
        fn rho_u(&mut self, _: u64, _: usize) -> hachi_co_core::Result<Vec<i64>> {
            unreachable!()
        }
        fn rho_w(&mut self, _: &Fqk<K>) -> hachi_co_core::Result<Ext<K>> {
            unreachable!()
        }
        fn rho_w_residues(&mut self, _: u64, _: usize) -> hachi_co_core::Result<Vec<u64>> {
            unreachable!()
        }
        fn rho_v(&mut self, b: u64, len: usize) -> hachi_co_core::Result<Vec<i64>> {
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
    // All-zero masks and rho are in the support (a terrible draw, but a valid one): z_1 is the
    // fold itself, which Rej accepts in every iteration, and the relation still holds.
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
    assert_eq!(out.j_star, 0);
    assert_eq!(
        out.accept_sets,
        vec![vec![(1u64 << ctx.layout.rej_iterations) - 1]]
    );
    let rel = linear_relation(&ctx, &pv, &out.v, &out.c, out.j_star, &u).unwrap();
    assert!(relation_holds(&ctx, &rel, &out.share.z_prime).unwrap());
}
