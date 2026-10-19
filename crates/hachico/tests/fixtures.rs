//! The composed PCS against the Sage oracle's fixtures:
//! - toy-n1 replayed end to end (`ReplayRandomness`, `ReplayChallenger`, the fixture's input,
//!   `Loopback`) through Commit, Eval and the composed verifier: every value of the proof equals
//!   the fixture, Open = 1 after Commit, and the final claims are the provers' polynomials;
//! - every fixture with n > 1 replayed the same way on the in-memory network, with both
//!   zero-sharing modes of its F_AH, at every party;
//! - the composed verifier accepts the proofs assembled from the three fixtures' recorded values
//!   with replayed coins, and returns the recorded final openings as claims.

mod common;

use common::net::run_with;
use common::{
    K, assert_claims_match, fixture_claims, fixture_proof, fixture_statement, load, recorded_state,
};
use hachi_co::{EvalOutput, EvalProof, commit, encoded_len, eval_with, verify_with};
use hachi_co_commit::open;
use hachi_co_core::{Fixture, ReplayChallenger, ReplayRandomness};
use hachi_co_mpc::{Loopback, MemoryConfig, ZeroSharing};

/// Every public value of the proof against the fixture, component by component (for readable
/// failures), then as a whole.
fn compare_proof(fx: &Fixture, proof: &EvalProof<K>, what: &str) {
    let want = fixture_proof(fx);
    assert_eq!(proof.e_u, want.e_u, "{what}: E(u)");
    assert_eq!(proof.attempt, want.attempt, "{what}: a*");
    assert_eq!(proof.v, want.v, "{what}: v");
    let (got, want_t) = (&proof.sumcheck, &want.sumcheck);
    assert_eq!(got.g0_sums, want_t.g0_sums, "{what}: G0");
    assert_eq!(got.g1_sums, want_t.g1_sums, "{what}: G1");
    assert_eq!(got.rounds.len(), want_t.rounds.len(), "{what}: rounds");
    for (r, (a, b)) in got.rounds.iter().zip(&want_t.rounds).enumerate() {
        assert_eq!(a.h0, b.h0, "{what}: H0 of round {}", r + 1);
        assert_eq!(a.h1, b.h1, "{what}: H1 of round {}", r + 1);
    }
    assert_eq!(got.finals, want_t.finals, "{what}: final values");
    assert_eq!(proof, &want, "{what}");
}

/// The attempts of the ring-to-linear step as the fixture records them (v, c, this party's
/// accept bit, the abort), and the accepted attempt's c.
fn compare_attempts(fx: &Fixture, out: &EvalOutput<K>, p: usize, what: &str) {
    let recorded = fx
        .get("/ring_to_linear/attempts")
        .unwrap()
        .as_array()
        .unwrap()
        .len();
    assert_eq!(out.r2l.attempts.len(), recorded, "{what}");
    for (a, rec) in out.r2l.attempts.iter().enumerate() {
        let at = |k: &str| format!("/ring_to_linear/attempts/{a}/{k}");
        assert_eq!(
            rec.v,
            fx.polys(&at("v")).unwrap(),
            "{what}: v of attempt {a}"
        );
        assert_eq!(
            rec.c,
            fx.int_rows(&at("c")).unwrap(),
            "{what}: c of attempt {a}"
        );
        let accept = fx.bool(&at(&format!("provers/{}/accept", p - 1))).unwrap();
        assert_eq!(rec.accepted, accept, "{what}: accept of attempt {a}");
        assert_eq!(
            rec.aborted,
            fx.bool(&at("abort")).unwrap(),
            "{what}: abort {a}"
        );
    }
    let a = out.proof.attempt;
    assert_eq!(
        out.r2l.c,
        fx.int_rows(&format!("/ring_to_linear/attempts/{a}/c"))
            .unwrap()
    );
    assert_eq!(out.conv.e_u, fx.poly("/conv/E_u").unwrap(), "{what}");
    assert_eq!(
        out.conv.u_share,
        fx.poly(&format!("/conv/u_shares/{}", p - 1)).unwrap(),
        "{what}: <u>_p"
    );
}

#[test]
fn toy_n1_replay_reproduces_the_fixture_end_to_end() {
    let (fx, ctx) = load("toy-n1");
    let q = ctx.q();
    let mut comm = Loopback::new(q).unwrap();
    let mut rng = ReplayRandomness::new(&fx, 1).unwrap();
    // Commit with the fixture's input.
    let f_init = fx.exts::<K>("/input/f_init_shares/0").unwrap();
    let (u, state) = commit(&ctx, &mut comm, &mut rng, &f_init).unwrap();
    assert_eq!(u, fx.polys("/commit/u").unwrap(), "u");
    assert_eq!(state, recorded_state(&fx, &ctx, 1), "st_1");
    let f = fx.exts::<K>("/input/f").unwrap();
    let report = open(&ctx, core::slice::from_ref(&state), &u, &f, None).unwrap();
    assert!(
        report.bit,
        "Open(pp, st, u, f, c*): {:?}",
        report.checks.failing()
    );
    // Eval with the replayed coins.
    let (u_fx, x, y) = fixture_statement(&fx);
    assert_eq!(u, u_fx);
    let mut ch = ReplayChallenger::new(&fx).unwrap();
    let out = eval_with(&ctx, &mut comm, &mut rng, &mut ch, &state, &u, &x, &y).unwrap();
    compare_proof(&fx, &out.proof, "toy-n1");
    compare_attempts(&fx, &out, 1, "toy-n1");
    assert_eq!(
        out.r2l.attempts.iter().filter(|a| a.aborted).count(),
        4,
        "toy-n1 has four aborted attempts (ref/README.md, choice 19)"
    );
    assert_eq!(
        out.sumcheck.point,
        fx.exts::<K>("/sumcheck/coins/a").unwrap()
    );
    // The composed verifier, with the replayed coins.
    let mut vch = ReplayChallenger::new(&fx).unwrap();
    let claims = verify_with(&ctx, &mut vch, &u, &x, &y, &out.proof).unwrap();
    assert_eq!(claims, fixture_claims(&fx));
    assert_claims_match(ctx.field(), &claims, &[&out.sumcheck], "toy-n1");
    // The encoding of the replayed proof.
    let bytes = out.proof.to_bytes();
    assert_eq!(bytes.len(), encoded_len(&ctx.layout).unwrap());
    assert_eq!(EvalProof::from_bytes(&ctx, &bytes).unwrap(), out.proof);
}

/// Every fixture with $n > 1$ replayed through Commit and Eval on the in-memory network, with
/// each zero-sharing mode: every party reproduces u, its state and the fixture's proof; Open = 1
/// on all states; the verifier accepts and its claims are the provers' polynomials.
#[test]
fn threaded_replay_reproduces_every_multi_party_fixture() {
    for name in Fixture::TOY {
        for mode in [ZeroSharing::PairwiseSeeds, ZeroSharing::Fresh] {
            let (fx, ctx) = load(name);
            let n = fx.n();
            if n == 1 {
                continue;
            }
            let (u_fx, x, y) = fixture_statement(&fx);
            let config = MemoryConfig::new(n, ctx.q(), [n as u8; 32]).zero_sharing(mode);
            let outputs: Vec<_> = run_with(&config, |p, comm| {
                let mut rng = ReplayRandomness::new(&fx, p).unwrap();
                let f_init = fx
                    .exts::<K>(&format!("/input/f_init_shares/{}", p - 1))
                    .unwrap();
                let (u, state) = commit(&ctx, comm, &mut rng, &f_init).unwrap();
                let mut ch = ReplayChallenger::new(&fx).unwrap();
                let out = eval_with(&ctx, comm, &mut rng, &mut ch, &state, &u, &x, &y).unwrap();
                (u, state, out)
            })
            .into_iter()
            .map(|run| run.expect("every party connects").output)
            .collect();
            for (p0, (u, state, out)) in outputs.iter().enumerate() {
                let what = format!("{name}, {mode:?}, party {}", p0 + 1);
                assert_eq!(u, &u_fx, "{what}: u");
                assert_eq!(state, &recorded_state(&fx, &ctx, p0 + 1), "{what}: st");
                compare_proof(&fx, &out.proof, &what);
                compare_attempts(&fx, out, p0 + 1, &what);
            }
            let states: Vec<_> = outputs.iter().map(|(_, st, _)| st.clone()).collect();
            let f = fx.exts::<K>("/input/f").unwrap();
            assert!(open(&ctx, &states, &u_fx, &f, None).unwrap().bit, "{name}");
            let mut vch = ReplayChallenger::new(&fx).unwrap();
            let claims = verify_with(&ctx, &mut vch, &u_fx, &x, &y, &outputs[0].2.proof).unwrap();
            assert_eq!(claims, fixture_claims(&fx), "{name}");
            let privates: Vec<_> = outputs.iter().map(|(_, _, out)| &out.sumcheck).collect();
            assert_claims_match(ctx.field(), &claims, &privates, name);
        }
    }
}

/// The proofs assembled from the recorded values of every fixture are accepted by the composed
/// verifier with replayed coins, and the claims are the recorded final openings, at the
/// recorded point.
#[test]
fn composed_verifier_accepts_the_three_fixture_transcripts() {
    for name in Fixture::TOY {
        let (fx, ctx) = load(name);
        let proof = fixture_proof(&fx);
        proof.validate(&ctx).unwrap();
        let (u, x, y) = fixture_statement(&fx);
        let mut ch = ReplayChallenger::new(&fx).unwrap();
        let claims = verify_with(&ctx, &mut ch, &u, &x, &y, &proof).unwrap();
        assert_eq!(claims, fixture_claims(&fx), "{name}");
        assert_eq!(claims.claims.len(), 3 * fx.n(), "{name}");
        assert!(fx.bool("/verifier/accept").unwrap());
        // The same proof through its encoding.
        let decoded = EvalProof::from_bytes(&ctx, &proof.to_bytes()).unwrap();
        let mut ch = ReplayChallenger::new(&fx).unwrap();
        assert_eq!(
            verify_with(&ctx, &mut ch, &u, &x, &y, &decoded).unwrap(),
            claims
        );
    }
}
