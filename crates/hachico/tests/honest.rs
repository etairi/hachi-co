//! Honest runs with Fiat-Shamir (`PrgRandomness` with fresh seeds, `FsChallenger` on
//! `EVAL_DOMAIN`) and the reference base PCS `ClearPcs`: fresh public parameters from `setup`, a
//! random f and a random x, and y = f(x) computed independently (an explicit table of
//! monomials). Every run verifies, openings included, its checked claims are the provers'
//! polynomials (ideal PCS), the provers' and the verifier's transcripts end in the same state
//! after the same challenger calls, the encoding round-trips and has the length of the size
//! formula with `ClearPcs`'s sizes.
//!
//! - toy-n1 (four seeds) and dev-l12, dev-l16 on the loopback, in the default run;
//! - dev-l20 on the loopback, `#[ignore]`d (run with `-- --ignored --nocapture`);
//! - toy-n2, n3, n4 and n8 on the in-memory network.
//!
//! Seeds: setup `seed("setup/<preset>", [i])`, instance `ChaCha8Rng::seed_from_u64(s)`, party
//! `seed("party/<preset>", [i, p])` (common::seed is SHA-256 of the tag and the indices); the
//! values are in each test. Timings are printed (`--nocapture`).

mod common;

use common::net::run_parties;
use common::{
    Call, Recorder, additive_shares, assert_claims_match, clear_pcs, clear_words, fresh_context,
    random_instance, seed,
};
use hachi_co::{
    BasePcs, ClaimedPoly, EVAL_DOMAIN, EvalProof, LABEL_PP_A, LABEL_PP_BASE, LABELS_PARAMS, commit,
    encoded_len, eval, eval_with, verify, verify_claims_with, verify_with,
};
use hachi_co_commit::open;
use hachi_co_core::{FsChallenger, PrgRandomness};
use hachi_co_mpc::Loopback;
use hachi_co_params::sizes_with_base;
use hachi_co_sumcheck::opening_label;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

/// One honest run on the loopback: setup (index `i`), a random instance (seed `instance`),
/// Commit and Open, Eval with `eval_with` on a fresh `FsChallenger`, the verifier with
/// `verify_claims_with` on another, and every check of the module documentation; the timings
/// of Commit, Eval and the verifier are printed. With `repeat`, the proof is made again with
/// the public `eval` (the same proof for the same seed) and checked with the public `verify`
/// and `verify_with`; the heavy dev-l20 run skips this, so that its peak memory is that of one
/// Eval.
fn honest_loopback_run(preset: &str, i: u64, instance: u64, repeat: bool) {
    let ctx = fresh_context(preset, i);
    let pcs = clear_pcs(&ctx);
    let (f, x, y) = random_instance(&ctx, instance);
    let party_seed = seed(&format!("party/{preset}"), &[i, 1]);
    let mut comm = Loopback::new(ctx.q()).unwrap();
    let mut rng = PrgRandomness::new(party_seed);
    let start = Instant::now();
    let (u, state) = commit(&ctx, &mut comm, &mut rng, &f).unwrap();
    let commit_time = start.elapsed();
    let report = open(&ctx, core::slice::from_ref(&state), &u, &f, None).unwrap();
    assert!(report.bit, "{preset}: Open: {:?}", report.checks.failing());
    let mut ch = FsChallenger::new(EVAL_DOMAIN);
    let start = Instant::now();
    let out = eval_with(&ctx, &pcs, &mut comm, &mut rng, &mut ch, &state, &u, &x, &y).unwrap();
    let prove_time = start.elapsed();
    let mut vch = FsChallenger::new(EVAL_DOMAIN);
    let start = Instant::now();
    let claims = verify_claims_with(&ctx, &pcs, &mut vch, &u, &x, &y, &out.proof).unwrap();
    let verify_time = start.elapsed();
    assert_eq!(
        vch.transcript().digest(),
        ch.transcript().digest(),
        "{preset}: the provers' and the verifier's transcripts"
    );
    assert_claims_match(ctx.field(), &claims, &[&out.sumcheck], preset);
    assert_eq!(claims.point, out.sumcheck.point);
    if repeat {
        // The public API: eval on a fresh FsChallenger gives the same proof for the same seed
        // (the randomness streams of a PrgRandomness are served once, so a new one with the
        // same seed), and verify and verify_with accept it.
        let mut rng2 = PrgRandomness::new(party_seed);
        let mut comm = Loopback::new(ctx.q()).unwrap();
        let proof = eval(&ctx, &pcs, &mut comm, &mut rng2, &state, &u, &x, &y).unwrap();
        assert_eq!(proof, out.proof, "{preset}: eval = eval_with");
        verify(&ctx, &pcs, &u, &x, &y, &proof).unwrap();
        let mut vch = FsChallenger::new(EVAL_DOMAIN);
        verify_with(&ctx, &pcs, &mut vch, &u, &x, &y, &proof).unwrap();
    }
    // The encoding, with the sizes of ClearPcs's items.
    let bytes = out.proof.to_bytes();
    let base = clear_words(&ctx.layout);
    let formula = sizes_with_base(&ctx.params, &base).unwrap();
    assert_eq!(bytes.len() as u128, formula.total, "{preset}: size formula");
    assert_eq!(bytes.len(), encoded_len(&ctx.layout, &base).unwrap());
    assert_eq!(bytes.len(), out.proof.encoded_len());
    assert_eq!(EvalProof::from_bytes(&ctx, &bytes).unwrap(), out.proof);
    eprintln!(
        "{preset} (setup {i}, instance {instance}): q = {}, l'' = {}, a* = {}, \
         commit {commit_time:?}, prove {prove_time:?}, verify {verify_time:?}, proof {} bytes \
         ({} without the base PCS's items)",
        ctx.q(),
        ctx.layout.ell_pp,
        out.proof.attempt,
        bytes.len(),
        hachi_co_params::sizes(&ctx.params).unwrap().total
    );
}

#[test]
fn honest_runs_at_toy_n1_verify() {
    for i in 0..4u64 {
        honest_loopback_run("toy-n1", i, 1000 + i, true);
    }
}

#[test]
fn honest_run_at_dev_l12_verifies() {
    honest_loopback_run("dev-l12", 0, 1200, true);
}

#[test]
fn honest_run_at_dev_l16_verifies() {
    honest_loopback_run("dev-l16", 0, 1600, true);
}

/// Heavy (tens of seconds and more than 1 GB): see the WP11 report for the measured figures.
#[test]
#[ignore = "heavy: run with --ignored --nocapture"]
fn honest_run_at_dev_l20_verifies() {
    honest_loopback_run("dev-l20", 0, 2000, false);
}

/// Honest runs of n = 2, 3, 4 and 8 parties on the in-memory network: each party commits its additive
/// share, runs Eval with its own randomness and challenger, and obtains the same proof; Open = 1
/// on the states; the verifier accepts; every party's transcript and the verifier's end in the
/// same state; the claims are the parties' polynomials.
#[test]
fn honest_multi_party_runs_on_the_network() {
    for (preset, n) in [
        ("toy-n2", 2usize),
        ("toy-n3", 3),
        ("toy-n4", 4),
        ("toy-n8", 8),
    ] {
        for i in 0..2u64 {
            let ctx = fresh_context(preset, i);
            let pcs = clear_pcs(&ctx);
            let (f, x, y) = random_instance(&ctx, 3000 + 10 * n as u64 + i);
            let mut rng = ChaCha8Rng::seed_from_u64(4000 + 10 * n as u64 + i);
            let f_init = additive_shares(&ctx, &mut rng, &f, n);
            let outputs = run_parties(n, ctx.q(), |p, comm| {
                let mut prg = PrgRandomness::new(seed(&format!("party/{preset}"), &[i, p as u64]));
                let (u, state) = commit(&ctx, comm, &mut prg, &f_init[p - 1]).unwrap();
                let mut ch = FsChallenger::new(EVAL_DOMAIN);
                let out =
                    eval_with(&ctx, &pcs, comm, &mut prg, &mut ch, &state, &u, &x, &y).unwrap();
                (u, state, out, ch.transcript().digest())
            });
            let (u, _, first, digest) = &outputs[0];
            for (u_p, _, out, digest_p) in &outputs[1..] {
                assert_eq!(u_p, u);
                assert_eq!(
                    out.proof, first.proof,
                    "{preset}: every party has the same proof"
                );
                assert_eq!(digest_p, digest);
            }
            let states: Vec<_> = outputs.iter().map(|(_, st, _, _)| st.clone()).collect();
            assert!(open(&ctx, &states, u, &f, None).unwrap().bit, "{preset}");
            let mut vch = FsChallenger::new(EVAL_DOMAIN);
            let claims = verify_claims_with(&ctx, &pcs, &mut vch, u, &x, &y, &first.proof).unwrap();
            assert_eq!(&vch.transcript().digest(), digest, "{preset}");
            let privates: Vec<_> = outputs.iter().map(|(_, _, out, _)| &out.sumcheck).collect();
            assert_claims_match(ctx.field(), &claims, &privates, preset);
            let bytes = first.proof.to_bytes();
            let base = clear_words(&ctx.layout);
            assert_eq!(
                bytes.len() as u128,
                sizes_with_base(&ctx.params, &base).unwrap().total
            );
            assert_eq!(bytes.len(), encoded_len(&ctx.layout, &base).unwrap());
            assert_eq!(EvalProof::from_bytes(&ctx, &bytes).unwrap(), first.proof);
            eprintln!(
                "{preset} (setup {i}): a* = {}, proof {} bytes",
                first.proof.attempt,
                bytes.len()
            );
        }
    }
}

/// Whether `call` absorbs `count` polynomials under `label`.
fn is_polys(call: &Call, label: &str, count: usize) -> bool {
    match call {
        Call::Polys(lab, n, _) => *lab == label && *n == count,
        _ => false,
    }
}

/// Removes the calls of the aborted attempts (`Begin(a)`, the absorb of v and `R2l(a, ..)` for
/// every a below the accepted one) from a provers' log.
fn without_aborted_attempts(log: &[Call], accepted: u64) -> Vec<Call> {
    let mut out = Vec::with_capacity(log.len());
    let mut skipping = false;
    for call in log {
        match call {
            Call::Begin(a) => skipping = *a != accepted,
            Call::R2l(a, ..) if *a != accepted => {
                skipping = false;
                continue;
            }
            _ => {}
        }
        if !skipping {
            out.push(call.clone());
        }
    }
    out
}

/// The provers and the verifier make the same challenger calls in the same order, the verifier
/// replaying only the accepted attempt; the sequence is the one of the crate documentation,
/// with the commitments absorbed after G and before the coins and the three contexts of the
/// openings (in the order g0, g1, w) after the final values. The first toy-n1 instance (setup
/// index 100 + j, instance seed 5000 + j) with an aborted attempt is used, so that the attempts
/// are exercised; the absorb sequence is printed.
#[test]
fn provers_and_verifier_make_the_same_challenger_calls() {
    for j in 0..32u64 {
        let ctx = fresh_context("toy-n1", 100 + j);
        let pcs = clear_pcs(&ctx);
        let (f, x, y) = random_instance(&ctx, 5000 + j);
        let mut comm = Loopback::new(ctx.q()).unwrap();
        let mut rng = PrgRandomness::new(seed("party/toy-n1/calls", &[j]));
        let (u, state) = commit(&ctx, &mut comm, &mut rng, &f).unwrap();
        let mut ch = Recorder::new(FsChallenger::new(EVAL_DOMAIN));
        let out = eval_with(&ctx, &pcs, &mut comm, &mut rng, &mut ch, &state, &u, &x, &y).unwrap();
        if out.proof.attempt == 0 {
            continue;
        }
        let mut vch = Recorder::new(FsChallenger::new(EVAL_DOMAIN));
        verify_with(&ctx, &pcs, &mut vch, &u, &x, &y, &out.proof).unwrap();
        assert_eq!(
            vch.log,
            without_aborted_attempts(&ch.log, out.proof.attempt)
        );
        assert_eq!(
            vch.inner.transcript().digest(),
            ch.inner.transcript().digest()
        );
        // The provers' log has one (Begin, v, R2l) triple per attempt.
        let begins: Vec<u64> = ch
            .log
            .iter()
            .filter_map(|c| match c {
                Call::Begin(a) => Some(*a),
                _ => None,
            })
            .collect();
        assert_eq!(begins, (0..=out.proof.attempt).collect::<Vec<_>>());
        // The documented order: 15 parameter words, A, B, D, the digest of the base PCS's
        // parameters, u, x, y, E(u), then the step.
        let l = &ctx.layout;
        let head: Vec<String> = vch.log.iter().take(23).map(|c| format!("{c:?}")).collect();
        for (k, label) in LABELS_PARAMS.iter().enumerate() {
            assert!(
                matches!(&vch.log[k], Call::U64(lab, _) if lab == label),
                "{k}"
            );
        }
        assert!(is_polys(&vch.log[15], LABEL_PP_A, l.n_a * l.width_a));
        let digest = pcs
            .params_digest()
            .as_chunks::<8>()
            .0
            .iter()
            .fold(0u64, |acc, b| acc.wrapping_add(u64::from_le_bytes(*b)));
        assert_eq!(vch.log[18], Call::Words(LABEL_PP_BASE, 1, digest));
        assert!(is_polys(&vch.log[19], hachi_co_commit::LABEL_U, l.n_b));
        assert!(is_polys(&vch.log[22], hachi_co_conv::LABEL_E_U, 1));
        assert_eq!(vch.log[23], Call::Begin(out.proof.attempt));
        assert!(is_polys(&vch.log[24], hachi_co_r2l::LABEL_V, l.n_d));
        assert!(matches!(vch.log[25], Call::R2l(a, ..) if a == out.proof.attempt));
        // G0, G1, the three commitments, coins, l'' x (H0, H1, a_r), three final absorbs, three
        // contexts.
        assert_eq!(vch.log.len(), 26 + 6 + 3 * l.ell_pp + 3 + 3);
        let words =
            |k: usize, label: &str| matches!(&vch.log[k], Call::Words(lab, 1, _) if *lab == label);
        assert!(words(28, hachi_co_sumcheck::LABEL_COMMIT_W));
        assert!(words(29, hachi_co_sumcheck::LABEL_COMMIT_G0));
        assert!(words(30, hachi_co_sumcheck::LABEL_COMMIT_G1));
        assert!(matches!(vch.log[31], Call::Coins(..)));
        let tail = &vch.log[vch.log.len() - 3..];
        for (call, poly) in tail.iter().zip(ClaimedPoly::ORDER) {
            assert!(
                matches!(call, Call::Context(label, _) if *label == opening_label(1, poly)),
                "{call:?}"
            );
        }
        eprintln!(
            "toy-n1 (setup {}, a* = {}): verifier's calls: {head:?} ... {:?}",
            100 + j,
            out.proof.attempt,
            &vch.log[23..]
        );
        return;
    }
    panic!("no instance with an aborted attempt among 32");
}
