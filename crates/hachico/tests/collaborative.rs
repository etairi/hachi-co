//! Hachi-Co with n provers on threads: every party runs
//! Commit and Eval on its own thread over the in-memory network of `hachi-co-mpc`, whose
//! F_AH is the masked broadcast (each party adds its share of a random additive sharing of 0
//! to its output share, broadcasts it, and everyone adds; as the authors specified, 2026-10-19).
//!
//! - honest runs at n = 2, 3, 4, 8 with both zero-sharing modes: every party has the same u and
//!   the same proof, Open = 1 on the states, the verifier accepts, every party's transcript ends
//!   in the verifier's state, and the claims are the parties' polynomials;
//! - the proof does not depend on the network: the same protocol seeds give the same proof
//!   bytes with another network seed and with the other zero-sharing mode;
//! - n parties against one party over the same field: both prove the same true statements, and
//!   every prover refuses a false y in both;
//! - a party that leaves after any number of frames: the run ends at once (no party waits for
//!   the timeout), the leaving party fails, and a party that finishes has the fault-free proof;
//! - the payload bytes of every collective call against the protocol's message sizes, in both
//!   modes; an ignored test prints the report at l = 12 for n = 2, 4, 8.
//!
//! Seeds: setup `seed("setup/collaborative", [n, i])`, instance
//! `ChaCha8Rng::seed_from_u64(6000 + 10 n + i)` and shares `seed_from_u64(7000 + 10 n + i)`,
//! party `seed("party/collaborative", [n, i, p])`, network `seed("network", [n, i])`
//! (`common::seed` is SHA-256 of the tag and the indices).

mod common;

use common::net::run_with;
use common::{
    K, additive_shares, assert_claims_match, clear_pcs, clear_words, random_instance, seed,
};
use hachi_co::{EVAL_DOMAIN, EvalOutput, EvalProof, commit, eval_with, verify_claims_with};
use hachi_co_commit::{LABEL_FAH_B, LABEL_RESHARE, open};
use hachi_co_conv::LABEL_FAH_E;
use hachi_co_core::{Context, Ext, FsChallenger, PartyState, Poly, PrgRandomness};
use hachi_co_mpc::{Comm, Fault, LABEL_SETUP, Loopback, MemoryConfig, Stats, ZeroSharing};
use hachi_co_params::presets::toy_choices;
use hachi_co_params::{Choices, QChoice, derive};
use hachi_co_r2l::{COMM_ABORT, COMM_FAH_D, COMM_GATHER_Z, COMM_PROVER1, response_len};
use hachi_co_sumcheck::{
    ALL_GATHER_COMMITMENTS, ALL_GATHER_FINAL, ALL_GATHER_OPENINGS, ALL_GATHER_SUMS, FAH_ROUND,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::{Duration, Instant};

const MODES: [ZeroSharing; 2] = [ZeroSharing::PairwiseSeeds, ZeroSharing::Fresh];

/// A fresh context for `choices`, from Setup with the seed of ($n$, `index`).
fn context(choices: &Choices, index: u64) -> Context<K> {
    let params = derive(choices).unwrap().0;
    hachi_co::setup(params, &seed("setup/collaborative", &[choices.n, index])).unwrap()
}

/// A statement and the parties' initial additive shares of $\mathbf f$.
struct Instance {
    ctx: Context<K>,
    f: Vec<Ext<K>>,
    x: Vec<Ext<K>>,
    y: Ext<K>,
    shares: Vec<Vec<Ext<K>>>,
}

fn instance(choices: &Choices, i: u64) -> Instance {
    let ctx = context(choices, i);
    let n = choices.n;
    let (f, x, y) = random_instance(&ctx, 6000 + 10 * n + i);
    let mut rng = ChaCha8Rng::seed_from_u64(7000 + 10 * n + i);
    let shares = additive_shares(&ctx, &mut rng, &f, n as usize);
    Instance {
        ctx,
        f,
        x,
        y,
        shares,
    }
}

/// The network of run `i` of `inst`, with a 30 s timeout.
fn network(inst: &Instance, i: u64, mode: ZeroSharing) -> MemoryConfig {
    let n = inst.shares.len();
    MemoryConfig::new(n, inst.ctx.q(), seed("network", &[n as u64, i]))
        .zero_sharing(mode)
        .timeout(Duration::from_secs(30))
}

/// What one party has after Commit and Eval.
#[derive(Clone, Debug)]
struct PartyOutput {
    u: Vec<Poly>,
    state: PartyState<K>,
    out: EvalOutput<K>,
    digest: [u8; 32],
}

/// Commit, then Eval with Fiat-Shamir and `ClearPcs`, for the party `comm.id()` with its initial
/// share `f_p` and the party seed of ($n$, `i`, $p$).
fn prove(
    ctx: &Context<K>,
    comm: &mut impl Comm,
    i: u64,
    f_p: &[Ext<K>],
    x: &[Ext<K>],
    y: &Ext<K>,
) -> Result<PartyOutput, hachi_co::Error> {
    let (n, p) = (comm.n() as u64, comm.id() as u64);
    let mut rng = PrgRandomness::new(seed("party/collaborative", &[n, i, p]));
    let (u, state) = commit(ctx, comm, &mut rng, f_p)?;
    let mut ch = FsChallenger::new(EVAL_DOMAIN);
    let out = eval_with(
        ctx,
        &clear_pcs(ctx),
        comm,
        &mut rng,
        &mut ch,
        &state,
        &u,
        x,
        y,
    )?;
    Ok(PartyOutput {
        u,
        state,
        out,
        digest: ch.transcript().digest(),
    })
}

/// Every party of `inst` (run `i`) on `config`; every party must succeed.
fn run_honest(inst: &Instance, i: u64, config: &MemoryConfig) -> (Vec<PartyOutput>, Vec<Stats>) {
    run_with(config, |p, comm| {
        prove(&inst.ctx, comm, i, &inst.shares[p - 1], &inst.x, &inst.y)
    })
    .into_iter()
    .map(|run| {
        let run = run.expect("every party connects");
        (run.output.expect("every honest party succeeds"), run.stats)
    })
    .unzip()
}

/// The checks of an honest run (see the module documentation).
fn check_honest(inst: &Instance, outs: &[PartyOutput], what: &str) {
    let first = &outs[0];
    for (i, o) in outs.iter().enumerate() {
        let p = i + 1;
        assert_eq!(o.u, first.u, "{what}: party {p} has u");
        assert_eq!(
            o.out.proof, first.out.proof,
            "{what}: party {p} has the proof"
        );
        assert_eq!(o.digest, first.digest, "{what}: party {p}'s transcript");
    }
    let states: Vec<_> = outs.iter().map(|o| o.state.clone()).collect();
    let report = open(&inst.ctx, &states, &first.u, &inst.f, None).unwrap();
    assert!(report.bit, "{what}: Open: {:?}", report.checks.failing());
    let mut vch = FsChallenger::new(EVAL_DOMAIN);
    let claims = verify_claims_with(
        &inst.ctx,
        &clear_pcs(&inst.ctx),
        &mut vch,
        &first.u,
        &inst.x,
        &inst.y,
        &first.out.proof,
    )
    .unwrap();
    assert_eq!(
        vch.transcript().digest(),
        first.digest,
        "{what}: the verifier's transcript"
    );
    let privates: Vec<_> = outs.iter().map(|o| &o.out.sumcheck).collect();
    assert_claims_match(inst.ctx.field(), &claims, &privates, what);
}

#[test]
fn honest_runs_with_2_3_4_and_8_parties_are_accepted() {
    for n in [2u64, 3, 4, 8] {
        for (i, mode) in (0u64..).zip(MODES) {
            let inst = instance(&toy_choices(n), i);
            let start = Instant::now();
            let (outs, stats) = run_honest(&inst, i, &network(&inst, i, mode));
            let elapsed = start.elapsed();
            let what = format!("toy n = {n}, {mode:?}");
            check_honest(&inst, &outs, &what);
            let bytes: u64 = stats.iter().map(|s| s.total().bytes_sent).sum();
            eprintln!(
                "{what}: a* = {}, Commit and Eval {elapsed:.2?}, {bytes} payload bytes sent in all",
                outs[0].out.proof.attempt
            );
        }
    }
}

#[test]
fn the_proof_does_not_depend_on_the_network() {
    // The masks of F_AH cancel and the other calls carry no network randomness, so the proof is
    // a function of the statement and the parties' seeds alone.
    let n = 4;
    let inst = instance(&toy_choices(n), 0);
    let (outs, _) = run_honest(&inst, 0, &network(&inst, 0, ZeroSharing::PairwiseSeeds));
    let reference = outs[0].out.proof.to_bytes();
    for mode in MODES {
        for index in [100u64, 101] {
            let config = MemoryConfig::new(n as usize, inst.ctx.q(), seed("network", &[n, index]))
                .zero_sharing(mode);
            let (outs, _) = run_honest(&inst, 0, &config);
            for (i, o) in outs.iter().enumerate() {
                assert_eq!(
                    o.out.proof.to_bytes(),
                    reference,
                    "{mode:?}, network seed {index}, party {}",
                    i + 1
                );
            }
        }
    }
}

#[test]
fn n_parties_and_one_party_prove_the_same_statements() {
    for n in [2u64, 3, 4, 8] {
        let inst = instance(&toy_choices(n), 0);
        let q = inst.ctx.q();
        // One party over the same field: the toy choices of n = 1 with the q of n parties.
        let one = Choices {
            q: QChoice::Fixed(q),
            ..toy_choices(1)
        };
        let ctx1 = context(&one, 0);
        assert_eq!(ctx1.q(), q);
        let (outs, _) = run_honest(&inst, 0, &network(&inst, 0, ZeroSharing::PairwiseSeeds));
        check_honest(&inst, &outs, &format!("n = {n}"));
        let mut comm = Loopback::new(q).unwrap();
        let single = prove(&ctx1, &mut comm, 0, &inst.f, &inst.x, &inst.y).unwrap();
        let mut vch = FsChallenger::new(EVAL_DOMAIN);
        verify_claims_with(
            &ctx1,
            &clear_pcs(&ctx1),
            &mut vch,
            &single.u,
            &inst.x,
            &inst.y,
            &single.out.proof,
        )
        .expect("the single party's proof of the same statement");
        // A false claim is refused by every prover, in both settings.
        let field = inst.ctx.field();
        let wrong = field.add(&inst.y, &field.one());
        let runs = run_with(&network(&inst, 0, ZeroSharing::PairwiseSeeds), |p, comm| {
            prove(&inst.ctx, comm, 0, &inst.shares[p - 1], &inst.x, &wrong).err()
        });
        for (i, run) in runs.into_iter().enumerate() {
            let err = run.unwrap().output;
            assert!(
                matches!(err, Some(hachi_co::Error::ClaimMismatch(_))),
                "n = {n}, party {}: {err:?}",
                i + 1
            );
        }
        let mut comm = Loopback::new(q).unwrap();
        assert!(matches!(
            prove(&ctx1, &mut comm, 0, &inst.f, &inst.x, &wrong),
            Err(hachi_co::Error::ClaimMismatch(_))
        ));
    }
}

#[test]
fn a_party_that_leaves_ends_the_run_at_once() {
    // Each party in turn leaves after k frames, for every k below the number of frames it sends
    // in the fault-free run. The timeout is 30 s, so a run that ends within 10 s had no party
    // waiting for it.
    let n = 3u64;
    let inst = instance(&toy_choices(n), 0);
    let base = network(&inst, 0, ZeroSharing::PairwiseSeeds);
    let (reference, stats) = run_honest(&inst, 0, &base);
    let proof = &reference[0].out.proof;
    let mut runs = 0;
    let mut finished = 0;
    for faulty in 1..=n as usize {
        let frames = stats[faulty - 1].total().frames_sent as usize;
        for k in 0..frames {
            let config = base.clone().fault(faulty, Fault::Leave { after_frames: k });
            let start = Instant::now();
            let results = run_with(&config, |p, comm| {
                prove(&inst.ctx, comm, 0, &inst.shares[p - 1], &inst.x, &inst.y)
                    .map(|o| o.out.proof)
            });
            let elapsed = start.elapsed();
            assert!(
                elapsed < Duration::from_secs(10),
                "party {faulty} leaves after {k} frames: {elapsed:.2?}"
            );
            for (i, result) in results.into_iter().enumerate() {
                let p = i + 1;
                if let Ok(Ok(got)) = result.map(|r| r.output) {
                    assert_ne!(p, faulty, "the leaving party finished (k = {k})");
                    assert_eq!(
                        &got, proof,
                        "party {p} finished with another proof (k = {k})"
                    );
                    finished += 1;
                }
            }
            runs += 1;
        }
    }
    eprintln!("{runs} runs with a party leaving; {finished} other parties still finished");
}

/// The payload bytes that party $p$ sends under each label in an honest run with `attempts`
/// attempts, from the protocol's message sizes (8 bytes per word): the resharing sends a share of
/// $2^\ell$ field elements to every other party; F_AH(B), F_AH(E), F_AH(D) and the round calls
/// broadcast $n\_B d$, $d$, $n\_D d$ and $(d\_0 + d\_1 + 2) K$ words (twice with fresh zero
/// sharings); the abort and prover-1 flags are one word; the responses go to prover 1; the
/// all-gathers broadcast $2K$ and $3K$ words, and the three base-PCS commitments and the three
/// opening proofs of `ClearPcs`, each with its length word.
fn expected_bytes_sent(
    ctx: &Context<K>,
    n: usize,
    p: usize,
    attempts: u64,
    mode: ZeroSharing,
) -> Vec<(&'static str, u64)> {
    let l = &ctx.layout;
    let base = clear_words(l);
    let commitments = 3 + (base.commitment_w + base.commitment_g0 + base.commitment_g1) as u64;
    let openings = 3 + (base.proof_w + base.proof_g0 + base.proof_g1) as u64;
    let peers = (n - 1) as u64;
    let fah = if mode == ZeroSharing::Fresh { 2 } else { 1 };
    let (k, d) = (K as u64, l.d as u64);
    let seeds = if mode == ZeroSharing::PairwiseSeeds {
        4 * (n - p) as u64
    } else {
        0
    };
    vec![
        (LABEL_SETUP, 8 * (3 * peers + seeds)),
        (LABEL_RESHARE, 8 * peers * k * (1u64 << l.ell)),
        (LABEL_FAH_B, 8 * peers * fah * l.n_b as u64 * d),
        (LABEL_FAH_E, 8 * peers * fah * d),
        (COMM_FAH_D, 8 * peers * fah * l.n_d as u64 * d * attempts),
        (COMM_ABORT, 8 * peers * attempts),
        (
            COMM_GATHER_Z,
            if p == 1 {
                0
            } else {
                8 * response_len(ctx) as u64
            },
        ),
        (COMM_PROVER1, 8 * peers),
        (ALL_GATHER_SUMS, 8 * peers * 2 * k),
        (ALL_GATHER_COMMITMENTS, 8 * peers * commitments),
        (
            FAH_ROUND,
            8 * peers * fah * (l.d0 + l.d1 + 2) as u64 * k * l.ell_pp as u64,
        ),
        (ALL_GATHER_FINAL, 8 * peers * 3 * k),
        (ALL_GATHER_OPENINGS, 8 * peers * openings),
    ]
}

/// Checks every party's accounting against [`expected_bytes_sent`], and that the bytes received
/// in all equal the bytes sent in all, label by label.
fn check_bytes(ctx: &Context<K>, proof: &EvalProof<K>, stats: &[Stats], mode: ZeroSharing) {
    let n = stats.len();
    let attempts = proof.attempt + 1;
    let mut total = Stats::default();
    for (i, s) in stats.iter().enumerate() {
        let p = i + 1;
        let expected = expected_bytes_sent(ctx, n, p, attempts, mode);
        for &(label, bytes) in &expected {
            assert_eq!(
                s.label(label).bytes_sent,
                bytes,
                "n = {n}, {mode:?}, party {p}, {label}"
            );
        }
        let listed: u64 = expected.iter().map(|(_, b)| b).sum();
        assert_eq!(
            s.total().bytes_sent,
            listed,
            "n = {n}, {mode:?}, party {p}: unlisted labels"
        );
        total += s;
    }
    for (label, s) in total.iter() {
        assert_eq!(s.bytes_sent, s.bytes_received, "n = {n}, {mode:?}, {label}");
        assert_eq!(
            s.frames_sent, s.frames_received,
            "n = {n}, {mode:?}, {label}"
        );
    }
}

#[test]
fn the_bytes_of_every_call_match_the_message_sizes() {
    for n in [2u64, 4, 8] {
        for mode in MODES {
            let inst = instance(&toy_choices(n), 0);
            let (outs, stats) = run_honest(&inst, 0, &network(&inst, 0, mode));
            check_bytes(&inst.ctx, &outs[0].out.proof, &stats, mode);
        }
    }
}

/// The report at l = 12 (the toy choices of each n with l = 12): per label, the calls, rounds and
/// payload bytes of party 1 and of party n, and the total over the parties; timings of the run.
#[test]
#[ignore = "a report: run with --ignored --nocapture"]
fn byte_report_at_l_12() {
    for n in [2u64, 4, 8] {
        let choices = Choices {
            ell: 12,
            ..toy_choices(n)
        };
        let inst = instance(&choices, 0);
        let start = Instant::now();
        let (outs, stats) = run_honest(&inst, 0, &network(&inst, 0, ZeroSharing::PairwiseSeeds));
        let elapsed = start.elapsed();
        check_honest(&inst, &outs, &format!("l = 12, n = {n}"));
        check_bytes(
            &inst.ctx,
            &outs[0].out.proof,
            &stats,
            ZeroSharing::PairwiseSeeds,
        );
        let mut total = Stats::default();
        for s in &stats {
            total += s;
        }
        eprintln!(
            "\nl = 12, n = {n}, q = {}, a* = {}, Commit and Eval {elapsed:.2?} (threads), proof {} bytes",
            inst.ctx.q(),
            outs[0].out.proof.attempt,
            outs[0].out.proof.to_bytes().len()
        );
        eprintln!(
            "{:<36} {:>6} {:>6} {:>14} {:>14} {:>14}",
            "label", "calls", "rounds", "sent by P1", "sent by Pn", "sent in all"
        );
        for (label, all) in total.iter() {
            let first = stats[0].label(label);
            let last = stats[n as usize - 1].label(label);
            eprintln!(
                "{label:<36} {:>6} {:>6} {:>14} {:>14} {:>14}",
                first.calls, first.rounds, first.bytes_sent, last.bytes_sent, all.bytes_sent
            );
        }
    }
}
