//! Hachi-Co with Hachi as the base PCS (`HachiBase`), with Fiat-Shamir, over the in-memory
//! network of `hachi-co-mpc`:
//!
//! - honest runs at every toy preset (n = 1, 2, 3, 4, 8) with Hachi's automatic schedule, which
//!   runs no iteration on polynomials this small, and at n = 1, 2 with one iteration per opening:
//!   every party has the same proof, the verifier accepts it, the claims it checked are the
//!   parties' polynomials, every base-PCS item has the size of `HachiBase::words`, every opening
//!   has the schedule's iterations, and the encoding has the size of the formula;
//! - the proof is bound to the base PCS: it is rejected under `ClearPcs`, under Hachi from
//!   another seed, and under Hachi with other choices but the same schedule (the parameter digest
//!   is part of the statement);
//! - tampered base-PCS items are rejected: changed, truncated or extended words of every
//!   commitment and every opening, openings or commitments swapped between parties, and an
//!   opening of the right polynomial made under another context;
//! - an ignored test prints timings and sizes.
//!
//! Seeds: setup `seed("setup/<preset>", [index])` (also the source of Hachi's matrices,
//! `hachi_co::base_pcs_seed`), instance `ChaCha8Rng::seed_from_u64(8000 + 10 n + index)`, shares
//! `seed_from_u64(8500 + 10 n + index)`, party `seed("party/hachi-base", [n, index, p])`, with
//! `index` 0 to 4 for the five tests in their order.

mod common;

use common::net::run_parties;
use common::{
    K, additive_shares, assert_claims_match, clear_pcs, fresh_context, hachi_base, random_instance,
    seed,
};
use hachi_co::{
    BasePcs, ClaimedPoly, Context, EVAL_DOMAIN, EvalOutput, EvalProof, HachiBase, PcsWords,
    VerifyError, commit, encoded_len, eval_with, verify, verify_claims_with,
};
use hachi_co_core::base_pcs::mobius;
use hachi_co_core::{Ext, FsChallenger, Poly, PrgRandomness};
use hachi_co_params::sizes_with_base;
use hachi_co_sumcheck::{blinder_coefficients, blinder_weights};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::{Duration, Instant};

/// One run of every party: the context, the base PCS, the statement and the parties' outputs.
struct Run {
    name: String,
    ctx: Context<K>,
    pcs: HachiBase<K>,
    u: Vec<Poly>,
    x: Vec<Ext<K>>,
    y: Ext<K>,
    outs: Vec<EvalOutput<K>>,
    prove_time: Duration,
}

impl Run {
    fn proof(&self) -> &EvalProof<K> {
        &self.outs[0].proof
    }

    fn verify_with(&self, pcs: &impl BasePcs<K>, proof: &EvalProof<K>) -> Result<(), VerifyError> {
        verify(&self.ctx, pcs, &self.u, &self.x, &self.y, proof)
    }

    fn check(&self, proof: &EvalProof<K>) -> Result<(), VerifyError> {
        self.verify_with(&self.pcs, proof)
    }
}

/// Commit and Eval at the preset `name` with $n$ parties on threads, Hachi running `iterations`
/// iterations per opening (`None`: its automatic schedule).
fn run(name: &str, n: usize, iterations: Option<usize>, index: u64) -> Run {
    let ctx = fresh_context(name, index);
    assert_eq!(ctx.layout.n, n, "{name}");
    let pcs = hachi_base(&ctx, name, index, iterations);
    let tag = 10 * n as u64 + index;
    let (f, x, y) = random_instance(&ctx, 8000 + tag);
    let shares = additive_shares(&ctx, &mut ChaCha8Rng::seed_from_u64(8500 + tag), &f, n);
    let start = Instant::now();
    let results = run_parties(n, ctx.q(), |p, comm| {
        let mut rng = PrgRandomness::new(seed("party/hachi-base", &[n as u64, index, p as u64]));
        let (u, state) = commit(&ctx, comm, &mut rng, &shares[p - 1]).unwrap();
        let mut ch = FsChallenger::new(EVAL_DOMAIN);
        let out = eval_with(&ctx, &pcs, comm, &mut rng, &mut ch, &state, &u, &x, &y)
            .unwrap_or_else(|e| panic!("{name}, party {p}: {e}"));
        (u, out)
    });
    let prove_time = start.elapsed();
    let u = results[0].0.clone();
    assert!(results.iter().all(|(v, _)| *v == u), "{name}: one u");
    Run {
        name: name.to_owned(),
        ctx,
        pcs,
        u,
        x,
        y,
        outs: results.into_iter().map(|(_, out)| out).collect(),
        prove_time,
    }
}

/// The checks of an honest run (see the module documentation); `iterations` is the number of
/// iterations every opening must have.
fn check_honest(r: &Run, iterations: usize) {
    let (name, ctx, pcs) = (&r.name, &r.ctx, &r.pcs);
    let proof = r.proof();
    assert!(
        r.outs.iter().all(|o| o.proof == *proof),
        "{name}: one proof"
    );
    let start = Instant::now();
    let claims = verify_claims_with(
        ctx,
        pcs,
        &mut FsChallenger::new(EVAL_DOMAIN),
        &r.u,
        &r.x,
        &r.y,
        proof,
    )
    .unwrap_or_else(|e| panic!("{name}: {e}"));
    let verify_time = start.elapsed();
    let privates: Vec<_> = r.outs.iter().map(|o| &o.sumcheck).collect();
    assert_claims_match(ctx.field(), &claims, &privates, name);
    // A false evaluation is rejected with this proof.
    let y_false = ctx.field().add(&r.y, &ctx.field().one());
    assert!(
        verify(ctx, pcs, &r.u, &r.x, &y_false, proof).is_err(),
        "{name}"
    );

    // Every item has the size of HachiBase::words, every opening the expected iterations.
    let words = pcs.words(&ctx.layout).unwrap();
    let t = &proof.sumcheck;
    for poly in ClaimedPoly::ORDER {
        let (commitment, opening) = match poly {
            ClaimedPoly::W => (words.commitment_w, words.proof_w),
            ClaimedPoly::G0 => (words.commitment_g0, words.proof_g0),
            ClaimedPoly::G1 => (words.commitment_g1, words.proof_g1),
        };
        assert_eq!(t.commitments.of(poly).len(), ctx.layout.n, "{name}");
        assert_eq!(t.openings.of(poly).len(), ctx.layout.n, "{name}");
        for (c, o) in t.commitments.of(poly).iter().zip(t.openings.of(poly)) {
            assert_eq!(c.len() as u128, commitment, "{name}, {poly:?}");
            assert_eq!(o.len() as u128, opening, "{name}, {poly:?}");
            let decoded = HachiBase::<K>::proof_from_words(o).unwrap();
            assert_eq!(decoded.iterations.len(), iterations, "{name}, {poly:?}");
        }
    }
    let bytes = proof.to_bytes();
    assert_eq!(
        bytes.len(),
        encoded_len(&ctx.layout, &words).unwrap(),
        "{name}"
    );
    assert_eq!(
        bytes.len() as u128,
        sizes_with_base(&ctx.params, &words).unwrap().total,
        "{name}"
    );
    assert_eq!(bytes.len(), proof.encoded_len(), "{name}");
    assert_eq!(
        EvalProof::from_bytes(ctx, &bytes).unwrap(),
        *proof,
        "{name}"
    );
    let pcs_bytes = 8 * (t.commitments.total_words() + t.openings.total_words());
    eprintln!(
        "{name}, {iterations} iterations: proof {} bytes, of which base PCS {pcs_bytes}; \
         per party: commitments 3 x {} words, openings {} + {} + {} words; \
         prove {:.2?} (all parties), verify {verify_time:.2?}",
        bytes.len(),
        words.commitment_w,
        words.proof_w,
        words.proof_g0,
        words.proof_g1,
        r.prove_time,
    );
}

#[test]
fn honest_runs_with_hachi_are_accepted_at_every_toy_preset() {
    for n in [1, 2, 3, 4, 8] {
        let r = run(&format!("toy-n{n}"), n, None, 0);
        check_honest(&r, 0);
    }
}

#[test]
fn honest_runs_with_an_iteration_of_hachi_are_accepted() {
    for n in [1, 2] {
        let r = run(&format!("toy-n{n}"), n, Some(1), 1);
        check_honest(&r, 1);
    }
}

#[test]
fn the_proof_is_bound_to_the_base_pcs() {
    let r = run("toy-n2", 2, None, 2);
    let proof = r.proof();
    r.check(proof).unwrap();
    let rejected =
        |result: Result<(), VerifyError>| matches!(result, Err(VerifyError::Sumcheck(_)));
    // Another scheme, Hachi from another seed, and Hachi whose choices differ (an explicit 0
    // iterations instead of the automatic schedule) but whose openings are the same: the digest
    // in the statement changes the coins, and the round checks fail.
    assert!(rejected(r.verify_with(&clear_pcs(&r.ctx), proof)));
    assert!(rejected(
        r.verify_with(&hachi_base(&r.ctx, "toy-n2", 99, None), proof)
    ));
    assert!(rejected(
        r.verify_with(&hachi_base(&r.ctx, "toy-n2", 2, Some(0)), proof)
    ));
}

/// `words` with the word at `i` changed (`+ 1`, wrapping).
fn changed(words: &[u64], i: usize) -> Vec<u64> {
    let mut w = words.to_vec();
    w[i] = w[i].wrapping_add(1);
    w
}

/// The verdict is a rejected opening of `poly` of `party`.
fn rejected_opening(result: Result<(), VerifyError>, party: usize, poly: ClaimedPoly) -> bool {
    matches!(
        result,
        Err(VerifyError::Sumcheck(hachi_co_sumcheck::VerifyError::Opening {
            party: p,
            poly: q,
            ..
        })) if p == party && q == poly
    )
}

#[test]
fn tampered_hachi_items_are_rejected() {
    let r = run("toy-n2", 2, Some(1), 3);
    let proof = r.proof();
    r.check(proof).unwrap();
    let n = r.ctx.layout.n;
    let with = |edit: &dyn Fn(&mut PcsWords, &mut PcsWords)| {
        let mut p = proof.clone();
        let t = &mut p.sumcheck;
        edit(&mut t.commitments, &mut t.openings);
        p
    };
    let mut cases = 0;
    for p0 in 0..n {
        for poly in ClaimedPoly::ORDER {
            // A changed commitment word changes every coin after it: the rounds fail.
            let len = proof.sumcheck.commitments.of(poly)[p0].len();
            for i in [0, 1, 2, 3, len / 2, len - 1] {
                let tampered = with(&|c, _| c.of_mut(poly)[p0] = changed(&c.of(poly)[p0], i));
                assert!(
                    matches!(r.check(&tampered), Err(VerifyError::Sumcheck(_))),
                    "party {}, {poly:?}, commitment word {i}",
                    p0 + 1
                );
                cases += 1;
            }
            // A changed, missing or extra opening word: that opening is rejected.
            let len = proof.sumcheck.openings.of(poly)[p0].len();
            let step = (len / 40).max(1);
            for i in (0..len).step_by(step).chain([len - 1]) {
                let tampered = with(&|_, o| o.of_mut(poly)[p0] = changed(&o.of(poly)[p0], i));
                assert!(
                    rejected_opening(r.check(&tampered), p0 + 1, poly),
                    "party {}, {poly:?}, opening word {i} of {len}",
                    p0 + 1
                );
                cases += 1;
            }
            let shorter = with(&|_, o| {
                o.of_mut(poly)[p0].pop();
            });
            assert!(rejected_opening(r.check(&shorter), p0 + 1, poly));
            let longer = with(&|_, o| o.of_mut(poly)[p0].push(0));
            assert!(rejected_opening(r.check(&longer), p0 + 1, poly));
            cases += 2;
        }
    }
    // Swapped between the parties: the openings fail at party 1; the commitments change the
    // coins.
    for poly in ClaimedPoly::ORDER {
        let swapped = with(&|_, o| o.of_mut(poly).swap(0, 1));
        assert!(rejected_opening(r.check(&swapped), 1, poly), "{poly:?}");
        let swapped = with(&|c, _| c.of_mut(poly).swap(0, 1));
        assert!(matches!(r.check(&swapped), Err(VerifyError::Sumcheck(_))));
        cases += 2;
    }
    // Party 1's polynomials, committed and opened again (Hachi's commitment is deterministic)
    // under another context: the commitments are the proof's, the values are right, but the
    // openings' Fiat-Shamir starts from the wrong context.
    let (field, l) = (r.ctx.field(), &r.ctx.layout);
    let private = &r.outs[0].sumcheck;
    let point = &private.point;
    let mut coeffs = private.w_table.clone();
    mobius(field.zq(), &mut coeffs).unwrap();
    let other = [0x5au8; 32];
    for poly in ClaimedPoly::ORDER {
        let (com, opening) = match poly {
            ClaimedPoly::W => {
                let (com, st) = r.pcs.commit_multilinear(&coeffs).unwrap();
                (com, r.pcs.prove_multilinear(&st, point, &other).unwrap().1)
            }
            ClaimedPoly::G0 | ClaimedPoly::G1 => {
                let (g, degree) = match poly {
                    ClaimedPoly::G0 => (&private.g0, l.d0),
                    _ => (&private.g1, l.d1),
                };
                let (com, st) = r.pcs.commit_vector(&blinder_coefficients(g)).unwrap();
                let weights = blinder_weights(field, point, degree);
                (
                    com,
                    r.pcs.prove_inner_product(&st, &weights, &other).unwrap().1,
                )
            }
        };
        assert_eq!(
            HachiBase::<K>::commitment_to_words(&com),
            proof.sumcheck.commitments.of(poly)[0],
            "{poly:?}: the same commitment"
        );
        let words = HachiBase::<K>::proof_to_words(&opening);
        let tampered = with(&|_, o| o.of_mut(poly)[0].clone_from(&words));
        assert!(rejected_opening(r.check(&tampered), 1, poly), "{poly:?}");
        cases += 1;
    }
    eprintln!("{cases} tampered proofs rejected");
}

#[test]
#[ignore = "timings and sizes at toy-n8 and dev-l12: run with --ignored --nocapture"]
fn timings_with_hachi() {
    for (name, n, iterations) in [
        ("toy-n8", 8, None),
        ("toy-n8", 8, Some(1)),
        ("dev-l12", 1, None),
        ("dev-l12", 1, Some(1)),
    ] {
        let r = run(name, n, iterations, 4);
        check_honest(&r, iterations.unwrap_or(0));
    }
}
