//! Hachi-Co with Hachi as the base PCS (`HachiBase`), with Fiat-Shamir, over the in-memory
//! network of `hachi-co-mpc`:
//!
//! - honest runs at every toy preset (n = 1, 2, 3, 4, 8) with Hachi's automatic schedule (per
//!   polynomial, the number of iterations whose proof is smallest), and at n = 1, 2 with one
//!   iteration per opening: every party has the same proof, the verifier accepts it, the claims
//!   it checked are the parties' polynomials, every base-PCS item has the size of
//!   `HachiBase::words`, every opening has the iterations of Hachi's schedule, and the encoding
//!   has the size of the formula;
//! - the proof is bound to the base PCS: it is rejected under `ClearPcs`, under Hachi from
//!   another seed, and under Hachi with other choices but the same schedules (the parameter
//!   digest is part of the statement);
//! - tampered base-PCS items are rejected: changed, truncated or extended words of every
//!   commitment and every opening, openings or commitments swapped between parties, and an
//!   opening of the right polynomial made under another context;
//! - the default choices randomize the commitments (`HachiChoices::hiding`, Gaussian randomness,
//!   `docs/hachi-hiding.md` §6), so the tests above run with them; honest runs with
//!   deterministic commitments at n = 1, 2, 3 (automatic schedule) and with digit randomness at
//!   n = 2 (one iteration) pass the same checks, and the proof is bound to the hiding choices;
//! - an ignored test prints timings and sizes.
//!
//! Seeds: setup `seed("setup/<preset>", [index])` (also the source of Hachi's matrices,
//! `hachi_co::base_pcs_seed`), instance `ChaCha8Rng::seed_from_u64(8000 + 10 n + index)`, shares
//! `seed_from_u64(8500 + 10 n + index)`, party `seed("party/hachi-base", [n, index, p])`, with
//! `index` 0 to 4 for the first five tests in their order, and 5, 6 for the hiding ones.

mod common;

use common::net::run_parties;
use common::{
    K, additive_shares, assert_claims_match, clear_pcs, fresh_context, hachi_base, random_instance,
    seed, setup_seed,
};

use hachi_co::{
    BasePcs, ClaimedPoly, Context, EVAL_DOMAIN, EvalOutput, EvalProof, HachiBase, HachiChoices,
    Hiding, Noise, PcsWords, VerifyError, commit, encoded_len, eval_with, verify,
    verify_claims_with,
};
use hachi_co_core::base_pcs::mobius;
use hachi_co_core::{Ext, FsChallenger, PartyRandomness, Poly, PrgRandomness};
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
    run_with(
        name,
        n,
        index,
        |ctx| hachi_base(ctx, name, index, iterations),
        ctx,
    )
}

/// [`run`] with the base PCS that `pcs` makes for the context.
fn run_with(
    name: &str,
    n: usize,
    index: u64,
    pcs: impl FnOnce(&Context<K>) -> HachiBase<K>,
    ctx: Context<K>,
) -> Run {
    assert_eq!(ctx.layout.n, n, "{name}");
    let pcs = pcs(&ctx);
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

/// The number of iterations of Hachi's schedule for the openings of `poly` at the layout of
/// `ctx`.
fn scheduled_iterations(pcs: &HachiBase<K>, ctx: &Context<K>, poly: ClaimedPoly) -> usize {
    let (l, hachi) = (&ctx.layout, pcs.pcs());
    let schedule = match poly {
        ClaimedPoly::W => hachi.multilinear_schedule(l.ell_pp),
        ClaimedPoly::G0 => hachi.vector_schedule(1 + l.ell_pp * l.d0),
        ClaimedPoly::G1 => hachi.vector_schedule(1 + l.ell_pp * l.d1),
    };
    schedule.unwrap().iterations.len()
}

/// The checks of an honest run (see the module documentation). Returns the numbers of
/// iterations of the openings, in the order of [`ClaimedPoly::ORDER`] ($g\_0$, $g\_1$,
/// $\langle \widetilde w \rangle$).
fn check_honest(r: &Run) -> [usize; 3] {
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

    // Every item has the size of HachiBase::words, every opening the scheduled iterations.
    let words = pcs.words(&ctx.layout).unwrap();
    let t = &proof.sumcheck;
    let iterations = ClaimedPoly::ORDER.map(|poly| scheduled_iterations(pcs, ctx, poly));
    for (poly, iterations) in ClaimedPoly::ORDER.into_iter().zip(iterations) {
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
            let decoded = pcs.proof_from_words(o).unwrap();
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
        "{name}, iterations {iterations:?} ({:?}): proof {} bytes, of which base PCS \
         {pcs_bytes}; per party: commitments 3 x {} words, openings {} + {} + {} words; \
         prove {:.2?} (all parties), verify {verify_time:.2?}",
        ClaimedPoly::ORDER,
        bytes.len(),
        words.commitment_w,
        words.proof_w,
        words.proof_g0,
        words.proof_g1,
        r.prove_time,
    );
    iterations
}

#[test]
fn honest_runs_with_hachi_are_accepted_at_every_toy_preset() {
    for n in [1, 2, 3, 4, 8] {
        let r = run(&format!("toy-n{n}"), n, None, 0);
        check_honest(&r);
    }
}

#[test]
fn honest_runs_with_an_iteration_of_hachi_are_accepted() {
    for n in [1, 2] {
        let r = run(&format!("toy-n{n}"), n, Some(1), 1);
        assert_eq!(check_honest(&r), [1; 3]);
    }
}

#[test]
fn the_proof_is_bound_to_the_base_pcs() {
    let r = run("toy-n2", 2, None, 2);
    let proof = r.proof();
    r.check(proof).unwrap();
    let rejected =
        |result: Result<(), VerifyError>| matches!(result, Err(VerifyError::Sumcheck(_)));
    // Another scheme, Hachi from another seed, and Hachi whose choices differ (a smaller
    // maximum number of iterations, which the automatic schedules do not reach) but whose
    // openings are the same: the digest in the statement changes the coins, and the round
    // checks fail.
    assert!(rejected(r.verify_with(&clear_pcs(&r.ctx), proof)));
    assert!(rejected(
        r.verify_with(&hachi_base(&r.ctx, "toy-n2", 99, None), proof)
    ));
    let choices = HachiChoices {
        max_iterations: 8,
        ..HachiBase::default_choices(&r.ctx, &setup_seed("toy-n2", 2))
    };
    let other = HachiBase::new(r.ctx.q(), choices).unwrap();
    for poly in ClaimedPoly::ORDER {
        assert_eq!(
            scheduled_iterations(&other, &r.ctx, poly),
            scheduled_iterations(&r.pcs, &r.ctx, poly),
            "{poly:?}: the same schedule"
        );
    }
    assert!(rejected(r.verify_with(&other, proof)));
}

/// Hachi for the context of (`name`, `index`): the default choices with `hiding` (`None`:
/// deterministic commitments) and the given number of iterations.
fn hiding_base(
    ctx: &Context<K>,
    name: &str,
    index: u64,
    hiding: Option<Hiding>,
    iterations: Option<usize>,
) -> HachiBase<K> {
    let choices = HachiChoices {
        hiding,
        fixed_iterations: iterations,
        ..HachiBase::default_choices(ctx, &setup_seed(name, index))
    };
    HachiBase::new(ctx.q(), choices).unwrap()
}

/// The hiding choices of the default ([`HachiChoices::toy_hiding`]).
const GAUSS8: Hiding = Hiding {
    nu: 1,
    noise: Noise::Gaussian { log_sigma: 3 },
};
const DIGITS2: Hiding = Hiding {
    nu: 2,
    noise: Noise::Digits,
};

#[test]
fn honest_runs_with_other_commitment_choices_are_accepted() {
    let ctx = fresh_context("toy-n1", 5);
    let default = HachiBase::default_choices(&ctx, &setup_seed("toy-n1", 5));
    assert_eq!(
        default.hiding,
        Some(GAUSS8),
        "the default randomizes the commitments"
    );
    // Deterministic commitments.
    for n in [1, 2, 3] {
        let name = format!("toy-n{n}");
        let ctx = fresh_context(&name, 5);
        let r = run_with(&name, n, 5, |c| hiding_base(c, &name, 5, None, None), ctx);
        check_honest(&r);
    }
    // Uniform digits as the randomness, with one iteration.
    let ctx = fresh_context("toy-n2", 5);
    let r = run_with(
        "toy-n2",
        2,
        5,
        |c| hiding_base(c, "toy-n2", 5, Some(DIGITS2), Some(1)),
        ctx,
    );
    assert_eq!(check_honest(&r), [1; 3]);
}

#[test]
fn the_proof_is_bound_to_the_hiding_choices() {
    let ctx = fresh_context("toy-n2", 6);
    let r = run_with(
        "toy-n2",
        2,
        6,
        |c| hiding_base(c, "toy-n2", 6, Some(GAUSS8), Some(1)),
        ctx,
    );
    let proof = r.proof();
    r.check(proof).unwrap();
    let rejected =
        |result: Result<(), VerifyError>| matches!(result, Err(VerifyError::Sumcheck(_)));
    // The same Hachi without hiding, or with other hiding choices: other digests, other coins.
    let plain = hiding_base(&r.ctx, "toy-n2", 6, None, Some(1));
    assert!(rejected(r.verify_with(&plain, proof)));
    for other in [
        Hiding { nu: 2, ..GAUSS8 },
        Hiding {
            noise: Noise::Gaussian { log_sigma: 4 },
            ..GAUSS8
        },
        Hiding {
            nu: 1,
            noise: Noise::Digits,
        },
    ] {
        let pcs = hiding_base(&r.ctx, "toy-n2", 6, Some(other), Some(1));
        assert_ne!(pcs.params_digest(), r.pcs.params_digest());
        assert!(rejected(r.verify_with(&pcs, proof)), "{other:?}");
    }
    assert_ne!(plain.params_digest(), r.pcs.params_digest());
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
    // Party 1's polynomials, committed and opened again under another context, with party 1's
    // base-PCS seeds (`PartyRandomness::base_pcs` draws each from its own stream, so they can be
    // drawn again here): the commitments are the proof's, the values are right, but the
    // openings' Fiat-Shamir starts from the wrong context.
    let (field, l) = (r.ctx.field(), &r.ctx.layout);
    let private = &r.outs[0].sumcheck;
    let point = &private.point;
    let mut coeffs = private.w_table.clone();
    mobius(field.zq(), &mut coeffs).unwrap();
    let other = [0x5au8; 32];
    let mut party = PrgRandomness::new(seed("party/hachi-base", &[n as u64, 3, 1]));
    for poly in ClaimedPoly::ORDER {
        let mut draw =
            |opening| PartyRandomness::<K>::base_pcs(&mut party, poly.seed_index(opening)).unwrap();
        let (commit_seed, open_seed) = (draw(false), draw(true));
        let (com, opening) = match poly {
            ClaimedPoly::W => {
                let (com, st) = r.pcs.commit_multilinear(&coeffs, &commit_seed).unwrap();
                (
                    com,
                    r.pcs
                        .prove_multilinear(&st, point, &other, &open_seed)
                        .unwrap()
                        .1,
                )
            }
            ClaimedPoly::G0 | ClaimedPoly::G1 => {
                let (g, degree) = match poly {
                    ClaimedPoly::G0 => (&private.g0, l.d0),
                    _ => (&private.g1, l.d1),
                };
                let (com, st) = r
                    .pcs
                    .commit_vector(&blinder_coefficients(g), &commit_seed)
                    .unwrap();
                let weights = blinder_weights(field, point, degree);
                (
                    com,
                    r.pcs
                        .prove_inner_product(&st, &weights, &other, &open_seed)
                        .unwrap()
                        .1,
                )
            }
        };
        assert_eq!(
            r.pcs.commitment_to_words(&com),
            proof.sumcheck.commitments.of(poly)[0],
            "{poly:?}: the same commitment"
        );
        let words = r.pcs.proof_to_words(&opening);
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
        check_honest(&r);
    }
    // With deterministic commitments, on the automatic schedule (the runs above use the default,
    // randomized commitments).
    for (name, n) in [("toy-n8", 8), ("dev-l12", 1)] {
        eprintln!("{name} with deterministic commitments:");
        let ctx = fresh_context(name, 4);
        let r = run_with(name, n, 4, |c| hiding_base(c, name, 4, None, None), ctx);
        check_honest(&r);
    }
}
