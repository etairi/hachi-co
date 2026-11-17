//! Hachi-Co with Hachi's tail (`HachiBase::with_tail`, not the draft's protocol), with the stub
//! tail of `hachi` (feature `stub-tail`), over the in-memory network of `hachi-co-mpc`:
//!
//! - Honest runs at toy-n2 and toy-n3, with a tail that takes every last table of the default
//!   schedules and with one that takes tables of up to $2^{30}$ digits, are accepted: one proof
//!   at every party, the verifier outputs $f(\mathbf x)$, its claims are the parties'
//!   polynomials, every opening has the iterations of the tail's rule, only the share
//!   polynomial's opening carries a tail proof, every item has the size of `HachiBase::words`,
//!   and the encoding has the size of the formula.
//! - A changed word of a tail commitment, of the party's tail proof, or a tail proof moved to a
//!   blinder's opening, is rejected, and so are all three openings of that party.
//! - The proof is bound to the tail: it is rejected without a tail and with another tail.
//! - Merged commitments refuse a tail.
//! - An ignored test prints the schedules and sizes of the secure presets with tails of
//!   $2^{21}$, $2^{24}$ and $2^{28}$ digits.
//!
//! Seeds: setup `seed("setup/<preset>", [index])`, instance `ChaCha8Rng::seed_from_u64(9000 + 10
//! n + index)`, shares `seed_from_u64(9500 + 10 n + index)`, party `seed("party/tail", [n,
//! index, p])`, with `index` 0 and 1 for the two tails of the honest runs and 2 for the others.

mod common;

use common::net::run_parties;
use common::{
    K, additive_shares, assert_claims_match, fresh_context, hachi_base, random_instance, seed,
};

use hachi::tail::StubTail;
use hachi::{Proof, Tail};
use hachi_co::{
    BasePcs, ClaimedPoly, Context, EVAL_DOMAIN, EvalOutput, EvalProof, HachiBase, VerifyError,
    commit, encoded_len, eval_with, verify_claims_with, verify_value,
};
use hachi_co_core::{Ext, FsChallenger, Poly, PrgRandomness};
use hachi_co_params::sizes_with_base;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;

/// One run of every party with a tail.
struct Run {
    name: String,
    ctx: Context<K>,
    pcs: HachiBase<K>,
    u: Vec<Poly>,
    x: Vec<Ext<K>>,
    y: Ext<K>,
    outs: Vec<EvalOutput<K>>,
}

impl Run {
    fn proof(&self) -> &EvalProof<K> {
        &self.outs[0].proof
    }

    fn verify_with(&self, pcs: &impl BasePcs<K>, proof: &EvalProof<K>) -> Result<(), VerifyError> {
        verify_value(&self.ctx, pcs, &self.u, &self.x, &self.y, proof)
    }

    fn check(&self, proof: &EvalProof<K>) -> Result<(), VerifyError> {
        self.verify_with(&self.pcs, proof)
    }
}

/// The stub tail for the base PCS `pcs`, with tables of at most $2^{\mathtt{max\_vars}}$ digits.
fn stub(pcs: &HachiBase<K>, max_vars: usize) -> Arc<dyn Tail<K>> {
    Arc::new(StubTail::new(pcs.pcs().ctx().q, pcs.choices().b, max_vars).unwrap())
}

/// The default base PCS of (`name`, `index`) with the stub tail of `max_vars`.
fn tail_base(ctx: &Context<K>, name: &str, index: u64, max_vars: usize) -> HachiBase<K> {
    let base = hachi_base(ctx, name, index, None);
    let tail = stub(&base, max_vars);
    base.with_tail(tail).unwrap()
}

/// Commit and Eval at the preset `name` with $n$ parties on threads, with the base PCS that
/// `pcs` makes for the context.
fn run(name: &str, n: usize, index: u64, pcs: impl FnOnce(&Context<K>) -> HachiBase<K>) -> Run {
    let ctx = fresh_context(name, index);
    assert_eq!(ctx.layout.n, n, "{name}");
    let pcs = pcs(&ctx);
    let tag = 10 * n as u64 + index;
    let (f, x, y) = random_instance(&ctx, 9000 + tag);
    let shares = additive_shares(&ctx, &mut ChaCha8Rng::seed_from_u64(9500 + tag), &f, n);
    let results = run_parties(n, ctx.q(), |p, comm| {
        let mut rng = PrgRandomness::new(seed("party/tail", &[n as u64, index, p as u64]));
        let (u, state) = commit(&ctx, comm, &mut rng, &shares[p - 1]).unwrap();
        let mut ch = FsChallenger::new(EVAL_DOMAIN);
        let out = eval_with(&ctx, &pcs, comm, &mut rng, &mut ch, &state, &u, &x)
            .unwrap_or_else(|e| panic!("{name}, party {p}: {e}"));
        (u, out)
    });
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
    }
}

/// The decoded opening of `poly` of party `p0 + 1`.
fn opening(r: &Run, proof: &EvalProof<K>, poly: ClaimedPoly, p0: usize) -> Proof<K> {
    r.pcs
        .proof_from_words(&proof.sumcheck.openings.of(poly)[p0])
        .unwrap()
}

/// The proof with the opening of `poly` of party `p0 + 1` replaced by `edit` of its decoding.
fn edited(r: &Run, poly: ClaimedPoly, p0: usize, edit: impl FnOnce(&mut Proof<K>)) -> EvalProof<K> {
    let mut proof = r.proof().clone();
    let mut decoded = opening(r, &proof, poly, p0);
    edit(&mut decoded);
    proof.sumcheck.openings.of_mut(poly)[p0] = r.pcs.proof_to_words(&decoded);
    proof
}

/// The verdict is a rejected opening of party `party` (the first of its three in the
/// verifier's order).
fn rejected_party(result: Result<(), VerifyError>, party: usize) -> bool {
    matches!(
        result,
        Err(VerifyError::Sumcheck(hachi_co_sumcheck::VerifyError::Opening { party: p, .. }))
            if p == party
    )
}

/// The checks of an honest run with a tail (see the module documentation).
fn check_honest(r: &Run) {
    let (name, ctx, pcs) = (&r.name, &r.ctx, &r.pcs);
    let proof = r.proof();
    assert!(
        r.outs.iter().all(|o| o.proof == *proof),
        "{name}: one proof"
    );
    let (value, claims) = verify_claims_with(
        ctx,
        pcs,
        &mut FsChallenger::new(EVAL_DOMAIN),
        &r.u,
        &r.x,
        proof,
    )
    .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(value, r.y, "{name}: the verifier outputs f(x)");
    let privates: Vec<_> = r.outs.iter().map(|o| &o.sumcheck).collect();
    assert_claims_match(ctx.field(), &claims, &privates, name);
    let y_false = ctx.field().add(&r.y, &ctx.field().one());
    assert_eq!(
        verify_value(ctx, pcs, &r.u, &r.x, &y_false, proof),
        Err(VerifyError::ValueMismatch),
        "{name}"
    );
    // The items: the sizes of HachiBase::words, the iterations of the tail's rule, and the
    // party's tail proof in the share polynomial's opening only.
    let l = &ctx.layout;
    let words = pcs.words(l).unwrap();
    let schedules = pcs.schedules(l).unwrap();
    let tail = pcs.tail().unwrap();
    let sizes = pcs.last_tables(l).unwrap();
    assert!(sizes.iter().all(|v| *v <= tail.max_vars()), "{name}");
    let t = &proof.sumcheck;
    for (i, poly) in [ClaimedPoly::W, ClaimedPoly::G0, ClaimedPoly::G1]
        .into_iter()
        .enumerate()
    {
        let (commitment, words) = match poly {
            ClaimedPoly::W => (words.commitment_w, words.proof_w),
            ClaimedPoly::G0 => (words.commitment_g0, words.proof_g0),
            ClaimedPoly::G1 => (words.commitment_g1, words.proof_g1),
        };
        for p0 in 0..l.n {
            assert_eq!(t.commitments.of(poly)[p0].len() as u128, commitment);
            assert_eq!(
                t.openings.of(poly)[p0].len() as u128,
                words,
                "{name}, {poly:?}"
            );
            let decoded = opening(r, proof, poly, p0);
            assert_eq!(decoded.iterations.len(), schedules[i].iterations.len());
            let carried = &decoded.tail.as_ref().unwrap().proof;
            let expected = if poly == ClaimedPoly::W {
                tail.proof_words(&sizes)
            } else {
                0
            };
            assert_eq!(carried.len(), expected, "{name}, {poly:?}");
            let last = decoded.iterations.last().unwrap();
            assert_eq!(
                last.tail_commitment.as_ref().unwrap().len(),
                tail.commitment_words(sizes[i])
            );
        }
    }
    let bytes = proof.to_bytes();
    assert_eq!(bytes.len(), encoded_len(l, &words).unwrap(), "{name}");
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
    eprintln!(
        "{name}, tail of 2^{}: iterations {:?}, last tables {sizes:?}, proof {} bytes, openings \
         {} + {} + {} words per party",
        tail.max_vars(),
        schedules.each_ref().map(|s| s.iterations.len()),
        bytes.len(),
        words.proof_w,
        words.proof_g0,
        words.proof_g1,
    );
}

#[test]
fn honest_runs_with_the_stub_tail_are_accepted() {
    for n in [2, 3] {
        let name = format!("toy-n{n}");
        // A tail that takes the largest last table of the default schedules, so that every
        // opening stops at least as early as it does without a tail, and one that takes every
        // table, so that every opening has one iteration.
        let ctx = fresh_context(&name, 0);
        let plain = hachi_base(&ctx, &name, 0, None);
        let largest = *plain
            .last_tables(&ctx.layout)
            .unwrap()
            .iter()
            .max()
            .unwrap();
        for (index, max_vars) in [(0, largest), (1, 30)] {
            let r = run(&name, n, index, |c| tail_base(c, &name, index, max_vars));
            check_honest(&r);
            let iterations = r
                .pcs
                .schedules(&r.ctx.layout)
                .unwrap()
                .map(|s| s.iterations.len());
            if max_vars == 30 {
                assert_eq!(iterations, [1; 3], "{name}");
            }
        }
    }
}

#[test]
fn changed_tail_words_are_rejected() {
    let r = run("toy-n2", 2, 2, |c| tail_base(c, "toy-n2", 2, 30));
    let proof = r.proof();
    r.check(proof).unwrap();
    let mut cases = 0;
    for p0 in 0..r.ctx.layout.n {
        let party = p0 + 1;
        for poly in [ClaimedPoly::W, ClaimedPoly::G0, ClaimedPoly::G1] {
            // A word of the tail's commitment: the opening's transcript changes, and the stub's
            // digest no longer matches.
            let len = opening(&r, proof, poly, p0).iterations[0]
                .tail_commitment
                .as_ref()
                .unwrap()
                .len();
            for at in [0, len - 1] {
                let tampered = edited(&r, poly, p0, |p| {
                    let last = p.iterations.last_mut().unwrap();
                    last.tail_commitment.as_mut().unwrap()[at] ^= 1;
                });
                assert!(
                    rejected_party(r.check(&tampered), party),
                    "party {party}, {poly:?}, tail commitment word {at}"
                );
                cases += 1;
            }
        }
        // A word of the party's tail proof, which the share polynomial's opening carries.
        let len = opening(&r, proof, ClaimedPoly::W, p0)
            .tail
            .unwrap()
            .proof
            .len();
        for at in [0, len / 2, len - 1] {
            let tampered = edited(&r, ClaimedPoly::W, p0, |p| {
                p.tail.as_mut().unwrap().proof[at] ^= 1 << 5;
            });
            assert!(
                rejected_party(r.check(&tampered), party),
                "party {party}, tail proof word {at}"
            );
            cases += 1;
        }
        // The tail proof moved to a blinder's opening, or copied there.
        let carried = opening(&r, proof, ClaimedPoly::W, p0).tail.unwrap().proof;
        for (moved, poly) in [(true, ClaimedPoly::G0), (false, ClaimedPoly::G1)] {
            let mut tampered = edited(&r, poly, p0, |p| {
                p.tail.as_mut().unwrap().proof.clone_from(&carried);
            });
            if moved {
                let mut w = opening(&r, proof, ClaimedPoly::W, p0);
                w.tail.as_mut().unwrap().proof.clear();
                tampered.sumcheck.openings.w[p0] = r.pcs.proof_to_words(&w);
            }
            assert!(
                rejected_party(r.check(&tampered), party),
                "party {party}, the tail proof in {poly:?} (moved {moved})"
            );
            cases += 1;
        }
    }
    // The parties' tail proofs swapped.
    let (w0, w1) = (
        opening(&r, proof, ClaimedPoly::W, 0),
        opening(&r, proof, ClaimedPoly::W, 1),
    );
    let mut swapped = proof.clone();
    let mut a = w0.clone();
    a.tail
        .as_mut()
        .unwrap()
        .proof
        .clone_from(&w1.tail.as_ref().unwrap().proof);
    swapped.sumcheck.openings.w[0] = r.pcs.proof_to_words(&a);
    assert!(rejected_party(r.check(&swapped), 1));
    cases += 1;
    eprintln!("{cases} tampered proofs rejected");
}

#[test]
fn the_proof_is_bound_to_the_tail() {
    let r = run("toy-n2", 2, 2, |c| tail_base(c, "toy-n2", 2, 30));
    let proof = r.proof();
    r.check(proof).unwrap();
    let plain = hachi_base(&r.ctx, "toy-n2", 2, None);
    let other = tail_base(&r.ctx, "toy-n2", 2, 29);
    assert_ne!(plain.params_digest(), r.pcs.params_digest());
    assert_ne!(other.params_digest(), r.pcs.params_digest());
    // The same schedules under the other tail, but another digest: the coins change.
    assert_eq!(
        other.schedules(&r.ctx.layout).unwrap(),
        r.pcs.schedules(&r.ctx.layout).unwrap()
    );
    for pcs in [&plain, &other] {
        assert!(matches!(
            r.verify_with(pcs, proof),
            Err(VerifyError::Sumcheck(_))
        ));
    }
    // Merged commitments do not take a tail.
    let base = hachi_base(&r.ctx, "toy-n2", 2, None);
    let tail = stub(&base, 30);
    assert!(
        base.clone()
            .merged()
            .unwrap()
            .with_tail(tail.clone())
            .is_err()
    );
    assert!(base.with_tail(tail).unwrap().merged().is_err());
}

/// The default base PCS of (`name`, `index`) with the Greyhound tail of the suffix `-greyhound`
/// (tables of at most $2^{28}$ digits, the preset's $\lambda$, the setup seed of the context).
#[cfg(feature = "greyhound")]
fn greyhound_base(ctx: &Context<K>, name: &str, index: u64) -> HachiBase<K> {
    use hachi_co::greyhound::GreyhoundTail;
    use hachi_co::harness::DEFAULT_TAIL_VARS;
    let base = hachi_base(ctx, name, index, None);
    let lambda = common::params_of(name).choices.lambda;
    let setup = common::setup_seed(name, index);
    let tail = GreyhoundTail::for_base(&base, &ctx.layout, DEFAULT_TAIL_VARS, lambda, &setup)
        .unwrap_or_else(|e| panic!("{name}: Greyhound refuses the tail: {e}"));
    base.with_tail(Arc::new(tail)).unwrap()
}

/// Honest runs at toy-n2 and toy-n3 with the Greyhound tail: the checks of the stub's runs, a
/// changed word of a party's Greyhound proof or of a Greyhound commitment rejected, and the
/// suffix of the harness giving the same base PCS.
#[cfg(feature = "greyhound")]
#[test]
fn honest_runs_with_the_greyhound_tail_are_accepted() {
    use hachi_co::harness::Setup;
    for (n, index) in [(2, 3), (3, 3)] {
        let name = format!("toy-n{n}");
        let r = run(&name, n, index, |c| greyhound_base(c, &name, index));
        check_honest(&r);
        let proof = r.proof();
        let len = opening(&r, proof, ClaimedPoly::W, 0)
            .tail
            .unwrap()
            .proof
            .len();
        for at in [0, 1, len / 2, len - 1] {
            let tampered = edited(&r, ClaimedPoly::W, 0, |p| {
                p.tail.as_mut().unwrap().proof[at] ^= 1;
            });
            assert!(
                rejected_party(r.check(&tampered), 1),
                "{name}: Greyhound proof word {at} of {len}"
            );
        }
        for poly in [ClaimedPoly::W, ClaimedPoly::G1] {
            let tampered = edited(&r, poly, n - 1, |p| {
                let last = p.iterations.last_mut().unwrap();
                last.tail_commitment.as_mut().unwrap()[0] ^= 1;
            });
            assert!(
                rejected_party(r.check(&tampered), n),
                "{name}: a word of the Greyhound commitment of {poly:?}"
            );
        }
        // The harness's suffix makes the same base PCS from the same setup seed.
        let setup = Setup::new(
            &format!("{name}-greyhound"),
            n,
            &common::setup_seed(&name, index),
        )
        .unwrap();
        assert_eq!(setup.pcs.params_digest(), r.pcs.params_digest(), "{name}");
        assert_eq!(setup.pcs.words(&r.ctx.layout), r.pcs.words(&r.ctx.layout));
    }
}

/// The schedules and sizes of the secure presets with the Greyhound tail of the harness's suffix
/// `-greyhound`, next to the presets without it: per opening ($\widetilde w$, $g\_0$, $g\_1$) the
/// iterations and the last table's $L$, the words of the openings, and the proof's bytes of
/// [`sizes_with_base`] (those of `hachi-co-party --sizes`). Run with `cargo test --release -p
/// hachi-co --test tail greyhound_schedules -- --ignored --nocapture`.
#[cfg(feature = "greyhound")]
#[test]
#[ignore]
fn greyhound_schedules_of_the_secure_presets() {
    use hachi_co::harness::Setup;
    for (preset, n) in [
        ("secure100-l20-n8-a8", 8),
        ("secure100-l22-n8-a8", 8),
        ("secure100-l24-n8-a8", 8),
        ("secure100-l20-n4-a8", 4),
        ("secure100-l22-n4-a8", 4),
        ("secure100-l24-n4-a8", 4),
    ] {
        for name in [preset.to_owned(), format!("{preset}-greyhound")] {
            let setup = Setup::new(&name, n, &[0u8; 32]).unwrap();
            let (l, pcs) = (&setup.ctx.layout, &setup.pcs);
            let schedules = pcs.schedules(l).unwrap();
            let words = pcs.words(l).unwrap();
            let sizes = sizes_with_base(&setup.ctx.params, &words).unwrap();
            let per = |s: &hachi::Schedule| {
                (
                    s.iterations.len(),
                    s.iterations.last().map(|it| it.table_vars),
                )
            };
            let tail = pcs
                .tail()
                .map(|t| {
                    let sizes = pcs.last_tables(l).unwrap();
                    format!(
                        ", tail proof {} words, tail commitments {:?} words",
                        t.proof_words(&sizes),
                        sizes.map(|v| t.commitment_words(v))
                    )
                })
                .unwrap_or_default();
            println!(
                "{name}: (iterations, last L) w {:?} g0 {:?} g1 {:?}; opening words w {} g0 {} g1 \
                 {}{tail}; proof {} bytes ({:.2} MB)",
                per(&schedules[0]),
                per(&schedules[1]),
                per(&schedules[2]),
                words.proof_w,
                words.proof_g0,
                words.proof_g1,
                sizes.total,
                sizes.total as f64 / 1e6,
            );
        }
    }
}

/// The schedules and sizes of the secure presets with stub tails of $2^{21}$, $2^{24}$ and
/// $2^{28}$ digits, against Hachi without a tail: per opening ($\widetilde w$, $g\_0$, $g\_1$)
/// the iterations and the last table's $L$, and the words of the openings without the tail's
/// proof. Run with `cargo test --release -p hachi-co --test tail schedules_of_the_secure_presets
/// -- --ignored --nocapture`.
#[test]
#[ignore]
fn schedules_of_the_secure_presets() {
    use hachi_co::harness::Setup;
    for preset in [
        "secure100-l20-n8-a8",
        "secure100-l22-n8-a8",
        "secure100-l24-n8-a8",
    ] {
        let setup = Setup::new(preset, 8, &[0u8; 32]).unwrap();
        let (l, base) = (&setup.ctx.layout, &setup.pcs);
        let describe = |pcs: &HachiBase<K>| {
            let schedules = pcs.schedules(l).unwrap();
            let words = pcs.words(l).unwrap();
            let carried = match pcs.tail() {
                Some(tail) => tail.proof_words(&pcs.last_tables(l).unwrap()) as u128,
                None => 0,
            };
            let per = |s: &hachi::Schedule| {
                (
                    s.iterations.len(),
                    s.iterations.last().map(|it| it.table_vars),
                )
            };
            format!(
                "(iterations, last L) w {:?} g0 {:?} g1 {:?}; opening words without the tail's \
                 proof w {} g0 {} g1 {} ({:.1} KiB per party); commitment {} words",
                per(&schedules[0]),
                per(&schedules[1]),
                per(&schedules[2]),
                words.proof_w - carried,
                words.proof_g0,
                words.proof_g1,
                8.0 * (words.proof_w - carried + words.proof_g0 + words.proof_g1) as f64 / 1024.0,
                words.commitment_w,
            )
        };
        println!("{preset}, no tail: {}", describe(base));
        for max_vars in [21, 24, 28] {
            let pcs = base.clone().with_tail(stub(base, max_vars)).unwrap();
            println!("{preset}, tail 2^{max_vars}: {}", describe(&pcs));
        }
    }
}
