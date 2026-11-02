//! Several evaluations of one commitment, and the two rules of the pad.
//!
//! - One Commit and four Evals, with a randomness source of its own per Eval, at three distinct
//!   points and one point twice ($t$-hiding is stated for $K$ adaptive evaluations,
//!   `t-hiding.tex`:19): every proof verifies under the same commitment, Open = 1 on the states,
//!   $\boldsymbol\rho^u\_p$ stays in the state, and the two proofs of one claim differ. Reusing
//!   a source is refused.
//! - The default rule of the pad, one element of $\mathbb F\_{q^K}$ (`5_Building_Blocks.tex`:252),
//!   and the hidden one, $2k$ residues of $\mathbb Z\_q$ at row $\mu$ (`docs/hachi-zk-options.md`):
//!   honest runs are accepted at $n = 1, 2, 3$ with `ClearPcs` and with Hachi, the pad is where
//!   its rule puts it, and a proof under one rule does not verify under the other.
//!
//! Seeds: setup `seed("setup/evaluations", [n, pad, index])`, instance
//! `ChaCha8Rng::seed_from_u64(8000 + 10 n + index)` and shares
//! `seed_from_u64(8500 + 10 n + index)`, party `seed("party/evaluations", [n, index, p])` for
//! Commit and `seed("party/evaluations/eval", [n, index, p, k])` for the $k$-th Eval, network
//! `seed("network/evaluations", [n, index])` (`common::seed` is SHA-256 of the tag and the indices).

mod common;

use common::net::run_with;
use common::{
    K, additive_shares, clear_pcs, clear_words, evaluate_by_monomials, random_instance, seed,
};
use hachi::HachiChoices;
use hachi_co::{
    BasePcsWords, EVAL_DOMAIN, EvalOutput, HachiBase, commit, encoded_len, eval, eval_with, verify,
};
use hachi_co_commit::open;
use hachi_co_core::{BasePcs, Context, Ext, FsChallenger, Poly, PrgRandomness};
use hachi_co_mpc::{Comm, MemoryConfig, ZeroSharing};
use hachi_co_params::presets::toy_choices;
use hachi_co_params::{Choices, PadRule, derive};
use hachi_co_sumcheck::pad_index;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::time::Duration;

fn pad_word(pad: PadRule) -> u64 {
    match pad {
        PadRule::Field => 0,
        PadRule::Residues => 1,
    }
}

/// A fresh context for $n$ parties and the rule `pad`.
fn context(n: u64, pad: PadRule, index: u64) -> Context<K> {
    let choices = Choices {
        pad,
        ..toy_choices(n)
    };
    let params = derive(&choices).unwrap().0;
    let setup = seed("setup/evaluations", &[n, pad_word(pad), index]);
    hachi_co::setup(params, &setup).unwrap()
}

/// Hachi with its default choices, from the setup seed of `context`.
fn hachi(ctx: &Context<K>, n: u64, pad: PadRule, index: u64) -> HachiBase<K> {
    let setup = seed("setup/evaluations", &[n, pad_word(pad), index]);
    let choices: HachiChoices = HachiBase::default_choices(ctx, &setup);
    HachiBase::new(ctx.q(), choices).unwrap()
}

/// $\mathbf f$, the parties' initial shares of it, and the network of run `index`.
fn instance(ctx: &Context<K>, index: u64) -> (Vec<Ext<K>>, Vec<Vec<Ext<K>>>, MemoryConfig) {
    let n = ctx.layout.n as u64;
    let (f, _, _) = random_instance(ctx, 8000 + 10 * n + index);
    let mut rng = ChaCha8Rng::seed_from_u64(8500 + 10 * n + index);
    let shares = additive_shares(ctx, &mut rng, &f, n as usize);
    let network = MemoryConfig::new(
        n as usize,
        ctx.q(),
        seed("network/evaluations", &[n, index]),
    )
    .zero_sharing(ZeroSharing::PairwiseSeeds)
    .timeout(Duration::from_secs(60));
    (f, shares, network)
}

/// The commitment, every party's state and the proofs of `claims`, one per claim.
#[allow(clippy::type_complexity)] // the outputs of the parties, in order
fn commit_and_prove<P: BasePcs<K> + Sync>(
    ctx: &Context<K>,
    pcs: &P,
    shares: &[Vec<Ext<K>>],
    network: &MemoryConfig,
    claims: &[(Vec<Ext<K>>, Ext<K>)],
    index: u64,
) -> (
    Vec<Poly>,
    Vec<hachi_co_core::PartyState<K>>,
    Vec<hachi_co::EvalProof<K>>,
) {
    let runs = run_with(network, |p, comm| {
        let n = comm.n() as u64;
        let mut rng = PrgRandomness::new(seed("party/evaluations", &[n, index, p as u64]));
        let (u, state) = commit(ctx, comm, &mut rng, &shares[p - 1])?;
        let proofs = claims
            .iter()
            .enumerate()
            .map(|(k, (x, y))| {
                let tag = [n, index, p as u64, k as u64];
                let mut rng = PrgRandomness::new(seed("party/evaluations/eval", &tag));
                eval(ctx, pcs, comm, &mut rng, &state, &u, x, y)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok::<_, hachi_co::Error>((u, state, proofs))
    });
    let outputs: Vec<_> = runs
        .into_iter()
        .map(|run| run.expect("every party connects").output.unwrap())
        .collect();
    let (u, _, proofs) = outputs[0].clone();
    for (other_u, _, other_proofs) in &outputs {
        assert_eq!(*other_u, u, "every party has the same commitment");
        assert_eq!(*other_proofs, proofs, "every party has the same proofs");
    }
    let states = outputs.into_iter().map(|(_, state, _)| state).collect();
    (u, states, proofs)
}

fn several_evaluations<P: BasePcs<K> + Sync>(ctx: &Context<K>, pcs: &P, index: u64) {
    let field = ctx.field();
    let (f, shares, network) = instance(ctx, index);
    let mut rng = ChaCha8Rng::seed_from_u64(9000 + index);
    let mut claims: Vec<(Vec<Ext<K>>, Ext<K>)> = (0..3)
        .map(|_| {
            let x = common::random_vector(&mut rng, ctx.q(), ctx.layout.ell);
            let y = evaluate_by_monomials(field, &f, &x);
            (x, y)
        })
        .collect();
    claims.push(claims[0].clone());
    let (u, states, proofs) = commit_and_prove(ctx, pcs, &shares, &network, &claims, index);
    assert!(open(ctx, &states, &u, &f, None).unwrap().bit);
    for ((x, y), proof) in claims.iter().zip(&proofs) {
        verify(ctx, pcs, &u, x, y, proof).unwrap();
    }
    // Everything but rho^u is fresh: one claim proved twice has two different proofs.
    assert_ne!(proofs[0], proofs[3]);
    // Each proof is about its own claim.
    assert!(verify(ctx, pcs, &u, &claims[1].0, &claims[1].1, &proofs[0]).is_err());
}

#[test]
fn several_evaluations_of_one_commitment_verify() {
    for n in [1u64, 2, 3] {
        let ctx = context(n, PadRule::Field, 0);
        several_evaluations(&ctx, &clear_pcs(&ctx), n);
    }
}

#[test]
fn several_evaluations_of_one_commitment_verify_with_hachi() {
    for n in [1u64, 2] {
        let ctx = context(n, PadRule::Field, 1);
        let pcs = hachi(&ctx, n, PadRule::Field, 1);
        several_evaluations(&ctx, &pcs, 10 + n);
    }
}

/// A source serves each of its streams once, so a second Eval with one source stops at its first
/// draw instead of repeating the masks.
#[test]
fn a_reused_source_is_refused() {
    let ctx = context(1, PadRule::Field, 2);
    let pcs = clear_pcs(&ctx);
    let (f, shares, network) = instance(&ctx, 2);
    let x = common::random_vector(
        &mut ChaCha8Rng::seed_from_u64(9100),
        ctx.q(),
        ctx.layout.ell,
    );
    let y = evaluate_by_monomials(ctx.field(), &f, &x);
    let runs = run_with(&network, |_, comm| {
        let mut rng = PrgRandomness::new(seed("party/evaluations/reused", &[]));
        let (u, state) = commit(&ctx, comm, &mut rng, &shares[0]).unwrap();
        eval(&ctx, &pcs, comm, &mut rng, &state, &u, &x, &y).unwrap();
        eval(&ctx, &pcs, comm, &mut rng, &state, &u, &x, &y)
    });
    let output = runs.into_iter().next().unwrap().unwrap().output;
    let error = output.expect_err("the second Eval must fail");
    assert!(
        format!("{error:?}").contains("RandomnessReuse"),
        "{error:?}"
    );
}

/// One run for party $p$ with the private data of the sumcheck.
fn prove_with_private<P: BasePcs<K> + Sync>(
    ctx: &Context<K>,
    pcs: &P,
    index: u64,
) -> (Vec<Poly>, Vec<Ext<K>>, Ext<K>, Vec<EvalOutput<K>>) {
    let (f, shares, network) = instance(ctx, index);
    let field = ctx.field();
    let x = common::random_vector(
        &mut ChaCha8Rng::seed_from_u64(9200 + index),
        ctx.q(),
        ctx.layout.ell,
    );
    let y = evaluate_by_monomials(field, &f, &x);
    let runs = run_with(&network, |p, comm| {
        let n = comm.n() as u64;
        let mut rng = PrgRandomness::new(seed("party/evaluations", &[n, index, p as u64]));
        let (u, state) = commit(ctx, comm, &mut rng, &shares[p - 1]).unwrap();
        let mut ch = FsChallenger::new(EVAL_DOMAIN);
        let out = eval_with(ctx, pcs, comm, &mut rng, &mut ch, &state, &u, &x, &y).unwrap();
        (u, out)
    });
    let outputs: Vec<(Vec<Poly>, EvalOutput<K>)> = runs
        .into_iter()
        .map(|run| run.expect("every party connects").output)
        .collect();
    let u = outputs[0].0.clone();
    (
        u,
        x,
        y,
        outputs.into_iter().map(|(_, out)| out).collect::<Vec<_>>(),
    )
}

fn both_pads<P: BasePcs<K> + Sync>(ctx: &Context<K>, pcs: &P, words: &BasePcsWords, index: u64) {
    let l = &ctx.layout;
    let (u, x, y, outs) = prove_with_private(ctx, pcs, index);
    verify(ctx, pcs, &u, &x, &y, &outs[0].proof).unwrap();
    let at = pad_index(l);
    for out in &outs {
        assert_eq!(out.proof, outs[0].proof);
        let private = &out.sumcheck;
        match l.pad_rule {
            PadRule::Field => {
                assert_eq!(private.pad.map(|(entry, _)| entry), Some(at));
                assert_eq!(private.w_table[at], 0, "the pad is not in the table of Z_q");
                assert_ne!(private.w_field_table(ctx.field())[at].0, [0; K]);
            }
            PadRule::Residues => {
                assert_eq!(private.pad, None);
                assert!(
                    private.w_table[at..at + 2 * l.k].iter().any(|x| *x != 0),
                    "2k residues in row mu"
                );
            }
        }
    }
    assert_eq!(
        outs[0].proof.to_bytes().len(),
        encoded_len(l, words).unwrap()
    );
}

#[test]
fn the_residue_pad_is_refused_with_masks() {
    // That pad opens through the Z_q shortcut, whose partial evaluations reveal more than the value.
    // So its default choices have no masks, and asking for them is refused.
    let ctx = context(2, PadRule::Residues, 6);
    let setup = seed("setup/evaluations", &[2, pad_word(PadRule::Residues), 6]);
    let plain = HachiBase::default_choices(&ctx, &setup);
    assert_eq!(plain.hiding, None);
    let field_ctx = context(2, PadRule::Field, 6);
    assert!(
        HachiBase::default_choices(&field_ctx, &setup)
            .hiding
            .is_some()
    );
    let masked = HachiBase::<K>::new(
        ctx.q(),
        HachiChoices {
            hiding: HachiChoices::toy_hiding(ctx.layout.d).hiding,
            ..plain
        },
    )
    .unwrap();
    assert!(masked.words(&ctx.layout).is_err());
    assert!(masked.commit_multilinear(&[0; 1 << 8], &[1; 32]).is_err());
}

#[test]
fn both_pads_are_accepted_with_clear_pcs() {
    for n in [1u64, 2, 3] {
        for pad in [PadRule::Field, PadRule::Residues] {
            let ctx = context(n, pad, 3);
            both_pads(&ctx, &clear_pcs(&ctx), &clear_words(&ctx.layout), 20 + n);
        }
    }
}

#[test]
fn both_pads_are_accepted_with_hachi() {
    for n in [1u64, 2] {
        for pad in [PadRule::Field, PadRule::Residues] {
            let ctx = context(n, pad, 4);
            let pcs = hachi(&ctx, n, pad, 4);
            both_pads(&ctx, &pcs, &pcs.words(&ctx.layout).unwrap(), 30 + n);
        }
    }
}

/// The rule of the pad is part of the statement, and the opening of $\langle \widetilde w
/// \rangle\_p$ differs in kind: a proof under one rule is refused under the other.
#[test]
fn the_proof_is_bound_to_the_rule_of_the_pad() {
    let n = 2;
    let setup = seed("setup/evaluations/same", &[n]);
    let context = |pad| {
        let choices = Choices {
            pad,
            ..toy_choices(n)
        };
        hachi_co::setup(derive(&choices).unwrap().0, &setup).unwrap()
    };
    let (field_ctx, residues_ctx) = (context(PadRule::Field), context(PadRule::Residues));
    let (pcs_f, pcs_r) = (clear_pcs(&field_ctx), clear_pcs(&residues_ctx));
    let (u, x, y, outs) = prove_with_private(&field_ctx, &pcs_f, 40);
    verify(&field_ctx, &pcs_f, &u, &x, &y, &outs[0].proof).unwrap();
    assert!(verify(&residues_ctx, &pcs_r, &u, &x, &y, &outs[0].proof).is_err());
    let (u, x, y, outs) = prove_with_private(&residues_ctx, &pcs_r, 41);
    verify(&residues_ctx, &pcs_r, &u, &x, &y, &outs[0].proof).unwrap();
    assert!(verify(&field_ctx, &pcs_f, &u, &x, &y, &outs[0].proof).is_err());
}
