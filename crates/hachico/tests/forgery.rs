//! The forgery against the protocol without a base PCS, turned around: with a base PCS it is
//! rejected.
//!
//! **The forging strategy.** Without a base PCS, a prover could make a proof of a **false**
//! evaluation $y' = f(\mathbf x) + 1$ from public data alone, and the composed verifier accepted
//! it: $E(u)$ is $y'$ embedded in $\mathcal R\_q^H$ (which passes $\Pi\_{\mathsf{conv}}$'s check
//! for $y'$), $a^\ast = 0$ and $\mathbf v = 0$; the hypercube sums are 0; each round message is
//! the constant polynomial with half of the previous round's value (so $H(0) + H(1)$ is right
//! and the value at the coin is known in advance); the final values are $\langle \widetilde w
//! \rangle\_1(\mathbf a) = 0$ (the range product vanishes at 0) and $g\_{0,1}(\mathbf a)$,
//! $g\_{1,1}(\mathbf a)$ solved from the two final checks with $\rho^{-1}$. It worked because the
//! final values were unproven claims, chosen after the last coin.
//!
//! **Why it now fails.** The final values must be openings of polynomials committed before the
//! coins: the commitments are absorbed into the transcript before $\alpha, \rho, \tau\_0,
//! \tau\_1$ and the round coins are drawn, and the verifier checks every opening against them.
//! The strategy needs $g\_{1,1}(\mathbf a) = H\_{1,\ell''}(a\_{\ell''}) / \rho$, a value fixed by the
//! coins, from a polynomial fixed before them. The forger can:
//! 1. commit to polynomials before the coins (here the zero polynomials, which fit the rest of
//!    the strategy: $\langle \widetilde w \rangle\_1 = 0$, and $G\_{0,1} = 0$ makes the solved
//!    $g\_{0,1}(\mathbf a)$ zero) and open them honestly: every round and final check of the
//!    sumcheck still passes, but the committed $g\_{1,1}$ opens to $0$, not to the solved value
//!    (equal only if the claimed sum $a$ is $0$, probability about $q^{-k}$), and the verifier
//!    rejects that opening;
//! 2. open another vector that takes the solved value: that opening does not open the
//!    commitment (with `ClearPcs`, its hash is not the commitment, which would be a SHA-256
//!    collision; with Hachi, its final witness does not recompute the commitment, or, with an
//!    iteration, its first sumcheck round fails under the verifier's coins), and the verifier
//!    rejects the same opening;
//! 3. commit to that vector instead: the commitment is absorbed before the coins, so every coin
//!    changes. Kept as they were, the round messages fail the first round check of $H\_1$ (its
//!    claimed sum $a$ depends on $\alpha$ and $\tau\_1$); forged again for the new coins, they
//!    pass, but the new solved value differs and the new commitment opens to the old one. A
//!    commitment consistent with its own coins is a fixed point of the hash, which Fiat-Shamir
//!    in the random-oracle model rules out (except with negligible probability per query).
//!
//! The tests run each variant at toy-n1 with Fiat-Shamir, against `ClearPcs` and against Hachi
//! (`HachiBase` with its default, automatic schedule, and with one iteration), and check the
//! verifier's reason. That the strategy fails follows from the binding of the base PCS and the
//! order of the transcript; it is evidence, not a proof of soundness (see the crate
//! documentation, "What an accepted proof means"). Hachi at the toy parameters is not binding
//! in any cryptographic sense (its matrices are far too small for Module-SIS): what the tests
//! show is that the verifier makes the checks that binding would make meaningful.

mod common;

use common::{K, clear_pcs, fresh_context, hachi_base, random_instance, seed, without_pcs};
use hachi_co::{
    BasePcs, ClaimedPoly, Context, EVAL_DOMAIN, EvalProof, PcsWords, ProofError, VerifyError,
    absorb_statement, commit, verify,
};
use hachi_co_core::{Challenger, Ext, FsChallenger, Poly, PrgRandomness};
use hachi_co_mpc::Loopback;
use hachi_co_sumcheck::{
    FinalValues, LABEL_COMMIT_G0, LABEL_COMMIT_G1, LABEL_COMMIT_W, LABEL_FINAL_G0, LABEL_FINAL_G1,
    LABEL_FINAL_W, LABEL_G0, LABEL_G1, LABEL_H0, LABEL_H1, RoundMessage, SumcheckTranscript,
    blinder_weights, claimed_sum, eq_table, opening_label,
};

/// The forged proof for given commitments, without openings, and what the forger learns while
/// making it: the point $\mathbf a$ and the contexts of the three openings (in the order $g\_0$,
/// $g\_1$, $\widetilde w$).
struct Forged {
    proof: EvalProof<K>,
    point: Vec<Ext<K>>,
    contexts: [[u8; 32]; 3],
}

/// The forging strategy of the module documentation for the false claim `y_false`, with the
/// base PCS `pcs` and its commitments `commitments` absorbed where the verifier absorbs them;
/// the coins are derived exactly as the verifier will derive them.
fn forge<P: BasePcs<K>>(
    ctx: &Context<K>,
    pcs: &P,
    u: &[Poly],
    x: &[Ext<K>],
    y_false: &Ext<K>,
    commitments: &PcsWords,
) -> Forged {
    let (l, field) = (&ctx.layout, ctx.field());
    let mut ch = FsChallenger::new(EVAL_DOMAIN);
    absorb_statement(&mut ch, ctx, &pcs.params_digest(), u, x, y_false);
    let e_u = ctx.sub.embed(y_false);
    Challenger::<K>::absorb_polys(
        &mut ch,
        hachi_co_conv::LABEL_E_U,
        core::slice::from_ref(&e_u),
    );
    let pv = hachi_co_conv::public_vectors(ctx, x).unwrap();
    let v = vec![Poly(vec![0; l.d]); l.n_d];
    let c = hachi_co_r2l::verifier_challenges(ctx, &mut ch, 0, &v).unwrap();
    let rel = hachi_co_r2l::linear_relation(ctx, &pv, &v, &c, u).unwrap();
    let zero = field.zero();
    let (g0_sums, g1_sums) = (vec![zero; l.n], vec![zero; l.n]);
    ch.absorb_exts(LABEL_G0, &g0_sums);
    ch.absorb_exts(LABEL_G1, &g1_sums);
    Challenger::<K>::absorb_words(&mut ch, LABEL_COMMIT_W, &commitments.w);
    Challenger::<K>::absorb_words(&mut ch, LABEL_COMMIT_G0, &commitments.g0);
    Challenger::<K>::absorb_words(&mut ch, LABEL_COMMIT_G1, &commitments.g1);
    let coins = ch
        .sumcheck_coins(field, l.ell_pp, l.log_n, l.log_nu)
        .unwrap();
    let eq1 = eq_table(field, &coins.tau1).unwrap();
    let a_claim = claimed_sum(ctx, &eq1, &rel.y, &e_u, &coins.alpha).unwrap();
    let half = field.inv(&field.from_base(2)).unwrap();
    // H_{0,0} = rho * sum G_0 = 0 and H_{1,0} = a + rho * sum G_1 = a.
    let (mut prev0, mut prev1) = (zero, a_claim);
    let mut rounds = Vec::with_capacity(l.ell_pp);
    let mut point = Vec::with_capacity(l.ell_pp);
    for r in 1..=l.ell_pp {
        let msg = RoundMessage {
            h0: vec![field.mul(&prev0, &half); l.d0 + 1],
            h1: vec![field.mul(&prev1, &half); l.d1 + 1],
        };
        ch.absorb_exts(LABEL_H0, &msg.h0);
        ch.absorb_exts(LABEL_H1, &msg.h1);
        point.push(ch.round_coin(field, r).unwrap());
        // A constant polynomial takes its value at the coin.
        (prev0, prev1) = (msg.h0[0], msg.h1[0]);
        rounds.push(msg);
    }
    let rho_inv = field.inv(&coins.rho).expect("rho = 0 has probability q^-k");
    let finals = FinalValues {
        w: vec![zero; l.n],
        g0: vec![field.mul(&prev0, &rho_inv)],
        g1: vec![field.mul(&prev1, &rho_inv)],
    };
    ch.absorb_exts(LABEL_FINAL_W, &finals.w);
    ch.absorb_exts(LABEL_FINAL_G0, &finals.g0);
    ch.absorb_exts(LABEL_FINAL_G1, &finals.g1);
    let contexts =
        ClaimedPoly::ORDER.map(|poly| Challenger::<K>::context(&mut ch, &opening_label(1, poly)));
    Forged {
        proof: EvalProof {
            e_u,
            attempt: 0,
            v,
            sumcheck: SumcheckTranscript {
                g0_sums,
                g1_sums,
                commitments: commitments.clone(),
                rounds,
                finals,
                openings: PcsWords::default(),
            },
        },
        point,
        contexts,
    }
}

/// The commitments (as words) to $\langle \widetilde w \rangle\_1$ by its coefficients and to the
/// two blinder vectors, and the provers' states.
fn commit_all<P: BasePcs<K>>(
    pcs: &P,
    w: &[u64],
    g0: &[Ext<K>],
    g1: &[Ext<K>],
) -> (PcsWords, [P::State; 3]) {
    let (cw, sw) = pcs.commit_multilinear(w).unwrap();
    let (c0, s0) = pcs.commit_vector(g0).unwrap();
    let (c1, s1) = pcs.commit_vector(g1).unwrap();
    let words = PcsWords {
        w: vec![P::commitment_to_words(&cw)],
        g0: vec![P::commitment_to_words(&c0)],
        g1: vec![P::commitment_to_words(&c1)],
    };
    (words, [sw, s0, s1])
}

/// The honest openings of the states (w, g0, g1) at the forger's point, under its contexts.
fn open_all<P: BasePcs<K>>(
    ctx: &Context<K>,
    pcs: &P,
    forged: &Forged,
    states: &[P::State; 3],
) -> PcsWords {
    let (l, field) = (&ctx.layout, ctx.field());
    let [sw, s0, s1] = states;
    let [ctx_g0, ctx_g1, ctx_w] = &forged.contexts;
    let (_, pw) = pcs.prove_multilinear(sw, &forged.point, ctx_w).unwrap();
    let w0 = blinder_weights(field, &forged.point, l.d0);
    let w1 = blinder_weights(field, &forged.point, l.d1);
    let (_, p0) = pcs.prove_inner_product(s0, &w0, ctx_g0).unwrap();
    let (_, p1) = pcs.prove_inner_product(s1, &w1, ctx_g1).unwrap();
    PcsWords {
        w: vec![P::proof_to_words(&pw)],
        g0: vec![P::proof_to_words(&p0)],
        g1: vec![P::proof_to_words(&p1)],
    }
}

/// The verifier's verdict is a rejected opening of `poly` of party 1 whose reason contains
/// `reason`.
fn rejected_opening(result: Result<(), VerifyError>, poly: ClaimedPoly, reason: &str) -> bool {
    matches!(
        result,
        Err(VerifyError::Sumcheck(hachi_co_sumcheck::VerifyError::Opening {
            party: 1,
            poly: got,
            reason: why,
        })) if got == poly && why.contains(reason)
    )
}

/// The verifier's reasons for the two kinds of rejected openings, which depend on the base PCS:
/// an honest opening of a polynomial that does not take the claimed value, and an opening of
/// another polynomial than the committed one.
struct Reasons {
    wrong_value: &'static str,
    other_polynomial: &'static str,
}

/// Every variant of the module documentation against the base PCS `pcs`, at the context `ctx`.
fn forgery_is_rejected<P: BasePcs<K>>(ctx: &Context<K>, pcs: &P, reasons: &Reasons) {
    let (l, field) = (&ctx.layout, ctx.field());
    // An honest commitment to a random f, and a false claim y' = f(x) + 1.
    let (f, x, y) = random_instance(ctx, 9000);
    let mut comm = Loopback::new(ctx.q()).unwrap();
    let mut rng = PrgRandomness::new(seed("party/forgery", &[9]));
    let (u, _state) = commit(ctx, &mut comm, &mut rng, &f).unwrap();
    let y_false = field.add(&y, &field.one());
    let check = |proof: &EvalProof<K>| verify(ctx, pcs, &u, &x, &y_false, proof);
    let zero = field.zero();
    let (len0, len1) = (1 + l.ell_pp * l.d0, 1 + l.ell_pp * l.d1);

    // Variant 1: zero polynomials committed before the coins, opened honestly. The sumcheck's
    // own checks pass (the verdict would be Rejected otherwise); the opening of g_{1,1} does not.
    let (commitments, states) = commit_all(
        pcs,
        &vec![0; 1 << l.ell_pp],
        &vec![zero; len0],
        &vec![zero; len1],
    );
    let mut forged = forge(ctx, pcs, &u, &x, &y_false, &commitments);
    let solved_g1 = forged.proof.sumcheck.finals.g1[0];
    assert_ne!(solved_g1, zero, "the solved value is a / (2^l'' rho)");
    assert_eq!(forged.proof.sumcheck.finals.g0[0], zero);
    forged.proof.sumcheck.openings = open_all(ctx, pcs, &forged, &states);
    assert!(rejected_opening(
        check(&forged.proof),
        ClaimedPoly::G1,
        reasons.wrong_value
    ));
    // Without commitments and openings (the proof of the protocol without a base PCS), the
    // proof is malformed.
    assert!(matches!(
        check(&without_pcs(&forged.proof)),
        Err(VerifyError::Proof(ProofError::Shape { .. }))
    ));
    // The honest claim y is rejected with this proof (Pi_conv's check).
    assert!(matches!(
        verify(ctx, pcs, &u, &x, &y, &forged.proof),
        Err(VerifyError::Conv(_))
    ));

    // Variant 2: open, under the same commitment, a vector that takes the solved value at a
    // (s_0 = the value, every other coefficient 0).
    let mut g1_solved = vec![zero; len1];
    g1_solved[0] = solved_g1;
    let (_, state_solved) = pcs.commit_vector(&g1_solved).unwrap();
    let weights1 = blinder_weights(field, &forged.point, l.d1);
    let (value, proof) = pcs
        .prove_inner_product(&state_solved, &weights1, &forged.contexts[1])
        .unwrap();
    assert_eq!(value, solved_g1, "the vector takes the solved value");
    let mut variant2 = forged.proof.clone();
    variant2.sumcheck.openings.g1 = vec![P::proof_to_words(&proof)];
    assert!(rejected_opening(
        check(&variant2),
        ClaimedPoly::G1,
        reasons.other_polynomial
    ));

    // Variant 3: commit to that vector instead. Keeping the messages, the coins change under
    // them: the first round check of H_1 fails.
    let (commitments3, states3) =
        commit_all(pcs, &vec![0; 1 << l.ell_pp], &vec![zero; len0], &g1_solved);
    let mut variant3 = forged.proof.clone();
    variant3.sumcheck.commitments = commitments3.clone();
    variant3.sumcheck.openings.g1 = vec![P::proof_to_words(&proof)];
    match check(&variant3) {
        Err(VerifyError::Sumcheck(hachi_co_sumcheck::VerifyError::Rejected { failing })) => {
            assert!(failing.contains(&"round_01_H1".to_owned()), "{failing:?}");
        }
        other => panic!("variant 3 with the old messages: {other:?}"),
    }
    // Forged again for the new coins: the round checks pass, but the new solved value is not the
    // value the new commitment opens to.
    let mut reforged = forge(ctx, pcs, &u, &x, &y_false, &commitments3);
    assert_ne!(reforged.proof.sumcheck.finals.g1[0], solved_g1);
    reforged.proof.sumcheck.openings = open_all(ctx, pcs, &reforged, &states3);
    assert!(rejected_opening(
        check(&reforged.proof),
        ClaimedPoly::G1,
        reasons.wrong_value
    ));
}

#[test]
fn the_forgery_is_rejected_with_clear_pcs() {
    let ctx = fresh_context("toy-n1", 9);
    let reasons = Reasons {
        wrong_value: "does not take the claimed value",
        other_polynomial: "does not match the commitment",
    };
    forgery_is_rejected(&ctx, &clear_pcs(&ctx), &reasons);
}

#[test]
fn the_forgery_is_rejected_with_hachi() {
    let ctx = fresh_context("toy-n1", 9);
    let pcs = hachi_base(&ctx, "toy-n1", 9, None);
    // The automatic schedule runs no iteration here: the final witness is the whole vector.
    let reasons = Reasons {
        wrong_value: "the trace of the inner product",
        other_polynomial: "the final witness does not open the commitment",
    };
    forgery_is_rejected(&ctx, &pcs, &reasons);
}

#[test]
fn the_forgery_is_rejected_with_hachi_after_an_iteration() {
    let ctx = fresh_context("toy-n1", 9);
    let pcs = hachi_base(&ctx, "toy-n1", 9, Some(1));
    // With an iteration, the other vector's proof was made on the transcript of another
    // commitment: under the verifier's coins, the first round of the iteration's linear sumcheck
    // fails (its claimed sum depends on the coins; the range sumcheck's is 0 for any coins).
    let reasons = Reasons {
        wrong_value: "the trace of the inner product",
        other_polynomial: "linear sumcheck, round 1",
    };
    forgery_is_rejected(&ctx, &pcs, &reasons);
}
