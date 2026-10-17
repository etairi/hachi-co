//! The limitation the crate documentation states ("What an accepted proof means in milestone
//! 1"), checked: without the base PCS, a proof for a **false** evaluation can be made from public
//! data alone, and the composed verifier accepts it. Its final claims are not evaluations of
//! committed polynomials; in milestone 3 the base-PCS commitments (absorbed before the
//! sumcheck's coins) and the openings of these claims must make this test fail, and it will have
//! to be turned around then.
//!
//! The forgery: $E(u)$ is $y'$ embedded in $\mathcal R\_q^H$, $a^\ast = 0$ and $\mathbf v = 0$;
//! the hypercube sums are 0; each round message is the constant polynomial with half of the
//! previous round's value (so that $H(0) + H(1)$ is right); the final values are
//! $\langle \widetilde w \rangle\_1(\mathbf a) = 0$ (the range product vanishes at 0) and
//! $g\_{0,1}(\mathbf a), g\_{1,1}(\mathbf a)$ solved from the final checks with $\rho^{-1}$.

mod common;

use common::{K, fresh_context, random_instance, seed};
use hachi_co::{EVAL_DOMAIN, EvalProof, absorb_statement, commit, verify};
use hachi_co_core::{Challenger, FsChallenger, Poly, PrgRandomness};
use hachi_co_mpc::Loopback;
use hachi_co_sumcheck::{
    FinalValues, LABEL_G0, LABEL_G1, LABEL_H0, LABEL_H1, RoundMessage, SumcheckTranscript,
    claimed_sum, eq_table,
};

#[test]
fn m1_proofs_without_the_base_pcs_are_forgeable() {
    let ctx = fresh_context("toy-n1", 9);
    let (l, field) = (&ctx.layout, ctx.field());
    // An honest commitment to a random f, and a false claim y' = f(x) + 1.
    let (f, x, y) = random_instance(&ctx, 9000);
    let mut comm = Loopback::new(ctx.q()).unwrap();
    let mut rng = PrgRandomness::new(seed("party/forgery", &[9]));
    let (u, _state) = commit(&ctx, &mut comm, &mut rng, &f).unwrap();
    let y_false = field.add(&y, &field.one());

    // The verifier's coins, derived exactly as the verifier will.
    let mut ch = FsChallenger::new(EVAL_DOMAIN);
    absorb_statement(&mut ch, &ctx, &u, &x, &y_false);
    let e_u = ctx.sub.embed(&y_false);
    Challenger::<K>::absorb_polys(
        &mut ch,
        hachi_co_conv::LABEL_E_U,
        core::slice::from_ref(&e_u),
    );
    let pv = hachi_co_conv::public_vectors(&ctx, &x).unwrap();
    let v = vec![Poly(vec![0; l.d]); l.n_d];
    let c = hachi_co_r2l::verifier_challenges(&ctx, &mut ch, 0, &v).unwrap();
    let rel = hachi_co_r2l::linear_relation(&ctx, &pv, &v, &c, &u).unwrap();
    let zero = field.zero();
    let (g0_sums, g1_sums) = (vec![zero; l.n], vec![zero; l.n]);
    ch.absorb_exts(LABEL_G0, &g0_sums);
    ch.absorb_exts(LABEL_G1, &g1_sums);
    let coins = ch
        .sumcheck_coins(field, l.ell_pp, l.log_n, l.log_nu)
        .unwrap();
    let eq1 = eq_table(field, &coins.tau1).unwrap();
    let a_claim = claimed_sum(&ctx, &eq1, &rel.y, &e_u, &coins.alpha).unwrap();
    let half = field.inv(&field.from_base(2)).unwrap();
    // H_{0,0} = rho * sum G_0 = 0 and H_{1,0} = a + rho * sum G_1 = a.
    let (mut prev0, mut prev1) = (zero, a_claim);
    let mut rounds = Vec::with_capacity(l.ell_pp);
    for r in 1..=l.ell_pp {
        let msg = RoundMessage {
            h0: vec![field.mul(&prev0, &half); l.d0 + 1],
            h1: vec![field.mul(&prev1, &half); l.d1 + 1],
        };
        ch.absorb_exts(LABEL_H0, &msg.h0);
        ch.absorb_exts(LABEL_H1, &msg.h1);
        ch.round_coin(field, r).unwrap();
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
    let forged = EvalProof {
        e_u,
        attempt: 0,
        v,
        sumcheck: SumcheckTranscript {
            g0_sums,
            g1_sums,
            rounds,
            finals,
        },
    };
    let claims = verify(&ctx, &u, &x, &y_false, &forged).expect("the forgery is accepted");
    assert_eq!(claims.claims.len(), 3);
    // The honest claim y is now rejected with this proof (Pi_conv's check).
    assert!(verify(&ctx, &u, &x, &y, &forged).is_err());
}
