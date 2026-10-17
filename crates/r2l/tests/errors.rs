//! The party-local functions, prover 1's steps, the share assembly and the relation checkers
//! refuse malformed inputs with errors (never panics), at the exact boundaries where the inputs
//! become invalid.

mod common;

use common::{load, u_com, u_share};
use hachi_co_core::{Fixture, LinearShare, Poly, ReplayChallenger, ReplayRandomness, short};
use hachi_co_r2l::{
    Error, LinearRelation, aggregate, block_w, check_challenges, d_slice, fah_d_input, fold,
    linear_relation, mask_images, norm_check, owned_w, relation_holds, residual, respond,
    response_len, sample_attempt, sum_shares, y_share, z_hat, z_prime_share,
};
use hachi_co_sampling::Error as SamplingError;

fn shape<T: core::fmt::Debug>(r: hachi_co_r2l::Result<T>) -> bool {
    matches!(r, Err(Error::Shape { .. }))
}

#[test]
fn party_local_functions_refuse_malformed_inputs() {
    let (fx, ctx, pv, states) = load("toy-n3");
    let l = &ctx.layout;
    let q = ctx.q();
    let n_len = response_len(&ctx);
    let mut rng = ReplayRandomness::new(&fx, 2).unwrap();
    let secrets = sample_attempt(&ctx, &pv, &mut rng, 0).unwrap();
    let w2 = owned_w(&ctx, &pv, &states[1]).unwrap();
    d_slice(&ctx, 2, &w2, &secrets).unwrap();

    // block_w / owned_w: shapes of the block and of the public vectors.
    let mut block = states[1].blocks[0].clone();
    block.s.pop();
    assert!(matches!(block_w(&ctx, &pv, &block), Err(Error::Core(_))));
    let mut pv_bad = pv.clone();
    pv_bad.a_ring.pop();
    assert!(matches!(
        owned_w(&ctx, &pv_bad, &states[1]),
        Err(Error::Core(_))
    ));

    // mask_images: length, and the support [-gamma, gamma] at its boundary.
    assert!(shape(mask_images(&ctx, &pv, &secrets.mask[1..])));
    let gamma = ctx.gamma() as i64;
    let mut mask = vec![0i64; n_len];
    mask[n_len - 1] = gamma;
    mask[0] = -gamma;
    mask_images(&ctx, &pv, &mask).unwrap();
    for bad in [gamma + 1, -gamma - 1, i64::MIN, i64::MAX] {
        mask[7] = bad;
        assert_eq!(
            mask_images(&ctx, &pv, &mask).map(|_| ()),
            Err(Error::Sampling(SamplingError::MaskOutOfRange))
        );
    }

    // d_slice: the party, the blocks it holds, their order and digits.
    for p in [0, l.n + 1] {
        assert_eq!(
            d_slice(&ctx, p, &w2, &secrets),
            Err(Error::InvalidParty { party: p, n: l.n })
        );
    }
    assert!(
        shape(d_slice(&ctx, 1, &w2, &secrets)),
        "p = 1 holds blocks 0, 1"
    );
    assert!(shape(d_slice(&ctx, 2, &w2[..1], &secrets)));
    let mut swapped = w2.clone();
    swapped.swap(0, 1);
    assert!(shape(d_slice(&ctx, 2, &swapped, &secrets)));
    let mut digits = w2.clone();
    digits[1].w_hat.pop();
    assert!(shape(d_slice(&ctx, 2, &digits, &secrets)));
    let mut digits = w2.clone();
    digits[0].w_hat[0].0[0] = q;
    assert!(matches!(
        d_slice(&ctx, 2, &digits, &secrets),
        Err(Error::NonCanonical { .. })
    ));
    assert!(shape(fah_d_input(&ctx, &vec![ctx.ring.zero(); l.t - 1])));

    // check_challenges and fold: count, length, the l1 ball at its boundary.
    let c = fx.int_rows("/ring_to_linear/attempts/0/c").unwrap();
    check_challenges(&ctx, &c).unwrap();
    assert!(shape(check_challenges(&ctx, &c[1..])));
    assert!(shape(fold(&ctx, &states[0], &c[1..])));
    let mut c_bad = c.clone();
    c_bad[5] = vec![0; l.d];
    c_bad[5][0] = ctx.omega() as i64;
    check_challenges(&ctx, &c_bad).unwrap();
    c_bad[5][l.d - 1] = 1;
    assert!(matches!(
        fold(&ctx, &states[0], &c_bad),
        Err(Error::ChallengeNotInC { index: 5, .. })
    ));
    // A non-canonical digit, and a state of the wrong shape.
    let mut st = states[0].clone();
    st.blocks[1].s[3].0[2] = q + 5;
    assert!(matches!(
        fold(&ctx, &st, &c),
        Err(Error::NonCanonical {
            what: "s_i (digits)",
            ..
        })
    ));
    let mut st = states[0].clone();
    st.blocks.pop();
    assert!(matches!(fold(&ctx, &st, &c), Err(Error::Core(_))));
    // Digits far outside the digit set give a fold above beta_1, which Rej refuses (in a block
    // whose challenge is not zero).
    let mut st = states[0].clone();
    let j = st
        .blocks
        .iter()
        .position(|b| c[b.index].iter().any(|x| *x != 0))
        .expect("a block of prover 1 with a nonzero challenge");
    st.blocks[j].s[0].0[0] = q / 2;
    let big = fold(&ctx, &st, &c).unwrap();
    assert!(short::inf_norm(&big) > ctx.beta1());
    assert_eq!(
        respond(&ctx, &big, &secrets.mask),
        Err(Error::Sampling(SamplingError::SecretTooLarge))
    );
}

#[test]
fn prover1_steps_refuse_malformed_responses() {
    let (_, ctx, _, _) = load("toy-n2");
    let n_len = response_len(&ctx);
    let ok = vec![vec![1i64; n_len], vec![-1i64; n_len]];
    assert_eq!(aggregate(&ctx, &ok).unwrap(), vec![0i64; n_len]);
    assert!(shape(aggregate(&ctx, &ok[..1])));
    let mut more = ok.clone();
    more.push(vec![0; n_len]);
    assert!(shape(aggregate(&ctx, &more)));
    let mut short_resp = ok.clone();
    short_resp[1].pop();
    assert!(shape(aggregate(&ctx, &short_resp)));
    let huge = vec![vec![i64::MAX; n_len], vec![1; n_len]];
    assert_eq!(
        aggregate(&ctx, &huge),
        Err(Error::Overflow("z = sum of the responses"))
    );
    // z_hat: the window of tau digits at its boundary, which contains [-beta, beta].
    let l = &ctx.layout;
    let b = ctx.b() as i64;
    let s_tau = (0..l.tau).fold(0i64, |s, _| s * b + 1);
    let (lo, hi) = (-(b / 2) * s_tau, ((b - 1) / 2) * s_tau);
    assert!(hi >= ctx.beta() as i64 && lo <= -(ctx.beta() as i64));
    let mut z = vec![0i64; n_len];
    z[0] = hi;
    z[1] = lo;
    // The window is wider than [-beta, beta], so these pass J^-1 but not the norm check.
    assert!(matches!(norm_check(&ctx, &z), Err(Error::NormCheck { .. })));
    let digits = z_hat(&ctx, &z).unwrap();
    assert_eq!(digits.len(), l.width_a * l.tau);
    for bad in [hi + 1, lo - 1] {
        z[0] = bad;
        assert!(matches!(
            z_hat(&ctx, &z),
            Err(Error::Math(hachi_co_math::Error::NotRepresentable { .. }))
        ));
    }
    assert!(shape(z_hat(&ctx, &z[1..])));
    assert!(shape(respond(&ctx, &z[1..], &z)));
}

#[test]
fn share_assembly_and_relation_checkers_refuse_malformed_inputs() {
    let name = Fixture::TOY[1];
    let (fx, ctx, pv, states) = load(name);
    let l = &ctx.layout;
    let q = ctx.q();
    let a = fx.usize("/ring_to_linear/accepted_attempt").unwrap() as u64;
    let mut rngs: Vec<_> = (1..=l.n)
        .map(|p| ReplayRandomness::new(&fx, p).unwrap())
        .collect();
    let secrets: Vec<_> = rngs
        .iter_mut()
        .map(|rng| sample_attempt(&ctx, &pv, rng, a).unwrap())
        .collect();
    let w: Vec<_> = states
        .iter()
        .map(|st| owned_w(&ctx, &pv, st).unwrap())
        .collect();
    let z_prime_rec = fx.polys("/ring_to_linear/z_prime").unwrap();
    let z_hat_rec = &z_prime_rec[l.z_hat_range()];
    let us = |p| u_share(&fx, p);
    z_prime_share(
        &ctx,
        &states[0],
        &w[0],
        &secrets[0],
        Some(z_hat_rec),
        &us(1),
    )
    .unwrap();
    z_prime_share(&ctx, &states[1], &w[1], &secrets[1], None, &us(2)).unwrap();
    // z_hat exactly at prover 1, with the right length and canonical residues.
    assert!(shape(z_prime_share(
        &ctx,
        &states[0],
        &w[0],
        &secrets[0],
        None,
        &us(1)
    )));
    assert!(shape(z_prime_share(
        &ctx,
        &states[1],
        &w[1],
        &secrets[1],
        Some(z_hat_rec),
        &us(2)
    )));
    assert!(shape(z_prime_share(
        &ctx,
        &states[0],
        &w[0],
        &secrets[0],
        Some(&z_hat_rec[1..]),
        &us(1)
    )));
    let mut zh = z_hat_rec.to_vec();
    zh[0].0[0] = q;
    assert!(matches!(
        z_prime_share(&ctx, &states[0], &w[0], &secrets[0], Some(&zh), &us(1)),
        Err(Error::NonCanonical { .. })
    ));
    // The blocks of w and of the state must be the party's; <u>_p must be a ring element.
    assert!(shape(z_prime_share(
        &ctx,
        &states[1],
        &w[0],
        &secrets[1],
        None,
        &us(2)
    )));
    assert!(shape(z_prime_share(
        &ctx,
        &states[1],
        &w[1],
        &secrets[1],
        None,
        &Poly(vec![0; l.d + 1])
    )));
    let mut st = states[1].clone();
    st.party = 1;
    assert!(matches!(
        z_prime_share(&ctx, &st, &w[1], &secrets[1], None, &us(2)),
        Err(Error::Core(_))
    ));

    // y_share: the party, and v and u for every party.
    let v = fx
        .polys(&format!("/ring_to_linear/attempts/{a}/v"))
        .unwrap();
    let u = u_com(&fx);
    assert_eq!(
        y_share(&ctx, 2, &v, &u).unwrap(),
        vec![ctx.ring.zero(); l.n_prime]
    );
    assert_eq!(
        y_share(&ctx, 1, &v, &u).unwrap(),
        fx.polys("/ring_to_linear/y").unwrap()
    );
    for p in [0, l.n + 1] {
        assert_eq!(
            y_share(&ctx, p, &v, &u),
            Err(Error::InvalidParty { party: p, n: l.n })
        );
    }
    assert!(shape(y_share(&ctx, 2, &v[..0], &u)));
    assert!(shape(y_share(&ctx, 1, &v, &u[..0])));

    // The relation checkers.
    let c = fx
        .int_rows(&format!("/ring_to_linear/attempts/{a}/c"))
        .unwrap();
    let rel = linear_relation(&ctx, &pv, &v, &c, &u).unwrap();
    assert!(relation_holds(&ctx, &rel, &z_prime_rec).unwrap());
    assert!(shape(residual(&ctx, &rel, &z_prime_rec[1..])));
    let mut rel_bad = rel.clone();
    rel_bad.m[2].pop();
    assert!(shape(relation_holds(&ctx, &rel_bad, &z_prime_rec)));
    let rel_bad = LinearRelation {
        m: rel.m[1..].to_vec(),
        y: rel.y.clone(),
    };
    assert!(shape(residual(&ctx, &rel_bad, &z_prime_rec)));
    let rel_bad = LinearRelation {
        m: rel.m.clone(),
        y: rel.y[1..].to_vec(),
    };
    assert!(shape(residual(&ctx, &rel_bad, &z_prime_rec)));
    assert!(shape(sum_shares(&ctx, &[])));
    let mut bad_share = LinearShare::zero(l);
    bad_share.y.pop();
    assert!(matches!(
        sum_shares(&ctx, &[bad_share]),
        Err(Error::Core(_))
    ));
    // The replayed challenger refuses an attempt the oracle did not run.
    let mut ch = ReplayChallenger::new(&fx).unwrap();
    assert!(matches!(
        hachi_co_r2l::verifier_challenges(&ctx, &mut ch, a + 1, &v),
        Err(Error::Core(hachi_co_core::Error::Replay(_)))
    ));
}
