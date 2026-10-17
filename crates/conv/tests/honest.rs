//! Fresh honest runs of $\Pi\_{\mathsf{conv}}$ (random $f$ and $\mathbf x$, fresh public
//! parameters, the parties' resharing from `PrgRandomness`): at toy-n1 on the loopback and at
//! toy-n2 and toy-n3 on the in-memory network, V's check accepts $y = f(\mathbf x)$ and the
//! relation checker holds; tampered witnesses break the relation where they should. Seeds are
//! recorded in each test.
mod common;

use common::{K, additive_shares, random_vector, run_parties, state_from_share};
use hachi_co_conv::{
    VerifyError, check, check_relation, evaluate, pi_conv, public_vectors, u_share, verify,
};
use hachi_co_core::{Context, PartyRandomness, PrgRandomness};
use hachi_co_math::Ext;
use hachi_co_mpc::Loopback;
use hachi_co_params::{derive, presets::toy_choices};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn context(n: u64, seed: u64) -> Context<K> {
    let (params, _) = derive(&toy_choices(n)).unwrap();
    let mut setup_seed = [0u8; 32];
    setup_seed[..8].copy_from_slice(&seed.to_le_bytes());
    setup_seed[8..18].copy_from_slice(b"conv-setup");
    Context::setup(params, &setup_seed).unwrap()
}

fn party_seed(seed: u64, p: usize) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&seed.to_le_bytes());
    out[8..16].copy_from_slice(&(p as u64).to_le_bytes());
    out[16..26].copy_from_slice(b"conv-party");
    out
}

/// The parties' shares after Commit's resharing with `PrgRandomness` (AS.Share of every initial
/// share, then the local sums), computed here for all parties.
fn reshared(ctx: &Context<K>, seed: u64, f_init: &[Vec<Ext<K>>]) -> Vec<Vec<Ext<K>>> {
    let (n, field) = (f_init.len(), ctx.field());
    let sent: Vec<Vec<Vec<Ext<K>>>> = f_init
        .iter()
        .enumerate()
        .map(|(i, fi)| {
            let mut prg = PrgRandomness::new(party_seed(seed, i + 1));
            prg.reshare(field, fi, n).unwrap()
        })
        .collect();
    (0..n)
        .map(|p| {
            (0..fi_len(f_init))
                .map(|t| (0..n).fold(field.zero(), |acc, i| field.add(&acc, &sent[i][p][t])))
                .collect()
        })
        .collect()
}

fn fi_len(f_init: &[Vec<Ext<K>>]) -> usize {
    f_init[0].len()
}

#[test]
fn fresh_runs_at_toy_n1_pass_the_check_on_the_loopback() {
    for seed in 0..8u64 {
        let ctx = context(1, seed);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let f = random_vector(&mut rng, ctx.q(), 1 << ctx.layout.ell);
        let x = random_vector(&mut rng, ctx.q(), ctx.layout.ell);
        let y = evaluate(&ctx, &f, &x).unwrap();
        let share = reshared(&ctx, seed, core::slice::from_ref(&f)).remove(0);
        assert_eq!(share, f, "n = 1: the resharing is the identity");
        let st = state_from_share(&ctx, 1, share);
        let mut comm = Loopback::new(ctx.q()).unwrap();
        let out = pi_conv(&ctx, &mut comm, &st, &x).unwrap();
        assert_eq!(check(&ctx, &y, &out.e_u), Ok(()), "seed {seed}");
        let rel = check_relation(&ctx, &x, &y, &[st], &out.u_share).unwrap();
        assert!(rel.holds(), "seed {seed}: {rel:?}");
    }
}

#[test]
fn fresh_runs_at_toy_n2_and_n3_pass_the_check_on_the_network() {
    for n in [2u64, 3] {
        for seed in 0..4u64 {
            let seed = 200 + 10 * n + seed;
            let ctx = context(n, seed);
            let (n, q) = (n as usize, ctx.q());
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let f = random_vector(&mut rng, q, 1 << ctx.layout.ell);
            let x = random_vector(&mut rng, q, ctx.layout.ell);
            let y = evaluate(&ctx, &f, &x).unwrap();
            let f_init = additive_shares(&ctx, &mut rng, &f, n);
            let states: Vec<_> = reshared(&ctx, seed, &f_init)
                .into_iter()
                .enumerate()
                .map(|(i, share)| state_from_share(&ctx, i + 1, share))
                .collect();
            let outputs = run_parties(n, q, |p, comm| {
                pi_conv(&ctx, comm, &states[p - 1], &x).unwrap()
            });
            let e_u = &outputs[0].e_u;
            assert!(outputs.iter().all(|o| o.e_u == *e_u));
            assert!(verify(&ctx, &y, e_u), "n = {n}, seed {seed}");
            let ring = &ctx.ring;
            let u = outputs
                .iter()
                .fold(ring.zero(), |acc, o| ring.add(&acc, &o.u_share).unwrap());
            let rel = check_relation(&ctx, &x, &y, &states, &u).unwrap();
            assert!(rel.holds(), "n = {n}, seed {seed}: {rel:?}");
            // A wrong claim fails the check and the input relation only.
            let y_bad = ctx.field().add(&y, &ctx.field().one());
            assert_eq!(check(&ctx, &y_bad, e_u), Err(VerifyError::Mismatch));
            let rel = check_relation(&ctx, &x, &y_bad, &states, &u).unwrap();
            assert_eq!((rel.input, rel.output, rel.link), (false, true, true));
        }
    }
}

/// A tampered block of $\mathbf f'$: with the shares of $u$ recomputed from it, the output
/// relation holds but $E(u) \ne f(\mathbf x)$ (the link fails, and so does V's check); with the
/// old $u$ the output relation fails.
#[test]
fn a_tampered_witness_breaks_the_link() {
    let seed = 300u64;
    let ctx = context(2, seed);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let q = ctx.q();
    let f = random_vector(&mut rng, q, 1 << ctx.layout.ell);
    let x = random_vector(&mut rng, q, ctx.layout.ell);
    let y = evaluate(&ctx, &f, &x).unwrap();
    let f_init = additive_shares(&ctx, &mut rng, &f, 2);
    let mut states: Vec<_> = f_init
        .into_iter()
        .enumerate()
        .map(|(i, share)| state_from_share(&ctx, i + 1, share))
        .collect();
    let pv = public_vectors(&ctx, &x).unwrap();
    let ring = &ctx.ring;
    let sum_u = |sts: &[hachi_co_core::PartyState<K>]| {
        sts.iter().fold(ring.zero(), |acc, st| {
            ring.add(&acc, &u_share(&ctx, st, &pv).unwrap()).unwrap()
        })
    };
    let u_old = sum_u(&states);
    assert!(
        check_relation(&ctx, &x, &y, &states, &u_old)
            .unwrap()
            .holds()
    );
    let coefficient = &mut states[1].blocks[0].f_prime[0].0[rng.random_range(0..ctx.layout.d)];
    *coefficient = (*coefficient + 1) % q;
    let u_new = sum_u(&states);
    let rel = check_relation(&ctx, &x, &y, &states, &u_new).unwrap();
    assert_eq!((rel.input, rel.output, rel.link), (true, true, false));
    let e_u = hachi_co_conv::e_map(&ctx, &u_new, &pv.v_point).unwrap();
    assert_eq!(check(&ctx, &y, &e_u), Err(VerifyError::Mismatch));
    let rel = check_relation(&ctx, &x, &y, &states, &u_old).unwrap();
    assert_eq!((rel.input, rel.output, rel.link), (true, false, true));
}

#[test]
fn the_driver_refuses_a_mismatched_network_or_state() {
    let ctx = context(2, 400);
    let mut rng = ChaCha8Rng::seed_from_u64(400);
    let q = ctx.q();
    let f = random_vector(&mut rng, q, 1 << ctx.layout.ell);
    let x = random_vector(&mut rng, q, ctx.layout.ell);
    let st = state_from_share(&ctx, 1, f);
    // A loopback is a one-party network; the context has n = 2.
    let mut comm = Loopback::new(q).unwrap();
    assert!(matches!(
        pi_conv(&ctx, &mut comm, &st, &x),
        Err(hachi_co_conv::Error::Network { what: "n", .. })
    ));
    let ctx1 = context(1, 400);
    let mut comm = Loopback::new(ctx.q()).unwrap();
    let st1 = state_from_share(&ctx1, 1, random_vector(&mut rng, ctx1.q(), 64));
    assert!(matches!(
        pi_conv(&ctx1, &mut comm, &st1, &x),
        Err(hachi_co_conv::Error::Network { what: "q", .. })
    ));
    let mut st_bad = st1.clone();
    st_bad.party = 2;
    let mut comm = Loopback::new(ctx1.q()).unwrap();
    assert!(matches!(
        pi_conv(&ctx1, &mut comm, &st_bad, &x),
        Err(hachi_co_conv::Error::InvalidParty { .. })
    ));
    let mut st_short = st1.clone();
    st_short.blocks.pop();
    assert!(matches!(
        pi_conv(&ctx1, &mut comm, &st_short, &x),
        Err(hachi_co_conv::Error::Core(_))
    ));
}
