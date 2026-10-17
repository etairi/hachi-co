//! Honest Commit runs with fresh randomness (`PrgRandomness`) and fresh public parameters
//! (`Context::setup`): at toy-n1 on the loopback, and at toy-n2 and toy-n3 on the in-memory
//! network; every run passes Open with $c^\*$. Seeds are recorded in each test (inputs from
//! `ChaCha8Rng::seed_from_u64(seed)`, party seeds and setup seeds derived from it).
mod common;

use common::{K, run_parties};
use hachi_co_commit::{commit, f_hat, open, pconv};
use hachi_co_core::{Context, Error as CoreError, PrgRandomness};
use hachi_co_math::Ext;
use hachi_co_mpc::Loopback;
use hachi_co_params::{derive, presets::toy_choices};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn context(n: u64, seed: u64) -> Context<K> {
    let (params, _) = derive(&toy_choices(n)).unwrap();
    let mut setup_seed = [0u8; 32];
    setup_seed[..8].copy_from_slice(&seed.to_le_bytes());
    setup_seed[8..20].copy_from_slice(b"commit-setup");
    Context::setup(params, &setup_seed).unwrap()
}

fn party_seed(seed: u64, p: usize) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[..8].copy_from_slice(&seed.to_le_bytes());
    out[8..16].copy_from_slice(&(p as u64).to_le_bytes());
    out[16..28].copy_from_slice(b"commit-party");
    out
}

fn random_vector(rng: &mut ChaCha8Rng, q: u64, len: usize) -> Vec<Ext<K>> {
    (0..len)
        .map(|_| Ext(core::array::from_fn(|_| rng.random_range(0..q))))
        .collect()
}

#[test]
fn fresh_runs_at_toy_n1_pass_open_on_the_loopback() {
    for seed in 0..8u64 {
        let ctx = context(1, seed);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let f = random_vector(&mut rng, ctx.q(), 1 << ctx.layout.ell);
        let mut comm = Loopback::new(ctx.q()).unwrap();
        let mut prg = PrgRandomness::new(party_seed(seed, 1));
        let (u, st) = commit(&ctx, &mut comm, &mut prg, &f).unwrap();
        // n = 1: the share is f itself, and the blocks are the global PConv of f.
        assert_eq!(st.share, f);
        let (_, f_prime) = pconv(&ctx, core::slice::from_ref(&f)).unwrap();
        let bl = ctx.layout.block_len;
        for block in &st.blocks {
            assert_eq!(
                block.f_prime,
                f_prime[block.index * bl..(block.index + 1) * bl]
            );
        }
        let report = open(&ctx, &[st], &u, &f, None).unwrap();
        assert!(report.bit, "seed {seed}: {:?}", report.checks.failing());
        // The digits are balanced: ||s||, ||t-hat|| <= ceil(b/2) = 2.
        assert!(report.max_norm_c_s <= 2 && report.max_norm_t_hat <= 2);
        // A PrgRandomness serves its resharing stream once (plan finding F5).
        let mut comm = Loopback::new(ctx.q()).unwrap();
        assert!(matches!(
            commit(&ctx, &mut comm, &mut prg, &f),
            Err(hachi_co_commit::Error::Core(
                CoreError::RandomnessReuse { .. }
            ))
        ));
    }
}

#[test]
fn fresh_runs_at_toy_n2_and_n3_pass_open_on_the_network() {
    for n in [2u64, 3] {
        for seed in 0..4u64 {
            let ctx = context(n, 100 + seed);
            let n = n as usize;
            let q = ctx.q();
            let len = 1usize << ctx.layout.ell;
            let mut rng = ChaCha8Rng::seed_from_u64(100 + seed);
            let f = random_vector(&mut rng, q, len);
            // The initial additive shares f_1, ..., f_n of f (6_CH:301).
            let field = ctx.field();
            let mut f_init: Vec<Vec<Ext<K>>> =
                (1..n).map(|_| random_vector(&mut rng, q, len)).collect();
            let last = (0..len)
                .map(|t| f_init.iter().fold(f[t], |acc, s| field.sub(&acc, &s[t])))
                .collect();
            f_init.push(last);
            let outputs = run_parties(n, q, |p, comm| {
                let mut prg = PrgRandomness::new(party_seed(100 + seed, p));
                commit(&ctx, comm, &mut prg, &f_init[p - 1]).unwrap()
            });
            let u = outputs[0].0.clone();
            assert!(outputs.iter().all(|(v, _)| *v == u), "every party gets u");
            let states: Vec<_> = outputs.into_iter().map(|(_, st)| st).collect();
            // The resharing is fresh: the shares differ from the initial ones but add up to f.
            let shares: Vec<_> = states.iter().map(|st| st.share.clone()).collect();
            assert_ne!(shares, f_init, "n = {n}, seed {seed}");
            let sum = f_hat(&ctx, &shares).unwrap();
            let total: Vec<Ext<K>> = (0..len)
                .map(|t| (0..n).fold(field.zero(), |acc, p| field.add(&acc, &sum[p * len + t])))
                .collect();
            assert_eq!(total, f);
            let report = open(&ctx, &states, &u, &f, None).unwrap();
            assert!(
                report.bit,
                "n = {n}, seed {seed}: {:?}",
                report.checks.failing()
            );
        }
    }
}
