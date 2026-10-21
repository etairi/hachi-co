//! Honest runs on synthetic instances at larger sizes, to check that the prover and the verifier
//! scale linearly with the hypercube (`docs/m1-spec.md`, WP10 performance; WP11 runs
//! l in {12, 16, 20} with n = 1, d = 16, b = 4, where l'' reaches 23).
//!
//! A synthetic instance: the parameters of `toy_choices(1)` with another l
//! (`hachi_co_params::derive`), A, B, D from `Context::setup`, random b, a (embedded field
//! elements) and short c, M = `build_m`, a random z' with every digit in [-b+1, b-1] and a
//! random <u>, y = M z', a random point v of E and y = E(u). It satisfies every condition the
//! sumcheck checks (for n = 1 every position below mu - 1 is in S_1), so honest runs must be
//! accepted. The randomness is a splitmix64 stream with the seed printed by each test.
//!
//! The runs at l = 16 and 20 are `#[ignore]`d (run with `-- --ignored --nocapture`); the runs at
//! l = 8 and 12 are part of the regular suite.

mod common;

use common::{K, clear_pcs, ideal_pcs};
use hachi_co_core::{Context, FsChallenger, LinearShare, Matrix, Poly, PrgRandomness, build_m};
use hachi_co_math::Ext;
use hachi_co_mpc::Loopback;
use hachi_co_params::{derive, presets::toy_choices};
use hachi_co_sumcheck::{
    PublicInput, eq_table, mstar_table, prove, quotient_share, verify_claims, w_table,
};
use std::time::Instant;

/// splitmix64.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

struct Synthetic {
    ctx: Context<K>,
    m: Matrix,
    e: Vec<Vec<u64>>,
    y: Vec<Poly>,
    y_ring: Poly,
    share: LinearShare,
}

fn synthetic(ell: u32, seed: u64) -> Synthetic {
    let mut choices = toy_choices(1);
    choices.ell = ell;
    let (params, _report) = derive(&choices).unwrap();
    let ctx = Context::<K>::setup(params, &[seed as u8; 32]).unwrap();
    let l = &ctx.layout;
    let (q, d, b) = (ctx.q(), l.d, ctx.b() as i64);
    let zq = *ctx.ring.zq();
    let mut rng = SplitMix(seed);
    let ext = |rng: &mut SplitMix| Ext::<K>(core::array::from_fn(|_| rng.next() % q));
    let b_ring: Vec<Poly> = (0..l.n_blocks)
        .map(|_| ctx.sub.embed(&ext(&mut rng)))
        .collect();
    let a_ring: Vec<Poly> = (0..l.block_len)
        .map(|_| ctx.sub.embed(&ext(&mut rng)))
        .collect();
    let c: Vec<Vec<i64>> = (0..l.n_blocks)
        .map(|_| (0..d).map(|_| (rng.next() % 3) as i64 - 1).collect())
        .collect();
    let m = build_m(&ctx, &b_ring, &a_ring, &c).unwrap();
    let span = (2 * b - 1) as u64;
    let mut z_prime: Vec<Poly> = (0..l.mu - 1)
        .map(|_| {
            Poly(
                (0..d)
                    .map(|_| zq.from_i64((rng.next() % span) as i64 - (b - 1)))
                    .collect(),
            )
        })
        .collect();
    let u = Poly((0..d).map(|_| rng.next() % q).collect());
    z_prime.push(u.clone());
    let y = ctx.ring.mat_vec(&m, &z_prime).unwrap();
    let v = Poly((0..d).map(|_| rng.next() % q).collect());
    let e = ctx.sub.e_matrix(&v).unwrap();
    let y_ring = ctx.sub.e_map(&u, &v).unwrap();
    let share = LinearShare {
        z_prime,
        y: y.clone(),
    };
    Synthetic {
        ctx,
        m,
        e,
        y,
        y_ring,
        share,
    }
}

/// One honest run with timings: the quotient, the M* table, the whole prover, the verifier.
fn honest_run(ell: u32, seed: u64) {
    let start = Instant::now();
    let s = synthetic(ell, seed);
    let built = start.elapsed();
    let l = &s.ctx.layout;
    let f = s.ctx.field();
    eprintln!(
        "l = {ell} (seed {seed}): q = {}, mu = {}, n' = {}, l'' = {} (2^l'' = {}), d0 = {}, \
         log nu = {}; instance built in {built:?}",
        s.ctx.q(),
        l.mu,
        l.n_prime,
        l.ell_pp,
        1usize << l.ell_pp,
        l.d0,
        l.log_nu
    );
    // The steps on their own (the driver repeats them).
    let t = Instant::now();
    let r = quotient_share(&s.ctx, &s.m, &s.share).unwrap();
    let quotient = t.elapsed();
    let t = Instant::now();
    let table = w_table(&s.ctx, &s.share.z_prime, &r).unwrap();
    let eq1 = eq_table(f, &vec![f.one(); l.log_nu]).unwrap();
    let mstar = mstar_table(&s.ctx, &s.m, &s.e, &eq1, &Ext([3, 1, 4, 1])).unwrap();
    let tables = t.elapsed();
    drop((table, eq1, mstar));
    // The whole prover and the verifier, with Fiat-Shamir and ClearPcs (whose openings are as
    // large as the polynomials, and cost the verifier linear time).
    let domain = b"hachi-co/test/sumcheck/scaling";
    let pcs = clear_pcs(f);
    let t = Instant::now();
    let mut comm = Loopback::new(s.ctx.q()).unwrap();
    let mut rng = PrgRandomness::new([seed as u8 ^ 0x5a; 32]);
    let mut ch = FsChallenger::new(domain);
    let (transcript, private) = prove(
        &s.ctx, &pcs, &mut comm, &mut rng, &mut ch, &s.m, &s.e, &s.share,
    )
    .unwrap();
    let proved = t.elapsed();
    let t = Instant::now();
    let public = PublicInput::new(&s.ctx, &s.m, &s.y, &s.e, &s.y_ring);
    let mut vch = FsChallenger::new(domain);
    let claims = verify_claims(&s.ctx, &pcs, &public, &transcript, &mut vch).unwrap();
    let verified = t.elapsed();
    eprintln!(
        "l = {ell}: quotient {quotient:?}, w and M* tables {tables:?}, prove (all steps, \
         ClearPcs) {proved:?}, verify {verified:?}, base-PCS words {}",
        transcript.commitments.total_words() + transcript.openings.total_words()
    );
    assert_eq!(ch.transcript().digest(), vch.transcript().digest());
    let pcs = ideal_pcs(f, &claims, &[private]);
    assert!(pcs.iter().all(|(_, ok)| *ok), "{pcs:?}");
}

#[test]
fn honest_run_at_l_8() {
    honest_run(8, 0x8888);
}

#[test]
fn honest_run_at_l_12() {
    honest_run(12, 0x1212);
}

#[test]
#[ignore = "about 2 s; run with --ignored"]
fn honest_run_at_l_16() {
    honest_run(16, 0x1616);
}

#[test]
#[ignore = "about 6 s and 0.7 GB; run with --ignored"]
fn honest_run_at_l_20() {
    honest_run(20, 0x2020);
}
