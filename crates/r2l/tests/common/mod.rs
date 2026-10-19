//! Shared helpers of the integration tests: the inputs of the ring-to-linear step built from a
//! fixture's `commit` and `conv` sections, and the n parties run on threads over the in-memory
//! network of `hachi-co-mpc` ([`run_parties`]).
#![allow(dead_code)]

use hachi_co_core::{Context, Fixture, PartyState, Poly, PublicVectors};
use hachi_co_mpc::{Comm, MemoryConfig, MemoryEndpoint, PartyId, run_in_memory};

pub const K: usize = 4;

/// Party p's state after Commit, as recorded by the oracle ([`Fixture::party_state`]).
pub fn state(fx: &Fixture, ctx: &Context<K>, p: usize) -> PartyState<K> {
    fx.party_state(ctx, p).unwrap()
}

/// The public vectors of Pi_conv as recorded ([`Fixture::public_vectors`]).
pub fn public_vectors(fx: &Fixture, ctx: &Context<K>) -> PublicVectors<K> {
    fx.public_vectors(ctx).unwrap()
}

/// Party p's share <u>_p of Pi_conv (`conv/u_shares`).
pub fn u_share(fx: &Fixture, p: usize) -> Poly {
    fx.u_share(p).unwrap()
}

/// The commitment u (`commit/u`).
pub fn u_com(fx: &Fixture) -> Vec<Poly> {
    fx.commitment().unwrap()
}

/// The number of recorded attempts.
pub fn attempts(fx: &Fixture) -> usize {
    fx.attempt_count().unwrap()
}

/// The fixture, its context and every party's state.
pub fn load(name: &str) -> (Fixture, Context<K>, PublicVectors<K>, Vec<PartyState<K>>) {
    let fx = Fixture::load(name).unwrap();
    let ctx = fx.context::<K>().unwrap();
    let pv = public_vectors(&fx, &ctx);
    let states = (1..=fx.n()).map(|p| state(&fx, &ctx, p)).collect();
    (fx, ctx, pv, states)
}

// ---------------------------------------------------------------- the in-memory network ----

/// Runs `party(p, comm)` for $p = 1, \dots, n$, one thread each, over a fresh in-memory network
/// ([`run_in_memory`], default zero sharing, a network seed fixed by $n$ and $q$) and returns the
/// results in party order.
pub fn run_parties<T: Send>(
    n: usize,
    q: u64,
    party: impl Fn(PartyId, &mut MemoryEndpoint) -> T + Sync,
) -> Vec<T> {
    let mut seed = [n as u8; 32];
    seed[..8].copy_from_slice(&q.to_le_bytes());
    run_in_memory(&MemoryConfig::new(n, q, seed), |comm| {
        party(comm.id(), comm)
    })
    .expect("a valid network")
    .into_iter()
    .map(|run| run.expect("every party connects").output)
    .collect()
}
