//! The $n$ parties on threads over the in-memory network of `hachi-co-mpc`, whose
//! $\mathcal F\_{\mathsf{AH}}$ is the masked broadcast ([`run_parties`]).

use hachi_co_mpc::{Comm, MemoryConfig, MemoryEndpoint, PartyId, run_in_memory};

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
