//! Shared helpers of the integration tests: the fixtures with their context, the parties' states
//! as the oracle's Commit recorded them, an independent $\mathsf{PConv}$ for fresh runs (this
//! crate cannot depend on `hachi-co-commit`), seeded random inputs, and the in-memory network.
#![allow(dead_code)]

mod net;

pub use net::run_parties;

use hachi_co_core::{BlockState, Context, Fixture, PartyId, PartyState};
use hachi_co_math::{Ext, Poly, decompose};
use rand::RngExt;
use rand_chacha::ChaCha8Rng;

/// The extension degree of every toy preset.
pub const K: usize = 4;

/// The fixture `name` and its context (with the recorded matrices).
pub fn load(name: &str) -> (Fixture, Context<K>) {
    let fx = Fixture::load(name).unwrap();
    let ctx = fx.context::<K>().unwrap();
    (fx, ctx)
}

/// Party $p$'s state as recorded by the oracle's Commit ([`Fixture::party_state`]).
pub fn recorded_state(fx: &Fixture, ctx: &Context<K>, p: PartyId) -> PartyState<K> {
    fx.party_state(ctx, p).unwrap()
}

/// Party $p$'s state from its share $\langle \mathbf f \rangle\_p$, written out from the draft
/// (`field-to-ring-rok.tex`:47-80, `6_Collaborative_Hachi.tex`:370-376) independently of
/// `hachi-co-commit`: entry $j$ of block $i$ is $\psi$ of $\hat f\[(i 2^m + j) g + \kappa'\]$,
/// $\kappa' \lt g = d/k$, where $\hat f\[(p-1) 2^\ell + t\] = \langle f \rangle\_p\[t\]$.
pub fn state_from_share(ctx: &Context<K>, p: PartyId, share: Vec<Ext<K>>) -> PartyState<K> {
    let l = &ctx.layout;
    let g = l.d / l.k;
    let base = (p - 1) << l.ell;
    let (ring, b, delta) = (&ctx.ring, ctx.b(), l.delta);
    let blocks = l
        .blocks_of(p)
        .map(|index| {
            let f_prime: Vec<Poly> = (0..l.block_len)
                .map(|j| {
                    let chunk: Vec<Ext<K>> = (0..g)
                        .map(|kk| share[(index * l.block_len + j) * g + kk - base])
                        .collect();
                    ctx.sub.psi(&chunk).unwrap()
                })
                .collect();
            let s = decompose::g_inv(ring, b, delta, &f_prime).unwrap();
            let t = ring.mat_vec(&ctx.pp.a, &s).unwrap();
            let t_hat = decompose::g_inv(ring, b, delta, &t).unwrap();
            BlockState {
                index,
                f_prime,
                s,
                t,
                t_hat,
            }
        })
        .collect();
    PartyState {
        party: p,
        share,
        blocks,
    }
}

/// `len` uniform elements of $\mathbb F\_{q^K}$.
pub fn random_vector(rng: &mut ChaCha8Rng, q: u64, len: usize) -> Vec<Ext<K>> {
    (0..len)
        .map(|_| Ext(core::array::from_fn(|_| rng.random_range(0..q))))
        .collect()
}

/// An additive sharing of $\mathbf f$ among $n$ parties: $n - 1$ uniform shares and the
/// difference.
pub fn additive_shares(
    ctx: &Context<K>,
    rng: &mut ChaCha8Rng,
    f: &[Ext<K>],
    n: usize,
) -> Vec<Vec<Ext<K>>> {
    let field = ctx.field();
    let mut shares: Vec<Vec<Ext<K>>> = (1..n)
        .map(|_| random_vector(rng, ctx.q(), f.len()))
        .collect();
    let last = (0..f.len())
        .map(|t| shares.iter().fold(f[t], |acc, s| field.sub(&acc, &s[t])))
        .collect();
    shares.push(last);
    shares
}
