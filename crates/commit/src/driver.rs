//! The Commit driver of one party (`6_Collaborative_Hachi.tex`:305-401), run against a
//! [`Comm`].

use crate::error::{Error, Result};
use crate::inner::{block_state, fah_b_input, party_state, sample_rho_u};
use crate::share::{decode_chunk, decode_received, reshare, sum_chunk, sum_shares};
use hachi_co_core::wire::{flatten_exts, flatten_polys, unflatten_polys};
use hachi_co_core::{Context, PartyRandomness, PartyState, SpillFile};
use hachi_co_math::{Ext, Poly};
use hachi_co_mpc::Comm;
use std::path::Path;

/// The [`Comm::exchange`] label of the resharing (step 1, `6_Collaborative_Hachi.tex`:308-310).
pub const LABEL_RESHARE: &str = "hachi-co/commit/reshare";
/// The [`Comm::fah`] label of $\mathcal F\_{\mathsf{AH}}(\mathbf B)$ (step 6,
/// `6_Collaborative_Hachi.tex`:383-397).
pub const LABEL_FAH_B: &str = "hachi-co/commit/F_AH(B)";
/// The label under which the commitment $\mathbf u$ is absorbed into a
/// [`hachi_co_core::Challenger`] (with `absorb_polys`) when it enters a statement; Commit itself
/// draws no coins.
pub const LABEL_U: &str = "hachi-co/commit/u";

/// Checks that `comm` has the context's $n$ and $q$ and a valid party index.
pub(crate) fn check_comm<const K: usize, C: Comm>(ctx: &Context<K>, comm: &C) -> Result<()> {
    let n = ctx.layout.n;
    if comm.n() != n {
        return Err(Error::Network {
            what: "n",
            expected: n as u64,
            actual: comm.n() as u64,
        });
    }
    if comm.modulus() != ctx.q() {
        return Err(Error::Network {
            what: "q",
            expected: ctx.q(),
            actual: comm.modulus(),
        });
    }
    if comm.id() == 0 || comm.id() > n {
        return Err(Error::InvalidParty {
            party: comm.id(),
            n,
        });
    }
    Ok(())
}

/// $\mathsf{Commit}\langle \mathcal P\_1(\mathbf f\_1), \dots, \mathcal P\_n(\mathbf f\_n)
/// \rangle(\mathsf{pp})$ for the party $p$ = `comm.id()` with initial share `f_init`
/// $= \mathbf f\_p \in \mathbb F\_{q^k}^{2^\ell}$ (`6_Collaborative_Hachi.tex`:305-401). Every
/// party calls it with its own `comm` and `rng`; the collective calls are, in order:
///
/// 1. [`reshare`] with `rng` ($\mathsf{AS.Share}$), then [`Comm::exchange`] with
///    [`LABEL_RESHARE`]: share $j$ goes to party $j$ (lines 308-310), and the received shares are
///    checked ([`decode_received`]);
/// 2. [`sum_shares`]: $\langle \mathbf f \rangle\_p = \sum\_i \langle \mathbf f\_i \rangle\_p$
///    (lines 311-313);
/// 3. [`sample_rho_u`] and [`party_state`]: $\boldsymbol\rho^u\_p$ (line 382), $\mathsf{PConv}$
///    of $p$'s blocks and their inner commitments
///    $\mathbf s\_i, \mathbf t\_i, \hat{\mathbf t}\_i$ (lines 315-381);
/// 4. [`fah_b_input`] and [`Comm::fah`] with [`LABEL_FAH_B`]: every party receives
///    $\mathbf u = \mathbf B (\hat{\mathbf t}, \boldsymbol\rho^u)$ (lines 383-397).
///
/// Returns the commitment $\mathbf u \in \mathcal R\_q^{n\_B}$ and $p$'s state $\mathsf{st}\_p$
/// (line 400, with $\mathbf t\_i$ and the share; [`PartyState`]). With [`hachi_co_mpc::Loopback`]
/// ($n = 1$) every collective call is the identity.
///
/// # Errors
/// [`Error::Network`] or [`Error::InvalidParty`] if `comm` does not match the context;
/// [`Error::Shape`] for an `f_init` without $2^\ell$ entries; [`Error::NonCanonical`] or
/// [`Error::Shape`] for malformed shares from another party; the errors of `rng` and `comm`.
pub fn commit<const K: usize, C: Comm, R: PartyRandomness<K>>(
    ctx: &Context<K>,
    comm: &mut C,
    rng: &mut R,
    f_init: &[Ext<K>],
) -> Result<(Vec<Poly>, PartyState<K>)> {
    check_comm(ctx, comm)?;
    // Step 1: AS.Share and the pairwise exchange.
    let shares = reshare(ctx, rng, f_init)?;
    let outgoing = shares.iter().map(|s| flatten_exts(s)).collect();
    let incoming = comm.exchange(LABEL_RESHARE, outgoing)?;
    let received = decode_received(ctx, &incoming)?;
    // Step 2: the local sum.
    let share = sum_shares(ctx, &received)?;
    // Steps 3-4: PConv of the party's blocks and the inner commitments.
    let rho_u = sample_rho_u(ctx, rng)?;
    let state = party_state(ctx, comm.id(), share, rho_u)?;
    let u = fah_b(ctx, comm, &state)?;
    Ok((u, state))
}

/// Step 6: $\mathbf u$ from $\mathcal F\_{\mathsf{AH}}(\mathbf B)$.
fn fah_b<const K: usize, C: Comm>(
    ctx: &Context<K>,
    comm: &mut C,
    state: &PartyState<K>,
) -> Result<Vec<Poly>> {
    let l = &ctx.layout;
    let local = fah_b_input(ctx, state.party, &state.blocks, &state.rho_u)?;
    let output = comm.fah(LABEL_FAH_B, &flatten_polys(&local))?;
    unflatten_polys(&output, l.n_b, l.d).map_err(|_| Error::Shape {
        what: "F_AH(B) output (n_B d residues)",
        expected: l.n_b.saturating_mul(l.d),
        actual: output.len(),
    })
}

/// [`commit`] for a polynomial too large for memory: the same messages and the same commitment,
/// with the party's initial share given block by block and the blocks of $\mathbf f'$ kept on
/// disk ([`SpillFile`], in the directory `dir`). `input(j)` returns the $j$-th chunk of
/// $\mathbf f\_p$, of $2^m d/k$ elements, $j \lt$ `blocks_per_party`; its entries are the
/// coefficients of the $j$-th of the party's blocks. Every chunk goes through the steps of
/// [`commit`] on its own, so that memory holds one block of $\mathbf s\_i$ at a time; `exchange`
/// runs once per chunk, and with [`PartyRandomness::reshare_stream`] the shares do not depend on
/// the chunks.
///
/// The state has no `share` and no `f_prime` or `s` in its blocks
/// ([`PartyState::for_each_block`]).
///
/// # Errors
/// As [`commit`]; [`Error::Core`] if the file fails; `input`'s errors.
pub fn commit_streaming<const K: usize, C: Comm, R: PartyRandomness<K>>(
    ctx: &Context<K>,
    comm: &mut C,
    rng: &mut R,
    dir: &Path,
    mut input: impl FnMut(usize) -> Result<Vec<Ext<K>>>,
) -> Result<(Vec<Poly>, PartyState<K>)> {
    check_comm(ctx, comm)?;
    let (l, party) = (&ctx.layout, comm.id());
    let g = l.d / l.k.max(1);
    let chunk_len = l.block_len * g;
    let mut stream = rng.reshare_stream(ctx.field(), l.n)?;
    let spill = SpillFile::create(dir, party, l)?;
    let mut blocks = Vec::with_capacity(l.blocks_per_party);
    for (slot, index) in l.blocks_of(party).enumerate() {
        let chunk = input(slot)?;
        crate::error::check_len("a chunk of f_p (2^m d/k entries)", chunk_len, chunk.len())?;
        let outgoing = stream
            .next_chunk(&chunk)?
            .iter()
            .map(|s| flatten_exts(s))
            .collect();
        let incoming = comm.exchange(LABEL_RESHARE, outgoing)?;
        let received = decode_chunk(ctx, &incoming, chunk_len)?;
        let share = sum_chunk(ctx, &received, chunk_len)?;
        let f_prime = share
            .chunks_exact(g)
            .map(|c| ctx.sub.psi(c))
            .collect::<hachi_co_math::Result<Vec<_>>>()?;
        let mut block = block_state(ctx, index, f_prime)?;
        spill.write_block(slot, &block.f_prime)?;
        block.f_prime = Vec::new();
        block.s = Vec::new();
        blocks.push(block);
    }
    let state = PartyState {
        party,
        share: Vec::new(),
        blocks,
        rho_u: sample_rho_u(ctx, rng)?,
        spill: Some(spill),
    };
    let u = fah_b(ctx, comm, &state)?;
    Ok((u, state))
}
