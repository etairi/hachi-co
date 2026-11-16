//! The Commit driver of one party (`6_Collaborative_Hachi.tex`:305-401, without the resharing
//! of lines 308-313 since the draft's commit `1096ea2`), run against a [`Comm`].

use crate::error::{Error, Result};
use crate::inner::{block_state_streamed, fah_b_input, party_state, sample_rho_u};
use crate::share::{input_share, reduce_chunk};
use hachi_co_core::wire::{flatten_polys, unflatten_polys};
use hachi_co_core::{Context, PartyRandomness, PartyState, SpillFile};
use hachi_co_math::{Ext, Poly};
use hachi_co_mpc::Comm;
use rayon::prelude::*;
use std::path::Path;

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

/// $\mathsf{Commit}\langle \mathcal P\_1(\langle \mathbf f \rangle\_1), \dots,
/// \mathcal P\_n(\langle \mathbf f \rangle\_n) \rangle(\mathsf{pp})$ for the party $p$ =
/// `comm.id()` with its share `f_init` $= \langle \mathbf f \rangle\_p \in
/// \mathbb F\_{q^k}^{2^\ell}$ (`6_Collaborative_Hachi.tex`:305-401 as the draft's commit
/// `1096ea2` revises it: no resharing). Every party calls it with its own `comm` and `rng`:
///
/// 1. [`input_share`]: the share, its coordinates modulo $q$;
/// 2. [`sample_rho_u`] and [`party_state`]: $\boldsymbol\rho^u\_p$ (line 382), $\mathsf{PConv}$
///    of $p$'s blocks and their inner commitments
///    $\mathbf s\_i, \mathbf t\_i, \hat{\mathbf t}\_i$ (lines 315-381);
/// 3. [`fah_b_input`] and [`Comm::fah`] with [`LABEL_FAH_B`], the only collective call: every
///    party receives $\mathbf u = \mathbf B (\hat{\mathbf t}, \boldsymbol\rho^u)$
///    (lines 383-397).
///
/// Returns the commitment $\mathbf u \in \mathcal R\_q^{n\_B}$ and $p$'s state $\mathsf{st}\_p$
/// (line 400, with $\mathbf t\_i$ and the share; [`PartyState`]). With [`hachi_co_mpc::Loopback`]
/// ($n = 1$) every collective call is the identity.
///
/// # Errors
/// [`Error::Network`] or [`Error::InvalidParty`] if `comm` does not match the context;
/// [`Error::Shape`] for an `f_init` without $2^\ell$ entries; the errors of `rng` and `comm`.
pub fn commit<const K: usize, C: Comm, R: PartyRandomness<K>>(
    ctx: &Context<K>,
    comm: &mut C,
    rng: &mut R,
    f_init: &[Ext<K>],
) -> Result<(Vec<Poly>, PartyState<K>)> {
    check_comm(ctx, comm)?;
    let share = input_share(ctx, f_init)?;
    // PConv of the party's blocks and the inner commitments.
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
/// [`commit`] on its own, without $\mathbf s\_i$, and $2 \times$ the threads' chunks at a time, so
/// that memory holds a few blocks at a time.
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
    let spill = SpillFile::create(dir, party, l)?;
    let mut blocks = Vec::with_capacity(l.blocks_per_party);
    // The chunks are read in order, a group at a time, and the blocks of a group are formed in
    // parallel, each sequentially ([`block_state_streamed`]); then written in order. A is
    // prepared first, by the whole pool, rather than by the first blocks.
    ctx.a_prepared()?;
    let slots: Vec<(usize, usize)> = l.blocks_of(party).enumerate().collect();
    let group = 2 * rayon::current_num_threads().max(1);
    for part in slots.chunks(group) {
        let chunks = part
            .iter()
            .map(|&(slot, index)| {
                let chunk = input(slot)?;
                crate::error::check_len(
                    "a chunk of <f>_p (2^m d/k entries)",
                    chunk_len,
                    chunk.len(),
                )?;
                Ok((slot, index, chunk))
            })
            .collect::<Result<Vec<_>>>()?;
        let formed = chunks
            .into_par_iter()
            .map(|(slot, index, chunk)| {
                let share = reduce_chunk(ctx, &chunk);
                let f_prime = share
                    .chunks_exact(g)
                    .map(|c| ctx.sub.psi(c))
                    .collect::<hachi_co_math::Result<Vec<_>>>()?;
                Ok((slot, block_state_streamed(ctx, index, f_prime)?))
            })
            .collect::<Result<Vec<_>>>()?;
        for (slot, mut block) in formed {
            spill.write_block(slot, &block.f_prime)?;
            block.f_prime = Vec::new();
            blocks.push(block);
        }
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
