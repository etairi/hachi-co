//! The Commit driver of one party (`6_Collaborative_Hachi.tex`:300-391), run against a
//! [`Comm`].

use crate::error::{Error, Result};
use crate::inner::{fah_b_input, party_state};
use crate::share::{decode_received, reshare, sum_shares};
use hachi_co_core::wire::{flatten_exts, flatten_polys, unflatten_polys};
use hachi_co_core::{Context, PartyRandomness, PartyState};
use hachi_co_math::{Ext, Poly};
use hachi_co_mpc::Comm;

/// The [`Comm::exchange`] label of the resharing (step 1, `6_Collaborative_Hachi.tex`:303-305).
pub const LABEL_RESHARE: &str = "hachi-co/commit/reshare";
/// The [`Comm::fah`] label of $\mathcal F\_{\mathsf{AH}}(\mathbf B)$ (step 5,
/// `6_Collaborative_Hachi.tex`:377-387).
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
/// $= \mathbf f\_p \in \mathbb F\_{q^k}^{2^\ell}$ (`6_Collaborative_Hachi.tex`:300-391). Every
/// party calls it with its own `comm` and `rng`; the collective calls are, in order:
///
/// 1. [`reshare`] with `rng` ($\mathsf{AS.Share}$), then [`Comm::exchange`] with
///    [`LABEL_RESHARE`]: share $j$ goes to party $j$ (lines 303-305), and the received shares are
///    checked ([`decode_received`]);
/// 2. [`sum_shares`]: $\langle \mathbf f \rangle\_p = \sum\_i \langle \mathbf f\_i \rangle\_p$
///    (lines 306-308);
/// 3. [`party_state`]: $\mathsf{PConv}$ of $p$'s blocks and their inner commitments
///    $\mathbf s\_i, \mathbf t\_i, \hat{\mathbf t}\_i$ (lines 310-376);
/// 4. [`fah_b_input`] and [`Comm::fah`] with [`LABEL_FAH_B`]: every party receives
///    $\mathbf u = \mathbf B \hat{\mathbf t}$ (lines 377-387).
///
/// Returns the commitment $\mathbf u \in \mathcal R\_q^{n\_B}$ and $p$'s state $\mathsf{st}\_p$
/// (line 390, with $\mathbf t\_i$ and the share; [`PartyState`]). With [`hachi_co_mpc::Loopback`]
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
    let l = &ctx.layout;
    // Step 1: AS.Share and the pairwise exchange.
    let shares = reshare(ctx, rng, f_init)?;
    let outgoing = shares.iter().map(|s| flatten_exts(s)).collect();
    let incoming = comm.exchange(LABEL_RESHARE, outgoing)?;
    let received = decode_received(ctx, &incoming)?;
    // Step 2: the local sum.
    let share = sum_shares(ctx, &received)?;
    // Steps 3-4: PConv of the party's blocks and the inner commitments.
    let state = party_state(ctx, comm.id(), share)?;
    // Step 5: F_AH(B).
    let local = fah_b_input(ctx, &state.blocks)?;
    let output = comm.fah(LABEL_FAH_B, &flatten_polys(&local))?;
    let u = unflatten_polys(&output, l.n_b, l.d).map_err(|_| Error::Shape {
        what: "F_AH(B) output (n_B d residues)",
        expected: l.n_b.saturating_mul(l.d),
        actual: output.len(),
    })?;
    Ok((u, state))
}
