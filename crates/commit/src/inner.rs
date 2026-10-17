//! Steps 3-5 of Commit for one party (`6_Collaborative_Hachi.tex`:310-387): its blocks of
//! $\mathbf f'$ ($\mathsf{PConv}$), the inner commitments $\mathbf s\_i = \mathbf G\_{2^m}^{-1}
//! (\mathbf f'\_i)$, $\mathbf t\_i = \mathbf A \mathbf s\_i$, $\hat{\mathbf t}\_i =
//! \mathbf G\_{n\_A}^{-1}(\mathbf t\_i)$ of its blocks, and its input to
//! $\mathcal F\_{\mathsf{AH}}(\mathbf B)$, the image $\mathbf B \lbrace \hat{\mathbf t} \rbrace\_p$
//! of its slice of $\hat{\mathbf t}$ (the oracle's `commit.commit`, steps 3-5).

use crate::error::{Error, Result, check_len};
use crate::polyconv::pconv_blocks;
use hachi_co_core::{BlockState, Context, PartyId, PartyState};
use hachi_co_math::{Ext, Poly, decompose};

/// The inner commitment of block $i$ (`6_Collaborative_Hachi.tex`:370-376):
/// $\mathbf s\_i = \mathbf G\_{2^m}^{-1}(\mathbf f'\_i) \in \mathcal R\_q^{2^m \delta}$,
/// $\mathbf t\_i = \mathbf A \mathbf s\_i \in \mathcal R\_q^{n\_A}$ (exact product) and
/// $\hat{\mathbf t}\_i = \mathbf G\_{n\_A}^{-1}(\mathbf t\_i) \in \mathcal R\_q^{n\_A \delta}$,
/// with the balanced digits and the representative rule of `ref/README.md` ("Gadgets", choice 2;
/// [`decompose::g_inv`]) in the element-major layout, stored as residues.
///
/// # Errors
/// [`Error::Shape`] for $i \ge 2^r$ or unless $\mathbf f'\_i$ has $2^m$ entries; [`Error::Math`]
/// for an entry without $d$ coefficients.
pub fn block_state<const K: usize>(
    ctx: &Context<K>,
    index: usize,
    f_prime: Vec<Poly>,
) -> Result<BlockState> {
    let l = &ctx.layout;
    if index >= l.n_blocks {
        return Err(Error::Shape {
            what: "block index (must be below 2^r)",
            expected: l.n_blocks,
            actual: index,
        });
    }
    check_len("f'_i (2^m entries)", l.block_len, f_prime.len())?;
    let (ring, b, delta) = (&ctx.ring, ctx.b(), l.delta);
    let s = decompose::g_inv(ring, b, delta, &f_prime)?;
    let t = ring.mat_vec(&ctx.pp.a, &s)?;
    let t_hat = decompose::g_inv(ring, b, delta, &t)?;
    Ok(BlockState {
        index,
        f_prime,
        s,
        t,
        t_hat,
    })
}

/// Party $p$'s state after steps 2-4, $\mathsf{st}\_p$ of `6_Collaborative_Hachi.tex`:390 with
/// its share: its blocks of $\mathbf f'$ by [`pconv_blocks`] and their inner commitments by
/// [`block_state`], in increasing order of index.
///
/// # Errors
/// As [`pconv_blocks`] and [`block_state`].
pub fn party_state<const K: usize>(
    ctx: &Context<K>,
    party: PartyId,
    share: Vec<Ext<K>>,
) -> Result<PartyState<K>> {
    let blocks = pconv_blocks(ctx, party, &share)?
        .into_iter()
        .map(|(i, f_prime)| block_state(ctx, i, f_prime))
        .collect::<Result<Vec<_>>>()?;
    Ok(PartyState {
        party,
        share,
        blocks,
    })
}

/// Checks the shape of every block and that no index repeats.
fn check_blocks<const K: usize>(ctx: &Context<K>, blocks: &[BlockState]) -> Result<()> {
    let mut seen = vec![false; ctx.layout.n_blocks];
    for block in blocks {
        block.validate(&ctx.layout)?;
        if core::mem::replace(&mut seen[block.index], true) {
            return Err(Error::DuplicateBlock { index: block.index });
        }
    }
    Ok(())
}

/// The slice $\lbrace \hat{\mathbf t} \rbrace\_p \in \mathcal R\_q^{2^r n\_A \delta}$ of a party
/// (`6_Collaborative_Hachi.tex`:377-383; `4_preliminaries.tex`:167-177): $\hat{\mathbf t}\_i$ at
/// $\[i n\_A \delta, (i+1) n\_A \delta)$ for the given blocks, zero elsewhere. Summed over the
/// parties it is $\hat{\mathbf t}$ (fixture key `commit/t_hat`).
///
/// # Errors
/// [`Error::Core`] for a block of the wrong shape; [`Error::DuplicateBlock`].
pub fn t_hat_slice<const K: usize>(ctx: &Context<K>, blocks: &[BlockState]) -> Result<Vec<Poly>> {
    check_blocks(ctx, blocks)?;
    let l = &ctx.layout;
    let width = l.n_a * l.delta;
    let mut out = vec![ctx.ring.zero(); l.width_b];
    for block in blocks {
        out[block.index * width..(block.index + 1) * width].clone_from_slice(&block.t_hat);
    }
    Ok(out)
}

/// Party $p$'s input to $\mathcal F\_{\mathsf{AH}}(\mathbf B)$ (`6_Collaborative_Hachi.tex`:
/// 377-387): $\mathbf B \lbrace \hat{\mathbf t} \rbrace\_p = \sum\_i \mathbf B\_i \hat{\mathbf
/// t}\_i \in \mathcal R\_q^{n\_B}$ over the given blocks, with $\mathbf B\_i$ the $n\_A \delta$
/// columns of $\mathbf B$ that act on $\hat{\mathbf t}\_i$ (exact products). The sum over the
/// parties is $\mathbf u = \mathbf B \hat{\mathbf t}$.
///
/// # Errors
/// [`Error::Core`] for a block of the wrong shape; [`Error::DuplicateBlock`].
pub fn fah_b_input<const K: usize>(ctx: &Context<K>, blocks: &[BlockState]) -> Result<Vec<Poly>> {
    check_blocks(ctx, blocks)?;
    let (l, ring) = (&ctx.layout, &ctx.ring);
    let width = l.n_a * l.delta;
    let mut out = vec![ring.zero(); l.n_b];
    for block in blocks {
        let cols = block.index * width..(block.index + 1) * width;
        for (o, row) in out.iter_mut().zip(&ctx.pp.b) {
            let part = ring.inner(&row[cols.clone()], &block.t_hat)?;
            *o = ring.add(o, &part)?;
        }
    }
    Ok(out)
}
