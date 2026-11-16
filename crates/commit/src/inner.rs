//! Steps 3-5 of Commit for one party (`6_Collaborative_Hachi.tex`:315-397): its blocks of
//! $\mathbf f'$ ($\mathsf{PConv}$), the inner commitments $\mathbf s\_i = \mathbf G\_{2^m}^{-1}
//! (\mathbf f'\_i)$, $\mathbf t\_i = \mathbf A \mathbf s\_i$, $\hat{\mathbf t}\_i =
//! \mathbf G\_{n\_A}^{-1}(\mathbf t\_i)$ of its blocks, and its input to
//! $\mathcal F\_{\mathsf{AH}}(\mathbf B)$, the image $\mathbf B \lbrace \hat{\mathbf t} \rbrace\_p$
//! of its slice of $\hat{\mathbf t}$ (the oracle's `commit.commit`, steps 3-5).

use crate::error::{Error, Result, check_len};
use crate::polyconv::pconv_blocks;
use hachi_co_core::{BlockState, Context, PartyId, PartyRandomness, PartyState};
use hachi_co_math::rns::NADDS;
use hachi_co_math::{Ext, Poly, decompose};
use rayon::prelude::*;

/// The inner commitment of block $i$ (`6_Collaborative_Hachi.tex`:375-381):
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
    let t = ring.mat_vec_prepared(&*ctx.a_prepared()?, &s)?;
    let t_hat = decompose::g_inv(ring, b, delta, &t)?;
    Ok(BlockState {
        index,
        f_prime,
        s,
        t,
        t_hat,
    })
}

/// [`block_state`] without $\mathbf s\_i$, for the streaming Commit, which drops it: $\mathbf t\_i
/// = \mathbf A \mathbf G\_{2^m}^{-1}(\mathbf f'\_i)$ by the prepared $\mathbf A$'s products on
/// pieces of whole elements whose digits are formed in a buffer
/// ([`hachi_co_math::Ring::mat_vec_prepared_pieces`], [`decompose::g_inv_rows`]), so that the
/// digits are never ring elements. The pieces hold at most [`NADDS`] digit vectors, so the sums
/// are exact as in [`block_state`]: the same $\mathbf t\_i$ and $\hat{\mathbf t}\_i$.
/// Sequential, for blocks formed in parallel.
///
/// # Errors
/// As [`block_state`].
pub fn block_state_streamed<const K: usize>(
    ctx: &Context<K>,
    index: usize,
    f_prime: Vec<Poly>,
) -> Result<BlockState> {
    block_state_in_pieces(ctx, index, f_prime, (NADDS / ctx.layout.delta).max(1))
}

/// [`block_state_streamed`] with pieces of `step` elements (at most [`NADDS`] digit vectors for
/// exact sums), so that the tests can cut the toy presets' blocks, which fit in one piece.
fn block_state_in_pieces<const K: usize>(
    ctx: &Context<K>,
    index: usize,
    f_prime: Vec<Poly>,
    step: usize,
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
    let (ring, b, delta, d) = (&ctx.ring, ctx.b(), l.delta, l.d);
    let zq = ring.zq();
    debug_assert!((1..=(NADDS / delta).max(1)).contains(&step));
    let len = f_prime.len();
    let pieces = (0..len)
        .step_by(step)
        .map(|e| e * delta..(e + step).min(len) * delta);
    let mut scratch = vec![0i64; d];
    let t = ring.mat_vec_prepared_pieces(&*ctx.a_prepared()?, pieces, |piece, out| {
        let (first, last) = (piece.start / delta, piece.end / delta);
        for (element, digits) in f_prime[first..last]
            .iter()
            .zip(out.chunks_exact_mut(delta * d))
        {
            decompose::g_inv_rows(zq, b, delta, &element.0, digits, &mut scratch)?;
        }
        Ok(())
    })?;
    let t_hat = decompose::g_inv(ring, b, delta, &t)?;
    Ok(BlockState {
        index,
        f_prime,
        s: Vec::new(),
        t,
        t_hat,
    })
}

/// $\boldsymbol\rho^u\_p \leftarrow S\_b^{m\_u}$ (`6_Collaborative_Hachi.tex`:382), as $m\_u$ ring
/// elements of residues, from [`PartyRandomness::rho_u`].
///
/// # Errors
/// The errors of `rng`; [`Error::Shape`] if it does not return $m\_u d$ integers;
/// [`Error::Math`] for an integer it cannot reduce.
pub fn sample_rho_u<const K: usize, R: PartyRandomness<K>>(
    ctx: &Context<K>,
    rng: &mut R,
) -> Result<Vec<Poly>> {
    let (m_u, d) = (ctx.layout.m_u, ctx.layout.d);
    let flat = rng.rho_u(ctx.b(), m_u * d)?;
    check_len("rho^u_p (m_u d integers)", m_u * d, flat.len())?;
    flat.chunks_exact(d)
        .map(|c| Ok(ctx.ring.from_i64(c)?))
        .collect()
}

/// Party $p$'s state after steps 2-5, $\mathsf{st}\_p$ of `6_Collaborative_Hachi.tex`:400 with
/// its share: its blocks of $\mathbf f'$ by [`pconv_blocks`] and their inner commitments by
/// [`block_state`], in increasing order of index, and its `rho_u` ([`sample_rho_u`]).
///
/// # Errors
/// As [`pconv_blocks`] and [`block_state`].
pub fn party_state<const K: usize>(
    ctx: &Context<K>,
    party: PartyId,
    share: Vec<Ext<K>>,
    rho_u: Vec<Poly>,
) -> Result<PartyState<K>> {
    // The blocks are independent: formed in parallel, collected in order, with A prepared
    // first by the whole pool.
    ctx.a_prepared()?;
    let blocks = pconv_blocks(ctx, party, &share)?
        .into_par_iter()
        .map(|(i, f_prime)| block_state(ctx, i, f_prime))
        .collect::<Result<Vec<_>>>()?;
    Ok(PartyState {
        party,
        share,
        blocks,
        rho_u,
        spill: None,
    })
}

/// Checks the shape of every block and that no index repeats.
fn check_blocks<const K: usize>(ctx: &Context<K>, blocks: &[BlockState]) -> Result<()> {
    let mut seen = vec![false; ctx.layout.n_blocks];
    for block in blocks {
        block.validate_commitment(&ctx.layout)?;
        if core::mem::replace(&mut seen[block.index], true) {
            return Err(Error::DuplicateBlock { index: block.index });
        }
    }
    Ok(())
}

/// The slice $\lbrace \hat{\mathbf t} \rbrace\_p \in \mathcal R\_q^{2^r n\_A \delta}$ of a party
/// (`6_Collaborative_Hachi.tex`:383-393; `4_preliminaries.tex`:167-177): $\hat{\mathbf t}\_i$ at
/// $\[i n\_A \delta, (i+1) n\_A \delta)$ for the given blocks, zero elsewhere. Summed over the
/// parties it is $\hat{\mathbf t}$ (fixture key `commit/t_hat`).
///
/// # Errors
/// [`Error::Core`] for a block of the wrong shape; [`Error::DuplicateBlock`].
pub fn t_hat_slice<const K: usize>(ctx: &Context<K>, blocks: &[BlockState]) -> Result<Vec<Poly>> {
    check_blocks(ctx, blocks)?;
    let l = &ctx.layout;
    let width = l.n_a * l.delta;
    let mut out = vec![ctx.ring.zero(); l.n_blocks * width];
    for block in blocks {
        out[block.index * width..(block.index + 1) * width].clone_from_slice(&block.t_hat);
    }
    Ok(out)
}

/// Party $p$'s input to $\mathcal F\_{\mathsf{AH}}(\mathbf B)$ (`6_Collaborative_Hachi.tex`:
/// 383-397): $\mathbf B \lbrace (\hat{\mathbf t}, \boldsymbol\rho^u) \rbrace\_p = \sum\_i \mathbf
/// B\_i \hat{\mathbf t}\_i + \mathbf B\_{\boldsymbol\rho, p} \boldsymbol\rho^u\_p \in
/// \mathcal R\_q^{n\_B}$ over the given blocks, with $\mathbf B\_i$ the $n\_A \delta$ columns of
/// $\mathbf B$ that act on $\hat{\mathbf t}\_i$ and $\mathbf B\_{\boldsymbol\rho, p}$ the $m\_u$
/// columns that act on $\boldsymbol\rho^u\_p$ (exact products). The sum over the parties is
/// $\mathbf u = \mathbf B (\hat{\mathbf t}, \boldsymbol\rho^u)$.
///
/// # Errors
/// [`Error::Core`] for a block of the wrong shape; [`Error::DuplicateBlock`];
/// [`Error::Shape`] unless `rho_u` has $m\_u$ entries.
pub fn fah_b_input<const K: usize>(
    ctx: &Context<K>,
    party: PartyId,
    blocks: &[BlockState],
    rho_u: &[Poly],
) -> Result<Vec<Poly>> {
    check_blocks(ctx, blocks)?;
    let (l, ring) = (&ctx.layout, &ctx.ring);
    check_len("rho^u_p (m_u entries)", l.m_u, rho_u.len())?;
    let width = l.n_a * l.delta;
    let mut out = vec![ring.zero(); l.n_b];
    for block in blocks {
        let cols = block.index * width..(block.index + 1) * width;
        for (o, row) in out.iter_mut().zip(&ctx.pp().b) {
            let part = ring.inner(&row[cols.clone()], &block.t_hat)?;
            *o = ring.add(o, &part)?;
        }
    }
    // The columns of B act on z'[t, z).
    let range = l.rho_u_range(party);
    let cols = range.start - l.t..range.end - l.t;
    for (o, row) in out.iter_mut().zip(&ctx.pp().b) {
        let part = ring.inner(&row[cols.clone()], rho_u)?;
        *o = ring.add(o, &part)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hachi_co_core::Fixture;

    /// The streamed block of the toy fixtures' recorded blocks, in pieces of every size up to the
    /// default: the $\mathbf t\_i$ and $\hat{\mathbf t}\_i$ of [`block_state`], without
    /// $\mathbf s\_i$.
    #[test]
    fn the_streamed_block_is_the_block_in_memory() {
        for name in Fixture::TOY {
            let fx = Fixture::load(name).unwrap();
            let ctx = fx.context::<4>().unwrap();
            let l = &ctx.layout;
            let share = fx.exts::<4>("/commit/f_shares/0").unwrap();
            let rho_u = vec![ctx.ring.zero(); l.m_u];
            let state = party_state(&ctx, 1, share, rho_u).unwrap();
            let default = (NADDS / l.delta).max(1);
            assert!(
                default >= l.block_len,
                "{name}: a block is one piece by default"
            );
            for block in &state.blocks {
                let want = BlockState {
                    s: Vec::new(),
                    ..block.clone()
                };
                for step in (1..=l.block_len).chain([default]) {
                    let got = block_state_in_pieces(&ctx, block.index, block.f_prime.clone(), step)
                        .unwrap();
                    assert_eq!(
                        got, want,
                        "{name}: block {} in pieces of {step}",
                        block.index
                    );
                }
                assert_eq!(
                    block_state_streamed(&ctx, block.index, block.f_prime.clone()).unwrap(),
                    want
                );
            }
            // The refusals of block_state.
            let f_prime = state.blocks[0].f_prime.clone();
            assert!(matches!(
                block_state_streamed(&ctx, l.n_blocks, f_prime.clone()),
                Err(Error::Shape { .. })
            ));
            assert!(matches!(
                block_state_streamed(&ctx, 0, f_prime[1..].to_vec()),
                Err(Error::Shape { .. })
            ));
            let mut bad_poly = f_prime;
            bad_poly[0].0.pop();
            assert!(matches!(
                block_state_streamed(&ctx, 0, bad_poly),
                Err(Error::Math(_))
            ));
        }
    }
}
