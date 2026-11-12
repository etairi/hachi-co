//! The party's share of the committed polynomial. Since the draft's commit `1096ea2`, Commit
//! starts from each party's additive share $\langle \mathbf f \rangle\_p$ of $\mathbf f$ as it
//! is (`6_Collaborative_Hachi.tex`, Commit: "Each party $i \in \[n\]$ holds an additive share
//! $\langle \mathbf f \rangle\_i$ of $\mathbf f$"); the resharing by $\mathsf{AS.Share}$ of the
//! earlier drafts (lines 308-313 at `8834c9f`) is gone.

use crate::error::{Result, check_len, pow2};
use hachi_co_core::Context;
use hachi_co_math::Ext;

/// Party $p$'s share $\langle \mathbf f \rangle\_p \in \mathbb F\_{q^K}^{2^\ell}$ of its input
/// `f_init`, the input itself with every coordinate read modulo $q$. The oracle's
/// `f_shares[p - 1]`, which equals `input/f_init_shares[p - 1]` reduced.
///
/// # Errors
/// [`Error::Shape`](crate::Error::Shape) unless `f_init` has $2^\ell$ entries.
pub fn input_share<const K: usize>(ctx: &Context<K>, f_init: &[Ext<K>]) -> Result<Vec<Ext<K>>> {
    check_len(
        "<f>_p (2^l coefficients)",
        pow2(ctx.layout.ell, "2^l")?,
        f_init.len(),
    )?;
    Ok(reduce_chunk(ctx, f_init))
}

/// The coordinates of `chunk` modulo $q$.
pub(crate) fn reduce_chunk<const K: usize>(ctx: &Context<K>, chunk: &[Ext<K>]) -> Vec<Ext<K>> {
    let field = ctx.field();
    chunk.iter().map(|x| field.reduce(x)).collect()
}
