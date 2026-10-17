//! Steps 1 and 2 of Commit (`6_Collaborative_Hachi.tex`:303-308): every party $i$ reshares its
//! initial share $\mathbf f\_i$ by $\mathsf{AS.Share}$ (`4_preliminaries.tex`:154), sends
//! $\langle \mathbf f\_i \rangle\_j$ to party $j$, and every party $j$ adds what it received:
//! $\langle \mathbf f \rangle\_j = \sum\_{i \in \[n\]} \langle \mathbf f\_i \rangle\_j$.

use crate::error::{Error, Result, check_len, pow2};
use hachi_co_core::{Context, PartyRandomness};
use hachi_co_math::Ext;

/// Step 1 for one party: $\langle \mathbf f\_i \rangle = \mathsf{AS.Share}(\mathbf f\_i, n)$
/// (`6_Collaborative_Hachi.tex`:303-305; `4_preliminaries.tex`:154), drawn from `rng`
/// ([`PartyRandomness::reshare`]: shares $1, \dots, n-1$ uniform, share $n$ the difference).
/// Returns the $n$ shares in order; share $j$ goes to party $j$ (the party's own share
/// included). The oracle's `commit.as_share`.
///
/// The coordinates of `f_init` are read modulo $q$.
///
/// # Errors
/// [`Error::Shape`] unless `f_init` has $2^\ell$ entries or if `rng` returns shares of the wrong
/// shape; the errors of `rng` as [`Error::Core`].
pub fn reshare<const K: usize, R: PartyRandomness<K>>(
    ctx: &Context<K>,
    rng: &mut R,
    f_init: &[Ext<K>],
) -> Result<Vec<Vec<Ext<K>>>> {
    let l = &ctx.layout;
    let len = pow2(l.ell, "2^l")?;
    check_len("f_i (2^l coefficients)", len, f_init.len())?;
    let shares = rng.reshare(ctx.field(), f_init, l.n)?;
    check_len("AS.Share output (n shares)", l.n, shares.len())?;
    for share in &shares {
        check_len("AS.Share output share (2^l entries)", len, share.len())?;
    }
    Ok(shares)
}

/// Decodes what a party received in the exchange of step 1: `incoming[j - 1]` is
/// $\langle \mathbf f\_j \rangle\_p$ from party $j$, as the $2^\ell K$ coordinates of its
/// $2^\ell$ elements ([`hachi_co_core::wire::flatten_exts`]).
///
/// # Errors
/// [`Error::Shape`] unless there are $n$ vectors of $2^\ell K$ entries; [`Error::NonCanonical`]
/// (naming the sender) for a residue $\ge q$.
pub fn decode_received<const K: usize>(
    ctx: &Context<K>,
    incoming: &[Vec<u64>],
) -> Result<Vec<Vec<Ext<K>>>> {
    let l = &ctx.layout;
    let len = pow2(l.ell, "2^l")?;
    let q = ctx.q();
    check_len("received shares (one per party)", l.n, incoming.len())?;
    incoming
        .iter()
        .enumerate()
        .map(|(j, flat)| {
            check_len(
                "received share (2^l K coordinates)",
                len.saturating_mul(K),
                flat.len(),
            )?;
            if let Some(value) = flat.iter().find(|x| **x >= q) {
                return Err(Error::NonCanonical {
                    from: j + 1,
                    what: "a reshared share",
                    value: *value,
                    q,
                });
            }
            Ok(hachi_co_core::wire::unflatten_exts::<K>(flat, len)?)
        })
        .collect()
}

/// Step 2 for party $p$: $\langle \mathbf f \rangle\_p = \sum\_{i \in \[n\]}
/// \langle \mathbf f\_i \rangle\_p$ (`6_Collaborative_Hachi.tex`:306-308), from the $n$ received
/// shares (`received[i - 1]` from party $i$). The oracle's `f_shares[p - 1]`.
///
/// # Errors
/// [`Error::Shape`] unless there are $n$ shares of $2^\ell$ entries.
pub fn sum_shares<const K: usize>(
    ctx: &Context<K>,
    received: &[Vec<Ext<K>>],
) -> Result<Vec<Ext<K>>> {
    let l = &ctx.layout;
    let len = pow2(l.ell, "2^l")?;
    check_len("received shares (one per party)", l.n, received.len())?;
    let field = ctx.field();
    let mut sum = vec![field.zero(); len];
    for share in received {
        check_len("received share (2^l entries)", len, share.len())?;
        for (s, x) in sum.iter_mut().zip(share) {
            *s = field.add(s, x);
        }
    }
    Ok(sum)
}
