//! $\mathsf{PConv}$ (`field-to-ring-rok.tex`:36-87; the oracle's `commit.pconv`): the secret-shared
//! coefficient vector $\langle \mathbf f \rangle$ over $\mathbb F\_{q^k}$ becomes the vector
//! $\mathbf f' \in \mathcal R\_q^{2^{\ell'}}$, whose block $\mathbf f'\_{\mathbf i}$ is held by a
//! single prover.
//!
//! With $\hat\ell = \lceil \log n \rceil + \ell$ and $g = d/k = 2^{\alpha - \kappa}$
//! (`ref/README.md`, "PConv"):
//! $$\hat f\[\mathrm{nat}(\mathbf p) 2^\ell + \mathrm{nat}(\mathbf q)\] =
//! \begin{cases} \langle \mathbf f \rangle\_{\mathrm{nat}(\mathbf p) + 1}\[\mathrm{nat}(\mathbf q)\]
//! & \mathrm{nat}(\mathbf p) \lt n \\\\ 0 & \text{otherwise,} \end{cases} \qquad
//! f'\[t\] = \psi\bigl(\hat f\[t g\], \dots, \hat f\[t g + g - 1\]\bigr).$$
//! Block $i$ of $\mathbf f'$ is $f'\[i 2^m .. (i+1) 2^m)$. Since $\lceil \log n \rceil \le r$, the
//! top $\lceil \log n \rceil$ bits of $i$ name the slot $\mathbf p$ that the whole block comes
//! from (`field-to-ring-rok.tex`:74), so prover $p$ computes its blocks
//! ([`hachi_co_core::Layout::blocks_of`]) from $\langle \mathbf f \rangle\_p$ alone
//! ([`pconv_blocks`]); the blocks of the empty slots ($n$ not a power of two) are zero and held by
//! nobody. The other functions are the global view, for tests and for Open's decoding.
//!
//! **Choice** (the oracle's, `ref/README.md`, choice 3): $\mathsf{PConv}$ runs once, in Commit;
//! $\Pi\_{\mathsf{conv}}$ reuses $\mathbf f'$ from the states.

use crate::error::{Error, Result, check_len, pow2};
use hachi_co_core::{Context, Layout, PartyId};
use hachi_co_math::{Ext, Poly};

/// The sizes of the conversion, checked against each other.
#[derive(Clone, Copy, Debug)]
struct Sizes {
    /// $2^\ell$, the entries of a share.
    coeffs: usize,
    /// $2^{\hat\ell} = 2^{\lceil \log n \rceil} 2^\ell$, the entries of $\hat{\mathbf f}$.
    f_hat: usize,
    /// $2^{\ell'}$, the entries of $\mathbf f'$.
    f_prime: usize,
    /// $g = d/k$, the field elements packed into one ring element.
    g: usize,
}

fn sizes(l: &Layout) -> Result<Sizes> {
    let coeffs = pow2(l.ell, "2^l")?;
    let f_hat = pow2(l.log_n, "2^ceil(log n)")?
        .checked_mul(coeffs)
        .ok_or(hachi_co_core::Error::Overflow("2^(ceil(log n) + l)"))?;
    let f_prime = pow2(l.ell_prime, "2^l'")?;
    // Context::new guarantees k | d/2 (hachi_co_math::Subfield::new).
    let g = l.d / l.k.max(1);
    // field-to-ring-rok.tex:75: l-hat - l' = alpha - kappa, i.e. 2^l' g = 2^l-hat.
    if f_prime.checked_mul(g) != Some(f_hat) {
        return Err(Error::Shape {
            what: "2^l' d/k (must equal 2^(ceil(log n) + l), field-to-ring-rok.tex:75)",
            expected: f_hat,
            actual: f_prime.saturating_mul(g),
        });
    }
    // Each prover's blocks cover exactly its share: 2^(r - ceil(log n)) 2^m g = 2^l.
    let per_party = l
        .blocks_per_party
        .checked_mul(l.block_len)
        .and_then(|x| x.checked_mul(g));
    if per_party != Some(coeffs) {
        return Err(Error::Shape {
            what: "2^(r - ceil(log n)) 2^m d/k (must equal 2^l)",
            expected: coeffs,
            actual: per_party.unwrap_or(usize::MAX),
        });
    }
    Ok(Sizes {
        coeffs,
        f_hat,
        f_prime,
        g,
    })
}

/// $\hat{\mathbf f} \in \mathbb F\_{q^k}^{2^{\hat\ell}}$ from all $n$ shares
/// (`field-to-ring-rok.tex`:47-59): share $p$ in slot $p - 1$, zeros in the slots $\ge n$. The
/// oracle's `f_hat` (fixture key `commit/f_hat`). Coordinates are reduced modulo $q$.
///
/// # Errors
/// [`Error::Shape`] unless there are $n$ shares of $2^\ell$ entries.
pub fn f_hat<const K: usize>(ctx: &Context<K>, shares: &[Vec<Ext<K>>]) -> Result<Vec<Ext<K>>> {
    let l = &ctx.layout;
    let s = sizes(l)?;
    check_len("shares (one per party)", l.n, shares.len())?;
    let field = ctx.field();
    let mut out = vec![field.zero(); s.f_hat];
    for (slot, share) in out.chunks_mut(s.coeffs).zip(shares) {
        check_len("<f>_p (2^l entries)", s.coeffs, share.len())?;
        for (o, x) in slot.iter_mut().zip(share) {
            *o = field.reduce(x);
        }
    }
    Ok(out)
}

/// $\mathbf f'$ from $\hat{\mathbf f}$: $f'\[t\] = \psi(\hat f\[tg .. tg + g))$
/// (`field-to-ring-rok.tex`:70-80), for all $2^{\ell'}$ entries.
///
/// # Errors
/// [`Error::Shape`] unless $\hat{\mathbf f}$ has $2^{\hat\ell}$ entries.
pub fn f_prime_of_f_hat<const K: usize>(ctx: &Context<K>, f_hat: &[Ext<K>]) -> Result<Vec<Poly>> {
    let s = sizes(&ctx.layout)?;
    check_len("f-hat (2^(ceil(log n) + l) entries)", s.f_hat, f_hat.len())?;
    Ok(f_hat
        .chunks(s.g)
        .map(|chunk| ctx.sub.psi(chunk))
        .collect::<hachi_co_math::Result<Vec<_>>>()?)
}

/// The global $\mathsf{PConv}$: $(\hat{\mathbf f}, \mathbf f')$ from all $n$ shares
/// ([`f_hat`], then [`f_prime_of_f_hat`]); the oracle's `commit.pconv`.
///
/// # Errors
/// As [`f_hat`].
pub fn pconv<const K: usize>(
    ctx: &Context<K>,
    shares: &[Vec<Ext<K>>],
) -> Result<(Vec<Ext<K>>, Vec<Poly>)> {
    let f_hat = f_hat(ctx, shares)?;
    let f_prime = f_prime_of_f_hat(ctx, &f_hat)?;
    Ok((f_hat, f_prime))
}

/// Party $p$'s part of $\mathsf{PConv}$ (`field-to-ring-rok.tex`:64-80;
/// `6_Collaborative_Hachi.tex`:310-313): the blocks $\mathbf f'\_i \in \mathcal R\_q^{2^m}$ for $i$
/// in [`Layout::blocks_of`]`(p)`, in increasing order, computed from $\langle \mathbf f \rangle\_p$
/// alone: entry $j$ of block $i$ is $\psi$ of the $g$ entries of $\langle \mathbf f \rangle\_p$
/// from $(i 2^m + j) g - (p - 1) 2^\ell$ on.
///
/// # Errors
/// [`Error::InvalidParty`] for $p \notin \lbrace 1, \dots, n \rbrace$; [`Error::Shape`] unless
/// the share has $2^\ell$ entries.
pub fn pconv_blocks<const K: usize>(
    ctx: &Context<K>,
    party: PartyId,
    share: &[Ext<K>],
) -> Result<Vec<(usize, Vec<Poly>)>> {
    let l = &ctx.layout;
    if party == 0 || party > l.n {
        return Err(Error::InvalidParty { party, n: l.n });
    }
    let s = sizes(l)?;
    check_len("<f>_p (2^l entries)", s.coeffs, share.len())?;
    // Party p's blocks start at f-hat index (p - 1) 2^l (sizes() checked the identities).
    let mut chunks = share.chunks(s.g);
    l.blocks_of(party)
        .map(|i| {
            let block = (0..l.block_len)
                .map(|_| {
                    let chunk = chunks.next().ok_or(Error::Shape {
                        what: "<f>_p (too short for the blocks of the party)",
                        expected: s.coeffs,
                        actual: share.len(),
                    })?;
                    Ok(ctx.sub.psi(chunk)?)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((i, block))
        })
        .collect()
}

/// $\hat{\mathbf f}$ from $\mathbf f'$ by $\psi^{-1}$ entry by entry: $(\hat f\[tg + j\])\_{j \lt
/// g} = \psi^{-1}(f'\[t\])$, which inverts [`f_prime_of_f_hat`] because $\psi$ is a bijection
/// (`4_preliminaries.tex`:81-84, Hachi Theorem 2); the decoding step of Open
/// (`6_Collaborative_Hachi.tex`:398).
///
/// # Errors
/// [`Error::Shape`] unless there are $2^{\ell'}$ entries; [`Error::Math`] for an entry without
/// $d$ coefficients.
pub fn f_hat_of_f_prime<const K: usize>(ctx: &Context<K>, f_prime: &[Poly]) -> Result<Vec<Ext<K>>> {
    let s = sizes(&ctx.layout)?;
    check_len("f' (2^l' entries)", s.f_prime, f_prime.len())?;
    let mut out = Vec::with_capacity(s.f_hat);
    for entry in f_prime {
        out.extend(ctx.sub.psi_inv(entry)?);
    }
    Ok(out)
}

/// $\bar f\_{\mathbf q} = \sum\_{\mathbf p \in \lbrace 0,1 \rbrace^{\lceil \log n \rceil}} \hat
/// f\_{\mathbf p \Vert \mathbf q}$ for every $\mathbf q \in \lbrace 0,1 \rbrace^\ell$
/// (`6_Collaborative_Hachi.tex`:398): the sum of the slots, over all $2^{\lceil \log n \rceil}$
/// of them.
///
/// # Errors
/// [`Error::Shape`] unless $\hat{\mathbf f}$ has $2^{\hat\ell}$ entries.
pub fn fold_slots<const K: usize>(ctx: &Context<K>, f_hat: &[Ext<K>]) -> Result<Vec<Ext<K>>> {
    let s = sizes(&ctx.layout)?;
    check_len("f-hat (2^(ceil(log n) + l) entries)", s.f_hat, f_hat.len())?;
    let field = ctx.field();
    let mut out = vec![field.zero(); s.coeffs];
    for slot in f_hat.chunks(s.coeffs) {
        for (o, x) in out.iter_mut().zip(slot) {
            *o = field.add(o, x);
        }
    }
    Ok(out)
}
