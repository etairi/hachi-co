//! Steps 1-2 of $\Pi\_{\mathsf{conv}}$ (`field-to-ring-rok.tex`:105-123): the public vectors
//! $\mathbf x', \mathbf b, \mathbf a, v, \mathbf d$, computed by every prover and (for $v$, and
//! $\mathbf b, \mathbf a$ in the matrix $\mathbf M$ of the ring-to-linear step) by the verifier.

use crate::error::{Error, Result, check_len};
use hachi_co_core::{Context, PublicVectors};
use hachi_co_math::Ext;

/// The public vectors of $\Pi\_{\mathsf{conv}}$ for the evaluation point $\mathbf x \in
/// \mathbb F\_{q^k}^\ell$ (`field-to-ring-rok.tex`:105-123; the oracle's `conv.public_vectors`,
/// `ref/README.md`, "Πconv"), with $\ell\_t = \ell - \alpha + \kappa$:
///
/// - $\mathbf x' = \mathbf 1^{\lceil \log n \rceil} \Vert (x\_1, \dots, x\_{\ell\_t}) \in
///   \mathbb F\_{q^k}^{\ell'}$ (line 113);
/// - $\mathbf b = \mathrm{monomials}(x'\_1, \dots, x'\_r) \in \mathbb F\_{q^k}^{2^r}$ and
///   $\mathbf a = \mathrm{monomials}(x'\_{r+1}, \dots, x'\_{\ell'}) \in \mathbb F\_{q^k}^{2^m}$
///   (lines 114-115), ordered by the MSB-first $\mathrm{nat}$
///   ([`hachi_co_math::Subfield::monomials`]);
/// - $v = \psi(\mathrm{monomials}(x\_{\ell\_t + 1}, \dots, x\_\ell)) \in \mathcal R\_q$ (line 116);
/// - $d\[\mathrm{nat}(\mathbf i) 2^m + \mathrm{nat}(\mathbf j)\] = b\_{\mathbf i} a\_{\mathbf j}$,
///   i.e. $\mathbf d^T = \mathbf b^T (\mathbf I\_{2^r} \otimes \mathbf a^T)$ (lines 117-121);
/// - $\mathbf b$ and $\mathbf a$ embedded in $\mathcal R\_q$ (Hachi Eq. (7);
///   [`hachi_co_math::Subfield::embed`]), as they enter $\mathbf M$
///   (`5_Building_Blocks.tex`:160-161).
///
/// The coordinates of $\mathbf x$ are read modulo $q$; every output is reduced. This is
/// verifier-side code: malformed input gives an error, never a panic.
///
/// # Errors
/// [`Error::Shape`] unless $\mathbf x$ has $\ell$ entries (or if the layout's
/// $\ell' \ne \lceil \log n \rceil + \ell - \alpha + \kappa$).
pub fn public_vectors<const K: usize>(ctx: &Context<K>, x: &[Ext<K>]) -> Result<PublicVectors<K>> {
    let l = &ctx.layout;
    let (field, sub) = (ctx.field(), &ctx.sub);
    check_len("x (l coordinates)", l.ell, x.len())?;
    // alpha - kappa = log2(d/k): the trailing coordinates packed by psi (Context: k | d/2).
    let packed = (l.d / l.k.max(1)).trailing_zeros() as usize;
    let lt = l.ell.checked_sub(packed).ok_or(Error::Shape {
        what: "l (must be at least alpha - kappa, 6_Collaborative_Hachi.tex:273)",
        expected: packed,
        actual: l.ell,
    })?;
    check_len(
        "l' (ceil(log n) + l - alpha + kappa)",
        l.log_n + lt,
        l.ell_prime,
    )?;
    if l.r > l.ell_prime {
        return Err(Error::Shape {
            what: "r (at most l')",
            expected: l.ell_prime,
            actual: l.r,
        });
    }
    let x: Vec<Ext<K>> = x.iter().map(|xi| field.reduce(xi)).collect();
    let mut x_prime = vec![field.one(); l.log_n];
    x_prime.extend_from_slice(&x[..lt]);
    let b = sub.monomials(&x_prime[..l.r])?;
    let a = sub.monomials(&x_prime[l.r..])?;
    let v_point = sub.psi(&sub.monomials(&x[lt..])?)?;
    let d = b
        .iter()
        .flat_map(|bi| a.iter().map(move |aj| field.mul(bi, aj)))
        .collect();
    let b_ring = b.iter().map(|e| sub.embed(e)).collect();
    let a_ring = a.iter().map(|e| sub.embed(e)).collect();
    Ok(PublicVectors {
        x_prime,
        b,
        a,
        d,
        v_point,
        b_ring,
        a_ring,
    })
}
