//! The verifier's check of $\Pi\_{\mathsf{conv}}$ (`field-to-ring-rok.tex`:160-164): $E(u) \in
//! \mathcal R\_q^H$ and, read in $\mathbb F\_{q^k}$, $y = E(u)$.

use crate::error::VerifyError;
use hachi_co_core::Context;
use hachi_co_math::{Ext, Poly};

/// V's check of $\Pi\_{\mathsf{conv}}$ with reasons (`field-to-ring-rok.tex`:163-164; the oracle's
/// `conv.verify_conv`): the message $E(u)$ must have $d$ canonical coefficients and lie in
/// $\mathcal R\_q^H$ (Hachi Eq. (7)), and its coordinates in Hachi's basis must equal the claimed
/// evaluation $y \in \mathbb F\_{q^k}$.
///
/// **Choice** (the oracle reduces both sides modulo $q$ first; with canonical inputs the result
/// is the same): residues of $E(u)$ or $y$ that are not below $q$ are rejected, so that every
/// accepted message has exactly one encoding.
///
/// # Errors
/// The first failing condition, in the order [`VerifyError::Shape`],
/// [`VerifyError::NonCanonical`] (for $E(u)$, then for $y$), [`VerifyError::NotInSubfield`],
/// [`VerifyError::Mismatch`]. Never panics.
pub fn check<const K: usize>(ctx: &Context<K>, y: &Ext<K>, e_u: &Poly) -> Result<(), VerifyError> {
    let (d, q) = (ctx.layout.d, ctx.q());
    if e_u.0.len() != d {
        return Err(VerifyError::Shape {
            expected: d,
            actual: e_u.0.len(),
        });
    }
    for (what, residues) in [("E(u)", &e_u.0[..]), ("y", &y.0[..])] {
        if let Some(value) = residues.iter().find(|x| **x >= q) {
            return Err(VerifyError::NonCanonical {
                what,
                value: *value,
                q,
            });
        }
    }
    let value = ctx
        .sub
        .extract(e_u)
        .map_err(|_| VerifyError::NotInSubfield)?;
    if value != *y {
        return Err(VerifyError::Mismatch);
    }
    Ok(())
}

/// V's check of $\Pi\_{\mathsf{conv}}$ as a bit (`docs/m1-spec.md`, WP8): `true` iff [`check`]
/// accepts, i.e. $E(u) \in \mathcal R\_q^H$ and $y = E(u)$. Never panics.
pub fn verify<const K: usize>(ctx: &Context<K>, y: &Ext<K>, e_u: &Poly) -> bool {
    check(ctx, y, e_u).is_ok()
}
