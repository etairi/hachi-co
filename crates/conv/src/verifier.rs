//! The verifier's step of $\Pi\_A$ (`5_Building_Blocks.tex`, $\Pi\_A$, step 5 at the draft's
//! commit `c071784`): it reads $E(u) \in \mathcal R\_q^H$ as the value $y \in \mathbb F\_{q^k}$
//! ([`read_value`]), and $y$ is an output of $\mathsf{Eval}$. Until that commit the verifier
//! received a claimed $y$ and checked $y = E(u)$ (`field-to-ring-rok.tex`:189-193 at `8834c9f`),
//! which [`check`] keeps for tests and for callers that compare against an expected value.

use crate::error::VerifyError;
use hachi_co_core::Context;
use hachi_co_math::{Ext, Poly};

/// V's step of $\Pi\_A$ with reasons: the message $E(u)$ must have $d$ canonical coefficients and
/// lie in $\mathcal R\_q^H$ (Hachi Eq. (7)), and its coordinates in Hachi's basis are the value
/// $y \in \mathbb F\_{q^k}$, which this returns.
///
/// **Choice** (the oracle reduces modulo $q$ first; with canonical inputs the result is the
/// same): residues of $E(u)$ that are not below $q$ are rejected, so that every accepted message
/// has exactly one encoding.
///
/// # Errors
/// The first failing condition, in the order [`VerifyError::Shape`],
/// [`VerifyError::NonCanonical`], [`VerifyError::NotInSubfield`]. Never panics.
pub fn read_value<const K: usize>(ctx: &Context<K>, e_u: &Poly) -> Result<Ext<K>, VerifyError> {
    let (d, q) = (ctx.layout.d, ctx.q());
    if e_u.0.len() != d {
        return Err(VerifyError::Shape {
            expected: d,
            actual: e_u.0.len(),
        });
    }
    if let Some(value) = e_u.0.iter().find(|x| **x >= q) {
        return Err(VerifyError::NonCanonical {
            what: "E(u)",
            value: *value,
            q,
        });
    }
    ctx.sub.extract(e_u).map_err(|_| VerifyError::NotInSubfield)
}

/// [`read_value`] compared with an expected value $y$ (the oracle's `conv.verify_conv`, and the
/// verifier's check $y = E(u)$ of the draft at `8834c9f`).
///
/// # Errors
/// The errors of [`read_value`], then [`VerifyError::NonCanonical`] for a residue of $y$ that
/// is not below $q$, then [`VerifyError::Mismatch`]. Never panics.
pub fn check<const K: usize>(ctx: &Context<K>, y: &Ext<K>, e_u: &Poly) -> Result<(), VerifyError> {
    let value = read_value(ctx, e_u)?;
    let q = ctx.q();
    if let Some(residue) = y.0.iter().find(|x| **x >= q) {
        return Err(VerifyError::NonCanonical {
            what: "y",
            value: *residue,
            q,
        });
    }
    if value != *y {
        return Err(VerifyError::Mismatch);
    }
    Ok(())
}

/// [`check`] as a bit (`docs/m1-spec.md`, WP8): `true` iff $E(u) \in \mathcal R\_q^H$ and it
/// equals $y$. Never panics.
pub fn verify<const K: usize>(ctx: &Context<K>, y: &Ext<K>, e_u: &Poly) -> bool {
    check(ctx, y, e_u).is_ok()
}
