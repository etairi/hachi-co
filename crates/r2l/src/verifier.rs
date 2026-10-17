//! The verifier's side of the ring-to-linear step: the challenges of the accepted attempt by
//! Fiat-Shamir, and the public matrix $\mathbf M$ and vector $\mathbf y$ of the linear relation
//! (`5_Building_Blocks.tex`:122-126, 154-179). The provers compute $\mathbf M$ and $\mathbf y$ the
//! same way.
//!
//! These functions take proof data ($a^\ast$, $\mathbf v$, $\mathbf c$) and never panic on it:
//! every malformed or out-of-range value is an error.

use crate::local::{check_challenges, check_polys};
use crate::{LABEL_V, Result};
use hachi_co_core::{Challenger, Context, Matrix, Poly, PublicVectors, build_m, build_y};

/// The challenges of the accepted attempt $a^\ast$ as a Fiat-Shamir verifier derives them:
/// [`Challenger::begin_attempt`]`(a*)`, [`Challenger::absorb_polys`] of $\mathbf v$ under
/// [`LABEL_V`], then [`Challenger::r2l_challenges`]`(a*, 2^r, d, ω)`, exactly the calls the
/// provers make in attempt $a^\ast$ ([`crate::run`]); the result is checked to lie in
/// $\mathcal C^{2^r}$. With [`hachi_co_core::FsChallenger`] in the state the provers' challenger
/// had before the step, it returns the provers' $\mathbf c$ (the documentation of
/// [`Challenger`]: the challenges of attempt $a$ do not depend on the aborted attempts).
///
/// # Errors
/// [`Error::Shape`](crate::Error::Shape) or [`Error::NonCanonical`](crate::Error::NonCanonical)
/// unless $\mathbf v$ has $n\_D$ polynomials of $d$ canonical residues; the challenger's errors;
/// [`Error::ChallengeNotInC`](crate::Error::ChallengeNotInC).
pub fn verifier_challenges<const K: usize, H: Challenger<K>>(
    ctx: &Context<K>,
    ch: &mut H,
    attempt: u64,
    v: &[Poly],
) -> Result<Vec<Vec<i64>>> {
    let l = &ctx.layout;
    check_polys("v", v, l.n_d, l.d, ctx.q())?;
    ch.begin_attempt(attempt)?;
    ch.absorb_polys(LABEL_V, v);
    let c = ch.r2l_challenges(attempt, l.n_blocks, l.d, ctx.omega())?;
    check_challenges(ctx, &c)?;
    Ok(c)
}

/// The linear relation $\mathbf M \mathbf z' = \mathbf y$ over $\mathcal R\_q$ produced by the
/// ring-to-linear step (`5_Building_Blocks.tex`:154-182).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearRelation {
    /// $\mathbf M \in \mathcal R\_q^{n' \times \mu}$, dense and row-major (lines 157-163).
    pub m: Matrix,
    /// $\mathbf y = (\mathbf v, \mathbf u, 0, 0, \mathbf 0) \in \mathcal R\_q^{n'}$
    /// (lines 173-179).
    pub y: Vec<Poly>,
}

/// $\mathbf M$ and $\mathbf y$ from public data (`5_Building_Blocks.tex`:157-179): the public
/// vectors of $\Pi\_{\mathsf{conv}}$ ($\mathbf b$ and $\mathbf a$ in $\mathcal R\_q$), the
/// accepted attempt's $\mathbf v$ and $\mathbf c$, and the commitment $\mathbf u$; after checking
/// $\mathbf c \in \mathcal C^{2^r}$ ([`check_challenges`]) and the shapes and canonical residues
/// of $\mathbf v$ and $\mathbf u$. [`build_m`] and [`build_y`] of `hachi-co-core` build them.
///
/// # Errors
/// [`Error::Core`](crate::Error::Core) if the public vectors have the wrong shape;
/// [`Error::Shape`](crate::Error::Shape), [`Error::NonCanonical`](crate::Error::NonCanonical) or
/// [`Error::ChallengeNotInC`](crate::Error::ChallengeNotInC) for malformed proof values.
pub fn linear_relation<const K: usize>(
    ctx: &Context<K>,
    pv: &PublicVectors<K>,
    v: &[Poly],
    c: &[Vec<i64>],
    u_com: &[Poly],
) -> Result<LinearRelation> {
    let l = &ctx.layout;
    pv.validate(l)?;
    check_challenges(ctx, c)?;
    check_polys("v", v, l.n_d, l.d, ctx.q())?;
    check_polys("u (commitment)", u_com, l.n_b, l.d, ctx.q())?;
    Ok(LinearRelation {
        m: build_m(ctx, &pv.b_ring, &pv.a_ring, c)?,
        y: build_y(ctx, v, u_com)?,
    })
}
