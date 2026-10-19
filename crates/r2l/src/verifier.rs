//! The verifier's side of the ring-to-linear step: the challenges of the accepted attempt by
//! Fiat-Shamir, and the public matrix $\mathbf M$ and vector $\mathbf y$ of the linear relation
//! (`5_Building_Blocks.tex`:122-126, 154-179). The provers compute $\mathbf M$ and $\mathbf y$ the
//! same way.
//!
//! These functions take proof data ($a^\ast$, $\mathbf v$, $\mathbf c$) and never panic on it:
//! every malformed or out-of-range value is an error.

use crate::local::{check_challenges, check_polys};
use crate::{Error, LABEL_V, MAX_ATTEMPTS, Result};
use hachi_co_core::{Challenger, Context, Matrix, Poly, PublicVectors, build_m, build_y};

/// The challenges of the accepted attempt $a^\ast$ as a Fiat-Shamir verifier derives them:
/// [`Challenger::begin_attempt`]`(a*)`, [`Challenger::absorb_polys`] of $\mathbf v$ under
/// [`LABEL_V`], then [`Challenger::r2l_challenges`]`(a*, 2^r, d, ω)`, exactly the calls the
/// provers make in attempt $a^\ast$ ([`crate::run`]); the result is checked to lie in
/// $\mathcal C^{2^r}$. With [`hachi_co_core::FsChallenger`] in the state the provers' challenger
/// had before the step, it returns the provers' $\mathbf c$ (the documentation of
/// [`Challenger`]: the challenges of attempt $a$ do not depend on the aborted attempts).
///
/// The attempt index is a proof value and is checked first: an index at or above the cap
/// [`MAX_ATTEMPTS`] is refused before the challenger is called (honest provers stop after
/// [`MAX_ATTEMPTS`] attempts, [`crate::run`]), so a refused call leaves the challenger as it was.
///
/// # Errors
/// [`Error::AttemptOutOfRange`] if `attempt` $\ge$ [`MAX_ATTEMPTS`];
/// [`Error::Shape`] or [`Error::NonCanonical`] unless $\mathbf v$ has $n\_D$ polynomials of $d$
/// canonical residues; the challenger's errors; [`Error::ChallengeNotInC`].
pub fn verifier_challenges<const K: usize, H: Challenger<K>>(
    ctx: &Context<K>,
    ch: &mut H,
    attempt: u64,
    v: &[Poly],
) -> Result<Vec<Vec<i64>>> {
    if attempt >= MAX_ATTEMPTS {
        return Err(Error::AttemptOutOfRange {
            attempt,
            cap: MAX_ATTEMPTS,
        });
    }
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
/// [`Error::Core`] if the public vectors have the wrong shape; [`Error::Shape`],
/// [`Error::NonCanonical`] or [`Error::ChallengeNotInC`] for malformed proof values.
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

#[cfg(test)]
mod tests {
    //! The attempt cap of [`verifier_challenges`] at its boundary, at toy-n1 with public
    //! parameters from the seed `[7; 32]` and Fiat-Shamir on the domain [`DOMAIN`].
    use super::*;
    use hachi_co_core::FsChallenger;
    use hachi_co_params::{derive, presets::toy_choices};

    /// The extension degree of the toy presets.
    const K: usize = 4;
    const DOMAIN: &[u8] = b"hachi-co-r2l/verifier/unit-tests";

    fn context() -> Context<K> {
        let (params, _) = derive(&toy_choices(1)).unwrap();
        Context::setup(params, &[7; 32]).unwrap()
    }

    /// A challenger in a state of its own (as after a statement), and a canonical $\mathbf v$.
    fn inputs(ctx: &Context<K>) -> (FsChallenger, Vec<Poly>) {
        let l = &ctx.layout;
        let mut ch = FsChallenger::new(DOMAIN);
        Challenger::<K>::absorb_u64(&mut ch, "test/statement", 1);
        let v = (0..l.n_d)
            .map(|i| {
                Poly(
                    (0..l.d as u64)
                        .map(|j| (7 * i as u64 + j) % ctx.q())
                        .collect(),
                )
            })
            .collect();
        (ch, v)
    }

    #[test]
    fn every_attempt_below_the_cap_is_accepted() {
        let ctx = context();
        let l = &ctx.layout;
        for attempt in [0, 1, MAX_ATTEMPTS / 2, MAX_ATTEMPTS - 1] {
            let (ch, v) = inputs(&ctx);
            let c = verifier_challenges(&ctx, &mut ch.clone(), attempt, &v).unwrap();
            // The provers' calls of attempt a (crate::run), made directly.
            let mut provers = ch;
            Challenger::<K>::begin_attempt(&mut provers, attempt).unwrap();
            Challenger::<K>::absorb_polys(&mut provers, LABEL_V, &v);
            let expected = Challenger::<K>::r2l_challenges(
                &mut provers,
                attempt,
                l.n_blocks,
                l.d,
                ctx.omega(),
            )
            .unwrap();
            assert_eq!(c, expected, "attempt {attempt}");
            check_challenges(&ctx, &c).unwrap();
        }
    }

    #[test]
    fn attempts_at_or_beyond_the_cap_are_refused_before_the_challenger_is_called() {
        let ctx = context();
        for attempt in [MAX_ATTEMPTS, MAX_ATTEMPTS + 1, 1 << 32, u64::MAX] {
            let (ch, v) = inputs(&ctx);
            let mut used = ch.clone();
            assert_eq!(
                verifier_challenges(&ctx, &mut used, attempt, &v),
                Err(Error::AttemptOutOfRange {
                    attempt,
                    cap: MAX_ATTEMPTS
                })
            );
            assert_eq!(used, ch, "attempt {attempt}: the challenger was called");
            // The index is checked before v: a malformed v does not change the error.
            assert_eq!(
                verifier_challenges(&ctx, &mut used, attempt, &v[..0]),
                Err(Error::AttemptOutOfRange {
                    attempt,
                    cap: MAX_ATTEMPTS
                })
            );
        }
        // Below the cap a malformed v is still reported as before.
        let (mut ch, v) = inputs(&ctx);
        assert!(matches!(
            verifier_challenges(&ctx, &mut ch, MAX_ATTEMPTS - 1, &v[..0]),
            Err(Error::Shape { .. })
        ));
    }
}
