//! The verifier's side of the ring-to-linear step: the challenges by Fiat-Shamir, the selected
//! iteration $j^\ast$ from the provers' sets of accepted iterations, and the public matrix
//! $\mathbf M$ and vector $\mathbf y$ of the linear relation (`5_Building_Blocks.tex`:134-138,
//! 167-193; `fig:ring-to-linear`, steps 4-6, since the draft's commit `1096ea2`). The provers
//! compute $\mathbf M$ and $\mathbf y$ the same way.
//!
//! These functions take proof data ($\mathbf v$, the sets, $\mathbf c$) and never panic on it:
//! every malformed or out-of-range value is an error.

use crate::accept::select_iteration;
use crate::local::{check_challenges, check_polys};
use crate::{Error, LABEL_ACCEPT, LABEL_V, Result};
use hachi_co_core::{Challenger, Context, Matrix, Poly, PublicVectors, build_m, build_y};

/// The challenges as a Fiat-Shamir verifier derives them: [`Challenger::absorb_polys`] of
/// $\mathbf v$ under [`LABEL_V`], then [`Challenger::r2l_challenges`]`(2^r, d, ω)`, exactly the
/// calls the provers make ([`crate::run`]); the result is checked to lie in
/// $\mathcal C^{2^r}$. With [`hachi_co_core::FsChallenger`] in the state the provers' challenger
/// had before the step, it returns the provers' $\mathbf c$.
///
/// # Errors
/// [`Error::Shape`] or [`Error::NonCanonical`] unless $\mathbf v$ has $n\_D$ polynomials of $d$
/// canonical residues (checked before the challenger is called); the challenger's errors;
/// [`Error::ChallengeNotInC`].
pub fn verifier_challenges<const K: usize, H: Challenger<K>>(
    ctx: &Context<K>,
    ch: &mut H,
    v: &[Poly],
) -> Result<Vec<Vec<i64>>> {
    let l = &ctx.layout;
    check_polys("v", v, l.n_d, l.d, ctx.q())?;
    ch.absorb_polys(LABEL_V, v);
    let c = ch.r2l_challenges(l.n_blocks, l.d, ctx.omega())?;
    check_challenges(ctx, &c)?;
    Ok(c)
}

/// The selected iteration as the verifier computes it from the provers' sets of accepted
/// iterations (`fig:ring-to-linear`, step 6): every set is checked
/// ([`check_accept_set`](crate::check_accept_set)), the sets are absorbed under
/// [`LABEL_ACCEPT`] (the provers' call after their broadcast, [`crate::run`]), and $j^\ast$ is
/// the smallest iteration in every set ([`select_iteration`]). The sets are checked before the
/// challenger is called, so a refused call leaves it as it was.
///
/// # Errors
/// [`Error::Shape`] unless there are $n$ sets; [`Error::AcceptSet`] for a malformed set;
/// [`Error::NoCommonIteration`] if no iteration is in every set (the verifier outputs $\bot$).
pub fn verifier_iteration<const K: usize, H: Challenger<K>>(
    ctx: &Context<K>,
    ch: &mut H,
    accept_sets: &[Vec<u64>],
) -> Result<usize> {
    let j_star = select_iteration(ctx, accept_sets)?;
    ch.absorb_words(LABEL_ACCEPT, accept_sets);
    j_star.ok_or(Error::NoCommonIteration {
        iterations: ctx.layout.rej_iterations,
    })
}

/// The linear relation $\mathbf M \mathbf z' = \mathbf y$ over $\mathcal R\_q$ produced by the
/// ring-to-linear step (`5_Building_Blocks.tex`:167-196).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearRelation {
    /// $\mathbf M \in \mathcal R\_q^{n' \times (\mu - 1)}$, dense and row-major (lines 170-176).
    pub m: Matrix,
    /// $\mathbf y = (\mathbf v, \mathbf u, 0, 0, \mathbf 0) \in \mathcal R\_q^{n'}$
    /// (lines 187-193).
    pub y: Vec<Poly>,
}

/// $\mathbf M$ and $\mathbf y$ from public data (`5_Building_Blocks.tex`:170-193,
/// `fig:ring-to-linear-2`): the public vectors of $\Pi\_{\mathsf{conv}}$ ($\mathbf b$ and
/// $\mathbf a$ in $\mathcal R\_q$), $\mathbf v$, $\mathbf c$, the selected iteration $j^\ast$ and
/// the commitment $\mathbf u$; after checking $\mathbf c \in \mathcal C^{2^r}$
/// ([`check_challenges`]) and the shapes and canonical residues of $\mathbf v$ and $\mathbf u$.
/// [`build_m`] and [`build_y`] of `hachi-co-core` build them.
///
/// # Errors
/// [`Error::Core`] if the public vectors have the wrong shape or $j^\ast \ge L$; [`Error::Shape`],
/// [`Error::NonCanonical`] or [`Error::ChallengeNotInC`] for malformed proof values.
pub fn linear_relation<const K: usize>(
    ctx: &Context<K>,
    pv: &PublicVectors<K>,
    v: &[Poly],
    c: &[Vec<i64>],
    j_star: usize,
    u_com: &[Poly],
) -> Result<LinearRelation> {
    check_relation(ctx, pv, v, c, u_com)?;
    Ok(LinearRelation {
        m: build_m(ctx, &pv.b_ring, &pv.a_ring, c, j_star)?,
        y: build_y(ctx, v, u_com)?,
    })
}

/// The checks of [`linear_relation`] without forming $\mathbf M$ and $\mathbf y$: the shapes of the
/// public vectors, $\mathbf c \in \mathcal C^{2^r}$ ([`check_challenges`]), and the shapes and
/// canonical residues of $\mathbf v$ and $\mathbf u$; for a prover that then reads $\mathbf M$ one row
/// at a time ([`hachi_co_core::MRows`], which checks $j^\ast \lt L$ and the other inputs of
/// [`build_m`]).
///
/// # Errors
/// As [`linear_relation`], except for $j^\ast$.
pub fn check_relation<const K: usize>(
    ctx: &Context<K>,
    pv: &PublicVectors<K>,
    v: &[Poly],
    c: &[Vec<i64>],
    u_com: &[Poly],
) -> Result<()> {
    let l = &ctx.layout;
    pv.validate(l)?;
    check_challenges(ctx, c)?;
    check_polys("v", v, l.n_d, l.d, ctx.q())?;
    check_polys("u (commitment)", u_com, l.n_b, l.d, ctx.q())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    //! The verifier's Fiat-Shamir calls against the provers' calls, and the selection of
    //! $j^\ast$ at its edges, at toy-n2 with public parameters from the seed `[7; 32]` and
    //! Fiat-Shamir on the domain [`DOMAIN`].
    use super::*;
    use crate::accept::encode_accept_set;
    use hachi_co_core::FsChallenger;
    use hachi_co_params::{derive, presets::toy_choices};

    /// The extension degree of the toy presets.
    const K: usize = 4;
    const DOMAIN: &[u8] = b"hachi-co-r2l/verifier/unit-tests";

    fn context() -> Context<K> {
        let (params, _) = derive(&toy_choices(2)).unwrap();
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
    fn the_verifier_makes_the_provers_calls() {
        let ctx = context();
        let l = &ctx.layout;
        let (ch, v) = inputs(&ctx);
        let mut verifier = ch.clone();
        let c = verifier_challenges(&ctx, &mut verifier, &v).unwrap();
        let mut provers = ch;
        Challenger::<K>::absorb_polys(&mut provers, LABEL_V, &v);
        let expected =
            Challenger::<K>::r2l_challenges(&mut provers, l.n_blocks, l.d, ctx.omega()).unwrap();
        assert_eq!(c, expected);
        check_challenges(&ctx, &c).unwrap();
        // The sets: iteration 3 is the first in both.
        let set = |js: &[usize]| {
            let flags: Vec<bool> = (0..l.rej_iterations).map(|j| js.contains(&j)).collect();
            encode_accept_set(&ctx, &flags).unwrap()
        };
        let sets = vec![set(&[0, 3, 5]), set(&[1, 3, 4, 5])];
        assert_eq!(verifier_iteration(&ctx, &mut verifier, &sets), Ok(3));
        Challenger::<K>::absorb_words(&mut provers, LABEL_ACCEPT, &sets);
        assert_eq!(verifier, provers);
    }

    #[test]
    fn malformed_or_disjoint_sets_are_refused() {
        let ctx = context();
        let l = &ctx.layout;
        assert!(l.rej_iterations < 64, "the toys have L = 16");
        let (ch, _) = inputs(&ctx);
        let full = (1u64 << l.rej_iterations) - 1;
        let cases: [(Vec<Vec<u64>>, Error); 4] = [
            (
                vec![vec![1 << l.rej_iterations], vec![full]],
                Error::AcceptSet {
                    party: 1,
                    what: "it names an iteration at or beyond L",
                },
            ),
            (
                vec![vec![full], vec![full, 0]],
                Error::AcceptSet {
                    party: 2,
                    what: "it does not have ceil(L/64) words",
                },
            ),
            (
                vec![vec![full]],
                Error::Shape {
                    what: "accept sets (one per prover)",
                    expected: 2,
                    actual: 1,
                },
            ),
            (
                vec![vec![0b0101], vec![0b1010]],
                Error::NoCommonIteration {
                    iterations: l.rej_iterations,
                },
            ),
        ];
        for (sets, expected) in cases {
            let mut used = ch.clone();
            assert_eq!(verifier_iteration(&ctx, &mut used, &sets), Err(expected));
        }
        // A malformed set leaves the challenger as it was; the empty intersection is decided
        // after the absorb, like the provers' (both stop there).
        let mut used = ch.clone();
        assert!(
            verifier_iteration(&ctx, &mut used, &[vec![full << 1 | 1 << 40], vec![full]]).is_err()
        );
        assert_eq!(used, ch);
        // The last iteration is selected when it is the only common one.
        let last = 1u64 << (l.rej_iterations - 1);
        let mut used = ch.clone();
        assert_eq!(
            verifier_iteration(&ctx, &mut used, &[vec![last | 1], vec![last]]),
            Ok(l.rej_iterations - 1)
        );
    }
}
