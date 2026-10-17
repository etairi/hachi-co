//! The verifier of the PCS with Fiat-Shamir (`docs/m1-spec.md`, WP11): $\Pi\_{\mathsf{conv}}$'s
//! check, the challenges and the linear relation of the ring-to-linear step, and the sumcheck
//! verifier, composed.

use crate::error::VerifyError;
use crate::proof::EvalProof;
use crate::statement::{EVAL_DOMAIN, absorb_statement, check_statement};
use hachi_co_conv::{LABEL_E_U, e_matrix, public_vectors};
use hachi_co_core::{Challenger, Context, Ext, FsChallenger, Poly};
use hachi_co_r2l::{linear_relation, verifier_challenges};
use hachi_co_sumcheck::{FinalClaims, PublicInput};

/// The verifier with Fiat-Shamir: [`verify_with`] on a fresh [`FsChallenger`] with the domain
/// [`EVAL_DOMAIN`].
///
/// # Errors
/// As [`verify_with`].
pub fn verify<const K: usize>(
    ctx: &Context<K>,
    u: &[Poly],
    x: &[Ext<K>],
    y: &Ext<K>,
    proof: &EvalProof<K>,
) -> Result<FinalClaims<K>, VerifyError> {
    let mut ch = FsChallenger::new(EVAL_DOMAIN);
    verify_with(ctx, &mut ch, u, x, y, proof)
}

/// The verifier of $\mathsf{Eval}$ for the statement "the polynomial committed in $\mathbf u$
/// takes the value $y$ at $\mathbf x$", with the coins drawn from `ch`, which must be fresh (an
/// [`FsChallenger`]`::new(`[`EVAL_DOMAIN`]`)` for a real proof, [`verify`]; a replay challenger in
/// fixture tests). It makes exactly the challenger calls of the provers' [`eval_with`] for the
/// accepted attempt:
///
/// 1. [`check_statement`] and [`EvalProof::validate`] (shapes, canonical residues,
///    $a^\ast$ below the cap), then [`absorb_statement`];
/// 2. $\Pi\_{\mathsf{conv}}$'s check $E(u) \in \mathcal R\_q^H$ and $y = E(u)$
///    ([`hachi_co_conv::check`], `field-to-ring-rok.tex`:163-164); $E(u)$ is absorbed under
///    [`LABEL_E_U`];
/// 3. the public vectors of $\mathbf x$ ([`public_vectors`]), the challenges of the accepted
///    attempt ([`verifier_challenges`]: `begin_attempt(a*)`, absorb $\mathbf v$, draw
///    $\mathbf c$, check $\mathbf c \in \mathcal C^{2^r}$), and $\mathbf M$, $\mathbf y$
///    ([`linear_relation`], `5_Building_Blocks.tex`:157-179);
/// 4. the matrix of $E$ at the point $v$ ([`e_matrix`]) and $y$ embedded in $\mathcal R\_q^H$
///    (`ref/README.md`, choice 12);
/// 5. the sumcheck verifier ([`hachi_co_sumcheck::verify`], `5_Building_Blocks.tex`:276-334):
///    every round check and both final checks.
///
/// Returns the point $\mathbf a$ and the $3n$ final claims $g\_{0,p}(\mathbf a)$,
/// $g\_{1,p}(\mathbf a)$, $\langle \widetilde w \rangle\_p(\mathbf a)$ (party by party). **They
/// are unproven:** in milestone 1 the base PCS is absent, and acceptance means something only
/// together with commitments to these polynomials and openings of the claims; without them a
/// proof can be made for any statement (see the crate documentation). Never panics on the
/// statement or the proof.
///
/// # Errors
/// The first failing step: [`VerifyError::Statement`], [`VerifyError::Proof`],
/// [`VerifyError::Conv`], [`VerifyError::PublicVectors`], [`VerifyError::RingToLinear`],
/// [`VerifyError::Sumcheck`].
///
/// [`eval_with`]: crate::eval_with
pub fn verify_with<const K: usize, H: Challenger<K>>(
    ctx: &Context<K>,
    ch: &mut H,
    u: &[Poly],
    x: &[Ext<K>],
    y: &Ext<K>,
    proof: &EvalProof<K>,
) -> Result<FinalClaims<K>, VerifyError> {
    check_statement(ctx, u, x, y)?;
    proof.validate(ctx)?;
    absorb_statement(ch, ctx, u, x, y);
    hachi_co_conv::check(ctx, y, &proof.e_u).map_err(VerifyError::Conv)?;
    ch.absorb_polys(LABEL_E_U, core::slice::from_ref(&proof.e_u));
    let pv = public_vectors(ctx, x).map_err(VerifyError::PublicVectors)?;
    let c =
        verifier_challenges(ctx, ch, proof.attempt, &proof.v).map_err(VerifyError::RingToLinear)?;
    let rel = linear_relation(ctx, &pv, &proof.v, &c, u).map_err(VerifyError::RingToLinear)?;
    let e = e_matrix(ctx, &pv.v_point).map_err(VerifyError::PublicVectors)?;
    let y_ring = ctx.sub.embed(y);
    let public = PublicInput::new(ctx, &rel.m, &rel.y, &e, &y_ring);
    hachi_co_sumcheck::verify(ctx, &public, &proof.sumcheck, ch).map_err(VerifyError::Sumcheck)
}
