//! The verifier of the PCS with Fiat-Shamir (`docs/m1-spec.md`, WP11): $\Pi\_{\mathsf{conv}}$'s
//! step, the challenges and the linear relation of the ring-to-linear step, and the sumcheck
//! verifier with the openings of the base PCS, composed. As in the draft's $\mathsf{Eval}$ at
//! commit `c071784` (`6_Collaborative_Hachi.tex`, $\mathsf{Eval}$; its syntax
//! $\mathsf{Eval} \to (\bot, \dots, \bot; (y, b))$ in the definition of a collaborative PCS), the
//! verifier outputs the value $y$ together with acceptance: `Ok(y)` is $(y, 1)$.

use crate::error::VerifyError;
use crate::proof::EvalProof;
use crate::statement::{EVAL_DOMAIN, absorb_statement, check_statement};
use hachi_co_conv::{LABEL_E_U, e_matrix, public_vectors};
use hachi_co_core::{BasePcs, Challenger, Context, Ext, FsChallenger, Poly};
use hachi_co_r2l::{linear_relation, verifier_challenges};
use hachi_co_sumcheck::{FinalClaims, PublicInput};

/// The verifier with Fiat-Shamir: [`verify_with`] on a fresh [`FsChallenger`] with the domain
/// [`EVAL_DOMAIN`]. `Ok(y)` means that the proof is accepted and that the polynomial committed in
/// $\mathbf u$ takes the value $y$ at $\mathbf x$, with the openings checked by the base PCS
/// `pcs` (see the crate documentation, "What an accepted proof means").
///
/// # Errors
/// As [`verify_with`].
pub fn verify<const K: usize, P: BasePcs<K>>(
    ctx: &Context<K>,
    pcs: &P,
    u: &[Poly],
    x: &[Ext<K>],
    proof: &EvalProof<K>,
) -> Result<Ext<K>, VerifyError> {
    let mut ch = FsChallenger::new(EVAL_DOMAIN);
    verify_with(ctx, pcs, &mut ch, u, x, proof)
}

/// [`verify`], then a comparison of the value the proof establishes with an expected value $y$:
/// for callers that know which value to expect.
///
/// # Errors
/// As [`verify`], and [`VerifyError::ValueMismatch`] if the proof is valid for another value.
pub fn verify_value<const K: usize, P: BasePcs<K>>(
    ctx: &Context<K>,
    pcs: &P,
    u: &[Poly],
    x: &[Ext<K>],
    y: &Ext<K>,
    proof: &EvalProof<K>,
) -> Result<(), VerifyError> {
    let value = verify(ctx, pcs, u, x, proof)?;
    if value != ctx.field().reduce(y) {
        return Err(VerifyError::ValueMismatch);
    }
    Ok(())
}

/// The verifier of $\mathsf{Eval}$ for the statement $(\mathbf u, \mathbf x)$, with the base PCS
/// `pcs` (the provers' one) and the coins drawn from `ch`, which must be fresh (an
/// [`FsChallenger`]`::new(`[`EVAL_DOMAIN`]`)` for a real proof, [`verify`]; a replay challenger in
/// fixture tests): [`verify_claims_with`] without the claims. Returns the value $y$.
///
/// # Errors
/// As [`verify_claims_with`].
pub fn verify_with<const K: usize, P: BasePcs<K>, H: Challenger<K>>(
    ctx: &Context<K>,
    pcs: &P,
    ch: &mut H,
    u: &[Poly],
    x: &[Ext<K>],
    proof: &EvalProof<K>,
) -> Result<Ext<K>, VerifyError> {
    verify_claims_with(ctx, pcs, ch, u, x, proof).map(|(y, _)| y)
}

/// The verifier of [`verify_with`], returning the value $y$, the point $\mathbf a$ and the $3n$
/// final claims $g\_{0,p}(\mathbf a)$, $g\_{1,p}(\mathbf a)$, $\langle \widetilde w \rangle\_p(\mathbf a)$
/// (party by party) that the base PCS checked: for tests and diagnostics. It makes exactly the
/// challenger calls of the provers' [`eval_with`] for the accepted attempt:
///
/// 1. [`check_statement`] and [`EvalProof::validate`] (shapes, canonical residues,
///    $a^\ast$ below the cap, one base-PCS commitment and proof per prover and polynomial), then
///    [`absorb_statement`];
/// 2. the verifier's step of $\Pi\_{\mathsf{conv}}$ ($\Pi\_A$ at the draft's commit `c071784`):
///    $E(u) \in \mathcal R\_q^H$, read as $y \in \mathbb F\_{q^k}$ ([`hachi_co_conv::read_value`]);
///    $E(u)$ is absorbed under [`LABEL_E_U`];
/// 3. the public vectors of $\mathbf x$ ([`public_vectors`]), the challenges of the accepted
///    attempt ([`verifier_challenges`]: `begin_attempt(a*)`, absorb $\mathbf v$, draw
///    $\mathbf c$, check $\mathbf c \in \mathcal C^{2^r}$), and $\mathbf M$, $\mathbf y$
///    ([`linear_relation`], `5_Building_Blocks.tex`:170-193);
/// 4. the matrix of $E$ at the point $v$ ([`e_matrix`]) and $y$ embedded in $\mathcal R\_q^H$
///    (`ref/README.md`, choice 12);
/// 5. the sumcheck verifier with the base PCS ([`hachi_co_sumcheck::verify_claims`],
///    `5_Building_Blocks.tex`:308-366): every round check, the $3n$ openings (each commitment,
///    absorbed before the sumcheck's coins, opened to the final value at $\mathbf a$ under its
///    context) and both final checks.
///
/// Never panics on the statement or the proof.
///
/// # Errors
/// The first failing step: [`VerifyError::Statement`], [`VerifyError::Proof`],
/// [`VerifyError::Conv`], [`VerifyError::PublicVectors`], [`VerifyError::RingToLinear`],
/// [`VerifyError::Sumcheck`] (a failing round or final check,
/// [`Rejected`](hachi_co_sumcheck::VerifyError::Rejected), before a failing opening,
/// [`Opening`](hachi_co_sumcheck::VerifyError::Opening), which names the party and the
/// polynomial).
///
/// [`eval_with`]: crate::eval_with
pub fn verify_claims_with<const K: usize, P: BasePcs<K>, H: Challenger<K>>(
    ctx: &Context<K>,
    pcs: &P,
    ch: &mut H,
    u: &[Poly],
    x: &[Ext<K>],
    proof: &EvalProof<K>,
) -> Result<(Ext<K>, FinalClaims<K>), VerifyError> {
    check_statement(ctx, u, x)?;
    proof.validate(ctx)?;
    absorb_statement(ch, ctx, &pcs.params_digest(), u, x);
    let y = hachi_co_conv::read_value(ctx, &proof.e_u).map_err(VerifyError::Conv)?;
    ch.absorb_polys(LABEL_E_U, core::slice::from_ref(&proof.e_u));
    let pv = public_vectors(ctx, x).map_err(VerifyError::PublicVectors)?;
    let c =
        verifier_challenges(ctx, ch, proof.attempt, &proof.v).map_err(VerifyError::RingToLinear)?;
    let rel = linear_relation(ctx, &pv, &proof.v, &c, u).map_err(VerifyError::RingToLinear)?;
    let e = e_matrix(ctx, &pv.v_point).map_err(VerifyError::PublicVectors)?;
    let y_ring = ctx.sub.embed(&y);
    let public = PublicInput::new(ctx, &rel.m, &rel.y, &e, &y_ring);
    let claims = hachi_co_sumcheck::verify_claims(ctx, pcs, &public, &proof.sumcheck, ch)
        .map_err(VerifyError::Sumcheck)?;
    Ok((y, claims))
}
