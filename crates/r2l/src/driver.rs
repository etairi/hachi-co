//! The driver of the ring-to-linear step for one party (`5_Building_Blocks.tex`:114-196), with
//! the attempts of Fiat-Shamir with aborts (`ref/README.md`, choice 5; plan
//! `docs/hachi-co-plan.md`, finding F5).

use crate::local::{
    check_challenges, check_polys, d_slice, fah_d_input, fold, owned_w, respond, sample_attempt,
};
use crate::prover1::{norm_check, z_hat, z_of_sum};
use crate::share::{y_share, z_prime_share};
use crate::{
    COMM_ABORT, COMM_FAH_D, COMM_FAH_Z, COMM_PROVER1, Error, LABEL_V, MAX_ATTEMPTS, Result,
};
use hachi_co_core::wire::{flatten_polys, unflatten_polys};
use hachi_co_core::{
    Challenger, Context, LinearShare, PartyRandomness, PartyState, Poly, PublicVectors,
};
use hachi_co_mpc::Comm;

/// The public record of one attempt, as this party saw it (for tests and diagnostics).
///
/// It holds only what the protocol makes public: the first message $\mathbf v$ (sent to the
/// verifier in every attempt, line 133), the challenges, this party's accept bit (a rejecting
/// prover announces the abort, line 141) and the agreed abort. Masks, $\boldsymbol\rho$, folds,
/// norms and responses are never recorded: a rejected mask must never be revealed (plan,
/// finding F5). Tests reproduce those through the party-local functions instead.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttemptRecord {
    /// The attempt index $a$.
    pub attempt: u64,
    /// $\mathbf v \in \mathcal R\_q^{n\_D}$, the output of $\mathcal F\_{\mathsf{AH}}(\mathbf D)$
    /// (`5_Building_Blocks.tex`:132).
    pub v: Vec<Poly>,
    /// $\mathbf c \in \mathcal C^{2^r}$ (`5_Building_Blocks.tex`:138), $2^r$ vectors of $d$
    /// signed coefficients.
    pub c: Vec<Vec<i64>>,
    /// Whether this party's $\mathsf{Rej}$ accepted (line 140).
    pub accepted: bool,
    /// Whether some party rejected, so that the attempt aborted (line 141).
    pub aborted: bool,
}

/// What one party keeps from the ring-to-linear step: what the sumcheck (WP10) and the PCS
/// composition (WP11) need.
///
/// The proof carries only `accepted_attempt`, `v` and `c` (plan, finding F5: never
/// $\mathbf z\_i$, $\hat{\mathbf z}$, $\mathbf r\_i$ or $\boldsymbol\rho^v\_i$); `share` is the
/// party's private input to the sumcheck.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R2lOutput {
    /// The index $a^\ast$ of the accepted attempt (the first attempt in which no party
    /// aborted). A Fiat-Shamir verifier calls
    /// [`Challenger::begin_attempt`]`(a*)` before absorbing $\mathbf v$
    /// ([`crate::verifier_challenges`]).
    pub accepted_attempt: u64,
    /// The accepted attempt's first message $\mathbf v \in \mathcal R\_q^{n\_D}$.
    pub v: Vec<Poly>,
    /// The accepted attempt's challenges $\mathbf c \in \mathcal C^{2^r}$.
    pub c: Vec<Vec<i64>>,
    /// The party's shares $\langle \mathbf z' \rangle\_p$ and $\lbrace \mathbf y \rbrace\_p$
    /// ([`crate::z_prime_share`], [`crate::y_share`]), with
    /// $\mathbf M \sum\_p \langle \mathbf z' \rangle\_p = \sum\_p \lbrace \mathbf y \rbrace\_p$
    /// for $\mathbf M$ of `c` ([`crate::linear_relation`]).
    pub share: LinearShare,
    /// One record per attempt, the aborted ones first and the accepted one last.
    pub attempts: Vec<AttemptRecord>,
}

/// [`run_with_cap`] with the cap [`MAX_ATTEMPTS`].
///
/// # Errors
/// As [`run_with_cap`].
#[allow(clippy::too_many_arguments)] // the signature of docs/m1-spec.md, WP9
pub fn run<const K: usize, C: Comm, R: PartyRandomness<K>, H: Challenger<K>>(
    ctx: &Context<K>,
    comm: &mut C,
    rng: &mut R,
    ch: &mut H,
    state: &PartyState<K>,
    pv: &PublicVectors<K>,
    u_share: &Poly,
    u_com: &[Poly],
) -> Result<R2lOutput> {
    run_with_cap(ctx, comm, rng, ch, state, pv, u_share, u_com, MAX_ATTEMPTS)
}

/// The ring-to-linear step (`5_Building_Blocks.tex`:114-196) for the party `comm.id()`, with up
/// to `max_attempts` attempts.
///
/// Inputs: the party's state after Commit, the public vectors of $\Pi\_{\mathsf{conv}}$, its
/// share $\langle u \rangle\_p$ of $u$ from $\Pi\_{\mathsf{conv}}$, and the commitment
/// $\mathbf u \in \mathcal R\_q^{n\_B}$. Every party calls it with the same `ctx`, `pv` and
/// `u_com` and its own state, randomness and challenger replica.
///
/// Step 1 ($w\_i$, $\hat{\mathbf w}\_i$, [`owned_w`]) runs once. Then, for $a = 0, 1, \dots$:
/// 1. [`Challenger::begin_attempt`]`(a)`;
/// 2. [`sample_attempt`]: fresh $\mathbf r\_p$, $\boldsymbol\rho^v\_p$ (index $a$) and their images
///    (lines 122-126);
/// 3. $\mathcal F\_{\mathsf{AH}}(\mathbf D)$ on the [`d_slice`] ([`fah_d_input`],
///    [`Comm::fah`] with label [`COMM_FAH_D`]) gives $\mathbf v$ (lines 127-133); $\mathbf v$
///    is absorbed under [`LABEL_V`];
/// 4. $\mathbf c$ = [`Challenger::r2l_challenges`]`(a, 2^r, d, ω)`, checked by
///    [`check_challenges`] (lines 134-138);
/// 5. [`fold`] and [`respond`] ($\mathsf{Rej}$, line 140); the accept bits are agreed with
///    [`Comm::any`] (label [`COMM_ABORT`]) before any response is sent (plan, finding F5). If
///    some party rejected, the attempt aborts (line 141): its mask is wiped, its other values
///    dropped, and the next attempt starts;
/// 6. otherwise the provers sum the responses with $\mathcal F\_{\mathsf{AH}}(1^n)$
///    ([`Comm::fah`], label [`COMM_FAH_Z`]), and prover 1 runs [`z_of_sum`], [`norm_check`] and
///    [`z_hat`] (lines 142-146) and announces whether these succeeded ([`Comm::any`], label
///    [`COMM_PROVER1`]);
/// 7. every party assembles its [`LinearShare`] ([`z_prime_share`], [`y_share`],
///    lines 147-193).
///
/// **Choices** (the draft is silent): a failure of prover 1's check is not retried (it never
/// happens for honest provers, line 207) but ends the step with [`Error::NormCheck`] at prover 1
/// and [`Error::Prover1Abort`] elsewhere; attempts are capped (plan, finding F5). The verifier
/// refuses an attempt index at or above [`MAX_ATTEMPTS`] ([`crate::verifier_challenges`]), so
/// with a larger `max_attempts` an output accepted at such an index does not verify.
///
/// **Secrets.** The mask is used only by [`sample_attempt`] (draw and images) and [`respond`]
/// ($\mathsf{Rej}$), and is wiped right after $\mathsf{Rej}$, as are the fold and the response of
/// an aborted attempt; the rest of the attempt's [`AttemptSecrets`](crate::AttemptSecrets) is wiped when it is dropped at
/// the end of the attempt. Nothing in the driver branches on secret values. See the crate
/// documentation for what is and is not constant time.
///
/// # Errors
/// [`Error::PartyMismatch`] if `comm` disagrees with the context or the state (number of
/// parties, party index, modulus); [`Error::Core`], [`Error::Shape`] or [`Error::NonCanonical`]
/// for malformed inputs; the errors of the randomness source, the challenger and the
/// communication layer; [`Error::ChallengeNotInC`] if the challenger returns a challenge outside
/// $\mathcal C$; [`Error::Protocol`] if `comm` breaks its contract; [`Error::NormCheck`] or
/// [`Error::Prover1Abort`] as above; [`Error::TooManyAttempts`] if all `max_attempts` attempts
/// abort.
#[allow(clippy::too_many_arguments)] // run's arguments and the cap
pub fn run_with_cap<const K: usize, C: Comm, R: PartyRandomness<K>, H: Challenger<K>>(
    ctx: &Context<K>,
    comm: &mut C,
    rng: &mut R,
    ch: &mut H,
    state: &PartyState<K>,
    pv: &PublicVectors<K>,
    u_share: &Poly,
    u_com: &[Poly],
    max_attempts: u64,
) -> Result<R2lOutput> {
    let l = &ctx.layout;
    let q = ctx.q();
    let p = state.party;
    if comm.n() != l.n {
        return Err(Error::PartyMismatch {
            what: "number of parties (context, communication layer)",
            expected: l.n,
            actual: comm.n(),
        });
    }
    if comm.id() != p {
        return Err(Error::PartyMismatch {
            what: "party index (state, communication layer)",
            expected: p,
            actual: comm.id(),
        });
    }
    if comm.modulus() != q {
        return Err(Error::Parameter(format!(
            "the communication layer adds modulo {}, the context has q = {q}",
            comm.modulus()
        )));
    }
    state.validate(l)?;
    pv.validate(l)?;
    check_polys("<u>_p", core::slice::from_ref(u_share), 1, l.d, q)?;
    check_polys("u (commitment)", u_com, l.n_b, l.d, q)?;

    let w = owned_w(ctx, pv, state)?;
    let mut attempts = Vec::new();
    for attempt in 0..max_attempts {
        ch.begin_attempt(attempt)?;
        let mut secrets = sample_attempt(ctx, pv, rng, attempt)?;
        let local = fah_d_input(ctx, &d_slice(ctx, p, &w, &secrets)?)?;
        let v = unflatten_polys(&comm.fah(COMM_FAH_D, &flatten_polys(&local))?, l.n_d, l.d)?;
        check_polys("v (output of F_AH(D))", &v, l.n_d, l.d, q)?;
        ch.absorb_polys(LABEL_V, &v);
        let c = ch.r2l_challenges(attempt, l.n_blocks, l.d, ctx.omega())?;
        check_challenges(ctx, &c)?;
        let mut folded = fold(ctx, state, &c)?;
        let response = respond(ctx, &folded, &secrets.mask);
        crate::ct::wipe(&mut folded);
        secrets.wipe_mask();
        let response = response?;
        let accepted = response.is_some();
        let aborted = comm.any(COMM_ABORT, !accepted)?;
        attempts.push(AttemptRecord {
            attempt,
            v: v.clone(),
            c: c.clone(),
            accepted,
            aborted,
        });
        let mut z_p = match (aborted, response) {
            (true, None) => continue,
            (true, Some(mut z_p)) => {
                // Another party rejected: this response is never sent (plan, finding F5).
                crate::ct::wipe(&mut z_p);
                continue;
            }
            (false, Some(z_p)) => z_p,
            (false, None) => {
                return Err(Error::Protocol(format!(
                    "attempt {attempt}: party {p} rejected but the abort agreement reported none"
                )));
            }
        };
        let mut local: Vec<u64> = z_p.iter().map(|x| ctx.ring.zq().from_i64(*x)).collect();
        let sum = comm.fah(COMM_FAH_Z, &local)?;
        crate::ct::wipe(&mut z_p);
        local.fill(0);
        let at_prover1 = (p == 1).then(|| prover1_steps(ctx, &sum));
        let failed = matches!(at_prover1, Some(Err(_)));
        let some_failed = comm.any(COMM_PROVER1, failed)?;
        let z_hat_digits = match (at_prover1, some_failed) {
            (Some(Err(e)), _) => return Err(e),
            (_, true) if p == 1 => {
                return Err(Error::Protocol(
                    "prover 1 succeeded but the agreement reported a failure".into(),
                ));
            }
            (_, true) => return Err(Error::Prover1Abort),
            (Some(Ok(digits)), false) => Some(digits),
            (None, false) => None,
        };
        let z_prime = z_prime_share(ctx, state, &w, &secrets, z_hat_digits.as_deref(), u_share)?;
        let y = y_share(ctx, p, &v, u_com)?;
        return Ok(R2lOutput {
            accepted_attempt: attempt,
            v,
            c,
            share: LinearShare { z_prime, y },
            attempts,
        });
    }
    Err(Error::TooManyAttempts {
        attempts: max_attempts,
    })
}

/// Prover 1's steps after $\mathcal F\_{\mathsf{AH}}(1^n)$ (`5_Building_Blocks.tex`:142-146):
/// [`z_of_sum`], [`norm_check`] and [`z_hat`]; $\mathbf z$ itself is wiped (best effort), as
/// $\hat{\mathbf z}$ carries it on.
fn prover1_steps<const K: usize>(ctx: &Context<K>, sum: &[u64]) -> Result<Vec<Poly>> {
    let mut z = z_of_sum(ctx, sum)?;
    let digits = norm_check(ctx, &z).and_then(|()| z_hat(ctx, &z));
    crate::ct::wipe(&mut z);
    digits
}
