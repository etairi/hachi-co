//! The driver of the ring-to-linear step for one party (`5_Building_Blocks.tex`:114-196), with the
//! $L$ parallel iterations of the rejection sampling of `fig:ring-to-linear` (since the draft's
//! commit `1096ea2`, in place of the earlier attempts of Fiat-Shamir with aborts).

use crate::accept::{check_accept_set, encode_accept_set, select_iteration};
use crate::local::{
    check_challenges, check_len, check_polys, d_slice, fah_d_input, fold, mask_groups, owned_w,
    respond, response_len, sample_mask_images,
};
use crate::prover1::{norm_check, z_hat, z_of_sum};
use crate::share::{y_share, z_prime_share};
use crate::{COMM_ACCEPT, COMM_FAH_D, COMM_FAH_Z, Error, LABEL_ACCEPT, LABEL_V, Result};
use hachi_co_core::wire::{flatten_polys, unflatten_polys};
use hachi_co_core::{
    Challenger, Context, LinearShare, PartyRandomness, PartyState, Poly, PublicVectors,
};
use hachi_co_mpc::Comm;
use rayon::prelude::*;

/// What one party keeps from the ring-to-linear step: what the sumcheck (WP10) and the PCS
/// composition (WP11) need.
///
/// The proof carries `v` and `accept_sets` (plan, finding F5: never $\mathbf z\_i$,
/// $\hat{\mathbf z}$, $\mathbf r\_i^j$ or $\boldsymbol\rho^v\_i$); the verifier re-derives `c` and
/// `j_star`. `share` is the party's private input to the sumcheck.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R2lOutput {
    /// The first message $\mathbf v \in \mathcal R\_q^{n\_D}$, which commits to the images of
    /// all $L$ masks of every prover.
    pub v: Vec<Poly>,
    /// The challenges $\mathbf c \in \mathcal C^{2^r}$.
    pub c: Vec<Vec<i64>>,
    /// The provers' sets $R\_1, \dots, R\_n$ of accepted iterations ([`crate::accept`]),
    /// `accept_sets[p - 1]` of prover $p$, as broadcast.
    pub accept_sets: Vec<Vec<u64>>,
    /// $j^\ast$, the smallest iteration in every set (0-indexed, the draft's $j^\ast + 1$).
    pub j_star: usize,
    /// The party's shares $\langle \mathbf z' \rangle\_p$ and $\lbrace \mathbf y \rbrace\_p$
    /// ([`crate::z_prime_share`], [`crate::y_share`]), with
    /// $\mathbf M \sum\_p \langle \mathbf z' \rangle\_p = \sum\_p \lbrace \mathbf y \rbrace\_p$
    /// for $\mathbf M$ of `c` and `j_star` ([`crate::linear_relation`]).
    pub share: LinearShare,
}

/// The ring-to-linear step (`5_Building_Blocks.tex`:114-196 and `fig:ring-to-linear`,
/// `fig:ring-to-linear-2` of the draft since its commit `1096ea2`) for the party `comm.id()`.
///
/// Inputs: the party's state after Commit, the public vectors of $\Pi\_{\mathsf{conv}}$, its
/// share $\langle u \rangle\_p$ of $u$ from $\Pi\_{\mathsf{conv}}$, and the commitment
/// $\mathbf u \in \mathcal R\_q^{n\_B}$. Every party calls it with the same `ctx`, `pv` and
/// `u_com` and its own state, randomness and challenger replica.
///
/// 1. [`owned_w`]: $w\_i$, $\hat{\mathbf w}\_i$ of the owned blocks (step 2);
/// 2. [`sample_masks`]: the $L$ masks $\mathbf r\_p^j$, one $\boldsymbol\rho^v\_p$, and the images
///    of every mask (step 3);
/// 3. $\mathcal F\_{\mathsf{AH}}(\mathbf D)$ on the [`d_slice`] ([`fah_d_input`],
///    [`Comm::fah`] with label [`COMM_FAH_D`]) gives $\mathbf v$ (step 4); $\mathbf v$ is
///    absorbed under [`LABEL_V`];
/// 4. $\mathbf c$ = [`Challenger::r2l_challenges`]`(2^r, d, ω)`, checked by
///    [`check_challenges`];
/// 5. [`fold`] once and [`respond`] ($\mathsf{Rej}$) with each of the $L$ masks; the masks and the
///    fold are wiped right after (step 5);
/// 6. the set $R\_p$ of accepted iterations is broadcast ([`Comm::all_gather`], label
///    [`COMM_ACCEPT`]) and the $n$ sets are absorbed under [`LABEL_ACCEPT`]; every party computes
///    $j^\ast$ = [`select_iteration`] and stops with [`Error::NoCommonIteration`] if no iteration
///    is in every set (step 6). Only then is a response sent, and only that of $j^\ast$: the
///    others are wiped (plan, finding F5);
/// 7. the provers sum the responses $\mathbf z\_p^{j^\ast}$ with $\mathcal F\_{\mathsf{AH}}(1^n)$
///    ([`Comm::fah`], label [`COMM_FAH_Z`]); every prover runs [`z_of_sum`] and [`norm_check`]
///    and stops with [`Error::NormCheck`] if the check fails, and prover 1 runs [`z_hat`]
///    (`fig:ring-to-linear-2`, steps 8-10, where every prover checks since the draft's commit
///    `0337dfb`, and before it only prover 1 did). The check never fails for honest provers
///    (line 207);
/// 8. every party assembles its [`LinearShare`] ([`z_prime_share`], [`y_share`],
///    steps 10-14).
///
/// **Choices** (the draft is silent): a received set that is malformed, or a party's own set
/// that comes back changed, is an error ([`Error::AcceptSet`], [`Error::Protocol`]).
///
/// **Secrets.** The masks are used only by [`sample_masks`] (draws and images) and [`respond`]
/// ($\mathsf{Rej}$), and are wiped right after $\mathsf{Rej}$, as are the fold and every
/// response but that of $j^\ast$; the rest of the [`MaskSecrets`](crate::MaskSecrets) is wiped
/// when it is dropped at the end of the step. The broadcast sets depend only on the
/// rejection-sampling outcomes, which are independent of the witness
/// (`lem:rejection-sampling`). See the crate documentation for what is and is not constant time.
///
/// # Errors
/// [`Error::PartyMismatch`] if `comm` disagrees with the context or the state (number of
/// parties, party index, modulus); [`Error::Core`], [`Error::Shape`] or [`Error::NonCanonical`]
/// for malformed inputs; the errors of the randomness source, the challenger and the
/// communication layer; [`Error::ChallengeNotInC`] if the challenger returns a challenge outside
/// $\mathcal C$; [`Error::AcceptSet`] or [`Error::Protocol`] as above;
/// [`Error::NoCommonIteration`]; [`Error::NormCheck`] as above. At prover 1 also the errors of
/// [`z_hat`], which cannot occur after the check when
/// $(\lceil b/2 \rceil - 1)(b^\tau - 1)/(b - 1) \ge \beta$, as the draft's $\tau$ ensures (the
/// others then fail at their next collective call).
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
    let mut secrets = sample_mask_images(ctx, pv, rng)?;
    let local = fah_d_input(ctx, &d_slice(ctx, p, &w, &secrets)?)?;
    let v = unflatten_polys(&comm.fah(COMM_FAH_D, &flatten_polys(&local))?, l.n_d, l.d)?;
    check_polys("v (output of F_AH(D))", &v, l.n_d, l.d, q)?;
    ch.absorb_polys(LABEL_V, &v);
    let c = ch.r2l_challenges(l.n_blocks, l.d, ctx.omega())?;
    check_challenges(ctx, &c)?;
    let mut folded = fold(ctx, state, &c)?;
    let n = response_len(ctx);
    // The rest uses the fold, which is wiped once z_p is formed, on errors as well.
    let selected = (|| {
        // One Rej per mask, each mask read again from its stream (`PartyRandomness::masks_again`)
        // a group at a time, in parallel: only whether a response is accepted is kept, the
        // accepted ones being wiped, and the selected one is formed again below.
        let mut accepted = Vec::with_capacity(l.rej_iterations);
        for range in mask_groups(l.rej_iterations) {
            let mut masks = rng.masks_again(range.clone(), ctx.gamma(), n)?;
            check_len(
                "masks r_p^j read again (one per iteration)",
                range.len(),
                masks.len(),
            )?;
            let verdicts = masks
                .par_iter()
                .map(|mask| {
                    Ok(respond(ctx, &folded, mask)?.is_some_and(|mut z| {
                        crate::ct::wipe(&mut z);
                        true
                    }))
                })
                .collect::<Result<Vec<bool>>>();
            for mask in &mut masks {
                crate::ct::wipe(mask);
            }
            accepted.extend(verdicts?);
        }
        let own = encode_accept_set(ctx, &accepted)?;
        let accept_sets = comm.all_gather(COMM_ACCEPT, &own)?;
        if accept_sets.len() != l.n {
            return Err(Error::Protocol(format!(
                "the broadcast of the accept sets returned {} sets for n = {}",
                accept_sets.len(),
                l.n
            )));
        }
        for (p0, set) in accept_sets.iter().enumerate() {
            check_accept_set(ctx, p0 + 1, set)?;
        }
        if accept_sets[p - 1] != own {
            return Err(Error::Protocol(format!(
                "party {p}'s own set of accepted iterations came back changed"
            )));
        }
        ch.absorb_words(LABEL_ACCEPT, &accept_sets);
        let j_star = select_iteration(ctx, &accept_sets)?;
        // The selected response, formed again from its mask; every other one is never sent
        // (plan, finding F5).
        let (j_star, z_p) = match j_star {
            None => {
                return Err(Error::NoCommonIteration {
                    iterations: l.rej_iterations,
                });
            }
            Some(j) if accepted[j] => {
                let mut mask = rng.masks_again(j..j + 1, ctx.gamma(), n)?;
                let z = match mask.first() {
                    Some(mask) => respond(ctx, &folded, mask)?,
                    None => None,
                };
                for mask in &mut mask {
                    crate::ct::wipe(mask);
                }
                let z = z.ok_or_else(|| {
                    Error::Protocol(format!(
                        "iteration {j} was accepted, but its response formed again was not"
                    ))
                })?;
                (j, z)
            }
            Some(j) => {
                return Err(Error::Protocol(format!(
                    "iteration {j} is in every set, but party {p} rejected it"
                )));
            }
        };
        Ok((accept_sets, j_star, z_p))
    })();
    crate::ct::wipe(&mut folded);
    secrets.wipe_masks();
    let (accept_sets, j_star, mut z_p) = selected?;
    let mut local: Vec<u64> = z_p.iter().map(|x| ctx.ring.zq().from_i64(*x)).collect();
    let sum = comm.fah(COMM_FAH_Z, &local);
    crate::ct::wipe(&mut z_p);
    local.fill(0);
    let z_hat_digits = after_sum(ctx, p, &sum?)?;
    let z_prime = z_prime_share(ctx, state, &w, &secrets, z_hat_digits.as_deref(), u_share)?;
    let y = y_share(ctx, p, &v, u_com)?;
    Ok(R2lOutput {
        v,
        c,
        accept_sets,
        j_star,
        share: LinearShare { z_prime, y },
    })
}

/// The steps after $\mathcal F\_{\mathsf{AH}}(1^n)$ (`fig:ring-to-linear-2`, steps 9-10): every
/// prover forms $\mathbf z$ ([`z_of_sum`]) and checks $\Vert \mathbf z \Vert\_\infty \le \beta$
/// ([`norm_check`]), and prover 1 computes $\hat{\mathbf z}$ ([`z_hat`]), returned as `Some` at
/// prover 1 and `None` at the others. $\mathbf z$ itself is wiped (best effort), as
/// $\hat{\mathbf z}$ carries it on.
fn after_sum<const K: usize>(ctx: &Context<K>, p: usize, sum: &[u64]) -> Result<Option<Vec<Poly>>> {
    let mut z = z_of_sum(ctx, sum)?;
    let digits = norm_check(ctx, &z).and_then(|()| (p == 1).then(|| z_hat(ctx, &z)).transpose());
    crate::ct::wipe(&mut z);
    digits
}
