//! The provers' side of the PCS: $\mathsf{Setup}$, $\mathsf{Commit}$ and $\mathsf{Eval}$
//! (`6_Collaborative_Hachi.tex`:263-391, 408-419) with Fiat-Shamir.

use crate::error::Error;
use crate::proof::EvalProof;
use crate::statement::{EVAL_DOMAIN, absorb_statement, check_statement};
use hachi_co_conv::{ConvOutput, LABEL_E_U, e_matrix, pi_conv};
use hachi_co_core::{
    BasePcs, Challenger, Context, Ext, FsChallenger, PartyRandomness, PartyState, Poly,
};
use hachi_co_mpc::Comm;
use hachi_co_params::Params;
use hachi_co_r2l::{R2lOutput, linear_relation};
use hachi_co_sumcheck::SumcheckPrivate;

/// $\mathsf{Setup}$ (`6_Collaborative_Hachi.tex`:263-298) from derived parameters and a 32-byte
/// seed: [`Context::setup`], which expands $\mathbf A, \mathbf B, \mathbf D$ from the seed. The
/// base PCS (lines 289-292) is set up on its own, over Hachi-Co's $q$ and $k$: Hachi from the
/// same seed with [`HachiBase::setup`](crate::HachiBase::setup), or `ClearPcs` in the tests. It
/// is passed to [`eval`] and [`verify`](crate::verify).
///
/// # Errors
/// [`Error::Core`] as [`Context::setup`] (for example $K \ne k$).
pub fn setup<const K: usize>(params: Params, seed: &[u8; 32]) -> Result<Context<K>, Error> {
    Ok(Context::setup(params, seed)?)
}

/// $\mathsf{Commit}$ (`6_Collaborative_Hachi.tex`:300-391) for the party `comm.id()` with
/// initial share `f_init` $= \mathbf f\_p \in \mathbb F\_{q^k}^{2^\ell}$:
/// [`hachi_co_commit::commit`]. Returns the commitment $\mathbf u$ (the same at every party) and
/// the party's state $\mathsf{st}\_p$. Commit draws no coins.
///
/// # Errors
/// [`Error::Commit`] as [`hachi_co_commit::commit`].
pub fn commit<const K: usize, C: Comm, R: PartyRandomness<K>>(
    ctx: &Context<K>,
    comm: &mut C,
    rng: &mut R,
    f_init: &[Ext<K>],
) -> Result<(Vec<Poly>, PartyState<K>), Error> {
    Ok(hachi_co_commit::commit(ctx, comm, rng, f_init)?)
}

/// Everything one party has after [`eval_with`]: the proof (the same at every party) and the
/// party's private data (the tests check the claims against it). Apart from `proof`, it holds
/// the party's secrets (its share of $u$, of $\mathbf z'$ and of $\widetilde w$, its blinders):
/// keep it private, and do not log it (`Debug` shows everything).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalOutput<const K: usize> {
    /// The proof.
    pub proof: EvalProof<K>,
    /// $\Pi\_{\mathsf{conv}}$'s output: the public vectors, the party's share $\langle u
    /// \rangle\_p$ (private) and $E(u)$.
    pub conv: ConvOutput<K>,
    /// The ring-to-linear step's output: $a^\ast$, $\mathbf v$, $\mathbf c$, the record of every
    /// attempt, and the party's shares $\langle \mathbf z' \rangle\_p$, $\lbrace \mathbf y
    /// \rbrace\_p$ (private).
    pub r2l: R2lOutput,
    /// The party's private data of the sumcheck: the table of $\langle \widetilde w \rangle\_p$,
    /// the blinders $g\_{0,p}, g\_{1,p}$, the coins and the point $\mathbf a$.
    pub sumcheck: SumcheckPrivate<K>,
}

/// $\mathsf{Eval}$ for the party `comm.id()` with Fiat-Shamir: [`eval_with`] on a fresh
/// [`FsChallenger`] with the domain [`EVAL_DOMAIN`], keeping only the proof. Every party calls it
/// with the same base PCS `pcs`, its own network end, randomness and state and the same
/// statement $(\mathbf u, \mathbf x, y)$; every party obtains the same proof.
///
/// # Errors
/// As [`eval_with`].
#[allow(clippy::too_many_arguments)] // eval's arguments and the base PCS
pub fn eval<const K: usize, P: BasePcs<K>, C: Comm, R: PartyRandomness<K>>(
    ctx: &Context<K>,
    pcs: &P,
    comm: &mut C,
    rng: &mut R,
    state: &PartyState<K>,
    u: &[Poly],
    x: &[Ext<K>],
    y: &Ext<K>,
) -> Result<EvalProof<K>, Error> {
    let mut ch = FsChallenger::new(EVAL_DOMAIN);
    Ok(eval_with(ctx, pcs, comm, rng, &mut ch, state, u, x, y)?.proof)
}

/// $\mathsf{Eval}\langle \mathcal P\_1(\mathsf{st}\_1), \dots, \mathcal P\_n(\mathsf{st}\_n)
/// \rangle(\mathsf{pp}, \mathbf u, \mathbf x)$ (`6_Collaborative_Hachi.tex`:408-419) for the party
/// `comm.id()`, proving the claim $f(\mathbf x) = y$ for the polynomial committed in
/// $\mathbf u$, with the base PCS `pcs` (the same at every party and at the verifier) and the
/// coins drawn from `ch`, which must be fresh: an [`FsChallenger`]`::new(`[`EVAL_DOMAIN`]`)` for a
/// real proof ([`eval`]), or a replay challenger in fixture tests. Steps:
///
/// 1. [`check_statement`], then [`absorb_statement`] (parameters, $\mathbf A, \mathbf B,
///    \mathbf D$, the digest of `pcs`'s parameters, $\mathbf u$, $\mathbf x$, $y$);
/// 2. $\Pi\_{\mathsf{conv}}$ ([`hachi_co_conv::pi_conv`], line 411): the public vectors,
///    $\langle u \rangle\_p$ and $E(u)$; the provers check $y = E(u)$
///    ([`hachi_co_conv::check`]) and stop with [`Error::ClaimMismatch`] if it fails; $E(u)$ is
///    absorbed under [`LABEL_E_U`];
/// 3. the ring-to-linear step with its attempts ([`hachi_co_r2l::run`], line 417), which calls
///    `begin_attempt`, absorbs each attempt's $\mathbf v$ and draws $\mathbf c$ itself;
///    $\mathbf M$ and $\mathbf y$ from the accepted attempt ([`linear_relation`]);
/// 4. the sumcheck ([`hachi_co_sumcheck::prove`], line 419) with the matrix of $E$ at the point
///    $v$ ([`e_matrix`]), which makes its own absorbs and contexts, and in which each party
///    commits to $\langle \widetilde w \rangle\_p$, $g\_{0,p}$, $g\_{1,p}$ with `pcs` before the
///    coins and opens them at the end (`5_Building_Blocks.tex`:209-241, 314-318).
///
/// The proof holds $E(u)$, $a^\ast$, $\mathbf v$ and the sumcheck transcript (with the base
/// PCS's commitments and opening proofs), never $\mathbf c$ (the verifier re-derives it) nor
/// the masks, $\boldsymbol\rho\_p$, $\mathbf z\_p$ or $\hat{\mathbf z}$ (plan
/// `docs/hachi-co-plan.md`, finding F5). The per-party final values do reveal one evaluation
/// $\langle \widetilde w \rangle\_p(\mathbf a)$ of each party's share polynomial, as the draft
/// prescribes (`5_Building_Blocks.tex`:314-320; plan, finding F4, whose impact on $t$-hiding is
/// open; the authors' answer 7 of 2026-10-21 lets them be revealed for now). The base PCS's
/// commitments and openings reveal whatever the base PCS reveals: `ClearPcs` reveals the
/// polynomials themselves, and Hachi ([`HachiBase`](crate::HachiBase)) is not hiding either.
/// An abort of every attempt is [`hachi_co_r2l::Error::TooManyAttempts`]; the draft's Eval ends
/// with $(\bot, \bot)$ on an abort (line 417), and under Fiat-Shamir the provers retry with fresh
/// randomness instead (`ref/README.md`, choice 5).
///
/// # Errors
/// [`Error::Statement`] for a statement that does not fit the context;
/// [`Error::ClaimMismatch`] if $y \ne E(u)$; the errors of the building blocks
/// ([`Error::Conv`], [`Error::R2l`], [`Error::Sumcheck`]), which include those of `comm`,
/// `rng`, `ch` and `pcs`.
#[allow(clippy::too_many_arguments)] // eval's arguments, the base PCS and the challenger
pub fn eval_with<
    const K: usize,
    P: BasePcs<K>,
    C: Comm,
    R: PartyRandomness<K>,
    H: Challenger<K>,
>(
    ctx: &Context<K>,
    pcs: &P,
    comm: &mut C,
    rng: &mut R,
    ch: &mut H,
    state: &PartyState<K>,
    u: &[Poly],
    x: &[Ext<K>],
    y: &Ext<K>,
) -> Result<EvalOutput<K>, Error> {
    check_statement(ctx, u, x, y)?;
    absorb_statement(ch, ctx, &pcs.params_digest(), u, x, y);
    let conv = pi_conv(ctx, comm, state, x)?;
    hachi_co_conv::check(ctx, y, &conv.e_u).map_err(Error::ClaimMismatch)?;
    ch.absorb_polys(LABEL_E_U, core::slice::from_ref(&conv.e_u));
    let r2l = hachi_co_r2l::run(ctx, comm, rng, ch, state, &conv.public, &conv.u_share, u)?;
    let (transcript, private) = {
        let rel = linear_relation(ctx, &conv.public, &r2l.v, &r2l.c, u)?;
        let e = e_matrix(ctx, &conv.public.v_point)?;
        hachi_co_sumcheck::prove(ctx, pcs, comm, rng, ch, &rel.m, &e, &r2l.share)?
    };
    let proof = EvalProof {
        e_u: conv.e_u.clone(),
        attempt: r2l.accepted_attempt,
        v: r2l.v.clone(),
        sumcheck: transcript,
    };
    Ok(EvalOutput {
        proof,
        conv,
        r2l,
        sumcheck: private,
    })
}
