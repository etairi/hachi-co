//! The provers' side of the PCS: $\mathsf{Setup}$, $\mathsf{Commit}$ and $\mathsf{Eval}$
//! (`6_Collaborative_Hachi.tex`:265-401, 426-437) with Fiat-Shamir.

use crate::error::Error;
use crate::proof::EvalProof;
use crate::statement::{EVAL_DOMAIN, absorb_statement, check_statement};
use hachi_co_conv::{ConvOutput, LABEL_E_U, e_matrix, pi_conv};
use hachi_co_core::{
    BasePcs, Challenger, Context, Ext, FsChallenger, MRows, PartyRandomness, PartyState, Poly,
};
use hachi_co_mpc::Comm;
use hachi_co_params::Params;
use hachi_co_r2l::{R2lOutput, check_relation};
use hachi_co_sumcheck::SumcheckPrivate;

/// $\mathsf{Setup}$ (`6_Collaborative_Hachi.tex`:265-303) from derived parameters and a 32-byte
/// seed: [`Context::setup`], which expands $\mathbf A, \mathbf B, \mathbf D$ from the seed. The
/// base PCS (lines 294-297) is set up on its own, over Hachi-Co's $q$ and $k$: Hachi from the
/// same seed with [`HachiBase::setup`](crate::HachiBase::setup), or `ClearPcs` in the tests. It
/// is passed to [`eval`] and [`verify`](crate::verify).
///
/// # Errors
/// [`Error::Core`] as [`Context::setup`] (for example $K \ne k$).
pub fn setup<const K: usize>(params: Params, seed: &[u8; 32]) -> Result<Context<K>, Error> {
    Ok(Context::setup(params, seed)?)
}

/// $\mathsf{Commit}$ (`6_Collaborative_Hachi.tex`:305-401) for the party `comm.id()` with
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

/// [`commit`] for a polynomial too large for memory: [`hachi_co_commit::commit_streaming`], with
/// the party's initial share read through `input` chunk by chunk and the blocks of
/// $\mathbf f'$ kept in a file in `dir`. It gives the commitment of [`commit`] for the same
/// share and randomness, and [`eval`] takes the state as usual, reading one block at a time.
///
/// # Errors
/// [`Error::Commit`] as [`hachi_co_commit::commit_streaming`].
pub fn commit_streaming<const K: usize, C: Comm, R: PartyRandomness<K>>(
    ctx: &Context<K>,
    comm: &mut C,
    rng: &mut R,
    dir: &std::path::Path,
    input: impl FnMut(usize) -> hachi_co_commit::Result<Vec<Ext<K>>>,
) -> Result<(Vec<Poly>, PartyState<K>), Error> {
    Ok(hachi_co_commit::commit_streaming(
        ctx, comm, rng, dir, input,
    )?)
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
    /// The ring-to-linear step's output: $\mathbf v$, $\mathbf c$, the sets of accepted
    /// iterations, $j^\ast$, and the party's shares $\langle \mathbf z' \rangle\_p$,
    /// $\lbrace \mathbf y \rbrace\_p$ (private).
    pub r2l: R2lOutput,
    /// The party's private data of the sumcheck: the table of $\langle \widetilde w \rangle\_p$,
    /// the blinders $g\_{0,p}, g\_{1,p}$, the coins and the point $\mathbf a$.
    pub sumcheck: SumcheckPrivate<K>,
}

/// $\mathsf{Eval}$ for the party `comm.id()` with Fiat-Shamir: [`eval_with`] on a fresh
/// [`FsChallenger`] with the domain [`EVAL_DOMAIN`], keeping only the proof. Every party calls it
/// with the same base PCS `pcs`, its own network end, randomness and state and the same
/// statement $(\mathbf u, \mathbf x)$; every party obtains the same proof, from which the verifier
/// reads the value $y$ ([`verify`](crate::verify)).
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
) -> Result<EvalProof<K>, Error> {
    let mut ch = FsChallenger::new(EVAL_DOMAIN);
    Ok(eval_with(ctx, pcs, comm, rng, &mut ch, state, u, x)?.proof)
}

/// [`eval`] for a caller that keeps only the proof, as the benchmark harness does: each large
/// value is freed after its last use. The context's $\mathbf A$ in the NTT form after the
/// ring-to-linear step ([`Context::release_a_prepared`]), its public matrices after the last row of
/// $\mathbf M$ ([`Context::release_public_params`]), both formed again if the context is read
/// again, and the share $\mathbf z'$ after the sumcheck's table, whose private copy is not formed
/// ([`hachi_co_sumcheck::prove_rebuilding_lean`]). The same proof as [`eval`].
///
/// # Errors
/// As [`eval_with`].
#[allow(clippy::too_many_arguments)] // eval's arguments and the base PCS
pub fn eval_lean<const K: usize, P: BasePcs<K>, C: Comm, R: PartyRandomness<K>>(
    ctx: &Context<K>,
    pcs: &P,
    comm: &mut C,
    rng: &mut R,
    state: &PartyState<K>,
    u: &[Poly],
    x: &[Ext<K>],
) -> Result<EvalProof<K>, Error> {
    let mut ch = FsChallenger::new(EVAL_DOMAIN);
    Ok(eval_inner(ctx, pcs, comm, rng, &mut ch, state, u, x, true)?.proof)
}

/// $\mathsf{Eval}\langle \mathcal P\_1(\mathsf{st}\_1), \dots, \mathcal P\_n(\mathsf{st}\_n)
/// \rangle(\mathsf{pp}, \mathbf u, \mathbf x)$ (`6_Collaborative_Hachi.tex`:426-437; at the draft's
/// commit `c071784` the composition of $\Pi\_A$, $\Pi\_B$, $\Pi\_C$) for the party `comm.id()`,
/// proving that $y = E(u)$ is the value $f(\mathbf x)$ of the polynomial committed in
/// $\mathbf u$, with the base PCS `pcs` (the same at every party and at the verifier) and the
/// coins drawn from `ch`, which must be fresh: an [`FsChallenger`]`::new(`[`EVAL_DOMAIN`]`)` for a
/// real proof ([`eval`]), or a replay challenger in fixture tests. Steps:
///
/// 1. [`check_statement`], then [`absorb_statement`] (parameters, $\mathbf A, \mathbf B,
///    \mathbf D$, the digest of `pcs`'s parameters, $\mathbf u$, $\mathbf x$);
/// 2. $\Pi\_{\mathsf{conv}}$ ([`hachi_co_conv::pi_conv`], line 429; $\Pi\_A$ at `c071784`): the
///    public vectors, $\langle u \rangle\_p$ and $E(u)$, which is the value $y$; $E(u)$ is
///    absorbed under [`LABEL_E_U`];
/// 3. the ring-to-linear step with its $L$ iterations of the rejection sampling
///    ([`hachi_co_r2l::run`], line 435; `fig:ring-to-linear` since the draft's commit `1096ea2`),
///    which absorbs $\mathbf v$, draws $\mathbf c$ and absorbs the sets of accepted iterations
///    itself; the checks of the linear relation for the selected iteration $j^\ast$
///    ([`check_relation`]), whose matrix $\mathbf M$ the sumcheck reads one row at a time
///    ([`MRows`]);
/// 4. the sumcheck ([`hachi_co_sumcheck::prove`], line 437) with the matrix of $E$ at the point
///    $v$ ([`e_matrix`]), which makes its own absorbs and contexts, and in which each party
///    commits to $\langle \widetilde w \rangle\_p$, $g\_{0,p}$, $g\_{1,p}$ with `pcs` before the
///    coins and opens them at the end (`5_Building_Blocks.tex`:238-273, 346-350).
///
/// The proof holds $E(u)$, the sets of accepted iterations, $\mathbf v$ and the sumcheck
/// transcript (with the base PCS's commitments and opening proofs), never $\mathbf c$ or
/// $j^\ast$ (the verifier re-derives them) nor the masks, $\boldsymbol\rho^v\_p$, $\boldsymbol\rho^u\_p$, $\mathbf z\_p$ or
/// $\hat{\mathbf z}$ (plan `docs/hachi-co-plan.md`, finding F5). The per-party final values
/// reveal one evaluation $\langle \widetilde w \rangle\_p(\mathbf a)$ of each party's share
/// polynomial, which the pad $\rho^w\_p$ masks (`5_Building_Blocks.tex`:252-256, 346-352). The
/// base PCS's commitments and openings reveal whatever the base PCS reveals: `ClearPcs` reveals
/// the polynomials themselves, and Hachi ([`HachiBase`](crate::HachiBase)) is not yet shown to
/// have the evaluation hiding that the draft assumes of the base PCS.
///
/// **Several evaluations of one commitment.** Each claim is one call with the same `state` and
/// a source `rng` of its own: $\boldsymbol\rho^u\_p$ comes from the state, and everything else
/// is drawn afresh. A [`PrgRandomness`](hachi_co_core::PrgRandomness) refuses to serve a stream
/// twice ([`RandomnessReuse`](hachi_co_core::Error::RandomnessReuse)), so reusing a source fails
/// instead of repeating a mask.
/// If no iteration is accepted by every prover the step aborts with
/// [`hachi_co_r2l::Error::NoCommonIteration`], the draft's Eval ending with $(\bot, \bot)$ (line
/// 435); for honest provers this has probability $(1 - M^{-n})^L$, negligible for $L = \lambda$
/// (`5_Building_Blocks.tex`, the paragraph before `fig:ring-to-linear`).
///
/// # Errors
/// [`Error::Statement`] for a statement that does not fit the context;
/// the errors of the building blocks
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
) -> Result<EvalOutput<K>, Error> {
    eval_inner(ctx, pcs, comm, rng, ch, state, u, x, false)
}

/// [`eval_with`], or with `lean` the work of [`eval_lean`].
#[allow(clippy::too_many_arguments)] // eval_with's arguments and the mode
fn eval_inner<const K: usize, P: BasePcs<K>, C: Comm, R: PartyRandomness<K>, H: Challenger<K>>(
    ctx: &Context<K>,
    pcs: &P,
    comm: &mut C,
    rng: &mut R,
    ch: &mut H,
    state: &PartyState<K>,
    u: &[Poly],
    x: &[Ext<K>],
    lean: bool,
) -> Result<EvalOutput<K>, Error> {
    check_statement(ctx, u, x)?;
    absorb_statement(ch, ctx, &pcs.params_digest(), u, x);
    let conv = pi_conv(ctx, comm, state, x)?;
    ch.absorb_polys(LABEL_E_U, core::slice::from_ref(&conv.e_u));
    let mut r2l = hachi_co_r2l::run(ctx, comm, rng, ch, state, &conv.public, &conv.u_share, u)?;
    if lean {
        // The ring-to-linear step was the last to multiply by A in the NTT form.
        ctx.release_a_prepared();
    }
    let (transcript, private) = {
        // The checks of `linear_relation`; M itself is read by rows below.
        check_relation(ctx, &conv.public, &r2l.v, &r2l.c, u)?;
        let e = e_matrix(ctx, &conv.public.v_point)?;
        // The driver reads M twice, for the quotient and for M*, with the commitments in between:
        // it builds each row when it reads it (`MRows`, the rows of `build_m`, from public
        // values), so that M is never held whole.
        let pv = &conv.public;
        let rows = MRows::new(ctx, &pv.b_ring, &pv.a_ring, &r2l.c, r2l.j_star)?;
        if lean {
            // The driver takes the share over and drops it after its last reader.
            let share = core::mem::take(&mut r2l.share);
            hachi_co_sumcheck::prove_rebuilding_lean(
                ctx,
                pcs,
                comm,
                rng,
                ch,
                |i| Ok(rows.row(i)?),
                &e,
                share,
            )?
        } else {
            hachi_co_sumcheck::prove_rebuilding(
                ctx,
                pcs,
                comm,
                rng,
                ch,
                |i| Ok(rows.row(i)?),
                &e,
                &r2l.share,
            )?
        }
    };
    let proof = EvalProof {
        e_u: conv.e_u.clone(),
        accept_sets: r2l.accept_sets.clone(),
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
