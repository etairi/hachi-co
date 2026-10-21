//! The driver of one party: the whole sumcheck of `5_Building_Blocks.tex`:195-326 against a
//! [`Comm`], a [`PartyRandomness`], a [`Challenger`] (`docs/m1-spec.md`, "Randomness and
//! coins") and a base PCS ([`BasePcs`]), with the absorb labels and the collective calls of the
//! crate.

use crate::blinder::{blinder_coefficients, blinder_weights};
use crate::eq::eq_table;
use crate::error::{Error, Result, check_len};
use crate::prover::PartyProver;
use crate::quotient::quotient_share;
use crate::tables::{mstar_table, w_table};
use crate::transcript::{
    ClaimedPoly, FinalValues, PcsWords, RoundMessage, SumcheckPrivate, SumcheckTranscript,
};
use hachi_co_core::base_pcs::mobius;
use hachi_co_core::wire::{flatten_exts, join_words, split_words, unflatten_exts};
use hachi_co_core::{
    BasePcs, BlinderKind, Challenger, Context, LinearShare, PartyRandomness, Poly,
};
use hachi_co_math::Ext;
use hachi_co_mpc::{Comm, PartyId};

/// The absorb label of the hypercube sums $G\_{0,1}, \dots, G\_{0,n}$ (absorbed before
/// [`Challenger::sumcheck_coins`]).
pub const LABEL_G0: &str = "hachi-co/sumcheck/G0";
/// The absorb label of $G\_{1,1}, \dots, G\_{1,n}$.
pub const LABEL_G1: &str = "hachi-co/sumcheck/G1";
/// The absorb label of the base-PCS commitments to $\langle \widetilde w \rangle\_p$,
/// $p = 1, \dots, n$ (absorbed after $G\_{0,\cdot}$ and $G\_{1,\cdot}$, before
/// [`Challenger::sumcheck_coins`]).
pub const LABEL_COMMIT_W: &str = "hachi-co/sumcheck/commit/w";
/// The absorb label of the base-PCS commitments to $g\_{0,p}$.
pub const LABEL_COMMIT_G0: &str = "hachi-co/sumcheck/commit/g0";
/// The absorb label of the base-PCS commitments to $g\_{1,p}$.
pub const LABEL_COMMIT_G1: &str = "hachi-co/sumcheck/commit/g1";
/// The absorb label of a round message $H\_{0,r}(1), \dots, H\_{0,r}(d\_0 + 1)$ (absorbed before
/// [`Challenger::round_coin`]`(r)`).
pub const LABEL_H0: &str = "hachi-co/sumcheck/H0";
/// The absorb label of $H\_{1,r}(1), \dots, H\_{1,r}(d\_1 + 1)$.
pub const LABEL_H1: &str = "hachi-co/sumcheck/H1";
/// The absorb label of the final values $\langle \widetilde w \rangle\_p(\mathbf a)$, $p = 1, \dots,
/// n$ (absorbed after the last round coin, before the contexts of the openings).
pub const LABEL_FINAL_W: &str = "hachi-co/sumcheck/final/w";
/// The absorb label of the final values $g\_{0,p}(\mathbf a)$.
pub const LABEL_FINAL_G0: &str = "hachi-co/sumcheck/final/g0";
/// The absorb label of the final values $g\_{1,p}(\mathbf a)$.
pub const LABEL_FINAL_G1: &str = "hachi-co/sumcheck/final/g1";
/// The label of the [`Comm::all_gather`] call that makes every $(G\_{0,p}, G\_{1,p})$ public.
pub const ALL_GATHER_SUMS: &str = "hachi-co/sumcheck/all-gather/G";
/// The label of the [`Comm::all_gather`] call that makes every party's three base-PCS
/// commitments public.
pub const ALL_GATHER_COMMITMENTS: &str = "hachi-co/sumcheck/all-gather/commitments";
/// The label of the $\mathcal F\_{\mathsf{AH}}(\mathbb I)$ call of each round
/// (`5_Building_Blocks.tex`:293-300).
pub const FAH_ROUND: &str = "hachi-co/sumcheck/fah/I";
/// The label of the [`Comm::all_gather`] call that makes every party's final values public.
pub const ALL_GATHER_FINAL: &str = "hachi-co/sumcheck/all-gather/final";
/// The label of the [`Comm::all_gather`] call that makes every party's three base-PCS opening
/// proofs public.
pub const ALL_GATHER_OPENINGS: &str = "hachi-co/sumcheck/all-gather/openings";

/// The label of the context ([`Challenger::context`]) of the base-PCS opening of `poly` of
/// party `party`: `hachi-co/sumcheck/open/<w|g0|g1>/p<party>`, for example
/// `hachi-co/sumcheck/open/g0/p2`. The provers and the verifier derive it after absorbing the
/// final values.
pub fn opening_label(party: PartyId, poly: ClaimedPoly) -> String {
    format!("hachi-co/sumcheck/open/{}/p{party}", poly.name())
}

/// Every party's values through one [`Comm::all_gather`] call: `count` values per party, party
/// by party.
fn all_gather<const K: usize, C: Comm>(
    comm: &mut C,
    label: &'static str,
    n: usize,
    values: &[Ext<K>],
) -> Result<Vec<Vec<Ext<K>>>> {
    let all = comm.all_gather(label, &flatten_exts(values))?;
    check_len("all-gathered vectors (one per party)", n, all.len())?;
    all.iter()
        .map(|v| Ok(unflatten_exts::<K>(v, values.len())?))
        .collect()
}

/// Every party's three word vectors $(\langle \widetilde w \rangle\_p, g\_{0,p}, g\_{1,p})$
/// through one [`Comm::all_gather`] call, each vector preceded by its length
/// ([`join_words`]); the parties' joined vectors must have the same length (the base PCS's
/// sizes depend only on the sizes of the polynomials, [`BasePcs`], "Words").
fn all_gather_words<C: Comm>(
    comm: &mut C,
    label: &'static str,
    n: usize,
    local: &[Vec<u64>; 3],
) -> Result<PcsWords> {
    let all = comm.all_gather(label, &join_words(local))?;
    check_len("all-gathered word vectors (one per party)", n, all.len())?;
    let mut out = PcsWords::default();
    for joined in &all {
        let mut parts = split_words(joined, 3)?.into_iter();
        for poly in [ClaimedPoly::W, ClaimedPoly::G0, ClaimedPoly::G1] {
            let part = parts.next().ok_or(Error::Shape {
                what: "all-gathered word vectors (three per party)",
                expected: 3,
                actual: 0,
            })?;
            out.of_mut(poly).push(part);
        }
    }
    Ok(out)
}

/// The index of `poly` in the arrays of this module: $\widetilde w$, $g\_0$, $g\_1$ (the order of
/// [`PcsWords`] and [`FinalValues`]).
fn slot(poly: ClaimedPoly) -> usize {
    match poly {
        ClaimedPoly::W => 0,
        ClaimedPoly::G0 => 1,
        ClaimedPoly::G1 => 2,
    }
}

/// Column `index` of per-party rows.
fn column<const K: usize>(rows: &[Vec<Ext<K>>], index: usize) -> Vec<Ext<K>> {
    rows.iter().map(|row| row[index]).collect()
}

/// Party $p$ = `comm.id()` runs the collaborative sumcheck (`5_Building_Blocks.tex`:195-326) on
/// its shares $\langle \mathbf z' \rangle\_p, \lbrace \mathbf y \rbrace\_p$ of the relation
/// $\mathbf M \mathbf z' = \mathbf y$, with the public $\mathbf M$ ([`hachi_co_core::build_m`]),
/// the $d \times d$ matrix of $E$ ([`hachi_co_math::Subfield::e_matrix`] at the point $v$ of
/// $\Pi\_{\mathsf{conv}}$) and the base PCS `pcs`. Every party obtains the same public
/// transcript.
///
/// **Steps** (each party; the coins come from `ch`, the blinders from `rng`):
/// 1. $\langle \mathbf r \rangle\_p$ ([`quotient_share`]) and the table of
///    $\langle \widetilde w \rangle\_p$ ([`w_table`]), lines 198-229;
/// 2. $g\_{0,p}, g\_{1,p}$ from [`PartyRandomness::blinder`] ($\ell'' \times d\_0$ and
///    $\ell'' \times d\_1$) and their hypercube sums, lines 209-219;
/// 3. the commitments (lines 220-221, 230): $\langle \widetilde w \rangle\_p$ by its monomial
///    coefficients ([`mobius`] of the table) with [`BasePcs::commit_multilinear`], and
///    $g\_{0,p}$, $g\_{1,p}$ by their coefficient vectors ([`blinder_coefficients`]) with
///    [`BasePcs::commit_vector`];
/// 4. [`Comm::all_gather`] (label [`ALL_GATHER_SUMS`]) of $(G\_{0,p}, G\_{1,p})$ and
///    (label [`ALL_GATHER_COMMITMENTS`]) of the three commitments in words
///    ([`BasePcs::commitment_to_words`]), so that every party learns them all: they are the
///    public messages of line 241, and the transcript holds them per party; absorb
///    $G\_{0,1..n}$ ([`LABEL_G0`]), $G\_{1,1..n}$ ([`LABEL_G1`]) and the commitments
///    ([`Challenger::absorb_words`]: [`LABEL_COMMIT_W`], [`LABEL_COMMIT_G0`],
///    [`LABEL_COMMIT_G1`], one word vector per party each); draw $\alpha, \rho, \tau\_0,
///    \tau\_1$ ([`Challenger::sumcheck_coins`]), lines 243-250;
/// 5. the tables $\widetilde{eq}(\tau\_1, \cdot)$ and $M^\ast$ ([`mstar_table`]);
/// 6. for $r = 1, \dots, \ell''$: the round values ([`PartyProver::round_values`]),
///    $\mathcal F\_{\mathsf{AH}}(\mathbb I)$ on $H\_{0,p,r}(1..d\_0+1) \Vert H\_{1,p,r}(1..d\_1+1)$
///    (label [`FAH_ROUND`], lines 293-300), absorb $H\_{0,r}$ ([`LABEL_H0`]) and $H\_{1,r}$
///    ([`LABEL_H1`]), draw $a\_r$ ([`Challenger::round_coin`]), bind;
/// 7. the final values $\langle \widetilde w \rangle\_p(\mathbf a), g\_{0,p}(\mathbf a),
///    g\_{1,p}(\mathbf a)$ through [`Comm::all_gather`] (label [`ALL_GATHER_FINAL`]), absorbed
///    with [`LABEL_FINAL_W`], [`LABEL_FINAL_G0`], [`LABEL_FINAL_G1`] (lines 314-320);
/// 8. the openings (lines 314-318), in the order [`ClaimedPoly::ORDER`] ($g\_0$, $g\_1$,
///    $\widetilde w$): for each, the context [`Challenger::context`]`(`[`opening_label`]`(p,
///    poly))`, then $g\_{b,p}(\mathbf a)$ by [`BasePcs::prove_inner_product`] with the weights
///    $(1, (a\_j^k)\_{j,k})$ ([`blinder_weights`]) and $\langle \widetilde w \rangle\_p(\mathbf a)$
///    by [`BasePcs::prove_multilinear`] at $\mathbf a$; each opened value must equal the final
///    value of step 7 ([`Error::OpeningMismatch`] otherwise); the proofs in words
///    ([`BasePcs::proof_to_words`]) through [`Comm::all_gather`] (label
///    [`ALL_GATHER_OPENINGS`]). Nothing follows them in the transcript, so they are not
///    absorbed.
///
/// With the replay providers `ReplayRandomness` and `ReplayChallenger` of `hachi-co-core`
/// (feature `fixtures`) the driver reproduces the oracle's run, which has no commitments or
/// openings (`ref/README.md`, choice 13): every other value of the transcript is the oracle's.
///
/// # Errors
/// [`Error::Parameter`] unless `comm` has the layout's $n$ parties and modulus $q$;
/// [`Error::InvalidParty`] for an index outside $1, \dots, n$; the shape errors of the steps;
/// [`Error::OpeningMismatch`]; the errors of `rng`, `ch`, `comm` and `pcs` (the last as
/// [`Error::Core`]).
#[allow(clippy::too_many_arguments)] // the inputs of the driver and the base PCS
pub fn prove<const K: usize, P: BasePcs<K>, C: Comm, R: PartyRandomness<K>, H: Challenger<K>>(
    ctx: &Context<K>,
    pcs: &P,
    comm: &mut C,
    rng: &mut R,
    ch: &mut H,
    m: &[Vec<Poly>],
    e_matrix: &[Vec<u64>],
    share: &LinearShare,
) -> Result<(SumcheckTranscript<K>, SumcheckPrivate<K>)> {
    let l = &ctx.layout;
    let field = ctx.field();
    let (n, party) = (comm.n(), comm.id());
    if n != l.n || comm.modulus() != ctx.q() {
        return Err(Error::Parameter(format!(
            "the network has n = {n} and q = {}, the context n = {} and q = {}",
            comm.modulus(),
            l.n,
            ctx.q()
        )));
    }
    if party == 0 || party > n {
        return Err(Error::InvalidParty { party, n });
    }
    // Steps 1-2.
    let quotient = quotient_share(ctx, m, share)?;
    let table = w_table(ctx, &share.z_prime, &quotient)?;
    let g0 = rng.blinder(field, BlinderKind::G0, l.ell_pp, l.d0)?;
    let g1 = rng.blinder(field, BlinderKind::G1, l.ell_pp, l.d1)?;
    let mut prover = PartyProver::new(ctx, party, table, g0, g1)?;
    // Step 3.
    let (st_w, com_w) = {
        let mut coeffs = prover.w_table().to_vec();
        mobius(field.zq(), &mut coeffs)?;
        let (com, st) = pcs.commit_multilinear(&coeffs)?;
        (st, pcs.commitment_to_words(&com))
    };
    let (g0, g1) = prover.blinders();
    let (com_g0, st_g0) = pcs.commit_vector(&blinder_coefficients(g0))?;
    let (com_g1, st_g1) = pcs.commit_vector(&blinder_coefficients(g1))?;
    // Step 4.
    let (s0, s1) = prover.hypercube_sums();
    let sums = all_gather(comm, ALL_GATHER_SUMS, n, &[s0, s1])?;
    let (g0_sums, g1_sums) = (column(&sums, 0), column(&sums, 1));
    let local = [
        com_w,
        pcs.commitment_to_words(&com_g0),
        pcs.commitment_to_words(&com_g1),
    ];
    let commitments = all_gather_words(comm, ALL_GATHER_COMMITMENTS, n, &local)?;
    ch.absorb_exts(LABEL_G0, &g0_sums);
    ch.absorb_exts(LABEL_G1, &g1_sums);
    ch.absorb_words(LABEL_COMMIT_W, &commitments.w);
    ch.absorb_words(LABEL_COMMIT_G0, &commitments.g0);
    ch.absorb_words(LABEL_COMMIT_G1, &commitments.g1);
    let coins = ch.sumcheck_coins(field, l.ell_pp, l.log_n, l.log_nu)?;
    coins.validate(l, ctx.q())?;
    // Step 5.
    let eq1 = eq_table(field, &coins.tau1)?;
    let mstar = mstar_table(ctx, m, e_matrix, &eq1, &coins.alpha)?;
    drop(eq1);
    prover.start(&coins, mstar)?;
    // Step 6.
    let mut rounds = Vec::with_capacity(l.ell_pp);
    let mut values = Vec::with_capacity(l.ell_pp);
    for r in 1..=l.ell_pp {
        let v = prover.round_values()?;
        let mut local = flatten_exts(&v.h0);
        local.extend(flatten_exts(&v.h1));
        let sum = unflatten_exts::<K>(&comm.fah(FAH_ROUND, &local)?, l.d0 + 1 + l.d1 + 1)?;
        let (h0, h1) = sum.split_at(l.d0 + 1);
        ch.absorb_exts(LABEL_H0, h0);
        ch.absorb_exts(LABEL_H1, h1);
        let a = ch.round_coin(field, r)?;
        prover.bind(&a)?;
        rounds.push(RoundMessage {
            h0: h0.to_vec(),
            h1: h1.to_vec(),
        });
        values.push(v);
    }
    // Step 7.
    let mine = prover.final_values()?;
    let finals = all_gather(comm, ALL_GATHER_FINAL, n, &[mine.0, mine.1, mine.2])?;
    let finals = FinalValues {
        w: column(&finals, 0),
        g0: column(&finals, 1),
        g1: column(&finals, 2),
    };
    check_len("final values (one per party)", n, finals.w.len())?;
    ch.absorb_exts(LABEL_FINAL_W, &finals.w);
    ch.absorb_exts(LABEL_FINAL_G0, &finals.g0);
    ch.absorb_exts(LABEL_FINAL_G1, &finals.g1);
    // Step 8.
    let point = prover.point().to_vec();
    let expected = [mine.0, mine.1, mine.2];
    let mut proofs: [Vec<u64>; 3] = Default::default();
    for poly in ClaimedPoly::ORDER {
        let context = ch.context(&opening_label(party, poly));
        let (value, words) = match poly {
            ClaimedPoly::W => {
                let (value, proof) = pcs.prove_multilinear(&st_w, &point, &context)?;
                (value, pcs.proof_to_words(&proof))
            }
            ClaimedPoly::G0 | ClaimedPoly::G1 => {
                let (st, degree) = if poly == ClaimedPoly::G0 {
                    (&st_g0, l.d0)
                } else {
                    (&st_g1, l.d1)
                };
                let weights = blinder_weights(field, &point, degree);
                let (value, proof) = pcs.prove_inner_product(st, &weights, &context)?;
                (value, pcs.proof_to_words(&proof))
            }
        };
        if field.reduce(&value) != expected[slot(poly)] {
            return Err(Error::OpeningMismatch { party, poly });
        }
        proofs[slot(poly)] = words;
    }
    let openings = all_gather_words(comm, ALL_GATHER_OPENINGS, n, &proofs)?;
    let (w_table, g0, g1, point) = prover.into_parts();
    Ok((
        SumcheckTranscript {
            g0_sums,
            g1_sums,
            commitments,
            rounds,
            finals,
            openings,
        },
        SumcheckPrivate {
            party,
            quotient,
            w_table,
            g0,
            g1,
            coins,
            point,
            rounds: values,
        },
    ))
}
