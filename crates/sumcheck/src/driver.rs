//! The driver of one party: the whole sumcheck of `5_Building_Blocks.tex`:195-320 against a
//! [`Comm`], a [`PartyRandomness`] and a [`Challenger`] (`docs/m1-spec.md`, "Randomness and
//! coins"), with the absorb labels and the $\mathcal F\_{\mathsf{AH}}$ calls of the crate.

use crate::eq::eq_table;
use crate::error::{Error, Result, check_len};
use crate::prover::PartyProver;
use crate::quotient::quotient_share;
use crate::tables::{mstar_table, w_table};
use crate::transcript::{FinalValues, RoundMessage, SumcheckPrivate, SumcheckTranscript};
use hachi_co_core::wire::{flatten_exts, unflatten_exts};
use hachi_co_core::{BlinderKind, Challenger, Context, LinearShare, PartyRandomness, Poly};
use hachi_co_math::Ext;
use hachi_co_mpc::Comm;

/// The absorb label of the hypercube sums $G\_{0,1}, \dots, G\_{0,n}$ (absorbed before
/// [`Challenger::sumcheck_coins`]).
pub const LABEL_G0: &str = "hachi-co/sumcheck/G0";
/// The absorb label of $G\_{1,1}, \dots, G\_{1,n}$.
pub const LABEL_G1: &str = "hachi-co/sumcheck/G1";
/// The absorb label of a round message $H\_{0,r}(1), \dots, H\_{0,r}(d\_0 + 1)$ (absorbed before
/// [`Challenger::round_coin`]`(r)`).
pub const LABEL_H0: &str = "hachi-co/sumcheck/H0";
/// The absorb label of $H\_{1,r}(1), \dots, H\_{1,r}(d\_1 + 1)$.
pub const LABEL_H1: &str = "hachi-co/sumcheck/H1";
/// The absorb label of the final values $\langle \widetilde w \rangle\_p(\mathbf a)$, $p = 1, \dots,
/// n$ (absorbed after the last round coin, for the coins that follow the sumcheck).
pub const LABEL_FINAL_W: &str = "hachi-co/sumcheck/final/w";
/// The absorb label of the final values $g\_{0,p}(\mathbf a)$.
pub const LABEL_FINAL_G0: &str = "hachi-co/sumcheck/final/g0";
/// The absorb label of the final values $g\_{1,p}(\mathbf a)$.
pub const LABEL_FINAL_G1: &str = "hachi-co/sumcheck/final/g1";
/// The label of the $\mathcal F\_{\mathsf{AH}}$ call that makes every $(G\_{0,p}, G\_{1,p})$ public.
pub const FAH_SUMS: &str = "hachi-co/sumcheck/fah/G";
/// The label of the $\mathcal F\_{\mathsf{AH}}(\mathbb I)$ call of each round
/// (`5_Building_Blocks.tex`:293-300).
pub const FAH_ROUND: &str = "hachi-co/sumcheck/fah/I";
/// The label of the $\mathcal F\_{\mathsf{AH}}$ call that makes every party's final values public.
pub const FAH_FINAL: &str = "hachi-co/sumcheck/fah/final";

/// Every party's values through one $\mathcal F\_{\mathsf{AH}}$ call: party $p$ places its
/// `values` in slot $p - 1$ of a vector of $n$ slots, zero elsewhere, so that the sum is the
/// concatenation of all parties' values. Returns `count` values per party, party by party.
fn all_gather<const K: usize, C: Comm>(
    comm: &mut C,
    label: &'static str,
    party: usize,
    n: usize,
    values: &[Ext<K>],
) -> Result<Vec<Vec<Ext<K>>>> {
    let count = values.len();
    let slot = count * K;
    let mut local = vec![0u64; n * slot];
    local[(party - 1) * slot..party * slot].copy_from_slice(&flatten_exts(values));
    let sum = comm.fah(label, &local)?;
    let all = unflatten_exts::<K>(&sum, n * count)?;
    Ok(all.chunks(count.max(1)).map(<[Ext<K>]>::to_vec).collect())
}

/// Column `index` of per-party rows.
fn column<const K: usize>(rows: &[Vec<Ext<K>>], index: usize) -> Vec<Ext<K>> {
    rows.iter().map(|row| row[index]).collect()
}

/// Party $p$ = `comm.id()` runs the collaborative sumcheck (`5_Building_Blocks.tex`:195-320) on
/// its shares $\langle \mathbf z' \rangle\_p, \lbrace \mathbf y \rbrace\_p$ of the relation
/// $\mathbf M \mathbf z' = \mathbf y$, with the public $\mathbf M$ ([`hachi_co_core::build_m`]) and
/// the $d \times d$ matrix of $E$ ([`hachi_co_math::Subfield::e_matrix`] at the point $v$ of
/// $\Pi\_{\mathsf{conv}}$). Every party obtains the same public transcript.
///
/// **Steps** (each party; the coins come from `ch`, the blinders from `rng`):
/// 1. $\langle \mathbf r \rangle\_p$ ([`quotient_share`]) and the table of
///    $\langle \widetilde w \rangle\_p$ ([`w_table`]), lines 198-229;
/// 2. $g\_{0,p}, g\_{1,p}$ from [`PartyRandomness::blinder`] ($\ell'' \times d\_0$ and
///    $\ell'' \times d\_1$) and their hypercube sums, lines 209-219;
/// 3. $\mathcal F\_{\mathsf{AH}}$ (label [`FAH_SUMS`]) with $(G\_{0,p}, G\_{1,p})$ in slot $p$, so
///    that every party learns all $n$ pairs, which are public messages (line 241; the transcript
///    holds the per-party values, as the oracle's); absorb $G\_{0,1..n}$ ([`LABEL_G0`]) and
///    $G\_{1,1..n}$ ([`LABEL_G1`]); draw $\alpha, \rho, \tau\_0, \tau\_1$
///    ([`Challenger::sumcheck_coins`]), lines 243-250;
/// 4. the tables $\widetilde{eq}(\tau\_1, \cdot)$ and $M^\ast$ ([`mstar_table`]);
/// 5. for $r = 1, \dots, \ell''$: the round values ([`PartyProver::round_values`]),
///    $\mathcal F\_{\mathsf{AH}}(\mathbb I)$ on $H\_{0,p,r}(1..d\_0+1) \Vert H\_{1,p,r}(1..d\_1+1)$
///    (label [`FAH_ROUND`], lines 293-300), absorb $H\_{0,r}$ ([`LABEL_H0`]) and $H\_{1,r}$
///    ([`LABEL_H1`]), draw $a\_r$ ([`Challenger::round_coin`]), bind;
/// 6. the openings $\langle \widetilde w \rangle\_p(\mathbf a), g\_{0,p}(\mathbf a), g\_{1,p}(\mathbf a)$
///    through $\mathcal F\_{\mathsf{AH}}$ (label [`FAH_FINAL`], slot $p$), absorbed with
///    [`LABEL_FINAL_W`], [`LABEL_FINAL_G0`], [`LABEL_FINAL_G1`] (lines 314-320).
///
/// The commitments of the base PCS (lines 220-221, 229) are not modelled in milestone 1
/// (`ref/README.md`, choice 13); milestone 3 absorbs them with the hypercube sums. With the
/// replay providers `ReplayRandomness` and `ReplayChallenger` of `hachi-co-core` (feature
/// `fixtures`) the driver reproduces the oracle's run.
///
/// # Errors
/// [`Error::Parameter`] unless `comm` has the layout's $n$ parties and modulus $q$;
/// [`Error::InvalidParty`] for an index outside $1, \dots, n$; the shape errors of the steps; the
/// errors of `rng`, `ch` and `comm`.
pub fn prove<const K: usize, C: Comm, R: PartyRandomness<K>, H: Challenger<K>>(
    ctx: &Context<K>,
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
    let (s0, s1) = prover.hypercube_sums();
    let sums = all_gather(comm, FAH_SUMS, party, n, &[s0, s1])?;
    let (g0_sums, g1_sums) = (column(&sums, 0), column(&sums, 1));
    ch.absorb_exts(LABEL_G0, &g0_sums);
    ch.absorb_exts(LABEL_G1, &g1_sums);
    let coins = ch.sumcheck_coins(field, l.ell_pp, l.log_n, l.log_nu)?;
    coins.validate(l)?;
    // Step 4.
    let eq1 = eq_table(field, &coins.tau1)?;
    let mstar = mstar_table(ctx, m, e_matrix, &eq1, &coins.alpha)?;
    drop(eq1);
    prover.start(&coins, mstar)?;
    // Step 5.
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
    // Step 6.
    let (w_a, g0_a, g1_a) = prover.final_values()?;
    let finals = all_gather(comm, FAH_FINAL, party, n, &[w_a, g0_a, g1_a])?;
    let finals = FinalValues {
        w: column(&finals, 0),
        g0: column(&finals, 1),
        g1: column(&finals, 2),
    };
    check_len("final values (one per party)", n, finals.w.len())?;
    ch.absorb_exts(LABEL_FINAL_W, &finals.w);
    ch.absorb_exts(LABEL_FINAL_G0, &finals.g0);
    ch.absorb_exts(LABEL_FINAL_G1, &finals.g1);
    let (w_table, g0, g1, point) = prover.into_parts();
    Ok((
        SumcheckTranscript {
            g0_sums,
            g1_sums,
            rounds,
            finals,
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
