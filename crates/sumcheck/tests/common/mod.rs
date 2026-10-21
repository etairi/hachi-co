//! Shared helpers of the integration tests: the fixture instances (inputs built as
//! `docs/m1-spec.md` prescribes), the fixture transcripts, the parties run in lockstep with
//! $\mathcal F\_{\mathsf{AH}}$ emulated by summing their inputs, the base-PCS commitments and
//! openings of `ClearPcs` attached to a transcript, the ideal PCS (the oracle's `IdealPCS`) and
//! a stand-in for $\Pi\_{\mathsf{conv}}$'s check $y = E(u)$.
#![allow(dead_code)]

use hachi_co_core::base_pcs::mobius;
use hachi_co_core::{
    BasePcs, BlinderKind, Challenger, ClearPcs, Context, Fixture, LinearShare, Matrix,
    PartyRandomness, Poly, ReplayChallenger, ReplayRandomness, build_m, build_y, sha256_residues,
};
use hachi_co_math::{Ext, Fqk};
use hachi_co_sumcheck::{
    Claim, ClaimedPoly, FinalClaims, FinalValues, PartyProver, PcsWords, RoundMessage, RoundValues,
    SumcheckPrivate, SumcheckTranscript, blinder_coefficients, blinder_weights, eq_table,
    mstar_table, opening_label, quotient_share, w_table,
};
use std::sync::OnceLock;

pub const K: usize = 4;

/// The reference base PCS of the tests.
pub type Pcs = ClearPcs<K>;

/// `ClearPcs` over the instance's field.
pub fn clear_pcs(field: &Fqk<K>) -> Pcs {
    ClearPcs::from_field(field)
}

/// The transcript without the base-PCS words: what the oracle records (`ref/README.md`,
/// choice 13: no commitments and no openings).
pub fn without_pcs(t: &SumcheckTranscript<K>) -> SumcheckTranscript<K> {
    SumcheckTranscript {
        commitments: PcsWords::default(),
        openings: PcsWords::default(),
        ..t.clone()
    }
}

/// The `ClearPcs` commitments and openings of the parties' polynomials (from their private
/// data: the table of $\langle \widetilde w \rangle\_p$, through the Möbius transform, and the
/// blinders' coefficient vectors), opened at `point` under the contexts of `ch`
/// (`opening_label`), set into `transcript`. This is the driver's step 3 and step 8 done
/// independently, for transcripts built without the driver (the fixtures' and the lockstep
/// runs'); the contexts of `ReplayChallenger` do not depend on the transcript.
pub fn attach_openings<H: Challenger<K>>(
    ctx: &Context<K>,
    transcript: &mut SumcheckTranscript<K>,
    privates: &[SumcheckPrivate<K>],
    point: &[Ext<K>],
    ch: &mut H,
) {
    let field = ctx.field();
    let l = &ctx.layout;
    let pcs = clear_pcs(field);
    let mut commitments = PcsWords::default();
    let mut openings = PcsWords::default();
    for private in privates {
        let mut coeffs = private.w_table.clone();
        mobius(field.zq(), &mut coeffs).unwrap();
        let (com_w, st_w) = pcs.commit_multilinear(&coeffs).unwrap();
        let (com_g0, st_g0) = pcs
            .commit_vector(&blinder_coefficients(&private.g0))
            .unwrap();
        let (com_g1, st_g1) = pcs
            .commit_vector(&blinder_coefficients(&private.g1))
            .unwrap();
        for (poly, com) in [
            (ClaimedPoly::W, com_w),
            (ClaimedPoly::G0, com_g0),
            (ClaimedPoly::G1, com_g1),
        ] {
            commitments
                .of_mut(poly)
                .push(Pcs::commitment_to_words(&com));
        }
        for poly in [ClaimedPoly::W, ClaimedPoly::G0, ClaimedPoly::G1] {
            let context = ch.context(&opening_label(private.party, poly));
            let (_, proof) = match poly {
                ClaimedPoly::W => pcs.prove_multilinear(&st_w, point, &context).unwrap(),
                ClaimedPoly::G0 => pcs
                    .prove_inner_product(&st_g0, &blinder_weights(field, point, l.d0), &context)
                    .unwrap(),
                ClaimedPoly::G1 => pcs
                    .prove_inner_product(&st_g1, &blinder_weights(field, point, l.d1), &context)
                    .unwrap(),
            };
            openings.of_mut(poly).push(Pcs::proof_to_words(&proof));
        }
    }
    transcript.commitments = commitments;
    transcript.openings = openings;
}

/// [`attach_openings`] with the replay challenger of the instance's fixture.
pub fn with_replay_openings(
    inst: &Instance,
    transcript: &SumcheckTranscript<K>,
    privates: &[SumcheckPrivate<K>],
) -> SumcheckTranscript<K> {
    let mut t = transcript.clone();
    let point = inst.fx.exts::<K>("/sumcheck/coins/a").unwrap();
    let mut ch = ReplayChallenger::new(&inst.fx).unwrap();
    attach_openings(&inst.ctx, &mut t, privates, &point, &mut ch);
    t
}

/// One fixture with the inputs of the sumcheck built from it.
pub struct Instance {
    pub name: &'static str,
    pub fx: Fixture,
    pub ctx: Context<K>,
    /// M from `core::build_m` with b, a from `conv` and c from the accepted attempt.
    pub m: Matrix,
    /// y from `core::build_y` with the accepted attempt's v and the commitment u.
    pub y: Vec<Poly>,
    /// The matrix of E at the point v of Pi_conv (`Subfield::e_matrix`).
    pub e: Vec<Vec<u64>>,
    /// The claimed evaluation y embedded in R_q^H (`input/y_ring`).
    pub y_ring: Poly,
    /// The parties' shares <z'>_p, {y}_p (`LinearShare::slice` of `ring_to_linear/z_prime`).
    pub shares: Vec<LinearShare>,
}

impl Instance {
    pub fn n(&self) -> usize {
        self.fx.n()
    }

    pub fn field(&self) -> &Fqk<K> {
        self.ctx.field()
    }
}

fn build(name: &'static str) -> Instance {
    let fx = Fixture::load(name).unwrap();
    let ctx = fx.context::<K>().unwrap();
    let embed = |pointer: &str| -> Vec<Poly> {
        fx.exts::<K>(pointer)
            .unwrap()
            .iter()
            .map(|x| ctx.sub.embed(x))
            .collect()
    };
    let accepted = fx.usize("/ring_to_linear/accepted_attempt").unwrap();
    let c = fx
        .int_rows(&format!("/ring_to_linear/attempts/{accepted}/c"))
        .unwrap();
    let m = build_m(&ctx, &embed("/conv/b"), &embed("/conv/a"), &c).unwrap();
    assert_eq!(
        sha256_residues(m.iter().flatten().flat_map(|p| p.0.iter().copied())),
        fx.str("/ring_to_linear/M_sha256").unwrap(),
        "{name}: M"
    );
    let v = fx
        .polys(&format!("/ring_to_linear/attempts/{accepted}/v"))
        .unwrap();
    let y = build_y(&ctx, &v, &fx.polys("/commit/u").unwrap()).unwrap();
    assert_eq!(y, fx.polys("/ring_to_linear/y").unwrap(), "{name}: y");
    let e = ctx
        .sub
        .e_matrix(&fx.poly("/conv/v_point").unwrap())
        .unwrap();
    assert_eq!(e, fx.residue_rows("/conv/E_matrix").unwrap(), "{name}: E");
    let y_ring = fx.poly("/input/y_ring").unwrap();
    assert_eq!(
        y_ring,
        ctx.sub.embed(&fx.ext::<K>("/input/y").unwrap()),
        "{name}: y as a ring element"
    );
    let z_prime = fx.polys("/ring_to_linear/z_prime").unwrap();
    let u_shares = fx.polys("/conv/u_shares").unwrap();
    let shares = (1..=fx.n())
        .map(|p| LinearShare::slice(&ctx.layout, &z_prime, &y, &u_shares[p - 1], p).unwrap())
        .collect();
    Instance {
        name,
        fx,
        ctx,
        m,
        y,
        e,
        y_ring,
        shares,
    }
}

/// The three fixture instances, built once per test binary.
pub fn instances() -> &'static [Instance] {
    static CELL: OnceLock<Vec<Instance>> = OnceLock::new();
    CELL.get_or_init(|| Fixture::TOY.iter().map(|name| build(name)).collect())
}

/// The transcript recorded in a fixture (`sumcheck/G0`, `G1`, `rounds`, `final`), without
/// base-PCS words (the oracle does not model them).
pub fn fixture_transcript(fx: &Fixture) -> SumcheckTranscript<K> {
    let rounds = fx
        .get("/sumcheck/rounds")
        .unwrap()
        .as_array()
        .unwrap()
        .len();
    SumcheckTranscript {
        g0_sums: fx.exts("/sumcheck/G0").unwrap(),
        g1_sums: fx.exts("/sumcheck/G1").unwrap(),
        commitments: PcsWords::default(),
        rounds: (0..rounds)
            .map(|r| RoundMessage {
                h0: fx.exts(&format!("/sumcheck/rounds/{r}/H0")).unwrap(),
                h1: fx.exts(&format!("/sumcheck/rounds/{r}/H1")).unwrap(),
            })
            .collect(),
        finals: FinalValues {
            w: fx.exts("/sumcheck/final/w").unwrap(),
            g0: fx.exts("/sumcheck/final/g0").unwrap(),
            g1: fx.exts("/sumcheck/final/g1").unwrap(),
        },
        openings: PcsWords::default(),
    }
}

/// The result of the parties run in lockstep.
pub struct Run {
    /// The transcript, with the `ClearPcs` commitments and openings ([`attach_openings`]).
    pub transcript: SumcheckTranscript<K>,
    pub privates: Vec<SumcheckPrivate<K>>,
}

/// Runs the $n$ parties' [`PartyProver`]s in lockstep on `shares` (and, if given, the quotient
/// shares `quotients` instead of the honest ones), with the fixture's recorded blinders
/// (`ReplayRandomness`) and coins (`ReplayChallenger`), emulating $\mathcal F\_{\mathsf{AH}}$ by
/// summing the parties' inputs; the parties' polynomials are committed and opened with
/// `ClearPcs` under the replay challenger's contexts ([`attach_openings`]).
pub fn run_parties(
    inst: &Instance,
    shares: &[LinearShare],
    quotients: Option<&[Vec<Poly>]>,
) -> Run {
    let ctx = &inst.ctx;
    let l = &ctx.layout;
    let f = ctx.field();
    let n = inst.n();
    let mut provers = Vec::with_capacity(n);
    let mut quotient_list = Vec::with_capacity(n);
    for p in 1..=n {
        let share = &shares[p - 1];
        let quotient = match quotients {
            Some(qs) => qs[p - 1].clone(),
            None => quotient_share(ctx, &inst.m, share).unwrap(),
        };
        let table = w_table(ctx, &share.z_prime, &quotient).unwrap();
        let mut rng = ReplayRandomness::new(&inst.fx, p).unwrap();
        let g0 = rng.blinder(f, BlinderKind::G0, l.ell_pp, l.d0).unwrap();
        let g1 = rng.blinder(f, BlinderKind::G1, l.ell_pp, l.d1).unwrap();
        provers.push(PartyProver::new(ctx, p, table, g0, g1).unwrap());
        quotient_list.push(quotient);
    }
    let g0_sums: Vec<Ext<K>> = provers.iter().map(|p| p.hypercube_sums().0).collect();
    let g1_sums: Vec<Ext<K>> = provers.iter().map(|p| p.hypercube_sums().1).collect();
    let mut ch = ReplayChallenger::new(&inst.fx).unwrap();
    let coins = ch.sumcheck_coins(f, l.ell_pp, l.log_n, l.log_nu).unwrap();
    let eq1 = eq_table(f, &coins.tau1).unwrap();
    let mstar = mstar_table(ctx, &inst.m, &inst.e, &eq1, &coins.alpha).unwrap();
    for prover in &mut provers {
        prover.start(&coins, mstar.clone()).unwrap();
    }
    let mut rounds = Vec::new();
    let mut values: Vec<Vec<RoundValues<K>>> = vec![Vec::new(); n];
    for r in 1..=l.ell_pp {
        let inputs: Vec<RoundValues<K>> =
            provers.iter().map(|p| p.round_values().unwrap()).collect();
        let add = |get: fn(&RoundValues<K>) -> &Vec<Ext<K>>| -> Vec<Ext<K>> {
            let len = get(&inputs[0]).len();
            (0..len)
                .map(|t| {
                    inputs
                        .iter()
                        .fold(f.zero(), |acc, v| f.add(&acc, &get(v)[t]))
                })
                .collect()
        };
        rounds.push(RoundMessage {
            h0: add(|v| &v.h0),
            h1: add(|v| &v.h1),
        });
        let a = ch.round_coin(f, r).unwrap();
        for prover in &mut provers {
            prover.bind(&a).unwrap();
        }
        for (p, v) in inputs.into_iter().enumerate() {
            values[p].push(v);
        }
    }
    let finals: Vec<(Ext<K>, Ext<K>, Ext<K>)> =
        provers.iter().map(|p| p.final_values().unwrap()).collect();
    let mut transcript = SumcheckTranscript {
        g0_sums,
        g1_sums,
        commitments: PcsWords::default(),
        rounds,
        finals: FinalValues {
            w: finals.iter().map(|v| v.0).collect(),
            g0: finals.iter().map(|v| v.1).collect(),
            g1: finals.iter().map(|v| v.2).collect(),
        },
        openings: PcsWords::default(),
    };
    let privates: Vec<SumcheckPrivate<K>> = provers
        .into_iter()
        .zip(quotient_list)
        .zip(values)
        .enumerate()
        .map(|(p0, ((prover, quotient), rounds))| {
            let (w_table, g0, g1, point) = prover.into_parts();
            SumcheckPrivate {
                party: p0 + 1,
                quotient,
                w_table,
                g0,
                g1,
                coins: coins.clone(),
                point,
                rounds,
            }
        })
        .collect();
    let point = privates[0].point.clone();
    let mut ch = ReplayChallenger::new(&inst.fx).unwrap();
    attach_openings(ctx, &mut transcript, &privates, &point, &mut ch);
    Run {
        transcript,
        privates,
    }
}

/// The honest run of each fixture, computed once per test binary.
pub fn honest_runs() -> &'static [Run] {
    static CELL: OnceLock<Vec<Run>> = OnceLock::new();
    CELL.get_or_init(|| {
        instances()
            .iter()
            .map(|inst| run_parties(inst, &inst.shares, None))
            .collect()
    })
}

/// The oracle's `IdealPCS` (`ref/README.md`, choice 13): an opening is accepted iff the claimed
/// value equals the committed polynomial evaluated directly, on a path independent of the
/// provers' folding: $\langle \widetilde w \rangle\_p(\mathbf a) = \sum\_x \widetilde{eq}(\mathbf a,
/// x) W\_p\[x\]$, and $g(\mathbf a) = s\_0 + \sum\_{j,k} s\_{j,k} a\_j^k$ with every power computed by
/// square and multiply. Returns the named checks `pcs_open_*` of the claims.
pub fn ideal_pcs(
    field: &Fqk<K>,
    claims: &FinalClaims<K>,
    privates: &[SumcheckPrivate<K>],
) -> Vec<(String, bool)> {
    let eq = eq_table(field, &claims.point).unwrap();
    claims
        .claims
        .iter()
        .map(|claim: &Claim<K>| {
            let private = &privates[claim.party - 1];
            let value = match claim.poly {
                ClaimedPoly::W => private
                    .w_table
                    .iter()
                    .zip(&eq)
                    .fold(field.zero(), |acc, (w, e)| {
                        field.add(&acc, &field.scale(e, *w))
                    }),
                ClaimedPoly::G0 | ClaimedPoly::G1 => {
                    let g = if claim.poly == ClaimedPoly::G0 {
                        &private.g0
                    } else {
                        &private.g1
                    };
                    let mut acc = g.s0;
                    for (row, a) in g.s.iter().zip(&claims.point) {
                        for (k, s) in row.iter().enumerate() {
                            let power = field.pow(a, &[k as u64 + 1]);
                            acc = field.add(&acc, &field.mul(s, &power));
                        }
                    }
                    acc
                }
            };
            (claim.check_name(), value == claim.value)
        })
        .collect()
}

/// A stand-in for V's check $y = E(u)$ of $\Pi\_{\mathsf{conv}}$ (`field-to-ring-rok.tex`:164; the
/// oracle's `conv.verify_conv`; WP8 implements it in `hachi-co-conv`, which this crate does not
/// depend on): $E(u) \in \mathcal R\_q^H$ and its coordinates equal $y$.
pub fn conv_check(ctx: &Context<K>, y: &Ext<K>, e_u: &Poly) -> bool {
    ctx.sub
        .extract(e_u)
        .is_ok_and(|x| x == ctx.field().reduce(y))
}

/// The fixture's per-party F_AH inputs of round `r` (1-based) for party `p`.
pub fn fah_inputs(fx: &Fixture, r: usize, p: usize) -> (Vec<Ext<K>>, Vec<Ext<K>>) {
    (
        fx.exts(&format!(
            "/sumcheck/rounds/{}/F_AH_inputs_H0/{}",
            r - 1,
            p - 1
        ))
        .unwrap(),
        fx.exts(&format!(
            "/sumcheck/rounds/{}/F_AH_inputs_H1/{}",
            r - 1,
            p - 1
        ))
        .unwrap(),
    )
}
