//! Shared helpers of the integration tests: the fixtures (context, statement, the proof built
//! from the recorded values, with `ClearPcs` commitments and openings of the recorded
//! polynomials), the reference base PCS and its sizes, seeded inputs and an independent
//! evaluation of $f(\mathbf x)$, an ideal PCS for the final claims (the oracle's `IdealPCS`, as
//! in the sumcheck crate's tests), a challenger that records the calls made to it, and the
//! parties run over the in-memory network.
#![allow(dead_code)]

pub mod net;

use hachi_co::{BasePcsWords, EvalProof, FinalClaims, HachiBase, HachiChoices};
use hachi_co_core::base_pcs::mobius;
use hachi_co_core::{
    BasePcs, Blinder, BlinderKind, Challenger, ClearPcs, Context, Ext, Fixture, Fqk, Layout,
    LinearShare, PartyRandomness, PartyState, Poly, ReplayChallenger, ReplayRandomness,
    SumcheckCoins, build_m,
};
use hachi_co_params::{Params, derive, preset};
use hachi_co_sumcheck::{
    Claim, ClaimedPoly, FinalValues, PcsWords, RoundMessage, SumcheckPrivate, SumcheckTranscript,
    blinder_coefficients, blinder_weights, eq_table, opening_label, quotient_share, w_table,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};

/// The extension degree of every toy and development preset.
pub const K: usize = 4;

/// The reference base PCS of the tests.
pub type Pcs = ClearPcs<K>;

/// `ClearPcs` over the field of `ctx`.
pub fn clear_pcs(ctx: &Context<K>) -> Pcs {
    ClearPcs::from_field(ctx.field())
}

/// The words of `ClearPcs`'s items at a layout, from its size formulas: 4-word commitments, a
/// multilinear proof of $6 + 2^{\ell''}$ words and vector proofs of $6 + K(1 + \ell'' d\_b)$
/// words.
pub fn clear_words(l: &Layout) -> BasePcsWords {
    let c = Pcs::COMMITMENT_WORDS as u128;
    let vector = |degree: usize| Pcs::vector_proof_words(1 + l.ell_pp * degree).unwrap() as u128;
    BasePcsWords {
        commitment_w: c,
        commitment_g0: c,
        commitment_g1: c,
        proof_w: Pcs::multilinear_proof_words(l.ell_pp).unwrap() as u128,
        proof_g0: vector(l.d0),
        proof_g1: vector(l.d1),
    }
}

/// A proof without its base-PCS words: what the oracle records.
pub fn without_pcs(proof: &EvalProof<K>) -> EvalProof<K> {
    let mut p = proof.clone();
    p.sumcheck.commitments = PcsWords::default();
    p.sumcheck.openings = PcsWords::default();
    p
}

/// The fixture `name` and its context (with the recorded matrices).
pub fn load(name: &str) -> (Fixture, Context<K>) {
    let fx = Fixture::load(name).unwrap();
    let ctx = fx.context::<K>().unwrap();
    (fx, ctx)
}

/// A 32-byte seed from a tag and indices: SHA-256 of the tag and the indices as LE64.
pub fn seed(tag: &str, indices: &[u64]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(tag.as_bytes());
    for i in indices {
        h.update(i.to_le_bytes());
    }
    h.finalize().into()
}

/// The derived parameters of a named preset.
pub fn params_of(name: &str) -> Params {
    derive(&preset(name).unwrap().choices).unwrap().0
}

/// The setup seed of (`name`, `index`).
pub fn setup_seed(name: &str, index: u64) -> [u8; 32] {
    seed(&format!("setup/{name}"), &[index])
}

/// A fresh context of the preset `name` from `Setup` with the seed of (`name`, `index`).
pub fn fresh_context(name: &str, index: u64) -> Context<K> {
    hachi_co::setup(params_of(name), &setup_seed(name, index)).unwrap()
}

/// Hachi as the base PCS of the context of (`name`, `index`) ([`fresh_context`]), from the same
/// setup seed, with its default choices and the given number of iterations (`None`: the
/// automatic schedule, the number of iterations whose proof is smallest).
pub fn hachi_base(
    ctx: &Context<K>,
    name: &str,
    index: u64,
    iterations: Option<usize>,
) -> HachiBase<K> {
    let choices = HachiChoices {
        fixed_iterations: iterations,
        ..HachiBase::default_choices(ctx, &setup_seed(name, index))
    };
    HachiBase::new(ctx.q(), choices).unwrap()
}

/// `len` uniform elements of $\mathbb F\_{q^K}$.
pub fn random_vector(rng: &mut ChaCha8Rng, q: u64, len: usize) -> Vec<Ext<K>> {
    (0..len)
        .map(|_| Ext(core::array::from_fn(|_| rng.random_range(0..q))))
        .collect()
}

/// An additive sharing of $\mathbf f$ among $n$ parties: $n - 1$ uniform shares and the
/// difference.
pub fn additive_shares(
    ctx: &Context<K>,
    rng: &mut ChaCha8Rng,
    f: &[Ext<K>],
    n: usize,
) -> Vec<Vec<Ext<K>>> {
    let field = ctx.field();
    let mut shares: Vec<Vec<Ext<K>>> = (1..n)
        .map(|_| random_vector(rng, ctx.q(), f.len()))
        .collect();
    let last = (0..f.len())
        .map(|t| shares.iter().fold(f[t], |acc, s| field.sub(&acc, &s[t])))
        .collect();
    shares.push(last);
    shares
}

/// $f(\mathbf x) = \sum\_{\mathbf q} f\_{\mathbf q} \prod\_j x\_j^{q\_j}$ by the explicit table of
/// monomials (MSB-first $\mathrm{nat}$: $x\_1$ goes with the most significant bit), a path
/// independent of the protocol and of `hachi_co_conv::evaluate` (which folds).
pub fn evaluate_by_monomials(field: &Fqk<K>, f: &[Ext<K>], x: &[Ext<K>]) -> Ext<K> {
    let mut monomials = vec![field.one()];
    for xj in x {
        monomials = monomials
            .iter()
            .flat_map(|m| [*m, field.mul(m, xj)])
            .collect();
    }
    assert_eq!(monomials.len(), f.len());
    monomials.iter().zip(f).fold(field.zero(), |acc, (m, c)| {
        field.add(&acc, &field.mul(m, c))
    })
}

/// A random instance for `ctx`: the coefficients $\mathbf f$, a point $\mathbf x$ and
/// $y = f(\mathbf x)$ ([`evaluate_by_monomials`]), from `ChaCha8Rng::seed_from_u64(seed)`.
pub fn random_instance(ctx: &Context<K>, seed: u64) -> (Vec<Ext<K>>, Vec<Ext<K>>, Ext<K>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let l = &ctx.layout;
    let f = random_vector(&mut rng, ctx.q(), 1 << l.ell);
    let x = random_vector(&mut rng, ctx.q(), l.ell);
    let y = evaluate_by_monomials(ctx.field(), &f, &x);
    (f, x, y)
}

/// The statement of a fixture: the commitment `commit/u`, the point `input/x` and the claimed
/// evaluation `input/y`.
pub fn fixture_statement(fx: &Fixture) -> (Vec<Poly>, Vec<Ext<K>>, Ext<K>) {
    (
        fx.polys("/commit/u").unwrap(),
        fx.exts("/input/x").unwrap(),
        fx.ext("/input/y").unwrap(),
    )
}

/// The sumcheck transcript recorded in a fixture (`sumcheck/G0`, `G1`, `rounds`, `final`),
/// without base-PCS words (the oracle does not model them).
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

/// The proof assembled from a fixture's recorded values (`conv/E_u`, the accepted attempt and
/// its `v`, and the sumcheck transcript), without base-PCS words: what the oracle records.
pub fn fixture_proof_without_pcs(fx: &Fixture) -> EvalProof<K> {
    let accepted = fx.u64("/ring_to_linear/accepted_attempt").unwrap();
    EvalProof {
        e_u: fx.poly("/conv/E_u").unwrap(),
        attempt: accepted,
        v: fx
            .polys(&format!("/ring_to_linear/attempts/{accepted}/v"))
            .unwrap(),
        sumcheck: fixture_transcript(fx),
    }
}

/// The parties' committed polynomials as the oracle records them, rebuilt from the recorded
/// inputs: the table of $\langle \widetilde w \rangle\_p$ from $\mathbf z'$ (`ring_to_linear/
/// z_prime`), $\mathbf y$, the shares $\langle u \rangle\_p$ (`conv/u_shares`) and $\mathbf M$
/// ([`LinearShare::slice`], [`quotient_share`], [`w_table`]), and the recorded blinders
/// ([`ReplayRandomness`]), party by party.
pub fn recorded_polynomials(
    fx: &Fixture,
    ctx: &Context<K>,
) -> Vec<(Vec<u64>, Blinder<K>, Blinder<K>)> {
    let l = &ctx.layout;
    let m = fixture_m(fx, ctx);
    let z_prime = fx.polys("/ring_to_linear/z_prime").unwrap();
    let y = fx.polys("/ring_to_linear/y").unwrap();
    let u_shares = fx.polys("/conv/u_shares").unwrap();
    (1..=fx.n())
        .map(|p| {
            let share = LinearShare::slice(l, &z_prime, &y, &u_shares[p - 1], p).unwrap();
            let quotient = quotient_share(ctx, &m, &share).unwrap();
            let table = w_table(ctx, &share.z_prime, &quotient).unwrap();
            let mut rng = ReplayRandomness::new(fx, p).unwrap();
            let g0 = rng
                .blinder(ctx.field(), BlinderKind::G0, l.ell_pp, l.d0)
                .unwrap();
            let g1 = rng
                .blinder(ctx.field(), BlinderKind::G1, l.ell_pp, l.d1)
                .unwrap();
            (table, g0, g1)
        })
        .collect()
}

/// The `ClearPcs` commitments and openings of the parties' polynomials (the table of
/// $\langle \widetilde w \rangle\_p$ through the Möbius transform, and the blinders' coefficient
/// vectors) at `point`, under the contexts of `ch` ([`opening_label`]), done independently of
/// the driver.
pub fn clear_pcs_words<H: Challenger<K>>(
    ctx: &Context<K>,
    polynomials: &[(Vec<u64>, Blinder<K>, Blinder<K>)],
    point: &[Ext<K>],
    ch: &mut H,
) -> (PcsWords, PcsWords) {
    let (field, l) = (ctx.field(), &ctx.layout);
    let pcs = clear_pcs(ctx);
    let (mut commitments, mut openings) = (PcsWords::default(), PcsWords::default());
    for (p0, (table, g0, g1)) in polynomials.iter().enumerate() {
        let mut coeffs = table.clone();
        mobius(field.zq(), &mut coeffs).unwrap();
        let (cw, sw) = pcs.commit_multilinear(&coeffs).unwrap();
        let (c0, s0) = pcs.commit_vector(&blinder_coefficients(g0)).unwrap();
        let (c1, s1) = pcs.commit_vector(&blinder_coefficients(g1)).unwrap();
        for (poly, com) in [
            (ClaimedPoly::W, cw),
            (ClaimedPoly::G0, c0),
            (ClaimedPoly::G1, c1),
        ] {
            commitments.of_mut(poly).push(pcs.commitment_to_words(&com));
            let context = ch.context(&opening_label(p0 + 1, poly));
            let (_, proof) = match poly {
                ClaimedPoly::W => pcs.prove_multilinear(&sw, point, &context).unwrap(),
                ClaimedPoly::G0 => pcs
                    .prove_inner_product(&s0, &blinder_weights(field, point, l.d0), &context)
                    .unwrap(),
                ClaimedPoly::G1 => pcs
                    .prove_inner_product(&s1, &blinder_weights(field, point, l.d1), &context)
                    .unwrap(),
            };
            openings.of_mut(poly).push(pcs.proof_to_words(&proof));
        }
    }
    (commitments, openings)
}

/// The proof assembled from a fixture's recorded values ([`fixture_proof_without_pcs`]) with the
/// `ClearPcs` commitments and openings of the recorded polynomials
/// ([`recorded_polynomials`]) at the recorded point, under the replay challenger's contexts.
pub fn fixture_proof(fx: &Fixture) -> EvalProof<K> {
    let ctx = fx.context::<K>().unwrap();
    let mut proof = fixture_proof_without_pcs(fx);
    let point = fx.exts::<K>("/sumcheck/coins/a").unwrap();
    let mut ch = ReplayChallenger::new(fx).unwrap();
    let (commitments, openings) =
        clear_pcs_words(&ctx, &recorded_polynomials(fx, &ctx), &point, &mut ch);
    proof.sumcheck.commitments = commitments;
    proof.sumcheck.openings = openings;
    proof
}

/// The claims the verifier must return for the fixture: the point `sumcheck/coins/a` and, party
/// by party, $g\_{0,p}(\mathbf a)$, $g\_{1,p}(\mathbf a)$,
/// $\langle \widetilde w \rangle\_p(\mathbf a)$ from `sumcheck/final`.
pub fn fixture_claims(fx: &Fixture) -> FinalClaims<K> {
    let t = fixture_transcript(fx);
    let claims = (0..fx.n())
        .flat_map(|p0| {
            [
                (ClaimedPoly::G0, t.finals.g0[p0]),
                (ClaimedPoly::G1, t.finals.g1[p0]),
                (ClaimedPoly::W, t.finals.w[p0]),
            ]
            .map(|(poly, value)| Claim {
                party: p0 + 1,
                poly,
                value,
            })
        })
        .collect();
    FinalClaims {
        point: fx.exts("/sumcheck/coins/a").unwrap(),
        claims,
    }
}

/// Party $p$'s state as recorded by the oracle's Commit ([`Fixture::party_state`]).
pub fn recorded_state(fx: &Fixture, ctx: &Context<K>, p: usize) -> PartyState<K> {
    fx.party_state(ctx, p).unwrap()
}

/// The oracle's `IdealPCS` (`ref/README.md`, choice 13), as in the sumcheck crate's tests: a
/// claim is accepted iff its value equals the committed polynomial of its party evaluated
/// directly at the point, on a path independent of the provers' folding:
/// $\langle \widetilde w \rangle\_p(\mathbf a) = \sum\_x \widetilde{eq}(\mathbf a, x)
/// W\_p\[x\]$ and $g(\mathbf a) = s\_0 + \sum\_{j,k} s\_{j,k} a\_j^k$ with every power by
/// square and multiply.
/// Returns the oracle's check name of each claim with the outcome.
pub fn ideal_pcs(
    field: &Fqk<K>,
    claims: &FinalClaims<K>,
    privates: &[&SumcheckPrivate<K>],
) -> Vec<(String, bool)> {
    let eq = eq_table(field, &claims.point).unwrap();
    claims
        .claims
        .iter()
        .map(|claim| {
            let private = privates[claim.party - 1];
            assert_eq!(private.party, claim.party);
            assert_eq!(private.point, claims.point, "the provers' point");
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

/// Asserts that every claim matches the provers' polynomials ([`ideal_pcs`]) and that there are
/// $3n$ of them, in the order $g\_0, g\_1, \widetilde w$ party by party.
pub fn assert_claims_match(
    field: &Fqk<K>,
    claims: &FinalClaims<K>,
    privates: &[&SumcheckPrivate<K>],
    what: &str,
) {
    let n = privates.len();
    assert_eq!(claims.claims.len(), 3 * n, "{what}");
    for (i, claim) in claims.claims.iter().enumerate() {
        assert_eq!(claim.party, i / 3 + 1, "{what}");
        let poly = [ClaimedPoly::G0, ClaimedPoly::G1, ClaimedPoly::W][i % 3];
        assert_eq!(claim.poly, poly, "{what}");
    }
    let pcs = ideal_pcs(field, claims, privates);
    assert!(pcs.iter().all(|(_, ok)| *ok), "{what}: {pcs:?}");
}

/// The matrix $\mathbf M$ of a fixture's accepted attempt, rebuilt from the recorded public
/// vectors and challenges (for the tests that compare the provers' relation).
pub fn fixture_m(fx: &Fixture, ctx: &Context<K>) -> Vec<Vec<Poly>> {
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
    build_m(ctx, &embed("/conv/b"), &embed("/conv/a"), &c).unwrap()
}

/// One call made to a [`Recorder`]: the method, the label (for absorbs), the number of absorbed
/// items, and a fingerprint of the absorbed values (the sum of the residues with wrapping), or
/// the arguments of a coin request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Call {
    /// `absorb_polys(label, polys)`: label, number of polynomials, fingerprint.
    Polys(&'static str, usize, u64),
    /// `absorb_exts(label, exts)`: label, number of elements, fingerprint.
    Exts(&'static str, usize, u64),
    /// `absorb_words(label, words)`: label, number of word vectors, fingerprint.
    Words(&'static str, usize, u64),
    /// `absorb_u64(label, value)`.
    U64(&'static str, u64),
    /// `context(label)`: the label and the context.
    Context(String, [u8; 32]),
    /// `begin_attempt(a)`.
    Begin(u64),
    /// `r2l_challenges(a, count, d, omega)`.
    R2l(u64, usize, usize, u64),
    /// `sumcheck_coins(.., l'', log n, log nu)`.
    Coins(usize, usize, usize),
    /// `round_coin(.., r)`.
    Round(usize),
}

/// A challenger that forwards every call to `inner` and records it in `log`.
pub struct Recorder<H> {
    pub inner: H,
    pub log: Vec<Call>,
}

impl<H> Recorder<H> {
    pub fn new(inner: H) -> Self {
        Self {
            inner,
            log: Vec::new(),
        }
    }
}

fn fingerprint(values: impl Iterator<Item = u64>) -> u64 {
    values.fold(0u64, u64::wrapping_add)
}

impl<H: Challenger<K>> Challenger<K> for Recorder<H> {
    fn absorb_polys(&mut self, label: &'static str, polys: &[Poly]) {
        let fp = fingerprint(polys.iter().flat_map(|p| p.0.iter().copied()));
        self.log.push(Call::Polys(label, polys.len(), fp));
        Challenger::<K>::absorb_polys(&mut self.inner, label, polys);
    }

    fn absorb_exts(&mut self, label: &'static str, exts: &[Ext<K>]) {
        let fp = fingerprint(exts.iter().flat_map(|e| e.0));
        self.log.push(Call::Exts(label, exts.len(), fp));
        self.inner.absorb_exts(label, exts);
    }

    fn absorb_words(&mut self, label: &'static str, words: &[Vec<u64>]) {
        let fp = fingerprint(words.iter().flatten().copied());
        self.log.push(Call::Words(label, words.len(), fp));
        Challenger::<K>::absorb_words(&mut self.inner, label, words);
    }

    fn absorb_u64(&mut self, label: &'static str, value: u64) {
        self.log.push(Call::U64(label, value));
        Challenger::<K>::absorb_u64(&mut self.inner, label, value);
    }

    fn context(&mut self, label: &str) -> [u8; 32] {
        let context = Challenger::<K>::context(&mut self.inner, label);
        self.log.push(Call::Context(label.to_owned(), context));
        context
    }

    fn begin_attempt(&mut self, attempt: u64) -> hachi_co_core::Result<()> {
        self.log.push(Call::Begin(attempt));
        Challenger::<K>::begin_attempt(&mut self.inner, attempt)
    }

    fn r2l_challenges(
        &mut self,
        attempt: u64,
        count: usize,
        d: usize,
        omega: u64,
    ) -> hachi_co_core::Result<Vec<Vec<i64>>> {
        self.log.push(Call::R2l(attempt, count, d, omega));
        Challenger::<K>::r2l_challenges(&mut self.inner, attempt, count, d, omega)
    }

    fn sumcheck_coins(
        &mut self,
        field: &Fqk<K>,
        ell_pp: usize,
        log_n: usize,
        log_nu: usize,
    ) -> hachi_co_core::Result<SumcheckCoins<K>> {
        self.log.push(Call::Coins(ell_pp, log_n, log_nu));
        self.inner.sumcheck_coins(field, ell_pp, log_n, log_nu)
    }

    fn round_coin(&mut self, field: &Fqk<K>, round: usize) -> hachi_co_core::Result<Ext<K>> {
        self.log.push(Call::Round(round));
        self.inner.round_coin(field, round)
    }
}
