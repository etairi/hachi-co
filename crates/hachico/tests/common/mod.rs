//! Shared helpers of the integration tests: the fixtures (context, statement, the proof built
//! from the recorded values), seeded inputs and an independent evaluation of $f(\mathbf x)$, an
//! ideal PCS for the final claims (the oracle's `IdealPCS`, as in the sumcheck crate's tests), a
//! challenger that records the calls made to it, and the threaded $n$-party test network.
#![allow(dead_code)]

pub mod net;

use hachi_co::{EvalProof, FinalClaims};
use hachi_co_core::{
    Challenger, Context, Ext, Fixture, Fqk, PartyState, Poly, SumcheckCoins, build_m,
};
use hachi_co_params::{Params, derive, preset};
use hachi_co_sumcheck::{
    Claim, ClaimedPoly, FinalValues, RoundMessage, SumcheckPrivate, SumcheckTranscript, eq_table,
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};

/// The extension degree of every toy and development preset.
pub const K: usize = 4;

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

/// A fresh context of the preset `name` from `Setup` with the seed of (`name`, `index`).
pub fn fresh_context(name: &str, index: u64) -> Context<K> {
    hachi_co::setup(params_of(name), &seed(&format!("setup/{name}"), &[index])).unwrap()
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

/// The sumcheck transcript recorded in a fixture (`sumcheck/G0`, `G1`, `rounds`, `final`).
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
    }
}

/// The proof assembled from a fixture's recorded values: `conv/E_u`, the accepted attempt and
/// its `v`, and the sumcheck transcript.
pub fn fixture_proof(fx: &Fixture) -> EvalProof<K> {
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

/// Party $p$'s state as recorded by the oracle's Commit (`commit/f_shares`, `commit/f_prime`,
/// `commit/blocks`).
pub fn recorded_state(fx: &Fixture, ctx: &Context<K>, p: usize) -> PartyState<K> {
    let l = &ctx.layout;
    let f_prime = fx.polys("/commit/f_prime").unwrap();
    let recorded = fx.get("/commit/blocks").unwrap().as_array().unwrap().len();
    let blocks = l
        .blocks_of(p)
        .map(|i| {
            let j = (0..recorded)
                .find(|j| fx.usize(&format!("/commit/blocks/{j}/index")).unwrap() == i)
                .unwrap();
            let key = |k: &str| fx.polys(&format!("/commit/blocks/{j}/{k}")).unwrap();
            hachi_co_core::BlockState {
                index: i,
                f_prime: f_prime[i * l.block_len..(i + 1) * l.block_len].to_vec(),
                s: key("s"),
                t: key("t"),
                t_hat: key("t_hat"),
            }
        })
        .collect();
    PartyState {
        party: p,
        share: fx.exts(&format!("/commit/f_shares/{}", p - 1)).unwrap(),
        blocks,
    }
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
    /// `absorb_u64(label, value)`.
    U64(&'static str, u64),
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

    fn absorb_u64(&mut self, label: &'static str, value: u64) {
        self.log.push(Call::U64(label, value));
        Challenger::<K>::absorb_u64(&mut self.inner, label, value);
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
