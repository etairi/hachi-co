//! The verifier of the collaborative sumcheck (`5_Building_Blocks.tex`:276-334; oracle
//! `sumcheck.verify` and `sumcheck.mstar_eval_factorized`), with the openings of the base PCS.

use crate::blinder::blinder_weights;
use crate::driver::{
    LABEL_COMMIT_G0, LABEL_COMMIT_G1, LABEL_COMMIT_W, LABEL_FINAL_G0, LABEL_FINAL_G1,
    LABEL_FINAL_W, LABEL_G0, LABEL_G1, LABEL_H0, LABEL_H1, opening_label,
};
use crate::eq::{eq_eval, eq_point, eq_table, msb_bits};
use crate::error::{Error, Result, VerifyError, check_len, pow2};
use crate::interp::Nodes;
use crate::quotient::check_matrix;
use crate::tables::{RangeProduct, claimed_sum, column_weights, ind_rows, powers};
use crate::transcript::{Claim, ClaimedPoly, FinalClaims, SumcheckTranscript};
use core::ops::Range;
use hachi_co_core::{BasePcs, Challenger, Context, Layout, Poly};
use hachi_co_math::{Ext, Fqk};

/// The public inputs of the verifier, all computed from public data
/// (`5_Building_Blocks.tex`:154-182, 251-266; the oracle's `oracle.verifier_public`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicInput<'a> {
    /// $\mathbf M \in \mathcal R\_q^{n' \times \mu}$ of the accepted attempt
    /// ([`hachi_co_core::build_m`]).
    pub m: &'a [Vec<Poly>],
    /// $\mathbf y = (\mathbf v, \mathbf u, 0, 0, \mathbf 0) \in \mathcal R\_q^{n'}$
    /// ([`hachi_co_core::build_y`]).
    pub y: &'a [Poly],
    /// The $d \times d$ matrix of $E$: `e_matrix[i][j]` is coefficient $i$ of $E(X^j)$
    /// ([`hachi_co_math::Subfield::e_matrix`] at the point $v$ of $\Pi\_{\mathsf{conv}}$).
    pub e_matrix: &'a [Vec<u64>],
    /// The claimed evaluation $y \in \mathbb F\_{q^k}$ embedded in $\mathcal R\_q^H$
    /// ([`hachi_co_math::Subfield::embed`]; `ref/README.md`, choice 12).
    pub y_ring: &'a Poly,
    /// $S\_1, \dots, S\_n$ ([`Layout::s_ranges`]), entry $p - 1$ for party $p$.
    pub partition: Vec<Vec<Range<usize>>>,
}

impl<'a> PublicInput<'a> {
    /// The public input with the partition of `ctx.layout`.
    pub fn new<const K: usize>(
        ctx: &Context<K>,
        m: &'a [Vec<Poly>],
        y: &'a [Poly],
        e_matrix: &'a [Vec<u64>],
        y_ring: &'a Poly,
    ) -> Self {
        Self {
            m,
            y,
            e_matrix,
            y_ring,
            partition: (1..=ctx.layout.n).map(|p| ctx.layout.s_ranges(p)).collect(),
        }
    }

    /// Checks every shape ($\mathbf M$: $n' \times \mu$; $\mathbf y$: $n'$; $E$: $d \times d$; $d$
    /// coefficients per polynomial; $n$ lists of ranges inside $\[0, 2^{\ell\_U})$) and that every
    /// residue is canonical.
    ///
    /// # Errors
    /// [`VerifyError::Shape`] or [`VerifyError::NonCanonical`].
    pub fn validate(&self, layout: &Layout, q: u64) -> core::result::Result<(), VerifyError> {
        let d = layout.d;
        check_matrix(self.m, layout.n_prime, layout.mu, d)?;
        check_len("y (one entry per row of M)", layout.n_prime, self.y.len())?;
        check_len("E matrix (rows)", d, self.e_matrix.len())?;
        check_len("y as a ring element", d, self.y_ring.0.len())?;
        for p in self.y {
            check_len("y (coefficients)", d, p.0.len())?;
        }
        for row in self.e_matrix {
            check_len("E matrix (columns)", d, row.len())?;
        }
        let residues = self
            .m
            .iter()
            .flatten()
            .chain(self.y)
            .chain(core::iter::once(self.y_ring))
            .flat_map(|p| p.0.iter())
            .chain(self.e_matrix.iter().flatten());
        for value in residues {
            if *value >= q {
                return Err(VerifyError::NonCanonical {
                    what: "public input (M, y, E or y as a ring element)",
                    value: *value,
                    q,
                });
            }
        }
        check_len(
            "partition (one list per party)",
            layout.n,
            self.partition.len(),
        )?;
        let rows = pow2(layout.ell_u, "the U part of the hypercube")?;
        for r in self.partition.iter().flatten() {
            if r.start > r.end || r.end > rows {
                return Err(VerifyError::Shape {
                    what: "partition range (within the U part of the hypercube)",
                    expected: rows,
                    actual: r.end,
                });
            }
        }
        Ok(())
    }
}

/// One named check of the verifier and its outcome.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Check {
    /// The oracle's name of the check (see [`check_names`]).
    pub name: String,
    /// Whether it holds.
    pub passed: bool,
}

/// One base-PCS opening as the verifier checked it (`5_Building_Blocks.tex`:321-326; the
/// oracle's check `pcs_open_<poly>_p<party>`, [`Claim::check_name`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningCheck<const K: usize> {
    /// The claim: the party, the polynomial and the final value the opening must prove.
    pub claim: Claim<K>,
    /// `None` if the base PCS accepted the opening, otherwise the reason of the rejection (a
    /// commitment or proof that does not decode, or a proof that does not verify).
    pub error: Option<String>,
}

impl<const K: usize> OpeningCheck<K> {
    /// The named check: [`Claim::check_name`] and whether the opening was accepted.
    pub fn check(&self) -> Check {
        Check {
            name: self.claim.check_name(),
            passed: self.error.is_none(),
        }
    }
}

/// Everything the verifier computed: every named check, the openings of the base PCS, the
/// checked claims, and the two public values the fixtures record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierReport<const K: usize> {
    /// The checks of the sumcheck itself: the round checks `round_01_H0`, `round_01_H1`, …,
    /// `round_<l''>_H1` and the final checks `final_H0`, `final_H1`, in the oracle's order.
    pub checks: Vec<Check>,
    /// The $3n$ openings of the base PCS (`pcs_open_*`), party by party in the order
    /// [`ClaimedPoly::ORDER`].
    pub openings: Vec<OpeningCheck<K>>,
    /// The point $\mathbf a$ and the $3n$ evaluation claims that the openings check.
    pub claims: FinalClaims<K>,
    /// The claimed sum $a$ (`5_Building_Blocks.tex`:263-264; fixture key `sumcheck/a_claim`).
    pub claimed_sum: Ext<K>,
    /// $\widetilde M^\ast(\mathbf a)$ (fixture key `sumcheck/mstar_at_a`).
    pub mstar_at_a: Ext<K>,
}

impl<const K: usize> VerifierReport<K> {
    /// Every named check with its outcome, in the oracle's order ([`check_names`]): the round
    /// checks, the $3n$ openings, the final checks.
    pub fn named_checks(&self) -> Vec<Check> {
        let split = self.checks.len().saturating_sub(2);
        let (rounds, finals) = self.checks.split_at(split);
        rounds
            .iter()
            .cloned()
            .chain(self.openings.iter().map(OpeningCheck::check))
            .chain(finals.iter().cloned())
            .collect()
    }

    /// The names of the failing checks, openings included, in the oracle's order.
    pub fn failing(&self) -> Vec<String> {
        self.named_checks()
            .into_iter()
            .filter(|c| !c.passed)
            .map(|c| c.name)
            .collect()
    }

    /// Whether every check holds, openings included.
    pub fn accepted(&self) -> bool {
        self.checks.iter().all(|c| c.passed) && self.openings.iter().all(|o| o.error.is_none())
    }

    /// The verdict of [`verify_claims`]: [`VerifyError::Rejected`] with the failing round and
    /// final checks, if any; otherwise [`VerifyError::Opening`] for the first failing opening,
    /// if any; otherwise the checked claims.
    ///
    /// # Errors
    /// As described.
    pub fn verdict(&self) -> core::result::Result<FinalClaims<K>, VerifyError> {
        let failing: Vec<String> = self
            .checks
            .iter()
            .filter(|c| !c.passed)
            .map(|c| c.name.clone())
            .collect();
        if !failing.is_empty() {
            return Err(VerifyError::Rejected { failing });
        }
        if let Some((claim, reason)) = self
            .openings
            .iter()
            .find_map(|o| o.error.as_ref().map(|e| (&o.claim, e)))
        {
            return Err(VerifyError::Opening {
                party: claim.party,
                poly: claim.poly,
                reason: reason.clone(),
            });
        }
        Ok(self.claims.clone())
    }
}

/// The oracle's name of the round check of $H\_{i,r}$, $i \in \lbrace 0, 1 \rbrace$:
/// `round_<rr>_H<i>` with $r$ in two digits (`round_01_H0`).
pub fn round_check_name(r: usize, which: usize) -> String {
    format!("round_{r:02}_H{which}")
}

/// The name of the final check of $H\_0$ (`5_Building_Blocks.tex`:331).
pub const FINAL_H0: &str = "final_H0";
/// The name of the final check of $H\_1$ (`5_Building_Blocks.tex`:332).
pub const FINAL_H1: &str = "final_H1";

/// Every check of the oracle's verifier of the sumcheck, in its order
/// (`ref/hachico_ref/sumcheck.py`, `verify`; fixture key `verifier/checks` without the
/// $\Pi\_{\mathsf{conv}}$ check `conv_y_eq_E(u)`): the $2\ell''$ round checks, the $3n$ PCS openings
/// `pcs_open_g0_p<p>`, `pcs_open_g1_p<p>`, `pcs_open_w_p<p>` ([`Claim::check_name`]), and
/// `final_H0`, `final_H1`. [`verify_checks`] evaluates all of them, the openings with the base
/// PCS ([`VerifierReport::named_checks`]); the oracle checks the openings with its ideal PCS.
pub fn check_names(layout: &Layout) -> Vec<String> {
    let mut names = Vec::with_capacity(2 * layout.ell_pp + 3 * layout.n + 2);
    for r in 1..=layout.ell_pp {
        names.push(round_check_name(r, 0));
        names.push(round_check_name(r, 1));
    }
    for p in 1..=layout.n {
        for poly in ClaimedPoly::ORDER {
            let claim = Claim::<1> {
                party: p,
                poly,
                value: Ext([0]),
            };
            names.push(claim.check_name());
        }
    }
    names.push(FINAL_H0.to_owned());
    names.push(FINAL_H1.to_owned());
    names
}

/// $\widetilde M^\ast(\mathbf a)$, the multilinear extension of the table of
/// [`mstar_table`](crate::mstar_table) at $\mathbf a = (\mathbf a\_U, \mathbf a\_L)$, evaluated from
/// the definition (`5_Building_Blocks.tex`:254-260) term by term with
/// $\widetilde{eq}(\mathbf a\_U, \cdot)$ and $\widetilde{eq}(\mathbf a\_L, \cdot)$ factored out, without
/// the $2^{\ell''}$ table (the oracle's `mstar_eval_factorized`): with
/// $A\_L = \sum\_{l\_0 \lt 2d} \widetilde{eq}(\mathbf a\_L, \mathsf{bit}(l\_0)) \alpha^{l\_0} =
/// \prod\_{j=1}^{\ell\_L} \bigl(1 - a\_{L,j} + a\_{L,j} \alpha^{2^{\ell\_L - j}}\bigr)$,
/// $$\widetilde M^\ast(\mathbf a) = A\_L \sum\_{u \lt \mu} \mathrm{eq}\_U\[u\] c\_u -
/// (\alpha^d + 1) A\_L \sum\_{i\_0 \lt n'} \mathrm{eq}\_1\[i\_0\] \mathrm{eq}\_U\[\mu + i\_0\] +
/// \mathrm{eq}\_U\[\mu - 1\] \sum\_{j\_0 \lt d} \mathrm{eq}\_1\[n' + j\_0\] \sum\_{l\_0 \lt d}
/// E\[j\_0\]\[l\_0\] \mathrm{eq}\_L\[l\_0\] +
/// \sum\_{d \le l\_0 \lt 2d} \mathrm{eq}\_L\[l\_0\] \sum\_{u \lt \mu + n'} \mathrm{eq}\_U\[u\]
/// \mathrm{eq}\_1\[n' + d + u + (l\_0 - d)(\mu + n')\],$$
/// with $c\_u$ and $\mathrm{eq}\_1$ as in [`mstar_table`](crate::mstar_table). The cost is
/// $O(\text{nonzero coefficients of } \mathbf M + 2^{\ell\_U} + d(\mu + n') + d^2)$, linear in the
/// size of the relation (the verifier evaluates $\mathbf M$ at $\alpha$ anyway).
///
/// # Errors
/// [`Error::Shape`] for inputs of the wrong shape (as [`mstar_table`](crate::mstar_table)) or a
/// point without $\ell''$ coordinates.
pub fn mstar_eval_factorized<const K: usize>(
    ctx: &Context<K>,
    m: &[Vec<Poly>],
    e_matrix: &[Vec<u64>],
    eq1: &[Ext<K>],
    alpha: &Ext<K>,
    point: &[Ext<K>],
) -> Result<Ext<K>> {
    let l = &ctx.layout;
    let f = ctx.field();
    let (d, mu, np) = (l.d, l.mu, l.n_prime);
    check_matrix(m, np, mu, d)?;
    check_len("E matrix (rows)", d, e_matrix.len())?;
    for row in e_matrix {
        check_len("E matrix (columns)", d, row.len())?;
    }
    check_len(
        "eq(tau_1) table (2^ceil(log nu) entries)",
        pow2(l.log_nu, "the eq(tau_1) table")?,
        eq1.len(),
    )?;
    check_len("point (l'' coordinates)", l.ell_pp, point.len())?;
    let (a_u, a_l) = point.split_at(l.ell_u);
    let eq_u = eq_table(f, a_u)?;
    let eq_l = eq_table(f, a_l)?;
    let one = f.one();
    // A_L, with alpha^{2^{l_L - j}} by repeated squaring from the last coordinate.
    let mut a_sum = one;
    let mut square = *alpha;
    for a in a_l.iter().rev() {
        let factor = f.add(&f.sub(&one, a), &f.mul(a, &square));
        a_sum = f.mul(&a_sum, &factor);
        square = f.square(&square);
    }
    let apow = powers(f, alpha, d + 1);
    let c = column_weights(f, m, eq1, &apow[..d]);
    let dot = |x: &[Ext<K>], y: &[Ext<K>]| {
        x.iter()
            .zip(y)
            .fold(f.zero(), |acc, (a, b)| f.add(&acc, &f.mul(a, b)))
    };
    let t1a = f.mul(&dot(&eq_u[..mu], &c), &a_sum);
    let ad1 = f.add(&apow[d], &one);
    let t1b = f.neg(&f.mul(&f.mul(&ad1, &a_sum), &dot(&eq1[..np], &eq_u[mu..mu + np])));
    let zq = f.zq();
    let mut inner = f.zero();
    for (j0, row) in e_matrix.iter().enumerate() {
        let mut row_sum = f.zero();
        for (coef, e) in row.iter().zip(&eq_l[..d]) {
            let coef = zq.reduce(*coef);
            if coef != 0 {
                row_sum = f.add(&row_sum, &f.scale(e, coef));
            }
        }
        inner = f.add(&inner, &f.mul(&eq1[np + j0], &row_sum));
    }
    let t2 = f.mul(&eq_u[mu - 1], &inner);
    let mut t3 = f.zero();
    for (l0, el) in eq_l.iter().enumerate().skip(d) {
        let start = np + d + (l0 - d) * (mu + np);
        t3 = f.add(
            &t3,
            &f.mul(el, &dot(&eq_u[..mu + np], &eq1[start..start + mu + np])),
        );
    }
    Ok(f.add(&f.add(&t1a, &t1b), &f.add(&t2, &t3)))
}

/// $\sum$ of field elements.
fn sum<const K: usize>(f: &Fqk<K>, values: &[Ext<K>]) -> Ext<K> {
    values.iter().fold(f.zero(), |acc, v| f.add(&acc, v))
}

/// One opening checked with the base PCS: decode the commitment and the proof, then verify the
/// multilinear evaluation ($\widetilde w$) or the inner product with `weights` (the blinders).
#[allow(clippy::too_many_arguments)] // one opening's instance
fn check_opening<const K: usize, P: BasePcs<K>>(
    pcs: &P,
    poly: ClaimedPoly,
    com_words: &[u64],
    proof_words: &[u64],
    point: &[Ext<K>],
    weights: &[Ext<K>],
    value: &Ext<K>,
    context: &[u8; 32],
) -> core::result::Result<(), String> {
    let com = pcs
        .commitment_from_words(com_words)
        .map_err(|e| format!("the commitment does not decode: {e}"))?;
    let proof = pcs
        .proof_from_words(proof_words)
        .map_err(|e| format!("the proof does not decode: {e}"))?;
    match poly {
        ClaimedPoly::W => pcs.verify_multilinear(&com, point.len(), point, value, &proof, context),
        ClaimedPoly::G0 | ClaimedPoly::G1 => {
            pcs.verify_inner_product(&com, weights.len(), weights, value, &proof, context)
        }
    }
}

/// The verifier of `5_Building_Blocks.tex`:276-334 with the base PCS `pcs`, returning every
/// named check.
///
/// It makes the same challenger calls as [`prove`](crate::prove): absorb $G\_{0,\cdot}$,
/// $G\_{1,\cdot}$ and the commitments ([`LABEL_COMMIT_W`], [`LABEL_COMMIT_G0`],
/// [`LABEL_COMMIT_G1`]), draw the coins; per round absorb $H\_{0,r}, H\_{1,r}$ and draw $a\_r$;
/// absorb the final values; derive the context of every opening ([`Challenger::context`] with
/// [`opening_label`]). It computes:
/// - $H\_{0,0} = \rho \sum\_p G\_{0,p}$ and $H\_{1,0} = a + \rho \sum\_p G\_{1,p}$, with the claimed sum
///   $a$ of [`claimed_sum`] (lines 276-280);
/// - for every round, `round_<r>_H0`: $H\_{0,r-1}(a\_{r-1}) = H\_{0,r}(0) + H\_{0,r}(1)$, and
///   `round_<r>_H1` likewise, with $H\_{i,r}(0)$ and $H\_{i,r}(a\_r)$ interpolated from the values at
///   $1, \dots, d\_i + 1$ ([`Nodes`]; lines 301-311);
/// - the $3n$ openings `pcs_open_<poly>_p<p>` (lines 321-326), party by party in the order
///   [`ClaimedPoly::ORDER`]: the words of the commitment and of the proof are decoded with
///   `pcs` ([`BasePcs::commitment_from_words`], [`BasePcs::proof_from_words`]), and the proof
///   must open the commitment to the final value: $\langle \widetilde w \rangle\_p(\mathbf a)$ by
///   [`BasePcs::verify_multilinear`] in $\ell''$ variables at $\mathbf a$, $g\_{b,p}(\mathbf a)$ by
///   [`BasePcs::verify_inner_product`] with the weights $(1, (a\_j^k)\_{j,k})$ of
///   [`blinder_weights`] ($1 + \ell'' d\_b$ entries);
/// - `final_H0`: $H\_{0,\ell''}(a\_{\ell''}) = \sum\_p \bigl(F\_{0,p}(\mathbf a) + \rho\,
///   g\_{0,p}(\mathbf a)\bigr)$ with $F\_{0,p}(\mathbf a) = \widetilde{eq}(\tau\_0, (\mathbf a,
///   \mathsf{bit}(p))) \bigl(e\_p \prod\_c (w\_p - c) + (1 - e\_p - \mathrm{ind}) w\_p\bigr)$, where
///   $w\_p = \langle \widetilde w \rangle\_p(\mathbf a)$ is the opened value, $e\_p =
///   \sum\_{u \in S\_p} \widetilde{eq}(\mathbf a\_U, \mathsf{bit}(u))$ and $\mathrm{ind}$ likewise over
///   [`ind_rows`]; `final_H1`: $H\_{1,\ell''}(a\_{\ell''}) = \sum\_p \bigl(w\_p
///   \widetilde M^\ast(\mathbf a) + \rho\, g\_{1,p}(\mathbf a)\bigr)$ with
///   [`mstar_eval_factorized`] (lines 327-333).
///
/// Every check is evaluated, also after a failure, so that the set of failing checks can be
/// compared with the oracle's (whose ideal PCS accepts an opening exactly when the value is the
/// committed polynomial's).
///
/// # Errors
/// [`VerifyError::Shape`] or [`VerifyError::NonCanonical`] for a malformed transcript
/// ([`SumcheckTranscript::validate`]) or public input ([`PublicInput::validate`]);
/// [`VerifyError::Parameter`] if $q \le d\_0$ (the messages cannot be interpolated);
/// [`VerifyError::Challenger`] if the challenger fails. A failing check, opening included, is
/// not an error here.
pub fn verify_checks<const K: usize, P: BasePcs<K>, H: Challenger<K>>(
    ctx: &Context<K>,
    pcs: &P,
    public: &PublicInput<'_>,
    transcript: &SumcheckTranscript<K>,
    ch: &mut H,
) -> core::result::Result<VerifierReport<K>, VerifyError> {
    let l = &ctx.layout;
    let f = ctx.field();
    let q = ctx.q();
    transcript.validate(l, q)?;
    public.validate(l, q)?;
    let param = |e: Error| VerifyError::Parameter(e.to_string());
    let nodes0 = Nodes::new(f.zq(), l.d0 + 1).map_err(param)?;
    let nodes1 = Nodes::new(f.zq(), l.d1 + 1).map_err(param)?;
    ch.absorb_exts(LABEL_G0, &transcript.g0_sums);
    ch.absorb_exts(LABEL_G1, &transcript.g1_sums);
    ch.absorb_words(LABEL_COMMIT_W, &transcript.commitments.w);
    ch.absorb_words(LABEL_COMMIT_G0, &transcript.commitments.g0);
    ch.absorb_words(LABEL_COMMIT_G1, &transcript.commitments.g1);
    let coins = ch
        .sumcheck_coins(f, l.ell_pp, l.log_n, l.log_nu)
        .map_err(VerifyError::Challenger)?;
    coins.validate(l, q).map_err(VerifyError::Challenger)?;
    let rho = f.reduce(&coins.rho);
    let alpha = f.reduce(&coins.alpha);
    let tau0: Vec<Ext<K>> = coins.tau0.iter().map(|t| f.reduce(t)).collect();
    let eq1 = eq_table(f, &coins.tau1)?;
    let a_claim = claimed_sum(ctx, &eq1, public.y, public.y_ring, &alpha)?;
    let mut prev0 = f.mul(&rho, &sum(f, &transcript.g0_sums));
    let mut prev1 = f.add(&a_claim, &f.mul(&rho, &sum(f, &transcript.g1_sums)));
    let mut checks = Vec::with_capacity(2 * l.ell_pp + 2);
    let mut point = Vec::with_capacity(l.ell_pp);
    let zero = f.zero();
    for (i, msg) in transcript.rounds.iter().enumerate() {
        let r = i + 1;
        for (which, (values, nodes, prev)) in
            [(&msg.h0, &nodes0, &prev0), (&msg.h1, &nodes1, &prev1)]
                .into_iter()
                .enumerate()
        {
            let at0 = nodes.eval(f, values, &zero)?;
            checks.push(Check {
                name: round_check_name(r, which),
                passed: *prev == f.add(&at0, &values[0]),
            });
        }
        ch.absorb_exts(LABEL_H0, &msg.h0);
        ch.absorb_exts(LABEL_H1, &msg.h1);
        let a = f.reduce(&ch.round_coin(f, r).map_err(VerifyError::Challenger)?);
        prev0 = nodes0.eval(f, &msg.h0, &a)?;
        prev1 = nodes1.eval(f, &msg.h1, &a)?;
        point.push(a);
    }
    let finals = &transcript.finals;
    ch.absorb_exts(LABEL_FINAL_W, &finals.w);
    ch.absorb_exts(LABEL_FINAL_G0, &finals.g0);
    ch.absorb_exts(LABEL_FINAL_G1, &finals.g1);
    // The openings of the base PCS.
    let weights0 = blinder_weights(f, &point, l.d0);
    let weights1 = blinder_weights(f, &point, l.d1);
    let mut openings = Vec::with_capacity(3 * l.n);
    for p0 in 0..l.n {
        let party = p0 + 1;
        for poly in ClaimedPoly::ORDER {
            let value = finals.of(poly)[p0];
            let context = ch.context(&opening_label(party, poly));
            let weights = if poly == ClaimedPoly::G1 {
                &weights1
            } else {
                &weights0
            };
            let error = check_opening(
                pcs,
                poly,
                &transcript.commitments.of(poly)[p0],
                &transcript.openings.of(poly)[p0],
                &point,
                weights,
                &value,
                &context,
            )
            .err();
            openings.push(OpeningCheck {
                claim: Claim { party, poly, value },
                error,
            });
        }
    }
    // V's own F_{0,p}(a) and F_{1,p}(a).
    let eq_u = eq_table(f, &point[..l.ell_u])?;
    let range_sum = |rows: &Range<usize>| sum(f, &eq_u[rows.clone()]);
    let ind = range_sum(&ind_rows(l));
    let eq_tau_a = eq_eval(f, &tau0[..l.ell_pp], &point)?;
    let mstar = mstar_eval_factorized(ctx, public.m, public.e_matrix, &eq1, &alpha, &point)?;
    let range = RangeProduct::new(f.zq(), ctx.b());
    let one = f.one();
    let (mut total0, mut total1) = (f.zero(), f.zero());
    for (p0, rows) in public.partition.iter().enumerate() {
        let (w, g0, g1) = (finals.w[p0], finals.g0[p0], finals.g1[p0]);
        let e_p = rows
            .iter()
            .fold(f.zero(), |acc, r| f.add(&acc, &range_sum(r)));
        let eq_f = f.mul(
            &eq_tau_a,
            &eq_point(f, &tau0[l.ell_pp..], &msb_bits(p0, l.log_n))?,
        );
        let bracket = f.sub(&f.sub(&one, &e_p), &ind);
        let term = f.add(&f.mul(&e_p, &range.eval(f, &w)), &f.mul(&bracket, &w));
        total0 = f.add(&total0, &f.add(&f.mul(&eq_f, &term), &f.mul(&rho, &g0)));
        total1 = f.add(&total1, &f.add(&f.mul(&w, &mstar), &f.mul(&rho, &g1)));
    }
    checks.push(Check {
        name: FINAL_H0.to_owned(),
        passed: prev0 == total0,
    });
    checks.push(Check {
        name: FINAL_H1.to_owned(),
        passed: prev1 == total1,
    });
    let claims = openings.iter().map(|o| o.claim).collect();
    Ok(VerifierReport {
        checks,
        openings,
        claims: FinalClaims { point, claims },
        claimed_sum: a_claim,
        mstar_at_a: mstar,
    })
}

/// The verifier of the collaborative sumcheck with the base PCS `pcs`
/// (`5_Building_Blocks.tex`:276-334): [`verify_claims`] without the claims. It accepts if every
/// round check, every opening of the base PCS and both final checks hold.
///
/// # Errors
/// As [`verify_claims`].
pub fn verify<const K: usize, P: BasePcs<K>, H: Challenger<K>>(
    ctx: &Context<K>,
    pcs: &P,
    public: &PublicInput<'_>,
    transcript: &SumcheckTranscript<K>,
    ch: &mut H,
) -> core::result::Result<(), VerifyError> {
    verify_claims(ctx, pcs, public, transcript, ch).map(|_| ())
}

/// The verifier of [`verify`], returning the point $\mathbf a$ and the $3n$ claims
/// $g\_{0,p}(\mathbf a)$, $g\_{1,p}(\mathbf a)$, $\langle \widetilde w \rangle\_p(\mathbf a)$
/// (`5_Building_Blocks.tex`:316-326) that the base PCS checked ([`verify_checks`], then
/// [`VerifierReport::verdict`]): for tests and diagnostics.
///
/// # Errors
/// The errors of [`verify_checks`]; [`VerifyError::Rejected`] with the names of the failing
/// round and final checks; otherwise [`VerifyError::Opening`] for the first opening the base
/// PCS rejects.
pub fn verify_claims<const K: usize, P: BasePcs<K>, H: Challenger<K>>(
    ctx: &Context<K>,
    pcs: &P,
    public: &PublicInput<'_>,
    transcript: &SumcheckTranscript<K>,
    ch: &mut H,
) -> core::result::Result<FinalClaims<K>, VerifyError> {
    verify_checks(ctx, pcs, public, transcript, ch)?.verdict()
}
