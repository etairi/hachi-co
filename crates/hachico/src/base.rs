//! Hachi as the base PCS of Hachi-Co: [`HachiBase`], the adapter between [`hachi::HachiPcs`] and
//! the [`BasePcs`] trait through which each prover commits to its share polynomial
//! $\langle \widetilde w \rangle\_p$ and its blinders $g\_{0,p}, g\_{1,p}$ and opens them at the
//! sumcheck's point (`5_Building_Blocks.tex`:238-273, 346-365).
//!
//! **Setup.** The draft samples $\mathsf{pp}\_{\mathsf{PCS}} \gets \mathsf{PCS}.\mathsf{Setup}
//! (1^\lambda, 1^{\ell''}, 1^{d\_0})$ as part of $\mathsf{Setup}$
//! (`6_Collaborative_Hachi.tex`:294-297). [`HachiBase::setup`] derives the seed of Hachi's public
//! matrices from Hachi-Co's setup seed ([`base_pcs_seed`]), so that one seed fixes the whole of
//! $\mathsf{pp}$. Hachi expands its matrices from that seed, per iteration, to the widths an
//! opening needs, so it needs neither $\ell''$ nor $d\_0$ in advance. Its other parameters are
//! its own (the authors' answer 3 of 2026-10-21): only $q$ and $k$ are Hachi-Co's.
//!
//! **Parameters.** [`HachiBase::setup`] uses [`HachiChoices::toy_hiding`] at Hachi-Co's ring
//! degree when the share polynomial is over the field ([`PadRule::Field`], the draft's pad): the
//! toy choices with masks. With [`PadRule::Residues`] it uses [`HachiChoices::toy`], since that
//! pad opens through the $\mathbb Z_q$ shortcut, which is not hiding. They are **insecure** (test
//! parameters, like the toy presets of Hachi-Co). Each opening runs the number of iterations whose
//! proof is smallest (`hachi::pcs`, "Schedule"), and with masks at least one: at the toy presets,
//! two or three for the share polynomial, and for the blinders one with masks and none without,
//! as their unmasked proofs are smallest in the clear.
//! [`HachiBase::new`] takes any choices, for example [`HachiChoices::fixed_iterations`] to fix the
//! number of iterations.
//!
//! **Hiding.** With masks, the share polynomial and the blinders are each committed and opened as
//! Appendix A.2 of the draft specifies (`hachi::pcs`, "Hiding"): three commitments and three
//! openings per prover, as the draft's protocol has them. [`HachiBase::merged`] selects instead
//! Hachi's merged commitment (`hachi::pcs`, "Merged commitments"): one commitment and one opening
//! per prover, with the same three values. That option is not in the draft, and its digest differs
//! from the default's.
//!
//! [`HachiBase::new`] with `hiding: None` gives the published scheme. The provers draw the seeds of
//! the randomness from their [`PartyRandomness`](hachi_co_core::PartyRandomness) (`base_pcs`).
//!
//! **Tail**, optional ([`HachiBase::with_tail`], `hachi::tail`): an option that is not in the
//! draft, approved on 2026-11-17. Hachi runs with the tail in batched mode, so each of a prover's
//! three openings ends with the tail's claim on its last table instead of the table in the clear,
//! and the party-level step ([`BasePcs::finish_openings`]) proves the three claims, in the order
//! $\langle \widetilde w \rangle\_p$, $g\_{0,p}$, $g\_{1,p}$, with one tail proof. That proof is
//! carried in the final step of the share polynomial's opening, and the blinders' openings carry
//! none. Its context is `hachi::tail::batch_context` of the three openings' contexts, each the
//! seed that the opening's transcript gives after its last iteration, so that the tail proof is
//! bound to the three transcripts, and through them to the sumcheck's. The verifier
//! ([`BasePcs::verify_openings`]) checks the three openings up to their claims and then the one
//! tail proof, and rejects all three if any of them fails. With a tail every opening runs the
//! fewest iterations whose last table fits the tail (`hachi::pcs`, "Tail"). Merged commitments do
//! not take a tail, and the digest of an instance with a tail differs from the one without.
//!
//! **Encodings.** Commitments and proofs travel as the words of
//! [`HachiPcs::commitment_to_words`] and [`HachiPcs::proof_to_words`]: packed, with
//! $\lceil \log\_2 q \rceil$ bits per residue and $\lceil \log\_2 b \rceil$ bits per digit of a
//! final witness. Their lengths depend only on the length of the committed polynomial or vector
//! (through Hachi's layout and schedule), as [`BasePcs`] requires for the provers' all-gathers.
//! The decoders check every length and value against the parameters and the header; Hachi's
//! verifier checks the counts of the header (iterations, rounds) against the schedule.

use crate::error::Error;
use hachi::pcs::ExtSource;
use hachi::tail::{Deferred, DeferredClaim, batch_context};
use hachi::{Commitment, HachiChoices, HachiPcs, Hiding, Proof, Schedule, State, Tail, TailMode};
use hachi_co_core::base_pcs::{
    LiftedCoeffs, OpeningKind, OpeningWords, PartedCoeffs, check_opening_words,
};
use hachi_co_core::{BasePcs, Context, Layout};
use hachi_co_math::Ext;
use hachi_co_params::{BasePcsWords, PadRule};
use hachi_co_sampling::ShakePrg;
use hachi_co_transcript::Transcript;
use rand::Rng;
use std::sync::Arc;

/// The PRG domain of the seed of Hachi's public matrices ([`base_pcs_seed`]), next to the
/// domains of $\mathbf A$, $\mathbf B$, $\mathbf D$ in [`Context::setup`].
pub const SETUP_DOMAIN_BASE_PCS: &[u8] = b"hachi-co/setup/base-pcs";

/// The transcript domain of [`HachiBase`]'s parameter digest ([`BasePcs::params_digest`]). The
/// digest itself absorbs [`hachi::VERSION`], so it tracks Hachi's encoding version; the `v2` in
/// the domain string is only a fixed label and does not track that version.
pub const HACHI_BASE_DIGEST_DOMAIN: &[u8] = b"hachi-co/base-pcs/hachi/v2";

/// The seed of Hachi's public matrices for Hachi-Co's setup seed `seed`: the first 32 bytes of
/// the PRG stream of (`seed`, [`SETUP_DOMAIN_BASE_PCS`], 0).
pub fn base_pcs_seed(seed: &[u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    ShakePrg::new(seed, SETUP_DOMAIN_BASE_PCS, 0).fill_bytes(&mut out);
    out
}

/// Hachi ([`HachiPcs`]) as a [`BasePcs`] of Hachi-Co, with fixed parameters at Hachi-Co's $q$
/// and $k = K$. See the module documentation.
#[derive(Clone, Debug)]
pub struct HachiBase<const K: usize> {
    pcs: HachiPcs<K>,
    digest: [u8; 32],
    merged: bool,
}

impl<const K: usize> HachiBase<K> {
    /// Hachi over $\mathbb F\_{q^K}$ with the given choices.
    ///
    /// # Errors
    /// [`Error::Hachi`] for choices Hachi cannot use at $q$ and $K$ (for example a ring degree
    /// that $2K$ does not divide, or $q \not\equiv 5 \pmod 8$).
    pub fn new(q: u64, choices: HachiChoices) -> Result<Self, Error> {
        let pcs = HachiPcs::new(q, choices)?;
        let digest = digest(q, K, &pcs.ctx().choices, false, None);
        Ok(Self {
            pcs,
            digest,
            merged: false,
        })
    }

    /// This base PCS with merged commitments ([`BasePcs::merges`]): each prover commits to its
    /// three polynomials in one commitment and opens them with one proof (`hachi::pcs`, "Merged
    /// commitments"). Not the draft's protocol: an option to compare against it.
    ///
    /// # Errors
    /// [`Error::Hachi`] without masks: Hachi merges only with [`HachiChoices::hiding`]. Also with
    /// a tail ([`HachiBase::with_tail`]), which merged commitments do not take.
    pub fn merged(self) -> Result<Self, Error> {
        let c = &self.pcs.ctx().choices;
        if c.hiding.is_none() {
            return Err(
                hachi::Error::Parameter("merged commitments need hiding choices".into()).into(),
            );
        }
        if self.pcs.tail().is_some() {
            return Err(
                hachi::Error::Parameter("merged commitments do not take a tail".into()).into(),
            );
        }
        let digest = digest(self.pcs.ctx().q, K, c, true, None);
        Ok(Self {
            digest,
            merged: true,
            ..self
        })
    }

    /// This base PCS with the tail `tail` (module documentation, "Tail"): Hachi in batched mode,
    /// and one tail proof per prover for the claims of its three openings. Not the draft's
    /// protocol: an option, approved on 2026-11-17.
    ///
    /// # Errors
    /// [`Error::Hachi`] for merged commitments, which do not take a tail, or a tail that Hachi
    /// refuses ([`HachiPcs::with_tail`]).
    pub fn with_tail(self, tail: Arc<dyn Tail<K>>) -> Result<Self, Error> {
        if self.merged {
            return Err(
                hachi::Error::Parameter("merged commitments do not take a tail".into()).into(),
            );
        }
        let named = (tail.max_vars(), tail.params_digest());
        let pcs = self.pcs.with_tail(tail, TailMode::Batched)?;
        let digest = digest(pcs.ctx().q, K, &pcs.ctx().choices, false, Some(named));
        Ok(Self {
            pcs,
            digest,
            merged: false,
        })
    }

    /// The tail, if this base PCS has one ([`HachiBase::with_tail`]).
    pub fn tail(&self) -> Option<&Arc<dyn Tail<K>>> {
        self.pcs.tail().map(|(tail, _)| tail)
    }

    /// The schedules of a prover's three openings at the layout `layout`, in the order
    /// $\langle \widetilde w \rangle\_p$, $g\_{0,p}$, $g\_{1,p}$: the share polynomial's in
    /// $\ell''$ variables and the blinders' of $1 + \ell'' d\_b$ coefficients. With a tail they
    /// follow the tail's rule (`hachi::pcs`, "Tail").
    ///
    /// # Errors
    /// As [`HachiBase::words`].
    pub fn schedules(&self, layout: &Layout) -> Result<[Schedule; 3], Error> {
        let pcs = &self.pcs;
        let w = match layout.pad_rule {
            PadRule::Field => pcs.multilinear_ext_schedule(layout.ell_pp)?,
            PadRule::Residues => pcs.multilinear_schedule(layout.ell_pp)?,
        };
        Ok([
            w,
            pcs.vector_schedule(blinder_len(layout, layout.d0)?)?,
            pcs.vector_schedule(blinder_len(layout, layout.d1)?)?,
        ])
    }

    /// The sizes $L$ of the last tables of a prover's three openings at the layout `layout`, in
    /// the order of [`HachiBase::schedules`]: with a tail, the sizes of the claims of its proof.
    ///
    /// # Errors
    /// As [`HachiBase::schedules`], and [`Error::Hachi`] for an opening without iterations.
    pub fn last_tables(&self, layout: &Layout) -> Result<[usize; 3], Error> {
        let [w, g0, g1] = self.schedules(layout)?;
        let last = |s: Schedule| -> Result<usize, Error> {
            s.iterations.last().map(|it| it.table_vars).ok_or_else(|| {
                hachi::Error::Parameter("an opening without iterations has no table".into()).into()
            })
        };
        Ok([last(w)?, last(g0)?, last(g1)?])
    }

    /// The base PCS of $\mathsf{Setup}$ (`6_Collaborative_Hachi.tex`:294-297) for the context
    /// `ctx` made from the setup seed `seed`: [`HachiBase::new`] at Hachi-Co's $q$ with
    /// [`HachiBase::default_choices`].
    ///
    /// # Errors
    /// As [`HachiBase::new`].
    pub fn setup(ctx: &Context<K>, seed: &[u8; 32]) -> Result<Self, Error> {
        Self::new(ctx.q(), Self::default_choices(ctx, seed))
    }

    /// The choices of [`HachiBase::setup`] at Hachi-Co's ring degree $d$: [`HachiChoices::toy_hiding`]
    /// for [`PadRule::Field`], and [`HachiChoices::toy`] for [`PadRule::Residues`], whose shortcut
    /// cannot hide. The seed of the public matrices comes from the setup seed ([`base_pcs_seed`]).
    /// **Insecure** (see the module documentation).
    pub fn default_choices(ctx: &Context<K>, seed: &[u8; 32]) -> HachiChoices {
        let base = match ctx.layout.pad_rule {
            PadRule::Field => HachiChoices::toy_hiding(ctx.layout.d),
            PadRule::Residues => HachiChoices::toy(ctx.layout.d),
        };
        HachiChoices {
            seed: base_pcs_seed(seed),
            ..base
        }
    }

    /// The choices of the preset `name` from [`hachi_preset`](hachi_co_params::presets::hachi_preset),
    /// with the masks of its entry if it has them and the pad is [`PadRule::Field`], or
    /// [`HachiBase::default_choices`] for a preset without an entry. The seed of the public matrices
    /// is derived from the setup seed ([`base_pcs_seed`]).
    ///
    /// # Errors
    /// As [`hachi_preset`](hachi_co_params::presets::hachi_preset).
    pub fn preset_choices(
        ctx: &Context<K>,
        name: &str,
        seed: &[u8; 32],
    ) -> Result<HachiChoices, hachi_co_params::Error> {
        let Some(entry) = hachi_co_params::presets::hachi_preset(name)? else {
            return Ok(Self::default_choices(ctx, seed));
        };
        let hiding = match ctx.layout.pad_rule {
            PadRule::Field => entry.hiding.map(|h| Hiding {
                nu: h.nu,
                n_a: h.n_a,
                lambda: h.lambda,
                phi: h.phi,
            }),
            PadRule::Residues => None,
        };
        Ok(HachiChoices {
            b: entry.b,
            omega: entry.omega,
            n_a: entry.n_a,
            n_b: entry.n_b,
            n_d: entry.n_d,
            seed: base_pcs_seed(seed),
            hiding,
            ..HachiChoices::toy(ctx.layout.d)
        })
    }

    /// The scheme.
    pub fn pcs(&self) -> &HachiPcs<K> {
        &self.pcs
    }

    /// The choices of the scheme.
    pub fn choices(&self) -> &HachiChoices {
        &self.pcs.ctx().choices
    }

    /// The words of one prover's base-PCS items at the layout `layout`, for
    /// [`hachi_co_params::sizes_with_base`] and [`encoded_len`](crate::encoded_len): three
    /// commitments of [`HachiPcs::commitment_words`], the multilinear opening of
    /// $\langle \widetilde w \rangle\_p$ in $\ell''$ variables and the inner-product openings of
    /// the blinders' $1 + \ell'' d\_b$ coefficients ([`HachiPcs::multilinear_ext_proof_words`],
    /// [`HachiPcs::vector_proof_words`]). With a tail (module documentation, "Tail") the openings
    /// are those of batched mode, and the share polynomial's carries the party's tail proof,
    /// [`Tail::proof_words`] at the sizes of [`HachiBase::last_tables`].
    ///
    /// # Errors
    /// [`Error::Hachi`] if Hachi cannot open polynomials of these sizes (fewer than $\log\_2 d$
    /// variables), or for [`PadRule::Residues`] with hiding, whose $\mathbb Z_q$ shortcut Hachi
    /// refuses.
    pub fn words(&self, layout: &Layout) -> Result<BasePcsWords, Error> {
        let pcs = &self.pcs;
        if layout.pad_rule == PadRule::Residues && pcs.ctx().hiding().is_some() {
            return Err(hachi::Error::Parameter(
                "the residue pad opens through the Z_q shortcut, which is not hiding".into(),
            )
            .into());
        }
        let commitment = pcs.commitment_words() as u128;
        if self.merged {
            // One commitment and one proof per prover, and empty items for the blinders.
            let lens = [
                blinder_len(layout, layout.d0)?,
                blinder_len(layout, layout.d1)?,
            ];
            return Ok(BasePcsWords {
                commitment_w: commitment,
                proof_w: pcs.merged_proof_words(layout.ell_pp, lens)? as u128,
                ..BasePcsWords::default()
            });
        }
        let vector = |degree: usize| -> Result<u128, Error> {
            Ok(pcs.vector_proof_words(blinder_len(layout, degree)?)? as u128)
        };
        let carried = match self.tail() {
            Some(tail) => tail.proof_words(&self.last_tables(layout)?) as u128,
            None => 0,
        };
        Ok(BasePcsWords {
            commitment_w: commitment,
            commitment_g0: commitment,
            commitment_g1: commitment,
            proof_w: match layout.pad_rule {
                PadRule::Field => pcs.multilinear_ext_proof_words(layout.ell_pp)?,
                PadRule::Residues => pcs.multilinear_proof_words(layout.ell_pp)?,
            } as u128
                + carried,
            proof_g0: vector(layout.d0)?,
            proof_g1: vector(layout.d1)?,
        })
    }

    /// The claim that a batched opening leaves, from the words of its commitment and proof: the
    /// deferred verifier of its kind (`hachi::pcs`, "Tail").
    ///
    /// # Errors
    /// The reason of the rejection, as [`check_opening_words`] gives it.
    fn deferred_claim(
        &self,
        opening: &OpeningWords<'_, K>,
    ) -> core::result::Result<DeferredClaim<K>, String> {
        let com = self
            .pcs
            .commitment_from_words(opening.commitment)
            .map_err(|e| format!("the commitment does not decode: Hachi: {e}"))?;
        let proof = self
            .pcs
            .proof_from_words(opening.proof)
            .map_err(|e| format!("the proof does not decode: Hachi: {e}"))?;
        let (value, context, pcs) = (opening.value, opening.context, &self.pcs);
        match opening.kind {
            OpeningKind::Multilinear(point) => {
                pcs.verify_multilinear_deferred(&com, point.len(), point, value, &proof, context)
            }
            OpeningKind::MultilinearExt(point) => pcs.verify_multilinear_ext_deferred(
                &com,
                point.len(),
                point,
                value,
                &proof,
                context,
            ),
            OpeningKind::InnerProduct(weights) => pcs.verify_inner_product_deferred(
                &com,
                weights.len(),
                weights,
                value,
                &proof,
                context,
            ),
        }
        .map_err(|e| format!("Hachi: {e}"))
    }
}

/// The number of coefficients of a blinder of degree `degree` at the layout `layout`: $1 + \ell''
/// d\_b$.
///
/// # Errors
/// [`Error::Hachi`] on overflow.
fn blinder_len(layout: &Layout, degree: usize) -> Result<usize, Error> {
    Ok(layout
        .ell_pp
        .checked_mul(degree)
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| hachi::Error::Parameter("the blinder length overflows usize".into()))?)
}

/// The parameter digest: the challenge seed of a transcript with the domain
/// [`HACHI_BASE_DIGEST_DOMAIN`] that absorbs $q$, $K$, every choice, Hachi's version
/// ([`hachi::VERSION`]) and the seed of the matrices, then, only with masks
/// ([`HachiChoices::hiding`]), the masks' four words ([`Hiding::words`]), only for merged
/// commitments ([`HachiBase::merged`]) the word 1 under `merged`, and only with a tail
/// ([`HachiBase::with_tail`]) the word 1 under `tail`, its largest table and its digest
/// (`tail`): the digests of the other choices are unchanged.
fn digest(
    q: u64,
    k: usize,
    c: &HachiChoices,
    merged: bool,
    tail: Option<(usize, [u8; 32])>,
) -> [u8; 32] {
    let mut t = Transcript::new(HACHI_BASE_DIGEST_DOMAIN);
    let words: [(&[u8], u64); 11] = [
        (b"q", q),
        (b"k", k as u64),
        (b"d", c.d as u64),
        (b"b", c.b),
        (b"omega", c.omega),
        (b"n_A", c.n_a as u64),
        (b"n_B", c.n_b as u64),
        (b"n_D", c.n_d as u64),
        (b"version", hachi::VERSION),
        (b"max_iterations", c.max_iterations as u64),
        (
            b"fixed_iterations",
            c.fixed_iterations.map_or(u64::MAX, |n| n as u64),
        ),
    ];
    for (label, value) in words {
        t.append_u64(label, value);
    }
    t.append_message(b"seed", &c.seed);
    if let Some(h) = c.hiding {
        let [nu, n_a, lambda, phi] = h.words();
        t.append_u64(b"nu", nu);
        t.append_u64(b"n_A-0", n_a);
        t.append_u64(b"lambda", lambda);
        t.append_u64(b"phi", phi);
    }
    if merged {
        t.append_u64(b"merged", 1);
    }
    if let Some((max_vars, params)) = tail {
        t.append_u64(b"tail", 1);
        t.append_u64(b"tail/max_vars", max_vars as u64);
        t.append_message(b"tail/params", &params);
    }
    t.challenge_seed(b"digest")
}

/// A refusal of Hachi as an error of the base PCS.
fn base_error(e: hachi::Error) -> hachi_co_core::Error {
    hachi_co_core::Error::BasePcs(format!("Hachi: {e}"))
}

/// The number of variables and the fingerprint of `coeffs`, for their [`ExtSource`].
///
/// # Errors
/// [`hachi_co_core::Error::Shape`] unless there are $2^\ell$ coefficients.
fn lifted_shape<const K: usize>(coeffs: &LiftedCoeffs<K>) -> hachi_co_core::Result<(usize, u64)> {
    if !coeffs.len().is_power_of_two() {
        return Err(hachi_co_core::Error::Shape {
            what: "lifted coefficients (2^l)",
            expected: coeffs.len().next_power_of_two(),
            actual: coeffs.len(),
        });
    }
    Ok((coeffs.len().trailing_zeros() as usize, coeffs.fingerprint()))
}

impl<const K: usize> BasePcs<K> for HachiBase<K> {
    type Commitment = Commitment;
    type State = State<K>;
    type Proof = Proof<K>;

    /// [`HachiPcs::commit_multilinear`]: $2^\ell$ coefficients with $2^\ell \ge d$.
    fn commit_multilinear(
        &self,
        coeffs: &[u64],
        seed: &[u8; 32],
    ) -> hachi_co_core::Result<(Commitment, State<K>)> {
        self.pcs
            .commit_multilinear(coeffs, seed)
            .map_err(base_error)
    }

    /// [`HachiPcs::commit_vector`]: at least one element.
    fn commit_vector(
        &self,
        v: &[Ext<K>],
        seed: &[u8; 32],
    ) -> hachi_co_core::Result<(Commitment, State<K>)> {
        self.pcs.commit_vector(v, seed).map_err(base_error)
    }

    /// [`HachiPcs::prove_multilinear`].
    fn prove_multilinear(
        &self,
        st: &State<K>,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> hachi_co_core::Result<(Ext<K>, Proof<K>)> {
        self.pcs
            .prove_multilinear(st, point, context, seed)
            .map_err(base_error)
    }

    /// [`HachiPcs::commit_multilinear_ext`]: $2^\ell$ coefficients with $2^\ell \ge d/K$.
    fn commit_multilinear_ext(
        &self,
        coeffs: &[Ext<K>],
        seed: &[u8; 32],
    ) -> hachi_co_core::Result<(Commitment, State<K>)> {
        self.pcs
            .commit_multilinear_ext(coeffs, seed)
            .map_err(base_error)
    }

    /// [`HachiPcs::commit_multilinear_ext_owned`], which frees the coefficients as it packs them.
    fn commit_multilinear_ext_owned(
        &self,
        coeffs: Vec<Ext<K>>,
        seed: &[u8; 32],
    ) -> hachi_co_core::Result<(Commitment, State<K>)> {
        self.pcs
            .commit_multilinear_ext_owned(coeffs, seed)
            .map_err(base_error)
    }

    /// [`HachiPcs::commit_multilinear_ext_source`] on the coefficients read one run at a time from
    /// `coeffs`: neither they nor their packed message are held over the field.
    fn commit_multilinear_ext_lifted(
        &self,
        coeffs: &LiftedCoeffs<K>,
        seed: &[u8; 32],
    ) -> hachi_co_core::Result<(Commitment, State<K>)> {
        let (vars, fingerprint) = lifted_shape(coeffs)?;
        let source = ExtSource {
            vars,
            read: std::sync::Arc::new(|start: usize, out: &mut [Ext<K>]| {
                coeffs.coeffs_into(start, out);
            }),
            fingerprint,
            parts: None,
        };
        self.pcs
            .commit_multilinear_ext_source(&source, seed)
            .map_err(base_error)
    }

    /// [`HachiPcs::commit_multilinear_ext_source`] on the coefficients read one part at a time
    /// ([`hachi::pcs::SourceParts`]): neither they nor their packed message are held whole, only
    /// the part `coeffs` forms.
    fn commit_multilinear_ext_parts(
        &self,
        coeffs: &PartedCoeffs<'_, K>,
        seed: &[u8; 32],
    ) -> hachi_co_core::Result<(Commitment, State<K>)> {
        let source = ExtSource {
            vars: coeffs.vars,
            read: std::sync::Arc::new(|start: usize, out: &mut [Ext<K>]| (coeffs.read)(start, out)),
            fingerprint: 0,
            parts: Some(hachi::pcs::SourceParts {
                count: coeffs.parts,
                prepare: std::sync::Arc::new(|p: usize| {
                    (coeffs.prepare)(p)
                        .map_err(|e| hachi::Error::Protocol(format!("forming part {p}: {e}")))
                }),
                fingerprint: std::sync::Arc::new(|| (coeffs.fingerprint)()),
            }),
        };
        self.pcs
            .commit_multilinear_ext_source(&source, seed)
            .map_err(base_error)
    }

    /// [`HachiPcs::prove_multilinear_ext_source`] with the coefficients read one part at a time
    /// ([`hachi::pcs::SourceParts`]), each pass over the message preparing the parts in order.
    fn prove_multilinear_ext_parts(
        &self,
        st: State<K>,
        coeffs: &PartedCoeffs<'_, K>,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> hachi_co_core::Result<(Ext<K>, Proof<K>)> {
        let source = ExtSource {
            vars: coeffs.vars,
            read: std::sync::Arc::new(|start: usize, out: &mut [Ext<K>]| (coeffs.read)(start, out)),
            fingerprint: (coeffs.fingerprint)(),
            parts: Some(hachi::pcs::SourceParts {
                count: coeffs.parts,
                prepare: std::sync::Arc::new(|p: usize| {
                    (coeffs.prepare)(p)
                        .map_err(|e| hachi::Error::Protocol(format!("forming part {p}: {e}")))
                }),
                fingerprint: std::sync::Arc::new(|| (coeffs.fingerprint)()),
            }),
        };
        self.pcs
            .prove_multilinear_ext_source(st, source, point, context, seed)
            .map_err(base_error)
    }

    /// [`HachiPcs::prove_multilinear_ext_source`] with the coefficients read from `coeffs`.
    fn prove_multilinear_ext_lifted(
        &self,
        st: State<K>,
        coeffs: LiftedCoeffs<K>,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> hachi_co_core::Result<(Ext<K>, Proof<K>)> {
        // The source owns the coefficients, so that the opening frees them after its first
        // iteration has read them.
        let (vars, fingerprint) = lifted_shape(&coeffs)?;
        let source = ExtSource {
            vars,
            read: std::sync::Arc::new(move |start: usize, out: &mut [Ext<K>]| {
                coeffs.coeffs_into(start, out);
            }),
            fingerprint,
            parts: None,
        };
        self.pcs
            .prove_multilinear_ext_source(st, source, point, context, seed)
            .map_err(base_error)
    }

    /// [`HachiPcs::shed_message`]: the state keeps its masking rows and a fingerprint of the
    /// packed coefficients.
    fn shed(&self, st: &mut State<K>) -> bool {
        self.pcs.shed_message(st)
    }

    /// [`HachiPcs::prove_multilinear_ext_rebuilt`] with the coefficients from `coeffs` if the
    /// state was shed.
    fn prove_multilinear_ext_rebuilt(
        &self,
        st: State<K>,
        coeffs: impl FnOnce() -> hachi_co_core::Result<Vec<Ext<K>>>,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> hachi_co_core::Result<(Ext<K>, Proof<K>)> {
        let coeffs = if st.is_shed() { coeffs()? } else { Vec::new() };
        self.pcs
            .prove_multilinear_ext_rebuilt(st, coeffs, point, context, seed)
            .map_err(base_error)
    }

    /// [`HachiPcs::prove_multilinear_ext`].
    fn prove_multilinear_ext(
        &self,
        st: &State<K>,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> hachi_co_core::Result<(Ext<K>, Proof<K>)> {
        self.pcs
            .prove_multilinear_ext(st, point, context, seed)
            .map_err(base_error)
    }

    /// [`HachiPcs::prove_multilinear_ext_owned`], which frees the committed message once the first
    /// iteration has read it.
    fn prove_multilinear_ext_owned(
        &self,
        st: State<K>,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> hachi_co_core::Result<(Ext<K>, Proof<K>)> {
        self.pcs
            .prove_multilinear_ext_owned(st, point, context, seed)
            .map_err(base_error)
    }

    /// [`HachiPcs::verify_multilinear_ext`]; the reason is Hachi's [`hachi::VerifyError`].
    fn verify_multilinear_ext(
        &self,
        com: &Commitment,
        num_vars: usize,
        point: &[Ext<K>],
        value: &Ext<K>,
        proof: &Proof<K>,
        context: &[u8; 32],
    ) -> core::result::Result<(), String> {
        self.pcs
            .verify_multilinear_ext(com, num_vars, point, value, proof, context)
            .map_err(|e| e.to_string())
    }

    /// [`HachiPcs::prove_inner_product`].
    fn prove_inner_product(
        &self,
        st: &State<K>,
        weights: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> hachi_co_core::Result<(Ext<K>, Proof<K>)> {
        self.pcs
            .prove_inner_product(st, weights, context, seed)
            .map_err(base_error)
    }

    /// [`HachiPcs::verify_multilinear`]; the reason is Hachi's [`hachi::VerifyError`].
    fn verify_multilinear(
        &self,
        com: &Commitment,
        num_vars: usize,
        point: &[Ext<K>],
        value: &Ext<K>,
        proof: &Proof<K>,
        context: &[u8; 32],
    ) -> Result<(), String> {
        self.pcs
            .verify_multilinear(com, num_vars, point, value, proof, context)
            .map_err(|e| format!("Hachi: {e}"))
    }

    /// [`HachiPcs::verify_inner_product`]; the reason is Hachi's [`hachi::VerifyError`].
    fn verify_inner_product(
        &self,
        com: &Commitment,
        len: usize,
        weights: &[Ext<K>],
        value: &Ext<K>,
        proof: &Proof<K>,
        context: &[u8; 32],
    ) -> Result<(), String> {
        self.pcs
            .verify_inner_product(com, len, weights, value, proof, context)
            .map_err(|e| format!("Hachi: {e}"))
    }

    fn merges(&self) -> bool {
        self.merged
    }

    /// [`HachiPcs::commit_merged`].
    fn commit_merged(
        &self,
        coeffs: &[Ext<K>],
        g0: &[Ext<K>],
        g1: &[Ext<K>],
        seed: &[u8; 32],
    ) -> hachi_co_core::Result<(Commitment, State<K>)> {
        if !self.merged {
            return Err(hachi_co_core::Error::BasePcs(
                "Hachi: this instance does not merge commitments".into(),
            ));
        }
        self.pcs
            .commit_merged(coeffs, [g0, g1], seed)
            .map_err(base_error)
    }

    /// [`HachiPcs::prove_merged`].
    fn prove_merged(
        &self,
        st: &State<K>,
        point: &[Ext<K>],
        weights0: &[Ext<K>],
        weights1: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> hachi_co_core::Result<([Ext<K>; 3], Proof<K>)> {
        self.pcs
            .prove_merged(st, point, [weights0, weights1], context, seed)
            .map_err(base_error)
    }

    /// [`HachiPcs::verify_merged`], with Hachi's [`hachi::VerifyError`] as the reason.
    fn verify_merged(
        &self,
        com: &Commitment,
        num_vars: usize,
        point: &[Ext<K>],
        weights0: &[Ext<K>],
        weights1: &[Ext<K>],
        values: &[Ext<K>; 3],
        proof: &Proof<K>,
        context: &[u8; 32],
    ) -> core::result::Result<(), String> {
        if !self.merged {
            return Err("Hachi: this instance does not merge commitments".into());
        }
        self.pcs
            .verify_merged(
                com,
                num_vars,
                point,
                [weights0, weights1],
                values,
                proof,
                context,
            )
            .map_err(|e| format!("Hachi: {e}"))
    }

    /// Without a tail, each proof encoded on its own. With a tail (module documentation,
    /// "Tail"), the three claims that the batched openings left, in the order
    /// $\langle \widetilde w \rangle\_p$, $g\_{0,p}$, $g\_{1,p}$, proved with one tail proof under
    /// `hachi::tail::batch_context` of their contexts, which the share polynomial's opening then
    /// carries.
    fn finish_openings(&self, proofs: [Proof<K>; 3]) -> hachi_co_core::Result<[Vec<u64>; 3]> {
        let Some(tail) = self.tail() else {
            return Ok(proofs.map(|p| self.pcs.proof_to_words(&p)));
        };
        let mut proofs = proofs;
        let mut claims: Vec<Deferred<K>> = Vec::with_capacity(3);
        for p in &mut proofs {
            let claim = p.tail.as_mut().and_then(|ending| ending.deferred.take());
            claims.push(claim.ok_or_else(|| {
                hachi_co_core::Error::BasePcs(
                    "Hachi: an opening with a tail leaves its claim to the party".into(),
                )
            })?);
        }
        let contexts: Vec<[u8; 32]> = claims.iter().map(|c| c.context).collect();
        let vars: Vec<usize> = claims.iter().map(|c| c.vars).collect();
        let prover: Vec<_> = claims.iter().map(Deferred::prover_claim).collect();
        let words = tail
            .prove(&prover, &batch_context(&contexts))
            .map_err(base_error)?;
        if words.len() != tail.proof_words(&vars) {
            return Err(hachi_co_core::Error::BasePcs(format!(
                "Hachi: the tail's proof has {} words, {} expected",
                words.len(),
                tail.proof_words(&vars)
            )));
        }
        drop(prover);
        drop(claims);
        if let Some(ending) = proofs[0].tail.as_mut() {
            ending.proof = words;
        }
        Ok(proofs.map(|p| self.pcs.proof_to_words(&p)))
    }

    /// Without a tail, each opening checked on its own. With a tail (module documentation,
    /// "Tail"), each opening up to its claim (the deferred verifiers of `hachi::pcs`), then the
    /// one tail proof that the share polynomial's opening carries on the three claims, the
    /// blinders' openings carrying none. All three are rejected if one of them is, as the tail
    /// proof covers their claims together.
    fn verify_openings(
        &self,
        openings: [OpeningWords<'_, K>; 3],
    ) -> [core::result::Result<(), String>; 3] {
        let Some(tail) = self.tail() else {
            return openings.map(|opening| check_opening_words(self, &opening));
        };
        let claims = openings.map(|opening| self.deferred_claim(&opening));
        let names = ["the share polynomial", "g0", "g1"];
        if let Some((i, Err(e))) = claims.iter().enumerate().find(|(_, c)| c.is_err()) {
            let reason = format!(
                "Hachi: the party's tail proof is not checked, as the opening of {} is rejected: {e}",
                names[i]
            );
            return core::array::from_fn(|j| match &claims[j] {
                Err(own) => Err(own.clone()),
                Ok(_) => Err(reason.clone()),
            });
        }
        let claims: Vec<DeferredClaim<K>> = claims.into_iter().filter_map(Result::ok).collect();
        let verdict = (|| {
            if !claims[1].carried.is_empty() || !claims[2].carried.is_empty() {
                return Err(
                    "Hachi: only the share polynomial's opening carries a tail proof".into(),
                );
            }
            let vars: Vec<usize> = claims.iter().map(|c| c.vars).collect();
            if claims[0].carried.len() != tail.proof_words(&vars) {
                return Err("Hachi: the party's tail proof does not have its length".into());
            }
            let contexts: Vec<[u8; 32]> = claims.iter().map(|c| c.context).collect();
            let checked: Vec<_> = claims.iter().map(DeferredClaim::claim).collect();
            tail.verify(&checked, &claims[0].carried, &batch_context(&contexts))
                .map_err(|e| format!("Hachi: the party's tail proof: {e}"))
        })();
        [verdict.clone(), verdict.clone(), verdict]
    }

    /// A transcript hash of $q$, $K$ and every choice of Hachi, the seed of its matrices
    /// included, under [`HACHI_BASE_DIGEST_DOMAIN`].
    fn params_digest(&self) -> [u8; 32] {
        self.digest
    }

    /// [`HachiPcs::commitment_to_words`].
    fn commitment_to_words(&self, com: &Commitment) -> Vec<u64> {
        self.pcs.commitment_to_words(com)
    }

    /// [`HachiPcs::commitment_from_words`].
    fn commitment_from_words(&self, words: &[u64]) -> hachi_co_core::Result<Commitment> {
        self.pcs.commitment_from_words(words).map_err(base_error)
    }

    /// [`HachiPcs::proof_to_words`].
    fn proof_to_words(&self, p: &Proof<K>) -> Vec<u64> {
        self.pcs.proof_to_words(p)
    }

    /// [`HachiPcs::proof_from_words`].
    fn proof_from_words(&self, words: &[u64]) -> hachi_co_core::Result<Proof<K>> {
        self.pcs.proof_from_words(words).map_err(base_error)
    }
}
