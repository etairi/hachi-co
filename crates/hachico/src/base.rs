//! Hachi as the base PCS of Hachi-Co: [`HachiBase`], the adapter between [`hachi::HachiPcs`] and
//! the [`BasePcs`] trait through which each prover commits to its share polynomial
//! $\langle \widetilde w \rangle\_p$ and its blinders $g\_{0,p}, g\_{1,p}$ and opens them at the
//! sumcheck's point (`5_Building_Blocks.tex`:209-241, 314-333).
//!
//! **Setup.** The draft samples $\mathsf{pp}\_{\mathsf{PCS}} \gets \mathsf{PCS}.\mathsf{Setup}
//! (1^\lambda, 1^{\ell''}, 1^{d\_0})$ as part of $\mathsf{Setup}$
//! (`6_Collaborative_Hachi.tex`:289-292). [`HachiBase::setup`] derives the seed of Hachi's public
//! matrices from Hachi-Co's setup seed ([`base_pcs_seed`]), so that one seed fixes the whole of
//! $\mathsf{pp}$. Hachi expands its matrices from that seed, per iteration, to the widths an
//! opening needs, so it needs neither $\ell''$ nor $d\_0$ in advance. Its other parameters are
//! its own (the authors' answer 3 of 2026-10-21): only $q$ and $k$ are Hachi-Co's.
//!
//! **Parameters.** [`HachiBase::setup`] uses [`HachiChoices::toy`] at Hachi-Co's ring degree,
//! which is **insecure** (test parameters, like the toy presets of Hachi-Co). Each opening runs
//! the number of iterations whose proof is smallest (`hachi::pcs`, "Schedule"): at the toy
//! presets, two or three for the share polynomial and none for the blinders, whose proofs are
//! smallest in the clear. [`HachiBase::new`] takes any choices, for example
//! [`HachiChoices::fixed_iterations`] to fix the number of iterations.
//!
//! **Not hiding.** Hachi as implemented is not hiding: the last witness of every opening is
//! sent in the clear, and with no iteration that is the committed polynomial itself. Choices
//! with [`HachiChoices::hiding`] randomize every commitment (`docs/hachi-hiding.md` §6):
//! - at an MLWE rank of real security, a commitment then hides the committed polynomial under
//!   MLWE until it is opened;
//! - the openings reveal as much as before, and Hachi-Co opens every commitment, so its proofs
//!   are not hiding yet (with no iteration, as for the blinders at the toy presets, the opening
//!   sends the blinder and its randomness).
//!
//! The default choices do not use randomized commitments yet. The provers draw the seeds of the
//! randomness from their [`PartyRandomness`](hachi_co_core::PartyRandomness) (`base_pcs`).
//!
//! **Encodings.** Commitments and proofs travel as the words of
//! [`HachiPcs::commitment_to_words`] and [`HachiPcs::proof_to_words`]: packed, with
//! $\lceil \log\_2 q \rceil$ bits per residue and $\lceil \log\_2 b \rceil$ bits per digit of a
//! final witness. Their lengths depend only on the length of the committed polynomial or vector
//! (through Hachi's layout and schedule), as [`BasePcs`] requires for the provers' all-gathers.
//! The decoders check every length and value against the parameters and the header; Hachi's
//! verifier checks the counts of the header (iterations, rounds) against the schedule.

use crate::error::Error;
use hachi::{Commitment, HachiChoices, HachiPcs, Proof, State};
use hachi_co_core::{BasePcs, Context, Layout};
use hachi_co_math::Ext;
use hachi_co_params::BasePcsWords;
use hachi_co_sampling::ShakePrg;
use hachi_co_transcript::Transcript;
use rand::Rng;

/// The PRG domain of the seed of Hachi's public matrices ([`base_pcs_seed`]), next to the
/// domains of $\mathbf A$, $\mathbf B$, $\mathbf D$ in [`Context::setup`].
pub const SETUP_DOMAIN_BASE_PCS: &[u8] = b"hachi-co/setup/base-pcs";

/// The transcript domain of [`HachiBase`]'s parameter digest ([`BasePcs::params_digest`]).
/// Version 2: Hachi's schedule minimises the proof's size and its encodings are packed
/// ([`hachi::VERSION`]).
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
}

impl<const K: usize> HachiBase<K> {
    /// Hachi over $\mathbb F\_{q^K}$ with the given choices.
    ///
    /// # Errors
    /// [`Error::Hachi`] for choices Hachi cannot use at $q$ and $K$ (for example a ring degree
    /// that $2K$ does not divide, or $q \not\equiv 5 \pmod 8$).
    pub fn new(q: u64, choices: HachiChoices) -> Result<Self, Error> {
        let pcs = HachiPcs::new(q, choices)?;
        let digest = digest(q, K, &pcs.ctx().choices);
        Ok(Self { pcs, digest })
    }

    /// The base PCS of $\mathsf{Setup}$ (`6_Collaborative_Hachi.tex`:289-292) for the context
    /// `ctx` made from the setup seed `seed`: [`HachiBase::new`] at Hachi-Co's $q$ with
    /// [`HachiBase::default_choices`].
    ///
    /// # Errors
    /// As [`HachiBase::new`].
    pub fn setup(ctx: &Context<K>, seed: &[u8; 32]) -> Result<Self, Error> {
        Self::new(ctx.q(), Self::default_choices(ctx, seed))
    }

    /// The choices of [`HachiBase::setup`]: [`HachiChoices::toy`] at Hachi-Co's ring degree $d$,
    /// with the seed of the public matrices derived from the setup seed ([`base_pcs_seed`]).
    /// **Insecure** (see the module documentation).
    pub fn default_choices(ctx: &Context<K>, seed: &[u8; 32]) -> HachiChoices {
        HachiChoices {
            seed: base_pcs_seed(seed),
            ..HachiChoices::toy(ctx.layout.d)
        }
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
    /// the blinders' $1 + \ell'' d\_b$ coefficients ([`HachiPcs::multilinear_proof_words`],
    /// [`HachiPcs::vector_proof_words`]).
    ///
    /// # Errors
    /// [`Error::Hachi`] if Hachi cannot open polynomials of these sizes (fewer than $\log\_2 d$
    /// variables).
    pub fn words(&self, layout: &Layout) -> Result<BasePcsWords, Error> {
        let pcs = &self.pcs;
        let commitment = pcs.commitment_words() as u128;
        let vector = |degree: usize| -> Result<u128, Error> {
            let len = layout
                .ell_pp
                .checked_mul(degree)
                .and_then(|n| n.checked_add(1))
                .ok_or_else(|| {
                    hachi::Error::Parameter("the blinder length overflows usize".into())
                })?;
            Ok(pcs.vector_proof_words(len)? as u128)
        };
        Ok(BasePcsWords {
            commitment_w: commitment,
            commitment_g0: commitment,
            commitment_g1: commitment,
            proof_w: pcs.multilinear_proof_words(layout.ell_pp)? as u128,
            proof_g0: vector(layout.d0)?,
            proof_g1: vector(layout.d1)?,
        })
    }
}

/// The parameter digest: the challenge seed of a transcript with the domain
/// [`HACHI_BASE_DIGEST_DOMAIN`] that absorbs $q$, $K$, every choice, Hachi's version
/// ([`hachi::VERSION`]) and the seed of the matrices, then, only for randomized commitments
/// ([`HachiChoices::hiding`]), $\nu$ and the distribution's two words
/// ([`hachi::Noise::words`]): the digest of non-hiding choices is unchanged.
fn digest(q: u64, k: usize, c: &HachiChoices) -> [u8; 32] {
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
        let [kind, param] = h.noise.words();
        t.append_u64(b"nu", h.nu as u64);
        t.append_u64(b"noise", kind);
        t.append_u64(b"noise-param", param);
    }
    t.challenge_seed(b"digest")
}

/// A refusal of Hachi as an error of the base PCS.
fn base_error(e: hachi::Error) -> hachi_co_core::Error {
    hachi_co_core::Error::BasePcs(format!("Hachi: {e}"))
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
