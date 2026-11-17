//! The polynomial commitment scheme: [`HachiPcs`].
//!
//! Three kinds of committed objects, all with evaluation points in $\mathbb F_{q^K}$:
//! - **Multilinear polynomials** $f \in \mathbb Z_q^{\le 1}\[X_1, \dots, X_\ell\]$ given by
//!   their $2^\ell$ monomial coefficients ([`HachiPcs::commit_multilinear`]); coefficient $i$
//!   multiplies $X_1^{i_1} \cdots X_\ell^{i_\ell}$ with $\mathrm{bit}(i) = (i_1, \dots, i_\ell)$
//!   most significant first. The commitment holds the packed polynomial $f' = \sum_i f_i Z_i$
//!   of the paper's Section 3.2, in ring elements by $\psi$. An opening reduces $f(\mathbf x) =
//!   y$ to a claim over $\mathcal R_q$:
//!   1. the prover sends $K - 1$ of the partial evaluations $y_i = f_i(\mathbf x')$ over the
//!      first $\kappa$ variables, $\mathbf x' = (x_{\kappa+1}, \dots, x_\ell)$, and the verifier
//!      derives the other from $y = \sum_i \mathrm{mono}\_i(x_1, \dots, x_\kappa) y_i$
//!      (Section 3.2 and footnote 5);
//!   2. a sumcheck binds the $y_i$ to $f'$ ([`crate::partials`]) and leaves the claim
//!      $f'(\mathbf r) = v_{\mathbf r}$ at the point $\mathbf r$ of its challenges. This
//!      deviates from Section 3.2, which proves $f'(\mathbf x') = \sum_i y_i Z_i$ only;
//!   3. the prover sends $Y = F(\mathbf r_{\mathrm{high}})$, checked by $E(Y) =
//!      (K/d)\,\mathrm{Tr}\_H(Y \sigma_{-1}(v)) = v_{\mathbf r}$ with $v = \psi$ of the
//!      monomials of $\mathbf r_{\mathrm{low}}$ (Section 3.1, Theorem 2).
//! - **Vectors** $\mathbf s \in \mathbb F_{q^K}^N$ ([`HachiPcs::commit_vector`]), opened as an
//!   inner product $\langle \mathbf s, \mathbf e \rangle$ with any public weights: by Theorem 2,
//!   $\langle \mathbf s, \mathbf e \rangle = (K/d)\,\mathrm{Tr}\_H(\sum_t \psi(\mathbf s_t)
//!   \sigma_{-1}(\psi(\mathbf e_t)))$ over blocks of $d/K$. The prover sends that sum $Y$; the
//!   weights are not a tensor, so the claim has a single chunk ($r = 0$).
//! - **Multilinear polynomials over the field** $f \in \mathbb F_{q^K}^{\le 1}\[X_1, \dots,
//!   X_\ell\]$ ([`HachiPcs::commit_multilinear_ext`]), with the same order of coefficients. The
//!   commitment holds $\psi$ of $d/K$ consecutive coefficients. An opening is the paper's Section
//!   3.1 directly: no partial evaluations and no batching sumcheck, only $Y$. Such a polynomial
//!   has $K$ times the residues of one over $\mathbb Z_q$ in as many variables.
//!
//! **Bit order, a deviation from `hachi-pcs.tex`.** An index here has the first variable as its
//! most significant bit, and the appendix's `nat` and `bits` put it in the least significant one.
//! So the message of a commitment is the appendix's with the bits of the chunk, element and $\psi$
//! indices reversed, and the weights $\mathbf b$, $\mathbf a$ and $v$ of an opening follow the
//! same order. Prover and verifier agree, and the scheme is the appendix's up to that relabelling
//! (`tests/fixtures/check_masked_spec.py` checks both on its shared instance).
//!
//! Then the iterations of the [`Schedule`] ([`crate::iteration`]) and the final step, which
//! sends the last witness in the clear (paper, Section 4.4): the verifier recomputes its
//! commitment and checks the claim directly.
//!
//! **Hiding.** With [`HachiChoices::hiding`], multilinear polynomials over the field and vectors
//! are committed and opened as the blue parts of `hachi-pcs.tex`, Appendix A.2, specify (after
//! Jindo Section 2.4). One opening of each kind, with its secrets and challenges, is checked
//! against the appendix's equations in Sage (`tests/fixtures/check_masked_spec.py`, on the
//! instance that the test `spec_fixture` writes):
//! - **Commit.** Every chunk gets a uniform masking row $\mathfrak m_i$, and $\mathbf t_i =
//!   \mathbf A \mathbf s_i + \mathbf A' \boldsymbol\varrho_i$ with $\boldsymbol\varrho_i$ of
//!   uniform digits. Under MLWE the $\mathbf t_i$ are pseudorandom, at an MLWE rank $\nu$ of real
//!   security (the toy choices have $\nu = 1$).
//! - **Step 1 of an opening.** With $Y = \sum_i b_i \mathbf a^\top \mathbf f_i$ and $Y_{\mathfrak
//!   m} = \sum_i b_i \mathfrak m_i$, the prover sends $y_{\mathfrak m} = E(Y_{\mathfrak m})$, then
//!   receives $x^* \in \mathbb F_{q^K} \setminus \lbrace 0 \rbrace$ and sends $Y^* = Y + x^*
//!   Y_{\mathfrak m}$. The verifier checks $E(Y^*) = y + x^* y_{\mathfrak m}$, and the first claim
//!   has the weights $(\mathbf a, x^*)$ and the value $Y^*$.
//! - **Vectors** are the vector chains of the paragraph "Sums of univariate polynomials" of
//!   Appendix A.2: one chunk ($r = 0$) with its masking row, $\mathbf b = (1)$, $v = 1$ and the
//!   weights $\sigma_{-1}(\psi(\cdot))$ of the blocks of $\mathbf e$, and step 1 as above.
//! - **The first iteration** is masked and repeated until it is accepted
//!   ([`crate::iteration`]). The later ones are unchanged, and there is at least one iteration.
//!
//! One limit: the polynomials over $\mathbb Z_q$ of [`HachiPcs::commit_multilinear`] are
//! **refused**, since their partial evaluations reveal more than the value, so that opening cannot
//! hide. Commit such a polynomial over the field instead.
//!
//! **Merged commitments** ([`HachiPcs::commit_merged`], with hiding only). An option that is not in
//! the draft, designed and tested here: one commitment to a multilinear polynomial over the field
//! and two vectors, Hachi-Co's share polynomial and its two blinders, and one opening for all
//! three, at one point and two sets of weights. The vectors fill extra rows of chunk 0
//! ([`Layout::extra`]), each followed by its own masking row, and the other chunks hold zeros
//! there. An opening sends the three values with the statement, then three masks' values, then,
//! after $x^*$, one ring value per object, each checked by its own trace check (the share's with
//! its $v$, the vectors' with $v = 1$: one trace check on a combined ring value would not give
//! the vectors' values when $v \ne 1$). Then $\lambda \ne 0$ combines the three ring values into
//! the one claim the iterations prove ([`HachiPcs::prove_merged`]). What is not enforced:
//! nothing makes a prover put zeros in the extra rows of the other chunks. One that fills them
//! binds vectors whose inner products come out as $\langle \mathbf g, \mathbf e \rangle$ plus
//! $\sum_{i \ge 2} b_i \langle \mathbf g_i, \mathbf e \rangle$, with the $b_i$ the monomials of the
//! first $r$ coordinates of the point: for Hachi-Co's blinders, polynomials one degree higher in
//! those coordinates, still fixed before the coins. No proof of soundness or hiding covers this
//! option.
//!
//! **Randomness.** The prover's methods take a secret 32-byte seed, which only hiding uses. A
//! commitment draws from the PRG streams `(seed, "hachi/commit/r", 0)` and `(seed,
//! "hachi/commit/mask", 0)`, an opening from the streams of [`crate::iteration`]. Every commitment
//! and every opening needs a fresh seed: an opening of the same state with a reused seed repeats
//! its masks. The state of a masked commitment opens once, and a second opening of it, or of a
//! clone, is refused: `Eval` in `hachi-pcs.tex`, Appendix A.2, opens a commitment once, since its
//! masking rows hide the values of step 2(a) for one evaluation only. Committing again with the
//! same seed gives the same commitment, so that is a second opening too.
//!
//! **Schedule.** Unless [`HachiChoices::fixed_iterations`] fixes it, an opening runs the number
//! of iterations, from 0 to [`HachiChoices::max_iterations`], whose proof has the fewest words
//! (the fewer iterations on a tie; [`crate::params`], "How many iterations"). The prover and the
//! verifier derive it from the size of the object and the parameters.
//!
//! **Encodings** ([`HachiPcs::commitment_to_words`], [`HachiPcs::proof_to_words`]). A header of
//! plain 64-bit words, then fields packed least significant bit first into 64-bit words, the
//! last word padded with zero bits. A residue takes $\lceil \log_2 q \rceil$ bits. After at
//! least one iteration the final witness is the next witness $\tilde w$ of the last iteration,
//! its $2^L$ digits of base $b$ in the order of the table (their range was proved by that
//! iteration's sumcheck), each sent as its index in the balanced digit set of
//! [`crate::sumcheck`] (a deviation from the appendix's set, noted there): $\lceil \log_2 b
//! \rceil$ bits. Without iterations the final witness is the input's message, sent as residues. The decoders know the parameters, so a header holds only
//! what they do not fix:
//! - commitment: `[VERSION]`, then the $n_B d$ coefficients of $\mathbf u$;
//! - proof: `[VERSION, P, B, k, L_1, …, L_k, N]` for $P$ partial evaluations and $B$ rounds of
//!   the input batching sumcheck ($K - 1$ and $\ell - \kappa$ for a multilinear opening, 0 and 0
//!   for an inner product), $k$ iterations of $L_i$ sumcheck rounds and a final witness of $N$
//!   entries: ring elements without iterations, digits after them. Then the input reduction's
//!   partial evaluations, its batching rounds (2 field elements each) and $f'(\mathbf r)$ (if $B >
//!   0$); $Y$; per iteration $\mathbf v$, $\mathbf u'$, the rounds ($b + 2$ and 3 field elements
//!   each), $y'$ and, except after the last iteration, $p'$; and the final witness. With hiding
//!   the header ends with a word $M$: bit 0
//!   is set if the proof carries $y_{\mathfrak m}$, which then comes just before $Y$, and bit 1 if
//!   it is a merged opening, whose vectors' two masks' values follow $y_{\mathfrak m}$ and whose two
//!   more ring values follow $Y$. So $M$ is 0, 1 or 3. Without hiding the encodings are those of
//!   the non-hiding scheme, unchanged.
//!
//! Every value has a single encoding: the decoders reject residues not below $q$, digit indices
//! outside the digit set, non-zero padding, and any length other than the one the header and the
//! parameters imply. The verifier checks the counts of a header: $B$ against $\ell - \kappa$, the
//! others against the schedule.

use crate::commit::{Committed, Ctx, commit, commit_source, commit_source_parts, commit_table};
use crate::error::VerifyError;
use crate::fs::Fs;
use crate::iteration::{
    self, Claim, IterationProof, MessageView, OwnedParts, OwnedSource, Reduced, Witness,
    check_exts, check_polys, commitment_randomness, evaluate_claim, table_eval,
};
use crate::params::{HachiChoices, Layout, Schedule, best_layout, max_digit};
use crate::partials::{self, RoundValues};
use crate::sumcheck::RoundMessage;
use crate::table::{DigitTable, ResidueTable};
use crate::{Error, Result};
use hachi_co_math::decompose::digit_range;
use hachi_co_math::{Ext, Poly};
use hachi_co_sampling::ShakePrg;
use hachi_co_sampling::uniform::uniform_mod_q;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// The version of the encodings ([`HachiPcs::commitment_to_words`],
/// [`HachiPcs::proof_to_words`]), also absorbed with the statement: 8 since every iteration
/// commits its next witness as Section 3.1 commits a polynomial over the field (`hachi-pcs.tex`,
/// Appendix A): $d/K$ digits per ring element, no partial evaluations and no batching sumcheck per
/// iteration, and no reduction after the last one, whose table is the final witness, sent as its
/// digits. Version 7 packed $K$ digits into each element of the field (Section 4.5), bound the
/// partial evaluations by ring-switching after every iteration, and since then the masks' width
/// uses the digit bound $\lceil b/2 \rceil$ (Setup item 2(e), [`crate::params`]), which changes
/// $\mathfrak s$, $\beta^{(0)}$ and with them $\tau^{(0)}$ and the shape of a masked proof for the same
/// choices. Version 6 masked the openings of vectors as well, with the width from $b$. Version 5
/// masked the first iteration of polynomials over the field only, with a masks' value in place of a
/// final randomness. Version 4 randomized every level,
/// and its sumcheck that binds the partial evaluations was already ring-switching, with 2 field
/// elements per round and $K - 1$ partial evaluations ([`crate::partials`]). Version 3 batched the
/// same claims on the orbit of the point, with 3 field elements per round and $K$ partial
/// evaluations. Version 2 had no such sumcheck. It packed the values and
/// chose the schedule that minimises the proof's size (version 1 had a 64-bit word per value and
/// stopped at the first iteration that did not shrink the witness).
pub const VERSION: u64 = 8;

/// A commitment $\mathbf u \in \mathcal R_q^{n_B}$.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commitment {
    /// $\mathbf u$.
    pub u: Vec<Poly>,
}

/// What the input of a committed object was.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Input<const K: usize> {
    Multilinear {
        vars: usize,
        coeffs: Vec<u64>,
    },
    MultilinearExt {
        vars: usize,
    },
    Vector {
        values: Vec<Ext<K>>,
    },
    Merged {
        vars: usize,
        values: [Vec<Ext<K>>; 2],
    },
}

/// The prover's state after committing.
///
/// The state of a masked commitment opens once (module documentation, "Randomness"). Its clones
/// share that one opening, and two states are equal when their inputs and commitments are.
#[derive(Clone, Debug)]
pub struct State<const K: usize> {
    input: Input<K>,
    committed: Committed,
    /// Set by the opening of a masked commitment, and shared by the clones of the state.
    opened: Arc<AtomicBool>,
    /// What is left of the message after [`HachiPcs::shed_message`]; the message itself is then
    /// empty until [`HachiPcs::prove_multilinear_ext_rebuilt`] packs it again.
    shed: Option<Shed>,
}

/// The part of a message that [`HachiPcs::shed_message`] keeps: each chunk's rows after its
/// packed coefficients (the masking row of a masked layout), and a fingerprint of the packed
/// coefficients (`message_fingerprint`), against which the coefficients given at the opening are
/// checked. The fingerprint guards against handing over other coefficients by mistake; it is not
/// a cryptographic binding (the commitment is).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Shed {
    tails: Vec<Vec<Poly>>,
    fingerprint: u64,
    /// Whether the state was committed from an [`ExtSource`], whose fingerprint `fingerprint` is
    /// then (the message was never formed), so that it opens by
    /// [`HachiPcs::prove_multilinear_ext_source`].
    from_source: bool,
}

/// Writes coefficients `start .. start + out.len()` of a polynomial over the field into `out`
/// ([`ExtSource`]).
pub type CoeffReader<'a, const K: usize> = dyn Fn(usize, &mut [Ext<K>]) + Send + Sync + 'a;

/// The monomial coefficients of a multilinear polynomial over the field, read in runs as they are
/// needed, for [`HachiPcs::commit_multilinear_ext_source`] and
/// [`HachiPcs::prove_multilinear_ext_source`]: the commitment and the opening of the coefficients
/// read, without them or their packed message held whole.
#[derive(Clone)]
pub struct ExtSource<'a, const K: usize> {
    /// The number of variables, $2^\ell$ coefficients.
    pub vars: usize,
    /// Writes coefficients `start .. start + out.len()` into `out`. Shared, so that an opening
    /// can own it with what it reads, and drop both once its first iteration has read them.
    pub read: Arc<CoeffReader<'a, K>>,
    /// A fingerprint of the coefficients, against which the opening checks that it was given
    /// those of the commitment (not a binding: the commitment is).
    pub fingerprint: u64,
    /// For the commitment only: the coefficients in parts, which the commitment reads one at a
    /// time ([`SourceParts`]); `None` reads them in any order.
    pub parts: Option<SourceParts<'a>>,
}

/// The coefficients of an [`ExtSource`] in `count` parts of $2^\ell / \mathrm{count}$
/// consecutive coefficients: [`HachiPcs::commit_multilinear_ext_source`] calls `prepare(p)` for
/// $p = 0, 1, \dots$ in order and reads only part $p$'s coefficients after it, so that a source
/// can form one part at a time ([`crate::commit::commit_source_parts`]). `fingerprint` is read after the
/// commitment, when every part was formed, in place of [`ExtSource::fingerprint`].
#[derive(Clone)]
pub struct SourceParts<'a> {
    /// The number of parts, a power of two dividing the chunks.
    pub count: usize,
    /// Forms part `p`.
    pub prepare: Arc<dyn Fn(usize) -> Result<()> + Send + Sync + 'a>,
    /// The fingerprint of all the coefficients, once every part was formed.
    pub fingerprint: Arc<dyn Fn() -> u64 + Send + Sync + 'a>,
}

/// FNV-1a over the residues of the first `len` ring elements of each chunk, chunk by chunk (in
/// parallel), then over the chunks' values in order.
fn message_fingerprint(message: &[Vec<Poly>], len: usize) -> u64 {
    use rayon::prelude::*;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let fnv = |h: u64, w: u64| (h ^ w).wrapping_mul(PRIME);
    message
        .par_iter()
        .map(|chunk| {
            chunk[..len.min(chunk.len())]
                .iter()
                .flat_map(|p| p.0.iter())
                .fold(0xcbf2_9ce4_8422_2325, |h, w| fnv(h, *w))
        })
        .collect::<Vec<u64>>()
        .into_iter()
        .fold(0xcbf2_9ce4_8422_2325, fnv)
}

impl<const K: usize> PartialEq for State<K> {
    fn eq(&self, other: &Self) -> bool {
        self.input == other.input && self.committed == other.committed && self.shed == other.shed
    }
}

impl<const K: usize> Eq for State<K> {}

/// A state borrowed from the caller, or owned by an opening that may free parts of it
/// ([`HachiPcs::prove_multilinear_ext_owned`]).
enum Held<'a, T> {
    Borrowed(&'a T),
    Owned(T),
}

impl<const K: usize> State<K> {
    /// The state of a new commitment, not opened.
    fn new(input: Input<K>, committed: Committed) -> Self {
        Self {
            input,
            committed,
            opened: Arc::new(AtomicBool::new(false)),
            shed: None,
        }
    }

    /// Whether [`HachiPcs::shed_message`] dropped the message: the state then opens only by
    /// [`HachiPcs::prove_multilinear_ext_rebuilt`] with the committed coefficients.
    pub fn is_shed(&self) -> bool {
        self.shed.is_some()
    }

    /// Takes the one opening of a masked commitment, after the inputs of the opening are checked
    /// and before anything is computed from the secrets. An unmasked commitment opens any number
    /// of times.
    ///
    /// # Errors
    /// [`Error::Protocol`] if the commitment is masked and this state, or a clone of it, was
    /// opened before.
    fn take_opening(&self) -> Result<()> {
        if self.committed.layout.masked && self.opened.swap(true, Ordering::SeqCst) {
            return Err(Error::Protocol(
                "a masked commitment opens once, and this state was opened".into(),
            ));
        }
        Ok(())
    }
}

/// An evaluation proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof<const K: usize> {
    /// The $K - 1$ partial evaluations of a multilinear opening that the verifier does not
    /// derive from the value ([`partials::complete`]); empty for an inner product.
    pub partials: Vec<Ext<K>>,
    /// The rounds of the sumcheck that binds the partial evaluations to $f'$
    /// ([`crate::partials`]): $\ell - \kappa$ for a multilinear opening, none for an inner
    /// product.
    pub batch_rounds: Vec<RoundValues<K>>,
    /// $f'(\mathbf r)$ at the point of those rounds; `None` for an inner product.
    pub batch_eval: Option<Ext<K>>,
    /// With a masked opening, $y_{\mathfrak m}$, and `None` otherwise.
    pub mask_eval: Option<Ext<K>>,
    /// The ring element $Y$ of the reduction to $\mathcal R_q$, and $Y^*$ in a masked opening.
    pub ring_value: Poly,
    /// The iterations.
    pub iterations: Vec<IterationProof<K>>,
    /// The final witness ([`FinalWitness`]).
    pub final_witness: FinalWitness,
    /// With a merged opening ([`HachiPcs::prove_merged`]), $y_{\mathfrak m}$ of the two vectors, in
    /// order, and empty otherwise.
    pub blinder_mask_evals: Vec<Ext<K>>,
    /// With a merged opening, $Y^*$ of the two vectors, in order, and empty otherwise.
    pub blinder_ring_values: Vec<Poly>,
}

/// What the final step of an opening sends (module documentation, "Encodings").
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinalWitness {
    /// Without iterations: the committed witness's message, chunk after chunk.
    Message(Vec<Poly>),
    /// After the iterations: the next witness $\tilde w$ of the last one, its digits on the
    /// hypercube in the clear (`hachi-pcs.tex`, step 3 of `Eval`).
    Digits(DigitTable),
}

/// What a proof carries ahead of its iterations besides $Y$, by the kind of opening; the choice
/// of the schedule counts it (module documentation, "Schedule" and "Encodings").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reduction {
    /// A multilinear opening: $K - 1$ partial evaluations, the rounds of the batching sumcheck
    /// and $f'(\mathbf r)$.
    Multilinear {
        /// The rounds, $\ell - \kappa$.
        rounds: usize,
    },
    /// An inner product, or a polynomial over $\mathbb F_{q^K}$: none of these.
    Direct,
    /// A masked polynomial over $\mathbb F_{q^K}$: $y_{\mathfrak m}$.
    Masked,
    /// A merged opening: three masks' values and two more ring values.
    Merged,
}

impl Reduction {
    /// The number of partial evaluations sent: $K - 1$ for a multilinear opening, else 0.
    fn partials(self, k: usize) -> usize {
        match self {
            Self::Multilinear { .. } => k - 1,
            Self::Direct | Self::Masked | Self::Merged => 0,
        }
    }

    /// The number of rounds of the batching sumcheck.
    fn rounds(self) -> usize {
        match self {
            Self::Multilinear { rounds } => rounds,
            Self::Direct | Self::Masked | Self::Merged => 0,
        }
    }

    /// The residues ahead of the iterations, $Y$ included: $K(K - 1) + 2K(\ell - \kappa) + K +
    /// d$ for a multilinear opening, $K + d$ for a masked one, $3K + 3d$ for a merged one, $d$
    /// otherwise, and `None` on overflow.
    fn residues(self, k: usize, d: usize) -> Option<usize> {
        match self {
            Self::Multilinear { rounds } => {
                rounds.checked_mul(2 * k)?.checked_add(k * (k - 1) + k + d)
            }
            Self::Direct => Some(d),
            Self::Masked => Some(k + d),
            Self::Merged => Some(3 * k + 3 * d),
        }
    }
}

/// Hachi with fixed parameters.
#[derive(Clone, Debug)]
pub struct HachiPcs<const K: usize> {
    ctx: Ctx<K>,
}

/// Transcript labels.
const LABEL_STATEMENT: &[u8] = b"hachi/statement";
const LABEL_COMMITMENT: &[u8] = b"hachi/commitment";
const LABEL_POINT: &[u8] = b"hachi/point";
const LABEL_VALUE: &[u8] = b"hachi/value";
const LABEL_PARTIALS: &[u8] = b"hachi/input/partials";
const LABEL_RING_VALUE: &[u8] = b"hachi/input/ring-value";
const LABEL_MASK_EVAL: &[u8] = b"hachi/input/mask-eval";
const LABEL_MASK_COIN: &[u8] = b"hachi/input/mask-coin";
const LABEL_MERGED_SHAPE: &[u8] = b"hachi/merged/lengths";
const LABEL_MERGED_WEIGHTS: &[u8] = b"hachi/merged/weights";
const LABEL_MERGED_VALUES: &[u8] = b"hachi/merged/values";
const LABEL_MERGED_COIN: &[u8] = b"hachi/merged/lambda";

/// The PRG domain of a commitment's randomness (index 0).
pub const DOMAIN_COMMIT_R: &[u8] = b"hachi/commit/r";
/// The PRG domain of a commitment's masking rows (index 0).
pub const DOMAIN_COMMIT_MASK: &[u8] = b"hachi/commit/mask";

impl<const K: usize> HachiPcs<K> {
    /// Hachi over $\mathbb F_{q^K}$ with the given choices.
    ///
    /// # Errors
    /// [`Error::Parameter`] for unusable choices (see [`crate::params::check`]).
    pub fn new(q: u64, choices: HachiChoices) -> Result<Self> {
        Ok(Self {
            ctx: Ctx::new(q, choices)?,
        })
    }

    /// The arithmetic context.
    pub fn ctx(&self) -> &Ctx<K> {
        &self.ctx
    }

    /// The layout of a multilinear polynomial in `vars` $\ge \alpha$ variables: $2^{\ell-\alpha}$
    /// ring elements, decomposed, split to minimise the first iteration's next witness.
    ///
    /// # Errors
    /// [`Error::Parameter`] if `vars` $< \alpha$.
    pub fn multilinear_layout(&self, vars: usize) -> Result<Layout> {
        let n = vars.checked_sub(self.ctx.alpha()).ok_or_else(|| {
            Error::Parameter(format!(
                "a polynomial needs at least alpha = {} variables",
                self.ctx.alpha()
            ))
        })?;
        best_layout(
            &self.ctx.choices,
            self.ctx.q,
            n,
            true,
            max_digit(self.ctx.choices.b),
            false,
            0,
        )
    }

    /// The layout of a vector of `len` $\ge 1$ field elements: one chunk of
    /// $2^{\lceil \log_2 \lceil len K/d \rceil \rceil}$ ring elements, decomposed, and with
    /// [`HachiChoices::hiding`] its masking row (module documentation, "Hiding").
    ///
    /// # Errors
    /// [`Error::Shape`] for an empty vector.
    pub fn vector_layout(&self, len: usize) -> Result<Layout> {
        if len == 0 {
            return Err(Error::Shape("an empty vector cannot be committed".into()));
        }
        let blocks = len.div_ceil(self.ctx.d_over_k());
        Ok(Layout {
            r: 0,
            m: blocks.next_power_of_two().trailing_zeros() as usize,
            decomposed: true,
            beta_s: max_digit(self.ctx.choices.b),
            masked: self.ctx.hiding().is_some(),
            extra: 0,
        })
    }

    /// The schedule of an opening of a multilinear polynomial in `vars` variables (module
    /// documentation, "Schedule").
    ///
    /// # Errors
    /// As [`HachiPcs::multilinear_layout`]; [`Error::Parameter`] if the fixed number of
    /// iterations cannot be parametrised.
    pub fn multilinear_schedule(&self, vars: usize) -> Result<Schedule> {
        self.schedule(
            self.multilinear_layout(vars)?,
            self.multilinear_reduction(vars),
        )
    }

    /// The schedule of an inner-product opening of a vector of `len` elements (module
    /// documentation, "Schedule").
    ///
    /// # Errors
    /// As [`HachiPcs::vector_layout`]; [`Error::Parameter`] if the fixed number of iterations
    /// cannot be parametrised.
    pub fn vector_schedule(&self, len: usize) -> Result<Schedule> {
        self.schedule(self.vector_layout(len)?, self.masks_or_direct())
    }

    /// The layout of a multilinear polynomial over $\mathbb F_{q^K}$ in `vars` $\ge \alpha -
    /// \kappa$ variables: $2^{\ell - \alpha + \kappa}$ ring elements (paper, Section 3.1),
    /// decomposed, split as [`HachiPcs::multilinear_layout`] splits them, and masked with
    /// [`HachiChoices::hiding`].
    ///
    /// # Errors
    /// [`Error::Parameter`] if `vars` $< \alpha - \kappa$.
    pub fn multilinear_ext_layout(&self, vars: usize) -> Result<Layout> {
        let n = (vars + self.ctx.kappa())
            .checked_sub(self.ctx.alpha())
            .ok_or_else(|| {
                Error::Parameter(format!(
                    "a polynomial over the field needs at least alpha - kappa = {} variables",
                    self.ctx.alpha() - self.ctx.kappa()
                ))
            })?;
        best_layout(
            &self.ctx.choices,
            self.ctx.q,
            n,
            true,
            max_digit(self.ctx.choices.b),
            self.ctx.hiding().is_some(),
            0,
        )
    }

    /// The reduction of an opening of a polynomial over the field or of a vector: masked with
    /// [`HachiChoices::hiding`], direct otherwise.
    fn masks_or_direct(&self) -> Reduction {
        if self.ctx.hiding().is_some() {
            Reduction::Masked
        } else {
            Reduction::Direct
        }
    }

    /// The schedule of an opening of a multilinear polynomial over $\mathbb F_{q^K}$ in `vars`
    /// variables (module documentation, "Schedule").
    ///
    /// # Errors
    /// As [`HachiPcs::multilinear_ext_layout`]; [`Error::Parameter`] if the fixed number of
    /// iterations cannot be parametrised.
    pub fn multilinear_ext_schedule(&self, vars: usize) -> Result<Schedule> {
        self.schedule(self.multilinear_ext_layout(vars)?, self.masks_or_direct())
    }

    /// The reduction of a multilinear opening in `vars` variables: $\ell - \kappa$ batching
    /// rounds.
    fn multilinear_reduction(&self, vars: usize) -> Reduction {
        Reduction::Multilinear {
            rounds: vars.saturating_sub(self.ctx.kappa()),
        }
    }

    /// The schedule of an opening of a first witness of layout `first` whose proof carries
    /// `reduction` ahead of its iterations: [`HachiChoices::fixed_iterations`] iterations if
    /// set, otherwise the candidate of [`Schedule::candidates`] whose proof has the fewest
    /// words, the one with fewer iterations on a tie.
    fn schedule(&self, first: Layout, reduction: Reduction) -> Result<Schedule> {
        let (c, q) = (&self.ctx.choices, self.ctx.q);
        if let Some(n) = c.fixed_iterations {
            return Schedule::fixed(c, q, K, first, n);
        }
        let mut best: Option<(usize, Schedule)> = None;
        for schedule in Schedule::candidates(c, q, K, first) {
            let words = self.proof_words(&schedule, reduction)?;
            if best.as_ref().is_none_or(|(least, _)| words < *least) {
                best = Some((words, schedule));
            }
        }
        best.map(|(_, s)| s)
            .ok_or_else(|| Error::Parameter("no schedule to choose from".into()))
    }

    /// The bits of a residue in the encodings: $\lceil \log_2 q \rceil$.
    fn residue_bits(&self) -> usize {
        (u64::BITS - (self.ctx.q - 1).leading_zeros()) as usize
    }

    /// The bits of a digit of the final witness: $\lceil \log_2 b \rceil$ (the digit set has
    /// $b$ elements).
    fn digit_bits(&self) -> usize {
        (u64::BITS - (self.ctx.choices.b - 1).leading_zeros()) as usize
    }

    /// The number of words of [`HachiPcs::commitment_to_words`]: the version, then $n_B d$
    /// packed residues.
    pub fn commitment_words(&self) -> usize {
        1 + (self.ctx.choices.n_b * self.ctx.d() * self.residue_bits()).div_ceil(64)
    }

    /// The number of words of [`HachiPcs::proof_to_words`] for an opening of a multilinear
    /// polynomial in `vars` variables. It depends only on `vars` and the parameters.
    ///
    /// # Errors
    /// As [`HachiPcs::multilinear_schedule`].
    pub fn multilinear_proof_words(&self, vars: usize) -> Result<usize> {
        self.proof_words(
            &self.multilinear_schedule(vars)?,
            self.multilinear_reduction(vars),
        )
    }

    /// The number of words of [`HachiPcs::proof_to_words`] for an inner-product opening of a
    /// vector of `len` elements. It depends only on `len` and the parameters.
    ///
    /// # Errors
    /// As [`HachiPcs::vector_schedule`].
    pub fn vector_proof_words(&self, len: usize) -> Result<usize> {
        self.proof_words(&self.vector_schedule(len)?, self.masks_or_direct())
    }

    /// The number of words of [`HachiPcs::proof_to_words`] for an opening of a multilinear
    /// polynomial over $\mathbb F_{q^K}$ in `vars` variables. It depends only on `vars` and the
    /// parameters.
    ///
    /// # Errors
    /// As [`HachiPcs::multilinear_ext_schedule`].
    pub fn multilinear_ext_proof_words(&self, vars: usize) -> Result<usize> {
        self.proof_words(
            &self.multilinear_ext_schedule(vars)?,
            self.masks_or_direct(),
        )
    }

    /// The words of [`HachiPcs::proof_to_words`] for the schedule `schedule` and the reduction
    /// `reduction` (module documentation, "Encodings"): the header ($5 + k$ words for $k$
    /// iterations, one more with hiding), then $\lceil \log_2 q \rceil$ bits per residue: for a
    /// multilinear opening the $K - 1$ partial evaluations ($K$ residues each), the $\ell -
    /// \kappa$ batching rounds ($2K$ each) and $f'(\mathbf r)$ ($K$), for a masked opening
    /// $y_{\mathfrak m}$ ($K$), $Y$ ($d$), per iteration $\mathbf v$ ($n_D d$), $\mathbf u'$
    /// ($n_B d$), the rounds ($L$ rounds of $b + 2$ and 3 field elements), $y'$ ($K$) and, except
    /// after the last iteration, $p'$ ($d$); the final witness, $2^{r+m} d$ residues without
    /// iterations, else the $2^{r+m} d/K = 2^L$ digits of the last table, of $\lceil \log_2 b
    /// \rceil$ bits each.
    ///
    /// # Errors
    /// [`Error::Parameter`] if the count of the reduction's residues overflows.
    fn proof_words(&self, schedule: &Schedule, reduction: Reduction) -> Result<usize> {
        let (d, c) = (self.ctx.d(), &self.ctx.choices);
        let round = K * (c.b as usize + 2 + 3);
        let mut residues = reduction
            .residues(K, d)
            .ok_or_else(|| Error::Parameter("the size of the proof overflows".into()))?;
        let count = schedule.iterations.len();
        for (index, it) in schedule.iterations.iter().enumerate() {
            // Per iteration: v, u', the L sumcheck rounds and y', then p' except after the last.
            residues += (c.n_d + c.n_b) * d + it.table_vars * round + K;
            if index + 1 < count {
                residues += d;
            }
        }
        let last = schedule.final_layout;
        let (entries, entry_bits) = if count == 0 {
            (last.chunks() * last.chunk_len() * d, self.residue_bits())
        } else {
            (
                last.chunks() * last.chunk_len() * self.ctx.d_over_k(),
                self.digit_bits(),
            )
        };
        let bits = residues * self.residue_bits() + entries * entry_bits;
        let header = 5 + schedule.iterations.len() + usize::from(self.ctx.hiding().is_some());
        Ok(header + bits.div_ceil(64))
    }

    /// Commits to the multilinear polynomial with monomial coefficients `coeffs` ($2^\ell$
    /// residues, $\ell \ge \alpha$, in the order of the module documentation).
    ///
    /// # Errors
    /// [`Error::Parameter`] with hiding: this opening's partial evaluations reveal more than the
    /// value (module documentation, "Hiding"). [`Error::Shape`] unless the length is a power of
    /// two of at least $d$. Arithmetic errors.
    pub fn commit_multilinear(
        &self,
        coeffs: &[u64],
        _seed: &[u8; 32],
    ) -> Result<(Commitment, State<K>)> {
        let ctx = &self.ctx;
        if ctx.hiding().is_some() {
            return Err(Error::Parameter(
                "polynomials over Z_q cannot be opened with hiding: commit them over the field"
                    .into(),
            ));
        }
        if !coeffs.len().is_power_of_two() || coeffs.len() < ctx.d() {
            return Err(Error::Shape(format!(
                "2^l coefficients with l >= alpha = {} are expected",
                ctx.alpha()
            )));
        }
        let vars = coeffs.len().trailing_zeros() as usize;
        let layout = self.multilinear_layout(vars)?;
        let coeffs: Vec<u64> = coeffs.iter().map(|c| ctx.ring.zq().reduce(*c)).collect();
        let message = self.multilinear_message(&layout, vars, &coeffs)?;
        let committed = commit(ctx, 0, layout, message, Vec::new())?;
        Ok((
            Commitment {
                u: committed.u.clone(),
            },
            State::new(Input::Multilinear { vars, coeffs }, committed),
        ))
    }

    /// The monomial coefficients $f'\_j = \sum_i f_{i \Vert j} Z_i$ of $f' = \sum_i f_i Z_i$,
    /// $j \in \lbrace 0, 1 \rbrace^{\ell - \kappa}$, of a polynomial in `vars` $\ge \kappa$
    /// variables.
    fn packed(&self, vars: usize, coeffs: &[u64]) -> Vec<Ext<K>> {
        let inner = 1usize << (vars - self.ctx.kappa());
        (0..inner)
            .map(|j| Ext(core::array::from_fn(|i| coeffs[i * inner + j])))
            .collect()
    }

    /// The message chunks of a multilinear polynomial: the coefficients $f'\_j$
    /// ([`HachiPcs::packed`]), then $\psi$ of $d/K$ consecutive $f'\_j$.
    fn multilinear_message(
        &self,
        layout: &Layout,
        vars: usize,
        coeffs: &[u64],
    ) -> Result<Vec<Vec<Poly>>> {
        let ctx = &self.ctx;
        let elements = self
            .packed(vars, coeffs)
            .chunks_exact(ctx.d_over_k())
            .map(|block| ctx.sub.psi(block))
            .collect::<hachi_co_math::Result<Vec<_>>>()?;
        Ok(elements
            .chunks_exact(layout.message_len())
            .map(<[Poly]>::to_vec)
            .collect())
    }

    /// The masking rows of a masked layout, one per chunk, from the secret `seed`: uniform in
    /// $\mathcal R_q$. None otherwise.
    fn masking_rows(&self, layout: &Layout, seed: &[u8; 32]) -> Result<Vec<Poly>> {
        if !layout.masked {
            return Ok(Vec::new());
        }
        let d = self.ctx.d();
        let mut prg = ShakePrg::new(seed, DOMAIN_COMMIT_MASK, 0);
        let residues = uniform_mod_q(&mut prg, self.ctx.q, layout.chunks() * d)?;
        Ok(residues.chunks_exact(d).map(|c| Poly(c.to_vec())).collect())
    }

    /// One chunk of the message of a polynomial over the field: $\psi$ of each block of $d/K$
    /// coefficients of `chunk`, reduced, $2^m$ ring elements for a chunk of $2^m d/K$ coefficients.
    fn pack_ext_chunk(&self, chunk: &[Ext<K>]) -> Result<Vec<Poly>> {
        let ctx = &self.ctx;
        let g = ctx.d_over_k();
        let mut block = Vec::with_capacity(g);
        Ok(chunk
            .chunks_exact(g)
            .map(|c| {
                block.clear();
                block.extend(c.iter().map(|x| ctx.field().reduce(x)));
                ctx.sub.psi(&block)
            })
            .collect::<hachi_co_math::Result<Vec<Poly>>>()?)
    }

    /// The message chunks of the coefficients `coeffs` in `layout`, without masking rows, packed
    /// from the last chunks to the first in batches of about 64 MiB of coefficients, the packed
    /// ones freed after each batch, so that the coefficients and the message are not held whole at
    /// once. The chunks are those of [`HachiPcs::pack_ext_chunk`], in order.
    fn pack_ext_owned(&self, coeffs: Vec<Ext<K>>, layout: &Layout) -> Result<Vec<Vec<Poly>>> {
        self.pack_ext_owned_in(coeffs, layout, 64 << 20)
    }

    /// [`HachiPcs::pack_ext_owned`] with batches of at most `batch_bytes` of coefficients (at
    /// least one chunk), a parameter for the tests.
    fn pack_ext_owned_in(
        &self,
        mut coeffs: Vec<Ext<K>>,
        layout: &Layout,
        batch_bytes: usize,
    ) -> Result<Vec<Vec<Poly>>> {
        use rayon::prelude::*;
        let per_chunk = self.ctx.d_over_k() * layout.message_len();
        let chunks = coeffs.len() / per_chunk;
        let batch = (batch_bytes / (per_chunk * core::mem::size_of::<Ext<K>>())).max(1);
        let mut message: Vec<Vec<Poly>> = Vec::with_capacity(chunks);
        let mut end = chunks;
        while end > 0 {
            let start = end.saturating_sub(batch);
            let packed = coeffs[start * per_chunk..end * per_chunk]
                .par_chunks_exact(per_chunk)
                .map(|chunk| self.pack_ext_chunk(chunk))
                .collect::<Result<Vec<_>>>()?;
            message.extend(packed.into_iter().rev());
            coeffs.truncate(start * per_chunk);
            coeffs.shrink_to_fit();
            end = start;
        }
        message.reverse();
        Ok(message)
    }

    /// Commits to the multilinear polynomial over $\mathbb F_{q^K}$ with monomial coefficients
    /// `coeffs` ($2^\ell$ field elements, $\ell \ge \alpha - \kappa$; module documentation for
    /// the order), with randomness from the secret `seed` if the choices hide.
    ///
    /// # Errors
    /// [`Error::Shape`] unless the length is a power of two of at least $d/K$; arithmetic errors.
    pub fn commit_multilinear_ext(
        &self,
        coeffs: &[Ext<K>],
        seed: &[u8; 32],
    ) -> Result<(Commitment, State<K>)> {
        let ctx = &self.ctx;
        if !coeffs.len().is_power_of_two() || coeffs.len() < ctx.d_over_k() {
            return Err(Error::Shape(format!(
                "2^l coefficients with l >= alpha - kappa = {} are expected",
                ctx.alpha() - ctx.kappa()
            )));
        }
        let vars = coeffs.len().trailing_zeros() as usize;
        let layout = self.multilinear_ext_layout(vars)?;
        let message = {
            use rayon::prelude::*;
            coeffs
                .par_chunks_exact(ctx.d_over_k() * layout.message_len())
                .map(|chunk| self.pack_ext_chunk(chunk))
                .collect::<Result<Vec<_>>>()?
        };
        self.commit_ext_message(vars, layout, message, seed)
    }

    /// [`HachiPcs::commit_multilinear_ext`] with the coefficients handed over: they are freed as
    /// they are packed, from the last chunks to the first in batches of about 64 MiB, before the
    /// products of the commitment. The
    /// commitment and the state are those of [`HachiPcs::commit_multilinear_ext`].
    ///
    /// # Errors
    /// As [`HachiPcs::commit_multilinear_ext`].
    pub fn commit_multilinear_ext_owned(
        &self,
        coeffs: Vec<Ext<K>>,
        seed: &[u8; 32],
    ) -> Result<(Commitment, State<K>)> {
        let ctx = &self.ctx;
        if !coeffs.len().is_power_of_two() || coeffs.len() < ctx.d_over_k() {
            return Err(Error::Shape(format!(
                "2^l coefficients with l >= alpha - kappa = {} are expected",
                ctx.alpha() - ctx.kappa()
            )));
        }
        let vars = coeffs.len().trailing_zeros() as usize;
        let layout = self.multilinear_ext_layout(vars)?;
        let message = self.pack_ext_owned(coeffs, &layout)?;
        self.commit_ext_message(vars, layout, message, seed)
    }

    /// The commitment of a polynomial over the field to its packed `message`: the masking rows
    /// and the randomness from `seed`, then [`commit`].
    fn commit_ext_message(
        &self,
        vars: usize,
        layout: Layout,
        mut message: Vec<Vec<Poly>>,
        seed: &[u8; 32],
    ) -> Result<(Commitment, State<K>)> {
        let ctx = &self.ctx;
        for (chunk, row) in message.iter_mut().zip(self.masking_rows(&layout, seed)?) {
            chunk.push(row);
        }
        let r = commitment_randomness(ctx, &layout, seed, DOMAIN_COMMIT_R)?;
        let committed = commit(ctx, 0, layout, message, r)?;
        Ok((
            Commitment {
                u: committed.u.clone(),
            },
            State::new(Input::MultilinearExt { vars }, committed),
        ))
    }

    /// Drops the packed coefficients of the message of a state of
    /// [`HachiPcs::commit_multilinear_ext`], until [`HachiPcs::prove_multilinear_ext_rebuilt`]
    /// packs them again from the coefficients (the opening reads them, nothing before it does):
    /// the state keeps the rest of each chunk (the masking row) and a fingerprint of what it
    /// dropped. Returns whether it dropped them (not for other kinds of states, or twice).
    pub fn shed_message(&self, state: &mut State<K>) -> bool {
        if !matches!(state.input, Input::MultilinearExt { .. }) || state.shed.is_some() {
            return false;
        }
        let len = state.committed.layout.message_len();
        let fingerprint = message_fingerprint(&state.committed.message, len);
        let tails = state
            .committed
            .message
            .iter_mut()
            .map(|chunk| chunk.split_off(len.min(chunk.len())))
            .collect();
        state.committed.message = Vec::new();
        state.shed = Some(Shed {
            tails,
            fingerprint,
            from_source: false,
        });
        true
    }

    /// [`HachiPcs::prove_multilinear_ext_owned`] on a state that [`HachiPcs::shed_message`] may
    /// have reduced: `coeffs`, the committed coefficients, are packed again (and freed as they are
    /// packed) if the state was shed, and ignored otherwise. The value and the proof are those of
    /// [`HachiPcs::prove_multilinear_ext`] on the state before it was shed.
    ///
    /// # Errors
    /// As [`HachiPcs::prove_multilinear_ext`]; [`Error::Shape`] for coefficients of another number,
    /// and [`Error::Protocol`] for coefficients that do not pack to the shed message.
    pub fn prove_multilinear_ext_rebuilt(
        &self,
        mut state: State<K>,
        coeffs: Vec<Ext<K>>,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> Result<(Ext<K>, Proof<K>)> {
        if let Some(shed) = state.shed.take() {
            let Input::MultilinearExt { vars } = state.input else {
                return Err(Error::Shape(
                    "a shed state is a polynomial over the field".into(),
                ));
            };
            if shed.from_source {
                return Err(Error::Protocol(
                    "this state was committed from a source: open it with \
                     prove_multilinear_ext_source"
                        .into(),
                ));
            }
            if coeffs.len() != 1usize << vars {
                return Err(Error::Shape(format!(
                    "2^{vars} coefficients are expected to open this state"
                )));
            }
            let layout = state.committed.layout;
            let mut message = self.pack_ext_owned(coeffs, &layout)?;
            if message_fingerprint(&message, layout.message_len()) != shed.fingerprint {
                return Err(Error::Protocol(
                    "the coefficients are not those of the commitment".into(),
                ));
            }
            for (chunk, tail) in message.iter_mut().zip(shed.tails) {
                chunk.extend(tail);
            }
            state.committed.message = message;
        }
        self.prove_multilinear_ext_owned(state, point, context, seed)
    }

    /// Commits as [`HachiPcs::commit_multilinear_ext`] to the coefficients that `coeffs` reads,
    /// without holding them or their packed message: the commitment packs the elements of each
    /// block of its products as it reads them ([`commit_source`]). The state keeps the masking
    /// rows and the source's fingerprint, and opens by [`HachiPcs::prove_multilinear_ext_source`]
    /// with the same coefficients. The commitment is that of [`HachiPcs::commit_multilinear_ext`]
    /// on those coefficients.
    ///
    /// # Errors
    /// [`Error::Shape`] unless $\ell \ge \alpha - \kappa$; arithmetic errors.
    pub fn commit_multilinear_ext_source(
        &self,
        coeffs: &ExtSource<'_, K>,
        seed: &[u8; 32],
    ) -> Result<(Commitment, State<K>)> {
        let ctx = &self.ctx;
        let vars = coeffs.vars;
        if vars >= usize::BITS as usize || (1usize << vars) < ctx.d_over_k() {
            return Err(Error::Shape(format!(
                "2^l coefficients with l >= alpha - kappa = {} are expected",
                ctx.alpha() - ctx.kappa()
            )));
        }
        let layout = self.multilinear_ext_layout(vars)?;
        let rows = self.masking_rows(&layout, seed)?;
        let elements = |c: usize, range: core::ops::Range<usize>| {
            self.source_elements(coeffs, &layout, &rows, c, range)
        };
        let r = commitment_randomness(ctx, &layout, seed, DOMAIN_COMMIT_R)?;
        let (committed, fingerprint) = match &coeffs.parts {
            None => (
                commit_source(ctx, 0, layout, &elements, r)?,
                coeffs.fingerprint,
            ),
            Some(parts) => {
                // Part p of the coefficients is chunks p chunks / count .. of the message.
                let message = layout.chunks() * layout.message_len() * ctx.d_over_k();
                if message != 1 << vars || !layout.chunks().is_multiple_of(parts.count) {
                    return Err(Error::Shape(format!(
                        "{} parts of the 2^{vars} coefficients do not split the {} chunks",
                        parts.count,
                        layout.chunks()
                    )));
                }
                let prepare = |p: usize| (parts.prepare)(p);
                let committed = commit_source_parts(
                    ctx,
                    0,
                    layout,
                    &elements,
                    r,
                    Some((parts.count, &prepare)),
                )?;
                (committed, (parts.fingerprint)())
            }
        };
        let tails = if layout.masked {
            rows.into_iter().map(|row| Vec::from([row])).collect()
        } else {
            vec![Vec::new(); layout.chunks()]
        };
        let mut state = State::new(Input::MultilinearExt { vars }, committed);
        state.shed = Some(Shed {
            tails,
            fingerprint,
            from_source: true,
        });
        Ok((
            Commitment {
                u: state.committed.u.clone(),
            },
            state,
        ))
    }

    /// Elements `range` of chunk `c` of the message of `coeffs` in `layout`, those of
    /// [`HachiPcs::pack_ext_chunk`]: $\psi$ of each block of $d/K$ coefficients, and at index
    /// [`Layout::message_len`] the chunk's masking row from `rows`.
    fn source_elements(
        &self,
        coeffs: &ExtSource<'_, K>,
        layout: &Layout,
        rows: &[Poly],
        c: usize,
        range: core::ops::Range<usize>,
    ) -> Result<Vec<Poly>> {
        let ctx = &self.ctx;
        let field = ctx.field();
        let (g, len) = (ctx.d_over_k(), layout.message_len());
        let mut block = vec![field.zero(); g];
        range
            .map(|e| {
                if e < len {
                    (coeffs.read.as_ref())((c * len + e) * g, &mut block);
                    for x in &mut block {
                        *x = field.reduce(x);
                    }
                    Ok(ctx.sub.psi(&block)?)
                } else {
                    rows.get(c).cloned().ok_or_else(|| {
                        Error::Shape("an element past the message of a chunk".into())
                    })
                }
            })
            .collect()
    }

    /// Opens a state of [`HachiPcs::commit_multilinear_ext_source`] with the same coefficients
    /// `coeffs` (checked by their fingerprint): the value and the proof of
    /// [`HachiPcs::prove_multilinear_ext`] on the state of [`HachiPcs::commit_multilinear_ext`].
    ///
    /// # Errors
    /// As [`HachiPcs::prove_multilinear_ext`]; [`Error::Protocol`] for another kind of state or
    /// other coefficients.
    pub fn prove_multilinear_ext_source(
        &self,
        mut state: State<K>,
        coeffs: ExtSource<'_, K>,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> Result<(Ext<K>, Proof<K>)> {
        let shed = state
            .shed
            .take()
            .ok_or_else(|| Error::Protocol("this state was not committed from a source".into()))?;
        let vars_match =
            matches!(state.input, Input::MultilinearExt { vars } if vars == coeffs.vars);
        if !shed.from_source || !vars_match || shed.fingerprint != coeffs.fingerprint {
            return Err(Error::Protocol(
                "the coefficients are not those of the commitment".into(),
            ));
        }
        // The message is read from the source when the opening reads it: the coefficients'
        // elements, then the chunk's masking row from the kept tails. The source owns them, and
        // the first iteration drops it after its step 2 (`Witness::Sourced`).
        let layout = state.committed.layout;
        let len = layout.message_len();
        let tails = shed.tails;
        // Coefficient parts are chunk parts (`commit_multilinear_ext_source` checks the split).
        let parts = coeffs.parts.as_ref().map(|parts| {
            let prepare = parts.prepare.clone();
            OwnedParts {
                count: parts.count,
                prepare: Box::new(move |p: usize| prepare(p)),
            }
        });
        let elements = move |c: usize, range: core::ops::Range<usize>| -> Result<Vec<Poly>> {
            let head = range.start.min(len)..range.end.min(len);
            let mut out = self.source_elements(&coeffs, &layout, &[], c, head)?;
            for e in range.start.max(len)..range.end.max(len) {
                out.push(
                    tails
                        .get(c)
                        .and_then(|t| t.get(e - len))
                        .cloned()
                        .ok_or_else(|| {
                            Error::Shape("an element past the chunk of the message".into())
                        })?,
                );
            }
            Ok(out)
        };
        let source = OwnedSource {
            chunks: layout.chunks(),
            chunk_len: layout.chunk_len(),
            elements: Box::new(elements),
            parts,
        };
        self.prove_multilinear_ext_from(Held::Owned(state), Some(source), point, context, seed)
    }

    /// Commits to the vector `values` ($N \ge 1$ field elements), with its masking row and
    /// randomness from the secret `seed` if the choices hide (module documentation, "Hiding").
    ///
    /// # Errors
    /// [`Error::Shape`] for an empty vector; arithmetic errors.
    pub fn commit_vector(
        &self,
        values: &[Ext<K>],
        seed: &[u8; 32],
    ) -> Result<(Commitment, State<K>)> {
        let ctx = &self.ctx;
        let layout = self.vector_layout(values.len())?;
        let values: Vec<Ext<K>> = values.iter().map(|v| ctx.field().reduce(v)).collect();
        let mut chunk = self.vector_blocks(&layout, &values)?;
        chunk.extend(self.masking_rows(&layout, seed)?);
        let r = commitment_randomness(ctx, &layout, seed, DOMAIN_COMMIT_R)?;
        let committed = commit(ctx, 0, layout, vec![chunk], r)?;
        Ok((
            Commitment {
                u: committed.u.clone(),
            },
            State::new(Input::Vector { values }, committed),
        ))
    }

    /// $\psi$ of the blocks of $d/K$ entries of `values`, zero-padded to $2^m$ blocks (the
    /// message, without a masking row).
    fn vector_blocks(&self, layout: &Layout, values: &[Ext<K>]) -> Result<Vec<Poly>> {
        let ctx = &self.ctx;
        let per = ctx.d_over_k();
        let mut padded = values.to_vec();
        padded.resize(layout.message_len() * per, ctx.field().zero());
        Ok(padded
            .chunks_exact(per)
            .map(|block| ctx.sub.psi(block))
            .collect::<hachi_co_math::Result<Vec<_>>>()?)
    }

    /// The transcript after the statement: the choices, the object's shape, the commitment,
    /// the point or weights, and the value.
    fn statement(
        &self,
        context: &[u8; 32],
        kind: u64,
        size: usize,
        commitment: &[Poly],
        point: &[Ext<K>],
        value: &Ext<K>,
    ) -> Fs {
        let c = &self.ctx.choices;
        let mut fs = Fs::new(context);
        let mut words = vec![
            VERSION,
            kind,
            size as u64,
            self.ctx.q,
            K as u64,
            c.d as u64,
            c.b,
            c.omega,
            c.n_a as u64,
            c.n_b as u64,
            c.n_d as u64,
            c.max_iterations as u64,
            c.fixed_iterations.map_or(u64::MAX, |n| n as u64),
        ];
        words.extend(
            c.seed
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| u64::from_le_bytes(*b)),
        );
        // Only with hiding, after the other words: a non-hiding statement is unchanged.
        if let Some(h) = c.hiding {
            words.extend(h.words());
        }
        fs.words(LABEL_STATEMENT, &words);
        fs.polys(LABEL_COMMITMENT, commitment);
        fs.exts(LABEL_POINT, point);
        fs.exts(LABEL_VALUE, core::slice::from_ref(value));
        fs
    }

    /// The value $\sum_i \mathrm{mono}\_i(x_1, \dots, x_\kappa) y_i$ of the partial evaluations
    /// `partials` at `point` (paper, Section 3.2).
    fn value_of_partials(&self, point: &[Ext<K>], partials: &[Ext<K>]) -> Result<Ext<K>> {
        let field = self.ctx.field();
        let mono = self.ctx.sub.monomials(&point[..self.ctx.kappa()])?;
        Ok(mono.iter().zip(partials).fold(field.zero(), |acc, (m, y)| {
            field.add(&acc, &field.mul(m, y))
        }))
    }

    /// The weights of the reduction of $f'(\mathbf r)$ to $\mathcal R_q$ (paper, Section 3.1)
    /// at a point $\mathbf r \in \mathbb F_{q^K}^{\ell - \kappa}$ of a multilinear opening in
    /// `layout`, split into $\mathbf r_{\mathrm{high}}$ and its last $\alpha - \kappa$
    /// coordinates $\mathbf r_{\mathrm{low}}$: $\mathbf b$ and $\mathbf a$, the embedded
    /// monomials of the first $r$ and of the other coordinates of $\mathbf r_{\mathrm{high}}$,
    /// and $v = \psi$ of the monomials of $\mathbf r_{\mathrm{low}}$.
    ///
    /// # Errors
    /// [`Error::Shape`] unless $\mathbf r$ has $r + m + \alpha - \kappa$ coordinates.
    fn ring_weights(&self, layout: &Layout, r: &[Ext<K>]) -> Result<(Vec<Poly>, Vec<Poly>, Poly)> {
        let sub = &self.ctx.sub;
        let low = self.ctx.alpha() - self.ctx.kappa();
        if r.len() != layout.r + layout.m + low {
            return Err(Error::Shape(format!(
                "a point of {} coordinates is expected",
                layout.r + layout.m + low
            )));
        }
        let (high, low) = r.split_at(layout.r + layout.m);
        let embedded = |x: &[Ext<K>]| -> Result<Vec<Poly>> {
            Ok(sub.monomials(x)?.iter().map(|e| sub.embed(e)).collect())
        };
        Ok((
            embedded(&high[..layout.r])?,
            embedded(&high[layout.r..])?,
            sub.psi(&sub.monomials(low)?)?,
        ))
    }

    /// $E(Y) = (K/d)\,\mathrm{Tr}\_H(Y \sigma_{-1}(v))$ as a field element (Theorem 2).
    fn traced(&self, ring_value: &Poly, v: &Poly) -> Result<Ext<K>> {
        Ok(self.ctx.sub.extract(&self.ctx.sub.e_map(ring_value, v)?)?)
    }

    /// The claim that the iterations prove for a multilinear opening: $Y = \sum_i b_i \mathbf
    /// a^\top \mathbf f_i$ with the weights of [`HachiPcs::ring_weights`] at the batching
    /// sumcheck's point $\mathbf r$, once the honest prover checked $E(Y) = f'(\mathbf r)$ =
    /// `at_r`.
    ///
    /// # Errors
    /// [`Error::Protocol`] if that check fails; as [`HachiPcs::ring_weights`].
    fn ring_claim(&self, state: &State<K>, r: &[Ext<K>], at_r: &Ext<K>) -> Result<Claim> {
        let (b, a, v) = self.ring_weights(&state.committed.layout, r)?;
        let target = evaluate_claim(&self.ctx, &b, &a, &state.committed.message)?;
        if self.traced(&target, &v)? != *at_r {
            return Err(Error::Protocol(
                "the trace check of the input reduction fails".into(),
            ));
        }
        Ok(Claim { b, a, target })
    }

    /// $x^* \in \mathbb F_{q^K} \setminus \lbrace 0 \rbrace$ from the transcript, which has absorbed
    /// $y_{\mathfrak m}$: uniform field elements until one is not zero.
    fn mask_coin(&self, fs: &mut Fs) -> Result<Ext<K>> {
        self.nonzero_coin(fs, LABEL_MASK_COIN)
    }

    /// A nonzero field element from the transcript under `label`: uniform field elements until one
    /// is not zero.
    fn nonzero_coin(&self, fs: &mut Fs, label: &[u8]) -> Result<Ext<K>> {
        let field = self.ctx.field();
        for _ in 0..hachi_co_sampling::uniform::MAX_TRIES {
            let coin = fs.field_coins::<K>(label, self.ctx.q, 1)?[0];
            if !field.is_zero(&coin) {
                return Ok(coin);
            }
        }
        Err(Error::Protocol("no nonzero challenge".into()))
    }

    /// Step 1 of `Eval` with masks, for the prover: from the weights $\mathbf b, \mathbf a, v$ it
    /// computes $Y$ and $Y_{\mathfrak m}$, absorbs $y_{\mathfrak m}$, draws $x^*$ and absorbs
    /// $Y^*$. Returns $y_{\mathfrak m}$ and the first claim, whose target is $Y^*$.
    ///
    /// # Errors
    /// [`Error::Protocol`] if the honest check fails. Arithmetic errors.
    #[allow(clippy::too_many_arguments)] // the state, its message and the claim's parts
    fn masked_claim(
        &self,
        state: &State<K>,
        message: MessageView<'_>,
        b: Vec<Poly>,
        a: Vec<Poly>,
        v: &Poly,
        value: &Ext<K>,
        fs: &mut Fs,
    ) -> Result<(Ext<K>, Claim)> {
        let len = state.committed.layout.message_len();
        let products = self.chunk_products(&a, message, len)?;
        let (mask_eval, claim, _) =
            self.masked_claim_from(message, &products, b, a, v, value, fs)?;
        Ok((mask_eval, claim))
    }

    /// $\langle \mathbf a, \mathbf f_i \rangle$ over the first `len` elements of every chunk of
    /// `message` (in parallel, $\mathbf a$ transformed once): the products that the claim's value,
    /// the masked claim and step 1 of the first iteration read.
    ///
    /// # Errors
    /// [`Error::Shape`] for a chunk shorter than `len`; arithmetic errors.
    fn chunk_products(
        &self,
        a: &[Poly],
        message: MessageView<'_>,
        len: usize,
    ) -> Result<Vec<Poly>> {
        let ring = &self.ctx.ring;
        let a_row = ring.prepare(core::slice::from_ref(&a.to_vec()))?;
        // The chunks of each part after it is prepared, in order.
        let mut out = Vec::with_capacity(message.chunks());
        for p in 0..message.part_count() {
            let range = message.prepare_part(p)?;
            out.extend(self.chunk_products_of(&a_row, message, len, range)?);
        }
        Ok(out)
    }

    /// [`HachiPcs::chunk_products`] on the chunks `range` (of the part prepared last).
    fn chunk_products_of(
        &self,
        a_row: &hachi_co_math::rns::Prepared,
        message: MessageView<'_>,
        len: usize,
        range: core::ops::Range<usize>,
    ) -> Result<Vec<Poly>> {
        use rayon::prelude::*;
        let ring = &self.ctx.ring;
        range
            .into_par_iter()
            .map(|c| {
                let chunk = message.chunk(c)?;
                let part = chunk
                    .get(..len)
                    .ok_or_else(|| Error::Shape("a chunk is shorter than its weights".into()))?;
                ring.mat_vec_prepared(a_row, part)?
                    .pop()
                    .ok_or_else(|| Error::Shape("a^T f_i is one row".into()))
            })
            .collect()
    }

    /// The masked claim of step 1 from the chunks' `products` with $\mathbf a$
    /// ([`HachiPcs::chunk_products`]) and their masking rows: absorbs $y_{\mathfrak m}$, draws
    /// $x^*$, forms and absorbs $Y^* = y + x^* y_{\mathfrak m}$, and returns the mask's value, the
    /// claim ($\mathbf a$ extended by $x^*$) and the first iteration's
    /// $w_i = \langle \mathbf a, \mathbf f_i \rangle + x^* m_i$ for the masking rows $m_i$.
    #[allow(clippy::too_many_arguments)] // the message, its products and the claim's parts
    fn masked_claim_from(
        &self,
        message: MessageView<'_>,
        products: &[Poly],
        b: Vec<Poly>,
        a: Vec<Poly>,
        v: &Poly,
        value: &Ext<K>,
        fs: &mut Fs,
    ) -> Result<(Ext<K>, Claim, Vec<Poly>)> {
        let (ctx, ring) = (&self.ctx, &self.ctx.ring);
        let field = ctx.field();
        let len = a.len();
        let mut y = ring.zero();
        let mut y_mask = ring.zero();
        let mut rows = Vec::with_capacity(message.chunks());
        for (c, bi) in b.iter().enumerate().take(message.chunks()) {
            let product = products
                .get(c)
                .ok_or_else(|| Error::Shape("one product per chunk is expected".into()))?;
            y = ring.add(&y, &ring.mul(bi, product)?)?;
            let row = message.element(c, len)?;
            y_mask = ring.add(&y_mask, &ring.mul(bi, &row)?)?;
            rows.push(row.into_owned());
        }
        let mask_eval = self.traced(&y_mask, v)?;
        fs.exts(LABEL_MASK_EVAL, core::slice::from_ref(&mask_eval));
        let coin = self.mask_coin(fs)?;
        let embedded = ctx.sub.embed(&coin);
        let target = ring.add(&y, &ring.mul(&embedded, &y_mask)?)?;
        fs.polys(LABEL_RING_VALUE, core::slice::from_ref(&target));
        if self.traced(&target, v)? != field.add(value, &field.mul(&coin, &mask_eval)) {
            return Err(Error::Protocol(
                "the trace check of the masked claim fails".into(),
            ));
        }
        let w = products
            .iter()
            .zip(&rows)
            .map(|(p, row)| Ok(ring.add(p, &ring.mul(&embedded, row)?)?))
            .collect::<Result<Vec<_>>>()?;
        let mut a = a;
        a.push(embedded);
        Ok((mask_eval, Claim { b, a, target }, w))
    }

    /// The verifier's step 1 with masks: absorbs $y_{\mathfrak m}$, draws $x^*$, checks
    /// $E(Y^*) = y + x^* y_{\mathfrak m}$, absorbs $Y^*$, and returns the first claim.
    ///
    /// # Errors
    /// [`VerifyError::Check`] if the trace check fails.
    fn verify_masked_claim(
        &self,
        proof: &Proof<K>,
        b: Vec<Poly>,
        a: Vec<Poly>,
        v: &Poly,
        value: &Ext<K>,
        fs: &mut Fs,
    ) -> core::result::Result<Claim, VerifyError> {
        let ctx = &self.ctx;
        let field = ctx.field();
        let Some(mask_eval) = &proof.mask_eval else {
            return Err(VerifyError::Malformed(
                "a masked opening carries the masks' value".into(),
            ));
        };
        check_exts(ctx, "masks' value", core::slice::from_ref(mask_eval), 1)?;
        fs.exts(LABEL_MASK_EVAL, core::slice::from_ref(mask_eval));
        let coin = self.mask_coin(fs)?;
        if self.traced(&proof.ring_value, v)? != field.add(value, &field.mul(&coin, mask_eval)) {
            return Err(VerifyError::Check("the trace of the masked claim".into()));
        }
        fs.polys(LABEL_RING_VALUE, core::slice::from_ref(&proof.ring_value));
        let mut a = a;
        a.push(ctx.sub.embed(&coin));
        Ok(Claim {
            b,
            a,
            target: proof.ring_value.clone(),
        })
    }

    /// The weights $\sigma_{-1}(\psi(\mathbf e_t))$ of an inner product, zero-padded.
    fn vector_weights(&self, layout: &Layout, weights: &[Ext<K>]) -> Result<Vec<Poly>> {
        self.vector_blocks(layout, weights)?
            .iter()
            .map(|p| Ok(self.ctx.ring.automorphism(p, -1)?))
            .collect()
    }

    /// Runs the iterations and the final step for the prover; `reduction` is what the proof
    /// carries ahead of them (it enters the choice of the schedule). Returns the iterations and
    /// the final witness.
    ///
    /// The witness of each later level is owned here and handed to its iteration, which frees its
    /// message after step 2; so is the first one when `committed` is owned ([`Witness`]).
    fn prove_rest<'w>(
        &self,
        layout: Layout,
        reduction: Reduction,
        committed: impl Into<Witness<'w>>,
        claim: Claim,
        fs: &mut Fs,
        seed: &[u8; 32],
    ) -> Result<(Vec<IterationProof<K>>, FinalWitness)> {
        self.prove_rest_with(layout, reduction, committed, claim, None, fs, seed)
    }

    /// [`HachiPcs::prove_rest`] with the first iteration's $w_i$ given if `first_w` is
    /// ([`iteration::prove_with_w`]).
    #[allow(clippy::too_many_arguments)] // the inputs of prove_rest and the given w
    fn prove_rest_with<'w>(
        &self,
        layout: Layout,
        reduction: Reduction,
        committed: impl Into<Witness<'w>>,
        claim: Claim,
        mut first_w: Option<Vec<Poly>>,
        fs: &mut Fs,
        seed: &[u8; 32],
    ) -> Result<(Vec<IterationProof<K>>, FinalWitness)> {
        let schedule = self.schedule(layout, reduction)?;
        let first: Witness<'w> = committed.into();
        // The witness of each later iteration is sourced from the previous one's table
        // (`iteration::Next`), which it reads in its steps 1 and 2 and then drops, instead of its
        // message packed whole. The last iteration's table is the final witness (step 3).
        let mut current: Witness<'_> = first;
        let mut claim = claim;
        let count = schedule.iterations.len();
        let mut proofs = Vec::with_capacity(count);
        let protocol = |what: &str| Error::Protocol(what.into());
        for (index, params) in schedule.iterations.iter().enumerate() {
            let last = index + 1 == count;
            let (proof, next, next_claim) = iteration::prove_next(
                &self.ctx,
                index,
                params,
                current,
                &claim,
                first_w.take(),
                fs,
                seed,
                last,
            )?;
            proofs.push(proof);
            if last {
                let table = next
                    .into_table()
                    .ok_or_else(|| protocol("the last iteration keeps its table"))?;
                return Ok((proofs, FinalWitness::Digits(table)));
            }
            current = next.into_witness(&self.ctx)?;
            claim =
                next_claim.ok_or_else(|| protocol("an iteration before the last has a claim"))?;
        }
        Ok((
            proofs,
            FinalWitness::Message(current.committed().message.concat()),
        ))
    }

    /// Proves $f(\mathbf x)$ for a committed multilinear polynomial; returns the value and the
    /// proof. `context` binds the proof to the caller's transcript; `seed` is the opening's
    /// fresh secret seed (module documentation, "Randomness").
    ///
    /// The transcript absorbs the statement with the value, the partial evaluations sent, the
    /// batching sumcheck ($\boldsymbol\eta$, each round's values then its challenge,
    /// $f'(\mathbf r)$; [`crate::partials`]), $Y$, then the iterations.
    ///
    /// # Errors
    /// [`Error::Shape`] for a point of the wrong length or a state of a vector; prover errors.
    pub fn prove_multilinear(
        &self,
        state: &State<K>,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> Result<(Ext<K>, Proof<K>)> {
        let ctx = &self.ctx;
        let field = ctx.field();
        let Input::Multilinear { vars, coeffs } = &state.input else {
            return Err(Error::Shape(
                "this state is not a multilinear polynomial".into(),
            ));
        };
        if point.len() != *vars {
            return Err(Error::Shape(format!(
                "a point of {vars} coordinates is expected"
            )));
        }
        let point: Vec<Ext<K>> = point.iter().map(|x| field.reduce(x)).collect();
        let kappa = ctx.kappa();
        let inner = 1usize << (vars - kappa);
        let x = &point[kappa..];
        let mono = ctx.sub.monomials(x)?;
        let partials: Vec<Ext<K>> = (0..K)
            .map(|i| {
                coeffs[i * inner..(i + 1) * inner]
                    .iter()
                    .zip(&mono)
                    .fold(field.zero(), |acc, (c, m)| {
                        field.add(&acc, &field.scale(m, *c))
                    })
            })
            .collect();
        let value = self.value_of_partials(&point, &partials)?;
        let sent = partials::omit(field, &ctx.sub.monomials(&point[..kappa])?, &partials)?;
        let mut fs = self.statement(context, 1, *vars, &state.committed.u, &point, &value);
        fs.exts(LABEL_PARTIALS, &sent);
        let (batch_rounds, r, at_r) =
            partials::prove(field, &self.packed(*vars, coeffs), x, &partials, &mut fs)?;
        let claim = self.ring_claim(state, &r, &at_r)?;
        let ring_value = claim.target.clone();
        fs.polys(LABEL_RING_VALUE, core::slice::from_ref(&ring_value));
        let (iterations, final_witness) = self.prove_rest(
            state.committed.layout,
            self.multilinear_reduction(*vars),
            &state.committed,
            claim,
            &mut fs,
            seed,
        )?;
        Ok((
            value,
            Proof {
                partials: sent,
                batch_rounds,
                batch_eval: Some(at_r),
                mask_eval: None,
                ring_value,
                iterations,
                final_witness,
                blinder_mask_evals: Vec::new(),
                blinder_ring_values: Vec::new(),
            },
        ))
    }

    /// Proves $f(\mathbf x)$ for a committed multilinear polynomial over $\mathbb F_{q^K}$;
    /// returns the value and the proof. `context` and `seed` as for
    /// [`HachiPcs::prove_multilinear`]. The paper's Section 3.1: the weights at $\mathbf x$ give
    /// $Y$, and $f(\mathbf x) = E(Y)$.
    ///
    /// The transcript absorbs the statement with the value, $Y$, then the iterations.
    ///
    /// # Errors
    /// [`Error::Shape`] for a point of the wrong length or a state of another kind; prover
    /// errors. [`Error::Protocol`] for a second opening of a masked commitment.
    pub fn prove_multilinear_ext(
        &self,
        state: &State<K>,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> Result<(Ext<K>, Proof<K>)> {
        self.prove_multilinear_ext_from(Held::Borrowed(state), None, point, context, seed)
    }

    /// [`HachiPcs::prove_multilinear_ext`] on a state that the caller does not need afterwards: the
    /// opening frees the committed message once the first iteration has read it (steps 1 and 2,
    /// [`Witness`]), instead of holding it until the proof is done. The value and the proof are the
    /// same.
    ///
    /// # Errors
    /// As [`HachiPcs::prove_multilinear_ext`].
    pub fn prove_multilinear_ext_owned(
        &self,
        state: State<K>,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> Result<(Ext<K>, Proof<K>)> {
        self.prove_multilinear_ext_from(Held::Owned(state), None, point, context, seed)
    }

    /// [`HachiPcs::prove_multilinear_ext`] on a borrowed or an owned state.
    fn prove_multilinear_ext_from(
        &self,
        held: Held<'_, State<K>>,
        source: Option<OwnedSource<'_>>,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> Result<(Ext<K>, Proof<K>)> {
        let state: &State<K> = match &held {
            Held::Borrowed(s) => s,
            Held::Owned(s) => s,
        };
        let ctx = &self.ctx;
        let field = ctx.field();
        let Input::MultilinearExt { vars } = &state.input else {
            return Err(Error::Shape(
                "this state is not a polynomial over the field".into(),
            ));
        };
        if state.shed.is_some() {
            return Err(Error::Protocol(
                "this state's message was shed: open it with prove_multilinear_ext_rebuilt".into(),
            ));
        }
        // The message as the opening reads it: the state's, or formed from a source.
        let message = source.as_ref().map_or(
            MessageView::Held(&state.committed.message),
            OwnedSource::view,
        );
        if point.len() != *vars {
            return Err(Error::Shape(format!(
                "a point of {vars} coordinates is expected"
            )));
        }
        state.take_opening()?;
        let point: Vec<Ext<K>> = point.iter().map(|x| field.reduce(x)).collect();
        let layout = state.committed.layout;
        let (b, a, v) = self.ring_weights(&layout, &point)?;
        // <a, f_i> over the message of every chunk, once: the claim's value, the masked claim and
        // step 1 of the first iteration (w_i) are formed from them.
        if !layout.masked && a.len() != layout.chunk_len() {
            return Err(Error::Shape(
                "the claim's weights do not cover a chunk".into(),
            ));
        }
        let products = self.chunk_products(&a, message, a.len())?;
        let weighted = |b: &[Poly]| -> Result<Poly> {
            let ring = &ctx.ring;
            let mut acc = ring.zero();
            for (bi, p) in b.iter().zip(&products) {
                acc = ring.add(&acc, &ring.mul(bi, p)?)?;
            }
            Ok(acc)
        };
        let (value, mask_eval, claim, first_w, mut fs) = if layout.masked {
            // The value from the message alone: the masking rows are not part of it.
            let y = weighted(&b)?;
            let value = self.traced(&y, &v)?;
            let mut fs = self.statement(context, 3, *vars, &state.committed.u, &point, &value);
            let (mask_eval, claim, w) =
                self.masked_claim_from(message, &products, b, a, &v, &value, &mut fs)?;
            (value, Some(mask_eval), claim, w, fs)
        } else {
            let target = weighted(&b)?;
            let value = self.traced(&target, &v)?;
            let mut fs = self.statement(context, 3, *vars, &state.committed.u, &point, &value);
            fs.polys(LABEL_RING_VALUE, core::slice::from_ref(&target));
            (value, None, Claim { b, a, target }, products.clone(), fs)
        };
        drop(products);
        let ring_value = claim.target.clone();
        let committed = match (held, source) {
            (Held::Borrowed(s), None) => Witness::Borrowed(&s.committed),
            (Held::Owned(s), None) => Witness::Owned(s.committed),
            (Held::Owned(s), Some(source)) => Witness::Sourced {
                committed: s.committed,
                source: Some(source),
            },
            (Held::Borrowed(_), Some(_)) => {
                return Err(Error::Protocol(
                    "a message from a source opens an owned state".into(),
                ));
            }
        };
        let (iterations, final_witness) = self.prove_rest_with(
            layout,
            self.masks_or_direct(),
            committed,
            claim,
            Some(first_w),
            &mut fs,
            seed,
        )?;
        Ok((
            value,
            Proof {
                partials: Vec::new(),
                batch_rounds: Vec::new(),
                batch_eval: None,
                mask_eval,
                ring_value,
                iterations,
                final_witness,
                blinder_mask_evals: Vec::new(),
                blinder_ring_values: Vec::new(),
            },
        ))
    }

    /// Proves $\langle \mathbf s, \mathbf e \rangle$ for a committed vector $\mathbf s$ and
    /// public weights $\mathbf e$ of the same length; returns the value and the proof. `seed` as
    /// for [`HachiPcs::prove_multilinear`]. With hiding, step 1 is masked as for a polynomial over
    /// the field, with $\mathbf b = (1)$ and $v = 1$ (module documentation, "Hiding").
    ///
    /// # Errors
    /// [`Error::Shape`] for weights of the wrong length or a state of a polynomial; prover
    /// errors. [`Error::Protocol`] for a second opening of a masked commitment.
    pub fn prove_inner_product(
        &self,
        state: &State<K>,
        weights: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> Result<(Ext<K>, Proof<K>)> {
        let ctx = &self.ctx;
        let field = ctx.field();
        let Input::Vector { values } = &state.input else {
            return Err(Error::Shape("this state is not a vector".into()));
        };
        if weights.len() != values.len() {
            return Err(Error::Shape(format!(
                "{} weights are expected",
                values.len()
            )));
        }
        state.take_opening()?;
        let weights: Vec<Ext<K>> = weights.iter().map(|x| field.reduce(x)).collect();
        let value = values
            .iter()
            .zip(&weights)
            .fold(field.zero(), |acc, (v, e)| {
                field.add(&acc, &field.mul(v, e))
            });
        let layout = state.committed.layout;
        let a = self.vector_weights(&layout, &weights)?;
        let b = vec![ctx.ring.monomial(0)];
        let one = ctx.ring.monomial(0);
        // The value from the message alone: a masking row is not part of it.
        let message = vec![state.committed.message[0][..layout.message_len()].to_vec()];
        let y = evaluate_claim(ctx, &b, &a, &message)?;
        if self.traced(&y, &one)? != value {
            return Err(Error::Protocol(
                "the trace check of the inner product fails".into(),
            ));
        }
        let mut fs = self.statement(
            context,
            2,
            values.len(),
            &state.committed.u,
            &weights,
            &value,
        );
        let (mask_eval, claim) = if layout.masked {
            let message = MessageView::Held(&state.committed.message);
            let (mask_eval, claim) =
                self.masked_claim(state, message, b, a, &one, &value, &mut fs)?;
            (Some(mask_eval), claim)
        } else {
            fs.polys(LABEL_RING_VALUE, core::slice::from_ref(&y));
            (None, Claim { b, a, target: y })
        };
        let ring_value = claim.target.clone();
        let (iterations, final_witness) = self.prove_rest(
            layout,
            self.masks_or_direct(),
            &state.committed,
            claim,
            &mut fs,
            seed,
        )?;
        Ok((
            value,
            Proof {
                partials: Vec::new(),
                batch_rounds: Vec::new(),
                batch_eval: None,
                mask_eval,
                ring_value,
                iterations,
                final_witness,
                blinder_mask_evals: Vec::new(),
                blinder_ring_values: Vec::new(),
            },
        ))
    }

    /// Runs the iterations' verifiers and the final step; `reduction` as for
    /// [`HachiPcs::prove_rest`].
    fn verify_rest(
        &self,
        layout: Layout,
        reduction: Reduction,
        commitment: &[Poly],
        claim: Claim,
        proof: &Proof<K>,
        fs: &mut Fs,
    ) -> core::result::Result<(), VerifyError> {
        let ctx = &self.ctx;
        let schedule = self.schedule(layout, reduction)?;
        if proof.iterations.len() != schedule.iterations.len() {
            return Err(VerifyError::Malformed(format!(
                "{} iterations, {} expected",
                proof.iterations.len(),
                schedule.iterations.len()
            )));
        }
        let mut u = commitment.to_vec();
        let mut claim = claim;
        let mut evaluation = None;
        let count = schedule.iterations.len();
        for (index, (params, it)) in schedule
            .iterations
            .iter()
            .zip(&proof.iterations)
            .enumerate()
        {
            match iteration::verify(ctx, index, params, &u, &claim, it, index + 1 == count, fs)? {
                Reduced::Claim(next) => claim = next,
                Reduced::Evaluation { point, value } => evaluation = Some((point, value)),
            }
            u.clone_from(&it.next_commitment);
        }
        let last = schedule.final_layout;
        let opens = |recomputed: &Committed| -> core::result::Result<(), VerifyError> {
            if recomputed.u != u {
                return Err(VerifyError::Check(
                    "the final witness does not open the commitment".into(),
                ));
            }
            Ok(())
        };
        match (&proof.final_witness, evaluation) {
            (FinalWitness::Message(final_message), None) => {
                check_polys(
                    ctx,
                    "final message",
                    final_message,
                    last.chunks() * last.chunk_len(),
                )?;
                if !last.decomposed {
                    for p in final_message {
                        if ctx.ring.inf_norm(p)? > last.beta_s {
                            return Err(VerifyError::Check(
                                "the final witness is not short".into(),
                            ));
                        }
                    }
                }
                let message: Vec<Vec<Poly>> = final_message
                    .chunks_exact(last.chunk_len())
                    .map(<[Poly]>::to_vec)
                    .collect();
                opens(&commit(ctx, count, last, message.clone(), Vec::new())?)?;
                if evaluate_claim(ctx, &claim.b, &claim.a, &message)? != claim.target {
                    return Err(VerifyError::Check(
                        "the final witness does not satisfy the claim".into(),
                    ));
                }
            }
            // Step 3: every value lies in the digit set, the commitment recomputed from the table
            // is the last one received, and the table's value at the point is y'.
            (FinalWitness::Digits(table), Some((point, value))) => {
                let entries = last.chunks() * last.chunk_len() * ctx.d_over_k();
                if table.len() != entries {
                    return Err(VerifyError::Malformed(format!(
                        "a final witness of {} digits, {entries} expected",
                        table.len()
                    )));
                }
                let (zq, (lo, hi)) = (ctx.ring.zq(), digit_range(ctx.choices.b));
                if (0..table.len()).any(|i| {
                    let v = i128::from(zq.to_centered(table[i]));
                    v < lo || v > hi
                }) {
                    return Err(VerifyError::Check(
                        "the final witness is not a table of digits".into(),
                    ));
                }
                opens(&commit_table(ctx, count, last, table)?)?;
                if table_eval(ctx.field(), table, &point)? != value {
                    return Err(VerifyError::Check(
                        "the final witness does not have the last iteration's value".into(),
                    ));
                }
            }
            _ => {
                return Err(VerifyError::Malformed(
                    "the final witness does not fit the schedule".into(),
                ));
            }
        }
        Ok(())
    }

    /// Checks the commitment's shape.
    fn check_commitment(&self, commitment: &Commitment) -> core::result::Result<(), VerifyError> {
        check_polys(&self.ctx, "commitment", &commitment.u, self.ctx.choices.n_b)
    }

    /// Verifies a multilinear opening: $f(\mathbf x) = $ `value` for the polynomial in `vars`
    /// variables committed in `commitment`.
    ///
    /// # Errors
    /// [`VerifyError`] for a malformed or rejected proof.
    pub fn verify_multilinear(
        &self,
        commitment: &Commitment,
        vars: usize,
        point: &[Ext<K>],
        value: &Ext<K>,
        proof: &Proof<K>,
        context: &[u8; 32],
    ) -> core::result::Result<(), VerifyError> {
        let ctx = &self.ctx;
        self.check_commitment(commitment)?;
        if point.len() != vars {
            return Err(VerifyError::Malformed(format!(
                "a point of {vars} coordinates is expected"
            )));
        }
        check_exts(ctx, "point", point, vars)?;
        check_exts(ctx, "value", core::slice::from_ref(value), 1)?;
        if !proof.blinder_mask_evals.is_empty() || !proof.blinder_ring_values.is_empty() {
            return Err(VerifyError::Malformed(
                "only a merged opening carries the values of two vectors".into(),
            ));
        }
        check_exts(ctx, "partial evaluations", &proof.partials, K - 1)?;
        if ctx.hiding().is_some() || proof.mask_eval.is_some() {
            return Err(VerifyError::Malformed(
                "polynomials over Z_q are not opened with hiding".into(),
            ));
        }
        // The layout exists only for vars >= alpha > kappa.
        let layout = self.multilinear_layout(vars)?;
        let x = &point[ctx.kappa()..];
        if proof.batch_rounds.len() != x.len() {
            return Err(VerifyError::Malformed(format!(
                "{} rounds of the batching sumcheck, {} expected",
                proof.batch_rounds.len(),
                x.len()
            )));
        }
        for round in &proof.batch_rounds {
            check_exts(ctx, "batching sumcheck", round, 2)?;
        }
        let Some(at_r) = &proof.batch_eval else {
            return Err(VerifyError::Malformed(
                "the batching sumcheck's final value is missing".into(),
            ));
        };
        check_exts(
            ctx,
            "batching sumcheck's final value",
            core::slice::from_ref(at_r),
            1,
        )?;
        check_polys(
            ctx,
            "ring value",
            core::slice::from_ref(&proof.ring_value),
            1,
        )?;
        let mono = ctx.sub.monomials(&point[..ctx.kappa()])?;
        let ys = partials::complete(ctx.field(), &mono, value, &proof.partials)?;
        let mut fs = self.statement(context, 1, vars, &commitment.u, point, value);
        fs.exts(LABEL_PARTIALS, &proof.partials);
        let r = partials::verify(
            ctx.field(),
            x,
            &ys,
            &proof.batch_rounds,
            at_r,
            &partials::INPUT_LABELS,
            &mut fs,
        )?;
        let (b, a, v) = self.ring_weights(&layout, &r)?;
        if self.traced(&proof.ring_value, &v)? != *at_r {
            return Err(VerifyError::Check(
                "the trace of the input reduction".into(),
            ));
        }
        fs.polys(LABEL_RING_VALUE, core::slice::from_ref(&proof.ring_value));
        let claim = Claim {
            b,
            a,
            target: proof.ring_value.clone(),
        };
        self.verify_rest(
            layout,
            self.multilinear_reduction(vars),
            &commitment.u,
            claim,
            proof,
            &mut fs,
        )
    }

    /// Verifies an opening of a multilinear polynomial over $\mathbb F_{q^K}$: $f(\mathbf x) = $
    /// `value` for the polynomial in `vars` variables committed in `commitment`.
    ///
    /// # Errors
    /// [`VerifyError`] for a malformed or rejected proof.
    pub fn verify_multilinear_ext(
        &self,
        commitment: &Commitment,
        vars: usize,
        point: &[Ext<K>],
        value: &Ext<K>,
        proof: &Proof<K>,
        context: &[u8; 32],
    ) -> core::result::Result<(), VerifyError> {
        let ctx = &self.ctx;
        self.check_commitment(commitment)?;
        if point.len() != vars {
            return Err(VerifyError::Malformed(format!(
                "a point of {vars} coordinates is expected"
            )));
        }
        check_exts(ctx, "point", point, vars)?;
        if !proof.blinder_mask_evals.is_empty() || !proof.blinder_ring_values.is_empty() {
            return Err(VerifyError::Malformed(
                "only a merged opening carries the values of two vectors".into(),
            ));
        }
        check_exts(ctx, "value", core::slice::from_ref(value), 1)?;
        if !proof.partials.is_empty()
            || !proof.batch_rounds.is_empty()
            || proof.batch_eval.is_some()
        {
            return Err(VerifyError::Malformed(
                "a polynomial over the field has no partial evaluations and no batching sumcheck"
                    .into(),
            ));
        }
        check_polys(
            ctx,
            "ring value",
            core::slice::from_ref(&proof.ring_value),
            1,
        )?;
        let layout = self.multilinear_ext_layout(vars)?;
        let (b, a, v) = self.ring_weights(&layout, point)?;
        let mut fs = self.statement(context, 3, vars, &commitment.u, point, value);
        let claim = if layout.masked {
            self.verify_masked_claim(proof, b, a, &v, value, &mut fs)?
        } else {
            if proof.mask_eval.is_some() {
                return Err(VerifyError::Malformed(
                    "only a masked opening carries the masks' value".into(),
                ));
            }
            if self.traced(&proof.ring_value, &v)? != *value {
                return Err(VerifyError::Check(
                    "the trace of the input reduction".into(),
                ));
            }
            fs.polys(LABEL_RING_VALUE, core::slice::from_ref(&proof.ring_value));
            Claim {
                b,
                a,
                target: proof.ring_value.clone(),
            }
        };
        self.verify_rest(
            layout,
            self.masks_or_direct(),
            &commitment.u,
            claim,
            proof,
            &mut fs,
        )
    }

    /// Verifies an inner-product opening: $\langle \mathbf s, \mathbf e \rangle = $ `value` for
    /// the vector of length `len` committed in `commitment`.
    ///
    /// # Errors
    /// [`VerifyError`] for a malformed or rejected proof.
    pub fn verify_inner_product(
        &self,
        commitment: &Commitment,
        len: usize,
        weights: &[Ext<K>],
        value: &Ext<K>,
        proof: &Proof<K>,
        context: &[u8; 32],
    ) -> core::result::Result<(), VerifyError> {
        let ctx = &self.ctx;
        self.check_commitment(commitment)?;
        check_exts(ctx, "weights", weights, len)?;
        if !proof.blinder_mask_evals.is_empty() || !proof.blinder_ring_values.is_empty() {
            return Err(VerifyError::Malformed(
                "only a merged opening carries the values of two vectors".into(),
            ));
        }
        check_exts(ctx, "value", core::slice::from_ref(value), 1)?;
        if !proof.partials.is_empty()
            || !proof.batch_rounds.is_empty()
            || proof.batch_eval.is_some()
        {
            return Err(VerifyError::Malformed(
                "an inner product has no partial evaluations and no batching sumcheck".into(),
            ));
        }
        check_polys(
            ctx,
            "ring value",
            core::slice::from_ref(&proof.ring_value),
            1,
        )?;
        let layout = self.vector_layout(len)?;
        let one = ctx.ring.monomial(0);
        let a = self.vector_weights(&layout, weights)?;
        let mut fs = self.statement(context, 2, len, &commitment.u, weights, value);
        let claim = if layout.masked {
            self.verify_masked_claim(proof, vec![one.clone()], a, &one, value, &mut fs)?
        } else {
            if proof.mask_eval.is_some() {
                return Err(VerifyError::Malformed(
                    "only a masked opening carries the masks' value".into(),
                ));
            }
            if self.traced(&proof.ring_value, &one)? != *value {
                return Err(VerifyError::Check("the trace of the inner product".into()));
            }
            fs.polys(LABEL_RING_VALUE, core::slice::from_ref(&proof.ring_value));
            Claim {
                b: vec![one],
                a,
                target: proof.ring_value.clone(),
            }
        };
        self.verify_rest(
            layout,
            self.masks_or_direct(),
            &commitment.u,
            claim,
            proof,
            &mut fs,
        )
    }

    // --------------------------------------------------------------- merged commitments ----

    /// The layout of a merged commitment ([`HachiPcs::commit_merged`]) to a multilinear polynomial
    /// over $\mathbb F_{q^K}$ in `vars` variables and two vectors of `lens` elements: the split of
    /// [`HachiPcs::multilinear_ext_layout`], made again with the extra rows, and per chunk the
    /// blocks of $d/K$ entries of each vector, each followed by its masking row
    /// ([`Layout::extra`]), before the chunk's own masking row.
    ///
    /// # Errors
    /// [`Error::Parameter`] without hiding (a merged commitment exists only with masks) or with
    /// fewer than $\alpha - \kappa$ variables, [`Error::Shape`] for an empty vector.
    pub fn merged_layout(&self, vars: usize, lens: [usize; 2]) -> Result<Layout> {
        if self.ctx.hiding().is_none() {
            return Err(Error::Parameter(
                "a merged commitment needs hiding choices".into(),
            ));
        }
        let n = (vars + self.ctx.kappa())
            .checked_sub(self.ctx.alpha())
            .ok_or_else(|| {
                Error::Parameter(format!(
                    "a polynomial over the field needs at least alpha - kappa = {} variables",
                    self.ctx.alpha() - self.ctx.kappa()
                ))
            })?;
        let mut extra = 0;
        for len in lens {
            if len == 0 {
                return Err(Error::Shape("an empty vector cannot be committed".into()));
            }
            extra += len.div_ceil(self.ctx.d_over_k()) + 1;
        }
        best_layout(
            &self.ctx.choices,
            self.ctx.q,
            n,
            true,
            max_digit(self.ctx.choices.b),
            true,
            extra,
        )
    }

    /// The schedule of a merged opening (module documentation, "Schedule").
    ///
    /// # Errors
    /// As [`HachiPcs::merged_layout`], and [`Error::Parameter`] if the fixed number of iterations
    /// cannot be parametrised.
    pub fn merged_schedule(&self, vars: usize, lens: [usize; 2]) -> Result<Schedule> {
        self.schedule(self.merged_layout(vars, lens)?, Reduction::Merged)
    }

    /// The number of words of [`HachiPcs::proof_to_words`] for a merged opening. It depends only
    /// on `vars`, `lens` and the parameters.
    ///
    /// # Errors
    /// As [`HachiPcs::merged_schedule`].
    pub fn merged_proof_words(&self, vars: usize, lens: [usize; 2]) -> Result<usize> {
        self.proof_words(&self.merged_schedule(vars, lens)?, Reduction::Merged)
    }

    /// $\psi$ of the blocks of $d/K$ entries of `values`, the last one zero-padded.
    fn blocks(&self, values: &[Ext<K>]) -> Result<Vec<Poly>> {
        let per = self.ctx.d_over_k();
        let mut padded = values.to_vec();
        padded.resize(values.len().div_ceil(per) * per, self.ctx.field().zero());
        Ok(padded
            .chunks_exact(per)
            .map(|block| self.ctx.sub.psi(block))
            .collect::<hachi_co_math::Result<Vec<_>>>()?)
    }

    /// The weights $\sigma_{-1}(\psi(\cdot))$ of the blocks of `weights` ([`HachiPcs::blocks`]).
    fn block_weights(&self, weights: &[Ext<K>]) -> Result<Vec<Poly>> {
        self.blocks(weights)?
            .iter()
            .map(|p| Ok(self.ctx.ring.automorphism(p, -1)?))
            .collect()
    }

    /// The masking rows of the two vectors of a merged commitment, from the stream
    /// `(seed, "hachi/commit/mask", 1)`: uniform in $\mathcal R_q$.
    fn vector_masking_rows(&self, seed: &[u8; 32]) -> Result<Vec<Poly>> {
        let d = self.ctx.d();
        let mut prg = ShakePrg::new(seed, DOMAIN_COMMIT_MASK, 1);
        let residues = uniform_mod_q(&mut prg, self.ctx.q, 2 * d)?;
        Ok(residues.chunks_exact(d).map(|c| Poly(c.to_vec())).collect())
    }

    /// Commits, in one commitment, to the multilinear polynomial over $\mathbb F_{q^K}$ with
    /// monomial coefficients `coeffs` (as [`HachiPcs::commit_multilinear_ext`]) and to the two
    /// vectors `vectors` (as [`HachiPcs::commit_vector`]): Hachi-Co's share polynomial and its two
    /// blinders. The share fills the message rows of every chunk, the vectors the extra rows of
    /// chunk 0, each followed by its masking row, and the other chunks hold zeros there
    /// ([`HachiPcs::merged_layout`]). Randomness and masking rows come from the secret `seed`.
    ///
    /// # Errors
    /// As [`HachiPcs::merged_layout`], [`Error::Shape`] unless the length of `coeffs` is a power of
    /// two of at least $d/K$, and arithmetic errors.
    pub fn commit_merged(
        &self,
        coeffs: &[Ext<K>],
        vectors: [&[Ext<K>]; 2],
        seed: &[u8; 32],
    ) -> Result<(Commitment, State<K>)> {
        let ctx = &self.ctx;
        if !coeffs.len().is_power_of_two() || coeffs.len() < ctx.d_over_k() {
            return Err(Error::Shape(format!(
                "2^l coefficients with l >= alpha - kappa = {} are expected",
                ctx.alpha() - ctx.kappa()
            )));
        }
        let vars = coeffs.len().trailing_zeros() as usize;
        let layout = self.merged_layout(vars, [vectors[0].len(), vectors[1].len()])?;
        let g = ctx.d_over_k();
        let mut block = Vec::with_capacity(g);
        let mut message = coeffs
            .chunks_exact(g * layout.message_len())
            .map(|chunk| {
                chunk
                    .chunks_exact(g)
                    .map(|c| {
                        block.clear();
                        block.extend(c.iter().map(|x| ctx.field().reduce(x)));
                        ctx.sub.psi(&block)
                    })
                    .collect::<hachi_co_math::Result<Vec<Poly>>>()
            })
            .collect::<hachi_co_math::Result<Vec<_>>>()?;
        let values: [Vec<Ext<K>>; 2] =
            vectors.map(|v| v.iter().map(|x| ctx.field().reduce(x)).collect());
        let mut extra = Vec::with_capacity(layout.extra);
        for (v, row) in values.iter().zip(self.vector_masking_rows(seed)?) {
            extra.extend(self.blocks(v)?);
            extra.push(row);
        }
        let zeros = vec![ctx.ring.zero(); layout.extra];
        for (i, chunk) in message.iter_mut().enumerate() {
            chunk.extend_from_slice(if i == 0 { &extra } else { &zeros });
        }
        for (chunk, row) in message.iter_mut().zip(self.masking_rows(&layout, seed)?) {
            chunk.push(row);
        }
        let r = commitment_randomness(ctx, &layout, seed, DOMAIN_COMMIT_R)?;
        let committed = commit(ctx, 0, layout, message, r)?;
        Ok((
            Commitment {
                u: committed.u.clone(),
            },
            State::new(Input::Merged { vars, values }, committed),
        ))
    }

    /// The transcript of a merged opening after its statement: the statement of the share's
    /// opening (kind 4), then the lengths and weights of the two vectors and their values.
    fn merged_statement(
        &self,
        context: &[u8; 32],
        vars: usize,
        commitment: &[Poly],
        point: &[Ext<K>],
        weights: [&[Ext<K>]; 2],
        values: &[Ext<K>; 3],
    ) -> Fs {
        let mut fs = self.statement(context, 4, vars, commitment, point, &values[0]);
        fs.words(
            LABEL_MERGED_SHAPE,
            &[weights[0].len() as u64, weights[1].len() as u64],
        );
        fs.exts(LABEL_MERGED_WEIGHTS, weights[0]);
        fs.exts(LABEL_MERGED_WEIGHTS, weights[1]);
        fs.exts(LABEL_MERGED_VALUES, &values[1..]);
        fs
    }

    /// The first claim of a merged opening: the weights $\mathbf a$ of the share, then $\lambda$
    /// times the block weights of the first vector and $\lambda x^*$ for its masking row,
    /// $\lambda^2$ times those of the second and $\lambda^2 x^*$, and $x^*$ for the chunk's masking
    /// row, with the target $Y^*_w + \lambda Y^*_0 + \lambda^2 Y^*_1$. Chunk 0 has $b_1 = 1$, and the
    /// other chunks hold zeros in the extra rows, so an honest witness meets it.
    #[allow(clippy::too_many_arguments)]
    fn merged_claim(
        &self,
        layout: &Layout,
        b: Vec<Poly>,
        a: Vec<Poly>,
        vector_weights: &[Vec<Poly>; 2],
        coin: &Ext<K>,
        lambda: &Ext<K>,
        targets: &[Poly; 3],
    ) -> Result<Claim> {
        let (field, ring, sub) = (self.ctx.field(), &self.ctx.ring, &self.ctx.sub);
        let mut weights = a;
        let mut target = targets[0].clone();
        let mut power = *lambda;
        for (w, t) in vector_weights.iter().zip(&targets[1..]) {
            let p = sub.embed(&power);
            for x in w {
                weights.push(ring.mul(&p, x)?);
            }
            weights.push(sub.embed(&field.mul(&power, coin)));
            target = ring.add(&target, &ring.mul(&p, t)?)?;
            power = field.mul(&power, lambda);
        }
        weights.push(sub.embed(coin));
        if weights.len() != layout.chunk_len() {
            return Err(Error::Shape(
                "the merged weights do not fit the layout".into(),
            ));
        }
        Ok(Claim {
            b,
            a: weights,
            target,
        })
    }

    /// Proves, for a merged commitment, the share's value at `point` and the two vectors' inner
    /// products with `weights`, and returns the three values and one proof. `seed` as for
    /// [`HachiPcs::prove_multilinear`].
    ///
    /// The prover sends the three values with the statement, then the three masks' values
    /// $y_{\mathfrak m}$ (the share's for $v$, the vectors' for $v = 1$), receives $x^*$, and sends
    /// $Y^*_w, Y^*_0, Y^*_1$, one per committed object, which the verifier checks by three trace
    /// checks. Only then is $\lambda \ne 0$ drawn, and the iterations prove one claim, with the
    /// weights of the share, $\lambda$ times those of the first vector and $\lambda^2$ times those of
    /// the second, and the target $Y^*_w + \lambda Y^*_0 + \lambda^2 Y^*_1$. A false value with a ring
    /// value fitted to its trace check makes that claim false unless $\lambda$ is one of at most 2
    /// roots.
    ///
    /// # Errors
    /// [`Error::Shape`] for a state of another kind, a point or weights of the wrong length, and
    /// prover errors. [`Error::Protocol`] for a second opening.
    pub fn prove_merged(
        &self,
        state: &State<K>,
        point: &[Ext<K>],
        weights: [&[Ext<K>]; 2],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> Result<([Ext<K>; 3], Proof<K>)> {
        let ctx = &self.ctx;
        let (field, ring) = (ctx.field(), &ctx.ring);
        let Input::Merged { vars, values } = &state.input else {
            return Err(Error::Shape("this state is not a merged commitment".into()));
        };
        if point.len() != *vars {
            return Err(Error::Shape(format!(
                "a point of {vars} coordinates is expected"
            )));
        }
        for (v, e) in values.iter().zip(weights) {
            if e.len() != v.len() {
                return Err(Error::Shape(format!("{} weights are expected", v.len())));
            }
        }
        state.take_opening()?;
        let point: Vec<Ext<K>> = point.iter().map(|x| field.reduce(x)).collect();
        let weights: [Vec<Ext<K>>; 2] =
            weights.map(|e| e.iter().map(|x| field.reduce(x)).collect());
        let layout = state.committed.layout;
        let chunks = &state.committed.message;
        let (b, a, v) = self.ring_weights(&layout, &point)?;
        let one = ring.monomial(0);
        let len = layout.message_len();
        // The share: its ring value from the message rows alone, and its masks.
        let share: Vec<Vec<Poly>> = chunks.iter().map(|c| c[..len].to_vec()).collect();
        let y_share = evaluate_claim(ctx, &b, &a, &share)?;
        let mut y_mask = ring.zero();
        for (bi, chunk) in b.iter().zip(chunks) {
            let row = chunk
                .last()
                .ok_or_else(|| Error::Shape("an empty chunk".into()))?;
            y_mask = ring.add(&y_mask, &ring.mul(bi, row)?)?;
        }
        // The vectors: the extra rows of chunk 0, each followed by its masking row.
        let mut at = len;
        let mut ys = Vec::with_capacity(2);
        let mut rows = Vec::with_capacity(2);
        let mut vector_weights = Vec::with_capacity(2);
        for (vals, e) in values.iter().zip(&weights) {
            let n = vals.len().div_ceil(ctx.d_over_k());
            let w = self.block_weights(e)?;
            ys.push(ring.inner(&w, &chunks[0][at..at + n])?);
            rows.push(chunks[0][at + n].clone());
            vector_weights.push(w);
            at += n + 1;
        }
        let direct = |vals: &[Ext<K>], e: &[Ext<K>]| {
            vals.iter().zip(e).fold(field.zero(), |acc, (x, y)| {
                field.add(&acc, &field.mul(x, y))
            })
        };
        let out = [
            self.traced(&y_share, &v)?,
            direct(&values[0], &weights[0]),
            direct(&values[1], &weights[1]),
        ];
        for (y, value) in ys.iter().zip(&out[1..]) {
            if self.traced(y, &one)? != *value {
                return Err(Error::Protocol(
                    "the trace check of a merged vector fails".into(),
                ));
            }
        }
        let mut fs = self.merged_statement(
            context,
            *vars,
            &state.committed.u,
            &point,
            [&weights[0], &weights[1]],
            &out,
        );
        let mask_evals = [
            self.traced(&y_mask, &v)?,
            self.traced(&rows[0], &one)?,
            self.traced(&rows[1], &one)?,
        ];
        fs.exts(LABEL_MASK_EVAL, &mask_evals);
        let coin = self.mask_coin(&mut fs)?;
        let x = ctx.sub.embed(&coin);
        let targets = [
            ring.add(&y_share, &ring.mul(&x, &y_mask)?)?,
            ring.add(&ys[0], &ring.mul(&x, &rows[0])?)?,
            ring.add(&ys[1], &ring.mul(&x, &rows[1])?)?,
        ];
        for ((t, vv), (value, m)) in targets
            .iter()
            .zip([&v, &one, &one])
            .zip(out.iter().zip(&mask_evals))
        {
            if self.traced(t, vv)? != field.add(value, &field.mul(&coin, m)) {
                return Err(Error::Protocol(
                    "the trace check of the merged claim fails".into(),
                ));
            }
        }
        fs.polys(LABEL_RING_VALUE, &targets);
        let lambda = self.nonzero_coin(&mut fs, LABEL_MERGED_COIN)?;
        let vector_weights: [Vec<Poly>; 2] = vector_weights
            .try_into()
            .map_err(|_| Error::Shape("two vectors are expected".into()))?;
        let claim = self.merged_claim(&layout, b, a, &vector_weights, &coin, &lambda, &targets)?;
        let (iterations, final_witness) = self.prove_rest(
            layout,
            Reduction::Merged,
            &state.committed,
            claim,
            &mut fs,
            seed,
        )?;
        let [ring_value, r0, r1] = targets;
        Ok((
            out,
            Proof {
                partials: Vec::new(),
                batch_rounds: Vec::new(),
                batch_eval: None,
                mask_eval: Some(mask_evals[0]),
                ring_value,
                iterations,
                final_witness,
                blinder_mask_evals: vec![mask_evals[1], mask_evals[2]],
                blinder_ring_values: vec![r0, r1],
            },
        ))
    }

    /// Verifies a merged opening ([`HachiPcs::prove_merged`]): the polynomial in `vars` variables
    /// takes `values[0]` at `point`, and the two vectors, of the lengths of `weights`, have the
    /// inner products `values[1]` and `values[2]` with them.
    ///
    /// # Errors
    /// [`VerifyError`] for a malformed or rejected proof.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_merged(
        &self,
        commitment: &Commitment,
        vars: usize,
        point: &[Ext<K>],
        weights: [&[Ext<K>]; 2],
        values: &[Ext<K>; 3],
        proof: &Proof<K>,
        context: &[u8; 32],
    ) -> core::result::Result<(), VerifyError> {
        let ctx = &self.ctx;
        let field = ctx.field();
        self.check_commitment(commitment)?;
        if point.len() != vars {
            return Err(VerifyError::Malformed(format!(
                "a point of {vars} coordinates is expected"
            )));
        }
        check_exts(ctx, "point", point, vars)?;
        check_exts(ctx, "values", values, 3)?;
        for e in weights {
            check_exts(ctx, "weights", e, e.len())?;
        }
        if !proof.partials.is_empty()
            || !proof.batch_rounds.is_empty()
            || proof.batch_eval.is_some()
        {
            return Err(VerifyError::Malformed(
                "a merged opening has no partial evaluations and no batching sumcheck".into(),
            ));
        }
        let (Some(mask_eval), [m0, m1], [r0, r1]) = (
            &proof.mask_eval,
            proof.blinder_mask_evals.as_slice(),
            proof.blinder_ring_values.as_slice(),
        ) else {
            return Err(VerifyError::Malformed(
                "a merged opening carries three masks' values and three ring values".into(),
            ));
        };
        let mask_evals = [*mask_eval, *m0, *m1];
        check_exts(ctx, "masks' values", &mask_evals, 3)?;
        let targets = [proof.ring_value.clone(), r0.clone(), r1.clone()];
        check_polys(ctx, "ring values", &targets, 3)?;
        let layout = self.merged_layout(vars, [weights[0].len(), weights[1].len()])?;
        let (b, a, v) = self.ring_weights(&layout, point)?;
        let one = ctx.ring.monomial(0);
        let vector_weights = [
            self.block_weights(weights[0])?,
            self.block_weights(weights[1])?,
        ];
        let mut fs = self.merged_statement(context, vars, &commitment.u, point, weights, values);
        fs.exts(LABEL_MASK_EVAL, &mask_evals);
        let coin = self.mask_coin(&mut fs)?;
        for ((t, vv), (value, m)) in targets
            .iter()
            .zip([&v, &one, &one])
            .zip(values.iter().zip(&mask_evals))
        {
            if self.traced(t, vv)? != field.add(value, &field.mul(&coin, m)) {
                return Err(VerifyError::Check("the trace of the merged claim".into()));
            }
        }
        fs.polys(LABEL_RING_VALUE, &targets);
        let lambda = self.nonzero_coin(&mut fs, LABEL_MERGED_COIN)?;
        let claim = self.merged_claim(&layout, b, a, &vector_weights, &coin, &lambda, &targets)?;
        self.verify_rest(
            layout,
            Reduction::Merged,
            &commitment.u,
            claim,
            proof,
            &mut fs,
        )
    }
}

// ------------------------------------------------------------------------------- encodings ----

/// Fields packed least significant bit first into 64-bit words (module documentation,
/// "Encodings").
#[derive(Default)]
struct BitWriter {
    words: Vec<u64>,
    bits: usize,
}

impl BitWriter {
    /// Appends the low `width` bits of `value` (`width` $\le 64$; 0 appends nothing).
    fn push(&mut self, value: u64, width: usize) {
        if width == 0 {
            return;
        }
        let value = if width >= 64 {
            value
        } else {
            value & ((1u64 << width) - 1)
        };
        let at = self.bits % 64;
        match self.words.last_mut() {
            Some(last) if at != 0 => {
                *last |= value << at;
                if at + width > 64 {
                    self.words.push(value >> (64 - at));
                }
            }
            _ => self.words.push(value),
        }
        self.bits += width;
    }
}

/// A reader of the fields of [`BitWriter`] that never panics.
struct BitReader<'a> {
    words: &'a [u64],
    bit: usize,
}

impl BitReader<'_> {
    /// The next field of `width` bits (`width` $\le 64$).
    fn take(&mut self, width: usize) -> Result<u64> {
        if width == 0 {
            return Ok(0);
        }
        let end = self
            .bit
            .checked_add(width)
            .filter(|end| *end <= self.words.len().saturating_mul(64))
            .ok_or_else(|| Error::Shape("the encoding ends early".into()))?;
        let (i, at) = (self.bit / 64, self.bit % 64);
        let mut value = self.words.get(i).copied().unwrap_or(0) >> at;
        if at + width > 64 {
            value |= self.words.get(i + 1).copied().unwrap_or(0) << (64 - at);
        }
        self.bit = end;
        Ok(if width >= 64 {
            value
        } else {
            value & ((1u64 << width) - 1)
        })
    }

    /// Checks that the fields fill the words exactly, up to zero padding in the last word.
    fn finish(&self) -> Result<()> {
        if self.bit.div_ceil(64) != self.words.len() {
            return Err(Error::Shape("trailing words after the encoding".into()));
        }
        let at = self.bit % 64;
        if at != 0 && self.words.last().is_some_and(|w| w >> at != 0) {
            return Err(Error::Shape("non-zero padding bits".into()));
        }
        Ok(())
    }
}

impl<const K: usize> HachiPcs<K> {
    /// Appends the coefficients of `polys` as residues.
    fn put_polys(&self, w: &mut BitWriter, polys: &[Poly]) {
        for x in polys.iter().flat_map(|p| &p.0) {
            w.push(*x, self.residue_bits());
        }
    }

    /// Appends the coordinates of `exts` as residues.
    fn put_exts(&self, w: &mut BitWriter, exts: &[Ext<K>]) {
        for x in exts.iter().flat_map(|e| e.0) {
            w.push(x, self.residue_bits());
        }
    }

    /// The next residue.
    ///
    /// # Errors
    /// [`Error::Shape`] if the encoding ends or the residue is not below $q$.
    fn take_residue(&self, r: &mut BitReader<'_>) -> Result<u64> {
        let x = r.take(self.residue_bits())?;
        if x >= self.ctx.q {
            return Err(Error::Shape("a residue is not below q".into()));
        }
        Ok(x)
    }

    /// The next ring element, as $d$ residues.
    fn take_poly(&self, r: &mut BitReader<'_>) -> Result<Poly> {
        (0..self.ctx.d())
            .map(|_| self.take_residue(r))
            .collect::<Result<Vec<_>>>()
            .map(Poly)
    }

    /// The next `count` ring elements.
    fn take_polys(&self, r: &mut BitReader<'_>, count: usize) -> Result<Vec<Poly>> {
        (0..count).map(|_| self.take_poly(r)).collect()
    }

    /// The next field element, as $K$ residues.
    fn take_ext(&self, r: &mut BitReader<'_>) -> Result<Ext<K>> {
        let mut e = Ext([0u64; K]);
        for x in &mut e.0 {
            *x = self.take_residue(r)?;
        }
        Ok(e)
    }

    /// The next `count` field elements.
    fn take_exts(&self, r: &mut BitReader<'_>, count: usize) -> Result<Vec<Ext<K>>> {
        (0..count).map(|_| self.take_ext(r)).collect()
    }

    /// The final witness after the iterations: `count` digits, each an index into the digit set.
    ///
    /// # Errors
    /// [`Error::Shape`] if the encoding ends or an index is outside the digit set.
    fn take_digit_table(&self, r: &mut BitReader<'_>, count: usize) -> Result<DigitTable> {
        let b = self.ctx.choices.b;
        let (lo, _) = digit_range(b);
        let values = (0..count)
            .map(|_| {
                let index = r.take(self.digit_bits())?;
                if index >= b {
                    return Err(Error::Shape("a digit index outside the digit set".into()));
                }
                i8::try_from(lo + i128::from(index))
                    .map_err(|_| Error::Shape("a digit beyond a byte".into()))
            })
            .collect::<Result<Vec<i8>>>()?;
        let mut table = DigitTable::zeros(self.ctx.ring.zq(), count);
        table.set_values(0, &values)?;
        Ok(table)
    }

    /// The encoding of a commitment: `[VERSION]`, then the $n_B d$ coefficients of
    /// $\mathbf u$ packed (module documentation, "Encodings"); [`HachiPcs::commitment_words`]
    /// words.
    pub fn commitment_to_words(&self, com: &Commitment) -> Vec<u64> {
        let mut w = BitWriter::default();
        self.put_polys(&mut w, &com.u);
        let mut out = vec![VERSION];
        out.extend(w.words);
        out
    }

    /// The inverse of [`HachiPcs::commitment_to_words`].
    ///
    /// # Errors
    /// [`Error::Shape`] for any input that [`HachiPcs::commitment_to_words`] does not produce
    /// from a commitment of $n_B$ ring elements: another version or length, a residue not below
    /// $q$, or non-zero padding.
    pub fn commitment_from_words(&self, words: &[u64]) -> Result<Commitment> {
        let Some((&version, stream)) = words.split_first() else {
            return Err(Error::Shape("the encoding ends early".into()));
        };
        if version != VERSION {
            return Err(Error::Shape("unknown commitment version".into()));
        }
        if self.commitment_words() != words.len() {
            return Err(Error::Shape(format!(
                "a commitment has {} words",
                self.commitment_words()
            )));
        }
        let mut r = BitReader {
            words: stream,
            bit: 0,
        };
        let u = self.take_polys(&mut r, self.ctx.choices.n_b)?;
        r.finish()?;
        Ok(Commitment { u })
    }

    /// The canonical encoding of a proof made by this scheme's prover (module documentation,
    /// "Encodings"). For any other value it returns some words without panicking, but the
    /// decoder need not give the value back: values are cut to their fields, and a final witness
    /// that is not a table of digits is not kept.
    pub fn proof_to_words(&self, proof: &Proof<K>) -> Vec<u64> {
        let mut out = vec![
            VERSION,
            proof.partials.len() as u64,
            proof.batch_rounds.len() as u64,
            proof.iterations.len() as u64,
        ];
        out.extend(proof.iterations.iter().map(|it| it.rounds.len() as u64));
        out.push(match &proof.final_witness {
            FinalWitness::Message(message) => message.len(),
            FinalWitness::Digits(table) => table.len(),
        } as u64);
        if self.ctx.hiding().is_some() {
            let merged = !proof.blinder_ring_values.is_empty();
            out.push(u64::from(proof.mask_eval.is_some()) | (u64::from(merged) << 1));
        }
        let mut w = BitWriter::default();
        self.put_exts(&mut w, &proof.partials);
        for round in &proof.batch_rounds {
            self.put_exts(&mut w, round);
        }
        if let Some(at_r) = &proof.batch_eval {
            self.put_exts(&mut w, core::slice::from_ref(at_r));
        }
        if let Some(y_mask) = &proof.mask_eval {
            self.put_exts(&mut w, core::slice::from_ref(y_mask));
        }
        self.put_exts(&mut w, &proof.blinder_mask_evals);
        self.put_polys(&mut w, core::slice::from_ref(&proof.ring_value));
        self.put_polys(&mut w, &proof.blinder_ring_values);
        for it in &proof.iterations {
            self.put_polys(&mut w, &it.v);
            self.put_polys(&mut w, &it.next_commitment);
            for round in &it.rounds {
                self.put_exts(&mut w, &round.h0);
                self.put_exts(&mut w, &round.h_alpha);
            }
            self.put_exts(&mut w, core::slice::from_ref(&it.final_eval));
            if let Some(target) = &it.next_target {
                self.put_polys(&mut w, core::slice::from_ref(target));
            }
        }
        match &proof.final_witness {
            FinalWitness::Message(message) => self.put_polys(&mut w, message),
            FinalWitness::Digits(table) => {
                let zq = self.ctx.ring.zq();
                let (lo, _) = digit_range(self.ctx.choices.b);
                for i in 0..table.len() {
                    let index = i128::from(zq.to_centered(table[i])) - lo;
                    w.push(index as u64, self.digit_bits());
                }
            }
        }
        out.extend(w.words);
        out
    }

    /// The inverse of [`HachiPcs::proof_to_words`]. The counts of the header (batching rounds,
    /// iterations, rounds, final ring elements) are checked by the verifier, against $\ell -
    /// \kappa$ and the schedule.
    ///
    /// # Errors
    /// [`Error::Shape`] for any input that [`HachiPcs::proof_to_words`] does not produce from a
    /// proof of this scheme's shape: another version, a number of partial evaluations other
    /// than 0 and $K - 1$, batching rounds without partial evaluations, a masks' value next to
    /// partial evaluations, a length other than the one the header implies, a residue not below
    /// $q$, a digit index outside the digit set, or non-zero padding.
    pub fn proof_from_words(&self, words: &[u64]) -> Result<Proof<K>> {
        let shape = |what: &str| Error::Shape(what.into());
        let too_large = || shape("a count in the header is too large");
        let [version, partials, batch_rounds, count, rest @ ..] = words else {
            return Err(shape("the encoding ends early"));
        };
        if *version != VERSION {
            return Err(shape("unknown proof version"));
        }
        let mut reduction = match (*partials, *batch_rounds) {
            (0, 0) => Reduction::Direct,
            (p, rounds) if p == K as u64 - 1 => Reduction::Multilinear {
                rounds: usize::try_from(rounds).map_err(|_| too_large())?,
            },
            (0, _) => return Err(shape("batching rounds without partial evaluations")),
            _ => return Err(shape("a proof has 0 or K - 1 partial evaluations")),
        };
        // The header: k round counts and the final count.
        let count = usize::try_from(*count)
            .ok()
            .filter(|k| *k < rest.len())
            .ok_or_else(|| shape("the encoding ends early"))?;
        let (round_counts, rest) = rest.split_at(count);
        let Some((&final_count, stream)) = rest.split_first() else {
            return Err(shape("the encoding ends early"));
        };
        let stream = if self.ctx.hiding().is_some() {
            let Some((&masked, stream)) = stream.split_first() else {
                return Err(shape("the encoding ends early"));
            };
            match (masked, reduction) {
                (0, _) => {}
                (1, Reduction::Direct) => reduction = Reduction::Masked,
                (3, Reduction::Direct) => reduction = Reduction::Merged,
                (1 | 3, _) => return Err(shape("a masks' value next to partial evaluations")),
                _ => return Err(shape("the masked word of the header is not 0, 1 or 3")),
            }
            stream
        } else {
            stream
        };
        let round_counts = round_counts
            .iter()
            .map(|l| usize::try_from(*l).map_err(|_| too_large()))
            .collect::<Result<Vec<_>>>()?;
        let final_count = usize::try_from(final_count).map_err(|_| too_large())?;
        // The length the header implies, checked before anything is allocated.
        let (d, c) = (self.ctx.d(), &self.ctx.choices);
        let round = K * (c.b as usize + 2 + 3);
        let fixed = (c.n_d + c.n_b) * d + K;
        let mut residues = reduction.residues(K, d);
        for (index, l) in round_counts.iter().enumerate() {
            // Per iteration: l sumcheck rounds and fixed, then p' except after the last.
            let target = if index + 1 < count { d } else { 0 };
            residues = residues.and_then(|r| {
                l.checked_mul(round)?
                    .checked_add(fixed)?
                    .checked_add(target)?
                    .checked_add(r)
            });
        }
        // The final witness: N ring elements of residues without iterations, N digits after them.
        let (entry, entry_bits) = if count == 0 {
            (d, self.residue_bits())
        } else {
            (1, self.digit_bits())
        };
        let bits = residues
            .and_then(|r| r.checked_mul(self.residue_bits()))
            .and_then(|b| {
                final_count
                    .checked_mul(entry)?
                    .checked_mul(entry_bits)?
                    .checked_add(b)
            })
            .ok_or_else(too_large)?;
        if bits.div_ceil(64) != stream.len() {
            return Err(shape("the length does not match the header"));
        }
        let mut r = BitReader {
            words: stream,
            bit: 0,
        };
        let partials = self.take_exts(&mut r, reduction.partials(K))?;
        let batch_rounds = (0..reduction.rounds())
            .map(|_| Ok([self.take_ext(&mut r)?, self.take_ext(&mut r)?]))
            .collect::<Result<Vec<_>>>()?;
        let batch_eval = match reduction {
            Reduction::Multilinear { .. } => Some(self.take_ext(&mut r)?),
            Reduction::Direct | Reduction::Masked | Reduction::Merged => None,
        };
        let mask_eval = match reduction {
            Reduction::Masked | Reduction::Merged => Some(self.take_ext(&mut r)?),
            _ => None,
        };
        let merged = usize::from(reduction == Reduction::Merged) * 2;
        let blinder_mask_evals = self.take_exts(&mut r, merged)?;
        let ring_value = self.take_poly(&mut r)?;
        let blinder_ring_values = self.take_polys(&mut r, merged)?;
        let mut iterations = Vec::with_capacity(count);
        for (index, l) in round_counts.into_iter().enumerate() {
            let v = self.take_polys(&mut r, c.n_d)?;
            let next_commitment = self.take_polys(&mut r, c.n_b)?;
            let rounds = (0..l)
                .map(|_| {
                    Ok(RoundMessage {
                        h0: self.take_exts(&mut r, c.b as usize + 2)?,
                        h_alpha: self.take_exts(&mut r, 3)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let final_eval = self.take_ext(&mut r)?;
            let next_target = if index + 1 < count {
                Some(self.take_poly(&mut r)?)
            } else {
                None
            };
            iterations.push(IterationProof {
                v,
                next_commitment,
                rounds,
                final_eval,
                next_target,
            });
        }
        let final_witness = if count == 0 {
            FinalWitness::Message(self.take_polys(&mut r, final_count)?)
        } else {
            FinalWitness::Digits(self.take_digit_table(&mut r, final_count)?)
        };
        r.finish()?;
        Ok(Proof {
            partials,
            batch_rounds,
            batch_eval,
            mask_eval,
            ring_value,
            iterations,
            final_witness,
            blinder_mask_evals,
            blinder_ring_values,
        })
    }
}

#[cfg(test)]
mod spec_fixture;

#[cfg(test)]
mod tests {
    use super::{
        BitReader, BitWriter, Commitment, HachiPcs, Input, LABEL_MASK_EVAL, LABEL_MERGED_COIN,
        LABEL_PARTIALS, LABEL_RING_VALUE, Proof, State,
    };
    use crate::error::VerifyError;
    use crate::fs::Fs;
    use crate::iteration::{Claim, basis, evaluate_claim};
    use crate::mle::mle_eval;
    use crate::params::{HachiChoices, Hiding};
    use crate::partials::{self, RoundValues};
    use hachi_co_math::{Ext, Poly};

    const K: usize = 4;
    /// The modulus of Hachi-Co's toy-n1 preset (24 bits).
    const Q: u64 = 8_388_637;
    const VARS: usize = 11;
    const CONTEXT: [u8; 32] = [9u8; 32];
    const SEED: [u8; 32] = [3u8; 32];

    /// An honest multilinear opening at the toy choices with a fixed number of iterations: a
    /// polynomial in 11 variables, a point of $\mathbb F_{q^4}$, the value and the proof.
    struct Opening {
        pcs: HachiPcs<K>,
        coeffs: Vec<u64>,
        com: Commitment,
        state: State<K>,
        point: Vec<Ext<K>>,
        value: Ext<K>,
        proof: Proof<K>,
    }

    impl Opening {
        fn new(iterations: usize) -> Self {
            let choices = HachiChoices {
                fixed_iterations: Some(iterations),
                ..HachiChoices::toy(16)
            };
            let pcs = HachiPcs::<K>::new(Q, choices).unwrap();
            let field = pcs.ctx().field();
            let coeffs: Vec<u64> = (0..1u64 << VARS)
                .map(|i| (i.wrapping_mul(2_654_435_761) + 12_345) % Q)
                .collect();
            let (com, state) = pcs.commit_multilinear(&coeffs, &SEED).unwrap();
            let point: Vec<Ext<K>> = (0..VARS as u64)
                .map(|i| field.reduce(&Ext([7 * i + 1, 13 * i + 5, 17 + i, 19 + 3 * i])))
                .collect();
            let (value, proof) = pcs
                .prove_multilinear(&state, &point, &CONTEXT, &SEED)
                .unwrap();
            pcs.verify_multilinear(&com, VARS, &point, &value, &proof, &CONTEXT)
                .unwrap();
            Self {
                pcs,
                coeffs,
                com,
                state,
                point,
                value,
                proof,
            }
        }

        /// The verifier's verdict on `proof` for `value`.
        fn verify(&self, value: &Ext<K>, proof: &Proof<K>) -> Result<(), VerifyError> {
            self.pcs
                .verify_multilinear(&self.com, VARS, &self.point, value, proof, &CONTEXT)
        }

        /// $\mathbf x'$.
        fn x(&self) -> &[Ext<K>] {
            &self.point[self.pcs.ctx().kappa()..]
        }

        /// The monomial coefficients of $f'$.
        fn packed(&self) -> Vec<Ext<K>> {
            self.pcs.packed(VARS, &self.coeffs)
        }

        /// $\mathrm{mono}\_i(x_1, \dots, x_\kappa)$.
        fn mono(&self) -> Vec<Ext<K>> {
            let ctx = self.pcs.ctx();
            ctx.sub.monomials(&self.point[..ctx.kappa()]).unwrap()
        }

        /// The $K$ true partial evaluations.
        fn true_partials(&self) -> Vec<Ext<K>> {
            let field = self.pcs.ctx().field();
            partials::complete(field, &self.mono(), &self.value, &self.proof.partials).unwrap()
        }

        /// The transcript after the statement with `value` and the partial evaluations `sent`.
        fn transcript(&self, value: &Ext<K>, sent: &[Ext<K>]) -> Fs {
            let mut fs = self.pcs.statement(
                &CONTEXT,
                1,
                VARS,
                &self.state.committed.u,
                &self.point,
                value,
            );
            fs.exts(LABEL_PARTIALS, sent);
            fs
        }

        /// The proof with the partial evaluations `sent` and the batching sumcheck `batch`
        /// (rounds, point $\mathbf r$, final value), whose transcript `fs` absorbed them: $Y$
        /// at $\mathbf r$ plus `shift` (embedded in $\mathcal R_q^H$), then the iterations and
        /// the final step of the honest prover.
        fn complete(
            &self,
            mut fs: Fs,
            sent: &[Ext<K>],
            batch: (Vec<RoundValues<K>>, Vec<Ext<K>>, Ext<K>),
            shift: &Ext<K>,
        ) -> Proof<K> {
            let (pcs, ctx) = (&self.pcs, self.pcs.ctx());
            let committed = &self.state.committed;
            let (batch_rounds, r, at_r) = batch;
            let (b, a, _) = pcs.ring_weights(&committed.layout, &r).unwrap();
            let y = evaluate_claim(ctx, &b, &a, &committed.message).unwrap();
            let ring_value = ctx.ring.add(&y, &ctx.sub.embed(shift)).unwrap();
            fs.polys(LABEL_RING_VALUE, core::slice::from_ref(&ring_value));
            let claim = Claim {
                b,
                a,
                target: ring_value.clone(),
            };
            let (iterations, final_witness) = pcs
                .prove_rest(
                    committed.layout,
                    pcs.multilinear_reduction(VARS),
                    committed,
                    claim,
                    &mut fs,
                    &SEED,
                )
                .unwrap();
            Proof {
                partials: sent.to_vec(),
                batch_rounds,
                batch_eval: Some(at_r),
                mask_eval: None,
                ring_value,
                iterations,
                final_witness,
                blinder_mask_evals: Vec::new(),
                blinder_ring_values: Vec::new(),
            }
        }

        /// The proof of a prover that claims `value` and sends the partial evaluations `sent`,
        /// then runs the honest prover's steps: the batching sumcheck on the true tables (its
        /// invariant checked against the true partial evaluations), $Y$ and the rest.
        fn honest_after(&self, value: &Ext<K>, sent: &[Ext<K>]) -> Proof<K> {
            let field = self.pcs.ctx().field();
            let mut fs = self.transcript(value, sent);
            let batch = partials::prove(
                field,
                &self.packed(),
                self.x(),
                &self.true_partials(),
                &mut fs,
            )
            .unwrap();
            self.complete(fs, sent, batch, &field.zero())
        }

        /// The partial evaluations moved by $\boldsymbol\delta$ with $\sum_i \delta_i Z_i = 0$
        /// and $\sum_i \mathrm{mono}\_i \delta_i = 1$ ($\delta_2 = \delta_3 = 0$), and the
        /// value they give, $f(\mathbf x) + 1$. $\sum_i y_i Z_i$
        /// is unchanged. All $K$ of them; a proof sends the last $K - 1$.
        fn kernel_forgery(&self) -> (Vec<Ext<K>>, Ext<K>) {
            let field = self.pcs.ctx().field();
            let mono = self.mono();
            let ratio = field.mul(&basis::<K>(1), &field.inv(&basis::<K>(0)).unwrap());
            let denom = field.sub(&mono[1], &field.mul(&mono[0], &ratio));
            let d1 = field.inv(&denom).unwrap();
            let d0 = field.neg(&field.mul(&d1, &ratio));
            let honest = self.true_partials();
            let mut moved = honest.clone();
            moved[0] = field.add(&moved[0], &d0);
            moved[1] = field.add(&moved[1], &d1);
            let packed = |ys: &[Ext<K>]| {
                ys.iter().enumerate().fold(field.zero(), |acc, (i, y)| {
                    field.add(&acc, &field.mul(y, &basis::<K>(i)))
                })
            };
            assert_eq!(packed(&moved), packed(&honest));
            let forged = self.pcs.value_of_partials(&self.point, &moved).unwrap();
            assert_eq!(forged, field.add(&self.value, &field.one()));
            (moved, forged)
        }
    }

    /// Asserts that `verdict` is the rejection by the check `check`.
    fn rejected_by(verdict: &Result<(), VerifyError>, check: &str, what: &str) {
        assert!(
            matches!(verdict, Err(VerifyError::Check(m)) if m == check),
            "{what}: {verdict:?}, expected the check \"{check}\""
        );
    }

    /// The verifier's message when the batching sumcheck ends on a false claim.
    const BATCH_CHECK: &str = "the batching sumcheck's final value";

    #[test]
    fn packing_by_batches_from_the_end_is_the_packing_in_order() {
        // Batches of one chunk, a few chunks and everything, at plain and masked layouts.
        for hiding in [false, true] {
            let choices = if hiding {
                HachiChoices::toy_hiding(16)
            } else {
                HachiChoices::toy(16)
            };
            let pcs = HachiPcs::<K>::new(Q, choices).unwrap();
            let field = pcs.ctx().field();
            let vars = 11;
            let coeffs: Vec<Ext<K>> = (0..1u64 << vars)
                .map(|i| field.reduce(&Ext([i * 7 + 3, i * 11 + 5, i + Q - 9, i * i])))
                .collect();
            let layout = pcs.multilinear_ext_layout(vars).unwrap();
            let per_chunk = pcs.ctx().d_over_k() * layout.message_len();
            let want: Vec<Vec<Poly>> = coeffs
                .chunks_exact(per_chunk)
                .map(|chunk| pcs.pack_ext_chunk(chunk).unwrap())
                .collect();
            assert!(want.len() > 2, "several chunks");
            let bytes = per_chunk * core::mem::size_of::<Ext<K>>();
            for batch in [1, bytes, 2 * bytes + 1, usize::MAX / 2] {
                let got = pcs
                    .pack_ext_owned_in(coeffs.clone(), &layout, batch)
                    .unwrap();
                assert_eq!(got, want, "hiding {hiding}, batches of {batch} bytes");
            }
        }
    }

    #[test]
    fn partials_moved_along_the_trace_kernel_are_rejected() {
        // With 0, 1 and 2 iterations: partial evaluations moved along the kernel of
        // sum_i y_i Z_i = f'(x'), the value moved
        // with them, and the rest of the proof made honestly. The batching sumcheck, run
        // honestly from there, sums to the batched rows of the true partial evaluations, not
        // of the moved ones, so the verifier ends on a false claim.
        for iterations in [0usize, 1, 2] {
            let o = Opening::new(iterations);
            assert_eq!(o.honest_after(&o.value, &o.proof.partials), o.proof);
            let (moved, forged) = o.kernel_forgery();
            let verdict = o.verify(&forged, &o.honest_after(&forged, &moved[1..]));
            rejected_by(&verdict, BATCH_CHECK, &format!("{iterations} iterations"));
        }
    }

    #[test]
    fn a_single_partial_or_the_value_alone_moved_is_rejected() {
        for iterations in [0usize, 1, 2] {
            let o = Opening::new(iterations);
            let field = o.pcs.ctx().field();
            let mono = o.mono();
            for i in 0..K - 1 {
                let what = format!("{iterations} iterations, partial evaluation {i}");
                let mut moved = o.proof.partials.clone();
                moved[i] = field.add(&moved[i], &field.one());
                // With the value unchanged, the derived partial evaluation moves too.
                let proof = Proof {
                    partials: moved.clone(),
                    ..o.proof.clone()
                };
                let verdict = o.verify(&o.value, &proof);
                rejected_by(&verdict, BATCH_CHECK, &what);
                // With the value moved by mono_{i+1}, and the rest made honestly from there.
                let value = field.add(&o.value, &mono[i + 1]);
                let verdict = o.verify(&value, &o.honest_after(&value, &moved));
                rejected_by(&verdict, BATCH_CHECK, &what);
            }
            // The value alone, with the honest proof and with a proof made for it.
            let what = format!("{iterations} iterations, the value");
            let value = field.add(&o.value, &field.one());
            let verdict = o.verify(&value, &o.proof);
            rejected_by(&verdict, BATCH_CHECK, &what);
            let verdict = o.verify(&value, &o.honest_after(&value, &o.proof.partials));
            rejected_by(&verdict, BATCH_CHECK, &what);
        }
    }

    #[test]
    fn partials_fitted_by_the_rounds_are_rejected() {
        // From the kernel forgery, a prover that makes every batching round add up to its
        // running claim and fits f'(r) to the last one passes the batching sumcheck
        // (`partials::prove_fitted`), with a value that is not f'(r). With Y at r the trace
        // check rejects; with the true f'(r) instead, the sumcheck's final check rejects; and
        // without iterations, with Y moved so that its trace is the fitted value, the final
        // witness does not satisfy the claim. (With iterations, that last case is a false claim
        // for the iterations to refute.)
        for iterations in [0usize, 1, 2] {
            let what = format!("{iterations} iterations");
            let o = Opening::new(iterations);
            let field = o.pcs.ctx().field();
            let (moved, forged) = o.kernel_forgery();
            let fitted = |fs: &mut Fs| {
                partials::prove_fitted(
                    field,
                    &o.packed(),
                    o.x(),
                    &moved,
                    &partials::INPUT_LABELS,
                    fs,
                )
            };
            let mut fs = o.transcript(&forged, &moved[1..]);
            let batch = fitted(&mut fs);
            let table = partials::hypercube_values(field, &o.packed()).unwrap();
            let true_at_r = mle_eval(field, &table, &batch.1);
            assert_ne!(batch.2, true_at_r, "{what}");
            let shift = field.sub(&batch.2, &true_at_r);
            let proof = o.complete(fs, &moved[1..], batch, &field.zero());
            let verdict = o.verify(&forged, &proof);
            rejected_by(&verdict, "the trace of the input reduction", &what);
            let proof = Proof {
                batch_eval: Some(true_at_r),
                ..proof
            };
            let verdict = o.verify(&forged, &proof);
            rejected_by(&verdict, BATCH_CHECK, &what);
            if iterations == 0 {
                let mut fs = o.transcript(&forged, &moved[1..]);
                let batch = fitted(&mut fs);
                let proof = o.complete(fs, &moved[1..], batch, &shift);
                let verdict = o.verify(&forged, &proof);
                rejected_by(
                    &verdict,
                    "the final witness does not satisfy the claim",
                    &what,
                );
            }
        }
    }

    /// The low `width` bits.
    fn low(value: u64, width: usize) -> u64 {
        if width >= 64 {
            value
        } else {
            value & ((1u64 << width) - 1)
        }
    }

    #[test]
    fn fields_of_every_width_round_trip() {
        // 4000 fields of widths 0 to 64 in a pseudorandom order, so that every offset in a
        // word meets every width.
        let mut state = 0x5eed_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state ^ (state >> 29)
        };
        let fields: Vec<(u64, usize)> = (0..4000)
            .map(|_| {
                let width = (next() % 65) as usize;
                (next(), width)
            })
            .collect();
        let mut w = BitWriter::default();
        for (value, width) in &fields {
            w.push(*value, *width);
        }
        let total: usize = fields.iter().map(|(_, width)| width).sum();
        assert_eq!(w.words.len(), total.div_ceil(64));
        let mut r = BitReader {
            words: &w.words,
            bit: 0,
        };
        for (value, width) in &fields {
            assert_eq!(
                r.take(*width).unwrap(),
                low(*value, *width),
                "width {width}"
            );
        }
        r.finish().unwrap();
        let mut end = BitReader {
            words: &w.words,
            bit: 64 * w.words.len(),
        };
        assert!(end.take(1).is_err());
        assert_eq!(end.take(0).unwrap(), 0);
    }

    #[test]
    fn values_are_cut_and_padding_and_trailing_words_are_refused() {
        let mut w = BitWriter::default();
        w.push(u64::MAX, 3);
        w.push(0, 61);
        assert_eq!(w.words, vec![7]);
        w.push(u64::MAX, 64);
        assert_eq!(w.words, vec![7, u64::MAX]);
        // A field across two words.
        let mut w = BitWriter::default();
        w.push(0, 60);
        w.push(0b1_0110_1011, 9);
        assert_eq!(w.words, vec![0b1011 << 60, 0b1_0110]);
        let mut r = BitReader {
            words: &w.words,
            bit: 60,
        };
        assert_eq!(r.take(9).unwrap(), 0b1_0110_1011);
        r.finish().unwrap();
        // Padding bits must be zero, and no word may be left over.
        let mut r = BitReader {
            words: &[1 << 63],
            bit: 0,
        };
        r.take(3).unwrap();
        assert!(r.finish().is_err());
        let r = BitReader {
            words: &[0, 0],
            bit: 64,
        };
        assert!(r.finish().is_err());
        let r = BitReader { words: &[], bit: 0 };
        r.finish().unwrap();
    }

    /// Step 1 of a merged opening for a prover that claims `values` and adds `shifts` to its three
    /// ring values after the masking challenge of its own transcript: whether its three trace
    /// checks pass, its ring values, and the combined claim the iterations would have to prove.
    #[allow(clippy::too_many_arguments)]
    fn lying_merged_claim(
        pcs: &HachiPcs<K>,
        state: &State<K>,
        point: &[Ext<K>],
        weights: [&[Ext<K>]; 2],
        values: &[Ext<K>; 3],
        shifts: &[Poly; 3],
        context: &[u8; 32],
    ) -> (bool, [Poly; 3], Claim) {
        let ctx = pcs.ctx();
        let (field, ring) = (ctx.field(), &ctx.ring);
        let Input::Merged {
            vars,
            values: vectors,
        } = &state.input
        else {
            panic!("a merged state")
        };
        let layout = state.committed.layout;
        let chunks = &state.committed.message;
        let (b, a, v) = pcs.ring_weights(&layout, point).unwrap();
        let one = ring.monomial(0);
        let len = layout.message_len();
        let share: Vec<Vec<Poly>> = chunks.iter().map(|c| c[..len].to_vec()).collect();
        let y_share = evaluate_claim(ctx, &b, &a, &share).unwrap();
        let mut y_mask = ring.zero();
        for (bi, chunk) in b.iter().zip(chunks) {
            y_mask = ring
                .add(&y_mask, &ring.mul(bi, chunk.last().unwrap()).unwrap())
                .unwrap();
        }
        let mut at = len;
        let (mut ys, mut rows, mut block_weights) = (Vec::new(), Vec::new(), Vec::new());
        for (vals, e) in vectors.iter().zip(weights) {
            let n = vals.len().div_ceil(ctx.d_over_k());
            let w = pcs.block_weights(e).unwrap();
            ys.push(ring.inner(&w, &chunks[0][at..at + n]).unwrap());
            rows.push(chunks[0][at + n].clone());
            block_weights.push(w);
            at += n + 1;
        }
        let mut fs =
            pcs.merged_statement(context, *vars, &state.committed.u, point, weights, values);
        let mask_evals = [
            pcs.traced(&y_mask, &v).unwrap(),
            pcs.traced(&rows[0], &one).unwrap(),
            pcs.traced(&rows[1], &one).unwrap(),
        ];
        fs.exts(LABEL_MASK_EVAL, &mask_evals);
        let coin = pcs.mask_coin(&mut fs).unwrap();
        let x = ctx.sub.embed(&coin);
        let honest = [
            ring.add(&y_share, &ring.mul(&x, &y_mask).unwrap()).unwrap(),
            ring.add(&ys[0], &ring.mul(&x, &rows[0]).unwrap()).unwrap(),
            ring.add(&ys[1], &ring.mul(&x, &rows[1]).unwrap()).unwrap(),
        ];
        let targets: [Poly; 3] =
            core::array::from_fn(|i| ring.add(&honest[i], &shifts[i]).unwrap());
        let traces_pass = targets
            .iter()
            .zip([&v, &one, &one])
            .zip(values.iter().zip(&mask_evals))
            .all(|((t, vv), (value, m))| {
                pcs.traced(t, vv).unwrap() == field.add(value, &field.mul(&coin, m))
            });
        fs.polys(LABEL_RING_VALUE, &targets);
        let lambda = pcs.nonzero_coin(&mut fs, LABEL_MERGED_COIN).unwrap();
        let block_weights: [Vec<Poly>; 2] = block_weights.try_into().unwrap();
        let claim = pcs
            .merged_claim(&layout, b, a, &block_weights, &coin, &lambda, &targets)
            .unwrap();
        (traces_pass, targets, claim)
    }

    #[test]
    fn a_false_merged_value_fitted_to_its_trace_check_fails_after_lambda() {
        let choices = HachiChoices {
            hiding: Some(Hiding {
                nu: 1,
                n_a: 1,
                lambda: 128,
                phi: 13,
            }),
            fixed_iterations: Some(1),
            ..HachiChoices::toy(16)
        };
        let pcs = HachiPcs::<K>::new(Q, choices).unwrap();
        let ctx = pcs.ctx();
        let (field, ring) = (ctx.field(), &ctx.ring);
        let ext = |i: u64| Ext([i % Q, (3 * i + 1) % Q, (7 * i + 2) % Q, (11 * i + 5) % Q]);
        let coeffs: Vec<Ext<K>> = (0..256).map(ext).collect();
        let g0: Vec<Ext<K>> = (300..333).map(ext).collect();
        let g1: Vec<Ext<K>> = (400..409).map(ext).collect();
        let point: Vec<Ext<K>> = (500..508).map(ext).collect();
        let e0: Vec<Ext<K>> = (600..633).map(ext).collect();
        let e1: Vec<Ext<K>> = (700..709).map(ext).collect();
        let context = [3u8; 32];
        let (_, state) = pcs.commit_merged(&coeffs, [&g0, &g1], &[1u8; 32]).unwrap();
        let (values, honest) = pcs
            .prove_merged(&state, &point, [&e0, &e1], &context, &[2u8; 32])
            .unwrap();
        let holds = |claim: &Claim| {
            evaluate_claim(ctx, &claim.b, &claim.a, &state.committed.message).unwrap()
                == claim.target
        };
        let zero = [ring.zero(), ring.zero(), ring.zero()];
        // Control: the true values and no shift give the honest prover's ring values and a claim
        // the witness meets.
        let (passes, targets, claim) =
            lying_merged_claim(&pcs, &state, &point, [&e0, &e1], &values, &zero, &context);
        assert!(passes && holds(&claim));
        assert_eq!(targets[0], honest.ring_value);
        assert_eq!(targets[1..], honest.blinder_ring_values[..]);
        // A false vector value with the ring value moved by 1, whose trace is 1 for v = 1: every
        // trace check passes under the prover's own x*, and the combined claim is false.
        for i in 1..3 {
            let mut lie = values;
            lie[i] = field.add(&lie[i], &field.one());
            let mut shifts = zero.clone();
            shifts[i] = ring.monomial(0);
            let (passes, _, claim) =
                lying_merged_claim(&pcs, &state, &point, [&e0, &e1], &lie, &shifts, &context);
            assert!(
                passes,
                "vector {i}: the fitted ring value passes its trace check"
            );
            assert!(!holds(&claim), "vector {i}: the combined claim is false");
        }
        // Both vectors lying at once, by amounts that would cancel for one value of lambda only.
        let mut lie = values;
        lie[1] = field.add(&lie[1], &field.one());
        lie[2] = field.sub(&lie[2], &field.one());
        let shifts = [
            ring.zero(),
            ring.monomial(0),
            ring.neg(&ring.monomial(0)).unwrap(),
        ];
        let (passes, _, claim) =
            lying_merged_claim(&pcs, &state, &point, [&e0, &e1], &lie, &shifts, &context);
        assert!(passes && !holds(&claim), "two lies");
    }
}
