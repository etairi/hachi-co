//! An optional tail for the openings ([`Tail`], [`HachiPcs::with_tail`](crate::HachiPcs::with_tail)):
//! a scheme that commits to the next witness of an opening's last iteration, a table of $2^L$
//! digits, and proves the claim that the final step checks today on that table sent in the clear
//! (`hachi-pcs.tex`, step 3 of `Eval`, and `pcs`, "Tail"). It is not in the paper or the draft: an
//! option approved on 2026-11-17, whose instance is Greyhound with LaBRADOR (crate
//! `hachi-co-greyhound`). Without a tail nothing changes, proofs and encodings included.
//!
//! **Claims** ([`TailClaim`]). A claim names the table's size $L$ (`vars`), the tail's commitment
//! to it, a point $\mathbf a \in \mathbb F_{q^K}^L$ and a value $y'$. It holds if every entry of the
//! committed table is a digit of base $b$, the range that the last iteration's sumcheck proved
//! for its table, and the multilinear polynomial whose values on the hypercube are the entries
//! takes $y'$ at $\mathbf a$. The first coordinate of $\mathbf a$ goes with the most significant
//! bit of an index, as [`table_eval`](crate::iteration::table_eval) reads a table. A proof covers
//! one claim or several, and its length depends only on their sizes ([`Tail::proof_words`]).
//!
//! **Modes** ([`TailMode`]). In single mode the proof of an opening carries the tail's proof of
//! its own claim. In batched mode it carries none: the prover's proof hands the claim to the
//! caller ([`TailFinal::deferred`]), and the verifier's deferred methods (for example
//! [`HachiPcs::verify_multilinear_ext_deferred`](crate::HachiPcs::verify_multilinear_ext_deferred))
//! return it ([`DeferredClaim`]), so that the caller proves and checks several claims with one
//! tail proof. Hachi-Co does this for the three openings of a prover (`hachi_co::HachiBase`).
//!
//! **Contexts.** A tail proof is made under a 32-byte context. Each opening derives its own from
//! its transcript after the last iteration (label `hachi/tail/context`), which binds the tail
//! proof to the whole opening. Single mode proves under it, and batched mode under
//! [`batch_context`] of the claims' contexts in order.
//!
//! **The stub** (`StubTail`, with the feature `stub-tail` or in this crate's tests) commits by a
//! SHAKE digest and proves by sending the tables: a stand-in that tests the plumbing, neither
//! succinct nor hiding.

use crate::{Error, Result};
use core::any::Any;
use hachi_co_math::Ext;
use hachi_co_transcript::Transcript;
use std::sync::Arc;

/// What [`Tail::commit`] keeps for proving: any value of the tail's own type, which its
/// [`Tail::prove`] reads back by downcasting.
pub type TailOpening = Box<dyn Any + Send + Sync>;

/// A claim on a committed table of digits (module documentation, "Claims").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TailClaim<'a, const K: usize> {
    /// $L$: the table has $2^L$ digits and the point $L$ coordinates.
    pub vars: usize,
    /// The tail's commitment to the table ([`Tail::commitment_words`] words).
    pub commitment: &'a [u64],
    /// $\mathbf a$, canonical.
    pub point: &'a [Ext<K>],
    /// $y'$, canonical.
    pub value: Ext<K>,
}

/// A claim with what proving it needs: the table and the opening of its commitment.
#[derive(Clone, Copy)]
pub struct TailProverClaim<'a, const K: usize> {
    /// The claim.
    pub claim: TailClaim<'a, K>,
    /// The table, $2^L$ digits in the order of the indices.
    pub digits: &'a [i8],
    /// What [`Tail::commit`] returned with the commitment.
    pub opening: &'a (dyn Any + Send + Sync),
}

impl<const K: usize> core::fmt::Debug for TailProverClaim<'_, K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TailProverClaim")
            .field("claim", &self.claim)
            .field("digits", &self.digits.len())
            .finish_non_exhaustive()
    }
}

/// A scheme that proves the claims of the final step in place of the table sent in the clear
/// (module documentation). It is dyn-compatible, so that Hachi holds any tail as an
/// `Arc<dyn Tail<K>>`.
///
/// **Words.** Commitments and proofs are `u64` words. [`Tail::commitment_words`] and
/// [`Tail::proof_words`] give their lengths, which must depend only on the sizes, and what
/// [`Tail::commit`] and [`Tail::prove`] return must have exactly these lengths: Hachi encodes
/// them as fields of 64 bits and counts them in its proof sizes. [`Tail::verify`] must accept any
/// words without panicking and reject what it cannot check.
pub trait Tail<const K: usize>: Send + Sync + core::fmt::Debug {
    /// The largest table the tail takes: $2^{\mathtt{max\_vars}}$ digits. Hachi's schedule ends
    /// with the first iteration whose table fits ([`HachiPcs::with_tail`](crate::HachiPcs::with_tail)).
    fn max_vars(&self) -> usize;

    /// The words of a commitment to a table of $2^{\mathtt{vars}}$ digits.
    fn commitment_words(&self, vars: usize) -> usize;

    /// The words of one proof of claims on tables of $2^{v_1}, 2^{v_2}, \dots$ digits, the sizes
    /// $v_i$ of `vars` in this order.
    fn proof_words(&self, vars: &[usize]) -> usize;

    /// A digest of the tail's public parameters, which Hachi absorbs into the statement of every
    /// opening with this tail. Two tails with different parameters must have different digests.
    fn params_digest(&self) -> [u8; 32];

    /// Commits to the table `digits` of $2^{\mathtt{vars}}$ digits.
    ///
    /// # Errors
    /// For a table of another length, more than $2^{\mathtt{max\_vars}}$ digits, or entries
    /// outside the digit set.
    fn commit(&self, vars: usize, digits: &[i8]) -> Result<(Vec<u64>, TailOpening)>;

    /// One proof of all of `claims` under `context`.
    ///
    /// # Errors
    /// For claims the tail cannot prove (an opening of another tail, a table that does not
    /// satisfy its claim).
    fn prove(&self, claims: &[TailProverClaim<'_, K>], context: &[u8; 32]) -> Result<Vec<u64>>;

    /// Checks that `proof` proves all of `claims` under `context`.
    ///
    /// # Errors
    /// The reason of the rejection.
    fn verify(
        &self,
        claims: &[TailClaim<'_, K>],
        proof: &[u64],
        context: &[u8; 32],
    ) -> core::result::Result<(), String>;
}

/// How an opening with a tail ends (module documentation, "Modes").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TailMode {
    /// The proof carries the tail's proof of its own claim.
    Single,
    /// The proof carries no tail proof of its own: the claim goes to the caller.
    Batched,
}

impl TailMode {
    /// The word of the mode in statements and in the header of a proof: 1 for
    /// [`TailMode::Single`], 2 for [`TailMode::Batched`].
    pub fn word(self) -> u64 {
        match self {
            Self::Single => 1,
            Self::Batched => 2,
        }
    }
}

/// A claim that a batched opening leaves to its caller, on the prover's side: the claim, the
/// table and the opening of its commitment, and the opening's context.
#[derive(Clone)]
pub struct Deferred<const K: usize> {
    /// $L$.
    pub vars: usize,
    /// The tail's commitment to the table.
    pub commitment: Vec<u64>,
    /// $\mathbf a$.
    pub point: Vec<Ext<K>>,
    /// $y'$.
    pub value: Ext<K>,
    /// The table.
    pub digits: Vec<i8>,
    /// The opening of the commitment ([`Tail::commit`]).
    pub opening: Arc<dyn Any + Send + Sync>,
    /// The opening's context for the tail (module documentation, "Contexts").
    pub context: [u8; 32],
}

impl<const K: usize> Deferred<K> {
    /// The claim.
    pub fn claim(&self) -> TailClaim<'_, K> {
        TailClaim {
            vars: self.vars,
            commitment: &self.commitment,
            point: &self.point,
            value: self.value,
        }
    }

    /// The claim with its table and opening, for [`Tail::prove`].
    pub fn prover_claim(&self) -> TailProverClaim<'_, K> {
        TailProverClaim {
            claim: self.claim(),
            digits: &self.digits,
            opening: self.opening.as_ref(),
        }
    }
}

impl<const K: usize> core::fmt::Debug for Deferred<K> {
    /// Everything but the table and the opening, of which only the table's length is shown.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Deferred")
            .field("vars", &self.vars)
            .field("commitment", &self.commitment)
            .field("point", &self.point)
            .field("value", &self.value)
            .field("digits", &self.digits.len())
            .field("context", &self.context)
            .finish_non_exhaustive()
    }
}

impl<const K: usize> PartialEq for Deferred<K> {
    /// The same claim, table and context, and the same opening (the same allocation).
    fn eq(&self, other: &Self) -> bool {
        self.vars == other.vars
            && self.commitment == other.commitment
            && self.point == other.point
            && self.value == other.value
            && self.digits == other.digits
            && self.context == other.context
            && Arc::ptr_eq(&self.opening, &other.opening)
    }
}

impl<const K: usize> Eq for Deferred<K> {}

/// What the final step of an opening with a tail sends in place of its table
/// ([`Proof::tail`](crate::Proof::tail)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TailFinal<const K: usize> {
    /// The tail proof words of the final step: in single mode the tail's proof of the opening's
    /// claim, in batched mode none, or the proof of the caller's batch if the caller puts it here
    /// (as `hachi_co::HachiBase` does in the share polynomial's opening).
    pub proof: Vec<u64>,
    /// In batched mode, on the prover's side, the claim left to the caller, which takes it to
    /// prove it. Not encoded, and `None` in a decoded proof.
    pub deferred: Option<Deferred<K>>,
}

/// A claim that a batched opening leaves to its caller, on the verifier's side: every check of
/// the opening but the tail's passed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeferredClaim<const K: usize> {
    /// $L$.
    pub vars: usize,
    /// The tail's commitment to the table, as the last iteration sent it.
    pub commitment: Vec<u64>,
    /// $\mathbf a$.
    pub point: Vec<Ext<K>>,
    /// $y'$.
    pub value: Ext<K>,
    /// The opening's context for the tail (module documentation, "Contexts").
    pub context: [u8; 32],
    /// The tail proof words that the opening's final step carries ([`TailFinal::proof`]),
    /// unchecked: the caller checks them, with the batch they belong to.
    pub carried: Vec<u64>,
}

impl<const K: usize> DeferredClaim<K> {
    /// The claim.
    pub fn claim(&self) -> TailClaim<'_, K> {
        TailClaim {
            vars: self.vars,
            commitment: &self.commitment,
            point: &self.point,
            value: self.value,
        }
    }
}

/// The transcript domain of [`batch_context`].
pub const DOMAIN_BATCH: &[u8] = b"hachi/tail/batch";

/// The context of one tail proof of several deferred claims: the challenge seed `context` of a
/// transcript on [`DOMAIN_BATCH`] that absorbed the number of claims and each claim's context in
/// order (module documentation, "Contexts").
pub fn batch_context(contexts: &[[u8; 32]]) -> [u8; 32] {
    let mut t = Transcript::new(DOMAIN_BATCH);
    t.append_u64(b"count", contexts.len() as u64);
    for context in contexts {
        t.append_message(b"context", context);
    }
    t.challenge_seed(b"context")
}

/// Checks that a tail returned `count` words where `expected` were due ([`Tail`], "Words").
///
/// # Errors
/// [`Error::Protocol`] otherwise.
pub(crate) fn check_words(what: &str, count: usize, expected: usize) -> Result<()> {
    if count != expected {
        return Err(Error::Protocol(format!(
            "the tail returned a {what} of {count} words, {expected} expected"
        )));
    }
    Ok(())
}

#[cfg(any(test, feature = "stub-tail"))]
pub use stub::StubTail;

/// The stub tail of the tests (module documentation, "The stub").
#[cfg(any(test, feature = "stub-tail"))]
mod stub {
    use super::{Tail, TailClaim, TailOpening, TailProverClaim};
    use crate::iteration::table_eval;
    use crate::{Error, Result};
    use hachi_co_math::decompose::digit_range;
    use hachi_co_math::{Ext, Fqk};
    use hachi_co_transcript::Transcript;

    /// The transcript domain of the stub's digests.
    const DOMAIN: &[u8] = b"hachi/tail/stub";

    /// A tail for tests only: the commitment to a table is a SHAKE digest of its size and its
    /// digits (4 words), and a proof is the tables of its claims, 8 digits per word, so its
    /// verifier recomputes each digest, checks the digits and evaluates the table. Binding as the
    /// digest is, but neither succinct nor hiding: it tests the plumbing of [`Tail`], not a
    /// scheme.
    #[derive(Clone, Debug)]
    pub struct StubTail<const K: usize> {
        field: Fqk<K>,
        b: u64,
        max_vars: usize,
        /// Whether the commitment and the proof get a trailing word that the verifier ignores,
        /// so that tests see what an unchecked word costs ([`StubTail::sloppy`]).
        sloppy: bool,
    }

    impl<const K: usize> StubTail<K> {
        /// The stub for $\mathbb F_{q^K}$, digits of base `b` and tables of at most
        /// $2^{\mathtt{max\_vars}}$ digits.
        ///
        /// # Errors
        /// [`Error::Math`] for a modulus the field does not take.
        pub fn new(q: u64, b: u64, max_vars: usize) -> Result<Self> {
            Ok(Self {
                field: Fqk::new(q)?,
                b,
                max_vars,
                sloppy: false,
            })
        }

        /// This stub, with one more word at the end of every commitment and every proof, which
        /// its verifier does not read: a negative control for the tests that change words.
        pub fn sloppy(self) -> Self {
            Self {
                sloppy: true,
                ..self
            }
        }

        /// The digest of a table of $2^{\mathtt{vars}}$ digits: 4 words.
        fn digest(&self, vars: usize, digits: &[i8]) -> Vec<u64> {
            let mut t = Transcript::new(DOMAIN);
            t.append_u64(b"vars", vars as u64);
            let bytes: Vec<u8> = digits.iter().map(|d| *d as u8).collect();
            t.append_message(b"digits", &bytes);
            let seed = t.challenge_seed(b"commitment");
            seed.as_chunks::<8>()
                .0
                .iter()
                .map(|w| u64::from_le_bytes(*w))
                .collect()
        }

        /// The number of words of a table of $2^{\mathtt{vars}}$ digits, 8 per word.
        fn table_words(vars: usize) -> usize {
            (1usize << vars).div_ceil(8)
        }

        /// Whether every entry lies in the digit set of base $b$.
        fn digits_fit(&self, digits: &[i8]) -> bool {
            let (lo, hi) = digit_range(self.b);
            digits.iter().all(|d| (lo..=hi).contains(&i128::from(*d)))
        }

        /// The value of the table at `point` ([`table_eval`] on the digits as residues).
        fn eval(&self, digits: &[i8], point: &[Ext<K>]) -> Result<Ext<K>> {
            let zq = self.field.zq();
            let residues: Vec<u64> = digits.iter().map(|d| zq.from_i64(i64::from(*d))).collect();
            table_eval(&self.field, &residues, point)
        }
    }

    impl<const K: usize> Tail<K> for StubTail<K> {
        fn max_vars(&self) -> usize {
            self.max_vars
        }

        fn commitment_words(&self, _vars: usize) -> usize {
            4 + usize::from(self.sloppy)
        }

        fn proof_words(&self, vars: &[usize]) -> usize {
            vars.iter().map(|v| Self::table_words(*v)).sum::<usize>() + usize::from(self.sloppy)
        }

        fn params_digest(&self) -> [u8; 32] {
            let mut t = Transcript::new(DOMAIN);
            t.append_u64(b"q", self.field.modulus());
            t.append_u64(b"k", K as u64);
            t.append_u64(b"b", self.b);
            t.append_u64(b"max_vars", self.max_vars as u64);
            t.append_u64(b"sloppy", u64::from(self.sloppy));
            t.challenge_seed(b"params")
        }

        fn commit(&self, vars: usize, digits: &[i8]) -> Result<(Vec<u64>, TailOpening)> {
            if vars > self.max_vars
                || u32::try_from(vars).ok().and_then(|v| 1usize.checked_shl(v))
                    != Some(digits.len())
            {
                return Err(Error::Shape(format!(
                    "the stub tail commits to 2^vars digits, vars <= {}",
                    self.max_vars
                )));
            }
            if !self.digits_fit(digits) {
                return Err(Error::Shape("the stub tail commits to digits".into()));
            }
            let mut words = self.digest(vars, digits);
            if self.sloppy {
                words.push(0);
            }
            Ok((words, Box::new(())))
        }

        fn prove(
            &self,
            claims: &[TailProverClaim<'_, K>],
            _context: &[u8; 32],
        ) -> Result<Vec<u64>> {
            let mut out = Vec::new();
            for c in claims {
                if c.opening.downcast_ref::<()>().is_none() {
                    return Err(Error::Protocol(
                        "an opening of another tail than the stub".into(),
                    ));
                }
                if self.eval(c.digits, c.claim.point)? != c.claim.value {
                    return Err(Error::Protocol(
                        "the table does not take the claimed value".into(),
                    ));
                }
                for word in c.digits.chunks(8) {
                    let mut bytes = [0u8; 8];
                    for (b, d) in bytes.iter_mut().zip(word) {
                        *b = *d as u8;
                    }
                    out.push(u64::from_le_bytes(bytes));
                }
            }
            if self.sloppy {
                out.push(0);
            }
            Ok(out)
        }

        fn verify(
            &self,
            claims: &[TailClaim<'_, K>],
            proof: &[u64],
            _context: &[u8; 32],
        ) -> core::result::Result<(), String> {
            let vars: Vec<usize> = claims.iter().map(|c| c.vars).collect();
            if vars.iter().any(|v| *v > self.max_vars) {
                return Err(format!(
                    "the stub tail takes tables of at most 2^{} digits",
                    self.max_vars
                ));
            }
            if proof.len() != self.proof_words(&vars) {
                return Err("the stub's proof does not have the length of its claims".into());
            }
            let mut at = 0;
            for (i, c) in claims.iter().enumerate() {
                let n = Self::table_words(c.vars);
                let digits: Vec<i8> = proof[at..at + n]
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .take(1 << c.vars)
                    .map(|b| b as i8)
                    .collect();
                at += n;
                let (padding, used) = (proof[at - 1].to_le_bytes(), (1usize << c.vars) % 8);
                if used != 0 && padding[used..].iter().any(|b| *b != 0) {
                    return Err(format!("claim {i}: non-zero padding in the stub's proof"));
                }
                if !self.digits_fit(&digits) {
                    return Err(format!("claim {i}: the table is not made of digits"));
                }
                let digest = self.digest(c.vars, &digits);
                if c.commitment.len() != self.commitment_words(c.vars)
                    || c.commitment[..4] != digest[..]
                {
                    return Err(format!("claim {i}: the table does not open the commitment"));
                }
                match self.eval(&digits, c.point) {
                    Ok(value) if value == c.value => {}
                    Ok(_) => return Err(format!("claim {i}: the table does not take the value")),
                    Err(e) => return Err(format!("claim {i}: {e}")),
                }
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    //! The tail at Hachi's level, with [`StubTail`]. Openings of polynomials over the field and
    //! of vectors, with and without masks, in single and batched mode, verify. The decoders give
    //! back what the encoders made, at the lengths of the size formulas. A changed word of the
    //! tail's commitment or proof is rejected. The schedule takes the fewest iterations whose last
    //! table fits. Without a tail every proof is the one of the scheme without the option.

    use super::{Deferred, DeferredClaim, StubTail, Tail, TailMode, batch_context};
    use crate::params::HachiChoices;
    use crate::{Commitment, HachiPcs, Proof, Schedule, VerifyError};
    use hachi_co_math::Ext;
    use hachi_co_sampling::ShakePrg;
    use hachi_co_sampling::uniform::uniform_mod_q;
    use std::sync::Arc;

    const K: usize = 4;
    /// The modulus of Hachi-Co's toy-n1 preset (24 bits).
    const Q: u64 = 8_388_637;
    const VARS: usize = 11;
    const LEN: usize = 33;
    const CONTEXT: [u8; 32] = [6u8; 32];

    fn exts(tag: &str, len: usize) -> Vec<Ext<K>> {
        uniform_mod_q(
            &mut ShakePrg::new(&[5u8; 32], tag.as_bytes(), 0),
            Q,
            len * K,
        )
        .unwrap()
        .as_chunks::<K>()
        .0
        .iter()
        .map(|c| Ext(*c))
        .collect()
    }

    fn choices(hiding: bool) -> HachiChoices {
        if hiding {
            HachiChoices::toy_hiding(16)
        } else {
            HachiChoices::toy(16)
        }
    }

    fn stub(max_vars: usize) -> StubTail<K> {
        StubTail::new(Q, 4, max_vars).unwrap()
    }

    /// The scheme with the stub in `mode`.
    fn with_stub(hiding: bool, max_vars: usize, mode: TailMode) -> HachiPcs<K> {
        HachiPcs::<K>::new(Q, choices(hiding))
            .unwrap()
            .with_tail(Arc::new(stub(max_vars)), mode)
            .unwrap()
    }

    /// The table size $L$ of the last iteration of each candidate with iterations, for the
    /// polynomial over the field and the vector, in the order of the candidates.
    fn candidate_sizes(hiding: bool) -> [Vec<usize>; 2] {
        let pcs = HachiPcs::<K>::new(Q, choices(hiding)).unwrap();
        let (c, q) = (&pcs.ctx().choices, pcs.ctx().q);
        [
            pcs.multilinear_ext_layout(VARS).unwrap(),
            pcs.vector_layout(LEN).unwrap(),
        ]
        .map(|first| {
            Schedule::candidates(c, q, K, first)
                .iter()
                .filter_map(|s| s.iterations.last().map(|it| it.table_vars))
                .collect()
        })
    }

    /// One opening of each kind under a scheme: the polynomial over the field at a point, and
    /// the vector against weights, with their commitments, values and proofs.
    struct Openings {
        point: Vec<Ext<K>>,
        weights: Vec<Ext<K>>,
        commitments: [Commitment; 2],
        values: [Ext<K>; 2],
        proofs: [Proof<K>; 2],
    }

    fn open(pcs: &HachiPcs<K>) -> Openings {
        let (coeffs, vector) = (exts("coeffs", 1 << VARS), exts("vector", LEN));
        let (point, weights) = (exts("point", VARS), exts("weights", LEN));
        let (com_f, st_f) = pcs.commit_multilinear_ext(&coeffs, &[1u8; 32]).unwrap();
        let (com_v, st_v) = pcs.commit_vector(&vector, &[2u8; 32]).unwrap();
        let (y_f, proof_f) = pcs
            .prove_multilinear_ext(&st_f, &point, &CONTEXT, &[3u8; 32])
            .unwrap();
        let (y_v, proof_v) = pcs
            .prove_inner_product(&st_v, &weights, &CONTEXT, &[4u8; 32])
            .unwrap();
        Openings {
            point,
            weights,
            commitments: [com_f, com_v],
            values: [y_f, y_v],
            proofs: [proof_f, proof_v],
        }
    }

    impl Openings {
        /// The verdicts of the two openings with the proofs `proofs`.
        fn verify(
            &self,
            pcs: &HachiPcs<K>,
            proofs: &[Proof<K>; 2],
        ) -> [Result<(), VerifyError>; 2] {
            [
                pcs.verify_multilinear_ext(
                    &self.commitments[0],
                    VARS,
                    &self.point,
                    &self.values[0],
                    &proofs[0],
                    &CONTEXT,
                ),
                pcs.verify_inner_product(
                    &self.commitments[1],
                    LEN,
                    &self.weights,
                    &self.values[1],
                    &proofs[1],
                    &CONTEXT,
                ),
            ]
        }

        /// The deferred claims of the two openings with the proofs `proofs`, in batched mode.
        fn deferred(
            &self,
            pcs: &HachiPcs<K>,
            proofs: &[Proof<K>; 2],
        ) -> [Result<DeferredClaim<K>, VerifyError>; 2] {
            [
                pcs.verify_multilinear_ext_deferred(
                    &self.commitments[0],
                    VARS,
                    &self.point,
                    &self.values[0],
                    &proofs[0],
                    &CONTEXT,
                ),
                pcs.verify_inner_product_deferred(
                    &self.commitments[1],
                    LEN,
                    &self.weights,
                    &self.values[1],
                    &proofs[1],
                    &CONTEXT,
                ),
            ]
        }
    }

    /// The proof as its decoder gives it back: without the prover's deferred claim.
    fn decoded_form(proof: &Proof<K>) -> Proof<K> {
        let mut proof = proof.clone();
        if let Some(ending) = proof.tail.as_mut() {
            ending.deferred = None;
        }
        proof
    }

    /// The words of the encoding of `proof` that change when word `at` of the tail's commitment
    /// (`commitment`) or of its proof changes: the tail's words are fields of 64 bits, so each
    /// lies across one or two words of the encoding.
    fn encoded_words(
        pcs: &HachiPcs<K>,
        proof: &Proof<K>,
        commitment: bool,
        at: usize,
    ) -> Vec<usize> {
        let words = pcs.proof_to_words(proof);
        let mut p = decoded_form(proof);
        if commitment {
            let last = p.iterations.last_mut().unwrap();
            last.tail_commitment.as_mut().unwrap()[at] ^= u64::MAX;
        } else {
            p.tail.as_mut().unwrap().proof[at] ^= u64::MAX;
        }
        let other = pcs.proof_to_words(&p);
        (0..words.len())
            .filter(|i| words[*i] != other[*i])
            .collect()
    }

    #[test]
    fn single_mode_openings_verify_round_trip_and_have_their_sizes() {
        for hiding in [false, true] {
            let sizes = candidate_sizes(hiding);
            // The tail takes the second candidate's last table of the polynomial, so that the
            // first candidate's does not fit if it is larger.
            let max_vars = sizes[0][1.min(sizes[0].len() - 1)];
            let pcs = with_stub(hiding, max_vars, TailMode::Single);
            let o = open(&pcs);
            for verdict in o.verify(&pcs, &o.proofs) {
                assert_eq!(verdict, Ok(()), "hiding {hiding}");
            }
            for (i, proof) in o.proofs.iter().enumerate() {
                let words = pcs.proof_to_words(proof);
                assert_eq!(
                    pcs.proof_from_words(&words).as_ref(),
                    Ok(proof),
                    "hiding {hiding}"
                );
                let expected = if i == 0 {
                    pcs.multilinear_ext_proof_words(VARS)
                } else {
                    pcs.vector_proof_words(LEN)
                };
                assert_eq!(
                    words.len(),
                    expected.unwrap(),
                    "hiding {hiding}, opening {i}"
                );
                let ending = proof.tail.as_ref().unwrap();
                assert!(ending.deferred.is_none() && !ending.proof.is_empty());
                assert!(
                    proof.iterations.iter().rev().skip(1).all(|it| {
                        it.tail_commitment.is_none() && !it.next_commitment.is_empty()
                    })
                );
            }
            // The deferred verifiers refuse an opening that proves its own claim.
            assert!(o.deferred(&pcs, &o.proofs).iter().all(Result::is_err));
        }
    }

    #[test]
    fn batched_openings_leave_their_claims_and_one_proof_checks_them() {
        for hiding in [false, true] {
            let pcs = with_stub(hiding, 20, TailMode::Batched);
            let tail = stub(20);
            let o = open(&pcs);
            // The whole-opening verifiers refuse an opening that leaves its claim.
            assert!(o.verify(&pcs, &o.proofs).iter().all(Result::is_err));
            let claims: Vec<DeferredClaim<K>> = o
                .deferred(&pcs, &o.proofs)
                .into_iter()
                .map(Result::unwrap)
                .collect();
            let prover: Vec<&Deferred<K>> = o
                .proofs
                .iter()
                .map(|p| p.tail.as_ref().unwrap().deferred.as_ref().unwrap())
                .collect();
            for (claim, mine) in claims.iter().zip(&prover) {
                assert_eq!(claim.claim(), mine.claim(), "hiding {hiding}");
                assert_eq!(claim.context, mine.context);
                assert!(claim.carried.is_empty());
            }
            assert_ne!(claims[0].context, claims[1].context);
            let context = batch_context(&[claims[0].context, claims[1].context]);
            let prover_claims: Vec<_> = prover.iter().map(|d| d.prover_claim()).collect();
            let proof = tail.prove(&prover_claims, &context).unwrap();
            let checked: Vec<_> = claims.iter().map(DeferredClaim::claim).collect();
            assert_eq!(tail.verify(&checked, &proof, &context), Ok(()));
            // A claim moved to another value is refused.
            let mut moved = checked.clone();
            moved[1].value.0[0] ^= 1;
            assert!(tail.verify(&moved, &proof, &context).is_err());
            for (i, p) in o.proofs.iter().enumerate() {
                // No tail proof in batched mode.
                let words = pcs.proof_to_words(p);
                assert_eq!(pcs.proof_from_words(&words), Ok(decoded_form(p)));
                let expected = if i == 0 {
                    pcs.multilinear_ext_proof_words(VARS)
                } else {
                    pcs.vector_proof_words(LEN)
                }
                .unwrap();
                assert_eq!(words.len(), expected, "hiding {hiding}, opening {i}");
                // A batch's proof carried in the final step costs its words and comes back to the
                // caller with the same claim.
                let mut carrying = decoded_form(p);
                carrying.tail.as_mut().unwrap().proof.clone_from(&proof);
                let words = pcs.proof_to_words(&carrying);
                assert_eq!(words.len(), expected + proof.len());
                let mut proofs = [decoded_form(&o.proofs[0]), decoded_form(&o.proofs[1])];
                proofs[i] = pcs.proof_from_words(&words).unwrap();
                let again = o.deferred(&pcs, &proofs)[i].clone().unwrap();
                assert_eq!(again.carried, proof);
                assert_eq!(again.claim(), claims[i].claim());
            }
        }
    }

    #[test]
    fn a_changed_tail_word_is_rejected() {
        for hiding in [false, true] {
            let pcs = with_stub(hiding, 20, TailMode::Single);
            let o = open(&pcs);
            for i in 0..2 {
                let proof = &o.proofs[i];
                let last = proof.iterations.last().unwrap();
                let commitment_len = last.tail_commitment.as_ref().unwrap().len();
                let proof_len = proof.tail.as_ref().unwrap().proof.len();
                let words = pcs.proof_to_words(proof);
                let fields = (0..commitment_len)
                    .map(|at| (true, at))
                    .chain([0, proof_len / 2, proof_len - 1].map(|at| (false, at)));
                for (commitment, at) in fields {
                    for word in encoded_words(&pcs, proof, commitment, at) {
                        for bit in [0, 17, 63] {
                            let mut changed = words.clone();
                            changed[word] ^= 1 << bit;
                            let what = format!(
                                "hiding {hiding}, opening {i}, tail commitment {commitment}, \
                                 field {at}, word {word}, bit {bit}"
                            );
                            let Ok(decoded) = pcs.proof_from_words(&changed) else {
                                continue;
                            };
                            let mut proofs = o.proofs.clone();
                            proofs[i] = decoded;
                            assert!(o.verify(&pcs, &proofs)[i].is_err(), "{what}");
                        }
                    }
                }
            }
            // A sloppy stub ignores the last word of its commitments and of its proofs. The
            // commitment's is still bound, by the transcript, and the proof's is not: the tail
            // must check every word of its proofs.
            let sloppy = HachiPcs::<K>::new(Q, choices(hiding))
                .unwrap()
                .with_tail(Arc::new(stub(20).sloppy()), TailMode::Single)
                .unwrap();
            let o = open(&sloppy);
            assert!(o.verify(&sloppy, &o.proofs).iter().all(Result::is_ok));
            let mut changed = decoded_form(&o.proofs[0]);
            let last = changed.iterations.last_mut().unwrap();
            *last.tail_commitment.as_mut().unwrap().last_mut().unwrap() ^= 1;
            let verdict = o.verify(&sloppy, &[changed, o.proofs[1].clone()]);
            assert!(verdict[0].is_err(), "hiding {hiding}");
            let mut changed = decoded_form(&o.proofs[0]);
            *changed.tail.as_mut().unwrap().proof.last_mut().unwrap() ^= 1;
            let verdict = o.verify(&sloppy, &[changed, o.proofs[1].clone()]);
            assert_eq!(verdict[0], Ok(()), "hiding {hiding}: the ignored word");
        }
    }

    #[test]
    fn the_schedule_takes_the_fewest_iterations_whose_table_fits() {
        for hiding in [false, true] {
            let sizes = candidate_sizes(hiding);
            for (kind, sizes) in sizes.iter().enumerate() {
                let smallest = *sizes.iter().min().unwrap();
                for max_vars in smallest..=sizes[0] + 1 {
                    let pcs = with_stub(hiding, max_vars, TailMode::Single);
                    let schedule = if kind == 0 {
                        pcs.multilinear_ext_schedule(VARS)
                    } else {
                        pcs.vector_schedule(LEN)
                    }
                    .unwrap();
                    let want = 1 + sizes.iter().position(|l| *l <= max_vars).unwrap();
                    assert_eq!(schedule.iterations.len(), want, "hiding {hiding}, {kind}");
                    assert!(schedule.iterations.last().unwrap().table_vars <= max_vars);
                }
                let pcs = with_stub(hiding, smallest - 1, TailMode::Single);
                let schedule = if kind == 0 {
                    pcs.multilinear_ext_schedule(VARS)
                } else {
                    pcs.vector_schedule(LEN)
                };
                assert!(schedule.is_err(), "hiding {hiding}, {kind}: no table fits");
            }
            // A fixed number of iterations is kept if its table fits, and refused otherwise.
            let fixed = |n: usize, max_vars: usize| {
                HachiPcs::<K>::new(
                    Q,
                    HachiChoices {
                        fixed_iterations: Some(n),
                        ..choices(hiding)
                    },
                )
                .unwrap()
                .with_tail(Arc::new(stub(max_vars)), TailMode::Single)
                .unwrap()
                .multilinear_ext_schedule(VARS)
            };
            let first = sizes[0][0];
            assert_eq!(fixed(1, first).unwrap().iterations.len(), 1);
            assert!(fixed(1, first - 1).is_err());
            assert!(fixed(0, 30).is_err());
        }
    }

    #[test]
    fn without_a_tail_the_proofs_are_those_of_the_scheme() {
        for hiding in [false, true] {
            let plain = HachiPcs::<K>::new(Q, choices(hiding)).unwrap();
            let dropped = with_stub(hiding, 20, TailMode::Single).without_tail();
            let (a, b) = (open(&plain), open(&dropped));
            assert_eq!(a.proofs, b.proofs, "hiding {hiding}");
            for proof in &a.proofs {
                assert!(proof.tail.is_none());
                assert!(
                    proof
                        .iterations
                        .iter()
                        .all(|it| it.tail_commitment.is_none())
                );
            }
            // A proof made with a tail is refused without one, and the other way round, and
            // neither decodes under the other scheme.
            let tailed = with_stub(hiding, 20, TailMode::Single);
            let c = open(&tailed);
            assert!(c.verify(&plain, &c.proofs).iter().all(Result::is_err));
            assert!(a.verify(&tailed, &a.proofs).iter().all(Result::is_err));
            for i in 0..2 {
                let with_tail = tailed.proof_to_words(&c.proofs[i]);
                let without = plain.proof_to_words(&a.proofs[i]);
                assert!(plain.proof_from_words(&with_tail).is_err());
                assert!(tailed.proof_from_words(&without).is_err());
                // Nor under the other mode.
                let batched = with_stub(hiding, 20, TailMode::Batched);
                assert!(batched.proof_from_words(&with_tail).is_err());
            }
        }
    }
}
