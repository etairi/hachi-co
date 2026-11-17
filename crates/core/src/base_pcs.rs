//! The interface to the base polynomial commitment scheme of the collaborative sumcheck
//! (`5_Building_Blocks.tex`:238-273, 346-365), and the conversions around it.
//!
//! Each prover commits, on its own, to three polynomials before the sumcheck's coins and opens
//! them at the sumcheck's point $\mathbf a$ after the last round (the authors' answers of
//! 2026-10-21, plan `docs/hachi-co-plan.md`, §0):
//! - its share polynomial $\langle \widetilde w \rangle\_p$, multilinear in $\ell''$ variables
//!   and defined by its values on the hypercube (`5_Building_Blocks.tex`:252-259): the values are
//!   converted to monomial coefficients ([`mobius_ext`]), committed with
//!   [`BasePcs::commit_multilinear_ext`] and opened as a multilinear evaluation
//!   ([`BasePcs::prove_multilinear_ext`]). Its pad puts one entry in $\mathbb F\_{q^k}$, so the
//!   coefficients lie there; a polynomial over $\mathbb Z\_q$ has the cheaper
//!   [`BasePcs::commit_multilinear`] and [`BasePcs::prove_multilinear`];
//! - its blinders $g\_{0,p}$ and $g\_{1,p}$ (lines 238-243): their coefficient vectors over
//!   $\mathbb F\_{q^k}$ are committed with [`BasePcs::commit_vector`] and opened as inner products
//!   with public weights ([`BasePcs::prove_inner_product`]).
//!
//! The trait is implemented by `ClearPcs` (feature `fixtures`), a reference stand-in that
//! reveals the committed vector, and by an adapter around Hachi.
//!
//! **Merged commitments**, optional ([`BasePcs::merges`]): a scheme may instead commit to the
//! three polynomials of a prover in one commitment and open them with one proof, which still gives
//! the three values. This is not the draft's protocol, which commits and opens them separately,
//! and the methods default to "not supported".
//!
//! **Party-level steps**, optional ([`BasePcs::finish_openings`], [`BasePcs::verify_openings`]):
//! after a prover's three openings, a scheme may prove something about them together, for example
//! one proof that ends all three (Hachi with a tail, `hachi_co::HachiBase`), and its verifier then
//! checks the three openings together. By default the three proofs are encoded and checked one by
//! one, as the draft has them.

use crate::data::check_len;
use crate::{Error, Result};
use core::fmt::Debug;
use hachi_co_math::{Ext, Fqk, Zq};
use rayon::prelude::*;

/// A base polynomial commitment scheme over $\mathbb F\_{q^K}$, as the collaborative sumcheck
/// uses it (`5_Building_Blocks.tex`:238-273, 346-365): every prover commits on its own, before
/// the sumcheck's coins, and opens at the public point $\mathbf a$ (or against public weights
/// derived from it) after the last round; the verifier checks every opening against the final
/// value the provers announced.
///
/// **Coefficient convention.** A multilinear polynomial $f \in \mathbb Z\_q\[X\_1, \dots,
/// X\_\ell\]$ is given by its $2^\ell$ monomial coefficients: `coeffs[i]` multiplies
/// $X\_1^{i\_1} \cdots X\_\ell^{i\_\ell}$, where $(i\_1, \dots, i\_\ell) = \mathsf{bit}(i)$ is
/// MSB-first, so $X\_1$ goes with the most significant bit of $i$:
/// $$f(\mathbf a) = \sum\_{i \lt 2^\ell} \mathtt{coeffs}\[i\] \prod\_{j=1}^{\ell} a\_j^{i\_j}$$
/// ([`evaluate_multilinear`]). This is the `nat` convention of Hachi-Co's hypercube
/// (`ref/README.md`, "Hypercube"; `4_preliminaries.tex`:44): the point $\mathbf x \in \lbrace 0,
/// 1 \rbrace^\ell$ has index $\mathrm{nat}(\mathbf x) = \sum\_j x\_j 2^{\ell - j}$, and
/// [`mobius`] turns the table of the values $f(\mathbf x)$, at index $\mathrm{nat}(\mathbf x)$,
/// into these coefficients. A vector $\mathbf v \in \mathbb F\_{q^K}^{len}$ is opened against
/// public weights $\mathbf e$ as $\langle \mathbf v, \mathbf e \rangle = \sum\_j v\_j e\_j$
/// ([`inner_product`]). Residues are canonical, in $\[0, q)$; a field element is its $K$
/// coordinates in Hachi's basis ([`Ext`]).
///
/// **Contexts.** Every opening is made and checked under a 32-byte `context`. The caller
/// derives it from its Fiat-Shamir transcript ([`Challenger::context`](crate::Challenger::context))
/// with a label that names the instance (in the sumcheck: the party and the polynomial), so that
/// the prover and the verifier pass the same value, and the openings of different instances, or
/// of different proofs, have different contexts. A scheme with its own Fiat-Shamir transcript
/// starts it from the context (before absorbing the commitment, the point or the weights and the
/// value), which binds each opening proof to its instance and to the outer transcript: a proof
/// made under one context must not verify under another. A scheme without coins does not need
/// it for soundness; `ClearPcs` checks it anyway, so that the tests catch a context mismatch.
///
/// **Randomness.** The provers' methods take a secret 32-byte `seed` for the scheme's own
/// randomness (a hiding commitment's, `docs/hachi-hiding.md` 6.5): each commitment and each opening
/// needs a fresh seed, which the sumcheck draws from
/// [`PartyRandomness::base_pcs`](crate::PartyRandomness::base_pcs). A deterministic scheme, such
/// as `ClearPcs` or Hachi without hiding, ignores it.
///
/// **Words.** Commitments and proofs travel as `u64` words: through
/// [`hachi_co_mpc::Comm::all_gather`], in the Fiat-Shamir transcript (the commitments) and in
/// the proof encoding. The encodings belong to an instance of the scheme, so that they can use
/// its parameters (Hachi packs residues in $\lceil \log\_2 q \rceil$ bits). On one instance,
/// `commitment_from_words` and `proof_from_words` must give back every commitment and proof the
/// instance makes from its encoding, accept at most one encoding of any value (whatever they
/// decode re-encodes to the same words), and return an error, never panic, on any input they
/// cannot decode: the verifier decodes words it reads from a proof. A decoder may accept values
/// of a shape that no statement has (Hachi's checks only the lengths its header implies); the
/// verify methods reject those. The provers publish their
/// commitments and their proofs with one `all_gather` call each, which needs the same length at
/// every party: the number of words of a commitment or of a proof must depend only on the size
/// of the committed polynomial (as for `ClearPcs`).
///
/// **Errors.** The provers' methods return an [`Error`] (for example [`Error::Shape`] or
/// [`Error::BasePcs`]); the verifier's methods return the reason of a rejection as a `String`,
/// which the caller reports together with the party and the polynomial. No method panics on
/// malformed input.
pub trait BasePcs<const K: usize>: Send + Sync {
    /// A commitment: public, sent to the verifier and absorbed before the sumcheck's coins
    /// (`5_Building_Blocks.tex`:249-250, 260, 273).
    type Commitment: Clone + Debug + PartialEq + Eq + Send + Sync;
    /// What a prover keeps after committing, to open later (private). Hachi-Co opens each state
    /// once, and a hiding scheme may refuse a second opening, as Hachi's masked states do.
    type State: Send + Sync;
    /// An opening proof (`5_Building_Blocks.tex`:348-350).
    type Proof: Clone + Debug + PartialEq + Eq + Send + Sync;

    /// Commits to the multilinear polynomial with the monomial coefficients `coeffs`
    /// ($2^\ell$ residues of $\mathbb Z\_q$, in the convention of the trait), with the secret
    /// `seed` (see "Randomness").
    ///
    /// # Errors
    /// Implementation-specific, for example [`Error::Shape`] unless the length is a power of
    /// two.
    fn commit_multilinear(
        &self,
        coeffs: &[u64],
        seed: &[u8; 32],
    ) -> Result<(Self::Commitment, Self::State)>;

    /// Commits to the vector `v` over $\mathbb F\_{q^K}$ (a blinder's coefficients), with the
    /// secret `seed`.
    ///
    /// # Errors
    /// Implementation-specific.
    fn commit_vector(
        &self,
        v: &[Ext<K>],
        seed: &[u8; 32],
    ) -> Result<(Self::Commitment, Self::State)>;

    /// Commits to the multilinear polynomial with the monomial coefficients `coeffs`
    /// ($2^\ell$ elements of $\mathbb F\_{q^K}$, in the convention of the trait), with the secret
    /// `seed`.
    ///
    /// # Errors
    /// Implementation-specific, for example [`Error::Shape`] unless the length is a power of
    /// two.
    fn commit_multilinear_ext(
        &self,
        coeffs: &[Ext<K>],
        seed: &[u8; 32],
    ) -> Result<(Self::Commitment, Self::State)>;

    /// [`commit_multilinear_ext`](Self::commit_multilinear_ext) with the coefficients handed over,
    /// so that an implementation may free them while it commits. The commitment and the state are
    /// those of [`commit_multilinear_ext`](Self::commit_multilinear_ext); by default it commits
    /// from a borrow.
    ///
    /// # Errors
    /// As [`commit_multilinear_ext`](Self::commit_multilinear_ext).
    fn commit_multilinear_ext_owned(
        &self,
        coeffs: Vec<Ext<K>>,
        seed: &[u8; 32],
    ) -> Result<(Self::Commitment, Self::State)> {
        self.commit_multilinear_ext(&coeffs, seed)
    }

    /// Lets the state of [`commit_multilinear_ext`](Self::commit_multilinear_ext) drop what it can
    /// form again from the committed coefficients, until its opening by
    /// [`prove_multilinear_ext_rebuilt`](Self::prove_multilinear_ext_rebuilt), which is then the
    /// only way to open it. Returns whether it did; by default nothing is dropped.
    fn shed(&self, st: &mut Self::State) -> bool {
        let _ = st;
        false
    }

    /// [`prove_multilinear_ext_owned`](Self::prove_multilinear_ext_owned) on a state that
    /// [`shed`](Self::shed) may have reduced: `coeffs` gives the committed coefficients again,
    /// and is called only if the state needs them. The value and the proof are those of
    /// [`prove_multilinear_ext`](Self::prove_multilinear_ext) on the whole state; by default
    /// `coeffs` is not called.
    ///
    /// # Errors
    /// As [`prove_multilinear_ext`](Self::prove_multilinear_ext), those of `coeffs`, and
    /// implementation-specific ones for coefficients other than the committed ones.
    fn prove_multilinear_ext_rebuilt(
        &self,
        st: Self::State,
        coeffs: impl FnOnce() -> Result<Vec<Ext<K>>>,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> Result<(Ext<K>, Self::Proof)> {
        let _ = coeffs;
        self.prove_multilinear_ext_owned(st, point, context, seed)
    }

    /// [`commit_multilinear_ext`](Self::commit_multilinear_ext) on the coefficients `coeffs`
    /// given as [`LiftedCoeffs`], which an implementation may read one at a time instead of
    /// holding them over the field. The commitment is that of [`LiftedCoeffs::to_vec`], and the
    /// state opens by [`prove_multilinear_ext_lifted`](Self::prove_multilinear_ext_lifted); by
    /// default the coefficients are formed and committed.
    ///
    /// # Errors
    /// As [`commit_multilinear_ext`](Self::commit_multilinear_ext).
    fn commit_multilinear_ext_lifted(
        &self,
        coeffs: &LiftedCoeffs<K>,
        seed: &[u8; 32],
    ) -> Result<(Self::Commitment, Self::State)> {
        self.commit_multilinear_ext_owned(coeffs.to_vec(), seed)
    }

    /// [`commit_multilinear_ext_lifted`](Self::commit_multilinear_ext_lifted) on coefficients
    /// formed one part at a time ([`PartedCoeffs`]), which an implementation may commit part by
    /// part without holding them all. The commitment is that of the coefficients, and the state
    /// opens by [`prove_multilinear_ext_lifted`](Self::prove_multilinear_ext_lifted) given them as
    /// [`LiftedCoeffs`] (whose [`LiftedCoeffs::fingerprint`] is `coeffs.fingerprint()`); by
    /// default every part is formed and read, then committed.
    ///
    /// # Errors
    /// As [`commit_multilinear_ext`](Self::commit_multilinear_ext); those of `coeffs.prepare`.
    fn commit_multilinear_ext_parts(
        &self,
        coeffs: &PartedCoeffs<'_, K>,
        seed: &[u8; 32],
    ) -> Result<(Self::Commitment, Self::State)> {
        let n = 1usize << coeffs.vars;
        let per = n / coeffs.parts;
        let mut all = vec![Ext([0u64; K]); n];
        for (p, part) in all.chunks_mut(per).enumerate() {
            (coeffs.prepare)(p)?;
            (coeffs.read)(p * per, part);
        }
        self.commit_multilinear_ext_owned(all, seed)
    }

    /// [`prove_multilinear_ext_rebuilt`](Self::prove_multilinear_ext_rebuilt) on a state of
    /// [`commit_multilinear_ext_lifted`](Self::commit_multilinear_ext_lifted), given the same
    /// coefficients again. The value and the proof are those of
    /// [`prove_multilinear_ext`](Self::prove_multilinear_ext) on the state of
    /// [`commit_multilinear_ext`](Self::commit_multilinear_ext); by default the coefficients are
    /// formed if the state needs them.
    ///
    /// # Errors
    /// As [`prove_multilinear_ext_rebuilt`](Self::prove_multilinear_ext_rebuilt).
    fn prove_multilinear_ext_lifted(
        &self,
        st: Self::State,
        coeffs: LiftedCoeffs<K>,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> Result<(Ext<K>, Self::Proof)> {
        self.prove_multilinear_ext_rebuilt(st, move || Ok(coeffs.to_vec()), point, context, seed)
    }

    /// [`prove_multilinear_ext_lifted`](Self::prove_multilinear_ext_lifted) given the committed
    /// coefficients in parts ([`PartedCoeffs`], whose `fingerprint()` is the commitment's), which
    /// an implementation may read one part at a time. The value and the proof are those of
    /// [`prove_multilinear_ext`](Self::prove_multilinear_ext) on the state; by default every part
    /// is formed and read, then the state is opened with them.
    ///
    /// # Errors
    /// As [`prove_multilinear_ext_lifted`](Self::prove_multilinear_ext_lifted); those of
    /// `coeffs.prepare`.
    fn prove_multilinear_ext_parts(
        &self,
        st: Self::State,
        coeffs: &PartedCoeffs<'_, K>,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> Result<(Ext<K>, Self::Proof)> {
        let n = 1usize << coeffs.vars;
        let per = n / coeffs.parts;
        let mut all = vec![Ext([0u64; K]); n];
        for (p, part) in all.chunks_mut(per).enumerate() {
            (coeffs.prepare)(p)?;
            (coeffs.read)(p * per, part);
        }
        self.prove_multilinear_ext_rebuilt(st, move || Ok(all), point, context, seed)
    }

    /// Opens a commitment of [`commit_multilinear`](Self::commit_multilinear) at `point`
    /// ($\ell$ coordinates) under `context`, with the secret `seed`: returns $f(\mathbf a)$ and
    /// a proof of it.
    ///
    /// # Errors
    /// Implementation-specific, for example a state of a vector commitment or a point of the
    /// wrong length.
    fn prove_multilinear(
        &self,
        st: &Self::State,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> Result<(Ext<K>, Self::Proof)>;

    /// Opens a commitment of [`commit_multilinear_ext`](Self::commit_multilinear_ext) at `point`
    /// ($\ell$ coordinates) under `context`, with the secret `seed`: returns $f(\mathbf a)$ and
    /// a proof of it.
    ///
    /// # Errors
    /// Implementation-specific, for example a state of another kind of commitment or a point of
    /// the wrong length.
    fn prove_multilinear_ext(
        &self,
        st: &Self::State,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> Result<(Ext<K>, Self::Proof)>;

    /// [`prove_multilinear_ext`](Self::prove_multilinear_ext) on a state that the caller does not
    /// need afterwards, so that an implementation may free parts of it while it proves. The value
    /// and the proof are those of [`prove_multilinear_ext`](Self::prove_multilinear_ext); by
    /// default it proves from a borrow of the state.
    ///
    /// # Errors
    /// As [`prove_multilinear_ext`](Self::prove_multilinear_ext).
    fn prove_multilinear_ext_owned(
        &self,
        st: Self::State,
        point: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> Result<(Ext<K>, Self::Proof)> {
        self.prove_multilinear_ext(&st, point, context, seed)
    }

    /// Opens a commitment of [`commit_vector`](Self::commit_vector) against `weights` (one per
    /// entry) under `context`, with the secret `seed`: returns $\langle \mathbf v, \mathbf e
    /// \rangle$ and a proof of it.
    ///
    /// # Errors
    /// Implementation-specific, for example a state of a multilinear commitment or weights of
    /// the wrong length.
    fn prove_inner_product(
        &self,
        st: &Self::State,
        weights: &[Ext<K>],
        context: &[u8; 32],
        seed: &[u8; 32],
    ) -> Result<(Ext<K>, Self::Proof)>;

    /// Checks that `proof` opens `com`, a commitment to a multilinear polynomial in `num_vars`
    /// variables, to `value` at `point`, under `context`.
    ///
    /// # Errors
    /// The reason of the rejection.
    fn verify_multilinear(
        &self,
        com: &Self::Commitment,
        num_vars: usize,
        point: &[Ext<K>],
        value: &Ext<K>,
        proof: &Self::Proof,
        context: &[u8; 32],
    ) -> core::result::Result<(), String>;

    /// Checks that `proof` opens `com`, a commitment to a multilinear polynomial over
    /// $\mathbb F\_{q^K}$ in `num_vars` variables, to `value` at `point`, under `context`.
    ///
    /// # Errors
    /// The reason of the rejection.
    fn verify_multilinear_ext(
        &self,
        com: &Self::Commitment,
        num_vars: usize,
        point: &[Ext<K>],
        value: &Ext<K>,
        proof: &Self::Proof,
        context: &[u8; 32],
    ) -> core::result::Result<(), String>;

    /// Checks that `proof` opens `com`, a commitment to a vector of `len` elements, to the
    /// inner product `value` with `weights`, under `context`.
    ///
    /// # Errors
    /// The reason of the rejection.
    fn verify_inner_product(
        &self,
        com: &Self::Commitment,
        len: usize,
        weights: &[Ext<K>],
        value: &Ext<K>,
        proof: &Self::Proof,
        context: &[u8; 32],
    ) -> core::result::Result<(), String>;

    /// Whether each prover commits to its three polynomials in one commitment
    /// ([`commit_merged`](Self::commit_merged)) and opens them with one proof, in place of the
    /// draft's three commitments and three openings. `false` unless the scheme supports and
    /// selects it. The choice must enter [`params_digest`](Self::params_digest), so that a proof
    /// is bound to it.
    fn merges(&self) -> bool {
        false
    }

    /// Commits, in one commitment, to the multilinear polynomial over $\mathbb F\_{q^K}$ with the
    /// monomial coefficients `coeffs` and to the two blinder vectors `g0` and `g1`, with the secret
    /// `seed`.
    ///
    /// # Errors
    /// [`Error::BasePcs`] for a scheme that does not merge, and implementation-specific errors
    /// otherwise.
    fn commit_merged(
        &self,
        _coeffs: &[Ext<K>],
        _g0: &[Ext<K>],
        _g1: &[Ext<K>],
        _seed: &[u8; 32],
    ) -> Result<(Self::Commitment, Self::State)> {
        Err(Error::BasePcs(
            "this base PCS does not merge commitments".into(),
        ))
    }

    /// Opens a commitment of [`commit_merged`](Self::commit_merged): the polynomial at `point` and
    /// the vectors against `weights0` and `weights1`, under `context`, with the secret `seed`.
    /// Returns the three values, in the order polynomial, `g0`, `g1`, and one proof of them.
    ///
    /// # Errors
    /// [`Error::BasePcs`] for a scheme that does not merge, and implementation-specific errors
    /// otherwise.
    fn prove_merged(
        &self,
        _st: &Self::State,
        _point: &[Ext<K>],
        _weights0: &[Ext<K>],
        _weights1: &[Ext<K>],
        _context: &[u8; 32],
        _seed: &[u8; 32],
    ) -> Result<([Ext<K>; 3], Self::Proof)> {
        Err(Error::BasePcs(
            "this base PCS does not merge commitments".into(),
        ))
    }

    /// Checks that `proof` opens `com`, a merged commitment to a polynomial in `num_vars`
    /// variables and two vectors of the lengths of the weights, to `values`: the polynomial's
    /// value at `point` and the vectors' inner products with `weights0` and `weights1`.
    ///
    /// # Errors
    /// The reason of the rejection, also for a scheme that does not merge.
    #[allow(clippy::too_many_arguments)] // one opening's instance
    fn verify_merged(
        &self,
        _com: &Self::Commitment,
        _num_vars: usize,
        _point: &[Ext<K>],
        _weights0: &[Ext<K>],
        _weights1: &[Ext<K>],
        _values: &[Ext<K>; 3],
        _proof: &Self::Proof,
        _context: &[u8; 32],
    ) -> core::result::Result<(), String> {
        Err("this base PCS does not merge commitments".into())
    }

    /// The party-level step after a prover's three openings (see "Party-level steps"): their
    /// proofs, in the order $\langle \widetilde w \rangle\_p$, $g\_{0,p}$, $g\_{1,p}$, become the
    /// words the prover publishes for them, in the same order. Each item must still have the
    /// length the sizes of the polynomials fix (see "Words"). By default each proof is encoded on
    /// its own ([`proof_to_words`](Self::proof_to_words)).
    ///
    /// # Errors
    /// Implementation-specific, for example a proof that does not belong to this prover's
    /// openings.
    fn finish_openings(&self, proofs: [Self::Proof; 3]) -> Result<[Vec<u64>; 3]> {
        Ok(proofs.map(|p| self.proof_to_words(&p)))
    }

    /// The party-level check of a prover's three openings (see "Party-level steps"), in the
    /// order $\langle \widetilde w \rangle\_p$, $g\_{0,p}$, $g\_{1,p}$: the verdict of each, in the
    /// same order. By default each is checked on its own ([`check_opening_words`]). A scheme that
    /// checks the three together must reject each opening whose claim it could not check.
    fn verify_openings(
        &self,
        openings: [OpeningWords<'_, K>; 3],
    ) -> [core::result::Result<(), String>; 3] {
        openings.map(|opening| check_opening_words(self, &opening))
    }

    /// A digest of the scheme's own public parameters. Hachi-Co absorbs it into the statement
    /// of every proof (`6_Collaborative_Hachi.tex`:294-297: $\mathsf{pp}\_{\mathsf{PCS}}$ is part
    /// of $\mathsf{pp}$), so that a proof is bound to the base PCS it was made with. Two
    /// instances with different parameters must have different digests.
    fn params_digest(&self) -> [u8; 32];

    /// The words of a commitment (see "Words").
    fn commitment_to_words(&self, com: &Self::Commitment) -> Vec<u64>;

    /// The commitment of `words`, the inverse of
    /// [`commitment_to_words`](Self::commitment_to_words).
    ///
    /// # Errors
    /// For any input that `commitment_to_words` does not produce.
    fn commitment_from_words(&self, words: &[u64]) -> Result<Self::Commitment>;

    /// The words of a proof (see "Words").
    fn proof_to_words(&self, p: &Self::Proof) -> Vec<u64>;

    /// The proof of `words`, the inverse of [`proof_to_words`](Self::proof_to_words).
    ///
    /// # Errors
    /// For any input that `proof_to_words` does not produce.
    fn proof_from_words(&self, words: &[u64]) -> Result<Self::Proof>;
}

/// What one opening of a party proves, for [`BasePcs::verify_openings`]: the verify method that
/// checks it, with its point or weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpeningKind<'a, const K: usize> {
    /// A polynomial over $\mathbb Z\_q$ at the point ([`BasePcs::verify_multilinear`]).
    Multilinear(&'a [Ext<K>]),
    /// A polynomial over $\mathbb F\_{q^K}$ at the point ([`BasePcs::verify_multilinear_ext`]).
    MultilinearExt(&'a [Ext<K>]),
    /// A vector against the weights ([`BasePcs::verify_inner_product`]).
    InnerProduct(&'a [Ext<K>]),
}

/// One opening of a party as the verifier received it, for [`BasePcs::verify_openings`]: the
/// words of its commitment and of its proof, what it proves, the value and the context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpeningWords<'a, const K: usize> {
    /// The words of the commitment.
    pub commitment: &'a [u64],
    /// The words of the proof.
    pub proof: &'a [u64],
    /// What the opening proves.
    pub kind: OpeningKind<'a, K>,
    /// The value it must prove.
    pub value: &'a Ext<K>,
    /// The opening's context (see "Contexts").
    pub context: &'a [u8; 32],
}

/// One opening checked on its own, as [`BasePcs::verify_openings`] checks each by default: decode
/// the commitment and the proof ([`BasePcs::commitment_from_words`],
/// [`BasePcs::proof_from_words`]), then the verify method of its kind.
///
/// # Errors
/// The reason of the rejection: a commitment or a proof that does not decode, or the verify
/// method's.
pub fn check_opening_words<const K: usize, P: BasePcs<K> + ?Sized>(
    pcs: &P,
    opening: &OpeningWords<'_, K>,
) -> core::result::Result<(), String> {
    let com = pcs
        .commitment_from_words(opening.commitment)
        .map_err(|e| format!("the commitment does not decode: {e}"))?;
    let proof = pcs
        .proof_from_words(opening.proof)
        .map_err(|e| format!("the proof does not decode: {e}"))?;
    let (value, context) = (opening.value, opening.context);
    match opening.kind {
        OpeningKind::Multilinear(point) => {
            pcs.verify_multilinear(&com, point.len(), point, value, &proof, context)
        }
        OpeningKind::MultilinearExt(point) => {
            pcs.verify_multilinear_ext(&com, point.len(), point, value, &proof, context)
        }
        OpeningKind::InnerProduct(weights) => {
            pcs.verify_inner_product(&com, weights.len(), weights, value, &proof, context)
        }
    }
}

/// Checks that a table of a multilinear polynomial has $2^\ell$ entries for some $\ell \ge 0$.
fn check_power_of_two(what: &'static str, len: usize) -> Result<()> {
    if !len.is_power_of_two() {
        return Err(Error::Shape {
            what,
            expected: len.checked_next_power_of_two().unwrap_or(0),
            actual: len,
        });
    }
    Ok(())
}

/// One butterfly pass per variable: `op(low, high)` replaces `high` for every pair of entries
/// that differ only in one bit, the entry with the bit unset being `low`.
fn butterflies<T: Copy>(table: &mut [T], op: impl Fn(T, T) -> T) {
    let mut half = 1;
    while half < table.len() {
        for chunk in table.chunks_exact_mut(2 * half) {
            let (low, high) = chunk.split_at_mut(half);
            for (l, h) in low.iter().zip(high.iter_mut()) {
                *h = op(*l, *h);
            }
        }
        half *= 2;
    }
}

/// The entries of a block of [`butterflies_par`]: its passes run within blocks of this many
/// entries (128 KiB of residues), so that they stay in a core's cache.
const BUTTERFLY_BLOCK: usize = 1 << 14;

/// The passes over higher bits that [`butterflies_par`] runs in one pass over the table: at 2^27
/// residues on the shared instance, three took 0.63 to 0.72 s against 0.86 to 0.94 s one at a
/// time, and four (16 rows at once) 0.90 to 0.94 s.
const FUSED_LEVELS: u32 = 3;

/// [`butterflies`] on the Rayon pool: the passes over the bits below [`BUTTERFLY_BLOCK`] pair
/// entries of the same block, so they run block by block, the blocks in parallel, each as
/// [`butterflies`] runs them. The higher passes then run [`FUSED_LEVELS`] at a time
/// ([`fused_passes`]), so that a table larger than the caches is read and written once for every
/// [`FUSED_LEVELS`] passes instead of once per pass. Every pair gets the same operation after the
/// same passes, so the table is the same.
fn butterflies_par<T: Copy + Send + Sync>(table: &mut [T], op: impl Fn(T, T) -> T + Sync) {
    let n = table.len();
    let block = BUTTERFLY_BLOCK.min(n).max(1);
    table
        .par_chunks_mut(block)
        .for_each(|chunk| butterflies(chunk, &op));
    let mut half = block;
    while half < n {
        let levels = (n / half).trailing_zeros().min(FUSED_LEVELS);
        let span = half << levels;
        table.par_chunks_mut(span).for_each(|group| match levels {
            1 => fused_passes::<T, 2>(group, half, &op),
            2 => fused_passes::<T, 4>(group, half, &op),
            _ => fused_passes::<T, 8>(group, half, &op),
        });
        half = span;
    }
}

/// The passes over the bits of `half`, `2 half`, ... (`R = 2^levels` rows) on `group`, whose `R`
/// rows of `half` entries each hold the entries that these passes pair: the entries of a column
/// go through the passes in order, as [`butterflies`] runs them (the pass over the bit of
/// `half · step` pairs rows `r` and `r + step`, for the `r` without that bit). The columns run
/// in parallel, by blocks of [`BUTTERFLY_BLOCK`].
fn fused_passes<T: Copy + Send + Sync, const R: usize>(
    group: &mut [T],
    half: usize,
    op: &(impl Fn(T, T) -> T + Sync),
) {
    let cols = half.div_ceil(BUTTERFLY_BLOCK);
    let mut tasks: Vec<Vec<&mut [T]>> = (0..cols).map(|_| Vec::with_capacity(R)).collect();
    for row in group.chunks_mut(half) {
        for (task, part) in tasks.iter_mut().zip(row.chunks_mut(BUTTERFLY_BLOCK)) {
            task.push(part);
        }
    }
    tasks.into_par_iter().for_each(|mut rows| {
        debug_assert_eq!(rows.len(), R, "a group has R rows");
        let len = rows[0].len();
        for o in 0..len {
            let mut v: [T; R] = core::array::from_fn(|r| rows[r][o]);
            let mut step = 1;
            while step < R {
                for r in 0..R {
                    if r & step == 0 {
                        v[r + step] = op(v[r], v[r + step]);
                    }
                }
                step <<= 1;
            }
            for (row, x) in rows.iter_mut().zip(v) {
                row[o] = x;
            }
        }
    });
}

/// The Möbius transform over $\mathbb Z\_q$, in place: the values $f(\mathbf x)$ of a multilinear
/// polynomial $f$ on $\lbrace 0, 1 \rbrace^\ell$, at index $\mathrm{nat}(\mathbf x)$, become its
/// monomial coefficients in the convention of [`BasePcs`]. With $S(i)$ the set of positions of
/// the bits of $i$,
/// $$c\_i = \sum\_{x \,:\, S(x) \subseteq S(i)} (-1)^{\lvert S(i) \rvert - \lvert S(x) \rvert} f(x),$$
/// the inverse of $f(x) = \sum\_{i \,:\, S(i) \subseteq S(x)} c\_i$ ([`zeta`]): the monomial
/// $\prod\_j X\_j^{i\_j}$ is 1 at a point of the hypercube if every variable in it is 1 there,
/// and 0 otherwise. The bits of the index play the same role in values and coefficients, so the
/// MSB-first convention carries over. $\ell 2^{\ell - 1}$ subtractions; every entry is reduced
/// modulo $q$. Hachi-Co converts the table of $\langle \widetilde w \rangle\_p$
/// (`5_Building_Blocks.tex`:252-259) with it before committing (the authors' answer 2 of
/// 2026-10-21, plan `docs/hachi-co-plan.md`, §0).
///
/// # Errors
/// [`Error::Shape`] unless the length is a power of two.
pub fn mobius(zq: &Zq, table: &mut [u64]) -> Result<()> {
    check_power_of_two("hypercube table (2^l entries)", table.len())?;
    table.par_iter_mut().for_each(|x| *x = zq.reduce(*x));
    butterflies_par(table, |low, high| zq.sub(high, low));
    Ok(())
}

/// The Möbius transform over $\mathbb F\_{q^K}$, in place: [`mobius`] for a table with values in
/// the field. Every entry is reduced.
///
/// # Errors
/// [`Error::Shape`] unless the length is a power of two.
pub fn mobius_ext<const K: usize>(field: &Fqk<K>, table: &mut [Ext<K>]) -> Result<()> {
    check_power_of_two("hypercube table (2^l entries)", table.len())?;
    table.par_iter_mut().for_each(|x| *x = field.reduce(x));
    butterflies_par(table, |low, high| field.sub(&high, &low));
    Ok(())
}

/// The monomial coefficients of a multilinear polynomial over $\mathbb F\_{q^K}$ whose values on
/// the hypercube lie in $\mathbb Z\_q$ but at one entry, the pad: the Möbius transform over
/// $\mathbb Z\_q$ of the base values ([`mobius`], with 0 at the pad's entry) and the pad's index
/// $p$ and value $v$. Since the transform is linear, coefficient $i$ is the base coefficient
/// lifted to the field, plus $(-1)^{\lvert S(i) \rvert - \lvert S(p) \rvert} v$ when
/// $S(p) \subseteq S(i)$ (the Möbius transform of $v$ at $p$): the coefficients of [`mobius_ext`]
/// on the table over the field, held in a quarter of the memory at $K = 4$ and formed one at a time.
#[derive(Clone, Debug)]
pub struct LiftedCoeffs<const K: usize> {
    field: Fqk<K>,
    /// The Möbius coefficients of the base values.
    base: Vec<u64>,
    /// The pad's index and value, reduced.
    pad: Option<(usize, Ext<K>)>,
    /// The table's entry at the pad's index, reduced, which [`LiftedCoeffs::from_table`] set to
    /// 0 and [`LiftedCoeffs::into_table`] puts back (0 without a pad).
    replaced: u64,
}

/// The monomial coefficients of a multilinear polynomial in `vars` variables over the field,
/// formed in `parts` parts of $2^{\mathtt{vars}} / \mathtt{parts}$ consecutive coefficients, for
/// [`BasePcs::commit_multilinear_ext_parts`]: `prepare(p)` forms part $p$ (called for $p = 0, 1,
/// \dots$ in order), after which `read(start, out)` writes coefficients of that part into `out`;
/// `fingerprint()`, once every part was formed, is [`LiftedCoeffs::fingerprint`] of the
/// coefficients.
pub struct PartedCoeffs<'a, const K: usize> {
    /// The number of variables.
    pub vars: usize,
    /// The number of parts, a power of two.
    pub parts: usize,
    /// Forms part `p`.
    pub prepare: &'a (dyn Fn(usize) -> Result<()> + Sync),
    /// Coefficients `start .. start + out.len()` of the part formed last.
    pub read: &'a (dyn Fn(usize, &mut [Ext<K>]) + Sync),
    /// The fingerprint of all the coefficients.
    pub fingerprint: &'a (dyn Fn() -> u64 + Sync),
}

impl<const K: usize> LiftedCoeffs<K> {
    /// The coefficients of the hypercube table that is `table` (values in $\mathbb Z\_q$) with the
    /// field element `pad.1` at the index `pad.0`, which `table`'s entry there does not count for
    /// (as when the pad replaces it): `table` becomes the Möbius transform of its base values in
    /// place.
    ///
    /// # Errors
    /// [`Error::Shape`] unless the length is a power of two and the pad's index lies in it.
    pub fn from_table(
        field: &Fqk<K>,
        mut table: Vec<u64>,
        pad: Option<(usize, Ext<K>)>,
    ) -> Result<Self> {
        let mut replaced = 0;
        if let Some((at, _)) = pad {
            let len = table.len();
            let entry = table.get_mut(at).ok_or(Error::Shape {
                what: "hypercube table holding the pad's index",
                expected: at + 1,
                actual: len,
            })?;
            replaced = field.zq().reduce(*entry);
            *entry = 0;
        }
        mobius(field.zq(), &mut table)?;
        Ok(Self {
            field: field.clone(),
            base: table,
            pad: pad.map(|(at, value)| (at, field.reduce(&value))),
            replaced,
        })
    }

    /// The table given to [`LiftedCoeffs::from_table`], every entry reduced modulo $q$: the zeta
    /// transform of the base coefficients ([`zeta`], the inverse of [`mobius`]) with the entry at
    /// the pad's index put back. A caller that keeps its table (reduced) can lend it for the
    /// coefficients instead of copying it.
    ///
    /// # Errors
    /// None for coefficients of [`LiftedCoeffs::from_table`] (a power-of-two length).
    pub fn into_table(self) -> Result<Vec<u64>> {
        let mut table = self.base;
        zeta(self.field.zq(), &mut table)?;
        if let Some((at, _)) = self.pad {
            table[at] = self.replaced;
        }
        Ok(table)
    }

    /// The number of coefficients, $2^\ell$.
    pub fn len(&self) -> usize {
        self.base.len()
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.base.is_empty()
    }

    /// Coefficient `i` (reduced).
    pub fn coeff(&self, i: usize) -> Ext<K> {
        let f = &self.field;
        let x = f.from_base(self.base[i]);
        match self.pad {
            Some((p, v)) if i & p == p => {
                if (i ^ p).count_ones().is_multiple_of(2) {
                    f.add(&x, &v)
                } else {
                    f.sub(&x, &v)
                }
            }
            _ => x,
        }
    }

    /// Coefficients `start .. start + out.len()` into `out`.
    pub fn coeffs_into(&self, start: usize, out: &mut [Ext<K>]) {
        for (i, o) in (start..).zip(out.iter_mut()) {
            *o = self.coeff(i);
        }
    }

    /// All the coefficients, those of [`mobius_ext`] on the table over the field.
    pub fn to_vec(&self) -> Vec<Ext<K>> {
        (0..self.len()).map(|i| self.coeff(i)).collect()
    }

    /// A fingerprint of the coefficients (FNV-1a over the base coefficients and the pad), against
    /// which an opening checks that it was given those of the commitment. Not a cryptographic
    /// binding (the commitment is).
    pub fn fingerprint(&self) -> u64 {
        const PRIME: u64 = 0x0000_0100_0000_01b3;
        let fnv = |h: u64, w: u64| (h ^ w).wrapping_mul(PRIME);
        // The hashes of the chunks in parallel, folded in order: the same value.
        let mut h = self
            .base
            .par_chunks(1 << 16)
            .map(|chunk| {
                chunk
                    .iter()
                    .fold(0xcbf2_9ce4_8422_2325u64, |h, x| fnv(h, *x))
            })
            .collect::<Vec<u64>>()
            .into_iter()
            .fold(0xcbf2_9ce4_8422_2325u64, fnv);
        h = fnv(h, self.len() as u64);
        if let Some((at, value)) = self.pad {
            h = value
                .0
                .iter()
                .fold(fnv(h, at as u64 + 1), |h, x| fnv(h, *x));
        }
        h
    }
}

/// The zeta transform over $\mathbb Z\_q$, in place: the inverse of [`mobius`], from the
/// monomial coefficients of a multilinear polynomial to its values on the hypercube,
/// $f(x) = \sum\_{i \,:\, S(i) \subseteq S(x)} c\_i$. Every entry is reduced modulo $q$. On the
/// Rayon pool, as [`mobius`].
///
/// # Errors
/// [`Error::Shape`] unless the length is a power of two.
pub fn zeta(zq: &Zq, table: &mut [u64]) -> Result<()> {
    check_power_of_two("monomial coefficients (2^l entries)", table.len())?;
    table.par_iter_mut().for_each(|x| *x = zq.reduce(*x));
    butterflies_par(table, |low, high| zq.add(high, low));
    Ok(())
}

/// $f(\mathbf a) = \sum\_i \mathtt{coeffs}\[i\] \prod\_j a\_j^{i\_j}$ for the multilinear polynomial
/// with the monomial coefficients `coeffs` (the convention of [`BasePcs`]) at a point of
/// $\mathbb F\_{q^K}^\ell$. Binds $X\_1$ first: $f = f\_0 + X\_1 f\_1$, where $f\_0$ and $f\_1$
/// are given by the first and the second half of the coefficients. $2^\ell$ multiplications.
///
/// # Errors
/// [`Error::Shape`] unless `coeffs` has $2^\ell$ entries for a point of $\ell$ coordinates;
/// [`Error::Overflow`] if $2^\ell$ does not fit in `usize`.
pub fn evaluate_multilinear<const K: usize>(
    field: &Fqk<K>,
    coeffs: &[u64],
    point: &[Ext<K>],
) -> Result<Ext<K>> {
    let size = u32::try_from(point.len())
        .ok()
        .and_then(|e| 1usize.checked_shl(e))
        .ok_or(Error::Overflow("2^l coefficients"))?;
    check_len(
        "multilinear coefficients (2^l for a point of l coordinates)",
        size,
        coeffs.len(),
    )?;
    let Some((first, rest)) = point.split_first() else {
        return Ok(field.from_base(coeffs[0]));
    };
    let (low, high) = coeffs.split_at(size / 2);
    let mut table: Vec<Ext<K>> = low
        .iter()
        .zip(high)
        .map(|(c0, c1)| field.add(&field.from_base(*c0), &field.scale(first, *c1)))
        .collect();
    for a in rest {
        let half = table.len() / 2;
        let (low, high) = table.split_at_mut(half);
        for (l, h) in low.iter_mut().zip(high.iter()) {
            *l = field.add(l, &field.mul(a, h));
        }
        table.truncate(half);
    }
    Ok(table[0])
}

/// [`evaluate_multilinear`] for coefficients in $\mathbb F\_{q^K}$.
///
/// # Errors
/// [`Error::Shape`] unless `coeffs` has $2^\ell$ entries for a point of $\ell$ coordinates;
/// [`Error::Overflow`] if $2^\ell$ does not fit in `usize`.
pub fn evaluate_multilinear_ext<const K: usize>(
    field: &Fqk<K>,
    coeffs: &[Ext<K>],
    point: &[Ext<K>],
) -> Result<Ext<K>> {
    let size = u32::try_from(point.len())
        .ok()
        .and_then(|e| 1usize.checked_shl(e))
        .ok_or(Error::Overflow("2^l coefficients"))?;
    check_len(
        "multilinear coefficients (2^l for a point of l coordinates)",
        size,
        coeffs.len(),
    )?;
    let mut table = coeffs.to_vec();
    for a in point {
        let half = table.len() / 2;
        let (low, high) = table.split_at_mut(half);
        for (l, h) in low.iter_mut().zip(high.iter()) {
            *l = field.add(l, &field.mul(a, h));
        }
        table.truncate(half);
    }
    Ok(table[0])
}

/// $\langle \mathbf v, \mathbf e \rangle = \sum\_j v\_j e\_j$ over $\mathbb F\_{q^K}$.
///
/// # Errors
/// [`Error::Shape`] unless the vectors have the same length.
pub fn inner_product<const K: usize>(
    field: &Fqk<K>,
    v: &[Ext<K>],
    weights: &[Ext<K>],
) -> Result<Ext<K>> {
    check_len(
        "inner product (vectors of equal length)",
        v.len(),
        weights.len(),
    )?;
    Ok(v.iter().zip(weights).fold(field.zero(), |acc, (x, e)| {
        field.add(&acc, &field.mul(x, e))
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The toy-n2 modulus.
    const Q: u64 = 33554501;

    /// A splitmix64 stream (seeded per test).
    struct Stream(u64);

    impl Stream {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn residues(&mut self, len: usize) -> Vec<u64> {
            (0..len).map(|_| self.next() % Q).collect()
        }

        fn ext(&mut self) -> Ext<4> {
            Ext(core::array::from_fn(|_| self.next() % Q))
        }
    }

    /// Bit $j$ ($1$-based, MSB-first) of an index of $\ell$ bits.
    fn bit(i: usize, j: usize, ell: usize) -> bool {
        (i >> (ell - j)) & 1 == 1
    }

    /// The passes on the Rayon pool against the passes in order, for tables below, at and above
    /// the block of [`BUTTERFLY_BLOCK`] entries (so that the passes over the higher bits run),
    /// over Z_q and over the field: the same tables.
    #[test]
    fn the_parallel_butterflies_are_the_butterflies() {
        let q = 15395032342003157u64; // secure-l24
        let zq = Zq::new(q).unwrap();
        let field = Fqk::<4>::new(q).unwrap();
        let mut state = 0x5eed_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // 2^17 to 2^20: three to six passes above the blocks, so groups of 8, 2 and 4 rows.
        for log in [0usize, 1, 5, 14, 15, 16, 17, 18, 19, 20] {
            let table: Vec<u64> = (0..1usize << log).map(|_| next() % q).collect();
            let mut want = table.clone();
            butterflies(&mut want, |low, high| zq.sub(high, low));
            let mut got = table.clone();
            butterflies_par(&mut got, |low, high| zq.sub(high, low));
            assert_eq!(got, want, "2^{log} residues");
            let mut by_mobius = table;
            mobius(&zq, &mut by_mobius).unwrap();
            assert_eq!(by_mobius, want, "mobius, 2^{log} residues");
        }
        for log in [3usize, 15, 18] {
            let table: Vec<Ext<4>> = (0..1usize << log)
                .map(|_| Ext(core::array::from_fn(|_| next() % q)))
                .collect();
            let mut want = table.clone();
            butterflies(&mut want, |low, high| field.sub(&high, &low));
            let mut got = table;
            mobius_ext(&field, &mut got).unwrap();
            assert_eq!(got, want, "mobius_ext, 2^{log} entries");
        }
    }

    /// The higher passes one at a time, as [`butterflies_par`] ran them before its passes were
    /// fused: the reference of [`timings_transforms`].
    fn levelwise_par<T: Copy + Send + Sync>(table: &mut [T], op: impl Fn(T, T) -> T + Sync) {
        let n = table.len();
        let block = BUTTERFLY_BLOCK.min(n).max(1);
        table
            .par_chunks_mut(block)
            .for_each(|chunk| butterflies(chunk, &op));
        let mut half = block;
        while half < n {
            table.par_chunks_mut(2 * half).for_each(|chunk| {
                let (low, high) = chunk.split_at_mut(half);
                low.par_iter()
                    .zip(high.par_iter_mut())
                    .with_min_len(BUTTERFLY_BLOCK)
                    .for_each(|(l, h)| *h = op(*l, *h));
            });
            half *= 2;
        }
    }

    /// Timings of the two transforms of a table of `2^HCO_TRANSFORM_LOG` residues (27 by default,
    /// the share's table at secure-l24-n2-a8), the passes one at a time against the fused passes,
    /// with the same tables; the zeta transform of the Möbius transform is the table. Run with
    /// `cargo test --release -p hachi-co-core --lib timings_transforms -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn timings_transforms() {
        use std::time::Instant;
        let log: u32 = std::env::var("HCO_TRANSFORM_LOG")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(27);
        let q = 15395032342003157u64; // secure-l24
        let zq = Zq::new(q).unwrap();
        let table: Vec<u64> = (0..1u64 << log)
            .into_par_iter()
            .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15) % q)
            .collect();
        for round in 1..=3 {
            let mut old = table.clone();
            let mut new = table.clone();
            let t = Instant::now();
            levelwise_par(&mut old, |low, high| zq.sub(high, low));
            let old_mobius = t.elapsed().as_secs_f64();
            let t = Instant::now();
            butterflies_par(&mut new, |low, high| zq.sub(high, low));
            let new_mobius = t.elapsed().as_secs_f64();
            assert_eq!(old, new);
            let t = Instant::now();
            levelwise_par(&mut old, |low, high| zq.add(high, low));
            let old_zeta = t.elapsed().as_secs_f64();
            let t = Instant::now();
            butterflies_par(&mut new, |low, high| zq.add(high, low));
            let new_zeta = t.elapsed().as_secs_f64();
            assert_eq!(old, new);
            assert_eq!(new, table);
            println!(
                "2^{log} residues, {} threads, round {round}: mobius {old_mobius:.3} -> {new_mobius:.3} s, zeta {old_zeta:.3} -> {new_zeta:.3} s",
                rayon::current_num_threads()
            );
        }
    }

    /// The lifted coefficients are those of `mobius_ext` on the table over the field with the
    /// pad at its index, for pads at every index of small tables and none, whatever the table's
    /// entry at the pad was; their fingerprint follows the table and the pad.
    #[test]
    fn lifted_coefficients_are_the_mobius_transform_over_the_field() {
        let field = Fqk::<4>::new(Q).unwrap();
        let mut rng = Stream(0x6c69_6674_6564);
        for ell in 0..=6usize {
            let size = 1usize << ell;
            let table = rng.residues(size);
            let value = rng.ext();
            let pads = (0..size).map(Some).chain([None]);
            for pad in pads {
                let mut over_field: Vec<Ext<4>> =
                    table.iter().map(|x| field.from_base(*x)).collect();
                if let Some(at) = pad {
                    over_field[at] = value;
                }
                mobius_ext(&field, &mut over_field).unwrap();
                let lifted =
                    LiftedCoeffs::from_table(&field, table.clone(), pad.map(|at| (at, value)))
                        .unwrap();
                assert_eq!(lifted.len(), size);
                assert_eq!(lifted.to_vec(), over_field, "ell {ell} pad {pad:?}");
                let mut run = vec![field.zero(); size / 2];
                lifted.coeffs_into(size - size / 2, &mut run);
                assert_eq!(run, over_field[size - size / 2..], "ell {ell} pad {pad:?}");
            }
            let a = LiftedCoeffs::from_table(&field, table.clone(), Some((0, value))).unwrap();
            let b = LiftedCoeffs::from_table(&field, table.clone(), Some((0, rng.ext()))).unwrap();
            let c = LiftedCoeffs::from_table(&field, table.clone(), None).unwrap();
            assert_ne!(a.fingerprint(), b.fingerprint());
            assert_ne!(a.fingerprint(), c.fingerprint());
            assert_eq!(
                a.fingerprint(),
                LiftedCoeffs::from_table(&field, table.clone(), Some((0, value)))
                    .unwrap()
                    .fingerprint()
            );
            assert!(LiftedCoeffs::from_table(&field, table, Some((size, value))).is_err());
        }
    }

    /// The Möbius transform from its definition, $c\_i = \sum\_{S(x) \subseteq S(i)}
    /// (-1)^{|S(i)| - |S(x)|} f(x)$, and the zeta transform as the subset sums, at every
    /// $\ell \le 6$; the two are inverse; the transforms of an unreduced table are those of its
    /// residues.
    #[test]
    fn mobius_and_zeta_match_brute_force() {
        let zq = Zq::new(Q).unwrap();
        let mut rng = Stream(0x4d6f_6269_7573);
        for ell in 0..=6usize {
            let size = 1usize << ell;
            let values = rng.residues(size);
            let mut coeffs = values.clone();
            mobius(&zq, &mut coeffs).unwrap();
            for (i, c) in coeffs.iter().enumerate() {
                let mut want = 0u64;
                for (x, v) in values.iter().enumerate() {
                    if x & !i == 0 {
                        let sign = (i.count_ones() - x.count_ones()) % 2;
                        want = if sign == 0 {
                            zq.add(want, *v)
                        } else {
                            zq.sub(want, *v)
                        };
                    }
                }
                assert_eq!(*c, want, "l = {ell}, i = {i}");
            }
            let mut back = coeffs.clone();
            zeta(&zq, &mut back).unwrap();
            assert_eq!(back, values, "l = {ell}");
            for (x, v) in values.iter().enumerate() {
                let sum = coeffs
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i & !x == 0)
                    .fold(0, |acc, (_, c)| zq.add(acc, *c));
                assert_eq!(sum, *v, "l = {ell}, x = {x}");
            }
            let mut unreduced: Vec<u64> = values.iter().map(|v| v + Q).collect();
            mobius(&zq, &mut unreduced).unwrap();
            assert_eq!(unreduced, coeffs);
        }
    }

    /// The table comes back from its lifted coefficients ([`LiftedCoeffs::into_table`]): with and
    /// without a pad, whatever the entry at the pad's index, reduced if it was not; the parallel
    /// zeta transform is the serial one, also at sizes whose passes leave the blocks of
    /// [`butterflies_par`].
    #[test]
    fn the_table_comes_back_from_its_coefficients() {
        let zq = Zq::new(Q).unwrap();
        let field = Fqk::<4>::new(Q).unwrap();
        let mut rng = Stream(0x7a65_7461);
        for ell in [0usize, 1, 2, 5, 15, 16] {
            let size = 1usize << ell;
            let table = rng.residues(size);
            let value = rng.ext();
            for pad in [
                None,
                Some((0, value)),
                Some((size - 1, value)),
                Some((size / 2, value)),
            ] {
                let lifted = LiftedCoeffs::from_table(&field, table.clone(), pad).unwrap();
                assert_eq!(
                    lifted.into_table().unwrap(),
                    table,
                    "l = {ell}, pad {pad:?}"
                );
            }
            let unreduced: Vec<u64> = table.iter().map(|v| v + Q).collect();
            let lifted = LiftedCoeffs::from_table(&field, unreduced, Some((size - 1, value)));
            assert_eq!(lifted.unwrap().into_table().unwrap(), table, "l = {ell}");
            let mut coeffs = table.clone();
            mobius(&zq, &mut coeffs).unwrap();
            let mut serial = coeffs.clone();
            butterflies(&mut serial, |low, high| zq.add(high, low));
            zeta(&zq, &mut coeffs).unwrap();
            assert_eq!(coeffs, serial, "l = {ell}");
            assert_eq!(coeffs, table, "l = {ell}");
        }
    }

    /// The coefficients of the Möbius transform evaluate (in the convention of the trait) to the
    /// multilinear extension of the values: $f(\mathbf a) = \sum\_x \widetilde{eq}(\mathbf a,
    /// \mathbf x) f(\mathbf x)$, with $\widetilde{eq}$ as a product over MSB-first bits; at a
    /// hypercube point, the value of the table; and [`evaluate_multilinear`] against the sum
    /// over the monomials with every power computed separately.
    #[test]
    fn coefficients_evaluate_to_the_multilinear_extension() {
        let zq = Zq::new(Q).unwrap();
        let field = Fqk::<4>::new(Q).unwrap();
        let mut rng = Stream(0x6576_616c);
        for ell in 0..=6usize {
            let size = 1usize << ell;
            let values = rng.residues(size);
            let mut coeffs = values.clone();
            mobius(&zq, &mut coeffs).unwrap();
            let point: Vec<Ext<4>> = (0..ell).map(|_| rng.ext()).collect();
            let one = field.one();
            let mle = values.iter().enumerate().fold(field.zero(), |acc, (x, v)| {
                let eq = (1..=ell).fold(one, |e, j| {
                    let a = &point[j - 1];
                    let factor = if bit(x, j, ell) {
                        *a
                    } else {
                        field.sub(&one, a)
                    };
                    field.mul(&e, &factor)
                });
                field.add(&acc, &field.scale(&eq, *v))
            });
            let monomials = coeffs.iter().enumerate().fold(field.zero(), |acc, (i, c)| {
                let term = (1..=ell)
                    .filter(|j| bit(i, *j, ell))
                    .fold(field.from_base(*c), |t, j| field.mul(&t, &point[j - 1]));
                field.add(&acc, &term)
            });
            let fast = evaluate_multilinear(&field, &coeffs, &point).unwrap();
            assert_eq!(fast, mle, "l = {ell}");
            assert_eq!(fast, monomials, "l = {ell}");
            // At a hypercube point: the table entry, with X_1 the most significant bit.
            if ell > 0 {
                let x = size - 2;
                let corner: Vec<Ext<4>> = (1..=ell)
                    .map(|j| field.from_base(u64::from(bit(x, j, ell))))
                    .collect();
                assert_eq!(
                    evaluate_multilinear(&field, &coeffs, &corner).unwrap(),
                    field.from_base(values[x])
                );
            }
        }
    }

    #[test]
    fn shapes_are_checked() {
        let zq = Zq::new(Q).unwrap();
        let field = Fqk::<4>::new(Q).unwrap();
        for len in [0usize, 3, 5, 6, 12] {
            let mut t = vec![1u64; len];
            assert!(
                matches!(mobius(&zq, &mut t), Err(Error::Shape { actual, .. }) if actual == len)
            );
            assert!(matches!(zeta(&zq, &mut t), Err(Error::Shape { .. })));
        }
        let mut one = vec![Q + 5];
        mobius(&zq, &mut one).unwrap();
        assert_eq!(one, vec![5]);
        let point = [field.one(); 2];
        assert!(evaluate_multilinear(&field, &[1, 2, 3], &point).is_err());
        assert!(evaluate_multilinear(&field, &[1, 2, 3, 4, 5, 6, 7, 8], &point).is_err());
        assert_eq!(
            evaluate_multilinear(&field, &[7], &[]).unwrap(),
            field.from_base(7)
        );
        let long = vec![field.one(); 64];
        assert_eq!(
            evaluate_multilinear(&field, &[1], &long),
            Err(Error::Overflow("2^l coefficients"))
        );
        assert!(inner_product(&field, &[field.one()], &[]).is_err());
        let v = [Ext([1, 2, 3, 4]), Ext([5, 6, 7, 8])];
        let e = [Ext([2, 0, 0, 0]), Ext([0, 1, 0, 0])];
        let want = field.add(&field.mul(&v[0], &e[0]), &field.mul(&v[1], &e[1]));
        assert_eq!(inner_product(&field, &v, &e).unwrap(), want);
        assert_eq!(inner_product::<4>(&field, &[], &[]).unwrap(), field.zero());
    }
}
