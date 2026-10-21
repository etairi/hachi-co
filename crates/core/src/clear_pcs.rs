//! [`ClearPcs`], a reference base PCS (feature `fixtures`): the commitment is the SHA-256 hash
//! of the committed polynomial or vector, and an opening reveals it.

use crate::base_pcs::{BasePcs, evaluate_multilinear, inner_product};
use crate::{Error, Result};
use hachi_co_math::{Ext, Fqk};
use sha2::{Digest, Sha256};

/// The domain of the commitments of [`ClearPcs`]: the first field of every hashed message.
pub const CLEAR_PCS_DOMAIN: &str = "hachi-co/clear-pcs/v1";

/// The words of a proof before the opened entries: the context (4), the kind (1) and the number
/// of entries (1).
const PROOF_HEADER_WORDS: usize = 6;

/// The words of a commitment: the 32 bytes of the hash.
const COMMITMENT_WORDS: usize = 4;

/// What a [`ClearPcs`] commitment is to; also the prover's state, and the content of an
/// opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClearPolynomial<const K: usize> {
    /// The $2^\ell$ monomial coefficients of a multilinear polynomial over $\mathbb Z\_q$ (the
    /// convention of [`BasePcs`]), canonical.
    Multilinear(Vec<u64>),
    /// A vector over $\mathbb F\_{q^K}$, canonical.
    Vector(Vec<Ext<K>>),
}

impl<const K: usize> ClearPolynomial<K> {
    /// The kind's word in a proof and its name in the hash.
    fn kind(&self) -> (u64, &'static str) {
        match self {
            Self::Multilinear(_) => (0, "multilinear"),
            Self::Vector(_) => (1, "vector"),
        }
    }

    /// The number of entries: coefficients or field elements.
    fn len(&self) -> usize {
        match self {
            Self::Multilinear(c) => c.len(),
            Self::Vector(v) => v.len(),
        }
    }

    /// The residues, entry after entry (a field element is its $K$ coordinates).
    fn residues(&self) -> Box<dyn Iterator<Item = u64> + '_> {
        match self {
            Self::Multilinear(c) => Box::new(c.iter().copied()),
            Self::Vector(v) => Box::new(v.iter().flat_map(|x| x.0)),
        }
    }
}

/// A [`ClearPcs`] commitment: the SHA-256 hash of the committed polynomial or vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClearCommitment(pub [u8; 32]);

/// A [`ClearPcs`] opening: the whole committed polynomial or vector, and the context it was
/// made for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClearOpening<const K: usize> {
    /// The context of the opening ([`BasePcs`], "Contexts").
    pub context: [u8; 32],
    /// The committed polynomial or vector.
    pub polynomial: ClearPolynomial<K>,
}

/// A reference base PCS ([`BasePcs`]) for the tests: binding and sound, but neither succinct
/// nor hiding. The fixture tests use it, since the Sage oracle does not model the base PCS;
/// Hachi is the base PCS otherwise.
///
/// - **Commit.** The commitment is the SHA-256 hash of `LE64(len(D)) ‖ D ‖ LE64(len(kind)) ‖
///   kind ‖ LE64(q) ‖ LE64(K) ‖ LE64(entries) ‖ residues`, with the domain $D$ =
///   [`CLEAR_PCS_DOMAIN`], the kind `multilinear` or `vector`, the number of entries
///   (coefficients or field elements) and the residues as `LE64`, entry after entry (a field
///   element is its $K$ coordinates). The inputs are reduced modulo $q$ first.
/// - **Open.** The proof is the whole committed polynomial or vector, with the context; the
///   value is computed from it ([`evaluate_multilinear`], [`inner_product`]).
/// - **Verify.** The verifier checks that the proof was made under its context, that it opens
///   the expected kind and size, that its residues are canonical, that its hash is the
///   commitment, and that it takes the claimed value.
///
/// It is **binding** (two vectors with one commitment are a SHA-256 collision) and **sound**
/// (the verifier recomputes the value from the committed vector itself), but neither
/// **succinct** (a proof is as large as the polynomial, $2^\ell$ words for $\langle \widetilde w
/// \rangle\_p$, and the verifier's work is linear in it) nor **hiding** (the commitment is a
/// deterministic hash, against which a guessed vector can be checked, and every opening reveals
/// the polynomial: an honest prover's share polynomial and blinders become public). Its
/// soundness does not depend on the context; the context check makes it as strict as a scheme
/// with Fiat-Shamir about where an opening may be used, so that the tests catch a prover and a
/// verifier that derive different contexts, or an opening moved to another party, polynomial or
/// proof.
///
/// **Words.** A commitment is the 32 bytes of the hash as 4 little-endian words. A proof is the
/// context as 4 little-endian words, the kind (0 multilinear, 1 vector), the number of entries,
/// and the residues ([`ClearPcs::multilinear_proof_words`], [`ClearPcs::vector_proof_words`]).
/// Decoding accepts exactly these encodings; the residues are checked against $q$ by the
/// verifier (decoding does not know $q$).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClearPcs<const K: usize> {
    field: Fqk<K>,
}

/// Absorbs `x` as $\mathrm{LE64}(|x|) \Vert x$.
fn frame(h: &mut Sha256, x: &[u8]) {
    h.update((x.len() as u64).to_le_bytes());
    h.update(x);
}

/// The hash of [`ClearPcs`] (the formula in its documentation).
fn digest<const K: usize>(q: u64, poly: &ClearPolynomial<K>) -> [u8; 32] {
    const CHUNK: usize = 1024;
    let mut h = Sha256::new();
    frame(&mut h, CLEAR_PCS_DOMAIN.as_bytes());
    frame(&mut h, poly.kind().1.as_bytes());
    h.update(q.to_le_bytes());
    h.update((K as u64).to_le_bytes());
    h.update((poly.len() as u64).to_le_bytes());
    let mut buf = Vec::with_capacity(8 * CHUNK);
    for r in poly.residues() {
        buf.extend_from_slice(&r.to_le_bytes());
        if buf.len() == 8 * CHUNK {
            h.update(&buf);
            buf.clear();
        }
    }
    h.update(&buf);
    h.finalize().into()
}

/// 32 bytes as 4 little-endian words.
fn bytes_to_words(bytes: &[u8; 32]) -> [u64; 4] {
    let (chunks, _) = bytes.as_chunks::<8>();
    core::array::from_fn(|i| u64::from_le_bytes(chunks[i]))
}

/// 4 little-endian words as 32 bytes.
fn words_to_bytes(words: &[u64; 4]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (chunk, w) in out.as_chunks_mut::<8>().0.iter_mut().zip(words) {
        *chunk = w.to_le_bytes();
    }
    out
}

impl<const K: usize> ClearPcs<K> {
    /// The scheme over $\mathbb F\_{q^K}$ ([`Fqk::new`]).
    ///
    /// # Errors
    /// [`Error::Math`] for a $q$ or a $K$ that [`Fqk::new`] refuses.
    pub fn new(q: u64) -> Result<Self> {
        Ok(Self {
            field: Fqk::new(q)?,
        })
    }

    /// The scheme over `field`.
    pub fn from_field(field: &Fqk<K>) -> Self {
        Self {
            field: field.clone(),
        }
    }

    /// The field $\mathbb F\_{q^K}$.
    pub fn field(&self) -> &Fqk<K> {
        &self.field
    }

    /// The number of words of a commitment: 4.
    pub const COMMITMENT_WORDS: usize = COMMITMENT_WORDS;

    /// The number of words of an opening of a multilinear polynomial in `num_vars` variables:
    /// $6 + 2^{\mathrm{num\\_vars}}$ (`None` on overflow).
    pub fn multilinear_proof_words(num_vars: usize) -> Option<usize> {
        u32::try_from(num_vars)
            .ok()
            .and_then(|e| 1usize.checked_shl(e))
            .and_then(|size| size.checked_add(PROOF_HEADER_WORDS))
    }

    /// The number of words of an opening of a vector of `len` elements: $6 + K \cdot len$
    /// (`None` on overflow).
    pub fn vector_proof_words(len: usize) -> Option<usize> {
        len.checked_mul(K)?.checked_add(PROOF_HEADER_WORDS)
    }

    /// The checks every opening shares: the context, canonical residues, and the hash.
    fn check_opening(
        &self,
        com: &ClearCommitment,
        proof: &ClearOpening<K>,
        context: &[u8; 32],
    ) -> core::result::Result<(), String> {
        if proof.context != *context {
            return Err("the opening was made under another context".into());
        }
        let q = self.field.modulus();
        if let Some(r) = proof.polynomial.residues().find(|r| *r >= q) {
            return Err(format!(
                "the opening has the residue {r}, not below q = {q}"
            ));
        }
        if digest(q, &proof.polynomial) != com.0 {
            return Err("the opened vector does not match the commitment".into());
        }
        Ok(())
    }

    /// The value check: `got` against the claimed `value`.
    fn check_value(&self, got: &Ext<K>, value: &Ext<K>) -> core::result::Result<(), String> {
        if *got != self.field.reduce(value) {
            return Err("the opened polynomial does not take the claimed value".into());
        }
        Ok(())
    }
}

impl<const K: usize> BasePcs<K> for ClearPcs<K> {
    type Commitment = ClearCommitment;
    type State = ClearPolynomial<K>;
    type Proof = ClearOpening<K>;

    /// SHA-256 of [`CLEAR_PCS_DOMAIN`], the tag `params`, $q$ and $K$: the scheme has no other
    /// parameter.
    fn params_digest(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        frame(&mut h, CLEAR_PCS_DOMAIN.as_bytes());
        frame(&mut h, b"params");
        h.update(self.field.modulus().to_le_bytes());
        h.update((K as u64).to_le_bytes());
        h.finalize().into()
    }

    /// The hash of the coefficients reduced modulo $q$.
    ///
    /// # Errors
    /// [`Error::Shape`] unless there are $2^\ell$ coefficients for some $\ell \ge 0$.
    fn commit_multilinear(&self, coeffs: &[u64]) -> Result<(ClearCommitment, ClearPolynomial<K>)> {
        if !coeffs.len().is_power_of_two() {
            return Err(Error::Shape {
                what: "multilinear coefficients (2^l entries)",
                expected: coeffs.len().checked_next_power_of_two().unwrap_or(0),
                actual: coeffs.len(),
            });
        }
        let zq = self.field.zq();
        let poly = ClearPolynomial::Multilinear(coeffs.iter().map(|c| zq.reduce(*c)).collect());
        Ok((ClearCommitment(digest(self.field.modulus(), &poly)), poly))
    }

    /// The hash of the vector reduced modulo $q$ (any length).
    fn commit_vector(&self, v: &[Ext<K>]) -> Result<(ClearCommitment, ClearPolynomial<K>)> {
        let poly = ClearPolynomial::Vector(v.iter().map(|x| self.field.reduce(x)).collect());
        Ok((ClearCommitment(digest(self.field.modulus(), &poly)), poly))
    }

    /// # Errors
    /// [`Error::BasePcs`] for the state of a vector commitment; [`Error::Shape`] unless the
    /// point has $\ell$ coordinates for $2^\ell$ coefficients.
    fn prove_multilinear(
        &self,
        st: &ClearPolynomial<K>,
        point: &[Ext<K>],
        context: &[u8; 32],
    ) -> Result<(Ext<K>, ClearOpening<K>)> {
        let ClearPolynomial::Multilinear(coeffs) = st else {
            return Err(Error::BasePcs(
                "a multilinear opening of a vector commitment".into(),
            ));
        };
        let value = evaluate_multilinear(&self.field, coeffs, point)?;
        Ok((
            value,
            ClearOpening {
                context: *context,
                polynomial: st.clone(),
            },
        ))
    }

    /// # Errors
    /// [`Error::BasePcs`] for the state of a multilinear commitment; [`Error::Shape`] unless
    /// there is one weight per entry.
    fn prove_inner_product(
        &self,
        st: &ClearPolynomial<K>,
        weights: &[Ext<K>],
        context: &[u8; 32],
    ) -> Result<(Ext<K>, ClearOpening<K>)> {
        let ClearPolynomial::Vector(v) = st else {
            return Err(Error::BasePcs(
                "an inner-product opening of a multilinear commitment".into(),
            ));
        };
        let value = inner_product(&self.field, v, weights)?;
        Ok((
            value,
            ClearOpening {
                context: *context,
                polynomial: st.clone(),
            },
        ))
    }

    fn verify_multilinear(
        &self,
        com: &ClearCommitment,
        num_vars: usize,
        point: &[Ext<K>],
        value: &Ext<K>,
        proof: &ClearOpening<K>,
        context: &[u8; 32],
    ) -> core::result::Result<(), String> {
        let ClearPolynomial::Multilinear(coeffs) = &proof.polynomial else {
            return Err("the opening is of a vector, not of a multilinear polynomial".into());
        };
        if point.len() != num_vars {
            return Err(format!(
                "the point has {} coordinates for a polynomial in {num_vars} variables",
                point.len()
            ));
        }
        let size = Self::multilinear_proof_words(num_vars).map(|w| w - PROOF_HEADER_WORDS);
        if size != Some(coeffs.len()) {
            return Err(format!(
                "the opening has {} coefficients for a polynomial in {num_vars} variables",
                coeffs.len()
            ));
        }
        self.check_opening(com, proof, context)?;
        let got = evaluate_multilinear(&self.field, coeffs, point).map_err(|e| e.to_string())?;
        self.check_value(&got, value)
    }

    fn verify_inner_product(
        &self,
        com: &ClearCommitment,
        len: usize,
        weights: &[Ext<K>],
        value: &Ext<K>,
        proof: &ClearOpening<K>,
        context: &[u8; 32],
    ) -> core::result::Result<(), String> {
        let ClearPolynomial::Vector(v) = &proof.polynomial else {
            return Err("the opening is of a multilinear polynomial, not of a vector".into());
        };
        if v.len() != len || weights.len() != len {
            return Err(format!(
                "the opening has {} entries and {} weights for a vector of {len}",
                v.len(),
                weights.len()
            ));
        }
        self.check_opening(com, proof, context)?;
        let got = inner_product(&self.field, v, weights).map_err(|e| e.to_string())?;
        self.check_value(&got, value)
    }

    fn commitment_to_words(&self, com: &ClearCommitment) -> Vec<u64> {
        bytes_to_words(&com.0).to_vec()
    }

    /// # Errors
    /// [`Error::BasePcs`] unless there are exactly 4 words.
    fn commitment_from_words(&self, words: &[u64]) -> Result<ClearCommitment> {
        let words: &[u64; COMMITMENT_WORDS] = words.try_into().map_err(|_| {
            Error::BasePcs(format!(
                "a ClearPcs commitment has {COMMITMENT_WORDS} words, got {}",
                words.len()
            ))
        })?;
        Ok(ClearCommitment(words_to_bytes(words)))
    }

    fn proof_to_words(&self, p: &ClearOpening<K>) -> Vec<u64> {
        let mut out = Vec::with_capacity(PROOF_HEADER_WORDS + K * p.polynomial.len());
        out.extend_from_slice(&bytes_to_words(&p.context));
        out.push(p.polynomial.kind().0);
        out.push(p.polynomial.len() as u64);
        out.extend(p.polynomial.residues());
        out
    }

    /// # Errors
    /// [`Error::BasePcs`] for fewer than 6 words, an unknown kind, or a number of entries that
    /// does not match the number of words.
    fn proof_from_words(&self, words: &[u64]) -> Result<ClearOpening<K>> {
        let Some((header, data)) = words.split_at_checked(PROOF_HEADER_WORDS) else {
            return Err(Error::BasePcs(format!(
                "a ClearPcs proof has at least {PROOF_HEADER_WORDS} words, got {}",
                words.len()
            )));
        };
        let context = words_to_bytes(&[header[0], header[1], header[2], header[3]]);
        let (kind, count) = (header[4], header[5]);
        let entries = |per_entry: usize| -> Result<usize> {
            usize::try_from(count)
                .ok()
                .and_then(|c| c.checked_mul(per_entry))
                .filter(|words| *words == data.len())
                .ok_or_else(|| {
                    Error::BasePcs(format!(
                        "a ClearPcs proof of {count} entries of {per_entry} words has {} words \
                         of entries",
                        data.len()
                    ))
                })
        };
        let polynomial = match kind {
            0 => {
                entries(1)?;
                ClearPolynomial::Multilinear(data.to_vec())
            }
            1 if K > 0 => {
                entries(K)?;
                ClearPolynomial::Vector(data.as_chunks::<K>().0.iter().map(|c| Ext(*c)).collect())
            }
            other => {
                return Err(Error::BasePcs(format!(
                    "a ClearPcs proof of the unknown kind {other}"
                )));
            }
        };
        Ok(ClearOpening {
            context,
            polynomial,
        })
    }
}
