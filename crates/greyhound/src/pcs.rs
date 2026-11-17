//! The scheme: public parameters, the prover's key, proofs, proving and verifying.
//!
//! A proof is the front end's messages ([`FrontProof`]: $u_2$ and $\beta^2$) followed by a
//! LaBRADOR proof of the reduced relation ([`crate::relation`]) by jali, under jali's default
//! policy (Module-SIS block size at least 439), with public matrices from a seed derived from
//! the scheme's seed and the transcript's digest as jali's context (the reference passes its
//! statement hash on to its LaBRADOR the same way). The verifier checks the reference's
//! conditions on $\beta^2$ ([`Params::check_norm`]), rebuilds the relation and verifies the
//! LaBRADOR proof.
use crate::commit::{Commitment, Opening};
use crate::error::{Error, Result};
use crate::eval::{FrontEnd, FrontProof, absorb_norm, prove_front, transcript};
use crate::key::Key;
use crate::params::{Params, Security};
use crate::relation::{Layout, Relation};
use hachi_co_math::{Prepared, Ring};
use hachi_co_sampling::ShakePrg;
use hachi_co_transcript::Transcript;
use rand::Rng;
use std::sync::Arc;

/// The domain of jali's seed and context.
pub const LABRADOR_DOMAIN: &[u8] = b"hachi-co-greyhound/v1/labrador";

/// The public parameters: the parameters, the security rule, the seed, the matrices and the
/// layout of jali's statement.
#[derive(Clone, Debug)]
pub struct PublicParams {
    params: Params,
    security: Security,
    seed: [u8; 32],
    ring: Ring,
    key: Key,
    layout: Layout,
}

impl PublicParams {
    /// The parameters of [`Params::derive`] for `len` ring elements over $\mathcal R_q$ and the
    /// matrices from `seed`.
    ///
    /// # Errors
    /// [`Error::Parameter`] or [`Error::Math`] for $q$ not a prime $\equiv 5 \pmod 8$ in
    /// $(2^{16}, 2^{62})$ or a length without parameters.
    pub fn new(len: usize, q: u64, security: Security, seed: [u8; 32]) -> Result<Self> {
        Self::with_params(Params::derive(len, q, security)?, security, seed)
    }

    /// Given parameters (validated) and the matrices from `seed`.
    ///
    /// # Errors
    /// As [`PublicParams::new`].
    pub fn with_params(params: Params, security: Security, seed: [u8; 32]) -> Result<Self> {
        params.validate()?;
        let ring = Ring::new(params.q, crate::params::N)?;
        let key = Key::expand(&params, &seed)?;
        Ok(Self {
            params,
            security,
            seed,
            ring,
            key,
            layout: Layout::choose(&params.witness_lengths()),
        })
    }

    /// The same parameters with another layout of jali's statement (the default is
    /// [`Layout::choose`]); prover and verifier must use the same.
    ///
    /// # Errors
    /// [`Error::Parameter`] for a layout that does not hold the witness or exceeds jali's limits
    /// (1024 vectors, $2^{24}$ coefficients).
    pub fn with_layout(mut self, layout: Layout) -> Result<Self> {
        let total: usize = self.params.witness_lengths().iter().sum();
        // jali's largest multiplicity (`statement::MAX_MULTIPLICITY`, not exported).
        if layout.n == 0
            || layout.r == 0
            || layout.r > 1024
            || layout.r.checked_mul(layout.n).is_none_or(|x| x < total)
            || layout.r * layout.n * crate::params::N > jali::labrador::MAX_WITNESS_COEFFICIENTS
        {
            return Err(Error::Parameter("layout"));
        }
        self.layout = layout;
        Ok(self)
    }

    /// The parameters.
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// The security rule.
    pub fn security(&self) -> Security {
        self.security
    }

    /// The seed of the matrices.
    pub fn seed(&self) -> &[u8; 32] {
        &self.seed
    }

    /// $\mathcal R_q$ with its exact products.
    pub fn ring(&self) -> &Ring {
        &self.ring
    }

    /// The matrices.
    pub fn key(&self) -> &Key {
        &self.key
    }

    /// The prover's key: $A$ in the transform domain, with the primes that products by digits
    /// of the polynomial need.
    ///
    /// # Errors
    /// Arithmetic errors.
    pub fn prover_key(&self) -> Result<ProverKey> {
        let p = &self.params;
        let norm = crate::commit::digit_bound(p.q, p.b, p.f);
        Ok(ProverKey {
            a: self.ring.prepare_short(&self.key.a, norm)?,
        })
    }

    /// The layout of the witness in jali's statement.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// jali's ring $\mathcal R_q$.
    ///
    /// # Errors
    /// [`Error::Labrador`] for $q \ge 2^{62}$, which jali 0.3.1 does not take.
    pub fn labrador_ring(&self) -> Result<Arc<jali::labrador::Rq>> {
        Ok(jali::labrador::Rq::new(self.params.q, crate::params::N)?)
    }

    /// jali's seed for its own public matrices.
    pub fn labrador_seed(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        ShakePrg::new(&self.seed, LABRADOR_DOMAIN, 0).fill_bytes(&mut out);
        out
    }
}

/// The prover's key.
#[derive(Clone, Debug)]
pub struct ProverKey {
    /// $A$ in the transform domain.
    pub(crate) a: Prepared,
}

/// jali's context for a transcript: [`LABRADOR_DOMAIN`] and the digest.
pub fn labrador_context(t: &Transcript) -> Vec<u8> {
    let mut out = LABRADOR_DOMAIN.to_vec();
    out.extend_from_slice(&t.digest());
    out
}

/// An evaluation proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    /// $u_2$ and $\beta^2$.
    pub front: FrontProof,
    /// jali's LaBRADOR proof of the reduced relation.
    pub labrador: jali::labrador::Proof,
}

impl Proof {
    /// The front end's encoding, then jali's.
    pub fn to_bytes(&self, params: &Params) -> Vec<u8> {
        let mut out = self.front.to_bytes(params);
        out.extend_from_slice(self.labrador.as_bytes());
        out
    }

    /// The proof of [`Proof::to_bytes`].
    ///
    /// # Errors
    /// [`Error::Encoding`].
    pub fn from_bytes(params: &Params, bytes: &[u8]) -> Result<Self> {
        let at = FrontProof::encoded_len(params);
        if bytes.len() <= at {
            return Err(Error::Encoding);
        }
        Ok(Self {
            front: FrontProof::from_bytes(params, &bytes[..at])?,
            labrador: jali::labrador::Proof::from_bytes(bytes[at..].to_vec())?,
        })
    }
}

/// The LaBRADOR part of the prover, from the front end's output: the reduced relation, its
/// jali form, the witness's and jali's proof.
///
/// # Errors
/// [`Error::Labrador`] (among them $q \ge 2^{48}$).
pub fn prove_labrador(
    pp: &PublicParams,
    com: &Commitment,
    x: u64,
    front: &FrontEnd,
) -> Result<jali::labrador::Proof> {
    let ring = pp.labrador_ring()?;
    let layout = pp.layout();
    let relation = Relation::with_challenges(pp, com, x, front.y, &front.proof, &front.challenges)?;
    let statement = relation.into_jali(&ring, &layout)?;
    let witness = layout.jali_witness(&ring, pp.params(), &front.witness)?;
    Ok(jali::labrador::prove(
        &statement,
        &witness,
        &pp.labrador_seed(),
        &labrador_context(&front.transcript),
        &jali::labrador::Policy::default(),
    )?)
}

/// Prove the evaluation of the committed polynomial at `x`: returns $y = f(x)$ and the proof.
///
/// # Errors
/// [`Error::Shape`] for inputs of the wrong size; [`Error::Labrador`] (among them $q \ge
/// 2^{48}$).
pub fn prove(
    pp: &PublicParams,
    com: &Commitment,
    opening: &Opening,
    poly: &[u64],
    x: u64,
) -> Result<(u64, Proof)> {
    let front = prove_front(pp, com, opening, poly, x)?;
    let labrador = prove_labrador(pp, com, x, &front)?;
    Ok((
        front.y,
        Proof {
            front: front.proof,
            labrador,
        },
    ))
}

/// Verify that `proof` shows $f(x) = y$ for the polynomial committed in `com`.
///
/// # Errors
/// [`Error::InvalidProof`] or [`Error::Labrador`] when the proof is rejected; [`Error::Shape`]
/// for inputs of the wrong size.
pub fn verify(pp: &PublicParams, com: &Commitment, x: u64, y: u64, proof: &Proof) -> Result<()> {
    let p = pp.params();
    p.check_norm(proof.front.norm_sq, pp.security())?;
    let (mut t, cs) = transcript(pp, com, x, y, &proof.front.u2);
    absorb_norm(&mut t, proof.front.norm_sq);
    let relation = Relation::with_challenges(pp, com, x, y, &proof.front, &cs)?;
    let ring = pp.labrador_ring()?;
    let statement = relation.into_jali(&ring, &pp.layout())?;
    jali::labrador::verify(
        &statement,
        &proof.labrador,
        &pp.labrador_seed(),
        &labrador_context(&t),
        &jali::labrador::Policy::default(),
    )?;
    Ok(())
}
