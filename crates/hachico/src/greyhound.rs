//! The Greyhound tail (feature `greyhound`, on by default): Greyhound with LaBRADOR, crate
//! `hachi-co-greyhound`, behind Hachi's [`Tail`] trait, for [`HachiBase::with_tail`] (batched
//! mode, one tail proof per prover for its three openings, `crate::base`, "Tail"). Not in the
//! draft: an option, approved on 2026-11-17.
//!
//! **Parameters.** Greyhound's [`TailParams`] are made for a fixed list of claim sizes, so the
//! tail is made for one base PCS and one layout ([`GreyhoundTail::for_base`]): the three last
//! tables' sizes $L$ in the order $\langle \widetilde w \rangle\_p$, $g\_{0,p}$, $g\_{1,p}$, as
//! the schedules give them under the tail's rule with `max_vars` (`hachi::pcs`, "Tail"), the
//! digit bound $\lfloor b/2 \rfloor$ of Hachi's base (8 for $b = 16$), the preset's $\lambda$ and
//! a seed from the setup seed ([`tail_seed`]). Making them takes seconds at $L = 27$ or 28, once
//! per setup. Only batched mode with these three sizes is supported: a list of other sizes has no
//! proof ([`Tail::proof_words`] is 0 for it, and the prover and the verifier refuse it).
//!
//! **Contexts.** Hachi passes `hachi::tail::batch_context` of the three openings' contexts,
//! each the seed of the opening's transcript after its last iteration (`hachi::tail`,
//! "Contexts"), so that the proof is bound to the three transcripts.

use crate::{Error, HachiBase};
use hachi::tail::{TailClaim, TailOpening, TailProverClaim};
use hachi::{Tail, params::max_digit};
use hachi_co_core::Layout;
use hachi_co_greyhound::{Claim, ProverClaim, TailParams};
use hachi_co_math::Fqk;
use hachi_co_sampling::ShakePrg;
use hachi_co_transcript::Transcript;
use rand::Rng;
use std::sync::Arc;

/// The PRG domain of the seed of the tail's parameters ([`tail_seed`]).
pub const SETUP_DOMAIN_TAIL: &[u8] = b"hachi-co/setup/tail/greyhound";

/// The transcript domain of [`GreyhoundTail`]'s digest ([`Tail::params_digest`]).
pub const GREYHOUND_DIGEST_DOMAIN: &[u8] = b"hachi-co/tail/greyhound/v1";

/// The seed of the tail's parameters for Hachi-Co's setup seed `seed`: the first 32 bytes of the
/// PRG stream of (`seed`, [`SETUP_DOMAIN_TAIL`], 0).
pub fn tail_seed(seed: &[u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    ShakePrg::new(seed, SETUP_DOMAIN_TAIL, 0).fill_bytes(&mut out);
    out
}

/// A tail that only names its largest table, to derive the schedules under the tail's rule
/// before the real tail is made ([`GreyhoundTail::for_base`]). It commits and proves nothing.
#[derive(Debug)]
struct Sizing {
    max_vars: usize,
}

impl<const K: usize> Tail<K> for Sizing {
    fn max_vars(&self) -> usize {
        self.max_vars
    }

    fn commitment_words(&self, _vars: usize) -> usize {
        0
    }

    fn proof_words(&self, _vars: &[usize]) -> usize {
        0
    }

    fn params_digest(&self) -> [u8; 32] {
        [0u8; 32]
    }

    fn commit(&self, _vars: usize, _digits: &[i8]) -> hachi::Result<(Vec<u64>, TailOpening)> {
        Err(hachi::Error::Protocol(
            "the sizing tail does not commit".into(),
        ))
    }

    fn prove(
        &self,
        _claims: &[TailProverClaim<'_, K>],
        _context: &[u8; 32],
    ) -> hachi::Result<Vec<u64>> {
        Err(hachi::Error::Protocol(
            "the sizing tail does not prove".into(),
        ))
    }

    fn verify(
        &self,
        _claims: &[TailClaim<'_, K>],
        _proof: &[u64],
        _context: &[u8; 32],
    ) -> core::result::Result<(), String> {
        Err("the sizing tail does not verify".into())
    }
}

/// Greyhound as Hachi's tail (module documentation).
pub struct GreyhoundTail<const K: usize> {
    params: TailParams,
    field: Fqk<K>,
    /// The sizes of the claims, in the party's order.
    sizes: Vec<usize>,
    max_vars: usize,
    digest: [u8; 32],
}

impl<const K: usize> core::fmt::Debug for GreyhoundTail<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GreyhoundTail")
            .field("sizes", &self.sizes)
            .field("max_vars", &self.max_vars)
            .finish_non_exhaustive()
    }
}

/// A refusal of Greyhound as an error of Hachi.
fn greyhound_error(e: impl core::fmt::Display) -> hachi::Error {
    hachi::Error::Protocol(format!("Greyhound: {e}"))
}

impl<const K: usize> GreyhoundTail<K> {
    /// The tail for the base PCS `base`, without a tail, at the layout `layout`, with tables of
    /// at most $2^{\mathtt{max\_vars}}$ digits, the preset's `lambda` and the setup seed `seed`
    /// (module documentation, "Parameters").
    ///
    /// # Errors
    /// [`Error::Hachi`] if no schedule fits the tail, or if Greyhound refuses the parameters.
    pub fn for_base(
        base: &HachiBase<K>,
        layout: &Layout,
        max_vars: usize,
        lambda: u32,
        seed: &[u8; 32],
    ) -> Result<Self, Error> {
        let sizing = base
            .clone()
            .with_tail(Arc::new(Sizing { max_vars }) as Arc<dyn Tail<K>>)?;
        let sizes = sizing.last_tables(layout)?.to_vec();
        let q = base.pcs().ctx().q;
        let bound = max_digit(base.choices().b);
        let seed = tail_seed(seed);
        let params = TailParams::new(q, &sizes, bound, lambda, seed).map_err(greyhound_error)?;
        let field = Fqk::new(q).map_err(hachi::Error::from)?;
        let mut t = Transcript::new(GREYHOUND_DIGEST_DOMAIN);
        t.append_u64(b"q", q);
        t.append_u64(b"k", K as u64);
        t.append_u64(b"claims", sizes.len() as u64);
        for vars in &sizes {
            t.append_u64(b"vars", *vars as u64);
        }
        t.append_u64(b"bound", bound);
        t.append_u64(b"lambda", u64::from(lambda));
        t.append_u64(b"max_vars", max_vars as u64);
        t.append_message(b"seed", &seed);
        let digest = t.challenge_seed(b"digest");
        Ok(Self {
            params,
            field,
            sizes,
            max_vars,
            digest,
        })
    }

    /// The sizes of the claims the tail proves, in the party's order.
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// Whether `claims` have the sizes of the parameters, in order.
    fn fits(&self, vars: impl Iterator<Item = usize>) -> bool {
        vars.eq(self.sizes.iter().copied())
    }
}

/// Greyhound's claim of Hachi's.
fn claim<'a, const K: usize>(c: &TailClaim<'a, K>) -> Claim<'a, K> {
    Claim {
        vars: c.vars,
        commitment: c.commitment,
        point: c.point,
        value: c.value,
    }
}

impl<const K: usize> Tail<K> for GreyhoundTail<K> {
    fn max_vars(&self) -> usize {
        self.max_vars
    }

    fn commitment_words(&self, vars: usize) -> usize {
        self.params.commitment_words(vars)
    }

    /// The words of the parameters' proof for the claims of their sizes, and 0 for any other
    /// list (module documentation, "Parameters").
    fn proof_words(&self, vars: &[usize]) -> usize {
        if self.fits(vars.iter().copied()) {
            self.params.proof_words()
        } else {
            0
        }
    }

    fn params_digest(&self) -> [u8; 32] {
        self.digest
    }

    fn commit(&self, vars: usize, digits: &[i8]) -> hachi::Result<(Vec<u64>, TailOpening)> {
        let (words, opening) = self.params.commit(vars, digits).map_err(greyhound_error)?;
        Ok((words, Box::new(opening)))
    }

    fn prove(
        &self,
        claims: &[TailProverClaim<'_, K>],
        context: &[u8; 32],
    ) -> hachi::Result<Vec<u64>> {
        if !self.fits(claims.iter().map(|c| c.claim.vars)) {
            return Err(greyhound_error(
                "the claims do not have the sizes of the parameters",
            ));
        }
        let openings = claims
            .iter()
            .map(|c| {
                c.opening
                    .downcast_ref::<hachi_co_greyhound::TailOpening>()
                    .ok_or_else(|| greyhound_error("an opening of another tail"))
            })
            .collect::<hachi::Result<Vec<_>>>()?;
        let claims: Vec<ProverClaim<'_, K>> = claims
            .iter()
            .zip(openings)
            .map(|(c, opening)| ProverClaim {
                claim: claim(&c.claim),
                digits: c.digits,
                opening,
            })
            .collect();
        self.params
            .prove(&self.field, &claims, context)
            .map_err(greyhound_error)
    }

    fn verify(
        &self,
        claims: &[TailClaim<'_, K>],
        proof: &[u64],
        context: &[u8; 32],
    ) -> core::result::Result<(), String> {
        if !self.fits(claims.iter().map(|c| c.vars)) {
            return Err("Greyhound: the claims do not have the sizes of the parameters".into());
        }
        let claims: Vec<Claim<'_, K>> = claims.iter().map(claim).collect();
        self.params
            .verify(&self.field, &claims, proof, context)
            .map_err(|e| format!("Greyhound: {e}"))
    }
}
