//! Fiat-Shamir for Hachi on the framed transcript of `hachi-co-transcript`: every message is
//! absorbed under a label before the coins that follow it are squeezed, and every opening starts
//! from a caller-supplied 32-byte context, so that the proofs of different instances (Hachi-Co
//! runs $3n$ of them per proof) are bound to their place in the caller's transcript.

use crate::Result;
use hachi_co_math::{Ext, Poly};
use hachi_co_sampling::ShakePrg;
use hachi_co_sampling::l1ball::L1Ball;
use hachi_co_sampling::uniform::uniform_mod_q;
use hachi_co_transcript::Transcript;

/// The transcript domain of Hachi's evaluation proofs.
pub const DOMAIN: &[u8] = b"hachi/eval/v1";

/// The Fiat-Shamir state of one evaluation proof.
#[derive(Clone, Debug)]
pub struct Fs {
    transcript: Transcript,
}

/// Little-endian bytes of words.
fn bytes(words: impl IntoIterator<Item = u64>) -> Vec<u8> {
    words.into_iter().flat_map(u64::to_le_bytes).collect()
}

impl Fs {
    /// A transcript on [`DOMAIN`] that has absorbed `context`.
    pub fn new(context: &[u8; 32]) -> Self {
        let mut transcript = Transcript::new(DOMAIN);
        transcript.append_message(b"context", context);
        Self { transcript }
    }

    /// Absorbs words under `label`.
    pub fn words(&mut self, label: &[u8], words: &[u64]) {
        self.transcript
            .append_message(label, &bytes(words.iter().copied()));
    }

    /// Absorbs ring elements under `label` (their count is framed with the message).
    pub fn polys(&mut self, label: &[u8], polys: &[Poly]) {
        self.transcript.append_message(
            label,
            &bytes(polys.iter().flat_map(|p| p.0.iter().copied())),
        );
    }

    /// Absorbs field elements under `label`.
    pub fn exts<const K: usize>(&mut self, label: &[u8], exts: &[Ext<K>]) {
        self.transcript
            .append_message(label, &bytes(exts.iter().flat_map(|e| e.0)));
    }

    /// `count` field elements, uniform in $\mathbb F_{q^K}$, squeezed under `label`.
    ///
    /// # Errors
    /// A sampling error (none for $q \ge 2$).
    pub fn field_coins<const K: usize>(
        &mut self,
        label: &[u8],
        q: u64,
        count: usize,
    ) -> Result<Vec<Ext<K>>> {
        let seed = self.transcript.challenge_seed(label);
        let mut prg = ShakePrg::new(&seed, b"hachi/fs/field", 0);
        let residues = uniform_mod_q(&mut prg, q, count * K)?;
        Ok(residues
            .as_chunks::<K>()
            .0
            .iter()
            .map(|c| Ext(*c))
            .collect())
    }

    /// `count` challenges, exactly uniform in the $\ell_1$ ball of radius $\omega$ in
    /// $\mathbb Z^d$, squeezed under `label`.
    ///
    /// # Errors
    /// A sampling error (an unusable radius).
    pub fn challenges(
        &mut self,
        label: &[u8],
        count: usize,
        d: usize,
        omega: u64,
    ) -> Result<Vec<Vec<i64>>> {
        let ball = L1Ball::new(d, omega)?;
        let seed = self.transcript.challenge_seed(label);
        let mut prg = ShakePrg::new(&seed, b"hachi/fs/challenges", 0);
        Ok((0..count)
            .map(|_| ball.sample(&mut prg))
            .collect::<hachi_co_sampling::Result<Vec<_>>>()?)
    }

    /// The digest of the state, for tests that compare the prover's and the verifier's
    /// transcripts.
    pub fn digest(&self) -> [u8; 32] {
        self.transcript.digest()
    }
}
