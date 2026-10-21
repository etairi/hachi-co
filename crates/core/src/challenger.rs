//! The verifier's public coins: the trait the protocol code draws them from, and its Fiat-Shamir
//! implementation on `hachi-co-transcript` (`docs/m1-spec.md`, "Randomness and coins").

use crate::data::SumcheckCoins;
use crate::randomness::uniform_ext;
use crate::{Error, Result};
use hachi_co_math::{Ext, Fqk, Poly};
use hachi_co_sampling::ShakePrg;
use hachi_co_sampling::l1ball::L1Ball;
use hachi_co_transcript::Transcript;

/// The transcript label of the seed of the ring-to-linear challenges.
const LABEL_R2L_C: &[u8] = b"hachi-co/r2l/c";
/// The transcript label of the seed of the sumcheck coins $\alpha, \rho, \tau\_0, \tau\_1$.
const LABEL_SC_COINS: &[u8] = b"hachi-co/sumcheck/coins";
/// The transcript label of the seed of a round coin $a\_r$.
const LABEL_SC_ROUND: &[u8] = b"hachi-co/sumcheck/a";
/// The transcript label of a base-PCS context, squeezed from the fork of the context's label
/// ([`FsChallenger`], [`Challenger::context`]).
pub const FS_LABEL_CONTEXT: &str = "hachi-co/context";
/// The PRG domain that expands the seed of the ring-to-linear challenges (index = attempt).
pub const FS_DOMAIN_R2L_C: &str = "hachi-co/fs/r2l/c";
/// The PRG domain that expands the seed of the sumcheck coins (index 0).
pub const FS_DOMAIN_SC_COINS: &str = "hachi-co/fs/sumcheck/coins";
/// The PRG domain that expands the seed of a round coin (index = round).
pub const FS_DOMAIN_SC_ROUND: &str = "hachi-co/fs/sumcheck/a";

/// The verifier's public coins (`docs/m1-spec.md`, "Randomness and coins" and `hachi-co-core`).
///
/// Protocol code never samples the verifier's coins itself: it draws them from a `Challenger`.
/// Real runs use [`FsChallenger`] (Fiat-Shamir on `hachi-co-transcript`); fixture tests use the
/// replay challenger of the feature `fixtures` (`ReplayChallenger`), which returns the Sage
/// oracle's recorded coins and ignores the absorbs (the oracle's coins are public coins that do
/// not depend on the messages, `ref/README.md`, choice 14). The same driver code runs with both.
///
/// **Call order.** Prover and verifier make the same calls in the same order. Absorb every public
/// message, with a fixed label (a constant of the crate that sends the message), before drawing
/// the coins that follow it. For the ring-to-linear step (`5_Building_Blocks.tex`:115-131), the
/// provers, for attempt $a = 0, 1, \dots$ until no party aborts, call
/// [`begin_attempt`](Self::begin_attempt)`(a)`, absorb that attempt's first message $\mathbf v$,
/// and draw [`r2l_challenges`](Self::r2l_challenges)`(a, 2^r, d, ω)`. A verifier that replays
/// the accepted attempt $a^\ast$ calls `begin_attempt(a*)` once, before absorbing $\mathbf v$ and
/// drawing the challenges.
///
/// The base-PCS openings at the end of the sumcheck (`5_Building_Blocks.tex`:314-326) draw no
/// coins of their own here: each one takes a 32-byte [`context`](Self::context) derived from
/// everything absorbed before it, under a label that names the opening.
///
/// [`FsChallenger`] and `ReplayChallenger` implement the trait for every `K`. Generic code with a
/// bound `H: Challenger<K>` calls the methods directly; on a concrete value, a method whose
/// signature does not mention `K` (`absorb_polys`, `absorb_words`, `absorb_u64`,
/// `begin_attempt`, `r2l_challenges`, `context`) needs it spelled out:
/// `Challenger::<4>::begin_attempt(&mut ch, 0)`.
pub trait Challenger<const K: usize> {
    /// Absorbs a public message made of ring elements (or of polynomials of another length),
    /// under `label`.
    fn absorb_polys(&mut self, label: &'static str, polys: &[Poly]);

    /// Absorbs a public message made of elements of $\mathbb F\_{q^K}$, under `label`.
    fn absorb_exts(&mut self, label: &'static str, exts: &[Ext<K>]);

    /// Absorbs a public message made of vectors of 64-bit words, under `label`: for example the
    /// base-PCS commitments of every party, encoded by
    /// [`BasePcs::commitment_to_words`](crate::BasePcs::commitment_to_words), one vector per
    /// party (`5_Building_Blocks.tex`:220-221, 230, 241).
    fn absorb_words(&mut self, label: &'static str, words: &[Vec<u64>]);

    /// Absorbs a public integer, under `label`.
    fn absorb_u64(&mut self, label: &'static str, value: u64);

    /// The 32-byte context of the base-PCS instance named `label`
    /// ([`BasePcs`](crate::BasePcs), "Contexts"), derived from everything absorbed so far.
    /// Prover and verifier call it at the same point of the protocol with the same label and
    /// obtain the same context; different labels give independent contexts. It draws no coin:
    /// the coins that follow, and the contexts of other labels, do not depend on whether or in
    /// which order contexts were derived (so that each prover can derive the contexts of its own
    /// openings and the verifier all of them). [`FsChallenger`] squeezes it from a fork of its
    /// transcript; `ReplayChallenger` returns a fixed hash of the label.
    fn context(&mut self, label: &str) -> [u8; 32];

    /// Starts attempt `attempt` of the ring-to-linear step (Fiat-Shamir with aborts; the attempt
    /// counter of `5_Building_Blocks.tex`:129, `6_Collaborative_Hachi.tex`:417 and
    /// `ref/README.md`, choice 5). For [`FsChallenger`], the first call (whatever its index) saves
    /// the current state as the base of the step, and every call restores that base and absorbs
    /// the attempt index, so that the challenges of attempt $a$ depend on everything absorbed
    /// before the step, on $a$ and on attempt $a$'s own messages, and not on earlier (aborted)
    /// attempts. The provers call it with $a = 0, 1, \dots$; a verifier calls it once with the
    /// accepted $a^\ast$ and reaches the same state.
    ///
    /// # Errors
    /// Implementation-specific errors; [`FsChallenger`] never fails.
    fn begin_attempt(&mut self, attempt: u64) -> Result<()>;

    /// The challenges $\mathbf c = (c\_1, \dots, c\_{\mathrm{count}})$ of attempt `attempt`, each
    /// uniform in $\mathcal C = \lbrace c \in \mathcal R\_q : \Vert c \Vert\_1 \le \omega \rbrace$
    /// (`5_Building_Blocks.tex`:122-126; `ref/README.md`, choice 15), given by its $d$ signed
    /// coefficients. The ring-to-linear step draws `count` $= 2^r$ of them.
    ///
    /// # Errors
    /// [`Error::Sampling`] if $\omega \gt 2^{63} - 1$ or the PRG fails, and
    /// implementation-specific errors.
    fn r2l_challenges(
        &mut self,
        attempt: u64,
        count: usize,
        d: usize,
        omega: u64,
    ) -> Result<Vec<Vec<i64>>>;

    /// The coins $\alpha, \rho \in \mathbb F\_{q^K}$, $\tau\_0 \in \mathbb F\_{q^K}^{\ell'' +
    /// \lceil \log n \rceil}$ and $\tau\_1 \in \mathbb F\_{q^K}^{\lceil \log \nu \rceil}$ of
    /// `5_Building_Blocks.tex`:243-250, uniform and independent.
    ///
    /// # Errors
    /// Implementation-specific errors.
    fn sumcheck_coins(
        &mut self,
        field: &Fqk<K>,
        ell_pp: usize,
        log_n: usize,
        log_nu: usize,
    ) -> Result<SumcheckCoins<K>>;

    /// The coin $a\_r \in \mathbb F\_{q^K}$ of round $r \in \lbrace 1, \dots, \ell'' \rbrace$ of the
    /// sumcheck (`5_Building_Blocks.tex`:312), uniform.
    ///
    /// # Errors
    /// [`Error::Challenger`] for round 0, and implementation-specific errors.
    fn round_coin(&mut self, field: &Fqk<K>, round: usize) -> Result<Ext<K>>;
}

/// The message bytes of [`Challenger::absorb_polys`]: `b"P"`, the number of polynomials as
/// $\mathrm{LE64}$, then for each polynomial its length as $\mathrm{LE64}$ and its residues as
/// $\mathrm{LE64}$.
fn encode_polys(polys: &[Poly]) -> Vec<u8> {
    let coeffs: usize = polys.iter().map(|p| p.0.len()).sum();
    let mut out = Vec::with_capacity(9 + 8 * (polys.len() + coeffs));
    out.push(b'P');
    out.extend_from_slice(&(polys.len() as u64).to_le_bytes());
    for p in polys {
        out.extend_from_slice(&(p.0.len() as u64).to_le_bytes());
        for c in &p.0 {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    out
}

/// The message bytes of [`Challenger::absorb_words`]: `b"W"`, the number of vectors as
/// $\mathrm{LE64}$, then for each vector its length and its words as $\mathrm{LE64}$.
fn encode_words(words: &[Vec<u64>]) -> Vec<u8> {
    let total: usize = words.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(9 + 8 * (words.len() + total));
    out.push(b'W');
    out.extend_from_slice(&(words.len() as u64).to_le_bytes());
    for v in words {
        out.extend_from_slice(&(v.len() as u64).to_le_bytes());
        for w in v {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }
    out
}

/// The message bytes of [`Challenger::absorb_exts`]: `b"E"`, $K$ and the number of elements as
/// $\mathrm{LE64}$, then the coordinates of each element as $\mathrm{LE64}$.
fn encode_exts<const K: usize>(exts: &[Ext<K>]) -> Vec<u8> {
    let mut out = Vec::with_capacity(17 + 8 * K * exts.len());
    out.push(b'E');
    out.extend_from_slice(&(K as u64).to_le_bytes());
    out.extend_from_slice(&(exts.len() as u64).to_le_bytes());
    for x in exts {
        for c in &x.0 {
            out.extend_from_slice(&c.to_le_bytes());
        }
    }
    out
}

/// [`Challenger`] by Fiat-Shamir on a [`Transcript`] (`docs/m1-spec.md`, `hachi-co-core`).
///
/// **Absorbing.** [`absorb_polys`](Challenger::absorb_polys),
/// [`absorb_exts`](Challenger::absorb_exts) and [`absorb_words`](Challenger::absorb_words) call
/// [`Transcript::append_message`] with the label and a self-describing encoding: `b"P"` ‖
/// LE64(#polynomials) ‖ (LE64(length) ‖ residues as LE64) per polynomial, resp. `b"E"` ‖
/// LE64($K$) ‖ LE64(#elements) ‖ coordinates as LE64, resp. `b"W"` ‖ LE64(#vectors) ‖
/// (LE64(length) ‖ words as LE64) per vector. A message of polynomials, of field elements, of
/// word vectors and an integer (8 bytes, [`absorb_u64`](Challenger::absorb_u64) =
/// [`Transcript::append_u64`]) therefore never encode alike. Residues are absorbed as given:
/// callers pass canonical residues, as every `hachi-co-math` operation returns.
///
/// **Contexts.** [`context`](Challenger::context)`(label)` is the 32-byte challenge seed, under
/// the label [`FS_LABEL_CONTEXT`], of the fork ([`Transcript::fork`]) of the current transcript
/// under `label`: a forked squeeze. It depends on everything absorbed so far and on the label,
/// and leaves the transcript itself unchanged, so the coins that follow do not depend on it and
/// prover and verifier stay in the same state whichever contexts each of them derives. The fork
/// has its own operation name, so no absorb or coin of the main transcript can imitate it.
///
/// **Coins.** Each draw takes a 32-byte seed from the transcript
/// ([`Transcript::challenge_seed`], which ratchets the state) and expands it with a [`ShakePrg`]:
///
/// | Method | seed label | PRG domain, index | sampler |
/// |---|---|---|---|
/// | `r2l_challenges` | `hachi-co/r2l/c` | [`FS_DOMAIN_R2L_C`], attempt | `count` draws of the exact $\ell\_1$-ball sampler [`L1Ball::sample`] |
/// | `sumcheck_coins` | `hachi-co/sumcheck/coins` | [`FS_DOMAIN_SC_COINS`], 0 | $\alpha$, $\rho$, $\tau\_0$, $\tau\_1$ in this order |
/// | `round_coin` | `hachi-co/sumcheck/a` | [`FS_DOMAIN_SC_ROUND`], round | one element |
///
/// A field element is $K$ coordinates, each uniform in $\[0, q)$ by exact rejection
/// ([`hachi_co_sampling::uniform::uniform_mod_q`]).
///
/// **Attempts.** [`begin_attempt`](Challenger::begin_attempt)`(0)` saves the state; every
/// `begin_attempt(a)` restores the saved state and calls [`Transcript::append_attempt`]`(a)`
/// (the attempt counter of Fiat-Shamir with aborts, plan `docs/hachi-co-plan.md`, finding F5).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FsChallenger {
    transcript: Transcript,
    attempt_base: Option<Transcript>,
}

impl FsChallenger {
    /// A challenger on a fresh transcript for the protocol instance `domain`
    /// ([`Transcript::new`]).
    pub fn new(domain: &[u8]) -> Self {
        Self::from_transcript(Transcript::new(domain))
    }

    /// A challenger that continues `transcript` (no attempt started).
    pub fn from_transcript(transcript: Transcript) -> Self {
        Self {
            transcript,
            attempt_base: None,
        }
    }

    /// The current transcript, e.g. to compare the provers' and the verifier's states
    /// ([`Transcript::digest`]).
    pub fn transcript(&self) -> &Transcript {
        &self.transcript
    }
}

impl<const K: usize> Challenger<K> for FsChallenger {
    fn absorb_polys(&mut self, label: &'static str, polys: &[Poly]) {
        self.transcript
            .append_message(label.as_bytes(), &encode_polys(polys));
    }

    fn absorb_exts(&mut self, label: &'static str, exts: &[Ext<K>]) {
        self.transcript
            .append_message(label.as_bytes(), &encode_exts(exts));
    }

    fn absorb_words(&mut self, label: &'static str, words: &[Vec<u64>]) {
        self.transcript
            .append_message(label.as_bytes(), &encode_words(words));
    }

    fn absorb_u64(&mut self, label: &'static str, value: u64) {
        self.transcript.append_u64(label.as_bytes(), value);
    }

    fn context(&mut self, label: &str) -> [u8; 32] {
        self.transcript
            .fork(label.as_bytes())
            .challenge_seed(FS_LABEL_CONTEXT.as_bytes())
    }

    fn begin_attempt(&mut self, attempt: u64) -> Result<()> {
        let base = self
            .attempt_base
            .get_or_insert_with(|| self.transcript.clone());
        self.transcript = base.clone();
        self.transcript.append_attempt(attempt);
        Ok(())
    }

    fn r2l_challenges(
        &mut self,
        attempt: u64,
        count: usize,
        d: usize,
        omega: u64,
    ) -> Result<Vec<Vec<i64>>> {
        let ball = L1Ball::new(d, omega)?;
        let seed = self.transcript.challenge_seed(LABEL_R2L_C);
        let mut prg = ShakePrg::new(&seed, FS_DOMAIN_R2L_C.as_bytes(), attempt);
        Ok((0..count)
            .map(|_| ball.sample(&mut prg))
            .collect::<hachi_co_sampling::Result<Vec<_>>>()?)
    }

    fn sumcheck_coins(
        &mut self,
        field: &Fqk<K>,
        ell_pp: usize,
        log_n: usize,
        log_nu: usize,
    ) -> Result<SumcheckCoins<K>> {
        let q = field.modulus();
        let seed = self.transcript.challenge_seed(LABEL_SC_COINS);
        let mut prg = ShakePrg::new(&seed, FS_DOMAIN_SC_COINS.as_bytes(), 0);
        let alpha = uniform_ext::<K>(&mut prg, q)?;
        let rho = uniform_ext::<K>(&mut prg, q)?;
        let tau0 = (0..ell_pp + log_n)
            .map(|_| uniform_ext::<K>(&mut prg, q))
            .collect::<Result<Vec<_>>>()?;
        let tau1 = (0..log_nu)
            .map(|_| uniform_ext::<K>(&mut prg, q))
            .collect::<Result<Vec<_>>>()?;
        Ok(SumcheckCoins {
            alpha,
            rho,
            tau0,
            tau1,
        })
    }

    fn round_coin(&mut self, field: &Fqk<K>, round: usize) -> Result<Ext<K>> {
        if round == 0 {
            return Err(Error::Challenger(
                "sumcheck rounds are numbered from 1".into(),
            ));
        }
        let seed = self.transcript.challenge_seed(LABEL_SC_ROUND);
        let mut prg = ShakePrg::new(&seed, FS_DOMAIN_SC_ROUND.as_bytes(), round as u64);
        uniform_ext::<K>(&mut prg, field.modulus())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodings_are_self_describing() {
        let p = Poly(vec![1, 2, 3, 4]);
        let e = Ext::<4>([1, 2, 3, 4]);
        let polys = encode_polys(core::slice::from_ref(&p));
        let exts = encode_exts(&[e]);
        assert_ne!(polys, exts);
        assert_eq!(polys.len(), 1 + 8 + 8 + 4 * 8);
        assert_eq!(exts.len(), 1 + 8 + 8 + 4 * 8);
        assert_eq!(&polys[..9], b"P\x01\0\0\0\0\0\0\0");
        assert_eq!(&exts[..17], b"E\x04\0\0\0\0\0\0\0\x01\0\0\0\0\0\0\0");
        // Splitting the coefficients differently changes the encoding.
        let split = encode_polys(&[Poly(vec![1, 2]), Poly(vec![3, 4])]);
        assert_ne!(polys, split);
        assert_eq!(encode_polys(&[]), b"P\0\0\0\0\0\0\0\0");
        // Word vectors: the same words as the polynomial above, another tag; the split matters.
        let words = encode_words(&[vec![1, 2, 3, 4]]);
        assert_eq!(&words[1..], &polys[1..]);
        assert_eq!(words[0], b'W');
        assert_ne!(words, encode_words(&[vec![1, 2], vec![3, 4]]));
        assert_ne!(
            encode_words(&[vec![], vec![1]]),
            encode_words(&[vec![1], vec![]])
        );
        assert_eq!(encode_words(&[]), b"W\0\0\0\0\0\0\0\0");
    }
}
