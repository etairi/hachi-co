//! Replay providers (feature `fixtures`): a [`PartyRandomness`] and a [`Challenger`] that serve
//! the values the Sage oracle recorded in a [`Fixture`], so that the block drivers reproduce the
//! fixture exactly (`docs/m1-spec.md`, "Randomness and coins").
//!
//! **Fixture keys** (JSON pointers; $p$ is the party, iterations and rounds as passed to the
//! methods; every key is read and checked when the provider is built):
//!
//! | Method | Fixture key | Shape |
//! |---|---|---|
//! | `mask(j, ..)` | `/ring_to_linear/provers/<p-1>/masks/<j>`, flattened | $\delta 2^m \times d$, for $j \lt L$ |
//! | `rho_u(..)` | `/commit/rho_u/<p-1>`, flattened | $m\_u \times d$ |
//! | `rho_v(..)` | `/ring_to_linear/provers/<p-1>/rho_v`, flattened | $m\_v \times d$ |
//! | `rho_w(..)` | `/sumcheck/rho_w/<p-1>` | $k$ |
//! | `blinder(G0, ..)` | `/sumcheck/blinders/<p-1>/g0/s0` and `.../g0/s` | $k$; $\ell'' \times d\_0 \times k$ |
//! | `blinder(G1, ..)` | `/sumcheck/blinders/<p-1>/g1/s0` and `.../g1/s` | $k$; $\ell'' \times d\_1 \times k$ |
//! | `r2l_challenges(..)` | `/ring_to_linear/c` | $2^r \times d$ |
//! | `sumcheck_coins` | `/sumcheck/coins/alpha`, `rho`, `tau0`, `tau1` | $k$; $k$; $(\ell'' + \lceil \log n \rceil) \times k$; $\lceil \log \nu \rceil \times k$ |
//! | `round_coin(r)` | `/sumcheck/coins/a/<r-1>` | $k$ |
//! | `context(label)` | none (the oracle does not model the base PCS): a fixed hash of the label, see [`ReplayChallenger`] | 32 bytes |
//! | `base_pcs(index)` | none (the oracle does not model the base PCS): a fixed hash of the party and the index, see [`REPLAY_BASE_PCS_TAG`] | 32 bytes |
//!
//! (`/ring_to_linear/provers/<p-1>/prover` is checked to be $p$.) Commit draws nothing: since the
//! draft's commit `1096ea2` party $p$ commits to its input share `input/f_init_shares[p-1]` as it
//! is (`ref/hachico_ref/commit.py`, `commit`).

use crate::data::{Blinder, BlinderKind, SumcheckCoins};
use crate::fixture::Fixture;
use crate::{Challenger, Error, PartyRandomness, Result};
use hachi_co_math::{Ext, Fqk, Poly};
use hachi_co_mpc::PartyId;
use sha2::{Digest, Sha256};

/// Coordinates as an [`Ext`], checking that there are $K$ of them.
fn to_ext<const K: usize>(coords: &[u64]) -> Result<Ext<K>> {
    if coords.len() != K {
        return Err(Error::Replay(format!(
            "recorded field elements have {} coordinates, requested K = {K}",
            coords.len()
        )));
    }
    Ok(Ext(core::array::from_fn(|c| coords[c])))
}

fn to_exts<const K: usize>(rows: &[Vec<u64>]) -> Result<Vec<Ext<K>>> {
    rows.iter().map(|c| to_ext(c)).collect()
}

fn check_modulus(field_q: u64, q: u64) -> Result<()> {
    if field_q != q {
        return Err(Error::Replay(format!(
            "the field has q = {field_q}, the recording q = {q}"
        )));
    }
    Ok(())
}

/// A recorded blinder: $s\_0$ and the $\ell'' \times D$ coefficients, as coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RecordedBlinder {
    s0: Vec<u64>,
    s: Vec<Vec<Vec<u64>>>,
}

impl RecordedBlinder {
    fn load(fx: &Fixture, prefix: &str) -> Result<Self> {
        let s0 = fx.residues(&format!("{prefix}/s0"))?;
        let rows = fx
            .get(&format!("{prefix}/s"))?
            .as_array()
            .map_or(0, Vec::len);
        let s = (0..rows)
            .map(|j| fx.residue_rows(&format!("{prefix}/s/{j}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { s0, s })
    }
}

/// The [`PartyRandomness`] of party $p$ replayed from a fixture (key mapping in the
/// [module documentation](self)).
///
/// It serves the recorded draws and checks that each request matches the recording ($\gamma$,
/// $b$, the lengths and shapes, the modulus); a request that does not match, or an iteration the
/// oracle did not draw, is an [`Error::Replay`]. Unlike [`crate::PrgRandomness`], a value may be
/// requested any number of times.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayRandomness {
    party: PartyId,
    q: u64,
    gamma: u64,
    b: u64,
    /// `masks[j]`: the mask of iteration $j$.
    masks: Vec<Vec<i64>>,
    rho_u: Vec<i64>,
    rho_v: Vec<i64>,
    rho_w: Vec<u64>,
    g0: RecordedBlinder,
    g1: RecordedBlinder,
}

impl ReplayRandomness {
    /// The recorded randomness of party `party` ($1, \dots, n$) in `fixture`.
    ///
    /// # Errors
    /// [`Error::InvalidParty`] for a party outside $1, \dots, n$; [`Error::Fixture`] if a key is
    /// missing or malformed.
    pub fn new(fixture: &Fixture, party: PartyId) -> Result<Self> {
        let n = fixture.n();
        if party == 0 || party > n {
            return Err(Error::InvalidParty { party, n });
        }
        let p0 = party - 1;
        let prefix = format!("/ring_to_linear/provers/{p0}");
        let recorded = fixture.usize(&format!("{prefix}/prover"))?;
        if recorded != party {
            return Err(Error::Fixture(format!(
                "{prefix}/prover is {recorded}, expected {party}"
            )));
        }
        let iterations = fixture.usize("/ring_to_linear/iterations")?;
        let masks = (0..iterations)
            .map(|j| fixture.ints(&format!("{prefix}/masks/{j}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            party,
            q: fixture.q(),
            gamma: fixture.u64("/params/gamma")?,
            b: fixture.u64("/params/b")?,
            masks,
            rho_u: fixture.ints(&format!("/commit/rho_u/{p0}"))?,
            rho_v: fixture.ints(&format!("{prefix}/rho_v"))?,
            rho_w: fixture.residues(&format!("/sumcheck/rho_w/{p0}"))?,
            g0: RecordedBlinder::load(fixture, &format!("/sumcheck/blinders/{p0}/g0"))?,
            g1: RecordedBlinder::load(fixture, &format!("/sumcheck/blinders/{p0}/g1"))?,
        })
    }

    /// The party whose randomness this is.
    pub fn party(&self) -> PartyId {
        self.party
    }

    /// The number $L$ of recorded masks (iterations of the rejection sampling).
    pub fn iterations(&self) -> usize {
        self.masks.len()
    }
}

/// `Err(Replay)` unless the recorded `values` have `len` entries.
fn check_recorded_len(values: &[i64], what: &str, len: usize) -> Result<()> {
    if values.len() != len {
        return Err(Error::Replay(format!(
            "{what} has {} entries, requested {len}",
            values.len()
        )));
    }
    Ok(())
}

impl<const K: usize> PartyRandomness<K> for ReplayRandomness {
    fn mask(&mut self, iteration: u64, gamma: u64, len: usize) -> Result<Vec<i64>> {
        if gamma != self.gamma {
            return Err(Error::Replay(format!(
                "mask width {gamma}, the recording has gamma = {}",
                self.gamma
            )));
        }
        let mask = usize::try_from(iteration)
            .ok()
            .and_then(|j| self.masks.get(j))
            .ok_or_else(|| {
                Error::Replay(format!(
                    "no mask recorded for iteration {iteration} (the oracle drew {})",
                    self.masks.len()
                ))
            })?;
        check_recorded_len(mask, &format!("the mask of iteration {iteration}"), len)?;
        Ok(mask.clone())
    }

    fn rho_u(&mut self, b: u64, len: usize) -> Result<Vec<i64>> {
        if b != self.b || len != self.rho_u.len() {
            return Err(Error::Replay(format!(
                "rho^u of {len} entries over S_{b}, the recording has {} over S_{}",
                self.rho_u.len(),
                self.b
            )));
        }
        Ok(self.rho_u.clone())
    }

    fn rho_v(&mut self, b: u64, len: usize) -> Result<Vec<i64>> {
        if b != self.b {
            return Err(Error::Replay(format!(
                "rho^v over S_{b}, the recording has b = {}",
                self.b
            )));
        }
        check_recorded_len(&self.rho_v, "rho^v", len)?;
        Ok(self.rho_v.clone())
    }

    fn rho_w(&mut self, field: &Fqk<K>) -> Result<Ext<K>> {
        check_modulus(field.modulus(), self.q)?;
        to_ext(&self.rho_w)
    }

    /// The fixtures record the draft's pad only.
    fn rho_w_residues(&mut self, _q: u64, _len: usize) -> Result<Vec<u64>> {
        Err(Error::Replay(
            "the fixtures record a field pad, not a pad of residues".into(),
        ))
    }

    fn blinder(
        &mut self,
        field: &Fqk<K>,
        kind: BlinderKind,
        ell_pp: usize,
        degree: usize,
    ) -> Result<Blinder<K>> {
        check_modulus(field.modulus(), self.q)?;
        let recorded = match kind {
            BlinderKind::G0 => &self.g0,
            BlinderKind::G1 => &self.g1,
        };
        if recorded.s.len() != ell_pp || recorded.s.iter().any(|row| row.len() != degree) {
            return Err(Error::Replay(format!(
                "blinder {kind:?} requested as {ell_pp} x {degree}, recorded as {} x {}",
                recorded.s.len(),
                recorded.s.first().map_or(0, Vec::len)
            )));
        }
        Ok(Blinder {
            s0: to_ext(&recorded.s0)?,
            s: recorded
                .s
                .iter()
                .map(|row| to_exts(row))
                .collect::<Result<_>>()?,
        })
    }

    /// Not recorded, since the oracle has no base PCS: the SHA-256 of the framed fields
    /// [`REPLAY_BASE_PCS_TAG`], LE64($p$) and LE64(`index`), each preceded by its LE64 length.
    /// The same request gives the same seed. These seeds are public and equal across fixtures:
    /// they reproduce runs, and must not be used to test hiding.
    fn base_pcs(&mut self, index: u64) -> Result<[u8; 32]> {
        let mut h = Sha256::new();
        let (party, index) = ((self.party as u64).to_le_bytes(), index.to_le_bytes());
        for field in [REPLAY_BASE_PCS_TAG.as_bytes(), &party, &index] {
            h.update((field.len() as u64).to_le_bytes());
            h.update(field);
        }
        Ok(h.finalize().into())
    }
}

/// The tag of the replayed base-PCS seeds ([`ReplayRandomness`]'s `base_pcs`).
pub const REPLAY_BASE_PCS_TAG: &str = "hachi-co/replay/base-pcs";

/// The tag of the contexts of [`ReplayChallenger`].
pub const REPLAY_CONTEXT_TAG: &str = "hachi-co/replay/context";

/// The verifier's coins replayed from a fixture (key mapping in the
/// [module documentation](self)). The absorbs are ignored: the oracle's coins are public coins
/// that do not depend on the messages (`ref/README.md`, choice 14). A request that does not match
/// the recording (shape, $\omega$, the modulus, a round the oracle did not run) is an
/// [`Error::Replay`].
///
/// The oracle does not model the base PCS (its `IdealPCS`, `ref/README.md`, choice 13), so it
/// records no contexts: [`context`](Challenger::context)`(label)` is the SHA-256 hash of
/// $\mathrm{LE64}(|T|) \Vert T \Vert \mathrm{LE64}(|\mathit{label}|) \Vert \mathit{label}$ with
/// the tag $T$ = [`REPLAY_CONTEXT_TAG`], a fixed value per label that does not depend on the
/// messages either.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayChallenger {
    q: u64,
    omega: u64,
    challenges: Vec<Vec<i64>>,
    alpha: Vec<u64>,
    rho: Vec<u64>,
    tau0: Vec<Vec<u64>>,
    tau1: Vec<Vec<u64>>,
    rounds: Vec<Vec<u64>>,
}

impl ReplayChallenger {
    /// The recorded coins of `fixture`.
    ///
    /// # Errors
    /// [`Error::Fixture`] if a key is missing or malformed.
    pub fn new(fixture: &Fixture) -> Result<Self> {
        Ok(Self {
            q: fixture.q(),
            omega: fixture.u64("/params/omega")?,
            challenges: fixture.int_rows("/ring_to_linear/c")?,
            alpha: fixture.residues("/sumcheck/coins/alpha")?,
            rho: fixture.residues("/sumcheck/coins/rho")?,
            tau0: fixture.residue_rows("/sumcheck/coins/tau0")?,
            tau1: fixture.residue_rows("/sumcheck/coins/tau1")?,
            rounds: fixture.residue_rows("/sumcheck/coins/a")?,
        })
    }

    /// The number of recorded sumcheck rounds ($\ell''$).
    pub fn rounds(&self) -> usize {
        self.rounds.len()
    }
}

impl<const K: usize> Challenger<K> for ReplayChallenger {
    fn absorb_polys(&mut self, _label: &'static str, _polys: &[Poly]) {}

    fn absorb_exts(&mut self, _label: &'static str, _exts: &[Ext<K>]) {}

    fn absorb_words(&mut self, _label: &'static str, _words: &[Vec<u64>]) {}

    fn absorb_u64(&mut self, _label: &'static str, _value: u64) {}

    fn context(&mut self, label: &str) -> [u8; 32] {
        let mut h = Sha256::new();
        for field in [REPLAY_CONTEXT_TAG.as_bytes(), label.as_bytes()] {
            h.update((field.len() as u64).to_le_bytes());
            h.update(field);
        }
        h.finalize().into()
    }

    fn r2l_challenges(&mut self, count: usize, d: usize, omega: u64) -> Result<Vec<Vec<i64>>> {
        let c = &self.challenges;
        if omega != self.omega {
            return Err(Error::Replay(format!(
                "challenges with omega = {omega}, the recording has omega = {}",
                self.omega
            )));
        }
        if c.len() != count || c.iter().any(|ci| ci.len() != d) {
            return Err(Error::Replay(format!(
                "challenges requested as {count} x {d}, recorded as {} x {}",
                c.len(),
                c.first().map_or(0, Vec::len)
            )));
        }
        Ok(c.clone())
    }

    fn sumcheck_coins(
        &mut self,
        field: &Fqk<K>,
        ell_pp: usize,
        log_n: usize,
        log_nu: usize,
    ) -> Result<SumcheckCoins<K>> {
        check_modulus(field.modulus(), self.q)?;
        if self.tau0.len() != ell_pp + log_n || self.tau1.len() != log_nu {
            return Err(Error::Replay(format!(
                "tau_0, tau_1 requested with {} and {log_nu} entries, recorded with {} and {}",
                ell_pp + log_n,
                self.tau0.len(),
                self.tau1.len()
            )));
        }
        Ok(SumcheckCoins {
            alpha: to_ext(&self.alpha)?,
            rho: to_ext(&self.rho)?,
            tau0: to_exts(&self.tau0)?,
            tau1: to_exts(&self.tau1)?,
        })
    }

    fn round_coin(&mut self, field: &Fqk<K>, round: usize) -> Result<Ext<K>> {
        check_modulus(field.modulus(), self.q)?;
        let coords = round
            .checked_sub(1)
            .and_then(|r| self.rounds.get(r))
            .ok_or_else(|| {
                Error::Replay(format!(
                    "no coin recorded for round {round} (rounds 1..={})",
                    self.rounds.len()
                ))
            })?;
        to_ext(coords)
    }
}
