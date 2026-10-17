//! Replay providers (feature `fixtures`): a [`PartyRandomness`] and a [`Challenger`] that serve
//! the values the Sage oracle recorded in a [`Fixture`], so that the block drivers reproduce the
//! fixture exactly (`docs/m1-spec.md`, "Randomness and coins").
//!
//! **Fixture keys** (JSON pointers; $p$ is the party, attempts and rounds as passed to the
//! methods; every key is read and checked when the provider is built):
//!
//! | Method | Fixture key | Shape |
//! |---|---|---|
//! | `reshare` | `/commit/reshare/<p-1>/<j-1>` for $j = 1, \dots, n-1$ (share $n$ is recomputed) | $n \times 2^\ell \times k$ |
//! | `mask(a, ..)` | `/ring_to_linear/attempts/<a>/provers/<p-1>/mask`, flattened | $\delta 2^m \times d$ |
//! | `rho(a, ..)` | `/ring_to_linear/attempts/<a>/provers/<p-1>/rho`, flattened | $m\_\rho \times d$ |
//! | `blinder(G0, ..)` | `/sumcheck/blinders/<p-1>/g0/s0` and `.../g0/s` | $k$; $\ell'' \times d\_0 \times k$ |
//! | `blinder(G1, ..)` | `/sumcheck/blinders/<p-1>/g1/s0` and `.../g1/s` | $k$; $\ell'' \times d\_1 \times k$ |
//! | `r2l_challenges(a, ..)` | `/ring_to_linear/attempts/<a>/c` | $2^r \times d$ |
//! | `sumcheck_coins` | `/sumcheck/coins/alpha`, `rho`, `tau0`, `tau1` | $k$; $k$; $(\ell'' + \lceil \log n \rceil) \times k$; $\lceil \log \nu \rceil \times k$ |
//! | `round_coin(r)` | `/sumcheck/coins/a/<r-1>` | $k$ |
//!
//! (`/ring_to_linear/attempts/<a>/provers/<p-1>/prover` is checked to be $p$.) The oracle's
//! resharing of party $p$ is `commit/reshare[p-1]` $= (\langle \mathbf f\_p \rangle\_1, \dots,
//! \langle \mathbf f\_p \rangle\_n)$ of its initial share `input/f_init_shares[p-1]`
//! (`ref/hachico_ref/commit.py`, `commit`).

use crate::data::{Blinder, BlinderKind, SumcheckCoins};
use crate::fixture::Fixture;
use crate::{Challenger, Error, PartyRandomness, Result};
use hachi_co_math::{Ext, Fqk, Poly};
use hachi_co_mpc::PartyId;

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
/// It serves the recorded draws and checks that each request matches the recording (the number of
/// parties, $\gamma$, $b$, the lengths and shapes, the modulus); a request that does not match,
/// or an attempt the oracle did not run, is an [`Error::Replay`]. `reshare` replays shares
/// $1, \dots, n-1$ and recomputes share $n = \mathbf x - \sum\_{j \lt n} \langle \mathbf x
/// \rangle\_j$ from the input it is given, exactly as [`crate::PrgRandomness`] does; for the
/// recorded input `input/f_init_shares[p-1]` this is the recorded share $n$. Unlike
/// [`crate::PrgRandomness`], a value may be requested any number of times.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayRandomness {
    party: PartyId,
    n: usize,
    q: u64,
    gamma: u64,
    b: u64,
    /// `reshare[j][t]`: coordinates of element $t$ of share $j + 1$.
    reshare: Vec<Vec<Vec<u64>>>,
    masks: Vec<Vec<i64>>,
    rhos: Vec<Vec<i64>>,
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
        let reshare = (0..n)
            .map(|j| fixture.residue_rows(&format!("/commit/reshare/{p0}/{j}")))
            .collect::<Result<Vec<_>>>()?;
        if reshare.iter().any(|s| s.len() != reshare[0].len()) {
            return Err(Error::Fixture(format!(
                "/commit/reshare/{p0}: the shares have different lengths"
            )));
        }
        let attempts = fixture
            .get("/ring_to_linear/attempts")?
            .as_array()
            .map_or(0, Vec::len);
        let mut masks = Vec::with_capacity(attempts);
        let mut rhos = Vec::with_capacity(attempts);
        for a in 0..attempts {
            let prefix = format!("/ring_to_linear/attempts/{a}/provers/{p0}");
            let recorded = fixture.usize(&format!("{prefix}/prover"))?;
            if recorded != party {
                return Err(Error::Fixture(format!(
                    "{prefix}/prover is {recorded}, expected {party}"
                )));
            }
            masks.push(fixture.ints(&format!("{prefix}/mask"))?);
            rhos.push(fixture.ints(&format!("{prefix}/rho"))?);
        }
        Ok(Self {
            party,
            n,
            q: fixture.q(),
            gamma: fixture.u64("/params/gamma")?,
            b: fixture.u64("/params/b")?,
            reshare,
            masks,
            rhos,
            g0: RecordedBlinder::load(fixture, &format!("/sumcheck/blinders/{p0}/g0"))?,
            g1: RecordedBlinder::load(fixture, &format!("/sumcheck/blinders/{p0}/g1"))?,
        })
    }

    /// The party whose randomness this is.
    pub fn party(&self) -> PartyId {
        self.party
    }

    /// The number of recorded attempts of the ring-to-linear step.
    pub fn attempts(&self) -> usize {
        self.masks.len()
    }

    fn recorded<'a>(
        &self,
        values: &'a [Vec<i64>],
        what: &str,
        attempt: u64,
        len: usize,
    ) -> Result<&'a Vec<i64>> {
        let v = usize::try_from(attempt)
            .ok()
            .and_then(|a| values.get(a))
            .ok_or_else(|| {
                Error::Replay(format!(
                    "no {what} recorded for attempt {attempt} (the oracle ran {} attempts)",
                    values.len()
                ))
            })?;
        if v.len() != len {
            return Err(Error::Replay(format!(
                "{what} of attempt {attempt} has {} entries, requested {len}",
                v.len()
            )));
        }
        Ok(v)
    }
}

impl<const K: usize> PartyRandomness<K> for ReplayRandomness {
    fn reshare(&mut self, field: &Fqk<K>, input: &[Ext<K>], n: usize) -> Result<Vec<Vec<Ext<K>>>> {
        check_modulus(field.modulus(), self.q)?;
        if n != self.n {
            return Err(Error::Replay(format!(
                "resharing among {n} parties, the recording has {}",
                self.n
            )));
        }
        let len = self.reshare.first().map_or(0, Vec::len);
        if input.len() != len {
            return Err(Error::Replay(format!(
                "resharing {} elements, the recording has {len}",
                input.len()
            )));
        }
        let mut shares = Vec::with_capacity(n);
        let mut last: Vec<Ext<K>> = input.iter().map(|x| field.reduce(x)).collect();
        for recorded in &self.reshare[..n - 1] {
            let share = to_exts::<K>(recorded)?;
            for (l, s) in last.iter_mut().zip(&share) {
                *l = field.sub(l, s);
            }
            shares.push(share);
        }
        shares.push(last);
        Ok(shares)
    }

    fn mask(&mut self, attempt: u64, gamma: u64, len: usize) -> Result<Vec<i64>> {
        if gamma != self.gamma {
            return Err(Error::Replay(format!(
                "mask width {gamma}, the recording has gamma = {}",
                self.gamma
            )));
        }
        Ok(self.recorded(&self.masks, "mask", attempt, len)?.clone())
    }

    fn rho(&mut self, attempt: u64, b: u64, len: usize) -> Result<Vec<i64>> {
        if b != self.b {
            return Err(Error::Replay(format!(
                "rho over S_{b}, the recording has b = {}",
                self.b
            )));
        }
        Ok(self.recorded(&self.rhos, "rho", attempt, len)?.clone())
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
}

/// The verifier's coins replayed from a fixture (key mapping in the
/// [module documentation](self)). The absorbs and
/// [`begin_attempt`](Challenger::begin_attempt) are ignored: the oracle's coins are public coins
/// that do not depend on the messages (`ref/README.md`, choice 14). A request that does not match
/// the recording (shape, $\omega$, the modulus, an attempt or round the oracle did not run) is an
/// [`Error::Replay`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayChallenger {
    q: u64,
    omega: u64,
    challenges: Vec<Vec<Vec<i64>>>,
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
        let attempts = fixture
            .get("/ring_to_linear/attempts")?
            .as_array()
            .map_or(0, Vec::len);
        let challenges = (0..attempts)
            .map(|a| fixture.int_rows(&format!("/ring_to_linear/attempts/{a}/c")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            q: fixture.q(),
            omega: fixture.u64("/params/omega")?,
            challenges,
            alpha: fixture.residues("/sumcheck/coins/alpha")?,
            rho: fixture.residues("/sumcheck/coins/rho")?,
            tau0: fixture.residue_rows("/sumcheck/coins/tau0")?,
            tau1: fixture.residue_rows("/sumcheck/coins/tau1")?,
            rounds: fixture.residue_rows("/sumcheck/coins/a")?,
        })
    }

    /// The number of recorded attempts of the ring-to-linear step.
    pub fn attempts(&self) -> usize {
        self.challenges.len()
    }

    /// The number of recorded sumcheck rounds ($\ell''$).
    pub fn rounds(&self) -> usize {
        self.rounds.len()
    }
}

impl<const K: usize> Challenger<K> for ReplayChallenger {
    fn absorb_polys(&mut self, _label: &'static str, _polys: &[Poly]) {}

    fn absorb_exts(&mut self, _label: &'static str, _exts: &[Ext<K>]) {}

    fn absorb_u64(&mut self, _label: &'static str, _value: u64) {}

    fn begin_attempt(&mut self, _attempt: u64) -> Result<()> {
        Ok(())
    }

    fn r2l_challenges(
        &mut self,
        attempt: u64,
        count: usize,
        d: usize,
        omega: u64,
    ) -> Result<Vec<Vec<i64>>> {
        let c = usize::try_from(attempt)
            .ok()
            .and_then(|a| self.challenges.get(a))
            .ok_or_else(|| {
                Error::Replay(format!(
                    "no challenges recorded for attempt {attempt} (the oracle ran {} attempts)",
                    self.challenges.len()
                ))
            })?;
        if omega != self.omega {
            return Err(Error::Replay(format!(
                "challenges with omega = {omega}, the recording has omega = {}",
                self.omega
            )));
        }
        if c.len() != count || c.iter().any(|ci| ci.len() != d) {
            return Err(Error::Replay(format!(
                "challenges of attempt {attempt} requested as {count} x {d}, recorded as {} x {}",
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
