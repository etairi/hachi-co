//! The $n$-party realization of [`Comm`] over a point-to-point [`Transport`].

use crate::{Comm, Error, Frame, PartyId, Result, Stats, Transport};
use hachi_co_sampling::ShakePrg;
use hachi_co_sampling::uniform::uniform_mod_q;
use rand::Rng;

/// The label of the set-up round of [`Endpoint::connect`].
pub const LABEL_SETUP: &str = "hachi-co/mpc/setup";

/// PRG domain of the pairwise seeds (index: the receiving party).
const DOMAIN_PAIRWISE: &[u8] = b"hachi-co/mpc/pairwise-seed";
/// PRG domain of the pairwise masks of [`ZeroSharing::PairwiseSeeds`] (index: the round).
const DOMAIN_ZERO: &[u8] = b"hachi-co/mpc/zero-sharing";
/// PRG domain of the sharings of [`ZeroSharing::Fresh`] (index: the round).
const DOMAIN_FRESH: &[u8] = b"hachi-co/mpc/fresh-zero-sharing";

/// How the parties of an [`Endpoint`] obtain the random additive sharings of 0 that mask their
/// $\mathcal F\_{\mathsf{AH}}$ output shares. Every party of a run must
/// use the same mode; [`Endpoint::connect`] checks it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ZeroSharing {
    /// Pseudorandom zero sharings from pairwise seeds: at connection, party $i$ sends a fresh
    /// 32-byte seed $s\_{ij}$ to every party $j \gt i$; in round $r$ party $i$'s share is
    /// $\mathbf z\_i = \sum\_{j \gt i} \mathrm{PRG}(s\_{ij}, r) - \sum\_{j \lt i}
    /// \mathrm{PRG}(s\_{ji}, r)$, each PRG output uniform in $\mathbb Z\_q^{len}$, so
    /// $\sum\_i \mathbf z\_i = 0$. No communication per call beyond the broadcast. Against a
    /// coalition that misses at least two parties, the honest shares are uniform subject to their
    /// sum, as far as the PRG (SHAKE128, [`ShakePrg`]) is pseudorandom.
    #[default]
    PairwiseSeeds,
    /// Exactly uniform zero sharings: in every $\mathcal F\_{\mathsf{AH}}$ call each party $i$
    /// samples $(\mathbf r\_{i,1}, \dots, \mathbf r\_{i,n})$ uniform subject to
    /// $\sum\_j \mathbf r\_{i,j} = 0$ and sends $\mathbf r\_{i,j}$ to party $j$; its share is
    /// $\mathbf z\_i = \sum\_j \mathbf r\_{j,i}$. This costs one more round and $(n-1) \cdot len$
    /// more words per party and call.
    Fresh,
}

impl ZeroSharing {
    /// The code of the mode in the set-up message.
    fn code(self) -> u64 {
        match self {
            Self::PairwiseSeeds => 1,
            Self::Fresh => 2,
        }
    }
}

/// One party's realization of [`Comm`] for $n$ parties over a [`Transport`].
///
/// - **Framing.** Every communication round has an index ($0$ for the set-up, then
///   $1, 2, \dots$); every frame carries its call's label and round index, and a frame with
///   another label or index is [`Error::Desync`]. Frames from one party to another arrive in
///   order, so the parties need no other synchronisation.
/// - **$\mathcal F\_{\mathsf{AH}}$** (draft `4_preliminaries.tex`:185-200; its realization is
///   left open at line 201) as the authors specified (2026-10-19): the parties apply the map
///   locally (the party code passes $\mathbf y\_i = L(\mathbf x\_i)$), each adds its share
///   $\mathbf z\_i$ of a random additive sharing of 0 ([`ZeroSharing`]), broadcasts
///   $\mathbf m\_i = \mathbf y\_i + \mathbf z\_i \bmod q$, and everyone outputs
///   $\sum\_i \mathbf m\_i = \sum\_i \mathbf y\_i \bmod q$. In the semi-honest model this reveals
///   only the sum: against a coalition that misses at least two parties, the honest
///   $\mathbf m\_i$ are uniform subject to their sum; with $n - 1$ corrupted parties the output
///   already determines the honest input.
/// - **The other calls** send their vectors as they are: [`Comm::gather`] to one party,
///   [`Comm::any`] and [`Comm::all_gather`] to everyone, [`Comm::exchange`] pairwise.
/// - **Accounting**: [`Endpoint::stats`], per label.
///
/// An `Endpoint` never panics on what it receives. After an error the run is broken: the
/// caller should stop and drop the endpoint, which closes its channels so that the other parties
/// fail too instead of waiting.
pub struct Endpoint<T> {
    transport: T,
    q: u64,
    mode: ZeroSharing,
    seed: [u8; 32],
    pairwise: Vec<Option<[u8; 32]>>,
    round: u64,
    stats: Stats,
}

impl<T: Transport> core::fmt::Debug for Endpoint<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Endpoint")
            .field("n", &self.transport.n())
            .field("id", &self.transport.id())
            .field("q", &self.q)
            .field("mode", &self.mode)
            .field("round", &self.round)
            .finish_non_exhaustive()
    }
}

/// The 32-byte seed in four little-endian words.
fn seed_to_words(seed: &[u8; 32]) -> [u64; 4] {
    let mut words = [0u64; 4];
    for (w, chunk) in words.iter_mut().zip(seed.as_chunks::<8>().0) {
        *w = u64::from_le_bytes(*chunk);
    }
    words
}

/// The inverse of [`seed_to_words`].
fn words_to_seed(words: &[u64]) -> [u8; 32] {
    let mut seed = [0u8; 32];
    for (chunk, w) in seed.as_chunks_mut::<8>().0.iter_mut().zip(words) {
        *chunk = w.to_le_bytes();
    }
    seed
}

impl<T: Transport> Endpoint<T> {
    /// Connects this party: checks $q$, then runs the set-up round ([`LABEL_SETUP`], round 0), in
    /// which every party tells every other one its $q$, its [`ZeroSharing`] mode and $n$, and,
    /// with [`ZeroSharing::PairwiseSeeds`], party $i$ sends $s\_{ij}$ to every $j \gt i$. All
    /// parties must call it at the same time. `seed` is this party's private 32-byte seed for the
    /// network's randomness (the pairwise seeds it chooses and the [`ZeroSharing::Fresh`]
    /// sharings); use a fresh one for every run.
    ///
    /// # Errors
    /// [`Error::InvalidModulus`] unless $2 \le q \lt 2^{63}$; [`Error::Malformed`] if another
    /// party uses another $q$, mode or $n$, or sends a set-up message of the wrong length; the
    /// transport's errors.
    pub fn connect(transport: T, q: u64, mode: ZeroSharing, seed: [u8; 32]) -> Result<Self> {
        if !(2..1 << 63).contains(&q) {
            return Err(Error::InvalidModulus(q));
        }
        let n = transport.n();
        let id = transport.id();
        let mut endpoint = Self {
            transport,
            q,
            mode,
            seed,
            pairwise: vec![None; n],
            round: 0,
            stats: Stats::default(),
        };
        let label = LABEL_SETUP;
        let entry = endpoint.stats.entry(label);
        entry.calls += 1;
        if n == 1 {
            return Ok(endpoint);
        }
        entry.rounds += 1;
        let header = [q, mode.code(), n as u64];
        for j in (1..=n).filter(|&j| j != id) {
            let mut payload = header.to_vec();
            if mode == ZeroSharing::PairwiseSeeds && j > id {
                let mut s = [0u8; 32];
                ShakePrg::new(&endpoint.seed, DOMAIN_PAIRWISE, j as u64).fill_bytes(&mut s);
                payload.extend(seed_to_words(&s));
                endpoint.pairwise[j - 1] = Some(s);
            }
            endpoint.send(label, j, 0, payload)?;
        }
        for j in (1..=n).filter(|&j| j != id) {
            let payload = endpoint.recv(label, j, 0)?;
            let with_seed = mode == ZeroSharing::PairwiseSeeds && j < id;
            let expected = header.len() + if with_seed { 4 } else { 0 };
            let malformed = |message: String| Error::Malformed {
                label,
                party: j,
                message,
            };
            if payload.len() != expected {
                return Err(malformed(format!(
                    "set-up message of {} words, expected {expected}",
                    payload.len()
                )));
            }
            if payload[..3] != header {
                return Err(malformed(format!(
                    "(q, zero-sharing mode, n) = ({}, {}, {}), this party has ({q}, {}, {n})",
                    payload[0],
                    payload[1],
                    payload[2],
                    mode.code()
                )));
            }
            if with_seed {
                endpoint.pairwise[j - 1] = Some(words_to_seed(&payload[3..]));
            }
        }
        Ok(endpoint)
    }

    /// This party's accounting so far (the set-up included).
    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    /// The index of the last communication round so far ($0$ after the set-up).
    pub fn round(&self) -> u64 {
        self.round
    }

    /// The zero-sharing mode.
    pub fn zero_sharing(&self) -> ZeroSharing {
        self.mode
    }

    /// The transport.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Starts a collective call of `rounds` communication rounds and returns the index of its
    /// first round.
    fn begin(&mut self, label: &'static str, rounds: u64) -> u64 {
        let communicates = self.transport.n() > 1;
        let entry = self.stats.entry(label);
        entry.calls += 1;
        if communicates {
            entry.rounds += rounds;
        }
        let first = self.round + 1;
        self.round += rounds;
        first
    }

    fn send(
        &mut self,
        label: &'static str,
        to: PartyId,
        round: u64,
        payload: Vec<u64>,
    ) -> Result<()> {
        let bytes = 8 * payload.len() as u64;
        self.transport
            .send(label, to, Frame::new(label, round, payload))?;
        let entry = self.stats.entry(label);
        entry.frames_sent += 1;
        entry.bytes_sent += bytes;
        Ok(())
    }

    fn recv(&mut self, label: &'static str, from: PartyId, round: u64) -> Result<Vec<u64>> {
        let frame = self.transport.recv(label, from)?;
        if frame.label != label || frame.round != round {
            return Err(Error::Desync {
                label,
                round,
                party: from,
                got: format!("{} (round {})", frame.label, frame.round),
            });
        }
        let entry = self.stats.entry(label);
        entry.frames_received += 1;
        entry.bytes_received += 8 * frame.payload.len() as u64;
        Ok(frame.payload)
    }

    /// Sends `payload` to every other party.
    fn broadcast(&mut self, label: &'static str, round: u64, payload: &[u64]) -> Result<()> {
        let id = self.transport.id();
        for j in (1..=self.transport.n()).filter(|&j| j != id) {
            self.send(label, j, round, payload.to_vec())?;
        }
        Ok(())
    }

    /// Receives a vector of `len` residues below $q$ from party `from`.
    fn recv_residues(
        &mut self,
        label: &'static str,
        from: PartyId,
        round: u64,
        len: usize,
    ) -> Result<Vec<u64>> {
        let v = self.recv(label, from, round)?;
        if v.len() != len {
            return Err(Error::LengthMismatch {
                label,
                expected: len,
                actual: v.len(),
            });
        }
        if let Some(x) = v.iter().find(|&&x| x >= self.q) {
            return Err(Error::Malformed {
                label,
                party: from,
                message: format!("{x} is not a residue below q = {}", self.q),
            });
        }
        Ok(v)
    }

    /// `len` residues uniform in $\[0, q)$ from the stream `(seed, domain, index)`.
    fn expand(
        &self,
        label: &'static str,
        seed: &[u8; 32],
        domain: &[u8],
        index: u64,
        len: usize,
    ) -> Result<Vec<u64>> {
        uniform_mod_q(&mut ShakePrg::new(seed, domain, index), self.q, len).map_err(|e| {
            Error::Transport {
                label,
                message: format!("sampling a zero sharing: {e}"),
            }
        })
    }

    /// This party's share of the pseudorandom zero sharing of `round`
    /// ([`ZeroSharing::PairwiseSeeds`]).
    fn zero_share_pairwise(&self, label: &'static str, round: u64, len: usize) -> Result<Vec<u64>> {
        let (n, id, q) = (self.transport.n(), self.transport.id(), self.q);
        let mut z = vec![0u64; len];
        for j in (1..=n).filter(|&j| j != id) {
            let seed = self.pairwise[j - 1].ok_or_else(|| Error::Transport {
                label,
                message: format!("no pairwise seed with party {j}"),
            })?;
            let r = self.expand(label, &seed, DOMAIN_ZERO, round, len)?;
            for (zi, ri) in z.iter_mut().zip(r) {
                *zi = if j > id {
                    (*zi + ri) % q
                } else {
                    (*zi + q - ri) % q
                };
            }
        }
        Ok(z)
    }

    /// This party's share of a fresh uniform zero sharing ([`ZeroSharing::Fresh`]): one round.
    fn zero_share_fresh(
        &mut self,
        label: &'static str,
        round: u64,
        len: usize,
    ) -> Result<Vec<u64>> {
        let (n, id, q) = (self.transport.n(), self.transport.id(), self.q);
        let mut prg = ShakePrg::new(&self.seed, DOMAIN_FRESH, round);
        let mut z = vec![0u64; len];
        for j in (1..=n).filter(|&j| j != id) {
            let r = uniform_mod_q(&mut prg, q, len).map_err(|e| Error::Transport {
                label,
                message: format!("sampling a zero sharing: {e}"),
            })?;
            for (zi, ri) in z.iter_mut().zip(&r) {
                *zi = (*zi + q - ri) % q;
            }
            self.send(label, j, round, r)?;
        }
        for j in (1..=n).filter(|&j| j != id) {
            let r = self.recv_residues(label, j, round, len)?;
            for (zi, ri) in z.iter_mut().zip(r) {
                *zi = (*zi + ri) % q;
            }
        }
        Ok(z)
    }
}

impl<T: Transport> Comm for Endpoint<T> {
    fn n(&self) -> usize {
        self.transport.n()
    }

    fn id(&self) -> PartyId {
        self.transport.id()
    }

    fn modulus(&self) -> u64 {
        self.q
    }

    /// The masked broadcast (see [`Endpoint`]).
    fn fah(&mut self, label: &'static str, local: &[u64]) -> Result<Vec<u64>> {
        let (n, id, q) = (self.transport.n(), self.transport.id(), self.q);
        let len = local.len();
        let rounds = if n > 1 && self.mode == ZeroSharing::Fresh {
            2
        } else {
            1
        };
        let first = self.begin(label, rounds);
        let mut sum: Vec<u64> = local.iter().map(|x| x % q).collect();
        if n == 1 {
            return Ok(sum);
        }
        let z = match self.mode {
            ZeroSharing::PairwiseSeeds => self.zero_share_pairwise(label, first, len)?,
            ZeroSharing::Fresh => self.zero_share_fresh(label, first, len)?,
        };
        for (s, zi) in sum.iter_mut().zip(z) {
            *s = (*s + zi) % q;
        }
        let round = first + rounds - 1;
        self.broadcast(label, round, &sum)?;
        for j in (1..=n).filter(|&j| j != id) {
            let m = self.recv_residues(label, j, round, len)?;
            for (s, mj) in sum.iter_mut().zip(m) {
                *s = (*s + mj) % q;
            }
        }
        Ok(sum)
    }

    fn gather(
        &mut self,
        label: &'static str,
        to: PartyId,
        local: &[i64],
    ) -> Result<Option<Vec<Vec<i64>>>> {
        let (n, id) = (self.transport.n(), self.transport.id());
        if to == 0 || to > n {
            return Err(Error::InvalidParty { party: to, n });
        }
        let round = self.begin(label, 1);
        if id != to {
            // Two's complement: `as` is the bit-for-bit reinterpretation, undone on receipt.
            self.send(label, to, round, local.iter().map(|&x| x as u64).collect())?;
            return Ok(None);
        }
        let mut all = Vec::with_capacity(n);
        for j in 1..=n {
            if j == id {
                all.push(local.to_vec());
            } else {
                let v = self.recv(label, j, round)?;
                all.push(v.into_iter().map(|w| w as i64).collect());
            }
        }
        Ok(Some(all))
    }

    fn any(&mut self, label: &'static str, local: bool) -> Result<bool> {
        let (n, id) = (self.transport.n(), self.transport.id());
        let round = self.begin(label, 1);
        if n == 1 {
            return Ok(local);
        }
        self.broadcast(label, round, &[u64::from(local)])?;
        let mut out = local;
        for j in (1..=n).filter(|&j| j != id) {
            let v = self.recv(label, j, round)?;
            match v.as_slice() {
                [0] => {}
                [1] => out = true,
                _ => {
                    return Err(Error::Malformed {
                        label,
                        party: j,
                        message: format!("a flag is one word, 0 or 1; got {v:?}"),
                    });
                }
            }
        }
        Ok(out)
    }

    fn exchange(&mut self, label: &'static str, outgoing: Vec<Vec<u64>>) -> Result<Vec<Vec<u64>>> {
        let (n, id) = (self.transport.n(), self.transport.id());
        if outgoing.len() != n {
            return Err(Error::WrongMessageCount {
                label,
                expected: n,
                actual: outgoing.len(),
            });
        }
        let round = self.begin(label, 1);
        let mut incoming = vec![Vec::new(); n];
        for (j, v) in (1..=n).zip(outgoing) {
            if j == id {
                incoming[j - 1] = v;
            } else {
                self.send(label, j, round, v)?;
            }
        }
        for j in (1..=n).filter(|&j| j != id) {
            incoming[j - 1] = self.recv(label, j, round)?;
        }
        Ok(incoming)
    }

    fn all_gather(&mut self, label: &'static str, local: &[u64]) -> Result<Vec<Vec<u64>>> {
        let (n, id) = (self.transport.n(), self.transport.id());
        let round = self.begin(label, 1);
        if n == 1 {
            return Ok(vec![local.to_vec()]);
        }
        self.broadcast(label, round, local)?;
        let mut all = Vec::with_capacity(n);
        for j in 1..=n {
            if j == id {
                all.push(local.to_vec());
            } else {
                let v = self.recv(label, j, round)?;
                if v.len() != local.len() {
                    return Err(Error::LengthMismatch {
                        label,
                        expected: local.len(),
                        actual: v.len(),
                    });
                }
                all.push(v);
            }
        }
        Ok(all)
    }
}
