//! $n$ parties on threads: one thread per party over a [`MemoryTransport`] network.

use crate::{
    Endpoint, Error, Fault, Faulty, MemoryTransport, PartyId, Result, Stats, Tap, Transport,
    ZeroSharing, memory_network,
};
use hachi_co_sampling::ShakePrg;
use rand::Rng;
use std::time::Duration;

/// The endpoint type of [`run_in_memory`].
pub type MemoryEndpoint = Endpoint<Faulty<MemoryTransport>>;

/// PRG domain of the parties' network seeds (index: the party).
const DOMAIN_PARTY_SEED: &[u8] = b"hachi-co/mpc/party-seed";

/// The configuration of [`run_in_memory`].
#[derive(Clone, Debug)]
pub struct MemoryConfig {
    /// The number of parties $n \ge 1$.
    pub n: usize,
    /// The modulus $q$ of $\mathcal F\_{\mathsf{AH}}$.
    pub q: u64,
    /// How the zero sharings of $\mathcal F\_{\mathsf{AH}}$ are made.
    pub zero_sharing: ZeroSharing,
    /// The seed from which every party's network seed is derived (the pairwise seeds and the
    /// fresh zero sharings; the protocol's own randomness is separate). Use a fresh one per run.
    ///
    /// Deriving all parties' seeds from one seed makes in-process runs reproducible, and it is
    /// only that: whoever knows this seed knows every mask. In a deployment each party draws its
    /// own secret seed and passes it to [`Endpoint::connect`].
    pub seed: [u8; 32],
    /// How long a party waits for a frame before it fails with [`Error::Timeout`].
    pub timeout: Duration,
    /// A recorder of every frame sent, if wanted.
    pub tap: Option<Tap>,
    /// Faults to inject, by party.
    pub faults: Vec<(PartyId, Fault)>,
}

impl MemoryConfig {
    /// $n$ parties, modulus $q$, network seed `seed`; [`ZeroSharing::PairwiseSeeds`], a timeout
    /// of 60 s, no tap and no faults.
    pub fn new(n: usize, q: u64, seed: [u8; 32]) -> Self {
        Self {
            n,
            q,
            zero_sharing: ZeroSharing::default(),
            seed,
            timeout: Duration::from_secs(60),
            tap: None,
            faults: Vec::new(),
        }
    }

    /// With another zero-sharing mode.
    #[must_use]
    pub fn zero_sharing(mut self, mode: ZeroSharing) -> Self {
        self.zero_sharing = mode;
        self
    }

    /// With another timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Recording every frame in `tap`.
    #[must_use]
    pub fn tap(mut self, tap: &Tap) -> Self {
        self.tap = Some(tap.clone());
        self
    }

    /// With one more fault, injected into the frames of `party`.
    #[must_use]
    pub fn fault(mut self, party: PartyId, fault: Fault) -> Self {
        self.faults.push((party, fault));
        self
    }

    /// Party $p$'s network seed: the stream `(seed, "hachi-co/mpc/party-seed", p)`.
    pub fn party_seed(&self, party: PartyId) -> [u8; 32] {
        let mut s = [0u8; 32];
        ShakePrg::new(&self.seed, DOMAIN_PARTY_SEED, party as u64).fill_bytes(&mut s);
        s
    }
}

/// What one party of [`run_in_memory`] produced.
#[derive(Clone, Debug)]
pub struct PartyRun<T> {
    /// The party.
    pub id: PartyId,
    /// What the party function returned.
    pub output: T,
    /// The party's accounting, the set-up included.
    pub stats: Stats,
}

/// Runs `party` for $p = 1, \dots, n$, each on its own thread with its own
/// [`MemoryEndpoint`] (connected with [`Endpoint::connect`]), and returns the parties' results
/// in party order: `Ok` with what `party` returned and the party's accounting, or the error of
/// the connection if the party could not connect. A party's endpoint is dropped as soon as its
/// function returns, so the other parties never wait for a party that has stopped. A panic in a
/// party function is propagated after all threads have finished.
///
/// # Errors
/// [`Error::Transport`] if $n = 0$; [`Error::InvalidParty`] if a fault names a party outside
/// $1, \dots, n$.
pub fn run_in_memory<T: Send>(
    config: &MemoryConfig,
    party: impl Fn(&mut MemoryEndpoint) -> T + Sync,
) -> Result<Vec<Result<PartyRun<T>>>> {
    let n = config.n;
    if let Some(&(p, _)) = config.faults.iter().find(|(p, _)| *p == 0 || *p > n) {
        return Err(Error::InvalidParty { party: p, n });
    }
    let transports = memory_network(n, config.timeout, config.tap.as_ref())?;
    let party = &party;
    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = transports
            .into_iter()
            .map(|transport| {
                let id = transport.id();
                let faults = config
                    .faults
                    .iter()
                    .filter(|(p, _)| *p == id)
                    .map(|(_, f)| f.clone())
                    .collect();
                let seed = config.party_seed(id);
                std::thread::Builder::new()
                    .name(format!("party-{id}"))
                    .spawn_scoped(scope, move || -> Result<PartyRun<T>> {
                        let transport = Faulty::new(transport, faults);
                        let mut endpoint =
                            Endpoint::connect(transport, config.q, config.zero_sharing, seed)?;
                        let output = party(&mut endpoint);
                        let stats = endpoint.stats().clone();
                        drop(endpoint);
                        Ok(PartyRun { id, output, stats })
                    })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.map(|h| h.join()))
            .collect::<Vec<_>>()
    });
    let mut out = Vec::with_capacity(n);
    let mut panic = None;
    for r in results {
        match r {
            Ok(Ok(run)) => out.push(run),
            Ok(Err(payload)) => {
                panic.get_or_insert(payload);
            }
            Err(e) => {
                return Err(Error::Transport {
                    label: "run_in_memory",
                    message: format!("could not spawn a party thread: {e}"),
                });
            }
        }
    }
    if let Some(payload) = panic {
        std::panic::resume_unwind(payload);
    }
    Ok(out)
}
