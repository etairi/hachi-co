//! A harness that runs each party of Hachi-Co as its own process, over TCP
//! ([`hachi_co_mpc::TcpTransport`]): one party's $\mathsf{Commit}$ and $\mathsf{Eval}$ with
//! Hachi as the base PCS ([`run_party`]), and the verifier of what a party wrote
//! ([`verify_output`]). The program `hachi-co-party` (`src/bin/hachi-co-party.rs`) is a command
//! line around it.
//!
//! **A test and benchmark harness, not a deployment.**
//! - The instance is dealer-free but not private. Every party derives the polynomial
//!   $\mathbf f$, the point $\mathbf x$ and the claim $y = f(\mathbf x)$ from the instance seed,
//!   which all parties share ([`Instance::derive`]), and computes its own additive share of
//!   $\mathbf f$ from the same seed ([`Instance::share`]). Whoever knows the instance seed knows
//!   $\mathbf f$ and every share; in a deployment each party would hold its share from elsewhere
//!   and the claim would be computed jointly.
//! - The connections are plaintext TCP ([`hachi_co_mpc::TcpTransport`], "Not a secure channel").
//! - The parameters are insecure test choices: those of the presets that are not `secure-*`,
//!   and those of [`HachiBase::setup`] at every preset.
//!
//! # Configuration
//!
//! One JSON file per party ([`PartyConfig`]):
//!
//! ```json
//! {
//!   "preset": "toy-n3",
//!   "n": 3,
//!   "party": 1,
//!   "addresses": ["127.0.0.1:7101", "127.0.0.1:7102", "127.0.0.1:7103"],
//!   "session": "5e55...",
//!   "setup_seed": "0101...",
//!   "instance_seed": "0202...",
//!   "party_seed": "1111...",
//!   "output": "out/party-1",
//!   "connect_timeout_ms": 30000,
//!   "recv_timeout_ms": 60000
//! }
//! ```
//!
//! where the session id and the seeds are 32 bytes in 64 hexadecimal digits (shortened here),
//! and the two timeouts are optional (with these defaults). All parties of a run share the
//! preset, $n$, the
//! addresses, the session id, the setup seed and the instance seed; each has its own index,
//! party seed and output directory.
//!
//! # Derivations
//!
//! | Value | Derived from |
//! |---|---|
//! | the context: the parameters and $\mathbf A, \mathbf B, \mathbf D$ | [`crate::setup`] on the preset's derived parameters, with the setup seed |
//! | the base PCS | [`HachiBase::setup`] for that context, with the setup seed |
//! | $\mathbf f \in \mathbb F\_{q^k}^{2^\ell}$, then $\mathbf x \in \mathbb F\_{q^k}^\ell$ | the PRG stream (instance seed, [`DOMAIN_INSTANCE`], 0), every coordinate by [`uniform_mod_q`] |
//! | $y = f(\mathbf x)$ | [`hachi_co_conv::evaluate`] |
//! | the share of party $p \lt n$ | the stream (instance seed, [`DOMAIN_SHARE`], $p$), $2^\ell$ elements as for $\mathbf f$ |
//! | the share of party $n$ | $\mathbf f$ minus the other shares |
//! | the party's protocol randomness | [`PrgRandomness`] with the party seed |
//! | the party's network seed ([`Endpoint::connect`]) | the first 32 bytes of the stream (party seed, [`DOMAIN_NETWORK_SEED`], 0) |
//!
//! The proof depends only on the context, the base PCS, the shares, the protocol randomness and
//! the statement: the network seeds change only the masks of $\mathcal F\_{\mathsf{AH}}$, which
//! cancel in its outputs, and the other collective calls deliver their vectors unchanged. A run
//! over TCP therefore gives the proof of the same parties over the
//! in-memory network ([`prove`] over [`hachi_co_mpc::run_in_memory`]), whatever the network
//! seeds; `tests/processes.rs` checks it.
//!
//! # Output
//!
//! [`run_party`] creates the output directory if needed and writes two files into it:
//! - [`PROOF_FILE`]: the proof's canonical encoding ([`EvalProof::to_bytes`]);
//! - [`STATEMENT_FILE`]: the public statement as JSON ([`Statement`]): the preset, $n$, the
//!   setup seed, the commitment $\mathbf u$, the point $\mathbf x$ and the claim $y$.
//!
//! Every party writes the same two files. [`verify_output`] reads them, rebuilds the context and
//! the base PCS from the preset and the setup seed, and runs [`verify`].

use crate::error::{DecodeError, Error, VerifyError};
use crate::proof::EvalProof;
use crate::prover::{commit, eval};
use crate::verifier::verify;
use crate::{Context, HachiBase};
use hachi_co_core::{Ext, Poly, PrgRandomness};
use hachi_co_mpc::{Comm, Endpoint, PartyId, Stats, TcpConfig, TcpTransport, ZeroSharing};
use hachi_co_sampling::ShakePrg;
use hachi_co_sampling::uniform::uniform_mod_q;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::net::{SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The extension degree $k$ of every preset (toy, development and secure).
pub const K: usize = 4;

/// The PRG domain of the instance (index 0): $\mathbf f$, then $\mathbf x$.
pub const DOMAIN_INSTANCE: &[u8] = b"hachi-co/harness/instance";
/// The PRG domain of the shares (index: the party $p \lt n$).
pub const DOMAIN_SHARE: &[u8] = b"hachi-co/harness/share";
/// The PRG domain of a party's network seed (index 0).
pub const DOMAIN_NETWORK_SEED: &[u8] = b"hachi-co/harness/network-seed";

/// The name of the proof's file in an output directory.
pub const PROOF_FILE: &str = "proof.bin";
/// The name of the statement's file in an output directory.
pub const STATEMENT_FILE: &str = "statement.json";

/// The default connect timeout of a [`PartyConfig`], in milliseconds.
pub const DEFAULT_CONNECT_TIMEOUT_MS: u64 = 30_000;
/// The default receive timeout of a [`PartyConfig`], in milliseconds.
pub const DEFAULT_RECV_TIMEOUT_MS: u64 = 60_000;

/// Why the harness stopped.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HarnessError {
    /// A configuration or a statement that does not parse, or does not fit: an unknown preset,
    /// an $n$ other than the preset's, a party index outside $1, \dots, n$, not $n$ addresses, a
    /// seed that is not 64 hexadecimal digits.
    #[error("configuration: {0}")]
    Config(String),
    /// A file could not be read or written.
    #[error("{path}: {message}")]
    Io {
        /// The file.
        path: PathBuf,
        /// What failed.
        message: String,
    },
    /// The party's address could not be bound, for example because another process uses it.
    #[error("binding {address}: {message}")]
    Bind {
        /// The address.
        address: SocketAddr,
        /// What failed.
        message: String,
    },
    /// The network failed outside the protocol: the connections, a handshake or the set-up
    /// round of [`Endpoint::connect`].
    #[error("network: {0}")]
    Network(#[from] hachi_co_mpc::Error),
    /// The protocol failed: $\mathsf{Setup}$, the instance, $\mathsf{Commit}$ or
    /// $\mathsf{Eval}$ (whose errors include those of the network during the protocol).
    #[error("protocol: {0}")]
    Protocol(#[from] Error),
    /// The proof's file is not the encoding of a proof for the statement's context.
    #[error("proof encoding: {0}")]
    Decode(#[from] DecodeError),
    /// The verifier rejected the proof.
    #[error("rejected: {0}")]
    Rejected(#[from] VerifyError),
}

/// 32 bytes as 64 lowercase hexadecimal digits.
pub fn to_hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// 64 hexadecimal digits (either case) as 32 bytes.
///
/// # Errors
/// [`HarnessError::Config`] for anything else.
pub fn parse_hex32(text: &str) -> Result<[u8; 32], HarnessError> {
    let digits = text.as_bytes();
    if digits.len() != 64 {
        return Err(HarnessError::Config(format!(
            "expected 64 hexadecimal digits, got {} characters",
            text.chars().count()
        )));
    }
    let digit = |c: u8| -> Result<u8, HarnessError> {
        char::from(c)
            .to_digit(16)
            .and_then(|d| u8::try_from(d).ok())
            .ok_or_else(|| {
                HarnessError::Config(format!("{:?} is not a hexadecimal digit", char::from(c)))
            })
    };
    let mut out = [0u8; 32];
    for (byte, pair) in out.iter_mut().zip(digits.as_chunks::<2>().0) {
        *byte = (digit(pair[0])? << 4) | digit(pair[1])?;
    }
    Ok(out)
}

/// Serde for a 32-byte value as 64 hexadecimal digits ([`to_hex`], [`parse_hex32`]).
mod hex32 {
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&super::to_hex(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let text = String::deserialize(d)?;
        super::parse_hex32(&text).map_err(D::Error::custom)
    }
}

fn default_connect_timeout_ms() -> u64 {
    DEFAULT_CONNECT_TIMEOUT_MS
}

fn default_recv_timeout_ms() -> u64 {
    DEFAULT_RECV_TIMEOUT_MS
}

/// One party's configuration, the JSON file of `hachi-co-party` (see the module
/// documentation). The party seed is the party's secret in a deployment; in this harness the
/// instance seed reveals every share anyway.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartyConfig {
    /// The preset ([`hachi_co_params::preset`]), for example `toy-n3`.
    pub preset: String,
    /// The number of parties $n$, which must be the preset's.
    pub n: usize,
    /// This party's index $p \in \lbrace 1, \dots, n \rbrace$.
    pub party: PartyId,
    /// The socket addresses of the $n$ parties, party $p$ at index $p - 1$
    /// ([`TcpConfig::addresses`]).
    pub addresses: Vec<SocketAddr>,
    /// The session id ([`TcpConfig::session`]): fresh for every run.
    #[serde(with = "hex32")]
    pub session: [u8; 32],
    /// The setup seed: the public matrices of the context and of the base PCS.
    #[serde(with = "hex32")]
    pub setup_seed: [u8; 32],
    /// The instance seed: $\mathbf f$, $\mathbf x$ and the shares.
    #[serde(with = "hex32")]
    pub instance_seed: [u8; 32],
    /// This party's seed: its protocol randomness and its network seed.
    #[serde(with = "hex32")]
    pub party_seed: [u8; 32],
    /// The output directory ([`PROOF_FILE`], [`STATEMENT_FILE`]).
    pub output: PathBuf,
    /// The connect timeout of the [`TcpConfig`], in milliseconds; optional, default
    /// [`DEFAULT_CONNECT_TIMEOUT_MS`].
    #[serde(default = "default_connect_timeout_ms")]
    pub connect_timeout_ms: u64,
    /// The receive timeout of the [`TcpConfig`], in milliseconds; optional, default
    /// [`DEFAULT_RECV_TIMEOUT_MS`].
    #[serde(default = "default_recv_timeout_ms")]
    pub recv_timeout_ms: u64,
}

/// Reads a whole file.
fn read(path: &Path) -> Result<Vec<u8>, HarnessError> {
    fs::read(path).map_err(|e| HarnessError::Io {
        path: path.to_owned(),
        message: e.to_string(),
    })
}

/// Writes a whole file.
fn write(path: &Path, bytes: &[u8]) -> Result<(), HarnessError> {
    fs::write(path, bytes).map_err(|e| HarnessError::Io {
        path: path.to_owned(),
        message: e.to_string(),
    })
}

impl PartyConfig {
    /// Reads and checks the configuration in the JSON file `path`.
    ///
    /// # Errors
    /// [`HarnessError::Io`] if the file cannot be read; [`HarnessError::Config`] if it does
    /// not parse (unknown fields included) or does not pass [`PartyConfig::validate`].
    pub fn load(path: &Path) -> Result<Self, HarnessError> {
        let config: Self = serde_json::from_slice(&read(path)?)
            .map_err(|e| HarnessError::Config(format!("{}: {e}", path.display())))?;
        config.validate()?;
        Ok(config)
    }

    /// Checks $n \ge 1$, $1 \le p \le n$ and $n$ addresses (the preset's $n$ is checked by
    /// [`Setup::new`]).
    ///
    /// # Errors
    /// [`HarnessError::Config`].
    pub fn validate(&self) -> Result<(), HarnessError> {
        if self.n == 0 || self.party == 0 || self.party > self.n {
            return Err(HarnessError::Config(format!(
                "party {} of n = {}: need 1 <= party <= n",
                self.party, self.n
            )));
        }
        if self.addresses.len() != self.n {
            return Err(HarnessError::Config(format!(
                "{} addresses for n = {} parties",
                self.addresses.len(),
                self.n
            )));
        }
        Ok(())
    }

    /// The configuration of this party's [`TcpTransport`].
    pub fn tcp_config(&self) -> TcpConfig {
        TcpConfig::new(self.n, self.party, self.addresses.clone(), self.session)
            .connect_timeout(Duration::from_millis(self.connect_timeout_ms))
            .recv_timeout(Duration::from_millis(self.recv_timeout_ms))
    }

    /// This party's network seed: the first 32 bytes of the stream (party seed,
    /// [`DOMAIN_NETWORK_SEED`], 0).
    pub fn network_seed(&self) -> [u8; 32] {
        let mut seed = [0u8; 32];
        ShakePrg::new(&self.party_seed, DOMAIN_NETWORK_SEED, 0).fill_bytes(&mut seed);
        seed
    }
}

/// The public parameters of a run: the context of $\mathsf{Setup}$ and the base PCS.
#[derive(Clone, Debug)]
pub struct Setup {
    /// The context: the preset's parameters and $\mathbf A, \mathbf B, \mathbf D$.
    pub ctx: Context<K>,
    /// Hachi as the base PCS ([`HachiBase::setup`]).
    pub pcs: HachiBase<K>,
}

impl Setup {
    /// $\mathsf{Setup}$ for the preset `preset` with `n` parties and the setup seed `seed`:
    /// [`crate::setup`] on the preset's derived parameters, and [`HachiBase::setup`] with the
    /// same seed.
    ///
    /// # Errors
    /// [`HarnessError::Config`] for an unknown preset or an $n$ other than the preset's;
    /// [`HarnessError::Protocol`] if the context or Hachi cannot be set up.
    pub fn new(preset: &str, n: usize, seed: &[u8; 32]) -> Result<Self, HarnessError> {
        let config = |e: hachi_co_params::Error| HarnessError::Config(format!("{preset}: {e}"));
        let preset_choices = hachi_co_params::preset(preset).map_err(config)?.choices;
        if preset_choices.n != n as u64 {
            return Err(HarnessError::Config(format!(
                "{preset} has n = {}, the configuration n = {n}",
                preset_choices.n
            )));
        }
        let (params, _report) = hachi_co_params::derive(&preset_choices).map_err(config)?;
        let ctx = crate::setup::<K>(params, seed)?;
        let pcs = HachiBase::setup(&ctx, seed)?;
        Ok(Self { ctx, pcs })
    }
}

/// The instance of a run: the polynomial $\mathbf f$ (its $2^\ell$ coefficients), the point
/// $\mathbf x$ and the claim $y = f(\mathbf x)$.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Instance {
    /// $\mathbf f \in \mathbb F\_{q^k}^{2^\ell}$.
    pub f: Vec<Ext<K>>,
    /// $\mathbf x \in \mathbb F\_{q^k}^\ell$.
    pub x: Vec<Ext<K>>,
    /// $y = f(\mathbf x)$.
    pub y: Ext<K>,
}

/// `count` uniform elements of $\mathbb F\_{q^K}$ from `prg`, coordinate by coordinate.
fn uniform_exts(prg: &mut ShakePrg, q: u64, count: usize) -> Result<Vec<Ext<K>>, HarnessError> {
    let len = count
        .checked_mul(K)
        .ok_or_else(|| HarnessError::Config(format!("{count} field elements overflow")))?;
    let residues = uniform_mod_q(prg, q, len).map_err(|e| Error::Core(e.into()))?;
    Ok(residues
        .as_chunks::<K>()
        .0
        .iter()
        .map(|c| Ext(*c))
        .collect())
}

/// $2^\ell$ for the context.
fn hypercube(ctx: &Context<K>) -> Result<usize, HarnessError> {
    let ell = ctx.layout.ell;
    u32::try_from(ell)
        .ok()
        .and_then(|e| 1usize.checked_shl(e))
        .ok_or_else(|| HarnessError::Config(format!("2^{ell} coefficients overflow")))
}

impl Instance {
    /// The instance of the seed `seed` (see the module documentation, "Derivations").
    ///
    /// # Errors
    /// [`HarnessError::Protocol`] if sampling or the evaluation fails (not expected for a
    /// valid context); [`HarnessError::Config`] if $2^\ell$ overflows.
    pub fn derive(ctx: &Context<K>, seed: &[u8; 32]) -> Result<Self, HarnessError> {
        let mut prg = ShakePrg::new(seed, DOMAIN_INSTANCE, 0);
        let f = uniform_exts(&mut prg, ctx.q(), hypercube(ctx)?)?;
        let x = uniform_exts(&mut prg, ctx.q(), ctx.layout.ell)?;
        let y = hachi_co_conv::evaluate(ctx, &f, &x).map_err(Error::from)?;
        Ok(Self { f, x, y })
    }

    /// Party `party`'s additive share of $\mathbf f$ for the instance seed `seed`: for
    /// $p \lt n$ the stream (`seed`, [`DOMAIN_SHARE`], $p$), for $p = n$ the difference between
    /// $\mathbf f$ and the other shares, so that the $n$ shares add up to $\mathbf f$. Each party
    /// computes only its own share (party $n$ computes the others to subtract them).
    ///
    /// # Errors
    /// [`HarnessError::Config`] unless $1 \le p \le n$; as [`Instance::derive`].
    pub fn share(
        &self,
        ctx: &Context<K>,
        seed: &[u8; 32],
        party: PartyId,
    ) -> Result<Vec<Ext<K>>, HarnessError> {
        let n = ctx.layout.n;
        if party == 0 || party > n {
            return Err(HarnessError::Config(format!(
                "party {party} of n = {n}: need 1 <= party <= n"
            )));
        }
        let stream = |p: PartyId| {
            uniform_exts(
                &mut ShakePrg::new(seed, DOMAIN_SHARE, p as u64),
                ctx.q(),
                self.f.len(),
            )
        };
        if party < n {
            return stream(party);
        }
        let field = ctx.field();
        let mut share = self.f.clone();
        for p in 1..n {
            for (s, r) in share.iter_mut().zip(stream(p)?) {
                *s = field.sub(s, &r);
            }
        }
        Ok(share)
    }
}

/// What one party publishes after [`prove`], with its timings.
#[derive(Clone, Debug)]
pub struct PartyOutput {
    /// The commitment $\mathbf u$, the same at every party.
    pub u: Vec<Poly>,
    /// The proof, the same at every party.
    pub proof: EvalProof<K>,
    /// The time of $\mathsf{Commit}$.
    pub commit_time: Duration,
    /// The time of $\mathsf{Eval}$.
    pub eval_time: Duration,
}

/// $\mathsf{Commit}$, then $\mathsf{Eval}$ of the claim of `instance`, for the party
/// `comm.id()` with its share `share` and the protocol randomness of `party_seed`
/// ([`PrgRandomness`]), over any network: a [`TcpTransport`] in [`run_party`], the in-memory
/// network in the tests.
///
/// # Errors
/// [`HarnessError::Protocol`] with the error of [`commit`] or [`eval`].
pub fn prove<C: Comm>(
    setup: &Setup,
    comm: &mut C,
    party_seed: &[u8; 32],
    share: &[Ext<K>],
    instance: &Instance,
) -> Result<PartyOutput, HarnessError> {
    let (ctx, pcs) = (&setup.ctx, &setup.pcs);
    let mut rng = PrgRandomness::new(*party_seed);
    let start = Instant::now();
    let (u, state) = commit(ctx, comm, &mut rng, share)?;
    let commit_time = start.elapsed();
    let start = Instant::now();
    let proof = eval(
        ctx,
        pcs,
        comm,
        &mut rng,
        &state,
        &u,
        &instance.x,
        &instance.y,
    )?;
    Ok(PartyOutput {
        u,
        proof,
        commit_time,
        eval_time: start.elapsed(),
    })
}

/// The public statement that every party writes next to its proof ([`STATEMENT_FILE`]): what
/// the verifier needs besides the proof.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Statement {
    /// The preset.
    pub preset: String,
    /// The number of parties $n$.
    pub n: usize,
    /// The setup seed, from which the verifier rebuilds the context and the base PCS.
    #[serde(with = "hex32")]
    pub setup_seed: [u8; 32],
    /// The commitment $\mathbf u$: $n\_B$ ring elements of $d$ residues.
    pub u: Vec<Vec<u64>>,
    /// The point $\mathbf x$: $\ell$ elements of $\mathbb F\_{q^k}$, each as its $k$
    /// coordinates.
    pub x: Vec<[u64; K]>,
    /// The claim $y$, as its $k$ coordinates.
    pub y: [u64; K],
}

impl Statement {
    /// The statement of a run of `config` with the commitment `u` and the claim of `instance`.
    pub fn new(config: &PartyConfig, u: &[Poly], instance: &Instance) -> Self {
        Self {
            preset: config.preset.clone(),
            n: config.n,
            setup_seed: config.setup_seed,
            u: u.iter().map(|p| p.0.clone()).collect(),
            x: instance.x.iter().map(|e| e.0).collect(),
            y: instance.y.0,
        }
    }

    /// $(\mathbf u, \mathbf x, y)$ as the verifier takes them; [`verify`] checks their shapes
    /// and residues.
    pub fn values(&self) -> (Vec<Poly>, Vec<Ext<K>>, Ext<K>) {
        (
            self.u.iter().cloned().map(Poly).collect(),
            self.x.iter().copied().map(Ext).collect(),
            Ext(self.y),
        )
    }

    /// Reads the statement in the JSON file `path`.
    ///
    /// # Errors
    /// [`HarnessError::Io`], or [`HarnessError::Config`] if it does not parse.
    pub fn load(path: &Path) -> Result<Self, HarnessError> {
        serde_json::from_slice(&read(path)?)
            .map_err(|e| HarnessError::Config(format!("{}: {e}", path.display())))
    }
}

/// What [`run_party`] did: the party's output, its timings and its accounting.
#[derive(Clone, Debug)]
pub struct PartyReport {
    /// The commitment, the proof and the times of Commit and Eval.
    pub output: PartyOutput,
    /// The length of the proof's encoding in bytes.
    pub proof_bytes: usize,
    /// The time of the local preparation: the context, the base PCS, the instance and the
    /// share.
    pub setup_time: Duration,
    /// The time of the connections, their handshakes and the set-up round of
    /// [`Endpoint::connect`].
    pub connect_time: Duration,
    /// The time from the start of the connections until the endpoint is closed, its last frames
    /// written out.
    pub network_time: Duration,
    /// The party's accounting ([`Endpoint::stats`]), the set-up round included.
    pub stats: Stats,
}

/// What a party computes before it connects: the public parameters, the instance and its
/// share, and the time they took.
struct Prepared {
    setup: Setup,
    instance: Instance,
    share: Vec<Ext<K>>,
    time: Duration,
}

impl Prepared {
    /// [`Setup::new`], [`Instance::derive`] and [`Instance::share`] for `config`, after
    /// [`PartyConfig::validate`].
    fn new(config: &PartyConfig) -> Result<Self, HarnessError> {
        config.validate()?;
        let start = Instant::now();
        let setup = Setup::new(&config.preset, config.n, &config.setup_seed)?;
        let instance = Instance::derive(&setup.ctx, &config.instance_seed)?;
        let share = instance.share(&setup.ctx, &config.instance_seed, config.party)?;
        Ok(Self {
            setup,
            instance,
            share,
            time: start.elapsed(),
        })
    }
}

/// Runs the party of `config`: [`run_party_with`] on a listener bound to the party's address,
/// which is bound only once the configuration has been checked and the local preparation is
/// done (so a configuration error is never reported as a network one).
///
/// # Errors
/// [`HarnessError::Bind`] if the address cannot be bound; as [`run_party_with`].
pub fn run_party(config: &PartyConfig) -> Result<PartyReport, HarnessError> {
    let prepared = Prepared::new(config)?;
    let address = config.addresses[config.party - 1];
    let listener = TcpListener::bind(address).map_err(|e| HarnessError::Bind {
        address,
        message: e.to_string(),
    })?;
    run_prepared(config, prepared, listener)
}

/// Runs the party of `config` with a listener already bound where the other parties reach it:
/// checks the configuration ([`PartyConfig::validate`]), prepares the context, the base PCS,
/// the instance and its share ([`Setup::new`], [`Instance::derive`], [`Instance::share`]),
/// connects ([`TcpTransport::connect_with`], [`Endpoint::connect`] with
/// [`ZeroSharing::PairwiseSeeds`] and the party's network seed), runs [`prove`], closes the
/// endpoint and writes [`PROOF_FILE`] and [`STATEMENT_FILE`] into the output directory.
///
/// # Errors
/// [`HarnessError::Config`] for an invalid configuration; [`HarnessError::Network`] if the
/// connections or the set-up round fail; [`HarnessError::Protocol`] if Setup, the instance,
/// Commit or Eval fail; [`HarnessError::Io`] if the output cannot be written.
pub fn run_party_with(
    config: &PartyConfig,
    listener: TcpListener,
) -> Result<PartyReport, HarnessError> {
    run_prepared(config, Prepared::new(config)?, listener)
}

/// The networked part of [`run_party_with`].
fn run_prepared(
    config: &PartyConfig,
    prepared: Prepared,
    listener: TcpListener,
) -> Result<PartyReport, HarnessError> {
    let Prepared {
        setup,
        instance,
        share,
        time: setup_time,
    } = prepared;
    let start = Instant::now();
    let transport = TcpTransport::connect_with(&config.tcp_config(), listener)?;
    let mut endpoint = Endpoint::connect(
        transport,
        setup.ctx.q(),
        ZeroSharing::PairwiseSeeds,
        config.network_seed(),
    )?;
    let connect_time = start.elapsed();
    let output = prove(&setup, &mut endpoint, &config.party_seed, &share, &instance)?;
    let stats = endpoint.stats().clone();
    drop(endpoint);
    let network_time = start.elapsed();
    let bytes = output.proof.to_bytes();
    let statement = Statement::new(config, &output.u, &instance);
    let json = serde_json::to_vec_pretty(&statement)
        .map_err(|e| HarnessError::Config(format!("the statement: {e}")))?;
    fs::create_dir_all(&config.output).map_err(|e| HarnessError::Io {
        path: config.output.clone(),
        message: e.to_string(),
    })?;
    write(&config.output.join(STATEMENT_FILE), &json)?;
    write(&config.output.join(PROOF_FILE), &bytes)?;
    Ok(PartyReport {
        output,
        proof_bytes: bytes.len(),
        setup_time,
        connect_time,
        network_time,
        stats,
    })
}

/// What [`verify_output`] checked.
#[derive(Clone, Debug)]
pub struct VerifyReport {
    /// The statement.
    pub statement: Statement,
    /// The length of the proof's encoding in bytes.
    pub proof_bytes: usize,
    /// The time of [`EvalProof::from_bytes`] and [`verify`].
    pub verify_time: Duration,
}

/// The verifier of an output directory: reads [`STATEMENT_FILE`] and [`PROOF_FILE`], rebuilds
/// the context and the base PCS from the statement's preset and setup seed ([`Setup::new`]),
/// decodes the proof ([`EvalProof::from_bytes`]) and runs [`verify`] on
/// $(\mathbf u, \mathbf x, y)$.
///
/// # Errors
/// [`HarnessError::Io`] or [`HarnessError::Config`] if the files cannot be read or the
/// statement does not parse; [`HarnessError::Decode`] if the proof does not decode;
/// [`HarnessError::Rejected`] if the verifier rejects.
pub fn verify_output(dir: &Path) -> Result<VerifyReport, HarnessError> {
    let statement = Statement::load(&dir.join(STATEMENT_FILE))?;
    let setup = Setup::new(&statement.preset, statement.n, &statement.setup_seed)?;
    let bytes = read(&dir.join(PROOF_FILE))?;
    let start = Instant::now();
    let proof = EvalProof::from_bytes(&setup.ctx, &bytes)?;
    let (u, x, y) = statement.values();
    verify(&setup.ctx, &setup.pcs, &u, &x, &y, &proof)?;
    Ok(VerifyReport {
        statement,
        proof_bytes: bytes.len(),
        verify_time: start.elapsed(),
    })
}
