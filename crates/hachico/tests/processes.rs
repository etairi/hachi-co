//! The parties of Hachi-Co as separate processes of `hachi-co-party` (`src/bin/hachi-co-party.rs`)
//! talking over TCP on 127.0.0.1, at the toy presets with n = 2 and n = 3:
//!
//! - every party exits successfully, reports the bytes it sent and received, and writes the same
//!   proof and the same statement;
//! - the statement is the instance of the seeds (`harness::Instance`), whose shares add up to f;
//! - the proof decodes and verifies, in this process and with `hachi-co-party --verify`;
//! - it is the proof of the same parties over the in-memory network with the same seeds
//!   (`harness::prove` over `run_in_memory`, whose network seeds differ: the proof does not
//!   depend on them), and so is the commitment;
//! - a proof with one bit flipped is rejected by `--verify`;
//! - the exit status tells usage and configuration errors (2), an address in use (3) and a
//!   failed verification (1) apart.
//!
//! Ports: the test binds port 0 for each party, reads the ports and releases them before the
//! processes start. If a party cannot bind its port all the same (exit status 3: another
//! process took it in between), the whole run is repeated once with new ports.
//!
//! Seeds: `seed("processes/<what>", [n, ...])` of `tests/common` for the session (with the
//! attempt), the setup, the instance and each party (with its index), and for the network
//! seed of the in-memory run.

mod common;

use common::seed;
use hachi_co::harness::{
    self, Instance, PROOF_FILE, PartyConfig, STATEMENT_FILE, Setup, Statement,
};
use hachi_co::{EvalProof, verify};
use hachi_co_mpc::{Comm, MemoryConfig, run_in_memory};
use std::fs::{self, File};
use std::net::{SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus};
use std::time::{Duration, Instant};

/// The program under test.
const EXE: &str = env!("CARGO_BIN_EXE_hachi-co-party");
/// The exit status of a party that could not bind its address.
const EXIT_BIND: i32 = 3;
/// The exit status of a usage or configuration error.
const EXIT_CONFIG: i32 = 2;
/// How long a run of the processes may take before they are killed.
const RUN_LIMIT: Duration = Duration::from_secs(120);

/// A fresh scratch directory for the test `name` under cargo's temporary directory for tests.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join("hachi-co-processes")
        .join(format!("{name}-{}", std::process::id()));
    if dir.exists() {
        fs::remove_dir_all(&dir).unwrap();
    }
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// `n` addresses on 127.0.0.1 whose ports were free a moment ago: all bound to port 0 at once
/// (so they differ), read, and released.
fn free_addresses(n: usize) -> Vec<SocketAddr> {
    let listeners: Vec<TcpListener> = (0..n)
        .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    listeners.iter().map(|l| l.local_addr().unwrap()).collect()
}

/// The configurations of the `n` parties at `toy-n{n}`, run number `attempt`, at `addresses`,
/// writing into `dir/party-p`.
fn configs(n: usize, attempt: u64, addresses: &[SocketAddr], dir: &Path) -> Vec<PartyConfig> {
    let tag = n as u64;
    (1..=n)
        .map(|p| PartyConfig {
            preset: format!("toy-n{n}"),
            n,
            party: p,
            addresses: addresses.to_vec(),
            session: seed("processes/session", &[tag, attempt]),
            setup_seed: seed("processes/setup", &[tag]),
            instance_seed: seed("processes/instance", &[tag]),
            party_seed: seed("processes/party", &[tag, p as u64]),
            output: dir.join(format!("party-{p}")),
            connect_timeout_ms: 20_000,
            recv_timeout_ms: 30_000,
        })
        .collect()
}

/// How one process ended.
struct Exit {
    status: ExitStatus,
    stdout: String,
    stderr: String,
}

impl Exit {
    fn describe(&self) -> String {
        format!(
            "{}\n--- stdout:\n{}--- stderr:\n{}",
            self.status, self.stdout, self.stderr
        )
    }
}

/// Starts `command` with its output in `dir/<name>.out` and `dir/<name>.err`, and returns the
/// wait for its end (which kills it after [`RUN_LIMIT`]).
fn spawn(mut command: Command, dir: &Path, name: &str) -> impl FnOnce() -> Exit + use<> {
    let out = dir.join(format!("{name}.out"));
    let err = dir.join(format!("{name}.err"));
    let mut child = command
        .stdout(File::create(&out).unwrap())
        .stderr(File::create(&err).unwrap())
        .spawn()
        .unwrap();
    let start = Instant::now();
    move || {
        let status = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status;
            }
            if start.elapsed() > RUN_LIMIT {
                child.kill().unwrap();
                break child.wait().unwrap();
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        Exit {
            status,
            stdout: fs::read_to_string(out).unwrap(),
            stderr: fs::read_to_string(err).unwrap(),
        }
    }
}

/// Writes each configuration to `dir/party-p.json`, starts one process per party (party n
/// first), waits for all of them and returns how they ended, in party order.
fn run_parties(configs: &[PartyConfig], dir: &Path) -> Vec<Exit> {
    let mut waits: Vec<_> = configs
        .iter()
        .rev()
        .map(|c| {
            let path = dir.join(format!("party-{}.json", c.party));
            fs::write(&path, serde_json::to_vec_pretty(c).unwrap()).unwrap();
            let mut command = Command::new(EXE);
            command.arg(&path);
            spawn(command, dir, &format!("party-{}", c.party))
        })
        .collect::<Vec<_>>();
    waits.reverse();
    waits.into_iter().map(|wait| wait()).collect()
}

/// `hachi-co-party` with `args`, to its end.
fn run(args: &[&std::ffi::OsStr], dir: &Path, name: &str) -> Exit {
    let mut command = Command::new(EXE);
    command.args(args);
    spawn(command, dir, name)()
}

/// The n parties of `toy-n{n}` as processes, with every check of the module documentation.
fn processes_agree_with_the_in_memory_run(n: usize) {
    let dir = scratch(&format!("n{n}"));
    let mut attempt = 0;
    let (configs, exits) = loop {
        let configs = configs(n, attempt, &free_addresses(n), &dir);
        let exits = run_parties(&configs, &dir);
        if attempt == 0 && exits.iter().any(|e| e.status.code() == Some(EXIT_BIND)) {
            eprintln!("n = {n}: a port was taken before its party bound it; another run");
            attempt += 1;
            continue;
        }
        break (configs, exits);
    };
    for (c, e) in configs.iter().zip(&exits) {
        assert!(
            e.status.success(),
            "n = {n}, party {}: {}",
            c.party,
            e.describe()
        );
        assert!(
            e.stdout.contains("payload bytes in") && e.stdout.contains("hachi-co/mpc/setup"),
            "n = {n}, party {}: no accounting in the report: {}",
            c.party,
            e.describe()
        );
    }
    eprintln!("n = {n}, party 1:\n{}", exits[0].stdout);

    // Every party wrote the same proof and the same statement.
    let proofs: Vec<Vec<u8>> = configs
        .iter()
        .map(|c| fs::read(c.output.join(PROOF_FILE)).unwrap())
        .collect();
    let statements: Vec<Statement> = configs
        .iter()
        .map(|c| Statement::load(&c.output.join(STATEMENT_FILE)).unwrap())
        .collect();
    assert!(proofs.iter().all(|p| *p == proofs[0]), "n = {n}: one proof");
    assert!(
        statements.iter().all(|s| *s == statements[0]),
        "n = {n}: one statement"
    );
    let (bytes, statement) = (&proofs[0], &statements[0]);

    // The statement is the instance of the seeds, and the shares add up to f.
    let c = &configs[0];
    let setup = Setup::new(&c.preset, n, &c.setup_seed).unwrap();
    let ctx = &setup.ctx;
    let instance = Instance::derive(ctx, &c.instance_seed).unwrap();
    assert_eq!(
        (&statement.preset, statement.n, statement.setup_seed),
        (&c.preset, n, c.setup_seed)
    );
    assert_eq!(
        statement.x,
        instance.x.iter().map(|e| e.0).collect::<Vec<_>>()
    );
    assert_eq!(statement.y, instance.y.0);
    let shares: Vec<_> = (1..=n)
        .map(|p| instance.share(ctx, &c.instance_seed, p).unwrap())
        .collect();
    let field = ctx.field();
    for (t, f) in instance.f.iter().enumerate() {
        let sum = shares
            .iter()
            .fold(field.zero(), |acc, s| field.add(&acc, &s[t]));
        assert_eq!(sum, *f, "n = {n}: the shares add up to f (coefficient {t})");
    }

    // The proof decodes and verifies, here and with `--verify`.
    let proof = EvalProof::from_bytes(ctx, bytes).unwrap();
    let (u, x, y) = statement.values();
    verify(ctx, &setup.pcs, &u, &x, &y, &proof).unwrap();
    let verified = run(&["--verify".as_ref(), c.output.as_os_str()], &dir, "verify");
    assert!(
        verified.status.success(),
        "n = {n}: {}",
        verified.describe()
    );
    assert!(
        verified.stdout.starts_with("accepted"),
        "{}",
        verified.describe()
    );

    // The in-memory run with the same seeds has the same commitment and proof.
    let memory = MemoryConfig::new(n, ctx.q(), seed("processes/network", &[n as u64]));
    let runs = run_in_memory(&memory, |ep| {
        let p = ep.id();
        let (party_seed, share) = (&configs[p - 1].party_seed, &shares[p - 1]);
        harness::prove(&setup, ep, party_seed, share, &instance).unwrap()
    })
    .unwrap();
    for run in runs {
        let out = run.unwrap().output;
        assert_eq!(out.u, u, "n = {n}: the commitment of the in-memory run");
        assert!(
            out.proof.to_bytes() == *bytes,
            "n = {n}: the proof of the in-memory run differs"
        );
    }

    // One flipped bit: rejected.
    let tampered = dir.join("tampered");
    fs::create_dir_all(&tampered).unwrap();
    fs::copy(c.output.join(STATEMENT_FILE), tampered.join(STATEMENT_FILE)).unwrap();
    let mut flipped = bytes.clone();
    let last = flipped.len() - 1;
    flipped[last] ^= 1;
    fs::write(tampered.join(PROOF_FILE), &flipped).unwrap();
    let rejected = run(
        &["--verify".as_ref(), tampered.as_os_str()],
        &dir,
        "tampered",
    );
    assert_eq!(rejected.status.code(), Some(1), "{}", rejected.describe());

    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn two_processes_prove_what_the_in_memory_network_proves() {
    processes_agree_with_the_in_memory_run(2);
}

#[test]
fn three_processes_prove_what_the_in_memory_network_proves() {
    processes_agree_with_the_in_memory_run(3);
}

#[test]
fn the_exit_status_tells_the_errors_apart() {
    let dir = scratch("errors");
    let code = |exit: Exit| exit.status.code();
    // Usage errors.
    assert_eq!(code(run(&[], &dir, "none")), Some(EXIT_CONFIG));
    assert_eq!(
        code(run(&["--verify".as_ref()], &dir, "verify-none")),
        Some(EXIT_CONFIG)
    );
    assert_eq!(code(run(&["--help".as_ref()], &dir, "help")), Some(0));
    // Configurations that do not parse or fit: an unknown preset, n other than the preset's (with
    // n addresses), a short seed, an unknown field. A missing file is a failure to read (1).
    let addresses = free_addresses(2);
    let good = configs(2, 0, &addresses, &dir).remove(0);
    let mut json = serde_json::to_value(&good).unwrap();
    let mut cases = Vec::new();
    for (key, value) in [
        ("preset", serde_json::json!("toy-n9")),
        ("n", serde_json::json!(3)),
        ("setup_seed", serde_json::json!("0123")),
        ("colour", serde_json::json!("blue")),
    ] {
        let mut bad = json.clone();
        bad[key] = value;
        if key == "n" {
            bad["addresses"] = serde_json::json!(["127.0.0.1:1", "127.0.0.1:2", "127.0.0.1:3"]);
        }
        cases.push((key, bad));
    }
    for (key, bad) in cases {
        let path = dir.join(format!("bad-{key}.json"));
        fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
        let exit = run(&[path.as_os_str()], &dir, &format!("bad-{key}"));
        assert_eq!(
            exit.status.code(),
            Some(EXIT_CONFIG),
            "{key}: {}",
            exit.describe()
        );
    }
    let missing = dir.join("missing.json");
    assert_eq!(code(run(&[missing.as_os_str()], &dir, "missing")), Some(1));
    // An address in use: party 1's port is held here.
    let held = TcpListener::bind("127.0.0.1:0").unwrap();
    json["addresses"] = serde_json::json!([held.local_addr().unwrap(), addresses[1]]);
    let path = dir.join("held.json");
    fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
    let exit = run(&[path.as_os_str()], &dir, "held");
    assert_eq!(exit.status.code(), Some(EXIT_BIND), "{}", exit.describe());
    // Nothing to verify.
    let empty = dir.join("empty");
    fs::create_dir_all(&empty).unwrap();
    let exit = run(&["--verify".as_ref(), empty.as_os_str()], &dir, "empty");
    assert_eq!(exit.status.code(), Some(1), "{}", exit.describe());
    drop(held);
    fs::remove_dir_all(&dir).unwrap();
}
