//! `hachi-co-party`: one party of Hachi-Co as its own process, over TCP, or the verifier of what
//! a party wrote. A command line around `hachi_co::harness`, a test and benchmark harness, not a
//! deployment (see there for the configuration file, the derivations and the output files).
//!
//! ```text
//! hachi-co-party <config.json>          run the party of the configuration
//! hachi-co-party --verify <directory>   verify the proof and the statement in an output directory
//! ```
//!
//! A party prints its timings and its accounting: the payload bytes and the frames it sent and
//! received, in total and per label (`Endpoint::stats`), and the bytes of its frames on the wire
//! (payloads and frame headers, without the handshakes). The exit status is 0 on success (the
//! proof was written, or it was accepted), 1 if the run or the verification failed, 2 for a
//! usage error or a configuration or statement that does not parse or fit, and 3 if the
//! party's address could not be bound (for example because another process uses it).
#![forbid(unsafe_code)]
#![deny(missing_docs)]

use hachi_co::harness::{self, HarnessError, PROOF_FILE, PartyConfig, PartyReport, STATEMENT_FILE};
use std::path::Path;
use std::process::ExitCode;

const USAGE: &str = "usage: hachi-co-party <config.json>
       hachi-co-party --verify <output directory>";

/// The bytes of a frame on the wire besides its label and its payload: the label length (2),
/// the round (8) and the payload length (8) (`hachi_co_mpc::TcpTransport`, "Frames").
const FRAME_HEADER_BYTES: u64 = 18;

/// The exit status of an error.
fn status(e: &HarnessError) -> ExitCode {
    match e {
        HarnessError::Config(_) => ExitCode::from(2),
        HarnessError::Bind { .. } => ExitCode::from(3),
        _ => ExitCode::FAILURE,
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.as_slice() {
        [flag] if flag == "-h" || flag == "--help" => {
            println!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        [flag, dir] if flag == "--verify" => verify(Path::new(dir)),
        [config] if !config.starts_with('-') => run(Path::new(config)),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("hachi-co-party: {e}");
            status(&e)
        }
    }
}

fn run(path: &Path) -> Result<(), HarnessError> {
    let config = PartyConfig::load(path)?;
    let report = harness::run_party(&config)?;
    print_report(&config, &report);
    Ok(())
}

fn print_report(config: &PartyConfig, r: &PartyReport) {
    let o = &r.output;
    println!(
        "party {} of {}, preset {}",
        config.party, config.n, config.preset
    );
    println!(
        "setup      {:>10.2?}  context, base PCS, instance and share",
        r.setup_time
    );
    println!(
        "connect    {:>10.2?}  connections, handshakes and the set-up round",
        r.connect_time
    );
    println!("commit     {:>10.2?}", o.commit_time);
    println!("eval       {:>10.2?}", o.eval_time);
    println!(
        "network    {:>10.2?}  from the first connection until the last frame is out",
        r.network_time
    );
    println!(
        "proof      {} bytes, {}",
        r.proof_bytes,
        config.output.join(PROOF_FILE).display()
    );
    println!(
        "statement  {}",
        config.output.join(STATEMENT_FILE).display()
    );
    let total = r.stats.total();
    let (mut wire_sent, mut wire_received) = (0u64, 0u64);
    for (label, s) in r.stats.iter() {
        let header = FRAME_HEADER_BYTES + label.len() as u64;
        wire_sent += s.bytes_sent + s.frames_sent * header;
        wire_received += s.bytes_received + s.frames_received * header;
    }
    println!(
        "sent       {} payload bytes in {} frames, {wire_sent} bytes on the wire",
        total.bytes_sent, total.frames_sent
    );
    println!(
        "received   {} payload bytes in {} frames, {wire_received} bytes on the wire",
        total.bytes_received, total.frames_received
    );
    println!(
        "{:<44} {:>5} {:>6} {:>12} {:>12}",
        "label", "calls", "rounds", "bytes sent", "bytes recv"
    );
    for (label, s) in r.stats.iter() {
        println!(
            "{label:<44} {:>5} {:>6} {:>12} {:>12}",
            s.calls, s.rounds, s.bytes_sent, s.bytes_received
        );
    }
}

fn verify(dir: &Path) -> Result<(), HarnessError> {
    let r = harness::verify_output(dir)?;
    println!(
        "accepted: {} ({} bytes) for the statement {} (preset {}, n = {}); decoded and verified in {:.2?}",
        dir.join(PROOF_FILE).display(),
        r.proof_bytes,
        dir.join(STATEMENT_FILE).display(),
        r.statement.preset,
        r.statement.n,
        r.verify_time
    );
    Ok(())
}
