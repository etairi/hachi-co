//! The TCP transport on 127.0.0.1, with every port chosen by the system before any party starts
//! (each party gets a listener bound to port 0), so the tests never race for ports:
//!
//! - n = 1, 2, 3 parties, one thread each and started in reverse order, connected through
//!   `Endpoint`, make the collective calls of `tests/endpoint.rs` (gather, abort agreement,
//!   pairwise exchange, all-gather, F_AH of a linear map) in both zero-sharing modes, with the
//!   outputs and the accounting of the in-memory network and the ideal outputs;
//! - the bytes on the wire are those of the documented format, in both directions (checked
//!   against an encoder written here from the documentation);
//! - the handshake rejects another session, another n, another wire format, and an unexpected
//!   or duplicate party index, at the accepting and at the connecting end;
//! - a peer that leaves gives `PeerLeft` after the frames it sent (an 8 MiB one included), well
//!   before the timeout; a silent peer gives `Timeout`; frames that break the wire format give
//!   `Malformed` and close the connection; a truncated frame gives `PeerLeft`;
//! - the set-up times out when a party never connects, never answers or cannot be reached, a
//!   party that starts late is waited for, and invalid configurations are errors.

use hachi_co_mpc::{
    Comm, Endpoint, Error, Frame, LABEL_SETUP, MemoryConfig, PartyId, Result, Stats, TcpConfig,
    TcpTransport, Transport, ZeroSharing, run_in_memory,
};
use hachi_co_sampling::ShakePrg;
use hachi_co_sampling::uniform::uniform_mod_q;
use rand::Rng;
use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

/// The toy-n1 modulus (`hachi_co_params::presets::toy_presets`).
const Q: u64 = 8388637;
/// The largest modulus the layer accepts.
const Q_MAX: u64 = (1 << 63) - 1;
const MODES: [ZeroSharing; 2] = [ZeroSharing::PairwiseSeeds, ZeroSharing::Fresh];
const SESSION: [u8; 32] = [0x5a; 32];
/// An address that party 1 never connects to (it accepts every other party).
const NEVER_DIALED: ([u8; 4], u16) = ([127, 0, 0, 1], 9);

fn seed(tag: &str, index: u64) -> [u8; 32] {
    let mut s = [0u8; 32];
    ShakePrg::new(&[7; 32], tag.as_bytes(), index).fill_bytes(&mut s);
    s
}

fn residues(tag: &str, index: u64, q: u64, len: usize) -> Vec<u64> {
    uniform_mod_q(
        &mut ShakePrg::new(&seed(tag, index), b"residues", 0),
        q,
        len,
    )
    .unwrap()
}

/// `n` listeners on 127.0.0.1, on ports chosen by the system, and their addresses.
fn listeners(n: usize) -> (Vec<TcpListener>, Vec<SocketAddr>) {
    let listeners: Vec<TcpListener> = (0..n)
        .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    let addresses = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
    (listeners, addresses)
}

/// The in-memory configuration whose parameters a TCP run copies (n, q, zero sharing, network
/// seeds, receive timeout).
fn config(n: usize, q: u64, mode: ZeroSharing, tag: &str) -> MemoryConfig {
    MemoryConfig::new(n, q, seed(tag, n as u64))
        .zero_sharing(mode)
        .timeout(Duration::from_secs(20))
}

/// Runs `party` for p = 1, ..., n, each on its own thread with an `Endpoint` over a
/// `TcpTransport` on its pre-bound listener, with the parameters and the network seeds of
/// `config`, and returns each party's output and accounting (or the error of its connection),
/// in party order. Party p starts 30 (n - p) ms after the first, so higher parties connect before
/// the lower ones accept.
fn run_tcp<T: Send>(
    config: &MemoryConfig,
    party: impl Fn(&mut Endpoint<TcpTransport>) -> T + Sync,
) -> Vec<Result<(T, Stats)>> {
    let n = config.n;
    let (listeners, addresses) = listeners(n);
    let party = &party;
    std::thread::scope(|s| {
        let handles: Vec<_> = listeners
            .into_iter()
            .zip(1..=n)
            .map(|(listener, id)| {
                let tcp = TcpConfig::new(n, id, addresses.clone(), SESSION)
                    .connect_timeout(Duration::from_secs(20))
                    .recv_timeout(config.timeout);
                s.spawn(move || -> Result<(T, Stats)> {
                    std::thread::sleep(Duration::from_millis(30 * (n - id) as u64));
                    let transport = TcpTransport::connect_with(&tcp, listener)?;
                    let mut ep = Endpoint::connect(
                        transport,
                        config.q,
                        config.zero_sharing,
                        config.party_seed(id),
                    )?;
                    let output = party(&mut ep);
                    Ok((output, ep.stats().clone()))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("no party panics"))
            .collect()
    })
}

/// What every party gets from one session of each call kind (as in `tests/endpoint.rs`).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Session {
    gathered: Option<Vec<Vec<i64>>>,
    any_false: bool,
    any_one: bool,
    exchanged: Vec<Vec<u64>>,
    all: Vec<Vec<u64>>,
    sum: Vec<u64>,
}

fn gather_input(p: PartyId) -> Vec<i64> {
    vec![-(p as i64), i64::MIN, i64::MAX, p as i64 * 1000]
}

fn exchange_input(p: PartyId, j: PartyId) -> Vec<u64> {
    // Different lengths per pair, values not reduced: exchange delivers them unchanged.
    (0..(p + j) % 4)
        .map(|k| (p * 100 + j * 10 + k) as u64 | 1 << 63)
        .collect()
}

fn session(ep: &mut impl Comm, to: PartyId) -> Result<Session> {
    let (n, p) = (ep.n(), ep.id());
    let gathered = ep.gather("gather", to, &gather_input(p))?;
    let any_false = ep.any("any/false", false)?;
    let any_one = ep.any("any/one", p == n)?;
    let outgoing = (1..=n).map(|j| exchange_input(p, j)).collect();
    let exchanged = ep.exchange("exchange", outgoing)?;
    let all = ep.all_gather("all", &[p as u64, u64::MAX - p as u64])?;
    let sum = ep.fah("fah", &[p as u64; 3])?;
    Ok(Session {
        gathered,
        any_false,
        any_one,
        exchanged,
        all,
        sum,
    })
}

fn ideal_session(n: usize, p: PartyId, to: PartyId) -> Session {
    let total = (n * (n + 1) / 2) as u64 % Q;
    Session {
        gathered: (p == to).then(|| (1..=n).map(gather_input).collect()),
        any_false: false,
        any_one: true,
        exchanged: (1..=n).map(|j| exchange_input(j, p)).collect(),
        all: (1..=n)
            .map(|j| vec![j as u64, u64::MAX - j as u64])
            .collect(),
        sum: vec![total; 3],
    }
}

#[test]
fn every_call_over_tcp_matches_the_in_memory_network() {
    for n in [1, 2, 3] {
        for mode in MODES {
            for to in [1, n] {
                let cfg = config(n, Q, mode, "calls");
                let tcp = run_tcp(&cfg, |ep| session(ep, to).unwrap());
                let memory = run_in_memory(&cfg, |ep| session(ep, to).unwrap()).unwrap();
                for (i, (tcp, memory)) in tcp.into_iter().zip(memory).enumerate() {
                    let p = i + 1;
                    let what = format!("n {n}, {mode:?}, to {to}, party {p}");
                    let (output, stats) = tcp.unwrap_or_else(|e| panic!("{what}: {e}"));
                    let memory = memory.unwrap();
                    assert_eq!(output, ideal_session(n, p, to), "{what}");
                    assert_eq!(output, memory.output, "{what}");
                    // The accounting counts payloads and frames, not the transport's headers.
                    assert_eq!(stats, memory.stats, "{what}");
                    assert_eq!(stats.label(LABEL_SETUP).calls, 1, "{what}");
                }
            }
        }
    }
}

#[test]
fn fah_over_tcp_outputs_the_map_of_the_reconstructed_input() {
    // F_AH(L) on additive shares x_p of x gives L(x) at every party, as in memory.
    let n = 3;
    let (m_in, m_out) = (7, 5);
    for q in [Q, Q_MAX] {
        for mode in MODES {
            let l = residues("L", n as u64, q, m_in * m_out);
            let shares: Vec<Vec<u64>> = (1..=n as u64)
                .map(|p| residues("x", 100 * n as u64 + p, q, m_in))
                .collect();
            let x: Vec<u64> = (0..m_in)
                .map(|t| {
                    let s = shares.iter().map(|s| u128::from(s[t])).sum::<u128>();
                    (s % u128::from(q)) as u64
                })
                .collect();
            let apply = |x: &[u64]| -> Vec<u64> {
                l.chunks(m_in)
                    .map(|row| {
                        let s = row
                            .iter()
                            .zip(x)
                            .map(|(a, b)| u128::from(*a) * u128::from(*b) % u128::from(q))
                            .sum::<u128>();
                        (s % u128::from(q)) as u64
                    })
                    .collect()
            };
            let want = apply(&x);
            let local = |p: PartyId| apply(&shares[p - 1]);
            let cfg = config(n, q, mode, "fah");
            let tcp = run_tcp(&cfg, |ep| {
                let p = ep.id();
                ep.fah("L", &local(p))
            });
            let memory = run_in_memory(&cfg, |ep| {
                let p = ep.id();
                ep.fah("L", &local(p))
            })
            .unwrap();
            for (i, (tcp, memory)) in tcp.into_iter().zip(memory).enumerate() {
                let (output, stats) = tcp.unwrap();
                let memory = memory.unwrap();
                assert_eq!(output, Ok(want.clone()), "q {q}, {mode:?}, party {}", i + 1);
                assert_eq!(memory.output, Ok(want.clone()));
                assert_eq!(stats, memory.stats);
            }
        }
    }
}

/// The frame encoding of the module documentation of `TcpTransport`, written here
/// independently: label length (u16 LE), label, round (u64 LE), payload length in words (u64
/// LE), payload words (u64 LE).
fn wire_frame(label: &[u8], round: u64, payload: &[u64]) -> Vec<u8> {
    let mut out = (label.len() as u16).to_le_bytes().to_vec();
    out.extend(label);
    out.extend(round.to_le_bytes());
    out.extend((payload.len() as u64).to_le_bytes());
    for w in payload {
        out.extend(w.to_le_bytes());
    }
    out
}

/// The handshake of the module documentation: magic, version (u64 LE), session, n (u64 LE),
/// the sender's index (u64 LE).
fn wire_hello(magic: [u8; 8], version: u64, session: [u8; 32], n: u64, id: u64) -> Vec<u8> {
    let mut out = magic.to_vec();
    out.extend(version.to_le_bytes());
    out.extend(session);
    out.extend(n.to_le_bytes());
    out.extend(id.to_le_bytes());
    out
}

fn good_hello(n: u64, id: u64) -> Vec<u8> {
    wire_hello(*b"HCO-TCP\0", 1, SESSION, n, id)
}

/// Party 1 of `n`, accepting on a pre-bound listener, while `peers` raw clients connect one
/// after the other, each sending its handshake and reading party 1's (which must be the
/// documented one). Returns party 1's result and the raw connections.
fn party_one_with_raw_peers(
    n: usize,
    timeout: Duration,
    peers: &[Vec<u8>],
) -> (Result<TcpTransport>, Vec<TcpStream>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let mut addresses = vec![address];
    addresses.resize(n, SocketAddr::from(NEVER_DIALED));
    let tcp = TcpConfig::new(n, 1, addresses, SESSION)
        .connect_timeout(timeout)
        .recv_timeout(Duration::from_secs(5));
    std::thread::scope(|s| {
        let party = s.spawn(|| TcpTransport::connect_with(&tcp, listener));
        let streams = peers
            .iter()
            .map(|hello| {
                let mut stream = TcpStream::connect(address).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                stream.write_all(hello).unwrap();
                let mut theirs = [0u8; 64];
                stream.read_exact(&mut theirs).unwrap();
                assert_eq!(
                    theirs.to_vec(),
                    good_hello(n as u64, 1),
                    "party 1's handshake"
                );
                stream
            })
            .collect();
        (party.join().unwrap(), streams)
    })
}

/// The error is a handshake failure.
fn is_handshake<T>(result: &Result<T>) -> bool {
    matches!(result, Err(Error::Handshake { .. }))
}

#[test]
fn the_handshake_rejects_another_session_or_another_n() {
    // Party 2 with another session id, then with n = 3: both ends refuse the connection.
    let mut other_session = SESSION;
    other_session[0] ^= 1;
    for (session, n2) in [(other_session, 2), (SESSION, 3)] {
        let (listeners, mut addresses) = listeners(2);
        let first = TcpConfig::new(2, 1, addresses.clone(), SESSION)
            .connect_timeout(Duration::from_secs(5));
        addresses.resize(n2, SocketAddr::from(NEVER_DIALED));
        let second =
            TcpConfig::new(n2, 2, addresses, session).connect_timeout(Duration::from_secs(5));
        let mut listeners = listeners.into_iter();
        let (l1, l2) = (listeners.next().unwrap(), listeners.next().unwrap());
        let (r1, r2) = std::thread::scope(|s| {
            let h1 = s.spawn(|| TcpTransport::connect_with(&first, l1));
            let h2 = s.spawn(|| TcpTransport::connect_with(&second, l2));
            (h1.join().unwrap(), h2.join().unwrap())
        });
        assert!(is_handshake(&r1), "party 1, n2 = {n2}: {r1:?}");
        assert!(is_handshake(&r2), "party 2, n2 = {n2}: {r2:?}");
    }
}

#[test]
fn the_handshake_rejects_unexpected_and_duplicate_indices_and_other_formats() {
    let timeout = Duration::from_secs(5);
    // Party 2 twice at party 1 of 3: the first is accepted, the second is a duplicate.
    let (result, _streams) =
        party_one_with_raw_peers(3, timeout, &[good_hello(3, 2), good_hello(3, 2)]);
    match result {
        Err(Error::Handshake { message, .. }) => {
            assert!(message.contains("connected already"), "{message}")
        }
        other => panic!("a duplicate party 2: {other:?}"),
    }
    // Party 1's own index, 0, and an index above n; another magic, another version.
    let mut magic = *b"HCO-TCP\0";
    magic[7] = b'!';
    for hello in [
        good_hello(3, 1),
        good_hello(3, 0),
        good_hello(3, 4),
        good_hello(3, u64::MAX),
        wire_hello(magic, 1, SESSION, 3, 2),
        wire_hello(*b"HCO-TCP\0", 2, SESSION, 3, 2),
    ] {
        let (result, _streams) = party_one_with_raw_peers(3, timeout, std::slice::from_ref(&hello));
        assert!(is_handshake(&result), "{hello:?}: {result:?}");
    }
    // The connecting end: party 2 of 2 reaches a raw "party 1" that answers with index 3, with
    // another session, or with another n.
    let mut other_session = SESSION;
    other_session[31] ^= 0x80;
    for answer in [
        good_hello(2, 3),
        wire_hello(*b"HCO-TCP\0", 1, other_session, 2, 1),
        good_hello(3, 1),
    ] {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addresses = vec![
            listener.local_addr().unwrap(),
            SocketAddr::from(NEVER_DIALED),
        ];
        let second = TcpConfig::new(2, 2, addresses, SESSION).connect_timeout(timeout);
        let result = std::thread::scope(|s| {
            let party = s.spawn(|| {
                TcpTransport::connect_with(&second, TcpListener::bind("127.0.0.1:0").unwrap())
            });
            let (mut stream, _) = listener.accept().unwrap();
            let mut theirs = [0u8; 64];
            stream.read_exact(&mut theirs).unwrap();
            assert_eq!(theirs.to_vec(), good_hello(2, 2), "party 2's handshake");
            stream.write_all(&answer).unwrap();
            party.join().unwrap()
        });
        assert!(is_handshake(&result), "{answer:?}: {result:?}");
    }
}

/// Party 1 of 2 connected to a raw client that plays party 2 (handshake done).
fn party_one_and_raw_peer() -> (TcpTransport, TcpStream) {
    let (result, mut streams) =
        party_one_with_raw_peers(2, Duration::from_secs(5), &[good_hello(2, 2)]);
    (result.unwrap(), streams.pop().unwrap())
}

#[test]
fn frames_on_the_wire_have_the_documented_format() {
    let (mut t1, mut raw) = party_one_and_raw_peer();
    // From the transport to the wire.
    let payload = vec![0, 1, u64::MAX, 0x0123_4567_89ab_cdef];
    t1.send("x", 2, Frame::new("hachi-co/test/out", 7, payload.clone()))
        .unwrap();
    let want = wire_frame(b"hachi-co/test/out", 7, &payload);
    let mut got = vec![0u8; want.len()];
    raw.read_exact(&mut got).unwrap();
    assert_eq!(got, want);
    // From the wire to the transport: labels arrive owned.
    raw.write_all(&wire_frame("é/in".as_bytes(), u64::MAX, &[42]))
        .unwrap();
    raw.write_all(&wire_frame(b"", 0, &[])).unwrap();
    let first = t1.recv("x", 2).unwrap();
    assert_eq!(first, Frame::new("é/in", u64::MAX, vec![42]));
    assert!(matches!(first.label, std::borrow::Cow::Owned(_)));
    assert_eq!(t1.recv("x", 2).unwrap(), Frame::new("", 0, vec![]));
}

#[test]
fn frames_that_break_the_wire_format_give_malformed_and_close_the_connection() {
    let mut long_label = wire_frame(&[b'a'; 255], 0, &[]);
    long_label[0] = 0;
    long_label[1] = 1; // 256 bytes announced, a label of 255 sent
    let mut huge = (1u16).to_le_bytes().to_vec();
    huge.extend(b"h");
    huge.extend(0u64.to_le_bytes());
    huge.extend((TcpTransport::MAX_PAYLOAD_WORDS + 1).to_le_bytes());
    for (bad, what) in [
        (long_label, "a label over 255 bytes"),
        (
            wire_frame(&[0xff, 0xfe], 0, &[1]),
            "a label that is not UTF-8",
        ),
        (huge, "a payload over the bound"),
    ] {
        let (mut t1, mut raw) = party_one_and_raw_peer();
        // A good frame first: it is delivered.
        raw.write_all(&wire_frame(b"ok", 1, &[5])).unwrap();
        raw.write_all(&bad).unwrap();
        assert_eq!(
            t1.recv("x", 2).unwrap(),
            Frame::new("ok", 1, vec![5]),
            "{what}"
        );
        let result = t1.recv("x", 2);
        assert!(
            matches!(result, Err(Error::Malformed { party: 2, .. })),
            "{what}: {result:?}"
        );
        // The receiver closed the connection: the peer sees its end, and the party treats the
        // peer as gone.
        raw.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut buf = [0u8; 1];
        match raw.read(&mut buf) {
            Ok(0) => {}
            Err(e) if e.kind() == ErrorKind::ConnectionReset => {}
            other => panic!("{what}: the connection is still open: {other:?}"),
        }
        assert!(matches!(
            t1.recv("x", 2),
            Err(Error::PeerLeft { party: 2, .. })
        ));
        assert!(matches!(
            t1.send("x", 2, Frame::new("x", 0, vec![])),
            Err(Error::PeerLeft { party: 2, .. })
        ));
    }
    // A frame cut short by the end of the connection: the peer left.
    let (mut t1, mut raw) = party_one_and_raw_peer();
    let frame = wire_frame(b"cut", 3, &[1, 2, 3]);
    raw.write_all(&frame[..frame.len() - 5]).unwrap();
    drop(raw);
    assert!(matches!(
        t1.recv("x", 2),
        Err(Error::PeerLeft { party: 2, .. })
    ));
}

/// Two connected transports, parties 1 and 2, with the receive timeout `timeout`.
fn pair(timeout: Duration) -> (TcpTransport, TcpTransport) {
    let (listeners, addresses) = listeners(2);
    let mut transports = std::thread::scope(|s| {
        let handles: Vec<_> = listeners
            .into_iter()
            .zip(1..)
            .map(|(listener, id)| {
                let tcp = TcpConfig::new(2, id, addresses.clone(), SESSION)
                    .connect_timeout(Duration::from_secs(10))
                    .recv_timeout(timeout);
                s.spawn(move || TcpTransport::connect_with(&tcp, listener).unwrap())
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    let second = transports.pop().unwrap();
    (transports.pop().unwrap(), second)
}

#[test]
fn a_peer_that_leaves_gives_peer_left_after_its_frames() {
    // Party 2 sends a small and an 8 MiB frame and leaves at once: dropping its transport
    // writes both out before it ends the connection.
    let (mut t1, mut t2) = pair(Duration::from_secs(20));
    let big: Vec<u64> = (0..(1u64 << 20) + 3)
        .map(|w| w.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .collect();
    t2.send("leave", 1, Frame::new("small", 1, vec![1, 2]))
        .unwrap();
    t2.send("leave", 1, Frame::new("big", 2, big.clone()))
        .unwrap();
    drop(t2);
    let start = Instant::now();
    assert_eq!(
        t1.recv("leave", 2).unwrap(),
        Frame::new("small", 1, vec![1, 2])
    );
    assert_eq!(t1.recv("leave", 2).unwrap(), Frame::new("big", 2, big));
    assert_eq!(
        t1.recv("leave", 2),
        Err(Error::PeerLeft {
            label: "leave",
            party: 2
        })
    );
    assert_eq!(
        t1.send("leave", 2, Frame::new("x", 3, vec![])),
        Err(Error::PeerLeft {
            label: "leave",
            party: 2
        })
    );
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "{:?}",
        start.elapsed()
    );
    // Through Endpoint: party 3 leaves right after the set-up; party 1 gathers and gets
    // PeerLeft from party 3 after party 2's vector, well before the timeout; party 2 only sends.
    let cfg = config(3, Q, ZeroSharing::PairwiseSeeds, "leave");
    let start = Instant::now();
    let runs = run_tcp(&cfg, |ep| match ep.id() {
        3 => Ok(None),
        _ => ep.gather("gather", 1, &[ep.id() as i64]),
    });
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "{:?}",
        start.elapsed()
    );
    let outputs: Vec<_> = runs.into_iter().map(|r| r.unwrap().0).collect();
    assert_eq!(
        outputs[0],
        Err(Error::PeerLeft {
            label: "gather",
            party: 3
        })
    );
    assert_eq!(outputs[1], Ok(None));
}

#[test]
fn a_silent_peer_gives_a_timeout() {
    let cfg =
        config(2, Q, ZeroSharing::PairwiseSeeds, "silent").timeout(Duration::from_millis(300));
    let runs = run_tcp(&cfg, |ep| {
        if ep.id() == 2 {
            std::thread::sleep(Duration::from_millis(900));
            Ok(false)
        } else {
            ep.any("x", false)
        }
    });
    let out = runs.into_iter().next().unwrap().unwrap().0;
    assert_eq!(
        out,
        Err(Error::Timeout {
            label: "x",
            party: 2,
            millis: 300
        })
    );
}

#[test]
fn the_set_up_times_out() {
    let timeout = Duration::from_millis(300);
    let millis = 300;
    // Party 1 of 2, and party 2 never connects.
    let (result, _) = party_one_with_raw_peers(2, timeout, &[]);
    assert_eq!(
        result.err(),
        Some(Error::Timeout {
            label: TcpTransport::LABEL_CONNECT,
            party: 2,
            millis
        })
    );
    // Party 2 of 2 reaches a listener that never answers: the handshake times out.
    let silent = TcpListener::bind("127.0.0.1:0").unwrap();
    let addresses = vec![silent.local_addr().unwrap(), SocketAddr::from(NEVER_DIALED)];
    let second = TcpConfig::new(2, 2, addresses, SESSION).connect_timeout(timeout);
    let start = Instant::now();
    let result = TcpTransport::connect_with(&second, TcpListener::bind("127.0.0.1:0").unwrap());
    assert!(is_handshake(&result), "{result:?}");
    assert!(start.elapsed() < Duration::from_secs(5));
    // Party 2 of 2 reaches no one (a port that was free a moment ago).
    let free = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap();
    let addresses = vec![free, SocketAddr::from(NEVER_DIALED)];
    let second = TcpConfig::new(2, 2, addresses, SESSION).connect_timeout(timeout);
    let result = TcpTransport::connect_with(&second, TcpListener::bind("127.0.0.1:0").unwrap());
    assert_eq!(
        result.err(),
        Some(Error::Timeout {
            label: TcpTransport::LABEL_CONNECT,
            party: 1,
            millis
        })
    );
}

#[test]
fn a_party_that_starts_late_is_waited_for() {
    // Party 2 starts first and retries until party 1, 400 ms later, binds its own address
    // (`TcpTransport::connect`). The address was free a moment before; if another process took
    // it in between, the attempt is repeated with another one.
    for attempt in 0..3 {
        let free = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let (l2, a2) = {
            let (mut l, a) = listeners(1);
            (l.pop().unwrap(), a[0])
        };
        let addresses = vec![free, a2];
        let first = TcpConfig::new(2, 1, addresses.clone(), SESSION);
        let second =
            TcpConfig::new(2, 2, addresses, SESSION).connect_timeout(Duration::from_secs(10));
        let (r1, r2) = std::thread::scope(|s| {
            let h2 = s.spawn(|| {
                let t = TcpTransport::connect_with(&second, l2)?;
                Endpoint::connect(t, Q, ZeroSharing::Fresh, [2; 32])?.any("late", true)
            });
            std::thread::sleep(Duration::from_millis(400));
            let r1 = TcpTransport::connect(&first)
                .and_then(|t| Endpoint::connect(t, Q, ZeroSharing::Fresh, [1; 32]))
                .and_then(|mut ep| ep.any("late", false));
            (r1, h2.join().unwrap())
        });
        if let Err(Error::Transport { message, .. }) = &r1
            && message.starts_with("binding")
            && attempt < 2
        {
            eprintln!("{free} was taken in between ({message}); another attempt");
            continue;
        }
        assert_eq!(r1, Ok(true));
        assert_eq!(r2, Ok(true));
        return;
    }
}

#[test]
fn invalid_configurations_are_errors() {
    let (_, addresses) = listeners(2);
    let bound = || TcpListener::bind("127.0.0.1:0").unwrap();
    let cases = [
        (TcpConfig::new(0, 1, vec![], SESSION), "n = 0"),
        (TcpConfig::new(2, 0, addresses.clone(), SESSION), "id 0"),
        (
            TcpConfig::new(2, 3, addresses.clone(), SESSION),
            "id 3 of 2",
        ),
        (
            TcpConfig::new(3, 1, addresses.clone(), SESSION),
            "2 addresses for 3",
        ),
    ];
    for (config, what) in cases {
        let result = TcpTransport::connect_with(&config, bound());
        assert!(
            matches!(
                result,
                Err(Error::Transport { .. } | Error::InvalidParty { .. })
            ),
            "{what}: {result:?}"
        );
        assert!(TcpTransport::connect(&config).is_err(), "{what}");
    }
    assert!(matches!(
        TcpTransport::connect_with(&TcpConfig::new(2, 3, addresses, SESSION), bound()),
        Err(Error::InvalidParty { party: 3, n: 2 })
    ));
    // One party alone: nothing to connect, every call is local.
    let (mut l, a) = listeners(1);
    let t =
        TcpTransport::connect_with(&TcpConfig::new(1, 1, a, SESSION), l.pop().unwrap()).unwrap();
    assert_eq!((t.n(), t.id()), (1, 1));
    let mut ep = Endpoint::connect(t, Q, ZeroSharing::PairwiseSeeds, [0; 32]).unwrap();
    assert_eq!(ep.fah("one", &[Q + 1]).unwrap(), vec![1]);
}
