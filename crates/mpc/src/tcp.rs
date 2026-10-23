//! The TCP transport: every party in its own process, or on its own machine, with one TCP
//! connection per pair of parties and the semantics of [`MemoryTransport`](crate::MemoryTransport).
//!
//! # Set-up
//!
//! [`TcpTransport::connect`] binds party $i$'s listener on its address
//! ([`TcpTransport::connect_with`] takes one that is already bound), connects to every party
//! $j \lt i$, in the order $j = 1, \dots, i - 1$, retrying while $j$ does not listen yet, and then
//! accepts a connection from every party $j \gt i$, in any order, all within the connect timeout
//! of the [`TcpConfig`]. The parties may start in any order: each binds its listener before it
//! connects anywhere, so the connections of higher parties wait in its backlog until it accepts;
//! party 1 connects to no one and accepts at once, and by induction on $i$ every party reaches
//! its accepting phase.
//!
//! On every new connection both ends set `TCP_NODELAY`, send their handshake at once and then
//! read the other's, 64 bytes ([`TcpTransport::HANDSHAKE_BYTES`]):
//!
//! | Bytes | Field |
//! |---|---|
//! | 0..8 | [`TcpTransport::MAGIC`], the ASCII bytes `HCO-TCP` and a zero byte |
//! | 8..16 | [`TcpTransport::VERSION`], `u64` little-endian |
//! | 16..48 | the session id ([`TcpConfig::session`]) |
//! | 48..56 | $n$, `u64` little-endian |
//! | 56..64 | the sender's party index, `u64` little-endian |
//!
//! A party rejects a handshake with other magic bytes or another version (another wire format),
//! another session id, or another $n$; on a connection that it opened to party $j$, a sender
//! index other than $j$; on a connection that it accepted, a sender index that is not in
//! $i + 1, \dots, n$ or that is connected already. Every failure to complete the handshake on an
//! open connection (one of these mismatches, the end of the stream, or silence until the connect
//! timeout) is [`Error::Handshake`]; a party that cannot be reached, or that never connects,
//! before the connect timeout is [`Error::Timeout`] with the label
//! [`TcpTransport::LABEL_CONNECT`]; a failing socket call is [`Error::Transport`].
//!
//! # Frames
//!
//! After the handshake a connection carries [`Frame`]s in both directions, each encoded as
//!
//! | Field | Encoding |
//! |---|---|
//! | label length $L$ | `u16` little-endian, $L \le$ [`TcpTransport::MAX_LABEL_BYTES`] $= 255$ |
//! | label | $L$ bytes of UTF-8 |
//! | round | `u64` little-endian |
//! | payload length $m$, in words | `u64` little-endian, $m \le$ [`TcpTransport::MAX_PAYLOAD_WORDS`] $= 2^{28}$ |
//! | payload | $m$ words, each `u64` little-endian |
//!
//! that is $18 + L + 8m$ bytes, of which [`Stats`](crate::Stats) counts the $8m$ of the payload.
//! A frame of $2^{28}$ words (2 GiB) carries the resharing of Commit up to $\ell = 26$ with
//! $k = 4$ ($2^\ell k$ words to each party, `6_Collaborative_Hachi.tex`:303-305); larger
//! witnesses need a streaming transfer, which this transport does not do. The receiver checks
//! both bounds before it allocates anything, and it allocates the payload as its words arrive,
//! so a peer cannot make it reserve memory for words that it does not send. A frame that breaks
//! the format (a label longer than the bound or not UTF-8, a payload longer than the bound) is
//! [`Error::Malformed`] at the receiver, which then closes the connection: the stream has no
//! markers from which to find the next frame. Received labels are [`Cow::Owned`].
//!
//! # Threads, timeouts and failures
//!
//! Each connection has a writer thread, fed by an unbounded channel, so [`Transport::send`] never
//! blocks, and a reader thread, which reads frames as they arrive and queues them, so
//! [`Transport::recv`] waits at most the receive timeout for the next frame of one peer
//! ([`Error::Timeout`] otherwise) and the frames of one peer arrive in the order they were sent.
//! The peer's end of the connection is drained continuously, so writes stall only if the peer
//! takes in nothing at all; a write that stalls for the receive timeout fails, and the writer
//! closes the connection.
//!
//! A peer whose connection ends (it dropped its transport, or its process died) has left:
//! [`Transport::recv`] returns its queued frames and then [`Error::PeerLeft`], and
//! [`Transport::send`] returns [`Error::PeerLeft`] as soon as the end of the connection has been
//! seen. As with any network, the end is seen only when it arrives: a frame sent to a peer that
//! has just left may be dropped without an error, and the failure shows at the next exchange
//! with it.
//!
//! Dropping a [`TcpTransport`] writes out every frame queued so far, ends each connection (so the
//! other parties receive those frames and then [`Error::PeerLeft`]) and joins its threads. A
//! party that stops early therefore never leaves the others waiting for the timeout, and a party
//! that finishes never cuts off its last frames.
//!
//! # Not a secure channel
//!
//! The connections are neither encrypted nor authenticated. Every frame travels in the clear,
//! among them the pairwise seeds of [`ZeroSharing::PairwiseSeeds`](crate::ZeroSharing::PairwiseSeeds), so an
//! eavesdropper learns every mask of $\mathcal F\_{\mathsf{AH}}$, and with it every party's
//! input to it; the plan (`docs/hachi-co-plan.md`, §5) requires these channels to be private.
//! Anyone who can reach a listener and knows the session id can take a free party index. Run it
//! on a trusted network, or tunnel the connections (SSH, WireGuard, a TLS proxy).

use crate::{Error, Frame, PartyId, Result, Transport};
use std::borrow::Cow;
use std::io::{self, BufReader, BufWriter, ErrorKind, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// The buffer size of the readers and writers of a connection.
const BUFFER_BYTES: usize = 1 << 16;
/// The number of payload words encoded or decoded at a time.
const CHUNK_WORDS: usize = 1 << 10;
/// The capacity reserved for a payload before its words arrive (512 KiB); beyond it the vector
/// grows with the words received.
const RESERVE_WORDS: usize = 1 << 16;
/// The pause between two polls of the listener while a party accepts.
const ACCEPT_POLL: Duration = Duration::from_millis(5);
/// The first pause between two attempts to reach a party that does not listen yet; it doubles
/// up to [`DIAL_PAUSE_MAX`].
const DIAL_PAUSE_MIN: Duration = Duration::from_millis(10);
/// The longest pause between two attempts to reach a party.
const DIAL_PAUSE_MAX: Duration = Duration::from_millis(100);
/// The shortest socket timeout (a zero timeout is not a valid socket option).
const MIN_SOCKET_TIMEOUT: Duration = Duration::from_millis(1);

/// The configuration of one party's [`TcpTransport`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TcpConfig {
    /// The number of parties $n \ge 1$.
    pub n: usize,
    /// This party's index $i \in \lbrace 1, \dots, n \rbrace$.
    pub id: PartyId,
    /// The socket addresses of the $n$ parties, party $p$ at index $p - 1$. Party $i$ listens on
    /// its own ([`TcpTransport::connect`]) and connects to those of the parties $j \lt i$.
    pub addresses: Vec<SocketAddr>,
    /// The session id, the same at every party of a run. A process of another run (a stale one,
    /// or one with another configuration) fails the handshake. Use a fresh one for every run; it
    /// is not secret.
    pub session: [u8; 32],
    /// How long the set-up may take, from the call of [`TcpTransport::connect`] until every
    /// connection has passed its handshake.
    pub connect_timeout: Duration,
    /// How long [`Transport::recv`] waits for a frame before it fails with [`Error::Timeout`];
    /// also how long a write may stall before the connection is given up.
    pub recv_timeout: Duration,
}

impl TcpConfig {
    /// Party `id` of the `n` parties at `addresses`, in the session `session`, with a connect
    /// timeout of 30 s and a receive timeout of 60 s.
    pub fn new(n: usize, id: PartyId, addresses: Vec<SocketAddr>, session: [u8; 32]) -> Self {
        Self {
            n,
            id,
            addresses,
            session,
            connect_timeout: Duration::from_secs(30),
            recv_timeout: Duration::from_secs(60),
        }
    }

    /// With another connect timeout.
    #[must_use]
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    /// With another receive timeout.
    #[must_use]
    pub fn recv_timeout(mut self, timeout: Duration) -> Self {
        self.recv_timeout = timeout;
        self
    }

    /// Checks $n \ge 1$, $1 \le i \le n$ and one address per party.
    fn validate(&self) -> Result<()> {
        if self.n == 0 {
            return Err(Error::Transport {
                label: TcpTransport::LABEL_CONNECT,
                message: "a network needs at least one party".to_owned(),
            });
        }
        if self.id == 0 || self.id > self.n {
            return Err(Error::InvalidParty {
                party: self.id,
                n: self.n,
            });
        }
        if self.addresses.len() != self.n {
            return Err(Error::Transport {
                label: TcpTransport::LABEL_CONNECT,
                message: format!("{} addresses for {} parties", self.addresses.len(), self.n),
            });
        }
        Ok(())
    }
}

/// A duration in milliseconds, saturating.
fn millis(d: Duration) -> u64 {
    u64::try_from(d.as_millis()).unwrap_or(u64::MAX)
}

/// The end of the set-up: `None` if the connect timeout is too large to add to the clock, and
/// then the set-up waits as long as it takes.
struct Deadline(Option<Instant>);

impl Deadline {
    fn after(timeout: Duration) -> Self {
        Self(Instant::now().checked_add(timeout))
    }

    /// The time left (zero once the deadline has passed), or `None` without a deadline.
    fn remaining(&self) -> Option<Duration> {
        self.0.map(|d| d.saturating_duration_since(Instant::now()))
    }

    fn passed(&self) -> bool {
        self.remaining().is_some_and(|left| left.is_zero())
    }

    /// The time left as a socket timeout (at least [`MIN_SOCKET_TIMEOUT`]; `None` blocks).
    fn socket_timeout(&self) -> Option<Duration> {
        self.remaining().map(|left| left.max(MIN_SOCKET_TIMEOUT))
    }
}

/// One end's handshake: the run it belongs to and its index (see the module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Hello {
    session: [u8; 32],
    n: u64,
    id: u64,
}

impl Hello {
    /// The 64 bytes of the handshake.
    fn encode(&self) -> [u8; TcpTransport::HANDSHAKE_BYTES] {
        let mut out = [0u8; TcpTransport::HANDSHAKE_BYTES];
        let fields = [TcpTransport::MAGIC, TcpTransport::VERSION.to_le_bytes()]
            .into_iter()
            .chain(self.session.as_chunks::<8>().0.iter().copied())
            .chain([self.n.to_le_bytes(), self.id.to_le_bytes()]);
        for (chunk, field) in out.as_chunks_mut::<8>().0.iter_mut().zip(fields) {
            *chunk = field;
        }
        out
    }

    /// Decodes a handshake, checking the wire format: the magic bytes and the version.
    fn decode(bytes: &[u8; TcpTransport::HANDSHAKE_BYTES]) -> core::result::Result<Self, String> {
        let &[magic, version, s0, s1, s2, s3, n, id] = bytes.as_chunks::<8>().0 else {
            return Err("a handshake of the wrong length".to_owned());
        };
        if magic != TcpTransport::MAGIC {
            return Err(format!(
                "it does not speak this wire format (magic bytes {magic:02x?})"
            ));
        }
        let version = u64::from_le_bytes(version);
        if version != TcpTransport::VERSION {
            return Err(format!(
                "it speaks version {version} of the wire format, this party version {}",
                TcpTransport::VERSION
            ));
        }
        let mut session = [0u8; 32];
        for (chunk, field) in session
            .as_chunks_mut::<8>()
            .0
            .iter_mut()
            .zip([s0, s1, s2, s3])
        {
            *chunk = field;
        }
        Ok(Self {
            session,
            n: u64::from_le_bytes(n),
            id: u64::from_le_bytes(id),
        })
    }

    /// Checks that the peer `theirs` belongs to the same run as `self`: the session id and $n$.
    fn check_run(&self, theirs: &Self) -> core::result::Result<(), String> {
        if theirs.session != self.session {
            return Err("it belongs to another session (another session id)".to_owned());
        }
        if theirs.n != self.n {
            return Err(format!(
                "it runs with n = {}, this party with n = {}",
                theirs.n, self.n
            ));
        }
        Ok(())
    }
}

/// Sends `ours` on `stream` and reads and decodes the peer's handshake, within the deadline.
/// Sets `TCP_NODELAY` first.
fn exchange_hellos(
    stream: &TcpStream,
    ours: &Hello,
    deadline: &Deadline,
) -> core::result::Result<Hello, String> {
    let failed = |e: io::Error| match e.kind() {
        ErrorKind::WouldBlock | ErrorKind::TimedOut => {
            "no handshake before the connect timeout".to_owned()
        }
        ErrorKind::UnexpectedEof => "the connection closed during the handshake".to_owned(),
        _ => format!("the handshake failed: {e}"),
    };
    let timeout = deadline.socket_timeout();
    stream.set_nodelay(true).map_err(failed)?;
    stream.set_write_timeout(timeout).map_err(failed)?;
    stream.set_read_timeout(timeout).map_err(failed)?;
    let mut writer = stream;
    writer.write_all(&ours.encode()).map_err(failed)?;
    let mut theirs = [0u8; TcpTransport::HANDSHAKE_BYTES];
    let mut reader = stream;
    reader.read_exact(&mut theirs).map_err(failed)?;
    Hello::decode(&theirs)
}

/// Whether a failed connection attempt is worth repeating: the party does not listen yet, or
/// the network is not ready.
fn retry_dial(kind: ErrorKind) -> bool {
    matches!(
        kind,
        ErrorKind::ConnectionRefused
            | ErrorKind::ConnectionReset
            | ErrorKind::ConnectionAborted
            | ErrorKind::TimedOut
            | ErrorKind::WouldBlock
            | ErrorKind::Interrupted
            | ErrorKind::HostUnreachable
            | ErrorKind::NetworkUnreachable
    )
}

/// Opens a connection to party `party` at `address`, retrying until the deadline.
fn dial(
    address: SocketAddr,
    party: PartyId,
    deadline: &Deadline,
    timeout: Duration,
) -> Result<TcpStream> {
    let mut pause = DIAL_PAUSE_MIN;
    loop {
        let attempt = match deadline.remaining() {
            Some(left) if left.is_zero() => {
                return Err(Error::Timeout {
                    label: TcpTransport::LABEL_CONNECT,
                    party,
                    millis: millis(timeout),
                });
            }
            Some(left) => TcpStream::connect_timeout(&address, left),
            None => TcpStream::connect(address),
        };
        match attempt {
            Ok(stream) => return Ok(stream),
            Err(e) if retry_dial(e.kind()) => {
                thread::sleep(deadline.remaining().map_or(pause, |left| pause.min(left)));
                pause = (pause * 2).min(DIAL_PAUSE_MAX);
            }
            Err(e) => {
                return Err(Error::Transport {
                    label: TcpTransport::LABEL_CONNECT,
                    message: format!("connecting to party {party} at {address}: {e}"),
                });
            }
        }
    }
}

/// Accepts a connection from every party $j \gt i$ on `listener` and checks its handshake,
/// filling `streams` (index $j - 1$).
fn accept_all(
    listener: &TcpListener,
    streams: &mut [Option<TcpStream>],
    ours: &Hello,
    config: &TcpConfig,
    deadline: &Deadline,
) -> Result<()> {
    let (n, id) = (config.n, config.id);
    let failed = |what: &str, e: io::Error| Error::Transport {
        label: TcpTransport::LABEL_CONNECT,
        message: format!("{what}: {e}"),
    };
    listener
        .set_nonblocking(true)
        .map_err(|e| failed("configuring the listener", e))?;
    while let Some(missing) = (id + 1..=n).find(|&j| streams[j - 1].is_none()) {
        match listener.accept() {
            Ok((stream, address)) => {
                // Some systems pass the listener's non-blocking mode on to accepted sockets.
                stream
                    .set_nonblocking(false)
                    .map_err(|e| failed("configuring an accepted connection", e))?;
                let rejected = |peer: String, message: String| Error::Handshake { peer, message };
                let theirs = exchange_hellos(&stream, ours, deadline)
                    .map_err(|m| rejected(address.to_string(), m))?;
                let peer = format!("party {} at {address}", theirs.id);
                ours.check_run(&theirs)
                    .map_err(|m| rejected(peer.clone(), m))?;
                let j = usize::try_from(theirs.id)
                    .ok()
                    .filter(|j| (id + 1..=n).contains(j))
                    .ok_or_else(|| {
                        rejected(
                            peer.clone(),
                            format!(
                                "party {id} of {n} accepts connections only from parties {}..={n}",
                                id + 1
                            ),
                        )
                    })?;
                if streams[j - 1].is_some() {
                    return Err(rejected(peer, format!("party {j} is connected already")));
                }
                streams[j - 1] = Some(stream);
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => {
                if deadline.passed() {
                    return Err(Error::Timeout {
                        label: TcpTransport::LABEL_CONNECT,
                        party: missing,
                        millis: millis(config.connect_timeout),
                    });
                }
                thread::sleep(ACCEPT_POLL);
            }
            Err(e)
                if matches!(
                    e.kind(),
                    ErrorKind::Interrupted | ErrorKind::ConnectionAborted
                ) => {}
            Err(e) => return Err(failed("accepting a connection", e)),
        }
    }
    Ok(())
}

/// Checks that a frame fits the wire format: a label of at most
/// [`TcpTransport::MAX_LABEL_BYTES`] bytes and a payload of at most
/// [`TcpTransport::MAX_PAYLOAD_WORDS`] words.
fn check_frame(label_bytes: usize, payload_words: usize) -> core::result::Result<(), String> {
    if label_bytes > TcpTransport::MAX_LABEL_BYTES {
        return Err(format!(
            "a label of {label_bytes} bytes, longer than the {} of the wire format",
            TcpTransport::MAX_LABEL_BYTES
        ));
    }
    if !u64::try_from(payload_words).is_ok_and(|m| m <= TcpTransport::MAX_PAYLOAD_WORDS) {
        return Err(format!(
            "a payload of {payload_words} words, longer than the {} of the wire format",
            TcpTransport::MAX_PAYLOAD_WORDS
        ));
    }
    Ok(())
}

/// Writes `frame` in the wire format (see the module documentation).
///
/// # Errors
/// [`ErrorKind::InvalidInput`] for a frame that [`check_frame`] refuses (nothing is written
/// then); the errors of `out`.
fn write_frame(out: &mut impl Write, frame: &Frame) -> io::Result<()> {
    let label = frame.label.as_bytes();
    check_frame(label.len(), frame.payload.len())
        .map_err(|m| io::Error::new(ErrorKind::InvalidInput, m))?;
    let label_len = u16::try_from(label.len())
        .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "label length"))?;
    out.write_all(&label_len.to_le_bytes())?;
    out.write_all(label)?;
    out.write_all(&frame.round.to_le_bytes())?;
    out.write_all(&(frame.payload.len() as u64).to_le_bytes())?;
    let mut buf = [0u8; 8 * CHUNK_WORDS];
    for words in frame.payload.chunks(CHUNK_WORDS) {
        let bytes = &mut buf[..8 * words.len()];
        for (chunk, word) in bytes.as_chunks_mut::<8>().0.iter_mut().zip(words) {
            *chunk = word.to_le_bytes();
        }
        out.write_all(bytes)?;
    }
    Ok(())
}

/// Why [`read_frame`] returned no frame.
#[derive(Debug)]
enum ReadError {
    /// The stream ended inside a frame.
    Truncated,
    /// The frame breaks the wire format.
    Invalid(String),
    /// Another I/O error.
    Io(io::Error),
}

impl From<io::Error> for ReadError {
    fn from(e: io::Error) -> Self {
        if e.kind() == ErrorKind::UnexpectedEof {
            Self::Truncated
        } else {
            Self::Io(e)
        }
    }
}

/// Reads one little-endian word.
fn read_word(input: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0u8; 8];
    input.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

/// Reads one frame in the wire format: `Ok(None)` if the stream ends before the frame's first
/// byte. The label length and the payload length are checked before anything is allocated, and
/// the payload grows as its words arrive.
fn read_frame(input: &mut impl Read) -> core::result::Result<Option<Frame>, ReadError> {
    let mut len = [0u8; 2];
    loop {
        match input.read(&mut len[..1]) {
            Ok(0) => return Ok(None),
            Ok(_) => break,
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e.into()),
        }
    }
    input.read_exact(&mut len[1..])?;
    let label_len = usize::from(u16::from_le_bytes(len));
    check_frame(label_len, 0).map_err(ReadError::Invalid)?;
    let mut label = vec![0u8; label_len];
    input.read_exact(&mut label)?;
    let label = String::from_utf8(label)
        .map_err(|e| ReadError::Invalid(format!("a label that is not UTF-8 ({e})")))?;
    let round = read_word(input)?;
    let words = read_word(input)?;
    if words > TcpTransport::MAX_PAYLOAD_WORDS {
        return Err(ReadError::Invalid(format!(
            "a payload of {words} words, longer than the {} of the wire format",
            TcpTransport::MAX_PAYLOAD_WORDS
        )));
    }
    let words = usize::try_from(words)
        .map_err(|_| ReadError::Invalid(format!("a payload of {words} words")))?;
    let mut payload = Vec::with_capacity(words.min(RESERVE_WORDS));
    let mut buf = [0u8; 8 * CHUNK_WORDS];
    while payload.len() < words {
        let take = (words - payload.len()).min(CHUNK_WORDS);
        let bytes = &mut buf[..8 * take];
        input.read_exact(bytes)?;
        payload.extend(
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|w| u64::from_le_bytes(*w)),
        );
    }
    Ok(Some(Frame {
        label: Cow::Owned(label),
        round,
        payload,
    }))
}

/// What the reader thread of a connection passes on.
#[derive(Debug)]
enum Incoming {
    /// A frame.
    Frame(Frame),
    /// A frame that breaks the wire format; the reader closed the connection after it.
    Malformed(String),
    /// An I/O error other than the end of the connection; the reader closed the connection
    /// after it.
    Failed(String),
}

/// The I/O errors that mean that the peer closed or reset the connection.
fn connection_ended(kind: ErrorKind) -> bool {
    matches!(
        kind,
        ErrorKind::UnexpectedEof
            | ErrorKind::ConnectionReset
            | ErrorKind::ConnectionAborted
            | ErrorKind::BrokenPipe
            | ErrorKind::NotConnected
    )
}

/// The reader thread: queues the frames of `stream` until the connection ends or a frame breaks
/// the wire format, then marks the peer as gone.
fn read_loop(stream: TcpStream, queue: Sender<Incoming>, left: Arc<AtomicBool>) {
    let mut input = BufReader::with_capacity(BUFFER_BYTES, &stream);
    let last = loop {
        match read_frame(&mut input) {
            Ok(Some(frame)) => {
                if queue.send(Incoming::Frame(frame)).is_err() {
                    break None;
                }
            }
            Ok(None) | Err(ReadError::Truncated) => break None,
            Err(ReadError::Io(e)) if connection_ended(e.kind()) => break None,
            Err(ReadError::Invalid(message)) => break Some(Incoming::Malformed(message)),
            Err(ReadError::Io(e)) => break Some(Incoming::Failed(e.to_string())),
        }
    };
    left.store(true, Ordering::Release);
    if let Some(event) = last {
        // The stream is out of step: close it, so that the peer stops too instead of waiting.
        let _ = stream.shutdown(Shutdown::Both);
        let _ = queue.send(event);
    }
}

/// The writer thread: writes the queued frames to `stream` until the queue closes (the
/// transport was dropped; then it ends the stream) or a write fails (then it closes the
/// connection).
fn write_loop(stream: TcpStream, queue: Receiver<Frame>) {
    let mut out = BufWriter::with_capacity(BUFFER_BYTES, &stream);
    for frame in queue.iter() {
        if write_frame(&mut out, &frame)
            .and_then(|()| out.flush())
            .is_err()
        {
            let _ = stream.shutdown(Shutdown::Both);
            return;
        }
    }
    let _ = out.flush();
    let _ = stream.shutdown(Shutdown::Write);
}

/// The connection to one peer and its two threads.
#[derive(Debug)]
struct Peer {
    stream: TcpStream,
    /// The writer's queue; `None` once the transport is being dropped.
    outgoing: Option<Sender<Frame>>,
    incoming: Receiver<Incoming>,
    /// Set by the reader when the connection has ended.
    left: Arc<AtomicBool>,
    writer: Option<JoinHandle<()>>,
    reader: Option<JoinHandle<()>>,
}

impl Peer {
    /// Starts the threads of the connection `stream` of party `id` to party `party`.
    fn start(
        stream: TcpStream,
        id: PartyId,
        party: PartyId,
        recv_timeout: Duration,
    ) -> Result<Self> {
        let failed = |what: &str, e: io::Error| Error::Transport {
            label: TcpTransport::LABEL_CONNECT,
            message: format!("connection to party {party}: {what}: {e}"),
        };
        stream
            .set_read_timeout(None)
            .and_then(|()| stream.set_write_timeout(Some(recv_timeout.max(MIN_SOCKET_TIMEOUT))))
            .map_err(|e| failed("configuring the socket", e))?;
        let (outgoing, to_write) = mpsc::channel();
        let (queue, incoming) = mpsc::channel();
        let left = Arc::new(AtomicBool::new(false));
        let writer = {
            let stream = stream
                .try_clone()
                .map_err(|e| failed("cloning the socket", e))?;
            thread::Builder::new()
                .name(format!("tcp-{id}-to-{party}"))
                .spawn(move || write_loop(stream, to_write))
                .map_err(|e| failed("spawning the writer thread", e))?
        };
        let reader = {
            let stream = stream
                .try_clone()
                .map_err(|e| failed("cloning the socket", e))?;
            let left = Arc::clone(&left);
            thread::Builder::new()
                .name(format!("tcp-{id}-from-{party}"))
                .spawn(move || read_loop(stream, queue, left))
                .map_err(|e| failed("spawning the reader thread", e))?
        };
        Ok(Self {
            stream,
            outgoing: Some(outgoing),
            incoming,
            left,
            writer: Some(writer),
            reader: Some(reader),
        })
    }
}

/// One party's end of a TCP network: one connection to every other party, each with a writer
/// thread and a reader thread. See the module documentation for the set-up, the wire format and
/// the failure semantics, which are those of [`MemoryTransport`](crate::MemoryTransport): `send`
/// never blocks, `recv` waits at most the receive timeout, the frames of one peer arrive in
/// order, a peer that left gives [`Error::PeerLeft`], and dropping the transport closes its
/// connections after writing out what it queued.
#[derive(Debug)]
pub struct TcpTransport {
    id: PartyId,
    n: usize,
    timeout: Duration,
    /// The connection to party $p$ at index $p - 1$; `None` at this party's own index.
    peers: Vec<Option<Peer>>,
}

impl TcpTransport {
    /// The first 8 bytes of a handshake: `HCO-TCP` and a zero byte.
    pub const MAGIC: [u8; 8] = *b"HCO-TCP\0";
    /// The version of the wire format (the handshake and the frames).
    pub const VERSION: u64 = 1;
    /// The length of a handshake in bytes.
    pub const HANDSHAKE_BYTES: usize = 64;
    /// The longest label of a frame, in bytes.
    pub const MAX_LABEL_BYTES: usize = 255;
    /// The longest payload of a frame, in words: $2^{28}$, that is 2 GiB.
    pub const MAX_PAYLOAD_WORDS: u64 = 1 << 28;
    /// The label of the errors of the set-up ([`TcpTransport::connect`]).
    pub const LABEL_CONNECT: &'static str = "hachi-co/mpc/tcp/connect";

    /// Connects party `config.id`: binds a listener on its address `config.addresses[id - 1]`,
    /// then runs [`TcpTransport::connect_with`]. All parties must call it within the connect
    /// timeout of each other.
    ///
    /// # Errors
    /// As [`TcpTransport::connect_with`]; [`Error::Transport`] if the address cannot be bound
    /// (for example because it is in use).
    pub fn connect(config: &TcpConfig) -> Result<Self> {
        config.validate()?;
        let address = config.addresses[config.id - 1];
        let listener = TcpListener::bind(address).map_err(|e| Error::Transport {
            label: Self::LABEL_CONNECT,
            message: format!("binding {address}: {e}"),
        })?;
        Self::connect_with(config, listener)
    }

    /// Connects party $i$ = `config.id` with a listener that is already bound, where the other
    /// parties reach it (for example to port 0, so that tests let the system choose the ports of
    /// all parties before any of them starts): connects to the parties $j \lt i$ and accepts the
    /// parties $j \gt i$, each with a handshake, within the connect timeout (see the module
    /// documentation), then starts the threads. The listener is closed when the set-up ends.
    ///
    /// # Errors
    /// [`Error::Transport`] if $n = 0$ or there are not $n$ addresses, or if a socket call
    /// fails; [`Error::InvalidParty`] unless $1 \le i \le n$; [`Error::Timeout`] (label
    /// [`TcpTransport::LABEL_CONNECT`]) if a party cannot be reached, or does not connect,
    /// before the connect timeout; [`Error::Handshake`] if a handshake fails.
    pub fn connect_with(config: &TcpConfig, listener: TcpListener) -> Result<Self> {
        config.validate()?;
        let (n, id) = (config.n, config.id);
        let deadline = Deadline::after(config.connect_timeout);
        let ours = Hello {
            session: config.session,
            n: n as u64,
            id: id as u64,
        };
        let mut streams: Vec<Option<TcpStream>> = (0..n).map(|_| None).collect();
        for j in 1..id {
            let address = config.addresses[j - 1];
            let stream = dial(address, j, &deadline, config.connect_timeout)?;
            let peer = format!("party {j} at {address}");
            let rejected = |message: String| Error::Handshake {
                peer: peer.clone(),
                message,
            };
            let theirs = exchange_hellos(&stream, &ours, &deadline).map_err(rejected)?;
            ours.check_run(&theirs).map_err(rejected)?;
            if theirs.id != j as u64 {
                return Err(rejected(format!("it says that it is party {}", theirs.id)));
            }
            streams[j - 1] = Some(stream);
        }
        accept_all(&listener, &mut streams, &ours, config, &deadline)?;
        drop(listener);
        let mut transport = Self {
            id,
            n,
            timeout: config.recv_timeout,
            peers: (0..n).map(|_| None).collect(),
        };
        for (slot, (j, stream)) in transport.peers.iter_mut().zip((1..=n).zip(streams)) {
            if let Some(stream) = stream {
                *slot = Some(Peer::start(stream, id, j, config.recv_timeout)?);
            }
        }
        Ok(transport)
    }

    /// The connection to `party`, which must be another party.
    fn peer(&self, party: PartyId) -> Result<&Peer> {
        let invalid = Error::InvalidParty { party, n: self.n };
        if party == 0 || party > self.n || party == self.id {
            return Err(invalid);
        }
        self.peers[party - 1].as_ref().ok_or(invalid)
    }
}

impl Transport for TcpTransport {
    fn n(&self) -> usize {
        self.n
    }

    fn id(&self) -> PartyId {
        self.id
    }

    /// Queues `frame` for the writer thread of the connection to `to`.
    ///
    /// # Errors
    /// [`Error::InvalidParty`]; [`Error::PeerLeft`] once the connection to `to` has ended;
    /// [`Error::Transport`] for a frame that the wire format cannot carry (a label longer than
    /// [`TcpTransport::MAX_LABEL_BYTES`], a payload longer than
    /// [`TcpTransport::MAX_PAYLOAD_WORDS`]).
    fn send(&mut self, label: &'static str, to: PartyId, frame: Frame) -> Result<()> {
        let peer = self.peer(to)?;
        check_frame(frame.label.len(), frame.payload.len()).map_err(|m| Error::Transport {
            label,
            message: format!("cannot send to party {to}: {m}"),
        })?;
        let left = Error::PeerLeft { label, party: to };
        if peer.left.load(Ordering::Acquire) {
            return Err(left);
        }
        match &peer.outgoing {
            Some(queue) => queue.send(frame).map_err(|_| left),
            None => Err(left),
        }
    }

    /// The next frame that the reader thread of the connection to `from` queued, waiting at most
    /// the receive timeout.
    ///
    /// # Errors
    /// [`Error::InvalidParty`]; [`Error::Timeout`]; [`Error::PeerLeft`] once the connection has
    /// ended and its frames have been received; [`Error::Malformed`] for a frame that breaks the
    /// wire format; [`Error::Transport`] for another failure of the connection.
    fn recv(&mut self, label: &'static str, from: PartyId) -> Result<Frame> {
        let timeout = self.timeout;
        let peer = self.peer(from)?;
        match peer.incoming.recv_timeout(timeout) {
            Ok(Incoming::Frame(frame)) => Ok(frame),
            Ok(Incoming::Malformed(message)) => Err(Error::Malformed {
                label,
                party: from,
                message: format!("a frame with {message}"),
            }),
            Ok(Incoming::Failed(message)) => Err(Error::Transport {
                label,
                message: format!("the connection to party {from} failed: {message}"),
            }),
            Err(RecvTimeoutError::Timeout) => Err(Error::Timeout {
                label,
                party: from,
                millis: millis(timeout),
            }),
            Err(RecvTimeoutError::Disconnected) => Err(Error::PeerLeft { label, party: from }),
        }
    }
}

impl Drop for TcpTransport {
    /// Closes the writers' queues, waits until every queued frame is written (a write stalls at
    /// most the receive timeout), ends the connections and joins the threads.
    fn drop(&mut self) {
        for peer in self.peers.iter_mut().flatten() {
            peer.outgoing = None;
        }
        for peer in self.peers.iter_mut().flatten() {
            if let Some(writer) = peer.writer.take() {
                let _ = writer.join();
            }
        }
        for peer in self.peers.iter_mut().flatten() {
            let _ = peer.stream.shutdown(Shutdown::Both);
            if let Some(reader) = peer.reader.take() {
                let _ = reader.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(label: &str, round: u64, payload: Vec<u64>) -> Frame {
        Frame {
            label: Cow::Owned(label.to_owned()),
            round,
            payload,
        }
    }

    fn encode(frame: &Frame) -> Vec<u8> {
        let mut out = Vec::new();
        write_frame(&mut out, frame).unwrap();
        out
    }

    /// The fixed part of a frame with a label of `label_len` bytes and `words` payload words, as
    /// the module documentation specifies it, with a label of `x`s.
    fn header(label_len: u16, round: u64, words: u64) -> Vec<u8> {
        let mut out = label_len.to_le_bytes().to_vec();
        out.extend(vec![b'x'; usize::from(label_len)]);
        out.extend(round.to_le_bytes());
        out.extend(words.to_le_bytes());
        out
    }

    #[test]
    fn the_encoding_is_the_documented_one() {
        // Label "ab", round 258, payload [1, 2^64 - 1]: 2 + 2 + 8 + 8 + 16 bytes.
        let bytes = encode(&frame("ab", 258, vec![1, u64::MAX]));
        let mut want = vec![
            2, 0, b'a', b'b', 2, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
        ];
        want.extend([1, 0, 0, 0, 0, 0, 0, 0]);
        want.extend([0xff; 8]);
        assert_eq!(bytes, want);
        // 18 + L + 8m bytes.
        let f = frame("hachi-co/sumcheck/all-gather/final", 7, vec![3; 5]);
        assert_eq!(encode(&f).len(), 18 + f.label.len() + 8 * 5);
    }

    #[test]
    fn frames_round_trip() {
        // Empty and longest labels (255 bytes, and multi-byte UTF-8), extreme rounds and words,
        // and payloads around and across the chunk size.
        let longest = "é".repeat(127) + "x";
        assert_eq!(longest.len(), TcpTransport::MAX_LABEL_BYTES);
        let frames = vec![
            frame("", 0, vec![]),
            frame("a", u64::MAX, vec![0]),
            frame(&longest, 1, vec![u64::MAX; 3]),
            frame("chunk", 2, (0..CHUNK_WORDS as u64).collect()),
            frame(
                "chunks",
                3,
                (0..3 * CHUNK_WORDS as u64 + 17)
                    .map(|w| w.wrapping_mul(0x9e37_79b9_7f4a_7c15))
                    .collect(),
            ),
            frame("big", 4, (0..RESERVE_WORDS as u64 + 5).collect()),
        ];
        // One at a time and back to back on one stream, which then ends cleanly.
        let mut stream = Vec::new();
        for f in &frames {
            let bytes = encode(f);
            assert_eq!(read_frame(&mut bytes.as_slice()).unwrap(), Some(f.clone()));
            stream.extend(bytes);
        }
        let mut input = stream.as_slice();
        for f in &frames {
            let got = read_frame(&mut input).unwrap().unwrap();
            assert!(matches!(got.label, Cow::Owned(_)));
            assert_eq!(got, *f);
        }
        assert!(read_frame(&mut input).unwrap().is_none());
        assert!(read_frame(&mut &[0u8; 0][..]).unwrap().is_none());
    }

    #[test]
    fn truncated_frames_are_detected_at_every_length() {
        let bytes = encode(&frame("label", 9, vec![1, 2, 3]));
        for cut in 1..bytes.len() {
            let result = read_frame(&mut &bytes[..cut]);
            assert!(
                matches!(result, Err(ReadError::Truncated)),
                "cut at {cut}: {result:?}"
            );
        }
    }

    #[test]
    fn frames_that_break_the_format_are_rejected() {
        // A label over the bound, at the bound + 1 and at the largest u16.
        for len in [256u16, u16::MAX] {
            let result = read_frame(&mut header(len, 0, 0).as_slice());
            assert!(
                matches!(result, Err(ReadError::Invalid(_))),
                "{len}: {result:?}"
            );
        }
        // A label that is not UTF-8.
        let mut bad = vec![2, 0, 0xff, 0xfe];
        bad.extend(0u64.to_le_bytes());
        bad.extend(0u64.to_le_bytes());
        assert!(matches!(
            read_frame(&mut bad.as_slice()),
            Err(ReadError::Invalid(_))
        ));
        // A payload over the bound: rejected from the header alone, before any allocation.
        for words in [TcpTransport::MAX_PAYLOAD_WORDS + 1, u64::MAX] {
            let result = read_frame(&mut header(1, 0, words).as_slice());
            assert!(
                matches!(result, Err(ReadError::Invalid(_))),
                "{words}: {result:?}"
            );
        }
        // At the bound the header is accepted, and the missing words are a truncation (the
        // reader reserves RESERVE_WORDS, not 2^28 words).
        let result = read_frame(&mut header(1, 0, TcpTransport::MAX_PAYLOAD_WORDS).as_slice());
        assert!(matches!(result, Err(ReadError::Truncated)), "{result:?}");
    }

    #[test]
    fn the_writer_refuses_what_the_reader_would_reject() {
        assert!(check_frame(TcpTransport::MAX_LABEL_BYTES, 0).is_ok());
        assert!(check_frame(TcpTransport::MAX_LABEL_BYTES + 1, 0).is_err());
        assert!(check_frame(0, TcpTransport::MAX_PAYLOAD_WORDS as usize).is_ok());
        assert!(check_frame(0, TcpTransport::MAX_PAYLOAD_WORDS as usize + 1).is_err());
        let mut out = Vec::new();
        let long = frame(&"x".repeat(256), 0, vec![1]);
        let e = write_frame(&mut out, &long).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidInput);
        assert!(out.is_empty(), "nothing is written");
    }

    #[test]
    fn handshakes_round_trip_and_wrong_formats_are_rejected() {
        let hello = Hello {
            session: core::array::from_fn(|i| i as u8),
            n: 3,
            id: 2,
        };
        let bytes = hello.encode();
        // The documented layout.
        assert_eq!(&bytes[..8], b"HCO-TCP\0");
        assert_eq!(bytes[8..16], 1u64.to_le_bytes());
        assert_eq!(bytes[16..48], hello.session);
        assert_eq!(bytes[48..56], 3u64.to_le_bytes());
        assert_eq!(bytes[56..64], 2u64.to_le_bytes());
        assert_eq!(Hello::decode(&bytes), Ok(hello));
        let mut magic = bytes;
        magic[0] ^= 1;
        assert!(Hello::decode(&magic).is_err());
        let mut version = bytes;
        version[8] = 2;
        assert!(Hello::decode(&version).is_err());
        // The same run, another session, another n.
        assert_eq!(hello.check_run(&Hello { id: 1, ..hello }), Ok(()));
        let mut session = hello.session;
        session[31] ^= 1;
        assert!(hello.check_run(&Hello { session, ..hello }).is_err());
        assert!(hello.check_run(&Hello { n: 4, ..hello }).is_err());
    }
}
