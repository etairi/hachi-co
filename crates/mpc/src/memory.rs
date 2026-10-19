//! The in-memory transport: $n$ parties of the same program, one channel per ordered pair.

use crate::{Error, Frame, PartyId, Result, Transport};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

/// One frame seen by a [`Tap`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TapRecord {
    /// The sender.
    pub from: PartyId,
    /// The receiver.
    pub to: PartyId,
    /// The frame, as sent.
    pub frame: Frame,
}

/// A recorder of every frame sent on an in-memory network, for tests: the view of a coalition
/// (the frames its members receive) and the distribution of the masked broadcasts. Cloning a
/// `Tap` gives another handle to the same record.
#[derive(Clone, Debug, Default)]
pub struct Tap(Arc<Mutex<Vec<TapRecord>>>);

impl Tap {
    /// An empty recorder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Every frame sent so far, in the order of the `send` calls.
    pub fn records(&self) -> Vec<TapRecord> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// Every frame sent so far to party `to`.
    pub fn received_by(&self, to: PartyId) -> Vec<TapRecord> {
        self.records().into_iter().filter(|r| r.to == to).collect()
    }

    /// Forgets every frame recorded so far.
    pub fn clear(&self) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }

    fn push(&self, record: TapRecord) {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(record);
    }
}

/// One party's end of an in-memory network ([`memory_network`]): an unbounded channel to and
/// from every other party. `send` never blocks; `recv` waits at most the network's timeout.
/// Dropping the transport closes its channels, so that the other parties get
/// [`Error::PeerLeft`] instead of waiting for it.
#[derive(Debug)]
pub struct MemoryTransport {
    id: PartyId,
    to: Vec<Option<Sender<Frame>>>,
    from: Vec<Option<Receiver<Frame>>>,
    timeout: Duration,
    tap: Option<Tap>,
}

/// The $n$ ends of a fresh in-memory network, party $p$ at index $p - 1$. Every `recv` gives up
/// after `timeout` with [`Error::Timeout`]; if `tap` is given, it records every frame sent.
///
/// # Errors
/// [`Error::Transport`] if $n = 0$.
pub fn memory_network(
    n: usize,
    timeout: Duration,
    tap: Option<&Tap>,
) -> Result<Vec<MemoryTransport>> {
    if n == 0 {
        return Err(Error::Transport {
            label: "memory_network",
            message: "a network needs at least one party".to_owned(),
        });
    }
    let mut to: Vec<Vec<Option<Sender<Frame>>>> = (0..n).map(|_| vec![None; n]).collect();
    let mut from: Vec<Vec<Option<Receiver<Frame>>>> =
        (0..n).map(|_| (0..n).map(|_| None).collect()).collect();
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let (tx, rx) = mpsc::channel();
                to[i][j] = Some(tx);
                from[j][i] = Some(rx);
            }
        }
    }
    Ok(to
        .into_iter()
        .zip(from)
        .enumerate()
        .map(|(i, (to, from))| MemoryTransport {
            id: i + 1,
            to,
            from,
            timeout,
            tap: tap.cloned(),
        })
        .collect())
}

impl MemoryTransport {
    /// The channel index of `party`, which must be another party.
    fn peer(&self, party: PartyId) -> Result<usize> {
        let n = self.to.len();
        if party == 0 || party > n || party == self.id {
            return Err(Error::InvalidParty { party, n });
        }
        Ok(party - 1)
    }
}

impl Transport for MemoryTransport {
    fn n(&self) -> usize {
        self.to.len()
    }

    fn id(&self) -> PartyId {
        self.id
    }

    fn send(&mut self, label: &'static str, to: PartyId, frame: Frame) -> Result<()> {
        let j = self.peer(to)?;
        if let Some(tap) = &self.tap {
            tap.push(TapRecord {
                from: self.id,
                to,
                frame: frame.clone(),
            });
        }
        let channel = self.to[j]
            .as_ref()
            .ok_or(Error::PeerLeft { label, party: to })?;
        channel
            .send(frame)
            .map_err(|_| Error::PeerLeft { label, party: to })
    }

    fn recv(&mut self, label: &'static str, from: PartyId) -> Result<Frame> {
        let j = self.peer(from)?;
        let channel = self.from[j]
            .as_ref()
            .ok_or(Error::PeerLeft { label, party: from })?;
        channel.recv_timeout(self.timeout).map_err(|e| match e {
            RecvTimeoutError::Timeout => Error::Timeout {
                label,
                party: from,
                millis: u64::try_from(self.timeout.as_millis()).unwrap_or(u64::MAX),
            },
            RecvTimeoutError::Disconnected => Error::PeerLeft { label, party: from },
        })
    }
}
