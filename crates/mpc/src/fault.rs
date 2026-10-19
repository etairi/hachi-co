//! Failure injection: a [`Transport`] wrapper that leaves the network or alters frames.

use crate::{Error, Frame, PartyId, Result, Transport};
use std::borrow::Cow;

/// A fault that [`Faulty`] injects into the frames its party sends. Frames are counted from 0
/// over every frame the party sends, the set-up included, so a sweep over the frame index covers
/// every point at which the other parties can observe a failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fault {
    /// The party leaves the network (its channels close) instead of sending frame
    /// `after_frames`, that is after sending exactly `after_frames` frames.
    Leave {
        /// The number of frames sent before leaving.
        after_frames: usize,
    },
    /// Frame `frame` loses its last payload word (or gains one if its payload is empty).
    Truncate {
        /// The index of the frame.
        frame: usize,
    },
    /// Frame `frame` gains a zero word at the end of its payload.
    Extend {
        /// The index of the frame.
        frame: usize,
    },
    /// Frame `frame` carries another label.
    Relabel {
        /// The index of the frame.
        frame: usize,
    },
    /// Frame `frame` carries the next round index.
    NextRound {
        /// The index of the frame.
        frame: usize,
    },
    /// Word `word` (modulo the payload length) of frame `frame` has its lowest bit flipped.
    FlipBit {
        /// The index of the frame.
        frame: usize,
        /// The index of the word.
        word: usize,
    },
}

/// The label that [`Fault::Relabel`] puts on a frame.
const RELABELLED: &str = "hachi-co/mpc/fault/relabelled";

/// A [`Transport`] that injects [`Fault`]s into the frames its party sends; without faults it
/// passes everything through. After [`Fault::Leave`] the inner transport is dropped (the other
/// parties see [`Error::PeerLeft`]) and every further call of this party fails with
/// [`Error::Transport`].
#[derive(Debug)]
pub struct Faulty<T> {
    inner: Option<T>,
    n: usize,
    id: PartyId,
    faults: Vec<Fault>,
    sent: usize,
}

impl<T: Transport> Faulty<T> {
    /// Wraps `inner` with the given faults.
    pub fn new(inner: T, faults: Vec<Fault>) -> Self {
        Self {
            n: inner.n(),
            id: inner.id(),
            inner: Some(inner),
            faults,
            sent: 0,
        }
    }

    /// Wraps `inner` without faults.
    pub fn honest(inner: T) -> Self {
        Self::new(inner, Vec::new())
    }

    /// Whether the party has left the network.
    pub fn has_left(&self) -> bool {
        self.inner.is_none()
    }

    /// The number of frames sent so far.
    pub fn frames_sent(&self) -> usize {
        self.sent
    }

    fn left(label: &'static str) -> Error {
        Error::Transport {
            label,
            message: "this party has left the network (injected fault)".to_owned(),
        }
    }

    /// Applies the faults that concern frame number `index`.
    fn tamper(&self, index: usize, mut frame: Frame) -> Frame {
        for fault in &self.faults {
            match *fault {
                Fault::Truncate { frame: f } if f == index => {
                    if frame.payload.pop().is_none() {
                        frame.payload.push(0);
                    }
                }
                Fault::Extend { frame: f } if f == index => frame.payload.push(0),
                Fault::Relabel { frame: f } if f == index => {
                    frame.label = Cow::Borrowed(RELABELLED)
                }
                Fault::NextRound { frame: f } if f == index => {
                    frame.round = frame.round.wrapping_add(1);
                }
                Fault::FlipBit { frame: f, word } if f == index => {
                    let len = frame.payload.len();
                    if len > 0 {
                        frame.payload[word % len] ^= 1;
                    }
                }
                _ => {}
            }
        }
        frame
    }
}

impl<T: Transport> Transport for Faulty<T> {
    fn n(&self) -> usize {
        self.n
    }

    fn id(&self) -> PartyId {
        self.id
    }

    fn send(&mut self, label: &'static str, to: PartyId, frame: Frame) -> Result<()> {
        if self.inner.is_none() {
            return Err(Self::left(label));
        }
        let index = self.sent;
        if self.faults.contains(&Fault::Leave {
            after_frames: index,
        }) {
            self.inner = None;
            return Err(Self::left(label));
        }
        let frame = self.tamper(index, frame);
        self.sent += 1;
        match self.inner.as_mut() {
            Some(inner) => inner.send(label, to, frame),
            None => Err(Self::left(label)),
        }
    }

    fn recv(&mut self, label: &'static str, from: PartyId) -> Result<Frame> {
        match self.inner.as_mut() {
            Some(inner) => inner.recv(label, from),
            None => Err(Self::left(label)),
        }
    }
}
