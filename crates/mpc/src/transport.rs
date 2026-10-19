//! Point-to-point transports: what an [`Endpoint`](crate::Endpoint) needs from a network.

use crate::{PartyId, Result};
use std::borrow::Cow;

/// One message from one party to another within a collective call.
///
/// The header (`label`, `round`) lets the receiver check that both parties are at the same call;
/// the payload is a vector of 64-bit words (residues, two's-complement `i64` values, flags).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    /// The label of the collective call.
    pub label: Cow<'static, str>,
    /// The sender's round index: 0 for the set-up, then $1, 2, \dots$, one per communication
    /// round (see [`Endpoint`](crate::Endpoint)).
    pub round: u64,
    /// The payload.
    pub payload: Vec<u64>,
}

impl Frame {
    /// A frame with a static label.
    pub fn new(label: &'static str, round: u64, payload: Vec<u64>) -> Self {
        Self {
            label: Cow::Borrowed(label),
            round,
            payload,
        }
    }
}

/// Reliable, ordered point-to-point channels between the $n$ parties of one run, seen from one
/// party.
///
/// Frames from one party to another arrive in the order they were sent. `send` does not wait
/// for the receiver; `recv` waits for the next frame from one given party. The `label` argument
/// is the label of the caller's collective call, used only to build errors.
pub trait Transport {
    /// The number of parties $n \ge 1$.
    fn n(&self) -> usize;

    /// This party's index in $1, \dots, n$.
    fn id(&self) -> PartyId;

    /// Sends `frame` to party `to` $\ne$ [`Transport::id`].
    ///
    /// # Errors
    /// [`Error::InvalidParty`](crate::Error::InvalidParty) for `to` outside $1, \dots, n$ or equal
    /// to this party; [`Error::PeerLeft`](crate::Error::PeerLeft) if `to` has left; another
    /// transport error for other failures.
    fn send(&mut self, label: &'static str, to: PartyId, frame: Frame) -> Result<()>;

    /// Receives the next frame from party `from` $\ne$ [`Transport::id`].
    ///
    /// # Errors
    /// [`Error::InvalidParty`](crate::Error::InvalidParty) for `from` outside $1, \dots, n$ or
    /// equal to this party; [`Error::PeerLeft`](crate::Error::PeerLeft) if `from` has left and no
    /// frame of it is pending; [`Error::Timeout`](crate::Error::Timeout) if nothing arrives in
    /// time; another transport error for other failures.
    fn recv(&mut self, label: &'static str, from: PartyId) -> Result<Frame>;
}
