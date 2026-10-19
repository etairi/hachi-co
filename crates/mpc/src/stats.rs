//! Per-label accounting of the collective calls of one party.

use std::collections::BTreeMap;
use std::ops::AddAssign;

/// What one party's calls under one label cost.
///
/// Bytes count payload words at 8 bytes each; frame headers depend on the transport and are not
/// counted, but frames are (so a header size can be added per frame).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LabelStats {
    /// The number of collective calls.
    pub calls: u64,
    /// The number of communication rounds (one per call, two for an
    /// $\mathcal F\_{\mathsf{AH}}$ call with [`ZeroSharing::Fresh`](crate::ZeroSharing::Fresh)).
    pub rounds: u64,
    /// The number of frames sent.
    pub frames_sent: u64,
    /// The payload bytes sent.
    pub bytes_sent: u64,
    /// The number of frames received.
    pub frames_received: u64,
    /// The payload bytes received.
    pub bytes_received: u64,
}

impl AddAssign for LabelStats {
    fn add_assign(&mut self, other: Self) {
        self.calls += other.calls;
        self.rounds += other.rounds;
        self.frames_sent += other.frames_sent;
        self.bytes_sent += other.bytes_sent;
        self.frames_received += other.frames_received;
        self.bytes_received += other.bytes_received;
    }
}

/// One party's accounting, by label (the set-up under [`LABEL_SETUP`](crate::LABEL_SETUP)).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    by_label: BTreeMap<&'static str, LabelStats>,
}

impl Stats {
    /// The cost of the calls under `label` (all zero if there were none).
    pub fn label(&self, label: &str) -> LabelStats {
        self.by_label.get(label).copied().unwrap_or_default()
    }

    /// The labels and their costs, in label order.
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, LabelStats)> + '_ {
        self.by_label.iter().map(|(l, s)| (*l, *s))
    }

    /// The sum over all labels.
    pub fn total(&self) -> LabelStats {
        let mut total = LabelStats::default();
        for s in self.by_label.values() {
            total += *s;
        }
        total
    }

    pub(crate) fn entry(&mut self, label: &'static str) -> &mut LabelStats {
        self.by_label.entry(label).or_default()
    }
}

impl AddAssign<&Stats> for Stats {
    /// Adds another party's accounting, label by label (for totals over the parties).
    fn add_assign(&mut self, other: &Stats) {
        for (label, s) in other.iter() {
            *self.entry(label) += s;
        }
    }
}
