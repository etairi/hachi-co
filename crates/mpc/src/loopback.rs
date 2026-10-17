//! The one-party network of milestone 1.

use crate::{Comm, Error, PartyId, Result};

/// The network of a single party ($n = 1$, [`Comm::id`] $= 1$): every collective call is the
/// identity (plan `docs/hachi-co-plan.md`, milestone M1: "with n = 1 on a loopback network, where
/// every F_AH call is the identity").
///
/// - [`Comm::fah`] returns its input reduced modulo $q$ (the sum over one party);
/// - [`Comm::gather`] at party 1 returns `Some(vec![local])`;
/// - [`Comm::any`] returns its input;
/// - [`Comm::exchange`] returns its single outgoing vector unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Loopback {
    q: u64,
}

impl Loopback {
    /// The loopback for the modulus $q$ of [`Comm::fah`].
    ///
    /// # Errors
    /// [`Error::InvalidModulus`] unless $2 \le q \lt 2^{63}$.
    pub fn new(q: u64) -> Result<Self> {
        if !(2..1 << 63).contains(&q) {
            return Err(Error::InvalidModulus(q));
        }
        Ok(Self { q })
    }
}

impl Comm for Loopback {
    /// Always 1.
    fn n(&self) -> usize {
        1
    }

    /// Always 1.
    fn id(&self) -> PartyId {
        1
    }

    fn modulus(&self) -> u64 {
        self.q
    }

    /// The sum over the single party: `local` reduced modulo $q$.
    fn fah(&mut self, _label: &'static str, local: &[u64]) -> Result<Vec<u64>> {
        Ok(local.iter().map(|x| x % self.q).collect())
    }

    /// `Some(vec![local])` for `to` $= 1$.
    fn gather(
        &mut self,
        _label: &'static str,
        to: PartyId,
        local: &[i64],
    ) -> Result<Option<Vec<Vec<i64>>>> {
        if to != 1 {
            return Err(Error::InvalidParty { party: to, n: 1 });
        }
        Ok(Some(vec![local.to_vec()]))
    }

    /// The flag itself.
    fn any(&mut self, _label: &'static str, local: bool) -> Result<bool> {
        Ok(local)
    }

    /// The single outgoing vector, unchanged.
    fn exchange(&mut self, label: &'static str, outgoing: Vec<Vec<u64>>) -> Result<Vec<Vec<u64>>> {
        if outgoing.len() != 1 {
            return Err(Error::WrongMessageCount {
                label,
                expected: 1,
                actual: outgoing.len(),
            });
        }
        Ok(outgoing)
    }
}
