//! An in-memory $n$-party network implementing [`Comm`] on threads, so that Commit, Eval and
//! the verifier run for $n = 2, 3$ as well (milestone 2 adds the real network; this one only
//! serves the tests). A copy of `crates/conv/tests/common/net.rs`, itself the helper of
//! `crates/commit/tests/common/mod.rs` (the crates share no test code).

use hachi_co_mpc::{Comm, Error as MpcError, PartyId, Result as MpcResult};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// One message of a collective call.
#[derive(Clone, Debug)]
enum Msg {
    Fah(Vec<u64>),
    Gather(PartyId, Vec<i64>),
    Any(bool),
    Exchange(Vec<Vec<u64>>),
}

/// The rendezvous of the current collective call.
struct Round {
    generation: u64,
    arrived: usize,
    inbox: Vec<Option<(&'static str, Msg)>>,
    published: Arc<Vec<(&'static str, Msg)>>,
}

struct Shared {
    n: usize,
    round: Mutex<Round>,
    cv: Condvar,
}

/// One party's end of the in-memory network: every collective call waits until all $n$ parties
/// have made it (with the same label), then returns the result computed from all inputs. A party
/// that waits more than 30 s gets a transport error instead of hanging the test.
pub struct NetParty {
    id: PartyId,
    q: u64,
    shared: Arc<Shared>,
}

/// The $n$ ends of a fresh network with modulus $q$, party $p$ at index $p - 1$.
pub fn network(n: usize, q: u64) -> Vec<NetParty> {
    let shared = Arc::new(Shared {
        n,
        round: Mutex::new(Round {
            generation: 0,
            arrived: 0,
            inbox: vec![None; n],
            published: Arc::new(Vec::new()),
        }),
        cv: Condvar::new(),
    });
    (1..=n)
        .map(|id| NetParty {
            id,
            q,
            shared: Arc::clone(&shared),
        })
        .collect()
}

fn transport(label: &'static str, message: &str) -> MpcError {
    MpcError::Transport {
        label,
        message: message.to_owned(),
    }
}

impl NetParty {
    fn collective(
        &self,
        label: &'static str,
        msg: Msg,
    ) -> MpcResult<Arc<Vec<(&'static str, Msg)>>> {
        let shared = &*self.shared;
        let mut round = shared.round.lock().unwrap();
        let generation = round.generation;
        round.inbox[self.id - 1] = Some((label, msg));
        round.arrived += 1;
        if round.arrived == shared.n {
            let all: Vec<_> = round.inbox.iter_mut().map(|m| m.take().unwrap()).collect();
            round.published = Arc::new(all);
            round.arrived = 0;
            round.generation += 1;
            shared.cv.notify_all();
        } else {
            while round.generation == generation {
                let (guard, timeout) = shared
                    .cv
                    .wait_timeout(round, Duration::from_secs(30))
                    .unwrap();
                round = guard;
                if timeout.timed_out() && round.generation == generation {
                    return Err(transport(label, "timeout: a party did not join the call"));
                }
            }
        }
        let all = Arc::clone(&round.published);
        drop(round);
        if all.iter().any(|(l, _)| *l != label) {
            return Err(transport(label, "the parties made different calls"));
        }
        Ok(all)
    }
}

impl Comm for NetParty {
    fn n(&self) -> usize {
        self.shared.n
    }

    fn id(&self) -> PartyId {
        self.id
    }

    fn modulus(&self) -> u64 {
        self.q
    }

    fn fah(&mut self, label: &'static str, local: &[u64]) -> MpcResult<Vec<u64>> {
        let all = self.collective(label, Msg::Fah(local.to_vec()))?;
        let q = u128::from(self.q);
        let mut sum = vec![0u128; local.len()];
        for (_, msg) in all.iter() {
            let Msg::Fah(v) = msg else {
                return Err(transport(label, "unexpected message"));
            };
            if v.len() != local.len() {
                return Err(MpcError::LengthMismatch {
                    label,
                    expected: local.len(),
                    actual: v.len(),
                });
            }
            for (s, x) in sum.iter_mut().zip(v) {
                *s = (*s + u128::from(*x)) % q;
            }
        }
        Ok(sum.into_iter().map(|x| x as u64).collect())
    }

    fn gather(
        &mut self,
        label: &'static str,
        to: PartyId,
        local: &[i64],
    ) -> MpcResult<Option<Vec<Vec<i64>>>> {
        if to == 0 || to > self.shared.n {
            return Err(MpcError::InvalidParty {
                party: to,
                n: self.shared.n,
            });
        }
        let all = self.collective(label, Msg::Gather(to, local.to_vec()))?;
        if self.id != to {
            return Ok(None);
        }
        all.iter()
            .map(|(_, msg)| match msg {
                Msg::Gather(t, v) if *t == to => Ok(v.clone()),
                _ => Err(transport(label, "unexpected message")),
            })
            .collect::<MpcResult<Vec<_>>>()
            .map(Some)
    }

    fn any(&mut self, label: &'static str, local: bool) -> MpcResult<bool> {
        let all = self.collective(label, Msg::Any(local))?;
        let mut out = false;
        for (_, msg) in all.iter() {
            let Msg::Any(b) = msg else {
                return Err(transport(label, "unexpected message"));
            };
            out |= *b;
        }
        Ok(out)
    }

    fn exchange(
        &mut self,
        label: &'static str,
        outgoing: Vec<Vec<u64>>,
    ) -> MpcResult<Vec<Vec<u64>>> {
        let n = self.shared.n;
        if outgoing.len() != n {
            return Err(MpcError::WrongMessageCount {
                label,
                expected: n,
                actual: outgoing.len(),
            });
        }
        let all = self.collective(label, Msg::Exchange(outgoing))?;
        all.iter()
            .map(|(_, msg)| match msg {
                Msg::Exchange(v) => Ok(v[self.id - 1].clone()),
                _ => Err(transport(label, "unexpected message")),
            })
            .collect()
    }
}

/// Runs `party(p, comm)` for $p = 1, \dots, n$ on threads over a fresh network and returns the
/// results in party order.
pub fn run_parties<T: Send>(
    n: usize,
    q: u64,
    party: impl Fn(PartyId, &mut NetParty) -> T + Sync,
) -> Vec<T> {
    let ends = network(n, q);
    std::thread::scope(|scope| {
        let handles: Vec<_> = ends
            .into_iter()
            .map(|mut comm| {
                let party = &party;
                scope.spawn(move || party(comm.id, &mut comm))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}
