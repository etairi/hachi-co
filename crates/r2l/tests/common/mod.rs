//! Shared helpers of the integration tests: the inputs of the ring-to-linear step built from a
//! fixture's `commit` and `conv` sections, and an in-memory n-party network (one thread per
//! party) implementing `Comm`, so that the driver can run for n = 2, 3.
#![allow(dead_code)]

use hachi_co_core::{BlockState, Context, Fixture, PartyState, Poly, PublicVectors};
use hachi_co_mpc::{Comm, Error as MpcError, PartyId, Result as MpcResult};
use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

pub const K: usize = 4;

/// Party p's state after Commit, as recorded by the oracle (`commit/f_prime`, `commit/blocks`,
/// `commit/f_shares`).
pub fn state(fx: &Fixture, ctx: &Context<K>, p: usize) -> PartyState<K> {
    let l = &ctx.layout;
    let f_prime = fx.polys("/commit/f_prime").unwrap();
    let recorded = fx.get("/commit/blocks").unwrap().as_array().unwrap().len();
    let blocks = l
        .blocks_of(p)
        .map(|i| {
            let j = (0..recorded)
                .find(|j| fx.usize(&format!("/commit/blocks/{j}/index")).unwrap() == i)
                .expect("every owned block is recorded");
            assert_eq!(fx.usize(&format!("/commit/blocks/{j}/owner")).unwrap(), p);
            let key = |k: &str| fx.polys(&format!("/commit/blocks/{j}/{k}")).unwrap();
            BlockState {
                index: i,
                f_prime: f_prime[i * l.block_len..(i + 1) * l.block_len].to_vec(),
                s: key("s"),
                t: key("t"),
                t_hat: key("t_hat"),
            }
        })
        .collect();
    let st = PartyState {
        party: p,
        share: fx.exts(&format!("/commit/f_shares/{}", p - 1)).unwrap(),
        blocks,
    };
    st.validate(l).unwrap();
    st
}

/// The public vectors of Pi_conv as recorded (`conv/x_prime`, `b`, `a`, `d`, `v_point`), with b
/// and a embedded in R_q.
pub fn public_vectors(fx: &Fixture, ctx: &Context<K>) -> PublicVectors<K> {
    let embed = |pointer: &str| -> Vec<Poly> {
        fx.exts::<K>(pointer)
            .unwrap()
            .iter()
            .map(|x| ctx.sub.embed(x))
            .collect()
    };
    let pv = PublicVectors {
        x_prime: fx.exts("/conv/x_prime").unwrap(),
        b: fx.exts("/conv/b").unwrap(),
        a: fx.exts("/conv/a").unwrap(),
        d: fx.exts("/conv/d").unwrap(),
        v_point: fx.poly("/conv/v_point").unwrap(),
        b_ring: embed("/conv/b"),
        a_ring: embed("/conv/a"),
    };
    pv.validate(&ctx.layout).unwrap();
    pv
}

/// Party p's share <u>_p of Pi_conv (`conv/u_shares`).
pub fn u_share(fx: &Fixture, p: usize) -> Poly {
    fx.poly(&format!("/conv/u_shares/{}", p - 1)).unwrap()
}

/// The commitment u (`commit/u`).
pub fn u_com(fx: &Fixture) -> Vec<Poly> {
    fx.polys("/commit/u").unwrap()
}

/// The number of recorded attempts.
pub fn attempts(fx: &Fixture) -> usize {
    fx.get("/ring_to_linear/attempts")
        .unwrap()
        .as_array()
        .unwrap()
        .len()
}

/// The fixture, its context and every party's state.
pub fn load(name: &str) -> (Fixture, Context<K>, PublicVectors<K>, Vec<PartyState<K>>) {
    let fx = Fixture::load(name).unwrap();
    let ctx = fx.context::<K>().unwrap();
    let pv = public_vectors(&fx, &ctx);
    let states = (1..=fx.n()).map(|p| state(&fx, &ctx, p)).collect();
    (fx, ctx, pv, states)
}

// ---------------------------------------------------------------- the threaded network ----

#[derive(Clone, Debug)]
enum Msg {
    Fah(Vec<u64>),
    Gather(PartyId, Vec<i64>),
    Any(bool),
    Exchange(Vec<Vec<u64>>),
}

/// Every party's labelled input of one round, in party order.
type Inputs = Arc<Vec<(&'static str, Msg)>>;

#[derive(Default)]
struct HubState {
    /// The round that is collecting inputs.
    round: u64,
    /// The inputs of that round, with their labels.
    pending: Vec<Option<(&'static str, Msg)>>,
    deposited: usize,
    /// Completed rounds that some party has not read yet: (inputs, readers left).
    done: HashMap<u64, (Inputs, usize)>,
}

/// An in-memory network of n parties for tests: every collective call is a synchronous round in
/// which each party deposits its input and reads everybody's. A party that waits more than the
/// timeout gets a transport error instead of hanging the test.
pub struct Hub {
    n: usize,
    q: u64,
    state: Mutex<HubState>,
    cv: Condvar,
    timeout: Duration,
}

impl Hub {
    pub fn new(n: usize, q: u64) -> Arc<Self> {
        Arc::new(Self {
            n,
            q,
            state: Mutex::new(HubState {
                pending: vec![None; n],
                ..HubState::default()
            }),
            cv: Condvar::new(),
            timeout: Duration::from_secs(20),
        })
    }

    /// One `Comm` endpoint per party.
    pub fn parties(self: &Arc<Self>) -> Vec<ThreadComm> {
        (1..=self.n)
            .map(|id| ThreadComm {
                id,
                hub: Arc::clone(self),
                calls: 0,
            })
            .collect()
    }
}

/// Party `id`'s endpoint of a [`Hub`].
pub struct ThreadComm {
    id: PartyId,
    hub: Arc<Hub>,
    calls: u64,
}

impl ThreadComm {
    fn round(&mut self, label: &'static str, msg: Msg) -> MpcResult<Inputs> {
        let k = self.calls;
        self.calls += 1;
        let hub = &self.hub;
        let timeout = |label| MpcError::Transport {
            label,
            message: "timed out waiting for the other parties".into(),
        };
        let mut st = hub.state.lock().unwrap();
        // Wait until round k collects inputs (the previous round is complete).
        while st.round != k {
            let (guard, res) = hub.cv.wait_timeout(st, hub.timeout).unwrap();
            st = guard;
            if res.timed_out() && st.round != k {
                return Err(timeout(label));
            }
        }
        st.pending[self.id - 1] = Some((label, msg));
        st.deposited += 1;
        if st.deposited == hub.n {
            let inputs: Vec<_> = st.pending.iter_mut().map(|m| m.take().unwrap()).collect();
            st.done.insert(k, (Arc::new(inputs), hub.n));
            st.deposited = 0;
            st.round += 1;
            hub.cv.notify_all();
        }
        while !st.done.contains_key(&k) {
            let (guard, res) = hub.cv.wait_timeout(st, hub.timeout).unwrap();
            st = guard;
            if res.timed_out() && !st.done.contains_key(&k) {
                return Err(timeout(label));
            }
        }
        let entry = st.done.get_mut(&k).unwrap();
        let inputs = Arc::clone(&entry.0);
        entry.1 -= 1;
        if entry.1 == 0 {
            st.done.remove(&k);
        }
        drop(st);
        if let Some((other, _)) = inputs.iter().find(|(l, _)| *l != label) {
            return Err(MpcError::Transport {
                label,
                message: format!("desynchronised: another party called {other}"),
            });
        }
        Ok(inputs)
    }
}

impl Comm for ThreadComm {
    fn n(&self) -> usize {
        self.hub.n
    }

    fn id(&self) -> PartyId {
        self.id
    }

    fn modulus(&self) -> u64 {
        self.hub.q
    }

    fn fah(&mut self, label: &'static str, local: &[u64]) -> MpcResult<Vec<u64>> {
        let q = self.hub.q;
        let inputs = self.round(label, Msg::Fah(local.to_vec()))?;
        let mut sum = vec![0u64; local.len()];
        for (_, msg) in inputs.iter() {
            let Msg::Fah(v) = msg else {
                return Err(MpcError::Transport {
                    label,
                    message: "mixed call kinds".into(),
                });
            };
            if v.len() != local.len() {
                return Err(MpcError::LengthMismatch {
                    label,
                    expected: local.len(),
                    actual: v.len(),
                });
            }
            for (s, x) in sum.iter_mut().zip(v) {
                *s = ((u128::from(*s) + u128::from(*x % q)) % u128::from(q)) as u64;
            }
        }
        Ok(sum)
    }

    fn gather(
        &mut self,
        label: &'static str,
        to: PartyId,
        local: &[i64],
    ) -> MpcResult<Option<Vec<Vec<i64>>>> {
        if to == 0 || to > self.hub.n {
            return Err(MpcError::InvalidParty {
                party: to,
                n: self.hub.n,
            });
        }
        let inputs = self.round(label, Msg::Gather(to, local.to_vec()))?;
        let mut all = Vec::with_capacity(inputs.len());
        for (_, msg) in inputs.iter() {
            match msg {
                Msg::Gather(t, v) if *t == to => all.push(v.clone()),
                _ => {
                    return Err(MpcError::Transport {
                        label,
                        message: "mixed call kinds or destinations".into(),
                    });
                }
            }
        }
        Ok((self.id == to).then_some(all))
    }

    fn any(&mut self, label: &'static str, local: bool) -> MpcResult<bool> {
        let inputs = self.round(label, Msg::Any(local))?;
        let mut out = false;
        for (_, msg) in inputs.iter() {
            let Msg::Any(b) = msg else {
                return Err(MpcError::Transport {
                    label,
                    message: "mixed call kinds".into(),
                });
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
        let n = self.hub.n;
        if outgoing.len() != n {
            return Err(MpcError::WrongMessageCount {
                label,
                expected: n,
                actual: outgoing.len(),
            });
        }
        let inputs = self.round(label, Msg::Exchange(outgoing))?;
        inputs
            .iter()
            .map(|(_, msg)| match msg {
                Msg::Exchange(out) => Ok(out[self.id - 1].clone()),
                _ => Err(MpcError::Transport {
                    label,
                    message: "mixed call kinds".into(),
                }),
            })
            .collect()
    }
}
