//! Malformed inputs and messages give errors, never panics: received shares of the wrong length or
//! with residues $\ge q$ (also through the driver, with a tampering network), networks that do not
//! match the context, invalid parties, block indices and duplicate blocks.
mod common;

use common::{K, load};
use hachi_co_commit::{
    Error, block_state, commit, decode_received, fah_b_input, party_state, pconv_blocks, reshare,
    sum_shares, t_hat_slice,
};
use hachi_co_core::wire::flatten_exts;
use hachi_co_core::{Fixture, ReplayRandomness};
use hachi_co_math::Ext;
use hachi_co_mpc::{Comm, Loopback, PartyId, Result as MpcResult};

fn f_init(fx: &Fixture, p: usize) -> Vec<Ext<K>> {
    fx.exts::<K>(&format!("/input/f_init_shares/{}", p - 1))
        .unwrap()
}

#[test]
fn received_shares_are_checked() {
    let (fx, ctx) = load("toy-n3");
    let q = ctx.q();
    let len = 1usize << ctx.layout.ell;
    let good: Vec<Vec<u64>> = (1..=3)
        .map(|i| {
            flatten_exts(
                &fx.ext_rows::<K>(&format!("/commit/reshare/{}", i - 1))
                    .unwrap()[0],
            )
        })
        .collect();
    assert!(decode_received::<K>(&ctx, &good).is_ok());
    // A residue >= q from party 2 is named.
    let mut bad = good.clone();
    bad[1][7] = q;
    assert_eq!(
        decode_received::<K>(&ctx, &bad),
        Err(Error::NonCanonical {
            from: 2,
            what: "a reshared share",
            value: q,
            q
        })
    );
    bad[1][7] = u64::MAX;
    assert!(matches!(
        decode_received::<K>(&ctx, &bad),
        Err(Error::NonCanonical { from: 2, .. })
    ));
    // Wrong lengths and counts.
    let mut short = good.clone();
    short[2].pop();
    assert!(matches!(
        decode_received::<K>(&ctx, &short),
        Err(Error::Shape { .. })
    ));
    assert!(matches!(
        decode_received::<K>(&ctx, &good[..2]),
        Err(Error::Shape { .. })
    ));
    let received = decode_received::<K>(&ctx, &good).unwrap();
    assert!(matches!(
        sum_shares(&ctx, &received[..2]),
        Err(Error::Shape { .. })
    ));
    let mut ragged = received.clone();
    ragged[0].truncate(len - 1);
    assert!(matches!(
        sum_shares(&ctx, &ragged),
        Err(Error::Shape { .. })
    ));
}

#[test]
fn party_local_functions_refuse_bad_arguments() {
    let (fx, ctx) = load("toy-n3");
    let l = &ctx.layout;
    let share = fx.exts::<K>("/commit/f_shares/0").unwrap();
    for p in [0, 4, usize::MAX] {
        assert_eq!(
            pconv_blocks(&ctx, p, &share),
            Err(Error::InvalidParty { party: p, n: 3 })
        );
    }
    assert!(matches!(
        pconv_blocks(&ctx, 1, &share[1..]),
        Err(Error::Shape { .. })
    ));
    let mut rng = ReplayRandomness::new(&fx, 1).unwrap();
    assert!(matches!(
        reshare(&ctx, &mut rng, &share[1..]),
        Err(Error::Shape { .. })
    ));
    let state = party_state(&ctx, 1, share).unwrap();
    let f_prime = state.blocks[0].f_prime.clone();
    assert!(matches!(
        block_state(&ctx, l.n_blocks, f_prime.clone()),
        Err(Error::Shape { .. })
    ));
    assert!(matches!(
        block_state(&ctx, 0, f_prime[1..].to_vec()),
        Err(Error::Shape { .. })
    ));
    let mut bad_poly = f_prime.clone();
    bad_poly[0].0.pop();
    assert!(matches!(
        block_state(&ctx, 0, bad_poly),
        Err(Error::Math(_))
    ));
    // Duplicate blocks in a slice of t-hat or an F_AH(B) input.
    let mut twice = state.blocks.clone();
    twice.push(state.blocks[0].clone());
    assert_eq!(
        t_hat_slice(&ctx, &twice),
        Err(Error::DuplicateBlock { index: 0 })
    );
    assert_eq!(
        fah_b_input(&ctx, &twice),
        Err(Error::DuplicateBlock { index: 0 })
    );
    let mut wrong = state.blocks.clone();
    wrong[1].t_hat.pop();
    assert!(matches!(fah_b_input(&ctx, &wrong), Err(Error::Core(_))));
}

/// A loopback whose exchange adds $q$ to one received residue.
struct Tampering(Loopback);

impl Comm for Tampering {
    fn n(&self) -> usize {
        self.0.n()
    }
    fn id(&self) -> PartyId {
        self.0.id()
    }
    fn modulus(&self) -> u64 {
        self.0.modulus()
    }
    fn fah(&mut self, label: &'static str, local: &[u64]) -> MpcResult<Vec<u64>> {
        self.0.fah(label, local)
    }
    fn gather(
        &mut self,
        label: &'static str,
        to: PartyId,
        local: &[i64],
    ) -> MpcResult<Option<Vec<Vec<i64>>>> {
        self.0.gather(label, to, local)
    }
    fn any(&mut self, label: &'static str, local: bool) -> MpcResult<bool> {
        self.0.any(label, local)
    }
    fn exchange(
        &mut self,
        label: &'static str,
        outgoing: Vec<Vec<u64>>,
    ) -> MpcResult<Vec<Vec<u64>>> {
        let mut incoming = self.0.exchange(label, outgoing)?;
        incoming[0][3] += self.0.modulus();
        Ok(incoming)
    }
    fn all_gather(&mut self, label: &'static str, local: &[u64]) -> MpcResult<Vec<Vec<u64>>> {
        self.0.all_gather(label, local)
    }
}

#[test]
fn the_driver_refuses_malformed_messages_and_mismatched_networks() {
    let (fx, ctx) = load("toy-n1");
    let q = ctx.q();
    let mut rng = ReplayRandomness::new(&fx, 1).unwrap();
    let mut comm = Tampering(Loopback::new(q).unwrap());
    assert!(matches!(
        commit(&ctx, &mut comm, &mut rng, &f_init(&fx, 1)),
        Err(Error::NonCanonical { from: 1, .. })
    ));
    // Another modulus, another number of parties, a short input.
    let mut comm = Loopback::new(33554501).unwrap();
    assert!(matches!(
        commit(&ctx, &mut comm, &mut rng, &f_init(&fx, 1)),
        Err(Error::Network { what: "q", .. })
    ));
    let (fx2, ctx2) = load("toy-n2");
    let mut comm = Loopback::new(ctx2.q()).unwrap();
    let mut rng2 = ReplayRandomness::new(&fx2, 1).unwrap();
    assert!(matches!(
        commit(&ctx2, &mut comm, &mut rng2, &f_init(&fx2, 1)),
        Err(Error::Network { what: "n", .. })
    ));
    let mut comm = Loopback::new(q).unwrap();
    assert!(matches!(
        commit(&ctx, &mut comm, &mut rng, &f_init(&fx, 1)[1..]),
        Err(Error::Shape { .. })
    ));
}
