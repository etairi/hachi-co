//! `SumcheckCoins::validate`: the lengths of tau_0 and tau_1, and canonical residues in every
//! coordinate of every coin (the gap "canonicity" left by the first version), at the boundary
//! q - 1 (accepted) and q (refused); honest Fiat-Shamir coins pass.

use hachi_co_core::{Challenger, Error, Ext, Fqk, FsChallenger, Layout, SumcheckCoins};
use hachi_co_params::{derive, presets::toy_choices};

const K: usize = 4;

fn layout_and_q(n: u64) -> (Layout, u64) {
    let params = derive(&toy_choices(n)).expect("toy presets derive").0;
    (Layout::new(&params).unwrap(), params.q)
}

fn coins(l: &Layout, value: u64) -> SumcheckCoins<K> {
    SumcheckCoins {
        alpha: Ext([value; K]),
        rho: Ext([value; K]),
        tau0: vec![Ext([value; K]); l.ell_pp + l.log_n],
        tau1: vec![Ext([value; K]); l.log_nu],
    }
}

#[test]
fn canonical_coins_of_the_right_lengths_pass() {
    for n in [1, 2, 3] {
        let (l, q) = layout_and_q(n);
        for value in [0, 1, q - 1] {
            assert_eq!(
                coins(&l, value).validate(&l, q),
                Ok(()),
                "n {n}, value {value}"
            );
        }
    }
}

#[test]
fn a_residue_equal_to_q_is_refused_in_every_coin() {
    let (l, q) = layout_and_q(2);
    type Edit = fn(&mut SumcheckCoins<K>, u64);
    let edits: [(&str, Edit); 4] = [
        ("alpha", |c, q| c.alpha.0[K - 1] = q),
        ("rho", |c, q| c.rho.0[0] = q),
        ("tau_0", |c, q| {
            let last = c.tau0.len() - 1;
            c.tau0[last].0[1] = q;
        }),
        ("tau_1", |c, q| c.tau1[0].0[2] = q + 5),
    ];
    for (what, edit) in edits {
        let mut c = coins(&l, q - 1);
        edit(&mut c, q);
        match c.validate(&l, q) {
            Err(Error::NonCanonical {
                what: w,
                value,
                q: qq,
            }) => {
                assert_eq!((w, qq), (what, q));
                assert!(value >= q);
            }
            other => panic!("{what}: {other:?}"),
        }
    }
}

#[test]
fn wrong_lengths_are_shape_errors() {
    let (l, q) = layout_and_q(3);
    let mut c = coins(&l, 0);
    c.tau0.pop();
    assert!(matches!(
        c.validate(&l, q),
        Err(Error::Shape { what: "tau_0", .. })
    ));
    let mut c = coins(&l, 0);
    c.tau1.push(Ext([0; K]));
    assert!(matches!(
        c.validate(&l, q),
        Err(Error::Shape { what: "tau_1", .. })
    ));
}

#[test]
fn fiat_shamir_coins_are_canonical() {
    for n in [1, 2, 3] {
        let params = derive(&toy_choices(n)).unwrap().0;
        let l = Layout::new(&params).unwrap();
        let field = Fqk::<K>::new(params.q).unwrap();
        let mut ch = FsChallenger::new(b"coins test");
        let c = ch
            .sumcheck_coins(&field, l.ell_pp, l.log_n, l.log_nu)
            .unwrap();
        assert_eq!(c.validate(&l, params.q), Ok(()), "n {n}");
    }
}
