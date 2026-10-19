//! The semantics of `Loopback` (`docs/m1-spec.md`, `hachi-co-mpc`): n = 1, id = 1, every call is
//! the identity, and malformed calls are errors.

use hachi_co_mpc::{Comm, Error, Loopback};

/// The toy-n1 modulus (`docs/m0-spec.md`, "Toy presets").
const Q: u64 = 8388637;

#[test]
fn constructor_accepts_exactly_two_to_two_pow_63_exclusive() {
    for q in [0, 1, 1 << 63, u64::MAX] {
        assert_eq!(Loopback::new(q), Err(Error::InvalidModulus(q)), "q = {q}");
    }
    for q in [2, 3, Q, (1 << 63) - 1] {
        let c = Loopback::new(q).expect("valid modulus");
        assert_eq!(c.modulus(), q);
    }
}

#[test]
fn one_party_with_index_one() {
    let c = Loopback::new(Q).unwrap();
    assert_eq!((c.n(), c.id(), c.modulus()), (1, 1, Q));
}

#[test]
fn fah_is_the_identity_on_residues_and_reduces_the_rest() {
    let mut c = Loopback::new(Q).unwrap();
    let canonical = vec![0, 1, Q / 2, Q - 1];
    assert_eq!(c.fah("canonical", &canonical).unwrap(), canonical);
    // Unreduced entries stand for their residue class (docs/m0-spec.md, "Conventions").
    let unreduced = [Q, Q + 1, 2 * Q - 1, u64::MAX];
    assert_eq!(
        c.fah("unreduced", &unreduced).unwrap(),
        vec![0, 1, Q - 1, u64::MAX % Q]
    );
    assert_eq!(c.fah("empty", &[]).unwrap(), Vec::<u64>::new());
}

#[test]
fn gather_returns_the_single_vector_at_party_one() {
    let mut c = Loopback::new(Q).unwrap();
    let z = vec![-5, 0, 7, i64::MIN, i64::MAX];
    assert_eq!(c.gather("z", 1, &z).unwrap(), Some(vec![z.clone()]));
    assert_eq!(c.gather("empty", 1, &[]).unwrap(), Some(vec![vec![]]));
}

#[test]
fn gather_to_any_other_party_is_an_error() {
    let mut c = Loopback::new(Q).unwrap();
    for to in [0, 2, 3, usize::MAX] {
        assert_eq!(
            c.gather("z", to, &[1, 2]),
            Err(Error::InvalidParty { party: to, n: 1 }),
            "to = {to}"
        );
    }
}

#[test]
fn any_returns_the_flag() {
    let mut c = Loopback::new(Q).unwrap();
    assert!(c.any("abort", true).unwrap());
    assert!(!c.any("abort", false).unwrap());
}

#[test]
fn exchange_returns_the_single_outgoing_vector_unchanged() {
    let mut c = Loopback::new(Q).unwrap();
    // Unreduced values are delivered as they are: exchange is a transport, not an addition.
    let out = vec![vec![1, Q, u64::MAX]];
    assert_eq!(c.exchange("reshare", out.clone()).unwrap(), out);
    assert_eq!(
        c.exchange("empty", vec![vec![]]).unwrap(),
        vec![Vec::<u64>::new()]
    );
}

#[test]
fn exchange_with_the_wrong_number_of_vectors_is_an_error() {
    let mut c = Loopback::new(Q).unwrap();
    for count in [0, 2, 3] {
        assert_eq!(
            c.exchange("reshare", vec![vec![1]; count]),
            Err(Error::WrongMessageCount {
                label: "reshare",
                expected: 1,
                actual: count
            }),
            "count = {count}"
        );
    }
}

/// A protocol written once against the trait (and usable as a trait object) runs on the loopback:
/// the shares of a sum, an exchange, a gather and an abort vote.
#[test]
fn generic_party_code_runs_on_the_loopback() {
    fn party<C: Comm + ?Sized>(c: &mut C, share: &[u64]) -> hachi_co_mpc::Result<(Vec<u64>, bool)> {
        let n = c.n();
        let incoming = c.exchange("demo/exchange", vec![share.to_vec(); n])?;
        let local: Vec<u64> = (0..share.len())
            .map(|j| incoming.iter().map(|v| v[j]).sum::<u64>())
            .collect();
        let sum = c.fah("demo/fah", &local)?;
        let gathered = c.gather("demo/gather", 1, &[c.id() as i64])?;
        if c.id() == 1 {
            assert_eq!(gathered.map(|g| g.len()), Some(n));
        }
        let abort = c.any("demo/abort", sum.contains(&0))?;
        Ok((sum, abort))
    }
    let mut c = Loopback::new(Q).unwrap();
    assert_eq!(party(&mut c, &[3, Q + 4]).unwrap(), (vec![3, 4], false));
    let dynamic: &mut dyn Comm = &mut c;
    assert_eq!(party(dynamic, &[0, 1]).unwrap(), (vec![0, 1], true));
}

#[test]
fn all_gather_returns_the_single_vector_unchanged() {
    let mut c = Loopback::new(Q).unwrap();
    let v = vec![0, Q, u64::MAX];
    assert_eq!(c.all_gather("all", &v).unwrap(), vec![v]);
}
