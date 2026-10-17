//! Agreement with the Sage implementation of the cascade in `parameters.sage` (Hachi-Co section,
//! function `hco_fixture`), which writes `tests/fixtures/sage_cascade.json`: every derived value
//! and every report flag, on the toy presets, a seeded grid of choices (every tau rule, both
//! challenge sets, fixed moduli at the boundaries of the q conditions), the secure grid at its
//! smallest admissible q, and moduli at the top of `u64`; plus the exact boundary of
//! $M \le e^{1/n}$ and primes $\equiv 5 \pmod 8$ above powers of two.

use hachi_co_params::rules::{next_prime_5_mod_8, rejection_constant_within};
use hachi_co_params::{ConstraintId, Params, QChoice, derive};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Entry {
    params: Params,
    report: BTreeMap<String, bool>,
}

#[derive(Deserialize)]
struct Boundary {
    n: u64,
    big_n: u128,
    beta1: u128,
    least_gamma: u128,
}

#[derive(Deserialize)]
struct Prime {
    above: u128,
    p: u64,
}

#[derive(Deserialize)]
struct Fixture {
    entries: Vec<Entry>,
    rejection_boundaries: Vec<Boundary>,
    primes_5_mod_8: Vec<Prime>,
}

fn fixture() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sage_cascade.json"
    );
    let text = std::fs::read_to_string(path).expect("run parameters.sage to create the fixture");
    serde_json::from_str(&text).expect("the fixture parses")
}

/// Exact equality of every field, except the two floats (relative 1e-12).
pub fn assert_same_params(expected: &Params, got: &Params, what: &str) {
    let rel = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs());
    assert!(
        rel(expected.rejection_m, got.rejection_m),
        "{what}: M {} vs {}",
        expected.rejection_m,
        got.rejection_m
    );
    assert!(
        rel(expected.log2_rejection_m, got.log2_rejection_m),
        "{what}: log2 M {} vs {}",
        expected.log2_rejection_m,
        got.log2_rejection_m
    );
    let mut expected = expected.clone();
    expected.rejection_m = got.rejection_m;
    expected.log2_rejection_m = got.log2_rejection_m;
    assert_eq!(&expected, got, "{what}");
}

#[test]
fn rust_and_sage_agree_on_every_entry() {
    let fx = fixture();
    assert!(fx.entries.len() >= 120, "{} entries", fx.entries.len());
    let mut flags_failed = BTreeMap::<String, usize>::new();
    for (i, e) in fx.entries.iter().enumerate() {
        let what = format!("entry {i}: {:?}", e.params.choices);
        let (p, report) = derive(&e.params.choices).unwrap_or_else(|err| panic!("{what}: {err}"));
        assert_same_params(&e.params, &p, &what);
        assert_eq!(e.report.len(), ConstraintId::ALL.len(), "{what}");
        for id in ConstraintId::ALL {
            let key = format!("{id:?}");
            assert_eq!(report.holds(id), e.report[&key], "{what}: {key}");
            if !e.report[&key] {
                *flags_failed.entry(key).or_default() += 1;
            }
        }
    }
    // The grid exercises a violation of every condition except M <= e^(1/n), which the
    // cascade cannot violate.
    for id in ConstraintId::ALL {
        let key = format!("{id:?}");
        if id == ConstraintId::RejectionConstant {
            assert!(!flags_failed.contains_key(&key));
        } else {
            assert!(
                flags_failed.get(&key).copied().unwrap_or(0) > 0,
                "no entry violates {key}"
            );
        }
    }
}

/// `QChoice::Smallest` scans the delta-bands; `docs/m0-spec.md` describes the fixed-point
/// iteration from q = 2^20. Both give the same modulus on every fixture entry whose answer is at
/// least 2^20.
#[test]
fn band_scan_equals_the_fixed_point_iteration() {
    let mut compared = 0;
    for e in fixture()
        .entries
        .iter()
        .filter(|e| e.params.choices.q == QChoice::Smallest)
    {
        if e.params.q < 1 << 20 {
            continue;
        }
        let mut q: u64 = 1 << 20;
        let iterated = loop {
            let (p, _) = derive(&hachi_co_params::Choices {
                q: QChoice::Fixed(q),
                ..e.params.choices
            })
            .unwrap();
            let bound = p.binding_guard.max(p.range_guard).max(2 * p.beta + 1);
            let next = next_prime_5_mod_8(bound).unwrap();
            let delta_next =
                hachi_co_params::rules::ceil_log(u128::from(p.choices.b), u128::from(next))
                    .unwrap();
            if delta_next == p.delta {
                break next;
            }
            q = next;
        };
        assert_eq!(iterated, e.params.q, "{:?}", e.params.choices);
        compared += 1;
    }
    assert!(compared >= 50, "{compared}");
}

#[test]
fn rejection_constant_boundaries_match_sage() {
    let fx = fixture();
    assert!(!fx.rejection_boundaries.is_empty());
    for b in fx.rejection_boundaries {
        assert!(
            rejection_constant_within(b.n, b.big_n, b.least_gamma, b.beta1).unwrap(),
            "holds at gamma = {}",
            b.least_gamma
        );
        assert!(
            !rejection_constant_within(b.n, b.big_n, b.least_gamma - 1, b.beta1).unwrap(),
            "fails at gamma = {} - 1",
            b.least_gamma
        );
    }
}

#[test]
fn primes_five_mod_eight_match_sage() {
    let fx = fixture();
    assert_eq!(fx.primes_5_mod_8.len(), 44);
    for p in fx.primes_5_mod_8 {
        assert_eq!(next_prime_5_mod_8(p.above), Some(p.p), "above {}", p.above);
    }
}
