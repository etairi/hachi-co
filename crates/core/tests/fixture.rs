//! The fixture loader: parameters cross-checked field by field, the typed decoders and their
//! failure cases, the digest convention and the JSON-pointer escaping; and, for every fixture
//! ($n = 1, 2, 3, 4, 8$), the per-party shapes and the oracle's recorded outcomes (relations,
//! verifier, negative tests, seed selection).
#![cfg(feature = "fixtures")]

use hachi_co_core::{Error, Fixture, Layout, escape_key, sha256_residues};
use hachi_co_params::presets::TOY_NS;
use serde_json::{Value, json};

fn fixture_err<T: core::fmt::Debug>(r: hachi_co_core::Result<T>) -> String {
    match r {
        Err(Error::Fixture(msg)) => msg,
        other => panic!("expected a fixture error, got {other:?}"),
    }
}

#[test]
fn every_fixture_loads_and_its_parameters_agree_with_rust() {
    assert_eq!(Fixture::TOY.len(), TOY_NS.len());
    for (name, n) in Fixture::TOY.into_iter().zip(TOY_NS) {
        assert_eq!(name, format!("toy-n{n}"));
        let fx = Fixture::load(name).unwrap();
        assert_eq!(fx.name(), name);
        assert_eq!(fx.n() as u64, n);
        assert_eq!((fx.d(), fx.k()), (16, 4));
        let params = fx.params().unwrap();
        assert_eq!(params.choices.n, n);
        assert_eq!(params.q, fx.q());
        assert_eq!(fx.str("/format").unwrap(), Fixture::FORMAT);
    }
}

/// The length of the list at `pointer`.
fn len(fx: &Fixture, pointer: &str) -> usize {
    fx.get(pointer)
        .unwrap()
        .as_array()
        .unwrap_or_else(|| panic!("{}: {pointer} is not a list", fx.name()))
        .len()
}

/// Every section that holds one entry per party has exactly $n$ of them, in party order, and the
/// owned blocks, the partition and the unheld positions follow the layout of $n$ parties.
#[test]
fn every_fixture_has_one_entry_per_party() {
    for name in Fixture::TOY {
        let fx = Fixture::load(name).unwrap();
        let n = fx.n();
        let l = Layout::new(&fx.params().unwrap()).unwrap();
        for pointer in [
            "/input/f_init_shares",
            "/commit/reshare",
            "/commit/f_shares",
            "/conv/u_shares",
            "/ring_to_linear/z_i",
            "/sumcheck/quotient_shares",
            "/sumcheck/blinders",
            "/sumcheck/G0",
            "/sumcheck/G1",
            "/sumcheck/final/w",
            "/sumcheck/final/g0",
            "/sumcheck/final/g1",
        ] {
            assert_eq!(len(&fx, pointer), n, "{name} {pointer}");
        }
        for i in 0..n {
            assert_eq!(len(&fx, &format!("/commit/reshare/{i}")), n, "{name}");
        }
        let attempts = len(&fx, "/ring_to_linear/attempts");
        for a in 0..attempts {
            let pointer = format!("/ring_to_linear/attempts/{a}/provers");
            assert_eq!(len(&fx, &pointer), n, "{name}");
            for p in 1..=n {
                assert_eq!(fx.usize(&format!("{pointer}/{}/prover", p - 1)).unwrap(), p);
            }
        }
        for r in 0..l.ell_pp {
            for key in ["F_AH_inputs_H0", "F_AH_inputs_H1"] {
                let pointer = format!("/sumcheck/rounds/{r}/{key}");
                assert_eq!(len(&fx, &pointer), n, "{name}");
            }
        }
        let partition = fx.get("/ring_to_linear/partition").unwrap();
        let parties: Vec<&String> = partition.as_object().unwrap().keys().collect();
        let expected: Vec<String> = (1..=n).map(|p| p.to_string()).collect();
        assert_eq!(parties, expected.iter().collect::<Vec<_>>(), "{name}");
        let digests = fx.get("/sumcheck/digests").unwrap().as_object().unwrap();
        for p in 1..=n + 1 {
            assert_eq!(
                digests.contains_key(&format!("w_table_p{p}")),
                p <= n,
                "{name}"
            );
        }
        assert_eq!(len(&fx, "/commit/block_owner"), l.n_blocks, "{name}");
        assert_eq!(len(&fx, "/commit/blocks"), n * l.blocks_per_party, "{name}");
        assert_eq!(l.blocks_per_party << l.log_n, l.n_blocks, "{name}");
        assert_eq!(
            len(&fx, "/ring_to_linear/unheld") == 0,
            n.is_power_of_two(),
            "{name}"
        );
    }
}

/// The oracle recorded a passing run: every relation holds, the verifier ran its
/// $1 + 2\ell'' + 3n + 2$ checks (36, 39, 44, 47 and 61 for $n = 1, 2, 3, 4, 8$) and accepted,
/// every negative test passed, and the seed record names the fixture's seeds
/// (`ref/README.md`, choice 19: the first seed index with an aborted attempt).
#[test]
fn every_fixture_records_a_passing_run() {
    let mut counts = Vec::new();
    for name in Fixture::TOY {
        let fx = Fixture::load(name).unwrap();
        let n = fx.n();
        let l = Layout::new(&fx.params().unwrap()).unwrap();
        let bools = |pointer: &str| -> Vec<(String, bool)> {
            let section = fx.get(pointer).unwrap().as_object().unwrap();
            section
                .iter()
                .map(|(k, v)| (k.clone(), v.as_bool().expect("a boolean")))
                .collect()
        };
        let relations = bools("/relations");
        assert!(relations.len() >= 29, "{name}");
        assert!(relations.iter().all(|(_, ok)| *ok), "{name}: {relations:?}");
        let checks = bools("/verifier/checks");
        assert_eq!(checks.len(), 1 + 2 * l.ell_pp + 3 * n + 2, "{name}");
        assert!(checks.iter().all(|(_, ok)| *ok), "{name}: {checks:?}");
        assert!(fx.bool("/verifier/accept").unwrap(), "{name}");
        counts.push(checks.len());
        for p in 1..=n {
            for kind in ["g0", "g1", "w"] {
                let key = format!("pcs_open_{kind}_p{p}");
                assert!(checks.iter().any(|(k, _)| *k == key), "{name}: {key}");
            }
        }
        // Negative tests: 4 per party and 16 others on the transcript; 6 witnesses (3 for n = 1,
        // which has no second party to shift shares to); 3 openings.
        let witnesses = if n >= 2 { 6 } else { 3 };
        for (group, count) in [
            ("transcript", 16 + 4 * n),
            ("witness", witnesses),
            ("open", 3),
        ] {
            let tests = fx.get(&format!("/negative_tests/{group}")).unwrap();
            let tests = tests.as_object().unwrap();
            assert_eq!(tests.len(), count, "{name} {group}");
            for (test, info) in tests {
                assert_eq!(info["passed"], json!(true), "{name} {group}: {test}");
            }
        }
        for pointer in [
            "/negative_tests/rej_boundary_toy/accepted_set_smaller",
            "/negative_tests/rej_boundary_toy/within_bound_full",
            "/negative_tests/abort_rate/within_3_sigma",
        ] {
            assert!(fx.bool(pointer).unwrap(), "{name} {pointer}");
        }
        // Seeds: the instance is the last run tried by the selection, which has an aborted attempt.
        let instance = fx.str("/seeds/instance").unwrap();
        assert!(
            instance.starts_with(&format!("hachi-co-ref/{name}/")),
            "{instance}"
        );
        assert_eq!(
            fx.str("/seeds/setup").unwrap(),
            format!("hachi-co-ref/setup/{name}")
        );
        let tried = len(&fx, "/seeds/selection/tried");
        assert_eq!(
            fx.str(&format!("/seeds/selection/tried/{}/seed", tried - 1))
                .unwrap(),
            instance
        );
        for t in 0..tried {
            let pointer = format!("/seeds/selection/tried/{t}/all_checks_pass");
            assert!(fx.bool(&pointer).unwrap(), "{name} {pointer}");
        }
        let accepted = fx.usize("/ring_to_linear/accepted_attempt").unwrap();
        assert!(accepted >= 1, "{name}: the fixture covers the retry path");
        assert_eq!(len(&fx, "/ring_to_linear/attempts"), accepted + 1, "{name}");
    }
    assert_eq!(counts, [36, 39, 44, 47, 61]);
}

/// Changing, adding or removing any parameter of a fixture is detected and named, for every
/// fixture.
#[test]
fn parameter_cross_check_detects_every_disagreement() {
    for name in Fixture::TOY {
        let fx = Fixture::load(name).unwrap();
        let keys: Vec<String> = fx
            .get("/params")
            .unwrap()
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(keys.len(), 44, "{name}");
        for key in &keys {
            let mut root = fx.json().clone();
            let entry = root.pointer_mut(&format!("/params/{key}")).unwrap();
            *entry = match entry {
                Value::Number(x) if x.is_u64() => json!(x.as_u64().unwrap() + 1),
                Value::Number(x) => json!(x.as_f64().unwrap() * (1.0 + 1e-9)),
                Value::String(s) => json!(format!("{s}x")),
                other => panic!("unexpected parameter value {other}"),
            };
            let tampered = Fixture::from_json(name, root);
            // n, q, d and k are also read by the loader; the cross-check still names the key.
            let msg = match tampered {
                Ok(t) => fixture_err(t.params()),
                Err(Error::Fixture(msg)) => msg,
                Err(e) => panic!("{name} {key}: {e}"),
            };
            assert!(msg.contains(key.as_str()), "{name} {key}: {msg}");
        }
        let mut root = fx.json().clone();
        root["params"]["extra"] = json!(1);
        let msg = fixture_err(Fixture::from_json(name, root).unwrap().params());
        assert!(msg.contains("extra: not cross-checked"), "{name}: {msg}");
        let mut root = fx.json().clone();
        root["params"].as_object_mut().unwrap().remove("tau");
        let msg = fixture_err(Fixture::from_json(name, root).unwrap().params());
        assert!(msg.contains("tau: fixture None"), "{name}: {msg}");
        let mut root = fx.json().clone();
        root["preset"] = json!("toy-n5");
        let msg = fixture_err(Fixture::from_json(name, root).unwrap().params());
        assert!(msg.contains("preset"), "{name}: {msg}");
    }
}

#[test]
fn loading_failures() {
    for bad in ["", "../Cargo", "toy n1", "toy/n1"] {
        assert!(
            fixture_err(Fixture::load(bad)).contains("invalid fixture name"),
            "{bad:?}"
        );
    }
    assert!(fixture_err(Fixture::load("toy-n9")).contains("cannot read"));
    let fx = Fixture::load("toy-n1").unwrap();
    let mut root = fx.json().clone();
    root["format"] = json!("hachi-co-ref-fixture/v2");
    assert!(fixture_err(Fixture::from_json("x", root)).contains("format"));
    let mut root = fx.json().clone();
    root["params"].as_object_mut().unwrap().remove("q");
    assert!(fixture_err(Fixture::from_json("x", root)).contains("/params/q"));
}

#[test]
fn decoders_check_types_shapes_and_residues() {
    let fx = Fixture::load("toy-n1").unwrap();
    let q = fx.q();
    // Shapes.
    assert_eq!(fx.poly("/commit/u/0").unwrap().0.len(), 16);
    assert_eq!(fx.polys("/commit/u").unwrap().len(), 1);
    assert_eq!(
        fx.polys_of_len("/sumcheck/quotient_shares/0", 32)
            .unwrap()
            .len(),
        5
    );
    assert_eq!(fx.matrix("/setup/A").unwrap().len(), 1);
    assert_eq!(fx.exts::<4>("/input/x").unwrap().len(), 6);
    assert_eq!(fx.ext_rows::<4>("/commit/reshare/0").unwrap().len(), 1);
    assert_eq!(fx.residue_rows("/conv/E_matrix").unwrap().len(), 16);
    assert_eq!(fx.residues("/setup/A").unwrap().len(), 48 * 16);
    assert_eq!(
        fx.ints("/ring_to_linear/attempts/0/provers/0/mask")
            .unwrap()
            .len(),
        48 * 16
    );
    assert_eq!(
        fx.int_rows("/ring_to_linear/attempts/0/c").unwrap().len(),
        4
    );
    assert_eq!(fx.usize("/ring_to_linear/accepted_attempt").unwrap(), 4);
    assert!(fx.bool("/verifier/accept").unwrap());
    // Wrong shapes and types.
    assert!(fixture_err(fx.poly("/commit/u")).contains("expected 16 residues"));
    assert!(fixture_err(fx.polys("/sumcheck/quotient_shares/0")).contains("expected 16"));
    assert!(fixture_err(fx.ext::<4>("/input/y_ring")).contains("expected 4"));
    assert!(fixture_err(fx.ext::<2>("/input/y")).contains("K = 2"));
    assert!(fixture_err(fx.bool("/params/n")).contains("boolean"));
    assert!(fixture_err(fx.str("/params/n")).contains("string"));
    assert!(fixture_err(fx.get("/no/such/key")).contains("missing"));
    assert!(fixture_err(fx.ints("/conventions/indexing")).contains("integer"));
    assert!(fixture_err(fx.int_rows("/params/q")).contains("list"));
    // Masks are signed: a negative entry is neither a u64 nor a residue.
    let mask = fx
        .int_rows("/ring_to_linear/attempts/0/provers/0/mask")
        .unwrap();
    let (i, j) = (0..mask.len())
        .flat_map(|i| (0..mask[i].len()).map(move |j| (i, j)))
        .find(|(i, j)| mask[*i][*j] < 0)
        .expect("a mask of 768 uniform entries has a negative one");
    let pointer = format!("/ring_to_linear/attempts/0/provers/0/mask/{i}/{j}");
    assert!(fixture_err(fx.u64(&pointer)).contains("non-negative integer"));
    assert!(
        fixture_err(fx.residues("/ring_to_linear/attempts/0/provers/0/mask"))
            .contains("not a residue")
    );
    // Residues must be below q.
    let mut root = fx.json().clone();
    root["commit"]["u"][0][3] = json!(q);
    let tampered = Fixture::from_json("toy-n1", root).unwrap();
    assert!(fixture_err(tampered.poly("/commit/u/0")).contains("not below q"));
    assert!(fixture_err(tampered.residues("/commit/u")).contains("not a residue"));
    let mut root = fx.json().clone();
    root["setup"]["D"][0][2][1] = json!(q + 5);
    let tampered = Fixture::from_json("toy-n1", root).unwrap();
    assert!(fixture_err(tampered.context::<4>()).contains("not below q"));
}

#[test]
fn digest_convention_and_pointer_escaping() {
    // Known answers from Python's hashlib (sha256 of little-endian u64 words).
    assert_eq!(
        sha256_residues([]),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_residues([1, 2, 8388637 - 1]),
        "f67555e7589cf6faef60fbad7b31b10feb11edb4384422cb6ebee9738ee8ef17"
    );
    assert_eq!(
        sha256_residues([u64::MAX]),
        "12a3ae445661ce5dee78d0650d33362dec29c4f82af05e7e57fb595bbbacf0ca"
    );
    assert_eq!(escape_key("a~b/c"), "a~0b~1c");
    let fx = Fixture::load("toy-n1").unwrap();
    let key = "sc.sum_x sum_p w~_p(x) M*(x) = a";
    assert!(fx.bool(&format!("/relations/{}", escape_key(key))).unwrap());
    // A key with `/`, or with `~0` / `~1` inside, is found only when escaped.
    let mut root = fx.json().clone();
    root["relations"]["a/b ~1"] = json!(true);
    let fx = Fixture::from_json("toy-n1", root).unwrap();
    assert!(
        fx.bool(&format!("/relations/{}", escape_key("a/b ~1")))
            .unwrap()
    );
    assert!(fx.get("/relations/a/b ~1").is_err());
}
