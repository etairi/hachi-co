//! The fixture loader: parameters cross-checked field by field, the typed decoders and their
//! failure cases, the digest convention and the JSON-pointer escaping.
#![cfg(feature = "fixtures")]

use hachi_co_core::{Error, Fixture, escape_key, sha256_residues};
use serde_json::{Value, json};

fn fixture_err<T: core::fmt::Debug>(r: hachi_co_core::Result<T>) -> String {
    match r {
        Err(Error::Fixture(msg)) => msg,
        other => panic!("expected a fixture error, got {other:?}"),
    }
}

#[test]
fn every_fixture_loads_and_its_parameters_agree_with_rust() {
    for (n, name) in Fixture::TOY.iter().enumerate() {
        let fx = Fixture::load(name).unwrap();
        assert_eq!(fx.name(), *name);
        assert_eq!(fx.n(), n + 1);
        assert_eq!((fx.d(), fx.k()), (16, 4));
        let params = fx.params().unwrap();
        assert_eq!(params.choices.n, n as u64 + 1);
        assert_eq!(params.q, fx.q());
        assert_eq!(fx.str("/format").unwrap(), Fixture::FORMAT);
    }
}

/// Changing, adding or removing any parameter of the fixture is detected and named.
#[test]
fn parameter_cross_check_detects_every_disagreement() {
    let fx = Fixture::load("toy-n2").unwrap();
    let keys: Vec<String> = fx
        .get("/params")
        .unwrap()
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    assert_eq!(keys.len(), 44);
    for key in &keys {
        let mut root = fx.json().clone();
        let entry = root.pointer_mut(&format!("/params/{key}")).unwrap();
        *entry = match entry {
            Value::Number(x) if x.is_u64() => json!(x.as_u64().unwrap() + 1),
            Value::Number(x) => json!(x.as_f64().unwrap() * (1.0 + 1e-9)),
            Value::String(s) => json!(format!("{s}x")),
            other => panic!("unexpected parameter value {other}"),
        };
        let tampered = Fixture::from_json("toy-n2", root);
        // n, q, d and k are also read by the loader; the cross-check still names the key.
        let msg = match tampered {
            Ok(t) => fixture_err(t.params()),
            Err(Error::Fixture(msg)) => msg,
            Err(e) => panic!("{key}: {e}"),
        };
        assert!(msg.contains(key.as_str()), "{key}: {msg}");
    }
    let mut root = fx.json().clone();
    root["params"]["extra"] = json!(1);
    let msg = fixture_err(Fixture::from_json("toy-n2", root).unwrap().params());
    assert!(msg.contains("extra: not cross-checked"), "{msg}");
    let mut root = fx.json().clone();
    root["params"].as_object_mut().unwrap().remove("tau");
    let msg = fixture_err(Fixture::from_json("toy-n2", root).unwrap().params());
    assert!(msg.contains("tau: fixture None"), "{msg}");
    let mut root = fx.json().clone();
    root["preset"] = json!("toy-n3");
    let msg = fixture_err(Fixture::from_json("toy-n2", root).unwrap().params());
    assert!(msg.contains("preset"), "{msg}");
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
