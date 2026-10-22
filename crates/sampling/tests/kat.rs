//! Known-answer tests: the PRG and every sampler (the Gaussian included) against
//! `tests/fixtures/kat.json`, which `tests/fixtures/gen_kat.py` computes with Python's
//! `hashlib.shake_128` and a Python transcription of the documented byte-level rules. Besides the
//! values, each case checks the number of 64-bit words read and the next word of the stream, so
//! the consumption matches too.

mod common;

use common::{Counting, hex, seed};
use hachi_co_sampling::{ShakePrg, gauss, l1ball, matrix, rej, uniform};
use num_bigint::BigUint;
use rand::Rng;
use serde_json::Value;

fn kat() -> Value {
    serde_json::from_str(include_str!("fixtures/kat.json")).expect("valid fixture")
}

fn cases<'a>(kat: &'a Value, key: &str) -> &'a Vec<Value> {
    let cases = kat[key].as_array().expect("array of cases");
    assert!(!cases.is_empty(), "no cases for {key}");
    cases
}

/// The counting PRG of a case's `(seed, domain, index)`.
fn prg(case: &Value) -> Counting<ShakePrg> {
    Counting::new(ShakePrg::new(
        &seed(case["seed"].as_str().unwrap()),
        &hex(case["domain"].as_str().unwrap()),
        case["index"].as_u64().unwrap(),
    ))
}

/// Check the words read and the next word against the case.
fn check_tail(mut prg: Counting<ShakePrg>, case: &Value, what: &str) {
    assert_eq!(
        prg.words as u64,
        case["words"].as_u64().unwrap(),
        "{what}: words read"
    );
    assert_eq!(
        prg.inner.next_u64(),
        case["next_u64"].as_u64().unwrap(),
        "{what}: next word"
    );
}

fn u64s(v: &Value) -> Vec<u64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap())
        .collect()
}

fn i64s(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

fn big(v: &Value) -> BigUint {
    v.as_str().unwrap().parse().unwrap()
}

#[test]
fn prg_streams_match_hashlib() {
    let kat = kat();
    for case in cases(&kat, "prg_streams") {
        let expected = hex(case["bytes"].as_str().unwrap());
        let mut out = vec![0u8; expected.len()];
        prg(case).inner.fill_bytes(&mut out);
        assert_eq!(out, expected, "stream {case}");
    }
}

#[test]
fn prg_words_are_little_endian_slices_of_one_stream() {
    let kat = kat();
    for case in cases(&kat, "prg_mixed") {
        let mut rng = prg(case).inner;
        for call in case["calls"].as_array().unwrap() {
            match call["op"].as_str().unwrap() {
                "u32" => assert_eq!(u64::from(rng.next_u32()), call["value"].as_u64().unwrap()),
                "u64" => assert_eq!(rng.next_u64(), call["value"].as_u64().unwrap()),
                "bytes" => {
                    let mut out = vec![0u8; call["len"].as_u64().unwrap() as usize];
                    rng.fill_bytes(&mut out);
                    assert_eq!(out, hex(call["value"].as_str().unwrap()));
                }
                op => panic!("unknown op {op}"),
            }
        }
    }
}

#[test]
fn uniform_samplers_match_the_python_reference() {
    let kat = kat();
    for case in cases(&kat, "uniform_below") {
        let mut rng = prg(case);
        let n = case["n"].as_u64().unwrap();
        let values: Vec<u64> = (0..case["values"].as_array().unwrap().len())
            .map(|_| uniform::uniform_below(&mut rng, n).unwrap())
            .collect();
        assert_eq!(values, u64s(&case["values"]), "uniform_below n = {n}");
        check_tail(rng, case, &format!("uniform_below n = {n}"));
    }
    for case in cases(&kat, "uniform_below_big") {
        let mut rng = prg(case);
        let n = big(&case["n"]);
        for (k, expected) in case["values"].as_array().unwrap().iter().enumerate() {
            assert_eq!(
                uniform::uniform_below_big(&mut rng, &n).unwrap(),
                big(expected),
                "n = {n}, draw {k}"
            );
        }
        check_tail(rng, case, &format!("uniform_below_big n = {n}"));
    }
    for case in cases(&kat, "uniform_range_i64") {
        let mut rng = prg(case);
        let (lo, hi) = (case["lo"].as_i64().unwrap(), case["hi"].as_i64().unwrap());
        let expected = i64s(&case["values"]);
        let values: Vec<i64> = (0..expected.len())
            .map(|_| uniform::uniform_range_i64(&mut rng, lo, hi).unwrap())
            .collect();
        assert_eq!(values, expected, "range [{lo}, {hi}]");
        check_tail(rng, case, &format!("range [{lo}, {hi}]"));
    }
    for case in cases(&kat, "uniform_mod_q") {
        let mut rng = prg(case);
        let q = case["q"].as_u64().unwrap();
        let len = case["len"].as_u64().unwrap() as usize;
        assert_eq!(
            uniform::uniform_mod_q(&mut rng, q, len).unwrap(),
            u64s(&case["values"]),
            "q = {q}"
        );
        check_tail(rng, case, &format!("uniform_mod_q q = {q}"));
    }
    for case in cases(&kat, "boxed") {
        let mut rng = prg(case);
        let gamma = case["gamma"].as_u64().unwrap();
        let len = case["len"].as_u64().unwrap() as usize;
        assert_eq!(
            uniform::boxed(&mut rng, gamma, len).unwrap(),
            i64s(&case["values"]),
            "gamma = {gamma}"
        );
        check_tail(rng, case, &format!("boxed gamma = {gamma}"));
        // mask is the same distribution with a CryptoRng bound: same values.
        let mut again = prg(case);
        assert_eq!(
            rej::mask(&mut again, gamma, len).unwrap(),
            i64s(&case["values"]),
            "mask gamma = {gamma}"
        );
    }
    for case in cases(&kat, "short") {
        let mut rng = prg(case);
        let beta = case["beta"].as_u64().unwrap();
        let len = case["len"].as_u64().unwrap() as usize;
        assert_eq!(
            uniform::short(&mut rng, beta, len).unwrap(),
            i64s(&case["values"]),
            "beta = {beta}"
        );
        check_tail(rng, case, &format!("short beta = {beta}"));
    }
}

#[test]
fn the_gaussian_sampler_matches_the_python_reference() {
    let kat = kat();
    for case in cases(&kat, "gaussian") {
        let mut rng = prg(case);
        let log_sigma = case["log_sigma"].as_u64().unwrap() as u32;
        let len = case["len"].as_u64().unwrap() as usize;
        assert_eq!(
            gauss::gaussian(&mut rng, log_sigma, len).unwrap(),
            i64s(&case["values"]),
            "log_sigma = {log_sigma}"
        );
        check_tail(rng, case, &format!("gaussian log_sigma = {log_sigma}"));
    }
}

#[test]
fn challenge_counts_match_python_big_integers() {
    let kat = kat();
    for case in cases(&kat, "l1ball_count") {
        let (d, omega) = (
            case["d"].as_u64().unwrap() as usize,
            case["omega"].as_u64().unwrap(),
        );
        assert_eq!(
            l1ball::count(d, omega),
            big(&case["count"]),
            "|C| for d = {d}, omega = {omega}"
        );
        assert_eq!(
            *l1ball::L1Ball::new(d, omega).unwrap().count(),
            big(&case["count"])
        );
    }
    for case in cases(&kat, "sparse_count") {
        let (d, omega) = (
            case["d"].as_u64().unwrap() as usize,
            case["omega"].as_u64().unwrap(),
        );
        assert_eq!(
            l1ball::count_sparse(d, omega),
            big(&case["count"]),
            "d = {d}, omega = {omega}"
        );
    }
}

#[test]
fn challenge_samplers_match_the_python_reference() {
    let kat = kat();
    for case in cases(&kat, "l1ball_sample") {
        let (d, omega) = (
            case["d"].as_u64().unwrap() as usize,
            case["omega"].as_u64().unwrap(),
        );
        let ball = l1ball::L1Ball::new(d, omega).unwrap();
        let mut rng = prg(case);
        for (k, expected) in case["draws"].as_array().unwrap().iter().enumerate() {
            let c = ball.sample(&mut rng).unwrap();
            assert_eq!(
                c,
                i64s(expected),
                "l1 ball d = {d}, omega = {omega}, draw {k}"
            );
            assert!(l1ball::contains(&c, omega));
        }
        check_tail(rng, case, &format!("l1ball d = {d}, omega = {omega}"));
        // The free function draws the same values.
        let mut rng = prg(case);
        let first = case["draws"].as_array().unwrap()[0].clone();
        assert_eq!(l1ball::sample(&mut rng, d, omega).unwrap(), i64s(&first));
    }
    for case in cases(&kat, "sparse_sample") {
        let (d, omega) = (
            case["d"].as_u64().unwrap() as usize,
            case["omega"].as_u64().unwrap(),
        );
        let mut rng = prg(case);
        for (k, expected) in case["draws"].as_array().unwrap().iter().enumerate() {
            let c = l1ball::sample_sparse(&mut rng, d, omega).unwrap();
            assert_eq!(
                c,
                i64s(expected),
                "sparse d = {d}, omega = {omega}, draw {k}"
            );
            assert!(l1ball::contains_sparse(&c, omega));
        }
        check_tail(rng, case, &format!("sparse d = {d}, omega = {omega}"));
    }
}

#[test]
fn sim_rej_matches_the_python_reference() {
    let kat = kat();
    for case in cases(&kat, "sim_rej") {
        let gamma = case["gamma"].as_u64().unwrap();
        let beta1 = case["beta1"].as_u64().unwrap();
        let n = case["n"].as_u64().unwrap() as usize;
        let mut rng = prg(case);
        for (k, expected) in case["draws"].as_array().unwrap().iter().enumerate() {
            let expected = if expected.is_null() {
                None
            } else {
                Some(i64s(expected))
            };
            assert_eq!(
                rej::sim_rej(&mut rng, gamma, beta1, n).unwrap(),
                expected,
                "draw {k} of {case}"
            );
        }
        check_tail(
            rng,
            case,
            &format!("sim_rej gamma = {gamma}, beta1 = {beta1}"),
        );
    }
}

#[test]
fn matrices_match_the_python_reference() {
    let kat = kat();
    for case in cases(&kat, "matrix") {
        let s = seed(case["seed"].as_str().unwrap());
        let domain = hex(case["domain"].as_str().unwrap());
        let get = |k: &str| case[k].as_u64().unwrap();
        let (rows, cols, d, q) = (
            get("rows") as usize,
            get("cols") as usize,
            get("d") as usize,
            get("q"),
        );
        let m = matrix::expand(&s, &domain, rows, cols, d, q).unwrap();
        let expected: Vec<Vec<Vec<u64>>> = case["entries"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row.as_array().unwrap().iter().map(u64s).collect())
            .collect();
        assert_eq!(m, expected, "matrix {rows} x {cols}, d = {d}, q = {q}");
    }
}
