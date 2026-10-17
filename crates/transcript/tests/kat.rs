//! Known-answer tests: replay `tests/fixtures/kat.json`, computed by `tests/fixtures/gen_kat.py`
//! with Python's `hashlib.shake_128`, and compare the state after every operation and every
//! challenge output.

use hachi_co_transcript::Transcript;
use serde_json::Value;

fn hex(s: &str) -> Vec<u8> {
    assert_eq!(s.len() % 2, 0, "odd hex length");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("hex digit"))
        .collect()
}

fn field(record: &Value, key: &str) -> Vec<u8> {
    hex(record[key]
        .as_str()
        .unwrap_or_else(|| panic!("missing {key} in {record}")))
}

/// Apply the recorded operations and check each digest and output; returns how many were checked.
fn replay(t: &mut Transcript, records: &[Value], path: &str) -> usize {
    let mut checked = 0;
    for (k, rec) in records.iter().enumerate() {
        let at = format!("{path}/{k} ({})", rec["op"]);
        match rec["op"].as_str().unwrap() {
            "message" => t.append_message(&field(rec, "label"), &field(rec, "data")),
            "u64" => t.append_u64(&field(rec, "label"), rec["value"].as_u64().unwrap()),
            "attempt" => t.append_attempt(rec["value"].as_u64().unwrap()),
            "challenge_bytes" => {
                let mut out = vec![0u8; rec["len"].as_u64().unwrap() as usize];
                t.challenge_bytes(&field(rec, "label"), &mut out);
                assert_eq!(out, field(rec, "out"), "{at}: output");
            }
            "challenge_seed" => {
                assert_eq!(
                    t.challenge_seed(&field(rec, "label")).to_vec(),
                    field(rec, "out"),
                    "{at}: output"
                );
            }
            "challenge_u64_below" => {
                let n = rec["n"].as_u64().unwrap();
                let x = t.challenge_u64_below(&field(rec, "label"), n).unwrap();
                assert_eq!(x, rec["out"].as_u64().unwrap(), "{at}: output");
            }
            "fork" => {
                let mut child = t.fork(&field(rec, "label"));
                assert_eq!(
                    child.digest().to_vec(),
                    field(rec, "child_digest"),
                    "{at}: child state"
                );
                checked += replay(&mut child, rec["child_ops"].as_array().unwrap(), &at);
            }
            op => panic!("unknown op {op}"),
        }
        assert_eq!(t.digest().to_vec(), field(rec, "digest"), "{at}: state");
        checked += 1;
    }
    checked
}

#[test]
fn transcripts_match_hashlib() {
    let kat: Value =
        serde_json::from_str(include_str!("fixtures/kat.json")).expect("valid fixture");
    let cases = kat["cases"].as_array().unwrap();
    assert!(cases.len() >= 7);
    let mut checked = 0;
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let mut t = Transcript::new(&field(case, "domain"));
        assert_eq!(
            t.digest().to_vec(),
            field(case, "initial_digest"),
            "{name}: initial state"
        );
        checked += replay(&mut t, case["ops"].as_array().unwrap(), name);
    }
    println!("{checked} recorded operations replayed");
    assert!(checked >= 40);
}
