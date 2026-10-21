//! Completeness and first soundness checks of [`HachiPcs`]: multilinear openings and inner
//! products at toy parameters, with and without iterations, against direct evaluations; false
//! values and altered proofs are rejected; the encodings round-trip and have the sizes of the
//! formulas; pinned outputs guard against unintended changes.

use hachi::{HachiChoices, HachiPcs, Proof};
use hachi_co_math::{Ext, Fqk};
use hachi_co_sampling::ShakePrg;
use hachi_co_sampling::uniform::uniform_mod_q;
use sha2::{Digest, Sha256};

const K: usize = 4;
/// The modulus of Hachi-Co's toy-n1 preset, 5 mod 8.
const Q: u64 = 8388637;

fn residues(tag: &str, index: u64, len: usize) -> Vec<u64> {
    uniform_mod_q(
        &mut ShakePrg::new(&[3u8; 32], tag.as_bytes(), index),
        Q,
        len,
    )
    .unwrap()
}

fn exts(tag: &str, index: u64, len: usize) -> Vec<Ext<K>> {
    residues(tag, index, len * K)
        .as_chunks::<K>()
        .0
        .iter()
        .map(|c| Ext(*c))
        .collect()
}

/// $f(\mathbf x) = \sum_i f_i \prod_j x_j^{i_j}$, $i_1$ the most significant bit.
fn evaluate(field: &Fqk<K>, coeffs: &[u64], x: &[Ext<K>]) -> Ext<K> {
    let mut mono = vec![field.one()];
    for xj in x {
        mono = mono.iter().flat_map(|m| [*m, field.mul(m, xj)]).collect();
    }
    coeffs.iter().zip(&mono).fold(field.zero(), |acc, (c, m)| {
        field.add(&acc, &field.scale(m, *c))
    })
}

/// Toy choices running exactly `iterations` iterations.
fn choices(iterations: usize) -> HachiChoices {
    HachiChoices {
        fixed_iterations: Some(iterations),
        ..HachiChoices::toy(16)
    }
}

#[test]
fn multilinear_openings_verify_with_and_without_iterations() {
    let field = Fqk::<K>::new(Q).unwrap();
    for (vars, iterations) in [(4usize, 0), (8, 0), (6, 1), (8, 1), (10, 2), (12, 1)] {
        let pcs = HachiPcs::<K>::new(Q, choices(iterations)).unwrap();
        let coeffs = residues("f", vars as u64, 1 << vars);
        let point = exts("x", vars as u64, vars);
        let (com, state) = pcs.commit_multilinear(&coeffs).unwrap();
        let context = [vars as u8; 32];
        let (value, proof) = pcs.prove_multilinear(&state, &point, &context).unwrap();
        assert_eq!(value, evaluate(&field, &coeffs, &point), "vars {vars}");
        let schedule = pcs.schedule(pcs.multilinear_layout(vars).unwrap()).unwrap();
        assert_eq!(proof.iterations.len(), schedule.iterations.len());
        pcs.verify_multilinear(&com, vars, &point, &value, &proof, &context)
            .unwrap_or_else(|e| panic!("vars {vars}: {e}"));
        eprintln!(
            "vars {vars}: {} iterations, proof {} words",
            proof.iterations.len(),
            proof.to_words().len()
        );
        // A false value and another point are rejected; so is another context once Fiat-Shamir
        // is used (a proof without iterations is the witness in the clear and uses no coins).
        let wrong = field.add(&value, &field.one());
        assert!(
            pcs.verify_multilinear(&com, vars, &point, &wrong, &proof, &context)
                .is_err()
        );
        if iterations > 0 {
            assert!(
                pcs.verify_multilinear(&com, vars, &point, &value, &proof, &[99u8; 32])
                    .is_err()
            );
        }
        let mut other = point.clone();
        other[0] = field.add(&other[0], &field.one());
        assert!(
            pcs.verify_multilinear(&com, vars, &other, &value, &proof, &context)
                .is_err()
        );
        // The encoding round-trips, with the sizes of the formulas.
        assert_eq!(Proof::<K>::from_words(&proof.to_words()).unwrap(), proof);
        assert_eq!(
            proof.to_words().len(),
            pcs.multilinear_proof_words(vars).unwrap(),
            "vars {vars}"
        );
        assert_eq!(com.to_words().len(), pcs.commitment_words());
    }
}

#[test]
fn inner_products_verify_with_and_without_iterations() {
    let field = Fqk::<K>::new(Q).unwrap();
    for (len, iterations) in [(1usize, 0), (37, 0), (1, 1), (37, 1), (211, 2)] {
        let pcs = HachiPcs::<K>::new(Q, choices(iterations)).unwrap();
        let values = exts("s", len as u64, len);
        let weights = exts("e", len as u64, len);
        let (com, state) = pcs.commit_vector(&values).unwrap();
        let context = [len as u8; 32];
        let (value, proof) = pcs.prove_inner_product(&state, &weights, &context).unwrap();
        let direct = values
            .iter()
            .zip(&weights)
            .fold(field.zero(), |acc, (v, e)| {
                field.add(&acc, &field.mul(v, e))
            });
        assert_eq!(value, direct, "len {len}");
        pcs.verify_inner_product(&com, len, &weights, &value, &proof, &context)
            .unwrap_or_else(|e| panic!("len {len}: {e}"));
        assert_eq!(
            proof.to_words().len(),
            pcs.vector_proof_words(len).unwrap(),
            "len {len}"
        );
        assert_eq!(com.to_words().len(), pcs.commitment_words());
        eprintln!(
            "len {len}: {} iterations, proof {} words",
            proof.iterations.len(),
            proof.to_words().len()
        );
        let wrong = field.add(&value, &field.one());
        assert!(
            pcs.verify_inner_product(&com, len, &weights, &wrong, &proof, &context)
                .is_err()
        );
        let mut other = weights.clone();
        other[0] = field.add(&other[0], &field.one());
        assert!(
            pcs.verify_inner_product(&com, len, &other, &value, &proof, &context)
                .is_err()
        );
    }
}

#[test]
fn every_altered_word_of_a_proof_is_rejected() {
    // A proof with two iterations: changing any single word must make decoding or verification
    // fail (a few hundred words are tried, spread over the whole encoding).
    let pcs = HachiPcs::<K>::new(Q, choices(2)).unwrap();
    let vars = 8;
    let coeffs = residues("f", 77, 1 << vars);
    let point = exts("x", 77, vars);
    let (com, state) = pcs.commit_multilinear(&coeffs).unwrap();
    let context = [1u8; 32];
    let (value, proof) = pcs.prove_multilinear(&state, &point, &context).unwrap();
    assert!(!proof.iterations.is_empty());
    let words = proof.to_words();
    let step = (words.len() / 300).max(1);
    let mut tried = 0;
    for i in (0..words.len()).step_by(step) {
        let mut altered = words.clone();
        altered[i] = altered[i].wrapping_add(1) % Q.max(altered[i].wrapping_add(2));
        if altered[i] == words[i] {
            altered[i] = words[i] ^ 1;
        }
        let ok = Proof::<K>::from_words(&altered).is_ok_and(|p| {
            pcs.verify_multilinear(&com, vars, &point, &value, &p, &context)
                .is_ok()
        });
        assert!(
            !ok,
            "word {i} of {} altered and still accepted",
            words.len()
        );
        tried += 1;
    }
    eprintln!("{tried} altered proofs rejected");
}

#[test]
#[ignore = "timings at Hachi-Co's sizes: run with --ignored --nocapture"]
fn timings_at_hachi_co_sizes() {
    use std::time::Instant;
    for (vars, iterations) in [(15usize, 0), (15, 1), (17, 0), (17, 1)] {
        let pcs = HachiPcs::<K>::new(Q, choices(iterations)).unwrap();
        let coeffs = residues("f", vars as u64, 1 << vars);
        let point = exts("x", vars as u64, vars);
        let t = Instant::now();
        let (com, state) = pcs.commit_multilinear(&coeffs).unwrap();
        let commit = t.elapsed();
        let t = Instant::now();
        let (value, proof) = pcs.prove_multilinear(&state, &point, &[0u8; 32]).unwrap();
        let prove = t.elapsed();
        let t = Instant::now();
        pcs.verify_multilinear(&com, vars, &point, &value, &proof, &[0u8; 32])
            .unwrap();
        let verify = t.elapsed();
        eprintln!(
            "vars {vars}, {iterations} iterations: commit {commit:.2?}, prove {prove:.2?}, verify {verify:.2?}, proof {} words",
            proof.to_words().len()
        );
    }
}

#[test]
fn other_choices_verify_as_well() {
    // Heights above 1 exercise every multi-row index of the relation; other bases and degrees
    // change delta, tau and the packing.
    let field = Fqk::<K>::new(Q).unwrap();
    let cases = [
        (16usize, 4u64, 2usize, 2usize, 2usize, 1usize),
        (32, 16, 2, 1, 3, 1),
        (16, 8, 3, 2, 1, 2),
        (64, 4, 1, 3, 2, 1),
    ];
    for (d, b, n_a, n_b, n_d, iterations) in cases {
        let choices = HachiChoices {
            d,
            b,
            n_a,
            n_b,
            n_d,
            omega: 3,
            fixed_iterations: Some(iterations),
            ..HachiChoices::toy(d)
        };
        let pcs = HachiPcs::<K>::new(Q, choices).unwrap();
        let vars = 9;
        let coeffs = residues("g", d as u64, 1 << vars);
        let point = exts("y", d as u64, vars);
        let (com, state) = pcs.commit_multilinear(&coeffs).unwrap();
        let (value, proof) = pcs.prove_multilinear(&state, &point, &[2u8; 32]).unwrap();
        assert_eq!(value, evaluate(&field, &coeffs, &point));
        pcs.verify_multilinear(&com, vars, &point, &value, &proof, &[2u8; 32])
            .unwrap_or_else(|e| panic!("d {d}, b {b}, heights {n_a} {n_b} {n_d}: {e}"));
        assert_eq!(
            proof.to_words().len(),
            pcs.multilinear_proof_words(vars).unwrap()
        );
        assert_eq!(com.to_words().len(), pcs.commitment_words());
        let len = 29;
        let values = exts("v", d as u64, len);
        let weights = exts("w", d as u64, len);
        let (com, state) = pcs.commit_vector(&values).unwrap();
        let (value, proof) = pcs
            .prove_inner_product(&state, &weights, &[4u8; 32])
            .unwrap();
        pcs.verify_inner_product(&com, len, &weights, &value, &proof, &[4u8; 32])
            .unwrap_or_else(|e| panic!("vector, d {d}, b {b}: {e}"));
        assert_eq!(proof.to_words().len(), pcs.vector_proof_words(len).unwrap());
    }
}

#[test]
fn a_proof_about_another_witness_is_rejected() {
    // The prover commits to f but proves an evaluation of f' (one coefficient changed), or of
    // another vector: the verifier, holding f's commitment, rejects with and without iterations.
    for iterations in [0usize, 1, 2] {
        let pcs = HachiPcs::<K>::new(Q, choices(iterations)).unwrap();
        let vars = 8;
        let coeffs = residues("f", 5, 1 << vars);
        let mut other = coeffs.clone();
        other[17] = (other[17] + 1) % Q;
        let point = exts("x", 5, vars);
        let (com, _) = pcs.commit_multilinear(&coeffs).unwrap();
        let (_, cheat) = pcs.commit_multilinear(&other).unwrap();
        let (value, proof) = pcs.prove_multilinear(&cheat, &point, &[0u8; 32]).unwrap();
        assert!(
            pcs.verify_multilinear(&com, vars, &point, &value, &proof, &[0u8; 32])
                .is_err(),
            "{iterations} iterations: a proof about f' passed for f"
        );
        let len = 21;
        let values = exts("s", 6, len);
        let mut changed = values.clone();
        changed[3].0[1] = (changed[3].0[1] + 1) % Q;
        let weights = exts("e", 6, len);
        let (com, _) = pcs.commit_vector(&values).unwrap();
        let (_, cheat) = pcs.commit_vector(&changed).unwrap();
        let (value, proof) = pcs
            .prove_inner_product(&cheat, &weights, &[0u8; 32])
            .unwrap();
        assert!(
            pcs.verify_inner_product(&com, len, &weights, &value, &proof, &[0u8; 32])
                .is_err(),
            "{iterations} iterations: a proof about another vector passed"
        );
    }
}

/// The SHA-256 of words as little-endian bytes, in hex.
fn sha256_hex(words: &[u64]) -> String {
    let mut h = Sha256::new();
    for w in words {
        h.update(w.to_le_bytes());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn outputs_are_pinned() {
    // Regression values, computed by this implementation on 2026-10-21 (not independent known
    // answers): the SHA-256 of the commitment's and the proof's words at fixed inputs, for a
    // multilinear opening and an inner product, with 0 and 1 iterations. A change of the
    // transcript, the expansion of the matrices, the encoding or the arithmetic changes them; it
    // must then be deliberate, and the values updated with it.
    let pinned: [(&str, usize, &str, &str); 4] = [
        (
            "multilinear",
            0,
            "9ab7f7a31c0fbd150667c5ddcfa93a6cd163a95d581c08951330d4127c74f52b",
            "5d259d551e7437e19d7375e610bd14b272c1a5219fdfddd46425a1f8fa94750d",
        ),
        (
            "multilinear",
            1,
            "9ab7f7a31c0fbd150667c5ddcfa93a6cd163a95d581c08951330d4127c74f52b",
            "0f5b61c5dc6201757f92c16646d504fb6412121011e059e209e0bbc54062edf6",
        ),
        (
            "vector",
            0,
            "3910f5fad59fc9986086d0b9016cac304f8a443f03eae44fc2c8d0afe8454254",
            "9cf3157c258cce8923725ef92e076a22a249d1af4b0255e8713f87caab229e65",
        ),
        (
            "vector",
            1,
            "3910f5fad59fc9986086d0b9016cac304f8a443f03eae44fc2c8d0afe8454254",
            "5c7f048a4b6e25c9c7a8ef280b254cf16dfa0228efd7f93ee65030a1fd066ec7",
        ),
    ];
    let mut failures = Vec::new();
    for (kind, iterations, commitment, proof) in pinned {
        let pcs = HachiPcs::<K>::new(Q, choices(iterations)).unwrap();
        let context = [9u8; 32];
        let (com, words) = if kind == "multilinear" {
            let (com, state) = pcs
                .commit_multilinear(&residues("pin/f", 0, 1 << 10))
                .unwrap();
            let point = exts("pin/x", 0, 10);
            (
                com,
                pcs.prove_multilinear(&state, &point, &context)
                    .unwrap()
                    .1
                    .to_words(),
            )
        } else {
            let (com, state) = pcs.commit_vector(&exts("pin/s", 0, 46)).unwrap();
            let weights = exts("pin/e", 0, 46);
            let proof = pcs
                .prove_inner_product(&state, &weights, &context)
                .unwrap()
                .1;
            (com, proof.to_words())
        };
        let got = (sha256_hex(&com.to_words()), sha256_hex(&words));
        if (got.0.as_str(), got.1.as_str()) != (commitment, proof) {
            failures.push(format!(
                "(\"{kind}\", {iterations}, \"{}\", \"{}\")",
                got.0, got.1
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "outputs changed:\n{}",
        failures.join(",\n")
    );
}
