//! Completeness and first soundness checks of [`HachiPcs`]: multilinear openings and inner
//! products at toy parameters, with and without iterations, against direct evaluations; false
//! values and altered proofs are rejected. The automatic schedule runs the number of iterations
//! whose proof is smallest. The packed encodings round-trip and have the sizes of the formulas
//! at moduli of 24, 34 and 63 bits and bases 4, 6 and 16; whatever decodes re-encodes to the same
//! words, and malformed encodings are rejected. Pinned outputs guard against unintended changes.

use hachi::params::MAX_ITERATIONS;
use hachi::{HachiChoices, HachiPcs, Proof, Schedule};
use hachi_co_math::{Ext, Fqk};
use hachi_co_sampling::ShakePrg;
use hachi_co_sampling::uniform::uniform_mod_q;
use sha2::{Digest, Sha256};

const K: usize = 4;
/// The modulus of Hachi-Co's toy-n1 preset, 5 mod 8 (24 bits).
const Q: u64 = 8388637;
/// The modulus of Hachi-Co's toy-n8 and dev-l12 presets (30 bits).
const Q30: u64 = 536871029;
/// The modulus of Hachi-Co's dev-l16 preset (34 bits).
const Q34: u64 = 8589934621;
/// The modulus of the secure presets with $\ell = 30$, $\alpha = 8$ (63 bits).
const Q63: u64 = 6618259129420972141;

/// The prover's seeds; these non-hiding choices do not use them, so the pinned outputs do not
/// depend on them.
const SEED: [u8; 32] = [5u8; 32];
const OPEN_SEED: [u8; 32] = [6u8; 32];

fn residues_mod(q: u64, tag: &str, index: u64, len: usize) -> Vec<u64> {
    uniform_mod_q(
        &mut ShakePrg::new(&[3u8; 32], tag.as_bytes(), index),
        q,
        len,
    )
    .unwrap()
}

fn residues(tag: &str, index: u64, len: usize) -> Vec<u64> {
    residues_mod(Q, tag, index, len)
}

fn exts_mod(q: u64, tag: &str, index: u64, len: usize) -> Vec<Ext<K>> {
    residues_mod(q, tag, index, len * K)
        .as_chunks::<K>()
        .0
        .iter()
        .map(|c| Ext(*c))
        .collect()
}

fn exts(tag: &str, index: u64, len: usize) -> Vec<Ext<K>> {
    exts_mod(Q, tag, index, len)
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

/// SplitMix64, for the mutations of the fuzz test.
struct Mix(u64);

impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

#[test]
fn multilinear_openings_verify_with_and_without_iterations() {
    let field = Fqk::<K>::new(Q).unwrap();
    for (vars, iterations) in [(4usize, 0), (8, 0), (6, 1), (8, 1), (10, 2), (12, 1)] {
        let pcs = HachiPcs::<K>::new(Q, choices(iterations)).unwrap();
        let coeffs = residues("f", vars as u64, 1 << vars);
        let point = exts("x", vars as u64, vars);
        let (com, state) = pcs.commit_multilinear(&coeffs, &SEED).unwrap();
        let context = [vars as u8; 32];
        let (value, proof) = pcs
            .prove_multilinear(&state, &point, &context, &OPEN_SEED)
            .unwrap();
        assert_eq!(value, evaluate(&field, &coeffs, &point), "vars {vars}");
        let schedule = pcs.multilinear_schedule(vars).unwrap();
        assert_eq!(proof.iterations.len(), schedule.iterations.len());
        pcs.verify_multilinear(&com, vars, &point, &value, &proof, &context)
            .unwrap_or_else(|e| panic!("vars {vars}: {e}"));
        let words = pcs.proof_to_words(&proof);
        eprintln!(
            "vars {vars}: {} iterations, proof {} words",
            proof.iterations.len(),
            words.len()
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
        // The encodings round-trip, with the sizes of the formulas.
        assert_eq!(pcs.proof_from_words(&words).unwrap(), proof);
        assert_eq!(
            words.len(),
            pcs.multilinear_proof_words(vars).unwrap(),
            "vars {vars}"
        );
        let com_words = pcs.commitment_to_words(&com);
        assert_eq!(com_words.len(), pcs.commitment_words());
        assert_eq!(pcs.commitment_from_words(&com_words).unwrap(), com);
    }
}

#[test]
fn inner_products_verify_with_and_without_iterations() {
    let field = Fqk::<K>::new(Q).unwrap();
    for (len, iterations) in [(1usize, 0), (37, 0), (1, 1), (37, 1), (211, 2)] {
        let pcs = HachiPcs::<K>::new(Q, choices(iterations)).unwrap();
        let values = exts("s", len as u64, len);
        let weights = exts("e", len as u64, len);
        let (com, state) = pcs.commit_vector(&values, &SEED).unwrap();
        let context = [len as u8; 32];
        let (value, proof) = pcs
            .prove_inner_product(&state, &weights, &context, &OPEN_SEED)
            .unwrap();
        let direct = values
            .iter()
            .zip(&weights)
            .fold(field.zero(), |acc, (v, e)| {
                field.add(&acc, &field.mul(v, e))
            });
        assert_eq!(value, direct, "len {len}");
        pcs.verify_inner_product(&com, len, &weights, &value, &proof, &context)
            .unwrap_or_else(|e| panic!("len {len}: {e}"));
        let words = pcs.proof_to_words(&proof);
        assert_eq!(
            words.len(),
            pcs.vector_proof_words(len).unwrap(),
            "len {len}"
        );
        assert_eq!(pcs.proof_from_words(&words).unwrap(), proof);
        assert_eq!(pcs.commitment_to_words(&com).len(), pcs.commitment_words());
        eprintln!(
            "len {len}: {} iterations, proof {} words",
            proof.iterations.len(),
            words.len()
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
    // Proofs without and with iterations (the final witness as residues, then as digits):
    // changing any single word, by one bit or by adding one, must make decoding or verification
    // fail (up to 150 words per proof, spread over the whole encoding).
    for iterations in [0usize, 2] {
        let pcs = HachiPcs::<K>::new(Q, choices(iterations)).unwrap();
        let vars = 8;
        let coeffs = residues("f", 77, 1 << vars);
        let point = exts("x", 77, vars);
        let (com, state) = pcs.commit_multilinear(&coeffs, &SEED).unwrap();
        let context = [1u8; 32];
        let (value, proof) = pcs
            .prove_multilinear(&state, &point, &context, &OPEN_SEED)
            .unwrap();
        assert_eq!(proof.iterations.len(), iterations);
        let words = pcs.proof_to_words(&proof);
        let step = (words.len() / 150).max(1);
        let mut tried = 0;
        for i in (0..words.len()).step_by(step) {
            for altered_word in [words[i] ^ (1 << (i % 64)), words[i].wrapping_add(1)] {
                let mut altered = words.clone();
                altered[i] = altered_word;
                let ok = pcs.proof_from_words(&altered).is_ok_and(|p| {
                    pcs.verify_multilinear(&com, vars, &point, &value, &p, &context)
                        .is_ok()
                });
                assert!(
                    !ok,
                    "{iterations} iterations: word {i} of {} altered and still accepted",
                    words.len()
                );
                tried += 1;
            }
        }
        eprintln!("{iterations} iterations: {tried} altered proofs rejected");
    }
}

#[test]
#[ignore = "timings at Hachi-Co's sizes: run with --ignored --nocapture"]
fn timings_at_hachi_co_sizes() {
    use std::time::Instant;
    for (vars, iterations) in [
        (15usize, Some(0)),
        (15, Some(1)),
        (15, None),
        (17, Some(0)),
        (17, Some(1)),
        (17, None),
    ] {
        let choices = HachiChoices {
            fixed_iterations: iterations,
            ..HachiChoices::toy(16)
        };
        let pcs = HachiPcs::<K>::new(Q, choices).unwrap();
        let coeffs = residues("f", vars as u64, 1 << vars);
        let point = exts("x", vars as u64, vars);
        let t = Instant::now();
        let (com, state) = pcs.commit_multilinear(&coeffs, &SEED).unwrap();
        let commit = t.elapsed();
        let t = Instant::now();
        let (value, proof) = pcs
            .prove_multilinear(&state, &point, &[0u8; 32], &OPEN_SEED)
            .unwrap();
        let prove = t.elapsed();
        let t = Instant::now();
        pcs.verify_multilinear(&com, vars, &point, &value, &proof, &[0u8; 32])
            .unwrap();
        let verify = t.elapsed();
        eprintln!(
            "vars {vars}, {} iterations ({}): commit {commit:.2?}, prove {prove:.2?}, verify {verify:.2?}, proof {} words",
            proof.iterations.len(),
            if iterations.is_some() {
                "fixed"
            } else {
                "automatic"
            },
            pcs.proof_to_words(&proof).len()
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
        let (com, state) = pcs.commit_multilinear(&coeffs, &SEED).unwrap();
        let (value, proof) = pcs
            .prove_multilinear(&state, &point, &[2u8; 32], &OPEN_SEED)
            .unwrap();
        assert_eq!(value, evaluate(&field, &coeffs, &point));
        pcs.verify_multilinear(&com, vars, &point, &value, &proof, &[2u8; 32])
            .unwrap_or_else(|e| panic!("d {d}, b {b}, heights {n_a} {n_b} {n_d}: {e}"));
        assert_eq!(
            pcs.proof_to_words(&proof).len(),
            pcs.multilinear_proof_words(vars).unwrap()
        );
        assert_eq!(pcs.commitment_to_words(&com).len(), pcs.commitment_words());
        let len = 29;
        let values = exts("v", d as u64, len);
        let weights = exts("w", d as u64, len);
        let (com, state) = pcs.commit_vector(&values, &SEED).unwrap();
        let (value, proof) = pcs
            .prove_inner_product(&state, &weights, &[4u8; 32], &OPEN_SEED)
            .unwrap();
        pcs.verify_inner_product(&com, len, &weights, &value, &proof, &[4u8; 32])
            .unwrap_or_else(|e| panic!("vector, d {d}, b {b}: {e}"));
        assert_eq!(
            pcs.proof_to_words(&proof).len(),
            pcs.vector_proof_words(len).unwrap()
        );
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
        let (com, _) = pcs.commit_multilinear(&coeffs, &SEED).unwrap();
        let (_, cheat) = pcs.commit_multilinear(&other, &SEED).unwrap();
        let (value, proof) = pcs
            .prove_multilinear(&cheat, &point, &[0u8; 32], &OPEN_SEED)
            .unwrap();
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
        let (com, _) = pcs.commit_vector(&values, &SEED).unwrap();
        let (_, cheat) = pcs.commit_vector(&changed, &SEED).unwrap();
        let (value, proof) = pcs
            .prove_inner_product(&cheat, &weights, &[0u8; 32], &OPEN_SEED)
            .unwrap();
        assert!(
            pcs.verify_inner_product(&com, len, &weights, &value, &proof, &[0u8; 32])
                .is_err(),
            "{iterations} iterations: a proof about another vector passed"
        );
    }
}

#[test]
fn the_automatic_schedule_minimises_the_proof_size() {
    // At the moduli of Hachi-Co's toy and dev presets and bases 4, 8, 16: the automatic
    // schedule's proof has the fewest words among all fixed numbers of iterations the
    // parameters allow, and it runs the fewest iterations among those that reach that size.
    // Sizes come from the word-count formulas, which the other tests check against encodings.
    let mut iterated = 0;
    let mut cases = 0;
    for q in [Q, Q30, Q34] {
        for b in [4u64, 8, 16] {
            let base = HachiChoices {
                b,
                ..HachiChoices::toy(16)
            };
            let auto = HachiPcs::<K>::new(q, base.clone()).unwrap();
            let fixed: Vec<HachiPcs<K>> = (0..=base.max_iterations)
                .map(|k| {
                    HachiPcs::<K>::new(
                        q,
                        HachiChoices {
                            fixed_iterations: Some(k),
                            ..base.clone()
                        },
                    )
                    .unwrap()
                })
                .collect();
            let check = |words: &[Option<usize>], got_words: usize, got_iterations: usize| {
                let least = words.iter().flatten().min().copied().unwrap();
                let first = words.iter().position(|w| *w == Some(least)).unwrap();
                assert_eq!(got_words, least, "q {q}, b {b}");
                assert_eq!(got_iterations, first, "q {q}, b {b}");
                first
            };
            for vars in [6usize, 10, 15, 17, 21] {
                let words: Vec<_> = fixed
                    .iter()
                    .map(|p| p.multilinear_proof_words(vars).ok())
                    .collect();
                let chosen = check(
                    &words,
                    auto.multilinear_proof_words(vars).unwrap(),
                    auto.multilinear_schedule(vars).unwrap().iterations.len(),
                );
                iterated += usize::from(chosen > 0);
                cases += 1;
            }
            for len in [1usize, 46, 211, 4000] {
                let words: Vec<_> = fixed
                    .iter()
                    .map(|p| p.vector_proof_words(len).ok())
                    .collect();
                let chosen = check(
                    &words,
                    auto.vector_proof_words(len).unwrap(),
                    auto.vector_schedule(len).unwrap().iterations.len(),
                );
                iterated += usize::from(chosen > 0);
                cases += 1;
            }
        }
    }
    eprintln!("{cases} cases, {iterated} with iterations");
    assert!(iterated > 0 && iterated < cases);
}

#[test]
fn the_candidates_stop_at_a_repeated_layout() {
    // With the largest maximum, the candidates still end where a final witness's layout would
    // repeat (every later schedule would be larger than an earlier one), no two candidates have
    // the same final layout, and the choice is that of the default maximum.
    let wide = HachiChoices {
        max_iterations: MAX_ITERATIONS,
        ..HachiChoices::toy(16)
    };
    let pcs = HachiPcs::<K>::new(Q, wide.clone()).unwrap();
    let default = HachiPcs::<K>::new(Q, HachiChoices::toy(16)).unwrap();
    for vars in [8usize, 12, 17, 21, 26] {
        let candidates = Schedule::candidates(&wide, Q, pcs.multilinear_layout(vars).unwrap());
        assert!(
            candidates.len() < 16,
            "vars {vars}: {} candidates",
            candidates.len()
        );
        let finals: Vec<_> = candidates.iter().map(|s| s.final_layout).collect();
        for (i, layout) in finals.iter().enumerate() {
            assert!(
                !finals[..i].contains(layout),
                "vars {vars}: layout {i} repeats"
            );
        }
        assert_eq!(
            pcs.multilinear_schedule(vars).unwrap(),
            default.multilinear_schedule(vars).unwrap(),
            "vars {vars}"
        );
    }
}

#[test]
fn unusable_parameters_are_refused() {
    // A base above q: its digits would not be distinct residues, and a final witness would have
    // two encodings (q = 5, b = 6: the digits -3 and 2 are the same residue). A fixed number of
    // iterations above the maximum, and a maximum above MAX_ITERATIONS.
    let refused = |q: u64, choices: HachiChoices| HachiPcs::<K>::new(q, choices).is_err();
    let toy = HachiChoices::toy(16);
    assert!(refused(
        5,
        HachiChoices {
            b: 6,
            ..toy.clone()
        }
    ));
    assert!(refused(
        Q,
        HachiChoices {
            b: Q + 1,
            ..toy.clone()
        }
    ));
    assert!(!refused(
        Q,
        HachiChoices {
            b: Q,
            ..toy.clone()
        }
    ));
    for (fixed, max, refuse) in [
        (Some(16), 16, false),
        (Some(17), 16, true),
        (Some(usize::MAX), 16, true),
        (None, MAX_ITERATIONS, false),
        (None, MAX_ITERATIONS + 1, true),
    ] {
        let choices = HachiChoices {
            fixed_iterations: fixed,
            max_iterations: max,
            ..toy.clone()
        };
        assert_eq!(refused(Q, choices), refuse, "fixed {fixed:?}, max {max}");
    }
    // Schedule::fixed refuses more iterations than the maximum, without allocating for them.
    let first = HachiPcs::<K>::new(Q, toy.clone())
        .unwrap()
        .multilinear_layout(12)
        .unwrap();
    assert!(Schedule::fixed(&toy, Q, first, usize::MAX).is_err());
    assert!(Schedule::fixed(&toy, Q, first, 16).is_ok());
}

#[test]
fn packed_encodings_have_the_sizes_of_the_formulas() {
    // Moduli of 24, 34 and 63 bits (fields that straddle words, padded last words), bases 4, 6
    // (not a power of two) and 16, degrees 16 and 32, heights above 1, and 0 to 2 iterations:
    // every commitment and proof has the size of its formula, decodes to itself and verifies.
    let cases: [(u64, usize, u64, usize, usize, usize, usize); 8] = [
        (Q, 16, 4, 1, 1, 1, 0),
        (Q, 16, 4, 1, 1, 1, 2),
        (Q34, 16, 6, 2, 1, 2, 1),
        (Q34, 32, 16, 1, 2, 1, 2),
        (Q34, 16, 6, 1, 1, 1, 0),
        (Q63, 16, 4, 1, 1, 1, 1),
        (Q63, 32, 6, 1, 1, 1, 2),
        (Q63, 16, 16, 2, 2, 1, 0),
    ];
    for (q, d, b, n_a, n_b, n_d, iterations) in cases {
        let what = format!("q {q}, d {d}, b {b}, heights {n_a} {n_b} {n_d}, {iterations} it.");
        let choices = HachiChoices {
            d,
            b,
            n_a,
            n_b,
            n_d,
            fixed_iterations: Some(iterations),
            ..HachiChoices::toy(d)
        };
        let pcs = HachiPcs::<K>::new(q, choices).unwrap();
        let field = Fqk::<K>::new(q).unwrap();
        let vars = 9;
        let coeffs = residues_mod(q, "packed/f", d as u64 + b, 1 << vars);
        let point = exts_mod(q, "packed/x", d as u64 + b, vars);
        let (com, state) = pcs.commit_multilinear(&coeffs, &SEED).unwrap();
        let (value, proof) = pcs
            .prove_multilinear(&state, &point, &[6u8; 32], &OPEN_SEED)
            .unwrap();
        assert_eq!(value, evaluate(&field, &coeffs, &point), "{what}");
        let words = pcs.proof_to_words(&proof);
        assert_eq!(
            words.len(),
            pcs.multilinear_proof_words(vars).unwrap(),
            "{what}"
        );
        let decoded = pcs.proof_from_words(&words).unwrap();
        assert_eq!(decoded, proof, "{what}");
        pcs.verify_multilinear(&com, vars, &point, &value, &decoded, &[6u8; 32])
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        let com_words = pcs.commitment_to_words(&com);
        assert_eq!(com_words.len(), pcs.commitment_words(), "{what}");
        assert_eq!(
            pcs.commitment_from_words(&com_words).unwrap(),
            com,
            "{what}"
        );
        let len = 37;
        let values = exts_mod(q, "packed/s", d as u64 + b, len);
        let weights = exts_mod(q, "packed/e", d as u64 + b, len);
        let (com, state) = pcs.commit_vector(&values, &SEED).unwrap();
        let (value, proof) = pcs
            .prove_inner_product(&state, &weights, &[7u8; 32], &OPEN_SEED)
            .unwrap();
        let words = pcs.proof_to_words(&proof);
        assert_eq!(words.len(), pcs.vector_proof_words(len).unwrap(), "{what}");
        let decoded = pcs.proof_from_words(&words).unwrap();
        assert_eq!(decoded, proof, "{what}");
        pcs.verify_inner_product(&com, len, &weights, &value, &decoded, &[7u8; 32])
            .unwrap_or_else(|e| panic!("vector, {what}: {e}"));
    }
}

/// The bits of the packed fields of a proof, counted from the proof itself (an independent
/// statement of the layout of `pcs.rs`, "Encodings"): $\lceil \log_2 q \rceil$ per residue and
/// $\lceil \log_2 b \rceil$ per digit of a final witness after iterations.
fn stream_bits(pcs: &HachiPcs<K>, proof: &Proof<K>) -> usize {
    let (c, d) = (&pcs.ctx().choices, pcs.ctx().d());
    let rho = (u64::BITS - (pcs.ctx().q - 1).leading_zeros()) as usize;
    let digit = (u64::BITS - (c.b - 1).leading_zeros()) as usize;
    let mut residues = K * proof.partials.len() + d;
    for it in &proof.iterations {
        residues += (it.v.len() + it.next_commitment.len()) * d;
        residues += it.rounds.len() * K * (c.b as usize + 2 + 3) + K + K * K + d;
    }
    let entries = proof.final_message.len() * d;
    let entry = if proof.iterations.is_empty() {
        rho
    } else {
        digit
    };
    residues * rho + entries * entry
}

#[test]
fn malformed_encodings_are_rejected() {
    // b = 6, so that a digit index can fall outside the digit set (3 bits, 6 digits); a 34-bit
    // modulus, so that fields straddle words and the last words are padded.
    let pcs = HachiPcs::<K>::new(
        Q34,
        HachiChoices {
            b: 6,
            fixed_iterations: Some(1),
            ..HachiChoices::toy(16)
        },
    )
    .unwrap();
    // The first number of variables whose proof leaves padding in its last word.
    let (vars, com, point, value, proof) = (8..12)
        .map(|vars| {
            let (com, state) = pcs
                .commit_multilinear(&residues_mod(Q34, "bad/f", 0, 1 << vars), &SEED)
                .unwrap();
            let point = exts_mod(Q34, "bad/x", 0, vars);
            let (value, proof) = pcs
                .prove_multilinear(&state, &point, &[5u8; 32], &OPEN_SEED)
                .unwrap();
            (vars, com, point, value, proof)
        })
        .find(|(_, _, _, _, proof)| !stream_bits(&pcs, proof).is_multiple_of(64))
        .expect("a case with padding");
    let words = pcs.proof_to_words(&proof);
    let com_words = pcs.commitment_to_words(&com);
    let header = 4 + proof.iterations.len();
    let bits = stream_bits(&pcs, &proof);
    assert_eq!(
        words.len(),
        header + bits.div_ceil(64),
        "the layout of the encoding"
    );
    let proof_err = |w: &[u64]| pcs.proof_from_words(w).err().map(|e| e.to_string());
    let com_err = |w: &[u64]| pcs.commitment_from_words(w).err().map(|e| e.to_string());
    let edited = |w: &[u64], i: usize, value: u64| {
        let mut w = w.to_vec();
        w[i] = value;
        w
    };
    // Every truncation and an extension.
    for cut in 0..words.len() {
        assert!(proof_err(&words[..cut]).is_some(), "proof cut at {cut}");
    }
    for cut in 0..com_words.len() {
        assert!(
            com_err(&com_words[..cut]).is_some(),
            "commitment cut at {cut}"
        );
    }
    assert!(proof_err(&[words.clone(), vec![0]].concat()).is_some());
    assert!(com_err(&[com_words.clone(), vec![0]].concat()).is_some());
    // The version.
    for version in [0u64, 1, 3] {
        assert!(proof_err(&edited(&words, 0, version)).is_some());
        assert!(com_err(&edited(&com_words, 0, version)).is_some());
    }
    // The number of partial evaluations: 0 or K only, and K for this proof's length.
    for partials in [0u64, 1, 3, 5, 8, u64::MAX] {
        assert!(
            proof_err(&edited(&words, 1, partials)).is_some(),
            "{partials}"
        );
    }
    // The counts of the header (iterations, the rounds of the iteration, the final ring
    // elements), including huge ones: errors, without allocating for them.
    for (i, value) in [
        (2, words[2] + 1),
        (2, words[2] - 1),
        (2, u64::MAX),
        (3, words[3] + 1),
        (3, words[3] - 1),
        (3, 1 << 62),
        (4, words[4] + 1),
        (4, u64::MAX),
    ] {
        assert!(
            proof_err(&edited(&words, i, value)).is_some(),
            "header {i} = {value}"
        );
    }
    // A residue not below q: the first field (34 bits) set to all ones.
    let all_ones = (1u64 << 34) - 1;
    let err = proof_err(&edited(&words, header, words[header] | all_ones)).unwrap();
    assert!(err.contains("not below q"), "{err}");
    let err = com_err(&edited(&com_words, 1, com_words[1] | all_ones)).unwrap();
    assert!(err.contains("not below q"), "{err}");
    // Non-zero padding.
    let last = words.len() - 1;
    let err = proof_err(&edited(&words, last, words[last] | 1 << 63)).unwrap();
    assert!(err.contains("padding"), "{err}");
    let last = com_words.len() - 1;
    let err = com_err(&edited(&com_words, last, com_words[last] | 1 << 63)).unwrap();
    assert!(err.contains("padding"), "{err}");
    // A digit index outside the digit set {-3, ..., 2}: a final table entry of 3 encodes as
    // index 6.
    let sub = &pcs.ctx().sub;
    let mut bad = proof.clone();
    let mut table = sub.psi_inv(proof.final_message.last().unwrap()).unwrap();
    let original = table[0].0[0];
    table[0].0[0] = 3;
    *bad.final_message.last_mut().unwrap() = sub.psi(&table).unwrap();
    let err = proof_err(&pcs.proof_to_words(&bad)).unwrap();
    assert!(err.contains("digit index"), "{err}");
    // Another digit inside the digit set decodes, and the verifier rejects it.
    table[0].0[0] = if original == Q34 - 3 {
        Q34 - 2
    } else {
        Q34 - 3
    };
    *bad.final_message.last_mut().unwrap() = sub.psi(&table).unwrap();
    let decoded = pcs.proof_from_words(&pcs.proof_to_words(&bad)).unwrap();
    assert_eq!(decoded, bad);
    assert!(
        pcs.verify_multilinear(&com, vars, &point, &value, &decoded, &[5u8; 32])
            .is_err()
    );
    // Random words: errors or proofs, never a panic.
    let mut rng = Mix(0x6261_6420_776f_7264);
    for len in 0..64 {
        let w: Vec<u64> = (0..len).map(|_| rng.next() >> (rng.next() % 64)).collect();
        if let Ok(p) = pcs.proof_from_words(&w) {
            assert_eq!(pcs.proof_to_words(&p), w);
        }
        let _ = pcs.commitment_from_words(&w);
    }
}

#[test]
fn whatever_decodes_reencodes_to_the_same_words() {
    // Random mutations of honest encodings (flipped bits, replaced words, header counts off by
    // one), with the final witness as residues and as digits of bases 4 and 6: every mutation
    // is rejected or decodes to a proof that encodes to the same words, so that every proof
    // has a single encoding.
    for (b, iterations) in [(4u64, 0usize), (4, 2), (6, 1)] {
        let pcs = HachiPcs::<K>::new(
            Q34,
            HachiChoices {
                b,
                fixed_iterations: Some(iterations),
                ..HachiChoices::toy(16)
            },
        )
        .unwrap();
        let vars = 8;
        let (_, state) = pcs
            .commit_multilinear(&residues_mod(Q34, "canon/f", b, 1 << vars), &SEED)
            .unwrap();
        let point = exts_mod(Q34, "canon/x", b, vars);
        let (_, proof) = pcs
            .prove_multilinear(&state, &point, &[8u8; 32], &OPEN_SEED)
            .unwrap();
        let words = pcs.proof_to_words(&proof);
        let mut rng = Mix(b * 100 + iterations as u64);
        let (mut decoded, mut rejected) = (0, 0);
        for _ in 0..3000 {
            let mut w = words.clone();
            let i = (rng.next() % w.len() as u64) as usize;
            match rng.next() % 4 {
                0 => w[i] ^= 1 << (rng.next() % 64),
                1 => w[i] ^= (1 << (rng.next() % 64)) | (1 << (rng.next() % 64)),
                2 => w[i] = rng.next(),
                _ => {
                    let j = (rng.next() % 5) as usize;
                    w[j] = if rng.next().is_multiple_of(2) {
                        w[j].wrapping_add(1)
                    } else {
                        w[j].wrapping_sub(1)
                    };
                }
            }
            match pcs.proof_from_words(&w) {
                Ok(p) => {
                    assert_eq!(pcs.proof_to_words(&p), w, "b {b}, {iterations} it.");
                    decoded += 1;
                }
                Err(_) => rejected += 1,
            }
        }
        eprintln!("b {b}, {iterations} iterations: {decoded} decoded, {rejected} rejected");
        assert!(decoded > 0 && rejected > 0);
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
    // Regression values, computed by this implementation (not independent known answers): the
    // SHA-256 of the commitment's and the proof's words at fixed inputs, for a multilinear
    // opening and an inner product, with 0 and 1 iterations. A change of the transcript, the
    // expansion of the matrices, the encoding or the arithmetic changes them; it must then be
    // deliberate, and the values updated with it. Updated on 2026-10-21 for the packed
    // encodings (version 2).
    let pinned: [(&str, usize, &str, &str); 4] = [
        (
            "multilinear",
            0,
            "eb9031220da297512c61355e7ca29b39f952282ee3b236bb13826256da6dcc59",
            "d142c668892b82588b22ac6d3e4382bae4489fa4495da3059338cbc12a8f456b",
        ),
        (
            "multilinear",
            1,
            "eb9031220da297512c61355e7ca29b39f952282ee3b236bb13826256da6dcc59",
            "0d83e0b5fe4eaab74a56a9f330026d9ddf4b45d9e0e2d48c4a17280e57969b55",
        ),
        (
            "vector",
            0,
            "4216c22b622198addd24bdc176e21e3c43ac838801e0a1319178fc5f0b70d32a",
            "c56109f5f239d36e64b1aec81d3014d2f696323dd9d3cf7f811d18b24dbeb17d",
        ),
        (
            "vector",
            1,
            "4216c22b622198addd24bdc176e21e3c43ac838801e0a1319178fc5f0b70d32a",
            "6a5da686fb462aee1276bf69c4437278bb16da90790ff114696a36bbf112cd4d",
        ),
    ];
    let mut failures = Vec::new();
    for (kind, iterations, commitment, proof) in pinned {
        let pcs = HachiPcs::<K>::new(Q, choices(iterations)).unwrap();
        let context = [9u8; 32];
        let (com, words) = if kind == "multilinear" {
            let (com, state) = pcs
                .commit_multilinear(&residues("pin/f", 0, 1 << 10), &SEED)
                .unwrap();
            let point = exts("pin/x", 0, 10);
            let proof = pcs
                .prove_multilinear(&state, &point, &context, &OPEN_SEED)
                .unwrap()
                .1;
            (com, pcs.proof_to_words(&proof))
        } else {
            let (com, state) = pcs.commit_vector(&exts("pin/s", 0, 46), &SEED).unwrap();
            let weights = exts("pin/e", 0, 46);
            let proof = pcs
                .prove_inner_product(&state, &weights, &context, &OPEN_SEED)
                .unwrap()
                .1;
            (com, pcs.proof_to_words(&proof))
        };
        let got = (
            sha256_hex(&pcs.commitment_to_words(&com)),
            sha256_hex(&words),
        );
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
