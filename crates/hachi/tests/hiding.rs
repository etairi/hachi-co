//! Randomized commitments ([`hachi::Hiding`], `docs/hachi-hiding.md` §6).
//!
//! What is tested:
//! - multilinear openings and inner products verify with both distributions of the randomness,
//!   several widths and MLWE ranks, 0 to 3 iterations and the automatic schedule, at moduli of
//!   24, 34 and 63 bits;
//! - the final randomness has its length and stays in its support;
//! - the encodings round-trip and have the sizes of the formulas, recomputed here;
//! - a commitment is randomized by its seed and determined by it, and so is an opening; without
//!   hiding the seeds change nothing;
//! - proofs do not verify across hiding choices;
//! - every altered word is rejected, and so is malformed randomness, by the decoder and by the
//!   verifier; whatever decodes re-encodes to the same words;
//! - the automatic schedule is still the smallest;
//! - unusable choices are refused, and the outputs are pinned.

use hachi::{HachiChoices, HachiPcs, Hiding, Noise, Proof, VerifyError};
use hachi_co_math::{Ext, Fqk};
use hachi_co_sampling::ShakePrg;
use hachi_co_sampling::gauss;
use hachi_co_sampling::uniform::uniform_mod_q;
use sha2::{Digest, Sha256};

const K: usize = 4;
/// The modulus of Hachi-Co's toy-n1 preset (24 bits).
const Q: u64 = 8388637;
/// The modulus of Hachi-Co's toy-n8 and dev-l12 presets (30 bits).
const Q30: u64 = 536871029;
/// The modulus of Hachi-Co's dev-l16 preset (34 bits).
const Q34: u64 = 8589934621;
/// The modulus of the secure presets with $\ell = 30$, $\alpha = 8$ (63 bits).
const Q63: u64 = 6618259129420972141;

const GAUSS1: Noise = Noise::Gaussian { log_sigma: 0 };
const GAUSS8: Noise = Noise::Gaussian { log_sigma: 3 };
const GAUSS16: Noise = Noise::Gaussian { log_sigma: 4 };

fn residues_mod(q: u64, tag: &str, index: u64, len: usize) -> Vec<u64> {
    uniform_mod_q(
        &mut ShakePrg::new(&[3u8; 32], tag.as_bytes(), index),
        q,
        len,
    )
    .unwrap()
}

fn exts_mod(q: u64, tag: &str, index: u64, len: usize) -> Vec<Ext<K>> {
    residues_mod(q, tag, index, len * K)
        .as_chunks::<K>()
        .0
        .iter()
        .map(|c| Ext(*c))
        .collect()
}

/// A seed per test case.
fn seed(tag: u8, index: usize) -> [u8; 32] {
    let mut s = [tag; 32];
    s[..8].copy_from_slice(&(index as u64).to_le_bytes());
    s
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

/// Toy choices with randomized commitments, and the number of iterations fixed if `Some`.
fn hiding(nu: usize, noise: Noise, iterations: Option<usize>) -> HachiChoices {
    HachiChoices {
        hiding: Some(Hiding { nu, noise }),
        fixed_iterations: iterations,
        ..HachiChoices::toy(16)
    }
}

/// The support of the noise at base `b`, recomputed: the digit set, or $\pm$ the table length.
fn support(noise: Noise, b: u64) -> (i64, i64) {
    match noise {
        Noise::Digits => (-((b / 2) as i64), ((b - 1) / 2) as i64),
        Noise::Gaussian { log_sigma } => {
            let bound = gauss::table(log_sigma).unwrap().len() as i64;
            (-bound, bound)
        }
    }
}

/// The words of a proof's encoding, recomputed from its parts (module documentation of
/// `hachi::pcs`, "Encodings"): $6 + k$ header words with hiding, then the packed fields.
fn expected_words(pcs: &HachiPcs<K>, proof: &Proof<K>, noise: Noise) -> usize {
    let (c, d) = (&pcs.ctx().choices, pcs.ctx().d());
    let bits_of = |x: u64| (u64::BITS - x.leading_zeros()) as usize;
    let residue = bits_of(pcs.ctx().q - 1);
    let digit = bits_of(c.b - 1);
    let mut residues = K * proof.partials.len() + d;
    residues += 3 * K * proof.batch_rounds.len() + K * usize::from(proof.batch_eval.is_some());
    for it in &proof.iterations {
        residues += (it.v.len() + it.next_commitment.len()) * d;
        residues += it.rounds.len() * K * (c.b as usize + 2 + 3) + K + K * K + d;
        // The per-iteration batching sumcheck: its rounds (3K each) and the packed value (K).
        residues += it.batch_rounds.len() * 3 * K + K;
    }
    let entry = if proof.iterations.is_empty() {
        residue
    } else {
        digit
    };
    let (lo, hi) = support(noise, c.b);
    let bits = residues * residue
        + proof.final_message.len() * d * entry
        + proof.final_randomness.len() * d * bits_of((hi - lo) as u64);
    6 + proof.iterations.len() + bits.div_ceil(64)
}

/// The checks of an honest proof's final randomness: its length and its support.
fn check_randomness(pcs: &HachiPcs<K>, proof: &Proof<K>, noise: Noise, chunks: usize) {
    let ctx = pcs.ctx();
    let nu = ctx.choices.hiding.unwrap().nu;
    assert_eq!(
        proof.final_randomness.len(),
        chunks * (nu + ctx.choices.n_a)
    );
    let (lo, hi) = support(noise, ctx.choices.b);
    let zq = ctx.ring.zq();
    for x in proof.final_randomness.iter().flat_map(|p| &p.0) {
        assert!((lo..=hi).contains(&zq.to_centered(*x)), "randomness {x}");
    }
}

#[test]
fn multilinear_openings_verify_with_randomized_commitments() {
    let cases = [
        (Q, 1, GAUSS8, Some(0), 8usize),
        (Q, 1, GAUSS8, Some(1), 8),
        (Q, 1, GAUSS8, Some(2), 10),
        (Q, 1, GAUSS8, Some(3), 12),
        (Q, 1, GAUSS8, None, 12),
        (Q, 2, Noise::Digits, Some(1), 8),
        (Q, 2, Noise::Digits, Some(2), 10),
        (Q, 1, GAUSS1, Some(2), 10),
        (Q, 1, GAUSS16, Some(2), 10),
        (Q34, 2, GAUSS8, Some(2), 10),
        (Q63, 1, GAUSS16, Some(1), 8),
        (Q63, 1, Noise::Digits, None, 10),
    ];
    for (i, (q, nu, noise, iterations, vars)) in cases.into_iter().enumerate() {
        let what = format!("q {q}, nu {nu}, {noise:?}, {iterations:?} iterations, {vars} vars");
        let field = Fqk::<K>::new(q).unwrap();
        let pcs = HachiPcs::<K>::new(q, hiding(nu, noise, iterations)).unwrap();
        let coeffs = residues_mod(q, "hide/f", i as u64, 1 << vars);
        let point = exts_mod(q, "hide/x", i as u64, vars);
        let (com, state) = pcs.commit_multilinear(&coeffs, &seed(1, i)).unwrap();
        let context = [i as u8; 32];
        let (value, proof) = pcs
            .prove_multilinear(&state, &point, &context, &seed(2, i))
            .unwrap();
        assert_eq!(value, evaluate(&field, &coeffs, &point), "{what}");
        let schedule = pcs.multilinear_schedule(vars).unwrap();
        assert_eq!(proof.iterations.len(), schedule.iterations.len(), "{what}");
        if let Some(n) = iterations {
            assert_eq!(proof.iterations.len(), n, "{what}");
        }
        check_randomness(&pcs, &proof, noise, schedule.final_layout.chunks());
        pcs.verify_multilinear(&com, vars, &point, &value, &proof, &context)
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        let wrong = field.add(&value, &field.one());
        assert!(
            pcs.verify_multilinear(&com, vars, &point, &wrong, &proof, &context)
                .is_err(),
            "{what}: a false value"
        );
        let words = pcs.proof_to_words(&proof);
        assert_eq!(pcs.proof_from_words(&words).unwrap(), proof, "{what}");
        assert_eq!(
            words.len(),
            pcs.multilinear_proof_words(vars).unwrap(),
            "{what}"
        );
        assert_eq!(words.len(), expected_words(&pcs, &proof, noise), "{what}");
        let com_words = pcs.commitment_to_words(&com);
        assert_eq!(com_words.len(), pcs.commitment_words());
        assert_eq!(pcs.commitment_from_words(&com_words).unwrap(), com);
        eprintln!(
            "{what}: {} iterations, proof {} words",
            proof.iterations.len(),
            words.len()
        );
    }
}

#[test]
fn inner_products_verify_with_randomized_commitments() {
    let cases = [
        (Q, 1, GAUSS8, Some(0), 46usize),
        (Q, 1, GAUSS8, Some(1), 211),
        (Q, 1, GAUSS8, Some(2), 4000),
        (Q, 1, GAUSS8, None, 4000),
        (Q34, 2, Noise::Digits, None, 211),
        (Q63, 2, GAUSS1, Some(1), 46),
    ];
    for (i, (q, nu, noise, iterations, len)) in cases.into_iter().enumerate() {
        let what = format!("q {q}, nu {nu}, {noise:?}, {iterations:?} iterations, len {len}");
        let field = Fqk::<K>::new(q).unwrap();
        let pcs = HachiPcs::<K>::new(q, hiding(nu, noise, iterations)).unwrap();
        let values = exts_mod(q, "hide/s", i as u64, len);
        let weights = exts_mod(q, "hide/e", i as u64, len);
        let (com, state) = pcs.commit_vector(&values, &seed(3, i)).unwrap();
        let context = [100 + i as u8; 32];
        let (value, proof) = pcs
            .prove_inner_product(&state, &weights, &context, &seed(4, i))
            .unwrap();
        let direct = values
            .iter()
            .zip(&weights)
            .fold(field.zero(), |acc, (v, e)| {
                field.add(&acc, &field.mul(v, e))
            });
        assert_eq!(value, direct, "{what}");
        let schedule = pcs.vector_schedule(len).unwrap();
        assert_eq!(proof.iterations.len(), schedule.iterations.len(), "{what}");
        check_randomness(&pcs, &proof, noise, schedule.final_layout.chunks());
        pcs.verify_inner_product(&com, len, &weights, &value, &proof, &context)
            .unwrap_or_else(|e| panic!("{what}: {e}"));
        let wrong = field.add(&value, &field.one());
        assert!(
            pcs.verify_inner_product(&com, len, &weights, &wrong, &proof, &context)
                .is_err()
        );
        let words = pcs.proof_to_words(&proof);
        assert_eq!(pcs.proof_from_words(&words).unwrap(), proof, "{what}");
        assert_eq!(words.len(), pcs.vector_proof_words(len).unwrap(), "{what}");
        assert_eq!(words.len(), expected_words(&pcs, &proof, noise), "{what}");
    }
}

#[test]
fn the_seeds_randomize_hiding_commitments_and_openings_only() {
    let vars = 8;
    let coeffs = residues_mod(Q, "seeds/f", 0, 1 << vars);
    let point = exts_mod(Q, "seeds/x", 0, vars);
    let context = [7u8; 32];
    let plain = HachiPcs::<K>::new(
        Q,
        HachiChoices {
            fixed_iterations: Some(1),
            ..HachiChoices::toy(16)
        },
    )
    .unwrap();
    let hide = HachiPcs::<K>::new(Q, hiding(1, GAUSS8, Some(1))).unwrap();
    let (a, b) = (seed(9, 0), seed(9, 1));
    // Without hiding the seeds change nothing.
    let (com_a, st_a) = plain.commit_multilinear(&coeffs, &a).unwrap();
    let (com_b, st_b) = plain.commit_multilinear(&coeffs, &b).unwrap();
    assert_eq!((&com_a, &st_a), (&com_b, &st_b));
    let (_, pa) = plain
        .prove_multilinear(&st_a, &point, &context, &a)
        .unwrap();
    let (_, pb) = plain
        .prove_multilinear(&st_a, &point, &context, &b)
        .unwrap();
    assert_eq!(pa, pb);
    assert!(pa.final_randomness.is_empty());
    // With hiding the commitment depends on the seed, and only on it.
    let (hcom_a, hst_a) = hide.commit_multilinear(&coeffs, &a).unwrap();
    let (hcom_a2, hst_a2) = hide.commit_multilinear(&coeffs, &a).unwrap();
    let (hcom_b, _) = hide.commit_multilinear(&coeffs, &b).unwrap();
    assert_eq!((&hcom_a, &hst_a), (&hcom_a2, &hst_a2));
    assert_ne!(hcom_a, hcom_b);
    assert_ne!(hcom_a, com_a);
    // So does an opening: v and the next commitment change with the seed; both verify.
    let (va, ha) = hide
        .prove_multilinear(&hst_a, &point, &context, &a)
        .unwrap();
    let (vb, hb) = hide
        .prove_multilinear(&hst_a, &point, &context, &b)
        .unwrap();
    let (_, ha2) = hide
        .prove_multilinear(&hst_a, &point, &context, &a)
        .unwrap();
    assert_eq!(ha, ha2);
    assert_eq!(va, vb);
    assert_ne!(ha.iterations[0].v, hb.iterations[0].v);
    assert_ne!(
        ha.iterations[0].next_commitment,
        hb.iterations[0].next_commitment
    );
    for p in [&ha, &hb] {
        hide.verify_multilinear(&hcom_a, vars, &point, &va, p, &context)
            .unwrap();
    }
}

#[test]
fn proofs_do_not_verify_across_hiding_choices() {
    let vars = 8;
    let coeffs = residues_mod(Q, "across/f", 0, 1 << vars);
    let point = exts_mod(Q, "across/x", 0, vars);
    let context = [8u8; 32];
    let variants = [
        HachiChoices {
            fixed_iterations: Some(1),
            ..HachiChoices::toy(16)
        },
        hiding(1, GAUSS8, Some(1)),
        hiding(2, GAUSS8, Some(1)),
        hiding(1, GAUSS16, Some(1)),
        hiding(1, Noise::Digits, Some(1)),
    ];
    let schemes: Vec<HachiPcs<K>> = variants
        .iter()
        .map(|c| HachiPcs::<K>::new(Q, c.clone()).unwrap())
        .collect();
    for (i, prover) in schemes.iter().enumerate() {
        let (com, state) = prover.commit_multilinear(&coeffs, &seed(5, i)).unwrap();
        let (value, proof) = prover
            .prove_multilinear(&state, &point, &context, &seed(6, i))
            .unwrap();
        let words = prover.proof_to_words(&proof);
        for (j, verifier) in schemes.iter().enumerate() {
            let accepted = verifier.proof_from_words(&words).is_ok_and(|p| {
                verifier
                    .verify_multilinear(&com, vars, &point, &value, &p, &context)
                    .is_ok()
            });
            assert_eq!(accepted, i == j, "proof of {i} checked by {j}");
        }
    }
}

#[test]
fn every_altered_word_is_rejected_with_hiding() {
    // As the non-hiding test: changing any single word of an encoding, by one bit or by adding
    // one, makes decoding or verification fail (up to 150 words per proof).
    for (iterations, noise) in [(0usize, GAUSS8), (2, GAUSS8), (2, Noise::Digits)] {
        let pcs = HachiPcs::<K>::new(Q, hiding(1, noise, Some(iterations))).unwrap();
        let vars = 8;
        let coeffs = residues_mod(Q, "alter/f", 0, 1 << vars);
        let point = exts_mod(Q, "alter/x", 0, vars);
        let (com, state) = pcs.commit_multilinear(&coeffs, &seed(7, 0)).unwrap();
        let context = [9u8; 32];
        let (value, proof) = pcs
            .prove_multilinear(&state, &point, &context, &seed(8, 0))
            .unwrap();
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
                    "{iterations} iterations, {noise:?}: word {i} of {} altered and accepted",
                    words.len()
                );
                tried += 1;
            }
        }
        // The last words hold the final randomness: alter each of them too.
        for i in words.len().saturating_sub(4)..words.len() {
            let mut altered = words.clone();
            altered[i] ^= 1;
            let ok = pcs.proof_from_words(&altered).is_ok_and(|p| {
                pcs.verify_multilinear(&com, vars, &point, &value, &p, &context)
                    .is_ok()
            });
            assert!(
                !ok,
                "{iterations} iterations: last word {i} altered and accepted"
            );
        }
        eprintln!("{iterations} iterations, {noise:?}: {tried} altered proofs rejected");
    }
}

#[test]
fn malformed_randomness_is_rejected() {
    let vars = 8;
    let coeffs = residues_mod(Q, "bad/f", 0, 1 << vars);
    let point = exts_mod(Q, "bad/x", 0, vars);
    let context = [10u8; 32];
    for (noise, outside) in [(GAUSS8, 150i64), (Noise::Digits, 2)] {
        let pcs = HachiPcs::<K>::new(Q, hiding(1, noise, Some(1))).unwrap();
        let zq = pcs.ctx().ring.zq();
        let (com, state) = pcs.commit_multilinear(&coeffs, &seed(11, 0)).unwrap();
        let (value, proof) = pcs
            .prove_multilinear(&state, &point, &context, &seed(12, 0))
            .unwrap();
        let verify = |p: &Proof<K>| pcs.verify_multilinear(&com, vars, &point, &value, p, &context);
        verify(&proof).unwrap();
        // A coefficient outside the support: the verifier rejects it; for the Gaussian, whose
        // field has room for it (8 bits for 211 values), the decoder rejects it too.
        let mut bad = proof.clone();
        bad.final_randomness[0].0[0] = zq.from_i64(outside);
        assert!(
            matches!(verify(&bad), Err(VerifyError::Check(_))),
            "{noise:?}: {:?}",
            verify(&bad)
        );
        if noise == GAUSS8 {
            assert!(pcs.proof_from_words(&pcs.proof_to_words(&bad)).is_err());
        }
        // One ring element too few or too many.
        let mut short = proof.clone();
        short.final_randomness.pop();
        assert!(matches!(verify(&short), Err(VerifyError::Malformed(_))));
        let mut long = proof.clone();
        long.final_randomness.push(long.final_randomness[0].clone());
        assert!(matches!(verify(&long), Err(VerifyError::Malformed(_))));
        // A header count R off by one is rejected: by the decoder if the length changes or a
        // ring element is left in the padding, else by the verifier. The digits' ring elements
        // take 32 bits, so one more of zero fields can fit in the last word's padding.
        let words = pcs.proof_to_words(&proof);
        let header_r = 6 + proof.iterations.len() - 1;
        assert_eq!(words[header_r], proof.final_randomness.len() as u64);
        for delta in [1u64, u64::MAX] {
            let mut w = words.clone();
            w[header_r] = w[header_r].wrapping_add(delta);
            assert!(
                !pcs.proof_from_words(&w).is_ok_and(|p| verify(&p).is_ok()),
                "{noise:?}: R off by one"
            );
        }
        // Other randomness inside the support does not open the commitment.
        let mut other = proof.clone();
        let x = &mut other.final_randomness[0].0[0];
        *x = if zq.to_centered(*x) == 0 {
            zq.from_i64(-1)
        } else {
            0
        };
        assert!(matches!(verify(&other), Err(VerifyError::Check(_))));
    }
}

#[test]
fn whatever_decodes_reencodes_to_the_same_words_with_hiding() {
    // As the non-hiding test, with the final randomness at the end of the stream.
    for (noise, nu, iterations) in [(GAUSS8, 1, 0usize), (GAUSS8, 1, 2), (Noise::Digits, 2, 1)] {
        let pcs = HachiPcs::<K>::new(Q34, hiding(nu, noise, Some(iterations))).unwrap();
        let vars = 8;
        let (_, state) = pcs
            .commit_multilinear(
                &residues_mod(Q34, "canon/f", iterations as u64, 1 << vars),
                &seed(13, 0),
            )
            .unwrap();
        let point = exts_mod(Q34, "canon/x", iterations as u64, vars);
        let (_, proof) = pcs
            .prove_multilinear(&state, &point, &[8u8; 32], &seed(14, 0))
            .unwrap();
        let words = pcs.proof_to_words(&proof);
        let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ iterations as u64;
        let mut next = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        };
        let (mut decoded, mut rejected) = (0, 0);
        for _ in 0..3000 {
            let mut w = words.clone();
            let i = (next() % w.len() as u64) as usize;
            match next() % 4 {
                0 => w[i] ^= 1 << (next() % 64),
                1 => w[i] ^= (1 << (next() % 64)) | (1 << (next() % 64)),
                2 => w[i] = next(),
                _ => {
                    let j = (next() % 7) as usize;
                    w[j] = if next().is_multiple_of(2) {
                        w[j].wrapping_add(1)
                    } else {
                        w[j].wrapping_sub(1)
                    };
                }
            }
            match pcs.proof_from_words(&w) {
                Ok(p) => {
                    assert_eq!(pcs.proof_to_words(&p), w, "{noise:?}, {iterations} it.");
                    decoded += 1;
                }
                Err(_) => rejected += 1,
            }
        }
        eprintln!("{noise:?}, {iterations} iterations: {decoded} decoded, {rejected} rejected");
        assert!(decoded > 0 && rejected > 0);
    }
}

#[test]
fn the_automatic_schedule_minimises_the_proof_size_with_hiding() {
    let mut cases = 0;
    for q in [Q, Q30, Q34] {
        for (nu, noise) in [(1, GAUSS8), (2, Noise::Digits)] {
            let auto = HachiPcs::<K>::new(q, hiding(nu, noise, None)).unwrap();
            let max = auto.ctx().choices.max_iterations;
            let fixed: Vec<HachiPcs<K>> = (0..=max)
                .map(|k| HachiPcs::<K>::new(q, hiding(nu, noise, Some(k))).unwrap())
                .collect();
            let check = |words: &[Option<usize>], got_words: usize, got_iterations: usize| {
                let least = words.iter().flatten().min().copied().unwrap();
                let first = words.iter().position(|w| *w == Some(least)).unwrap();
                assert_eq!(got_words, least, "q {q}, nu {nu}");
                assert_eq!(got_iterations, first, "q {q}, nu {nu}");
            };
            for vars in [6usize, 10, 15, 17] {
                let words: Vec<_> = fixed
                    .iter()
                    .map(|p| p.multilinear_proof_words(vars).ok())
                    .collect();
                check(
                    &words,
                    auto.multilinear_proof_words(vars).unwrap(),
                    auto.multilinear_schedule(vars).unwrap().iterations.len(),
                );
                cases += 1;
            }
            for len in [1usize, 211, 4000] {
                let words: Vec<_> = fixed
                    .iter()
                    .map(|p| p.vector_proof_words(len).ok())
                    .collect();
                check(
                    &words,
                    auto.vector_proof_words(len).unwrap(),
                    auto.vector_schedule(len).unwrap().iterations.len(),
                );
                cases += 1;
            }
        }
    }
    eprintln!("{cases} cases");
}

#[test]
fn unusable_hiding_choices_are_refused() {
    let toy = HachiChoices::toy_hiding(16);
    assert_eq!(
        toy.hiding,
        Some(Hiding {
            nu: 1,
            noise: Noise::Gaussian { log_sigma: 3 }
        })
    );
    assert!(HachiPcs::<K>::new(Q, toy).is_ok());
    for bad in [
        Hiding {
            nu: 0,
            noise: GAUSS8,
        },
        Hiding {
            nu: 1,
            noise: Noise::Gaussian {
                log_sigma: gauss::MAX_LOG_SIGMA + 1,
            },
        },
        Hiding {
            nu: hachi::params::MAX_NU + 1,
            noise: GAUSS8,
        },
    ] {
        let choices = HachiChoices {
            hiding: Some(bad),
            ..HachiChoices::toy(16)
        };
        assert!(
            matches!(
                HachiPcs::<K>::new(Q, choices),
                Err(hachi::Error::Parameter(_))
            ),
            "{bad:?}"
        );
    }
    // The support must fit the centred residues: q = 101 (5 mod 8) has residues -50..50, so
    // sigma = 8 (support +-105) and sigma = 4 (+-52) are refused, while sigma = 2 (+-26) and the
    // digits are accepted; q = 229 takes sigma = 8 (2 * 105 + 1 = 211 <= 229).
    let at = |q: u64, nu: usize, noise: Noise| {
        HachiPcs::<K>::new(
            q,
            HachiChoices {
                hiding: Some(Hiding { nu, noise }),
                ..HachiChoices::toy(16)
            },
        )
    };
    for noise in [GAUSS8, Noise::Gaussian { log_sigma: 2 }] {
        assert!(
            matches!(at(101, 1, noise), Err(hachi::Error::Parameter(m)) if m.contains("support")),
            "{noise:?} at q = 101"
        );
    }
    for noise in [Noise::Gaussian { log_sigma: 1 }, Noise::Digits] {
        assert!(at(101, 1, noise).is_ok(), "{noise:?} at q = 101");
    }
    assert!(at(229, 1, GAUSS8).is_ok());
    assert!(at(Q, hachi::params::MAX_NU, GAUSS8).is_ok());
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
fn hiding_outputs_are_pinned() {
    // Regression values computed by this implementation (not independent known answers): the
    // SHA-256 of the commitment's and the proof's words at fixed inputs and seeds, for a
    // multilinear opening (Gaussian, one iteration) and an inner product (digits, nu = 2, one
    // iteration). Pinned on 2026-10-21 with the randomized commitments; updated on 2026-10-23
    // for version 3, the input and per-iteration batching sumchecks (the commitments differ
    // only in their version word; both opened proofs carry one iteration).
    let ml = HachiPcs::<K>::new(Q, hiding(1, GAUSS8, Some(1))).unwrap();
    let vars = 8;
    let (com, state) = ml
        .commit_multilinear(&residues_mod(Q, "pin/f", 0, 1 << vars), &[21u8; 32])
        .unwrap();
    let (_, proof) = ml
        .prove_multilinear(
            &state,
            &exts_mod(Q, "pin/x", 0, vars),
            &[22u8; 32],
            &[23u8; 32],
        )
        .unwrap();
    let ip = HachiPcs::<K>::new(Q, hiding(2, Noise::Digits, Some(1))).unwrap();
    let (vcom, vstate) = ip
        .commit_vector(&exts_mod(Q, "pin/s", 0, 211), &[24u8; 32])
        .unwrap();
    let (_, vproof) = ip
        .prove_inner_product(
            &vstate,
            &exts_mod(Q, "pin/e", 0, 211),
            &[25u8; 32],
            &[26u8; 32],
        )
        .unwrap();
    let got = [
        sha256_hex(&ml.commitment_to_words(&com)),
        sha256_hex(&ml.proof_to_words(&proof)),
        sha256_hex(&ip.commitment_to_words(&vcom)),
        sha256_hex(&ip.proof_to_words(&vproof)),
    ];
    let pinned = [
        "add7900e6a5568ddcb3aff60425f21b5cf07f9be785f36b5515d7f64924e09c0",
        "90fed68d9e5909700417a9b5a06e1d5cfc2f60bb6d7d319b6803564cb31fc932",
        "68e24942df1f73be14bc4399c88f31841ca7df126d843a8b59e26052ea01824d",
        "0b169644c4549d2d2e3f1111a7039bac27e5e69602cfb5ca61af57bec5fda6cc",
    ];
    for (i, (g, p)) in got.iter().zip(pinned).enumerate() {
        assert_eq!(g, p, "output {i}");
    }
}
