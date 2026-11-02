//! Merged commitments ([`HachiPcs::commit_merged`]): one commitment to a polynomial over the field
//! and two vectors, and one opening for all three. An option that is not in the draft.
//!
//! What is tested:
//! - openings verify at several widths, ranks, heights, moduli, numbers of iterations and lengths,
//!   give the three values, and have the sizes of the formulas.
//! - each false value is rejected: with the honest ring value at the trace checks, and with a ring
//!   value fitted to its trace check in the iterations, where $\lambda$ combines the claims.
//! - a ring value moved along the kernel of the trace, another point, other weights and another
//!   context are rejected.
//! - commitments and openings are randomized by their seeds.
//! - without hiding the option is refused, and the proofs of the other kinds are not accepted as
//!   merged ones or the other way round.
//! - the encodings round-trip, every altered word is rejected, and the outputs are pinned.

use hachi::{HachiChoices, HachiPcs, Hiding, Proof, VerifyError};
use hachi_co_math::{Ext, Fqk};
use hachi_co_sampling::ShakePrg;
use hachi_co_sampling::uniform::uniform_mod_q;
use sha2::{Digest, Sha256};

const K: usize = 4;
/// The modulus of Hachi-Co's toy-n1 preset (24 bits).
const Q: u64 = 8388637;
/// The modulus of Hachi-Co's toy-n8 and dev-l12 presets (30 bits).
const Q30: u64 = 536871029;
/// The modulus of the secure presets with $\ell = 30$, $\alpha = 8$ (63 bits).
const Q63: u64 = 6618259129420972141;

const CONTEXT: [u8; 32] = [6u8; 32];

fn exts_mod(q: u64, tag: &str, index: u64, len: usize) -> Vec<Ext<K>> {
    uniform_mod_q(
        &mut ShakePrg::new(&[5u8; 32], tag.as_bytes(), index),
        q,
        len * K,
    )
    .unwrap()
    .as_chunks::<K>()
    .0
    .iter()
    .map(|c| Ext(*c))
    .collect()
}

fn seed(tag: u8, index: usize) -> [u8; 32] {
    let mut s = [tag; 32];
    s[..8].copy_from_slice(&(index as u64).to_le_bytes());
    s
}

fn evaluate(field: &Fqk<K>, coeffs: &[Ext<K>], x: &[Ext<K>]) -> Ext<K> {
    let mut mono = vec![field.one()];
    for xj in x {
        mono = mono.iter().flat_map(|m| [*m, field.mul(m, xj)]).collect();
    }
    coeffs.iter().zip(&mono).fold(field.zero(), |acc, (c, m)| {
        field.add(&acc, &field.mul(m, c))
    })
}

fn inner(field: &Fqk<K>, a: &[Ext<K>], b: &[Ext<K>]) -> Ext<K> {
    a.iter().zip(b).fold(field.zero(), |acc, (x, y)| {
        field.add(&acc, &field.mul(x, y))
    })
}

fn masked(nu: usize, n_a: usize, iterations: Option<usize>) -> HachiChoices {
    HachiChoices {
        hiding: Some(Hiding {
            nu,
            n_a,
            lambda: 128,
            phi: 13,
        }),
        fixed_iterations: iterations,
        ..HachiChoices::toy(16)
    }
}

/// The words of a proof's encoding, recomputed from its parts: $6 + k$ header words with hiding,
/// then the packed fields.
fn expected_words(pcs: &HachiPcs<K>, proof: &Proof<K>) -> usize {
    let (c, d) = (&pcs.ctx().choices, pcs.ctx().d());
    let bits_of = |x: u64| (u64::BITS - x.leading_zeros()) as usize;
    let residue = bits_of(pcs.ctx().q - 1);
    let digit = bits_of(c.b - 1);
    let mut residues = d + K * usize::from(proof.mask_eval.is_some());
    residues += K * proof.blinder_mask_evals.len() + d * proof.blinder_ring_values.len();
    for it in &proof.iterations {
        residues += (it.v.len() + it.next_commitment.len()) * d;
        residues += it.rounds.len() * K * (c.b as usize + 2 + 3) + K + K * it.partials.len() + d;
        residues += it.batch_rounds.len() * 2 * K + K;
    }
    let entry = if proof.iterations.is_empty() {
        residue
    } else {
        digit
    };
    6 + proof.iterations.len()
        + (residues * residue + proof.final_message.len() * d * entry).div_ceil(64)
}

/// An honest merged opening.
struct Merged {
    pcs: HachiPcs<K>,
    com: hachi::Commitment,
    vars: usize,
    point: Vec<Ext<K>>,
    weights: [Vec<Ext<K>>; 2],
    values: [Ext<K>; 3],
    proof: Proof<K>,
}

impl Merged {
    fn new(q: u64, choices: HachiChoices, vars: usize, lens: [usize; 2], index: usize) -> Self {
        let pcs = HachiPcs::<K>::new(q, choices).unwrap();
        let field = pcs.ctx().field().clone();
        let coeffs = exts_mod(q, "f", index as u64, 1 << vars);
        let g = [
            exts_mod(q, "g0", index as u64, lens[0]),
            exts_mod(q, "g1", index as u64, lens[1]),
        ];
        let point = exts_mod(q, "x", index as u64, vars);
        let weights = [
            exts_mod(q, "e0", index as u64, lens[0]),
            exts_mod(q, "e1", index as u64, lens[1]),
        ];
        let (com, state) = pcs
            .commit_merged(&coeffs, [&g[0], &g[1]], &seed(1, index))
            .unwrap();
        let (values, proof) = pcs
            .prove_merged(
                &state,
                &point,
                [&weights[0], &weights[1]],
                &CONTEXT,
                &seed(2, index),
            )
            .unwrap();
        assert_eq!(values[0], evaluate(&field, &coeffs, &point), "share value");
        assert_eq!(values[1], inner(&field, &g[0], &weights[0]), "first vector");
        assert_eq!(
            values[2],
            inner(&field, &g[1], &weights[1]),
            "second vector"
        );
        Self {
            pcs,
            com,
            vars,
            point,
            weights,
            values,
            proof,
        }
    }

    fn verify_with(&self, values: &[Ext<K>; 3], proof: &Proof<K>) -> Result<(), VerifyError> {
        self.pcs.verify_merged(
            &self.com,
            self.vars,
            &self.point,
            [&self.weights[0], &self.weights[1]],
            values,
            proof,
            &CONTEXT,
        )
    }

    fn verify(&self, proof: &Proof<K>) -> Result<(), VerifyError> {
        self.verify_with(&self.values, proof)
    }

    fn lens(&self) -> [usize; 2] {
        [self.weights[0].len(), self.weights[1].len()]
    }
}

/// (q, nu, n_A at level 0, fixed iterations, variables, lengths of the vectors).
type Case = (u64, usize, usize, Option<usize>, usize, [usize; 2]);

#[test]
fn merged_openings_verify() {
    // Lengths: Hachi-Co's blinders at l'' = 8 with d_0 = 4, d_1 = 1 (33 and 9), under one block,
    // exactly one block, and longer vectors.
    let cases: [Case; 7] = [
        (Q, 1, 1, None, 8, [33, 9]),
        (Q, 1, 1, Some(1), 6, [1, 4]),
        (Q, 2, 1, Some(2), 8, [60, 25]),
        (Q, 1, 2, None, 8, [33, 9]),
        (Q30, 1, 1, None, 10, [41, 11]),
        (Q30, 2, 2, Some(1), 10, [61, 21]),
        (Q63, 1, 1, Some(1), 10, [41, 11]),
    ];
    for (index, (q, nu, n_a, iterations, vars, lens)) in cases.into_iter().enumerate() {
        let o = Merged::new(q, masked(nu, n_a, iterations), vars, lens, index);
        let what = format!("q {q}, nu {nu}, n_A {n_a}, {iterations:?}, {vars} variables, {lens:?}");
        o.verify(&o.proof).unwrap_or_else(|e| panic!("{what}: {e}"));
        let layout = o.pcs.merged_layout(vars, lens).unwrap();
        let blocks = |len: usize| len.div_ceil(o.pcs.ctx().d_over_k());
        assert_eq!(
            layout.extra,
            blocks(lens[0]) + blocks(lens[1]) + 2,
            "{what}"
        );
        assert!(layout.masked && !o.proof.iterations.is_empty(), "{what}");
        assert_eq!(o.proof.blinder_ring_values.len(), 2, "{what}");
        let words = o.pcs.proof_to_words(&o.proof);
        assert_eq!(words.len(), expected_words(&o.pcs, &o.proof), "{what}");
        assert_eq!(
            words.len(),
            o.pcs.merged_proof_words(vars, lens).unwrap(),
            "{what}"
        );
        assert_eq!(o.pcs.proof_from_words(&words).unwrap(), o.proof, "{what}");
        let field = o.pcs.ctx().field();
        for i in 0..3 {
            let mut wrong = o.values;
            wrong[i] = field.add(&wrong[i], &field.one());
            assert!(
                o.verify_with(&wrong, &o.proof).is_err(),
                "{what}: value {i}"
            );
        }
        let mut other = o.point.clone();
        other[0] = field.add(&other[0], &field.one());
        let w = [&o.weights[0][..], &o.weights[1][..]];
        assert!(
            o.pcs
                .verify_merged(&o.com, vars, &other, w, &o.values, &o.proof, &CONTEXT)
                .is_err(),
            "{what}: point"
        );
        for i in 0..2 {
            let mut weights = o.weights.clone();
            weights[i][0] = field.add(&weights[i][0], &field.one());
            let w = [&weights[0][..], &weights[1][..]];
            assert!(
                o.pcs
                    .verify_merged(&o.com, vars, &o.point, w, &o.values, &o.proof, &CONTEXT)
                    .is_err(),
                "{what}: weights {i}"
            );
        }
        let w = [&o.weights[0][..], &o.weights[1][..]];
        assert!(
            o.pcs
                .verify_merged(&o.com, vars, &o.point, w, &o.values, &o.proof, &[9u8; 32])
                .is_err(),
            "{what}: context"
        );
    }
}

#[test]
fn false_values_are_caught_by_the_trace_checks_or_by_lambda() {
    let o = Merged::new(Q, masked(1, 1, Some(1)), 8, [33, 9], 20);
    let (field, ring) = (o.pcs.ctx().field(), &o.pcs.ctx().ring);
    let trace_check = |r: Result<(), VerifyError>, what: &str| {
        assert!(
            matches!(&r, Err(VerifyError::Check(m)) if m == "the trace of the merged claim"),
            "{what}: {r:?}"
        );
    };
    // A false value with the honest ring values: its trace check fails.
    for i in 0..3 {
        let mut wrong = o.values;
        wrong[i] = field.add(&wrong[i], &field.one());
        trace_check(o.verify_with(&wrong, &o.proof), &format!("value {i}"));
    }
    // A false value with a ring value fitted to its trace check is the unit test
    // `a_false_merged_value_fitted_to_its_trace_check_fails_after_lambda` of `hachi::pcs`: the
    // values enter the transcript before x*, so the fit needs the prover's own challenges.
    // Each masks' value and ring value is checked, and a ring value moved along the kernel of the
    // trace fails in the iterations.
    let mut p = o.proof.clone();
    p.blinder_mask_evals[1] = field.add(&p.blinder_mask_evals[1], &field.one());
    trace_check(o.verify(&p), "y_m of the second vector");
    let mut p = o.proof.clone();
    p.blinder_ring_values[0] = ring
        .add(&p.blinder_ring_values[0], &ring.monomial(1))
        .unwrap();
    let verdict = o.verify(&p);
    assert!(
        verdict.is_err()
            && !matches!(&verdict, Err(VerifyError::Check(m)) if m == "the trace of the merged claim")
    );
    let mut p = o.proof.clone();
    p.blinder_ring_values.clear();
    assert!(matches!(o.verify(&p), Err(VerifyError::Malformed(_))));
}

#[test]
fn merged_commitments_and_openings_are_randomized() {
    let pcs = HachiPcs::<K>::new(Q, masked(1, 1, Some(1))).unwrap();
    let coeffs = exts_mod(Q, "f", 30, 1 << 8);
    let (g0, g1) = (exts_mod(Q, "g0", 30, 33), exts_mod(Q, "g1", 30, 9));
    let (a, state) = pcs.commit_merged(&coeffs, [&g0, &g1], &seed(1, 0)).unwrap();
    let same = pcs
        .commit_merged(&coeffs, [&g0, &g1], &seed(1, 0))
        .unwrap()
        .0;
    let other = pcs
        .commit_merged(&coeffs, [&g0, &g1], &seed(1, 1))
        .unwrap()
        .0;
    assert_eq!(a, same, "determined by the seed");
    assert_ne!(a, other, "randomized by the seed");
    let point = exts_mod(Q, "x", 30, 8);
    let (e0, e1) = (exts_mod(Q, "e0", 30, 33), exts_mod(Q, "e1", 30, 9));
    let open = |s: &[u8; 32]| {
        pcs.prove_merged(&state, &point, [&e0, &e1], &CONTEXT, s)
            .unwrap()
    };
    assert_eq!(open(&seed(2, 0)), open(&seed(2, 0)));
    let ((values, p), (_, q)) = (open(&seed(2, 0)), open(&seed(2, 1)));
    // x* and lambda follow from the statement, so only the masks of the fold are fresh.
    assert_ne!(p.iterations[0].v, q.iterations[0].v, "fresh masks");
    for proof in [&p, &q] {
        pcs.verify_merged(&a, 8, &point, [&e0, &e1], &values, proof, &CONTEXT)
            .unwrap();
    }
}

#[test]
fn merged_commitments_need_hiding_and_kinds_do_not_mix() {
    let plain = HachiPcs::<K>::new(Q, HachiChoices::toy(16)).unwrap();
    assert!(plain.merged_layout(8, [33, 9]).is_err());
    let coeffs = exts_mod(Q, "f", 40, 1 << 8);
    let (g0, g1) = (exts_mod(Q, "g0", 40, 33), exts_mod(Q, "g1", 40, 9));
    let err = plain
        .commit_merged(&coeffs, [&g0, &g1], &seed(1, 0))
        .unwrap_err()
        .to_string();
    assert!(err.contains("hiding"), "{err}");
    // A merged proof is not an opening of the share polynomial, and the share's proof is not a
    // merged opening.
    let o = Merged::new(Q, masked(1, 1, Some(1)), 8, [33, 9], 41);
    assert!(
        o.pcs
            .verify_multilinear_ext(&o.com, 8, &o.point, &o.values[0], &o.proof, &CONTEXT)
            .is_err()
    );
    let (com, state) = o
        .pcs
        .commit_multilinear_ext(&exts_mod(Q, "f", 41, 1 << 8), &seed(1, 41))
        .unwrap();
    let (value, single) = o
        .pcs
        .prove_multilinear_ext(&state, &o.point, &CONTEXT, &seed(2, 41))
        .unwrap();
    let w = [&o.weights[0][..], &o.weights[1][..]];
    let values = [value, o.values[1], o.values[2]];
    assert!(matches!(
        o.pcs
            .verify_merged(&com, 8, &o.point, w, &values, &single, &CONTEXT),
        Err(VerifyError::Malformed(_))
    ));
    // And the merged state is not opened by the other provers.
    assert!(
        o.pcs
            .prove_multilinear_ext(&state, &o.point[..7], &CONTEXT, &seed(2, 0))
            .is_err()
    );
    assert_eq!(o.lens(), [33, 9]);
}

#[test]
fn every_altered_word_is_rejected_and_decodings_reencode() {
    let o = Merged::new(Q, masked(1, 1, Some(1)), 6, [9, 5], 50);
    let words = o.pcs.proof_to_words(&o.proof);
    // The masked word of the header is 3 for a merged opening.
    let header = 6 + o.proof.iterations.len() - 1;
    assert_eq!(words[header], 3);
    let mut checked = 0;
    for i in (0..words.len()).step_by(words.len() / 200 + 1) {
        for bit in [0, 1, 17, 63] {
            let mut bad = words.clone();
            bad[i] ^= 1 << bit;
            if let Ok(proof) = o.pcs.proof_from_words(&bad) {
                assert_eq!(o.pcs.proof_to_words(&proof), bad, "word {i}, bit {bit}");
                assert!(o.verify(&proof).is_err(), "word {i}, bit {bit}");
                checked += 1;
            }
        }
    }
    assert!(checked > 20, "only {checked} words decoded");
    // M = 2 (a merged opening without masks' value) is not an encoding.
    let mut bad = words.clone();
    bad[header] = 2;
    assert!(o.pcs.proof_from_words(&bad).is_err());
}

#[test]
fn a_merged_opening_is_smaller_than_three() {
    // One masked iteration instead of three: the share polynomial and two vectors opened apart.
    for (q, vars, lens) in [(Q, 8usize, [33usize, 9usize]), (Q30, 10, [41, 11])] {
        let pcs = HachiPcs::<K>::new(q, masked(1, 1, None)).unwrap();
        let merged = pcs.merged_proof_words(vars, lens).unwrap();
        let apart = pcs.multilinear_ext_proof_words(vars).unwrap()
            + pcs.vector_proof_words(lens[0]).unwrap()
            + pcs.vector_proof_words(lens[1]).unwrap();
        println!("q {q}, {vars} variables, {lens:?}: merged {merged} words, apart {apart}");
        assert!(merged < apart, "{merged} >= {apart}");
    }
}

fn sha256_hex(words: &[u64]) -> String {
    let mut hasher = Sha256::new();
    for w in words {
        hasher.update(w.to_le_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn outputs_are_pinned() {
    let o = Merged::new(Q, masked(1, 1, Some(1)), 8, [33, 9], 60);
    let commitment = sha256_hex(&o.pcs.commitment_to_words(&o.com));
    let proof = sha256_hex(&o.pcs.proof_to_words(&o.proof));
    println!("{commitment} {proof}");
    assert_eq!(
        commitment,
        "1bd8085c52524acadc26080ad45fcef97f7222ab08fea56b3979272ff70d7bd4"
    );
    assert_eq!(
        proof,
        "2b5d7925a93538b0875cfaf28e31e0836f99e4ce63c5e96189e3e978fb0635c2"
    );
}
