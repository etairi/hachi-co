//! Masking the first iteration ([`hachi::Hiding`], `hachi-pcs.tex`, the blue parts of Appendix
//! A.2).
//!
//! What is tested:
//! - openings of polynomials over the field verify, at several widths, ranks, heights, moduli and
//!   numbers of iterations, and a false value, another point or another context is rejected.
//! - the parameters are those of Setup, recomputed here from the appendix's formulas.
//! - the encodings round-trip and have the sizes of the formulas, recomputed here;
//! - a commitment and an opening are randomized by their seeds and determined by them.
//! - a masked state, and every clone of it, opens once, and an unmasked one opens any number of
//!   times.
//! - a masked witness needs an iteration, and the $\mathbb Z_q$ shortcut is refused.
//! - vectors are masked as the vector chains of Appendix A.2 (one chunk, $\mathbf b = (1)$,
//!   $v = 1$): their openings verify at several lengths, ranks and heights, are randomized by the
//!   seeds, carry the masks' value, which the verifier checks, and reject a false value, other
//!   weights or another context.
//! - proofs do not verify across hiding choices;
//! - every altered word is rejected, and each masked value is checked by the verifier, and whatever
//!   decodes re-encodes to the same words.
//! - the automatic schedule is still the smallest among those with an iteration.
//! - unusable choices are refused, and the outputs are pinned.

use hachi::{Error, HachiChoices, HachiPcs, Hiding, Proof, VerifyError};
use hachi_co_math::{Ext, Fqk};
use hachi_co_sampling::ShakePrg;
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

const CONTEXT: [u8; 32] = [4u8; 32];

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
fn evaluate(field: &Fqk<K>, coeffs: &[Ext<K>], x: &[Ext<K>]) -> Ext<K> {
    let mut mono = vec![field.one()];
    for xj in x {
        mono = mono.iter().flat_map(|m| [*m, field.mul(m, xj)]).collect();
    }
    coeffs.iter().zip(&mono).fold(field.zero(), |acc, (c, m)| {
        field.add(&acc, &field.mul(m, c))
    })
}

/// Toy choices with masks, and the number of iterations fixed if `Some`.
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

/// The words of a proof's encoding, recomputed from its parts (module documentation of
/// `hachi::pcs`, "Encodings"): $6 + k$ header words with hiding, then the packed fields.
fn expected_words(pcs: &HachiPcs<K>, proof: &Proof<K>) -> usize {
    let (c, d) = (&pcs.ctx().choices, pcs.ctx().d());
    let bits_of = |x: u64| (u64::BITS - x.leading_zeros()) as usize;
    let residue = bits_of(pcs.ctx().q - 1);
    let digit = bits_of(c.b - 1);
    let mut residues = K * proof.partials.len() + d;
    residues += 2 * K * proof.batch_rounds.len() + K * usize::from(proof.batch_eval.is_some());
    residues += K * usize::from(proof.mask_eval.is_some());
    for it in &proof.iterations {
        residues += (it.v.len() + it.next_commitment.len()) * d;
        residues += it.rounds.len() * K * (c.b as usize + 2 + 3) + K + K * it.partials.len() + d;
        // The per-iteration batching sumcheck: its rounds (2K each) and the packed value (K).
        residues += it.batch_rounds.len() * 2 * K + K;
    }
    let entry = if proof.iterations.is_empty() {
        residue
    } else {
        digit
    };
    let bits = residues * residue + proof.final_message.len() * d * entry;
    6 + proof.iterations.len() + bits.div_ceil(64)
}

/// An honest opening in `vars` variables.
struct Opening {
    pcs: HachiPcs<K>,
    com: hachi::Commitment,
    vars: usize,
    point: Vec<Ext<K>>,
    value: Ext<K>,
    proof: Proof<K>,
}

impl Opening {
    fn new(q: u64, choices: HachiChoices, vars: usize, index: usize) -> Self {
        let pcs = HachiPcs::<K>::new(q, choices).unwrap();
        let field = pcs.ctx().field().clone();
        let coeffs = exts_mod(q, "f", index as u64, 1 << vars);
        let point = exts_mod(q, "x", index as u64, vars);
        let (com, state) = pcs
            .commit_multilinear_ext(&coeffs, &seed(1, index))
            .unwrap();
        let (value, proof) = pcs
            .prove_multilinear_ext(&state, &point, &CONTEXT, &seed(2, index))
            .unwrap();
        assert_eq!(value, evaluate(&field, &coeffs, &point), "the value");
        Self {
            pcs,
            com,
            vars,
            point,
            value,
            proof,
        }
    }

    fn verify(&self, proof: &Proof<K>) -> Result<(), VerifyError> {
        self.pcs.verify_multilinear_ext(
            &self.com,
            self.vars,
            &self.point,
            &self.value,
            proof,
            &CONTEXT,
        )
    }
}

#[test]
fn masked_openings_verify() {
    let cases: [(u64, usize, usize, Option<usize>, usize); 10] = [
        (Q, 1, 1, Some(1), 6),
        (Q, 1, 1, Some(1), 8),
        (Q, 1, 1, Some(2), 8),
        (Q, 2, 1, Some(1), 8),
        (Q, 1, 2, Some(1), 8),
        (Q, 1, 1, None, 8),
        (Q30, 1, 1, None, 10),
        (Q30, 2, 2, Some(2), 10),
        (Q34, 1, 1, None, 12),
        (Q63, 1, 1, Some(1), 10),
    ];
    for (index, (q, nu, n_a, iterations, vars)) in cases.into_iter().enumerate() {
        let o = Opening::new(q, masked(nu, n_a, iterations), vars, index);
        let what = format!("q {q}, nu {nu}, n_A {n_a}, {iterations:?}, {vars} variables");
        o.verify(&o.proof).unwrap_or_else(|e| panic!("{what}: {e}"));
        assert!(o.proof.mask_eval.is_some(), "{what}");
        let schedule = o.pcs.multilinear_ext_schedule(vars).unwrap();
        assert_eq!(
            o.proof.iterations.len(),
            schedule.iterations.len(),
            "{what}"
        );
        let words = o.pcs.proof_to_words(&o.proof);
        assert_eq!(words.len(), expected_words(&o.pcs, &o.proof), "{what}");
        assert_eq!(
            words.len(),
            o.pcs.multilinear_ext_proof_words(vars).unwrap(),
            "{what}"
        );
        assert_eq!(o.pcs.proof_from_words(&words).unwrap(), o.proof, "{what}");
        // A false value, another point and another context.
        let field = o.pcs.ctx().field();
        let wrong = field.add(&o.value, &field.one());
        assert!(
            o.pcs
                .verify_multilinear_ext(&o.com, vars, &o.point, &wrong, &o.proof, &CONTEXT)
                .is_err(),
            "{what}"
        );
        let mut other = o.point.clone();
        other[0] = field.add(&other[0], &field.one());
        assert!(
            o.pcs
                .verify_multilinear_ext(&o.com, vars, &other, &o.value, &o.proof, &CONTEXT)
                .is_err(),
            "{what}"
        );
        assert!(
            o.pcs
                .verify_multilinear_ext(&o.com, vars, &o.point, &o.value, &o.proof, &[9u8; 32])
                .is_err(),
            "{what}"
        );
    }
}

#[test]
fn the_parameters_are_those_of_setup() {
    // W, N, s, beta, tau-independent mu and the width of D from the formulas of Appendix A.2,
    // recomputed here in floating point.
    for (nu, n_a, vars, q) in [
        (1usize, 1usize, 8usize, Q),
        (2, 1, 10, Q30),
        (3, 2, 12, Q34),
    ] {
        let pcs = HachiPcs::<K>::new(q, masked(nu, n_a, Some(1))).unwrap();
        let schedule = pcs.multilinear_ext_schedule(vars).unwrap();
        let it = &schedule.iterations[0];
        let (c, d) = (&pcs.ctx().choices, pcs.ctx().d());
        let l = it.layout;
        let (delta, chunks) = (it.delta, l.chunks());
        assert!(l.masked && l.decomposed);
        assert_eq!(l.r + l.m + pcs.ctx().alpha(), vars + pcs.ctx().kappa());
        assert_eq!(it.n_a, n_a);
        let width = delta * ((1 << l.m) + 1);
        assert_eq!(l.width(delta), width, "W");
        let n = (width + nu + n_a) * d;
        let target = 13.0 * (n as f64).sqrt() * chunks as f64 * (c.omega * c.b) as f64;
        assert!(it.sigma as f64 >= target, "s {} {target}", it.sigma);
        assert!(
            (it.sigma as f64) < target * 1.05 + 32.0,
            "s {} {target}",
            it.sigma
        );
        assert_eq!(it.sigma % 16, 0);
        let tail = 2.0 * (2.0 * n as f64).ln() + 2.0 * 128.0 * core::f64::consts::LN_2;
        assert!(it.beta as f64 >= it.sigma as f64 * tail.sqrt(), "beta");
        assert_eq!(
            it.mu,
            chunks * delta
                + delta
                + n_a * delta
                + chunks * n_a * delta
                + width * it.tau
                + (nu + n_a) * it.tau,
            "mu"
        );
        assert_eq!(it.d_cols(), delta * (chunks + 1 + n_a), "D");
        assert_eq!(it.rows, c.n_d + c.n_b + 2 + n_a, "rows");
    }
    // At lambda = 128: t = 14, and M = exp(365 / 338) for phi = 13.
    let h = masked(1, 1, None).hiding.unwrap();
    assert_eq!(h.t(), 14);
    assert_eq!(h.log_rejection(), (365, 338));
    // ceil(128 M) + 1 for M = 2.944..., so 378.
    assert_eq!(h.attempts(), 378);
    let high = Hiding { lambda: 256, ..h };
    assert_eq!(high.t(), 19);
}

#[test]
fn the_seeds_randomize_commitments_and_openings() {
    let pcs = HachiPcs::<K>::new(Q, masked(1, 1, Some(1))).unwrap();
    let coeffs = exts_mod(Q, "f", 7, 1 << 8);
    let point = exts_mod(Q, "x", 7, 8);
    let (a, _) = pcs.commit_multilinear_ext(&coeffs, &seed(1, 0)).unwrap();
    let (same, _) = pcs.commit_multilinear_ext(&coeffs, &seed(1, 0)).unwrap();
    let (other, _) = pcs.commit_multilinear_ext(&coeffs, &seed(1, 1)).unwrap();
    assert_eq!(a, same, "determined by the seed");
    assert_ne!(a, other, "randomized by the seed");
    // A masked state opens once, so each opening commits again with the seed of `a`.
    let open = |s: &[u8; 32]| {
        let (_, state) = pcs.commit_multilinear_ext(&coeffs, &seed(1, 0)).unwrap();
        pcs.prove_multilinear_ext(&state, &point, &CONTEXT, s)
            .unwrap()
            .1
    };
    assert_eq!(
        open(&seed(2, 0)),
        open(&seed(2, 0)),
        "determined by the seed"
    );
    let (p, q) = (open(&seed(2, 0)), open(&seed(2, 1)));
    assert_ne!(p.iterations[0].v, q.iterations[0].v, "fresh masks");
    for proof in [&p, &q] {
        let value = evaluate(pcs.ctx().field(), &coeffs, &point);
        pcs.verify_multilinear_ext(&a, 8, &point, &value, proof, &CONTEXT)
            .unwrap();
    }
}

#[test]
fn a_masked_witness_needs_an_iteration() {
    let pcs = HachiPcs::<K>::new(Q, masked(1, 1, Some(0))).unwrap();
    let err = pcs.multilinear_ext_schedule(8).unwrap_err().to_string();
    assert!(err.contains("at least one iteration"), "{err}");
    // The automatic schedule at a size that the schedule without iterations would win.
    let pcs = HachiPcs::<K>::new(Q, masked(1, 1, None)).unwrap();
    assert!(
        !pcs.multilinear_ext_schedule(4)
            .unwrap()
            .iterations
            .is_empty()
    );
}

#[test]
fn the_z_q_shortcut_is_refused() {
    let pcs = HachiPcs::<K>::new(Q, masked(1, 1, Some(1))).unwrap();
    let coeffs = residues_mod(Q, "z_q", 0, 1 << 8);
    let err = pcs
        .commit_multilinear(&coeffs, &seed(1, 0))
        .unwrap_err()
        .to_string();
    assert!(err.contains("over the field"), "{err}");
    // And a verifier with hiding refuses such a proof, whatever it holds.
    let plain = HachiPcs::<K>::new(
        Q,
        HachiChoices {
            fixed_iterations: Some(1),
            ..HachiChoices::toy(16)
        },
    )
    .unwrap();
    let (com, state) = plain.commit_multilinear(&coeffs, &seed(1, 0)).unwrap();
    let point = exts_mod(Q, "x", 0, 8);
    let (value, proof) = plain
        .prove_multilinear(&state, &point, &CONTEXT, &seed(2, 0))
        .unwrap();
    assert!(matches!(
        pcs.verify_multilinear(&com, 8, &point, &value, &proof, &CONTEXT),
        Err(VerifyError::Malformed(_))
    ));
}

/// An honest inner-product opening of a vector of `len` elements.
struct VectorOpening {
    pcs: HachiPcs<K>,
    com: hachi::Commitment,
    len: usize,
    weights: Vec<Ext<K>>,
    value: Ext<K>,
    proof: Proof<K>,
}

impl VectorOpening {
    fn new(q: u64, choices: HachiChoices, len: usize, index: usize) -> Self {
        let pcs = HachiPcs::<K>::new(q, choices).unwrap();
        let field = pcs.ctx().field().clone();
        let values = exts_mod(q, "s", index as u64, len);
        let weights = exts_mod(q, "e", index as u64, len);
        let (com, state) = pcs.commit_vector(&values, &seed(3, index)).unwrap();
        let (value, proof) = pcs
            .prove_inner_product(&state, &weights, &CONTEXT, &seed(4, index))
            .unwrap();
        let direct = values
            .iter()
            .zip(&weights)
            .fold(field.zero(), |acc, (v, e)| {
                field.add(&acc, &field.mul(v, e))
            });
        assert_eq!(value, direct, "the value");
        Self {
            pcs,
            com,
            len,
            weights,
            value,
            proof,
        }
    }

    fn verify(&self, proof: &Proof<K>) -> Result<(), VerifyError> {
        self.pcs.verify_inner_product(
            &self.com,
            self.len,
            &self.weights,
            &self.value,
            proof,
            &CONTEXT,
        )
    }
}

#[test]
fn masked_vector_openings_verify() {
    // Lengths: under one block of d/K = 4, exactly one block, 15 blocks padded to 16, exactly 16,
    // and the lengths 1 + 12 d_b of Hachi-Co's blinders at l'' = 12 and d_b = 3 and 1.
    let cases: [(u64, usize, usize, Option<usize>, usize); 9] = [
        (Q, 1, 1, None, 1),
        (Q, 1, 1, None, 4),
        (Q, 1, 1, None, 60),
        (Q, 1, 1, Some(2), 64),
        (Q, 2, 1, None, 37),
        (Q, 1, 2, None, 13),
        (Q30, 1, 1, None, 37),
        (Q30, 2, 2, Some(1), 60),
        (Q63, 1, 1, Some(1), 13),
    ];
    for (index, (q, nu, n_a, iterations, len)) in cases.into_iter().enumerate() {
        let o = VectorOpening::new(q, masked(nu, n_a, iterations), len, index);
        let what = format!("q {q}, nu {nu}, n_A {n_a}, {iterations:?}, length {len}");
        o.verify(&o.proof).unwrap_or_else(|e| panic!("{what}: {e}"));
        let layout = o.pcs.vector_layout(len).unwrap();
        assert!(layout.masked && layout.r == 0, "{what}");
        assert!(o.proof.mask_eval.is_some(), "{what}");
        assert!(!o.proof.iterations.is_empty(), "{what}: sent in the clear");
        let schedule = o.pcs.vector_schedule(len).unwrap();
        assert_eq!(
            o.proof.iterations.len(),
            schedule.iterations.len(),
            "{what}"
        );
        let words = o.pcs.proof_to_words(&o.proof);
        assert_eq!(words.len(), expected_words(&o.pcs, &o.proof), "{what}");
        assert_eq!(
            words.len(),
            o.pcs.vector_proof_words(len).unwrap(),
            "{what}"
        );
        assert_eq!(o.pcs.proof_from_words(&words).unwrap(), o.proof, "{what}");
        // A false value, other weights and another context.
        let field = o.pcs.ctx().field();
        let wrong = field.add(&o.value, &field.one());
        assert!(
            o.pcs
                .verify_inner_product(&o.com, len, &o.weights, &wrong, &o.proof, &CONTEXT)
                .is_err(),
            "{what}"
        );
        let mut other = o.weights.clone();
        other[0] = field.add(&other[0], &field.one());
        assert!(
            o.pcs
                .verify_inner_product(&o.com, len, &other, &o.value, &o.proof, &CONTEXT)
                .is_err(),
            "{what}"
        );
        assert!(
            o.pcs
                .verify_inner_product(&o.com, len, &o.weights, &o.value, &o.proof, &[9u8; 32])
                .is_err(),
            "{what}"
        );
    }
}

#[test]
fn vector_commitments_and_openings_are_randomized() {
    let pcs = HachiPcs::<K>::new(Q, masked(1, 1, None)).unwrap();
    let plain = HachiPcs::<K>::new(Q, HachiChoices::toy(16)).unwrap();
    let values = exts_mod(Q, "s", 30, 37);
    let weights = exts_mod(Q, "e", 30, 37);
    let (a, _) = pcs.commit_vector(&values, &seed(3, 0)).unwrap();
    assert_eq!(
        a,
        pcs.commit_vector(&values, &seed(3, 0)).unwrap().0,
        "determined"
    );
    assert_ne!(
        a,
        pcs.commit_vector(&values, &seed(3, 1)).unwrap().0,
        "randomized"
    );
    assert_ne!(
        a,
        plain.commit_vector(&values, &seed(3, 0)).unwrap().0,
        "masked"
    );
    // A masked state opens once, so each opening commits again with the seed of `a`.
    let open = |s: &[u8; 32]| {
        let (_, state) = pcs.commit_vector(&values, &seed(3, 0)).unwrap();
        pcs.prove_inner_product(&state, &weights, &CONTEXT, s)
            .unwrap()
    };
    assert_eq!(
        open(&seed(4, 0)),
        open(&seed(4, 0)),
        "determined by the seed"
    );
    let ((value, p), (_, q)) = (open(&seed(4, 0)), open(&seed(4, 1)));
    assert_ne!(p.iterations[0].v, q.iterations[0].v, "fresh masks");
    for proof in [&p, &q] {
        pcs.verify_inner_product(&a, 37, &weights, &value, proof, &CONTEXT)
            .unwrap();
    }
}

#[test]
fn each_masked_value_of_a_vector_opening_is_checked() {
    let o = VectorOpening::new(Q, masked(1, 1, Some(1)), 37, 40);
    let (field, ring) = (o.pcs.ctx().field(), &o.pcs.ctx().ring);
    let check = |proof: &Proof<K>, what: &str| {
        let verdict = o.verify(proof);
        assert!(
            matches!(&verdict, Err(VerifyError::Check(m)) if m == "the trace of the masked claim"),
            "{what}: {verdict:?}"
        );
    };
    let mut p = o.proof.clone();
    p.mask_eval = Some(field.add(&p.mask_eval.unwrap(), &field.one()));
    check(&p, "y_m");
    let mut p = o.proof.clone();
    p.ring_value = ring.add(&p.ring_value, &ring.monomial(0)).unwrap();
    check(&p, "Y*");
    let mut p = o.proof.clone();
    p.mask_eval = None;
    assert!(matches!(o.verify(&p), Err(VerifyError::Malformed(_))));
    // Y* moved along the kernel of the trace passes step 1 and fails in the iterations.
    let mut p = o.proof.clone();
    p.ring_value = ring.add(&p.ring_value, &ring.monomial(1)).unwrap();
    assert!(o.verify(&p).is_err(), "Y* in the kernel of the trace");
    // A vector proof without masks is refused by a verifier with hiding, and the other way round.
    let plain = VectorOpening::new(
        Q,
        HachiChoices {
            fixed_iterations: Some(1),
            ..HachiChoices::toy(16)
        },
        37,
        40,
    );
    assert!(o.verify(&plain.proof).is_err());
    assert!(plain.verify(&o.proof).is_err());
}

#[test]
fn proofs_do_not_verify_across_hiding_choices() {
    let o = Opening::new(Q, masked(1, 1, Some(1)), 8, 11);
    o.verify(&o.proof).unwrap();
    for choices in [
        masked(2, 1, Some(1)),
        masked(1, 2, Some(1)),
        HachiChoices {
            hiding: Some(Hiding {
                nu: 1,
                n_a: 1,
                lambda: 128,
                phi: 14,
            }),
            ..masked(1, 1, Some(1))
        },
        HachiChoices {
            hiding: Some(Hiding {
                nu: 1,
                n_a: 1,
                lambda: 100,
                phi: 13,
            }),
            ..masked(1, 1, Some(1))
        },
        HachiChoices {
            fixed_iterations: Some(1),
            ..HachiChoices::toy(16)
        },
    ] {
        let pcs = HachiPcs::<K>::new(Q, choices.clone()).unwrap();
        assert!(
            pcs.verify_multilinear_ext(&o.com, 8, &o.point, &o.value, &o.proof, &CONTEXT)
                .is_err(),
            "{:?}",
            choices.hiding
        );
    }
}

#[test]
fn each_masked_value_is_checked() {
    let o = Opening::new(Q, masked(1, 1, Some(1)), 8, 12);
    let (field, ring) = (o.pcs.ctx().field(), &o.pcs.ctx().ring);
    let check = |proof: &Proof<K>, message: &str, what: &str| {
        let verdict = o.verify(proof);
        assert!(
            matches!(&verdict, Err(VerifyError::Check(m)) if m == message),
            "{what}: {verdict:?}"
        );
    };
    let mut p = o.proof.clone();
    let y = p.mask_eval.unwrap();
    p.mask_eval = Some(field.add(&y, &field.one()));
    check(&p, "the trace of the masked claim", "y_m");
    let mut p = o.proof.clone();
    p.ring_value = ring.add(&p.ring_value, &ring.monomial(0)).unwrap();
    check(&p, "the trace of the masked claim", "Y*");
    let mut p = o.proof.clone();
    p.mask_eval = None;
    assert!(matches!(o.verify(&p), Err(VerifyError::Malformed(_))));
    // A y_m and a Y* that pass the trace check for another mask value still fail in the iterations:
    // Y* moved along the kernel of the trace is a false claim for the first fold.
    let mut p = o.proof.clone();
    let shift = ring.monomial(1);
    let traced = o.pcs.ctx().sub.e_map(&shift, &ring.monomial(0)).unwrap();
    assert_eq!(traced, ring.zero(), "X lies in the kernel of the trace");
    p.ring_value = ring.add(&p.ring_value, &shift).unwrap();
    assert!(o.verify(&p).is_err(), "Y* in the kernel of the trace");
}

#[test]
fn every_altered_word_is_rejected() {
    let o = Opening::new(Q, masked(1, 1, Some(1)), 6, 13);
    let words = o.pcs.proof_to_words(&o.proof);
    let mut checked = 0;
    for i in (0..words.len()).step_by(words.len() / 200 + 1) {
        let mut bad = words.clone();
        bad[i] ^= 1;
        if let Ok(proof) = o.pcs.proof_from_words(&bad) {
            assert!(o.verify(&proof).is_err(), "word {i}");
            checked += 1;
        }
    }
    assert!(checked > 20, "only {checked} words decoded");
}

#[test]
fn whatever_decodes_reencodes_to_the_same_words() {
    let o = Opening::new(Q, masked(1, 1, Some(1)), 6, 14);
    let words = o.pcs.proof_to_words(&o.proof);
    for i in 0..words.len() {
        for bit in [0, 1, 17, 63] {
            let mut bad = words.clone();
            bad[i] ^= 1 << bit;
            if let Ok(proof) = o.pcs.proof_from_words(&bad) {
                assert_eq!(o.pcs.proof_to_words(&proof), bad, "word {i}, bit {bit}");
            }
        }
    }
    // The masked word of the header is a bit, and a masked proof needs hiding choices.
    let plain = HachiPcs::<K>::new(Q, HachiChoices::toy(16)).unwrap();
    assert!(plain.proof_from_words(&words).is_err());
}

#[test]
fn the_automatic_schedule_is_the_smallest() {
    for vars in [6usize, 8, 10] {
        let automatic = HachiPcs::<K>::new(Q30, masked(1, 1, None)).unwrap();
        let words = automatic.multilinear_ext_proof_words(vars).unwrap();
        for n in 1..=4 {
            let Ok(fixed) = HachiPcs::<K>::new(Q30, masked(1, 1, Some(n))) else {
                continue;
            };
            if let Ok(w) = fixed.multilinear_ext_proof_words(vars) {
                assert!(words <= w, "{vars} variables, {n} iterations");
            }
        }
    }
}

#[test]
fn unusable_choices_are_refused() {
    let base = Hiding {
        nu: 1,
        n_a: 1,
        lambda: 128,
        phi: 13,
    };
    for hiding in [
        Hiding { nu: 0, ..base },
        Hiding { n_a: 0, ..base },
        Hiding { lambda: 0, ..base },
        Hiding { phi: 0, ..base },
        Hiding {
            nu: hachi::params::MAX_NU + 1,
            ..base
        },
    ] {
        let choices = HachiChoices {
            hiding: Some(hiding),
            ..HachiChoices::toy(16)
        };
        assert!(HachiPcs::<K>::new(Q, choices).is_err(), "{hiding:?}");
    }
    // b <= sqrt(q / 2) and omega < sqrt(q) / (2 sqrt(2)), Setup item 1(b).
    for (b, omega) in [(2049u64, 2u64), (4, 1025)] {
        let choices = HachiChoices {
            b,
            omega,
            ..masked(1, 1, None)
        };
        assert!(
            HachiPcs::<K>::new(Q, choices).is_err(),
            "b {b}, omega {omega}"
        );
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
    // Regression values of this implementation. Version 6 left the opening of a polynomial over
    // the field as it was: its pins of version 5 held before the version word moved. Updated on
    // 2026-11-02 for the 192-bit tables of the Gaussian sampler: the commitments held, and the
    // proofs changed with their masks.
    let o = Opening::new(Q, masked(1, 1, Some(2)), 8, 20);
    let commitment = sha256_hex(&o.pcs.commitment_to_words(&o.com));
    let proof = sha256_hex(&o.pcs.proof_to_words(&o.proof));
    let v = VectorOpening::new(Q, masked(1, 1, Some(1)), 37, 21);
    let vector_commitment = sha256_hex(&v.pcs.commitment_to_words(&v.com));
    let vector_proof = sha256_hex(&v.pcs.proof_to_words(&v.proof));
    println!("{commitment} {proof} {vector_commitment} {vector_proof}");
    assert_eq!(
        commitment,
        "206ee272e3358fe7bbc8c0c6def6f0442c2fc08e542594021688d52aebf15c26"
    );
    assert_eq!(
        proof,
        "81951a6db2944f4e74922758e1ba0e60bab09f87a038a0a6b10a6c8377419b65"
    );
    assert_eq!(
        vector_commitment,
        "afb30fd6f54cfeb87a8be627c6c70f291f5d5d5e62152246d422c9dbc3be3142"
    );
    assert_eq!(
        vector_proof,
        "8b456dfa21f6e6a70c16e54a9ef4ada15a69714e11e97eecc907cd3971f8e3cd"
    );
}

/// Whether `result` is the refusal of a second opening of a masked commitment.
fn refused<T>(result: Result<T, Error>) -> bool {
    matches!(result, Err(Error::Protocol(m)) if m.contains("opens once"))
}

#[test]
fn a_masked_state_opens_once() {
    // Appendix A.2 opens a commitment once (its masking rows hide one evaluation), so a masked
    // state, and every clone of it, opens once. An opening refused for its inputs does not count.
    // Without masks a state opens any number of times.
    let pcs = HachiPcs::<K>::new(Q, masked(1, 1, Some(1))).unwrap();
    let pcs_vectors = HachiPcs::<K>::new(Q, masked(1, 1, None)).unwrap();
    let plain = HachiPcs::<K>::new(Q, HachiChoices::toy(16)).unwrap();
    let coeffs = exts_mod(Q, "f", 50, 1 << 8);
    let point = exts_mod(Q, "x", 50, 8);
    let (values, weights) = (exts_mod(Q, "s", 50, 37), exts_mod(Q, "e", 50, 37));
    // A polynomial over the field.
    let (_, state) = pcs.commit_multilinear_ext(&coeffs, &seed(1, 50)).unwrap();
    let copy = state.clone();
    assert!(matches!(
        pcs.prove_multilinear_ext(&state, &point[..7], &CONTEXT, &seed(2, 50)),
        Err(Error::Shape(_))
    ));
    pcs.prove_multilinear_ext(&state, &point, &CONTEXT, &seed(2, 50))
        .unwrap();
    assert!(refused(pcs.prove_multilinear_ext(
        &state,
        &point,
        &CONTEXT,
        &seed(2, 51)
    )));
    assert!(refused(pcs.prove_multilinear_ext(
        &copy,
        &point,
        &CONTEXT,
        &seed(2, 52)
    )));
    // A vector.
    let (_, state) = pcs_vectors.commit_vector(&values, &seed(3, 50)).unwrap();
    assert!(matches!(
        pcs_vectors.prove_inner_product(&state, &weights[1..], &CONTEXT, &seed(4, 50)),
        Err(Error::Shape(_))
    ));
    pcs_vectors
        .prove_inner_product(&state, &weights, &CONTEXT, &seed(4, 50))
        .unwrap();
    assert!(refused(pcs_vectors.prove_inner_product(
        &state,
        &weights,
        &CONTEXT,
        &seed(4, 51)
    )));
    // Without masks.
    let (_, state) = plain.commit_multilinear_ext(&coeffs, &seed(1, 50)).unwrap();
    for s in [50, 51] {
        plain
            .prove_multilinear_ext(&state, &point, &CONTEXT, &seed(2, s))
            .unwrap();
    }
    let (_, state) = plain.commit_vector(&values, &seed(3, 50)).unwrap();
    for s in [50, 51] {
        plain
            .prove_inner_product(&state, &weights, &CONTEXT, &seed(4, s))
            .unwrap();
    }
}
