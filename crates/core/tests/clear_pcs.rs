//! `ClearPcs`, the reference base PCS: completeness for both kinds of openings (multilinear
//! evaluations against the multilinear extension of the hypercube values, inner products against
//! the direct sum), the word encodings (round trips, the sizes of the formulas, and every
//! malformed encoding refused without a panic, with a seeded fuzz loop), and the failure
//! boundary of the verifier: a wrong value, context, commitment, kind, size, a tampered or
//! non-canonical residue.
//!
//! Seeds: the splitmix64 streams below (seeds in each test).
#![cfg(feature = "fixtures")]

use hachi_co_core::base_pcs::{evaluate_multilinear, inner_product, mobius};
use hachi_co_core::{
    BasePcs, ClearCommitment, ClearOpening, ClearPcs, ClearPolynomial, Error, Ext, Fqk,
};

/// The base PCS's seed: `ClearPcs` is deterministic and ignores it.
const SEED: [u8; 32] = [0; 32];

const K: usize = 4;
/// The toy-n1 modulus.
const Q: u64 = 8388637;

type Pcs = ClearPcs<K>;

struct Stream(u64);

impl Stream {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn residues(&mut self, len: usize) -> Vec<u64> {
        (0..len).map(|_| self.next() % Q).collect()
    }

    fn ext(&mut self) -> Ext<K> {
        Ext(core::array::from_fn(|_| self.next() % Q))
    }

    fn exts(&mut self, len: usize) -> Vec<Ext<K>> {
        (0..len).map(|_| self.ext()).collect()
    }

    fn context(&mut self) -> [u8; 32] {
        let mut out = [0u8; 32];
        for chunk in out.as_chunks_mut::<8>().0 {
            *chunk = self.next().to_le_bytes();
        }
        out
    }
}

/// The multilinear extension of hypercube values at a point, from its definition
/// ($\sum\_x \prod\_j (x\_j a\_j + (1 - x\_j)(1 - a\_j)) f(x)$, $x\_1$ the most significant bit).
fn mle(field: &Fqk<K>, values: &[u64], point: &[Ext<K>]) -> Ext<K> {
    let ell = point.len();
    let one = field.one();
    values.iter().enumerate().fold(field.zero(), |acc, (x, v)| {
        let eq = (0..ell).fold(one, |e, j| {
            let bit = (x >> (ell - 1 - j)) & 1 == 1;
            let a = &point[j];
            field.mul(&e, &if bit { *a } else { field.sub(&one, a) })
        });
        field.add(&acc, &field.scale(&eq, *v))
    })
}

/// A multilinear commitment to hypercube values (through the Möbius transform) and an honest
/// opening at a random point.
fn multilinear_instance(
    pcs: &Pcs,
    rng: &mut Stream,
    ell: usize,
) -> (
    Vec<u64>,
    Vec<Ext<K>>,
    ClearCommitment,
    Ext<K>,
    ClearOpening<K>,
    [u8; 32],
) {
    let values = rng.residues(1 << ell);
    let mut coeffs = values.clone();
    mobius(pcs.field().zq(), &mut coeffs).unwrap();
    let (com, st) = pcs.commit_multilinear(&coeffs, &SEED).unwrap();
    let point = rng.exts(ell);
    let context = rng.context();
    let (value, proof) = pcs.prove_multilinear(&st, &point, &context, &SEED).unwrap();
    (values, point, com, value, proof, context)
}

#[test]
fn honest_multilinear_openings_verify() {
    let pcs = Pcs::new(Q).unwrap();
    let field = pcs.field().clone();
    let mut rng = Stream(0xc1ea_0001);
    for ell in [0usize, 1, 3, 6, 10] {
        let (values, point, com, value, proof, context) = multilinear_instance(&pcs, &mut rng, ell);
        assert_eq!(value, mle(&field, &values, &point), "l = {ell}");
        pcs.verify_multilinear(&com, ell, &point, &value, &proof, &context)
            .unwrap();
        // The words round-trip, with the sizes of the formulas.
        let words = pcs.commitment_to_words(&com);
        assert_eq!(words.len(), Pcs::COMMITMENT_WORDS);
        assert_eq!(pcs.commitment_from_words(&words).unwrap(), com);
        let words = pcs.proof_to_words(&proof);
        assert_eq!(Some(words.len()), Pcs::multilinear_proof_words(ell));
        assert_eq!(pcs.proof_from_words(&words).unwrap(), proof);
        // A non-canonical value stands for its residue class.
        let mut unreduced = value;
        unreduced.0[1] += Q;
        pcs.verify_multilinear(&com, ell, &point, &unreduced, &proof, &context)
            .unwrap();
    }
}

#[test]
fn honest_inner_product_openings_verify() {
    let pcs = Pcs::new(Q).unwrap();
    let field = pcs.field().clone();
    let mut rng = Stream(0xc1ea_0002);
    for len in [0usize, 1, 7, 61] {
        let v = rng.exts(len);
        let weights = rng.exts(len);
        let context = rng.context();
        let (com, st) = pcs.commit_vector(&v, &SEED).unwrap();
        let (value, proof) = pcs
            .prove_inner_product(&st, &weights, &context, &SEED)
            .unwrap();
        let direct = v.iter().zip(&weights).fold(field.zero(), |acc, (x, e)| {
            field.add(&acc, &field.mul(x, e))
        });
        assert_eq!(value, direct, "len = {len}");
        assert_eq!(value, inner_product(&field, &v, &weights).unwrap());
        pcs.verify_inner_product(&com, len, &weights, &value, &proof, &context)
            .unwrap();
        let words = pcs.proof_to_words(&proof);
        assert_eq!(Some(words.len()), Pcs::vector_proof_words(len));
        assert_eq!(pcs.proof_from_words(&words).unwrap(), proof);
    }
}

/// Inputs are reduced modulo q: an unreduced vector commits like its residues.
#[test]
fn commitments_are_to_residue_classes() {
    let pcs = Pcs::new(Q).unwrap();
    let mut rng = Stream(0xc1ea_0003);
    let coeffs = rng.residues(8);
    let shifted: Vec<u64> = coeffs.iter().map(|c| c + Q).collect();
    assert_eq!(
        pcs.commit_multilinear(&coeffs, &SEED).unwrap(),
        pcs.commit_multilinear(&shifted, &SEED).unwrap()
    );
    let v = rng.exts(3);
    let shifted: Vec<Ext<K>> = v.iter().map(|x| Ext(x.0.map(|c| c + 2 * Q))).collect();
    assert_eq!(
        pcs.commit_vector(&v, &SEED).unwrap(),
        pcs.commit_vector(&shifted, &SEED).unwrap()
    );
}

/// The hash separates the kinds, the moduli and the lengths: the same residues as four
/// coefficients, as one field element, under another q, or with a zero appended commit to
/// different values.
#[test]
fn commitments_are_domain_separated() {
    let pcs = Pcs::new(Q).unwrap();
    let other = Pcs::new(33554501).unwrap();
    let residues = vec![1u64, 2, 3, 4];
    let (multilinear, _) = pcs.commit_multilinear(&residues, &SEED).unwrap();
    let (vector, _) = pcs.commit_vector(&[Ext([1, 2, 3, 4])], &SEED).unwrap();
    let (other_q, _) = other.commit_multilinear(&residues, &SEED).unwrap();
    let mut longer = residues.clone();
    longer.extend([0, 0, 0, 0]);
    let (longer, _) = pcs.commit_multilinear(&longer, &SEED).unwrap();
    let all = [multilinear, vector, other_q, longer];
    for i in 0..all.len() {
        for j in i + 1..all.len() {
            assert_ne!(all[i], all[j], "{i} vs {j}");
        }
    }
}

/// Every single change the verifier must refuse, for both kinds: the value (each coordinate +1),
/// the context (each byte), the commitment (each byte), every residue of the opening (+1, and
/// +q, non-canonical), the size and the point.
#[test]
fn the_verifier_refuses_every_change() {
    let pcs = Pcs::new(Q).unwrap();
    let mut rng = Stream(0xc1ea_0004);
    let ell = 4;
    let (_, point, com, value, proof, context) = multilinear_instance(&pcs, &mut rng, ell);
    let check = |com: &ClearCommitment,
                 num_vars: usize,
                 point: &[Ext<K>],
                 value: &Ext<K>,
                 proof: &ClearOpening<K>,
                 context: &[u8; 32]| {
        pcs.verify_multilinear(com, num_vars, point, value, proof, context)
    };
    assert!(check(&com, ell, &point, &value, &proof, &context).is_ok());
    for c in 0..K {
        let mut v = value;
        v.0[c] = (v.0[c] + 1) % Q;
        assert!(check(&com, ell, &point, &v, &proof, &context).is_err());
    }
    for byte in 0..32 {
        let mut ctx = context;
        ctx[byte] ^= 1;
        assert!(check(&com, ell, &point, &value, &proof, &ctx).is_err());
        let mut c = com;
        c.0[byte] ^= 1;
        assert!(check(&c, ell, &point, &value, &proof, &context).is_err());
    }
    let ClearPolynomial::Multilinear(coeffs) = &proof.polynomial else {
        unreachable!()
    };
    for i in 0..coeffs.len() {
        for delta in [1, Q] {
            let mut p = proof.clone();
            let ClearPolynomial::Multilinear(c) = &mut p.polynomial else {
                unreachable!()
            };
            c[i] = if delta == Q { c[i] + Q } else { (c[i] + 1) % Q };
            let err = check(&com, ell, &point, &value, &p, &context).unwrap_err();
            if delta == Q {
                assert!(err.contains("not below q"), "{err}");
            }
        }
    }
    // Another size, another point length, the opening of another polynomial.
    assert!(check(&com, ell + 1, &point, &value, &proof, &context).is_err());
    assert!(check(&com, ell, &point[1..], &value, &proof, &context).is_err());
    let (_, point2, com2, value2, proof2, context2) = multilinear_instance(&pcs, &mut rng, ell);
    pcs.verify_multilinear(&com2, ell, &point2, &value2, &proof2, &context2)
        .unwrap();
    assert!(check(&com, ell, &point2, &value2, &proof2, &context2).is_err());
    let mut moved = proof2.clone();
    moved.context = context;
    assert!(check(&com, ell, &point, &value, &moved, &context).is_err());

    // Inner products.
    let v = rng.exts(9);
    let weights = rng.exts(9);
    let (vcom, st) = pcs.commit_vector(&v, &SEED).unwrap();
    let (vvalue, vproof) = pcs
        .prove_inner_product(&st, &weights, &context, &SEED)
        .unwrap();
    let vcheck = |len: usize, weights: &[Ext<K>], value: &Ext<K>, proof: &ClearOpening<K>| {
        pcs.verify_inner_product(&vcom, len, weights, value, proof, &context)
    };
    assert!(vcheck(9, &weights, &vvalue, &vproof).is_ok());
    let ClearPolynomial::Vector(entries) = &vproof.polynomial else {
        unreachable!()
    };
    for i in 0..entries.len() {
        for c in 0..K {
            let tampered = |value: u64| {
                let mut p = vproof.clone();
                if let ClearPolynomial::Vector(e) = &mut p.polynomial {
                    e[i].0[c] = value;
                }
                p
            };
            let original = entries[i].0[c];
            assert!(vcheck(9, &weights, &vvalue, &tampered((original + 1) % Q)).is_err());
            assert!(vcheck(9, &weights, &vvalue, &tampered(original + Q)).is_err());
        }
    }
    let mut w = weights.clone();
    w[3].0[0] = (w[3].0[0] + 1) % Q;
    assert!(vcheck(9, &w, &vvalue, &vproof).is_err());
    assert!(vcheck(8, &weights[1..], &vvalue, &vproof).is_err());
    assert!(vcheck(9, &weights[1..], &vvalue, &vproof).is_err());
    let mut wrong = vvalue;
    wrong.0[3] = (wrong.0[3] + 1) % Q;
    assert!(vcheck(9, &weights, &wrong, &vproof).is_err());
    // The other kind: a vector opening checked as a multilinear one and conversely.
    assert!(
        pcs.verify_multilinear(&vcom, 0, &[], &vvalue, &vproof, &context)
            .is_err()
    );
    assert!(
        pcs.verify_inner_product(&com, 16, &rng.exts(16), &value, &proof, &context)
            .is_err()
    );
}

#[test]
fn the_provers_refuse_misuse() {
    let pcs = Pcs::new(Q).unwrap();
    let context = [7u8; 32];
    for len in [0usize, 3, 6, 12] {
        assert!(matches!(
            pcs.commit_multilinear(&vec![1; len], &SEED),
            Err(Error::Shape { .. })
        ));
    }
    let (_, ml) = pcs.commit_multilinear(&[1, 2, 3, 4], &SEED).unwrap();
    let (_, vec_state) = pcs.commit_vector(&[Ext([1, 2, 3, 4])], &SEED).unwrap();
    let one = pcs.field().one();
    assert!(matches!(
        pcs.prove_multilinear(&vec_state, &[], &context, &SEED),
        Err(Error::BasePcs(_))
    ));
    assert!(matches!(
        pcs.prove_inner_product(&ml, &[one; 4], &context, &SEED),
        Err(Error::BasePcs(_))
    ));
    assert!(matches!(
        pcs.prove_multilinear(&ml, &[one; 3], &context, &SEED),
        Err(Error::Shape { .. })
    ));
    assert!(matches!(
        pcs.prove_inner_product(&vec_state, &[one; 2], &context, &SEED),
        Err(Error::Shape { .. })
    ));
    assert!(matches!(Pcs::new(8388636), Err(Error::Math(_))));
    assert!(matches!(ClearPcs::<3>::new(Q), Err(Error::Math(_))));
    assert_eq!(Pcs::multilinear_proof_words(64), None);
    assert_eq!(Pcs::vector_proof_words(usize::MAX), None);
}

/// Malformed words are errors, never panics; whatever decodes re-encodes to the same words.
#[test]
fn malformed_words_are_errors() {
    let pcs = Pcs::new(Q).unwrap();
    for len in [0usize, 1, 3, 5, 8] {
        assert!(matches!(
            pcs.commitment_from_words(&vec![1; len]),
            Err(Error::BasePcs(_))
        ));
    }
    let mut rng = Stream(0xc1ea_0005);
    let (_, _, _, _, ml, _) = multilinear_instance(&pcs, &mut rng, 3);
    let (_, st) = pcs.commit_vector(&rng.exts(2), &SEED).unwrap();
    let (_, vp) = pcs
        .prove_inner_product(&st, &rng.exts(2), &[1; 32], &SEED)
        .unwrap();
    for proof in [ml, vp] {
        let words = pcs.proof_to_words(&proof);
        // Every truncation and an extension.
        for cut in 0..words.len() {
            assert!(pcs.proof_from_words(&words[..cut]).is_err(), "cut {cut}");
        }
        let mut longer = words.clone();
        longer.push(0);
        assert!(pcs.proof_from_words(&longer).is_err());
        // The kind and the count.
        for kind in [2u64, 3, u64::MAX] {
            let mut w = words.clone();
            w[4] = kind;
            assert!(pcs.proof_from_words(&w).is_err());
        }
        for count in [words[5] + 1, words[5] - 1, u64::MAX, 1 << 62] {
            let mut w = words.clone();
            w[5] = count;
            assert!(pcs.proof_from_words(&w).is_err());
        }
    }
    // Random words: never a panic; a decoded proof re-encodes to the same words.
    let mut decoded = 0;
    for i in 0..20_000u64 {
        let len = (rng.next() % 40) as usize;
        let mut w: Vec<u64> = (0..len).map(|_| rng.next() % 4).collect();
        if len >= 6 && i % 2 == 0 {
            w[5] = (len as u64 - 6) / [1, K as u64][(w[4] % 2) as usize];
        }
        if let Ok(p) = pcs.proof_from_words(&w) {
            assert_eq!(pcs.proof_to_words(&p), w);
            decoded += 1;
        }
    }
    assert!(decoded > 1000, "{decoded} random encodings decoded");
}

/// The commitment of the zero polynomial and an opening of it, by hand: the evaluation at any
/// point is zero, and the formula of `evaluate_multilinear` on a monomial.
#[test]
fn a_monomial_by_hand() {
    let pcs = Pcs::new(Q).unwrap();
    let field = pcs.field().clone();
    // f = 5 X_1 X_3 in three variables: the coefficient index has bits 1 and 3 set (MSB-first),
    // i.e. 0b101 = 5.
    let mut coeffs = vec![0u64; 8];
    coeffs[0b101] = 5;
    let point = [Ext([2, 1, 0, 0]), Ext([9, 9, 9, 9]), Ext([3, 0, 0, 1])];
    let want = field.scale(&field.mul(&point[0], &point[2]), 5);
    assert_eq!(evaluate_multilinear(&field, &coeffs, &point).unwrap(), want);
    let (com, st) = pcs.commit_multilinear(&coeffs, &SEED).unwrap();
    let (value, proof) = pcs.prove_multilinear(&st, &point, &[0; 32], &SEED).unwrap();
    assert_eq!(value, want);
    pcs.verify_multilinear(&com, 3, &point, &want, &proof, &[0; 32])
        .unwrap();
    let (zero, st) = pcs.commit_multilinear(&[0; 8], &SEED).unwrap();
    let (value, proof) = pcs.prove_multilinear(&st, &point, &[0; 32], &SEED).unwrap();
    assert_eq!(value, field.zero());
    pcs.verify_multilinear(&zero, 3, &point, &field.zero(), &proof, &[0; 32])
        .unwrap();
}
