//! The proof encoding (format version 2) and its size: `from_bytes(to_bytes(p)) == p` for the
//! fixture proofs (with the `ClearPcs` commitments and openings of the recorded polynomials) and
//! for honest proofs; `to_bytes(p).len()` equals the size formula of `hachi-co-params` with
//! `ClearPcs`'s sizes for every preset run here (the toy presets; the development presets in
//! `tests/honest.rs`); the length computed from the layout (`encoded_len`) equals the formula
//! for every preset, secure ones included, with `ClearPcs`'s word counts and with arbitrary
//! ones; the layout of the bytes is the documented one.

mod common;

use common::{K, clear_words, fixture_proof, load};
use hachi_co::{BasePcsWords, ClaimedPoly, EvalProof, HEADER_LEN, MAGIC, VERSION, encoded_len};
use hachi_co_core::{Fixture, Layout};
use hachi_co_params::{presets, sizes, sizes_with_base};

#[test]
fn fixture_proofs_round_trip_with_the_size_of_the_formula() {
    for name in Fixture::TOY {
        let (fx, ctx) = load(name);
        let proof = fixture_proof(&fx);
        let bytes = proof.to_bytes();
        let base = clear_words(&ctx.layout);
        let formula = sizes_with_base(&ctx.params, &base).unwrap();
        assert_eq!(bytes.len() as u128, formula.total, "{name}");
        assert_eq!(
            bytes.len(),
            encoded_len(&ctx.layout, &base).unwrap(),
            "{name}"
        );
        assert_eq!(bytes.len(), proof.encoded_len(), "{name}");
        assert_eq!(
            EvalProof::from_bytes(&ctx, &bytes).unwrap(),
            proof,
            "{name}"
        );
        let plain = sizes(&ctx.params).unwrap();
        eprintln!(
            "{name}: {} bytes ({:?}); without the base PCS {} bytes; {} residues of {} bits, {} \
             bytes packed",
            formula.total,
            formula.components(),
            plain.total,
            formula.residues,
            formula.bits_per_residue,
            formula.packed_total
        );
    }
}

#[test]
fn encoded_len_is_the_size_formula_for_every_preset() {
    let arbitrary = BasePcsWords {
        commitment_w: 3,
        commitment_g0: 0,
        commitment_g1: 7,
        proof_w: 1234,
        proof_g0: 1,
        proof_g1: 99,
    };
    for p in presets().unwrap() {
        let (params, _) = p.derive().unwrap();
        let layout = Layout::new(&params).unwrap();
        for base in [clear_words(&layout), arbitrary, BasePcsWords::default()] {
            assert_eq!(
                encoded_len(&layout, &base).unwrap() as u128,
                sizes_with_base(&params, &base).unwrap().total,
                "{}: {base:?}",
                p.name
            );
        }
    }
}

/// The bytes of the toy-n2 fixture proof, word by word, against the documented layout: a*, the
/// 6n lengths, E(u), v, G, the commitments, the rounds, the final values, the openings.
#[test]
fn the_layout_of_the_bytes_is_the_documented_one() {
    let (fx, ctx) = load("toy-n2");
    let proof = fixture_proof(&fx);
    let bytes = proof.to_bytes();
    assert_eq!(&bytes[..8], &MAGIC);
    assert_eq!(&bytes[8..16], &VERSION.to_le_bytes());
    assert_eq!(VERSION, 2);
    let (chunks, rest) = bytes[HEADER_LEN..].as_chunks::<8>();
    assert!(rest.is_empty());
    let words: Vec<u64> = chunks.iter().map(|c| u64::from_le_bytes(*c)).collect();
    let t = &proof.sumcheck;
    let order = [ClaimedPoly::W, ClaimedPoly::G0, ClaimedPoly::G1];
    let items = |w: &hachi_co::PcsWords| -> Vec<Vec<u64>> {
        order
            .iter()
            .flat_map(|poly| w.of(*poly).iter().cloned())
            .collect()
    };
    let (commitments, openings) = (items(&t.commitments), items(&t.openings));
    let mut expected = vec![proof.attempt];
    expected.extend(
        commitments
            .iter()
            .chain(&openings)
            .map(|item| item.len() as u64),
    );
    expected.extend(&proof.e_u.0);
    expected.extend(proof.v.iter().flat_map(|p| p.0.iter()));
    let ext_words = |v: &[hachi_co_core::Ext<K>]| v.iter().flat_map(|e| e.0).collect::<Vec<_>>();
    expected.extend(ext_words(&t.g0_sums));
    expected.extend(ext_words(&t.g1_sums));
    expected.extend(commitments.iter().flatten());
    for r in &t.rounds {
        expected.extend(ext_words(&r.h0));
        expected.extend(ext_words(&r.h1));
    }
    expected.extend(ext_words(&t.finals.w));
    expected.extend(ext_words(&t.finals.g0));
    expected.extend(ext_words(&t.finals.g1));
    expected.extend(openings.iter().flatten());
    assert_eq!(words, expected);
    let l = &ctx.layout;
    let pcs_words: usize = commitments.iter().chain(&openings).map(Vec::len).sum();
    assert_eq!(
        words.len(),
        1 + 6 * l.n
            + l.d
            + l.n_d * l.d
            + K * (2 * l.n + l.ell_pp * (l.d0 + l.d1 + 2) + 3 * l.n)
            + pcs_words
    );
    assert_eq!(commitments.len(), 3 * l.n);
    // Decoding is the inverse, and re-encoding a decoded proof gives the same bytes.
    let decoded = EvalProof::<K>::from_bytes(&ctx, &bytes).unwrap();
    assert_eq!(decoded.to_bytes(), bytes);
}
