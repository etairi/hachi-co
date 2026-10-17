//! The proof encoding and its size: `from_bytes(to_bytes(p)) == p` for the fixture proofs and
//! for honest proofs; `to_bytes(p).len()` equals the size formula of `hachi-co-params` for every
//! preset run here (toy-n1, toy-n2, toy-n3; the development presets in `tests/honest.rs`); the
//! length computed from the layout (`encoded_len`) equals the formula for every preset, secure
//! ones included; the layout of the bytes is the documented one.

mod common;

use common::{K, fixture_proof, load};
use hachi_co::{EvalProof, HEADER_LEN, MAGIC, VERSION, encoded_len};
use hachi_co_core::{Fixture, Layout};
use hachi_co_params::{presets, sizes};

#[test]
fn fixture_proofs_round_trip_with_the_size_of_the_formula() {
    for name in Fixture::TOY {
        let (fx, ctx) = load(name);
        let proof = fixture_proof(&fx);
        let bytes = proof.to_bytes();
        let formula = sizes(&ctx.params).unwrap();
        assert_eq!(bytes.len() as u128, formula.total, "{name}");
        assert_eq!(bytes.len(), encoded_len(&ctx.layout).unwrap(), "{name}");
        assert_eq!(
            EvalProof::from_bytes(&ctx, &bytes).unwrap(),
            proof,
            "{name}"
        );
        eprintln!(
            "{name}: {} bytes ({:?}); {} residues of {} bits, {} bytes packed",
            formula.total,
            formula.components(),
            formula.residues,
            formula.bits_per_residue,
            formula.packed_total
        );
    }
}

#[test]
fn encoded_len_is_the_size_formula_for_every_preset() {
    for p in presets().unwrap() {
        let (params, _) = p.derive().unwrap();
        let layout = Layout::new(&params).unwrap();
        assert_eq!(
            encoded_len(&layout).unwrap() as u128,
            sizes(&params).unwrap().total,
            "{}",
            p.name
        );
    }
}

/// The bytes of the toy-n1 fixture proof, word by word, against the documented layout.
#[test]
fn the_layout_of_the_bytes_is_the_documented_one() {
    let (fx, ctx) = load("toy-n1");
    let proof = fixture_proof(&fx);
    let bytes = proof.to_bytes();
    assert_eq!(&bytes[..8], &MAGIC);
    assert_eq!(&bytes[8..16], &VERSION.to_le_bytes());
    let (chunks, rest) = bytes[HEADER_LEN..].as_chunks::<8>();
    assert!(rest.is_empty());
    let words: Vec<u64> = chunks.iter().map(|c| u64::from_le_bytes(*c)).collect();
    let mut expected = vec![proof.attempt];
    expected.extend(&proof.e_u.0);
    expected.extend(proof.v.iter().flat_map(|p| p.0.iter()));
    let t = &proof.sumcheck;
    let ext_words = |v: &[hachi_co_core::Ext<K>]| v.iter().flat_map(|e| e.0).collect::<Vec<_>>();
    expected.extend(ext_words(&t.g0_sums));
    expected.extend(ext_words(&t.g1_sums));
    for r in &t.rounds {
        expected.extend(ext_words(&r.h0));
        expected.extend(ext_words(&r.h1));
    }
    expected.extend(ext_words(&t.finals.w));
    expected.extend(ext_words(&t.finals.g0));
    expected.extend(ext_words(&t.finals.g1));
    assert_eq!(words, expected);
    let l = &ctx.layout;
    assert_eq!(
        words.len(),
        1 + l.d + l.n_d * l.d + K * (2 * l.n + l.ell_pp * (l.d0 + l.d1 + 2) + 3 * l.n)
    );
    // Decoding is the inverse, and re-encoding a decoded proof gives the same bytes.
    let decoded = EvalProof::<K>::from_bytes(&ctx, &bytes).unwrap();
    assert_eq!(decoded.to_bytes(), bytes);
}
