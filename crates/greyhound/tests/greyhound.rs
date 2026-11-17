//! Commitments, evaluation proofs, the reduced relation and jali's LaBRADOR proofs, at
//! $q = 2^{32} - 99$ (the reference's modulus for `LOGQ = 32`), $2^{56} - 27$ (its modulus for
//! `LOGQ = 56`) and the 57-bit Hachi-Co modulus 93266870910952069.
use hachi_co_greyhound::{
    Commitment, Error, Layout, Proof, ProverKey, PublicParams, Relation, Security, commit,
    evaluate, prove, prove_front, verify,
};
use hachi_co_sampling::ShakePrg;
use rand::Rng;

const Q32: u64 = 4294967197;
const Q56: u64 = 72057594037927909;
const Q57: u64 = 93266870910952069;

/// A polynomial of `len` ring elements with uniform coefficients modulo $q$.
fn random_poly(q: u64, len: usize, seed: u8) -> Vec<u64> {
    let mut prg = ShakePrg::new(&[seed; 32], b"test polynomial", 0);
    let bits = 64 - (q - 1).leading_zeros();
    let mask = (1u64 << bits) - 1;
    (0..len * 64)
        .map(|_| {
            loop {
                let v = prg.next_u64() & mask;
                if v < q {
                    break v;
                }
            }
        })
        .collect()
}

fn setup(len: usize, q: u64) -> (PublicParams, ProverKey) {
    let pp = PublicParams::new(len, q, Security::labrador(), [3u8; 32]).unwrap();
    let pk = pp.prover_key().unwrap();
    (pp, pk)
}

/// The front end's value is the polynomial's evaluation and its witness satisfies the reduced
/// relation, at every modulus; a wrong value, a changed witness and a smaller bound do not.
#[test]
fn front_end_satisfies_the_relation() {
    for q in [Q32, Q56, Q57] {
        for (len, x) in [
            (1usize, 43u64),
            (5, 43),
            (64, 2),
            (300, q - 1),
            (1000, 123456789),
        ] {
            let (pp, pk) = setup(len, q);
            let poly = random_poly(q, len, len as u8);
            let (com, opening) = commit(&pp, &pk, &poly).unwrap();
            let front = prove_front(&pp, &com, &opening, &poly, x).unwrap();
            assert_eq!(front.y, evaluate(q, &poly, x).unwrap(), "q {q} len {len}");
            let relation = Relation::new(&pp, &com, x, front.y, &front.proof).unwrap();
            relation.check(pp.ring(), &front.witness).unwrap();
            // The norm is the witness's.
            assert_eq!(front.proof.norm_sq, front.witness.norm_sq());

            let wrong = Relation::new(&pp, &com, x, (front.y + 1) % q, &front.proof).unwrap();
            assert!(matches!(
                wrong.check(pp.ring(), &front.witness),
                Err(Error::Witness(_))
            ));
            for v in 0..4 {
                let mut changed = front.witness.clone();
                changed.vectors[v][0] += 1;
                let mut relation = Relation::new(&pp, &com, x, front.y, &front.proof).unwrap();
                relation.norm_sq = changed.norm_sq();
                assert!(relation.check(pp.ring(), &changed).is_err(), "vector {v}");
            }
            let mut tight = Relation::new(&pp, &com, x, front.y, &front.proof).unwrap();
            tight.norm_sq -= 1;
            assert!(tight.check(pp.ring(), &front.witness).is_err());
        }
    }
}

/// jali's own check of its statement accepts the witness in jali's layout.
#[test]
fn jali_statement_holds() {
    let (len, x) = (200, 43);
    let (pp, pk) = setup(len, Q32);
    let poly = random_poly(Q32, len, 1);
    let (com, opening) = commit(&pp, &pk, &poly).unwrap();
    let front = prove_front(&pp, &com, &opening, &poly, x).unwrap();
    let ring = pp.labrador_ring().unwrap();
    for layout in [
        pp.layout(),
        Layout::with_multiplicity(&pp.params().witness_lengths(), 1),
    ] {
        let relation = Relation::new(&pp, &com, x, front.y, &front.proof).unwrap();
        let statement = relation.into_jali(&ring, &layout).unwrap();
        let witness = layout
            .jali_witness(&ring, pp.params(), &front.witness)
            .unwrap();
        statement.check(&witness).unwrap();
        assert_eq!(statement.multiplicity(), layout.r);
        assert_eq!(statement.rank(), layout.n);
    }
}

/// Honest proofs verify and survive their encoding; wrong values, points, commitments and
/// changed bytes are rejected.
#[test]
fn proofs_verify_and_tampering_fails() {
    let (len, x) = (256, 43);
    let (pp, pk) = setup(len, Q32);
    let poly = random_poly(Q32, len, 2);
    let (com, opening) = commit(&pp, &pk, &poly).unwrap();
    let (y, proof) = prove(&pp, &com, &opening, &poly, x).unwrap();
    assert_eq!(y, evaluate(Q32, &poly, x).unwrap());
    verify(&pp, &com, x, y, &proof).unwrap();
    let p = pp.params();
    let bytes = proof.to_bytes(p);
    assert_eq!(Proof::from_bytes(p, &bytes).unwrap(), proof);
    assert_eq!(Commitment::from_bytes(p, &com.to_bytes(p)).unwrap(), com);

    assert!(verify(&pp, &com, x, (y + 1) % Q32, &proof).is_err());
    assert!(verify(&pp, &com, x + 1, y, &proof).is_err());
    let other = commit(&pp, &pk, &random_poly(Q32, len, 3)).unwrap().0;
    assert!(verify(&pp, &other, x, y, &proof).is_err());
    let mut norm = proof.clone();
    norm.front.norm_sq += 1;
    assert!(verify(&pp, &com, x, y, &norm).is_err());
    let mut u2 = proof.clone();
    u2.front.u2[0].0[5] = (u2.front.u2[0].0[5] + 1) % Q32;
    assert!(verify(&pp, &com, x, y, &u2).is_err());
    // One flipped bit in each of a few places of the encoding.
    let front_len = bytes.len() - proof.labrador.len();
    for at in [
        0,
        front_len - 1,
        front_len,
        front_len + 100,
        bytes.len() / 2,
        bytes.len() - 1,
    ] {
        let mut changed = bytes.clone();
        changed[at] ^= 1;
        let rejected = match Proof::from_bytes(p, &changed) {
            Ok(proof) => verify(&pp, &com, x, y, &proof).is_err(),
            Err(_) => true,
        };
        assert!(rejected, "flipped bit at byte {at} of {}", bytes.len());
    }
    // A proof for another polynomial with the same commitment's parameters.
    let poly2 = random_poly(Q32, len, 4);
    let (com2, opening2) = commit(&pp, &pk, &poly2).unwrap();
    let (y2, proof2) = prove(&pp, &com2, &opening2, &poly2, x).unwrap();
    verify(&pp, &com2, x, y2, &proof2).unwrap();
    assert!(verify(&pp, &com, x, y2, &proof2).is_err());
}

/// jali 0.3.1 takes moduli up to 2^62 (0.3.0 stopped at 2^48): proofs at C's
/// LOGQ = 56 modulus and at three moduli of Hachi-Co's presets (53.3, 56.4 and 61.1 bits).
#[test]
fn labrador_at_large_moduli() {
    for q in [
        Q56,
        Q57,
        10916536653611741,
        93266870910952069,
        2402958527344632421,
    ] {
        let (pp, pk) = setup(16, q);
        let poly = random_poly(q, 16, 5);
        let (com, opening) = commit(&pp, &pk, &poly).unwrap();
        let (y, proof) = prove(&pp, &com, &opening, &poly, 43).unwrap();
        assert_eq!(y, evaluate(q, &poly, 43).unwrap());
        verify(&pp, &com, 43, y, &proof).unwrap();
        assert!(verify(&pp, &com, 43, (y + 1) % q, &proof).is_err(), "q {q}");
    }
    // At 2^62 and above the modulus is refused.
    assert!(matches!(
        jali::labrador::Rq::new((1 << 62) + 5, 64),
        Err(jali::Error::Parameter("LaBRADOR modulus"))
    ));
}

/// A moderate size, degree 2^18 at q = 2^32 - 99, end to end.
#[test]
fn moderate_size_end_to_end() {
    let (len, x) = (1 << 12, 43);
    let (pp, pk) = setup(len, Q32);
    let poly = random_poly(Q32, len, 6);
    let (com, opening) = commit(&pp, &pk, &poly).unwrap();
    let front = prove_front(&pp, &com, &opening, &poly, x).unwrap();
    Relation::new(&pp, &com, x, front.y, &front.proof)
        .unwrap()
        .check(pp.ring(), &front.witness)
        .unwrap();
    let (y, proof) = prove(&pp, &com, &opening, &poly, x).unwrap();
    assert_eq!(y, front.y);
    verify(&pp, &com, x, y, &proof).unwrap();
    assert!(verify(&pp, &com, x, (y + 1) % Q32, &proof).is_err());
}

/// The case that rejected an honest proof before the margin of `params::NORM_MARGIN`: the
/// 57-bit modulus, degree 2^21 and f = 4, where the predicted norm met the threshold of rank 14
/// and the actual norm did not. The verifier's check on the actual norm must pass.
#[test]
fn ranks_hold_for_the_actual_norm() {
    let (len, x) = (1 << 15, 43);
    let params =
        hachi_co_greyhound::Params::with_expansion(len, Q57, Security::labrador(), 4).unwrap();
    let pp = PublicParams::with_params(params, Security::labrador(), [0u8; 32]).unwrap();
    let pk = pp.prover_key().unwrap();
    for seed in 0..3 {
        let poly = random_poly(Q57, len, 10 + seed);
        let (com, opening) = commit(&pp, &pk, &poly).unwrap();
        let front = prove_front(&pp, &com, &opening, &poly, x).unwrap();
        pp.params()
            .check_norm(front.proof.norm_sq, pp.security())
            .unwrap();
    }
}

/// The front end at a moderate size and the 57-bit modulus.
#[test]
fn moderate_size_front_end_57_bits() {
    let (len, x) = (1 << 12, 43);
    let (pp, pk) = setup(len, Q57);
    let poly = random_poly(Q57, len, 7);
    let (com, opening) = commit(&pp, &pk, &poly).unwrap();
    let front = prove_front(&pp, &com, &opening, &poly, x).unwrap();
    assert_eq!(front.y, evaluate(Q57, &poly, x).unwrap());
    Relation::new(&pp, &com, x, front.y, &front.proof)
        .unwrap()
        .check(pp.ring(), &front.witness)
        .unwrap();
    pp.params()
        .check_norm(front.proof.norm_sq, pp.security())
        .unwrap();
}
