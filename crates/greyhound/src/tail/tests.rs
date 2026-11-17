use super::*;
use hachi_co_math::Fqk;

/// Hachi-Co's modulus of secure100-l20-n8-a8 and secure100-l24-n8-a8.
const Q53: u64 = 10916536653611741;
const Q56: u64 = 93266870910952069;

fn digits(vars: usize, seed: u64) -> Vec<i8> {
    let mut prg = ShakePrg::new(&[seed as u8; 32], b"tail digits", seed);
    (0..1usize << vars)
        .map(|_| (prg.next_u32() % 16) as i8 - 8)
        .collect()
}

fn point<const K: usize>(field: &Fqk<K>, vars: usize, seed: u64) -> Vec<Ext<K>> {
    let mut prg = ShakePrg::new(&[seed as u8; 32], b"tail point", seed);
    let q = field.modulus();
    (0..vars)
        .map(|_| Ext(core::array::from_fn(|_| prg.next_u64() % q)))
        .collect()
}

/// $\sum_x d_x \widetilde{eq}(x, \mathbf a)$ directly.
fn value<const K: usize>(field: &Fqk<K>, digits: &[i8], point: &[Ext<K>]) -> Ext<K> {
    let eq = eq_table(field, point);
    let zq = field.zq();
    digits.iter().zip(&eq).fold(field.zero(), |acc, (d, e)| {
        field.add(&acc, &field.scale(e, zq.from_i64(i64::from(*d))))
    })
}

#[test]
fn value_is_hachis_table_eval() {
    let field = Fqk::<4>::new(Q53).unwrap();
    for vars in [6, 7, 10, 13] {
        let d = digits(vars, vars as u64);
        let a = point(&field, vars, 100 + vars as u64);
        let table: Vec<u64> = d
            .iter()
            .map(|x| field.zq().from_i64(i64::from(*x)))
            .collect();
        let hachi = hachi::iteration::table_eval(&field, &table, &a).unwrap();
        assert_eq!(value(&field, &d, &a), hachi, "vars {vars}");
    }
}

/// The tables, commitments, openings and claims of a batch.
struct Batch {
    vars: Vec<usize>,
    digits: Vec<Vec<i8>>,
    commitments: Vec<Vec<u64>>,
    openings: Vec<TailOpening>,
    points: Vec<Vec<Ext<4>>>,
    values: Vec<Ext<4>>,
}

impl Batch {
    fn new(params: &TailParams, field: &Fqk<4>, vars: &[usize], seed: u64) -> Self {
        let digits: Vec<Vec<i8>> = vars
            .iter()
            .enumerate()
            .map(|(t, v)| self::digits(*v, seed + t as u64))
            .collect();
        Self::with_tables(params, field, vars, digits, seed)
    }

    fn with_tables(
        params: &TailParams,
        field: &Fqk<4>,
        vars: &[usize],
        digits: Vec<Vec<i8>>,
        seed: u64,
    ) -> Self {
        let (commitments, openings): (Vec<_>, Vec<_>) = vars
            .iter()
            .zip(&digits)
            .map(|(v, d)| params.commit(*v, d).unwrap())
            .unzip();
        let points: Vec<Vec<Ext<4>>> = vars
            .iter()
            .enumerate()
            .map(|(t, v)| point(field, *v, 50 + seed + t as u64))
            .collect();
        let values = digits
            .iter()
            .zip(&points)
            .map(|(d, a)| value(field, d, a))
            .collect();
        Self {
            vars: vars.to_vec(),
            digits,
            commitments,
            openings,
            points,
            values,
        }
    }

    fn claims(&self) -> Vec<Claim<'_, 4>> {
        (0..self.vars.len())
            .map(|t| Claim {
                vars: self.vars[t],
                commitment: &self.commitments[t],
                point: &self.points[t],
                value: self.values[t],
            })
            .collect()
    }

    fn prove(&self, params: &TailParams, field: &Fqk<4>, digits: &[Vec<i8>]) -> Result<Vec<u64>> {
        self.prove_in(params, field, digits, &[9u8; 32])
    }

    fn prove_in(
        &self,
        params: &TailParams,
        field: &Fqk<4>,
        digits: &[Vec<i8>],
        context: &[u8; 32],
    ) -> Result<Vec<u64>> {
        let claims = self.claims();
        let prover: Vec<ProverClaim<'_, 4>> = (0..self.vars.len())
            .map(|t| ProverClaim {
                claim: claims[t],
                digits: &digits[t],
                opening: &self.openings[t],
            })
            .collect();
        params.prove(field, &prover, context)
    }
}

/// Tables whose columns are all equal: all -8, and -8, 7 alternating.
fn degenerate(vars: usize, alternating: bool) -> Vec<i8> {
    (0..1usize << vars)
        .map(|x| if alternating && x % 2 == 1 { 7 } else { -8 })
        .collect()
}

/// The restart path: with the first three attempts forced to restart, a batch of all -8
/// tables proves at attempt 3 and verifies. Another attempt number in the honest proof fails
/// verification, and a number of 64 or more, or any other high bit of the length word, is an
/// encoding error.
#[test]
fn restarts_and_the_attempt_number() {
    for q in [Q53, Q56] {
        let field = Fqk::<4>::new(q).unwrap();
        let vars = [13, 10, 8];
        let params = TailParams::new(q, &vars, 8, 100, [4u8; 32])
            .unwrap()
            .with_forced_restarts(3);
        let tables: Vec<Vec<i8>> = vars.iter().map(|v| degenerate(*v, false)).collect();
        let batch = Batch::with_tables(&params, &field, &vars, tables.clone(), 60);
        let proof = batch.prove(&params, &field, &tables).unwrap();
        let at = params.length_word();
        assert_eq!(proof[at] >> ATTEMPT_SHIFT, 3);
        let claims = batch.claims();
        params.verify(&field, &claims, &proof, &[9u8; 32]).unwrap();
        for attempt in [0u64, 2, 4, 63] {
            let mut p = proof.clone();
            p[at] = (p[at] & ((1 << ATTEMPT_SHIFT) - 1)) | (attempt << ATTEMPT_SHIFT);
            assert!(
                params.verify(&field, &claims, &p, &[9u8; 32]).is_err(),
                "attempt {attempt}"
            );
        }
        for high in [64u64 << ATTEMPT_SHIFT, 1 << 54, 1 << 63] {
            let mut p = proof.clone();
            p[at] |= high;
            assert!(
                matches!(
                    params.verify(&field, &claims, &p, &[9u8; 32]),
                    Err(Error::Encoding)
                ),
                "high bits {high:#x}"
            );
        }
    }
}

/// Degenerate tables (all -8, and -8, 7 alternating) prove and verify in several contexts.
#[test]
fn degenerate_tables_prove_and_verify() {
    let field = Fqk::<4>::new(Q53).unwrap();
    let vars = [14, 10, 8];
    let params = TailParams::new(Q53, &vars, 8, 100, [2u8; 32]).unwrap();
    for alternating in [false, true] {
        let tables: Vec<Vec<i8>> = vars.iter().map(|v| degenerate(*v, alternating)).collect();
        let batch = Batch::with_tables(&params, &field, &vars, tables.clone(), 70);
        for c in 0..4u8 {
            let context = [c; 32];
            let proof = batch.prove_in(&params, &field, &tables, &context).unwrap();
            assert!(proof[params.length_word()] >> ATTEMPT_SHIFT < u64::from(TAIL_ATTEMPTS));
            params
                .verify(&field, &batch.claims(), &proof, &context)
                .unwrap();
        }
    }
}

/// prove() returns an error, without panicking, when all 64 attempts fail: on the binding
/// checks (with a root Hermite factor that no norm passes) and on the reserve (one word).
#[test]
fn every_attempt_failing_is_an_error() {
    let field = Fqk::<4>::new(Q53).unwrap();
    let vars = [7];
    let params = TailParams::new(Q53, &vars, 8, 100, [3u8; 32])
        .unwrap()
        .with_check_delta(1.000_000_1);
    let batch = Batch::new(&params, &field, &vars, 80);
    assert!(matches!(
        batch.prove(&params, &field, &batch.digits),
        Err(Error::Parameter(why)) if why.contains("norm")
    ));
    let params = TailParams::new(Q53, &vars, 8, 100, [3u8; 32])
        .unwrap()
        .with_reserve_words(1);
    let batch = Batch::new(&params, &field, &vars, 80);
    assert!(matches!(
        batch.prove(&params, &field, &batch.digits),
        Err(Error::Parameter(why)) if why.contains("fixed length")
    ));
}

/// A batch of three claims on tables of different sizes, in one proof, at both moduli: it
/// verifies, and is rejected for a changed value, point, commitment word, proof word, context
/// or claim order; a changed digit cannot be proven.
#[test]
fn batch_of_three_and_its_rejections() {
    for q in [Q53, Q56] {
        let field = Fqk::<4>::new(q).unwrap();
        let vars = [13, 10, 8];
        let params = TailParams::new(q, &vars, 8, 100, [4u8; 32]).unwrap();
        let batch = Batch::new(&params, &field, &vars, 20);
        let proof = batch.prove(&params, &field, &batch.digits).unwrap();
        assert_eq!(proof.len(), params.proof_words());
        let claims = batch.claims();
        params.verify(&field, &claims, &proof, &[9u8; 32]).unwrap();
        assert!(params.verify(&field, &claims, &proof, &[8u8; 32]).is_err());
        for t in 0..3 {
            let mut c = claims.clone();
            c[t].value = field.add(&c[t].value, &field.one());
            assert!(
                params.verify(&field, &c, &proof, &[9u8; 32]).is_err(),
                "value {t}"
            );
            let mut point = batch.points[t].clone();
            point[0] = field.add(&point[0], &field.one());
            let mut c = claims.clone();
            c[t].point = &point;
            assert!(
                params.verify(&field, &c, &proof, &[9u8; 32]).is_err(),
                "point {t}"
            );
            for at in [0, batch.commitments[t].len() - 1] {
                let mut com = batch.commitments[t].clone();
                com[at] ^= 1;
                let mut c = claims.clone();
                c[t].commitment = &com;
                assert!(
                    params.verify(&field, &c, &proof, &[9u8; 32]).is_err(),
                    "commitment {t}"
                );
            }
        }
        // Every region of the proof: the u2 words, the norm, the length, jali's words and the
        // padding.
        let u2_words: usize = vars.iter().map(|v| params.commitment_words(*v)).sum();
        let jali_words = ((proof[u2_words + 2] & ((1 << ATTEMPT_SHIFT) - 1)) as usize).div_ceil(8);
        let regions = [
            0,
            u2_words - 1,
            u2_words,
            u2_words + 1,
            u2_words + 2,
            u2_words + 3,
            u2_words + 3 + jali_words / 2,
            u2_words + 2 + jali_words,
            proof.len() - 1,
        ];
        for at in regions {
            for bit in [0, 63] {
                let mut p = proof.clone();
                p[at] ^= 1 << bit;
                assert!(
                    params.verify(&field, &claims, &p, &[9u8; 32]).is_err(),
                    "proof word {at} bit {bit}"
                );
            }
        }
        assert!(
            params
                .verify(&field, &claims, &proof[..proof.len() - 1], &[9u8; 32])
                .is_err()
        );
        let mut swapped = claims.clone();
        swapped.swap(1, 2);
        assert!(params.verify(&field, &swapped, &proof, &[9u8; 32]).is_err());
        // A digit that is not the committed one: no proof.
        let mut changed = batch.digits.clone();
        changed[1][5] = if changed[1][5] == 7 {
            6
        } else {
            changed[1][5] + 1
        };
        assert!(batch.prove(&params, &field, &changed).is_err());
    }
}

/// Decoders never panic: random words of the right and wrong lengths are rejected.
#[test]
fn decoders_reject_garbage() {
    let field = Fqk::<4>::new(Q53).unwrap();
    let vars = [9, 8];
    let params = TailParams::new(Q53, &vars, 8, 100, [5u8; 32]).unwrap();
    let batch = Batch::new(&params, &field, &vars, 30);
    let claims = batch.claims();
    let mut prg = ShakePrg::new(&[6u8; 32], b"garbage", 0);
    for len in [
        0,
        1,
        params.proof_words() - 1,
        params.proof_words(),
        params.proof_words() + 1,
    ] {
        for _ in 0..3 {
            let words: Vec<u64> = (0..len).map(|_| prg.next_u64()).collect();
            assert!(params.verify(&field, &claims, &words, &[1u8; 32]).is_err());
        }
    }
    // A commitment of residues of q or more, and of the wrong length.
    let mut com = batch.commitments[0].clone();
    com[0] = u64::MAX;
    let mut c = claims.clone();
    c[0].commitment = &com;
    assert!(
        params
            .verify(&field, &c, &vec![0; params.proof_words()], &[1u8; 32])
            .is_err()
    );
    let short = &batch.commitments[0][1..];
    let mut c = claims.clone();
    c[0].commitment = short;
    assert!(
        params
            .verify(&field, &c, &vec![0; params.proof_words()], &[1u8; 32])
            .is_err()
    );
    // Words round trip through the packing.
    let values: Vec<u64> = (0..100u64)
        .map(|i| i.wrapping_mul(0x9e37_79b9) % Q53)
        .collect();
    let bits = 64 - (Q53 - 1).leading_zeros();
    assert_eq!(
        unpack_words(&pack_words(values.iter().copied(), bits), bits, 100, Q53).unwrap(),
        values
    );
    // Claims that do not match the parameters, and digits beyond the bound.
    assert!(params.commit(10, &digits(10, 1)).is_err());
    assert!(params.commit(9, &vec![9i8; 1 << 9]).is_err());
    assert!(params.commit(9, &vec![0i8; 1 << 8]).is_err());
}

/// Single claims at Hachi-Co's 48.4-bit modulus (secure100-l20-n4-a8) and at the three moduli
/// jali 0.3.1 takes beyond 2^48 (53.3, 56.4 and 61.1 bits).
#[test]
fn single_claims_at_hachi_co_moduli() {
    for q in [366142346648669, Q53, Q56, 2402958527344632421] {
        let field = Fqk::<4>::new(q).unwrap();
        let params = TailParams::new(q, &[11], 8, 100, [7u8; 32]).unwrap();
        let batch = Batch::new(&params, &field, &[11], 40);
        let proof = batch.prove(&params, &field, &batch.digits).unwrap();
        params
            .verify(&field, &batch.claims(), &proof, &[9u8; 32])
            .unwrap();
    }
}

/// The ranks of every claim of a batch hold for the batch's joint norm, at lambda 100 and 115.
#[test]
fn ranks_hold_for_the_joint_norm() {
    for lambda in [100, 115] {
        let params = TailParams::new(Q53, &[24, 20, 18], 8, lambda, [8u8; 32]).unwrap();
        let joint = params.joint_norm_sq();
        for s in params.shapes() {
            let sec = params.security();
            let norm = joint.sqrt();
            assert!(sis_secure(
                Q53,
                s.kappa,
                6.0 * T * sec.slack * 2f64.powi(s.bu as i32) * norm,
                sec.delta
            ));
            assert!(sis_secure(Q53, s.kappa1, 2.0 * sec.slack * norm, sec.delta));
            assert_eq!(s.m * s.n * N, 1 << s.vars);
        }
        assert!(params.policy().min_block_size >= block_size(lambda).max(346));
    }
}

#[test]
fn one_claim_round_trip() {
    for q in [Q53, Q56] {
        let field = Fqk::<4>::new(q).unwrap();
        let vars = 12;
        let params = TailParams::new(q, &[vars], 8, 100, [1u8; 32]).unwrap();
        let d = digits(vars, 1);
        let (commitment, opening) = params.commit(vars, &d).unwrap();
        assert_eq!(commitment.len(), params.commitment_words(vars));
        let a = point(&field, vars, 2);
        let claim = Claim {
            vars,
            commitment: &commitment,
            point: &a,
            value: value(&field, &d, &a),
        };
        let proof = params
            .prove(
                &field,
                &[ProverClaim {
                    claim,
                    digits: &d,
                    opening: &opening,
                }],
                &[3u8; 32],
            )
            .unwrap();
        assert_eq!(proof.len(), params.proof_words());
        params.verify(&field, &[claim], &proof, &[3u8; 32]).unwrap();
        let mut wrong = claim;
        wrong.value = field.add(&claim.value, &field.one());
        assert!(params.verify(&field, &[wrong], &proof, &[3u8; 32]).is_err());
    }
}
