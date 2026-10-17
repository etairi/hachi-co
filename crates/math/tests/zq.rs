//! $\mathbb Z_q$ against num-bigint, on boundary, unreduced and random operands, for small primes,
//! the test moduli and the largest primes below $2^{63}$ (seed 1).
mod common;

use common::*;
use hachi_co_math::{Error, Zq, zq::is_prime};
use num_bigint::BigInt;
use rand::RngExt;

/// Small primes, the test moduli, and $2^{63} - 25$ (the largest prime below $2^{63}$, $\equiv 7
/// \pmod 8$).
const PRIMES: [u64; 8] = [3, 5, 13, Q32, Q52, Q61, Q63, 9223372036854775783];

fn operands(q: u64, rng: &mut rand_chacha::ChaCha8Rng) -> Vec<u64> {
    let mut v = vec![0, 1, 2, q / 2, q / 2 + 1, q - 2, q - 1];
    // Unreduced: read modulo q.
    v.extend([q, q + 1, 2 * q - 1, u64::MAX - 1, u64::MAX]);
    v.extend((0..20).map(|_| rng.random_range(0..q)));
    v.extend((0..5).map(|_| rng.random::<u64>()));
    v
}

#[test]
fn construction_accepts_exactly_the_odd_primes_below_2_to_63() {
    for q in PRIMES {
        assert_eq!(Zq::new(q).unwrap().modulus(), q);
    }
    for q in [
        0u64,
        1,
        2,
        4,
        9,
        15,
        561,
        (1 << 63) - 1,
        1 << 63,
        18446744073709551557,
    ] {
        assert_eq!(Zq::new(q), Err(Error::Modulus(q)), "q = {q}");
    }
    assert!(is_prime(18446744073709551557) && 18446744073709551557u64 >= 1u64 << 63);
}

#[test]
fn operations_equal_big_integer_arithmetic() {
    let mut rng = rng(1);
    for q in PRIMES {
        let zq = Zq::new(q).unwrap();
        let big_q = BigInt::from(q);
        let modq = |x: BigInt| -> u64 { u64::try_from(((x % &big_q) + &big_q) % &big_q).unwrap() };
        let values = operands(q, &mut rng);
        for &a in &values {
            let ba = BigInt::from(a);
            assert_eq!(zq.reduce(a), modq(ba.clone()));
            assert_eq!(zq.neg(a), modq(-ba.clone()));
            let centered = zq.to_centered(a);
            assert!(centered.unsigned_abs() <= (q - 1) / 2);
            assert_eq!(modq(BigInt::from(centered)), modq(ba.clone()));
            match zq.inv(a) {
                None => assert_eq!(a % q, 0),
                Some(inv) => {
                    assert!(inv < q);
                    assert_eq!(modq(&ba * BigInt::from(inv)), 1);
                    assert_eq!(inv, zq.pow(a, q - 2), "Fermat");
                }
            }
            for &b in &values {
                let bb = BigInt::from(b);
                assert_eq!(zq.add(a, b), modq(&ba + &bb), "q {q} a {a} b {b}");
                assert_eq!(zq.sub(a, b), modq(&ba - &bb), "q {q} a {a} b {b}");
                assert_eq!(zq.mul(a, b), modq(&ba * &bb), "q {q} a {a} b {b}");
            }
            for e in [0u64, 1, 2, 3, 64, q - 2, q - 1, q, u64::MAX] {
                assert_eq!(
                    zq.pow(a, e),
                    modq(ba.modpow(&BigInt::from(e), &big_q)),
                    "pow"
                );
            }
        }
        // Wide and signed inputs.
        for x in [
            0u128,
            1,
            u128::from(q),
            u128::from(u64::MAX),
            u128::MAX,
            rng.random(),
        ] {
            assert_eq!(zq.reduce_u128(x), modq(BigInt::from(x)));
        }
        for x in [
            0i64,
            1,
            -1,
            i64::MIN,
            i64::MAX,
            -(q as i64 / 2),
            rng.random(),
        ] {
            assert_eq!(zq.from_i64(x), modq(BigInt::from(x)));
        }
        for x in [0i128, -1, i128::MIN, i128::MAX, rng.random()] {
            assert_eq!(zq.from_i128(x), modq(BigInt::from(x)));
        }
        // The centered range: (q-1)/2 is positive, (q+1)/2 negative.
        assert_eq!(zq.to_centered((q - 1) / 2), ((q - 1) / 2) as i64);
        assert_eq!(zq.to_centered(q.div_ceil(2)), -(((q - 1) / 2) as i64));
        assert_eq!(zq.half(), (q - 1) / 2);
    }
}
