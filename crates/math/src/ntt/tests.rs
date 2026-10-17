//! The transform against two references (ported from jali's `ntt/oracle_tests.rs` and
//! extended): a reference transform with a remainder per product, the twiddles advanced by a
//! product per butterfly and every constant recomputed; and, for small sizes, direct evaluation
//! of the polynomial at $\psi^{2j+1}$ by Horner's rule. Equal outputs for all nine NTT primes
//! and every size from 2 to 4096, on boundary and random residues; the Montgomery form against
//! the standard one; the table against its definition; Montgomery products against remainders.
use super::*;
use crate::arith::tests::SplitMix;

/// The reference plan.
struct Reference {
    p: u64,
    d: usize,
    twist: Vec<u64>,
    untwist: Vec<u64>,
    root: u64,
}
impl Reference {
    fn new(p: u64, d: usize, psi: u64) -> Self {
        let inv_psi = pow_mod(psi, p - 2, p);
        let mut twist = vec![1; d];
        let mut untwist = vec![1; d];
        for i in 1..d {
            twist[i] = mul_mod(twist[i - 1], psi, p);
            untwist[i] = mul_mod(untwist[i - 1], inv_psi, p);
        }
        Self {
            p,
            d,
            twist,
            untwist,
            root: mul_mod(psi, psi, p),
        }
    }
    fn cyclic(&self, a: &mut [u64], root: u64) {
        let mut j = 0;
        for i in 1..self.d {
            let mut bit = self.d >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                a.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= self.d {
            let step = pow_mod(root, (self.d / len) as u64, self.p);
            for block in a.chunks_exact_mut(len) {
                let mut w = 1;
                for j in 0..len / 2 {
                    let u = block[j];
                    let v = mul_mod(block[j + len / 2], w, self.p);
                    block[j] = (u + v) % self.p;
                    block[j + len / 2] = (u + self.p - v) % self.p;
                    w = mul_mod(w, step, self.p);
                }
            }
            len *= 2;
        }
    }
    fn forward(&self, a: &mut [u64]) {
        for (x, t) in a.iter_mut().zip(&self.twist) {
            *x = mul_mod(*x, *t, self.p);
        }
        self.cyclic(a, self.root);
    }
    fn inverse(&self, a: &mut [u64]) {
        self.cyclic(a, pow_mod(self.root, self.p - 2, self.p));
        let inv_d = pow_mod(self.d as u64, self.p - 2, self.p);
        for (x, t) in a.iter_mut().zip(&self.untwist) {
            *x = mul_mod(mul_mod(*x, inv_d, self.p), *t, self.p);
        }
    }
}

/// $a(\psi^{2j+1})$ for $j < d$ by Horner's rule.
fn evaluate(a: &[u64], p: u64, psi: u64) -> Vec<u64> {
    let d = a.len();
    (0..d)
        .map(|j| {
            let x = pow_mod(psi, 2 * j as u64 + 1, p);
            a.iter()
                .rev()
                .fold(0, |acc, c| add_mod(mul_mod(acc, x, p), *c, p))
        })
        .collect()
}

/// Boundary vectors (zero, one, $p-1$, alternating, unit vectors at both ends) and random ones,
/// of length `d` over $[0, p)$.
fn inputs(p: u64, d: usize, rng: &mut SplitMix) -> Vec<Vec<u64>> {
    let mut out = vec![vec![0; d], vec![1; d], vec![p - 1; d]];
    out.push((0..d).map(|i| if i % 2 == 0 { p - 1 } else { 1 }).collect());
    for i in [0, d - 1] {
        let mut unit = vec![0; d];
        unit[i] = p - 1;
        out.push(unit);
    }
    for _ in 0..3 {
        out.push((0..d).map(|_| rng.next() % p).collect());
    }
    out
}

#[test]
fn the_table_matches_its_definition() {
    let mut last = 1u64 << 62;
    for (p, root) in NTT_PRIMES {
        assert!(p < last, "descending and below 2^62");
        last = p;
        assert!(is_prime(p));
        assert_eq!((p - 1) % (1 << 13), 0, "2^13 divides p - 1");
        // Order exactly 2^13.
        assert_eq!(pow_mod(root, 1 << 13, p), 1);
        assert_eq!(pow_mod(root, 1 << 12, p), p - 1);
    }
    // No prime of the required form is skipped between the first entry and 2^62, or between
    // consecutive entries.
    let mut candidate = NTT_PRIMES[0].0 + (1 << 13);
    while candidate < 1 << 62 {
        assert!(!is_prime(candidate), "{candidate} would precede the table");
        candidate += 1 << 13;
    }
    for pair in NTT_PRIMES.windows(2) {
        let mut candidate = pair[1].0 + (1 << 13);
        while candidate < pair[0].0 {
            assert!(
                !is_prime(candidate),
                "{candidate} is missing from the table"
            );
            candidate += 1 << 13;
        }
    }
}

#[test]
fn transforms_equal_the_reference_for_every_prime_and_degree() {
    let mut rng = SplitMix(1);
    for (index, (p, root)) in NTT_PRIMES.into_iter().enumerate() {
        let r = ((1u128 << 64) % u128::from(p)) as u64;
        for log_d in 1..=12 {
            let d = 1usize << log_d;
            let psi = root_of_unity(p, root, d);
            let plan = NttPlan::new(p, d, psi).unwrap();
            assert_eq!(plan.modulus(), p);
            assert_eq!(plan.degree(), d);
            let table = NttPlan::from_table(index, d).unwrap();
            let slow = Reference::new(p, d, psi);
            for a in inputs(p, d, &mut rng) {
                let (mut new, mut expected) = (a.clone(), a.clone());
                plan.forward(&mut new).unwrap();
                slow.forward(&mut expected);
                assert_eq!(new, expected, "forward p {p} d {d}");
                let mut from_table = a.clone();
                table.forward(&mut from_table).unwrap();
                assert_eq!(from_table, expected, "from_table p {p} d {d}");
                if d <= 64 {
                    assert_eq!(new, evaluate(&a, p, psi), "evaluation p {p} d {d}");
                }
                // The Montgomery form is the standard one times R.
                let mut mont = a.clone();
                plan.forward_mont(&mut mont);
                let scaled: Vec<u64> = new.iter().map(|x| mul_mod(*x, r, p)).collect();
                assert_eq!(mont, scaled, "forward_mont p {p} d {d}");
                plan.inverse_mont(&mut mont);
                assert_eq!(mont, a, "inverse_mont p {p} d {d}");
                // The inverse of an arbitrary vector, and the inverse of the transform.
                let (mut new_inv, mut expected_inv) = (a.clone(), a.clone());
                plan.inverse(&mut new_inv).unwrap();
                slow.inverse(&mut expected_inv);
                assert_eq!(new_inv, expected_inv, "inverse p {p} d {d}");
                plan.inverse(&mut new).unwrap();
                assert_eq!(new, a, "round trip p {p} d {d}");
            }
        }
    }
}

#[test]
fn unreduced_inputs_are_read_modulo_p_and_lengths_are_checked() {
    let (p, root) = NTT_PRIMES[0];
    let d = 8;
    let plan = NttPlan::new(p, d, root_of_unity(p, root, d)).unwrap();
    let a: Vec<u64> = vec![p, p + 1, u64::MAX, 2 * p - 1, 0, 1, p - 1, 3];
    let reduced: Vec<u64> = a.iter().map(|x| x % p).collect();
    let (mut x, mut y) = (a.clone(), reduced.clone());
    plan.forward(&mut x).unwrap();
    plan.forward(&mut y).unwrap();
    assert_eq!(x, y);
    let (mut x, mut y) = (a, reduced);
    plan.inverse(&mut x).unwrap();
    plan.inverse(&mut y).unwrap();
    assert_eq!(x, y);
    assert_eq!(
        plan.forward(&mut [0; 7]),
        Err(Error::Length {
            expected: 8,
            actual: 7
        })
    );
}

#[test]
fn invalid_plans_are_rejected() {
    let (p, root) = NTT_PRIMES[0];
    let psi = |d| root_of_unity(p, root, d);
    // Sizes: 1, 8192 and a non-power of two.
    assert!(NttPlan::new(p, 1, 1).is_err());
    assert!(NttPlan::new(p, 8192, psi(4096)).is_err());
    assert!(NttPlan::new(p, 12, psi(16)).is_err());
    assert!(NttPlan::from_table(0, 8192).is_err());
    assert!(NttPlan::from_table(9, 16).is_err());
    // A root of the wrong order: psi(8) has order 16, not 32.
    assert!(NttPlan::new(p, 16, psi(8)).is_err());
    // A root of order 2d that is not primitive in the required sense: psi^2 has order d.
    assert!(NttPlan::new(p, 16, mul_mod(psi(16), psi(16), p)).is_err());
    // A composite modulus with 2^13 | n - 1 (between the first two table primes).
    assert!(!is_prime(p - 8192));
    assert!(NttPlan::new(p - 8192, 16, psi(16) % (p - 8192)).is_err());
    // The smallest prime above 2^62 with 2^13 | p - 1, with a valid root of order 32 (SageMath
    // 10.9: root 795177494079348431 of order 2^13): rejected only for its size.
    let big = 4611686018427494401u64;
    let big_root = pow_mod(795177494079348431, 4096 / 16, big);
    assert!(is_prime(big) && big >= 1 << 62 && pow_mod(big_root, 16, big) == big - 1);
    assert!(NttPlan::new(big, 16, big_root).is_err());
    // A prime with 2d not dividing p - 1: 2^32 - 99 = 4 * odd + 1.
    assert!(NttPlan::new(4294967197, 4, 2).is_err());
    // The largest size is accepted.
    assert!(NttPlan::new(p, 4096, psi(4096)).is_ok());
}

#[test]
fn montgomery_products_equal_remainders() {
    let mut rng = SplitMix(2);
    for (p, root) in NTT_PRIMES {
        let plan = NttPlan::new(p, 2, root_of_unity(p, root, 2)).unwrap();
        assert_eq!(p.wrapping_mul(plan.neg_inv.wrapping_neg()), 1);
        let r_inv = pow_mod(((1u128 << 64) % u128::from(p)) as u64, p - 2, p);
        let mut values = vec![0, 1, 2, p / 2, p / 2 + 1, p - 2, p - 1];
        values.extend((0..40).map(|_| rng.next() % p));
        for &a in &values {
            for &b in &values {
                assert_eq!(plan.mont_mul(a, b), mul_mod(mul_mod(a, b, p), r_inv, p));
            }
        }
    }
}
