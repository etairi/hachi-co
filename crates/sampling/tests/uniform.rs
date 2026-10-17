//! The uniform samplers: exactness of the rejection rule on scripted words (every masked value of
//! small ranges, and the edges of large ones), the range and error boundaries, prefix
//! consistency, and chi-square tests on real streams (fixed seeds, threshold at
//! $1 - 10^{-4}$, see `common::chi_square_threshold`).

mod common;

use common::{Constant, Counting, Scripted, assert_chi_square};
use hachi_co_sampling::uniform::{
    MAX_TRIES, boxed, short, uniform_below, uniform_below_big, uniform_mod_q, uniform_range_i64,
};
use hachi_co_sampling::{Error, ShakePrg};
use num_bigint::BigUint;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// Bit length of $n - 1$.
fn bits(n: u64) -> u32 {
    64 - (n - 1).leading_zeros()
}

#[test]
fn every_masked_word_is_accepted_as_itself_or_rejected() {
    // For each n, every value w of the low bits(n) bits: w < n is returned as is after one word,
    // w >= n is rejected and the next word decides. High bits never matter. So the accepted
    // words map one-to-one onto [0, n): exact uniformity given uniform words.
    for n in 2u64..=70 {
        let b = bits(n);
        let mut hits = vec![0u32; n as usize];
        for w in 0..1u64 << b {
            for junk in [0u64, 1, u64::MAX >> b] {
                let word = w | (junk << b);
                let mut rng = Scripted::new(&[word, 0]);
                let x = uniform_below(&mut rng, n).unwrap();
                if w < n {
                    assert_eq!((x, rng.read), (w, 1), "n = {n}, word {word:#x}");
                    if junk == 0 {
                        hits[w as usize] += 1;
                    }
                } else {
                    assert_eq!(
                        (x, rng.read),
                        (0, 2),
                        "n = {n}, word {word:#x} must be rejected"
                    );
                }
            }
        }
        assert!(hits.iter().all(|h| *h == 1), "n = {n}");
    }
}

#[test]
fn large_ranges_accept_exactly_the_words_below_n() {
    for n in [
        1u64 << 32,
        (1 << 32) + 1,
        (1 << 63) - 1,
        1 << 63,
        (1 << 63) + 1,
        u64::MAX - 1,
        u64::MAX,
    ] {
        let mask = u64::MAX >> (n - 1).leading_zeros();
        for w in [0, 1, n - 2, n - 1, n, n.wrapping_add(1), mask - 1, mask] {
            if w > mask {
                continue;
            }
            let high = !mask; // bits above the mask are ignored
            let mut rng = Scripted::new(&[w | high, 7]);
            let x = uniform_below(&mut rng, n).unwrap();
            if w < n {
                assert_eq!((x, rng.read), (w, 1), "n = {n}, w = {w}");
            } else {
                assert_eq!((x, rng.read), (7, 2), "n = {n}, w = {w}");
            }
        }
    }
}

#[test]
fn degenerate_and_broken_cases() {
    let mut rng = Scripted::new(&[]);
    assert_eq!(uniform_below(&mut rng, 1), Ok(0));
    assert_eq!(uniform_below(&mut rng, 0), Err(Error::EmptyRange));
    assert_eq!(
        uniform_below_big(&mut rng, &BigUint::from(1u32)),
        Ok(BigUint::ZERO)
    );
    assert_eq!(
        uniform_below_big(&mut rng, &BigUint::ZERO),
        Err(Error::EmptyRange)
    );
    assert_eq!(uniform_range_i64(&mut rng, 5, 5), Ok(5));
    assert_eq!(
        uniform_range_i64(&mut rng, i64::MIN, i64::MIN),
        Ok(i64::MIN)
    );
    assert_eq!(uniform_range_i64(&mut rng, 1, 0), Err(Error::EmptyRange));
    assert_eq!(uniform_mod_q(&mut rng, 1, 5), Ok(vec![0; 5]));
    assert_eq!(uniform_mod_q(&mut rng, 0, 5), Err(Error::EmptyRange));
    assert_eq!(boxed(&mut rng, 0, 4), Ok(vec![0; 4]));
    assert_eq!(short(&mut rng, 1, 4), Ok(vec![0; 4]));
    assert_eq!(short(&mut rng, 0, 4), Err(Error::EmptyRange));
    assert!(matches!(
        boxed(&mut rng, 1 << 63, 1),
        Err(Error::InvalidParameter(_))
    ));
    assert_eq!(rng.read, 0, "none of these may read the stream");
    // A source stuck on a rejected word hits the cap after exactly MAX_TRIES words.
    let mut stuck = Constant(u64::MAX, 0);
    assert_eq!(
        uniform_below(&mut stuck, 3),
        Err(Error::RejectionLimit(MAX_TRIES))
    );
    assert_eq!(stuck.1, MAX_TRIES as usize);
    let mut stuck = Constant(u64::MAX, 0);
    let n = (BigUint::from(1u32) << 100u32) + 1u32;
    assert_eq!(
        uniform_below_big(&mut stuck, &n),
        Err(Error::RejectionLimit(MAX_TRIES))
    );
    assert_eq!(stuck.1, 2 * MAX_TRIES as usize, "two words per try");
}

#[test]
fn big_bounds_use_little_endian_limbs_masked_to_the_bit_length() {
    // n = 2^64 + 1: 65 bits, two words per try, the top word masked to one bit.
    let n = (BigUint::from(1u32) << 64u32) + 1u32;
    let two64 = BigUint::from(1u32) << 64u32;
    let cases: [(&[u64], BigUint, usize); 4] = [
        (&[5, 0xfffe], BigUint::from(5u32), 2),         // top bit 0
        (&[0, 0xffff], two64.clone(), 2),               // x = 2^64 < n
        (&[1, 1, 9, 2], BigUint::from(9u32), 4),        // x = 2^64 + 1 rejected
        (&[u64::MAX, 3, 3, 0], BigUint::from(3u32), 4), // rejected, then 3
    ];
    for (words, expected, read) in cases {
        let mut rng = Scripted::new(words);
        assert_eq!(
            uniform_below_big(&mut rng, &n).unwrap(),
            expected,
            "words {words:?}"
        );
        assert_eq!(rng.read, read);
    }
}

#[test]
fn big_bounds_up_to_2_64_draw_like_uniform_below() {
    for n in [2u64, 3, 1000, (1 << 63) + 1, u64::MAX] {
        let mut a = Counting::new(ShakePrg::new(&[3; 32], b"compare", n));
        let mut b = Counting::new(ShakePrg::new(&[3; 32], b"compare", n));
        for _ in 0..50 {
            assert_eq!(
                uniform_below_big(&mut a, &BigUint::from(n)).unwrap(),
                BigUint::from(uniform_below(&mut b, n).unwrap())
            );
        }
        assert_eq!(a.words, b.words, "n = {n}");
    }
    // n = 2^64 is not a u64: one word per try, always accepted.
    let mut rng = Scripted::new(&[u64::MAX, 0]);
    let two64 = BigUint::from(1u32) << 64u32;
    assert_eq!(
        uniform_below_big(&mut rng, &two64).unwrap(),
        BigUint::from(u64::MAX)
    );
    assert_eq!(rng.read, 1);
}

#[test]
fn ranges_are_closed_and_offset_from_lo() {
    // Full range: one word w, always accepted, mapped to i64::MIN + w.
    for (w, expected) in [(0u64, i64::MIN), (1 << 63, 0), (u64::MAX, i64::MAX)] {
        let mut rng = Scripted::new(&[w]);
        assert_eq!(
            uniform_range_i64(&mut rng, i64::MIN, i64::MAX),
            Ok(expected)
        );
    }
    // [-1, 1]: three values; word values 0, 1, 2 map to -1, 0, 1 and 3 is rejected.
    let mut rng = Scripted::new(&[0, 1, 2, 3, 2]);
    let got: Vec<i64> = (0..4)
        .map(|_| uniform_range_i64(&mut rng, -1, 1).unwrap())
        .collect();
    assert_eq!(got, vec![-1, 0, 1, 1]);
    // Near the ends of i64.
    let mut rng = Scripted::new(&[0, 2, 3, 1]);
    assert_eq!(
        uniform_range_i64(&mut rng, i64::MAX - 2, i64::MAX),
        Ok(i64::MAX - 2)
    );
    assert_eq!(
        uniform_range_i64(&mut rng, i64::MAX - 2, i64::MAX),
        Ok(i64::MAX)
    );
    assert_eq!(
        uniform_range_i64(&mut rng, i64::MIN, i64::MIN + 2),
        Ok(i64::MIN + 1)
    );
}

#[test]
fn short_and_boxed_ranges_match_the_draft() {
    // S_beta: [ceil(-beta/2), ceil(beta/2) - 1] (4_preliminaries.tex:88), beta values.
    for beta in 1u64..=9 {
        let lo = -((beta / 2) as i64);
        let hi = beta.div_ceil(2) as i64 - 1;
        assert_eq!(hi - lo + 1, beta as i64);
        // Scripted words 0..beta: the entries are lo, ..., hi in order.
        let words: Vec<u64> = (0..beta).collect();
        let mut rng = Scripted::new(&words);
        assert_eq!(
            short(&mut rng, beta, beta as usize).unwrap(),
            (lo..=hi).collect::<Vec<_>>()
        );
    }
    // The extreme betas: 2^64 - 1 gives [-(2^63 - 1), 2^63 - 1], and 2^63 gives [-2^62, 2^62 - 1].
    let mut rng = Scripted::new(&[0, u64::MAX - 1]);
    assert_eq!(
        short(&mut rng, u64::MAX, 2).unwrap(),
        vec![-i64::MAX, i64::MAX]
    );
    let mut rng = Scripted::new(&[0, (1 << 63) - 1]);
    assert_eq!(
        short(&mut rng, 1 << 63, 2).unwrap(),
        vec![-(1 << 62), (1 << 62) - 1]
    );
    // Boxes [-gamma, gamma]: words 0 and 2 gamma are the two ends.
    for gamma in [1u64, 2, 49152, i64::MAX as u64] {
        let mut rng = Scripted::new(&[0, 2 * gamma, gamma]);
        assert_eq!(
            boxed(&mut rng, gamma, 3).unwrap(),
            vec![-(gamma as i64), gamma as i64, 0]
        );
    }
}

#[test]
fn vector_draws_are_prefix_consistent() {
    let long = uniform_mod_q(&mut ShakePrg::new(&[1; 32], b"prefix", 0), 8388637, 100).unwrap();
    let short_ = uniform_mod_q(&mut ShakePrg::new(&[1; 32], b"prefix", 0), 8388637, 37).unwrap();
    assert_eq!(&long[..37], &short_[..]);
    let long = boxed(&mut ShakePrg::new(&[1; 32], b"prefix", 1), 5, 64).unwrap();
    let short_ = boxed(&mut ShakePrg::new(&[1; 32], b"prefix", 1), 5, 10).unwrap();
    assert_eq!(&long[..10], &short_[..]);
}

/// Counts of `samples` draws of a sampler with values in [lo, hi].
fn histogram(lo: i64, hi: i64, samples: usize, mut draw: impl FnMut() -> i64) -> Vec<u64> {
    let mut counts = vec![0u64; (hi - lo + 1) as usize];
    for _ in 0..samples {
        let x = draw();
        assert!((lo..=hi).contains(&x), "{x} outside [{lo}, {hi}]");
        counts[(x - lo) as usize] += 1;
    }
    counts
}

#[test]
fn small_ranges_pass_chi_square() {
    // ChaCha20, seed 20261017, an RNG independent of ShakePrg.
    let mut rng = ChaCha20Rng::seed_from_u64(20261017);
    for n in [2u64, 3, 5, 7, 10, 17] {
        let samples = 2000 * n as usize;
        let counts = histogram(0, n as i64 - 1, samples, || {
            uniform_below(&mut rng, n).unwrap() as i64
        });
        assert_chi_square(
            &format!("uniform_below({n})"),
            &counts,
            &vec![2000.0; n as usize],
        );
    }
    let mut rng = ShakePrg::new(&[0x11; 32], b"chi2/short", 0);
    for beta in [2u64, 3, 4, 5, 6] {
        let lo = -((beta / 2) as i64);
        let counts = histogram(lo, lo + beta as i64 - 1, 3000 * beta as usize, || {
            short(&mut rng, beta, 1).unwrap()[0]
        });
        assert_chi_square(
            &format!("short({beta})"),
            &counts,
            &vec![3000.0; beta as usize],
        );
    }
    let mut rng = ShakePrg::new(&[0x11; 32], b"chi2/boxed", 0);
    for gamma in [1u64, 2, 5] {
        let cells = 2 * gamma as usize + 1;
        let counts = histogram(-(gamma as i64), gamma as i64, 3000 * cells, || {
            boxed(&mut rng, gamma, 1).unwrap()[0]
        });
        assert_chi_square(&format!("boxed({gamma})"), &counts, &vec![3000.0; cells]);
    }
}
