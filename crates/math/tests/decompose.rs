//! Decompositions: $\delta$ and $\tau$ against brute force and the toy table of
//! `docs/m0-spec.md`; $\mathbf G^{-1}$ exhaustively for small moduli and at the window boundary
//! for the test moduli; $\mathbf J^{-1}$ at both ends of the representable window and one
//! beyond; the vector forms and their gadgets (seeds 30-39).
mod common;

use common::*;
use hachi_co_math::{
    Error, Poly, Ring, Zq,
    decompose::{
        digit_range, draft_tau, g_apply, g_inv, g_inv_scalar, j_apply, j_inv, j_inv_scalar,
        num_digits, representable_range, tight_tau,
    },
};
use num_bigint::{BigInt, BigUint};
use rand::RngExt;

/// $\sum_i a_i b^i$ over $\mathbb Z$.
fn recompose(digits: &[i64], b: u64) -> i128 {
    digits
        .iter()
        .rev()
        .fold(0i128, |acc, a| acc * i128::from(b) + i128::from(*a))
}

fn in_digit_range(digits: &[i64], b: u64) -> bool {
    let (lo, hi) = digit_range(b);
    digits.iter().all(|a| (lo..=hi).contains(&i128::from(*a)))
}

/// $S_k \lceil -b/2 \rceil$ and $S_k (\lceil b/2 \rceil - 1)$ by big integers, clamped to the
/// `i128` range.
fn window_big(b: u64, k: usize) -> (i128, i128) {
    let s: BigInt = (0..k).map(|i| BigInt::from(b).pow(i as u32)).sum();
    let clamp = |x: BigInt| -> i128 {
        i128::try_from(&x).unwrap_or(if x < BigInt::from(0) {
            i128::MIN
        } else {
            i128::MAX
        })
    };
    (
        clamp(-(&s * BigInt::from(b / 2))),
        clamp(&s * BigInt::from((b - 1) / 2)),
    )
}

#[test]
fn digit_counts_equal_brute_force_and_the_toy_table() {
    // The toy presets of docs/m0-spec.md (b = 4).
    for (q, delta, beta, tau) in [
        (8388637u64, 12usize, 49152u128, 9usize),
        (33554501, 13, 212992, 10),
        (134217757, 14, 516096, 11),
    ] {
        assert_eq!(num_digits(q, 4).unwrap(), delta);
        assert_eq!(tight_tau(beta, 4).unwrap(), tau);
    }
    for q in [2u64, 3, 5, 13, 16, 17, Q32, Q52, Q61, Q63, u64::MAX] {
        for b in [
            2u64,
            3,
            4,
            5,
            7,
            16,
            1 << 20,
            q.max(3) - 1,
            q,
            q.saturating_add(1),
        ] {
            let delta = num_digits(q, b).unwrap();
            let pow = |k: usize| BigUint::from(b).pow(k as u32);
            assert!(pow(delta) >= BigUint::from(q), "q {q} b {b}");
            assert!(
                delta == 0 || pow(delta - 1) < BigUint::from(q),
                "q {q} b {b}: least"
            );
        }
    }
    let mut betas: Vec<u128> = (0..2000).collect();
    betas.extend([
        49152,
        212992,
        516096,
        1 << 64,
        1 << 100,
        u128::MAX / 3,
        u128::MAX,
    ]);
    for b in [3u64, 4, 5, 7, 16, 255, 256] {
        for &beta in &betas {
            let tau = tight_tau(beta, b).unwrap();
            let top = |k: usize| -> BigUint {
                let s: BigUint = (0..k).map(|i| BigUint::from(b).pow(i as u32)).sum();
                s * BigUint::from((b - 1) / 2)
            };
            assert!(top(tau) >= BigUint::from(beta), "b {b} beta {beta}");
            assert!(
                tau == 0 || top(tau - 1) < BigUint::from(beta),
                "b {b} beta {beta}: least"
            );
            if beta >= 1 {
                let draft = draft_tau(beta, b).unwrap();
                let t = draft - 1;
                assert!(BigUint::from(b).pow(t as u32) >= BigUint::from(beta));
                assert!(t == 0 || BigUint::from(b).pow(t as u32 - 1) < BigUint::from(beta));
                // The draft's tau is representable for b >= 3 (6_CH:753-761), hence >= tight.
                assert!(draft >= tau, "b {b} beta {beta}");
            }
        }
    }
    assert_eq!(tight_tau(0, 2).unwrap(), 0);
    assert!(tight_tau(1, 2).is_err());
    assert!(tight_tau(1, 1).is_err());
    assert!(draft_tau(0, 4).is_err());
    assert!(draft_tau(5, 1).is_err());
    assert!(num_digits(1, 4).is_err());
    assert!(num_digits(13, 1).is_err());
}

#[test]
fn windows_equal_their_definition() {
    for b in [2u64, 3, 4, 5, 7, 16, 1 << 20, 1 << 40, u64::MAX] {
        for k in 0..=200 {
            assert_eq!(representable_range(b, k), window_big(b, k), "b {b} k {k}");
            let (min, max) = representable_range(b, k);
            // b^k integers, while the window fits.
            if min > i128::MIN && max < i128::MAX {
                assert_eq!(
                    BigInt::from(max) - BigInt::from(min) + 1,
                    BigInt::from(b).pow(k as u32)
                );
            }
        }
    }
    // S_k fits u128 but not i128: base 3 with 81 digits (S = (3^81 - 1)/2, about 2^127.9).
    let s81: BigInt = (0..81).map(|i| BigInt::from(3u32).pow(i)).sum();
    assert!(s81 > BigInt::from(i128::MAX) && s81 < BigInt::from(u128::MAX));
    assert_eq!(representable_range(3, 81), (i128::MIN, i128::MAX));
    // Saturation instead of overflow.
    assert_eq!(representable_range(4, 200), (i128::MIN, i128::MAX));
    assert_eq!(representable_range(2, 200).1, 0);
    // The spec's window for even b; for odd b the negative end is -(ceil(b/2) - 1) S.
    assert_eq!(digit_range(4), (-2, 1));
    assert_eq!(digit_range(5), (-2, 2));
    assert_eq!(digit_range(2), (-1, 0));
    assert_eq!(digit_range(3), (-1, 1));
}

/// Every residue of small moduli (and its unreduced copy), every base, and $\delta$ from the
/// least admissible value up: digits in range, recomposition congruent to $x$, and the chosen
/// representative ($x$ up to the window's top, $x - q$ above).
#[test]
fn g_inverse_is_exact_for_every_residue_of_small_moduli() {
    for q in [3u64, 5, 13, 29, 37, 101, 8191] {
        let zq = Zq::new(q).unwrap();
        for b in [2u64, 3, 4, 5, 7, 16] {
            let least = num_digits(q, b).unwrap();
            for delta in least..least + 3 {
                let (_, top) = representable_range(b, delta);
                let mut out = vec![0i64; delta];
                for x in 0..q {
                    for input in [x, x + q] {
                        g_inv_scalar(&zq, b, delta, input, &mut out).unwrap();
                        assert!(in_digit_range(&out, b), "q {q} b {b} delta {delta} x {x}");
                        let want = if i128::from(x) <= top {
                            i128::from(x)
                        } else {
                            i128::from(x) - i128::from(q)
                        };
                        assert_eq!(recompose(&out, b), want, "q {q} b {b} delta {delta} x {x}");
                    }
                }
            }
            // One digit fewer than needed: b^(delta - 1) < q.
            if least > 1 {
                let mut out = vec![0i64; least - 1];
                assert!(matches!(
                    g_inv_scalar(&zq, b, least - 1, 0, &mut out),
                    Err(Error::Parameter(_))
                ));
            }
        }
    }
}

/// The test moduli at the window boundary: the top of the window is decomposed as itself with
/// all digits $\lceil b/2 \rceil - 1$, the next residue as itself minus $q$, and the largest
/// residues near the bottom; plus random residues.
#[test]
fn g_inverse_at_the_window_boundary_of_the_test_moduli() {
    let mut rng = rng(30);
    for q in ALL_MODULI {
        let zq = Zq::new(q).unwrap();
        for b in [3u64, 4, 5, 16, 1 << 16] {
            let delta = num_digits(q, b).unwrap();
            let (min, top) = representable_range(b, delta);
            assert!(top - min + 1 >= i128::from(q));
            let mut xs = vec![0u64, 1, (q - 1) / 2, q.div_ceil(2), q - 2, q - 1];
            if top < i128::from(q) {
                xs.extend([top as u64 - 1, top as u64]);
                if top + 1 < i128::from(q) {
                    xs.push(top as u64 + 1);
                }
            }
            xs.extend((0..50).map(|_| rng.random_range(0..q)));
            let mut out = vec![0i64; delta];
            for x in xs {
                g_inv_scalar(&zq, b, delta, x, &mut out).unwrap();
                assert!(in_digit_range(&out, b));
                let value = recompose(&out, b);
                assert!(min <= value && value <= top);
                let want = if i128::from(x) <= top {
                    i128::from(x)
                } else {
                    i128::from(x) - i128::from(q)
                };
                assert_eq!(value, want, "q {q} b {b} x {x}");
                if i128::from(x) == top {
                    let (_, hi) = digit_range(b);
                    assert!(out.iter().all(|a| i128::from(*a) == hi));
                }
            }
            assert!(g_inv_scalar(&zq, b, delta, 0, &mut vec![0; delta + 1]).is_err());
            assert!(g_inv_scalar(&zq, b, 0, 0, &mut []).is_err());
            assert!(g_inv_scalar(&zq, 1, delta, 0, &mut vec![0; delta]).is_err());
            assert!(
                g_inv_scalar(&zq, b, delta - 1, 0, &mut vec![0; delta - 1]).is_err(),
                "b^(delta-1) < q"
            );
        }
    }
}

/// $\mathbf J^{-1}$: every integer from three below the window to three above, for small bases
/// and digit counts; the ends pass with constant digits and one beyond fails with the window in
/// the error; the `i64` extremes for long digit vectors.
#[test]
fn j_inverse_passes_on_the_window_and_fails_one_beyond() {
    for b in [2u64, 3, 4, 5, 7, 16] {
        for tau in 1..=5usize {
            let (min, max) = representable_range(b, tau);
            let mut out = vec![0i64; tau];
            for z in (min - 3)..=(max + 3) {
                let result = j_inv_scalar(b, tau, z as i64, &mut out);
                if (min..=max).contains(&z) {
                    result.unwrap();
                    assert!(in_digit_range(&out, b));
                    assert_eq!(recompose(&out, b), z, "b {b} tau {tau} z {z}");
                } else {
                    assert_eq!(result, Err(Error::NotRepresentable { value: z, min, max }));
                }
            }
            let (lo, hi) = digit_range(b);
            j_inv_scalar(b, tau, min as i64, &mut out).unwrap();
            assert!(out.iter().all(|a| i128::from(*a) == lo));
            j_inv_scalar(b, tau, max as i64, &mut out).unwrap();
            assert!(out.iter().all(|a| i128::from(*a) == hi));
        }
    }
    // Larger windows: the toy presets' tight tau at beta and the ends.
    for (beta, tau) in [(49152i64, 9usize), (212992, 10), (516096, 11)] {
        let (min, max) = representable_range(4, tau);
        let mut out = vec![0i64; tau];
        for z in [-beta, beta, min as i64, max as i64] {
            j_inv_scalar(4, tau, z, &mut out).unwrap();
            assert_eq!(recompose(&out, 4), i128::from(z));
        }
        assert!(j_inv_scalar(4, tau, max as i64 + 1, &mut out).is_err());
        assert!(j_inv_scalar(4, tau, min as i64 - 1, &mut out).is_err());
        // One digit fewer does not represent beta (tau is tight).
        let mut short = vec![0i64; tau - 1];
        assert!(j_inv_scalar(4, tau - 1, beta, &mut short).is_err());
    }
    let mut out = vec![0i64; 64];
    for z in [i64::MIN, i64::MIN + 1, -1, 0, 1, i64::MAX] {
        j_inv_scalar(4, 64, z, &mut out).unwrap();
        assert_eq!(recompose(&out, 4), i128::from(z));
    }
    assert!(j_inv_scalar(4, 0, 0, &mut []).is_err());
    assert!(j_inv_scalar(4, 3, 0, &mut [0; 2]).is_err());
    assert!(j_inv_scalar(1, 3, 0, &mut [0; 3]).is_err());
}

#[test]
fn vector_decompositions_have_the_element_major_layout_and_invert_their_gadgets() {
    let mut rng = rng(31);
    for q in MODULI {
        for d in [16usize, 64] {
            let ring = Ring::new(q, d).unwrap();
            let zq = *ring.zq();
            for b in [3u64, 4, 16] {
                // G^{-1} of three random entries and one at the window top.
                let delta = num_digits(q, b).unwrap();
                let (_, top) = representable_range(b, delta);
                let mut t: Vec<Poly> = (0..3).map(|_| Poly(residues(&mut rng, q, d))).collect();
                t.push(Poly(vec![zq.reduce(top.min(i128::from(q - 1)) as u64); d]));
                let s = g_inv(&ring, b, delta, &t).unwrap();
                assert_eq!(s.len(), t.len() * delta);
                let mut digits = vec![0i64; delta];
                for (i, entry) in t.iter().enumerate() {
                    for (j, x) in entry.0.iter().enumerate() {
                        g_inv_scalar(&zq, b, delta, *x, &mut digits).unwrap();
                        for (k, digit) in digits.iter().enumerate() {
                            assert_eq!(s[i * delta + k].0[j], zq.from_i64(*digit), "layout");
                        }
                    }
                }
                assert_eq!(g_apply(&ring, b, delta, &s).unwrap(), t, "G G^-1 = id");
                // J^{-1} of short entries, at the window ends, and one beyond.
                let tau = tight_tau(1000, b).unwrap();
                let (min, max) = representable_range(b, tau);
                let mut z: Vec<Poly> = (0..2)
                    .map(|_| {
                        let v: Vec<i64> = (0..d)
                            .map(|_| rng.random_range(min as i64..=max as i64))
                            .collect();
                        ring.from_i64(&v).unwrap()
                    })
                    .collect();
                z.push(ring.from_i64(&vec![min as i64; d]).unwrap());
                z.push(ring.from_i64(&vec![max as i64; d]).unwrap());
                let s = j_inv(&ring, b, tau, &z).unwrap();
                assert_eq!(s.len(), z.len() * tau);
                for (i, entry) in z.iter().enumerate() {
                    for (j, x) in entry.0.iter().enumerate() {
                        let mut digits = vec![0i64; tau];
                        j_inv_scalar(b, tau, zq.to_centered(*x), &mut digits).unwrap();
                        for (k, digit) in digits.iter().enumerate() {
                            assert_eq!(s[i * tau + k].0[j], zq.from_i64(*digit));
                        }
                    }
                }
                assert_eq!(j_apply(&ring, b, tau, &s).unwrap(), z, "J J^-1 = id");
                let mut beyond = z.clone();
                beyond[1].0[d - 1] = zq.from_i64(max as i64 + 1);
                assert_eq!(
                    j_inv(&ring, b, tau, &beyond),
                    Err(Error::NotRepresentable {
                        value: max + 1,
                        min,
                        max
                    })
                );
                beyond[1].0[d - 1] = zq.from_i64(min as i64 - 1);
                assert!(j_inv(&ring, b, tau, &beyond).is_err());
                // Shapes.
                assert!(delta >= 2);
                assert!(g_apply(&ring, b, delta, &vec![ring.zero(); delta + 1]).is_err());
                assert!(g_inv(&ring, b, delta, &[Poly(vec![0; d - 1])]).is_err());
                // Parameters are checked even without input.
                assert!(g_inv(&ring, b, delta - 1, &[]).is_err(), "b^(delta-1) < q");
                assert!(g_inv(&ring, b, 0, &[]).is_err());
                assert!(j_inv(&ring, b, 0, &[]).is_err());
                assert_eq!(g_inv(&ring, b, delta, &[]).unwrap(), Vec::<Poly>::new());
                assert!(j_apply(&ring, b, 0, &s).is_err());
            }
        }
    }
}
