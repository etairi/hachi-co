//! [`Ratio::log2`] against exact references: an independent big-integer evaluation of
//! $\log_2(a/b)$ to more than 300 significant bits (another normalization, the series
//! $\ln(1 + x) = \sum_{j \ge 1} (-1)^{j+1} x^j / j$ and $\ln 2 = \sum_{j \ge 1} 1/(j 2^j)$, no
//! floating point), SageMath's MPFR on named cases (the toy presets included), the special values,
//! the constant $2/\ln 2$, and a seeded sweep over long and short integers, ratios near 1, near
//! powers of two and near the switch points $2^{\pm 1/2}$ of the implementation, and results near
//! and below the subnormal range. Every case must meet the documented bound: below $0.52$ ulp of
//! the exact value for a normal result, below one ulp for a subnormal one.

use super::*;
use num_bigint::BigInt;
use num_traits::{One, ToPrimitive};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use std::sync::OnceLock;

/// The documented error bound of [`Ratio::log2`] for a normal result, in ulps.
const NORMAL_BOUND: f64 = 0.52;
/// The documented error bound of [`Ratio::log2`] for a subnormal result, in ulps ($2^{-1074}$).
const SUBNORMAL_BOUND: f64 = 1.0;

/// $\ln 2$ with `p` fractional bits from $\sum_{j \ge 1} 1/(j 2^j)$, each term truncated at
/// $2^{-p-16}$: the error is below $1 + (p + 17)/2^{16}$ units of $2^{-p}$.
fn ln2_series(p: u64) -> BigUint {
    const GUARD: u64 = 16;
    let one = BigUint::one() << (p + GUARD);
    let mut sum = BigUint::ZERO;
    for j in 1..=p + GUARD {
        sum += (&one >> j) / j;
    }
    sum >> GUARD
}

/// [`ln2_series`], truncated from one evaluation at 8192 bits when `p` is at most 8192.
fn ln2_fixed(p: u64) -> BigUint {
    const P_MAX: u64 = 8192;
    static LN2: OnceLock<BigUint> = OnceLock::new();
    if p > P_MAX {
        return ln2_series(p);
    }
    LN2.get_or_init(|| ln2_series(P_MAX)) >> (P_MAX - p)
}

/// $\log_2(a/b) = v / 2^p$ up to `err` units of $2^{-p}$ (`err` is 0 when the value is exact).
struct Exact {
    v: BigInt,
    p: u64,
    err: u64,
}

/// The reference: $\log_2(a/b) = e + \ln(1 + x) / \ln 2$ for $a, b \gt 0$, with $e$ chosen by
/// exact comparisons so that $1 + x = a/(2^e b) \in [3/4, 3/2)$, and $p = 320 + z$ fractional
/// bits for $|x| \lt 2^{1 - z}$, so that the relative error of $\ln(1 + x)/\ln 2$ is below
/// $2^{-300}$. The series terms and powers are truncated, each by less than a unit; with at most
/// $p$ terms of the series (as $|x| \lt 1/2$) the total error stays below $2^{10}$ units of
/// $2^{-p}$, and `err` is declared as $2^{12}$.
fn reference(a: &BigUint, b: &BigUint) -> Exact {
    let three = BigUint::from(3u32);
    let scaled = |e: i64| {
        if e >= 0 {
            (a.clone(), b << e.unsigned_abs())
        } else {
            (a << e.unsigned_abs(), b.clone())
        }
    };
    // a / (2^e b) is in (1/2, 2), and in [3/4, 3/2) after at most one step.
    let mut e = a.bits() as i64 - b.bits() as i64;
    let (mut aa, mut bb) = scaled(e);
    if (&aa << 2u32) < &bb * &three {
        e -= 1;
        (aa, bb) = scaled(e);
    } else if (&aa << 1u32) >= &bb * &three {
        e += 1;
        (aa, bb) = scaled(e);
    }
    if aa == bb {
        return Exact {
            v: BigInt::from(e),
            p: 0,
            err: 0,
        };
    }
    let negative = aa < bb;
    let diff = if negative { &bb - &aa } else { &aa - &bb };
    let z = bb.bits() - diff.bits();
    let p = 320 + z;
    // |x| 2^p, truncated; the terms of ln(1 + x) alternate for x > 0 and are all negative for
    // x < 0.
    let x = (diff << p) / &bb;
    let mut ln = BigInt::ZERO;
    let mut power = x.clone();
    let mut j = 1u64;
    while !power.is_zero() {
        let term = BigInt::from(&power / j);
        if negative || j.is_multiple_of(2) {
            ln -= term;
        } else {
            ln += term;
        }
        power = (&power * &x) >> p;
        j += 1;
    }
    let v = (BigInt::from(e) << p) + (ln << p) / BigInt::from(ln2_fixed(p));
    Exact { v, p, err: 1 << 12 }
}

/// $x \cdot 2^s$ as an `f64`, for error measurements (a relative error of $2^{-50}$ is
/// harmless there).
fn scaled_f64(x: &BigUint, s: i64) -> f64 {
    let drop = x.bits().saturating_sub(64);
    let top = (x >> drop).to_f64().unwrap_or(f64::INFINITY);
    let s = s + drop as i64;
    if top == 0.0 || s < -2000 {
        return 0.0;
    }
    if s > 2000 {
        return f64::INFINITY;
    }
    top * 2f64.powi((s / 2) as i32) * 2f64.powi((s - s / 2) as i32)
}

/// $|r - \mathrm{exact}|$ in ulps of the exact value ($2^{E - 52}$ for
/// $2^E \le |\mathrm{exact}| \lt 2^{E+1}$, with $E$ a lower bound that accounts for the
/// reference's error, and at least $2^{-1074}$), and whether the exact value is normal.
fn ulp_error(r: f64, exact: &Exact) -> (f64, bool) {
    assert!(r.is_finite(), "{r}");
    let bits = r.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i64;
    let fraction = bits & ((1 << 52) - 1);
    let (mantissa, q) = if biased == 0 {
        (fraction, -1074)
    } else {
        (fraction | 1 << 52, biased - 1075)
    };
    let m = if r.is_sign_negative() {
        -BigInt::from(mantissa)
    } else {
        BigInt::from(mantissa)
    };
    // (r - v / 2^p) 2^f, exactly.
    let p = exact.p as i64;
    let f = p.max(-q);
    let diff = (m << (q + f) as u64) - (&exact.v << (f - p) as u64);
    let magnitude = exact.v.magnitude();
    let lower = if magnitude > &BigUint::from(exact.err) {
        magnitude - exact.err
    } else {
        BigUint::ZERO
    };
    let exponent = if lower.is_zero() {
        -1075
    } else {
        lower.bits() as i64 - 1 - p
    };
    let ulp = exponent.max(-1022) - 52;
    (scaled_f64(diff.magnitude(), -(f + ulp)), exponent >= -1022)
}

/// Checks [`Ratio::log2`] of $a/b$ against [`reference`] and returns its error in ulps and
/// whether the exact value is normal.
fn check(a: &BigUint, b: &BigUint, what: &str) -> (f64, bool) {
    let r = Ratio {
        numerator: a.clone(),
        denominator: b.clone(),
    }
    .log2();
    let (error, normal) = ulp_error(r, &reference(a, b));
    let bound = if normal {
        NORMAL_BOUND
    } else {
        SUBNORMAL_BOUND
    };
    assert!(
        error < bound,
        "{what}: log2 = {r:e} is {error} ulp from the exact value (normal: {normal}); \
         bits of a and b: {}, {}",
        a.bits(),
        b.bits()
    );
    (error, normal)
}

#[test]
fn the_constant_is_two_over_ln_2() {
    // round(2^63 / ln 2) = floor((floor(2^64 / ln 2) + 1) / 2), with ln 2 from the series at 256
    // bits (an error near 2^-190 in 2^64 / ln 2, far from its distance to an integer).
    let twice = (BigUint::one() << 320u32) / ln2_fixed(256);
    assert_eq!((twice + 1u32) >> 1u32, BigUint::from(TWO_OVER_LN2_Q62));
}

#[test]
fn the_reference_knows_ln_2_and_simple_values() {
    // ln 2 = 0.6931471805599453094172321214581765680755..., first 128 bits (SageMath 10.9:
    // floor(log(RealField(4000)(2)) * 2^128) = 235865763225513294137944142764154484399).
    assert_eq!(
        ln2_fixed(128),
        BigUint::from(235_865_763_225_513_294_137_944_142_764_154_484_399u128)
    );
    // log2(3/2) and log2(1/3) through the reference, rounded to f64, are SageMath's values.
    let one = |a: u32, b: u32| reference(&BigUint::from(a), &BigUint::from(b));
    let to_f64 = |x: &Exact| scaled_f64(x.v.magnitude(), -(x.p as i64));
    assert_eq!(to_f64(&one(3, 2)).to_bits(), 0x3fe2_b803_473f_7ad1);
    assert_eq!(to_f64(&one(1, 3)).to_bits(), 0x3ff9_5c01_a39f_bd68);
}

#[test]
fn special_values_and_exact_powers_of_two() {
    let log2 = |a: &BigUint, b: &BigUint| {
        Ratio {
            numerator: a.clone(),
            denominator: b.clone(),
        }
        .log2()
    };
    let (zero, one, five) = (BigUint::ZERO, BigUint::one(), BigUint::from(5u32));
    assert_eq!(log2(&zero, &five), f64::NEG_INFINITY);
    assert_eq!(log2(&zero, &one), f64::NEG_INFINITY);
    assert_eq!(log2(&five, &zero), f64::INFINITY);
    assert!(log2(&zero, &zero).is_nan());
    let odd = [
        BigUint::one(),
        BigUint::from(3u32),
        BigUint::from(3u32).pow(500) + 2u32,
        (BigUint::one() << 5000u32) + 1u32,
    ];
    for x in &odd {
        // a / a = 1: +0 exactly.
        assert_eq!(log2(x, x).to_bits(), 0);
        for k in [1u64, 2, 52, 53, 63, 64, 65, 1000, 1074, 1075, 4096, 100_000] {
            let shifted = x << k;
            assert_eq!(log2(&shifted, x), k as f64, "2^{k} x / x");
            assert_eq!(log2(x, &shifted), -(k as f64), "x / (2^{k} x)");
        }
    }
}

/// $\log_2$ of the named cases, rounded to the nearest `f64` by MPFR (SageMath 10.9, 4000 bits
/// from the exact rational): the output of `tests/fixtures/log2_ratio.sage`.
const SAGE: [(&str, u64); 35] = [
    ("toy-n1", 0xbfe71724a67b9f85),
    ("toy-n2", 0xbfd716239db82a52),
    ("toy-n3", 0xbfcec761cd1e750e),
    ("2/1", 0x3ff0000000000000),
    ("1/2", 0xbff0000000000000),
    ("3/2", 0x3fe2b803473f7ad1),
    ("2/3", 0xbfe2b803473f7ad1),
    ("1/3", 0xbff95c01a39fbd68),
    ("7/5", 0x3fdf113baed305dc),
    ("1/2^3000", 0xc0a7700000000000),
    ("(2^1000+1)/2^1000", 0x01771547652b82fe),
    ("(2^1000-1)/2^1000", 0x81771547652b82fe),
    ("2^1000/(2^1000-1)", 0x01771547652b82fe),
    ("(2^1060+1)/2^1060", 0x0000000000005c55),
    ("(2^1050-1)/2^1050", 0x8000000001715476),
    ("(2^1100+1)/2^1100", 0x0000000000000000),
    ("3^2000/2^3170", 0xbfb3331b0083fc71),
    ("1/3^1000", 0xc098c3d999c9fef8),
    ("3^1000/1", 0x4098c3d999c9fef8),
    ("(2^64+1)2^5/2^64", 0x4014000000000000),
    ("1/(2^4000+1)", 0xc0af400000000000),
    ("isqrt(2^1001)/2^500", 0x3fe0000000000000),
    ("(isqrt(2^1001)+1)/2^500", 0x3fe0000000000000),
    ("2^500/isqrt(2^1001)", 0xbfe0000000000000),
    ("2^500/(isqrt(2^1001)+1)", 0xbfe0000000000000),
    ("65537^1000/65536^1000", 0x3f968abc73702518),
    ("F(5001)/F(5000)", 0x3fe6373ad151ca68),
    ("(10^300+3)/10^300", 0x01c7301a5d9b1c3d),
    ("(2^1022+1)/2^1022", 0x00171547652b82fe),
    ("(2^1023+1)/2^1023", 0x000b8aa3b295c17f),
    ("(2^1075+1)/2^1075", 0x0000000000000001),
    ("(2^1076+1)/2^1076", 0x0000000000000000),
    ("2^1100/(2^1100+1)", 0x8000000000000000),
    ("2^100000/3", 0x40f869e6a3fe5c60),
    ("3^100000/2^158496", 0x3fd0012e798e2c7c),
];

/// The numerator and denominator of a named case of [`SAGE`] (the definitions of
/// `tests/fixtures/log2_ratio.sage`).
fn named_case(name: &str) -> (BigUint, BigUint) {
    let n = |x: u64| BigUint::from(x);
    let pow = |base: u64, e: u32| BigUint::from(base).pow(e);
    let p2 = |e: u64| BigUint::one() << e;
    let toy = |gamma: u64, beta1: u64, big_n: usize| {
        let r = acceptance_probability(gamma, beta1, big_n).unwrap();
        (r.numerator, r.denominator)
    };
    let fibonacci = |k: usize| {
        let (mut f, mut g) = (BigUint::ZERO, BigUint::one());
        for _ in 0..k {
            (f, g) = (g.clone(), f + g);
        }
        f
    };
    let s1001 = p2(1001).sqrt();
    match name {
        "toy-n1" => toy(49152, 32, 768),
        "toy-n2" => toy(106496, 32, 832),
        "toy-n3" => toy(172032, 16, 1792),
        "2/1" => (n(2), n(1)),
        "1/2" => (n(1), n(2)),
        "3/2" => (n(3), n(2)),
        "2/3" => (n(2), n(3)),
        "1/3" => (n(1), n(3)),
        "7/5" => (n(7), n(5)),
        "1/2^3000" => (n(1), p2(3000)),
        "(2^1000+1)/2^1000" => (p2(1000) + 1u32, p2(1000)),
        "(2^1000-1)/2^1000" => (p2(1000) - 1u32, p2(1000)),
        "2^1000/(2^1000-1)" => (p2(1000), p2(1000) - 1u32),
        "(2^1060+1)/2^1060" => (p2(1060) + 1u32, p2(1060)),
        "(2^1050-1)/2^1050" => (p2(1050) - 1u32, p2(1050)),
        "(2^1100+1)/2^1100" => (p2(1100) + 1u32, p2(1100)),
        "3^2000/2^3170" => (pow(3, 2000), p2(3170)),
        "1/3^1000" => (n(1), pow(3, 1000)),
        "3^1000/1" => (pow(3, 1000), n(1)),
        "(2^64+1)2^5/2^64" => ((p2(64) + 1u32) << 5u32, p2(64)),
        "1/(2^4000+1)" => (n(1), p2(4000) + 1u32),
        "isqrt(2^1001)/2^500" => (s1001, p2(500)),
        "(isqrt(2^1001)+1)/2^500" => (s1001 + 1u32, p2(500)),
        "2^500/isqrt(2^1001)" => (p2(500), s1001),
        "2^500/(isqrt(2^1001)+1)" => (p2(500), s1001 + 1u32),
        "65537^1000/65536^1000" => (pow(65537, 1000), pow(65536, 1000)),
        "F(5001)/F(5000)" => (fibonacci(5001), fibonacci(5000)),
        "(10^300+3)/10^300" => (pow(10, 300) + 3u32, pow(10, 300)),
        "(2^1022+1)/2^1022" => (p2(1022) + 1u32, p2(1022)),
        "(2^1023+1)/2^1023" => (p2(1023) + 1u32, p2(1023)),
        "(2^1075+1)/2^1075" => (p2(1075) + 1u32, p2(1075)),
        "(2^1076+1)/2^1076" => (p2(1076) + 1u32, p2(1076)),
        "2^1100/(2^1100+1)" => (p2(1100), p2(1100) + 1u32),
        "2^100000/3" => (p2(100_000), n(3)),
        "3^100000/2^158496" => (pow(3, 100_000), p2(158_496)),
        _ => unreachable!("unknown case {name}"),
    }
}

/// The position of `x` among the doubles ordered by value, with $\pm 0$ at 0, so that adjacent
/// doubles differ by 1.
fn ordinal(x: f64) -> i64 {
    let magnitude = (x.to_bits() & !(1 << 63)) as i64;
    if x.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    }
}

#[test]
fn named_cases_match_sagemath_and_the_reference() {
    let mut exact = 0;
    for (name, sage_bits) in SAGE {
        let (a, b) = named_case(name);
        let sage = f64::from_bits(sage_bits);
        let r = Ratio {
            numerator: a.clone(),
            denominator: b.clone(),
        }
        .log2();
        // The reference agrees with MPFR: SageMath's double is within half an ulp of it.
        let (sage_error, _) = ulp_error(sage, &reference(&a, &b));
        assert!(
            sage_error <= 0.5 + 1e-9,
            "{name}: SageMath {sage_error} ulp"
        );
        // Ratio::log2 meets its bound, hence is SageMath's double or a neighbour.
        let (error, normal) = check(&a, &b, name);
        let distance = (ordinal(r) - ordinal(sage)).abs();
        assert!(distance <= 1, "{name}: {r:e} vs SageMath {sage:e}");
        exact += usize::from(distance == 0);
        println!(
            "{name:26} {r:<24e} {error:.4} ulp (normal: {normal}), SageMath distance {distance}"
        );
    }
    println!(
        "{exact} of {} named cases equal SageMath's correctly rounded double",
        SAGE.len()
    );
}

/// A uniform integer with exactly `bits` $\ge 1$ bits.
fn random_bits(rng: &mut ChaCha20Rng, bits: u64) -> BigUint {
    let words = bits.div_ceil(32);
    let digits = (0..words).map(|_| rng.random::<u32>()).collect();
    (BigUint::new(digits) >> (32 * words - bits)) | (BigUint::one() << (bits - 1))
}

/// $b \pm \delta$ for a random sign, or $b + \delta$ when $\delta \ge b$.
fn perturb(rng: &mut ChaCha20Rng, b: &BigUint, delta: &BigUint) -> BigUint {
    if rng.random::<bool>() && delta < b {
        b - delta
    } else {
        b + delta
    }
}

/// The seed of the sweep's ChaCha20 (`ChaCha20Rng::seed_from_u64`).
const SEED: u64 = 0x1092_a710;
/// The number of cases per category of the sweep.
const PER_CATEGORY: usize = 20_000;

/// One random case of the sweep's category `category`.
fn sweep_case(rng: &mut ChaCha20Rng, category: usize) -> (BigUint, BigUint) {
    let swap = rng.random::<bool>();
    let (a, b) = match category {
        // Independent integers of 1 to 2048 bits.
        0 => {
            let (x, y) = (rng.random_range(1..=2048), rng.random_range(1..=2048));
            (random_bits(rng, x), random_bits(rng, y))
        }
        // Ratios near 1: a = b +- delta with delta of 1 to bits(b) bits.
        1 => {
            let bits = rng.random_range(1..=2048);
            let b = random_bits(rng, bits);
            let delta_bits = rng.random_range(1..=bits);
            let delta = random_bits(rng, delta_bits);
            (perturb(rng, &b, &delta), b)
        }
        // Ratios near 2^k, 1 <= k <= 80.
        2 => {
            let bits = rng.random_range(1..=1024);
            let b = random_bits(rng, bits);
            let delta_bits = rng.random_range(1..=bits);
            let delta = random_bits(rng, delta_bits);
            let k = rng.random_range(1..=80u64);
            (perturb(rng, &b, &delta) << k, b)
        }
        // Near the switch points 2^(+-1/2) of the exponent choice, times 2^k: a = isqrt(2 b^2) +
        // delta with |delta| <= 3.
        3 => {
            // b >= 4, so that the root is at least 5 > |delta|.
            let bits = rng.random_range(3..=1024);
            let b = random_bits(rng, bits);
            let root = (&b * &b * 2u32).sqrt();
            let delta = rng.random_range(-3..=3i64);
            let a = if delta >= 0 {
                root + delta.unsigned_abs()
            } else {
                root - delta.unsigned_abs()
            };
            (a << rng.random_range(0..=8u64), b)
        }
        // Results near and below the subnormal range: b of 990 to 1160 bits, delta of 1 to 24.
        4 => {
            let bits = rng.random_range(990..=1160);
            let b = random_bits(rng, bits);
            let delta_bits = rng.random_range(1..=24);
            let delta = random_bits(rng, delta_bits);
            (perturb(rng, &b, &delta), b)
        }
        // Short integers.
        _ => (
            BigUint::from(rng.random_range(1..=1u64 << 20)),
            BigUint::from(rng.random_range(1..=1u64 << 20)),
        ),
    };
    if swap { (b, a) } else { (a, b) }
}

#[test]
fn a_seeded_sweep_stays_within_the_bound() {
    const NAMES: [&str; 6] = [
        "independent",
        "near 1",
        "near 2^k",
        "near 2^(+-1/2)",
        "near the subnormal range",
        "short integers",
    ];
    let mut rng = ChaCha20Rng::seed_from_u64(SEED);
    for (category, name) in NAMES.iter().enumerate() {
        let (mut max_normal, mut max_subnormal) = (0f64, 0f64);
        let (mut subnormal, mut above_half) = (0, 0);
        for i in 0..PER_CATEGORY {
            let (a, b) = sweep_case(&mut rng, category);
            let (error, normal) = check(&a, &b, &format!("{name}, case {i} (seed {SEED:#x})"));
            if normal {
                max_normal = max_normal.max(error);
            } else {
                subnormal += 1;
                max_subnormal = max_subnormal.max(error);
            }
            above_half += usize::from(error > 0.5);
        }
        println!(
            "{name}: {PER_CATEGORY} cases, max error {max_normal:.4} ulp (normal), \
             {max_subnormal:.4} ulp over {subnormal} subnormal or zero results, \
             {above_half} above 0.5 ulp"
        );
    }
}
