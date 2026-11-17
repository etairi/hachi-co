use super::*;

/// $|c(\zeta)|^2$ maximised over the primitive 128th roots of unity, with the platform's `cos`
/// and `sin`: an independent computation of [`opnorm_sq`].
fn opnorm_sq_libm(c: &[i8; N]) -> f64 {
    let mut best = 0f64;
    for k in 0..N {
        let theta = std::f64::consts::PI * (2 * k + 1) as f64 / 64.0;
        let (mut re, mut im) = (0f64, 0f64);
        for (j, v) in c.iter().enumerate() {
            re += f64::from(*v) * (theta * j as f64).cos();
            im += f64::from(*v) * (theta * j as f64).sin();
        }
        best = best.max(re * re + im * im);
    }
    best
}

/// What the reference's `poly_opnorm` computes, squared: its `poly_fft` reads $c_j + c_j i$ for
/// $j < 32$, so it evaluates $(1 + i) c_{\mathrm{lo}}$ at the 32 roots
/// $e^{i\pi(4l+1)/64}$ of $X^{32} = i$.
fn reference_check_sq(c: &[i8; N]) -> f64 {
    let mut best = 0f64;
    for l in 0..N / 2 {
        let (mut re, mut im) = (0f64, 0f64);
        for (j, v) in c[..N / 2].iter().enumerate() {
            let t = ((4 * l + 1) * j) % 128;
            re += f64::from(*v) * cos128(t);
            im += f64::from(*v) * sin128(t);
        }
        best = best.max(2.0 * (re * re + im * im));
    }
    best
}

#[test]
fn table_is_the_cosine() {
    // The platform's functions of the rounded argument differ from the correctly rounded table
    // by a few ulps for large t; an index or sign error would differ by far more.
    for t in 0..256 {
        let x = std::f64::consts::PI * t as f64 / 64.0;
        assert!((cos128(t) - x.cos()).abs() < 1e-14, "cos {t}");
        assert!((sin128(t) - x.sin()).abs() < 1e-14, "sin {t}");
    }
    for (t, c) in COS.iter().enumerate() {
        let x = std::f64::consts::PI * t as f64 / 64.0;
        assert!((c - x.cos()).abs() <= 2.0 * f64::EPSILON, "table {t}");
    }
}

#[test]
fn challenges_have_the_reference_weights_and_norm() {
    let seed = [7u8; 32];
    let mut draws = 0u64;
    let count = 2000u64;
    for i in 0..count {
        let (c, d) = Challenge::sample_counted(&seed, i);
        draws += u64::from(d);
        let ones = c.coeffs.iter().filter(|x| x.unsigned_abs() == 1).count();
        let twos = c.coeffs.iter().filter(|x| x.unsigned_abs() == 2).count();
        assert_eq!((ones, twos), (TAU1, TAU2));
        assert_eq!(c.l1_norm(), (TAU1 + 2 * TAU2) as u64);
        let norm = opnorm_sq(&c.coeffs);
        assert!(norm <= T * T);
        assert!((norm - opnorm_sq_libm(&c.coeffs)).abs() < 1e-9);
        assert_eq!(Challenge::sample(&seed, i), c);
    }
    assert_ne!(Challenge::sample(&seed, 0), Challenge::sample(&seed, 1));
    assert_ne!(
        Challenge::sample(&seed, 0),
        Challenge::sample(&[8u8; 32], 0)
    );
    let mean = draws as f64 / count as f64;
    eprintln!("draws per challenge: {mean:.2}");
    assert!((1.0..20.0).contains(&mean));
}

/// The reference's check against the operator norm on raw draws (before any check): it accepts
/// draws whose operator norm exceeds $T$ and rejects draws within it.
#[test]
fn reference_check_is_not_the_operator_norm() {
    let seed = [9u8; 32];
    let mut bytes = Bytes {
        prg: ShakePrg::new(&seed, b"test", 0),
        buf: [0; 168],
        pos: 168,
    };
    let (mut true_ok, mut ref_ok, mut ref_ok_true_bad, mut largest) = (0, 0, 0, 0f64);
    let count = 20000;
    for _ in 0..count {
        let c = Challenge::draw(&mut bytes);
        let truth = opnorm_sq(&c.coeffs);
        let reference = reference_check_sq(&c.coeffs);
        true_ok += usize::from(truth <= T * T);
        ref_ok += usize::from(reference <= T * T);
        if reference <= T * T && truth > T * T {
            ref_ok_true_bad += 1;
            largest = largest.max(truth.sqrt());
        }
    }
    eprintln!(
        "of {count} draws: operator norm <= 14: {true_ok}; reference check passes: {ref_ok}; \
         passes with operator norm > 14: {ref_ok_true_bad} (largest {largest:.2})"
    );
    assert!(ref_ok_true_bad > 0);
}
