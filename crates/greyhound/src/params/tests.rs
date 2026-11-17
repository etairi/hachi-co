use super::*;

/// $2^{32} - 99$, the reference's modulus for `LOGQ = 32`.
const Q32: u64 = 4294967197;

/// $2^{56} - 27$, the reference's modulus for `LOGQ = 56`.
const Q56: u64 = 72057594037927909;

/// One parameter set that the reference prints (`print_polcomctx_pp`).
struct Printed {
    q: u64,
    len: usize,
    m: usize,
    n: usize,
    kappa: usize,
    kappa1: usize,
    base: u64,
    base_u: u64,
    f: usize,
    fu: usize,
    norm: f64,
}

/// The reference's prints for `len` = $2^{15}$, $2^{17}$ and $2^{19}$ at `LOGQ = 32` (copies of
/// `test_greyhound.c`) and `LOGQ = 56` (its front end alone, built with a stand-in for
/// `jlproj.c`), measured on hco-dev1 on 17 November 2026.
const REFERENCE: [Printed; 6] = [
    Printed {
        q: Q32,
        len: 1 << 15,
        m: 572,
        n: 58,
        kappa: 19,
        kappa1: 7,
        base: 1 << 11,
        base_u: 1 << 8,
        f: 3,
        fu: 4,
        norm: 66300.87,
    },
    Printed {
        q: Q32,
        len: 1 << 17,
        m: 1086,
        n: 121,
        kappa: 17,
        kappa1: 6,
        base: 1 << 8,
        base_u: 1 << 7,
        f: 4,
        fu: 5,
        norm: 43144.83,
    },
    Printed {
        q: Q32,
        len: 1 << 19,
        m: 2232,
        n: 235,
        kappa: 18,
        kappa1: 7,
        base: 1 << 8,
        base_u: 1 << 7,
        f: 4,
        fu: 5,
        norm: 72190.34,
    },
    Printed {
        q: Q56,
        len: 1 << 15,
        m: 724,
        n: 46,
        kappa: 31,
        kappa1: 11,
        base: 1 << 28,
        base_u: 1 << 17,
        f: 2,
        fu: 3,
        norm: 372301181.24,
    },
    Printed {
        q: Q56,
        len: 1 << 17,
        m: 1471,
        n: 90,
        kappa: 32,
        kappa1: 11,
        base: 1 << 28,
        base_u: 1 << 17,
        f: 2,
        fu: 3,
        norm: 529011457.92,
    },
    Printed {
        q: Q56,
        len: 1 << 19,
        m: 2941,
        n: 179,
        kappa: 32,
        kappa1: 11,
        base: 1 << 28,
        base_u: 1 << 17,
        f: 2,
        fu: 3,
        norm: 746560781.95,
    },
];

/// The proof sizes the reference prints at `LOGQ = 32` ("Proof size", $u_1$ and $u_2$), in KB.
const REFERENCE_KB: [f64; 3] = [3.50, 3.00, 3.50];

#[test]
fn reference_rule_reproduces_the_reference_parameters() {
    for (i, r) in REFERENCE.iter().enumerate() {
        let (q, len) = (r.q, r.len);
        let p = Params::derive(len, q, Security::REFERENCE).unwrap();
        assert_eq!((p.m, p.n), (r.m, r.n), "q {q} len {len}");
        assert_eq!((p.kappa, p.kappa1), (r.kappa, r.kappa1), "q {q} len {len}");
        assert_eq!(
            (1u64 << p.b, 1u64 << p.bu),
            (r.base, r.base_u),
            "q {q} len {len}"
        );
        assert_eq!((p.f, p.fu), (r.f, r.fu), "q {q} len {len}");
        let predicted = (p.predicted_norm_sq() as f64).sqrt();
        assert!(
            (predicted - r.norm).abs() < 0.005,
            "q {q} len {len}: {predicted}"
        );
        if let Some(kb) = REFERENCE_KB.get(i) {
            assert_eq!(p.reference_proof_bytes() as f64 / 1024.0, *kb, "len {len}");
        }
    }
}

#[test]
fn labrador_slack_needs_at_least_the_reference_ranks() {
    for Printed { len, .. } in REFERENCE.iter().take(3) {
        let len = *len;
        let reference = Params::derive(len, Q32, Security::REFERENCE).unwrap();
        let labrador = Params::derive(len, Q32, Security::labrador()).unwrap();
        // A larger slack only raises the bounds, so the first secure rank cannot be smaller for
        // the same expansion factor.
        if labrador.f == reference.f {
            assert!(labrador.kappa >= reference.kappa, "len {len}");
            assert!(labrador.kappa1 >= reference.kappa1, "len {len}");
        }
    }
}

#[test]
fn sis_rule_matches_the_closed_form() {
    // maxlog = 2 sqrt(32 log2(1.00444) 64 k): rank 18 is hard up to 2^30.6993.
    let maxlog = 2.0 * (32.0 * 1.00444f64.log2() * 64.0 * 18.0).sqrt();
    assert!((maxlog - 30.6993).abs() < 0.0001);
    let q = 1u64 << 32;
    assert!(sis_secure(q, 18, 2f64.powf(maxlog - 1e-6), REFERENCE_DELTA));
    assert!(!sis_secure(
        q,
        18,
        2f64.powf(maxlog + 1e-6),
        REFERENCE_DELTA
    ));
    // The bound never exceeds q.
    assert!(!sis_secure(Q32, 64, Q32 as f64, REFERENCE_DELTA));
}

#[test]
fn the_reference_delta_is_about_block_size_342() {
    // delta_0(b) decreases in b; 1.00444 lies between block sizes 341 and 343.
    let d = Security::delta_for_block_size;
    assert!(d(341) > REFERENCE_DELTA && d(343) < REFERENCE_DELTA);
    // jali's default policy, block size 439.
    assert!((d(439) - 1.003732).abs() < 1e-6);
}

#[test]
fn parameters_for_other_moduli_and_lengths() {
    // C's LOGQ = 56 modulus and a 57-bit Hachi-Co modulus, both 5 mod 8.
    for q in [72057594037927909u64, 93266870910952069] {
        for len in [1usize, 2, 7, 64, 1000, 1 << 12, 1 << 19] {
            for security in [Security::REFERENCE, Security::labrador()] {
                let p = Params::derive(len, q, security).unwrap();
                assert!(p.m * p.n >= len);
                assert!(p.kappa <= MAX_RANK && p.kappa1 <= MAX_RANK);
                p.check_norm(u128::from(p.predicted_norm_sq()), security)
                    .unwrap();
            }
        }
    }
}

#[test]
fn a_fixed_expansion_factor_runs_one_iteration_of_the_rule() {
    // The reference takes f = 4 for 2^19 ring elements, where f = 2 and f = 3 have no ranks.
    let chosen = Params::derive(1 << 19, Q32, Security::REFERENCE).unwrap();
    assert_eq!(
        Params::with_expansion(1 << 19, Q32, Security::REFERENCE, 4).unwrap(),
        chosen
    );
    assert!(Params::with_expansion(1 << 19, Q32, Security::REFERENCE, 2).is_err());
    // At q = 2^56 - 27 the rule's first f is 2, at the largest rank; f = 4 needs far smaller
    // ranks and norms.
    let q56 = 72057594037927909u64;
    let first = Params::derive(1 << 15, q56, Security::labrador()).unwrap();
    let four = Params::with_expansion(1 << 15, q56, Security::labrador(), 4).unwrap();
    assert_eq!((first.f, first.kappa), (2, 32));
    assert!(four.kappa < first.kappa && four.predicted_norm_sq() < first.predicted_norm_sq());
}

#[test]
fn encoding_round_trips_and_rejects_garbage() {
    let p = Params::derive(1 << 10, Q32, Security::labrador()).unwrap();
    assert_eq!(Params::from_bytes(&p.to_bytes()).unwrap(), p);
    assert!(Params::from_bytes(&p.to_bytes()[..79]).is_err());
    let mut bad = p;
    bad.m = 1;
    bad.n = 1;
    assert!(Params::from_bytes(&bad.to_bytes()).is_err());
    assert!(Params::derive(1 << 10, Q32 + 2, Security::labrador()).is_err());
    assert!(Params::derive(0, Q32, Security::labrador()).is_err());
}
