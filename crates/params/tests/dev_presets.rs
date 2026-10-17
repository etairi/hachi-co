//! The development presets `dev-l12`, `dev-l16`, `dev-l20` (`docs/m1-spec.md`, WP11): the toy
//! choices with n = 1 and another l, kind `Dev`, found by name; every derived value checked
//! against a direct evaluation of the draft's formulas (6_Collaborative_Hachi.tex:274-288) with
//! big integers; q checked to be the smallest admissible prime by an independent search (trial
//! division); the report flags only the knowledge-error target, as for the toys.

use hachi_co_params::presets::{DEV_ELLS, dev_choices, dev_presets, toy_choices};
use hachi_co_params::{ConstraintId, Omega, PresetKind, QChoice, TauRule, derive, preset};
use num_bigint::BigUint;

#[test]
fn dev_presets_are_the_toy_choices_at_larger_l() {
    let presets = dev_presets();
    assert_eq!(DEV_ELLS, [12, 16, 20]);
    assert_eq!(
        presets.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
        ["dev-l12", "dev-l16", "dev-l20"]
    );
    for (p, ell) in presets.iter().zip(DEV_ELLS) {
        assert_eq!(p.kind, PresetKind::Dev);
        assert!(p.security.is_none());
        let mut expected = toy_choices(1);
        expected.ell = ell;
        assert_eq!(p.choices, expected, "{}", p.name);
        assert_eq!(p.choices, dev_choices(ell));
        // The choices of the brief, spelled out.
        let c = p.choices;
        assert_eq!(
            (
                c.n, c.alpha, c.kappa, c.b, c.omega, c.n_a, c.n_b, c.n_d, c.m_rho
            ),
            (1, 4, 2, 4, Omega::Fixed(2), 1, 1, 1, 2)
        );
        assert_eq!((c.tau_rule, c.q), (TauRule::Tight, QChoice::Smallest));
        assert_eq!(preset(&p.name).unwrap(), *p);
    }
}

#[test]
fn dev_reports_flag_only_the_knowledge_error() {
    for p in dev_presets() {
        let (_, report) = p.derive().unwrap();
        assert!(report.is_admissible(), "{}:\n{report}", p.name);
        let violations: Vec<_> = report.violations().map(|c| c.id).collect();
        assert_eq!(violations, [ConstraintId::KnowledgeError], "{}", p.name);
    }
}

/// The least tau with (ceil(b/2) - 1)(b^tau - 1)/(b - 1) >= beta (the tight rule,
/// `docs/m0-spec.md`), by direct search.
fn tight_tau(b: u128, beta: u128) -> u32 {
    let top = b.div_ceil(2) - 1;
    (0u32..)
        .find(|&t| top * (b.pow(t) - 1) / (b - 1) >= beta)
        .unwrap()
}

/// The lower bound on q for a given delta (6_Collaborative_Hachi.tex:272, 275-277;
/// `docs/m0-spec.md`, "Choosing q"), for n = 1, omega = 2, b = 4, d = 16 and the l, r, m of the
/// preset: max(16 omega b^tau, 1 + 2^(l+3) n omega^2 b, 2 beta + 1).
fn q_bound(ell: u32, r: u32, m: u32, delta: u32) -> u128 {
    let (n, omega, b, d) = (1u128, 2u128, 4u128, 16u128);
    let beta1 = (1u128 << r) * omega * b; // log n = 0
    let big_n = d * u128::from(delta) * (1u128 << m);
    let gamma = 2 * n * big_n * beta1;
    let beta = n * gamma;
    let tau = tight_tau(b, beta);
    let binding = 16 * omega * b.pow(tau);
    let range = 1 + (1u128 << (ell + 3)) * n * omega * omega * b;
    binding.max(range).max(2 * beta + 1)
}

fn is_prime(p: u64) -> bool {
    if p < 2 {
        return false;
    }
    if p.is_multiple_of(2) {
        return p == 2;
    }
    let mut f = 3u64;
    while f * f <= p {
        if p.is_multiple_of(f) {
            return false;
        }
        f += 2;
    }
    true
}

/// q is prime, 5 mod 8, above the bound of its delta-band, the smallest such prime of that band,
/// and every lower band has no admissible modulus because its bound already exceeds the band.
#[test]
fn dev_moduli_are_the_smallest_admissible_primes() {
    for p in dev_presets() {
        let (x, _) = p.derive().unwrap();
        let (ell, r, m) = (p.choices.ell, x.r, x.m);
        let q = u128::from(x.q);
        let delta = x.delta;
        assert!(
            4u128.pow(delta - 1) < q && q <= 4u128.pow(delta),
            "{}",
            p.name
        );
        assert!(is_prime(x.q) && x.q % 8 == 5, "{}", p.name);
        let low = q_bound(ell, r, m, delta).max(4u128.pow(delta - 1));
        assert!(q > low, "{}", p.name);
        for candidate in (low + 1..q).filter(|c| c % 8 == 5) {
            assert!(
                !is_prime(candidate as u64),
                "{}: {candidate} is a smaller admissible prime",
                p.name
            );
        }
        for lower in 1..delta {
            assert!(
                q_bound(ell, r, m, lower) >= 4u128.pow(lower),
                "{}: band {lower} might hold a modulus",
                p.name
            );
        }
    }
}

/// The derived quantities, recomputed from the draft's formulas with `BigUint` (as
/// `tests/toy_table.rs` does for the toys).
#[test]
fn dev_presets_match_the_formulas_with_big_integers() {
    for p in dev_presets() {
        let ell = p.choices.ell;
        let (x, _) = derive(&p.choices).unwrap();
        let big = |v: u128| BigUint::from(v);
        let (b, omega, d) = (big(4), big(2), big(16));
        let (n_a, n_b, n_d, m_rho, n) = (big(1), big(1), big(1), big(2), big(1));
        let delta = big(u128::from(x.delta));
        let pow2 = |e: u32| BigUint::from(1u8) << e;
        assert_eq!((x.d, x.k, x.log_n), (16, 4, 0), "{}", p.name);
        assert_eq!(x.ell_prime, ell - 4 + 2, "{}", p.name);
        assert_eq!(x.r, x.ell_prime.div_ceil(2), "{}", p.name);
        assert_eq!(x.m, x.ell_prime - x.r, "{}", p.name);
        assert_eq!(big(x.beta1), pow2(x.r) * &omega * &b);
        assert_eq!(big(x.big_n), &d * &delta * pow2(x.m));
        assert_eq!(big(x.gamma), big(2) * &n * big(x.big_n) * big(x.beta1));
        assert_eq!(big(x.beta), &n * big(x.gamma));
        assert_eq!(x.tau, tight_tau(4, x.beta), "{}", p.name);
        let b_tau = b.pow(x.tau);
        assert_eq!(big(x.beta_bar), big(2) * &b_tau);
        let mu = pow2(x.r) * &delta
            + &n * &delta
            + pow2(x.r) * &n_a * &delta
            + &n * &n_a * &delta
            + &n * &m_rho
            + pow2(x.m) * &delta * BigUint::from(x.tau)
            + 1u8;
        assert_eq!(big(x.mu), mu, "{}", p.name);
        let n_prime = &n_a + &n_b + &n_d + 2u8;
        assert_eq!(BigUint::from(x.n_prime), n_prime);
        let s = big(x.mu) + &n_prime;
        assert_eq!(x.ell_dprime, (s - 1u8).bits() as u32 + 5, "{}", p.name);
        assert_eq!(BigUint::from(x.d0), big(2) * &b + &n_prime + 1u8);
        assert_eq!(x.d1, 3);
        assert_eq!(big(x.nu), &n_prime + &d + (big(x.mu) + &n_prime) * &d);
        assert_eq!(big(x.width_a), &delta * pow2(x.m));
        assert_eq!(big(x.width_b), &n_a * &delta * pow2(x.r));
        assert_eq!(
            big(x.width_d),
            &delta * (pow2(x.r) + &n + &n * &n_a) + &n * &m_rho
        );
    }
    // The hypercube sizes of the rustdoc of `dev_choices`.
    let ell_pp: Vec<u32> = dev_presets()
        .iter()
        .map(|p| p.derive().unwrap().0.ell_dprime)
        .collect();
    assert_eq!(ell_pp, [18, 21, 23]);
}
