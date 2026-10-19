//! The toy presets reproduce the table of `docs/m0-spec.md` ("Toy presets") and the rows the Sage
//! oracle adds for $n = 4, 8$ (`ref/hachico_ref/params.py`, `TOY_TABLE`) exactly, and every other
//! derived value is checked against a direct evaluation of the draft's formulas
//! (6_Collaborative_Hachi.tex:274-288) with big integers.

use hachi_co_params::presets::{TOY_NS, toy_choices, toy_presets};
use hachi_co_params::{ConstraintId, PresetKind, Severity, derive, preset};
use num_bigint::BigUint;

/// One row of the table: name, n, q, delta, r, m, beta_1, N, gamma, beta, tau, mu, l'', d_0.
type Row = (
    &'static str,
    u64,
    u64,
    u32,
    u32,
    u32,
    u128,
    u128,
    u128,
    u128,
    u32,
    u128,
    u32,
    u64,
);

/// The table of `docs/m0-spec.md`, computed by the main session (independent of this crate), and
/// the rows `toy-n4`, `toy-n8` of the Sage oracle's `TOY_TABLE` (`ref/hachico_ref/params.py`:
/// $q$ by its `choose_q`, the rest by its `derive`, also independent of this crate).
const TABLE: [Row; 5] = [
    (
        "toy-n1", 1, 8388637, 12, 2, 2, 32, 768, 49152, 49152, 9, 555, 15, 14,
    ),
    (
        "toy-n2", 2, 33554501, 13, 3, 2, 32, 832, 106496, 212992, 10, 785, 15, 14,
    ),
    (
        "toy-n3", 3, 134217757, 14, 3, 3, 16, 1792, 172032, 516096, 11, 1547, 16, 14,
    ),
    (
        "toy-n4", 4, 134217757, 14, 3, 3, 16, 1792, 229376, 917504, 11, 1577, 16, 14,
    ),
    (
        "toy-n8", 8, 536871029, 15, 4, 3, 16, 1920, 491520, 3932160, 12, 2177, 17, 14,
    ),
];

#[test]
fn toy_presets_reproduce_the_brief_table() {
    let presets = toy_presets();
    assert_eq!(presets.len(), TABLE.len());
    assert_eq!(TOY_NS.len(), TABLE.len());
    for (preset, row) in presets.iter().zip(TABLE) {
        let (name, n, q, delta, r, m, beta1, big_n, gamma, beta, tau, mu, ell_dprime, d0) = row;
        assert_eq!(preset.name, name);
        assert_eq!(preset.kind, PresetKind::Toy);
        assert_eq!(preset.choices.n, n);
        let (p, _) = preset.derive().expect("toy presets derive");
        assert_eq!(
            (
                p.q,
                p.delta,
                p.r,
                p.m,
                p.beta1,
                p.big_n,
                p.gamma,
                p.beta,
                p.tau,
                p.mu,
                p.ell_dprime,
                p.d0
            ),
            (
                q, delta, r, m, beta1, big_n, gamma, beta, tau, mu, ell_dprime, d0
            ),
            "{name}"
        );
        // The fixed choices of the brief.
        assert_eq!((p.d, p.k, p.omega, p.choices.b), (16, 4, 2, 4), "{name}");
    }
}

/// The remaining quantities, recomputed from the draft's formulas with `BigUint`.
#[test]
fn toy_presets_match_the_formulas_with_big_integers() {
    for n in TOY_NS {
        let (p, _) = derive(&toy_choices(n)).unwrap();
        let big = |x: u128| BigUint::from(x);
        let (b, omega, d) = (big(4), big(2), big(16));
        let (n_a, n_b, n_d, m_rho) = (big(1), big(1), big(1), big(2));
        let nn = big(u128::from(n));
        let delta = big(u128::from(p.delta));
        let two = big(2);
        let pow2 = |e: u32| BigUint::from(1u8) << e;
        let log_n = (n - 1).checked_ilog2().map_or(0, |l| l + 1); // ceil(log2 n)
        assert_eq!(p.log_n, log_n);
        assert_eq!(p.ell_prime, log_n + 6 - 4 + 2);
        assert_eq!(big(p.beta1), pow2(p.r - log_n) * &omega * &b);
        assert_eq!(big(p.big_n), &d * &delta * pow2(p.m));
        assert_eq!(big(p.gamma), &two * &nn * big(p.big_n) * big(p.beta1));
        assert_eq!(big(p.beta), &nn * big(p.gamma));
        let b_tau = b.pow(p.tau);
        assert_eq!(big(p.beta_bar), &two * &b_tau);
        assert_eq!(p.omega_bar, 4);
        assert_eq!(p.gamma_bar, 4);
        let mu = pow2(p.r) * &delta
            + &nn * &delta
            + pow2(p.r) * &n_a * &delta
            + &nn * &n_a * &delta
            + &nn * &m_rho
            + pow2(p.m) * &delta * BigUint::from(p.tau)
            + 1u8;
        assert_eq!(big(p.mu), mu);
        let n_prime = &n_a + &n_b + &n_d + 2u8;
        assert_eq!(BigUint::from(p.n_prime), n_prime);
        let s = big(p.mu) + &n_prime;
        let ceil_log = (s - 1u8).bits() as u32; // ceil(log2 s) for s >= 2
        assert_eq!(p.ell_dprime, ceil_log + 5);
        assert_eq!(BigUint::from(p.d0), &two * &b + &n_prime + 1u8);
        assert_eq!(p.d1, 3);
        assert_eq!(big(p.nu), &n_prime + &d + (big(p.mu) + &n_prime) * &d);
        assert_eq!(big(p.width_a), &delta * pow2(p.m));
        assert_eq!(big(p.width_b), &n_a * &delta * pow2(p.r));
        assert_eq!(
            big(p.width_d),
            &delta * (pow2(p.r) + &nn + &nn * &n_a) + &nn * &m_rho
        );
        assert_eq!(big(p.binding_guard), big(16) * &omega * &b_tau);
        assert_eq!(big(p.msis_bound_a), big(8) * &omega * &b_tau);
        assert_eq!(p.msis_bound_b, 8);
        assert_eq!(
            big(p.range_guard),
            pow2(9) * &nn * &omega * &omega * &b + 1u8
        );
        // log2 M = N log2((2 gamma + 1)/(2 gamma - 2 beta_1 + 1)), via the exact ratio.
        let ratio =
            (2.0 * p.gamma as f64 + 1.0) / (2.0 * p.gamma as f64 - 2.0 * p.beta1 as f64 + 1.0);
        let expected = p.big_n as f64 * ratio.log2();
        assert!((p.log2_rejection_m - expected).abs() <= 1e-9 * expected);
        assert!((p.rejection_m - expected.exp2()).abs() <= 1e-12 * p.rejection_m);
    }
}

#[test]
fn toy_reports_pass_every_fatal_condition_and_warn_about_the_knowledge_error() {
    for preset in toy_presets() {
        let (_, report) = preset.derive().unwrap();
        assert!(report.is_admissible(), "{}:\n{report}", preset.name);
        let violations: Vec<_> = report.violations().map(|c| c.id).collect();
        assert_eq!(
            violations,
            [ConstraintId::KnowledgeError],
            "{}",
            preset.name
        );
        assert_eq!(
            report.get(ConstraintId::KnowledgeError).severity,
            Severity::Warning
        );
    }
}

#[test]
fn toy_presets_are_found_by_name() {
    for n in TOY_NS {
        let p = preset(&format!("toy-n{n}")).unwrap();
        assert_eq!((p.kind, p.choices), (PresetKind::Toy, toy_choices(n)));
    }
    // No toy preset between and beyond the listed party counts.
    for n in [0, 5, 6, 7, 9, 16] {
        let name = format!("toy-n{n}");
        assert!(matches!(
            preset(&name),
            Err(hachi_co_params::Error::UnknownPreset(unknown)) if unknown == name
        ));
    }
}
