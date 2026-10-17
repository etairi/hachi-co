//! The secure presets of `presets/secure.json` (written by `parameters.sage`): Rust re-derives
//! every preset from its choices and must reproduce every value Sage derived, every condition of
//! the report must hold, and the recorded estimates must be consistent with the parameters and
//! reach the target.

use hachi_co_params::presets::{SecurePresetFile, secure_preset_file, secure_presets};
use hachi_co_params::{ChallengeSet, Omega, Params, PresetKind, QChoice, TauRule, derive, presets};

/// Exact equality of every field, except the two floats (relative 1e-12).
fn assert_same_params(expected: &Params, got: &Params, what: &str) {
    let rel = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs());
    assert!(rel(expected.rejection_m, got.rejection_m), "{what}: M");
    assert!(
        rel(expected.log2_rejection_m, got.log2_rejection_m),
        "{what}: log2 M"
    );
    let mut expected = expected.clone();
    expected.rejection_m = got.rejection_m;
    expected.log2_rejection_m = got.log2_rejection_m;
    assert_eq!(&expected, got, "{what}");
}

fn file() -> SecurePresetFile {
    secure_preset_file().expect("secure.json parses")
}

#[test]
fn the_file_records_its_provenance() {
    let f = file();
    assert_eq!(f.format, 1);
    assert!(
        f.command.starts_with("sage parameters.sage"),
        "{}",
        f.command
    );
    assert!(f.sage_version.starts_with("SageMath"), "{}", f.sage_version);
    assert_eq!(
        f.estimator.repository,
        "https://github.com/malb/lattice-estimator"
    );
    assert_eq!(f.estimator.revision.len(), 40);
    assert!(f.estimator.revision.chars().all(|c| c.is_ascii_hexdigit()));
    assert!(
        !f.presets.is_empty(),
        "no secure presets: run parameters.sage"
    );
}

#[test]
fn rust_reproduces_every_secure_preset_exactly() {
    for e in file().presets {
        let (p, report) =
            derive(&e.params.choices).unwrap_or_else(|err| panic!("{}: {err}", e.name));
        assert_same_params(&e.params, &p, &e.name);
        assert!(report.all_hold(), "{}:\n{report}", e.name);
    }
}

#[test]
fn secure_presets_have_the_documented_choices() {
    for e in file().presets {
        let c = e.params.choices;
        let p = &e.params;
        assert_eq!(
            (
                c.lambda,
                c.kappa,
                c.b,
                c.tau_rule,
                c.challenge_set,
                c.omega,
                c.n_d
            ),
            (
                115,
                2,
                16,
                TauRule::Tight,
                ChallengeSet::L1Ball,
                Omega::MinForTarget,
                1
            ),
            "{}",
            e.name
        );
        assert_eq!(c.q, QChoice::Fixed(p.q), "{}", e.name);
        assert!(p.q < 1 << 63, "{}", e.name);
        assert_eq!(c.t, (c.n - 1) / 2, "{}", e.name);
        assert_eq!(e.name, format!("secure-l{}-n{}-a{}", c.ell, c.n, c.alpha));
    }
}

#[test]
fn estimates_are_consistent_and_reach_the_target() {
    for e in file().presets {
        let (p, s, c) = (&e.params, &e.security, &e.params.choices);
        let name = &e.name;
        assert_eq!(s.lambda, c.lambda, "{name}");
        let target = f64::from(s.lambda);
        // MSIS for A: rank n_A d, width delta 2^m, bound 2 omega_bar beta_bar.
        assert_eq!(
            u128::from(s.msis_a.rank),
            u128::from(c.n_a) * u128::from(p.d),
            "{name}"
        );
        assert_eq!(s.msis_a.columns, p.width_a * u128::from(p.d), "{name}");
        assert_eq!(s.msis_a.bound, p.msis_bound_a, "{name}");
        assert_eq!(
            s.msis_a.bound,
            2 * u128::from(p.omega_bar) * p.beta_bar,
            "{name}"
        );
        // MSIS for B: rank n_B d, width n_A delta 2^r, bound 2 gamma_bar.
        assert_eq!(
            u128::from(s.msis_b.rank),
            u128::from(c.n_b) * u128::from(p.d),
            "{name}"
        );
        assert_eq!(s.msis_b.columns, p.width_b * u128::from(p.d), "{name}");
        assert_eq!(s.msis_b.bound, 2 * u128::from(p.gamma_bar), "{name}");
        for (which, m) in [("A", &s.msis_a), ("B", &s.msis_b)] {
            // The trivial linear-algebra attack: rank log2(q / (2B + 1)), recomputed here.
            let trivial = m.rank as f64 * (p.q as f64 / (2.0 * m.bound as f64 + 1.0)).log2();
            assert!(
                (m.trivial_bits - trivial).abs() <= 1e-6 * trivial,
                "{name} {which}: {} vs {trivial}",
                m.trivial_bits
            );
            assert!(m.trivial_bits >= target, "{name} {which}");
            assert!(
                m.lattice_bits.is_none_or(|bits| bits >= target),
                "{name} {which}: {:?}",
                m.lattice_bits
            );
            assert!(
                u128::from(m.estimator_columns) <= m.columns,
                "{name} {which}"
            );
        }
        // MLWE_{q, d, n_D, m_rho - n_D, b}: S_b has coefficients in [ceil(-b/2), ceil(b/2) - 1].
        assert_eq!(s.mlwe.secret_dim, (c.m_rho - c.n_d) * p.d, "{name}");
        assert_eq!(s.mlwe.samples, c.n_d * p.d, "{name}");
        let b = i64::try_from(c.b).unwrap();
        assert_eq!(s.mlwe.digit_range, (-(b / 2), (b + 1) / 2 - 1), "{name}");
        assert!(
            s.mlwe.bits.is_none_or(|bits| bits >= target),
            "{name}: {:?}",
            s.mlwe.bits
        );
    }
}

#[test]
fn presets_lists_toys_then_dev_then_secure_presets() {
    let all = presets().unwrap();
    let secure = secure_presets().unwrap();
    assert_eq!(all.len(), 3 + 3 + secure.len());
    assert!(
        all[..3]
            .iter()
            .all(|p| p.kind == PresetKind::Toy && p.security.is_none())
    );
    assert!(
        all[3..6]
            .iter()
            .all(|p| p.kind == PresetKind::Dev && p.security.is_none())
    );
    assert!(
        all[6..]
            .iter()
            .all(|p| p.kind == PresetKind::Secure && p.security.is_some())
    );
    for p in &secure {
        assert_eq!(hachi_co_params::preset(&p.name).unwrap(), *p);
    }
    let mut names: Vec<_> = all.iter().map(|p| p.name.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), all.len(), "preset names are unique");
}

#[test]
fn params_round_trip_through_serde() {
    for e in file().presets {
        let text = serde_json::to_string(&e.params).unwrap();
        let back: Params = serde_json::from_str(&text).unwrap();
        assert_eq!(back, e.params, "{}", e.name);
    }
    // u128 values beyond 2^64 survive JSON too.
    let (mut p, _) = derive(&hachi_co_params::presets::toy_choices(1)).unwrap();
    p.nu = u128::MAX;
    let back: Params = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
    assert_eq!(back.nu, u128::MAX);
}
