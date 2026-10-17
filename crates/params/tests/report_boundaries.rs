//! Each condition of the report (`docs/m0-spec.md`, `hachi-co-params`) at its failure boundary:
//! the boundary itself passes, one step beyond fails, with the documented severity, and the
//! violation flags only the conditions it should.

use hachi_co_params::presets::toy_choices;
use hachi_co_params::rules::is_prime_u64;
use hachi_co_params::{Choices, ConstraintId, Omega, QChoice, Report, Severity, TauRule, derive};

fn report(c: &Choices) -> Report {
    derive(c).expect("the cascade is defined").1
}

fn violated(c: &Choices) -> Vec<ConstraintId> {
    report(c).violations().map(|v| v.id).collect()
}

fn with_q(c: Choices, q: u64) -> Choices {
    Choices {
        q: QChoice::Fixed(q),
        ..c
    }
}

/// The toy preset n = 1 with a target its challenge set meets (2^3/545 <= 2^-6), so that its
/// report is clean and each test below flips exactly one condition.
fn clean_toy() -> Choices {
    Choices {
        lambda: 6,
        ..toy_choices(1)
    }
}

#[test]
fn severities_are_as_documented() {
    use ConstraintId::*;
    for id in ConstraintId::ALL {
        let expected = match id {
            RejectionConstant | KnowledgeError => Severity::Warning,
            _ => Severity::Fatal,
        };
        assert_eq!(id.severity(), expected, "{id:?}");
    }
    let r = report(&clean_toy());
    assert!(r.all_hold(), "{r}");
    assert_eq!(r.constraints().len(), 13);
    for (entry, id) in r.constraints().iter().zip(ConstraintId::ALL) {
        assert_eq!((entry.id, entry.severity), (id, id.severity()));
    }
}

#[test]
fn q_prime() {
    // 8388637 is the toy modulus; 8388645 = 5 * 1677729 has the same residue mod 8 and delta.
    assert!(violated(&with_q(clean_toy(), 8388637)).is_empty());
    assert_eq!(
        violated(&with_q(clean_toy(), 8388645)),
        [ConstraintId::QPrime]
    );
    assert_eq!(
        report(&with_q(clean_toy(), 8388645))
            .get(ConstraintId::QPrime)
            .severity,
        Severity::Fatal
    );
}

#[test]
fn q_five_mod_eight() {
    // The first prime above the binding guard 2^23 in each other odd residue class (same delta).
    for residue in [1u64, 3, 7] {
        let p = (8388609u64..)
            .find(|&p| p % 8 == residue && is_prime_u64(p))
            .unwrap();
        assert_eq!(
            violated(&with_q(clean_toy(), p)),
            [ConstraintId::QFiveMod8],
            "q = {p}"
        );
    }
}

#[test]
fn q_above_range_guard() {
    // 1 + 2^(l+3) n omega^2 b = 1 + 2^9 * 4 * 4 = 8193 for the toy n = 1 (independent of delta).
    let at = report(&with_q(clean_toy(), 8193));
    let above = report(&with_q(clean_toy(), 8194));
    assert!(!at.holds(ConstraintId::QAboveRangeGuard));
    assert!(above.holds(ConstraintId::QAboveRangeGuard));
    assert_eq!(
        at.get(ConstraintId::QAboveRangeGuard).severity,
        Severity::Fatal
    );
}

#[test]
fn q_above_binding_guard() {
    // 16 omega b^tau = 32 * 4^9 = 2^23 at delta = 12, the delta of both moduli.
    let (p, r) = derive(&with_q(clean_toy(), 1 << 23)).unwrap();
    assert_eq!((p.delta, p.binding_guard), (12, 1 << 23));
    assert!(!r.holds(ConstraintId::QAboveBindingGuard));
    assert_eq!(
        r.get(ConstraintId::QAboveBindingGuard).severity,
        Severity::Fatal
    );
    let (p, r) = derive(&with_q(clean_toy(), (1 << 23) + 1)).unwrap();
    assert_eq!((p.delta, p.binding_guard), (12, 1 << 23));
    assert!(r.holds(ConstraintId::QAboveBindingGuard));
}

#[test]
fn centered_responses() {
    // With b = 2^20, d = 4, n = 1, omega = 1: beta = 2^25 at delta = 2, and 2 beta = 2^26 lies in
    // the band 2^20 < q <= 2^40 of delta = 2.
    let c = Choices {
        lambda: 1,
        n: 1,
        t: 0,
        ell: 2,
        alpha: 2,
        kappa: 1,
        b: 1 << 20,
        omega: Omega::Fixed(1),
        challenge_set: hachi_co_params::ChallengeSet::L1Ball,
        tau_rule: TauRule::Tight,
        n_a: 1,
        n_b: 1,
        n_d: 1,
        m_rho: 2,
        q: QChoice::Smallest,
    };
    let (p, r) = derive(&with_q(c, 1 << 26)).unwrap();
    assert_eq!((p.delta, p.beta), (2, 1 << 25));
    assert!(!r.holds(ConstraintId::CenteredResponses));
    assert_eq!(
        r.get(ConstraintId::CenteredResponses).severity,
        Severity::Fatal
    );
    let (p, r) = derive(&with_q(c, (1 << 26) + 1)).unwrap();
    assert_eq!((p.delta, p.beta), (2, 1 << 25));
    assert!(r.holds(ConstraintId::CenteredResponses));
}

#[test]
fn alpha_minus_kappa_and_k_divides_half_d() {
    // alpha - kappa = 1 (the boundary) holds; kappa = alpha (k = d) fails both, which are
    // equivalent for powers of two.
    let c = Choices {
        kappa: 3,
        ..clean_toy()
    };
    let r = report(&c);
    assert!(r.holds(ConstraintId::AlphaMinusKappa) && r.holds(ConstraintId::KDividesHalfD));
    let c = Choices {
        kappa: 4,
        ..clean_toy()
    };
    let v = violated(&c);
    assert!(v.contains(&ConstraintId::AlphaMinusKappa) && v.contains(&ConstraintId::KDividesHalfD));
    let r = report(&c);
    assert_eq!(
        r.get(ConstraintId::AlphaMinusKappa).severity,
        Severity::Fatal
    );
    assert_eq!(r.get(ConstraintId::KDividesHalfD).severity, Severity::Fatal);
    // The upper end alpha - kappa = l holds; alpha - kappa = l + 1 makes m < 0 (an error).
    let c = Choices {
        ell: 2,
        ..clean_toy()
    };
    assert!(report(&c).holds(ConstraintId::AlphaMinusKappa));
    for n in [1, 2, 3, 8] {
        let c = Choices {
            ell: 1,
            n,
            ..clean_toy()
        };
        assert!(
            matches!(derive(&c), Err(hachi_co_params::Error::NegativeExponent(_))),
            "n = {n}"
        );
    }
}

#[test]
fn b_at_least_three() {
    let three = Choices {
        b: 3,
        ..clean_toy()
    };
    assert!(report(&three).holds(ConstraintId::BAtLeast3));
    // b = 2 is computable with the draft's tau (the tight rule is undefined there).
    let two = Choices {
        b: 2,
        tau_rule: TauRule::Draft,
        ..clean_toy()
    };
    let r = report(&two);
    assert!(!r.holds(ConstraintId::BAtLeast3));
    assert_eq!(r.get(ConstraintId::BAtLeast3).severity, Severity::Fatal);
}

#[test]
fn m_rho_above_n_d() {
    assert!(
        report(&Choices {
            m_rho: 2,
            n_d: 1,
            ..clean_toy()
        })
        .all_hold()
    );
    assert_eq!(
        violated(&Choices {
            m_rho: 1,
            n_d: 1,
            ..clean_toy()
        }),
        [ConstraintId::MRhoAboveND]
    );
    assert_eq!(
        violated(&Choices {
            m_rho: 3,
            n_d: 3,
            ..clean_toy()
        }),
        [ConstraintId::MRhoAboveND]
    );
    assert!(
        report(&Choices {
            m_rho: 4,
            n_d: 3,
            ..clean_toy()
        })
        .all_hold()
    );
}

#[test]
fn q_below_two_pow_63() {
    let below = report(&with_q(clean_toy(), (1 << 63) - 1));
    let at = report(&with_q(clean_toy(), 1 << 63));
    assert!(below.holds(ConstraintId::QBelow2Pow63));
    assert!(!at.holds(ConstraintId::QBelow2Pow63));
    assert_eq!(at.get(ConstraintId::QBelow2Pow63).severity, Severity::Fatal);
    // The largest primes = 5 mod 8 on either side: only the modulus-size condition differs.
    let p_below = (1u64 << 63) - 25; // prime, = 7 mod 8
    assert!(is_prime_u64(p_below));
    let q = ((1u64 << 63) + 1..)
        .find(|&p| p % 8 == 5 && is_prime_u64(p))
        .unwrap();
    assert_eq!(
        violated(&with_q(clean_toy(), q)),
        [ConstraintId::QBelow2Pow63]
    );
}

#[test]
fn rejection_constant_holds_with_warning_severity() {
    // M <= e^(1/n) always holds for the cascade's gamma; its exact boundary is tested against
    // Sage in tests/sage_fixture.rs.
    for n in 1..=9 {
        let r = report(&Choices { n, ..clean_toy() });
        let c = r.get(ConstraintId::RejectionConstant);
        assert!(c.holds && c.severity == Severity::Warning, "n = {n}");
    }
}

#[test]
fn j_inv_representable() {
    // toy n = 1: beta = 49152; the plain ceiling gives tau = 8 with capacity (4^8 - 1)/3 = 21845.
    let ceil = Choices {
        tau_rule: TauRule::Ceil,
        ..clean_toy()
    };
    let (p, r) = derive(&ceil).unwrap();
    assert_eq!((p.tau, p.j_inv_capacity), (8, 21845));
    assert!(!r.holds(ConstraintId::JInvRepresentable));
    assert_eq!(
        r.get(ConstraintId::JInvRepresentable).severity,
        Severity::Fatal
    );
    let (p, r) = derive(&Choices {
        tau_rule: TauRule::Draft,
        ..clean_toy()
    })
    .unwrap();
    assert_eq!((p.tau, p.j_inv_capacity), (9, 87381));
    assert!(r.all_hold());
    let (p, r) = derive(&clean_toy()).unwrap();
    assert_eq!(p.tau, 9);
    assert!(r.holds(ConstraintId::JInvRepresentable));
}

#[test]
fn knowledge_error() {
    // |C| = 545 for d = 16, omega = 2 and r = 2: 2^(3 + 6) = 512 <= 545 < 1024 = 2^(3 + 7).
    let at = report(&Choices {
        lambda: 6,
        ..toy_choices(1)
    });
    let beyond = report(&Choices {
        lambda: 7,
        ..toy_choices(1)
    });
    assert!(at.holds(ConstraintId::KnowledgeError));
    assert!(!beyond.holds(ConstraintId::KnowledgeError));
    assert_eq!(
        beyond.get(ConstraintId::KnowledgeError).severity,
        Severity::Warning
    );
    assert!(beyond.is_admissible() && !beyond.all_hold());
    // MinForTarget picks the smallest omega that holds; omega - 1 fails.
    let c = Choices {
        lambda: 40,
        omega: Omega::MinForTarget,
        ..toy_choices(2)
    };
    let (p, r) = derive(&c).unwrap();
    assert!(r.holds(ConstraintId::KnowledgeError));
    let smaller = Choices {
        omega: Omega::Fixed(p.omega - 1),
        ..c
    };
    assert!(!report(&smaller).holds(ConstraintId::KnowledgeError));
}

#[test]
fn display_lists_every_condition() {
    let text = report(&Choices {
        tau_rule: TauRule::Ceil,
        ..toy_choices(1)
    })
    .to_string();
    assert_eq!(text.lines().count(), 13);
    assert!(
        text.contains("FATAL")
            && text.contains("warning")
            && text.contains("6_Collaborative_Hachi.tex:272")
    );
}
