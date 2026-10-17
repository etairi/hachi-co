//! `derive` returns errors (values, not panics) exactly when the cascade cannot be evaluated.

use hachi_co_params::presets::toy_choices;
use hachi_co_params::{ChallengeSet, Choices, Error, Omega, QChoice, TauRule, derive};

#[test]
fn invalid_choices() {
    let toy = toy_choices(2);
    for bad in [
        Choices { n: 0, ..toy },
        Choices { b: 0, ..toy },
        Choices { b: 1, ..toy },
        Choices { n_a: 0, ..toy },
        Choices { n_b: 0, ..toy },
        Choices { n_d: 0, ..toy },
        Choices {
            omega: Omega::Fixed(0),
            ..toy
        },
        Choices {
            q: QChoice::Fixed(0),
            ..toy
        },
        Choices {
            q: QChoice::Fixed(1),
            ..toy
        },
    ] {
        assert!(
            matches!(derive(&bad), Err(Error::InvalidChoice(_))),
            "{bad:?}"
        );
    }
    // The smallest values that are accepted.
    assert!(
        derive(&Choices {
            q: QChoice::Fixed(2),
            ..toy
        })
        .is_ok()
    );
    assert!(
        derive(&Choices {
            b: 2,
            tau_rule: TauRule::Ceil,
            ..toy
        })
        .is_ok()
    );
}

#[test]
fn tight_tau_needs_b_at_least_three() {
    let c = Choices {
        b: 2,
        ..toy_choices(1)
    };
    assert!(matches!(derive(&c), Err(Error::TightTauUndefined { b: 2 })));
}

#[test]
fn negative_exponents() {
    let c = Choices {
        alpha: 9,
        kappa: 2,
        ell: 6,
        ..toy_choices(1)
    };
    assert!(matches!(derive(&c), Err(Error::NegativeExponent(_))));
}

#[test]
fn overflow_and_missing_modulus() {
    // 2^(l+3) n omega^2 b and the mask width exceed u128 for l = 200.
    let huge = Choices {
        ell: 200,
        ..toy_choices(2)
    };
    assert!(matches!(
        derive(&Choices {
            q: QChoice::Fixed(1 << 40),
            ..huge
        }),
        Err(Error::Overflow(_))
    ));
    // With the smallest modulus the range guard 1 + 2^(l+3) n omega^2 b is computed first.
    assert!(matches!(derive(&huge), Err(Error::Overflow(_))));
    // l = 60: the range guard 1 + 2^63 * 2 * 4 * 4 is beyond 2^64.
    assert!(matches!(
        derive(&Choices {
            ell: 60,
            ..toy_choices(2)
        }),
        Err(Error::NoModulus(_))
    ));
    // d = 2^64 does not fit.
    assert!(matches!(
        derive(&Choices {
            alpha: 64,
            ..toy_choices(1)
        }),
        Err(Error::Overflow(_))
    ));
}

#[test]
fn no_omega_for_the_sparse_set() {
    // The sparse set of degree 16 has fewer than 2^24 elements.
    let c = Choices {
        omega: Omega::MinForTarget,
        challenge_set: ChallengeSet::Sparse,
        lambda: 30,
        ..toy_choices(1)
    };
    assert!(matches!(
        derive(&c),
        Err(Error::NoOmega {
            cap: 16,
            lambda: 30
        })
    ));
}

#[test]
fn errors_display() {
    let e = derive(&Choices {
        n: 0,
        ..toy_choices(1)
    })
    .unwrap_err();
    assert!(e.to_string().contains("number of provers"));
}
