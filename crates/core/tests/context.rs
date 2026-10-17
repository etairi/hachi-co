//! `Context`: built from every fixture, from a seed (`Context::setup`), and rejected when the
//! parameters or the public matrices are inconsistent.

use hachi_co_core::{
    Context, Error, Poly, PublicParams, SETUP_DOMAIN_A, SETUP_DOMAIN_B, SETUP_DOMAIN_D,
};
use hachi_co_params::{Params, derive, presets::toy_choices};
use hachi_co_sampling::matrix;

fn toy(n: u64) -> Params {
    derive(&toy_choices(n)).expect("toy presets derive").0
}

const SEED: [u8; 32] = *b"hachi-co core context test seed!";

#[test]
fn setup_expands_the_documented_streams() {
    for n in 1..=3 {
        let params = toy(n);
        let ctx = Context::<4>::setup(params.clone(), &SEED).unwrap();
        let l = &ctx.layout;
        let expected = |domain: &[u8], rows, cols| -> Vec<Vec<Poly>> {
            matrix::expand(&SEED, domain, rows, cols, l.d, ctx.q())
                .unwrap()
                .into_iter()
                .map(|r| r.into_iter().map(Poly).collect())
                .collect()
        };
        assert_eq!(ctx.pp.a, expected(SETUP_DOMAIN_A, l.n_a, l.width_a));
        assert_eq!(ctx.pp.b, expected(SETUP_DOMAIN_B, l.n_b, l.width_b));
        assert_eq!(ctx.pp.d, expected(SETUP_DOMAIN_D, l.n_d, l.width_d));
        // Deterministic, seed-dependent, and the three domains differ.
        let again = Context::<4>::setup(params.clone(), &SEED).unwrap();
        assert_eq!(again.pp, ctx.pp);
        let mut other_seed = SEED;
        other_seed[31] ^= 1;
        let other = Context::<4>::setup(params, &other_seed).unwrap();
        assert_ne!(other.pp.a, ctx.pp.a);
        assert_ne!(ctx.pp.a[0][0], ctx.pp.b[0][0]);
        assert_ne!(ctx.pp.b[0][0], ctx.pp.d[0][0]);
    }
}

#[test]
fn accessors_and_rings() {
    let params = toy(2);
    let ctx = Context::<4>::setup(params.clone(), &SEED).unwrap();
    assert_eq!(ctx.q(), params.q);
    assert_eq!(ctx.field().modulus(), params.q);
    assert_eq!(ctx.b(), params.choices.b);
    assert_eq!(ctx.omega(), params.omega);
    assert_eq!(u128::from(ctx.gamma()), params.gamma);
    assert_eq!(u128::from(ctx.beta1()), params.beta1);
    assert_eq!(u128::from(ctx.beta()), params.beta);
    assert_eq!(ctx.ring.degree(), 16);
    assert_eq!(ctx.ring2.degree(), 32);
    assert_eq!(ctx.ring2.modulus(), params.q);
    assert_eq!(ctx.sub.ring(), &ctx.ring);
    assert_eq!(ctx.params, params);
}

#[test]
fn inconsistent_inputs_are_rejected() {
    let params = toy(3);
    let ctx = Context::<4>::setup(params.clone(), &SEED).unwrap();
    let pp = ctx.pp.clone();
    // K must equal k.
    assert!(matches!(
        Context::<2>::new(params.clone(), pp.clone()),
        Err(Error::Parameter(msg)) if msg.contains("K = 2")
    ));
    // Shapes of A, B, D.
    let shape = |pp: PublicParams| {
        matches!(
            Context::<4>::new(params.clone(), pp),
            Err(Error::Shape { .. })
        )
    };
    let mut bad = pp.clone();
    bad.a.push(bad.a[0].clone());
    assert!(shape(bad));
    let mut bad = pp.clone();
    bad.b[0].pop();
    assert!(shape(bad));
    let mut bad = pp.clone();
    bad.d[0][3].0.push(0);
    assert!(shape(bad));
    // Residues must be canonical.
    let mut bad = pp.clone();
    bad.d[0][5].0[7] = ctx.q();
    assert!(matches!(
        Context::<4>::new(params.clone(), bad),
        Err(Error::NonCanonical { what: "D", .. })
    ));
    // Parameters the arithmetic cannot use.
    let mut p = params.clone();
    p.choices.b = 1;
    assert!(matches!(
        Context::<4>::new(p, pp.clone()),
        Err(Error::Parameter(_))
    ));
    let mut p = params.clone();
    p.gamma = u128::from(u64::MAX);
    assert!(matches!(
        Context::<4>::new(p, pp.clone()),
        Err(Error::Parameter(msg)) if msg.contains("gamma")
    ));
    let mut p = params.clone();
    p.q = 8388617; // prime, but 1 mod 8: no subfield
    assert!(matches!(
        Context::<4>::new(p, pp.clone()),
        Err(Error::Math(_))
    ));
    let mut p = params;
    p.mu += 1;
    assert!(matches!(Context::<4>::new(p, pp), Err(Error::Layout(_))));
}

#[cfg(feature = "fixtures")]
#[test]
fn context_from_every_fixture() {
    use hachi_co_core::Fixture;
    for name in Fixture::TOY {
        let fx = Fixture::load(name).unwrap();
        let ctx = fx.context::<4>().unwrap();
        assert_eq!(ctx.params, fx.params().unwrap());
        assert_eq!(ctx.pp.a, fx.matrix("/setup/A").unwrap());
        assert_eq!(ctx.pp.b, fx.matrix("/setup/B").unwrap());
        assert_eq!(ctx.pp.d, fx.matrix("/setup/D").unwrap());
        assert_eq!(ctx.layout.n, fx.n());
        assert_eq!(
            (ctx.q(), ctx.layout.d, ctx.layout.k),
            (fx.q(), fx.d(), fx.k())
        );
        // The embedding of the recorded y is the recorded y_ring (subfield of hachi-co-math).
        let y = fx.ext::<4>("/input/y").unwrap();
        assert_eq!(ctx.sub.embed(&y), fx.poly("/input/y_ring").unwrap());
        assert!(matches!(fx.context::<2>(), Err(Error::Parameter(_))));
    }
}

/// Milestone 2 runs the parties on threads: the shared types must be `Send + Sync`.
#[test]
fn shared_types_are_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<Context<4>>();
    check::<hachi_co_core::Layout>();
    check::<hachi_co_core::PrgRandomness>();
    check::<hachi_co_core::FsChallenger>();
    check::<hachi_co_core::PartyState<4>>();
    check::<hachi_co_core::LinearShare>();
    check::<hachi_co_core::Error>();
    check::<hachi_co_mpc::Loopback>();
    #[cfg(feature = "fixtures")]
    {
        check::<hachi_co_core::Fixture>();
        check::<hachi_co_core::ReplayRandomness>();
        check::<hachi_co_core::ReplayChallenger>();
    }
}
