//! $\Pi\_{\mathsf{conv}}$ against the Sage oracle's fixtures `ref/fixtures/toy-n{1,2,3}.json`
//! (section `conv`, the relation checks `conv.*`, V's check in `verifier/checks` and the tampered
//! $y$ and $E(u)$ of `negative_tests/transcript`), exact equality residue by residue.
//!
//! Every party's local values are computed one party at a time from the states that the oracle's
//! Commit recorded, with the $\mathcal F\_{\mathsf{AH}}(E)$ inputs summed here; the driver is then
//! run on the loopback (toy-n1) and on an in-memory network ($n = 1, 2, 3$).
mod common;

use common::{K, load, recorded_state, run_parties, state_from_share};
use hachi_co_conv::{
    VerifyError, check, check_relation, e_map, e_matrix, evaluate, f_prime_of_states, fah_e_input,
    pi_conv, public_vectors, u_share, verify,
};
use hachi_co_core::{Fixture, PartyState, escape_key};
use hachi_co_math::Poly;
use hachi_co_mpc::Loopback;

fn states(fx: &Fixture, ctx: &hachi_co_core::Context<K>) -> Vec<PartyState<K>> {
    (1..=ctx.layout.n)
        .map(|p| recorded_state(fx, ctx, p))
        .collect()
}

fn relation(fx: &Fixture, key: &str) -> bool {
    fx.bool(&format!("/relations/{}", escape_key(key))).unwrap()
}

#[test]
fn public_vectors_match_the_fixtures() {
    for name in Fixture::TOY {
        let (fx, ctx) = load(name);
        let x = fx.exts::<K>("/input/x").unwrap();
        let pv = public_vectors(&ctx, &x).unwrap();
        pv.validate(&ctx.layout).unwrap();
        assert_eq!(
            pv.x_prime,
            fx.exts::<K>("/conv/x_prime").unwrap(),
            "{name}: x'"
        );
        assert_eq!(pv.b, fx.exts::<K>("/conv/b").unwrap(), "{name}: b");
        assert_eq!(pv.a, fx.exts::<K>("/conv/a").unwrap(), "{name}: a");
        assert_eq!(pv.d, fx.exts::<K>("/conv/d").unwrap(), "{name}: d");
        assert_eq!(pv.v_point, fx.poly("/conv/v_point").unwrap(), "{name}: v");
        // The ring forms are the embeddings (Hachi Eq. (7)) and lie in R_q^H.
        for (e, r) in
            pv.b.iter()
                .zip(&pv.b_ring)
                .chain(pv.a.iter().zip(&pv.a_ring))
        {
            assert_eq!(ctx.sub.extract(r).unwrap(), *e, "{name}");
        }
        // d = b (x) a also holds in R_q: d_t embedded is b_i a_j as ring elements.
        let m = ctx.layout.block_len;
        for (t, dt) in pv.d.iter().enumerate() {
            let prod = ctx.ring.mul(&pv.b_ring[t / m], &pv.a_ring[t % m]).unwrap();
            assert_eq!(ctx.sub.embed(dt), prod, "{name}: d_{t}");
        }
    }
}

#[test]
fn party_shares_and_the_fah_e_inputs_match_the_fixtures() {
    for name in Fixture::TOY {
        let (fx, ctx) = load(name);
        let ring = &ctx.ring;
        let x = fx.exts::<K>("/input/x").unwrap();
        let pv = public_vectors(&ctx, &x).unwrap();
        let mut u = ring.zero();
        let mut e_u = ring.zero();
        for (p, st) in (1..=ctx.layout.n).zip(states(&fx, &ctx)) {
            let share = u_share(&ctx, &st, &pv).unwrap();
            let want = fx.poly(&format!("/conv/u_shares/{}", p - 1)).unwrap();
            assert_eq!(share, want, "{name} p = {p}: <u>_p");
            let input = fah_e_input(&ctx, &share, &pv.v_point).unwrap();
            u = ring.add(&u, &share).unwrap();
            e_u = ring.add(&e_u, &input).unwrap();
        }
        assert_eq!(u, fx.poly("/conv/u_inside_F_AH").unwrap(), "{name}: u");
        let recorded = fx.poly("/conv/E_u").unwrap();
        assert_eq!(e_u, recorded, "{name}: sum of the F_AH(E) inputs");
        assert_eq!(
            e_map(&ctx, &u, &pv.v_point).unwrap(),
            recorded,
            "{name}: E(u)"
        );
        assert_eq!(
            ctx.sub.is_in_subfield(&recorded),
            relation(&fx, "conv.E(u)_in_R_q^H")
        );
        assert!(ctx.sub.is_in_subfield(&recorded));
    }
}

#[test]
fn the_e_matrix_matches_and_acts_on_coefficients() {
    for name in Fixture::TOY {
        let (fx, ctx) = load(name);
        let zq = ctx.ring.zq();
        let v = fx.poly("/conv/v_point").unwrap();
        let e = e_matrix(&ctx, &v).unwrap();
        assert_eq!(e, fx.residue_rows("/conv/E_matrix").unwrap(), "{name}");
        // E cf(u) = cf(E(u)) for the recorded u (relation conv.E_matrix_times_u_eq_E(u)).
        let u = fx.poly("/conv/u_inside_F_AH").unwrap();
        let e_u: Vec<u64> = e
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&u.0)
                    .fold(0, |acc, (a, b)| zq.add(acc, zq.mul(*a, *b)))
            })
            .collect();
        assert_eq!(Poly(e_u.clone()), fx.poly("/conv/E_u").unwrap(), "{name}");
        assert_eq!(
            Poly(e_u) == e_map(&ctx, &u, &v).unwrap(),
            relation(&fx, "conv.E_matrix_times_u_eq_E(u)")
        );
    }
}

#[test]
fn the_verifier_accepts_and_the_relation_holds() {
    for name in Fixture::TOY {
        let (fx, ctx) = load(name);
        let y = fx.ext::<K>("/input/y").unwrap();
        let e_u = fx.poly("/conv/E_u").unwrap();
        assert_eq!(check(&ctx, &y, &e_u), Ok(()), "{name}");
        assert!(verify(&ctx, &y, &e_u));
        assert!(fx.bool("/conv/verifier_y_eq_E_u").unwrap());
        assert!(relation(&fx, "conv.verifier_y_eq_E(u)"));
        assert!(fx.bool("/verifier/checks/conv_y_eq_E(u)").unwrap());
        // y as a ring element is the embedding of y.
        assert_eq!(ctx.sub.embed(&y), fx.poly("/input/y_ring").unwrap());
        // The reduction on the reconstructed witnesses.
        let x = fx.exts::<K>("/input/x").unwrap();
        let sts = states(&fx, &ctx);
        let u = fx.poly("/conv/u_inside_F_AH").unwrap();
        let rel = check_relation(&ctx, &x, &y, &sts, &u).unwrap();
        assert!(rel.holds(), "{name}: {rel:?}");
        // f(x) by folding equals the oracle's y (computed with subfield.py), and f' from the
        // states equals the recorded f'.
        let f = fx.exts::<K>("/input/f").unwrap();
        assert_eq!(evaluate(&ctx, &f, &x).unwrap(), y, "{name}");
        assert_eq!(
            f_prime_of_states(&ctx, &sts).unwrap(),
            fx.polys("/commit/f_prime").unwrap()
        );
    }
}

/// The oracle's tampered public values (`negative_tests/transcript`, "tamper claimed y += 1:
/// Pi_conv check" and "tamper E(u) message: Pi_conv check"), and further tamperings.
#[test]
fn tampered_y_and_e_u_fail_the_check() {
    for name in Fixture::TOY {
        let (fx, ctx) = load(name);
        let (q, field) = (ctx.q(), ctx.field());
        let y = fx.ext::<K>("/input/y").unwrap();
        let e_u = fx.poly("/conv/E_u").unwrap();
        for key in [
            "tamper claimed y += 1: Pi_conv check",
            "tamper E(u) message: Pi_conv check",
        ] {
            let pointer = format!("/negative_tests/transcript/{}/passed", escape_key(key));
            assert!(fx.bool(&pointer).unwrap(), "{name}: {key}");
        }
        // y + 1 (the oracle's tampering).
        let y_bad = field.add(&y, &field.one());
        assert_eq!(check(&ctx, &y_bad, &e_u), Err(VerifyError::Mismatch));
        assert!(!verify(&ctx, &y_bad, &e_u));
        // E(u) with coefficient 0 plus 1 (the oracle's tampering): still in R_q^H.
        let mut e_bad = e_u.clone();
        e_bad.0[0] = (e_bad.0[0] + 1) % q;
        assert_eq!(check(&ctx, &y, &e_bad), Err(VerifyError::Mismatch));
        // Every other coordinate of y.
        for c in 1..K {
            let mut y_bad = y;
            y_bad.0[c] = (y_bad.0[c] + 1) % q;
            assert_eq!(check(&ctx, &y_bad, &e_u), Err(VerifyError::Mismatch));
        }
        // A change at a position outside the support of R_q^H leaves the subfield.
        let mut e_bad = e_u.clone();
        e_bad.0[1] = (e_bad.0[1] + 1) % q;
        assert_eq!(check(&ctx, &y, &e_bad), Err(VerifyError::NotInSubfield));
        // A change at position m d/(2k) without its mirror at (2k - m) d/(2k).
        let step = ctx.layout.d / (2 * K);
        let mut e_bad = e_u.clone();
        e_bad.0[step] = (e_bad.0[step] + 1) % q;
        assert_eq!(check(&ctx, &y, &e_bad), Err(VerifyError::NotInSubfield));
        // The same element of F_{q^k} in both places: the mirror changed consistently.
        let mut e_moved = e_u.clone();
        e_moved.0[step] = (e_moved.0[step] + 1) % q;
        e_moved.0[(2 * K - 1) * step] = (e_moved.0[(2 * K - 1) * step] + q - 1) % q;
        let mut y_moved = y;
        y_moved.0[1] = (y_moved.0[1] + 1) % q;
        assert_eq!(check(&ctx, &y_moved, &e_moved), Ok(()));
        assert_eq!(check(&ctx, &y, &e_moved), Err(VerifyError::Mismatch));
    }
}

#[test]
fn malformed_messages_are_rejected_without_panic() {
    let (fx, ctx) = load("toy-n2");
    let (q, d) = (ctx.q(), ctx.layout.d);
    let y = fx.ext::<K>("/input/y").unwrap();
    let e_u = fx.poly("/conv/E_u").unwrap();
    for len in [0usize, 1, d - 1, d + 1, 2 * d] {
        let e = Poly(vec![0; len]);
        assert_eq!(
            check(&ctx, &y, &e),
            Err(VerifyError::Shape {
                expected: d,
                actual: len
            })
        );
        assert!(!verify(&ctx, &y, &e));
    }
    // Non-canonical residues: the same residue class, rejected.
    let mut e_big = e_u.clone();
    e_big.0[0] += q;
    assert!(matches!(
        check(&ctx, &y, &e_big),
        Err(VerifyError::NonCanonical { what: "E(u)", .. })
    ));
    let mut y_big = y;
    y_big.0[2] += q;
    assert!(matches!(
        check(&ctx, &y_big, &e_u),
        Err(VerifyError::NonCanonical { what: "y", .. })
    ));
    let e_max = Poly(vec![u64::MAX; d]);
    assert!(matches!(
        check(&ctx, &y, &e_max),
        Err(VerifyError::NonCanonical { .. })
    ));
    // public_vectors (verifier side) refuses a point of the wrong length.
    let x = fx.exts::<K>("/input/x").unwrap();
    assert!(public_vectors(&ctx, &x[1..]).is_err());
    let mut x_long = x.clone();
    x_long.push(x[0]);
    assert!(public_vectors(&ctx, &x_long).is_err());
    // Non-canonical coordinates of x are read modulo q.
    let mut x_big = x.clone();
    x_big[0].0[0] += q;
    assert_eq!(
        public_vectors(&ctx, &x_big).unwrap(),
        public_vectors(&ctx, &x).unwrap()
    );
}

#[test]
fn the_driver_with_loopback_reproduces_toy_n1() {
    let (fx, ctx) = load("toy-n1");
    let st = recorded_state(&fx, &ctx, 1);
    let x = fx.exts::<K>("/input/x").unwrap();
    let mut comm = Loopback::new(ctx.q()).unwrap();
    let out = pi_conv(&ctx, &mut comm, &st, &x).unwrap();
    assert_eq!(out.public, public_vectors(&ctx, &x).unwrap());
    assert_eq!(out.u_share, fx.poly("/conv/u_shares/0").unwrap());
    assert_eq!(out.e_u, fx.poly("/conv/E_u").unwrap());
    assert!(verify(&ctx, &fx.ext::<K>("/input/y").unwrap(), &out.e_u));
}

#[test]
fn the_drivers_on_an_n_party_network_reproduce_the_fixtures() {
    for name in Fixture::TOY {
        let (fx, ctx) = load(name);
        let n = ctx.layout.n;
        let x = fx.exts::<K>("/input/x").unwrap();
        let sts = states(&fx, &ctx);
        let outputs = run_parties(n, ctx.q(), |p, comm| {
            pi_conv(&ctx, comm, &sts[p - 1], &x).unwrap()
        });
        let recorded = fx.poly("/conv/E_u").unwrap();
        for (p, out) in (1..=n).zip(&outputs) {
            let want = fx.poly(&format!("/conv/u_shares/{}", p - 1)).unwrap();
            assert_eq!(out.u_share, want, "{name} p = {p}");
            assert_eq!(out.e_u, recorded, "{name} p = {p}: E(u)");
        }
        assert!(verify(&ctx, &fx.ext::<K>("/input/y").unwrap(), &recorded));
    }
}

/// The test-only conversion `state_from_share` (used by the fresh runs) reproduces the states
/// that the oracle's Commit recorded.
#[test]
fn the_test_conversion_matches_the_recorded_states() {
    for name in Fixture::TOY {
        let (fx, ctx) = load(name);
        for p in 1..=ctx.layout.n {
            let share = fx
                .exts::<K>(&format!("/commit/f_shares/{}", p - 1))
                .unwrap();
            assert_eq!(
                state_from_share(&ctx, p, share),
                recorded_state(&fx, &ctx, p),
                "{name} p = {p}"
            );
        }
    }
}

/// V's check on 4000 arbitrary messages (seed 80): random lengths around $d$, residues below and
/// above $q$, elements of $\mathcal R\_q^H$ and arbitrary ring elements, against the documented
/// order of the reasons. Never panics.
#[test]
fn the_check_classifies_arbitrary_messages() {
    use rand::{RngExt, SeedableRng};
    let (fx, ctx) = load("toy-n3");
    let (q, d) = (ctx.q(), ctx.layout.d);
    let y = fx.ext::<K>("/input/y").unwrap();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(80);
    let mut seen = [0usize; 5];
    for _ in 0..4000 {
        let kind = rng.random_range(0..5u32);
        let len = if kind == 0 {
            rng.random_range(0..2 * d + 2)
        } else {
            d
        };
        let mut e: Vec<u64> = (0..len).map(|_| rng.random_range(0..q)).collect();
        match kind {
            1 if len > 0 => {
                let i = rng.random_range(0..len);
                e[i] = rng.random_range(q..=u64::MAX);
            }
            2 => {
                let x = hachi_co_math::Ext(core::array::from_fn(|_| rng.random_range(0..q)));
                e = ctx.sub.embed(&x).0;
            }
            3 => e = ctx.sub.embed(&y).0,
            _ => {}
        }
        let e = Poly(e);
        let want = if e.0.len() != d {
            Err(VerifyError::Shape {
                expected: d,
                actual: e.0.len(),
            })
        } else if e.0.iter().any(|c| *c >= q) {
            Err(VerifyError::NonCanonical {
                what: "E(u)",
                value: *e.0.iter().find(|c| **c >= q).unwrap(),
                q,
            })
        } else if !ctx.sub.is_in_subfield(&e) {
            Err(VerifyError::NotInSubfield)
        } else if ctx.sub.extract(&e).unwrap() != y {
            Err(VerifyError::Mismatch)
        } else {
            Ok(())
        };
        let got = check(&ctx, &y, &e);
        assert_eq!(got, want);
        assert_eq!(verify(&ctx, &y, &e), want.is_ok());
        seen[match got {
            Ok(()) => 0,
            Err(VerifyError::Shape { .. }) => 1,
            Err(VerifyError::NonCanonical { .. }) => 2,
            Err(VerifyError::NotInSubfield) => 3,
            Err(_) => 4,
        }] += 1;
    }
    assert!(
        seen.iter().all(|c| *c > 0),
        "every outcome occurs: {seen:?}"
    );
}
