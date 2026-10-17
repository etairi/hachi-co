//! `build_m` and `build_y` against the oracle's fixtures: the digest of M, the vector y, and the
//! draft's relation M z' = y (5_Building_Blocks.tex:154-182) checked directly on the recorded
//! witness, also through the parties' shares (ref/README.md, "Shares").
#![cfg(feature = "fixtures")]

use hachi_co_core::{
    Context, Error, Fixture, LinearShare, Matrix, Poly, PublicVectors, build_m, build_y,
    sha256_residues,
};

const K: usize = 4;

/// The inputs of `build_m` recorded in a fixture: b and a embedded, and the accepted attempt's c.
fn inputs(fx: &Fixture, ctx: &Context<K>) -> (Vec<Poly>, Vec<Poly>, Vec<Vec<i64>>) {
    let embed = |pointer: &str| -> Vec<Poly> {
        fx.exts::<K>(pointer)
            .unwrap()
            .iter()
            .map(|x| ctx.sub.embed(x))
            .collect()
    };
    let accepted = fx.usize("/ring_to_linear/accepted_attempt").unwrap();
    let c = fx
        .int_rows(&format!("/ring_to_linear/attempts/{accepted}/c"))
        .unwrap();
    (embed("/conv/b"), embed("/conv/a"), c)
}

fn digest(m: &Matrix) -> String {
    sha256_residues(m.iter().flatten().flat_map(|p| p.0.iter().copied()))
}

#[test]
fn m_matches_the_digest_of_every_fixture() {
    for name in Fixture::TOY {
        let fx = Fixture::load(name).unwrap();
        let ctx = fx.context::<K>().unwrap();
        let (b, a, c) = inputs(&fx, &ctx);
        let m = build_m(&ctx, &b, &a, &c).unwrap();
        assert_eq!(m.len(), ctx.layout.n_prime);
        assert!(m.iter().all(|row| row.len() == ctx.layout.mu));
        assert!(m.iter().flatten().all(|p| p.0.len() == ctx.layout.d));
        assert!(m.iter().flatten().flat_map(|p| &p.0).all(|x| *x < ctx.q()));
        assert_eq!(
            digest(&m),
            fx.str("/ring_to_linear/M_sha256").unwrap(),
            "{name}"
        );
        let nonzero = m
            .iter()
            .flatten()
            .filter(|p| p.0.iter().any(|x| *x != 0))
            .count();
        assert_eq!(
            nonzero,
            fx.usize("/ring_to_linear/M_nonzero_ring_entries").unwrap(),
            "{name}"
        );
    }
}

#[test]
fn y_matches_every_fixture_and_m_z_prime_equals_y() {
    for name in Fixture::TOY {
        let fx = Fixture::load(name).unwrap();
        let ctx = fx.context::<K>().unwrap();
        let (b, a, c) = inputs(&fx, &ctx);
        let accepted = fx.usize("/ring_to_linear/accepted_attempt").unwrap();
        let v = fx
            .polys(&format!("/ring_to_linear/attempts/{accepted}/v"))
            .unwrap();
        let u_com = fx.polys("/commit/u").unwrap();
        let y = build_y(&ctx, &v, &u_com).unwrap();
        assert_eq!(y, fx.polys("/ring_to_linear/y").unwrap(), "{name}");
        // The draft's relation, checked directly on the recorded z' (ring.mat_vec is the exact RNS
        // product of hachi-co-math, independent of the oracle's numpy code).
        let m = build_m(&ctx, &b, &a, &c).unwrap();
        let z_prime = fx.polys("/ring_to_linear/z_prime").unwrap();
        assert_eq!(z_prime.len(), ctx.layout.mu);
        assert_eq!(
            ctx.ring.mat_vec(&m, &z_prime).unwrap(),
            y,
            "{name}: M z' = y"
        );
        // The failure boundary: another challenge (an aborted attempt's, or one coefficient
        // changed) gives another M, for which the recorded z' is no solution.
        let mut c2 = c.clone();
        c2[0][0] += 1;
        let m2 = build_m(&ctx, &b, &a, &c2).unwrap();
        assert_ne!(digest(&m2), digest(&m));
        assert_ne!(ctx.ring.mat_vec(&m2, &z_prime).unwrap(), y);
        let mut v2 = v.clone();
        v2[0].0[0] = (v2[0].0[0] + 1) % ctx.q();
        assert_ne!(
            ctx.ring.mat_vec(&m, &z_prime).unwrap(),
            build_y(&ctx, &v2, &u_com).unwrap()
        );
    }
}

/// The parties' shares as the README defines them (z' on S_p, 0 elsewhere, <u>_p at mu - 1; y at
/// prover 1) add up to (z', y), the unheld positions of z' are zero, and summing the provers'
/// M <z'>_p - {y}_p gives 0.
#[test]
fn the_shares_of_the_layout_add_up_to_the_witness() {
    for name in Fixture::TOY {
        let fx = Fixture::load(name).unwrap();
        let ctx = fx.context::<K>().unwrap();
        let l = &ctx.layout;
        let ring = &ctx.ring;
        let z_prime = fx.polys("/ring_to_linear/z_prime").unwrap();
        let y = fx.polys("/ring_to_linear/y").unwrap();
        let u_shares = fx.polys("/conv/u_shares").unwrap();
        for r in l.unheld() {
            assert!(
                z_prime[r].iter().all(|p| p.0.iter().all(|x| *x == 0)),
                "{name}"
            );
        }
        let (b, a, c) = inputs(&fx, &ctx);
        let m = build_m(&ctx, &b, &a, &c).unwrap();
        let mut sum = LinearShare::zero(l);
        let mut residual = vec![ring.zero(); l.n_prime];
        for p in 1..=l.n {
            let mut share = LinearShare::zero(l);
            for r in l.s_ranges(p) {
                share.z_prime[r.clone()].clone_from_slice(&z_prime[r]);
            }
            share.z_prime[l.u] = u_shares[p - 1].clone();
            if p == 1 {
                share.y = y.clone();
            }
            share.validate(l).unwrap();
            assert_eq!(
                LinearShare::slice(l, &z_prime, &y, &u_shares[p - 1], p).unwrap(),
                share,
                "{name} p = {p}"
            );
            for (s, x) in sum.z_prime.iter_mut().zip(&share.z_prime) {
                *s = ring.add(s, x).unwrap();
            }
            for (s, x) in sum.y.iter_mut().zip(&share.y) {
                *s = ring.add(s, x).unwrap();
            }
            let mz = ring.mat_vec(&m, &share.z_prime).unwrap();
            for ((r, mzi), yi) in residual.iter_mut().zip(&mz).zip(&share.y) {
                *r = ring.add(r, &ring.sub(mzi, yi).unwrap()).unwrap();
            }
        }
        assert_eq!(sum.z_prime, z_prime, "{name}");
        assert_eq!(sum.y, y, "{name}");
        // Slice refuses other parties and shapes.
        let u0 = &u_shares[0];
        for p in [0, l.n + 1] {
            assert_eq!(
                LinearShare::slice(l, &z_prime, &y, u0, p),
                Err(Error::InvalidParty { party: p, n: l.n })
            );
        }
        let shape = |r| matches!(r, Err(Error::Shape { .. }));
        assert!(shape(LinearShare::slice(l, &z_prime[1..], &y, u0, 1)));
        assert!(shape(LinearShare::slice(l, &z_prime, &y[1..], u0, 1)));
        assert!(shape(LinearShare::slice(
            l,
            &z_prime,
            &y,
            &Poly(vec![0; 3]),
            1
        )));
        assert!(
            residual.iter().all(|r| r.0.iter().all(|x| *x == 0)),
            "{name}"
        );
        // The public vectors of the fixture have the shapes of the layout.
        let pv = PublicVectors::<K> {
            x_prime: fx.exts("/conv/x_prime").unwrap(),
            b: fx.exts("/conv/b").unwrap(),
            a: fx.exts("/conv/a").unwrap(),
            d: fx.exts("/conv/d").unwrap(),
            v_point: fx.poly("/conv/v_point").unwrap(),
            b_ring: b,
            a_ring: a,
        };
        pv.validate(l).unwrap();
        let mut bad = pv.clone();
        bad.d.pop();
        assert!(matches!(
            bad.validate(l),
            Err(Error::Shape { what: "d", .. })
        ));
    }
}

#[test]
fn wrong_shapes_are_errors() {
    let fx = Fixture::load("toy-n2").unwrap();
    let ctx = fx.context::<K>().unwrap();
    let (b, a, c) = inputs(&fx, &ctx);
    let shape = |r: hachi_co_core::Result<Matrix>| matches!(r, Err(Error::Shape { .. }));
    assert!(shape(build_m(&ctx, &b[1..], &a, &c)));
    assert!(shape(build_m(&ctx, &b, &a[1..], &c)));
    assert!(shape(build_m(&ctx, &b, &a, &c[1..])));
    let mut c_short = c.clone();
    c_short[2].pop();
    assert!(shape(build_m(&ctx, &b, &a, &c_short)));
    let mut b_short = b.clone();
    b_short[0].0.pop();
    assert!(shape(build_m(&ctx, &b_short, &a, &c)));
    let v = fx.polys("/ring_to_linear/attempts/0/v").unwrap();
    let u = fx.polys("/commit/u").unwrap();
    assert!(matches!(build_y(&ctx, &[], &u), Err(Error::Shape { .. })));
    assert!(matches!(build_y(&ctx, &v, &[]), Err(Error::Shape { .. })));
    let mut v_short = v.clone();
    v_short[0].0.pop();
    assert!(matches!(build_y(&ctx, &v_short, &u), Err(Error::Math(_))));
    // Unreduced inputs are reduced.
    let mut v_big = v.clone();
    v_big[0].0[0] += ctx.q();
    assert_eq!(
        build_y(&ctx, &v_big, &u).unwrap(),
        build_y(&ctx, &v, &u).unwrap()
    );
}
