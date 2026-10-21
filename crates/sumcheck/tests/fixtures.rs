//! Every value of the fixtures' section `sumcheck` (`ref/fixtures/toy-n{1,2,3}.json`), for every
//! party of n = 1, 2, 3, recomputed by this crate from inputs built as `docs/m1-spec.md`
//! prescribes, with exact equality residue by residue; plus the oracle's relations `sc.*` checked
//! directly on the recomputed values.

mod common;

use common::{K, fah_inputs, honest_runs, instances};
use hachi_co_core::{
    BlinderKind, Challenger, PartyRandomness, ReplayChallenger, ReplayRandomness, sha256_residues,
};
use hachi_co_math::{Ext, Poly};
use hachi_co_sumcheck::{
    RangeProduct, claimed_sum, degree_at_nodes, eq_table, hypercube_sum, ind_rows, mle_eval,
    mstar_eval_factorized, mstar_table, quotient_share, row_indicator, w_table,
};

fn ext_digest(values: &[Ext<K>]) -> String {
    sha256_residues(values.iter().flat_map(|x| x.0))
}

/// The quotient shares and their sum (low half), and the relations
/// `sc.quotient_high_half_of_sum_is_0`, `sc.(X^d+1) r = M z' - y over Z_q[X]` and
/// `sc.quotient_share_1_matches_literal_formula`, checked with the exact products of
/// `ctx.ring2`.
#[test]
fn quotient_shares_match_every_party() {
    for inst in instances() {
        let (ctx, fx) = (&inst.ctx, &inst.fx);
        let l = &ctx.layout;
        let (d, q) = (l.d, ctx.q());
        let recorded = fx
            .get("/sumcheck/quotient_shares")
            .unwrap()
            .as_array()
            .unwrap()
            .len();
        assert_eq!(recorded, inst.n());
        let mut sum = vec![Poly(vec![0; 2 * d]); l.n_prime];
        for p in 1..=inst.n() {
            let r = quotient_share(ctx, &inst.m, &inst.shares[p - 1]).unwrap();
            let want = fx
                .polys_of_len(&format!("/sumcheck/quotient_shares/{}", p - 1), 2 * d)
                .unwrap();
            assert_eq!(r, want, "{}: <r>_{p}", inst.name);
            // The literal product 1/2 (1 - X^d)(M <z'>_p - {y}_p) in Z_q[X]/(X^{2d} + 1).
            let half = q.div_ceil(2); // 1/2 mod q
            let mut factor = vec![0u64; 2 * d];
            factor[0] = half;
            factor[d] = q - half;
            let pad = |x: &Poly| {
                let mut v = x.0.clone();
                v.resize(2 * d, 0);
                Poly(v)
            };
            let share = &inst.shares[p - 1];
            let z2: Vec<Poly> = share.z_prime.iter().map(pad).collect();
            for (i, row) in inst.m.iter().enumerate() {
                let m2: Vec<Poly> = row.iter().map(pad).collect();
                let full = ctx.ring2.inner(&m2, &z2).unwrap();
                let diff = ctx.ring2.sub(&full, &pad(&share.y[i])).unwrap();
                let literal = ctx.ring2.mul(&Poly(factor.clone()), &diff).unwrap();
                assert_eq!(
                    literal, r[i],
                    "{}: literal formula, party {p}, row {i}",
                    inst.name
                );
            }
            for (acc, x) in sum.iter_mut().zip(&r) {
                *acc = ctx.ring2.add(acc, x).unwrap();
            }
        }
        let low: Vec<Poly> = sum.iter().map(|p| Poly(p.0[..d].to_vec())).collect();
        assert_eq!(
            low,
            fx.polys("/sumcheck/quotient").unwrap(),
            "{}: sum",
            inst.name
        );
        assert!(
            sum.iter().all(|p| p.0[d..].iter().all(|x| *x == 0)),
            "{}: high halves",
            inst.name
        );
        // (X^d + 1) r = M z' - y over Z_q[X]: the products of degree < 2d do not wrap in ring2.
        let z_prime = fx.polys("/ring_to_linear/z_prime").unwrap();
        let pad = |x: &Poly| {
            let mut v = x.0.clone();
            v.resize(2 * d, 0);
            Poly(v)
        };
        let mut xd1 = vec![0u64; 2 * d];
        xd1[0] = 1;
        xd1[d] = 1;
        let z2: Vec<Poly> = z_prime.iter().map(pad).collect();
        for (i, row) in inst.m.iter().enumerate() {
            let m2: Vec<Poly> = row.iter().map(pad).collect();
            let lhs = ctx.ring2.mul(&Poly(xd1.clone()), &pad(&low[i])).unwrap();
            let rhs = ctx
                .ring2
                .sub(&ctx.ring2.inner(&m2, &z2).unwrap(), &pad(&inst.y[i]))
                .unwrap();
            assert_eq!(lhs, rhs, "{}: (X^d + 1) r = M z' - y, row {i}", inst.name);
        }
    }
}

/// The table digests of the fixtures: the <w~>_p tables (2^l'' residues), the M* table
/// (2^l'' x k) and the eq(tau_1) table (2^ceil(log nu) x k), with the fixture's shapes.
#[test]
fn table_digests_match() {
    for inst in instances() {
        let (ctx, fx) = (&inst.ctx, &inst.fx);
        let l = &ctx.layout;
        let f = ctx.field();
        assert_eq!(
            fx.ints("/sumcheck/digest_shapes/w_table").unwrap(),
            vec![1i64 << l.ell_pp]
        );
        assert_eq!(
            fx.ints("/sumcheck/digest_shapes/mstar_table").unwrap(),
            vec![1i64 << l.ell_pp, K as i64]
        );
        assert_eq!(
            fx.ints("/sumcheck/digest_shapes/eq_tau1_table").unwrap(),
            vec![1i64 << l.log_nu, K as i64]
        );
        for p in 1..=inst.n() {
            let share = &inst.shares[p - 1];
            let r = quotient_share(ctx, &inst.m, share).unwrap();
            let table = w_table(ctx, &share.z_prime, &r).unwrap();
            assert_eq!(table.len(), 1 << l.ell_pp);
            assert_eq!(
                sha256_residues(table.iter().copied()),
                fx.str(&format!("/sumcheck/digests/w_table_p{p}")).unwrap(),
                "{}: <w~>_{p} table",
                inst.name
            );
        }
        let coins = ReplayChallenger::new(fx)
            .unwrap()
            .sumcheck_coins(f, l.ell_pp, l.log_n, l.log_nu)
            .unwrap();
        let eq1 = eq_table(f, &coins.tau1).unwrap();
        assert_eq!(eq1.len(), 1 << l.log_nu);
        assert_eq!(
            ext_digest(&eq1),
            fx.str("/sumcheck/digests/eq_tau1_table").unwrap(),
            "{}: eq(tau_1)",
            inst.name
        );
        let mstar = mstar_table(ctx, &inst.m, &inst.e, &eq1, &coins.alpha).unwrap();
        assert_eq!(
            ext_digest(&mstar),
            fx.str("/sumcheck/digests/mstar_table").unwrap(),
            "{}: M*",
            inst.name
        );
        // The failure boundary: another alpha gives another table.
        let alpha2 = f.add(&coins.alpha, &f.one());
        let other = mstar_table(ctx, &inst.m, &inst.e, &eq1, &alpha2).unwrap();
        assert_ne!(ext_digest(&other), ext_digest(&mstar));
    }
}

/// The blinders (served by `ReplayRandomness`) and G_{0,p}, G_{1,p}; the coins (served by
/// `ReplayChallenger`) and the claimed sum a.
#[test]
fn blinders_sums_coins_and_claimed_sum_match() {
    for (inst, run) in instances().iter().zip(honest_runs()) {
        let (ctx, fx) = (&inst.ctx, &inst.fx);
        let l = &ctx.layout;
        let f = ctx.field();
        for p in 1..=inst.n() {
            let mut rng = ReplayRandomness::new(fx, p).unwrap();
            for (kind, key, sums) in [
                (BlinderKind::G0, "g0", &run.transcript.g0_sums),
                (BlinderKind::G1, "g1", &run.transcript.g1_sums),
            ] {
                let g = rng.blinder(f, kind, l.ell_pp, kind.degree(l)).unwrap();
                let prefix = format!("/sumcheck/blinders/{}/{key}", p - 1);
                assert_eq!(g.s0, fx.ext::<K>(&format!("{prefix}/s0")).unwrap());
                assert_eq!(g.s, fx.ext_rows::<K>(&format!("{prefix}/s")).unwrap());
                let private = &run.privates[p - 1];
                assert_eq!(
                    &g,
                    if key == "g0" {
                        &private.g0
                    } else {
                        &private.g1
                    }
                );
                let sum = hypercube_sum(f, &g);
                assert_eq!(sums[p - 1], sum);
                let recorded = fx
                    .ext::<K>(&format!("/sumcheck/{}/{}", key.to_uppercase(), p - 1))
                    .unwrap();
                assert_eq!(sum, recorded, "{}: G of {key}, party {p}", inst.name);
            }
        }
        let coins = &run.privates[0].coins;
        assert_eq!(coins.alpha, fx.ext::<K>("/sumcheck/coins/alpha").unwrap());
        assert_eq!(coins.rho, fx.ext::<K>("/sumcheck/coins/rho").unwrap());
        assert_eq!(coins.tau0, fx.exts::<K>("/sumcheck/coins/tau0").unwrap());
        assert_eq!(coins.tau1, fx.exts::<K>("/sumcheck/coins/tau1").unwrap());
        assert_eq!(
            run.privates[0].point,
            fx.exts::<K>("/sumcheck/coins/a").unwrap()
        );
        let eq1 = eq_table(f, &coins.tau1).unwrap();
        let a = claimed_sum(ctx, &eq1, &inst.y, &inst.y_ring, &coins.alpha).unwrap();
        assert_eq!(
            a,
            fx.ext::<K>("/sumcheck/a_claim").unwrap(),
            "{}: a",
            inst.name
        );
    }
}

/// Per round, H0 and H1 and every party's F_AH(I) inputs; the final values; M~*(a) (factorized,
/// and as the MLE of the table: relation `sc.mstar_table_mle(a) = factorized M~*(a)`).
#[test]
fn rounds_final_values_and_mstar_at_a_match() {
    for (inst, run) in instances().iter().zip(honest_runs()) {
        let (ctx, fx) = (&inst.ctx, &inst.fx);
        let l = &ctx.layout;
        let f = ctx.field();
        let recorded = common::fixture_transcript(fx);
        assert_eq!(run.transcript.rounds.len(), l.ell_pp);
        for r in 1..=l.ell_pp {
            for p in 1..=inst.n() {
                let (h0, h1) = fah_inputs(fx, r, p);
                let mine = &run.privates[p - 1].rounds[r - 1];
                assert_eq!(
                    mine.h0, h0,
                    "{}: F_AH input H0, round {r}, party {p}",
                    inst.name
                );
                assert_eq!(
                    mine.h1, h1,
                    "{}: F_AH input H1, round {r}, party {p}",
                    inst.name
                );
                assert_eq!((mine.h0.len(), mine.h1.len()), (l.d0 + 1, l.d1 + 1));
            }
            assert_eq!(
                run.transcript.rounds[r - 1],
                recorded.rounds[r - 1],
                "{}: round {r}",
                inst.name
            );
        }
        assert_eq!(
            run.transcript.finals, recorded.finals,
            "{}: final values",
            inst.name
        );
        // The whole transcript, except the base-PCS words, which the oracle does not model.
        assert_eq!(
            common::without_pcs(&run.transcript),
            recorded,
            "{}: the whole transcript",
            inst.name
        );
        let coins = &run.privates[0].coins;
        let point = &run.privates[0].point;
        let eq1 = eq_table(f, &coins.tau1).unwrap();
        let at_a = mstar_eval_factorized(ctx, &inst.m, &inst.e, &eq1, &coins.alpha, point).unwrap();
        assert_eq!(
            at_a,
            fx.ext::<K>("/sumcheck/mstar_at_a").unwrap(),
            "{}: M*(a)",
            inst.name
        );
        let table = mstar_table(ctx, &inst.m, &inst.e, &eq1, &coins.alpha).unwrap();
        assert_eq!(
            mle_eval(f, &table, point).unwrap(),
            at_a,
            "{}: MLE of M* at a",
            inst.name
        );
        // The final value of each party is its table's MLE at a.
        for (p0, private) in run.privates.iter().enumerate() {
            let lifted: Vec<Ext<K>> = private.w_table.iter().map(|x| f.from_base(*x)).collect();
            assert_eq!(
                mle_eval(f, &lifted, point).unwrap(),
                run.transcript.finals.w[p0]
            );
        }
    }
}

/// The degree audit (`sumcheck/round_degrees_unblinded`): the degrees of the unblinded round
/// polynomials sum_p F_{0,p,r} and sum_p F_{1,p,r} at every round, and the relations
/// `sc.deg(...) <= d0` and `<= d1`.
#[test]
fn degree_audit_matches() {
    for (inst, run) in instances().iter().zip(honest_runs()) {
        let (ctx, fx) = (&inst.ctx, &inst.fx);
        let l = &ctx.layout;
        let f = ctx.field();
        let want0 = fx.ints("/sumcheck/round_degrees_unblinded/F0").unwrap();
        let want1 = fx.ints("/sumcheck/round_degrees_unblinded/F1").unwrap();
        for r in 1..=l.ell_pp {
            let sum = |get: fn(&hachi_co_sumcheck::RoundValues<K>) -> &Vec<Ext<K>>| -> Vec<Ext<K>> {
                let len = get(&run.privates[0].rounds[r - 1]).len();
                (0..len)
                    .map(|t| {
                        run.privates
                            .iter()
                            .fold(f.zero(), |acc, pr| f.add(&acc, &get(&pr.rounds[r - 1])[t]))
                    })
                    .collect()
            };
            let d0 = degree_at_nodes(f, &sum(|v| &v.f0)).map_or(-1, |x| x as i64);
            let d1 = degree_at_nodes(f, &sum(|v| &v.f1)).map_or(-1, |x| x as i64);
            assert_eq!(d0, want0[r - 1], "{}: degree of F_0, round {r}", inst.name);
            assert_eq!(d1, want1[r - 1], "{}: degree of F_1, round {r}", inst.name);
            assert!(d0 <= l.d0 as i64 && d1 <= l.d1 as i64);
            // H = F + rho g at every node.
            for pr in &run.privates {
                let v = &pr.rounds[r - 1];
                assert_eq!(v.f0.len(), v.h0.len());
                assert_eq!(v.f1.len(), v.h1.len());
            }
        }
        for key in [
            "sc.deg(sum_p F_0 round poly) <= d0",
            "sc.deg(sum_p F_1 round poly) <= d1",
        ] {
            assert!(
                fx.bool(&format!("/relations/{}", hachi_co_core::escape_key(key)))
                    .unwrap()
            );
        }
    }
}

/// The oracle-level relations on the hypercube (`sc.zero_check_P_p(x)=0_on_hypercube` and
/// `sc.sum_x sum_p w~_p(x) M*(x) = a`) recomputed from this crate's tables: before any round,
/// every party's P_p vanishes at every point and the linear sum is the claimed sum. The failure
/// boundary: a share changed at one point breaks the zero check there.
#[test]
fn hypercube_relations_hold() {
    for (inst, run) in instances().iter().zip(honest_runs()) {
        let (ctx, fx) = (&inst.ctx, &inst.fx);
        let l = &ctx.layout;
        let f = ctx.field();
        let two_d = 2 * l.d;
        let range = RangeProduct::new(f.zq(), ctx.b());
        let ind = row_indicator(l, &[ind_rows(l)]).unwrap();
        let p_at = |e: u64, i: u64, w: u64| -> Ext<K> {
            let w = f.from_base(w);
            let e = f.from_base(e);
            let bracket = f.sub(&f.sub(&f.one(), &e), &f.from_base(i));
            f.add(&f.mul(&e, &range.eval(f, &w)), &f.mul(&bracket, &w))
        };
        for p in 1..=inst.n() {
            let table = &run.privates[p - 1].w_table;
            let e = row_indicator(l, &l.s_ranges(p)).unwrap();
            let bad = table
                .iter()
                .enumerate()
                .filter(|(x, w)| !f.is_zero(&p_at(e[x / two_d], ind[x / two_d], **w)))
                .count();
            assert_eq!(bad, 0, "{}: P_{p} on the hypercube", inst.name);
            // A digit pushed out of range, or a value outside S_p, is caught at that point.
            let x = 0; // row 0 is in S_1 and in no other S_p
            let outside = if p == 1 { ctx.b() } else { 1 };
            assert!(
                !f.is_zero(&p_at(e[x / two_d], ind[x / two_d], outside)),
                "{}",
                inst.name
            );
        }
        let coins = &run.privates[0].coins;
        let eq1 = eq_table(f, &coins.tau1).unwrap();
        let mstar = mstar_table(ctx, &inst.m, &inst.e, &eq1, &coins.alpha).unwrap();
        let mut total = f.zero();
        for private in &run.privates {
            for (w, m) in private.w_table.iter().zip(&mstar) {
                total = f.add(&total, &f.scale(m, *w));
            }
        }
        let a = claimed_sum(ctx, &eq1, &inst.y, &inst.y_ring, &coins.alpha).unwrap();
        assert_eq!(total, a, "{}: sum of <w~>_p M* = a", inst.name);
        for key in [
            "sc.zero_check_P_p(x)=0_on_hypercube",
            "sc.sum_x sum_p w~_p(x) M*(x) = a",
            "sc.mstar_table_mle(a) = factorized M~*(a)",
            "sc.a_claim(V) = a(provers)",
        ] {
            assert!(
                fx.bool(&format!("/relations/{}", hachi_co_core::escape_key(key)))
                    .unwrap()
            );
        }
    }
}
