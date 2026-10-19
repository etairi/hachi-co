//! The verifier against the fixtures: it accepts every recorded transcript with replayed
//! coins, its named checks are the oracle's, the final claims match the provers' polynomials
//! (ideal PCS), and every case of `negative_tests.transcript` and `negative_tests.witness` fails
//! exactly as recorded (the same set of failing checks, with the PCS openings checked by the
//! ideal PCS and the Pi_conv check by its stand-in).

mod common;

use common::{
    Instance, K, Run, conv_check, fixture_transcript, honest_runs, ideal_pcs, instances,
    run_parties,
};
use hachi_co_core::{Fixture, LinearShare, Poly, ReplayChallenger, escape_key};
use hachi_co_math::Ext;
use hachi_co_sumcheck::{
    PublicInput, SumcheckPrivate, SumcheckTranscript, VerifyError, check_names, quotient_share,
    verify, verify_checks,
};
use std::collections::BTreeSet;

/// The failing checks of the verifier and of the ideal PCS (sorted), for a transcript, public
/// input and the provers' private data.
fn failing(
    inst: &Instance,
    transcript: &SumcheckTranscript<K>,
    public: &PublicInput<'_>,
    privates: &[SumcheckPrivate<K>],
) -> Vec<String> {
    let mut ch = ReplayChallenger::new(&inst.fx).unwrap();
    let report = verify_checks(&inst.ctx, public, transcript, &mut ch).unwrap();
    let mut out: Vec<String> = report.failing();
    out.extend(
        ideal_pcs(inst.field(), &report.claims, privates)
            .into_iter()
            .filter(|(_, ok)| !ok)
            .map(|(name, _)| name),
    );
    out.sort();
    out
}

fn public(inst: &Instance) -> PublicInput<'_> {
    PublicInput::new(&inst.ctx, &inst.m, &inst.y, &inst.e, &inst.y_ring)
}

/// The values $H\_{0,r}$ or $H\_{1,r}$ of round `r` (1-based) of a transcript.
fn message_mut<'a>(t: &'a mut SumcheckTranscript<K>, r: usize, which: &str) -> &'a mut Vec<Ext<K>> {
    let round = &mut t.rounds[r - 1];
    if which == "H0" {
        &mut round.h0
    } else {
        &mut round.h1
    }
}

/// The recorded failing checks of a negative test.
fn recorded(fx: &Fixture, section: &str, case: &str) -> Vec<String> {
    let pointer = format!(
        "/negative_tests/{section}/{}/failing_checks",
        escape_key(case)
    );
    let mut v: Vec<String> = fx
        .get(&pointer)
        .unwrap_or_else(|_| panic!("{pointer}"))
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_owned())
        .collect();
    v.sort();
    v
}

fn recorded_cases(fx: &Fixture, section: &str) -> BTreeSet<String> {
    fx.get(&format!("/negative_tests/{section}"))
        .unwrap()
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect()
}

/// The recorded transcripts are accepted with replayed coins; the named checks, together with
/// the ideal PCS openings and the Pi_conv check, are exactly the fixture's `verifier/checks`,
/// all passing, and there are 1 + 2 l'' + 3n + 2 of them (the conv check, two per round, three
/// openings per prover and the two final checks, as `ref/run_oracle.py` counts them: 36, 39, 44,
/// 47 and 61 for toy-n1, n2, n3, n4, n8); a and M~*(a) are the recorded ones; the claims are the
/// provers' polynomials at the point.
#[test]
fn fixture_transcripts_are_accepted() {
    for (inst, run) in instances().iter().zip(honest_runs()) {
        let (ctx, fx) = (&inst.ctx, &inst.fx);
        let transcript = fixture_transcript(fx);
        let pubin = public(inst);
        let mut ch = ReplayChallenger::new(fx).unwrap();
        let report = verify_checks(ctx, &pubin, &transcript, &mut ch).unwrap();
        assert!(report.accepted(), "{}: {:?}", inst.name, report.failing());
        assert_eq!(
            report.claimed_sum,
            fx.ext::<K>("/sumcheck/a_claim").unwrap()
        );
        assert_eq!(
            report.mstar_at_a,
            fx.ext::<K>("/sumcheck/mstar_at_a").unwrap()
        );
        assert_eq!(
            report.claims.point,
            fx.exts::<K>("/sumcheck/coins/a").unwrap()
        );
        assert_eq!(report.claims.claims.len(), 3 * inst.n());
        let pcs = ideal_pcs(inst.field(), &report.claims, &run.privates);
        assert!(pcs.iter().all(|(_, ok)| *ok), "{}: {pcs:?}", inst.name);
        let conv = conv_check(
            ctx,
            &fx.ext::<K>("/input/y").unwrap(),
            &fx.poly("/conv/E_u").unwrap(),
        );
        assert!(conv);
        // The set of names: conv_y_eq_E(u), the round checks, the PCS openings, the final checks.
        let mut names: Vec<(String, bool)> = vec![("conv_y_eq_E(u)".to_owned(), conv)];
        let checks = &report.checks;
        names.extend(
            checks[..checks.len() - 2]
                .iter()
                .map(|c| (c.name.clone(), c.passed)),
        );
        names.extend(pcs);
        names.extend(
            checks[checks.len() - 2..]
                .iter()
                .map(|c| (c.name.clone(), c.passed)),
        );
        // serde_json without `preserve_order` lists the keys sorted, so compare sorted lists;
        // the oracle's order (sumcheck.py, `verify`) is that of `check_names`.
        let want: Vec<(String, bool)> = fx
            .get("/verifier/checks")
            .unwrap()
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.as_bool().unwrap()))
            .collect();
        let mut sorted = names.clone();
        sorted.sort();
        let mut want_sorted = want.clone();
        want_sorted.sort();
        assert_eq!(sorted, want_sorted, "{}: the named checks", inst.name);
        assert_eq!(want.len(), 1 + 2 * ctx.layout.ell_pp + 3 * inst.n() + 2);
        assert!(fx.bool("/verifier/accept").unwrap());
        let mut all = vec!["conv_y_eq_E(u)".to_owned()];
        all.extend(check_names(&ctx.layout));
        assert_eq!(
            names.iter().map(|(k, _)| k.clone()).collect::<Vec<_>>(),
            all,
            "{}: the oracle's order",
            inst.name
        );
        // verify returns the same claims.
        let mut ch = ReplayChallenger::new(fx).unwrap();
        assert_eq!(
            verify(ctx, &pubin, &transcript, &mut ch).unwrap(),
            report.claims
        );
    }
}

/// Every case of `negative_tests.transcript`: the tampered transcript (or public input) fails
/// exactly the recorded checks.
#[test]
fn transcript_tampering_fails_as_recorded() {
    for (inst, run) in instances().iter().zip(honest_runs()) {
        let (ctx, fx) = (&inst.ctx, &inst.fx);
        let f = inst.field();
        let l = &ctx.layout;
        let honest = fixture_transcript(fx);
        let pubin = public(inst);
        assert!(failing(inst, &honest, &pubin, &run.privates).is_empty());
        let mut seen = BTreeSet::new();
        let mut check = |case: String, got: Vec<String>| {
            assert_eq!(
                got,
                recorded(fx, "transcript", &case),
                "{}: {case}",
                inst.name
            );
            assert!(
                fx.bool(&format!(
                    "/negative_tests/transcript/{}/passed",
                    escape_key(&case)
                ))
                .unwrap()
            );
            seen.insert(case);
        };
        let rounds: BTreeSet<usize> = [1, l.ell_pp.div_ceil(2), l.ell_pp].into_iter().collect();
        for r in rounds {
            for which in ["H0", "H1"] {
                // H(1) += 1.
                let mut t = honest.clone();
                let values = message_mut(&mut t, r, which);
                values[0] = f.add(&values[0], &f.one());
                check(
                    format!("tamper {which}(1) += 1 at round {r}"),
                    failing(inst, &t, &pubin, &run.privates),
                );
                // H += 5 X (X - 1): the sum at 0 and 1 is unchanged.
                let mut t = honest.clone();
                let values = message_mut(&mut t, r, which);
                for (i, v) in values.iter_mut().enumerate() {
                    let x = (i + 1) as u64;
                    *v = f.add(v, &f.from_base(5 * x * (x - 1)));
                }
                check(
                    format!("tamper {which} += 5 X(X-1) at round {r}"),
                    failing(inst, &t, &pubin, &run.privates),
                );
            }
        }
        for p in 1..=inst.n() {
            let mut t = honest.clone();
            t.finals.w[p - 1] = f.add(&t.finals.w[p - 1], &f.one());
            check(
                format!("tamper final <w~>_{p}(a) += 1"),
                failing(inst, &t, &pubin, &run.privates),
            );
            let mut t = honest.clone();
            t.finals.g0[p - 1] = f.add(&t.finals.g0[p - 1], &f.one());
            check(
                format!("tamper final g_0,{p}(a) += 1"),
                failing(inst, &t, &pubin, &run.privates),
            );
            let mut t = honest.clone();
            t.finals.g1[p - 1] = f.add(&t.finals.g1[p - 1], &f.one());
            check(
                format!("tamper final g_1,{p}(a) += 1"),
                failing(inst, &t, &pubin, &run.privates),
            );
            let mut t = honest.clone();
            t.g0_sums[p - 1] = f.add(&t.g0_sums[p - 1], &f.one());
            check(
                format!("tamper G_0,{p} += 1"),
                failing(inst, &t, &pubin, &run.privates),
            );
        }
        // The claimed y + 1: Pi_conv's check fails, and in the sumcheck round 1 of H1.
        let y = fx.ext::<K>("/input/y").unwrap();
        let y_bad = f.add(&y, &f.one());
        let e_u = fx.poly("/conv/E_u").unwrap();
        assert!(conv_check(ctx, &y, &e_u));
        assert!(!conv_check(ctx, &y_bad, &e_u));
        assert!(
            fx.bool("/negative_tests/transcript/tamper claimed y += 1: Pi_conv check/passed")
                .unwrap()
        );
        let mut conv_cases = vec!["tamper claimed y += 1: Pi_conv check".to_owned()];
        let y_ring_bad = ctx.sub.embed(&y_bad);
        let mut bad = public(inst);
        bad.y_ring = &y_ring_bad;
        check(
            "tamper claimed y += 1: sumcheck".to_owned(),
            failing(inst, &honest, &bad, &run.privates),
        );
        // A tampered E(u) message fails Pi_conv's check (it is not used by the sumcheck).
        let mut e_u_bad = e_u.clone();
        e_u_bad.0[0] = (e_u_bad.0[0] + 1) % ctx.q();
        assert!(!conv_check(ctx, &y, &e_u_bad));
        assert!(
            fx.bool("/negative_tests/transcript/tamper E(u) message: Pi_conv check/passed")
                .unwrap()
        );
        conv_cases.push("tamper E(u) message: Pi_conv check".to_owned());
        // v of the accepted attempt, coefficient 0 + 1, enters y and hence a.
        let mut y_vec = inst.y.clone();
        y_vec[0].0[0] = (y_vec[0].0[0] + 1) % ctx.q();
        let bad = PublicInput::new(ctx, &inst.m, &y_vec, &inst.e, &inst.y_ring);
        check(
            "tamper v (accepted attempt) coefficient 0 += 1".to_owned(),
            failing(inst, &honest, &bad, &run.privates),
        );
        seen.extend(conv_cases);
        assert_eq!(
            seen,
            recorded_cases(fx, "transcript"),
            "{}: every case",
            inst.name
        );
    }
}

/// Party p's share with `z'` changed by `edit` (position, coefficient, new residue).
fn edit_shares(shares: &[LinearShare], edits: &[(usize, usize, usize, u64)]) -> Vec<LinearShare> {
    let mut out = shares.to_vec();
    for (p, pos, coef, value) in edits {
        out[p - 1].z_prime[*pos].0[*coef] = *value;
    }
    out
}

/// Runs the parties on (possibly malicious) shares and returns the failing checks.
fn witness_failing(
    inst: &Instance,
    shares: &[LinearShare],
    quotients: Option<&[Vec<Poly>]>,
) -> Vec<String> {
    let Run {
        transcript,
        privates,
    } = run_parties(inst, shares, quotients);
    failing(inst, &transcript, &public(inst), &privates)
}

/// Every case of `negative_tests.witness`, rebuilt as `ref/hachico_ref/negative.py` does: the
/// provers run the sumcheck honestly on modified shares (same coins) and the verifier fails
/// exactly the recorded checks.
#[test]
fn malicious_witnesses_fail_as_recorded() {
    for inst in instances() {
        let (ctx, fx) = (&inst.ctx, &inst.fx);
        let l = &ctx.layout;
        let (q, b, d) = (ctx.q(), ctx.b(), l.d);
        let zq = ctx.ring.zq();
        let shares = &inst.shares;
        let mut seen = BTreeSet::new();
        let mut check = |case: &str, got: Vec<String>| {
            let pointer = format!("/negative_tests/witness/{}", escape_key(case));
            let want = fx
                .get(&format!("{pointer}/failing_checks"))
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s.as_str().unwrap().to_owned())
                .collect::<BTreeSet<_>>();
            assert_eq!(
                got.iter().cloned().collect::<BTreeSet<_>>(),
                want,
                "{}: {case}",
                inst.name
            );
            assert!(fx.bool(&format!("{pointer}/passed")).unwrap());
            seen.insert(case.to_owned());
        };
        // (a) An out-of-range z_hat digit of prover 1 with J z_hat unchanged: digit t -> x + b,
        //     digit t + 1 -> y - 1 at the first (index, coefficient) with t < tau - 1 and x >= 0.
        let zh = &shares[0].z_prime[l.z..l.u];
        let (idx, l0) = (0..zh.len())
            .filter(|i| i % l.tau != l.tau - 1)
            .find_map(|i| {
                (0..d)
                    .find(|j| zq.to_centered(zh[i].0[*j]) >= 0)
                    .map(|j| (i, j))
            })
            .unwrap();
        let x = zq.to_centered(zh[idx].0[l0]);
        let y = zq.to_centered(zh[idx + 1].0[l0]);
        let bad = edit_shares(
            shares,
            &[
                (1, l.z + idx, l0, zq.from_i64(x + b as i64)),
                (1, l.z + idx + 1, l0, zq.from_i64(y - 1)),
            ],
        );
        // M z' = y still holds (the recorded flag).
        let mut z_sum = vec![ctx.ring.zero(); l.mu];
        for s in &bad {
            for (acc, z) in z_sum.iter_mut().zip(&s.z_prime) {
                *acc = ctx.ring.add(acc, z).unwrap();
            }
        }
        let case = "witness: out-of-range z_hat digit (J z_hat unchanged)";
        assert_eq!(ctx.ring.mat_vec(&inst.m, &z_sum).unwrap(), inst.y);
        assert!(
            fx.bool(&format!(
                "/negative_tests/witness/{}/linear_relation_still_holds",
                escape_key(case)
            ))
            .unwrap()
        );
        check(case, witness_failing(inst, &bad, None));
        // (b) An in-range change of the first w_hat digit of prover 1's first block.
        let pos = l.w + l.blocks_of(1).start * l.delta;
        let x = zq.to_centered(shares[0].z_prime[pos].0[0]);
        let top = (b as i64 + 1) / 2 - 1;
        let new = if x < top { x + 1 } else { x - 1 };
        let bad = edit_shares(shares, &[(1, pos, 0, zq.from_i64(new))]);
        check(
            "witness: in-range w_hat digit changed (M z' != y)",
            witness_failing(inst, &bad, None),
        );
        if inst.n() >= 2 {
            // (c) Prover 2 writes into prover 1's slot z_hat, prover 1 compensates.
            let bad = edit_shares(
                shares,
                &[
                    (2, l.z, 0, zq.add(shares[1].z_prime[l.z].0[0], 1)),
                    (1, l.z, 0, zq.sub(shares[0].z_prime[l.z].0[0], 1)),
                ],
            );
            check(
                "witness: prover 2 nonzero outside its slice (aggregate unchanged)",
                witness_failing(inst, &bad, None),
            );
            // (d) Re-randomized <u> shares (coefficient 3, +-7) are accepted.
            let bad = edit_shares(
                shares,
                &[
                    (1, l.u, 3, zq.add(shares[0].z_prime[l.u].0[3], 7)),
                    (2, l.u, 3, zq.sub(shares[1].z_prime[l.u].0[3], 7)),
                ],
            );
            check(
                "witness: re-randomized <u> shares are accepted",
                witness_failing(inst, &bad, None),
            );
            // Re-randomized quotient shares (row 0, coefficient d + 2, +-11) are accepted.
            let mut qs: Vec<Vec<Poly>> = shares
                .iter()
                .map(|s| quotient_share(ctx, &inst.m, s).unwrap())
                .collect();
            qs[0][0].0[d + 2] = zq.add(qs[0][0].0[d + 2], 11);
            qs[1][0].0[d + 2] = zq.sub(qs[1][0].0[d + 2], 11);
            check(
                "witness: re-randomized quotient shares are accepted",
                witness_failing(inst, shares, Some(&qs)),
            );
        }
        // (e) A quotient whose sum has a nonzero coefficient of X^{d+1} (row 1).
        let mut qs: Vec<Vec<Poly>> = shares
            .iter()
            .map(|s| quotient_share(ctx, &inst.m, s).unwrap())
            .collect();
        qs[0][1].0[d + 1] = (qs[0][1].0[d + 1] + 1) % q;
        check(
            "witness: quotient sum with a nonzero coefficient of X^{d+1}",
            witness_failing(inst, shares, Some(&qs)),
        );
        assert_eq!(
            seen,
            recorded_cases(fx, "witness"),
            "{}: every case",
            inst.name
        );
    }
}

/// Malformed transcripts and public inputs give errors, never panics: missing or extra rounds,
/// messages of the wrong length, a wrong number of parties, non-canonical residues, and public
/// inputs of the wrong shape.
#[test]
fn malformed_transcripts_are_errors() {
    let inst = &instances()[1];
    let (ctx, fx) = (&inst.ctx, &inst.fx);
    let q = ctx.q();
    let honest = fixture_transcript(fx);
    let pubin = public(inst);
    let run = |t: &SumcheckTranscript<K>, p: &PublicInput<'_>| {
        let mut ch = ReplayChallenger::new(fx).unwrap();
        verify(ctx, p, t, &mut ch)
    };
    assert!(run(&honest, &pubin).is_ok());
    let shape = |r: Result<_, VerifyError>| matches!(r, Err(VerifyError::Shape { .. }));
    let noncanonical =
        |r: Result<_, VerifyError>| matches!(r, Err(VerifyError::NonCanonical { .. }));
    let mut t = honest.clone();
    t.rounds.pop();
    assert!(shape(run(&t, &pubin)));
    let mut t = honest.clone();
    t.rounds.push(t.rounds[0].clone());
    assert!(shape(run(&t, &pubin)));
    let mut t = honest.clone();
    t.rounds[3].h0.pop();
    assert!(shape(run(&t, &pubin)));
    let mut t = honest.clone();
    t.rounds[3].h1.push(Ext([0; K]));
    assert!(shape(run(&t, &pubin)));
    let mut t = honest.clone();
    t.g0_sums.pop();
    assert!(shape(run(&t, &pubin)));
    let mut t = honest.clone();
    t.g1_sums.push(Ext([0; K]));
    assert!(shape(run(&t, &pubin)));
    for which in 0..3 {
        let mut t = honest.clone();
        let v = [&mut t.finals.w, &mut t.finals.g0, &mut t.finals.g1];
        v.into_iter().nth(which).unwrap().clear();
        assert!(shape(run(&t, &pubin)));
    }
    let mut t = honest.clone();
    t.rounds = Vec::new();
    assert!(shape(run(&t, &pubin)));
    // Non-canonical residues: q itself, and u64::MAX.
    for bad in [q, u64::MAX] {
        let mut t = honest.clone();
        t.rounds[2].h1[1].0[2] = bad;
        assert!(noncanonical(run(&t, &pubin)));
        let mut t = honest.clone();
        t.g0_sums[0].0[0] = bad;
        assert!(noncanonical(run(&t, &pubin)));
        let mut t = honest.clone();
        t.finals.w[1].0[3] = bad;
        assert!(noncanonical(run(&t, &pubin)));
    }
    // Residues that are canonical but wrong are a rejection, not an error.
    let mut t = honest.clone();
    t.rounds[2].h1[1].0[2] = (t.rounds[2].h1[1].0[2] + 1) % q;
    assert!(matches!(run(&t, &pubin), Err(VerifyError::Rejected { .. })));
    // Public inputs of the wrong shape or with non-canonical residues.
    let m_short: Vec<Vec<Poly>> = inst.m[..inst.m.len() - 1].to_vec();
    let p = PublicInput::new(ctx, &m_short, &inst.y, &inst.e, &inst.y_ring);
    assert!(shape(run(&honest, &p)));
    let y_short = inst.y[1..].to_vec();
    let p = PublicInput::new(ctx, &inst.m, &y_short, &inst.e, &inst.y_ring);
    assert!(shape(run(&honest, &p)));
    let e_bad: Vec<Vec<u64>> = inst.e.iter().map(|r| r[1..].to_vec()).collect();
    let p = PublicInput::new(ctx, &inst.m, &inst.y, &e_bad, &inst.y_ring);
    assert!(shape(run(&honest, &p)));
    let y_ring_long = Poly(vec![0; ctx.layout.d + 1]);
    let p = PublicInput::new(ctx, &inst.m, &inst.y, &inst.e, &y_ring_long);
    assert!(shape(run(&honest, &p)));
    let mut y_nc = inst.y.clone();
    y_nc[0].0[0] = q;
    let p = PublicInput::new(ctx, &inst.m, &y_nc, &inst.e, &inst.y_ring);
    assert!(noncanonical(run(&honest, &p)));
    let mut p = public(inst);
    p.partition.pop();
    assert!(shape(run(&honest, &p)));
    let mut p = public(inst);
    p.partition[0].push(0..usize::MAX);
    assert!(shape(run(&honest, &p)));
    let mut p = public(inst);
    #[allow(clippy::reversed_empty_ranges)]
    p.partition[1].push(5..3);
    assert!(shape(run(&honest, &p)));
}

/// The exact boundary of the range check of F_0, {-b+1, ..., b-1} (`5_Building_Blocks.tex`:269),
/// which is wider than the digit set [ceil(-b/2), ceil(b/2) - 1] of the gadgets: moving a z_hat
/// digit of prover 1 by +-b (and the next digit by -+1, so that J z_hat and M z' = y are
/// unchanged) is accepted when the digit lands on +-(b - 1) and rejected (round_01_H0 only) when
/// it lands on +-b. At toy-n1 and toy-n3.
#[test]
fn range_check_boundary() {
    for inst in [&instances()[0], &instances()[2]] {
        let ctx = &inst.ctx;
        let l = &ctx.layout;
        let zq = ctx.ring.zq();
        let b = ctx.b() as i64;
        let zh = &inst.shares[0].z_prime[l.z..l.u];
        // The first coefficient of a non-top digit with the given centered value.
        let find = |value: i64| {
            (0..zh.len())
                .filter(|i| i % l.tau != l.tau - 1)
                .find_map(|i| {
                    (0..l.d)
                        .find(|j| zq.to_centered(zh[i].0[*j]) == value)
                        .map(|j| (i, j))
                })
                .unwrap()
        };
        for (value, shift, accepted) in
            [(-1, b, true), (1, -b, true), (0, b, false), (0, -b, false)]
        {
            let (idx, j) = find(value);
            let next = zq.to_centered(zh[idx + 1].0[j]);
            let carry = if shift > 0 { -1 } else { 1 };
            let bad = edit_shares(
                &inst.shares,
                &[
                    (1, l.z + idx, j, zq.from_i64(value + shift)),
                    (1, l.z + idx + 1, j, zq.from_i64(next + carry)),
                ],
            );
            assert!((next + carry).abs() < b, "the carried digit stays in range");
            let got = witness_failing(inst, &bad, None);
            let want: Vec<String> = if accepted {
                vec![]
            } else {
                vec!["round_01_H0".to_owned()]
            };
            assert_eq!(got, want, "{}: digit {value} moved by {shift}", inst.name);
        }
    }
}
