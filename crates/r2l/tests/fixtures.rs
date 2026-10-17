//! Every value of the fixtures' `ring_to_linear` section against this crate's party-local
//! functions, for every party of toy-n1, toy-n2 and toy-n3, attempt by attempt (aborted attempts
//! included). The network is emulated here: the F_AH(D) inputs are summed and the responses
//! gathered by the test. Then the relation checks of the oracle named `r2l.*`.

mod common;

use common::{K, attempts, load, u_com, u_share};
use hachi_co_core::{
    Challenger, Fixture, LinearShare, PartyRandomness, Poly, ReplayChallenger, ReplayRandomness,
    sha256_residues, short,
};
use hachi_co_r2l::{
    AttemptSecrets, BlockW, Error, UnpackedWitness, aggregate, check_challenges, d_slice,
    fah_d_input, fold, linear_relation, norm_check, owned_w, relation_holds, residual, respond,
    response_len, row_block_checks, sample_attempt, sum_shares, y_share, z_hat, z_prime_share,
};

/// Everything a test needs from one accepted run, recomputed by this crate.
struct Run {
    accepted: usize,
    v: Vec<Poly>,
    c: Vec<Vec<i64>>,
    shares: Vec<LinearShare>,
    witness: UnpackedWitness,
    masks: Vec<Vec<i64>>,
    z_hat: Vec<Poly>,
}

/// Replays the oracle's run of `name` through the party-local functions, comparing every recorded
/// value of `ring_to_linear`, and returns the accepted attempt's values.
fn replay_and_compare(name: &str) -> Run {
    let (fx, ctx, pv, states) = load(name);
    let l = &ctx.layout;
    let ring = &ctx.ring;
    let (n, d, delta) = (l.n, l.d, l.delta);
    let key = |k: &str| format!("/ring_to_linear/{k}");

    // Step 1: w and w_hat of every block, each computed by its owner (0 for unowned blocks).
    let ws: Vec<Vec<BlockW>> = states
        .iter()
        .map(|st| owned_w(&ctx, &pv, st).unwrap())
        .collect();
    let mut w = vec![ring.zero(); l.n_blocks];
    let mut w_hat = vec![ring.zero(); l.n_blocks * delta];
    for (p, bws) in (1..=n).zip(&ws) {
        assert_eq!(
            bws.iter().map(|b| b.index).collect::<Vec<_>>(),
            l.blocks_of(p).collect::<Vec<_>>()
        );
        for bw in bws {
            w[bw.index] = bw.w.clone();
            w_hat[l.w_hat_range(bw.index)].clone_from_slice(&bw.w_hat);
        }
    }
    assert_eq!(w, fx.polys(&key("w")).unwrap(), "{name}: w");
    assert_eq!(w_hat, fx.polys(&key("w_hat")).unwrap(), "{name}: w_hat");

    let accepted = fx.usize(&key("accepted_attempt")).unwrap();
    let recorded = attempts(&fx);
    assert_eq!(
        recorded,
        accepted + 1,
        "{name}: the accepted attempt is the last"
    );
    let mut ch = ReplayChallenger::new(&fx).unwrap();
    let mut rngs: Vec<ReplayRandomness> = (1..=n)
        .map(|p| ReplayRandomness::new(&fx, p).unwrap())
        .collect();
    let mut run = None;
    for a in 0..recorded {
        let at = |k: &str| key(&format!("attempts/{a}/{k}"));
        assert_eq!(fx.usize(&at("attempt")).unwrap(), a);
        // Every prover's step 2 and its F_AH(D) input; the test sums the inputs.
        let secrets: Vec<AttemptSecrets> = rngs
            .iter_mut()
            .map(|rng| sample_attempt(&ctx, &pv, rng, a as u64).unwrap())
            .collect();
        let mut v = vec![ring.zero(); l.n_d];
        for (p, s) in (1..=n).zip(&secrets) {
            let pk = |k: &str| at(&format!("provers/{}/{k}", p - 1));
            assert_eq!(fx.usize(&pk("prover")).unwrap(), p);
            assert_eq!(s.attempt, a as u64);
            assert_eq!(
                s.mask,
                fx.ints(&pk("mask")).unwrap(),
                "{name} a={a} p={p}: mask"
            );
            assert_eq!(
                s.rho,
                fx.ints(&pk("rho")).unwrap(),
                "{name} a={a} p={p}: rho"
            );
            assert_eq!(
                s.images.t0,
                fx.polys(&pk("t0")).unwrap(),
                "{name} a={a} p={p}: t0"
            );
            assert_eq!(s.images.t0_hat, fx.polys(&pk("t0_hat")).unwrap(), "t0_hat");
            assert_eq!(
                s.images.w0,
                fx.poly(&pk("w0")).unwrap(),
                "{name} a={a} p={p}: w0"
            );
            assert_eq!(s.images.w0_hat, fx.polys(&pk("w0_hat")).unwrap(), "w0_hat");
            let slice = d_slice(&ctx, p, &ws[p - 1], s).unwrap();
            let input = fah_d_input(&ctx, &slice).unwrap();
            for (vi, x) in v.iter_mut().zip(&input) {
                *vi = ring.add(vi, x).unwrap();
            }
        }
        assert_eq!(v, fx.polys(&at("v")).unwrap(), "{name} a={a}: v");
        // The verifier's challenges (replayed), checked by every prover.
        let c =
            Challenger::<K>::r2l_challenges(&mut ch, a as u64, l.n_blocks, d, ctx.omega()).unwrap();
        assert_eq!(c, fx.int_rows(&at("c")).unwrap(), "{name} a={a}: c");
        check_challenges(&ctx, &c).unwrap();
        assert!(fx.bool(&at("c_in_C")).unwrap());
        // Each prover's fold and Rej.
        let mut responses = Vec::new();
        let mut aborting = Vec::new();
        for (p, (st, s)) in (1..=n).zip(states.iter().zip(&secrets)) {
            let pk = |k: &str| at(&format!("provers/{}/{k}", p - 1));
            let f = fold(&ctx, st, &c).unwrap();
            assert_eq!(f.len(), response_len(&ctx));
            assert_eq!(
                short::inf_norm(&f),
                fx.u64(&pk("fold_norm")).unwrap(),
                "fold_norm"
            );
            assert!(short::inf_norm(&f) <= ctx.beta1());
            let candidate: Vec<i64> = f.iter().zip(&s.mask).map(|(x, r)| x + r).collect();
            assert_eq!(
                short::inf_norm(&candidate),
                fx.u64(&pk("z_norm")).unwrap(),
                "{name} a={a} p={p}: z_norm"
            );
            let response = respond(&ctx, &f, &s.mask).unwrap();
            let accept = fx.bool(&pk("accept")).unwrap();
            assert_eq!(response.is_some(), accept, "{name} a={a} p={p}: accept");
            assert_eq!(
                accept,
                short::inf_norm(&candidate) <= ctx.gamma() - ctx.beta1(),
                "Rej accepts iff ||z_p|| <= gamma - beta_1 (4_preliminaries.tex:119)"
            );
            match response {
                Some(z_p) => {
                    assert_eq!(z_p, candidate);
                    responses.push(z_p);
                }
                None => aborting.push(p as u64),
            }
        }
        let abort = !aborting.is_empty();
        assert_eq!(abort, fx.bool(&at("abort")).unwrap(), "{name} a={a}: abort");
        assert_eq!(
            aborting,
            fx.ints(&at("aborting_provers"))
                .unwrap()
                .iter()
                .map(|x| *x as u64)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            abort,
            a != accepted,
            "{name}: only the last attempt is accepted"
        );
        if abort {
            continue;
        }

        // The accepted attempt: the gather at prover 1 and prover 1's steps.
        for (p, z_p) in (1..=n).zip(&responses) {
            assert_eq!(
                *z_p,
                fx.ints(&key(&format!("z_i/{}", p - 1))).unwrap(),
                "{name}: z_{p}"
            );
        }
        let z = aggregate(&ctx, &responses).unwrap();
        assert_eq!(z, fx.ints(&key("z")).unwrap(), "{name}: z");
        assert_eq!(short::inf_norm(&z), fx.u64(&key("z_norm")).unwrap());
        assert_eq!(
            norm_check(&ctx, &z).is_ok(),
            fx.bool(&key("z_norm_le_beta")).unwrap()
        );
        let zh = z_hat(&ctx, &z).unwrap();
        let z_prime_rec = fx.polys(&key("z_prime")).unwrap();
        assert_eq!(zh, z_prime_rec[l.z_hat_range()], "{name}: z_hat");

        // The shares; their sums are the recorded z' and y.
        let shares: Vec<LinearShare> = (1..=n)
            .map(|p| LinearShare {
                z_prime: z_prime_share(
                    &ctx,
                    &states[p - 1],
                    &ws[p - 1],
                    &secrets[p - 1],
                    (p == 1).then_some(zh.as_slice()),
                    &u_share(&fx, p),
                )
                .unwrap(),
                y: y_share(&ctx, p, &v, &u_com(&fx)).unwrap(),
            })
            .collect();
        let sum = sum_shares(&ctx, &shares).unwrap();
        assert_eq!(sum.z_prime, z_prime_rec, "{name}: z' = sum of the shares");
        assert_eq!(sum.y, fx.polys(&key("y")).unwrap(), "{name}: y");
        // Party p's share is z' on S_p, <u>_p at mu - 1 and zero elsewhere; y sits at prover 1.
        for (p, share) in (1..=n).zip(&shares) {
            assert_eq!(
                *share,
                LinearShare::slice(l, &z_prime_rec, &sum.y, &u_share(&fx, p), p).unwrap(),
                "{name}: share of p = {p}"
            );
            for pos in 0..l.u {
                if !l.in_s(p, pos) {
                    assert!(share.z_prime[pos].0.iter().all(|x| *x == 0), "{name} p={p}");
                }
            }
        }

        let mut w0 = Vec::new();
        let mut t0 = Vec::new();
        for s in &secrets {
            w0.push(s.images.w0.clone());
            t0.push(s.images.t0.clone());
        }
        let mut t = vec![vec![ring.zero(); l.n_a]; l.n_blocks];
        for st in &states {
            for block in &st.blocks {
                t[block.index] = block.t.clone();
            }
        }
        let u = (1..=n).fold(ring.zero(), |acc, p| {
            ring.add(&acc, &u_share(&fx, p)).unwrap()
        });
        assert_eq!(u, fx.poly("/conv/u_inside_F_AH").unwrap());
        run = Some(Run {
            accepted: a,
            v: v.clone(),
            c: c.clone(),
            shares,
            witness: UnpackedWitness {
                w: w.clone(),
                w_hat: w_hat.clone(),
                w0,
                t0,
                t,
                z,
                u,
            },
            masks: secrets.iter().map(|s| s.mask.clone()).collect(),
            z_hat: zh,
        });
    }
    run.expect("an accepted attempt")
}

#[test]
fn every_ring_to_linear_value_matches_every_fixture() {
    for name in Fixture::TOY {
        replay_and_compare(name);
    }
}

#[test]
fn layout_partition_and_unheld_positions_match() {
    for name in Fixture::TOY {
        let (fx, ctx, _, _) = load(name);
        let l = &ctx.layout;
        for p in 1..=l.n {
            let recorded: Vec<std::ops::Range<usize>> = fx
                .int_rows(&format!("/ring_to_linear/partition/{p}"))
                .unwrap()
                .iter()
                .map(|r| r[0] as usize..r[1] as usize)
                .collect();
            assert_eq!(l.s_ranges(p), recorded, "{name}: S_{p}");
        }
        let unheld: Vec<std::ops::Range<usize>> = fx
            .int_rows("/ring_to_linear/unheld")
            .unwrap_or_default()
            .iter()
            .map(|r| r[0] as usize..r[1] as usize)
            .collect();
        assert_eq!(l.unheld(), unheld, "{name}: unheld");
        for (i, block) in l.named_blocks().iter().enumerate() {
            let pointer = format!("/ring_to_linear/layout/blocks/{i}");
            let rec = fx.get(&pointer).unwrap().as_array().unwrap();
            assert_eq!(rec[0].as_str().unwrap(), block.name);
            assert_eq!(rec[1].as_u64().unwrap() as usize, block.start);
            assert_eq!(rec[2].as_u64().unwrap() as usize, block.len);
        }
        let rows = |k: &str| {
            fx.usize(&format!("/ring_to_linear/layout/rows/{k}"))
                .unwrap()
        };
        assert_eq!(
            [rows("D"), rows("B"), rows("b"), rows("c"), rows("A")],
            [l.row_d, l.row_b, l.row_bg, l.row_c, l.row_a]
        );
    }
}

/// The oracle's relation checks `r2l.*` on this crate's values, and their failure boundary.
#[test]
fn relations_hold_and_fail_when_tampered() {
    for name in Fixture::TOY {
        let run = replay_and_compare(name);
        let (fx, ctx, pv, states) = load(name);
        let l = &ctx.layout;
        let ring = &ctx.ring;
        let u = u_com(&fx);
        let rel_key = |k: &str| fx.bool(&format!("/relations/{}", hachi_co_core::escape_key(k)));

        // r2l.challenges_in_C (every attempt) and the prover-1 norm check.
        for a in 0..attempts(&fx) {
            let c = fx
                .int_rows(&format!("/ring_to_linear/attempts/{a}/c"))
                .unwrap();
            check_challenges(&ctx, &c).unwrap();
        }
        assert!(rel_key("r2l.challenges_in_C").unwrap());
        norm_check(&ctx, &run.witness.z).unwrap();
        assert!(rel_key("r2l.prover1_norm_check_z_le_beta").unwrap());

        // r2l.J_z_hat_eq_z: J applied over the integers to the centered digits gives z back.
        let digits = short::centered(ring, &run.z_hat).unwrap();
        let b = ctx.b() as i64;
        let mut jz = Vec::with_capacity(run.witness.z.len());
        for entry in digits.chunks(l.tau) {
            // entry[theta] holds digit theta of every coefficient of one element of z.
            for j in 0..l.d {
                jz.push(entry.iter().rev().fold(0i64, |x, digit| x * b + digit[j]));
            }
        }
        assert_eq!(digits.len(), l.width_a * l.tau);
        assert_eq!(jz, run.witness.z, "{name}: J z_hat = z");
        assert!(rel_key("r2l.J_z_hat_eq_z").unwrap());

        // r2l.z_eq_sum_c_s_plus_sum_masks, block by block with the core's exact product.
        let mut total = vec![0i64; response_len(&ctx)];
        for st in &states {
            for block in &st.blocks {
                let s = short::centered(ring, &block.s).unwrap();
                let cs = short::negacyclic_mul_rows(&run.c[block.index], &s).unwrap();
                for (acc, x) in total.iter_mut().zip(cs.iter().flatten()) {
                    *acc += x;
                }
            }
        }
        for mask in &run.masks {
            for (acc, x) in total.iter_mut().zip(mask) {
                *acc += x;
            }
        }
        assert_eq!(total, run.witness.z, "{name}: z = sum c_i s_i + sum r_p");
        assert!(rel_key("r2l.z_eq_sum_c_s_plus_sum_masks").unwrap());

        // r2l.M_z'_eq_y: M from the verifier's side, z' from the shares.
        let rel = linear_relation(&ctx, &pv, &run.v, &run.c, &u).unwrap();
        let m_digest = sha256_residues(rel.m.iter().flatten().flat_map(|p| p.0.iter().copied()));
        assert_eq!(
            m_digest,
            fx.str("/ring_to_linear/M_sha256").unwrap(),
            "{name}: M"
        );
        assert_eq!(rel.y, fx.polys("/ring_to_linear/y").unwrap());
        let shape: Vec<usize> = fx
            .ints("/ring_to_linear/M_shape")
            .unwrap()
            .iter()
            .map(|x| *x as usize)
            .collect();
        assert_eq!(shape, [rel.m.len(), rel.m[0].len(), l.d]);
        let nonzero = rel
            .m
            .iter()
            .flatten()
            .filter(|e| e.0.iter().any(|x| *x != 0))
            .count();
        assert_eq!(
            nonzero,
            fx.usize("/ring_to_linear/M_nonzero_ring_entries").unwrap()
        );
        let z_prime = sum_shares(&ctx, &run.shares).unwrap().z_prime;
        assert!(
            relation_holds(&ctx, &rel, &z_prime).unwrap(),
            "{name}: M z' = y"
        );
        assert!(rel_key("r2l.M_z'_eq_y (Sage)").unwrap());

        // The row blocks of 5_Building_Blocks.tex:182 on the unpacked values.
        let rows = row_block_checks(&ctx, &pv, &run.c, &run.witness).unwrap();
        assert!(
            rows.row_b && rows.row_c && rows.row_a && rows.all(),
            "{name}: {rows:?}"
        );
        for k in [
            "r2l.row_b: sum_i b_i w_i = u",
            "r2l.row_c: sum c_i w_i + sum w0_p = a^T G z",
            "r2l.row_A: sum c_i t_i + sum t0_p = A z",
        ] {
            assert!(rel_key(k).unwrap(), "{name}: {k}");
        }

        // Failure boundary. Another attempt's challenges (or one coefficient changed) give
        // another M, for which z' is no solution; so does a changed v.
        if run.accepted > 0 {
            let c_old = fx.int_rows("/ring_to_linear/attempts/0/c").unwrap();
            let rel_old = linear_relation(&ctx, &pv, &run.v, &c_old, &u).unwrap();
            assert!(!relation_holds(&ctx, &rel_old, &z_prime).unwrap());
        }
        let mut c2 = run.c.clone();
        let j = c2[0].iter().position(|x| *x != 0).unwrap_or(0);
        c2[0][j] = if c2[0][j] > 0 {
            c2[0][j] - 1
        } else {
            c2[0][j] + 1
        };
        let rel2 = linear_relation(&ctx, &pv, &run.v, &c2, &u).unwrap();
        assert!(!relation_holds(&ctx, &rel2, &z_prime).unwrap());
        let mut v2 = run.v.clone();
        v2[0].0[0] = (v2[0].0[0] + 1) % ctx.q();
        let rel3 = linear_relation(&ctx, &pv, &v2, &run.c, &u).unwrap();
        let res = residual(&ctx, &rel3, &z_prime).unwrap();
        // Exactly the first D row is off, by -1 in coefficient 0.
        assert_eq!(res[l.row_d].0[0], ctx.q() - 1);
        assert!(res.iter().skip(1).all(|p| p.0.iter().all(|x| *x == 0)));
        // A changed digit of one party's share breaks the relation.
        for p in 1..=l.n {
            let mut shares = run.shares.clone();
            let pos = l.w0_hat_range(p).start;
            shares[p - 1].z_prime[pos].0[0] = (shares[p - 1].z_prime[pos].0[0] + 1) % ctx.q();
            let zs = sum_shares(&ctx, &shares).unwrap().z_prime;
            assert!(!relation_holds(&ctx, &rel, &zs).unwrap(), "{name} p={p}");
        }
        // Each row-block check fails on its own tampered value.
        let mut wit = run.witness.clone();
        wit.u.0[1] = (wit.u.0[1] + 1) % ctx.q();
        let r = row_block_checks(&ctx, &pv, &run.c, &wit).unwrap();
        assert!(!r.row_b && r.row_c && r.row_a);
        let mut wit = run.witness.clone();
        wit.w0[l.n - 1].0[0] = (wit.w0[l.n - 1].0[0] + 1) % ctx.q();
        let r = row_block_checks(&ctx, &pv, &run.c, &wit).unwrap();
        assert!(r.row_b && !r.row_c && r.row_a);
        let mut wit = run.witness.clone();
        wit.t0[0][0].0[3] = (wit.t0[0][0].0[3] + 1) % ctx.q();
        let r = row_block_checks(&ctx, &pv, &run.c, &wit).unwrap();
        assert!(r.row_b && r.row_c && !r.row_a);
        let mut wit = run.witness.clone();
        wit.z[5] += 1;
        let r = row_block_checks(&ctx, &pv, &run.c, &wit).unwrap();
        assert!(r.row_b && !r.row_c && !r.row_a && !r.all());
        // A response one step past prover 1's bound fails the norm check.
        let mut z = run.witness.z.clone();
        z[0] = ctx.beta() as i64 + 1;
        assert_eq!(
            norm_check(&ctx, &z),
            Err(Error::NormCheck {
                norm: ctx.beta() + 1,
                beta: ctx.beta()
            })
        );
        z[0] = -(ctx.beta() as i64);
        norm_check(&ctx, &z).unwrap();
    }
}

/// The replay providers are consistent with the driver's requests: the randomness of an attempt
/// the oracle did not run is refused, and so are wrong parameters.
#[test]
fn replayed_randomness_is_requested_with_the_recorded_parameters() {
    let (fx, ctx, pv, _) = load("toy-n2");
    let mut rng = ReplayRandomness::new(&fx, 2).unwrap();
    let a = attempts(&fx) as u64;
    assert!(matches!(
        sample_attempt(&ctx, &pv, &mut rng, a),
        Err(Error::Core(hachi_co_core::Error::Replay(_)))
    ));
    let mask = PartyRandomness::<K>::mask(&mut rng, 0, ctx.gamma(), response_len(&ctx)).unwrap();
    assert_eq!(
        mask,
        fx.ints("/ring_to_linear/attempts/0/provers/1/mask")
            .unwrap()
    );
}
