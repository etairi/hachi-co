//! The linear check constrains every column of the next witness: with random public data and
//! random $\tau_1, \alpha$, the weight $m^\ast(u)$ of every non-padding column is nonzero and every
//! padding column has weight zero, for decomposed and packed layouts, without hiding and with
//! randomized commitments (the columns of $\hat{\boldsymbol\rho}$ and $\hat{\mathbf r}_z$, for
//! both distributions and $\nu = 1, 2$). Completeness alone would not catch a row that the prover
//! and the verifier both forgot.
//!
//! With randomized commitments, on an honest witness, every row vanishes modulo $X^d + 1$ and the
//! linear sumcheck's identity $\sum_u m^\ast(u) P_u(\alpha) = \sum_i \widetilde{eq}(\tau_1, i)
//! y_i(\alpha)$ holds on the next witness's table; changing $\boldsymbol\rho$ or $\mathbf r_z$ by
//! one makes a row fail, and changing a digit of $\hat{\boldsymbol\rho}$ or $\hat{\mathbf r}_z$
//! in the table breaks the identity.

use hachi::Error;
use hachi::commit::{Ctx, commit};
use hachi::iteration::{commitment_randomness, evaluate_claim};
use hachi::mle::{eq_table, powers};
use hachi::params::{HachiChoices, Hiding, Layout, Noise, Schedule};
use hachi::relation::{Columns, Public, mstar, next_table, quotients};
use hachi_co_math::decompose::{g_inv, j_inv};
use hachi_co_math::{Ext, Poly};
use hachi_co_sampling::ShakePrg;
use hachi_co_sampling::uniform::uniform_mod_q;

const K: usize = 4;
const Q: u64 = 8388637;

fn polys(tag: &str, n: usize, d: usize) -> Vec<Poly> {
    let r = uniform_mod_q(&mut ShakePrg::new(&[5u8; 32], tag.as_bytes(), 0), Q, n * d).unwrap();
    r.chunks_exact(d).map(|c| Poly(c.to_vec())).collect()
}

#[test]
fn every_witness_column_has_a_nonzero_weight() {
    let base = HachiChoices {
        fixed_iterations: Some(1),
        ..HachiChoices::toy(16)
    };
    let digits = Hiding {
        nu: 2,
        noise: Noise::Digits,
    };
    for choices in [
        base.clone(),
        HachiChoices {
            fixed_iterations: Some(1),
            ..HachiChoices::toy_hiding(16)
        },
        HachiChoices {
            hiding: Some(digits),
            ..base.clone()
        },
    ] {
        check_weights(&choices);
    }
}

fn check_weights(choices: &HachiChoices) {
    let ctx = Ctx::<K>::new(Q, choices.clone()).unwrap();
    let d = ctx.d();
    let nu = choices.hiding.map_or(0, |h| h.nu);
    for decomposed in [true, false] {
        for (r, m) in [(0usize, 3usize), (2, 2), (3, 1)] {
            let layout = Layout {
                r,
                m,
                decomposed,
                beta_s: if decomposed { 2 } else { 4 },
            };
            let schedule = Schedule::fixed(choices, Q, layout, 1).unwrap();
            let p = &schedule.iterations[0];
            let (chunks, delta) = (layout.chunks(), p.delta);
            let width = layout.width(delta);
            let b = polys("b", chunks, d);
            let a = polys("a", layout.chunk_len(), d);
            let c = polys("c", chunks, d);
            let target = polys("T", 1, d).remove(0);
            let d_mat = vec![polys("D", chunks * delta, d)];
            let b_mat = vec![polys("B", chunks * delta, d)];
            let a_mat = vec![polys("A", width, d)];
            let v = polys("v", 1, d);
            let u = polys("u", 1, d);
            let (a_rand, d_rand) = if nu == 0 {
                (Vec::new(), Vec::new())
            } else {
                (vec![polys("A'", nu, d)], vec![polys("D'", nu, d)])
            };
            let public = Public {
                params: p,
                b: &b,
                a: &a,
                target: &target,
                c: &c,
                d_mat: &d_mat,
                b_mat: &b_mat,
                a_mat: &a_mat,
                a_rand: &a_rand,
                d_rand: &d_rand,
                v: &v,
                u: &u,
            };
            let field = ctx.field();
            let coins: Vec<Ext<K>> =
                uniform_mod_q(&mut ShakePrg::new(&[6u8; 32], b"coins", 0), Q, 9 * K)
                    .unwrap()
                    .as_chunks::<K>()
                    .0
                    .iter()
                    .map(|c| Ext(*c))
                    .collect();
            let eq1 = eq_table(field, &coins[1..4]);
            let alpha_pows = powers(field, &coins[0], d + 1);
            let (weights, _) = mstar(&ctx, &public, &eq1, &alpha_pows).unwrap();
            let cols = Columns::of(p, choices.n_a, ctx.alpha(), ctx.rand_len(choices.n_d));
            let used = cols.q + p.rows * delta;
            if nu > 0 {
                assert!(cols.z < cols.rho && cols.rho < cols.r && cols.r < cols.q);
                assert_eq!(
                    cols.q - cols.rho,
                    (nu + choices.n_d) * p.tau_rho + (nu + choices.n_a) * p.tau_r
                );
            } else {
                assert_eq!((cols.rho, cols.r), (cols.q, cols.q));
            }
            for (col, w) in weights.iter().enumerate() {
                if col < used {
                    assert!(
                        !field.is_zero(w),
                        "hiding {:?}, decomposed {decomposed}, r {r}, m {m}: column {col} of {used} has \
                         weight 0",
                        choices.hiding
                    );
                } else {
                    assert!(field.is_zero(w), "padding column {col} has a weight");
                }
            }
        }
    }
}

#[test]
fn the_randomness_columns_are_constrained() {
    let digits = Hiding {
        nu: 2,
        noise: Noise::Digits,
    };
    for choices in [
        HachiChoices {
            fixed_iterations: Some(1),
            ..HachiChoices::toy_hiding(16)
        },
        HachiChoices {
            fixed_iterations: Some(1),
            hiding: Some(digits),
            ..HachiChoices::toy(16)
        },
    ] {
        check_constrained(&choices);
    }
}

fn check_constrained(choices: &HachiChoices) {
    let ctx = Ctx::<K>::new(Q, choices.clone()).unwrap();
    let (ring, field, d) = (&ctx.ring, ctx.field(), ctx.d());
    let zq = ring.zq();
    let (n_a, n_b, n_d, b) = (choices.n_a, choices.n_b, choices.n_d, choices.b);
    let layout = Layout {
        r: 2,
        m: 2,
        decomposed: true,
        beta_s: 2,
    };
    let schedule = Schedule::fixed(choices, Q, layout, 1).unwrap();
    let p = &schedule.iterations[0];
    let (chunks, delta) = (layout.chunks(), p.delta);
    // An honest witness: a message committed with randomness.
    let message: Vec<Vec<Poly>> = (0..chunks)
        .map(|i| polys(&format!("f{i}"), layout.chunk_len(), d))
        .collect();
    let r = commitment_randomness(&ctx, &layout, &[9u8; 32], b"test/r", 0).unwrap();
    let witness = commit(&ctx, 0, layout, message.clone(), r).unwrap();
    // A claim, v with its randomness, and challenges c_i = X^i (l1 norm 1 <= omega).
    let a = polys("a", layout.chunk_len(), d);
    let bv = polys("b", chunks, d);
    let target = evaluate_claim(&ctx, &bv, &a, &message).unwrap();
    let w: Vec<Poly> = message.iter().map(|f| ring.inner(&a, f).unwrap()).collect();
    let w_hat = g_inv(ring, b, delta, &w).unwrap();
    let d_mat = ctx.matrix(b"D", 0, n_d, chunks * delta).unwrap();
    let (a_rand, d_rand) = (
        ctx.rand_matrix(b"A'", 0, n_a).unwrap(),
        ctx.rand_matrix(b"D'", 0, n_d).unwrap(),
    );
    let rho = ctx
        .noise(
            &mut ShakePrg::new(&[10u8; 32], b"test/rho", 0),
            ctx.rand_len(n_d),
        )
        .unwrap();
    let v: Vec<Poly> = ring
        .mat_vec(&d_mat, &w_hat)
        .unwrap()
        .iter()
        .zip(&ctx.rand_term(&d_rand, &rho).unwrap())
        .map(|(x, y)| ring.add(x, y).unwrap())
        .collect();
    let c: Vec<Poly> = (0..chunks).map(|i| ring.monomial(i as i64)).collect();
    let fold = |parts: &[Vec<Poly>]| -> Vec<Poly> {
        let mut out = vec![ring.zero(); parts[0].len()];
        for (ci, part) in c.iter().zip(parts) {
            for (o, x) in out.iter_mut().zip(part) {
                *o = ring.add(o, &ring.mul(ci, x).unwrap()).unwrap();
            }
        }
        out
    };
    let (z, r_z) = (fold(&witness.s), fold(&witness.r));
    let b_mat = ctx.matrix(b"B", 0, n_b, chunks * n_a * delta).unwrap();
    let a_mat = ctx.matrix(b"A", 0, n_a, layout.width(delta)).unwrap();
    let public = Public {
        params: p,
        b: &bv,
        a: &a,
        target: &target,
        c: &c,
        d_mat: &d_mat,
        b_mat: &b_mat,
        a_mat: &a_mat,
        a_rand: &a_rand,
        d_rand: &d_rand,
        v: &v,
        u: &witness.u,
    };
    let q = |rho: &[Poly], r_z: &[Poly]| {
        quotients(
            &ctx,
            &public,
            &w,
            &w_hat,
            &witness.t,
            &witness.t_hat,
            &z,
            rho,
            r_z,
        )
    };
    // Honest: every row vanishes. Changing rho or r_z by one (in the A1' or D1' part and in the
    // identity part) makes a row fail.
    let quot = q(&rho, &r_z).unwrap();
    let nu = choices.hiding.unwrap().nu;
    for entry in [0, nu] {
        let mut bad = rho.clone();
        bad[entry].0[0] = zq.add(bad[entry].0[0], 1);
        assert!(
            matches!(q(&bad, &r_z), Err(Error::Protocol(_))),
            "rho, entry {entry}"
        );
        let mut bad = r_z.clone();
        bad[entry].0[0] = zq.add(bad[entry].0[0], 1);
        assert!(
            matches!(q(&rho, &bad), Err(Error::Protocol(_))),
            "r_z, entry {entry}"
        );
    }
    // The table of the honest witness satisfies the linear sumcheck's identity.
    let mut z_prime = w_hat.clone();
    z_prime.extend(witness.t_hat.iter().cloned());
    z_prime.extend(j_inv(ring, b, p.tau, &z).unwrap());
    z_prime.extend(j_inv(ring, b, p.tau_rho, &rho).unwrap());
    z_prime.extend(j_inv(ring, b, p.tau_r, &r_z).unwrap());
    let table = next_table(&ctx, p, &z_prime, &quot).unwrap();
    let lr = p.rows.next_power_of_two().trailing_zeros() as usize;
    let coins: Vec<Ext<K>> = uniform_mod_q(
        &mut ShakePrg::new(&[11u8; 32], b"coins", 1),
        Q,
        (1 + lr) * K,
    )
    .unwrap()
    .as_chunks::<K>()
    .0
    .iter()
    .map(|c| Ext(*c))
    .collect();
    let alpha_pows = powers(field, &coins[0], d + 1);
    let (m, sum) = mstar(&ctx, &public, &eq_table(field, &coins[1..]), &alpha_pows).unwrap();
    let lhs = |table: &[u64]| {
        m.iter().enumerate().fold(field.zero(), |acc, (u, mu)| {
            let at_alpha = (0..d).fold(field.zero(), |acc, l| {
                field.add(&acc, &field.scale(&alpha_pows[l], table[u * d + l]))
            });
            field.add(&acc, &field.mul(mu, &at_alpha))
        })
    };
    assert_eq!(
        lhs(&table),
        sum,
        "{:?}: the honest identity",
        choices.hiding
    );
    // A digit of rho-hat or r_z-hat changed in the table breaks it.
    let cols = Columns::of(p, n_a, ctx.alpha(), ctx.rand_len(n_d));
    for col in [cols.rho, cols.r, cols.q - 1] {
        let mut bad = table.clone();
        bad[col * d] = zq.add(bad[col * d], 1);
        assert_ne!(lhs(&bad), sum, "{:?}: column {col}", choices.hiding);
    }
}
