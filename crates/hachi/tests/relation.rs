//! The linear check constrains every column of the next witness: with random public data and
//! random $\tau_1, \alpha$, the weight $m^\ast(u)$ of every non-padding column is nonzero and every
//! padding column has weight zero, for decomposed and packed layouts, without masks and at a masked
//! level (the columns of $\hat w_0$, $\hat{\mathbf t}_0$ and $\hat{\mathbf z}_\varrho$, for
//! $\nu = 1, 2$). Completeness alone would not catch a row that the prover and the verifier both
//! forgot.
//!
//! At a masked level, on an honest witness, every row vanishes modulo $X^d + 1$ and the linear
//! sumcheck's identity $\sum_u m^\ast(u) P_u(\alpha) = \sum_i \widetilde{eq}(\tau_1, i)
//! y_i(\alpha)$ holds on the next witness's table. Changing $w_0$, $\mathbf t_0$ or $\mathbf
//! z_\varrho$ by one makes a row fail, and changing a digit of any masked part in the table breaks
//! the identity.

use hachi::Error;
use hachi::commit::{Committed, Ctx, commit};
use hachi::iteration::{commitment_randomness, evaluate_claim};
use hachi::mle::{eq_table, powers};
use hachi::params::{HachiChoices, Hiding, Layout, Schedule};
use hachi::relation::{Columns, Masked, Public, mstar, next_table, quotients};
use hachi_co_math::decompose::{g_apply, g_inv, j_inv};
use hachi_co_math::{Ext, Poly};
use hachi_co_sampling::ShakePrg;
use hachi_co_sampling::uniform::uniform_mod_q;

const K: usize = 4;
const Q: u64 = 8388637;

fn polys(tag: &str, n: usize, d: usize) -> Vec<Poly> {
    let r = uniform_mod_q(&mut ShakePrg::new(&[5u8; 32], tag.as_bytes(), 0), Q, n * d).unwrap();
    r.chunks_exact(d).map(|c| Poly(c.to_vec())).collect()
}

/// The masks of the toy choices with rank `nu` and height `n_a` at level 0.
fn masked(nu: usize, n_a: usize) -> HachiChoices {
    HachiChoices {
        fixed_iterations: Some(1),
        hiding: Some(Hiding {
            nu,
            n_a,
            lambda: 128,
            phi: 13,
        }),
        ..HachiChoices::toy(16)
    }
}

#[test]
fn every_witness_column_has_a_nonzero_weight() {
    let plain = HachiChoices {
        fixed_iterations: Some(1),
        ..HachiChoices::toy(16)
    };
    for choices in [plain, masked(1, 1), masked(2, 1), masked(1, 2)] {
        check_weights(&choices);
    }
}

fn check_weights(choices: &HachiChoices) {
    let ctx = Ctx::<K>::new(Q, choices.clone()).unwrap();
    let d = ctx.d();
    let nu = choices.hiding.map_or(0, |h| h.nu);
    let is_masked = choices.hiding.is_some();
    for decomposed in [true, false] {
        // A masked witness is the input, so it is decomposed.
        if is_masked && !decomposed {
            continue;
        }
        for (r, m) in [(0usize, 3usize), (2, 2), (3, 1)] {
            let layout = Layout {
                r,
                m,
                decomposed,
                beta_s: if decomposed { 2 } else { 4 },
                masked: is_masked,
                extra: 0,
            };
            let schedule = Schedule::fixed(choices, Q, layout, 1).unwrap();
            let p = &schedule.iterations[0];
            let (chunks, delta, n_a) = (layout.chunks(), p.delta, p.n_a);
            let width = layout.width(delta);
            let b = polys("b", chunks, d);
            let a = polys("a", layout.chunk_len(), d);
            let c = polys("c", chunks, d);
            let target = polys("T", 1, d).remove(0);
            let d_mat = vec![polys("D", p.d_cols(), d)];
            let b_mat = vec![polys("B", chunks * n_a * delta, d)];
            let a_mat: Vec<Vec<Poly>> = (0..n_a)
                .map(|i| polys(&format!("A{i}"), width, d))
                .collect();
            let v = polys("v", 1, d);
            let u = polys("u", 1, d);
            let a_rand: Vec<Vec<Poly>> = if is_masked {
                (0..n_a).map(|i| polys(&format!("A'{i}"), nu, d)).collect()
            } else {
                Vec::new()
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
            let cols = Columns::of(p, ctx.alpha());
            let used = cols.q + p.rows * delta;
            if is_masked {
                assert_eq!(cols.w0, chunks * delta);
                assert_eq!(cols.t0, cols.w0 + delta);
                assert_eq!(cols.t, cols.t0 + n_a * delta);
                assert_eq!(cols.q - cols.r, (nu + n_a) * p.tau);
            } else {
                assert_eq!((cols.w0, cols.t0), (cols.t, cols.t));
                assert_eq!(cols.r, cols.q);
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
fn the_masked_columns_are_constrained() {
    for choices in [masked(1, 1), masked(2, 1), masked(1, 2)] {
        check_constrained(&choices);
    }
}

fn check_constrained(choices: &HachiChoices) {
    let ctx = Ctx::<K>::new(Q, choices.clone()).unwrap();
    let (ring, field, d) = (&ctx.ring, ctx.field(), ctx.d());
    let zq = ring.zq();
    let (n_b, n_d, b) = (choices.n_b, choices.n_d, choices.b);
    let layout = Layout {
        r: 2,
        m: 2,
        decomposed: true,
        beta_s: 2,
        masked: true,
        extra: 0,
    };
    let schedule = Schedule::fixed(choices, Q, layout, 1).unwrap();
    let p = &schedule.iterations[0];
    let (chunks, delta, n_a) = (layout.chunks(), p.delta, p.n_a);
    let width = layout.width(delta);
    // An honest witness: a message and a masking row per chunk, committed with randomness.
    let message: Vec<Vec<Poly>> = (0..chunks)
        .map(|i| polys(&format!("f{i}"), layout.chunk_len(), d))
        .collect();
    let r = commitment_randomness(&ctx, &layout, &[9u8; 32], b"test/r").unwrap();
    let witness = commit(&ctx, 0, layout, message.clone(), r).unwrap();
    // A claim on (a, x*), and challenges c_i = X^i (l1 norm 1 <= omega).
    let a = polys("a", layout.chunk_len(), d);
    let bv = polys("b", chunks, d);
    let target = evaluate_claim(&ctx, &bv, &a, &message).unwrap();
    let w: Vec<Poly> = message.iter().map(|f| ring.inner(&a, f).unwrap()).collect();
    let w_hat = g_inv(ring, b, delta, &w).unwrap();
    let a_mat = ctx.matrix(b"A", 0, n_a, width).unwrap();
    let a_rand = ctx.rand_matrix().unwrap();
    // Masks. Their distribution does not matter to the relation, so they are short here.
    let rand_len = ctx.rand_len().unwrap();
    let mut prg = ShakePrg::new(&[10u8; 32], b"test/masks", 0);
    let s0 = ctx.short(&mut prg, width).unwrap();
    let r0 = ctx.short(&mut prg, rand_len).unwrap();
    let w0 = ring
        .inner(&a, &g_apply(ring, b, delta, &s0).unwrap())
        .unwrap();
    let t0: Vec<Poly> = ring
        .mat_vec(&a_mat, &s0)
        .unwrap()
        .iter()
        .zip(&ctx.rand_term(&a_rand, &r0).unwrap())
        .map(|(x, y)| ring.add(x, y).unwrap())
        .collect();
    let w0_hat = g_inv(ring, b, delta, core::slice::from_ref(&w0)).unwrap();
    let t0_hat = g_inv(ring, b, delta, &t0).unwrap();
    let d_mat = ctx.matrix(b"D", 0, n_d, p.d_cols()).unwrap();
    let mut columns = w_hat.clone();
    columns.extend_from_slice(&w0_hat);
    columns.extend_from_slice(&t0_hat);
    let v = ring.mat_vec(&d_mat, &columns).unwrap();
    let c: Vec<Poly> = (0..chunks).map(|i| ring.monomial(i as i64)).collect();
    let fold = |parts: &[Vec<Poly>], mask: &[Poly]| -> Vec<Poly> {
        let mut out = mask.to_vec();
        for (ci, part) in c.iter().zip(parts) {
            for (o, x) in out.iter_mut().zip(part) {
                *o = ring.add(o, &ring.mul(ci, x).unwrap()).unwrap();
            }
        }
        out
    };
    let s: Vec<Vec<Poly>> = witness
        .message
        .iter()
        .map(|chunk| {
            Committed::digits(&ctx, &witness.layout, delta, chunk)
                .unwrap()
                .into_owned()
        })
        .collect();
    let (z, z_r) = (fold(&s, &s0), fold(&witness.r, &r0));
    let b_mat = ctx.matrix(b"B", 0, n_b, chunks * n_a * delta).unwrap();
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
        v: &v,
        u: &witness.u,
    };
    let q = |w0: &Poly, t0: &[Poly], z_r: &[Poly]| {
        quotients(
            &ctx,
            &public,
            &w,
            &w_hat,
            &witness.t,
            &witness.t_hat,
            &z,
            Some(&Masked {
                w0,
                w0_hat: &w0_hat,
                t0,
                t0_hat: &t0_hat,
                z_r,
            }),
        )
    };
    // Honest: every row vanishes. Changing a masked part by one makes a row fail.
    let quot = q(&w0, &t0, &z_r).unwrap();
    let bump = |p: &Poly| {
        let mut p = p.clone();
        p.0[0] = zq.add(p.0[0], 1);
        p
    };
    assert!(matches!(q(&bump(&w0), &t0, &z_r), Err(Error::Protocol(_))));
    let nu = choices.hiding.unwrap().nu;
    for i in 0..n_a {
        let mut bad = t0.clone();
        bad[i] = bump(&bad[i]);
        assert!(
            matches!(q(&w0, &bad, &z_r), Err(Error::Protocol(_))),
            "t0 {i}"
        );
    }
    for entry in [0, nu] {
        let mut bad = z_r.clone();
        bad[entry] = bump(&bad[entry]);
        assert!(
            matches!(q(&w0, &t0, &bad), Err(Error::Protocol(_))),
            "z_varrho, entry {entry}"
        );
    }
    // A level without masks refuses them, and a masked level needs them.
    assert!(matches!(
        quotients(
            &ctx,
            &public,
            &w,
            &w_hat,
            &witness.t,
            &witness.t_hat,
            &z,
            None
        ),
        Err(Error::Shape(_))
    ));
    // The table of the honest witness satisfies the linear sumcheck's identity.
    let mut z_prime = w_hat.clone();
    z_prime.extend_from_slice(&w0_hat);
    z_prime.extend_from_slice(&t0_hat);
    z_prime.extend(witness.t_hat.iter().cloned());
    z_prime.extend(j_inv(ring, b, p.tau, &z).unwrap());
    z_prime.extend(j_inv(ring, b, p.tau, &z_r).unwrap());
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
    // A digit of any part changed in the table breaks it.
    let cols = Columns::of(p, ctx.alpha());
    for col in [0, cols.w0, cols.t0, cols.t, cols.z, cols.r, cols.q - 1] {
        let mut bad = table.clone();
        bad[col * d] = zq.add(bad[col * d], 1);
        assert_ne!(lhs(&bad), sum, "{:?}: column {col}", choices.hiding);
    }
}
