//! The linear check constrains every column of the next witness: with random public data and
//! random $\tau_1, \alpha$, the weight $m^\ast(u)$ of every non-padding column is nonzero and every
//! padding column has weight zero, for decomposed and packed layouts. Completeness alone would not
//! catch a row that the prover and the verifier both forgot.

use hachi::commit::Ctx;
use hachi::mle::{eq_table, powers};
use hachi::params::{HachiChoices, Layout, Schedule};
use hachi::relation::{Columns, Public, mstar};
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
    let choices = HachiChoices {
        fixed_iterations: Some(1),
        ..HachiChoices::toy(16)
    };
    let ctx = Ctx::<K>::new(Q, choices.clone()).unwrap();
    let d = ctx.d();
    for decomposed in [true, false] {
        for (r, m) in [(0usize, 3usize), (2, 2), (3, 1)] {
            let layout = Layout {
                r,
                m,
                decomposed,
                beta_s: if decomposed { 2 } else { 4 },
            };
            let schedule = Schedule::derive(&choices, Q, layout).unwrap();
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
            let public = Public {
                params: p,
                b: &b,
                a: &a,
                target: &target,
                c: &c,
                d_mat: &d_mat,
                b_mat: &b_mat,
                a_mat: &a_mat,
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
            let cols = Columns::of(p, choices.n_a, ctx.alpha());
            let used = cols.q + p.rows * delta;
            for (col, w) in weights.iter().enumerate() {
                if col < used {
                    assert!(
                        !field.is_zero(w),
                        "decomposed {decomposed}, r {r}, m {m}: column {col} of {used} has weight 0"
                    );
                } else {
                    assert!(field.is_zero(w), "padding column {col} has a weight");
                }
            }
        }
    }
}
