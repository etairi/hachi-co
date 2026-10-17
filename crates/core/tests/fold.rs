//! The shared helpers and data types on the oracle's data: the commit states (`PartyState`), each
//! prover's fold of 5_Building_Blocks.tex:128 computed with `short::negacyclic_mul` on the
//! centered digits, its norm, Rej's decision, the responses z_i and their sum z, and the flat
//! encoding of a response for the gather.
#![cfg(feature = "fixtures")]

use hachi_co_core::wire::{flatten_rows, unflatten_rows};
use hachi_co_core::{
    BlockState, Context, Error, Fixture, PartyRandomness, PartyState, ReplayRandomness, short,
};
use hachi_co_sampling::rej;

const K: usize = 4;

/// Party p's state after Commit, as recorded by the oracle.
fn state(fx: &Fixture, ctx: &Context<K>, p: usize) -> PartyState<K> {
    let l = &ctx.layout;
    let f_prime = fx.polys("/commit/f_prime").unwrap();
    let recorded = fx.get("/commit/blocks").unwrap().as_array().unwrap().len();
    let blocks = l
        .blocks_of(p)
        .map(|i| {
            let j = (0..recorded)
                .find(|j| fx.usize(&format!("/commit/blocks/{j}/index")).unwrap() == i)
                .expect("every owned block is recorded");
            let key = |k: &str| fx.polys(&format!("/commit/blocks/{j}/{k}")).unwrap();
            BlockState {
                index: i,
                f_prime: f_prime[i * l.block_len..(i + 1) * l.block_len].to_vec(),
                s: key("s"),
                t: key("t"),
                t_hat: key("t_hat"),
            }
        })
        .collect();
    PartyState {
        party: p,
        share: fx.exts(&format!("/commit/f_shares/{}", p - 1)).unwrap(),
        blocks,
    }
}

#[test]
fn folds_norms_rej_and_responses_match_every_fixture() {
    for name in Fixture::TOY {
        let fx = Fixture::load(name).unwrap();
        let ctx = fx.context::<K>().unwrap();
        let l = &ctx.layout;
        let (gamma, beta1) = (ctx.gamma(), ctx.beta1());
        let states: Vec<PartyState<K>> = (1..=l.n).map(|p| state(&fx, &ctx, p)).collect();
        for st in &states {
            st.validate(l).unwrap();
        }
        let attempts = fx
            .get("/ring_to_linear/attempts")
            .unwrap()
            .as_array()
            .unwrap()
            .len();
        let accepted = fx.usize("/ring_to_linear/accepted_attempt").unwrap();
        let mut z_sum = vec![0i64; l.width_a * l.d];
        for a in 0..attempts {
            let c = fx
                .int_rows(&format!("/ring_to_linear/attempts/{a}/c"))
                .unwrap();
            for (st, p) in states.iter().zip(1..) {
                let prefix = format!("/ring_to_linear/attempts/{a}/provers/{}", p - 1);
                // fold = sum over p's blocks of c_i s_i over Z[X]/(X^d + 1).
                let mut fold = vec![vec![0i64; l.d]; l.width_a];
                for block in &st.blocks {
                    let s = short::centered(&ctx.ring, &block.s).unwrap();
                    assert!(short::inf_norm_rows(&s) <= ctx.b() / 2);
                    let cs = short::negacyclic_mul_rows(&c[block.index], &s).unwrap();
                    for (acc, row) in fold.iter_mut().zip(&cs) {
                        for (x, y) in acc.iter_mut().zip(row) {
                            *x += y;
                        }
                    }
                }
                let fold_norm = short::inf_norm_rows(&fold);
                assert_eq!(
                    fold_norm,
                    fx.u64(&format!("{prefix}/fold_norm")).unwrap(),
                    "{name}"
                );
                assert!(fold_norm <= beta1, "the bound of 5_Building_Blocks.tex:129");
                let mut rng = ReplayRandomness::new(&fx, p).unwrap();
                let mask =
                    PartyRandomness::<K>::mask(&mut rng, a as u64, gamma, l.width_a * l.d).unwrap();
                let fold = flatten_rows(&fold);
                let z: Vec<i64> = fold.iter().zip(&mask).map(|(x, r)| x + r).collect();
                assert_eq!(
                    short::inf_norm(&z),
                    fx.u64(&format!("{prefix}/z_norm")).unwrap()
                );
                let accept = fx.bool(&format!("{prefix}/accept")).unwrap();
                assert_eq!(
                    short::inf_norm(&z) <= gamma - beta1,
                    accept,
                    "{name} attempt {a} p {p}"
                );
                let decided = rej::rej(&fold, gamma, beta1, &mask).unwrap();
                assert_eq!(decided.is_some(), accept);
                if a == accepted {
                    assert!(accept);
                    assert_eq!(
                        z,
                        fx.ints(&format!("/ring_to_linear/z_i/{}", p - 1)).unwrap()
                    );
                    for (s, x) in z_sum.iter_mut().zip(&z) {
                        *s += x;
                    }
                }
            }
        }
        assert_eq!(z_sum, fx.ints("/ring_to_linear/z").unwrap(), "{name}");
        assert_eq!(
            short::inf_norm(&z_sum),
            fx.u64("/ring_to_linear/z_norm").unwrap()
        );
        assert!(short::inf_norm(&z_sum) <= ctx.beta());
        let rows = unflatten_rows(&z_sum, l.width_a, l.d).unwrap();
        assert_eq!(rows, fx.int_rows("/ring_to_linear/z").unwrap());
    }
}

#[test]
fn state_validation_catches_shape_errors() {
    let fx = Fixture::load("toy-n3").unwrap();
    let ctx = fx.context::<K>().unwrap();
    let l = &ctx.layout;
    let good = state(&fx, &ctx, 2);
    good.validate(l).unwrap();
    let mut bad = good.clone();
    bad.party = 4;
    assert_eq!(bad.validate(l), Err(Error::InvalidParty { party: 4, n: 3 }));
    let mut bad = good.clone();
    bad.party = 1; // holds blocks 0, 1, not 2, 3
    assert!(matches!(bad.validate(l), Err(Error::Shape { .. })));
    let mut bad = good.clone();
    bad.blocks.swap(0, 1);
    assert!(matches!(bad.validate(l), Err(Error::Shape { .. })));
    let mut bad = good.clone();
    bad.blocks.pop();
    assert!(matches!(bad.validate(l), Err(Error::Shape { .. })));
    let mut bad = good.clone();
    bad.share.pop();
    assert!(matches!(
        bad.validate(l),
        Err(Error::Shape { what: "<f>_p", .. })
    ));
    let mut bad = good.clone();
    bad.blocks[1].s.pop();
    assert!(matches!(
        bad.validate(l),
        Err(Error::Shape { what: "s_i", .. })
    ));
    let mut bad = good.clone();
    bad.blocks[0].t_hat[0].0.push(0);
    assert!(matches!(
        bad.validate(l),
        Err(Error::Shape {
            what: "t_hat_i",
            ..
        })
    ));
    let mut bad = good.blocks[0].clone();
    bad.index = l.n_blocks;
    assert!(matches!(bad.validate(l), Err(Error::Shape { .. })));
    // Party 3 of toy-n3 holds blocks 4 and 5; blocks 6 and 7 belong to nobody.
    assert_eq!(
        state(&fx, &ctx, 3)
            .blocks
            .iter()
            .map(|b| b.index)
            .collect::<Vec<_>>(),
        [4, 5]
    );
}
