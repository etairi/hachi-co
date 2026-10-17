//! `Layout` against the oracle's fixtures (offsets, rows, partition, unheld positions, owners,
//! hypercube sizes) and against its own definition (brute force), plus the failure boundary of
//! its invariants.

use hachi_co_core::{Error, Layout};
use hachi_co_params::{Params, derive, presets::toy_choices};

fn toy(n: u64) -> Params {
    derive(&toy_choices(n)).expect("toy presets derive").0
}

/// The partition, the ranges and the ownership agree with their definitions, position by position
/// (`ref/hachico_ref/r2l.py`, `Layout.partition`, reimplemented here independently).
#[test]
fn partition_matches_a_brute_force_reimplementation() {
    for n in 1..=8u64 {
        let l = Layout::new(&toy(n)).unwrap();
        let n = n as usize;
        let mut owner = vec![0usize; l.mu - 1]; // 0 = unheld
        for i in 0..l.n_blocks {
            // The oracle's rule: (i >> (r - ceil(log n))) + 1 if <= n.
            let p = (i >> (l.r - l.log_n)) + 1;
            let p = if p <= n { p } else { 0 };
            assert_eq!(l.owner_of_block(i), (p != 0).then_some(p), "block {i}");
            for e in 0..l.delta {
                owner[l.w + i * l.delta + e] = p;
            }
            for e in 0..l.n_a * l.delta {
                owner[l.t + i * l.n_a * l.delta + e] = p;
            }
        }
        for p in 1..=n {
            for e in 0..l.delta {
                owner[l.w0 + (p - 1) * l.delta + e] = p;
            }
            for e in 0..l.n_a * l.delta {
                owner[l.t0 + (p - 1) * l.n_a * l.delta + e] = p;
            }
            for e in 0..l.m_rho {
                owner[l.rho + (p - 1) * l.m_rho + e] = p;
            }
        }
        owner[l.z..l.u].fill(1);
        for p in 1..=n {
            let ranges = l.s_ranges(p);
            for w in ranges.windows(2) {
                assert!(
                    w[0].end < w[1].start,
                    "sorted, disjoint, merged: {ranges:?}"
                );
            }
            for (pos, o) in owner.iter().enumerate() {
                let in_ranges = ranges.iter().any(|r| r.contains(&pos));
                assert_eq!(in_ranges, *o == p, "n = {n}, p = {p}, pos = {pos}");
                assert_eq!(
                    l.in_s(p, pos),
                    *o == p,
                    "in_s n = {n}, p = {p}, pos = {pos}"
                );
            }
            let blocks = l.blocks_of(p);
            assert_eq!(blocks.len(), l.blocks_per_party);
            assert!(blocks.clone().all(|i| l.owner_of_block(i) == Some(p)));
        }
        let unheld = l.unheld();
        for (pos, o) in owner.iter().enumerate() {
            assert_eq!(
                unheld.iter().any(|r| r.contains(&pos)),
                *o == 0,
                "pos {pos}"
            );
        }
        assert_eq!(unheld.is_empty(), n.is_power_of_two(), "n = {n}");
        // No S_p contains u = mu - 1, and invalid parties hold nothing.
        for p in 0..=n + 1 {
            assert!(!l.in_s(p, l.u));
        }
        assert!(l.s_ranges(0).is_empty() && l.s_ranges(n + 1).is_empty());
        assert!(l.blocks_of(0).is_empty() && l.blocks_of(n + 1).is_empty());
        assert_eq!(l.owner_of_block(l.n_blocks), None);
    }
}

#[test]
fn ranges_follow_the_element_major_layout() {
    for n in 1..=3u64 {
        let l = Layout::new(&toy(n)).unwrap();
        let n = n as usize;
        for i in 0..l.n_blocks {
            assert_eq!(l.w_hat_range(i), l.w + i * l.delta..l.w + (i + 1) * l.delta);
            let t = l.t_hat_range(i);
            assert_eq!(t.len(), l.n_a * l.delta);
            assert_eq!(t.start, l.t + i * l.n_a * l.delta);
        }
        assert!(l.w_hat_range(l.n_blocks).is_empty() && l.t_hat_range(l.n_blocks).is_empty());
        for p in 1..=n {
            assert_eq!(l.w0_hat_range(p).len(), l.delta);
            assert_eq!(l.t0_hat_range(p).len(), l.n_a * l.delta);
            assert_eq!(l.rho_range(p).len(), l.m_rho);
        }
        assert_eq!(l.w0_hat_range(n).end, l.t0);
        assert_eq!(l.t0_hat_range(n).end, l.rho);
        assert_eq!(l.rho_range(n).end, l.t);
        assert_eq!(l.t_hat_range(l.n_blocks - 1).end, l.z);
        assert_eq!(l.z_hat_range().len(), l.width_a * l.tau);
        for p in [0, n + 1] {
            assert!(l.w0_hat_range(p).is_empty());
            assert!(l.t0_hat_range(p).is_empty());
            assert!(l.rho_range(p).is_empty());
        }
        // The named blocks tile [0, mu).
        let blocks = l.named_blocks();
        assert_eq!(blocks[0].start, 0);
        for w in blocks.windows(2) {
            assert_eq!(w[0].start + w[0].len, w[1].start);
        }
        assert_eq!(blocks[6].start + blocks[6].len, l.mu);
        assert_eq!((l.width_d, l.t), (l.t, l.w + l.t));
        assert_eq!(l.ell_pp, l.ell_u + l.ell_l);
        assert_eq!(l.n_prime, l.row_a + l.n_a);
    }
}

/// Each invariant that `Layout::new` checks, broken one at a time.
#[test]
fn inconsistent_parameters_are_rejected() {
    let base = toy(3);
    let layout_err = |p: &Params| match Layout::new(p) {
        Err(Error::Layout(msg)) => msg,
        other => panic!("expected a layout error, got {other:?}"),
    };
    let mut p = base.clone();
    p.mu += 1;
    assert!(layout_err(&p).contains("mu"));
    let mut p = base.clone();
    p.width_d -= 1;
    assert!(layout_err(&p).contains("width of D"));
    let mut p = base.clone();
    p.width_b += 1;
    assert!(layout_err(&p).contains("width of B"));
    let mut p = base.clone();
    p.width_a += 1;
    assert!(layout_err(&p).contains("width of A"));
    let mut p = base.clone();
    p.n_prime += 1;
    assert!(layout_err(&p).contains("n'"));
    let mut p = base.clone();
    p.ell_dprime += 1;
    assert!(layout_err(&p).contains("l''"));
    let mut p = base.clone();
    p.nu += 1;
    assert!(layout_err(&p).contains("nu"));
    let mut p = base.clone();
    p.log_n = 1;
    assert!(layout_err(&p).contains("ceil(log n)"));
    let mut p = base.clone();
    p.d = 24;
    assert!(layout_err(&p).contains("power of two"));
    let mut p = base.clone();
    p.r = 1;
    assert!(layout_err(&p).contains("r = 1"));
    let mut p = base.clone();
    p.choices.n = 0;
    assert!(layout_err(&p).contains("at least one prover"));
    let mut p = base.clone();
    p.mu = u128::MAX;
    assert!(layout_err(&p).contains("does not fit in usize"));
    let mut p = base;
    p.r = 200;
    assert!(layout_err(&p).contains("overflows"));
}

#[cfg(feature = "fixtures")]
mod fixtures {
    use super::*;
    use hachi_co_core::Fixture;

    fn ranges(fx: &Fixture, pointer: &str) -> Vec<core::ops::Range<usize>> {
        fx.int_rows(pointer)
            .unwrap()
            .into_iter()
            .map(|r| r[0] as usize..r[1] as usize)
            .collect()
    }

    #[test]
    fn layout_matches_every_fixture() {
        for name in Fixture::TOY {
            let fx = Fixture::load(name).unwrap();
            let l = Layout::new(&fx.params().unwrap()).unwrap();
            // Named blocks: [name, start, length].
            let recorded = fx.get("/ring_to_linear/layout/blocks").unwrap();
            let recorded = recorded.as_array().unwrap();
            assert_eq!(recorded.len(), 7);
            for (b, r) in l.named_blocks().iter().zip(recorded) {
                assert_eq!(r[0].as_str(), Some(b.name), "{name}");
                assert_eq!(r[1].as_u64(), Some(b.start as u64), "{name} {}", b.name);
                assert_eq!(r[2].as_u64(), Some(b.len as u64), "{name} {}", b.name);
            }
            // Rows of M.
            for (key, row) in [
                ("D", l.row_d),
                ("B", l.row_b),
                ("b", l.row_bg),
                ("c", l.row_c),
                ("A", l.row_a),
            ] {
                let pointer = format!("/ring_to_linear/layout/rows/{key}");
                assert_eq!(fx.usize(&pointer).unwrap(), row, "{name} {key}");
            }
            assert_eq!(
                fx.ints("/ring_to_linear/M_shape").unwrap(),
                vec![l.n_prime as i64, l.mu as i64, l.d as i64]
            );
            // Partition S_1..S_n (merged ranges) and the unheld positions.
            for p in 1..=fx.n() {
                let pointer = format!("/ring_to_linear/partition/{p}");
                assert_eq!(l.s_ranges(p), ranges(&fx, &pointer), "{name} S_{p}");
            }
            assert_eq!(l.unheld(), ranges(&fx, "/ring_to_linear/unheld"), "{name}");
            // Block owners (0 = nobody) and the owned blocks of the commit section.
            let owners = fx.ints("/commit/block_owner").unwrap();
            assert_eq!(owners.len(), l.n_blocks);
            for (i, o) in owners.iter().enumerate() {
                let expected = (*o != 0).then_some(*o as usize);
                assert_eq!(l.owner_of_block(i), expected, "{name} block {i}");
            }
            let blocks = fx.get("/commit/blocks").unwrap().as_array().unwrap().len();
            for j in 0..blocks {
                let i = fx.usize(&format!("/commit/blocks/{j}/index")).unwrap();
                let o = fx.usize(&format!("/commit/blocks/{j}/owner")).unwrap();
                assert_eq!(l.owner_of_block(i), Some(o));
                assert!(l.blocks_of(o).contains(&i));
            }
            assert_eq!(blocks, l.n * l.blocks_per_party);
            // Hypercube sizes and the dimensions the blocks index with.
            for (key, value) in [
                ("ell_U", l.ell_u),
                ("ell_L", l.ell_l),
                ("ell_pp", l.ell_pp),
                ("log_nu", l.log_nu),
                ("mu", l.mu),
                ("n_prime", l.n_prime),
                ("nu", l.nu),
                ("width_A", l.width_a),
                ("width_B", l.width_b),
                ("width_D", l.width_d),
                ("d0", l.d0),
                ("d1", l.d1),
                ("delta", l.delta),
                ("tau", l.tau),
                ("r", l.r),
                ("m", l.m),
                ("logn", l.log_n),
                ("ell_prime", l.ell_prime),
            ] {
                let pointer = format!("/params/{key}");
                assert_eq!(fx.usize(&pointer).unwrap(), value, "{name} {key}");
            }
        }
    }
}
