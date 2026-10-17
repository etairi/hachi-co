//! Open's failure boundary beyond the oracle's three negative tests: tamperings that each break
//! exactly one condition (with the other values made consistent again), the norm conditions at
//! their bounds $\bar\beta$, $\bar\gamma$ and $\bar\omega$ (passing at the bound, failing one
//! beyond), the blocks of the empty slots for $n = 3$, and malformed inputs (errors, no panic).
mod common;

use common::{K, load};
use hachi_co_commit::{Error, block_state, c_star, open, party_state, t_hat_slice};
use hachi_co_core::{Context, Fixture, PartyState};
use hachi_co_math::{Ext, Poly, decompose};

/// The honest states (from the recorded shares), the commitment and f.
fn honest(fx: &Fixture, ctx: &Context<K>) -> (Vec<PartyState<K>>, Vec<Poly>, Vec<Ext<K>>) {
    let states = (1..=ctx.layout.n)
        .map(|p| {
            let share = fx
                .exts::<K>(&format!("/commit/f_shares/{}", p - 1))
                .unwrap();
            party_state(ctx, p, share).unwrap()
        })
        .collect();
    (
        states,
        fx.polys("/commit/u").unwrap(),
        fx.exts::<K>("/input/f").unwrap(),
    )
}

/// $\mathbf u = \mathbf B \hat{\mathbf t}$ for whatever the states hold.
fn commitment(ctx: &Context<K>, states: &[PartyState<K>]) -> Vec<Poly> {
    let ring = &ctx.ring;
    let mut t_hat = vec![ring.zero(); ctx.layout.width_b];
    for st in states {
        for (acc, x) in t_hat.iter_mut().zip(t_hat_slice(ctx, &st.blocks).unwrap()) {
            *acc = ring.add(acc, &x).unwrap();
        }
    }
    ring.mat_vec(&ctx.pp.b, &t_hat).unwrap()
}

fn failing(
    ctx: &Context<K>,
    states: &[PartyState<K>],
    u: &[Poly],
    f: &[Ext<K>],
    c: Option<&[Vec<i64>]>,
) -> Vec<&'static str> {
    let report = open(ctx, states, u, f, c).unwrap();
    assert_eq!(report.bit, report.checks.failing().is_empty());
    report.checks.failing()
}

#[test]
fn each_tampering_breaks_exactly_its_condition() {
    for name in Fixture::TOY {
        let (fx, ctx) = load(name);
        let (l, ring, q) = (&ctx.layout, &ctx.ring, ctx.q());
        let (states, u, f) = honest(&fx, &ctx);
        let none: Vec<&str> = Vec::new();
        assert_eq!(failing(&ctx, &states, &u, &f, None), none, "{name}: honest");
        let several = vec!["no_block_in_several_states"];

        // The same block twice in one state, and in another state.
        let mut st = states.clone();
        st[0].blocks.push(states[0].blocks[0].clone());
        assert_eq!(failing(&ctx, &st, &u, &f, None), several, "{name}");
        if l.n > 1 {
            let mut st = states.clone();
            st[l.n - 1].blocks.push(states[0].blocks[0].clone());
            assert_eq!(failing(&ctx, &st, &u, &f, None), several, "{name}");
        }
        // A different later entry for block 0: the first one is used.
        let mut other_f_prime = states[0].blocks[0].f_prime.clone();
        other_f_prime[0].0[0] = (other_f_prime[0].0[0] + 1) % q;
        let mut st = states.clone();
        st[l.n - 1]
            .blocks
            .push(block_state(&ctx, 0, other_f_prime).unwrap());
        assert_eq!(failing(&ctx, &st, &u, &f, None), several, "{name}");

        // u changed, and u with a non-canonical residue (compared as given).
        let mut u_bad = u.clone();
        u_bad[0].0[3] = (u_bad[0].0[3] + 1) % q;
        assert_eq!(
            failing(&ctx, &states, &u_bad, &f, None),
            vec!["B_that_eq_u"]
        );
        let mut u_bad = u.clone();
        u_bad[0].0[3] += q;
        assert_eq!(
            failing(&ctx, &states, &u_bad, &f, None),
            vec!["B_that_eq_u"]
        );

        // f with a non-canonical residue is read modulo q.
        let mut f_unreduced = f.clone();
        f_unreduced[0].0[0] += q;
        assert_eq!(failing(&ctx, &states, &u, &f_unreduced, None), none);

        // One digit of t-hat changed inside the digit range, u recomputed: only A s = G t-hat.
        let mut st = states.clone();
        let digit = &mut st[0].blocks[0].t_hat[0].0[0];
        let centered = ring.zq().to_centered(*digit);
        *digit = ring.zq().from_i64(if centered <= 0 {
            centered + 1
        } else {
            centered - 1
        });
        let u2 = commitment(&ctx, &st);
        assert_eq!(failing(&ctx, &st, &u2, &f, None), vec!["A_s_eq_G_that"]);

        // A missing block is assembled as zero.
        let mut st = states.clone();
        st[0].blocks.remove(0);
        assert_eq!(
            failing(&ctx, &st, &u, &f, None),
            vec!["B_that_eq_u", "f_eq_f_bar"],
            "{name}"
        );
    }
}

/// The slack space: $\Vert c\_i \Vert\_1 \le \bar\omega$ passes at the bound and fails one beyond
/// (centered coefficients, read modulo $q$); $c\_i = 0$ has norm 0 but is not a unit.
#[test]
fn the_slack_space_at_its_bound() {
    for name in Fixture::TOY {
        let (fx, ctx) = load(name);
        let (l, q) = (&ctx.layout, ctx.q() as i64);
        let (states, u, f) = honest(&fx, &ctx);
        let omega_bar = ctx.params.omega_bar as i64;
        let none: Vec<&str> = Vec::new();
        let slack = vec!["c_in_slack_space"];
        let with_c0 = |c0: Vec<i64>| {
            let mut c = c_star(l);
            c[l.n_blocks - 1] = c0;
            failing(&ctx, &states, &u, &f, Some(&c))
        };
        let mut c0 = vec![0i64; l.d];
        c0[0] = omega_bar;
        assert_eq!(with_c0(c0.clone()), none, "{name}: ||c||_1 = omega-bar");
        c0[0] = omega_bar + 1;
        assert_eq!(
            with_c0(c0.clone()),
            slack,
            "{name}: ||c||_1 = omega-bar + 1"
        );
        // Spread over several coefficients, with signs, and given as residues.
        let mut c0 = vec![0i64; l.d];
        for (j, x) in c0.iter_mut().enumerate().take(omega_bar as usize) {
            *x = if j % 2 == 0 { -1 } else { q - 1 };
        }
        assert_eq!(with_c0(c0.clone()), none, "{name}");
        c0[l.d - 1] = 1;
        assert_eq!(with_c0(c0), slack, "{name}");
        // Zero: norm 0, not a unit.
        assert_eq!(with_c0(vec![0; l.d]), slack, "{name}: c_i = 0");
    }
}

/// Finds a block (in party order) and a coefficient where digit 0 of entry `entry` of
/// `field(block)` is congruent to `v` modulo $b$, sets that digit to $v$ and moves the
/// difference into digit 1, so that $\mathbf G$ of the digits is unchanged.
fn set_digit(
    ctx: &Context<K>,
    states: &mut [PartyState<K>],
    entry: usize,
    v: i64,
    field: fn(&mut hachi_co_core::BlockState) -> &mut Vec<Poly>,
) {
    let (zq, b, delta) = (*ctx.ring.zq(), ctx.b() as i64, ctx.layout.delta);
    for st in states.iter_mut() {
        for block in &mut st.blocks {
            let digits = field(block);
            for c in 0..ctx.layout.d {
                let x = zq.to_centered(digits[entry * delta].0[c]);
                if (v - x).rem_euclid(b) == 0 {
                    let carry = (v - x) / b;
                    let next = zq.to_centered(digits[entry * delta + 1].0[c]);
                    digits[entry * delta].0[c] = zq.from_i64(v);
                    digits[entry * delta + 1].0[c] = zq.from_i64(next - carry);
                    return;
                }
            }
        }
    }
    panic!("no digit congruent to {v} modulo b");
}

/// $\Vert c\_i \mathbf s\_i \Vert\_\infty \le \bar\beta$ (with $c^\*$) and
/// $\Vert \hat{\mathbf t} \Vert\_\infty \le \bar\gamma$ at their bounds: a digit set to
/// $\pm\bar\beta$ resp. $\pm\bar\gamma$ passes, $\pm(\bar\beta + 1)$ resp. $\pm(\bar\gamma + 1)$
/// fails, with $\mathbf G$ of the digits unchanged (so $\mathbf f$ decodes as before) and
/// $\mathbf t, \hat{\mathbf t}, \mathbf u$ recomputed, so that every other condition holds.
#[test]
fn the_norm_conditions_at_their_bounds() {
    for name in Fixture::TOY {
        let (fx, ctx) = load(name);
        let (honest_states, _, f) = honest(&fx, &ctx);
        let (ring, b, delta) = (&ctx.ring, ctx.b(), ctx.layout.delta);
        let beta_bar = ctx.params.beta_bar as i64;
        let gamma_bar = ctx.params.gamma_bar as i64;
        let none: Vec<&str> = Vec::new();
        for (v, want) in [
            (beta_bar, none.clone()),
            (-beta_bar, none.clone()),
            (beta_bar + 1, vec!["norm_c_s_le_beta_bar"]),
            (-beta_bar - 1, vec!["norm_c_s_le_beta_bar"]),
        ] {
            let mut st = honest_states.clone();
            set_digit(&ctx, &mut st, 0, v, |blk| &mut blk.s);
            // Recompute t = A s and t-hat = G^-1(t) of every block, then u.
            for block in st.iter_mut().flat_map(|s| s.blocks.iter_mut()) {
                assert_eq!(
                    decompose::g_apply(ring, b, delta, &block.s).unwrap(),
                    block.f_prime
                );
                block.t = ring.mat_vec(&ctx.pp.a, &block.s).unwrap();
                block.t_hat = decompose::g_inv(ring, b, delta, &block.t).unwrap();
            }
            let u = commitment(&ctx, &st);
            let report = open(&ctx, &st, &u, &f, None).unwrap();
            assert_eq!(report.checks.failing(), want, "{name}: digit {v} in s");
            assert_eq!(report.max_norm_c_s, v.unsigned_abs(), "{name}");
        }
        for (v, want) in [
            (gamma_bar, none.clone()),
            (-gamma_bar, none.clone()),
            (gamma_bar + 1, vec!["norm_that_le_gamma_bar"]),
            (-gamma_bar - 1, vec!["norm_that_le_gamma_bar"]),
        ] {
            let mut st = honest_states.clone();
            set_digit(&ctx, &mut st, 0, v, |blk| &mut blk.t_hat);
            let u = commitment(&ctx, &st);
            let report = open(&ctx, &st, &u, &f, None).unwrap();
            assert_eq!(report.checks.failing(), want, "{name}: digit {v} in t-hat");
            assert_eq!(report.max_norm_t_hat, v.unsigned_abs(), "{name}");
        }
    }
}

/// For $n = 3$ the slot $\mathbf p = 11$ is empty: blocks 6 and 7 are held by nobody and decode
/// as zero. A state that adds block 6 changes $\bar f$, because $\bar f\_{\mathbf q}$ sums all
/// $2^{\lceil \log n \rceil}$ slots (`6_Collaborative_Hachi.tex`:398).
#[test]
fn a_block_of_the_empty_slot_is_decoded_into_f_bar() {
    let (fx, ctx) = load("toy-n3");
    let l = &ctx.layout;
    let (states, u, f) = honest(&fx, &ctx);
    assert_eq!(l.owner_of_block(6), None);
    let mut st = states.clone();
    st[2]
        .blocks
        .push(block_state(&ctx, 6, states[0].blocks[0].f_prime.clone()).unwrap());
    assert_eq!(
        failing(&ctx, &st, &u, &f, None),
        vec!["B_that_eq_u", "f_eq_f_bar"]
    );
    let u2 = commitment(&ctx, &st);
    assert_eq!(failing(&ctx, &st, &u2, &f, None), vec!["f_eq_f_bar"]);
    // An all-zero block 6 changes nothing.
    let mut st = states.clone();
    st[2]
        .blocks
        .push(block_state(&ctx, 6, vec![ctx.ring.zero(); l.block_len]).unwrap());
    assert!(open(&ctx, &st, &u, &f, None).unwrap().bit);
}

#[test]
fn malformed_inputs_are_errors() {
    let (fx, ctx) = load("toy-n2");
    let l = &ctx.layout;
    let (states, u, f) = honest(&fx, &ctx);
    let shape = |r: Result<_, Error>| matches!(r, Err(Error::Shape { .. }));
    assert!(shape(open(&ctx, &states[..1], &u, &f, None)));
    assert!(shape(open(&ctx, &[], &u, &f, None)));
    assert!(shape(open(&ctx, &states, &u[..0], &f, None)));
    let mut u_short = u.clone();
    u_short[0].0.pop();
    assert!(shape(open(&ctx, &states, &u_short, &f, None)));
    assert!(shape(open(&ctx, &states, &u, &f[1..], None)));
    let c = c_star(l);
    assert!(shape(open(&ctx, &states, &u, &f, Some(&c[1..]))));
    let mut c_long = c.clone();
    c_long[0].push(0);
    assert!(shape(open(&ctx, &states, &u, &f, Some(&c_long))));
    // Blocks of the wrong shape: an index beyond 2^r, a short s.
    let mut st = states.clone();
    st[0].blocks[0].index = l.n_blocks;
    assert!(matches!(
        open(&ctx, &st, &u, &f, None),
        Err(Error::Core(hachi_co_core::Error::Shape { .. }))
    ));
    let mut st = states.clone();
    st[1].blocks[0].s.pop();
    assert!(matches!(
        open(&ctx, &st, &u, &f, None),
        Err(Error::Core(hachi_co_core::Error::Shape { .. }))
    ));
    // Extreme coefficients of c are read modulo q without overflow.
    let mut c_extreme = c.clone();
    c_extreme[0] = vec![i64::MIN; l.d];
    c_extreme[1] = vec![i64::MAX; l.d];
    let report = open(&ctx, &states, &u, &f, Some(&c_extreme)).unwrap();
    assert!(!report.checks.c_in_slack_space);
}
