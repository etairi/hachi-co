//! Commit and Open against the Sage oracle's fixtures `ref/fixtures/toy-n{1,2,3}.json` (sections
//! `input`, `commit`, `open`, `negative_tests/open` and the relation checks `open.*`), exact
//! equality residue by residue.
//!
//! Every party's local values are computed one party at a time, with the network emulated by
//! hand (the exchange through the wire format, the F_AH(B) inputs summed here); the drivers are
//! then run on the loopback (toy-n1) and on an in-memory network (n = 1, 2, 3) with the replayed
//! randomness.
mod common;

use common::{K, load, run_parties};
use hachi_co_commit::{
    OpenChecks, block_state, c_star, commit, decode_received, f_hat_of_f_prime, fah_b_input,
    fold_slots, open, party_state, pconv, pconv_blocks, reshare, sum_shares, t_hat_slice,
};
use hachi_co_core::wire::flatten_exts;
use hachi_co_core::{Context, Fixture, PartyState, ReplayRandomness, escape_key};
use hachi_co_math::{Ext, Poly};
use hachi_co_mpc::Loopback;
use std::collections::BTreeMap;

fn f_init(fx: &Fixture, p: usize) -> Vec<Ext<K>> {
    fx.exts::<K>(&format!("/input/f_init_shares/{}", p - 1))
        .unwrap()
}

/// A recorded block of `commit/blocks`: (owner, s, t, t_hat).
type RecordedBlock = (usize, Vec<Poly>, Vec<Poly>, Vec<Poly>);

/// The recorded blocks of `commit/blocks`, by index.
fn recorded_blocks(fx: &Fixture) -> BTreeMap<usize, RecordedBlock> {
    let count = fx.get("/commit/blocks").unwrap().as_array().unwrap().len();
    (0..count)
        .map(|k| {
            let key = |field: &str| format!("/commit/blocks/{k}/{field}");
            (
                fx.usize(&key("index")).unwrap(),
                (
                    fx.usize(&key("owner")).unwrap(),
                    fx.polys(&key("s")).unwrap(),
                    fx.polys(&key("t")).unwrap(),
                    fx.polys(&key("t_hat")).unwrap(),
                ),
            )
        })
        .collect()
}

/// Every party's state, computed party by party with the exchange emulated, checking each local
/// value against the fixture on the way. Returns the states and the F_AH(B) inputs.
fn local_run(fx: &Fixture, ctx: &Context<K>) -> (Vec<PartyState<K>>, Vec<Vec<Poly>>) {
    let name = fx.name().to_owned();
    let (l, ring) = (&ctx.layout, &ctx.ring);
    let n = l.n;
    let len = 1usize << l.ell;
    // Step 1: every party's AS.Share of its initial share (replayed randomness).
    let sent: Vec<Vec<Vec<Ext<K>>>> = (1..=n)
        .map(|p| {
            let mut rng = ReplayRandomness::new(fx, p).unwrap();
            let shares = reshare(ctx, &mut rng, &f_init(fx, p)).unwrap();
            let want = fx
                .ext_rows::<K>(&format!("/commit/reshare/{}", p - 1))
                .unwrap();
            assert_eq!(shares, want, "{name} p = {p}: reshare");
            // The shares add up to the initial share (AS.Share, 4_P:154).
            let field = ctx.field();
            for (t, x) in f_init(fx, p).iter().enumerate() {
                let sum = shares
                    .iter()
                    .fold(field.zero(), |a, s| field.add(&a, &s[t]));
                assert_eq!(sum, *x, "{name} p = {p}: AS.Share sums to f_p");
            }
            shares
        })
        .collect();
    let f_hat_rec = fx.exts::<K>("/commit/f_hat").unwrap();
    let f_prime_rec = fx.polys("/commit/f_prime").unwrap();
    let blocks_rec = recorded_blocks(fx);
    let mut states = Vec::new();
    let mut inputs = Vec::new();
    for p in 1..=n {
        // The exchange, through the wire format: party p receives share p of every party i.
        let incoming: Vec<Vec<u64>> = (1..=n).map(|i| flatten_exts(&sent[i - 1][p - 1])).collect();
        let received = decode_received::<K>(ctx, &incoming).unwrap();
        // Step 2: <f>_p.
        let share = sum_shares(ctx, &received).unwrap();
        let want = fx
            .exts::<K>(&format!("/commit/f_shares/{}", p - 1))
            .unwrap();
        assert_eq!(share, want, "{name} p = {p}: <f>_p");
        // p's slot of f-hat.
        assert_eq!(
            &f_hat_rec[(p - 1) * len..p * len],
            &share[..],
            "{name} p = {p}: f-hat"
        );
        // Step 3: PConv of p's blocks.
        let blocks = pconv_blocks(ctx, p, &share).unwrap();
        let indices: Vec<usize> = blocks.iter().map(|(i, _)| *i).collect();
        assert_eq!(
            indices,
            l.blocks_of(p).collect::<Vec<_>>(),
            "{name} p = {p}"
        );
        for (i, f_prime) in &blocks {
            let bl = l.block_len;
            assert_eq!(
                f_prime,
                &f_prime_rec[i * bl..(i + 1) * bl],
                "{name} p = {p}: f'_{i}"
            );
        }
        // Step 4: s, t, t-hat of p's blocks.
        let state = party_state(ctx, p, share).unwrap();
        state.validate(l).unwrap();
        for block in &state.blocks {
            let (owner, s, t, t_hat) = &blocks_rec[&block.index];
            assert_eq!(*owner, p, "{name}: owner of block {}", block.index);
            assert_eq!(&block.s, s, "{name} p = {p}: s_{}", block.index);
            assert_eq!(&block.t, t, "{name} p = {p}: t_{}", block.index);
            assert_eq!(&block.t_hat, t_hat, "{name} p = {p}: t-hat_{}", block.index);
            // block_state of the recorded f'_i gives the same block.
            assert_eq!(
                &block_state(ctx, block.index, block.f_prime.clone()).unwrap(),
                block
            );
        }
        // Step 5: the F_AH(B) input, against B times the slice (an independent path).
        let input = fah_b_input(ctx, &state.blocks).unwrap();
        let slice = t_hat_slice(ctx, &state.blocks).unwrap();
        assert_eq!(input, ring.mat_vec(&ctx.pp.b, &slice).unwrap());
        inputs.push(input);
        states.push(state);
    }
    (states, inputs)
}

#[test]
fn party_local_values_match_the_fixtures() {
    for name in Fixture::TOY {
        let (fx, ctx) = load(name);
        let (l, ring) = (&ctx.layout, &ctx.ring);
        let (states, inputs) = local_run(&fx, &ctx);
        // Ownership: the recorded owners (0 = nobody), and every owned block is in a state.
        let owners: Vec<u64> = fx
            .ints("/commit/block_owner")
            .unwrap()
            .into_iter()
            .map(|x| x as u64)
            .collect();
        for (i, owner) in owners.iter().enumerate() {
            assert_eq!(
                l.owner_of_block(i).map_or(0, |p| p as u64),
                *owner,
                "{name}: block {i}"
            );
        }
        let held: usize = states.iter().map(|s| s.blocks.len()).sum();
        assert_eq!(held, recorded_blocks(&fx).len(), "{name}: owned blocks");
        // The unowned blocks of f' and the empty slots of f-hat are zero.
        let f_prime_rec = fx.polys("/commit/f_prime").unwrap();
        for i in (0..l.n_blocks).filter(|i| l.owner_of_block(*i).is_none()) {
            for entry in &f_prime_rec[i * l.block_len..(i + 1) * l.block_len] {
                assert!(entry.0.iter().all(|c| *c == 0), "{name}: f'_{i} is zero");
            }
        }
        // The slices of t-hat add up to t-hat, and the F_AH(B) inputs to u.
        let mut t_hat = vec![ring.zero(); l.width_b];
        for state in &states {
            for (acc, x) in t_hat
                .iter_mut()
                .zip(t_hat_slice(&ctx, &state.blocks).unwrap())
            {
                *acc = ring.add(acc, &x).unwrap();
            }
        }
        assert_eq!(t_hat, fx.polys("/commit/t_hat").unwrap(), "{name}: t-hat");
        let mut u = vec![ring.zero(); l.n_b];
        for input in &inputs {
            for (acc, x) in u.iter_mut().zip(input) {
                *acc = ring.add(acc, x).unwrap();
            }
        }
        assert_eq!(u, fx.polys("/commit/u").unwrap(), "{name}: u");
        assert_eq!(
            u,
            ring.mat_vec(&ctx.pp.b, &t_hat).unwrap(),
            "{name}: u = B t-hat"
        );
    }
}

#[test]
fn global_pconv_and_its_inverse_match_the_fixtures() {
    for name in Fixture::TOY {
        let (fx, ctx) = load(name);
        let shares = fx.ext_rows::<K>("/commit/f_shares").unwrap();
        let (f_hat, f_prime) = pconv(&ctx, &shares).unwrap();
        assert_eq!(f_hat, fx.exts::<K>("/commit/f_hat").unwrap(), "{name}");
        assert_eq!(f_prime, fx.polys("/commit/f_prime").unwrap(), "{name}");
        // psi^-1 inverts psi, and the slots add up to f.
        assert_eq!(f_hat_of_f_prime(&ctx, &f_prime).unwrap(), f_hat, "{name}");
        assert_eq!(
            fold_slots(&ctx, &f_hat).unwrap(),
            fx.exts::<K>("/input/f").unwrap(),
            "{name}"
        );
    }
}

/// The honest states of the fixture, computed by the party-local functions.
fn honest(fx: &Fixture, ctx: &Context<K>) -> (Vec<PartyState<K>>, Vec<Poly>, Vec<Ext<K>>) {
    let (states, _) = local_run(fx, ctx);
    (
        states,
        fx.polys("/commit/u").unwrap(),
        fx.exts::<K>("/input/f").unwrap(),
    )
}

#[test]
fn open_accepts_the_honest_commitment_as_recorded() {
    for name in Fixture::TOY {
        let (fx, ctx) = load(name);
        let (states, u, f) = honest(&fx, &ctx);
        assert_eq!(fx.str("/open/c").unwrap(), "c* = (1, ..., 1)");
        let report = open(&ctx, &states, &u, &f, None).unwrap();
        assert_eq!(report.bit, fx.bool("/open/bit").unwrap(), "{name}");
        assert!(report.bit, "{name}: Open(pp, st, u, f, c*) = 1");
        for (key, ok) in report.checks.named() {
            assert_eq!(
                ok,
                fx.bool(&format!("/open/checks/{key}")).unwrap(),
                "{name}: {key}"
            );
            let relation = format!("/relations/{}", escape_key(&format!("open.{key}")));
            assert_eq!(ok, fx.bool(&relation).unwrap(), "{name}: {relation}");
        }
        let recorded = fx.get("/open/checks").unwrap().as_object().unwrap();
        assert_eq!(
            recorded.len(),
            OpenChecks::NAMES.len(),
            "{name}: every condition"
        );
        assert_eq!(
            report.bit,
            fx.bool(&format!(
                "/relations/{}",
                escape_key("open(pp, st, u, f, c*) = 1")
            ))
            .unwrap()
        );
        assert_eq!(report.max_norm_c_s, fx.u64("/open/max_norm_c_s").unwrap());
        assert_eq!(
            report.max_norm_t_hat,
            fx.u64("/open/max_norm_t_hat").unwrap()
        );
        // c* passed explicitly gives the same report.
        let c = c_star(&ctx.layout);
        assert_eq!(open(&ctx, &states, &u, &f, Some(&c)).unwrap(), report);
    }
}

/// The oracle's Open negative tests (`ref/hachico_ref/negative.py`, `open_tests`): each fails
/// with exactly the recorded conditions.
#[test]
fn open_negative_tests_fail_exactly_as_recorded() {
    for name in Fixture::TOY {
        let (fx, ctx) = load(name);
        let l = &ctx.layout;
        let q = ctx.q();
        let (states, u, f) = honest(&fx, &ctx);
        let recorded = fx.get("/negative_tests/open").unwrap().as_object().unwrap();
        assert_eq!(
            recorded.len(),
            3,
            "{name}: the oracle records three Open tests"
        );
        for key in recorded.keys() {
            let report = match key.as_str() {
                "open: one digit of s_i changed" => {
                    // Prover 1's first block, digit 0 of entry 0, coefficient 0, plus 1.
                    let mut st = states.clone();
                    let c = &mut st[0].blocks[0].s[0].0[0];
                    *c = (*c + 1) % q;
                    open(&ctx, &st, &u, &f, None).unwrap()
                }
                "open: f changed in one coordinate" => {
                    let mut f_bad = f.clone();
                    f_bad[5].0[1] = (f_bad[5].0[1] + 1) % q;
                    open(&ctx, &states, &u, &f_bad, None).unwrap()
                }
                "open: c outside the slack space" => {
                    // ||c_0||_1 = omega-bar + 1.
                    let mut c = c_star(l);
                    c[0][1] = ctx.params.omega_bar as i64;
                    open(&ctx, &states, &u, &f, Some(&c)).unwrap()
                }
                other => panic!("{name}: unknown Open negative test {other:?}"),
            };
            let pointer = format!("/negative_tests/open/{}", escape_key(key));
            assert!(fx.bool(&format!("{pointer}/passed")).unwrap());
            assert!(!report.bit, "{name}: {key}: Open outputs 0");
            let want: Vec<String> = fx
                .get(&format!("{pointer}/failing"))
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_owned())
                .collect();
            assert_eq!(report.checks.failing(), want, "{name}: {key}");
        }
    }
}

/// Compares a driver's output with the fixture.
fn check_driver_output(fx: &Fixture, ctx: &Context<K>, p: usize, u: &[Poly], st: &PartyState<K>) {
    let name = fx.name();
    assert_eq!(u, fx.polys("/commit/u").unwrap(), "{name} p = {p}: u");
    assert_eq!(st.party, p);
    assert_eq!(
        st.share,
        fx.exts::<K>(&format!("/commit/f_shares/{}", p - 1))
            .unwrap(),
        "{name} p = {p}: <f>_p"
    );
    st.validate(&ctx.layout).unwrap();
    let blocks_rec = recorded_blocks(fx);
    let f_prime_rec = fx.polys("/commit/f_prime").unwrap();
    let bl = ctx.layout.block_len;
    for block in &st.blocks {
        let i = block.index;
        let (owner, s, t, t_hat) = &blocks_rec[&i];
        assert_eq!(*owner, p);
        assert_eq!(&block.f_prime, &f_prime_rec[i * bl..(i + 1) * bl]);
        assert_eq!(
            (&block.s, &block.t, &block.t_hat),
            (s, t, t_hat),
            "{name} block {i}"
        );
    }
}

#[test]
fn the_driver_with_loopback_reproduces_toy_n1() {
    let (fx, ctx) = load("toy-n1");
    let mut comm = Loopback::new(ctx.q()).unwrap();
    let mut rng = ReplayRandomness::new(&fx, 1).unwrap();
    let (u, st) = commit(&ctx, &mut comm, &mut rng, &f_init(&fx, 1)).unwrap();
    check_driver_output(&fx, &ctx, 1, &u, &st);
    let f = fx.exts::<K>("/input/f").unwrap();
    assert!(open(&ctx, &[st], &u, &f, None).unwrap().bit);
}

#[test]
fn the_drivers_on_an_n_party_network_reproduce_the_fixtures() {
    for name in Fixture::TOY {
        let (fx, ctx) = load(name);
        let n = ctx.layout.n;
        let outputs = run_parties(n, ctx.q(), |p, comm| {
            let mut rng = ReplayRandomness::new(&fx, p).unwrap();
            commit(&ctx, comm, &mut rng, &f_init(&fx, p)).unwrap()
        });
        for (p, (u, st)) in (1..=n).zip(&outputs) {
            check_driver_output(&fx, &ctx, p, u, st);
        }
        let states: Vec<_> = outputs.iter().map(|(_, st)| st.clone()).collect();
        let f = fx.exts::<K>("/input/f").unwrap();
        let report = open(&ctx, &states, &outputs[0].0, &f, None).unwrap();
        assert!(report.bit, "{name}: {:?}", report.checks.failing());
    }
}
