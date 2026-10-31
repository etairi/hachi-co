//! A commitment with the blocks of $\mathbf f'$ on disk ([`commit_streaming`]) against the one in
//! memory ([`commit`]), over the in-memory network with the seeds of the harness: same commitment
//! and same proof bytes at $n = 1, 2, 3, 4$, whatever the chunks, with Hachi as the base PCS; the
//! states agree block by block, Open accepts, one Eval reads the file as many times as it needs,
//! and the file disappears with the state. Malformed chunks are refused.

mod common;

use common::K;
use common::net::run_with;
use hachi_co::harness::{Instance, PartyConfig, Setup, prove, prove_streaming};
use hachi_co::{commit, commit_streaming, eval, verify};
use hachi_co_commit::open;
use hachi_co_core::{BlockState, Context, Ext, PartyState, PrgRandomness};
use hachi_co_mpc::{Loopback, MemoryConfig};
use std::path::PathBuf;

const SETUP_SEED: [u8; 32] = [31; 32];
const INSTANCE_SEED: [u8; 32] = [32; 32];
const PARTY_SEED: [u8; 32] = [33; 32];

/// An empty scratch directory for `tag`, removed by the caller.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hachi-co-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn network(n: usize, q: u64) -> MemoryConfig {
    MemoryConfig::new(n, q, [34; 32])
}

fn party_seed(p: usize) -> [u8; 32] {
    let mut seed = PARTY_SEED;
    seed[0] = p as u8;
    seed
}

/// The blocks with $\mathbf f'\_i$ and $\mathbf s\_i$, read through [`PartyState::for_each_block`].
fn read_blocks(ctx: &Context<K>, state: &PartyState<K>) -> Vec<BlockState> {
    let mut out = Vec::new();
    state
        .for_each_block(ctx, |block| {
            out.push(block.clone());
            Ok::<(), hachi_co_core::Error>(())
        })
        .unwrap();
    out
}

#[test]
fn streaming_gives_the_commitment_and_the_proof_in_memory() {
    let dir = scratch("streaming");
    for (preset, n) in [
        ("toy-n1", 1usize),
        ("toy-n2", 2),
        ("toy-n3", 3),
        ("toy-n4", 4),
    ] {
        let setup = Setup::new(preset, n, &SETUP_SEED).unwrap();
        let ctx = &setup.ctx;
        let instance = Instance::derive(ctx, &INSTANCE_SEED).unwrap();
        let streamed = Instance::derive_streaming(ctx, &INSTANCE_SEED).unwrap();
        assert_eq!((&streamed.x, streamed.y), (&instance.x, instance.y));
        assert!(streamed.f.is_empty());
        let in_memory = run_with(&network(n, ctx.q()), |p, comm| {
            let share = instance.share(ctx, &INSTANCE_SEED, p).unwrap();
            prove(&setup, comm, &party_seed(p), &share, &instance).unwrap()
        });
        let spilled = run_with(&network(n, ctx.q()), |p, comm| {
            prove_streaming(
                &setup,
                comm,
                &party_seed(p),
                &INSTANCE_SEED,
                &dir,
                &streamed,
            )
            .unwrap()
        });
        for (a, b) in in_memory.into_iter().zip(spilled) {
            let (a, b) = (a.unwrap().output, b.unwrap().output);
            assert_eq!(a.u, b.u, "{preset}: the commitment");
            assert_eq!(
                a.proof.to_bytes(),
                b.proof.to_bytes(),
                "{preset}: the proof"
            );
        }
    }
    let left = std::fs::read_dir(&dir).unwrap().count();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(left, 0, "the spill files are gone with the states");
}

/// The state on disk has what the state in memory has: every block, $\mathbf t\_i$,
/// $\hat{\mathbf t}\_i$, $\boldsymbol\rho^u\_p$, for the chunks of a share in memory.
#[test]
fn the_state_on_disk_matches_the_state_in_memory() {
    let dir = scratch("state");
    for (preset, n) in [("toy-n2", 2usize), ("toy-n3", 3)] {
        let setup = Setup::new(preset, n, &SETUP_SEED).unwrap();
        let ctx = &setup.ctx;
        let instance = Instance::derive(ctx, &INSTANCE_SEED).unwrap();
        let len = ctx.layout.block_len * ctx.layout.d / ctx.layout.k;
        let runs = run_with(&network(n, ctx.q()), |p, comm| {
            let share = instance.share(ctx, &INSTANCE_SEED, p).unwrap();
            let mut rng = PrgRandomness::new(party_seed(p));
            commit(ctx, comm, &mut rng, &share).unwrap()
        });
        let spilled = run_with(&network(n, ctx.q()), |p, comm| {
            let share = instance.share(ctx, &INSTANCE_SEED, p).unwrap();
            let mut rng = PrgRandomness::new(party_seed(p));
            commit_streaming(ctx, comm, &mut rng, &dir, |j| {
                Ok(share[j * len..(j + 1) * len].to_vec())
            })
            .unwrap()
        });
        let states: Vec<PartyState<K>> = runs
            .into_iter()
            .zip(spilled)
            .map(|(a, b)| {
                let ((u, memory), (u_disk, disk)) = (a.unwrap().output, b.unwrap().output);
                assert_eq!(u, u_disk, "{preset}");
                assert!(disk.spill.is_some());
                assert!(disk.blocks.iter().all(|b| b.s.is_empty()));
                disk.validate(&ctx.layout).unwrap();
                assert_eq!(disk.rho_u, memory.rho_u, "{preset}");
                assert_eq!(read_blocks(ctx, &disk), memory.blocks, "{preset}");
                // A second pass reads the same blocks.
                assert_eq!(read_blocks(ctx, &disk), memory.blocks, "{preset}");
                memory
            })
            .collect();
        let u = run_with(&network(n, ctx.q()), |p, comm| {
            let share = instance.share(ctx, &INSTANCE_SEED, p).unwrap();
            let mut rng = PrgRandomness::new(party_seed(p));
            commit(ctx, comm, &mut rng, &share).unwrap().0
        })
        .into_iter()
        .next()
        .unwrap()
        .unwrap()
        .output;
        assert!(open(ctx, &states, &u, &instance.f, None).unwrap().bit);
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A state on disk serves several evaluations, each reading the file again.
#[test]
fn a_state_on_disk_serves_several_evaluations() {
    let dir = scratch("evaluations");
    let setup = Setup::new("toy-n2", 2, &SETUP_SEED).unwrap();
    let (ctx, pcs) = (&setup.ctx, &setup.pcs);
    let instance = Instance::derive(ctx, &INSTANCE_SEED).unwrap();
    let len = ctx.layout.block_len * ctx.layout.d / ctx.layout.k;
    let field = ctx.field();
    let mut points = [instance.x.clone(), instance.x.clone()];
    points[1][0] = field.add(&points[1][0], &field.one());
    let values: Vec<Ext<K>> = points
        .iter()
        .map(|x| hachi_co_conv::evaluate(ctx, &instance.f, x).unwrap())
        .collect();
    let runs = run_with(&network(2, ctx.q()), |p, comm| {
        let share = instance.share(ctx, &INSTANCE_SEED, p).unwrap();
        let mut rng = PrgRandomness::new(party_seed(p));
        let (u, state) = commit_streaming(ctx, comm, &mut rng, &dir, |j| {
            Ok(share[j * len..(j + 1) * len].to_vec())
        })
        .unwrap();
        let proofs: Vec<_> = points
            .iter()
            .zip(&values)
            .enumerate()
            .map(|(k, (x, y))| {
                let mut seed = party_seed(p);
                seed[1] = 1 + k as u8;
                let mut rng = PrgRandomness::new(seed);
                eval(ctx, pcs, comm, &mut rng, &state, &u, x, y).unwrap()
            })
            .collect();
        (u, proofs)
    });
    let (u, proofs) = runs.into_iter().next().unwrap().unwrap().output;
    for ((x, y), proof) in points.iter().zip(&values).zip(&proofs) {
        verify(ctx, pcs, &u, x, y, proof).unwrap();
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn malformed_chunks_are_refused() {
    let dir = scratch("malformed");
    let setup = Setup::new("toy-n1", 1, &SETUP_SEED).unwrap();
    let ctx = &setup.ctx;
    let short = |rng: &mut PrgRandomness, dir: &std::path::Path| {
        let mut comm = Loopback::new(ctx.q()).unwrap();
        commit_streaming(ctx, &mut comm, rng, dir, |_| Ok(vec![Ext::<K>([0; K])]))
    };
    assert!(short(&mut PrgRandomness::new(PARTY_SEED), &dir).is_err());
    // A directory that does not exist is an error, not a panic.
    assert!(short(&mut PrgRandomness::new(PARTY_SEED), &dir.join("missing")).is_err());
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The file of `hachi-co-party`: `spill_dir` is optional, and only that.
#[test]
fn the_configuration_takes_a_spill_directory() {
    let json = serde_json::json!({
        "preset": "toy-n1",
        "n": 1,
        "party": 1,
        "addresses": ["127.0.0.1:7101"],
        "session": "00".repeat(32),
        "setup_seed": "01".repeat(32),
        "instance_seed": "02".repeat(32),
        "party_seed": "03".repeat(32),
        "output": "out",
    });
    let config: PartyConfig = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(config.spill_dir, None);
    let mut with_dir = json;
    with_dir["spill_dir"] = "spill".into();
    let config: PartyConfig = serde_json::from_value(with_dir).unwrap();
    assert_eq!(config.spill_dir, Some(PathBuf::from("spill")));
    assert_eq!(
        serde_json::from_str::<PartyConfig>(&serde_json::to_string(&config).unwrap()).unwrap(),
        config
    );
}
