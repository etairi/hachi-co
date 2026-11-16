//! A commitment with the blocks of $\mathbf f'$ on disk ([`commit_streaming`]) against the one in
//! memory ([`commit`]), over the in-memory network with the seeds of the harness: same commitment
//! and same proof bytes at $n = 1, 2, 3, 4$, whatever the chunks, with Hachi as the base PCS; the
//! states agree block by block, Open accepts, one Eval reads the file as many times as it needs,
//! and the file disappears with the state. Malformed chunks are refused.

mod common;

use common::K;
use common::net::run_with;
use hachi_co::harness::{Instance, PartyConfig, Setup, prove, prove_streaming};
use hachi_co::{commit, commit_streaming, eval, eval_lean, verify_value};
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

/// The lean Eval (`eval_lean`) gives the proof of `eval`, for a state in memory and on disk, and
/// the context it released its matrices from still serves a second evaluation and the verifier.
#[test]
fn the_lean_evaluation_is_the_evaluation() {
    let dir = scratch("lean");
    for (preset, n) in [("toy-n2", 2usize), ("toy-n3", 3)] {
        let setup = Setup::new(preset, n, &SETUP_SEED).unwrap();
        let (ctx, pcs) = (&setup.ctx, &setup.pcs);
        let instance = Instance::derive(ctx, &INSTANCE_SEED).unwrap();
        let len = ctx.layout.block_len * ctx.layout.d / ctx.layout.k;
        let runs = run_with(&network(n, ctx.q()), |p, comm| {
            let share = instance.share(ctx, &INSTANCE_SEED, p).unwrap();
            let mut rng = PrgRandomness::new(party_seed(p));
            let (u, state) = if p % 2 == 0 {
                commit(ctx, comm, &mut rng, &share).unwrap()
            } else {
                commit_streaming(ctx, comm, &mut rng, &dir, |j| {
                    Ok(share[j * len..(j + 1) * len].to_vec())
                })
                .unwrap()
            };
            let mut seed = party_seed(p);
            seed[1] = 7;
            let full = eval(
                ctx,
                pcs,
                comm,
                &mut PrgRandomness::new(seed),
                &state,
                &u,
                &instance.x,
            )
            .unwrap();
            seed[1] = 8;
            let lean = eval_lean(
                ctx,
                pcs,
                comm,
                &mut PrgRandomness::new(seed),
                &state,
                &u,
                &instance.x,
            )
            .unwrap();
            seed[1] = 8;
            let again = eval_lean(
                ctx,
                pcs,
                comm,
                &mut PrgRandomness::new(seed),
                &state,
                &u,
                &instance.x,
            )
            .unwrap();
            seed[1] = 8;
            let full_again = eval(
                ctx,
                pcs,
                comm,
                &mut PrgRandomness::new(seed),
                &state,
                &u,
                &instance.x,
            )
            .unwrap();
            (u, full, lean, again, full_again)
        });
        let y = hachi_co_conv::evaluate(ctx, &instance.f, &instance.x).unwrap();
        for run in runs {
            let (u, full, lean, again, full_again) = run.unwrap().output;
            assert_eq!(
                lean.to_bytes(),
                full_again.to_bytes(),
                "{preset}: lean and full, same randomness"
            );
            assert_eq!(
                again.to_bytes(),
                lean.to_bytes(),
                "{preset}: after the release"
            );
            verify_value(ctx, pcs, &u, &instance.x, &y, &full).unwrap();
            verify_value(ctx, pcs, &u, &instance.x, &y, &lean).unwrap();
        }
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
            .enumerate()
            .map(|(k, x)| {
                let mut seed = party_seed(p);
                seed[1] = 1 + k as u8;
                let mut rng = PrgRandomness::new(seed);
                eval(ctx, pcs, comm, &mut rng, &state, &u, x).unwrap()
            })
            .collect();
        (u, proofs)
    });
    let (u, proofs) = runs.into_iter().next().unwrap().unwrap().output;
    for ((x, y), proof) in points.iter().zip(&values).zip(&proofs) {
        verify_value(ctx, pcs, &u, x, y, proof).unwrap();
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

/// The visits that read $\mathbf f'\_i$ only give the values of their definitions from
/// $\mathbf s\_i$: [`hachi_co_r2l::owned_w`] is [`hachi_co_r2l::block_w`] of every block,
/// [`hachi_co_conv::u_share`] is $\sum\_i \mathbf d\_i^T \mathbf f'\_i$ block by block, and
/// [`hachi_co_r2l::fold`] of a state on disk (its digits formed from $\mathbf f'\_i$) is the fold
/// of the state in memory (from $\mathbf s\_i$), at toy-n2 and toy-n3.
#[test]
fn the_visits_of_f_prime_give_the_values_of_the_blocks() {
    let dir = scratch("f-prime");
    for (preset, n) in [("toy-n2", 2usize), ("toy-n3", 3)] {
        let setup = Setup::new(preset, n, &SETUP_SEED).unwrap();
        let ctx = &setup.ctx;
        let (l, ring) = (&ctx.layout, &ctx.ring);
        let instance = Instance::derive(ctx, &INSTANCE_SEED).unwrap();
        let len = l.block_len * l.d / l.k;
        let disk = run_with(&network(n, ctx.q()), |p, comm| {
            let share = instance.share(ctx, &INSTANCE_SEED, p).unwrap();
            let mut rng = PrgRandomness::new(party_seed(p));
            commit_streaming(ctx, comm, &mut rng, &dir, |j| {
                Ok(share[j * len..(j + 1) * len].to_vec())
            })
            .unwrap()
            .1
        });
        let memory = run_with(&network(n, ctx.q()), |p, comm| {
            let share = instance.share(ctx, &INSTANCE_SEED, p).unwrap();
            let mut rng = PrgRandomness::new(party_seed(p));
            commit(ctx, comm, &mut rng, &share).unwrap().1
        });
        let pv = hachi_co_conv::public_vectors(ctx, &instance.x).unwrap();
        // Challenges of C: omega coefficients +-1 at distinct places (13 is odd, d a power of 2).
        let omega = (ctx.omega() as usize).min(l.d);
        let c: Vec<Vec<i64>> = (0..l.n_blocks)
            .map(|i| {
                let mut ci = vec![0i64; l.d];
                for k in 0..omega {
                    ci[(7 * i + 13 * k) % l.d] = if (i + k) % 2 == 0 { 1 } else { -1 };
                }
                ci
            })
            .collect();
        for (disk, memory) in disk.into_iter().zip(memory) {
            let (disk, memory) = (disk.unwrap().output, memory.unwrap().output);
            assert!(disk.spill.is_some() && memory.spill.is_none());
            let blocks = read_blocks(ctx, &disk);
            assert_eq!(blocks, memory.blocks, "{preset}");
            let want_w: Vec<_> = blocks
                .iter()
                .map(|b| hachi_co_r2l::block_w(ctx, &pv, b).unwrap())
                .collect();
            let mut want_u = ring.zero();
            for b in &blocks {
                let d_block = &pv.d[b.index * l.block_len..(b.index + 1) * l.block_len];
                let d_ring: Vec<_> = d_block.iter().map(|e| ctx.sub.embed(e)).collect();
                want_u = ring
                    .add(&want_u, &ring.inner(&d_ring, &b.f_prime).unwrap())
                    .unwrap();
            }
            let want_fold = hachi_co_r2l::fold(ctx, &memory, &c).unwrap();
            for state in [&disk, &memory] {
                assert_eq!(hachi_co_r2l::owned_w(ctx, &pv, state).unwrap(), want_w);
                assert_eq!(hachi_co_conv::u_share(ctx, state, &pv).unwrap(), want_u);
                assert_eq!(hachi_co_r2l::fold(ctx, state, &c).unwrap(), want_fold);
            }
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

/// Item 147's measurement of the spilled reads: one pass over party 1's state on disk at `PRESET`
/// (default secure-l22-n2-a8) by [`PartyState::for_each_f_prime_group`] (a group's blocks read
/// in parallel) against the blocks read one after the other, as before, alternating, `REPS` times.
#[test]
#[ignore = "timings: run with --release --ignored --nocapture"]
fn timings_spilled_reads() {
    use std::time::Instant;
    let preset = std::env::var("PRESET").unwrap_or_else(|_| "secure-l22-n2-a8".into());
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let dir = scratch("reads");
    let setup = Setup::new(&preset, 2, &SETUP_SEED).unwrap();
    let ctx = &setup.ctx;
    let instance = Instance::derive(ctx, &INSTANCE_SEED).unwrap();
    let len = ctx.layout.block_len * ctx.layout.d / ctx.layout.k;
    let disk = run_with(&network(2, ctx.q()), |p, comm| {
        let share = instance.share(ctx, &INSTANCE_SEED, p).unwrap();
        let mut rng = PrgRandomness::new(party_seed(p));
        commit_streaming(ctx, comm, &mut rng, &dir, |j| {
            Ok(share[j * len..(j + 1) * len].to_vec())
        })
        .unwrap()
        .1
    })
    .into_iter()
    .next()
    .unwrap()
    .unwrap()
    .output;
    let file = disk.spill.clone().unwrap();
    println!("{preset}: {} blocks of party 1", disk.blocks.len());
    for rep in 0..reps {
        let start = Instant::now();
        let mut sum = 0u64;
        disk.for_each_f_prime_group(ctx, |_, f_primes| {
            sum = sum.wrapping_add(f_primes.iter().map(|f| f[0].0[0]).sum::<u64>());
            Ok::<(), hachi_co_core::Error>(())
        })
        .unwrap();
        let grouped = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let mut again = 0u64;
        for slot in 0..disk.blocks.len() {
            again = again.wrapping_add(file.read_block(slot).unwrap()[0].0[0]);
        }
        let serial = start.elapsed().as_secs_f64();
        assert_eq!(sum, again);
        println!(
            "rep {rep}: one after the other {serial:.3} s, grouped in parallel {grouped:.3} s"
        );
    }
    drop((disk, file));
    std::fs::remove_dir_all(&dir).unwrap();
}
