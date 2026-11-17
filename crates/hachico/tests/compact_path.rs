//! The sumcheck's table formed compactly and committed in parts against its dense path
//! (`hachi_co_sumcheck::set_compact_from`): Eval gives the same proof both ways, lean or not, and
//! the proof verifies. A test binary of its own, since the threshold holds for the whole process.

mod common;

use common::net::run_with;
use hachi_co::harness::{Instance, Setup};
use hachi_co::{commit, eval, eval_lean, verify_value};
use hachi_co_core::PrgRandomness;
use hachi_co_mpc::MemoryConfig;
use hachi_co_params::PadRule;

const SETUP_SEED: [u8; 32] = [41; 32];
const INSTANCE_SEED: [u8; 32] = [42; 32];

fn party_seed(p: usize, tag: u8) -> [u8; 32] {
    let mut seed = [43; 32];
    seed[0] = p as u8;
    seed[1] = tag;
    seed
}

/// Every table compact and in parts (threshold 0), then none (`usize::MAX`): the same proof from
/// `eval` and `eval_lean` at two and three provers, and it verifies.
#[test]
fn the_compact_path_gives_the_proof_of_the_dense_path() {
    for (preset, n) in [("toy-n2", 2usize), ("toy-n3", 3)] {
        let setup = Setup::new(preset, n, &SETUP_SEED).unwrap();
        let (ctx, pcs) = (&setup.ctx, &setup.pcs);
        assert_eq!(
            ctx.layout.pad_rule,
            PadRule::Field,
            "{preset}: the compact path applies"
        );
        let instance = Instance::derive(ctx, &INSTANCE_SEED).unwrap();
        let y = hachi_co_conv::evaluate(ctx, &instance.f, &instance.x).unwrap();
        let mut proofs = Vec::new();
        for compact_from in [0, usize::MAX] {
            hachi_co_sumcheck::set_compact_from(compact_from);
            let runs = run_with(&MemoryConfig::new(n, ctx.q(), [44; 32]), |p, comm| {
                let share = instance.share(ctx, &INSTANCE_SEED, p).unwrap();
                let mut rng = PrgRandomness::new(party_seed(p, 1));
                let (u, state) = commit(ctx, comm, &mut rng, &share).unwrap();
                let full = eval(
                    ctx,
                    pcs,
                    comm,
                    &mut PrgRandomness::new(party_seed(p, 2)),
                    &state,
                    &u,
                    &instance.x,
                )
                .unwrap();
                let lean = eval_lean(
                    ctx,
                    pcs,
                    comm,
                    &mut PrgRandomness::new(party_seed(p, 2)),
                    &state,
                    &u,
                    &instance.x,
                )
                .unwrap();
                (u, full, lean)
            });
            for run in runs {
                let (u, full, lean) = run.unwrap().output;
                assert_eq!(
                    lean.to_bytes(),
                    full.to_bytes(),
                    "{preset}, from {compact_from}: lean and full"
                );
                verify_value(ctx, pcs, &u, &instance.x, &y, &full).unwrap();
                proofs.push((compact_from, full.to_bytes()));
            }
        }
        hachi_co_sumcheck::set_compact_from(1 << 28);
        let (compact, dense): (Vec<_>, Vec<_>) = proofs.iter().partition(|(from, _)| *from == 0);
        assert_eq!(compact.len(), n, "{preset}");
        for ((_, a), (_, b)) in compact.iter().zip(&dense) {
            assert_eq!(
                a, b,
                "{preset}: the compact and the dense path, the same proof"
            );
        }
    }
}
