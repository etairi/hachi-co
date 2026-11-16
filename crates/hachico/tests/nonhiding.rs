//! The suffix `-nonhiding` on a preset name ([`Setup::new`]): Hachi without its hiding choices,
//! the published scheme, under the same collaborative layer. Honest runs are accepted, with the
//! commitment of the preset's run and a smaller proof, and the suffix cannot be combined with
//! `-merged`, whose commitments need hiding. The single-prover baseline ([`run_single`]) proves
//! the instance's value with and without hiding.

mod common;

use common::net::run_with;
use hachi_co::harness::{Instance, Setup, prove, run_single};
use hachi_co::verify_value;
use hachi_co_mpc::MemoryConfig;

const SETUP_SEED: [u8; 32] = [41; 32];
const INSTANCE_SEED: [u8; 32] = [42; 32];

fn party_seed(p: usize) -> [u8; 32] {
    let mut seed = [43; 32];
    seed[0] = p as u8;
    seed
}

/// The commitment and party 1's proof bytes, after checking every party's proof with the
/// verifier of `setup`.
fn commitment_and_proof_bytes(setup: &Setup) -> (Vec<hachi_co_core::Poly>, usize) {
    let ctx = &setup.ctx;
    let n = ctx.layout.n;
    let instance = Instance::derive(ctx, &INSTANCE_SEED).unwrap();
    let runs = run_with(&MemoryConfig::new(n, ctx.q(), [44; 32]), |p, comm| {
        let share = instance.share(ctx, &INSTANCE_SEED, p).unwrap();
        prove(setup, comm, &party_seed(p), &share, &instance).unwrap()
    });
    let mut outputs = Vec::new();
    for run in runs {
        let out = run.unwrap().output;
        verify_value(
            ctx,
            &setup.pcs,
            &out.u,
            &instance.x,
            &instance.y,
            &out.proof,
        )
        .unwrap();
        outputs.push((out.u, out.proof.to_bytes().len()));
    }
    outputs.swap_remove(0)
}

#[test]
fn the_nonhiding_suffix_runs_hachi_without_its_hiding_choices() {
    for (preset, n) in [("toy-n2", 2usize), ("toy-n3", 3)] {
        let hiding = Setup::new(preset, n, &SETUP_SEED).unwrap();
        let plain = Setup::new(&format!("{preset}-nonhiding"), n, &SETUP_SEED).unwrap();
        assert!(hiding.pcs.choices().hiding.is_some(), "{preset}");
        assert!(plain.pcs.choices().hiding.is_none(), "{preset}");
        assert_eq!(
            hiding.ctx.params, plain.ctx.params,
            "{preset}: the same Hachi-Co parameters"
        );
        let (u_with, with) = commitment_and_proof_bytes(&hiding);
        let (u_without, without) = commitment_and_proof_bytes(&plain);
        assert_eq!(
            u_with, u_without,
            "{preset}: Commit does not use the base PCS"
        );
        assert!(
            without < with,
            "{preset}: {without} bytes without hiding against {with} with it"
        );
    }
    for name in ["toy-n2-merged-nonhiding", "toy-n2-nonhiding-merged"] {
        assert!(Setup::new(name, 2, &SETUP_SEED).is_err(), "{name}");
    }
    assert!(Setup::new("toy-n9-nonhiding", 2, &SETUP_SEED).is_err());
}

#[test]
fn the_single_prover_proves_the_value_with_and_without_hiding() {
    for (preset, n) in [("toy-n1", 1usize), ("toy-n3", 3)] {
        let mut proofs = Vec::new();
        for name in [preset.to_string(), format!("{preset}-nonhiding")] {
            let setup = Setup::new(&name, n, &SETUP_SEED).unwrap();
            let instance = Instance::derive(&setup.ctx, &INSTANCE_SEED).unwrap();
            let report = run_single(&setup, instance, &party_seed(1)).unwrap();
            assert_eq!(report.hiding, !name.ends_with("-nonhiding"), "{name}");
            proofs.push(report.proof_bytes);
        }
        assert!(proofs[1] < proofs[0], "{preset}: {proofs:?}");
    }
}
