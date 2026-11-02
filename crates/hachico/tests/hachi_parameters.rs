//! Hachi's parameters for the secure presets, in `crates/params/presets/hachi.json`
//! (`scripts/hachi-parameters/`): every secure preset has an entry, the entry builds a
//! [`HachiBase`] whose schedules and proof sizes are the recorded ones, masks included, and `Setup`
//! of the harness takes it. The ignored test prints what the script reads.

use hachi::{HachiChoices, HachiPcs, Hiding, Layout, Schedule};
use hachi_co::HachiBase;
use hachi_co::harness::Setup;
use hachi_co_core::Context;
use hachi_co_params::presets::{hachi_preset, secure_presets};

const K: usize = 4;

/// $\lambda$ and $\phi$ of the masks of the secure presets.
const LAMBDA: u32 = 128;
const PHI: u32 = 13;

/// The largest height of $\mathbf A$ at the masked level that the probe tries.
const MAX_N_A0: usize = 12;

/// The MLWE ranks of `scripts/hachi-parameters/mlwe_ranks.txt`: `(q, d, b) -> nu`.
fn mlwe_ranks() -> std::collections::BTreeMap<(u64, usize, u64), usize> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../scripts/hachi-parameters/mlwe_ranks.txt"
    );
    std::fs::read_to_string(path)
        .expect("the ranks of mlwe_ranks.sage")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            let key = (f[0].parse().ok()?, f[1].parse().ok()?, f[2].parse().ok()?);
            Some((key, f[3].parse().ok()?))
        })
        .collect()
}

/// The JSON of a layout.
fn layout_json(l: &Layout, delta: usize) -> String {
    format!(
        "{{\"r\":{},\"m\":{},\"decomposed\":{},\"beta_s\":{},\"width\":{}}}",
        l.r,
        l.m,
        l.decomposed,
        l.beta_s,
        l.width(delta)
    )
}

/// The JSON of a schedule.
fn schedule_json(s: &Schedule, delta: usize) -> String {
    let iterations: Vec<String> = s
        .iterations
        .iter()
        .map(|it| {
            format!(
                "{{\"layout\":{},\"delta\":{},\"beta\":{},\"tau\":{}}}",
                layout_json(&it.layout, it.delta),
                it.delta,
                it.beta,
                it.tau,
            )
        })
        .collect();
    format!(
        "{{\"iterations\":[{}],\"final\":{}}}",
        iterations.join(","),
        layout_json(&s.final_layout, delta)
    )
}

/// One line per secure preset, object (the share polynomial over the field, the two blinder
/// vectors) and candidate: the input of `scripts/hachi-parameters/choose.py`. The share polynomial
/// has one line without masks and one per height of the masked level, as the object `"masked"`.
#[test]
#[ignore = "a probe: run with --ignored --nocapture"]
fn hachi_layouts() {
    let ranks = mlwe_ranks();
    for preset in secure_presets().unwrap() {
        let (params, _) = preset.derive().unwrap();
        // Up to 8 rows at d = 256, so that n_A d reaches 2048 as at d = 1024 with n_A = 2.
        let max_n_a = if preset.name.ends_with("-a8") { 8 } else { 6 };
        let (q, d) = (params.q, params.d as usize);
        let vars = params.ell_dprime as usize;
        let lengths = [
            ("share", 0usize),
            ("g0", 1 + vars * params.d0 as usize),
            ("g1", 1 + vars * params.d1 as usize),
        ];
        for b in [4u64, 8, 16] {
            for omega in [8u64, 12, 16, 24, 32] {
                for n_a in 1..=max_n_a {
                    let choices = HachiChoices {
                        b,
                        omega,
                        n_a,
                        n_b: 1,
                        n_d: 1,
                        ..HachiChoices::toy(d)
                    };
                    let Ok(pcs) = HachiPcs::<K>::new(q, choices) else {
                        continue;
                    };
                    let delta = hachi_co_math::decompose::num_digits(q, b).unwrap();
                    for (object, len) in lengths {
                        let schedule = if object == "share" {
                            pcs.multilinear_ext_schedule(vars)
                        } else {
                            pcs.vector_schedule(len)
                        };
                        let Ok(schedule) = schedule else { continue };
                        let words = if object == "share" {
                            pcs.multilinear_ext_proof_words(vars)
                        } else {
                            pcs.vector_proof_words(len)
                        }
                        .unwrap();
                        println!(
                            "{{\"preset\":\"{}\",\"q\":{q},\"d\":{d},\"b\":{b},\"omega\":{omega},\
                             \"n_a\":{n_a},\"object\":\"{object}\",\"proof_words\":{words},\
                             \"schedule\":{}}}",
                            preset.name,
                            schedule_json(&schedule, delta)
                        );
                    }
                    let Some(&nu) = ranks.get(&(q, d, b)) else {
                        continue;
                    };
                    for n_a0 in n_a..=MAX_N_A0 {
                        let hiding = Hiding {
                            nu,
                            n_a: n_a0,
                            lambda: LAMBDA,
                            phi: PHI,
                        };
                        let choices = HachiChoices {
                            b,
                            omega,
                            n_a,
                            n_b: 1,
                            n_d: 1,
                            hiding: Some(hiding),
                            ..HachiChoices::toy(d)
                        };
                        let Ok(pcs) = HachiPcs::<K>::new(q, choices) else {
                            continue;
                        };
                        let delta = hachi_co_math::decompose::num_digits(q, b).unwrap();
                        let Ok(schedule) = pcs.multilinear_ext_schedule(vars) else {
                            continue;
                        };
                        let words = pcs.multilinear_ext_proof_words(vars).unwrap();
                        println!(
                            "{{\"preset\":\"{}\",\"q\":{q},\"d\":{d},\"b\":{b},\"omega\":{omega},\
                             \"n_a\":{n_a},\"object\":\"masked\",\"nu\":{nu},\"n_a0\":{n_a0},\
                             \"proof_words\":{words},\"schedule\":{}}}",
                            preset.name,
                            schedule_json(&schedule, delta)
                        );
                    }
                }
            }
        }
    }
}

/// The base PCS of a secure preset, without its matrices' use: the choices alone fix the sizes.
fn context_free_pcs(q: u64, d: usize, name: &str) -> HachiPcs<K> {
    let entry = hachi_preset(name).unwrap().expect("an entry");
    let choices = HachiChoices {
        b: entry.b,
        omega: entry.omega,
        n_a: entry.n_a,
        n_b: entry.n_b,
        n_d: entry.n_d,
        hiding: entry.hiding.map(|h| Hiding {
            nu: h.nu,
            n_a: h.n_a,
            lambda: h.lambda,
            phi: h.phi,
        }),
        ..HachiChoices::toy(d)
    };
    HachiPcs::new(q, choices).unwrap()
}

#[test]
fn every_secure_preset_has_hachi_parameters_of_the_recorded_sizes() {
    for preset in secure_presets().unwrap() {
        let (params, _) = preset.derive().unwrap();
        let (q, d, vars) = (params.q, params.d as usize, params.ell_dprime as usize);
        let pcs = context_free_pcs(q, d, &preset.name);
        let entry = hachi_preset(&preset.name).unwrap().unwrap();
        let name = &preset.name;
        let (len0, len1) = (1 + vars * params.d0 as usize, 1 + vars * params.d1 as usize);
        assert_eq!(
            pcs.multilinear_ext_proof_words(vars).unwrap(),
            entry.proof_words.w,
            "{name}"
        );
        assert_eq!(
            pcs.vector_proof_words(len0).unwrap(),
            entry.proof_words.g0,
            "{name}"
        );
        assert_eq!(
            pcs.vector_proof_words(len1).unwrap(),
            entry.proof_words.g1,
            "{name}"
        );
        assert_eq!(
            pcs.multilinear_ext_schedule(vars).unwrap().iterations.len(),
            entry.iterations.w,
            "{name}"
        );
    }
}

/// The harness takes the entry of a secure preset, its masks included, with the setup seed.
#[test]
fn the_setup_of_a_secure_preset_uses_its_hachi_parameters() {
    let preset = &secure_presets().unwrap()[0];
    let n = preset.choices.n as usize;
    let seed = [41u8; 32];
    let setup = Setup::new(&preset.name, n, &seed).unwrap();
    let entry = hachi_preset(&preset.name).unwrap().unwrap();
    let c: &HachiChoices = setup.pcs.choices();
    assert_eq!((c.b, c.omega, c.n_a), (entry.b, entry.omega, entry.n_a));
    assert_eq!(c.hiding.map(|h| h.nu), entry.hiding.map(|h| h.nu));
    assert_eq!(c.hiding.map(|h| h.n_a), entry.hiding.map(|h| h.n_a));
    let ctx: &Context<K> = &setup.ctx;
    assert_eq!(
        HachiBase::preset_choices(ctx, &preset.name, &seed)
            .unwrap()
            .seed,
        c.seed
    );
    // A toy preset has none, and keeps the default choices.
    assert_eq!(hachi_preset("toy-n2").unwrap(), None);
    let toy = Setup::new("toy-n2", 2, &seed).unwrap();
    assert_eq!(
        toy.pcs.choices(),
        &HachiBase::default_choices(&toy.ctx, &seed)
    );
}
