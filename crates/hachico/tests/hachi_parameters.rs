//! Hachi's parameters for the secure presets, in `crates/params/presets/hachi.json` and
//! `hachi100.json` (`scripts/hachi-parameters/`): every secure preset of every family has an
//! entry at its family's target, the entry builds a [`HachiBase`] whose schedules and proof sizes
//! are the recorded ones, masks included, and `Setup` of the harness takes it. The ignored test
//! prints what the script reads.

use hachi::{HachiChoices, HachiPcs, Hiding, Layout, Schedule};
use hachi_co::HachiBase;
use hachi_co::harness::Setup;
use hachi_co_core::Context;
use hachi_co_params::presets::{SECURE_FAMILIES, all_secure_presets, hachi_preset, secure_presets};

const K: usize = 4;

/// $\phi$ of the masks of the secure presets. Their $\lambda$ is the preset's.
const PHI: u32 = 13;

/// The largest height of $\mathbf A$ at the masked level that the probe tries.
const MAX_N_A0: usize = 12;

/// The bases $b$ that the probe tries.
const BASES: [u64; 3] = [4, 8, 16];

/// The radii $\omega$ that the probe tries: every integer from 8 to 32. The grid
/// $8, 12, 16, 24, 32$ used before found no masked choice at `secure-l16-n2-a8` and `-n4-a8` once
/// $L = \lambda$ raised $\ell''$: their first iteration has $r = 9$, so $\omega = 24$ leaves the
/// challenge set below $2^{\lambda + r + 1}$, while $\omega = 32$ puts the bound of the masked
/// level above what Module-SIS allows at their $q$, and $\omega = 25$ does neither.
const OMEGAS: std::ops::RangeInclusive<u64> = 8..=32;

/// The MLWE ranks at `lambda` bits of `scripts/hachi-parameters/mlwe_ranks-<lambda>.txt`:
/// `(q, d, b) -> nu`.
fn mlwe_ranks(lambda: u32) -> std::collections::BTreeMap<(u64, usize, u64), usize> {
    let path = format!(
        "{}/../../scripts/hachi-parameters/mlwe_ranks-{lambda}.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{path}, the ranks of mlwe_ranks.sage target={lambda}: {e}"))
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
/// vectors) and candidate: the input of `scripts/hachi-parameters/choose.py`. Each object has one
/// line without masks and one per height of the masked level, as the objects `"masked"` (the
/// share polynomial), `"masked-g0"` and `"masked-g1"`.
///
/// The candidates are every base in [`BASES`], every radius in [`OMEGAS`] and every height up to
/// 8 rows (`-a8`) or 6 (`-a10`). The family is `HACHI_PROBE_FAMILY` (default `secure`), and the
/// masks and MLWE ranks are at the family's $\lambda$.
#[test]
#[ignore = "a probe: run with --ignored --nocapture"]
fn hachi_layouts() {
    let name = std::env::var("HACHI_PROBE_FAMILY").unwrap_or_else(|_| "secure".to_owned());
    let family = SECURE_FAMILIES
        .iter()
        .find(|f| f.name == name)
        .unwrap_or_else(|| panic!("HACHI_PROBE_FAMILY={name} is not a family"));
    let ranks = mlwe_ranks(family.lambda);
    for preset in family.presets().unwrap() {
        assert_eq!(preset.choices.lambda, family.lambda, "{}", preset.name);
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
        for b in BASES {
            for omega in OMEGAS {
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
                            lambda: family.lambda,
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
                        // The blinders, masked by the same choices.
                        for (object, len) in &lengths[1..] {
                            let Ok(schedule) = pcs.vector_schedule(*len) else {
                                continue;
                            };
                            let words = pcs.vector_proof_words(*len).unwrap();
                            println!(
                                "{{\"preset\":\"{}\",\"q\":{q},\"d\":{d},\"b\":{b},\
                                 \"omega\":{omega},\"n_a\":{n_a},\"object\":\"masked-{object}\",\
                                 \"nu\":{nu},\"n_a0\":{n_a0},\"proof_words\":{words},\
                                 \"schedule\":{}}}",
                                preset.name,
                                schedule_json(&schedule, delta)
                            );
                        }
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

/// Each family's Hachi file is chosen at the family's target, and its masks at the preset's
/// $\lambda$.
#[test]
fn every_hachi_file_targets_its_family() {
    for family in &SECURE_FAMILIES {
        let file: serde_json::Value = serde_json::from_str(family.hachi_json).unwrap();
        assert_eq!(
            file["target_bits"],
            u64::from(family.lambda),
            "the Hachi file of {} (run choose.py --target {})",
            family.name,
            family.lambda
        );
    }
    for preset in all_secure_presets().unwrap() {
        let entry = hachi_preset(&preset.name).unwrap().expect("an entry");
        if let Some(h) = entry.hiding {
            assert_eq!(h.lambda, preset.choices.lambda, "{}", preset.name);
        }
    }
}

#[test]
fn every_secure_preset_has_hachi_parameters_of_the_recorded_sizes() {
    for preset in all_secure_presets().unwrap() {
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
        for (len, iterations) in [(len0, entry.iterations.g0), (len1, entry.iterations.g1)] {
            let schedule = pcs.vector_schedule(len).unwrap();
            assert_eq!(schedule.iterations.len(), iterations, "{name}");
            // With masks, no object is sent in the clear.
            if entry.hiding.is_some() {
                assert!(iterations >= 1, "{name}");
                assert!(entry.iterations.w >= 1, "{name}");
            }
        }
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
