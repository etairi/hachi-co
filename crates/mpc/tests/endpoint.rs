//! The n-party endpoint over the in-memory transport: every collective
//! call against its ideal semantics for n in {1, 2, 3, 4, 5, 8} and both zero-sharing modes; the
//! defining property of F_AH (its output equals the map applied to the reconstructed input); the
//! masked broadcasts are uniform whatever the input; the set-up checks; the accounting against
//! its formulas; reproducibility; and the failure boundary (a party that leaves after every
//! possible number of frames, altered frames, out-of-step calls), where no party hangs or panics
//! and no party that finishes returns a wrong value.

use hachi_co_mpc::{
    Comm, Error, Fault, LABEL_SETUP, MemoryConfig, MemoryEndpoint, PartyId, PartyRun, Tap,
    ZeroSharing, run_in_memory,
};
use hachi_co_sampling::ShakePrg;
use hachi_co_sampling::uniform::uniform_mod_q;
use rand::Rng;
use std::time::{Duration, Instant};

/// The toy-n1 modulus (`docs/m0-spec.md`, "Toy presets").
const Q: u64 = 8388637;
/// The largest modulus the layer accepts: sums of two residues come closest to 2^64 here.
const Q_MAX: u64 = (1 << 63) - 1;
const PARTIES: [usize; 6] = [1, 2, 3, 4, 5, 8];
const MODES: [ZeroSharing; 2] = [ZeroSharing::PairwiseSeeds, ZeroSharing::Fresh];

fn seed(tag: &str, index: u64) -> [u8; 32] {
    let mut s = [0u8; 32];
    ShakePrg::new(&[7; 32], tag.as_bytes(), index).fill_bytes(&mut s);
    s
}

fn residues(tag: &str, index: u64, q: u64, len: usize) -> Vec<u64> {
    uniform_mod_q(
        &mut ShakePrg::new(&seed(tag, index), b"residues", 0),
        q,
        len,
    )
    .unwrap()
}

fn add(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    a.iter()
        .zip(b)
        .map(|(x, y)| ((u128::from(*x) + u128::from(*y)) % u128::from(q)) as u64)
        .collect()
}

/// `L x` for the row-major `rows x x.len()` matrix `l`.
fn apply(l: &[u64], x: &[u64], q: u64) -> Vec<u64> {
    l.chunks(x.len())
        .map(|row| {
            let s = row
                .iter()
                .zip(x)
                .map(|(a, b)| u128::from(*a) * u128::from(*b) % u128::from(q))
                .sum::<u128>();
            (s % u128::from(q)) as u64
        })
        .collect()
}

fn outputs<T: Clone>(runs: Vec<hachi_co_mpc::Result<PartyRun<T>>>) -> Vec<T> {
    runs.into_iter()
        .map(|r| r.expect("party connected").output)
        .collect()
}

fn config(n: usize, q: u64, mode: ZeroSharing, tag: &str) -> MemoryConfig {
    MemoryConfig::new(n, q, seed(tag, n as u64))
        .zero_sharing(mode)
        .timeout(Duration::from_secs(20))
}

#[test]
fn fah_outputs_the_map_of_the_reconstructed_input() {
    // F_AH(L) on additive shares x_p of x gives L(x) at every party.
    for q in [Q, Q_MAX] {
        for n in PARTIES {
            for mode in MODES {
                let (m_in, m_out) = (7, 5);
                let l = residues("L", n as u64, q, m_in * m_out);
                let shares: Vec<Vec<u64>> = (1..=n as u64)
                    .map(|p| residues("x", 100 * n as u64 + p, q, m_in))
                    .collect();
                let x = shares.iter().fold(vec![0; m_in], |acc, s| add(&acc, s, q));
                let want = apply(&l, &x, q);
                let runs = run_in_memory(&config(n, q, mode, "fah"), |ep| {
                    let local = apply(&l, &shares[ep.id() - 1], q);
                    // An unreduced entry stands for its residue class.
                    let mut unreduced = local.clone();
                    unreduced[0] += q;
                    (
                        ep.fah("L", &local).unwrap(),
                        ep.fah("L", &unreduced).unwrap(),
                    )
                })
                .unwrap();
                for (p, (a, b)) in outputs(runs).into_iter().enumerate() {
                    assert_eq!(a, want, "q {q}, n {n}, {mode:?}, party {}", p + 1);
                    assert_eq!(
                        b,
                        want,
                        "q {q}, n {n}, {mode:?}, party {} (unreduced)",
                        p + 1
                    );
                }
            }
        }
    }
}

/// What every party gets from one session of each call kind.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Session {
    gathered: Option<Vec<Vec<i64>>>,
    any_false: bool,
    any_one: bool,
    exchanged: Vec<Vec<u64>>,
    all: Vec<Vec<u64>>,
    sum: Vec<u64>,
}

fn gather_input(p: PartyId) -> Vec<i64> {
    vec![-(p as i64), i64::MIN, i64::MAX, p as i64 * 1000]
}

fn exchange_input(p: PartyId, j: PartyId) -> Vec<u64> {
    // Different lengths per pair, values not reduced: exchange delivers them unchanged.
    (0..(p + j) % 4)
        .map(|k| (p * 100 + j * 10 + k) as u64 | 1 << 63)
        .collect()
}

fn session(ep: &mut MemoryEndpoint, to: PartyId) -> hachi_co_mpc::Result<Session> {
    let (n, p) = (ep.n(), ep.id());
    let gathered = ep.gather("gather", to, &gather_input(p))?;
    let any_false = ep.any("any/false", false)?;
    let any_one = ep.any("any/one", p == n)?;
    let outgoing = (1..=n).map(|j| exchange_input(p, j)).collect();
    let exchanged = ep.exchange("exchange", outgoing)?;
    let all = ep.all_gather("all", &[p as u64, u64::MAX - p as u64])?;
    let sum = ep.fah("fah", &[p as u64; 3])?;
    Ok(Session {
        gathered,
        any_false,
        any_one,
        exchanged,
        all,
        sum,
    })
}

fn ideal_session(n: usize, p: PartyId, to: PartyId) -> Session {
    let total = (n * (n + 1) / 2) as u64 % Q;
    Session {
        gathered: (p == to).then(|| (1..=n).map(gather_input).collect()),
        any_false: false,
        any_one: true,
        exchanged: (1..=n).map(|j| exchange_input(j, p)).collect(),
        all: (1..=n)
            .map(|j| vec![j as u64, u64::MAX - j as u64])
            .collect(),
        sum: vec![total; 3],
    }
}

#[test]
fn every_call_has_its_ideal_semantics() {
    for n in PARTIES {
        for mode in MODES {
            for to in [1, n] {
                let runs =
                    run_in_memory(&config(n, Q, mode, "calls"), |ep| session(ep, to)).unwrap();
                for (i, out) in outputs(runs).into_iter().enumerate() {
                    let p = i + 1;
                    assert_eq!(
                        out.unwrap(),
                        ideal_session(n, p, to),
                        "n {n}, {mode:?}, to {to}, party {p}"
                    );
                }
            }
        }
    }
}

#[test]
fn invalid_calls_are_errors() {
    let runs = run_in_memory(&config(3, Q, ZeroSharing::PairwiseSeeds, "invalid"), |ep| {
        let gather = ep.gather("g", 4, &[1]);
        let zero = ep.gather("g", 0, &[1]);
        let exchange = ep.exchange("x", vec![vec![1]; 2]);
        (gather, zero, exchange)
    })
    .unwrap();
    for (g, z, x) in outputs(runs) {
        assert_eq!(g, Err(Error::InvalidParty { party: 4, n: 3 }));
        assert_eq!(z, Err(Error::InvalidParty { party: 0, n: 3 }));
        assert_eq!(
            x,
            Err(Error::WrongMessageCount {
                label: "x",
                expected: 3,
                actual: 2
            })
        );
    }
    let bad = MemoryConfig::new(2, Q, [0; 32]).fault(3, Fault::Extend { frame: 0 });
    assert!(matches!(
        run_in_memory(&bad, |_| ()),
        Err(Error::InvalidParty { party: 3, n: 2 })
    ));
    assert!(run_in_memory(&MemoryConfig::new(0, Q, [0; 32]), |_| ()).is_err());
    for q in [0, 1, 1 << 63, u64::MAX] {
        let runs = run_in_memory(&MemoryConfig::new(2, q, [0; 32]), |_| ()).unwrap();
        for r in runs {
            assert_eq!(r.err(), Some(Error::InvalidModulus(q)));
        }
    }
}

/// Pearson's statistic of `counts` against the uniform distribution.
fn chi_square(counts: &[u64]) -> f64 {
    let total: u64 = counts.iter().sum();
    let expected = total as f64 / counts.len() as f64;
    counts
        .iter()
        .map(|&c| (c as f64 - expected).powi(2) / expected)
        .sum()
}

#[test]
fn masked_broadcasts_are_uniform_whatever_the_input() {
    // Party 1's broadcast m_1 = y_1 + z_1 must be uniform in Z_q for every input y_1. With
    // q = 13 and 13 * 400 samples, Pearson's statistic has 12 degrees of freedom; 32.91 is its
    // 0.999 quantile, so each of the 2 x 2 x 2 checks fails by chance with probability 0.001
    // (the seeds are fixed, so the outcome is reproducible).
    const SMALL_Q: u64 = 13;
    const CALLS: usize = 52;
    const LEN: usize = 100;
    for mode in MODES {
        for n in [2, 3] {
            for input in [0u64, 7] {
                let tap = Tap::new();
                let cfg = config(n, SMALL_Q, mode, "uniform").tap(&tap);
                let runs = run_in_memory(&cfg, |ep| {
                    for _ in 0..CALLS {
                        let local = if ep.id() == 1 {
                            vec![input; LEN]
                        } else {
                            vec![0; LEN]
                        };
                        ep.fah("uniform", &local).unwrap();
                    }
                })
                .unwrap();
                assert!(runs.iter().all(Result::is_ok));
                let mut counts = [0u64; SMALL_Q as usize];
                let rounds_per_call = if mode == ZeroSharing::Fresh { 2 } else { 1 };
                for r in tap.records() {
                    // The broadcast is the last round of each call; with Fresh the first round
                    // carries the zero shares, which are uniform by construction.
                    let broadcast = r.frame.round % rounds_per_call == 0;
                    if r.from == 1 && r.to == 2 && r.frame.label == "uniform" && broadcast {
                        for &m in &r.frame.payload {
                            counts[m as usize] += 1;
                        }
                    }
                }
                assert_eq!(counts.iter().sum::<u64>(), (CALLS * LEN) as u64);
                let stat = chi_square(&counts);
                assert!(
                    stat < 32.91,
                    "{mode:?}, n {n}, input {input}: chi-square {stat:.2}, counts {counts:?}"
                );
            }
        }
    }
}

#[test]
fn the_set_up_checks_q_mode_and_n() {
    // Two endpoints that disagree on q or on the zero-sharing mode do not connect.
    use hachi_co_mpc::{Endpoint, Faulty, memory_network};
    let cases: [(u64, u64, ZeroSharing, ZeroSharing); 2] = [
        (
            Q,
            Q + 2,
            ZeroSharing::PairwiseSeeds,
            ZeroSharing::PairwiseSeeds,
        ),
        (Q, Q, ZeroSharing::PairwiseSeeds, ZeroSharing::Fresh),
    ];
    for (q1, q2, m1, m2) in cases {
        let mut ends = memory_network(2, Duration::from_secs(5), None).unwrap();
        let b = ends.pop().unwrap();
        let a = ends.pop().unwrap();
        let (ra, rb) = std::thread::scope(|s| {
            let ha = s.spawn(|| Endpoint::connect(Faulty::honest(a), q1, m1, [1; 32]).map(|_| ()));
            let hb = s.spawn(|| Endpoint::connect(Faulty::honest(b), q2, m2, [2; 32]).map(|_| ()));
            (ha.join().unwrap(), hb.join().unwrap())
        });
        for r in [ra, rb] {
            assert!(
                matches!(
                    r,
                    Err(Error::Malformed {
                        label: LABEL_SETUP,
                        ..
                    })
                ),
                "{r:?}"
            );
        }
    }
}

#[test]
fn accounting_matches_the_formulas() {
    // Per party and call: fah (n-1)*8*len bytes each way (twice with Fresh, in two rounds);
    // all_gather the same; gather 8*len from every party but `to`; any (n-1)*8; exchange
    // 8 * sum_{j != p} len_{p,j}. The set-up sends 3 words to every party, plus 4 to every
    // higher-numbered party with pairwise seeds.
    for n in PARTIES {
        for mode in MODES {
            let to = 1;
            let runs = run_in_memory(&config(n, Q, mode, "stats"), |ep| {
                session(ep, to).map(|_| ())
            })
            .unwrap();
            for run in runs {
                let run = run.unwrap();
                run.output.unwrap();
                let (p, s) = (run.id, &run.stats);
                let n1 = (n - 1) as u64;
                let fresh = u64::from(mode == ZeroSharing::Fresh);
                let check = |label: &str, rounds, sent, received| {
                    let l = s.label(label);
                    assert_eq!(l.calls, 1, "n {n}, {mode:?}, party {p}, {label}");
                    assert_eq!(
                        (l.rounds, l.bytes_sent, l.bytes_received),
                        (rounds, sent, received),
                        "n {n}, {mode:?}, party {p}, {label}"
                    );
                };
                let r = u64::from(n > 1);
                check(
                    "fah",
                    r * (1 + fresh),
                    n1 * 8 * 3 * (1 + fresh),
                    n1 * 8 * 3 * (1 + fresh),
                );
                check("all", r, n1 * 16, n1 * 16);
                check("any/false", r, n1 * 8, n1 * 8);
                let gathered = if p == to { 0 } else { 32 };
                check("gather", r, gathered, if p == to { n1 * 32 } else { 0 });
                let ex_sent: u64 = (1..=n)
                    .filter(|&j| j != p)
                    .map(|j| 8 * exchange_input(p, j).len() as u64)
                    .sum();
                let ex_received: u64 = (1..=n)
                    .filter(|&j| j != p)
                    .map(|j| 8 * exchange_input(j, p).len() as u64)
                    .sum();
                check("exchange", r, ex_sent, ex_received);
                let setup = s.label(LABEL_SETUP);
                let higher = (n - p) as u64;
                let lower = (p - 1) as u64;
                let seeds = u64::from(mode == ZeroSharing::PairwiseSeeds);
                assert_eq!(
                    setup.bytes_sent,
                    8 * (3 * n1 + 4 * seeds * higher),
                    "n {n}, {mode:?}, party {p}"
                );
                assert_eq!(
                    setup.bytes_received,
                    8 * (3 * n1 + 4 * seeds * lower),
                    "n {n}, {mode:?}, party {p}"
                );
                assert_eq!(s.total().calls, 7, "six calls and the set-up");
            }
        }
    }
}

#[test]
fn runs_are_reproducible_from_the_seed() {
    for mode in MODES {
        let frames = |tag: &str| {
            let tap = Tap::new();
            run_in_memory(&config(4, Q, mode, tag).tap(&tap), |ep| session(ep, 2)).unwrap();
            let mut records: Vec<_> = tap
                .records()
                .into_iter()
                .map(|r| {
                    (
                        r.from,
                        r.to,
                        r.frame.round,
                        r.frame.label.to_string(),
                        r.frame.payload,
                    )
                })
                .collect();
            // Thread scheduling changes only the order in which frames are recorded.
            records.sort();
            records
        };
        assert_eq!(frames("same"), frames("same"), "{mode:?}");
        assert_ne!(frames("same"), frames("other"), "{mode:?}");
    }
}

/// The fault-free outputs of [`session`] at n parties.
fn reference(n: usize, mode: ZeroSharing) -> Vec<Session> {
    let runs = run_in_memory(&config(n, Q, mode, "faults"), |ep| session(ep, 1)).unwrap();
    outputs(runs).into_iter().map(Result::unwrap).collect()
}

#[test]
fn a_party_that_leaves_never_makes_the_others_hang_or_err_silently() {
    // Party `faulty` leaves after k frames, for every k up to the number of frames it sends in a
    // fault-free session. It always gets an error; every other party either finishes with the
    // fault-free output or gets an error, and none waits for the timeout.
    for n in [2, 3, 4] {
        for mode in MODES {
            let reference = reference(n, mode);
            for faulty in [1, n] {
                let frames = {
                    let runs = run_in_memory(&config(n, Q, mode, "faults"), |ep| {
                        session(ep, 1).map(|_| ())
                    })
                    .unwrap();
                    runs[faulty - 1].as_ref().unwrap().stats.total().frames_sent as usize
                };
                for k in 0..frames {
                    let cfg = config(n, Q, mode, "faults")
                        .timeout(Duration::from_secs(10))
                        .fault(faulty, Fault::Leave { after_frames: k });
                    let start = Instant::now();
                    let runs = run_in_memory(&cfg, |ep| session(ep, 1)).unwrap();
                    assert!(
                        start.elapsed() < Duration::from_secs(5),
                        "n {n}, {mode:?}, party {faulty} leaves after {k}: took {:?}",
                        start.elapsed()
                    );
                    let mut someone_failed = false;
                    for (i, run) in runs.into_iter().enumerate() {
                        let p = i + 1;
                        let result = run.and_then(|r| r.output);
                        if let Err(Error::Timeout { .. }) = result {
                            panic!(
                                "n {n}, {mode:?}, party {faulty} leaves after {k}: party {p} timed out"
                            );
                        }
                        match result {
                            Ok(out) => {
                                assert_ne!(
                                    p, faulty,
                                    "n {n}, {mode:?}: the leaving party finished (k = {k})"
                                );
                                assert_eq!(
                                    out, reference[i],
                                    "n {n}, {mode:?}, k {k}: party {p} finished with another output"
                                );
                            }
                            Err(_) => someone_failed = true,
                        }
                    }
                    assert!(someone_failed);
                }
            }
        }
    }
}

#[test]
fn altered_frames_are_detected_where_the_layer_can_see_them() {
    // Frame 0 of party 2 after the set-up is its first frame of `gather` (to party 1).
    let n = 3;
    let mode = ZeroSharing::PairwiseSeeds;
    let setup_frames = n - 1;
    // Faults on the first frame of `fah` (the last call of a session): the broadcast of party 2.
    let session_frames = {
        let runs = run_in_memory(&config(n, Q, mode, "alter"), |ep| {
            session(ep, 1).map(|_| ())
        })
        .unwrap();
        runs[1].as_ref().unwrap().stats.total().frames_sent as usize
    };
    let fah_frame = session_frames - (n - 1);
    let gather_frame = setup_frames;
    let cases: [(Fault, &str); 5] = [
        (Fault::Truncate { frame: fah_frame }, "length"),
        (Fault::Extend { frame: fah_frame }, "length"),
        (
            Fault::Relabel {
                frame: gather_frame,
            },
            "desync",
        ),
        (Fault::NextRound { frame: fah_frame }, "desync"),
        (
            Fault::Extend {
                frame: setup_frames - 1,
            },
            "malformed",
        ),
    ];
    for (fault, kind) in cases {
        let cfg = config(n, Q, mode, "alter").fault(2, fault.clone());
        let runs = run_in_memory(&cfg, |ep| session(ep, 1)).unwrap();
        let errors: Vec<Error> = runs
            .into_iter()
            .filter_map(|r| r.and_then(|r| r.output).err())
            .collect();
        let seen = errors.iter().any(|e| match kind {
            "length" => matches!(e, Error::LengthMismatch { label: "fah", .. }),
            "desync" => matches!(e, Error::Desync { party: 2, .. }),
            _ => matches!(
                e,
                Error::Malformed {
                    label: LABEL_SETUP,
                    party: 2,
                    ..
                }
            ),
        });
        assert!(seen, "{fault:?}: expected a {kind} error, got {errors:?}");
    }
    // A flipped bit in a masked broadcast is not detectable by the layer (the model is
    // semi-honest): the receiving party ends with another sum, the others are unaffected.
    let reference = reference(n, mode);
    let cfg = config(n, Q, mode, "faults").fault(
        2,
        Fault::FlipBit {
            frame: fah_frame,
            word: 1,
        },
    );
    let out: Vec<Session> = outputs(run_in_memory(&cfg, |ep| session(ep, 1)).unwrap())
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert_ne!(
        out[0].sum, reference[0].sum,
        "party 1 received the flipped frame"
    );
    assert_eq!(out[1], reference[1]);
    assert_eq!(out[2], reference[2]);
}

#[test]
fn out_of_step_parties_get_desync_errors() {
    let runs = run_in_memory(&config(3, Q, ZeroSharing::PairwiseSeeds, "step"), |ep| {
        if ep.id() == 2 {
            ep.any("other", false).map(|_| Vec::new())
        } else {
            ep.fah("this", &[1, 2])
        }
    })
    .unwrap();
    let errors: Vec<_> = outputs(runs).into_iter().filter_map(Result::err).collect();
    assert!(
        errors.iter().any(|e| matches!(e, Error::Desync { .. })),
        "{errors:?}"
    );
    // A party that skips a call: the others see the next call's frame.
    let runs = run_in_memory(&config(3, Q, ZeroSharing::PairwiseSeeds, "skip"), |ep| {
        if ep.id() != 3 {
            ep.any("first", false)?;
        }
        ep.any("second", false)
    })
    .unwrap();
    let errors: Vec<_> = outputs(runs).into_iter().filter_map(Result::err).collect();
    assert!(
        errors.iter().any(|e| matches!(e, Error::Desync { .. })),
        "{errors:?}"
    );
}

#[test]
fn a_panicking_party_is_propagated_without_hanging_the_others() {
    let start = Instant::now();
    let result = std::panic::catch_unwind(|| {
        run_in_memory(&config(3, Q, ZeroSharing::PairwiseSeeds, "panic"), |ep| {
            if ep.id() == 2 {
                panic!("party 2 panics");
            }
            ep.any("x", false)
        })
    });
    assert!(result.is_err());
    assert!(start.elapsed() < Duration::from_secs(5));
}

#[test]
fn a_silent_party_gives_a_timeout() {
    let cfg =
        config(2, Q, ZeroSharing::PairwiseSeeds, "silent").timeout(Duration::from_millis(200));
    let runs = run_in_memory(&cfg, |ep| {
        if ep.id() == 2 {
            std::thread::sleep(Duration::from_millis(600));
            Ok(false)
        } else {
            ep.any("x", false)
        }
    })
    .unwrap();
    let out = outputs(runs);
    assert!(
        matches!(
            out[0],
            Err(Error::Timeout {
                party: 2,
                millis: 200,
                ..
            })
        ),
        "{:?}",
        out[0]
    );
}
