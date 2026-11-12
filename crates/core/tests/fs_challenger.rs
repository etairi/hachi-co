//! `FsChallenger`: deterministic, sensitive to every absorbed byte, label and the domain; the
//! provers' and the verifier's calls of the ring-to-linear step agree; challenges in C; canonical
//! field coins.

use hachi_co_core::{Challenger, Error, Ext, Fqk, FsChallenger, Poly, SumcheckCoins};
use hachi_co_sampling::l1ball;

const K: usize = 4;
/// The toy-n2 modulus (`docs/m0-spec.md`, "Toy presets").
const Q: u64 = 33554501;
const DOMAIN: &[u8] = b"hachi-co core test";
/// Toy-n2 dimensions: 2^r = 8 challenges of d = 16 coefficients, omega = 2, l'' = 15,
/// ceil(log n) = 1, ceil(log nu) = 14.
const COUNT: usize = 8;
const D: usize = 16;
const OMEGA: u64 = 2;
const ELL_PP: usize = 15;
const LOG_N: usize = 1;
const LOG_NU: usize = 14;

fn field() -> Fqk<K> {
    Fqk::new(Q).unwrap()
}

/// A fixed sequence of absorbs.
#[derive(Clone, Debug)]
struct Script {
    domain: Vec<u8>,
    labels: [&'static str; 3],
    polys: Vec<Poly>,
    exts: Vec<Ext<K>>,
    value: u64,
}

type Outcome = (Vec<Vec<i64>>, SumcheckCoins<K>, Ext<K>, [u8; 32]);

impl Script {
    fn base() -> Self {
        Self {
            domain: DOMAIN.to_vec(),
            labels: ["test/polys", "test/exts", "test/u64"],
            polys: vec![Poly(vec![1, 2, 3]), Poly(vec![Q - 1, 0, 7])],
            exts: vec![Ext([5, 6, 7, 8]), Ext([Q - 2, 0, 1, 2])],
            value: 0x0123_4567_89ab_cdef,
        }
    }

    /// Absorb everything, then draw the ring-to-linear challenges, the sumcheck coins and a round
    /// coin.
    fn run(&self) -> Outcome {
        let f = field();
        let mut ch = FsChallenger::new(&self.domain);
        Challenger::<K>::absorb_polys(&mut ch, self.labels[0], &self.polys);
        ch.absorb_exts(self.labels[1], &self.exts);
        Challenger::<K>::absorb_u64(&mut ch, self.labels[2], self.value);
        let c = Challenger::<K>::r2l_challenges(&mut ch, COUNT, D, OMEGA).unwrap();
        let coins = ch.sumcheck_coins(&f, ELL_PP, LOG_N, LOG_NU).unwrap();
        let a1 = ch.round_coin(&f, 1).unwrap();
        (c, coins, a1, ch.transcript().digest())
    }
}

fn assert_all_differ(a: &Outcome, b: &Outcome, what: &str) {
    assert_ne!(a.0, b.0, "{what}: challenges");
    assert_ne!(a.1.alpha, b.1.alpha, "{what}: alpha");
    assert_ne!(a.1.tau0, b.1.tau0, "{what}: tau_0");
    assert_ne!(a.2, b.2, "{what}: round coin");
    assert_ne!(a.3, b.3, "{what}: state");
}

#[test]
fn deterministic() {
    let s = Script::base();
    assert_eq!(s.run(), s.run());
    assert_eq!(s.clone().run(), Script::base().run());
}

#[test]
fn sensitive_to_every_absorbed_byte() {
    let base = Script::base();
    let reference = base.run();
    for i in 0..base.polys.len() {
        for j in 0..base.polys[i].0.len() {
            for byte in 0..8 {
                let mut s = base.clone();
                s.polys[i].0[j] ^= 1 << (8 * byte);
                assert_all_differ(
                    &reference,
                    &s.run(),
                    &format!("poly {i} coeff {j} byte {byte}"),
                );
            }
        }
    }
    for i in 0..base.exts.len() {
        for j in 0..K {
            for byte in 0..8 {
                let mut s = base.clone();
                s.exts[i].0[j] ^= 1 << (8 * byte);
                assert_all_differ(
                    &reference,
                    &s.run(),
                    &format!("ext {i} coord {j} byte {byte}"),
                );
            }
        }
    }
    for byte in 0..8 {
        let mut s = base.clone();
        s.value ^= 1 << (8 * byte);
        assert_all_differ(&reference, &s.run(), &format!("u64 byte {byte}"));
    }
}

#[test]
fn sensitive_to_every_label_byte_and_the_domain() {
    let base = Script::base();
    let reference = base.run();
    for which in 0..3 {
        let label = base.labels[which];
        for pos in 0..label.len() {
            let mut bytes = label.as_bytes().to_vec();
            bytes[pos] ^= 1;
            let changed: &'static str =
                Box::leak(String::from_utf8(bytes).unwrap().into_boxed_str());
            let mut s = base.clone();
            s.labels[which] = changed;
            assert_all_differ(&reference, &s.run(), &format!("label {which} byte {pos}"));
        }
    }
    for pos in 0..base.domain.len() {
        let mut s = base.clone();
        s.domain[pos] ^= 1;
        assert_all_differ(&reference, &s.run(), &format!("domain byte {pos}"));
    }
}

#[test]
fn framing_separates_messages() {
    let f = field();
    let draw = |ch: &mut FsChallenger| ch.round_coin(&f, 1).unwrap();
    let p = [Poly(vec![1, 2]), Poly(vec![3, 4])];
    // One message of two polynomials vs two messages.
    let mut one = FsChallenger::new(DOMAIN);
    Challenger::<K>::absorb_polys(&mut one, "m", &p);
    let mut two = FsChallenger::new(DOMAIN);
    Challenger::<K>::absorb_polys(&mut two, "m", &p[..1]);
    Challenger::<K>::absorb_polys(&mut two, "m", &p[1..]);
    // The same residues as one polynomial of length 4.
    let mut flat = FsChallenger::new(DOMAIN);
    Challenger::<K>::absorb_polys(&mut flat, "m", &[Poly(vec![1, 2, 3, 4])]);
    // The same residues as a field element.
    let mut ext = FsChallenger::new(DOMAIN);
    ext.absorb_exts("m", &[Ext::<K>([1, 2, 3, 4])]);
    let coins = [
        draw(&mut one),
        draw(&mut two),
        draw(&mut flat),
        draw(&mut ext),
    ];
    for i in 0..4 {
        for j in i + 1..4 {
            assert_ne!(coins[i], coins[j], "{i} vs {j}");
        }
    }
    // Empty messages still count.
    let mut empty = FsChallenger::new(DOMAIN);
    Challenger::<K>::absorb_polys(&mut empty, "m", &[]);
    let mut nothing = FsChallenger::new(DOMAIN);
    assert_ne!(draw(&mut empty), draw(&mut nothing));
}

#[test]
fn the_verifier_reaches_the_provers_state_through_the_ring_to_linear_step() {
    let f = field();
    let v = |x: u64| vec![Poly(vec![x; D])];
    let mut prefix = FsChallenger::new(DOMAIN);
    Challenger::<K>::absorb_polys(&mut prefix, "statement", &v(42));
    // The provers: v, the challenges, the sets of accepted iterations, then coins.
    let sets = vec![vec![0b1011u64], vec![0b0110u64]];
    let mut provers = prefix.clone();
    Challenger::<K>::absorb_polys(&mut provers, "r2l/v", &v(100));
    let c = Challenger::<K>::r2l_challenges(&mut provers, COUNT, D, OMEGA).unwrap();
    Challenger::<K>::absorb_words(&mut provers, "r2l/accept-sets", &sets);
    let after_provers = provers.sumcheck_coins(&f, ELL_PP, LOG_N, LOG_NU).unwrap();
    // The verifier makes the same calls with the proof's v and sets.
    let mut verifier = prefix.clone();
    Challenger::<K>::absorb_polys(&mut verifier, "r2l/v", &v(100));
    assert_eq!(
        Challenger::<K>::r2l_challenges(&mut verifier, COUNT, D, OMEGA).unwrap(),
        c
    );
    Challenger::<K>::absorb_words(&mut verifier, "r2l/accept-sets", &sets);
    assert_eq!(
        verifier.sumcheck_coins(&f, ELL_PP, LOG_N, LOG_NU).unwrap(),
        after_provers
    );
    assert_eq!(verifier.transcript(), provers.transcript());
    // Another v gives other challenges, and other sets give other coins afterwards.
    let mut other_v = prefix.clone();
    Challenger::<K>::absorb_polys(&mut other_v, "r2l/v", &v(101));
    assert_ne!(
        Challenger::<K>::r2l_challenges(&mut other_v, COUNT, D, OMEGA).unwrap(),
        c
    );
    let mut other_sets = prefix;
    Challenger::<K>::absorb_polys(&mut other_sets, "r2l/v", &v(100));
    Challenger::<K>::r2l_challenges(&mut other_sets, COUNT, D, OMEGA).unwrap();
    Challenger::<K>::absorb_words(
        &mut other_sets,
        "r2l/accept-sets",
        &[vec![0b1011], vec![0b0111]],
    );
    assert_ne!(
        other_sets
            .sumcheck_coins(&f, ELL_PP, LOG_N, LOG_NU)
            .unwrap(),
        after_provers
    );
}

#[test]
fn challenges_lie_in_the_l1_ball() {
    let mut ch = FsChallenger::new(DOMAIN);
    let (mut support1, mut support2) = (false, false);
    for a in 0..200u64 {
        Challenger::<K>::absorb_u64(&mut ch, "test/draw", a);
        let c = Challenger::<K>::r2l_challenges(&mut ch, COUNT, D, OMEGA).unwrap();
        assert_eq!(c.len(), COUNT);
        for ci in &c {
            assert_eq!(ci.len(), D);
            assert!(l1ball::contains(ci, OMEGA), "{ci:?}");
            match ci.iter().filter(|x| **x != 0).count() {
                1 => support1 = true,
                2 => support2 = true,
                _ => {}
            }
        }
    }
    assert!(
        support1 && support2,
        "both support sizes appear in 1600 draws"
    );
    // Other radii and degrees, and the empty request.
    for (d, omega) in [(64, 10), (256, 60), (1, 1)] {
        let c = Challenger::<K>::r2l_challenges(&mut ch, 3, d, omega).unwrap();
        assert!(
            c.iter()
                .all(|ci| ci.len() == d && l1ball::contains(ci, omega))
        );
    }
    assert!(
        Challenger::<K>::r2l_challenges(&mut ch, 0, D, OMEGA)
            .unwrap()
            .is_empty()
    );
    assert!(matches!(
        Challenger::<K>::r2l_challenges(&mut ch, 1, D, u64::MAX),
        Err(Error::Sampling(_))
    ));
}

#[test]
fn field_coins_are_canonical() {
    let f = field();
    let mut ch = FsChallenger::new(DOMAIN);
    let coins = ch.sumcheck_coins(&f, ELL_PP, LOG_N, LOG_NU).unwrap();
    assert_eq!(coins.tau0.len(), ELL_PP + LOG_N);
    assert_eq!(coins.tau1.len(), LOG_NU);
    let mut all = vec![coins.alpha, coins.rho];
    all.extend(&coins.tau0);
    all.extend(&coins.tau1);
    let rounds: Vec<Ext<K>> = (1..=ELL_PP)
        .map(|r| ch.round_coin(&f, r).unwrap())
        .collect();
    all.extend(&rounds);
    assert!(all.iter().all(|x| x.0.iter().all(|c| *c < Q)));
    // All distinct (2^100-ish values each).
    for i in 0..all.len() {
        for j in i + 1..all.len() {
            assert_ne!(all[i], all[j], "{i} {j}");
        }
    }
    assert!(matches!(ch.round_coin(&f, 0), Err(Error::Challenger(_))));
    // A tiny field: every coordinate below q = 13.
    let small = Fqk::<K>::new(13).unwrap();
    for r in 1..200 {
        let x = ch.round_coin(&small, r).unwrap();
        assert!(x.0.iter().all(|c| *c < 13));
    }
}

/// `absorb_words`: sensitive to every word and to the split into vectors, and never encoded
/// like a message of polynomials with the same residues.
#[test]
fn word_messages_are_framed_and_sensitive() {
    let f = field();
    let run = |words: &[Vec<u64>]| {
        let mut ch = FsChallenger::new(DOMAIN);
        Challenger::<K>::absorb_words(&mut ch, "commitments", words);
        ch.round_coin(&f, 1).unwrap()
    };
    let base = vec![vec![1u64, 2, 3, 4], vec![u64::MAX, 0]];
    let reference = run(&base);
    assert_eq!(run(&base), reference);
    for i in 0..base.len() {
        for j in 0..base[i].len() {
            for bit in [0, 17, 63] {
                let mut w = base.clone();
                w[i][j] ^= 1 << bit;
                assert_ne!(run(&w), reference, "vector {i} word {j} bit {bit}");
            }
        }
    }
    assert_ne!(run(&[vec![1, 2], vec![3, 4, u64::MAX, 0]]), reference);
    assert_ne!(run(&[vec![1, 2, 3, 4, u64::MAX, 0]]), reference);
    assert_ne!(run(&[]), run(&[vec![]]));
    let mut polys = FsChallenger::new(DOMAIN);
    Challenger::<K>::absorb_polys(
        &mut polys,
        "commitments",
        &[Poly(vec![1, 2, 3, 4]), Poly(vec![u64::MAX, 0])],
    );
    assert_ne!(polys.round_coin(&f, 1).unwrap(), reference);
}

/// `context`: a forked squeeze. It is the seed of the fork of the transcript under the label;
/// deterministic; it depends on every byte of the label and on what was absorbed before; it
/// leaves the transcript, and hence every later coin, unchanged, whichever contexts were derived
/// and in which order.
#[test]
fn contexts_are_forked_squeezes() {
    let f = field();
    let s = Script::base();
    let prefix = |ch: &mut FsChallenger| {
        Challenger::<K>::absorb_polys(ch, s.labels[0], &s.polys);
        ch.absorb_exts(s.labels[1], &s.exts);
    };
    let mut ch = FsChallenger::new(DOMAIN);
    prefix(&mut ch);
    let before = ch.transcript().clone();
    let label = "hachi-co/sumcheck/open/w/p1";
    let c = Challenger::<K>::context(&mut ch, label);
    assert_eq!(ch.transcript(), &before, "the transcript is unchanged");
    assert_eq!(
        c,
        before
            .fork(label.as_bytes())
            .challenge_seed(hachi_co_core::FS_LABEL_CONTEXT.as_bytes())
    );
    assert_eq!(Challenger::<K>::context(&mut ch, label), c);
    for pos in 0..label.len() {
        let mut bytes = label.as_bytes().to_vec();
        bytes[pos] ^= 1;
        let other = String::from_utf8(bytes).unwrap();
        assert_ne!(Challenger::<K>::context(&mut ch, &other), c, "byte {pos}");
    }
    assert_ne!(Challenger::<K>::context(&mut ch, ""), c);
    // What was absorbed before matters.
    let mut other = FsChallenger::new(DOMAIN);
    prefix(&mut other);
    Challenger::<K>::absorb_u64(&mut other, "one more", 0);
    assert_ne!(Challenger::<K>::context(&mut other, label), c);
    // Later coins do not depend on the contexts derived before them, nor on their order.
    let mut plain = FsChallenger::new(DOMAIN);
    prefix(&mut plain);
    let mut forward = plain.clone();
    let mut backward = plain.clone();
    let labels = ["a", "b", "c"];
    let fwd: Vec<[u8; 32]> = labels
        .iter()
        .map(|l| Challenger::<K>::context(&mut forward, l))
        .collect();
    let bwd: Vec<[u8; 32]> = labels
        .iter()
        .rev()
        .map(|l| Challenger::<K>::context(&mut backward, l))
        .collect();
    assert_eq!(fwd, bwd.into_iter().rev().collect::<Vec<_>>());
    let coin = plain.round_coin(&f, 1).unwrap();
    assert_eq!(forward.round_coin(&f, 1).unwrap(), coin);
    assert_eq!(backward.round_coin(&f, 1).unwrap(), coin);
    assert_eq!(forward.transcript(), plain.transcript());
    // A context is not a coin of the main transcript under the same label.
    let mut main = FsChallenger::new(DOMAIN);
    prefix(&mut main);
    let mut t = main.transcript().clone();
    assert_ne!(t.challenge_seed(label.as_bytes()), c);
}
