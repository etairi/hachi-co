//! Properties of the transcript: framing, operation separation, forks, the attempt counter,
//! statement changes reaching every challenge, ratcheting, and exact small-range challenges.
//! "Every challenge changes" is checked on 32-byte challenges and on u64 challenges with
//! $n = 2^{64} - 1$, where an accidental equality has probability $2^{-64}$ or less.

use hachi_co_transcript::{Error, Transcript};
use std::collections::BTreeSet;

/// A protocol-shaped run: the statement, an optional attempt, a first message, then five
/// challenges interleaved with messages. Returns the challenges (u64 ones as 8 bytes).
fn run(domain: &[u8], statement: &[u8], attempt: Option<u64>) -> Vec<Vec<u8>> {
    let mut t = Transcript::new(domain);
    t.append_message(b"pp", b"public parameters");
    t.append_message(b"statement", statement);
    if let Some(i) = attempt {
        t.append_attempt(i);
    }
    t.append_message(b"v", b"first message");
    let mut out = vec![t.challenge_seed(b"c").to_vec()];
    out.push(t.challenge_seed(b"c").to_vec());
    t.append_message(b"round", b"1");
    let mut alpha = [0u8; 48];
    t.challenge_bytes(b"alpha", &mut alpha);
    out.push(alpha.to_vec());
    out.push(
        t.challenge_u64_below(b"index", u64::MAX)
            .unwrap()
            .to_le_bytes()
            .to_vec(),
    );
    t.append_u64(b"round", 2);
    out.push(t.challenge_seed(b"final").to_vec());
    out
}

/// Assert that every challenge of `a` differs from the challenge at the same position of `b`.
fn all_differ(a: &[Vec<u8>], b: &[Vec<u8>], what: &str) {
    assert_eq!(a.len(), b.len());
    for (k, (x, y)) in a.iter().zip(b).enumerate() {
        assert_ne!(x, y, "{what}: challenge {k} unchanged");
    }
}

#[test]
fn the_label_message_split_matters() {
    let digests: BTreeSet<[u8; 32]> = [
        (&b"ab"[..], &b"c"[..]),
        (b"a", b"bc"),
        (b"", b"abc"),
        (b"abc", b""),
    ]
    .into_iter()
    .map(|(label, message)| {
        let mut t = Transcript::new(b"framing");
        t.append_message(label, message);
        t.digest()
    })
    .collect();
    assert_eq!(digests.len(), 4);
    // Two messages differ from their concatenation.
    let mut two = Transcript::new(b"framing");
    two.append_message(b"m", b"ab");
    two.append_message(b"m", b"c");
    let mut one = Transcript::new(b"framing");
    one.append_message(b"m", b"abc");
    assert_ne!(two, one);
}

#[test]
fn operations_are_separated() {
    let base = Transcript::new(b"ops");
    let mut message = base.clone();
    message.append_message(b"attempt", &5u64.to_le_bytes());
    let mut attempt = base.clone();
    attempt.append_attempt(5);
    let fork = base.fork(b"attempt");
    let mut empty = base.clone();
    empty.append_message(b"attempt", b"");
    let states: BTreeSet<[u8; 32]> = [
        message.digest(),
        attempt.digest(),
        fork.digest(),
        empty.digest(),
        base.digest(),
    ]
    .into();
    assert_eq!(
        states.len(),
        5,
        "message, attempt, fork and the base must all differ"
    );
    // append_u64 is a message holding LE64(value).
    let mut u = base.clone();
    u.append_u64(b"attempt", 5);
    assert_eq!(u, message);
    // The domain is framed too.
    assert_ne!(Transcript::new(b"ab"), Transcript::new(b"a"));
}

#[test]
fn forks_are_independent() {
    let mut parent = Transcript::new(b"forks");
    parent.append_message(b"statement", b"s");
    let before = parent.clone();
    let mut p1 = parent.fork(b"prover-1");
    let mut p2 = parent.fork(b"prover-2");
    // The parent is untouched, and a fork is deterministic.
    assert_eq!(parent, before);
    assert_eq!(parent.fork(b"prover-1"), p1);
    // Distinct labels, and the parent, give distinct states and challenges.
    assert_ne!(p1, p2);
    assert_ne!(p1, parent);
    for t in [&mut p1, &mut p2, &mut parent] {
        t.append_message(b"z", b"same message");
    }
    let c1 = p1.challenge_seed(b"c");
    let c2 = p2.challenge_seed(b"c");
    let c0 = parent.challenge_seed(b"c");
    assert!(c1 != c2 && c1 != c0 && c2 != c0);
    // What a child does never reaches the parent.
    let mut q = before.clone();
    let mut child = q.fork(b"prover-1");
    child.append_message(b"z", b"anything");
    let _ = child.challenge_seed(b"c");
    assert_eq!(q, before);
    q.append_message(b"z", b"same message");
    assert_eq!(q.challenge_seed(b"c"), c0);
    // Nested forks differ from forks of the parent with the same label.
    assert_ne!(p1.fork(b"inner"), parent.fork(b"inner"));
}

#[test]
fn the_attempt_counter_changes_every_challenge() {
    let attempts: Vec<u64> = (0..16).chain([1 << 32, u64::MAX]).collect();
    let runs: Vec<Vec<Vec<u8>>> = attempts
        .iter()
        .map(|i| run(b"hachi-co", b"stmt", Some(*i)))
        .collect();
    let without = run(b"hachi-co", b"stmt", None);
    for (a, ra) in attempts.iter().zip(&runs) {
        all_differ(ra, &without, &format!("attempt {a} vs none"));
        for (b, rb) in attempts.iter().zip(&runs) {
            if a < b {
                all_differ(ra, rb, &format!("attempt {a} vs {b}"));
            }
        }
    }
}

#[test]
fn any_change_of_the_statement_changes_every_challenge() {
    let statement: Vec<u8> = (0..64).collect();
    let base = run(b"hachi-co", &statement, Some(0));
    // Every bit of every byte.
    for pos in 0..statement.len() {
        for bit in 0..8 {
            let mut s = statement.clone();
            s[pos] ^= 1 << bit;
            all_differ(
                &run(b"hachi-co", &s, Some(0)),
                &base,
                &format!("byte {pos} bit {bit}"),
            );
        }
    }
    // Truncation, extension, the empty statement, and the domain.
    all_differ(
        &run(b"hachi-co", &statement[..63], Some(0)),
        &base,
        "truncated",
    );
    let mut longer = statement.clone();
    longer.push(0);
    all_differ(&run(b"hachi-co", &longer, Some(0)), &base, "extended");
    all_differ(&run(b"hachi-co", b"", Some(0)), &base, "empty");
    all_differ(&run(b"hachi-co-v2", &statement, Some(0)), &base, "domain");
    // The same statement reproduces the same challenges.
    assert_eq!(run(b"hachi-co", &statement, Some(0)), base);
}

#[test]
fn challenges_ratchet() {
    let mut t = Transcript::new(b"ratchet");
    let a = t.challenge_seed(b"c");
    let b = t.challenge_seed(b"c");
    assert_ne!(a, b, "two challenges in a row must differ");
    // challenge_seed is challenge_bytes with 32 bytes.
    let mut u = Transcript::new(b"ratchet");
    let mut v = u.clone();
    let mut bytes = [0u8; 32];
    u.challenge_bytes(b"c", &mut bytes);
    assert_eq!(bytes, v.challenge_seed(b"c"));
    assert_eq!(u, v);
    // The output length is bound: 16 bytes are not a prefix of 32, and the states differ.
    let mut w = Transcript::new(b"ratchet");
    let mut short = [0u8; 16];
    w.challenge_bytes(b"c", &mut short);
    assert_ne!(&short[..], &bytes[..16]);
    assert_ne!(w, u);
    // A zero-length challenge still advances the state.
    let mut z = Transcript::new(b"ratchet");
    let start = z.clone();
    z.challenge_bytes(b"c", &mut []);
    assert_ne!(z, start);
}

#[test]
fn small_range_challenges_are_in_range_and_uniform() {
    let mut t = Transcript::new(b"u64-below");
    let before = t.clone();
    assert_eq!(t.challenge_u64_below(b"x", 0), Err(Error::EmptyRange));
    assert_eq!(t, before, "an error leaves the state unchanged");
    assert_eq!(t.challenge_u64_below(b"x", 1), Ok(0));
    assert_ne!(t, before, "n = 1 still advances the state");
    for n in [2u64, 3, 5, 1 << 32, (1 << 63) + 1, u64::MAX] {
        for _ in 0..200 {
            assert!(t.challenge_u64_below(b"x", n).unwrap() < n);
        }
    }
    // Chi-square over n = 6 with 60000 challenges; threshold 25.7448 is the 1 - 1e-4 quantile of
    // chi-square with 5 degrees of freedom (scipy 1.17.1, chi2.ppf(1 - 1e-4, 5)).
    let mut counts = [0u64; 6];
    for _ in 0..60_000 {
        counts[t.challenge_u64_below(b"die", 6).unwrap() as usize] += 1;
    }
    let stat: f64 = counts
        .iter()
        .map(|c| (*c as f64 - 10_000.0).powi(2) / 10_000.0)
        .sum();
    println!("chi2 challenge_u64_below(6): {stat:.2} (df 5, threshold 25.7448)");
    assert!(stat < 25.7448, "chi2 = {stat}");
}
