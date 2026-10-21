//! Tampering: every change of a proof component is rejected with an error, never a panic. The
//! base PCS is the reference `ClearPcs`.
//!
//! - An honest toy-n1 proof under Fiat-Shamir (setup index 7, instance seed 7000, party seed
//!   `seed("party/tamper", [7])`): E(u) (every coefficient), a* (every other value below the cap,
//!   and values at and beyond it), every residue of v, every coordinate of every G value, every
//!   word of every base-PCS commitment, every coordinate of every entry of H0 and H1 in the
//!   first, a middle and the last round, every coordinate of every final value, the words of
//!   every base-PCS opening proof (its context, kind and count, and its first, a middle and its
//!   last entry) and every proof given for another polynomial; another x, y or u;
//!   non-canonical residues and wrong shapes in the struct and in the statement.
//! - The toy-n3 fixture proof with replayed coins (the coins and the contexts do not depend on
//!   the messages, so the verifier's checks themselves must catch every change), for the same
//!   components with three provers, and the openings or commitments of two provers swapped.
//! - An honest toy-n3 proof under Fiat-Shamir on the in-memory network (setup index 7, instance
//!   seed 7300, party seeds `seed("party/tamper/n3", [p])`): every pair of provers' openings,
//!   or commitments, swapped.
//! - Encodings: every truncation, extensions, a wrong magic or version, a non-canonical residue
//!   in every residue word, a changed length of a base-PCS item, a* at and beyond the cap.
//! - A seeded fuzz loop (`ChaCha8Rng` seeds 0xF022_0001..=0xF022_0003): random byte strings,
//!   random canonical bodies, and random mutations of the honest encoding through `from_bytes`
//!   and `verify`; whatever decodes re-encodes to the same bytes, and every decoded proof other
//!   than the honest one is rejected.

mod common;

use common::net::run_parties;
use common::{
    K, Pcs, additive_shares, clear_pcs, clear_words, fixture_proof, fixture_statement,
    fresh_context, load, random_instance, seed,
};
use hachi_co::{
    BasePcsWords, ClaimedPoly, DecodeError, EVAL_DOMAIN, Error, EvalProof, HEADER_LEN, MAGIC,
    ProofError, StatementError, VERSION, VerifyError, commit, encoded_len, eval, eval_with, setup,
    verify, verify_with,
};
use hachi_co_core::{Context, Ext, Fixture, FsChallenger, Poly, PrgRandomness, ReplayChallenger};
use hachi_co_mpc::Loopback;
use hachi_co_r2l::MAX_ATTEMPTS;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::ops::Range;
use std::sync::OnceLock;

/// An honest proof with its statement.
struct Honest {
    ctx: Context<K>,
    pcs: Pcs,
    u: Vec<Poly>,
    x: Vec<Ext<K>>,
    y: Ext<K>,
    proof: EvalProof<K>,
    bytes: Vec<u8>,
}

/// The honest toy-n1 proof under Fiat-Shamir, computed once per test binary.
fn honest() -> &'static Honest {
    static CELL: OnceLock<Honest> = OnceLock::new();
    CELL.get_or_init(|| {
        let ctx = fresh_context("toy-n1", 7);
        let pcs = clear_pcs(&ctx);
        let (f, x, y) = random_instance(&ctx, 7000);
        let mut comm = Loopback::new(ctx.q()).unwrap();
        let mut rng = PrgRandomness::new(seed("party/tamper", &[7]));
        let (u, state) = commit(&ctx, &mut comm, &mut rng, &f).unwrap();
        let proof = eval(&ctx, &pcs, &mut comm, &mut rng, &state, &u, &x, &y).unwrap();
        verify(&ctx, &pcs, &u, &x, &y, &proof).unwrap();
        let bytes = proof.to_bytes();
        Honest {
            ctx,
            pcs,
            u,
            x,
            y,
            proof,
            bytes,
        }
    })
}

/// A verifier for a fixed context: proof and statement in, acceptance or error out.
type Verifier<'a> =
    dyn Fn(&EvalProof<K>, &[Poly], &[Ext<K>], &Ext<K>) -> Result<(), VerifyError> + 'a;

/// $x + 1$ in one coordinate, modulo $q$.
fn bump(x: &mut u64, q: u64) {
    *x = (*x + 1) % q;
}

/// Whether `err` is a rejected opening of `poly` of party `party`.
fn is_opening(err: &VerifyError, party: usize, poly: ClaimedPoly) -> bool {
    matches!(
        err,
        VerifyError::Sumcheck(hachi_co_sumcheck::VerifyError::Opening { party: p, poly: q, .. })
            if *p == party && *q == poly
    )
}

/// Runs every single-component change of the module documentation through `verifier` and
/// requires an error each time; returns the number of changes tried.
fn every_component_change_is_rejected(
    ctx: &Context<K>,
    statement: (&[Poly], &[Ext<K>], &Ext<K>),
    proof: &EvalProof<K>,
    verifier: &Verifier<'_>,
    name: &str,
) -> usize {
    let (u, x, y) = statement;
    let q = ctx.q();
    let l = &ctx.layout;
    assert!(verifier(proof, u, x, y).is_ok(), "{name}: the honest proof");
    let mut tried = 0usize;
    let mut reject = |p: &EvalProof<K>, what: String| -> VerifyError {
        tried += 1;
        assert_ne!(p, proof, "{name}: {what} must change the proof");
        let err = verifier(p, u, x, y).expect_err(&format!("{name}: {what} was accepted"));
        // A tampered E(u) fails Pi_conv's check before anything else.
        if what.starts_with("E(u)") {
            assert!(
                matches!(err, VerifyError::Conv(_)),
                "{name}: {what}: {err:?}"
            );
        }
        err
    };
    for i in 0..l.d {
        let mut p = proof.clone();
        bump(&mut p.e_u.0[i], q);
        reject(&p, format!("E(u)[{i}] + 1"));
    }
    for a in (0..MAX_ATTEMPTS).filter(|a| *a != proof.attempt) {
        let mut p = proof.clone();
        p.attempt = a;
        reject(&p, format!("a* = {a}"));
    }
    for (j, poly) in proof.v.iter().enumerate() {
        for i in 0..poly.0.len() {
            let mut p = proof.clone();
            bump(&mut p.v[j].0[i], q);
            reject(&p, format!("v[{j}][{i}] + 1"));
        }
    }
    for which in 0..2 {
        for party in 0..l.n {
            for c in 0..K {
                let mut p = proof.clone();
                let sums = if which == 0 {
                    &mut p.sumcheck.g0_sums
                } else {
                    &mut p.sumcheck.g1_sums
                };
                bump(&mut sums[party].0[c], q);
                reject(&p, format!("G_{which},{} coordinate {c} + 1", party + 1));
            }
        }
    }
    // Every word of every base-PCS commitment.
    for poly in ClaimedPoly::ORDER {
        for party in 0..l.n {
            for i in 0..proof.sumcheck.commitments.of(poly)[party].len() {
                let mut p = proof.clone();
                let word = &mut p.sumcheck.commitments.of_mut(poly)[party][i];
                *word = word.wrapping_add(1);
                let err = reject(
                    &p,
                    format!("commitment {} of party {} word {i}", poly.name(), party + 1),
                );
                assert!(
                    matches!(
                        err,
                        VerifyError::Sumcheck(
                            hachi_co_sumcheck::VerifyError::Rejected { .. }
                                | hachi_co_sumcheck::VerifyError::Opening { .. }
                        )
                    ),
                    "{name}: commitment: {err:?}"
                );
            }
        }
    }
    let mut rounds = vec![0, l.ell_pp / 2, l.ell_pp - 1];
    rounds.dedup();
    for r in rounds {
        for which in 0..2 {
            let len = [l.d0 + 1, l.d1 + 1][which];
            for t in 0..len {
                for c in 0..K {
                    let mut p = proof.clone();
                    let msg = &mut p.sumcheck.rounds[r];
                    let values = if which == 0 { &mut msg.h0 } else { &mut msg.h1 };
                    bump(&mut values[t].0[c], q);
                    reject(
                        &p,
                        format!("H_{which},{}({}) coordinate {c} + 1", r + 1, t + 1),
                    );
                }
            }
        }
    }
    for kind in 0..3 {
        for party in 0..l.n {
            for c in 0..K {
                let mut p = proof.clone();
                let f = &mut p.sumcheck.finals;
                let values = [&mut f.w, &mut f.g0, &mut f.g1]
                    .into_iter()
                    .nth(kind)
                    .unwrap();
                bump(&mut values[party].0[c], q);
                reject(
                    &p,
                    format!("final {kind} of party {} coordinate {c}", party + 1),
                );
            }
        }
    }
    // The base-PCS opening proofs: the context (words 0 to 3), the kind (4), the count (5), the
    // first, a middle and the last entry. Nothing after the proofs depends on them, so the
    // sumcheck's checks pass and the opening is what fails.
    for poly in ClaimedPoly::ORDER {
        for party in 0..l.n {
            let len = proof.sumcheck.openings.of(poly)[party].len();
            let mut words = vec![0, 3, 4, 5, 6, len / 2, len - 1];
            words.dedup();
            for i in words {
                let mut p = proof.clone();
                let word = &mut p.sumcheck.openings.of_mut(poly)[party][i];
                *word = word.wrapping_add(1);
                let err = reject(
                    &p,
                    format!("opening {} of party {} word {i}", poly.name(), party + 1),
                );
                assert!(
                    is_opening(&err, party + 1, poly),
                    "{name}: opening {} of party {} word {i}: {err:?}",
                    poly.name(),
                    party + 1
                );
            }
            // The proof of another polynomial of the same party.
            let other = match poly {
                ClaimedPoly::W => ClaimedPoly::G0,
                ClaimedPoly::G0 => ClaimedPoly::G1,
                ClaimedPoly::G1 => ClaimedPoly::W,
            };
            let mut p = proof.clone();
            p.sumcheck.openings.of_mut(poly)[party] =
                proof.sumcheck.openings.of(other)[party].clone();
            let err = reject(
                &p,
                format!(
                    "opening of {} given for {} of party {}",
                    other.name(),
                    poly.name(),
                    party + 1
                ),
            );
            assert!(is_opening(&err, party + 1, poly), "{name}: {err:?}");
        }
    }
    // Another statement with the same proof.
    let mut statement_changes = 0;
    for j in 0..x.len() {
        for c in 0..K {
            let mut x2 = x.to_vec();
            bump(&mut x2[j].0[c], q);
            assert!(
                verifier(proof, u, &x2, y).is_err(),
                "{name}: x[{j}][{c}] + 1"
            );
            statement_changes += 1;
        }
    }
    for c in 0..K {
        let mut y2 = *y;
        bump(&mut y2.0[c], q);
        assert!(
            matches!(
                verifier(proof, u, x, &y2),
                Err(VerifyError::Conv(hachi_co_conv::VerifyError::Mismatch))
            ),
            "{name}: y[{c}] + 1"
        );
        statement_changes += 1;
    }
    for j in 0..u.len() {
        for i in 0..l.d {
            let mut u2 = u.to_vec();
            bump(&mut u2[j].0[i], q);
            assert!(
                verifier(proof, &u2, x, y).is_err(),
                "{name}: u[{j}][{i}] + 1"
            );
            statement_changes += 1;
        }
    }
    tried + statement_changes
}

/// Every pair of parties' openings (or commitments) of each polynomial swapped is rejected:
/// a swapped opening fails at the first party of the pair (its context and its commitment are
/// another party's); swapped commitments fail too (with Fiat-Shamir the coins change; with
/// replayed coins the openings no longer match).
fn swapped_items_are_rejected(
    proof: &EvalProof<K>,
    n: usize,
    verifier: &Verifier<'_>,
    statement: (&[Poly], &[Ext<K>], &Ext<K>),
    name: &str,
) -> usize {
    let (u, x, y) = statement;
    let mut tried = 0;
    for poly in ClaimedPoly::ORDER {
        for a in 0..n {
            for b in a + 1..n {
                let mut p = proof.clone();
                p.sumcheck.openings.of_mut(poly).swap(a, b);
                let err = verifier(&p, u, x, y).expect_err(name);
                assert!(
                    is_opening(&err, a + 1, poly),
                    "{name}: openings of {} swapped between parties {} and {}: {err:?}",
                    poly.name(),
                    a + 1,
                    b + 1
                );
                let mut p = proof.clone();
                p.sumcheck.commitments.of_mut(poly).swap(a, b);
                assert!(verifier(&p, u, x, y).is_err(), "{name}: commitments");
                tried += 2;
            }
        }
    }
    tried
}

#[test]
fn tampered_fiat_shamir_proofs_are_rejected() {
    let h = honest();
    let ctx = &h.ctx;
    let verifier =
        |p: &EvalProof<K>, u: &[Poly], x: &[Ext<K>], y: &Ext<K>| verify(ctx, &h.pcs, u, x, y, p);
    let tried = every_component_change_is_rejected(
        ctx,
        (&h.u, &h.x, &h.y),
        &h.proof,
        &verifier,
        "toy-n1 (Fiat-Shamir)",
    );
    eprintln!("toy-n1 (Fiat-Shamir): {tried} tampered proofs and statements rejected");
}

#[test]
fn tampered_fixture_proofs_are_rejected_with_replayed_coins() {
    let (fx, ctx) = load("toy-n3");
    let pcs = clear_pcs(&ctx);
    let (u, x, y) = fixture_statement(&fx);
    let proof = fixture_proof(&fx);
    let verifier = |p: &EvalProof<K>, u: &[Poly], x: &[Ext<K>], y: &Ext<K>| {
        let mut ch = ReplayChallenger::new(&fx).unwrap();
        verify_with(&ctx, &pcs, &mut ch, u, x, y, p)
    };
    let tried = every_component_change_is_rejected(
        &ctx,
        (&u, &x, &y),
        &proof,
        &verifier,
        "toy-n3 (replay)",
    );
    let swapped = swapped_items_are_rejected(&proof, 3, &verifier, (&u, &x, &y), "toy-n3 (replay)");
    eprintln!("toy-n3 (replay): {tried} tampered proofs and statements, {swapped} swaps rejected");
    // With the replayed coins, a changed v or G fails a round check, a changed final value a
    // final check (the oracle's negative tests, `ref/README.md`, "Negative tests"), reported
    // before the opening that fails with it; a changed commitment no longer matches its
    // opening.
    let rejected = |p: &EvalProof<K>| match verifier(p, &u, &x, &y) {
        Err(VerifyError::Sumcheck(hachi_co_sumcheck::VerifyError::Rejected { failing })) => failing,
        other => panic!("{other:?}"),
    };
    let mut p = proof.clone();
    bump(&mut p.v[0].0[0], ctx.q());
    assert_eq!(rejected(&p), ["round_01_H1"]);
    let mut p = proof.clone();
    bump(&mut p.sumcheck.g0_sums[2].0[0], ctx.q());
    assert_eq!(rejected(&p), ["round_01_H0"]);
    let mut p = proof.clone();
    bump(&mut p.sumcheck.finals.g1[1].0[3], ctx.q());
    assert_eq!(rejected(&p), ["final_H1"]);
    let mut p = proof.clone();
    bump(&mut p.sumcheck.finals.w[0].0[0], ctx.q());
    assert_eq!(rejected(&p), ["final_H0", "final_H1"]);
    for poly in ClaimedPoly::ORDER {
        let mut p = proof.clone();
        p.sumcheck.commitments.of_mut(poly)[1][2] ^= 1 << 40;
        match verifier(&p, &u, &x, &y) {
            Err(VerifyError::Sumcheck(hachi_co_sumcheck::VerifyError::Opening {
                party: 2,
                poly: got,
                reason,
            })) if got == poly => assert!(reason.contains("does not match"), "{reason}"),
            other => panic!("{}: {other:?}", poly.name()),
        }
    }
}

/// Openings and commitments swapped between every pair of parties of an honest toy-n3 proof
/// under Fiat-Shamir, made on the in-memory network: all rejected.
#[test]
fn swapped_openings_between_parties_are_rejected() {
    let ctx = fresh_context("toy-n3", 7);
    let pcs = clear_pcs(&ctx);
    let (f, x, y) = random_instance(&ctx, 7300);
    let mut rng = ChaCha8Rng::seed_from_u64(7301);
    let f_init = additive_shares(&ctx, &mut rng, &f, 3);
    let outputs = run_parties(3, ctx.q(), |p, comm| {
        let mut prg = PrgRandomness::new(seed("party/tamper/n3", &[p as u64]));
        let (u, state) = commit(&ctx, comm, &mut prg, &f_init[p - 1]).unwrap();
        let mut ch = FsChallenger::new(EVAL_DOMAIN);
        let out = eval_with(&ctx, &pcs, comm, &mut prg, &mut ch, &state, &u, &x, &y).unwrap();
        (u, out.proof)
    });
    let (u, proof) = &outputs[0];
    let verifier =
        |p: &EvalProof<K>, u: &[Poly], x: &[Ext<K>], y: &Ext<K>| verify(&ctx, &pcs, u, x, y, p);
    assert!(verifier(proof, u, &x, &y).is_ok());
    let swapped = swapped_items_are_rejected(proof, 3, &verifier, (u, &x, &y), "toy-n3 (FS)");
    assert_eq!(swapped, 18);
}

#[test]
fn malformed_structs_and_statements_are_errors() {
    let h = honest();
    let (ctx, q) = (&h.ctx, h.ctx.q());
    let l = &ctx.layout;
    let check = |p: &EvalProof<K>| verify(ctx, &h.pcs, &h.u, &h.x, &h.y, p);
    let non_canonical = |r: Result<_, VerifyError>| {
        matches!(r, Err(VerifyError::Proof(ProofError::NonCanonical { .. })))
    };
    let shape =
        |r: Result<_, VerifyError>| matches!(r, Err(VerifyError::Proof(ProofError::Shape { .. })));
    // Non-canonical residues in every component.
    let mut p = h.proof.clone();
    p.e_u.0[3] = q;
    assert!(non_canonical(check(&p)));
    let mut p = h.proof.clone();
    p.v[0].0[15] = u64::MAX;
    assert!(non_canonical(check(&p)));
    let mut p = h.proof.clone();
    p.sumcheck.g1_sums[0].0[2] = q + 5;
    assert!(non_canonical(check(&p)));
    let mut p = h.proof.clone();
    p.sumcheck.rounds[l.ell_pp / 2].h1[2].0[3] = q;
    assert!(non_canonical(check(&p)));
    let mut p = h.proof.clone();
    p.sumcheck.finals.g0[0].0[0] = u64::MAX;
    assert!(non_canonical(check(&p)));
    // The attempt cap.
    for a in [MAX_ATTEMPTS, MAX_ATTEMPTS + 1, u64::MAX] {
        let mut p = h.proof.clone();
        p.attempt = a;
        assert_eq!(
            check(&p),
            Err(VerifyError::Proof(ProofError::AttemptOutOfRange {
                attempt: a,
                cap: MAX_ATTEMPTS
            }))
        );
    }
    // Wrong shapes.
    let mut p = h.proof.clone();
    p.e_u.0.pop();
    assert!(shape(check(&p)));
    let mut p = h.proof.clone();
    p.v.push(Poly(vec![0; l.d]));
    assert!(shape(check(&p)));
    let mut p = h.proof.clone();
    p.v.clear();
    assert!(shape(check(&p)));
    let mut p = h.proof.clone();
    p.sumcheck.g0_sums.push(Ext([0; K]));
    assert!(shape(check(&p)));
    let mut p = h.proof.clone();
    p.sumcheck.rounds.pop();
    assert!(shape(check(&p)));
    let mut p = h.proof.clone();
    let extra = p.sumcheck.rounds[0].clone();
    p.sumcheck.rounds.push(extra);
    assert!(shape(check(&p)));
    let mut p = h.proof.clone();
    p.sumcheck.rounds[0].h0.pop();
    assert!(shape(check(&p)));
    let mut p = h.proof.clone();
    p.sumcheck.finals.w.clear();
    assert!(shape(check(&p)));
    for poly in ClaimedPoly::ORDER {
        let mut p = h.proof.clone();
        p.sumcheck.commitments.of_mut(poly).clear();
        assert!(shape(check(&p)));
        let mut p = h.proof.clone();
        p.sumcheck.openings.of_mut(poly).push(Vec::new());
        assert!(shape(check(&p)));
        // Words the base PCS cannot decode: a rejected opening, not a panic.
        let mut p = h.proof.clone();
        p.sumcheck.commitments.of_mut(poly)[0].push(0);
        assert!(check(&p).is_err());
        let mut p = h.proof.clone();
        p.sumcheck.openings.of_mut(poly)[0].truncate(5);
        let err = check(&p).unwrap_err();
        assert!(is_opening(&err, 1, poly), "{err:?}");
    }
    // Malformed statements.
    let statement = |u: &[Poly], x: &[Ext<K>], y: &Ext<K>| verify(ctx, &h.pcs, u, x, y, &h.proof);
    assert!(matches!(
        statement(&h.u, &h.x[1..], &h.y),
        Err(VerifyError::Statement(StatementError::Shape { .. }))
    ));
    assert!(matches!(
        statement(&[], &h.x, &h.y),
        Err(VerifyError::Statement(StatementError::Shape { .. }))
    ));
    let mut u = h.u.clone();
    u[0].0.push(0);
    assert!(matches!(
        statement(&u, &h.x, &h.y),
        Err(VerifyError::Statement(StatementError::Shape { .. }))
    ));
    let mut x = h.x.clone();
    x[2].0[1] += q;
    assert_eq!(
        statement(&h.u, &x, &h.y),
        Err(VerifyError::Statement(StatementError::NonCanonical {
            what: "x",
            value: x[2].0[1],
            q
        }))
    );
    let mut y = h.y;
    y.0[0] = q;
    assert!(matches!(
        statement(&h.u, &h.x, &y),
        Err(VerifyError::Statement(StatementError::NonCanonical {
            what: "y",
            ..
        }))
    ));
    let mut u = h.u.clone();
    u[0].0[0] = u64::MAX;
    assert!(matches!(
        statement(&u, &h.x, &h.y),
        Err(VerifyError::Statement(StatementError::NonCanonical {
            what: "u",
            ..
        }))
    ));
}

/// The word indices of the encoding of `proof` (after the header) by kind: the table of the
/// base-PCS lengths, the residue words, and the base PCS's own words.
struct WordMap {
    lengths: Range<usize>,
    residues: Vec<Range<usize>>,
    pcs: Vec<Range<usize>>,
    total: usize,
}

fn word_map(ctx: &Context<K>, proof: &EvalProof<K>) -> WordMap {
    let l = &ctx.layout;
    let t = &proof.sumcheck;
    let lengths = 1..1 + 6 * l.n;
    let head = l.d + l.n_d * l.d + 2 * l.n * K;
    let commitments = t.commitments.total_words();
    let middle = (l.ell_pp * (l.d0 + l.d1 + 2) + 3 * l.n) * K;
    let openings = t.openings.total_words();
    let r1 = lengths.end..lengths.end + head;
    let c = r1.end..r1.end + commitments;
    let r2 = c.end..c.end + middle;
    let o = r2.end..r2.end + openings;
    WordMap {
        total: o.end,
        lengths,
        residues: vec![r1, r2],
        pcs: vec![c, o],
    }
}

#[test]
fn malformed_encodings_are_errors() {
    let h = honest();
    let (ctx, q) = (&h.ctx, h.ctx.q());
    let l = &ctx.layout;
    let len = h.bytes.len();
    let base = clear_words(l);
    assert_eq!(len, encoded_len(l, &base).unwrap());
    let shortest = encoded_len(l, &BasePcsWords::default()).unwrap();
    assert_eq!(EvalProof::from_bytes(ctx, &h.bytes).unwrap(), h.proof);
    let map = word_map(ctx, &h.proof);
    assert_eq!(HEADER_LEN + 8 * map.total, len);
    // Every truncation: below the table of lengths the expected length is that of empty
    // base-PCS items; from there on, the length the table implies.
    let table_end = HEADER_LEN + 8 * map.lengths.end;
    for cut in 0..len {
        let expected = if cut < table_end { shortest } else { len };
        assert_eq!(
            EvalProof::from_bytes(ctx, &h.bytes[..cut]),
            Err(DecodeError::Truncated {
                expected,
                actual: cut
            }),
            "truncated to {cut} bytes"
        );
    }
    // Extensions.
    for extra in [1usize, 7, 8, 9, 16, 1000] {
        let mut b = h.bytes.clone();
        b.extend(std::iter::repeat_n(0u8, extra));
        assert_eq!(
            EvalProof::from_bytes(ctx, &b),
            Err(DecodeError::TrailingBytes {
                expected: len,
                actual: len + extra
            })
        );
    }
    // The header.
    assert_eq!(&h.bytes[..8], &MAGIC);
    for i in 0..8 {
        let mut b = h.bytes.clone();
        b[i] ^= 0x20;
        assert_eq!(EvalProof::from_bytes(ctx, &b), Err(DecodeError::BadMagic));
    }
    for version in [0u64, 1, 3, u64::MAX] {
        let mut b = h.bytes.clone();
        b[8..16].copy_from_slice(&version.to_le_bytes());
        assert_eq!(
            EvalProof::from_bytes(ctx, &b),
            Err(DecodeError::UnsupportedVersion(version))
        );
    }
    let word_at = |w: usize| HEADER_LEN + 8 * w;
    // A non-canonical residue in every residue word.
    for w in map.residues.iter().flat_map(Clone::clone) {
        for bad in [q, q + 1, u64::MAX] {
            let mut b = h.bytes.clone();
            b[word_at(w)..word_at(w) + 8].copy_from_slice(&bad.to_le_bytes());
            assert!(
                matches!(
                    EvalProof::from_bytes(ctx, &b),
                    Err(DecodeError::Invalid(ProofError::NonCanonical { value, q: qq, .. }))
                        if value == bad && qq == q
                ),
                "word {w} = {bad}"
            );
        }
    }
    // A changed length of a base-PCS item: the length no longer matches; lengths that overflow.
    for w in map.lengths.clone() {
        let old = u64::from_le_bytes(h.bytes[word_at(w)..word_at(w) + 8].try_into().unwrap());
        for (new, err) in [
            (
                old + 1,
                DecodeError::Truncated {
                    expected: len + 8,
                    actual: len,
                },
            ),
            (
                old - 1,
                DecodeError::TrailingBytes {
                    expected: len - 8,
                    actual: len,
                },
            ),
            (u64::MAX, DecodeError::Overflow),
        ] {
            let mut b = h.bytes.clone();
            b[word_at(w)..word_at(w) + 8].copy_from_slice(&new.to_le_bytes());
            assert_eq!(EvalProof::from_bytes(ctx, &b), Err(err), "length word {w}");
        }
    }
    // A length moved from one item to the next: the encoding decodes, with the items' boundaries
    // moved, and the verifier rejects it.
    let mut b = h.bytes.clone();
    let (w0, w1) = (map.lengths.start, map.lengths.start + 1);
    let read =
        |b: &[u8], w: usize| u64::from_le_bytes(b[word_at(w)..word_at(w) + 8].try_into().unwrap());
    let (a0, a1) = (read(&b, w0), read(&b, w1));
    b[word_at(w0)..word_at(w0) + 8].copy_from_slice(&(a0 + 1).to_le_bytes());
    b[word_at(w1)..word_at(w1) + 8].copy_from_slice(&(a1 - 1).to_le_bytes());
    let moved = EvalProof::from_bytes(ctx, &b).unwrap();
    assert_ne!(moved, h.proof);
    assert!(verify(ctx, &h.pcs, &h.u, &h.x, &h.y, &moved).is_err());
    // a* at and beyond the cap; every value below it decodes.
    for a in [MAX_ATTEMPTS, MAX_ATTEMPTS + 1, 1 << 32, u64::MAX] {
        let mut b = h.bytes.clone();
        b[HEADER_LEN..HEADER_LEN + 8].copy_from_slice(&a.to_le_bytes());
        assert_eq!(
            EvalProof::from_bytes(ctx, &b),
            Err(DecodeError::Invalid(ProofError::AttemptOutOfRange {
                attempt: a,
                cap: MAX_ATTEMPTS
            }))
        );
    }
    for a in 0..MAX_ATTEMPTS {
        let mut b = h.bytes.clone();
        b[HEADER_LEN..HEADER_LEN + 8].copy_from_slice(&a.to_le_bytes());
        assert_eq!(EvalProof::from_bytes(ctx, &b).unwrap().attempt, a);
    }
    // An encoding for another context: toy-n2 has n = 2, so another table and length.
    let (_, ctx2) = load("toy-n2");
    assert!(matches!(
        EvalProof::from_bytes(&ctx2, &h.bytes),
        Err(DecodeError::Truncated { .. } | DecodeError::TrailingBytes { .. })
    ));
}

/// The fuzz loop of the module documentation.
#[test]
fn fuzzed_encodings_never_panic() {
    let h = honest();
    let (ctx, q) = (&h.ctx, h.ctx.q());
    let len = h.bytes.len();
    let map = word_map(ctx, &h.proof);
    let decode_and_verify = |b: &[u8]| -> bool {
        match EvalProof::from_bytes(ctx, b) {
            Ok(p) => {
                assert_eq!(p.to_bytes(), b, "a decoded encoding is canonical");
                let result = verify(ctx, &h.pcs, &h.u, &h.x, &h.y, &p);
                if b == h.bytes.as_slice() {
                    assert!(result.is_ok());
                } else {
                    assert!(result.is_err(), "a mutated proof was accepted");
                }
                true
            }
            Err(_) => false,
        }
    };
    // Random byte strings, with and without a valid header.
    let mut rng = ChaCha8Rng::seed_from_u64(0xF022_0001);
    let mut decoded = 0;
    for i in 0..3000 {
        let n = if i % 3 == 0 {
            len
        } else {
            rng.random_range(0..=2 * len)
        };
        let mut b = vec![0u8; n];
        rng.fill(&mut b[..]);
        if i % 2 == 0 && n >= HEADER_LEN {
            b[..8].copy_from_slice(&MAGIC);
            b[8..16].copy_from_slice(&VERSION.to_le_bytes());
        }
        decoded += usize::from(decode_and_verify(&b));
    }
    // Random canonical bodies (the honest table of lengths, random residues, random base-PCS
    // words): they decode, and the verifier rejects them.
    let mut rng = ChaCha8Rng::seed_from_u64(0xF022_0002);
    let words = (len - HEADER_LEN) / 8;
    let in_any = |ranges: &[Range<usize>], w: usize| ranges.iter().any(|r| r.contains(&w));
    for _ in 0..200 {
        let mut b = h.bytes[..HEADER_LEN].to_vec();
        for w in 0..words {
            let value = if w == 0 {
                rng.random_range(0..MAX_ATTEMPTS)
            } else if map.lengths.contains(&w) {
                let at = HEADER_LEN + 8 * w;
                u64::from_le_bytes(h.bytes[at..at + 8].try_into().unwrap())
            } else if in_any(&map.pcs, w) {
                rng.random()
            } else {
                rng.random_range(0..q)
            };
            b.extend_from_slice(&value.to_le_bytes());
        }
        assert!(decode_and_verify(&b));
        decoded += 1;
    }
    // Random mutations of the honest encoding.
    let mut rng = ChaCha8Rng::seed_from_u64(0xF022_0003);
    for _ in 0..3000 {
        let mut b = h.bytes.clone();
        match rng.random_range(0..6u32) {
            0 => {
                for _ in 0..rng.random_range(1..=4) {
                    let at = rng.random_range(0..len);
                    b[at] ^= rng.random_range(1..=255u8);
                }
            }
            1 => {
                let at = HEADER_LEN + 8 * rng.random_range(0..words);
                b[at..at + 8].copy_from_slice(&rng.random::<u64>().to_le_bytes());
            }
            2 => {
                // A canonical change: the word stays a valid a*, length, residue or base-PCS
                // word.
                let w = rng.random_range(0..words);
                let at = HEADER_LEN + 8 * w;
                let value = if w == 0 {
                    rng.random_range(0..MAX_ATTEMPTS)
                } else if map.lengths.contains(&w) {
                    u64::from_le_bytes(h.bytes[at..at + 8].try_into().unwrap())
                } else if in_any(&map.pcs, w) {
                    rng.random()
                } else {
                    rng.random_range(0..q)
                };
                b[at..at + 8].copy_from_slice(&value.to_le_bytes());
            }
            3 => b.truncate(rng.random_range(0..len)),
            4 => b.extend((0..rng.random_range(1..64)).map(|_| rng.random::<u8>())),
            _ => {
                let (i, j) = (rng.random_range(1..words), rng.random_range(1..words));
                let (a, c) = (HEADER_LEN + 8 * i, HEADER_LEN + 8 * j);
                let wi: [u8; 8] = b[a..a + 8].try_into().unwrap();
                let wj: [u8; 8] = b[c..c + 8].try_into().unwrap();
                b[a..a + 8].copy_from_slice(&wj);
                b[c..c + 8].copy_from_slice(&wi);
            }
        }
        decoded += usize::from(decode_and_verify(&b));
    }
    eprintln!("fuzz: {decoded} of 6200 inputs decoded (and were then verified)");
    assert!(decoded > 200);
}

/// The provers refuse what the verifier would reject, and report misconfigurations as errors.
#[test]
fn provers_refuse_wrong_claims_and_mismatched_inputs() {
    let ctx = fresh_context("toy-n1", 8);
    let pcs = clear_pcs(&ctx);
    let (f, x, y) = random_instance(&ctx, 8000);
    let q = ctx.q();
    let mut comm = Loopback::new(q).unwrap();
    let mut rng = PrgRandomness::new(seed("party/errors", &[8]));
    let (u, state) = commit(&ctx, &mut comm, &mut rng, &f).unwrap();
    let field = ctx.field();
    let fresh = || PrgRandomness::new(seed("party/errors", &[8]));
    let run = |u: &[Poly], x: &[Ext<K>], y: &Ext<K>, comm: &mut Loopback| {
        let mut ch = FsChallenger::new(EVAL_DOMAIN);
        eval_with(&ctx, &pcs, comm, &mut fresh(), &mut ch, &state, u, x, y).map(|o| o.proof)
    };
    let y_bad = field.add(&y, &field.one());
    assert_eq!(
        run(&u, &x, &y_bad, &mut comm),
        Err(Error::ClaimMismatch(hachi_co_conv::VerifyError::Mismatch))
    );
    let mut x_bad = x.clone();
    x_bad[0].0[0] = q;
    assert!(matches!(
        run(&u, &x_bad, &y, &mut comm),
        Err(Error::Statement(StatementError::NonCanonical {
            what: "x",
            ..
        }))
    ));
    assert!(matches!(
        run(&u[..0], &x, &y, &mut comm),
        Err(Error::Statement(StatementError::Shape { .. }))
    ));
    // A network with another modulus.
    let mut other = Loopback::new(q + 2).unwrap();
    assert!(matches!(
        run(&u, &x, &y, &mut other),
        Err(Error::Conv(hachi_co_conv::Error::Network { what: "q", .. }))
    ));
    // The honest claim goes through.
    assert!(run(&u, &x, &y, &mut comm).is_ok());
    // A base PCS over another field: the provers' check of the opened values stops them.
    let (_, ctx2) = load("toy-n2");
    let mut ch = FsChallenger::new(EVAL_DOMAIN);
    assert!(matches!(
        eval_with(
            &ctx,
            &clear_pcs(&ctx2),
            &mut comm,
            &mut fresh(),
            &mut ch,
            &state,
            &u,
            &x,
            &y
        ),
        Err(Error::Sumcheck(
            hachi_co_sumcheck::Error::OpeningMismatch { .. }
        ))
    ));
    // Commit and setup errors.
    assert!(matches!(
        commit(&ctx, &mut comm, &mut fresh(), &f[1..]),
        Err(Error::Commit(_))
    ));
    let params = ctx.params.clone();
    assert!(matches!(
        setup::<8>(params, &[0; 32]),
        Err(Error::Core(hachi_co_core::Error::Parameter(_)))
    ));
    // The context of a fixture with an unrelated proof: n = 2 needs two G values per kind.
    let (fx, ctx2): (Fixture, Context<K>) = load("toy-n2");
    let (u2, x2, y2) = fixture_statement(&fx);
    let proof1 = fixture_proof(&load("toy-n1").0);
    assert!(verify(&ctx2, &clear_pcs(&ctx2), &u2, &x2, &y2, &proof1).is_err());
}
