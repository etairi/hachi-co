//! The driver `prove` with the reference base PCS `ClearPcs`: with `Loopback`,
//! `ReplayRandomness` and `ReplayChallenger` it reproduces toy-n1 (every recorded value; the
//! oracle has no commitments or openings, and the driver's are those of the lockstep run with
//! the openings attached independently); on the in-memory network of `hachi-co-mpc` (n threads,
//! masked-sum F_AH) it reproduces the multi-party fixtures at every party; honest Fiat-Shamir
//! runs (`PrgRandomness`, `FsChallenger`) are accepted by the verifier, openings included, and
//! leave the provers' and the verifier's transcripts in the same state; tampering under
//! Fiat-Shamir is rejected, including the commitments and the opening proofs; a faulty base PCS
//! is caught by the provers.

mod common;

use common::{
    Instance, K, Pcs, clear_pcs, fixture_transcript, honest_runs, ideal_pcs, instances, without_pcs,
};
use hachi_co_core::{
    BasePcs, Challenger, ClearCommitment, ClearOpening, ClearPolynomial, Ext, FsChallenger,
    PartyRandomness, PrgRandomness, ReplayChallenger, ReplayRandomness,
};
use hachi_co_mpc::{Comm, Loopback, MemoryConfig, PartyId, run_in_memory};
use hachi_co_sumcheck::{
    ClaimedPoly, Error, PublicInput, SumcheckPrivate, SumcheckTranscript, VerifyError, prove,
    verify, verify_checks, verify_claims,
};
use std::time::Instant;

/// The Fiat-Shamir domain of the tests.
const DOMAIN: &[u8] = b"hachi-co/test/sumcheck";

/// Runs `prove` at every party of `inst` on threads over the in-memory network, with the
/// randomness and challenger made by `rng` and `ch` for each party and `ClearPcs`.
fn run_network<R, H>(
    inst: &Instance,
    rng: impl Fn(PartyId) -> R + Sync,
    ch: impl Fn(PartyId) -> H + Sync,
) -> Vec<(SumcheckTranscript<K>, SumcheckPrivate<K>, H)>
where
    R: PartyRandomness<K>,
    H: Challenger<K> + Send,
{
    let (n, q) = (inst.n(), inst.ctx.q());
    let mut seed = [n as u8; 32];
    seed[..8].copy_from_slice(&q.to_le_bytes());
    let pcs = clear_pcs(inst.field());
    run_in_memory(&MemoryConfig::new(n, q, seed), |net| {
        let p = net.id();
        let (mut r, mut c) = (rng(p), ch(p));
        let (t, private) = prove(
            &inst.ctx,
            &pcs,
            net,
            &mut r,
            &mut c,
            &inst.m,
            &inst.e,
            &inst.shares[p - 1],
        )
        .unwrap();
        (t, private, c)
    })
    .expect("a valid network")
    .into_iter()
    .map(|run| run.expect("every party connects").output)
    .collect()
}

fn public(inst: &Instance) -> PublicInput<'_> {
    PublicInput::new(&inst.ctx, &inst.m, &inst.y, &inst.e, &inst.y_ring)
}

/// The driver with `Loopback`, `ReplayRandomness` and `ReplayChallenger` reproduces toy-n1: the
/// transcript without the base-PCS words is the fixture's; the whole transcript (commitments
/// and openings included) and the private data are those of the lockstep run, whose openings
/// are made independently of the driver; the verifier accepts with `ClearPcs`.
#[test]
fn loopback_driver_reproduces_toy_n1() {
    let inst = &instances()[0];
    let fx = &inst.fx;
    let pcs = clear_pcs(inst.field());
    let mut comm = Loopback::new(inst.ctx.q()).unwrap();
    let mut rng = ReplayRandomness::new(fx, 1).unwrap();
    let mut ch = ReplayChallenger::new(fx).unwrap();
    let (transcript, private) = prove(
        &inst.ctx,
        &pcs,
        &mut comm,
        &mut rng,
        &mut ch,
        &inst.m,
        &inst.e,
        &inst.shares[0],
    )
    .unwrap();
    assert_eq!(without_pcs(&transcript), fixture_transcript(fx));
    assert_eq!(transcript, honest_runs()[0].transcript);
    assert_eq!(private, honest_runs()[0].privates[0]);
    assert_eq!(
        private.quotient,
        fx.polys_of_len("/sumcheck/quotient_shares/0", 2 * inst.ctx.layout.d)
            .unwrap()
    );
    // One commitment and one proof per polynomial, of the sizes of ClearPcs.
    let l = &inst.ctx.layout;
    for poly in ClaimedPoly::ORDER {
        assert_eq!(
            transcript.commitments.of(poly)[0].len(),
            Pcs::COMMITMENT_WORDS
        );
    }
    assert_eq!(
        Some(transcript.openings.w[0].len()),
        Pcs::multilinear_proof_words(l.ell_pp)
    );
    assert_eq!(
        Some(transcript.openings.g0[0].len()),
        Pcs::vector_proof_words(1 + l.ell_pp * l.d0)
    );
    assert_eq!(
        Some(transcript.openings.g1[0].len()),
        Pcs::vector_proof_words(1 + l.ell_pp * l.d1)
    );
    let mut ch = ReplayChallenger::new(fx).unwrap();
    let claims = verify_claims(&inst.ctx, &pcs, &public(inst), &transcript, &mut ch).unwrap();
    assert!(
        ideal_pcs(inst.field(), &claims, &[private])
            .iter()
            .all(|(_, ok)| *ok)
    );
}

/// The driver on n threads with replayed randomness and coins reproduces toy-n2, n3, n4 and n8
/// at every party (the per-party G, commitments, final values and proofs travel through one
/// all-gather call each), commitments and openings included (those of the lockstep run).
#[test]
fn threaded_driver_reproduces_the_multi_party_fixtures() {
    for (inst, run) in instances().iter().zip(honest_runs()).skip(1) {
        let fx = &inst.fx;
        let outputs = run_network(
            inst,
            |p| ReplayRandomness::new(fx, p).unwrap(),
            |_| ReplayChallenger::new(fx).unwrap(),
        );
        let recorded = fixture_transcript(fx);
        for (p0, (transcript, private, _)) in outputs.iter().enumerate() {
            let what = format!("{}: party {}", inst.name, p0 + 1);
            assert_eq!(without_pcs(transcript), recorded, "{what}");
            assert_eq!(transcript, &run.transcript, "{what}");
            assert_eq!(private, &run.privates[p0], "{what}");
        }
        let pcs = clear_pcs(inst.field());
        let mut ch = ReplayChallenger::new(fx).unwrap();
        verify(&inst.ctx, &pcs, &public(inst), &outputs[0].0, &mut ch).unwrap();
    }
}

/// Honest Fiat-Shamir runs (fresh `PrgRandomness` seeds 1, 2, 3 per party, `FsChallenger` on the
/// domain "hachi-co/test/sumcheck") at toy-n1 (Loopback) and at toy-n2, n3, n4, n8 (threads)
/// are accepted, openings included; the checked claims are the provers' polynomials (ideal
/// PCS); every party's transcript and the verifier's end in the same state. One run at toy-n1
/// is timed (printed with `--nocapture`).
#[test]
fn honest_fiat_shamir_runs_are_accepted() {
    for seed in 1u8..=3 {
        let inst = &instances()[0];
        let pcs = clear_pcs(inst.field());
        let start = Instant::now();
        let mut comm = Loopback::new(inst.ctx.q()).unwrap();
        let mut rng = PrgRandomness::new([seed; 32]);
        let mut ch = FsChallenger::new(DOMAIN);
        let (transcript, private) = prove(
            &inst.ctx,
            &pcs,
            &mut comm,
            &mut rng,
            &mut ch,
            &inst.m,
            &inst.e,
            &inst.shares[0],
        )
        .unwrap();
        let proved = start.elapsed();
        let start = Instant::now();
        let mut vch = FsChallenger::new(DOMAIN);
        let claims = verify_claims(&inst.ctx, &pcs, &public(inst), &transcript, &mut vch).unwrap();
        let verified = start.elapsed();
        eprintln!(
            "toy-n1, PrgRandomness seed [{seed}; 32]: prove {proved:?}, verify {verified:?} \
             (l'' = {}, 2^l'' = {} points)",
            inst.ctx.layout.ell_pp,
            1usize << inst.ctx.layout.ell_pp
        );
        assert_eq!(ch.transcript().digest(), vch.transcript().digest());
        assert!(
            ideal_pcs(inst.field(), &claims, &[private])
                .iter()
                .all(|(_, ok)| *ok)
        );
        assert_ne!(without_pcs(&transcript), fixture_transcript(&inst.fx));
    }
    for inst in &instances()[1..] {
        let outputs = run_network(
            inst,
            |p| PrgRandomness::new([p as u8 + 16; 32]),
            |_| FsChallenger::new(DOMAIN),
        );
        let transcript = &outputs[0].0;
        let pcs = clear_pcs(inst.field());
        let mut vch = FsChallenger::new(DOMAIN);
        let claims = verify_claims(&inst.ctx, &pcs, &public(inst), transcript, &mut vch).unwrap();
        let privates: Vec<SumcheckPrivate<K>> = outputs.iter().map(|o| o.1.clone()).collect();
        assert!(
            ideal_pcs(inst.field(), &claims, &privates)
                .iter()
                .all(|(_, ok)| *ok)
        );
        for (t, _, ch) in &outputs {
            assert_eq!(t, transcript);
            assert_eq!(ch.transcript().digest(), vch.transcript().digest());
        }
    }
}

/// Under Fiat-Shamir a tampered message also changes every later coin, and the contexts of the
/// openings; the verifier rejects, and a verifier on another domain (another statement) rejects
/// the honest transcript. A challenger that absorbed something else first, too. A tampered
/// commitment changes the coins (round checks fail); a tampered or moved opening proof fails
/// its opening (`VerifyError::Opening` with the party and the polynomial).
#[test]
fn fiat_shamir_tampering_is_rejected() {
    let inst = &instances()[1]; // n = 2
    let f = inst.field();
    let pcs = clear_pcs(f);
    let outputs = run_network(
        inst,
        |p| PrgRandomness::new([p as u8 + 7; 32]),
        |_| FsChallenger::new(DOMAIN),
    );
    let transcript = &outputs[0].0;
    let run = |t: &SumcheckTranscript<K>, mut vch: FsChallenger| {
        verify(&inst.ctx, &pcs, &public(inst), t, &mut vch)
    };
    let rejected = |t: &SumcheckTranscript<K>, vch: FsChallenger| {
        matches!(run(t, vch), Err(VerifyError::Rejected { .. }))
    };
    assert!(run(transcript, FsChallenger::new(DOMAIN)).is_ok());
    for r in [0, 7, transcript.rounds.len() - 1] {
        let mut t = transcript.clone();
        t.rounds[r].h1[2] = f.add(&t.rounds[r].h1[2], &f.one());
        assert!(rejected(&t, FsChallenger::new(DOMAIN)), "round {}", r + 1);
        // A consistent change (the sum at 0 and 1 unchanged) changes the next coins.
        let mut t = transcript.clone();
        for (i, v) in t.rounds[r].h0.iter_mut().enumerate() {
            let x = (i + 1) as u64;
            *v = f.add(v, &f.from_base(3 * x * (x - 1)));
        }
        let mut vch = FsChallenger::new(DOMAIN);
        let report = verify_checks(&inst.ctx, &pcs, &public(inst), &t, &mut vch).unwrap();
        assert!(!report.accepted(), "round {}", r + 1);
    }
    let mut t = transcript.clone();
    t.g1_sums[0] = f.add(&t.g1_sums[0], &f.one());
    assert!(rejected(&t, FsChallenger::new(DOMAIN)));
    let mut t = transcript.clone();
    t.finals.g0[0] = f.add(&t.finals.g0[0], &f.one());
    assert!(rejected(&t, FsChallenger::new(DOMAIN)));
    assert!(rejected(
        transcript,
        FsChallenger::new(b"another statement")
    ));
    let mut other = FsChallenger::new(DOMAIN);
    Challenger::<K>::absorb_u64(&mut other, "something else", 1);
    assert!(rejected(transcript, other));
    // Commitments: every word of every commitment; the coins change.
    for poly in ClaimedPoly::ORDER {
        for p0 in 0..inst.n() {
            for i in 0..Pcs::COMMITMENT_WORDS {
                let mut t = transcript.clone();
                t.commitments.of_mut(poly)[p0][i] ^= 1;
                assert!(
                    rejected(&t, FsChallenger::new(DOMAIN)),
                    "commitment {} of party {} word {i}",
                    poly.name(),
                    p0 + 1
                );
            }
        }
    }
    // Opening proofs: a changed word (context, kind, count, or an entry), or the proofs of the
    // two parties swapped.
    for poly in ClaimedPoly::ORDER {
        for p0 in 0..inst.n() {
            let len = transcript.openings.of(poly)[p0].len();
            for i in [0, 3, 4, 5, 6, len / 2, len - 1] {
                let mut t = transcript.clone();
                let word = &mut t.openings.of_mut(poly)[p0][i];
                *word = word.wrapping_add(1);
                let err = run(&t, FsChallenger::new(DOMAIN)).unwrap_err();
                assert!(
                    matches!(&err, VerifyError::Opening { party, poly: got, .. }
                        if *party == p0 + 1 && *got == poly),
                    "{} of party {} word {i}: {err:?}",
                    poly.name(),
                    p0 + 1
                );
            }
        }
        let mut t = transcript.clone();
        t.openings.of_mut(poly).swap(0, 1);
        assert!(
            matches!(
                run(&t, FsChallenger::new(DOMAIN)),
                Err(VerifyError::Opening { party: 1, poly: got, .. }) if got == poly
            ),
            "{} swapped",
            poly.name()
        );
        // Commitments and proofs swapped together: the coins and the contexts change.
        let mut t = transcript.clone();
        t.openings.of_mut(poly).swap(0, 1);
        t.commitments.of_mut(poly).swap(0, 1);
        assert!(run(&t, FsChallenger::new(DOMAIN)).is_err());
    }
    // A proof of one polynomial given for another of the same party.
    let mut t = transcript.clone();
    t.openings.g0[1] = t.openings.g1[1].clone();
    assert!(matches!(
        run(&t, FsChallenger::new(DOMAIN)),
        Err(VerifyError::Opening {
            party: 2,
            poly: ClaimedPoly::G0,
            ..
        })
    ));
}

/// The driver refuses a network that does not match the context, and a prover cannot draw a
/// blinder stream twice from one `PrgRandomness` (a second Eval needs a fresh seed).
#[test]
fn driver_refuses_mismatched_networks_and_reused_randomness() {
    let inst = &instances()[1]; // n = 2
    let pcs = clear_pcs(inst.field());
    let mut comm = Loopback::new(inst.ctx.q()).unwrap();
    let mut rng = PrgRandomness::new([1; 32]);
    let mut ch = FsChallenger::new(DOMAIN);
    let r = prove(
        &inst.ctx,
        &pcs,
        &mut comm,
        &mut rng,
        &mut ch,
        &inst.m,
        &inst.e,
        &inst.shares[0],
    );
    assert!(matches!(r, Err(Error::Parameter(_))));
    // A base PCS over another field (here toy-n2's q for a toy-n1 context) opens to other
    // values: the provers' check of the opened values stops them.
    let other_field = pcs;
    let inst = &instances()[0];
    let mut comm = Loopback::new(inst.ctx.q()).unwrap();
    let r = prove(
        &inst.ctx,
        &other_field,
        &mut comm,
        &mut PrgRandomness::new([3; 32]),
        &mut FsChallenger::new(DOMAIN),
        &inst.m,
        &inst.e,
        &inst.shares[0],
    );
    assert!(matches!(
        r,
        Err(Error::OpeningMismatch {
            party: 1,
            poly: ClaimedPoly::G0
        })
    ));
    let pcs = clear_pcs(inst.field());
    let mut comm = Loopback::new(inst.ctx.q() + 2).unwrap();
    let r = prove(
        &inst.ctx,
        &pcs,
        &mut comm,
        &mut rng,
        &mut ch,
        &inst.m,
        &inst.e,
        &inst.shares[0],
    );
    assert!(matches!(r, Err(Error::Parameter(_))));
    let mut comm = Loopback::new(inst.ctx.q()).unwrap();
    let mut rng = PrgRandomness::new([1; 32]);
    prove(
        &inst.ctx,
        &pcs,
        &mut comm,
        &mut rng,
        &mut ch,
        &inst.m,
        &inst.e,
        &inst.shares[0],
    )
    .unwrap();
    let again = prove(
        &inst.ctx,
        &pcs,
        &mut comm,
        &mut rng,
        &mut ch,
        &inst.m,
        &inst.e,
        &inst.shares[0],
    );
    assert!(matches!(
        again,
        Err(Error::Core(hachi_co_core::Error::RandomnessReuse { .. }))
    ));
    // A share of the wrong shape is an error.
    let mut bad = inst.shares[0].clone();
    bad.z_prime.pop();
    let mut rng = PrgRandomness::new([2; 32]);
    let r = prove(
        &inst.ctx, &pcs, &mut comm, &mut rng, &mut ch, &inst.m, &inst.e, &bad,
    );
    assert!(r.is_err());
}

/// A base PCS that misbehaves on the provers' side: it opens one kind of polynomial to another
/// value, or refuses to commit.
struct FaultyPcs {
    inner: Pcs,
    /// Which polynomial gets a wrong value: `W` (multilinear) or the vectors (`G0`, `G1`).
    wrong: Option<ClaimedPoly>,
    /// Refuse every vector commitment.
    refuse_vectors: bool,
}

impl BasePcs<K> for FaultyPcs {
    type Commitment = ClearCommitment;
    type State = ClearPolynomial<K>;
    type Proof = ClearOpening<K>;

    fn commit_multilinear(
        &self,
        coeffs: &[u64],
    ) -> hachi_co_core::Result<(ClearCommitment, ClearPolynomial<K>)> {
        self.inner.commit_multilinear(coeffs)
    }

    fn commit_vector(
        &self,
        v: &[Ext<K>],
    ) -> hachi_co_core::Result<(ClearCommitment, ClearPolynomial<K>)> {
        if self.refuse_vectors {
            return Err(hachi_co_core::Error::BasePcs("refused".into()));
        }
        self.inner.commit_vector(v)
    }

    fn prove_multilinear(
        &self,
        st: &ClearPolynomial<K>,
        point: &[Ext<K>],
        context: &[u8; 32],
    ) -> hachi_co_core::Result<(Ext<K>, ClearOpening<K>)> {
        let (mut value, proof) = self.inner.prove_multilinear(st, point, context)?;
        if self.wrong == Some(ClaimedPoly::W) {
            value.0[0] ^= 1;
        }
        Ok((value, proof))
    }

    fn prove_inner_product(
        &self,
        st: &ClearPolynomial<K>,
        weights: &[Ext<K>],
        context: &[u8; 32],
    ) -> hachi_co_core::Result<(Ext<K>, ClearOpening<K>)> {
        let (mut value, proof) = self.inner.prove_inner_product(st, weights, context)?;
        if matches!(self.wrong, Some(ClaimedPoly::G0 | ClaimedPoly::G1)) {
            value.0[0] ^= 1;
        }
        Ok((value, proof))
    }

    fn verify_multilinear(
        &self,
        com: &ClearCommitment,
        num_vars: usize,
        point: &[Ext<K>],
        value: &Ext<K>,
        proof: &ClearOpening<K>,
        context: &[u8; 32],
    ) -> Result<(), String> {
        self.inner
            .verify_multilinear(com, num_vars, point, value, proof, context)
    }

    fn verify_inner_product(
        &self,
        com: &ClearCommitment,
        len: usize,
        weights: &[Ext<K>],
        value: &Ext<K>,
        proof: &ClearOpening<K>,
        context: &[u8; 32],
    ) -> Result<(), String> {
        self.inner
            .verify_inner_product(com, len, weights, value, proof, context)
    }

    fn params_digest(&self) -> [u8; 32] {
        self.inner.params_digest()
    }

    fn commitment_to_words(com: &ClearCommitment) -> Vec<u64> {
        Pcs::commitment_to_words(com)
    }

    fn commitment_from_words(words: &[u64]) -> hachi_co_core::Result<ClearCommitment> {
        Pcs::commitment_from_words(words)
    }

    fn proof_to_words(p: &ClearOpening<K>) -> Vec<u64> {
        Pcs::proof_to_words(p)
    }

    fn proof_from_words(words: &[u64]) -> hachi_co_core::Result<ClearOpening<K>> {
        Pcs::proof_from_words(words)
    }
}

/// The provers check every opened value against their final value and stop with
/// `OpeningMismatch` (the first opening in the order g0, g1, w that differs); a base PCS that
/// fails is an error of the driver.
#[test]
fn a_faulty_base_pcs_is_caught_by_the_provers() {
    let inst = &instances()[0];
    let run = |pcs: &FaultyPcs| {
        let mut comm = Loopback::new(inst.ctx.q()).unwrap();
        let mut rng = ReplayRandomness::new(&inst.fx, 1).unwrap();
        let mut ch = ReplayChallenger::new(&inst.fx).unwrap();
        prove(
            &inst.ctx,
            pcs,
            &mut comm,
            &mut rng,
            &mut ch,
            &inst.m,
            &inst.e,
            &inst.shares[0],
        )
    };
    let faulty = |wrong, refuse_vectors| FaultyPcs {
        inner: clear_pcs(inst.field()),
        wrong,
        refuse_vectors,
    };
    assert!(run(&faulty(None, false)).is_ok());
    for (wrong, first) in [
        (ClaimedPoly::W, ClaimedPoly::W),
        (ClaimedPoly::G0, ClaimedPoly::G0),
    ] {
        assert!(matches!(
            run(&faulty(Some(wrong), false)),
            Err(Error::OpeningMismatch { party: 1, poly }) if poly == first
        ));
    }
    assert!(matches!(
        run(&faulty(None, true)),
        Err(Error::Core(hachi_co_core::Error::BasePcs(_)))
    ));
}
