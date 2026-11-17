//! Greyhound with jali's LaBRADOR: parameters, times and sizes, as the reference's
//! `test_greyhound` reports them.
//!
//! `cargo run --release -p hachi-co-greyhound --example greyhound -- LOG_DEGREE [Q] [REPS]
//! [labrador|reference]` commits to a polynomial of degree below $2^{\mathrm{LOG\_DEGREE}}$ with
//! uniform coefficients modulo $q$ (default $2^{32} - 99$), proves its evaluation at $x = 43$ and
//! verifies it, `REPS` times (default 1), under the security rule of [`Security::labrador`]
//! (default) or [`Security::REFERENCE`]. Threads: Rayon's default, or `RAYON_NUM_THREADS`.
//! For $q \ge 2^{62}$, which jali 0.3.1 does not take, it times the front end, the reduction and
//! the check of the reduced relation instead of LaBRADOR. `GREYHOUND_R=r` lays the witness out
//! in about $r$ vectors for jali ([`Layout::with_multiplicity`]) instead of [`Layout::choose`],
//! and `GREYHOUND_F=f` fixes the expansion factor ([`Params::with_expansion`]).
//!
//! Every run prints a `csv,` line: log degree, bits of $q$, threads, repetition, then seconds
//! (setup, commit, front end, reduction and jali form, LaBRADOR proving, proving in all, the
//! verifier's reduction and jali form, LaBRADOR verifying, verifying in all) and bytes
//! (commitment, front-end proof, LaBRADOR proof, proof in all, the reference's count of $u_1$
//! and $u_2$), the witness norm and jali's layout.
use hachi_co_greyhound::{
    Commitment, FrontProof, Layout, Params, Proof, PublicParams, Relation, Security, commit,
    eval::{absorb_norm, transcript},
    pcs::labrador_context,
    prove_front, verify,
};
use hachi_co_sampling::ShakePrg;
use rand::Rng;
use std::time::Instant;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let log_degree: u32 = args.first().map_or(Ok(21), |a| a.parse())?;
    let q: u64 = args.get(1).map_or(Ok(4294967197), |a| a.parse())?;
    let reps: usize = args.get(2).map_or(Ok(1), |a| a.parse())?;
    let security = match args.get(3).map(String::as_str) {
        Some("reference") => Security::REFERENCE,
        _ => Security::labrador(),
    };
    if log_degree < 6 {
        return Err("the degree must be at least 2^6".into());
    }
    let len = 1usize << (log_degree - 6);
    let threads = rayon::current_num_threads();
    let bits = 64 - q.leading_zeros();
    println!(
        "Greyhound, degree 2^{log_degree} ({len} ring elements), q = {q} ({bits} bits), \
         {threads} threads, vector instructions {}, slack {:.4}, delta {}",
        hachi_co_math::fma::vector_instructions(),
        security.slack,
        security.delta
    );

    let start = Instant::now();
    // GREYHOUND_F=f fixes the expansion factor (Params::with_expansion) instead of the first f.
    let pp = match std::env::var("GREYHOUND_F")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        Some(f) => {
            let params = Params::with_expansion(len, q, security, f)?;
            PublicParams::with_params(params, security, [0u8; 32])?
        }
        None => PublicParams::new(len, q, security, [0u8; 32])?,
    };
    // GREYHOUND_R=r takes about r vectors for jali's statement instead of Layout::choose.
    let pp = match std::env::var("GREYHOUND_R")
        .ok()
        .and_then(|v| v.parse().ok())
    {
        Some(r) => {
            let layout = Layout::with_multiplicity(&pp.params().witness_lengths(), r);
            pp.with_layout(layout)?
        }
        None => pp,
    };
    let pk = pp.prover_key()?;
    let setup = start.elapsed().as_secs_f64();
    let p = *pp.params();
    let layout = pp.layout();
    println!(
        "  Lengths decomposition: {} x {}\n  Commitment ranks: kappa = {}; kappa1 = {}\n  \
         Decomposition bases: b = {}; bu = {}\n  Expansion factors: f = {}; fu = {}\n  \
         Predicted witness norm: {:.2}\n  jali layout: {} x {} (witness {} polynomials)\n  \
         Setup (matrices, prover's transforms): {setup:.4}s",
        p.m,
        p.n,
        p.kappa,
        p.kappa1,
        1u64 << p.b,
        1u64 << p.bu,
        p.f,
        p.fu,
        (p.predicted_norm_sq() as f64).sqrt(),
        layout.r,
        layout.n,
        p.witness_lengths().iter().sum::<usize>(),
    );

    let mut prg = ShakePrg::new(&[1u8; 32], b"benchmark polynomial", 0);
    let mask = u64::MAX >> (q - 1).leading_zeros();
    let poly: Vec<u64> = (0..len * 64)
        .map(|_| {
            loop {
                let v = prg.next_u64() & mask;
                if v < q {
                    break v;
                }
            }
        })
        .collect();
    let x = 43u64;
    // Whether jali takes the modulus (jali 0.3.1: q < 2^62).
    let labrador = pp.labrador_ring().is_ok();

    for rep in 0..reps {
        let start = Instant::now();
        let (com, opening) = commit(&pp, &pk, &poly)?;
        let t_commit = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let front = prove_front(&pp, &com, &opening, &poly, x)?;
        let t_front = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let relation =
            Relation::with_challenges(&pp, &com, x, front.y, &front.proof, &front.challenges)?;
        let coefficients = relation.coefficients();
        if !labrador {
            relation.check(pp.ring(), &front.witness)?;
            let t_check = start.elapsed().as_secs_f64();
            let front_bytes = FrontProof::encoded_len(&p);
            println!(
                "  rep {rep}: commit {t_commit:.4}s, front end {t_front:.4}s, reduction and \
                 check {t_check:.4}s ({coefficients} coefficients of phi), witness norm {:.2}, \
                 front-end proof {front_bytes} B, the reference's u1 + u2 {:.2} KB; no LaBRADOR \
                 at {bits} bits (jali does not take this modulus)",
                (front.proof.norm_sq as f64).sqrt(),
                p.reference_proof_bytes() as f64 / 1024.0,
            );
            println!(
                "csv,{log_degree},{bits},{threads},{rep},{setup:.4},{t_commit:.4},{t_front:.4},\
                 {t_check:.4},,,,,,{},{front_bytes},,,{},{:.2},{}x{}",
                Commitment::encoded_len(&p),
                p.reference_proof_bytes(),
                (front.proof.norm_sq as f64).sqrt(),
                layout.r,
                layout.n
            );
            continue;
        }
        let ring = pp.labrador_ring()?;
        let statement = relation.into_jali(&ring, &layout)?;
        let witness = layout.jali_witness(&ring, &p, &front.witness)?;
        let t_reduce = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let lab = jali::labrador::prove(
            &statement,
            &witness,
            &pp.labrador_seed(),
            &labrador_context(&front.transcript),
            &jali::labrador::Policy::default(),
        )?;
        let t_lab = start.elapsed().as_secs_f64();
        drop(statement);
        drop(witness);
        let proof = Proof {
            front: front.proof.clone(),
            labrador: lab,
        };
        let t_prove = t_front + t_reduce + t_lab;

        // The verifier, step by step as in `verify`.
        let start = Instant::now();
        p.check_norm(proof.front.norm_sq, pp.security())?;
        let (mut t, cs) = transcript(&pp, &com, x, front.y, &proof.front.u2);
        absorb_norm(&mut t, proof.front.norm_sq);
        let relation = Relation::with_challenges(&pp, &com, x, front.y, &proof.front, &cs)?;
        let statement = relation.into_jali(&ring, &layout)?;
        let t_vreduce = start.elapsed().as_secs_f64();
        let start = Instant::now();
        jali::labrador::verify(
            &statement,
            &proof.labrador,
            &pp.labrador_seed(),
            &labrador_context(&t),
            &jali::labrador::Policy::default(),
        )?;
        let t_vlab = start.elapsed().as_secs_f64();
        let t_verify = t_vreduce + t_vlab;
        drop(statement);
        // The public entry point as well, untimed, as a check.
        verify(&pp, &com, x, front.y, &proof)?;

        let com_bytes = Commitment::encoded_len(&p);
        let front_bytes = FrontProof::encoded_len(&p);
        let lab_bytes = proof.labrador.len();
        let total = proof.to_bytes(&p).len();
        println!(
            "  rep {rep}: commit {t_commit:.4}s | prove {t_prove:.4}s (front end {t_front:.4}s, \
             reduction {t_reduce:.4}s, LaBRADOR {t_lab:.4}s) | verify {t_verify:.4}s \
             (reduction {t_vreduce:.4}s, LaBRADOR {t_vlab:.4}s) | commitment {com_bytes} B, \
             proof {total} B = front end {front_bytes} B + LaBRADOR {lab_bytes} B ({:.2} KB), \
             the reference's u1 + u2 {:.2} KB, u1 + proof {:.2} KB | witness norm {:.2} | \
             {coefficients} coefficients of phi",
            total as f64 / 1024.0,
            p.reference_proof_bytes() as f64 / 1024.0,
            (com_bytes + total) as f64 / 1024.0,
            (proof.front.norm_sq as f64).sqrt(),
        );
        println!(
            "csv,{log_degree},{bits},{threads},{rep},{setup:.4},{t_commit:.4},{t_front:.4},\
             {t_reduce:.4},{t_lab:.4},{t_prove:.4},{t_vreduce:.4},{t_vlab:.4},{t_verify:.4},\
             {com_bytes},{front_bytes},{lab_bytes},{total},{},{:.2},{}x{}",
            p.reference_proof_bytes(),
            (proof.front.norm_sq as f64).sqrt(),
            layout.r,
            layout.n
        );
    }
    Ok(())
}
