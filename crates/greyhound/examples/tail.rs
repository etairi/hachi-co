//! The tail: commitments to digit tables and one LaBRADOR proof of their evaluation claims.
//!
//! `cargo run --release -p hachi-co-greyhound --example tail -- Q LAMBDA REPS VARS...` commits
//! to tables of $2^{v}$ digits uniform in $[-8, 7]$ (bound 8) for each `VARS` entry, proves the
//! claims at random points of $\mathbb F_{q^4}$ in one proof and verifies it, `REPS` times, with
//! Rayon's threads (or `RAYON_NUM_THREADS`). Prints the shapes, then a `csv,` line per
//! repetition: q bits, lambda, the vars joined by `+`, threads, repetition, seconds (setup,
//! commit of all tables, prove, verify), bytes (commitments, proof as fixed, jali's part
//! before padding), the joint witness norm, and the attempt the proof took (0 unless the
//! prover restarted).
use hachi_co_greyhound::tail::{Claim, ProverClaim, TailParams};
use hachi_co_math::{Ext, Fqk};
use hachi_co_sampling::ShakePrg;
use rand::Rng;
use std::time::Instant;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let q: u64 = args.first().map_or(Ok(10916536653611741), |a| a.parse())?;
    let lambda: u32 = args.get(1).map_or(Ok(100), |a| a.parse())?;
    let reps: usize = args.get(2).map_or(Ok(1), |a| a.parse())?;
    let vars: Vec<usize> = args[3.min(args.len())..]
        .iter()
        .map(|a| a.parse())
        .collect::<Result<_, _>>()?;
    let vars = if vars.is_empty() {
        vec![21, 20, 18]
    } else {
        vars
    };
    let threads = rayon::current_num_threads();
    let bits = 64 - q.leading_zeros();
    let field = Fqk::<4>::new(q)?;

    let start = Instant::now();
    let params = TailParams::new(q, &vars, 8, lambda, [0u8; 32])?;
    let setup = start.elapsed().as_secs_f64();
    println!(
        "tail: q = {q} ({bits} bits), lambda {lambda}, vars {vars:?}, {threads} threads, \
         setup {setup:.3}s, proof {} words, jali predicted {} B, joint norm {:.1}",
        params.proof_words(),
        params.predicted_labrador_bytes(),
        params.joint_norm_sq().sqrt()
    );
    for s in params.shapes() {
        println!(
            "  vars {}: m = 2^{} x n = {}, kappa {}, kappa1 {}, b_u 2^{}, f_u {}, commitment {} words",
            s.vars,
            s.mbits,
            s.n,
            s.kappa,
            s.kappa1,
            s.bu,
            s.fu,
            params.commitment_words(s.vars)
        );
    }
    let tables: Vec<Vec<i8>> = vars
        .iter()
        .enumerate()
        .map(|(t, v)| {
            let mut prg = ShakePrg::new(&[t as u8; 32], b"tail benchmark digits", 0);
            (0..1usize << v)
                .map(|_| (prg.next_u32() % 16) as i8 - 8)
                .collect()
        })
        .collect();
    let points: Vec<Vec<Ext<4>>> = vars
        .iter()
        .enumerate()
        .map(|(t, v)| {
            let mut prg = ShakePrg::new(&[t as u8; 32], b"tail benchmark point", 1);
            (0..*v)
                .map(|_| Ext(core::array::from_fn(|_| prg.next_u64() % q)))
                .collect()
        })
        .collect();
    // Values: sum_x d_x eq(x, a), by the tensor of the halves (as Hachi's table_eval).
    let values: Vec<Ext<4>> = tables
        .iter()
        .zip(&points)
        .map(|(d, a)| eval(&field, d, a))
        .collect();
    for rep in 0..reps {
        let start = Instant::now();
        let mut commitments = Vec::new();
        let mut openings = Vec::new();
        for (v, d) in vars.iter().zip(&tables) {
            let (c, o) = params.commit(*v, d)?;
            commitments.push(c);
            openings.push(o);
        }
        let t_commit = start.elapsed().as_secs_f64();
        let claims: Vec<Claim<'_, 4>> = (0..vars.len())
            .map(|t| Claim {
                vars: vars[t],
                commitment: &commitments[t],
                point: &points[t],
                value: values[t],
            })
            .collect();
        let prover: Vec<ProverClaim<'_, 4>> = (0..vars.len())
            .map(|t| ProverClaim {
                claim: claims[t],
                digits: &tables[t],
                opening: &openings[t],
            })
            .collect();
        let context = [7u8; 32];
        let start = Instant::now();
        let proof = params.prove(&field, &prover, &context)?;
        let t_prove = start.elapsed().as_secs_f64();
        let start = Instant::now();
        params.verify(&field, &claims, &proof, &context)?;
        let t_verify = start.elapsed().as_secs_f64();
        let com_bytes: usize = commitments.iter().map(|c| 8 * c.len()).sum();
        let u2_words: usize = params
            .shapes()
            .iter()
            .map(|s| params.commitment_words(s.vars))
            .sum();
        // The length word: jali's bytes in the low 48 bits, the attempt in the high 16.
        let jali_bytes = (proof[u2_words + 2] & ((1 << 48) - 1)) as usize;
        let attempt = proof[u2_words + 2] >> 48;
        let norm =
            ((u128::from(proof[u2_words]) | (u128::from(proof[u2_words + 1]) << 64)) as f64).sqrt();
        let joined: Vec<String> = vars.iter().map(usize::to_string).collect();
        println!(
            "  rep {rep}: commit {t_commit:.3}s, prove {t_prove:.3}s, verify {t_verify:.3}s, \
             commitments {com_bytes} B, proof {} B (jali {jali_bytes} B before padding), \
             norm {norm:.1}, attempt {attempt}",
            8 * proof.len()
        );
        println!(
            "csv,{bits},{lambda},{},{threads},{rep},{setup:.4},{t_commit:.4},{t_prove:.4},\
             {t_verify:.4},{com_bytes},{},{jali_bytes},{norm:.1},{attempt}",
            joined.join("+"),
            8 * proof.len()
        );
    }
    Ok(())
}

/// $\sum_x d_x \widetilde{eq}(x, \mathbf a)$, $a_0$ the most significant bit.
fn eval(field: &Fqk<4>, digits: &[i8], a: &[Ext<4>]) -> Ext<4> {
    let eq = |a: &[Ext<4>]| {
        let mut out = vec![field.one()];
        for aj in a {
            let mut next = Vec::with_capacity(2 * out.len());
            for v in &out {
                let one = field.mul(v, aj);
                next.push(field.sub(v, &one));
                next.push(one);
            }
            out = next;
        }
        out
    };
    let (hi, lo) = a.split_at(a.len() / 2);
    let (rows, cols) = (eq(hi), eq(lo));
    let zq = field.zq();
    let mut acc = field.zero();
    for (u, eu) in rows.iter().enumerate() {
        let mut row = field.zero();
        for (l, el) in cols.iter().enumerate() {
            let d = digits[u * cols.len() + l];
            row = field.add(&row, &field.scale(el, zq.from_i64(i64::from(d))));
        }
        acc = field.add(&acc, &field.mul(eu, &row));
    }
    acc
}
