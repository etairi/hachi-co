//! `hachi <l> [d] [iterations]`: commits to a random multilinear polynomial in `l` variables over
//! $\mathbb Z_q$, $q = 2^{32} - 99$, proves and verifies its evaluation at a random point of
//! $\mathbb F_{q^4}$, and prints the timings and the proof size. `d` is the ring degree (default
//! 64); `iterations` fixes the number of iterations (default: the automatic rule). The parameters
//! are the insecure test choices of `HachiChoices::toy`.

use hachi::{HachiChoices, HachiPcs};
use hachi_co_math::Ext;
use hachi_co_sampling::ShakePrg;
use hachi_co_sampling::uniform::uniform_mod_q;
use std::time::Instant;

const Q: u64 = 4294967197;
const K: usize = 4;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let parse = |i: usize, default: usize| {
        args.get(i)
            .map_or(Ok(default), |a| a.parse::<usize>())
            .unwrap_or_else(|_| {
                eprintln!("usage: hachi <l> [d] [iterations]");
                std::process::exit(2)
            })
    };
    let vars = parse(1, 16);
    let d = parse(2, 64);
    let choices = HachiChoices {
        fixed_iterations: args.get(3).map(|_| parse(3, 0)),
        ..HachiChoices::toy(d)
    };
    let pcs = match HachiPcs::<K>::new(Q, choices) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("parameters: {e}");
            std::process::exit(1)
        }
    };
    let mut prg = ShakePrg::new(&[1u8; 32], b"hachi/cli", 0);
    let coeffs = uniform_mod_q(&mut prg, Q, 1 << vars).expect("q > 0");
    let point: Vec<Ext<K>> = uniform_mod_q(&mut prg, Q, vars * K)
        .expect("q > 0")
        .as_chunks::<K>()
        .0
        .iter()
        .map(|c| Ext(*c))
        .collect();
    let run = || -> Result<(), String> {
        let t = Instant::now();
        let (com, state) = pcs.commit_multilinear(&coeffs).map_err(|e| e.to_string())?;
        println!("commit  {:>10.2?}", t.elapsed());
        let t = Instant::now();
        let (value, proof) = pcs
            .prove_multilinear(&state, &point, &[0u8; 32])
            .map_err(|e| e.to_string())?;
        println!(
            "prove   {:>10.2?}  ({} iterations)",
            t.elapsed(),
            proof.iterations.len()
        );
        let t = Instant::now();
        pcs.verify_multilinear(&com, vars, &point, &value, &proof, &[0u8; 32])
            .map_err(|e| e.to_string())?;
        println!("verify  {:>10.2?}", t.elapsed());
        let words = pcs.proof_to_words(&proof).len();
        println!("proof   {words} words ({} bytes)", 8 * words);
        Ok(())
    };
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
