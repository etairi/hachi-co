mod arithmetic;
mod stream;
mod utils;
mod hachi;

#[cfg(feature = "gen_file")]
use crate::utils::gen_file::write_random_data;

use crate::stream::Stream;
use crate::stream::file_stream::U64FileStream;

use crate::arithmetic::utils::multi_lin_coeff_int;

use crate::hachi::Hachi;
use crate::hachi::setup::Setup;
use crate::hachi::commit::Commit;
use crate::hachi::prove::Prove;
use crate::hachi::verify::Verify;

fn main() {
    time_graph::enable_data_collection(true);

    // get witness file and length
    let args: Vec<String> = std::env::args().collect();
    let witness_file = &args[1];
    let l = *&args[2].parse::<usize>().unwrap();

    // public parameters
    let params = Hachi::setup(l, true);

    #[cfg(feature = "gen_file")]
    // Produce a dummy witness file containing random coefficients.
    write_random_data(witness_file, params.l, params.q);

    // commit to the witness
    let mut witness = U64FileStream::init(witness_file, 0);
    let com = Hachi::commit(&mut witness, &params);

    // create an evaluation point with distinct coordinates (an all-equal point would hide
    // any variable-order mismatch between prover, verifier and this reference evaluation)
    let x: Vec<u64> = (0..params.l).map(|i| (1234 + 7 * i as u64) % params.q).collect();
    let proof = Hachi::prove(&mut witness, &params, &x, &com);

    // compute claimed evaluation
    println!("\nCalculating evaluation...");
    let mut y = 0;

    witness.reset();

    let r = 20;
    let m = params.l - r;
    let mut buf = vec![0u64; 1 << r];

    // The scheme pairs the variables with a coefficient's position p = (chunk, element, ring index)
    // as chunk <-> x[0..params.r], element <-> x[params.r..params.r + params.m], ring index <-> the
    // last log d variables, while p has the ring index in its low bits; permute x accordingly.
    let (pr, pm) = (params.r, params.m);
    let mut x_perm: Vec<u64> = Vec::with_capacity(params.l);
    x_perm.extend_from_slice(&x[pr + pm..params.l]);
    x_perm.extend_from_slice(&x[pr..pr + pm]);
    x_perm.extend_from_slice(&x[0..pr]);

    for i in 0..1 << m {
        witness.read(&mut buf);

        for j in 0..1 << r {
            let a = multi_lin_coeff_int(&x_perm, i << r | j, params.l, params.q);
            y = (y + a * buf[j]) % params.q;
        }
    }

    // verification
    Hachi::verify(&params, &x, y, &com.u, &proof);

    let graph = time_graph::get_full_graph();
    println!("{}", graph.as_dot());    
}
