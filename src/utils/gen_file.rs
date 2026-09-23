//! Generation of a random witness file (feature `gen_file`): $2^\ell$ coefficients uniform in
//! $\[0, q)$, written as 32-bit little-endian integers, the format read by
//! [`crate::stream::file_stream::U64FileStream`]. `main` uses it to produce a benchmarking
//! witness of the requested size before committing to it.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use rand_chacha::ChaCha12Rng;

use crate::arithmetic::utils::{Logarithm, rand_int};
#[cfg(feature = "verbose")]
use crate::utils::verbose::progress_bar;

/// Bytes per integer in the file: 32-bit little-endian, matching `U64FileStream`.
const INT_WIDTH: usize = 4;

/// Write $2^l$ independent uniform residues in $\[0, q)$ to `filename` as 32-bit little-endian
/// integers, in $2^{l - 20}$ blocks of $2^{20}$ integers (4 MiB each), reporting progress with
/// `progress_bar` when feature `verbose` is on.
///
/// * Requires $l \ge 20$ (asserted) and $q \le 2^{32}$, since each residue is truncated to a
///   `u32` before writing; the default $q = 2^{32} - 99$ satisfies this.
/// * If `filename` already exists nothing is written and a message is printed: the existing
///   file is reused as it is, whatever its size or contents.
/// * Randomness comes from a `ChaCha12Rng` seeded by `rand::make_rng` from the process's
///   thread RNG, so the file is not reproducible from a recorded seed; each residue is drawn
///   with [`rand_int`] and `logq = q.log()`.
/// * Panics if the file cannot be created or written.
pub fn write_random_data(
    filename: &str,
    l: usize,
    q: u64
) {
    assert!(l >= 20);
    
    if Path::new(filename).exists() {
        println!("Not generating file: already exists");
        return;
    }

    // Create the file in append mode
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(filename)
        .expect("Failed to open the file");

    let reps = 1 << (l - 20);
    let buf_size = INT_WIDTH * (1 << 20);
    let mut arr = vec![0u64; 1 << 20];
    let mut buf = vec![0u8; buf_size];

    let mut rng: ChaCha12Rng = rand::make_rng();

    // Write all of the coefficients
    for rep in 0..reps {
        // Fill the array
        for i in 0..arr.len() {
            arr[i] = rand_int(q, q.log(), &mut rng);
        }

        // Iterate over every element and convert it to 4 bytes
        for i in 0..(1 << 20) {
            let bytes: [u8; INT_WIDTH] = (arr[i] as u32).to_le_bytes();

            for j in 0..INT_WIDTH {
                buf[INT_WIDTH * i + j] = bytes[j];
            }
        }

        file.write_all(&buf)
                .expect("Failed to write to the file");

        #[cfg(feature = "verbose")]
        progress_bar("Generating file", rep, reps);
    }
}
