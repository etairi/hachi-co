# Hachi PCS prototpye 

## Build and Run
Build: `cargo build --release`

Usage: `./target/release/hachi-co <witness-file> <l>`
    - Runs the PCS scheme for witness of size 2^l (streamed from the specified file)
    - Outputs benchmarks for Commit, Prove, Verify
    - Evaluates at the fixed point x_t = 1234 + 7t (mod q), t = 0..l-1 (distinct coordinates, so that
      any mismatch in the variable order between prover, verifier and the reference evaluation is caught)

### Generate witness file and run
- `cargo build --release --features=gen_file`

### Run with AVX-512 (requires compatible CPU)
- `cargo +nightly build --release --features=nightly`