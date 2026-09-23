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

## Documentation
- `cargo docs` builds the API documentation of this crate, including its private modules, into
  `target/doc/hachi_co/index.html`; `cargo docs-open` also opens it in the browser (both are aliases
  defined in `.cargo/config.toml`).
- Equations in the documentation are written in LaTeX and rendered by KaTeX, which `katex-header.html`
  loads from a CDN (network access is needed to render them). The same header is configured for docs.rs
  in `Cargo.toml`.

## Attribution
This repository started from the public Hachi prototype by the authors of the Hachi paper,
[georgeorourke/hachi-pcs](https://github.com/georgeorourke/hachi-pcs) (Ngoc Khanh Nguyen, George O'Rourke
and Jiapeng Zhang, "Hachi: Efficient Lattice-Based Multilinear Polynomial Commitments over Extension Fields",
Cryptology ePrint Archive 2026/156).


## License
The upstream repository does not include a license file, so no license is inherited from it. The MIT license in `LICENSE` covers the contributions
made in this repository. See `LICENSE` and the attribution above.
