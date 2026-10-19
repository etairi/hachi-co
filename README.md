# Hachi-Co

A Rust implementation, in progress, of **Hachi-Co**, the collaborative lattice-based polynomial commitment
scheme of the draft "Collaborative Polynomial Commitments from Lattices". Several provers hold additive shares
of a multilinear polynomial and jointly commit to it and prove its evaluations. The repository started from
the single-prover **Hachi** prototype, which is kept unchanged as its own crate.

## Workspace

| Crate | Package | Contents |
|---|---|---|
| `crates/hachi` | `hachi` | The legacy single-prover Hachi prototype: one round of Hachi over $\mathbb Z_q$ with $q = 2^{32} - 99$. Kept byte-identical; a golden test pins its output. |
| `crates/math` | `hachi-co-math` | Arithmetic for a modulus chosen at run time: $\mathbb Z_q$, exact products in $\mathbf R_q = \mathbb Z_q[X]/(X^d+1)$, gadget decompositions, $\mathbb F_{q^k}$ in Hachi's subfield basis, $\mathbf R_q^H$, the trace, $\psi$ and $E$. |
| `crates/sampling` | `hachi-co-sampling` | SHAKE-based PRG, exact uniform and short samplers, $\ell_1$-ball challenges, matrices from seeds, uniform rejection sampling. |
| `crates/transcript` | `hachi-co-transcript` | Framed, domain-separated Fiat–Shamir transcript with an attempt counter. |
| `crates/params` | `hachi-co-params` | The draft's Setup cascade, a constraint report, toy and secure presets. |
| `crates/mpc` | `hachi-co-mpc` | The communication interface of the party code (F_AH, gather, abort agreement, pairwise exchange, all-gather) and its implementations: a loopback for one party, and an n-party endpoint over any point-to-point transport, which realizes F_AH as a masked broadcast (pseudorandom zero sharings from pairwise seeds, or fresh uniform ones), checks the framing, injects failures and counts bytes per call. An in-memory transport runs the parties on threads. |
| `crates/core` | `hachi-co-core` | The shared protocol context: parameters and public matrices, the layout of the witness z′, the matrix M of the linear relation, the data passed between the building blocks, and the sources of randomness and Fiat–Shamir coins (with replay of the Sage fixtures for tests). |
| `crates/commit` | `hachi-co-commit` | Commit (resharing, PConv, inner Ajtai commitments, F_AH(B)) and Open. |
| `crates/conv` | `hachi-co-conv` | Πconv, the reduction from F_{q^k} to R_q, and the verifier's check y = E(u). |
| `crates/r2l` | `hachi-co-r2l` | The ring-to-linear step: masks, rejection sampling, aborts and retries, and the shares of the linear relation. |
| `crates/sumcheck` | `hachi-co-sumcheck` | The collaborative sumcheck and its verifier, which returns the final evaluation claims for the base PCS. |
| `crates/hachico` | `hachi-co` | The PCS: Setup, Commit, Eval and Verify with Fiat–Shamir, the proof format and its size. Its tests run Commit and Eval with 1 to 8 provers, one thread each. Hachi is not yet the base PCS: the verifier returns the final evaluation claims that Hachi will prove. |
| `ref/` | — | A SageMath reference implementation of the building blocks, used as a test oracle and fixture generator. |

`parameters.sage` computes parameters and security estimates. It clones the lattice estimator into a
temporary directory and removes it at exit.

## Build and test

```text
cargo build --release
cargo test --workspace --release
```

The legacy prototype's command-line program:

```text
cargo run --release -p hachi -- <witness-file> <l>
```

It runs commit, prove and verify for a witness of $2^\ell$ coefficients ($\ell \ge 20$) streamed from the
file, and prints timings. It evaluates at the point $x_t = 1234 + 7t \bmod q$, whose coordinates are distinct
so that a variable-order mismatch would be caught. The feature `gen_file` writes a random witness file first;
the feature `nightly` enables tfhe-ntt's AVX-512 path (nightly toolchain, compatible CPU).

## Documentation

- `cargo docs` builds the documentation of every workspace crate, including private items, into
  `target/doc/`; `cargo docs-open` also opens it. Both are aliases in `.cargo/config.toml`.
- Equations are written in LaTeX and rendered by KaTeX, which `katex-header.html` loads from a CDN.
- The implementation plan and design notes live in `docs/`, which is not tracked by git.

## Attribution

This repository started from the public Hachi prototype by the authors of the Hachi paper,
[georgeorourke/hachi-pcs](https://github.com/georgeorourke/hachi-pcs) (Ngoc Khanh Nguyen, George O'Rourke
and Jiapeng Zhang, "Hachi: Efficient Lattice-Based Multilinear Polynomial Commitments over Extension Fields",
Cryptology ePrint Archive 2026/156). That code is now `crates/hachi`.

Parts of the NTT, CRT, samplers and transcript are adapted from the author's own MIT-licensed library jali.

## License

The upstream repository does not include a license file, so no license is inherited from it. The MIT license
in `LICENSE` covers the contributions made in this repository.
