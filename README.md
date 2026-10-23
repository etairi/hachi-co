# Hachi-Co

A Rust implementation, in progress, of **Hachi-Co**, the collaborative lattice-based polynomial commitment
scheme of the draft "Collaborative Polynomial Commitments from Lattices". Several provers hold additive shares
of a multilinear polynomial and jointly commit to it and prove its evaluations. The repository started from
the single-prover **Hachi** prototype, which was ported onto the workspace's arithmetic and completed; it is the
base PCS of Hachi-Co.

## Workspace

| Crate | Package | Contents |
|---|---|---|
| `crates/hachi` | `hachi` | Hachi, the base PCS: inner-outer commitments to multilinear polynomials over $\mathbb Z_q$ and to vectors over $\mathbb F_{q^k}$, openings at points in $\mathbb F_{q^k}$ (multilinear evaluations and inner products with public weights; a multilinear opening binds its partial evaluations to the committed polynomial by a sumcheck over their Frobenius conjugates, a deviation from the paper's Section 3.2, whose check leaves them free), the recursive iterations of the paper (each iteration's reduction to the next claim binds its partial evaluations the same way, a deviation from Section 4.5) with the final witness in the clear (as many as give the smallest proof), Fiat–Shamir, and packed encodings ($\lceil \log_2 q \rceil$ bits per residue, $\lceil \log_2 b \rceil$ per digit of the final witness). Optionally randomized commitments ($\mathbf t_i = \mathbf A \mathbf s_i + \mathbf A' \mathbf r_i$, with discrete Gaussian or uniform-digit randomness): at an MLWE rank of real security a commitment then hides the committed object until it is opened; the openings are not hiding yet, and the toy choices are not secure. |
| `crates/math` | `hachi-co-math` | Arithmetic for a modulus chosen at run time: $\mathbb Z_q$, exact products in $\mathbf R_q = \mathbb Z_q[X]/(X^d+1)$, gadget decompositions, $\mathbb F_{q^k}$ in Hachi's subfield basis, $\mathbf R_q^H$, the trace, $\psi$ and $E$. |
| `crates/sampling` | `hachi-co-sampling` | SHAKE-based PRG, exact uniform and short samplers, $\ell_1$-ball challenges, matrices from seeds, uniform rejection sampling, and discrete Gaussians $D_{\mathbb Z,\sigma}$ ($\sigma = 1, 2, 4, 8, 16$) from certified 128-bit tables. |
| `crates/transcript` | `hachi-co-transcript` | Framed, domain-separated Fiat–Shamir transcript with an attempt counter. |
| `crates/params` | `hachi-co-params` | The draft's Setup cascade, a constraint report, toy and secure presets. |
| `crates/mpc` | `hachi-co-mpc` | The communication interface of the party code (F_AH, gather, abort agreement, pairwise exchange, all-gather) and its implementations: a loopback for one party, and an n-party endpoint over any point-to-point transport, which realizes F_AH as a masked broadcast (pseudorandom zero sharings from pairwise seeds, or fresh uniform ones), checks the framing, injects failures and counts bytes per call. An in-memory transport runs the parties on threads. |
| `crates/core` | `hachi-co-core` | The shared protocol context: parameters and public matrices, the layout of the witness z′, the matrix M of the linear relation, the data passed between the building blocks, and the sources of randomness and Fiat–Shamir coins (with replay of the Sage fixtures for tests). The interface to the base PCS (`BasePcs`), with the conversion of hypercube values to monomial coefficients, and `ClearPcs`, a reference base PCS for the tests that opens by revealing the polynomial. |
| `crates/commit` | `hachi-co-commit` | Commit (resharing, PConv, inner Ajtai commitments, F_AH(B)) and Open. |
| `crates/conv` | `hachi-co-conv` | Πconv, the reduction from F_{q^k} to R_q, and the verifier's check y = E(u). |
| `crates/r2l` | `hachi-co-r2l` | The ring-to-linear step: masks, rejection sampling, aborts and retries, and the shares of the linear relation. |
| `crates/sumcheck` | `hachi-co-sumcheck` | The collaborative sumcheck and its verifier. Each prover commits to its share polynomial and its two blinders with the base PCS before the coins and opens them at the sumcheck's point; the verifier checks the 3n openings against the final values. |
| `crates/hachico` | `hachi-co` | The PCS: Setup, Commit, Eval and Verify with Fiat–Shamir, with Hachi as the base PCS (`HachiBase`, whose public matrices come from the setup seed), the proof format and its size. Its tests run Commit and Eval with 1 to 8 provers, one thread each, with Hachi and with `ClearPcs`. Not hiding, and the parameters are insecure test choices. |
| `ref/` | — | A SageMath reference implementation of the building blocks, used as a test oracle and fixture generator. |

`parameters.sage` computes parameters and security estimates. It clones the lattice estimator into a
temporary directory and removes it at exit.

## Build and test

```text
cargo build --release
cargo test --workspace --release
```

Hachi's command-line program:

```text
cargo run --release -p hachi -- <l> [d] [iterations]
```

It commits to a random multilinear polynomial in $\ell$ variables over $\mathbb Z_q$, $q = 2^{32} - 99$, proves
and verifies its evaluation at a random point of $\mathbb F_{q^4}$, and prints the timings and the proof size.
`d` is the ring degree (default 64) and `iterations` fixes the number of iterations (default: the number whose
proof is smallest). The parameters are insecure test choices.

## Documentation

- `cargo docs` builds the documentation of every workspace crate, including private items, into
  `target/doc/`; `cargo docs-open` also opens it. Both are aliases in `.cargo/config.toml`.
- Equations are written in LaTeX and rendered by KaTeX, which `katex-header.html` loads from a CDN.
- The implementation plan and design notes live in `docs/`, which is not tracked by git.

## Attribution

This repository started from the public Hachi prototype by the authors of the Hachi paper,
[georgeorourke/hachi-pcs](https://github.com/georgeorourke/hachi-pcs) (Ngoc Khanh Nguyen, George O'Rourke
and Jiapeng Zhang, "Hachi: Efficient Lattice-Based Multilinear Polynomial Commitments over Extension Fields",
Cryptology ePrint Archive 2026/156). `crates/hachi` started from that code.

Parts of the NTT, CRT, samplers and transcript are adapted from the author's own MIT-licensed library jali.

## License

The upstream repository does not include a license file, so no license is inherited from it. The MIT license
in `LICENSE` covers the contributions made in this repository.
