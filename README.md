# Hachi-Co

Hachi-Co is a collaborative polynomial commitment scheme (PCS) from lattices. Several provers hold
additive shares of a multilinear polynomial. Together they commit to it and prove its evaluations at
public points, and any n − 1 of them learn nothing about the polynomial beyond their own shares. The
scheme is plausibly post-quantum secure.

This repository contains the Rust implementation of the scheme and the scripts of the paper's
benchmarks. Each prover opens its share with a hiding variant of [Hachi](https://eprint.iacr.org/2026/156),
a lattice-based PCS. As an option, the openings can be composed with Greyhound, which makes the proof
about 12 times smaller for 8 provers.

| Presets | Security | Used for |
|---|---|---|
| `secure-l<l>-n<n>-a8` | 115 bits | the main benchmarks |
| `secure100-l<l>-n<n>-a8` | 100 bits | the comparison with PPP26 |
| `toy-n<n>`, `bench-l<l>-n<n>` | none | tests and quick runs |

Here `l` is the number of variables of the polynomial (2^l coefficients) and `n` the number of
provers. A suffix changes how a preset runs on the same instance:

- `-greyhound` composes the openings with Greyhound.
- `-nonhiding` runs the base PCS without its hiding choices, to measure what hiding costs. The
  result is not hiding.
- `-merged` gives each prover one commitment and one opening instead of three, an option that is
  not in the paper.

## Requirements

- Rust 1.95 or newer.
- A C compiler, for the mimalloc allocator of the party binary. Build with
  `--no-default-features` to use the system allocator instead.
- Python 3, for the scripts in `scripts/`.
- Linux or macOS.

On x86-64, `.cargo/config.toml` builds for the CPU of the building machine, so a binary runs only on
machines with the same instructions. Build on the machine that you measure on.

Optional:

- SageMath, for the reference implementation in `ref/` and for recomputing parameters.
- Docker, to emulate a wide-area network on one machine.
- The Google Cloud SDK, to run the provers on separate cloud instances.
- CMake and MPI, to build and run PPP26.

## Quick start

```sh
cargo build --release
cargo test --workspace --release
python3 scripts/run_parties.py --preset bench-l16-n2 --n 2 --out /tmp/hachi-co-run
```

The last command starts two provers as separate processes on this machine. They run Commit and Eval
over TCP, the verifier checks the proof, and the script prints one JSON object with the timings, the
peak memory of each prover, the bytes sent and the proof size. The seeds are derived from a tag, so
the command line reproduces the run.

To try a secure instance, replace the preset by, for example, `secure-l20-n2-a8`. With one thread per
prover it takes a few minutes. Other useful options:

- `--threads <k>` gives each prover k threads.
- `--spill <directory>` keeps each prover's large blocks on disk, so that big instances fit in memory.
- `--single` runs the single-prover baseline: one prover commits to the whole polynomial with Hachi.

## Running the provers on several machines

`hachi-co-party` runs one prover as its own process, from a JSON configuration. Write one
configuration per prover, start the provers in any order, then verify the proof. For three provers
on one machine:

```sh
cargo build --release -p hachi-co --bin hachi-co-party
for p in 1 2 3; do
cat > party-$p.json <<EOF
{
  "preset": "toy-n3",
  "n": 3,
  "party": $p,
  "addresses": ["127.0.0.1:7101", "127.0.0.1:7102", "127.0.0.1:7103"],
  "session": "5e55105e55105e55105e55105e55105e55105e55105e55105e55105e55105e55",
  "setup_seed": "0101010101010101010101010101010101010101010101010101010101010101",
  "instance_seed": "0202020202020202020202020202020202020202020202020202020202020202",
  "party_seed": "$(printf '%064d' $p)",
  "output": "out/party-$p"
}
EOF
done
for p in 1 2 3; do target/release/hachi-co-party party-$p.json > party-$p.log & done; wait
cat party-1.log
target/release/hachi-co-party --verify out/party-1
```

Each prover prints its timings and the bytes it sent and received. It writes the proof
(`proof.bin`) and the statement (`statement.json`) into its output directory, and
`hachi-co-party --verify` needs only that directory.

On several machines, list the address at which each prover can be reached and use a fresh `session`
for every run. The optional fields `connect_timeout_ms` (default 30000) and `recv_timeout_ms`
(default 60000) set the timeouts. The exit status is 0 on success, 1 if the run or the verification
failed, 2 for a usage or configuration error, and 3 if the prover's address is in use.

## Reproducing the benchmarks

The results and notes on earlier runs are in `benchmarks/README.md`.

**All provers on one machine.** `scripts/local_grid.py` runs a plan of instances over the loopback and
appends one JSON line per run. An interrupted plan resumes where it stopped.

```sh
cargo build --release --locked -p hachi-co --bin hachi-co-party
python3 scripts/local_grid.py --plan phase1 --list        # what the plan runs
python3 scripts/local_grid.py --plan phase1 --out runs/local/phase1.jsonl --work runs/local/work
python3 scripts/local_grid.py --plan greyhound --out runs/local/greyhound.jsonl --work runs/local/work-gh
```

The plans are `smoke` (a quick check of the machine), `calibration`, `phase1` (l = 20, 22, 24 and
n = 2, 4, 8, with the baselines) and `greyhound` (the same instances composed with Greyhound).

**One prover per cloud instance.** `scripts/gcp.py` creates one Google Cloud instance per prover,
builds the code on each, writes the hosts file and can emulate a network with `tc netem`. It only
prints the `gcloud` commands unless you pass `--execute`. `scripts/wan_grid.py` then runs the presets
in PPP26's three network settings: a LAN (10 Gbps, 2 ms round trip) and two wide-area networks
(200 Mbps, 50 and 100 ms).

```sh
cp scripts/gcp.example.json runs/gcp.json           # set the project, zones and machine type
python3 scripts/gcp.py --config runs/gcp.json create --execute
python3 scripts/gcp.py --config runs/gcp.json build --execute
python3 scripts/gcp.py --config runs/gcp.json hosts --execute --out runs/gcp/hosts.json
python3 scripts/wan_grid.py --hosts runs/gcp/hosts.json --switch gcp --gcp-config runs/gcp.json \
    --settings lan wan50 wan100 --presets secure100-l20-n4-a8 --threads 1 --reps 1 --spill \
    --out runs/gcp-grid.jsonl
python3 scripts/gcp.py --config runs/gcp.json delete --execute
```

To emulate the network on one machine instead, start one Docker container per prover with
`scripts/wan_docker.py up` and pass `--switch docker` to `scripts/wan_grid.py`.

**PPP26.** `scripts/ppp26.py` fetches the implementation of PPP26 at a fixed commit, applies a small
patch that adds timers, and builds and runs it on the same cloud instances (`fetch`, `gcp-build`,
`gcp-mpi`, `gcp-run`).

**Tables and figures.** `scripts/paper_tables.py` reads the JSON lines of the runs and prints the
numbers and plot coordinates of the paper:

```sh
python3 scripts/paper_tables.py --local runs/local/phase1.jsonl --wan runs/gcp-grid.jsonl
```

## Parameters

The secure presets are in `crates/params/presets/`, each with a README that lists every parameter
and its security estimate. They were computed with `parameters.sage`, which uses SageMath and the
lattice estimator (it clones the estimator into a temporary directory and removes it at the end).
`scripts/hachi-parameters/` chooses the parameters of Hachi for these presets. The security levels
are estimates, not proofs.

## Repository layout

| Path | Contents |
|---|---|
| `crates/hachico` | The PCS: Setup, Commit, Eval and Verify, the proof format, and the `hachi-co-party` program |
| `crates/hachi` | Hachi, the base PCS, with its hiding variant |
| `crates/greyhound` | Greyhound with LaBRADOR, for the optional composition |
| `crates/commit` | Commit and Open |
| `crates/conv` | The conversion from F_{q^k} to R_q |
| `crates/r2l` | The ring-to-linear step with rejection sampling |
| `crates/sumcheck` | The collaborative sumcheck and its verifier |
| `crates/core` | The protocol context shared by the steps, and the interface to the base PCS |
| `crates/mpc` | Communication between the provers: in memory or over TCP |
| `crates/params` | The Setup, the presets and their constraint checks |
| `crates/math` | Arithmetic modulo q, in R_q and in F_{q^k} |
| `crates/sampling` | Samplers, including discrete Gaussians |
| `crates/transcript` | The Fiat–Shamir transcript |
| `ref/` | A SageMath reference implementation, used as a test oracle |
| `scripts/` | Benchmark, cluster and table scripts |
| `benchmarks/` | Results of earlier runs |

`cargo docs` builds the documentation of every crate, including private items, into `target/doc/`.
`cargo docs-open` also opens it. The equations are rendered by KaTeX, which `katex-header.html`
loads from a CDN.

## Differences from the Hachi paper

- A multilinear opening binds its partial evaluations to the committed polynomial with a
  ring-switching sumcheck (Hachi, Section 3.2).
- Every iteration except the last commits its next witness as Section 3.1 commits a polynomial and
  reduces its claim by Theorem 2, without partial evaluations (Section 4.5).
- Range checks use the b balanced digits instead of the set {−b + 1, …, b − 1}.
- The last iteration sends its witness in the clear, after as many iterations as give the smallest
  proof, unless the openings are composed with Greyhound.

## Caveats

This is research code. It has not been audited and does not run in constant time. The TCP transport
is neither encrypted nor authenticated. In the benchmark harness, the provers derive the instance and
their shares from shared seeds, so it is not a deployment. The `toy` and `bench` presets are
insecure.

## Attribution and license

`crates/hachi` started from the public Hachi prototype of its authors,
[georgeorourke/hachi-pcs](https://github.com/georgeorourke/hachi-pcs) (Ngoc Khanh Nguyen, George
O'Rourke and Jiapeng Zhang, "Hachi: Efficient Lattice-Based Multilinear Polynomial Commitments over
Extension Fields", Cryptology ePrint Archive 2026/156). Parts of the NTT, CRT, samplers and transcript
are adapted from the MIT-licensed library jali, whose LaBRADOR the Greyhound composition uses.

The upstream prototype has no license file, so no license is inherited from it. The MIT license in
`LICENSE` covers the contributions of this repository.
