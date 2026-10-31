# Benchmarks

Every party ran as its own process on one machine, over TCP on the loopback, with Hachi as the base PCS. These numbers
show how the code scales and where it spends memory. They are **not** a network measurement, and the machine was not
quiet, so read the times as indicative. Runs across real machines come later.

- **Machine.** Apple Silicon laptop, 10 cores, 16 GiB, macOS 27.0.1. Other applications were open and swap was in use.
  `parameters.sage` used one core during the grid's first runs.
- **Software.** rustc 1.99.1, release profile, the working tree of 2026-10-30 on top of commit `d4ea36a`.
- **Parameters.** `bench-l<l>-n<n>` are the insecure choices of `toy-n<n>` (d = 16, k = 4, b = 4, ω = 2) with l variables,
  with Hachi's default test choices. `secure-*` are the presets of `crates/params/presets/secure.json`, with Hachi's
  choices of `crates/params/presets/hachi.json`: estimates for binding, without hiding, and no proof of soundness.
- **Memory.** The peak footprint of `/usr/bin/time -l`. The resident size is useless here: under memory pressure it
  showed a 15.6 GiB run as 1.6 GiB.
- **Times.** The slowest party's Commit and Eval. A party uses about one core.
- **Proof.** The measured encoding, always equal to the prediction of `hachi_co_params::sizes_with_base`. The raw lines
  in `local-2026-10-30.jsonl` also give `packed_bytes`, the size with each residue in ⌈log₂ q⌉ bits.

Commands, from the repository root. The seeds are SHA-256 of the tag `hachi-co/bench` and an index, so each line
reproduces its instance:

```text
cargo build --release -p hachi-co --bin hachi-co-party
python3 scripts/bench_grid.py --out benchmarks/local-2026-10-30.jsonl --scratch <dir> --ells 16 20 --ns 2 4 8 --budget-gib 12
python3 scripts/run_parties.py --preset bench-l24-n2 --n 2 --out <dir> --spill <dir>
python3 scripts/run_parties.py --preset bench-l20-n2-residues --n 2 --out <dir> --spill <dir>
python3 scripts/run_parties.py --preset bench-l16-n2 --n 2 --out <dir>
python3 scripts/run_parties.py --preset bench-l20-n1 --n 1 --out <dir>
python3 scripts/run_parties.py --preset bench-l24-n1 --n 1 --out <dir> --spill <dir>
python3 scripts/run_parties.py --preset bench-l26-n1 --n 1 --out <dir> --spill <dir>
python3 scripts/run_parties.py --preset secure-l16-n2-a8 --n 2 --out <dir> --spill <dir>
python3 scripts/run_parties.py --preset secure-l16-n2-a10 --n 2 --out <dir> --spill <dir>
python3 scripts/bench_table.py benchmarks/local-2026-10-30.jsonl
```

"On disk" means that the party keeps its blocks of f′ in a file and recomputes their digits when needed. The cells not
run need more memory than this machine has; the estimate is the fit in `scripts/bench_grid.py`, for all the parties.

## Grid

| preset | parties | on disk | Commit (s) | Eval (s) | Verify (s) | peak per party (GiB) | sent per party (MiB) | proof (KiB) | tool's size (KiB) | verifier's peak (GiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| bench-l16-n2 | 2 | yes | 0.3 | 28.4 | 0.29 | 0.61 | 2.3 | 83.0 | 83.0 | 0.09 |
| bench-l16-n4 | 4 | yes | 0.5 | 73.5 | 0.77 | 1.02 | 7.9 | 182.3 | 182.3 | 0.18 |
| bench-l16-n8 | 8 | yes | 0.7 | 104.6 | 1.27 | 0.98 | 18.8 | 365.1 | 365.1 | 0.20 |
| bench-l20-n2 | 2 | yes | 5.7 | 122.7 | 0.86 | 1.56 | 33.3 | 101.9 | 101.9 | 0.36 |
| bench-l20-n4 | 4 | yes | 8.3 | 289.8 | 2.46 | 2.72 | 104.1 | 214.7 | 214.7 | 0.69 |
| bench-l20-n8 | 8 | not run: about 34.4 GiB in all | | | | | | | | |
| bench-l24-n2 | 2 | yes | 110.4 | 1144.0 | 4.81 | 7.44 | 517.8 | 133.4 | 133.4 | 2.44 |
| bench-l24-n4 | 4 | not run: about 65.2 GiB in all | | | | | | | | |
| bench-l24-n8 | 8 | not run: about 130.4 GiB in all | | | | | | | | |

## One party

| preset | parties | on disk | Commit (s) | Eval (s) | Verify (s) | peak per party (GiB) | sent per party (MiB) | proof (KiB) | tool's size (KiB) | verifier's peak (GiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| bench-l20-n1 | 1 | no | 4.0 | 96.2 | 0.69 | 2.19 | 0.0 | 56.2 | 56.2 | 0.31 |
| bench-l24-n1 | 1 | yes | 76.3 | 484.4 | 2.51 | 4.00 | 0.0 | 65.2 | 65.2 | 1.52 |
| bench-l26-n1 | 1 | yes | 371.5 | 1502.0 | 8.03 | 8.64 | 0.0 | 74.6 | 74.6 | 3.41 |

## The two pads

| preset | parties | on disk | Commit (s) | Eval (s) | Verify (s) | peak per party (GiB) | sent per party (MiB) | proof (KiB) | tool's size (KiB) | verifier's peak (GiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| bench-l20-n2 | 2 | yes | 5.7 | 122.7 | 0.86 | 1.56 | 33.3 | 101.9 | 101.9 | 0.36 |
| bench-l20-n2-residues | 2 | yes | 5.2 | 40.3 | 0.68 | 1.30 | 33.3 | 95.1 | 95.1 | 0.36 |

## In memory against on disk

| preset | parties | on disk | Commit (s) | Eval (s) | Verify (s) | peak per party (GiB) | sent per party (MiB) | proof (KiB) | tool's size (KiB) | verifier's peak (GiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| bench-l16-n2 | 2 | yes | 0.3 | 28.4 | 0.29 | 0.61 | 2.3 | 83.0 | 83.0 | 0.09 |
| bench-l16-n2 | 2 | no | 0.3 | 25.5 | 0.27 | 0.66 | 2.3 | 83.0 | 83.0 | 0.09 |

## Secure presets

| preset | parties | on disk | Commit (s) | Eval (s) | Verify (s) | peak per party (GiB) | sent per party (MiB) | proof (KiB) | tool's size (KiB) | verifier's peak (GiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| secure-l16-n2-a8 | 2 | yes | 0.5 | 335.7 | 1.85 | 3.39 | 3.2 | 970.4 | 970.4 | 0.49 |
| secure-l16-n2-a10 | 2 | yes | 0.3 | 486.6 | 1.33 | 3.20 | 4.1 | 1415.1 | 1415.1 | 0.40 |

## Reading the tables

- Every run was accepted by the verifier. Six of the nine grid cells ran; `bench-l20-n8`, `bench-l24-n4` and
  `bench-l24-n8` need a larger machine.
- Eval dominates. Commit is linear in 2^l and small; Eval follows the sumcheck's table, 2^l″, which grows with n as
  well as with l.
- Peak memory follows the same table, not the polynomial: `bench-l26-n1` holds 2^26 field elements (2 GiB) on disk and
  peaks at 8.64 GiB. In the one timeline sampled, `bench-l20-n1`, the peak is a short spike in Hachi's opening at the end
  of Eval, about 1.5 times the plateau of the collaborative sumcheck.
- With the digits kept in memory, s_i alone would take about 54 GiB at l = 26 [derived, not run].
- Keeping f′ on disk costs little: compare the two `bench-l16-n2` rows.
- The proof grows with n, as there are 3n base-PCS openings, and slowly with l.
- The draft's pad costs three times the Eval time of the hidden option here, as it makes Hachi commit to k times as
  many residues. The comparison is at insecure parameters.
