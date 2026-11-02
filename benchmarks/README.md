# Benchmarks

Every party ran as its own process on one machine, over TCP on the loopback, with Hachi as the base PCS. These numbers
show how the code scales and where it spends memory. They are **not** a network measurement, and the machine was not
quiet, so read the times as indicative. Runs across real machines come later.

- **Machine.** Apple Silicon laptop, 10 cores (4 performance, 6 efficiency), 16 GiB, macOS 27.0.1, NEON. Other
  applications were open and swap was in use.
- **Software.** rustc 1.99.1, release profile, the working tree of 2026-10-31 on top of commit `ae1e79e`, proof format
  version 4.
- **Parameters.** `bench-l<l>-n<n>` are the insecure choices of `toy-n<n>` (d = 16, k = 4, b = 4, ω = 2) with l variables,
  with Hachi's default test choices. `secure-*` are the presets of `crates/params/presets/secure.json`, with Hachi's
  choices of `crates/params/presets/hachi.json`: estimates for binding, without hiding, and no proof of soundness.
- **Memory.** The peak footprint of `/usr/bin/time -l`. The resident size is useless here: under memory pressure it
  showed a 15.6 GiB run as 1.6 GiB.
- **Times.** The slowest party's Commit and Eval. Each party had its share of the cores, 10 threads alone, 5 with two
  parties, 2 with four and 1 with eight, and the verifier all 10.
- **Proof.** The measured encoding, always equal to the prediction of `hachi_co_params::sizes_with_base`. The raw lines
  in `local-2026-10-31.jsonl` also give `packed_bytes`, the size with each residue in ⌈log₂ q⌉ bits. The grid of
  2026-10-30, with one core per party and proof format version 3, stays in `local-2026-10-30.jsonl`.
- **Masks.** These grids predate the masks of hiding Hachi, so the commands below no longer reproduce them: from proof
  format version 5 on, `hachi.json` sets masks at 24 of the 26 presets, and the `bench-*` presets over the field take
  the toy masks. Commit `b7a9477` repeats the grids. Runs of `secure-l16-n2-a8` with n = 2 with masks, against the
  line of the same preset here (Eval 33.6 s, Verify 1.04 s, the proof 989,976 bytes, the larger party's peak 3.0 GiB):
  - `local-2026-11-01.jsonl`: the share polynomial masked and the blinders not (proof format 5). Eval 79.7 s, Verify
    1.71 s, the proof 992,088 bytes, peak 8.5 GiB. An earlier run took 64.9 s for Eval.
  - `local-2026-11-02.jsonl`: every polynomial masked, as in the draft's protocol (proof format 6, with
    the presets chosen again for masked blinders). Eval 59.4 s, Verify 1.23 s, the proof 2,044,520 bytes, peak 5.2 GiB.
    This is the size of a hiding proof at this preset, 2.07 times the unmasked one.

  The parameters of the first run differ from the second (b = 8 and height 11 at the first level, against b = 16 and
  height 8). One run each, on a laptop that was not quiet.

Commands, from the repository root. The seeds are SHA-256 of the tag `hachi-co/bench` and an index, so each line
reproduces its instance:

```text
cargo build --release -p hachi-co --bin hachi-co-party
python3 scripts/bench_grid.py --out benchmarks/local-2026-10-31.jsonl --scratch <dir> --ells 16 20 --ns 2 4 8 --budget-gib 12
python3 scripts/run_parties.py --preset bench-l24-n2 --n 2 --out <dir> --spill <dir>
python3 scripts/run_parties.py --preset bench-l20-n2-residues --n 2 --out <dir> --spill <dir>
python3 scripts/run_parties.py --preset bench-l16-n2 --n 2 --out <dir>
python3 scripts/run_parties.py --preset bench-l20-n1 --n 1 --out <dir>
python3 scripts/run_parties.py --preset bench-l24-n1 --n 1 --out <dir> --spill <dir>
python3 scripts/run_parties.py --preset bench-l26-n1 --n 1 --out <dir> --spill <dir>
python3 scripts/run_parties.py --preset secure-l16-n2-a8 --n 2 --out <dir> --spill <dir>
python3 scripts/run_parties.py --preset secure-l16-n2-a10 --n 2 --out <dir> --spill <dir>
python3 scripts/bench_table.py benchmarks/local-2026-10-31.jsonl
```

"On disk" means that the party keeps its blocks of f′ in a file and recomputes their digits when needed. The cells not
run need more memory than this machine has; the estimate is the fit in `scripts/bench_grid.py`, for all the parties.

## Grid

| preset | parties | on disk | Commit (s) | Eval (s) | Verify (s) | peak per party (GiB) | sent per party (MiB) | proof (KiB) | tool's size (KiB) | verifier's peak (GiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| bench-l16-n2 | 2 | yes | 0.1 | 2.6 | 0.15 | 0.43 | 2.3 | 80.7 | 80.7 | 0.10 |
| bench-l16-n4 | 4 | yes | 0.1 | 13.1 | 0.43 | 0.85 | 7.9 | 176.1 | 176.1 | 0.19 |
| bench-l16-n8 | 8 | yes | 0.3 | 27.5 | 0.80 | 0.83 | 18.8 | 352.0 | 352.0 | 0.25 |
| bench-l20-n2 | 2 | yes | 1.3 | 14.2 | 0.43 | 1.27 | 33.3 | 98.6 | 98.6 | 0.35 |
| bench-l20-n4 | 4 | yes | 2.7 | 58.0 | 1.16 | 2.48 | 104.1 | 208.9 | 208.9 | 0.71 |
| bench-l20-n8 | 8 | not run: about 34.4 GiB in all | | | | | | | | |
| bench-l24-n2 | 2 | yes | 27.2 | 164.8 | 2.20 | 7.76 | 517.8 | 129.4 | 129.4 | 2.43 |
| bench-l24-n4 | 4 | not run: about 65.2 GiB in all | | | | | | | | |
| bench-l24-n8 | 8 | not run: about 130.4 GiB in all | | | | | | | | |

## One party

| preset | parties | on disk | Commit (s) | Eval (s) | Verify (s) | peak per party (GiB) | sent per party (MiB) | proof (KiB) | tool's size (KiB) | verifier's peak (GiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| bench-l20-n1 | 1 | no | 0.7 | 8.3 | 0.30 | 2.14 | 0.0 | 54.6 | 54.6 | 0.31 |
| bench-l24-n1 | 1 | yes | 15.8 | 52.8 | 1.16 | 4.32 | 0.0 | 63.7 | 63.7 | 1.50 |
| bench-l26-n1 | 1 | yes | 67.5 | 194.7 | 3.50 | 8.92 | 0.0 | 72.5 | 72.5 | 3.34 |

## The two pads

| preset | parties | on disk | Commit (s) | Eval (s) | Verify (s) | peak per party (GiB) | sent per party (MiB) | proof (KiB) | tool's size (KiB) | verifier's peak (GiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| bench-l20-n2 | 2 | yes | 1.3 | 14.2 | 0.43 | 1.27 | 33.3 | 98.6 | 98.6 | 0.35 |
| bench-l20-n2-residues | 2 | yes | 1.3 | 9.7 | 0.38 | 1.21 | 33.3 | 91.7 | 91.7 | 0.35 |

## In memory against on disk

| preset | parties | on disk | Commit (s) | Eval (s) | Verify (s) | peak per party (GiB) | sent per party (MiB) | proof (KiB) | tool's size (KiB) | verifier's peak (GiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| bench-l16-n2 | 2 | yes | 0.1 | 2.6 | 0.15 | 0.43 | 2.3 | 80.7 | 80.7 | 0.10 |
| bench-l16-n2 | 2 | no | 0.1 | 3.9 | 0.16 | 0.50 | 2.3 | 80.7 | 80.7 | 0.10 |

## Secure presets

| preset | parties | on disk | Commit (s) | Eval (s) | Verify (s) | peak per party (GiB) | sent per party (MiB) | proof (KiB) | tool's size (KiB) | verifier's peak (GiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| secure-l16-n2-a8 | 2 | yes | 0.1 | 33.6 | 1.04 | 3.01 | 3.2 | 966.8 | 966.8 | 0.51 |
| secure-l16-n2-a10 | 2 | yes | 0.0 | 33.9 | 0.67 | 3.01 | 4.1 | 1411.6 | 1411.6 | 0.41 |

## Reading the tables

- Every run was accepted by the verifier. Six of the nine grid cells ran; `bench-l20-n8`, `bench-l24-n4` and
  `bench-l24-n8` need a larger machine.
- Against the grid of 2026-10-30, Eval takes 4 to 14 times less time, with more threads for most cells. Peak memory
  fell by up to 30% on the smaller cells and rose by 3 to 8% on the three largest. The two grids ran on a machine that
  was not quiet, so the ratios are indicative. Seven presets, rerun with the same settings, give the same output files
  before and after the speed work.
- Eval dominates. Commit is linear in 2^l and small; Eval follows the sumcheck's table, 2^l″, which grows with n as
  well as with l.
- More parties here also means fewer threads per party, so these tables overstate the cost of n. Runs across machines
  will show it without that effect.
- Peak memory follows the same table, not the polynomial: `bench-l26-n1` holds 2^26 field elements (2 GiB) on disk and
  peaks at 8.92 GiB.
- With the digits kept in memory, s_i alone would take about 54 GiB at l = 26 [derived, not run].
- Keeping f′ on disk costs little. The `bench-l16-n2` row in memory ran soon after `bench-l24-n2`, with the machine
  in swap, which likely explains its 3.9 s: four other runs of that cell on the same day took 2.7 to 3.0 s.
- The proof grows with n, as there are 3n base-PCS openings, and slowly with l.
- The draft's pad costs about 1.5 times the Eval time of the hidden option here, as it makes Hachi commit to k times as
  many residues. It was 3 times with one core per party. The comparison is at insecure parameters.
- No x86 vector instructions ran: Rosetta on this machine has no AVX, so how the transforms do on AVX2 or AVX-512 is
  not measured.
