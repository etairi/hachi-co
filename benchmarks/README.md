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
  format version 5 on, `hachi.json` sets masks, at 24 of the 26 presets in versions 5 and 6 and at all 26 from version
  7 on, and the `bench-*` presets over the field take the toy masks. Commit `b7a9477` repeats the grids. Runs of
  `secure-l16-n2-a8` with n = 2 with masks, against the line of the same preset here (Eval 33.6 s, Verify 1.04 s, the
  proof 989,976 bytes, the larger party's peak 3.0 GiB):
  - `local-2026-11-01.jsonl`: the share polynomial masked and the blinders not (proof format 5). Eval 79.7 s, Verify
    1.71 s, the proof 992,088 bytes, peak 8.5 GiB. An earlier run took 64.9 s for Eval.
  - `local-2026-11-02.jsonl`, first line: every polynomial masked, as in the draft's protocol (proof format 6, with
    the presets chosen again for masked blinders). Eval 59.4 s, Verify 1.23 s, the proof 2,044,520 bytes, peak 5.2 GiB.
    This is the size of a hiding proof at this preset, 2.07 times the unmasked one.
  - its second line: the same with merged commitments (the preset `secure-l16-n2-a8-merged`), an option that is not
    in the draft. Eval 59.0 s, Verify 1.20 s, the proof 1,196,392 bytes, peak 5.3 GiB.

  The parameters of the first run differ from the others (b = 8 and height 11 at the first level, against b = 16 and
  height 8). One run each, on a laptop that was not quiet.

  Every run here predates two later changes of the parameters: proof format version 7 sizes the masks with ⌈b/2⌉
  instead of b, and the draft's commit `2ee6507` sets β₁ = 2^(r − ⌈log n⌉) ω ⌊b/2⌋ instead of 2^(r − ⌈log n⌉) ω b,
  which halves β₁, γ and β at every preset and lowers q at 7 of the 26 secure presets. The two measured here,
  `secure-l16-n2-a8` and `secure-l16-n2-a10`, keep their q, and this change leaves their proof sizes as they were.

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
python3 scripts/run_parties.py --preset secure-l16-n2-a8-merged --n 2 --out <dir> --spill <dir>
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

## Across machines and emulated networks

These tools, added on 2026-11-12, prepare the wide-area runs and the comparison with PPP26 (ePrint 2026/729). No grid
has been run with them yet.

- **`scripts/cluster.py`** runs the parties of one instance on the hosts of a hosts file (transport `ssh`, `docker` or
  `local`), then the verifier, and prints one JSON line per repetition (`--reps`). Each line records the slowest
  party's setup, connect, Commit, Eval and network times, every party's peak memory, payload and wire bytes and rounds
  (and party 1's table per label), the verifier's time and peak, the proof's size against the parameter tool's, and
  the setting: each host's kernel, cores, CPU and its flags, memory and memory limit, cloud machine type and zone,
  emulated network, the build (commit, local changes, rustc, source hash), and the round-trip times (`ping`, from party 1
  or `--rtt-all-pairs`) and TCP throughput (`--bandwidth`, iperf3, slow start omitted) between the hosts. The instance
  is the same in every repetition and the session is fresh.
- **`scripts/wan_docker.py`** emulates a wide-area network here: one container per party (`docker/Dockerfile`, built from
  the working tree) on a private bridge network, with `tc netem` delaying each container's outgoing traffic by half the
  round-trip time and capping its rate, each container pinned to its own cores. The containers share one Linux VM
  (arm64 on this Mac), so they measure the network's effect, not a cloud instance's speed.
- **`scripts/gcp.py`** creates one Google Cloud instance per party, builds `hachi-co-party` on each from a tarball of
  the working tree, writes the hosts file and an ssh configuration for `cluster.py`, optionally emulates a fixed RTT
  with `tc netem`, and deletes everything afterwards. It only prints the gcloud commands unless given `--execute`.
  `scripts/gcp.example.json` is a configuration to copy.
- **`scripts/bench_table.py --medians`** prints the lines of `cluster.py` as a table of medians, one row per preset,
  number of parties, setting and threads per party.
- **`scripts/wan_grid.py`** runs a grid: for each network setting it switches the emulation of the hosts (`docker`
  through `wan_docker.py netem`, `gcp` through `gcp.py netem`, or `none` for real regions), then runs `cluster.py` for
  each preset and, on the cloud instances, `ppp26.py gcp-run` for each PPP26 size, appending one line per repetition.
  A rerun skips the repetitions already recorded, so an interrupted grid resumes. The settings `lan` (RTT 2 ms,
  10 Gbit/s), `wan50` and `wan100` (RTT 50 and 100 ms, 200 Mbit/s) are those of PPP26.
- **`scripts/ppp26.py`** runs PPP26's collaborative PCS from their repository at commit `1c50a74` (the paper's reference
  [46]) the same way, in its PCS-only mode (`pigeon 1 <log2 M> <rate>`: commitment and zero-knowledge opening of a
  polynomial with M = 2^l coefficients over F_{p^2}, p = 2^61 - 1, packing factor n/4, so n >= 4). Two additions of ours,
  both in `scripts/ppp26/`, leave their computation unchanged: `timing.patch` prints, per rank, the commit and open
  times and the parts the mode's timer leaves out (`dummy_setup`, where rank 0 plays a dealer of the random shares,
  and `prepare_mask_shares`, every prover's encoding and commitment of its masks), and `mpi_count.c`, preloaded,
  counts the payload bytes each rank sends, since the mode prints "Com: 0 MB" (it sums a local variable that stays 0)
  and the code's own counter misses the PCS's messages. On the instances, `gcp-build` builds it, `gcp-mpi` sets up ssh
  between the instances, MPI's host file and a firewall rule for MPI's ports, and `gcp-run` runs it with one rank per
  instance and records the setting as `cluster.py` does. Its code needs x86-64 with AVX2 and MPI. `docker-build` and
  `docker-run` build and run it in one amd64 container, which on this Mac runs under Rosetta: a test of the build and
  the parsing, whose times mean nothing.

PPP26 measure a LAN of 10 Gbps with an RTT of 2 ms and a WAN emulated with tc at 200 Mbps with an RTT of 50 ms and of
100 ms, with one vCPU per prover (`--threads 1` here), and their implementation targets about 100 bits: the presets
`secure100-*` are chosen at that target for the comparison, and `secure-*` at 115 bits. Their code needs x86-64 with
AVX2 and MPI, so it runs on the cloud instances, not on this Mac.

```text
python3 scripts/wan_docker.py build
python3 scripts/wan_docker.py up --n 4 --rtt-ms 50 --rate-mbit 200 --cpus 2 --hosts runs/docker/hosts.json
python3 scripts/cluster.py --hosts runs/docker/hosts.json --preset bench-l16-n4 --out runs/docker --reps 3 >> runs/docker.jsonl
python3 scripts/wan_docker.py down
python3 scripts/bench_table.py --medians runs/docker.jsonl
```

Checked on 2026-11-12 on this Mac: `toy-n2` with the local transport (two repetitions), and `bench-l16-n4` (two
repetitions) and `toy-n4` in four containers with an RTT of 50 ms and 200 Mbit/s. Every run was accepted, and the
measured round trips (51 to 55 ms at the minimum) and throughput (165 to 191 Mbit/s) match the emulation. `gcp.py` was
checked only as a dry run, since nothing may be created without the project and the user's go-ahead.

Also checked on 2026-11-12: `wan_grid.py` over the three settings of PPP26 in four containers with `toy-n4` (two
repetitions each, Eval growing with the RTT, and a rerun that skipped every cell), and PPP26's PCS mode built and run
under Rosetta at n = 4 and 8 with 2^14 coefficients (508 queries, every rank's line parsed, the payload counter
working). The cloud commands of `ppp26.py` were checked as dry runs.
