#!/usr/bin/env python3
"""The local runs of the benchmarks: every party on this machine, over the loopback. Each cell of a plan runs
scripts/cluster.py with local parties (or scripts/run_parties.py --single for the single-prover baseline) and appends
one JSON line per repetition to --out, with the cell's "plan" and "cell" names added. Repetitions already in --out are
skipped, so an interrupted plan resumes where it stopped. The parties keep the blocks of f' on disk (--spill), so that
eight parties fit in the memory of one machine.

    cargo build --release --locked -p hachi-co --bin hachi-co-party
    python3 scripts/local_grid.py --plan calibration --out runs/local/calibration.jsonl --work runs/local/work
    python3 scripts/local_grid.py --plan phase1 --out runs/local/phase1.jsonl --work runs/local/work
    python3 scripts/local_grid.py --plan phase1 --list      # the cells and their repetitions, nothing is run

Plans (presets secure-l{l}-n{n}-a8, 115 bits; the 100-bit comparison with PPP26 runs across machines instead):
- smoke: the toy presets at n = 2 and 4 and the single prover, one repetition each, to check the machine.
- calibration: l = 20 and n = 2 at 1 and 4 threads per party, and l = 24 and n = 8 at 1 thread (the memory of eight
  parties), one repetition each.
- phase1: l = 20, 22, 24 and n = 2, 4, 8 at 1 thread per party; n = 2 at 4 threads per party; the same instance
  without hiding in the base PCS (-nonhiding) at n = 2, 4 and 8 and 1 thread; the single prover at 1 and 8
  threads. One repetition each, in increasing size within each part.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ELLS = (20, 22, 24)
REPS = 1


def preset(ell, n, suffix=""):
    return f"secure-l{ell}-n{n}-a8{suffix}"


def plan_cells(name):
    """The cells of a plan: (cell name, kind, preset, parties, threads, repetitions)."""
    if name == "smoke":
        return [("smoke-toy-n2", "parties", "toy-n2", 2, 1, 1), ("smoke-toy-n4", "parties", "toy-n4", 4, 1, 1),
                ("smoke-single", "single", "toy-n2", 1, 1, 1)]
    if name == "calibration":
        return [("cal-l20-n2-t1", "parties", preset(20, 2), 2, 1, 1),
                ("cal-l20-n2-t4", "parties", preset(20, 2), 2, 4, 1),
                ("cal-l24-n8-t1", "parties", preset(24, 8), 8, 1, 1)]
    if name == "phase1":
        cells = [(f"l{ell}-n{n}-t1", "parties", preset(ell, n), n, 1, REPS) for ell in ELLS for n in (2, 4, 8)]
        cells += [(f"l{ell}-n2-t4", "parties", preset(ell, 2), 2, 4, REPS) for ell in ELLS]
        cells += [(f"l{ell}-n2-t1-nonhiding", "parties", preset(ell, 2, "-nonhiding"), 2, 1, REPS) for ell in ELLS]
        cells += [(f"l{ell}-n{n}-t1-nonhiding", "parties", preset(ell, n, "-nonhiding"), n, 1, REPS)
                  for ell in ELLS for n in (4, 8)]
        cells += [(f"l{ell}-single-t{t}", "single", preset(ell, 2), 1, t, REPS) for ell in ELLS for t in (1, 8)]
        return cells
    raise SystemExit(f"unknown plan {name}")


def recorded(out, plan, cell):
    """The repetitions of a cell already in --out."""
    if not out.exists():
        return 0
    count = 0
    for line in out.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        count += d.get("plan") == plan and d.get("cell") == cell
    return count


def hosts_file(work, n, port):
    """A hosts file of scripts/cluster.py with n local parties, each with its own working directory."""
    binary = str(ROOT / "target" / "release" / "hachi-co-party")
    parties = []
    for p in range(1, n + 1):
        workdir = work / f"party-{p}"
        workdir.mkdir(parents=True, exist_ok=True)
        parties.append({"host": "local", "address": f"127.0.0.1:{port + p - 1}", "workdir": str(workdir),
                        "binary": binary})
    path = work / f"hosts-n{n}.json"
    path.write_text(json.dumps({"transport": "local", "setting": {"name": "local, loopback"}, "parties": parties},
                               indent=1) + "\n")
    return path


def run_cell(args, cell, kind, name, n, threads, first, reps):
    """Runs the repetitions first .. reps - 1 of a cell and returns their JSON lines."""
    lines = []
    if kind == "parties":
        command = [sys.executable, str(HERE / "cluster.py"), "--hosts", str(hosts_file(args.work, n, args.port)),
                   "--preset", name, "--out", str(args.work / cell), "--reps", str(reps - first), "--first-rep",
                   str(first), "--threads", str(threads), "--spill"]
        proc = subprocess.run(command, capture_output=True, text=True)
        sys.stderr.write(proc.stderr[-2000:])
        if proc.returncode != 0:
            raise SystemExit(f"{cell}: cluster.py failed ({proc.returncode})")
        lines = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    else:
        for rep in range(first, reps):
            out = args.work / cell / f"rep-{rep}"
            command = [sys.executable, str(HERE / "run_parties.py"), "--preset", name, "--n", "2", "--out", str(out),
                       "--single", "--threads", str(threads), "--port", str(args.port)]
            proc = subprocess.run(command, capture_output=True, text=True)
            sys.stderr.write(proc.stderr[-2000:])
            if proc.returncode != 0:
                raise SystemExit(f"{cell}: run_parties.py failed ({proc.returncode})")
            got = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
            lines += [{**d, "rep": rep, "threads": threads} for d in got]
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True, choices=["smoke", "calibration", "phase1"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--work", type=Path, help="working directories of the parties and the outputs of each cell")
    ap.add_argument("--port", type=int, default=7400)
    ap.add_argument("--list", action="store_true", help="print the cells and their repetitions, and run nothing")
    args = ap.parse_args()
    cells = plan_cells(args.plan)
    if args.list:
        for cell, kind, name, n, threads, reps in cells:
            done = recorded(args.out, args.plan, cell) if args.out else 0
            print(f"{cell:24} {kind:8} {name:32} n={n} threads={threads} reps={reps} recorded={done}")
        return 0
    if not args.out or not args.work:
        raise SystemExit("give --out and --work")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.work.mkdir(parents=True, exist_ok=True)
    for cell, kind, name, n, threads, reps in cells:
        first = recorded(args.out, args.plan, cell)
        if first >= reps:
            continue
        print(f"{cell}: repetitions {first} to {reps - 1}", file=sys.stderr, flush=True)
        for line in run_cell(args, cell, kind, name, n, threads, first, reps):
            with args.out.open("a") as fh:
                fh.write(json.dumps({**line, "plan": args.plan, "cell": cell}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
