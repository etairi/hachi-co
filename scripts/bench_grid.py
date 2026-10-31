#!/usr/bin/env python3
"""The benchmark grid: scripts/run_parties.py for each cell, one JSON line per cell, in increasing size.

A cell whose estimated memory exceeds --budget-gib is recorded as skipped with its estimate, not run. The estimate is
the fit noted at PEAK_A_GIB, for all the parties of the cell.

    cargo build --release -p hachi-co --bin hachi-co-party
    python3 scripts/bench_grid.py --out benchmarks/local.jsonl --scratch /tmp/grid --budget-gib 12
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Peak footprint of one party at l'' variables of the sumcheck's table: a + b 2^l'', fitted to bench-l16-n2 (0.55 GiB
# at l'' = 21) and bench-l20-n2 (1.3 GiB at l'' = 23).
PEAK_A_GIB, PEAK_B_GIB_AT_21 = 0.3, 0.25
# l'' of the bench presets, from hachi_co_params::derive.
ELL_DPRIME = {
    (16, 1): 21, (16, 2): 21, (16, 4): 22, (16, 8): 22,
    (20, 1): 23, (20, 2): 23, (20, 4): 24, (20, 8): 25,
    (24, 1): 25, (24, 2): 26, (24, 4): 27, (24, 8): 27,
    (26, 1): 26, (26, 2): 27, (26, 4): 28, (26, 8): 28,
}


def estimate(ell, n):
    table = ELL_DPRIME[(ell, n)]
    return (PEAK_A_GIB + PEAK_B_GIB_AT_21 * 2 ** (table - 21)) * n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--budget-gib", type=float, default=12.0)
    parser.add_argument("--ells", type=int, nargs="+", default=[16, 20, 24])
    parser.add_argument("--ns", type=int, nargs="+", default=[2, 4, 8])
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cells = sorted(((ell, n) for ell in args.ells for n in args.ns), key=lambda c: (ELL_DPRIME[c], c[1]))
    for ell, n in cells:
        preset = f"bench-l{ell}-n{n}"
        need = estimate(ell, n)
        if need > args.budget_gib:
            line = json.dumps({"preset": preset, "n": n, "skipped": True, "estimated_gib": round(need, 1)})
        else:
            run = subprocess.run(
                [
                    sys.executable,
                    str(HERE / "run_parties.py"),
                    "--preset", preset,
                    "--n", str(n),
                    "--out", str(args.scratch / preset),
                    "--spill", str(args.scratch / "spill"),
                ],
                capture_output=True,
                text=True,
            )
            line = run.stdout.strip().splitlines()[-1] if run.returncode == 0 else json.dumps(
                {"preset": preset, "n": n, "failed": True, "stderr": run.stderr[-500:]}
            )
        print(line, flush=True)
        with args.out.open("a") as out:
            out.write(line + "\n")


if __name__ == "__main__":
    main()
