#!/usr/bin/env python3
"""Prints the JSON lines of scripts/bench_grid.py and scripts/run_parties.py as Markdown tables, one per section.
With --medians, prints the lines of scripts/cluster.py instead: one row per preset, number of parties, setting and
threads, with the median of the repetitions (and the range of Eval).

    python3 scripts/bench_table.py benchmarks/local-2026-10-31.jsonl
    python3 scripts/bench_table.py --medians results.jsonl
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

GIB = 2 ** 30
HEADER = (
    "| preset | parties | on disk | Commit (s) | Eval (s) | Verify (s) | peak per party (GiB) | sent per party (MiB) "
    "| proof (KiB) | tool's size (KiB) | verifier's peak (GiB) |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|"
)


def row(cell):
    name = cell["preset"]
    if cell.get("skipped"):
        return f"| {name} | {cell['n']} | not run: about {cell['estimated_gib']} GiB in all | | | | | | | | |"
    return (
        f"| {name} | {cell['n']} | {'yes' if cell['spill'] else 'no'} | {cell['commit_s']:.1f} | {cell['eval_s']:.1f} "
        f"| {cell['verify_s']:.2f} | {max(cell['peak_bytes_per_party']) / GIB:.2f} "
        f"| {max(cell['sent_bytes_per_party']) / 2 ** 20:.1f} | {cell['proof_bytes'] / 1024:.1f} "
        f"| {cell['predicted_bytes'] / 1024:.1f} | {cell['verifier_peak_bytes'] / GIB:.2f} |"
    )


MEDIAN_HEADER = (
    "| preset | parties | setting | threads per party | runs | Commit (s) | Eval (s) | Eval range (s) | Verify (s) "
    "| rounds per party | peak per party (GiB) | sent per party (MiB) | proof (KiB) |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def medians(cells):
    """One row per (preset, n, setting, threads): the medians over the repetitions."""
    groups = {}
    for c in cells:
        if not c["accepted"]:
            raise SystemExit(f"{c['preset']} (rep {c.get('rep')}) was not accepted")
        threads = c.get("threads_per_party") or []
        setting = c.get("setting", {}).get("name") or c.get("setting", {}).get("transport", "local")
        key = (c["preset"], c["n"], setting, max(threads) if threads else None)
        groups.setdefault(key, []).append(c)
    print(MEDIAN_HEADER)
    for (preset, n, setting, threads), runs in groups.items():
        med = lambda f: statistics.median(f(r) for r in runs)
        evals = [r["eval_s"] for r in runs]
        print(
            f"| {preset} | {n} | {setting} | {threads} | {len(runs)} | {med(lambda r: r['commit_s']):.2f} "
            f"| {med(lambda r: r['eval_s']):.2f} | {min(evals):.2f} to {max(evals):.2f} "
            f"| {med(lambda r: r['verify_s']):.2f} | {med(lambda r: max(r['rounds_per_party'])):.0f} "
            f"| {med(lambda r: max(r['peak_bytes_per_party'])) / GIB:.2f} "
            f"| {med(lambda r: max(r['sent_bytes_per_party'])) / 2 ** 20:.1f} | {med(lambda r: r['proof_bytes']) / 1024:.1f} |"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--medians", action="store_true", help="the lines of scripts/cluster.py, with medians")
    ap.add_argument("file", type=Path)
    args = ap.parse_args()
    cells = [json.loads(line) for line in args.file.read_text().splitlines() if line.strip()]
    if args.medians:
        medians(cells)
        return
    for cell in cells:
        if not cell.get("skipped") and not cell["accepted"]:
            raise SystemExit(f"{cell['preset']} was not accepted")
    sections = [
        ("Grid", lambda c: c["preset"].startswith("bench") and c["n"] >= 2 and "residues" not in c["preset"]
         and c.get("spill", True)),
        ("One party", lambda c: c["preset"].startswith("bench") and c["n"] == 1),
        ("The two pads", lambda c: c["preset"] in ("bench-l20-n2", "bench-l20-n2-residues")),
        ("In memory against on disk", lambda c: c["preset"] == "bench-l16-n2"),
        ("Secure presets", lambda c: c["preset"].startswith("secure")),
    ]
    for title, keep in sections:
        print(f"\n## {title}\n\n{HEADER}")
        for cell in filter(keep, cells):
            print(row(cell))


if __name__ == "__main__":
    main()
