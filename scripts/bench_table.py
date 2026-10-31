#!/usr/bin/env python3
"""Prints the JSON lines of scripts/bench_grid.py and scripts/run_parties.py as Markdown tables, one per section."""
import json
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


def main():
    cells = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
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
