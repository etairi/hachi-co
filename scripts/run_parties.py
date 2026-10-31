#!/usr/bin/env python3
"""Runs the parties of one Hachi-Co instance as local processes over TCP, then the verifier.

Prints one JSON object: the commitment and evaluation times of the slowest party, the peak memory of
each party, the payload bytes and the proof's size. The seeds are SHA-256 of a tag, so a run is reproduced by its
command line. The peak memory is /usr/bin/time's: the peak footprint on macOS, the peak resident size on Linux. Each
party gets its share of the cores, as RAYON_NUM_THREADS, and the verifier gets all of them.

    cargo build --release -p hachi-co --bin hachi-co-party
    python3 scripts/run_parties.py --preset toy-n3 --n 3 --out /tmp/run
    python3 scripts/run_parties.py --preset bench-l20-n2 --n 2 --out /tmp/run --spill /tmp/spill
"""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

BINARY = Path(__file__).resolve().parent.parent / "target" / "release" / "hachi-co-party"


def seed(tag, *parts):
    return hashlib.sha256("/".join([tag, *map(str, parts)]).encode()).hexdigest()


def time_command():
    return ["/usr/bin/time", "-l"] if platform.system() == "Darwin" else ["/usr/bin/time", "-v"]


def peak_bytes(stderr):
    # On macOS the resident size leaves out compressed and swapped pages, so it needs the footprint.
    darwin = re.search(r"(\d+)\s+peak memory footprint", stderr)
    if darwin:
        return int(darwin.group(1))
    linux = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr)
    return int(linux.group(1)) * 1024 if linux else None


def seconds(text):
    match = re.fullmatch(r"([\d.]+)(ns|µs|ms|s|m)?", text.strip())
    scale = {"ns": 1e-9, "µs": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0}
    return float(match.group(1)) * scale[match.group(2) or "s"]


def report(stdout):
    get = lambda pattern: re.search(pattern, stdout, re.M)
    return {
        "setup_s": seconds(get(r"^setup\s+(\S+)").group(1)),
        "commit_s": seconds(get(r"^commit\s+(\S+)").group(1)),
        "eval_s": seconds(get(r"^eval\s+(\S+)").group(1)),
        "network_s": seconds(get(r"^network\s+(\S+)").group(1)),
        "proof_bytes": int(get(r"^proof\s+(\d+) bytes").group(1)),
        "sent_bytes": int(get(r"^sent\s+(\d+) payload").group(1)),
        "vector": get(r"^vector\s+(\S+)").group(1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True, help="directory of the configurations and outputs")
    parser.add_argument("--spill", type=Path, help="keep the blocks of f' on disk in this directory")
    parser.add_argument("--port", type=int, default=7300)
    parser.add_argument("--tag", default="hachi-co/bench", help="the seeds are SHA-256 of this tag and an index")
    parser.add_argument("--timeout-ms", type=int, default=3_600_000)
    parser.add_argument("--threads", type=int, help="threads of each party, by default its share of the cores")
    args = parser.parse_args()
    threads = args.threads or max(os.cpu_count() // args.n, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.spill:
        args.spill.mkdir(parents=True, exist_ok=True)
    addresses = [f"127.0.0.1:{args.port + p}" for p in range(args.n)]
    started = time.time()
    processes = []
    for p in range(1, args.n + 1):
        config = {
            "preset": args.preset,
            "n": args.n,
            "party": p,
            "addresses": addresses,
            "session": seed(args.tag, "session", args.preset),
            "setup_seed": seed(args.tag, "setup", args.preset),
            "instance_seed": seed(args.tag, "instance", args.preset),
            "party_seed": seed(args.tag, "party", args.preset, p),
            "output": str(args.out / f"party-{p}"),
            "connect_timeout_ms": args.timeout_ms,
            "recv_timeout_ms": args.timeout_ms,
        }
        if args.spill:
            config["spill_dir"] = str(args.spill)
        path = args.out / f"party-{p}.json"
        path.write_text(json.dumps(config, indent=1))
        processes.append(
            subprocess.Popen(
                [*time_command(), str(BINARY), str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "RAYON_NUM_THREADS": str(threads)},
            )
        )
    parties = []
    failed = False
    for p, process in enumerate(processes, 1):
        stdout, stderr = process.communicate()
        (args.out / f"party-{p}.stdout").write_text(stdout)
        (args.out / f"party-{p}.stderr").write_text(stderr)
        if process.returncode != 0:
            failed = True
            sys.stderr.write(f"party {p} failed:\n{stderr[-2000:]}\n")
            continue
        parties.append({**report(stdout), "peak_bytes": peak_bytes(stderr)})
    if failed:
        return 1
    verify = subprocess.run(
        [*time_command(), str(BINARY), "--verify", str(args.out / "party-1")],
        capture_output=True,
        text=True,
    )
    accepted = verify.returncode == 0 and verify.stdout.startswith("accepted")
    match = re.search(r"verified in (\S+)", verify.stdout)
    sizes = subprocess.run(
        [str(BINARY), "--sizes", args.preset, str(args.n)], capture_output=True, text=True, check=True
    )
    predicted, packed = map(int, sizes.stdout.split())
    result = {
        "preset": args.preset,
        "n": args.n,
        "spill": bool(args.spill),
        "tag": args.tag,
        "accepted": accepted,
        "wall_s": round(time.time() - started, 2),
        "setup_s": max(p["setup_s"] for p in parties),
        "commit_s": max(p["commit_s"] for p in parties),
        "eval_s": max(p["eval_s"] for p in parties),
        "verify_s": seconds(match.group(1)) if match else None,
        "verifier_peak_bytes": peak_bytes(verify.stderr),
        "proof_bytes": parties[0]["proof_bytes"],
        "predicted_bytes": predicted,
        "packed_bytes": packed,
        "sent_bytes_per_party": [p["sent_bytes"] for p in parties],
        "peak_bytes_per_party": [p["peak_bytes"] for p in parties],
        "cores": os.cpu_count(),
        "threads_per_party": threads,
        "vector": parties[0]["vector"],
        "platform": platform.platform(),
    }
    print(json.dumps(result))
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
