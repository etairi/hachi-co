#!/usr/bin/env python3
"""Runs the parties of one Hachi-Co instance as local processes over TCP, then the verifier.

Prints one JSON object: the commitment and evaluation times of the slowest party, the peak memory of
each party, the payload bytes and the proof's size. The seeds are SHA-256 of a tag, so a run is reproduced by its
command line. The peak memory is /usr/bin/time's: the peak footprint on macOS, the peak resident size on Linux. Each
party gets its share of the cores, as RAYON_NUM_THREADS, and the verifier gets all of them.

A preset name ending in -nonhiding runs Hachi, the base PCS, without its hiding choices (the published scheme), to
measure what hiding costs. It takes the seeds of the name without the suffix, so both runs prove the same polynomial
at the same point with the same public matrices. With --single the script runs the single-prover baseline instead
(hachi-co-party --single): one prover commits to the whole polynomial with Hachi and opens it, with the threads of one
party.

    cargo build --release -p hachi-co --bin hachi-co-party
    python3 scripts/run_parties.py --preset toy-n3 --n 3 --out /tmp/run
    python3 scripts/run_parties.py --preset bench-l20-n2 --n 2 --out /tmp/run --spill /tmp/spill
    python3 scripts/run_parties.py --preset secure-l16-n2-a8-nonhiding --n 2 --out /tmp/run
    python3 scripts/run_parties.py --preset secure-l16-n2-a8 --n 2 --out /tmp/run --single
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


def instance_name(preset):
    """The name the seeds are drawn from: the preset without the suffixes -nonhiding and -greyhound (in either order),
    which change only the base PCS, so that a variant proves the same instance as its preset."""
    name = preset
    while name.endswith(("-nonhiding", "-greyhound")):
        name = name.removesuffix("-nonhiding").removesuffix("-greyhound")
    return name


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


def hiding(stdout):
    match = re.search(r"^hiding\s+(yes|no)", stdout, re.M)
    return match.group(1) == "yes" if match else None


def report(stdout):
    get = lambda pattern: re.search(pattern, stdout, re.M)
    return {
        "hiding": hiding(stdout),
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
    parser.add_argument("--single", action="store_true", help="run the single-prover baseline instead of the parties")
    args = parser.parse_args()
    threads = args.threads or max(os.cpu_count() // args.n, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    name = instance_name(args.preset)
    if args.single:
        return single(args, name, threads)
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
            "session": seed(args.tag, "session", name),
            "setup_seed": seed(args.tag, "setup", name),
            "instance_seed": seed(args.tag, "instance", name),
            "party_seed": seed(args.tag, "party", name, p),
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
        "hiding": parties[0]["hiding"],
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


def single(args, name, threads):
    """The single-prover baseline on the instance of the run: party 1's configuration, one process."""
    config = {
        "preset": args.preset,
        "n": args.n,
        "party": 1,
        "addresses": [f"127.0.0.1:{args.port + p}" for p in range(args.n)],
        "session": seed(args.tag, "session", name),
        "setup_seed": seed(args.tag, "setup", name),
        "instance_seed": seed(args.tag, "instance", name),
        "party_seed": seed(args.tag, "party", name, 1),
        "output": str(args.out / "single"),
    }
    path = args.out / "single.json"
    path.write_text(json.dumps(config, indent=1))
    started = time.time()
    run = subprocess.run(
        [*time_command(), str(BINARY), "--single", str(path)],
        capture_output=True,
        text=True,
        env={**os.environ, "RAYON_NUM_THREADS": str(threads)},
    )
    (args.out / "single.stdout").write_text(run.stdout)
    (args.out / "single.stderr").write_text(run.stderr)
    if run.returncode != 0:
        sys.stderr.write(f"the single prover failed:\n{run.stderr[-2000:]}\n")
        return 1
    get = lambda pattern: re.search(pattern, run.stdout, re.M).group(1)
    result = {
        "preset": args.preset,
        "n": args.n,
        "single": True,
        "tag": args.tag,
        "accepted": True,
        "hiding": hiding(run.stdout),
        "wall_s": round(time.time() - started, 2),
        "setup_s": seconds(get(r"^setup\s+(\S+)")),
        "commit_s": seconds(get(r"^commit\s+(\S+)")),
        "eval_s": seconds(get(r"^eval\s+(\S+)")),
        "verify_s": seconds(get(r"^verify\s+(\S+)")),
        "iterations": int(get(r"^iterations\s+(\d+)")),
        "commitment_bytes": int(get(r"^commitment\s+(\d+) bytes")),
        "proof_bytes": int(get(r"^proof\s+(\d+) bytes")),
        "peak_bytes": peak_bytes(run.stderr),
        "cores": os.cpu_count(),
        "threads": threads,
        "vector": get(r"^vector\s+(\S+)"),
        "platform": platform.platform(),
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
