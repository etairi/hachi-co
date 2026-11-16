#!/usr/bin/env python3
"""A benchmark grid over network settings, presets and repetitions: for each setting it switches the emulated network
of the hosts, then runs scripts/cluster.py for each preset (and scripts/ppp26.py for each PPP26 size, on the cloud
instances), appending one JSON line per repetition to --out. A rerun skips the repetitions already in --out, so an
interrupted grid resumes where it stopped.

    python3 scripts/wan_docker.py up --n 4 --cpus 2 --hosts runs/docker/hosts.json
    python3 scripts/wan_grid.py --hosts runs/docker/hosts.json --switch docker --settings lan wan50 wan100 \\
        --presets bench-l16-n4 toy-n4 --reps 3 --out runs/docker-grid.jsonl
    python3 scripts/wan_grid.py --hosts runs/gcp/hosts.json --switch gcp --gcp-config runs/gcp.json \\
        --settings lan wan50 wan100 --presets secure100-l16-n4-a8 secure100-l20-n4-a8 \\
        --ppp26-log-ms 16 20 --threads 1 --reps 3 --out runs/gcp-grid.jsonl

A preset name ending in -nonhiding (for example secure100-l16-n4-a8-nonhiding next to secure100-l16-n4-a8) runs Hachi,
the base PCS, without its hiding choices on the same instance, to measure what hiding costs.

The named settings are those of PPP26 (ePrint 2026/729, Section 7): `lan` (RTT 2 ms, 10 Gbit/s), `wan50` (RTT 50 ms,
200 Mbit/s) and `wan100` (RTT 100 ms, 200 Mbit/s). Others are `name:rtt_ms:rate_mbit` (`asis` with --switch none, for
instances in different regions). With --switch gcp the netem commands run on the instances (scripts/gcp.py netem).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMED = {"lan": (2.0, 10000.0), "wan50": (50.0, 200.0), "wan100": (100.0, 200.0), "asis": (None, None)}


def parse_setting(text):
    if text in NAMED:
        return text, *NAMED[text]
    name, rtt, rate = text.split(":")
    return name, float(rtt), float(rate) if rate else None


def switch(args, name, rtt, rate):
    """Sets the emulated network of every host to (rtt, rate), or leaves it as it is."""
    if args.switch == "none" or rtt is None:
        return
    if args.switch == "docker":
        cmd = [sys.executable, str(HERE / "wan_docker.py"), "netem", "--rtt-ms", str(rtt), "--hosts", str(args.hosts)]
    else:
        cmd = [sys.executable, str(HERE / "gcp.py"), "--config", str(args.gcp_config), "netem", "--rtt-ms", str(rtt),
               "--execute"]
    if rate:
        cmd += ["--rate-mbit", str(rate)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"switching to {name} failed:\n{proc.stderr[-2000:]}")


def done(out, **key):
    """How many accepted repetitions of `key` --out already has."""
    if not out.exists():
        return 0
    count = 0
    for text in out.read_text().splitlines():
        line = json.loads(text)
        if line.get("accepted", True) and all(line.get(k) == v for k, v in key.items() if k != "setting_name") \
                and line.get("setting", {}).get("grid_setting") == key["setting_name"]:
            count += 1
    return count


def run(cmd, out):
    """Runs one cell, appends its lines to `out` and returns whether it succeeded."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    with out.open("a") as fh:
        fh.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(f"failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-2000:]}\n")
    elif proc.stderr.strip():
        sys.stderr.write(proc.stderr.strip().splitlines()[-1] + "\n")
    return proc.returncode == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts", type=Path, required=True, help="the hosts file of scripts/cluster.py")
    ap.add_argument("--switch", choices=["docker", "gcp", "none"], required=True,
                    help="how to set the emulated network of the hosts")
    ap.add_argument("--gcp-config", type=Path, help="the configuration of scripts/gcp.py (with --switch gcp)")
    ap.add_argument("--settings", nargs="+", default=["asis"])
    ap.add_argument("--presets", nargs="*", default=[], help="presets of Hachi-Co, in this order")
    ap.add_argument("--ppp26-log-ms", type=int, nargs="*", default=[],
                    help="log2 of the polynomial sizes for PPP26's PCS (cloud instances only)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, help="threads of each party (1 for the comparison with PPP26)")
    ap.add_argument("--spill", action="store_true")
    ap.add_argument("--bandwidth", action="store_true", help="measure TCP throughput in each setting")
    ap.add_argument("--out", type=Path, required=True, help="the JSON lines, appended")
    ap.add_argument("--logs", type=Path, help="the logs of cluster.py (default: next to --out)")
    args = ap.parse_args()
    if args.switch == "gcp" and not args.gcp_config:
        raise SystemExit("--switch gcp needs --gcp-config")
    if args.ppp26_log_ms and args.switch == "docker":
        raise SystemExit("PPP26 runs on the cloud instances only (scripts/ppp26.py docker-run tests it locally)")
    n_hosts = len(json.loads(args.hosts.read_text())["parties"])
    logs = args.logs or args.out.with_suffix(".logs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    for text in args.settings:
        name, rtt, rate = parse_setting(text)
        switch(args, name, rtt, rate)
        setting = json.dumps({"grid_setting": name, "grid_rtt_ms": rtt, "grid_rate_mbit": rate})
        sys.stderr.write(f"== {name}: RTT {rtt} ms, {rate} Mbit/s\n")
        for preset in args.presets:
            have = done(args.out, preset=preset, n=n_hosts, setting_name=name)
            if have >= args.reps:
                sys.stderr.write(f"{preset}: {have} repetitions already in {args.out}\n")
                continue
            cmd = [sys.executable, str(HERE / "cluster.py"), "--hosts", str(args.hosts), "--preset", preset,
                   "--out", str(logs / name / preset), "--reps", str(args.reps - have), "--first-rep", str(have),
                   "--setting-json", setting]
            if args.threads:
                cmd += ["--threads", str(args.threads)]
            if args.spill:
                cmd.append("--spill")
            if args.bandwidth:
                cmd.append("--bandwidth")
            failures += not run(cmd, args.out)
        for log_m in args.ppp26_log_ms:
            have = done(args.out, scheme="ppp26", log_m=log_m, n=n_hosts, setting_name=name)
            if have >= args.reps:
                continue
            cmd = [sys.executable, str(HERE / "ppp26.py"), "gcp-run", "--hosts", str(args.hosts), "--n", str(n_hosts),
                   "--log-m", str(log_m), "--reps", str(args.reps - have), "--first-rep", str(have),
                   "--setting-json", setting, "--execute"]
            failures += not run(cmd, args.out)
    sys.stderr.write(f"grid finished with {failures} failed cells\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
