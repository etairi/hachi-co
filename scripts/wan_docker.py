#!/usr/bin/env python3
"""Emulates a wide-area network on this machine: one Docker container per party on a private bridge network, where
`tc netem` delays each container's outgoing traffic by half the round-trip time and limits its rate, so that a round
trip between two parties takes --rtt-ms and each party sends at most --rate-mbit. The containers run the image of
docker/Dockerfile, and the hosts file written by `up` is the input of scripts/cluster.py (transport "docker").

    python3 scripts/wan_docker.py build
    python3 scripts/wan_docker.py up --n 4 --rtt-ms 50 --rate-mbit 200 --cpus 2 --hosts runs/docker-hosts.json
    python3 scripts/cluster.py --hosts runs/docker-hosts.json --preset bench-l16-n4 --out runs/docker --reps 3
    python3 scripts/wan_docker.py down

The settings of PPP26 (ePrint 2026/729, Section 7) are a LAN of 10 Gbps with an RTT of 2 ms, and a WAN of 200 Mbps with
an RTT of 50 ms or 100 ms, emulated with tc. Docker Desktop runs every container in one Linux VM, so the parties share
its CPUs and memory: --cpus pins each container to cores of its own, and --memory caps it. The numbers measure the
network's effect, not the speed of a cloud instance.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGE = "hachi-co-party:local"
NETWORK = "hachi-co-wan"
SUBNET = "172.31.250.0/24"
PREFIX = "hco-p"
PORT = 7300


def docker(*args, check=True, capture=True):
    proc = subprocess.run(["docker", *args], capture_output=capture, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"docker {' '.join(args)}: {proc.stderr.strip() if capture else proc.returncode}")
    return proc.stdout.strip() if capture else ""


def ip(p):
    return f"172.31.250.{10 + p}"


def build(args):
    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    commit = git("rev-parse", "HEAD")
    dirty = "true" if git("status", "--porcelain", "--untracked-files=no") else "false"
    rustc = subprocess.run(["rustc", "-V"], capture_output=True, text=True).stdout
    version = args.rust or re.search(r"rustc (\d+\.\d+\.\d+)", rustc).group(1)
    docker("build", "-f", str(ROOT / "docker" / "Dockerfile"), "-t", IMAGE, "--build-arg", f"RUST_VERSION={version}",
           "--build-arg", f"COMMIT={commit}", "--build-arg", f"DIRTY={dirty}", str(ROOT), capture=False)
    print(f"built {IMAGE} from {commit}{' with local changes' if dirty == 'true' else ''}, Rust {version}")


def existing():
    out = docker("ps", "-a", "--filter", f"name=^{PREFIX}", "--format", "{{.Names}}", check=False)
    return [name for name in out.splitlines() if re.fullmatch(rf"{PREFIX}\d+", name)]


def up(args):
    if args.n < 1:
        raise SystemExit("--n must be at least 1")
    vm_cpus = int(docker("info", "--format", "{{.NCPU}}"))
    if args.cpus and args.cpus * args.n > vm_cpus:
        raise SystemExit(f"--cpus {args.cpus} for {args.n} parties needs {args.cpus * args.n} CPUs, "
                         f"the Docker VM has {vm_cpus}")
    for name in existing():
        docker("rm", "-f", name)
    if docker("network", "ls", "--filter", f"name=^{NETWORK}$", "--format", "{{.Name}}") != NETWORK:
        docker("network", "create", "--subnet", SUBNET, NETWORK)
    netem = netem_spec(args)
    parties = []
    for p in range(1, args.n + 1):
        name = f"{PREFIX}{p}"
        run = ["run", "-d", "--name", name, "--hostname", name, "--network", NETWORK, "--ip", ip(p),
               "--cap-add", "NET_ADMIN"]
        if args.cpus:
            run += ["--cpuset-cpus", f"{(p - 1) * args.cpus}-{p * args.cpus - 1}"]
        if args.memory:
            run += ["--memory", args.memory]
        docker(*run, IMAGE)
        apply_netem(name, netem)
        parties.append({"host": name, "address": f"{ip(p)}:{PORT}", "workdir": "/run/hco",
                        "binary": "/usr/local/bin/hachi-co-party"})
        print(f"{name}: {ip(p)}, {docker('exec', name, 'tc', 'qdisc', 'show', 'dev', 'eth0')}")
    setting = {
        **emulated(args),
        "cpus_per_party": args.cpus, "memory_per_party": args.memory, "docker_vm_cpus": vm_cpus,
        "image": docker("image", "inspect", IMAGE, "--format", "{{.Id}}"),
    }
    args.hosts.parent.mkdir(parents=True, exist_ok=True)
    args.hosts.write_text(json.dumps({"transport": "docker", "setting": setting, "parties": parties}, indent=1) + "\n")
    print(f"wrote {args.hosts}")


def netem_spec(args):
    """The netem options of --rtt-ms, --jitter-ms, --rate-mbit and --loss-pct (empty: no emulation)."""
    spec = []
    if args.rtt_ms:
        spec += ["delay", f"{args.rtt_ms / 2:g}ms"]
        if args.jitter_ms:
            spec += [f"{args.jitter_ms / 2:g}ms"]
    if args.rate_mbit:
        spec += ["rate", f"{args.rate_mbit:g}mbit"]
    if args.loss_pct:
        spec += ["loss", f"{args.loss_pct:g}%"]
    return spec


def apply_netem(name, spec):
    """Sets the emulation of a container's eth0, or removes it for an empty spec."""
    if spec:
        # A queue long enough for the bandwidth-delay product of the slowest setting.
        docker("exec", name, "tc", "qdisc", "replace", "dev", "eth0", "root", "netem", *spec, "limit", "100000")
    else:
        docker("exec", name, "tc", "qdisc", "del", "dev", "eth0", "root", check=False)


def emulated(args):
    return {"name": f"docker netem, RTT {args.rtt_ms:g} ms" + (f", {args.rate_mbit:g} Mbit/s" if args.rate_mbit else ""),
            "emulated": {"rtt_ms": args.rtt_ms, "jitter_ms": args.jitter_ms, "rate_mbit": args.rate_mbit,
                         "loss_pct": args.loss_pct}}


def netem(args):
    """Changes the emulation of the running containers, and the setting in the hosts file if one is given."""
    names = existing()
    if not names:
        raise SystemExit("no containers: run `up` first")
    spec = netem_spec(args)
    for name in names:
        apply_netem(name, spec)
        print(f"{name}: {docker('exec', name, 'tc', 'qdisc', 'show', 'dev', 'eth0')}")
    if args.hosts:
        doc = json.loads(args.hosts.read_text())
        doc["setting"] = {**doc.get("setting", {}), **emulated(args)}
        args.hosts.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"updated {args.hosts}")


def down(_args):
    for name in existing():
        docker("rm", "-f", name)
        print(f"removed {name}")
    if docker("network", "ls", "--filter", f"name=^{NETWORK}$", "--format", "{{.Name}}") == NETWORK:
        docker("network", "rm", NETWORK)
        print(f"removed the network {NETWORK}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build the image from the working tree")
    b.add_argument("--rust", help="the Rust release of the image (default: the local rustc's)")
    u = sub.add_parser("up", help="start the containers and the emulated network, and write the hosts file")
    u.add_argument("--n", type=int, required=True)
    u.add_argument("--rtt-ms", type=float, default=0.0, help="round-trip time between two parties")
    u.add_argument("--jitter-ms", type=float, default=0.0, help="jitter of the round-trip time")
    u.add_argument("--rate-mbit", type=float, help="outgoing rate of each party")
    u.add_argument("--loss-pct", type=float, help="packet loss of each direction")
    u.add_argument("--cpus", type=int, help="cores of each party (pinned, distinct)")
    u.add_argument("--memory", help="memory limit of each party, for example 6g")
    u.add_argument("--hosts", type=Path, required=True, help="the hosts file to write")
    e = sub.add_parser("netem", help="change the emulation of the running containers")
    e.add_argument("--rtt-ms", type=float, default=0.0)
    e.add_argument("--jitter-ms", type=float, default=0.0)
    e.add_argument("--rate-mbit", type=float)
    e.add_argument("--loss-pct", type=float)
    e.add_argument("--hosts", type=Path, help="a hosts file of `up` whose setting to update")
    sub.add_parser("down", help="remove the containers and the network")
    args = ap.parse_args()
    {"build": build, "up": up, "netem": netem, "down": down}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
