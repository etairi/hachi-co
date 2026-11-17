#!/usr/bin/env python3
"""Runs the parties of one Hachi-Co instance on several hosts, then the verifier, and prints one JSON line per
repetition: the slowest party's times, every party's peak memory, bytes and rounds per label, the proof's size, and the
setting of the run (the hosts and their CPUs, the measured round-trip times and bandwidth, the emulated network, the
build). It generalizes scripts/run_parties.py, which keeps every party on one machine.

The hosts file (JSON) says how to reach each party and where it listens:

    {"transport": "ssh",                        # or "docker" (scripts/wan_docker.py writes such a file) or "local"
     "ssh_config": "ssh_config",                # ssh only, optional, relative to the hosts file (scripts/gcp.py)
     "setting": {"name": "gcp us-central1", ...},  # optional, copied into every line
     "parties": [
       {"host": "hco-p1", "address": "10.128.0.2:7300", "workdir": "/home/hco/run",
        "binary": "/home/hco/hachi-co/target/release/hachi-co-party"},
       ...]}

"host" is an ssh destination, a container's name, or ignored for "local". Party p listens on its address and every
party connects to the others' addresses, so each address must be reachable from every host (internal VPC addresses on
Google Cloud). The instance seeds are SHA-256 of --tag and the preset, the same in every repetition, and the session id
is fresh in each, so the repetitions measure the same instance. A preset name ending in -nonhiding runs Hachi, the base
PCS, without its hiding choices, and takes the seeds of the name without the suffix: the same instance as the preset.

    python3 scripts/cluster.py --hosts runs/hosts.json --preset secure-l16-n4-a8 --out runs/x --reps 3 >> results.jsonl
    python3 scripts/bench_table.py --medians results.jsonl
"""
import argparse
import hashlib
import io
import json
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_BINARY = ROOT / "target" / "release" / "hachi-co-party"


def seed(tag, *parts):
    return hashlib.sha256("/".join([tag, *map(str, parts)]).encode()).hexdigest()


def instance_name(preset):
    """The name the seeds are drawn from: the preset without the suffixes -nonhiding and -greyhound (in either order),
    which change only the base PCS, so that a variant proves the same instance as its preset."""
    name = preset
    while name.endswith(("-nonhiding", "-greyhound")):
        name = name.removesuffix("-nonhiding").removesuffix("-greyhound")
    return name


# ------------------------------------------------------------------------------------------- hosts and transports

class Host:
    """One party's machine: runs shell commands there and moves files in and out."""

    def __init__(self, transport, spec, ssh_config=None):
        self.transport = transport
        self.spec = spec
        self.name = spec.get("host", "localhost")
        self.address = spec["address"]
        self.ip = self.address.rsplit(":", 1)[0].strip("[]")
        self.workdir = spec.get("workdir", "/tmp/hachi-co-run")
        self.binary = spec.get("binary", str(LOCAL_BINARY))
        self.ssh_config = ssh_config
        self.env = {}

    def argv(self, command):
        """The local argv that runs the shell command `command` on the host."""
        if self.transport == "local":
            return ["sh", "-c", command]
        if self.transport == "ssh":
            opts = ["-o", "BatchMode=yes", "-o", "ServerAliveInterval=30"]
            if self.ssh_config:
                opts += ["-F", str(self.ssh_config)]
            return ["ssh", *opts, self.name, command]
        if self.transport == "docker":
            return ["docker", "exec", "-i", self.name, "sh", "-c", command]
        raise SystemExit(f"unknown transport {self.transport}")

    def run(self, command, stdin=None, check=True, timeout=None):
        proc = subprocess.run(self.argv(command), input=stdin, capture_output=True, timeout=timeout)
        if check and proc.returncode != 0:
            raise SystemExit(f"{self.name}: `{command}` failed ({proc.returncode}): "
                             f"{proc.stderr.decode(errors='replace')[-2000:]}")
        return proc

    def text(self, command, check=False, timeout=60):
        proc = self.run(command, check=check, timeout=timeout)
        return proc.stdout.decode(errors="replace").strip() if proc.returncode == 0 else ""

    def put_text(self, path, text):
        q = shlex.quote(path)
        self.run(f"mkdir -p $(dirname {q}) && cat > {q}", stdin=text.encode())

    def get_dir(self, remote_dir, local_dir):
        """Copies the directory `remote_dir` of the host into `local_dir` (through tar, for every transport)."""
        parent, name = os.path.split(remote_dir.rstrip("/"))
        proc = self.run(f"tar -C {shlex.quote(parent)} -cf - {shlex.quote(name)}")
        local_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
            try:
                tar.extractall(local_dir, filter="data")
            except TypeError:  # Python before 3.11.4 has no extraction filters
                tar.extractall(local_dir)
        return local_dir / name


def load_hosts(path):
    spec = json.loads(Path(path).read_text())
    transport = spec.get("transport", "local")
    ssh_config = spec.get("ssh_config")
    if ssh_config:
        ssh_config = (Path(path).parent / ssh_config).resolve()
    hosts = [Host(transport, p, ssh_config) for p in spec["parties"]]
    return spec, hosts


# ------------------------------------------------------------------------------------------- recording the setting

PROBE = r"""
echo "kernel=$(uname -srm)"
if [ "$(uname -s)" = Darwin ]; then
  echo "cores=$(sysctl -n hw.ncpu)"; echo "cpu=$(sysctl -n machdep.cpu.brand_string)"
  echo "memory_bytes=$(sysctl -n hw.memsize)"
else
  cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')
  [ -z "$cpu" ] && cpu=$(lscpu 2>/dev/null | grep -m1 'Model name' | cut -d: -f2- | sed 's/^ *//')
  echo "cores=$(nproc)"; echo "cpu=$cpu"
  echo "memory_bytes=$(( $(grep MemTotal /proc/meminfo | tr -s ' ' | cut -d' ' -f2) * 1024 ))"
  echo "memory_limit=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)"
  echo "cpu_flags=$(grep -m1 -o -w -E 'avx2|avx512f|avx512ifma|fma' /proc/cpuinfo 2>/dev/null | sort -u | tr '\n' ' ')"
  [ -n "$(command -v nproc)" ] && echo "cpu_quota=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
  echo "netem=$(tc qdisc show 2>/dev/null | grep -m1 netem)"
fi
md() { curl -s -m 1 -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1" 2>/dev/null; }
echo "gcp_machine_type=$(md machine-type | sed 's|.*/||')"
echo "gcp_zone=$(md zone | sed 's|.*/||')"
echo "time=$( [ -x /usr/bin/time ] && echo yes || echo no )"
echo "ping=$(command -v ping)"
echo "iperf3=$(command -v iperf3)"
"""


def probe(host):
    """The host's kernel, CPU, memory, cloud machine type and zone, emulated network, and the build of its binary."""
    env = {}
    for line in host.text(PROBE, timeout=60).splitlines():
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    info = host.text(f"cat {shlex.quote(host.binary)}.build.json 2>/dev/null")
    env["build"] = json.loads(info) if info.startswith("{") else None
    env["cores"] = int(env["cores"]) if env.get("cores", "").isdigit() else None
    env["memory_bytes"] = int(env["memory_bytes"]) if env.get("memory_bytes", "").isdigit() else None
    host.env = env
    return env


def ping_ms(src, dst_ip, count=5):
    out = src.text(f"ping -c {count} -q {shlex.quote(dst_ip)}", timeout=60)
    m = re.search(r"= ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms", out)
    return {"min": float(m.group(1)), "avg": float(m.group(2)), "max": float(m.group(3))} if m else None


def iperf_mbit(src, dst, seconds=6):
    """The TCP throughput from `src` to `dst` with iperf3 over `seconds`, the first two (slow start) omitted, or None
    if either host lacks iperf3."""
    if not src.env.get("iperf3") or not dst.env.get("iperf3"):
        return None
    dst.run("pkill -x iperf3 2>/dev/null; iperf3 -s -1 -D", check=False)
    for _ in range(20):  # until the server listens
        if dst.text("(ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -c ':5201 '") not in ("", "0"):
            break
        time.sleep(0.25)
    out = src.text(f"iperf3 -c {shlex.quote(dst.ip)} -t {seconds} -O 2 -J", timeout=seconds + 30)
    try:
        return round(json.loads(out)["end"]["sum_received"]["bits_per_second"] / 1e6, 1)
    except (ValueError, KeyError):
        return None


def measure_network(hosts, all_pairs, bandwidth):
    """Round-trip times (and optionally TCP throughput) from party 1 to every other party, or between all pairs."""
    if len(hosts) < 2 or hosts[0].transport == "local":
        return {"rtt_ms": {}, "bandwidth_mbit": {}}
    pairs = [(i, j) for i in range(len(hosts)) for j in range(i + 1, len(hosts))] if all_pairs else \
        [(0, j) for j in range(1, len(hosts))]
    rtt, bw = {}, {}
    for i, j in pairs:
        key = f"{i + 1}-{j + 1}"
        rtt[key] = ping_ms(hosts[i], hosts[j].ip)
        if bandwidth:
            bw[key] = iperf_mbit(hosts[i], hosts[j])
    return {"rtt_ms": rtt, "bandwidth_mbit": bw}


def local_build():
    """The commit of the local checkout and whether it has changes (for the local transport)."""
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "rustc": subprocess.run(["rustc", "-V"], capture_output=True, text=True).stdout.strip()}


# ------------------------------------------------------------------------------------------- one run

def seconds(text):
    match = re.fullmatch(r"([\d.]+)(ns|µs|ms|s|m)?", text.strip())
    scale = {"ns": 1e-9, "µs": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0}
    return float(match.group(1)) * scale[match.group(2) or "s"]


def peak_bytes(stderr):
    darwin = re.search(r"(\d+)\s+peak memory footprint", stderr)
    if darwin:
        return int(darwin.group(1))
    linux = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr)
    return int(linux.group(1)) * 1024 if linux else None


def report(stdout):
    get = lambda pattern: re.search(pattern, stdout, re.M)
    labels = {}
    table = stdout.split("bytes recv", 1)[1] if "bytes recv" in stdout else ""
    for line in table.strip().splitlines():
        f = line.split()
        if len(f) == 5 and f[1].isdigit():
            labels[f[0]] = {"calls": int(f[1]), "rounds": int(f[2]), "sent": int(f[3]), "received": int(f[4])}
    hiding = get(r"^hiding\s+(yes|no)")
    return {
        "hiding": hiding.group(1) == "yes" if hiding else None,
        "setup_s": seconds(get(r"^setup\s+(\S+)").group(1)),
        "connect_s": seconds(get(r"^connect\s+(\S+)").group(1)),
        "commit_s": seconds(get(r"^commit\s+(\S+)").group(1)),
        "eval_s": seconds(get(r"^eval\s+(\S+)").group(1)),
        "network_s": seconds(get(r"^network\s+(\S+)").group(1)),
        "proof_bytes": int(get(r"^proof\s+(\d+) bytes").group(1)),
        "sent_bytes": int(get(r"^sent\s+(\d+) payload").group(1)),
        "sent_wire_bytes": int(get(r"^sent\s+\d+ payload bytes in \d+ frames, (\d+) bytes on the wire").group(1)),
        "rounds": sum(v["rounds"] for v in labels.values()),
        "labels": labels,
        "vector": get(r"^vector\s+(\S+)").group(1),
    }


def time_prefix(host):
    if host.env.get("time") == "no":
        return ""
    return "/usr/bin/time -l " if host.env.get("kernel", "").startswith("Darwin") else "/usr/bin/time -v "


def run_once(args, hosts, rep, out):
    n = len(hosts)
    name = instance_name(args.preset)
    addresses = [h.address for h in hosts]
    run_dir = f"{args.run_name}/rep-{rep}"
    procs = []
    started = time.time()
    for p, host in enumerate(hosts, 1):
        threads = args.threads or (max((os.cpu_count() or 1) // n, 1) if host.transport == "local"
                                   else host.env.get("cores") or 1)
        base = f"{host.workdir}/{run_dir}"
        config = {
            "preset": args.preset, "n": n, "party": p, "addresses": addresses,
            "session": seed(args.tag, "session", name, rep),
            "setup_seed": seed(args.tag, "setup", name),
            "instance_seed": seed(args.tag, "instance", name),
            "party_seed": seed(args.tag, "party", name, p),
            "output": f"{base}/party-{p}",
            "connect_timeout_ms": args.timeout_ms, "recv_timeout_ms": args.timeout_ms,
        }
        if args.spill:
            config["spill_dir"] = f"{host.workdir}/spill-{p}"
        path = f"{base}/party-{p}.json"
        host.put_text(path, json.dumps(config, indent=1))
        if args.spill:
            host.run(f"mkdir -p {shlex.quote(config['spill_dir'])}")
        command = (f"RAYON_NUM_THREADS={threads} {time_prefix(host)}{shlex.quote(host.binary)} {shlex.quote(path)}")
        procs.append((host, threads, subprocess.Popen(host.argv(command), stdout=subprocess.PIPE,
                                                      stderr=subprocess.PIPE, text=True)))
    parties, failed = [], False
    for p, (host, threads, proc) in enumerate(procs, 1):
        stdout, stderr = proc.communicate()
        (out / f"rep-{rep}-party-{p}.stdout").write_text(stdout)
        (out / f"rep-{rep}-party-{p}.stderr").write_text(stderr)
        if proc.returncode != 0:
            failed = True
            sys.stderr.write(f"party {p} on {host.name} failed:\n{stderr[-2000:]}\n")
            continue
        parties.append({**report(stdout), "peak_bytes": peak_bytes(stderr), "threads": threads})
    if failed:
        return None
    wall = time.time() - started
    first = hosts[0]
    remote_out = f"{first.workdir}/{run_dir}/party-1"
    if args.verify_on == "local" or first.transport == "local":
        local = first.get_dir(remote_out, out / f"rep-{rep}") if first.transport != "local" else Path(remote_out)
        verify = subprocess.run(["/usr/bin/time", "-l" if platform.system() == "Darwin" else "-v",
                                 str(LOCAL_BINARY), "--verify", str(local)], capture_output=True, text=True)
        v_out, v_err, v_code = verify.stdout, verify.stderr, verify.returncode
    else:
        proc = first.run(f"{time_prefix(first)}{shlex.quote(first.binary)} --verify {shlex.quote(remote_out)}",
                         check=False, timeout=args.timeout_ms / 1000)
        v_out, v_err, v_code = proc.stdout.decode(), proc.stderr.decode(), proc.returncode
        first.get_dir(remote_out, out / f"rep-{rep}")
    (out / f"rep-{rep}-verify.stdout").write_text(v_out)
    (out / f"rep-{rep}-verify.stderr").write_text(v_err)
    accepted = v_code == 0 and v_out.startswith("accepted")
    match = re.search(r"verified in (\S+)", v_out)
    return {
        "accepted": accepted,
        "hiding": parties[0]["hiding"],
        "wall_s": round(wall, 2),
        "setup_s": max(p["setup_s"] for p in parties),
        "connect_s": max(p["connect_s"] for p in parties),
        "commit_s": max(p["commit_s"] for p in parties),
        "eval_s": max(p["eval_s"] for p in parties),
        "network_s": max(p["network_s"] for p in parties),
        "verify_s": seconds(match.group(1)) if match else None,
        "verifier_peak_bytes": peak_bytes(v_err),
        "proof_bytes": parties[0]["proof_bytes"],
        "sent_bytes_per_party": [p["sent_bytes"] for p in parties],
        "sent_wire_bytes_per_party": [p["sent_wire_bytes"] for p in parties],
        "rounds_per_party": [p["rounds"] for p in parties],
        "peak_bytes_per_party": [p["peak_bytes"] for p in parties],
        "threads_per_party": [p["threads"] for p in parties],
        "vector": parties[0]["vector"],
        "labels_party_1": parties[0]["labels"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts", type=Path, required=True, help="the hosts file (JSON)")
    ap.add_argument("--preset", required=True)
    ap.add_argument("--out", type=Path, required=True, help="local directory of the logs and the fetched proofs")
    ap.add_argument("--reps", type=int, default=1, help="repetitions of the run (one JSON line each)")
    ap.add_argument("--threads", type=int, help="threads of each party (default: the host's cores, or its share "
                                                "of the local cores)")
    ap.add_argument("--spill", action="store_true", help="keep the blocks of f' on disk on each host")
    ap.add_argument("--tag", default="hachi-co/bench", help="the seeds are SHA-256 of this tag and indices")
    ap.add_argument("--timeout-ms", type=int, default=3_600_000)
    ap.add_argument("--verify-on", choices=["party1", "local"], default="party1",
                    help="where the verifier runs (party1: on party 1's host, timed there)")
    ap.add_argument("--rtt-all-pairs", action="store_true", help="ping between all pairs, not only from party 1")
    ap.add_argument("--bandwidth", action="store_true", help="measure TCP throughput with iperf3 (6 s per pair)")
    ap.add_argument("--setting-json", default="{}", help="JSON merged into the recorded setting (scripts/wan_grid.py)")
    ap.add_argument("--first-rep", type=int, default=0, help="the index of the first repetition (its session seed)")
    args = ap.parse_args()
    spec, hosts = load_hosts(args.hosts)
    args.started = time.strftime("%Y%m%d-%H%M%S")
    args.run_name = f"{args.preset}-{args.started}"
    args.out.mkdir(parents=True, exist_ok=True)
    envs = [probe(h) for h in hosts]
    network = measure_network(hosts, args.rtt_all_pairs, args.bandwidth)
    if hosts[0].transport == "local":
        for env in envs:
            env["build"] = env["build"] or local_build()
    builds = {json.dumps(e["build"], sort_keys=True) for e in envs}
    if len(builds) > 1:
        sys.stderr.write("warning: the hosts run different builds\n")
    predicted, packed = None, None
    if LOCAL_BINARY.exists():
        sizes = subprocess.run([str(LOCAL_BINARY), "--sizes", args.preset, str(len(hosts))], capture_output=True,
                               text=True)
        if sizes.returncode == 0:
            predicted, packed = map(int, sizes.stdout.split())
    setting = {
        "transport": hosts[0].transport,
        **spec.get("setting", {}),
        **json.loads(args.setting_json),
        "hosts": [{"host": h.name, "address": h.address, **{k: v for k, v in e.items() if k != "build"}}
                  for h, e in zip(hosts, envs)],
        "build": envs[0]["build"],
        "network": network,
    }
    runs = []
    for rep in range(args.first_rep, args.first_rep + args.reps):
        result = run_once(args, hosts, rep, args.out)
        if result is None:
            return 1
        line = {"preset": args.preset, "n": len(hosts), "spill": args.spill, "tag": args.tag, "rep": rep,
                **result, "predicted_bytes": predicted, "packed_bytes": packed, "setting": setting,
                "started": args.started}
        print(json.dumps(line), flush=True)
        runs.append(line)
        if not result["accepted"]:
            sys.stderr.write(f"rep {rep}: the verifier rejected the proof\n")
            return 1
    if args.reps > 1:
        med = lambda key: statistics.median(r[key] for r in runs)
        sys.stderr.write(f"{args.preset}, n = {len(hosts)}, {args.reps} runs: median commit {med('commit_s'):.2f} s, "
                         f"eval {med('eval_s'):.2f} s, verify {med('verify_s'):.2f} s\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
