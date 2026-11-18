#!/usr/bin/env python3
"""PPP26's collaborative PCS (Pappas, Papadopoulos, Papamanthou, "Code-based Scalable Collaborative SNARKs", ePrint
2026/729), run the way scripts/cluster.py runs ours, for the comparison of the paper's benchmark section.

Their code (the paper's reference [46], pinned at COMMIT below) has a PCS-only mode, `pigeon 1 <log2 M> <rate>`: the
commitment and the zero-knowledge opening of a random multilinear polynomial with M coefficients over F_{p^2},
p = 2^61 - 1, packed into shares of M/k per prover with k = n/4 (so n >= 4, and at most n/4 corrupted provers), at
about 100 bits (the query count is computed for 100 bits in the code). It needs x86-64 with AVX2, and MPI.

scripts/ppp26/timing.patch adds one line per rank, `hco-patch Id : ...`, with its commit and open times and the parts
that the mode's timed region leaves out: `dummy_setup`, where rank 0 plays a dealer and sends every prover its random
shares (a deployment needs a distributed protocol for them), and `prepare_mask_shares`, every prover's encoding and
commitment of its masks. It also reads the code's own communication counter around both, since the mode's printed
"Com" sums a local variable that stays 0. Everything else is their code as published.

Their build has no -fopenmp, so the compiler ignores the OpenMP loops of their Merkle trees, the only parallel code in
it, and every prover runs one thread. With --openmp, fetch, gcp-build and gcp-run use a second tree
(target/ppp26/src-openmp) and a second build on the instances (~/ppp26-openmp) with scripts/ppp26/openmp.patch on top,
which adds -fopenmp and makes the hash buffers of those loops private to each thread (their code shares them between
the threads), and gcp-run gives every prover --threads OpenMP threads.

    python3 scripts/ppp26.py fetch                           # clone at COMMIT into target/ppp26/src, apply the patch
    python3 scripts/ppp26.py docker-build                    # an amd64 image (Rosetta on Apple Silicon: tests only)
    python3 scripts/ppp26.py docker-run --n 4 --log-m 14 --reps 2 >> runs/ppp26-docker.jsonl
    python3 scripts/ppp26.py gcp-build --config runs/gcp.json            # print, then again with --execute
    python3 scripts/ppp26.py gcp-mpi --config runs/gcp.json --hosts runs/gcp/hosts.json
    python3 scripts/ppp26.py gcp-run --hosts runs/gcp/hosts.json --n 4 --log-m 20 --reps 3 >> runs/ppp26-gcp.jsonl

The Google Cloud commands use the instances of scripts/gcp.py and only print what they would run without --execute.
MPI starts the remote ranks over ssh from party 1's instance, on the TCP ports of MPI_PORTS, which gcp-mpi opens
between the instances.
"""
import argparse
import hashlib
import io
import json
import re
import shlex
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import cluster  # noqa: E402  (the hosts file, the probe of each host, the network measurements)
import gcp      # noqa: E402  (the configuration and the gcloud commands)

REPO = "https://github.com/ChristodoulosPappas/Code-Based-Scalable-coSNARKs"
COMMIT = "1c50a7410fa464ad325e0014d0d5fedd196d86c5"   # main on 2026-11-12 (last push 2025-11-14)
PATCH = HERE / "ppp26" / "timing.patch"
PACKING = HERE / "ppp26" / "packing.patch"   # their packing factor from HCO_PPP26_K (unset: their N/4)
OPENMP = HERE / "ppp26" / "openmp.patch"   # with --openmp: -fopenmp, and thread-private buffers in their parallel loops
SHIM = HERE / "ppp26" / "mpi_count.c"   # copied into their tree as hco/mpi_count.c
SRC = ROOT / "target" / "ppp26" / "src"
SRC_OPENMP = ROOT / "target" / "ppp26" / "src-openmp"
IMAGE = "ppp26:local"
BINARY_IN_IMAGE = "/ppp26/build/src/pigeon"
SHIM_IN_IMAGE = "/ppp26/build/libhco_mpi_count.so"
MPI_PORTS = (7400, 7599)   # the BTL ports 7400-7499 and the out-of-band ports 7500-7599
MCA = ["--mca", "btl_tcp_port_min_v4", str(MPI_PORTS[0]), "--mca", "btl_tcp_port_range_v4", "100",
       "--mca", "oob_tcp_dynamic_ipv4_ports", f"{MPI_PORTS[0] + 100}-{MPI_PORTS[1]}"]


def sh(argv, **kw):
    return subprocess.run(argv, check=True, **kw)


def patch_sha(openmp=False):
    """The SHA-256 of the patches and the shim, which together are our changes to their tree."""
    extra = OPENMP.read_bytes() if openmp else b""
    return hashlib.sha256(PATCH.read_bytes() + PACKING.read_bytes() + extra + SHIM.read_bytes()).hexdigest()


def remote_dir(args):
    """The directory of their build in the home directory of every instance."""
    return "ppp26-openmp" if getattr(args, "openmp", False) else "ppp26"


# ------------------------------------------------------------------------------------------- the source

def fetch(args):
    if (args.dir / ".git").exists():
        sh(["git", "-C", str(args.dir), "fetch", "--quiet", "origin"])
        sh(["git", "-C", str(args.dir), "reset", "--quiet", "--hard", COMMIT])
        sh(["git", "-C", str(args.dir), "clean", "--quiet", "-fdx"])
    else:
        args.dir.parent.mkdir(parents=True, exist_ok=True)
        sh(["git", "clone", "--quiet", REPO, str(args.dir)])
        sh(["git", "-C", str(args.dir), "checkout", "--quiet", COMMIT])
    sh(["git", "-C", str(args.dir), "apply", str(PATCH)])
    sh(["git", "-C", str(args.dir), "apply", str(PACKING)])
    if args.openmp:
        sh(["git", "-C", str(args.dir), "apply", str(OPENMP)])
    (args.dir / "hco").mkdir(exist_ok=True)
    (args.dir / "hco" / "mpi_count.c").write_bytes(SHIM.read_bytes())
    patches = ", ".join(f.name for f in [PATCH, PACKING] + ([OPENMP] if args.openmp else []))
    print(f"{args.dir}: {COMMIT} with {patches} (sha256 {patch_sha(args.openmp)[:16]}) and hco/mpi_count.c")


# ------------------------------------------------------------------------------------------- parsing a run

def parse(stdout, stderr):
    """The provers' times, the proof size and the verifier's time from the output of `pigeon 1 ...`."""
    ranks = {}
    for m in re.finditer(r"hco-patch Id : (\d+), Commit time: ([\d.]+), Open time: ([\d.]+), Dealer setup: ([\d.]+), "
                         r"Mask commit: ([\d.]+), Comm KB setup: ([\d.eE+-]+), Comm KB timed: ([\d.eE+-]+), "
                         r"MPI KB setup: ([\d.eE+-]+), MPI KB timed: ([\d.eE+-]+)", stdout):
        r = int(m.group(1))
        ranks[r] = dict(commit_s=float(m.group(2)), open_s=float(m.group(3)), dealer_setup_s=float(m.group(4)),
                        mask_commit_s=float(m.group(5)), comm_kb_setup=float(m.group(6)),
                        comm_kb_timed=float(m.group(7)), mpi_kb_setup=float(m.group(8)),
                        mpi_kb_timed=float(m.group(9)))
    for m in re.finditer(r"Id : (\d+), Commit time: [\d.]+, Open time, [\d.]+, Total: [\d.]+, CPU Only Pt: ([\d.]+), "
                         r"Computation only: ([\d.]+)", stdout):
        ranks.setdefault(int(m.group(1)), {}).update(cpu_only_s=float(m.group(2)), computation_only_s=float(m.group(3)))
    m0 = re.search(r"Id : 0, Pt: ([\d.]+), CPU Only Pt: ([\d.]+), Computation only: ([\d.]+), Vt: ([\d.]+)", stdout)
    if m0:
        ranks.setdefault(0, {}).update(cpu_only_s=float(m0.group(2)), computation_only_s=float(m0.group(3)))
    tail = re.search(r"Vt : ([\d.]+) sec, Ps: ([\d.]+) KB, Com: ([\d.]+) MB, Queries: (\d+)", stdout)
    workers = re.search(r"Workers: (\d+)", stdout)
    peaks = [int(k) * 1024 for k in re.findall(r"Maximum resident set size \(kbytes\): (\d+)", stderr)]
    if not ranks or not tail:
        return None
    per = [ranks[r] for r in sorted(ranks)]
    return {
        "workers": int(workers.group(1)) if workers else None,
        "commit_s": max(r["commit_s"] for r in per),
        "open_s": max(r["open_s"] for r in per),
        "total_s": max(r["commit_s"] + r["open_s"] for r in per),
        "dealer_setup_s": max(r["dealer_setup_s"] for r in per),
        "mask_commit_s": max(r["mask_commit_s"] for r in per),
        "cpu_only_s": max(r.get("cpu_only_s", 0.0) for r in per),
        "computation_only_s": max(r.get("computation_only_s", 0.0) for r in per),
        "comm_kb_timed_per_prover": [r["comm_kb_timed"] for r in per],
        "comm_kb_setup_per_prover": [r["comm_kb_setup"] for r in per],
        "sent_bytes_per_prover": [round(r["mpi_kb_timed"] * 1024) if r["mpi_kb_timed"] >= 0 else None for r in per],
        "sent_bytes_setup_per_prover": [round(r["mpi_kb_setup"] * 1024) if r["mpi_kb_setup"] >= 0 else None
                                        for r in per],
        "verify_s_reported": float(tail.group(1)),
        "proof_kb": float(tail.group(2)),
        "com_mb_reported": float(tail.group(3)),
        "queries": int(tail.group(4)),
        "peak_bytes_per_prover": peaks,
        "per_prover": per,
    }


def line(args, rep, result, setting):
    packing = getattr(args, "packing", None) or args.n // 4
    openmp = getattr(args, "openmp", False)
    return {"scheme": "ppp26", "commit": COMMIT, "patch_sha256": patch_sha(openmp), "n": args.n, "log_m": args.log_m,
            "ell": args.log_m, "rate": args.rate, "packing": packing, "openmp": openmp,
            "threads_per_prover": args.threads if openmp else 1, "rep": rep, **result, "setting": setting,
            "started": time.strftime("%Y%m%d-%H%M%S")}


# ------------------------------------------------------------------------------------------- Docker (one container)

def docker_build(args):
    sh(["docker", "build", "--platform", "linux/amd64", "-f", str(ROOT / "docker" / "ppp26.Dockerfile"),
        "-t", IMAGE, str(args.dir)])
    print(f"built {IMAGE} from {args.dir}")


def docker_run(args):
    """Every rank in one amd64 container (mpirun --oversubscribe): a test of the build and the parsing."""
    image = subprocess.run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], capture_output=True,
                           text=True, check=True).stdout.strip()
    setting = {"name": "docker amd64, one container", "transport": "docker-mpi", "image": image,
               "note": "Rosetta on Apple Silicon: for testing, the times mean nothing", **json.loads(args.setting_json)}
    for rep in range(args.first_rep, args.first_rep + args.reps):
        proc = subprocess.run(["docker", "run", "--rm", "--platform", "linux/amd64", IMAGE, "mpirun", "--oversubscribe",
                               "-np", str(args.n), "/usr/bin/time", "-v", "env", f"LD_PRELOAD={SHIM_IN_IMAGE}",
                               BINARY_IN_IMAGE, "1", str(args.log_m), str(args.rate)], capture_output=True, text=True)
        result = parse(proc.stdout, proc.stderr)
        if proc.returncode != 0 or result is None:
            sys.stderr.write(f"rep {rep} failed ({proc.returncode}):\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}\n")
            return 1
        print(json.dumps(line(args, rep, result, setting)), flush=True)
    return 0


# ------------------------------------------------------------------------------------------- Google Cloud

def source_tarball(src):
    """The patched source, without git's data and the authors' build directory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for path in sorted(src.rglob("*")):
            rel = path.relative_to(src)
            if rel.parts[0] in (".git", "build") or not path.is_file():
                continue
            info = tar.gettarinfo(str(path), arcname=str(rel))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with open(path, "rb") as fh:
                tar.addfile(info, fh)
    return buf.getvalue()


BUILD = r"""set -e
sudo apt-get update -y -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q build-essential cmake libopenmpi-dev openmpi-bin time iproute2 iputils-ping iperf3
rm -rf "$HOME/REMOTE" && mkdir -p "$HOME/REMOTE" && tar -C "$HOME/REMOTE" -xzf "$HOME/REMOTE.tar.gz"
cd "$HOME/REMOTE" && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release > /dev/null && make -j"$(nproc)" > /dev/null
mpicc -shared -fPIC -O2 -o libhco_mpi_count.so ../hco/mpi_count.c
printf '{"scheme": "ppp26", "commit": "COMMIT", "patch_sha256": "PATCHSHA", "source_sha256": "SRCSHA", "openmp": OPENMPFLAG, "built_in": "gcp %s"}\n' "$(uname -m)" > src/pigeon.build.json
cat src/pigeon.build.json
"""


def gcp_build(args):
    if not (args.dir / "CMakeLists.txt").exists():
        raise SystemExit(f"{args.dir}: no source (run fetch{' --openmp' if args.openmp else ''} first)")
    cfg = gcp.load(args.config)
    data = source_tarball(args.dir)
    tarball = ROOT / "target" / "ppp26" / f"{remote_dir(args)}.tar.gz"
    tarball.parent.mkdir(parents=True, exist_ok=True)
    tarball.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    print(f"# {tarball}: {len(data)} bytes, sha256 {sha}")
    script = (BUILD.replace("COMMIT", COMMIT).replace("PATCHSHA", patch_sha(args.openmp)).replace("SRCSHA", sha)
              .replace("REMOTE", remote_dir(args)).replace("OPENMPFLAG", "true" if args.openmp else "false"))
    uploads = [gcp.gcloud(cfg, "compute", "scp", "--zone", i["zone"], str(tarball),
                          f"{cfg['ssh_user']}@{i['name']}:{remote_dir(args)}.tar.gz", "--quiet")
               for i in cfg["instances"]]
    gcp.execute(uploads, args.execute, parallel=True)
    gcp.execute([gcp.ssh(cfg, i, script) for i in cfg["instances"]], args.execute, parallel=True)


def gcp_mpi(args):
    """Passwordless ssh from party 1's instance to the others, the MPI host file there, and the firewall rule for
    MPI's ports between the instances."""
    cfg = gcp.load(args.config)
    _, hosts = cluster.load_hosts(args.hosts)
    first = cfg["instances"][0]
    keygen = gcp.ssh(cfg, first, "test -f ~/.ssh/id_mpi || ssh-keygen -q -t ed25519 -N '' -f ~/.ssh/id_mpi; "
                                 "cat ~/.ssh/id_mpi.pub")
    print("# 1. a key on party 1's instance")
    out = gcp.execute([keygen], args.execute)
    pub = out[0].strip().splitlines()[-1] if out else "<the public key printed by step 1>"
    print("# 2. the key on every instance, and the host file on party 1's")
    authorize = [gcp.ssh(cfg, i, f"grep -qF {shlex.quote(pub)} ~/.ssh/authorized_keys 2>/dev/null || "
                                 f"echo {shlex.quote(pub)} >> ~/.ssh/authorized_keys") for i in cfg["instances"]]
    ips = [h.ip for h in hosts]
    config = ("printf 'Host *\\n  IdentityFile ~/.ssh/id_mpi\\n  StrictHostKeyChecking accept-new\\n' > ~/.ssh/config; "
              "chmod 600 ~/.ssh/config; " + f"printf '{''.join(f'{ip} slots=1{chr(92)}n' for ip in ips)}' > ~/ppp26-hosts; "
              + "; ".join(f"ssh -o BatchMode=yes {ip} true" for ip in ips))
    gcp.execute(authorize + [gcp.ssh(cfg, first, config)], args.execute)
    print("# 3. MPI's ports between the instances")
    gcp.execute([gcp.gcloud(cfg, "compute", "firewall-rules", "create", f"{cfg['tag']}-mpi", "--network",
                            cfg["network"], "--direction", "INGRESS", "--allow",
                            f"tcp:{MPI_PORTS[0]}-{MPI_PORTS[1]},tcp:22", "--source-tags", cfg["tag"],
                            "--target-tags", cfg["tag"])], args.execute)


def iface_bytes(host):
    """rx and tx bytes of the host's interface towards the others, from /proc/net/dev."""
    out = host.text("dev=$(ip -o route get 10.0.0.1 | sed -n 's/.* dev \\([^ ]*\\).*/\\1/p'); "
                    "grep \"$dev:\" /proc/net/dev")
    f = out.replace(":", " ").split()
    return (int(f[1]), int(f[9])) if len(f) > 9 else (None, None)


def gcp_run(args):
    """mpirun on party 1's instance with one rank per instance, the setting recorded as by scripts/cluster.py, and
    the bytes each instance's interface moved during the run (all traffic, framing included)."""
    spec, hosts = cluster.load_hosts(args.hosts)
    if len(hosts) < args.n:
        raise SystemExit(f"{len(hosts)} hosts for n = {args.n}")
    hosts = hosts[:args.n]
    home = hosts[0].workdir.rsplit('/', 1)[0]
    binary = f"{home}/{remote_dir(args)}/build/src/pigeon"
    shim = f"{home}/{remote_dir(args)}/build/libhco_mpi_count.so"
    packing = ["-x", f"HCO_PPP26_K={args.packing}"] if args.packing else []
    threads = ["-x", f"OMP_NUM_THREADS={args.threads}", "--bind-to", "none"] if args.openmp else []
    command = " ".join(shlex.quote(a) for a in ["mpirun", "--hostfile", "ppp26-hosts", "--map-by", "node", "-np",
                                                 str(args.n), *MCA, *packing, *threads, "/usr/bin/time", "-v", "env",
                                                 f"LD_PRELOAD={shim}", binary, "1", str(args.log_m),
                                                 str(args.rate)])
    if not args.execute:
        print(f"# on {hosts[0].name}: {command}")
        print("# dry run: nothing was executed (pass --execute)")
        return 0
    info = hosts[0].text(f"cat {shlex.quote(binary)}.build.json 2>/dev/null")
    envs = [cluster.probe(h) for h in hosts]
    setting = {**spec.get("setting", {}), "transport": "ssh-mpi",
               "build": json.loads(info) if info.startswith("{") else None,
               "network": cluster.measure_network(hosts, False, args.bandwidth),
               "hosts": [{"host": h.name, "address": h.address, **{k: v for k, v in e.items() if k != "build"}}
                         for h, e in zip(hosts, envs)],
               **json.loads(args.setting_json)}
    for rep in range(args.first_rep, args.first_rep + args.reps):
        before = [iface_bytes(h) for h in hosts]
        proc = hosts[0].run(f"cd ~ && {command}", check=False, timeout=args.timeout_s)
        after = [iface_bytes(h) for h in hosts]
        stdout, stderr = proc.stdout.decode(errors="replace"), proc.stderr.decode(errors="replace")
        result = parse(stdout, stderr)
        if proc.returncode != 0 or result is None:
            sys.stderr.write(f"rep {rep} failed ({proc.returncode}):\n{stdout[-3000:]}\n{stderr[-3000:]}\n")
            return 1
        result["iface_rx_tx_bytes_per_host"] = [
            [a[0] - b[0], a[1] - b[1]] if None not in a + b else None for a, b in zip(after, before)]
        print(json.dumps(line(args, rep, result, setting)), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("fetch", "docker-build", "docker-run", "gcp-build", "gcp-mpi", "gcp-run"):
        p = sub.add_parser(name)
        p.add_argument("--dir", type=Path, help="their source, patched (default target/ppp26/src, or "
                       "target/ppp26/src-openmp with --openmp)")
        if name in ("fetch", "gcp-build", "gcp-run"):
            p.add_argument("--openmp", action="store_true", help="their code with OpenMP (scripts/ppp26/openmp.patch)")
        if name in ("docker-run", "gcp-run"):
            p.add_argument("--n", type=int, required=True, help="provers (at least 4: the packing factor is n/4)")
            p.add_argument("--log-m", type=int, required=True, help="log2 of the number of coefficients")
            p.add_argument("--rate", type=int, default=4, help="the inverse rate of their code (4 in their paper)")
            p.add_argument("--reps", type=int, default=1)
            p.add_argument("--first-rep", type=int, default=0)
            p.add_argument("--setting-json", default="{}", help="JSON merged into the recorded setting")
        if name == "gcp-run":
            p.add_argument("--threads", type=int, default=8, help="OpenMP threads of every prover, with --openmp")
            p.add_argument("--packing", type=int, help="their packing factor k (default n/4, their choice; 1 gives "
                           "their highest corruption threshold, t <= n/2)")
        if name.startswith("gcp"):
            p.add_argument("--execute", action="store_true", help="run the commands instead of printing them")
        if name in ("gcp-build", "gcp-mpi"):
            p.add_argument("--config", type=Path, required=True, help="the configuration of scripts/gcp.py")
        if name in ("gcp-mpi", "gcp-run"):
            p.add_argument("--hosts", type=Path, required=True, help="the hosts file of scripts/gcp.py hosts")
        if name == "gcp-run":
            p.add_argument("--bandwidth", action="store_true")
            p.add_argument("--timeout-s", type=int, default=7200)
    args = ap.parse_args()
    if args.dir is None:
        args.dir = SRC_OPENMP if getattr(args, "openmp", False) else SRC
    if getattr(args, "n", 4) < 4:
        raise SystemExit("PPP26 needs n >= 4 provers (packing factor n/4)")
    return {"fetch": fetch, "docker-build": docker_build, "docker-run": docker_run, "gcp-build": gcp_build,
            "gcp-mpi": gcp_mpi, "gcp-run": gcp_run}[args.command](args) or 0


if __name__ == "__main__":
    sys.exit(main())
