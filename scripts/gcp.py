#!/usr/bin/env python3
"""The Google Cloud side of the wide-area runs: create one instance per party, build hachi-co-party on each from the
working tree, write the hosts file of scripts/cluster.py, optionally emulate a fixed round-trip time with tc netem, and
delete everything afterwards.

Every command only prints the gcloud commands it would run. Nothing is created, changed or deleted without --execute.

    cp scripts/gcp.example.json runs/gcp.json          # set the project, zones and machine type
    python3 scripts/gcp.py --config runs/gcp.json create             # print the commands
    python3 scripts/gcp.py --config runs/gcp.json create --execute   # instances and the firewall rule
    python3 scripts/gcp.py --config runs/gcp.json build --execute    # tools, Rust, the source, cargo build
    python3 scripts/gcp.py --config runs/gcp.json hosts --execute --out runs/gcp/hosts.json
    python3 scripts/cluster.py --hosts runs/gcp/hosts.json --preset secure100-l16-n4-a8 --out runs/gcp/x --reps 3
    python3 scripts/gcp.py --config runs/gcp.json netem --rtt-ms 50 --rate-mbit 200 --execute   # optional
    python3 scripts/gcp.py --config runs/gcp.json delete --execute

Each instance builds for its own CPU (.cargo/config.toml sets target-cpu=native on x86-64), so use one machine type for
all parties. The source is a tarball of the files git tracks, with their local changes, and its commit, its changes
flag and its SHA-256 go into the build record that scripts/cluster.py copies into every result. Instances in one zone
see each other at LAN latency; zones in different regions give a real WAN, and `netem` gives a fixed emulated one, as
in the experiments of PPP26 (ePrint 2026/729, Section 7).
"""
import argparse
import getpass
import hashlib
import io
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    cfg = json.loads(Path(path).read_text())
    if "YOUR" in cfg.get("project", "YOUR"):
        sys.stderr.write("note: set \"project\" in the config before --execute\n")
    cfg.setdefault("network", "default")
    cfg.setdefault("tag", "hco-party")
    cfg.setdefault("port", 7300)
    cfg.setdefault("boot_disk_gb", 50)
    cfg["ssh_user"] = cfg.get("ssh_user") or getpass.getuser()
    return cfg


def gcloud(cfg, *args):
    return ["gcloud", *args, "--project", cfg["project"]]


def ssh(cfg, inst, command):
    return gcloud(cfg, "compute", "ssh", f"{cfg['ssh_user']}@{inst['name']}", "--zone", inst["zone"],
                  "--command", command, "--quiet")


def show(argv):
    print(" ".join(shlex.quote(a) for a in argv))


def execute(commands, really, parallel=False):
    """Prints each command, and runs it with --execute (all at once with parallel=True)."""
    for argv in commands:
        show(argv)
    if not really:
        print("# dry run: nothing was executed (pass --execute)")
        return []
    if parallel:
        procs = [subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                 for argv in commands]
        outs = []
        for argv, proc in zip(commands, procs):
            out, _ = proc.communicate()
            outs.append(out)
            if proc.returncode != 0:
                raise SystemExit(f"failed ({proc.returncode}): {' '.join(argv)}\n{out[-3000:]}")
        return outs
    outs = []
    for argv in commands:
        proc = subprocess.run(argv, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"failed ({proc.returncode}): {' '.join(argv)}\n{proc.stderr[-3000:]}")
        outs.append(proc.stdout)
    return outs


def create(cfg, args):
    commands = []
    for inst in cfg["instances"]:
        commands.append(gcloud(cfg, "compute", "instances", "create", inst["name"], "--zone", inst["zone"],
                               "--machine-type", inst.get("machine_type", cfg["machine_type"]),
                               "--image-family", cfg["image_family"], "--image-project", cfg["image_project"],
                               "--boot-disk-size", f"{cfg['boot_disk_gb']}GB", "--network", cfg["network"],
                               "--tags", cfg["tag"]))
    n = len(cfg["instances"])
    commands.append(gcloud(cfg, "compute", "firewall-rules", "create", f"{cfg['tag']}-parties",
                           "--network", cfg["network"], "--direction", "INGRESS",
                           "--allow", f"tcp:{cfg['port']}-{cfg['port'] + n},tcp:5201,icmp",
                           "--source-tags", cfg["tag"], "--target-tags", cfg["tag"]))
    execute(commands, args.execute)


def source_tarball():
    """The files git tracks, as they are in the working tree, gzipped, with the commit and the changes flag."""
    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, check=True).stdout
    files = [f for f in git("ls-files", "-z").decode().split("\0") if f and (ROOT / f).is_file()]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT) as tar:
        for f in sorted(files):
            info = tar.gettarinfo(str(ROOT / f), arcname=f)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with open(ROOT / f, "rb") as fh:
                tar.addfile(info, fh)
    data = buf.getvalue()
    commit = git("rev-parse", "HEAD").decode().strip()
    dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip())
    return data, commit, dirty, hashlib.sha256(data).hexdigest()


BUILD = r"""set -e
sudo apt-get update -y -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q build-essential pkg-config curl time iproute2 iputils-ping iperf3
[ -x "$HOME/.cargo/bin/cargo" ] || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain RUST
"$HOME/.cargo/bin/rustup" toolchain install RUST --profile minimal
rm -rf "$HOME/hachi-co" && mkdir -p "$HOME/hachi-co" "$HOME/run" && tar -C "$HOME/hachi-co" -xzf "$HOME/hachi-co.tar.gz"
cd "$HOME/hachi-co" && "$HOME/.cargo/bin/cargo" +RUST build --release --locked -p hachi-co --bin hachi-co-party
printf '{"commit": "COMMIT", "dirty": DIRTY, "source_sha256": "SHA", "rustc": "%s", "built_in": "gcp %s"}\n' \
  "$("$HOME/.cargo/bin/rustc" +RUST -V)" "$(uname -m)" > target/release/hachi-co-party.build.json
cat target/release/hachi-co-party.build.json
"""


def build(cfg, args):
    data, commit, dirty, sha = source_tarball()
    tarball = Path(args.tarball)
    tarball.parent.mkdir(parents=True, exist_ok=True)
    tarball.write_bytes(data)
    print(f"# {tarball}: {len(data)} bytes, commit {commit}{' with local changes' if dirty else ''}, sha256 {sha}")
    script = (BUILD.replace("RUST", cfg["rust"]).replace("COMMIT", commit)
              .replace("DIRTY", "true" if dirty else "false").replace("SHA", sha))
    uploads = [gcloud(cfg, "compute", "scp", "--zone", inst["zone"], str(tarball),
                      f"{cfg['ssh_user']}@{inst['name']}:hachi-co.tar.gz", "--quiet") for inst in cfg["instances"]]
    execute(uploads, args.execute, parallel=True)
    outs = execute([ssh(cfg, inst, script) for inst in cfg["instances"]], args.execute, parallel=True)
    for inst, out in zip(cfg["instances"], outs):
        print(f"{inst['name']}: {out.strip().splitlines()[-1] if out.strip() else ''}")


def hosts(cfg, args):
    describe = [gcloud(cfg, "compute", "instances", "describe", inst["name"], "--zone", inst["zone"],
                       "--format", "json") for inst in cfg["instances"]]
    outs = execute(describe, args.execute)
    if not outs:
        return
    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    home = f"/home/{cfg['ssh_user']}"
    parties, ssh_lines = [], []
    for p, (inst, text) in enumerate(zip(cfg["instances"], outs), 1):
        nic = json.loads(text)["networkInterfaces"][0]
        internal, external = nic["networkIP"], nic["accessConfigs"][0]["natIP"]
        parties.append({"host": inst["name"], "address": f"{internal}:{cfg['port'] + p - 1}",
                        "workdir": f"{home}/run", "binary": f"{home}/hachi-co/target/release/hachi-co-party"})
        ssh_lines += [f"Host {inst['name']}", f"  HostName {external}", f"  User {cfg['ssh_user']}",
                      "  IdentityFile ~/.ssh/google_compute_engine", "  StrictHostKeyChecking accept-new",
                      f"  UserKnownHostsFile {out_dir.resolve() / 'known_hosts'}", ""]
    (out_dir / "ssh_config").write_text("\n".join(ssh_lines))
    zones = sorted({inst["zone"] for inst in cfg["instances"]})
    setting = {"name": f"gcp {', '.join(zones)}", "provider": "gcp", "zones": [i["zone"] for i in cfg["instances"]],
               "machine_type": cfg["machine_type"], "image_family": cfg["image_family"]}
    args.out.write_text(json.dumps({"transport": "ssh", "ssh_config": "ssh_config", "setting": setting,
                                    "parties": parties}, indent=1) + "\n")
    print(f"wrote {args.out} and {out_dir / 'ssh_config'}")


def netem(cfg, args):
    dev = r"$(ip -o route get 10.0.0.1 | sed -n 's/.* dev \([^ ]*\).*/\1/p')"
    if args.clear:
        command = f"sudo tc qdisc del dev {dev} root 2>/dev/null || true; tc qdisc show dev {dev}"
    else:
        spec = []
        if args.rtt_ms:
            spec += ["delay", f"{args.rtt_ms / 2:g}ms"]
        if args.rate_mbit:
            spec += ["rate", f"{args.rate_mbit:g}mbit"]
        if not spec:
            raise SystemExit("give --rtt-ms, --rate-mbit or --clear")
        command = f"sudo tc qdisc replace dev {dev} root netem {' '.join(spec)} limit 100000 && tc qdisc show dev {dev}"
    execute([ssh(cfg, inst, command) for inst in cfg["instances"]], args.execute, parallel=True)


def delete(cfg, args):
    commands = []
    by_zone = {}
    for inst in cfg["instances"]:
        by_zone.setdefault(inst["zone"], []).append(inst["name"])
    for zone, names in by_zone.items():
        commands.append(gcloud(cfg, "compute", "instances", "delete", *names, "--zone", zone, "--quiet"))
    commands.append(gcloud(cfg, "compute", "firewall-rules", "delete", f"{cfg['tag']}-parties", "--quiet"))
    execute(commands, args.execute)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("create", "build", "hosts", "netem", "delete"):
        p = sub.add_parser(name)
        p.add_argument("--execute", action="store_true", help="run the commands instead of printing them")
        if name == "build":
            p.add_argument("--tarball", default=str(ROOT / "target" / "gcp" / "hachi-co.tar.gz"))
        if name == "hosts":
            p.add_argument("--out", type=Path, required=True, help="the hosts file to write (ssh_config beside it)")
        if name == "netem":
            p.add_argument("--rtt-ms", type=float)
            p.add_argument("--rate-mbit", type=float)
            p.add_argument("--clear", action="store_true", help="remove the emulation")
    args = ap.parse_args()
    cfg = load(args.config)
    {"create": create, "build": build, "hosts": hosts, "netem": netem, "delete": delete}[args.command](cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
