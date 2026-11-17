#!/usr/bin/env python3
"""The table, the plot coordinates and the numbers of the paper's benchmark section (`7_Benchmarks.tex`) from the
benchmark records: the local grid (`scripts/local_grid.py --plan phase1`, every prover on one machine over the
loopback) and the grid of PPP26's network settings (`scripts/wan_grid.py`, one prover per instance).

    python3 scripts/paper_tables.py --local benchmarks/gcp-2026-11-17-local.jsonl \\
        --wan benchmarks/gcp-2026-11-17-wan.jsonl

Definitions, as the section states them:
- per-prover time: Commit plus Eval of the slowest prover (`cluster.py` records the slowest of each), and for PPP26
  the commitment plus the opening that their code times (`scripts/ppp26.py`: `commit_s` and `open_s`, the slowest
  rank) plus the preparation of their masks (`mask_commit_s`, every prover's encoding and commitment of its masks),
  which their timer leaves out while our per-prover time includes the corresponding work of ours; their dealer
  (`dealer_setup_s`, which shares the polynomial, as our provers are given their shares) is left out;
- communication per prover: the payload bytes the busiest prover sends (`sent_bytes_per_party`), and for PPP26 the
  payload bytes of its timed part counted by our MPI shim;
- peak memory per prover: the largest resident set of a prover (`/usr/bin/time`);
- sizes in decimal units, 1 MB = 10^6 bytes and 1 GB = 10^9 bytes;
- repetitions averaged (mean), with their relative spread printed.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ELLS = (20, 22, 24)


def lines(path):
    if path is None or not Path(path).exists():
        return []
    out = []
    for text in Path(path).read_text().splitlines():
        text = text.strip()
        if text.startswith("{"):
            out.append(json.loads(text))
    return out


def mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def spread(xs):
    """The relative spread (max - min) / mean of the repetitions, in percent."""
    return 100 * (max(xs) - min(xs)) / mean(xs) if len(xs) > 1 else 0.0


def per_prover_time(d):
    return d["commit_s"] + d["eval_s"]


def fmt(x, digits=1):
    return f"{x:.{digits}f}"


def coords(points, digits=1):
    return " ".join(f"({x}, {fmt(y, digits)})" for x, y in points)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", help="the records of the local grid (JSON lines)")
    ap.add_argument("--wan", help="the records of the grid of PPP26's settings (JSON lines)")
    args = ap.parse_args()
    local, wan = lines(args.local), lines(args.wan)

    # The local grid, by cell.
    cells = defaultdict(list)
    for d in local:
        if d.get("accepted", True):
            cells[d.get("cell")].append(d)
    print(f"% local: {len(local)} lines, {len(cells)} cells; wan: {len(wan)} lines")
    print("% local cells and repetitions: " + ", ".join(f"{c} ({len(v)})" for c, v in sorted(cells.items())))
    rejected = [d.get("cell") for d in local if not d.get("accepted", True)]
    if rejected:
        print(f"% NOT ACCEPTED: {rejected}")

    def cell(ell, n, threads=1, kind="parties", suffix=""):
        name = f"l{ell}-single-t{threads}" if kind == "single" else f"l{ell}-n{n}-t{threads}{suffix}"
        return cells.get(name, [])

    # 1. The local runs at 115 bits, 1 thread per prover: every n, for the table and the text.
    print("\n% 1. Local, 115 bits, 1 thread per prover (mean of the repetitions, spread of the per-prover time)")
    print("% n ell | per-prover time s (Commit + Eval, Commit share) | sent MB | peak GB | proof MB | verify s "
          "| verifier GB")
    local_rows = {}
    for n in (2, 4, 8):
        for ell in ELLS:
            rs = cell(ell, n)
            if not rs:
                continue
            t = [per_prover_time(d) for d in rs]
            row = dict(
                time=mean(t), commit=mean([d["commit_s"] for d in rs]), eval=mean([d["eval_s"] for d in rs]),
                sent=mean([max(d["sent_bytes_per_party"]) for d in rs]) / 1e6,
                peak=mean([max(d["peak_bytes_per_party"]) for d in rs]) / 1e9,
                proof=mean([d["proof_bytes"] for d in rs]) / 1e6, verify=mean([d["verify_s"] for d in rs]),
                verifier=mean([d.get("verifier_peak_bytes", 0) for d in rs]) / 1e9, spread=spread(t), reps=len(rs))
            local_rows[(n, ell)] = row
            print(f"% {n} {ell} | {fmt(row['time'])} ({fmt(100 * row['commit'] / row['time'])}% Commit, spread "
                  f"{fmt(row['spread'])}%, {row['reps']} reps) | {fmt(row['sent'])} | {fmt(row['peak'], 2)} | "
                  f"{fmt(row['proof'], 2)} | {fmt(row['verify'], 2)} | {fmt(row['verifier'], 1)}")

    # The table: n = 8.
    n_tab = 8
    have = [ell for ell in ELLS if (n_tab, ell) in local_rows]
    if have:
        r = [local_rows[(n_tab, ell)] for ell in have]
        print(f"\n% tab:bench-size, n = {n_tab}, local, 115 bits, 1 thread per prover")
        print("        number of variables $\\ell$ & " + " & ".join(str(e) for e in have) + " \\\\")
        print("        \\midrule")
        print("        per-prover time (s)           & " + " & ".join(fmt(x["time"]) for x in r) + " \\\\")
        print("        communication per prover (MB) & " + " & ".join(fmt(x["sent"]) for x in r) + " \\\\")
        print("        peak memory per prover (GB)   & " + " & ".join(fmt(x["peak"], 2) for x in r) + " \\\\")
        print("        proof size (MB)               & " + " & ".join(fmt(x["proof"], 2) for x in r) + " \\\\")
        print("        verification time (s)         & " + " & ".join(fmt(x["verify"], 1) for x in r) + " \\\\")

    # The communication of prover 1 by message (local, every n): Commit, the ring-to-linear step's F_AH(1^n), the
    # openings' all-gather of the sumcheck, and the rest.
    print("\n% Prover 1's sent bytes by message, MB (local, 1 thread): total | Commit | r2l F_AH(1^n) | sumcheck openings")
    for n in (2, 4, 8):
        for ell in ELLS:
            rs = cell(ell, n)
            if not rs or "labels_party_1" not in rs[0]:
                continue
            lab = rs[0]["labels_party_1"]
            total = sum(v["sent"] for v in lab.values())
            commit = sum(v["sent"] for k, v in lab.items() if k.startswith(("hachi-co/commit/", "hachi-co/mpc/")))
            zsum = lab.get("hachi-co/r2l/F_AH(1^n)", {}).get("sent", 0)
            openings = lab.get("hachi-co/sumcheck/all-gather/openings", {}).get("sent", 0)
            print(f"% n{n} l{ell}: {fmt(total / 1e6, 2)} | {fmt(commit / 1e3, 1)} kB | {fmt(zsum / 1e6, 2)} | "
                  f"{fmt(openings / 1e6, 2)}")

    # 2. The single Hachi prover, 115 bits.
    print("\n% 2. Single Hachi prover, 115 bits (Commit + Eval, proof, peak), against ours at n = 2, 1 thread")
    single = {}
    for threads in (1, 8):
        for ell in ELLS:
            rs = cells.get(f"l{ell}-single-t{threads}", [])
            if not rs:
                continue
            t = mean([d["commit_s"] + d["eval_s"] for d in rs])
            single[(threads, ell)] = dict(time=t, proof=mean([d["proof_bytes"] for d in rs]) / 1e6,
                                          peak=mean([d["peak_bytes"] for d in rs]) / 1e9,
                                          verify=mean([d["verify_s"] for d in rs]))
            ours = local_rows.get((2, ell)) if threads == 1 else None
            ratio = f", ours n = 2 / single: time x{fmt(ours['time'] / t, 2)}, proof x{fmt(ours['proof'] / single[(threads, ell)]['proof'], 2)}" if ours else ""
            print(f"% t{threads} l{ell}: {fmt(t)} s, proof {fmt(single[(threads, ell)]['proof'], 2)} MB, peak "
                  f"{fmt(single[(threads, ell)]['peak'], 2)} GB, verify {fmt(single[(threads, ell)]['verify'], 2)} s{ratio}")

    # 3. The other local cells: 4 threads at n = 2, and every n without hiding in the base PCS (1 thread).
    print("\n% 3. Local: n = 2 at 4 threads, and hiding against non-hiding base PCS at 1 thread (time, Eval, proof, peak)")
    for ell in ELLS:
        base = local_rows.get((2, ell))
        rs = cells.get(f"l{ell}-n2-t4", [])
        if rs and base:
            t = mean([per_prover_time(d) for d in rs])
            print(f"% l{ell} n2 4 threads: {fmt(t)} s against {fmt(base['time'])} s at 1 thread (x{fmt(base['time'] / t, 2)} faster)")
    print("% hiding vs non-hiding: n ell | hiding s | non-hiding s | hiding / non-hiding | share of the time hiding costs | proof MB h/nh | peak GB h/nh")
    for n in (2, 4, 8):
        for ell in ELLS:
            base = local_rows.get((n, ell))
            rs = cells.get(f"l{ell}-n{n}-t1-nonhiding", [])
            if rs and base:
                t = mean([per_prover_time(d) for d in rs])
                pr = mean([d["proof_bytes"] for d in rs]) / 1e6
                pk = mean([max(d["peak_bytes_per_party"]) for d in rs]) / 1e9
                print(f"% {n} {ell} | {fmt(base['time'])} | {fmt(t)} | x{fmt(base['time'] / t, 2)} | "
                      f"{fmt(100 * (base['time'] - t) / base['time'])}% | {fmt(base['proof'], 2)}/{fmt(pr, 2)} | "
                      f"{fmt(base['peak'], 2)}/{fmt(pk, 2)}")

    # 4. PPP26's settings: ours at 100 bits and PPP26, 1 thread per prover; ours at 115 bits and 8 threads in LAN.
    ours = defaultdict(list)
    theirs = defaultdict(list)
    for d in wan:
        g = d.get("setting", {}).get("grid_setting")
        if d.get("scheme") == "ppp26":
            theirs[(g, d["n"], d["log_m"])].append(d)
        elif d.get("accepted", True):
            threads = max(d.get("threads_per_party", [1]))
            ours[(g, d["preset"], threads)].append(d)
    settings = ("lan", "wan50", "wan100")
    print("\n% 4. PPP26's settings, 1 thread per prover: per-prover time s (ours: Commit + Eval at 100 bits; PPP26: "
          "commit + open + their masks' preparation)")
    for g in settings:
        print(f"% -- {g}")
        for n in (4, 8):
            pts_ours = [(ell, mean([per_prover_time(d) for d in ours[(g, f'secure100-l{ell}-n{n}-a8', 1)]]))
                        for ell in ELLS if ours.get((g, f"secure100-l{ell}-n{n}-a8", 1))]
            pts_theirs = [(ell, mean([d["total_s"] + d["mask_commit_s"] for d in theirs[(g, n, ell)]
                                      if d.get("total_s") is not None]))
                          for ell in ELLS if theirs.get((g, n, ell))]
            print(f"\\addplot[mark=*] coordinates {{{coords(pts_ours)}}};  % ours, n = {n}, {g}")
            print(f"\\addplot[mark=o] coordinates {{{coords(pts_theirs)}}};  % PPP26, n = {n}, {g}")
    print("\n% Ours / PPP26 per-prover time (setting, n, ell): against their timed part | against it with their masks")
    for g in settings:
        for n in (4, 8):
            for ell in ELLS:
                o = ours.get((g, f"secure100-l{ell}-n{n}-a8", 1))
                t = theirs.get((g, n, ell))
                if o and t and t[0].get("total_s"):
                    to = mean([per_prover_time(d) for d in o])
                    tt = mean([d["total_s"] for d in t])
                    tm = mean([d["total_s"] + d["mask_commit_s"] for d in t])
                    print(f"% {g} n{n} l{ell}: ours {fmt(to)} s, PPP26 {fmt(tt, 2)} s timed, {fmt(tm, 2)} s with masks: "
                          f"x{fmt(to / tt)} | x{fmt(to / tm)}")
    print("\n% PPP26 details (setting, n, log M): timed total, commit, open | dealer, masks (untimed) | sent MB "
          "(timed, max) | peak GB (ranks seen) | proof KB | verify s")
    for (g, n, lm), rs in sorted(theirs.items()):
        for d in rs:
            sent = [x for x in d.get("sent_bytes_per_prover", []) if x is not None]
            peaks = d.get("peak_bytes_per_prover", [])
            print(f"% {g} n{n} 2^{lm}: {d.get('total_s')} = {d.get('commit_s')} + {d.get('open_s')} | "
                  f"{d.get('dealer_setup_s')}, {d.get('mask_commit_s')} | {fmt(max(sent) / 1e6) if sent else '-'} | "
                  f"{fmt(max(peaks) / 1e9, 2) if peaks else '-'} ({len(peaks)} of {n}) | {d.get('proof_kb')} | "
                  f"{d.get('verify_s_reported')}")
    print("\n% ours in PPP26's settings (setting, preset, threads): per-prover time, Commit, sent MB, peak GB, proof MB, "
          "verify s, measured RTT ms and Mbit/s (1-2)")
    for (g, preset, threads), rs in sorted(ours.items()):
        d = rs[0]
        net = d.get("setting", {}).get("network", {})
        rtt = net.get("rtt_ms", {}).get("1-2", {}).get("avg")
        bw = net.get("bandwidth_mbit", {}).get("1-2")
        print(f"% {g} {preset} t{threads}: {fmt(mean([per_prover_time(x) for x in rs]))} s, Commit "
              f"{fmt(mean([x['commit_s'] for x in rs]), 2)} s, {fmt(mean([max(x['sent_bytes_per_party']) for x in rs]) / 1e6)} MB, "
              f"{fmt(mean([max(x['peak_bytes_per_party']) for x in rs]) / 1e9, 2)} GB, "
              f"{fmt(mean([x['proof_bytes'] for x in rs]) / 1e6, 2)} MB, {fmt(mean([x['verify_s'] for x in rs]), 2)} s, "
              f"RTT {rtt} ms, {bw} Mbit/s")

    # 5. Against n at l = 24, 115 bits: local at 1 thread, LAN at 8 threads.
    print("\n% 5. fig:bench-provers, l = 24, 115 bits: per-prover time s and peak memory MB against n")
    loc_t = [(n, local_rows[(n, 24)]["time"]) for n in (2, 4, 8) if (n, 24) in local_rows]
    loc_m = [(n, 1000 * local_rows[(n, 24)]["peak"]) for n in (2, 4, 8) if (n, 24) in local_rows]
    lan = {n: ours.get(("lan", f"secure-l24-n{n}-a8", 8), []) for n in (2, 4, 8)}
    lan_t = [(n, mean([per_prover_time(d) for d in rs])) for n, rs in lan.items() if rs]
    lan_m = [(n, mean([max(d["peak_bytes_per_party"]) for d in rs]) / 1e6) for n, rs in lan.items() if rs]
    print(f"\\addplot[mark=*] coordinates {{{coords(loc_t)}}};  % local, 1 thread")
    print(f"\\addplot[mark=square*] coordinates {{{coords(lan_t)}}};  % LAN, 8 threads")
    print(f"\\addplot[mark=*] coordinates {{{coords(loc_m, 0)}}};  % local, 1 thread, MB")
    print(f"\\addplot[mark=square*] coordinates {{{coords(lan_m, 0)}}};  % LAN, 8 threads, MB")


if __name__ == "__main__":
    main()
