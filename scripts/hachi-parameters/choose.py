#!/usr/bin/env python3
"""Chooses Hachi's parameters for the secure presets of Hachi-Co.

Inputs:
- layouts.jsonl, from the ignored test `hachi_layouts` of crates/hachico/tests/hachi_parameters.rs: per secure preset,
  object (the share polynomial <w~>_p over F_{q^k} and the blinders g0, g1, each without masks and with them, as the
  objects masked, masked-g0 and masked-g1) and candidate choices (b, omega, n_A; n_B = n_D = 1; with masks also the
  height n_A0 of level 0 and the MLWE rank nu of mlwe_ranks.txt), the automatic schedule and the proof's words;
- thresholds.txt, from msis_thresholds.sage: the largest integer log2(beta) with infinity-norm Module-SIS at least
  115 / 128 bits hard, per modulus q and rank n d (lattice estimator 53da598, SIS.lattice).

Output: crates/params/presets/hachi.json. Hachi-Co's base PCS is one Hachi instance, so one choice serves all three
objects: per preset, among the choices where all three pass, the one with the smallest sum of proof words. A choice
with masks masks all three objects, and any such choice beats one without masks: it has the entry `hiding`. Without
one, the preset keeps the choice without masks and has no such entry.

Binding instances per iteration j (Hachi, Lemma 7 and Lemma 8, with the ranges the range check enforces):
- A: rank n_A d, norm 8 omega B_tau, B_tau = max|.| of the values tau balanced base-b digits represent
     (the paper's (beta-bar, omega-bar) = (2 b^tau, 2 omega) with b^tau replaced by the exact range);
- B: rank n_B d, norm 2b;   D: rank n_D d, norm 2b;
- the final witness, sent in the clear and checked against the last commitment: A at norm 2 beta_s (final layout).
At a masked level, A has rank n_A0 d instead, and tau is that of the mask bound beta^(0), which the digits of z and
of z_varrho share; A' adds an identity block, so the instance has the same rank.
Knowledge error of coordinate-wise special soundness: |C| >= 2^(lambda + r + 1) for the l1 ball of radius omega in
Z^d (|C| = sum_k C(d,k) C(omega,k) 2^k), and omega < sqrt(q)/(2 sqrt 2) (Lemma 8).
A rank missing from the table is read at the largest tabulated rank below it (conservative: Module-SIS gets harder
with the rank). A tabulated value equal to the search's cap, floor(log2 q) - 1, means "at least that" (conservative
too). These are estimates, not proofs. The MLWE rank is an estimate too (mlwe_ranks.sage).
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAMBDA = 128
PHI = 13
OBJECTS = ("share", "g0", "g1")
MASKED = {"masked": "share", "masked-g0": "g0", "masked-g1": "g1"}


def representable(b, tau):
    s = (b ** tau - 1) // (b - 1) if tau else 0
    return max((b // 2) * s, ((b - 1) // 2) * s)


def ball_bits(d, omega):
    total = sum(math.comb(d, k) * math.comb(omega, k) * 2 ** k for k in range(0, min(d, omega) + 1))
    return math.log2(total)


def thresholds():
    thr = {}
    for line in (HERE / "thresholds.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 7 and parts[3] == "|" and parts[0].isdigit():
            q, rank = int(parts[0]), int(parts[2])
            thr[(q, rank)] = tuple(None if v == "none" else int(v) for v in (parts[4], parts[6]))
    return thr


def max_norm_log2(thr, q, rank):
    ranks = sorted(r for (qq, r) in thr if qq == q and r <= rank)
    return thr[(q, ranks[-1])][1] if ranks else None


def failure(cfg, thr):
    """None if the binding instances of `cfg` reach the target, else the first that fails."""
    q, d, b, omega, n_a = cfg["q"], cfg["d"], cfg["b"], cfg["omega"], cfg["n_a"]
    its = cfg["schedule"]["iterations"]
    r_max = max([it["layout"]["r"] for it in its] + [cfg["schedule"]["final"]["r"]])
    if ball_bits(d, omega) < LAMBDA + r_max + 1:
        return "challenge set too small"
    if omega >= math.sqrt(q) / (2 * math.sqrt(2)):
        return "omega too large for Lemma 8"
    small = max_norm_log2(thr, q, d)
    if small is None or small < math.ceil(math.log2(2 * b)):
        return "B or D (norm 2b) not hard at rank d"
    have = max_norm_log2(thr, q, n_a * d)
    for j, it in enumerate(its):
        rank = cfg["n_a0"] * d if j == 0 and "n_a0" in cfg else n_a * d
        allowed = max_norm_log2(thr, q, rank)
        need = math.ceil(math.log2(8 * omega * representable(b, it["tau"])))
        if allowed is None or need > allowed:
            return f"A at iteration {j}: needs log2 norm {need}, rank {rank} allows {allowed}"
    need = math.ceil(math.log2(2 * cfg["schedule"]["final"]["beta_s"]))
    if have is None or need > have:
        return "final witness binding"
    return None


def main():
    layouts = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "layouts.jsonl"
    thr = thresholds()
    rows = [json.loads(line) for line in layouts.read_text().splitlines() if line.startswith("{")]
    groups = defaultdict(dict)
    masked = defaultdict(dict)
    for row in rows:
        key = (row["preset"], row["b"], row["omega"], row["n_a"])
        if row["object"] in MASKED:
            masked[key + (row["n_a0"],)][MASKED[row["object"]]] = row
        else:
            groups[key][row["object"]] = row
    candidates = defaultdict(list)
    for (preset, b, omega, n_a), objects in groups.items():
        if len(objects) == len(OBJECTS) and not any(failure(objects[o], thr) for o in OBJECTS):
            candidates[preset].append((1, b, omega, n_a, objects))
    for (preset, b, omega, n_a, n_a0), objects in masked.items():
        if len(objects) == len(OBJECTS) and not any(failure(objects[o], thr) for o in OBJECTS):
            candidates[preset].append((0, b, omega, n_a, objects))
    presets = {}
    for preset, found in sorted(candidates.items()):
        # Any candidate with masks beats one without, then fewer words win.
        def rank(c):
            unmasked, b, omega, n_a, objects = c
            words = sum(objects[o]["proof_words"] for o in OBJECTS)
            return (unmasked, words, n_a, b, omega, objects["share"].get("n_a0", 0))
        unmasked, b, omega, n_a, objects = min(found, key=rank)
        entry = dict(
            b=b,
            omega=omega,
            n_a=n_a,
            n_b=1,
            n_d=1,
            proof_words=dict(w=objects["share"]["proof_words"], g0=objects["g0"]["proof_words"],
                             g1=objects["g1"]["proof_words"]),
            iterations={("w" if o == "share" else o): len(objects[o]["schedule"]["iterations"]) for o in OBJECTS},
        )
        if not unmasked:
            share = objects["share"]
            entry["hiding"] = dict(nu=share["nu"], n_a=share["n_a0"], **{"lambda": LAMBDA, "phi": PHI})
        presets[preset] = entry
    names = sorted({row["preset"] for row in rows})
    missing = [name for name in names if name not in presets]
    print(json.dumps(dict(target_bits=LAMBDA, estimator_revision="53da5982597709ba0fdf94ea37a84d822310fd84",
                          presets=presets), indent=1, sort_keys=True))
    plain = [name for name, entry in presets.items() if "hiding" not in entry]
    print(f"{len(presets)} presets of {len(names)}; none found for {missing}; no masks for {plain}", file=sys.stderr)


if __name__ == "__main__":
    main()
