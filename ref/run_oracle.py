"""Hachi-Co Sage reference oracle (milestone 0): honest runs at the toy presets, every relation check,
negative tests, and the JSON fixtures in ref/fixtures/.

Usage (any working directory):
    sage -python ref/run_oracle.py                # self-tests, n = 1, 2, 3, negative tests, fixtures
    sage -python ref/run_oracle.py --n 2          # only toy-n2
    sage -python ref/run_oracle.py --no-write     # do not write fixtures
    sage -python ref/run_oracle.py --sweep 3      # also 3 extra honest runs per n (seeds .../1 .. .../3)
Exit status 0 iff every check passed.
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from hachico_ref import negative as neg  # noqa: E402
from hachico_ref.fixtures import build_fixture, write_fixture  # noqa: E402
from hachico_ref.oracle import run_instance  # noqa: E402
from hachico_ref.params import toy_preset  # noqa: E402
from hachico_ref.selftest import run_all  # noqa: E402


def seed_for(P, idx=0):
    return "hachi-co-ref/%s/%d" % (P.name, idx)


def select_run(P, seed_index=None, max_index=32):
    """Fixture seed: --seed-index if given; otherwise the smallest index s >= 0 whose run contains at least
    one aborted ring-to-linear attempt, so that the fixture covers the abort-and-retry path.  Every run tried
    on the way is fully checked and reported (no run is discarded silently)."""
    tried = []
    rng = [seed_index] if seed_index is not None else range(max_index)
    for s in rng:
        res = run_instance(P, seed_for(P, s))
        pattern = [a["aborting_provers"] for a in res["r2l"]["attempts"]]
        ok = all(res["relations"].values()) and res["verifier_accept"]
        tried.append(dict(seed=seed_for(P, s), aborting_provers_per_attempt=pattern, all_checks_pass=bool(ok)))
        if seed_index is not None or len(pattern) > 1:
            rule = ("given by --seed-index" if seed_index is not None else
                    "smallest index s >= 0 whose run has an aborted ring-to-linear attempt (retry-path coverage)")
            return res, dict(rule=rule, tried=tried)
    raise RuntimeError("no run with an abort among the first %d seeds" % max_index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--no-selftest", action="store_true")
    ap.add_argument("--no-negative", action="store_true")
    ap.add_argument("--sweep", type=int, default=0)
    ap.add_argument("--seed-index", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "fixtures"))
    args = ap.parse_args()
    failures = []
    command = "sage -python ref/run_oracle.py"

    if not args.no_selftest:
        print("== unit self-tests")
        t = time.time()
        for name, ok, det in run_all():
            print("  %s  %s%s" % ("PASS" if ok else "FAIL", name, ("  [" + det + "]") if det else ""))
            if not ok:
                failures.append("selftest: " + name)
        print("  (%.1fs)" % (time.time() - t))
        r = neg.rej_exhaustive()
        ok = r["within_bound_uniform"] and r["boundary_breaks_uniformity"]
        print("  %s  Rej exhaustive (N=%d, gamma=%d, beta1=%d): every v with ||v|| <= beta1 gives each z of the "
              "box exactly once; all %d v with ||v|| = beta1+1 give a different accepted set"
              % ("PASS" if ok else "FAIL", r["N"], r["gamma"], r["beta1"], r["boundary_v_count"]))
        if not ok:
            failures.append("rej exhaustive")

    for n in args.n:
        P = toy_preset(n)
        print("\n== %s: q=%d d=%d k=%d delta=%d r=%d m=%d tau=%d mu=%d n'=%d l''=%d d0=%d nu=%d"
              % (P.name, P.q, P.d, P.k, P.delta, P.r, P.m, P.tau, P.mu, P.n_prime, P.ell_pp, P.d0, P.nu))
        t_sel = time.time()
        res, selection = select_run(P, args.seed_index)
        for tr_ in selection["tried"][:-1]:
            print("  %s  run %s: attempts %s (no abort, not used for the fixture)"
                  % ("PASS" if tr_["all_checks_pass"] else "FAIL", tr_["seed"], tr_["aborting_provers_per_attempt"]))
            if not tr_["all_checks_pass"]:
                failures.append("%s run %s" % (P.name, tr_["seed"]))
        print("  info  fixture seed %s (%s); selection took %.1fs"
              % (selection["tried"][-1]["seed"], selection["rule"], time.time() - t_sel))
        for k_, v_ in res["relations"].items():
            print("  %s  relation  %s" % ("PASS" if v_ else "FAIL", k_))
            if not v_:
                failures.append("%s relation %s" % (P.name, k_))
        vc = res["verifier_checks"]
        nfail = [k_ for k_, v_ in vc.items() if not v_]
        print("  %s  verifier: %d checks (conv, %d rounds x 2, %d PCS openings, 2 final), accept=%s"
              % ("PASS" if not nfail else "FAIL", len(vc), P.ell_pp, 3 * P.n, res["verifier_accept"]))
        if nfail:
            failures.append("%s verifier %s" % (P.name, nfail))
        atts = res["r2l"]["attempts"]
        print("  info  ring-to-linear attempts: %d (aborting provers per attempt: %s); ||z|| = %d <= beta = %d"
              % (len(atts), [a["aborting_provers"] for a in atts], res["r2l"]["z_norm"], P.beta))
        d0s, d1s = res["hypercube"]["round_degrees_F0"], res["hypercube"]["round_degrees_F1"]
        print("  info  degree of sum_p F_0 round polys: U rounds %s, L rounds %s (2b+1 = %d, draft d0 = %d); "
              "F_1: max %d (d1 = %d)" % (sorted(set(d0s[:P.ell_U])), sorted(set(d0s[P.ell_U:])), 2 * P.b + 1,
                                         P.d0, max(d1s), P.d1))
        negres = {}
        if not args.no_negative:
            t = time.time()
            negres["transcript"] = neg.transcript_tests(res)
            negres["witness"] = neg.witness_tests(res)
            negres["open"] = neg.open_tests(res)
            negres["rej_boundary_toy"] = neg.rej_boundary_toy(P)
            negres["abort_rate"] = neg.abort_rate(P)
            for group in ("transcript", "witness", "open"):
                for name, info in negres[group].items():
                    ok = info["passed"]
                    extra = ""
                    if "failing_checks" in info:
                        extra = "  [failing: %s]" % ", ".join(info["failing_checks"]) if info["failing_checks"] else "  [no check fails]"
                    elif "failing" in info:
                        extra = "  [failing: %s]" % ", ".join(info["failing"])
                    print("  %s  negative  %s%s" % ("PASS" if ok else "FAIL", name, extra))
                    if not ok:
                        failures.append("%s negative %s" % (P.name, name))
            rb = negres["rej_boundary_toy"]
            ok = rb["accepted_set_smaller"] and rb["within_bound_full"]
            print("  %s  negative  Rej at ||v|| = beta1+1 (toy params, exact): first coordinate reaches %d of %d "
                  "box values (all %d at ||v|| = beta1)" % ("PASS" if ok else "FAIL",
                                                           rb["first_coordinate_reachable_at_beta1_plus_1"],
                                                           rb["box_side"], rb["first_coordinate_reachable_at_beta1"]))
            if not ok:
                failures.append("%s rej boundary" % P.name)
            ar = negres["abort_rate"]
            print("  %s  abort rate over %d attempts: %.4f vs exact 1 - M^-n = %.4f (3 sigma = %.4f, %s, seed %d)"
                  % ("PASS" if ar["within_3_sigma"] else "FAIL", ar["trials"], ar["rate"], ar["exact"],
                     3 * ar["sigma"], ar["rng"], ar["seed"]))
            if not ar["within_3_sigma"]:
                failures.append("%s abort rate" % P.name)
            print("  (negative tests %.1fs)" % (time.time() - t))
        print("  timings: " + ", ".join("%s %.2fs" % kv for kv in res["timings"].items()))
        if not args.no_write:
            os.makedirs(args.out, exist_ok=True)
            fx = build_fixture(res, negres, command, selection)
            path = os.path.join(args.out, P.name + ".json")
            size = write_fixture(fx, path)
            print("  wrote %s (%.0f KB)" % (os.path.relpath(path, os.path.dirname(HERE)), size / 1024))
        for s in range(1, args.sweep + 1):
            t = time.time()
            rs = run_instance(P, seed_for(P, s))
            okr = all(rs["relations"].values()) and rs["verifier_accept"]
            print("  %s  sweep seed %s: attempts %s, accept=%s (%.1fs)"
                  % ("PASS" if okr else "FAIL", seed_for(P, s), [a["aborting_provers"] for a in rs["r2l"]["attempts"]],
                     rs["verifier_accept"], time.time() - t))
            if not okr:
                failures.append("%s sweep %d" % (P.name, s))

    print("\n== %s" % ("ALL CHECKS PASSED" if not failures else "FAILURES: %s" % failures))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
