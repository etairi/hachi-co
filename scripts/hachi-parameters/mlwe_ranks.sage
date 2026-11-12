# The MLWE rank nu of the commitment randomness of hiding Hachi, per secure modulus, ring degree and base.
#
# The randomness and the error of t_i = A s_i + A' rho_i are uniform on the digit set S_b, the set of Definition 2,
# so hardness depends on b: a smaller b means less noise and a larger rank. For each (q, d, b) this prints the least
# nu such that LWE.estimate.rough (core-SVP, ADPS16: usvp and dual hybrid) reaches the target, with n = nu d, uniform
# digits as secret and error, and m = 2 n samples. A commitment gives the attacker n_A d samples per secret, which is
# fewer, so m = 2 n is conservative. These are estimates, not proofs.
#
# Run from a directory holding this script and lattice-estimator at the revision of secure.json, with the family's
# file and its lambda as the target (the probe reads mlwe_ranks-<target>.txt):
#   sage mlwe_ranks.sage [secure.json] [target=115] > mlwe_ranks-115.txt
#   sage mlwe_ranks.sage secure100.json target=100 > mlwe_ranks-100.txt
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "lattice-estimator"))
from estimator import *

TARGET = int(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("target=")), "128"))
BASES = [4, 8, 16]
MAX_NU = 32


def pairs(path):
    presets = json.load(open(path))["presets"]
    return sorted({(int(p["params"]["q"]), int(p["params"]["d"])) for p in presets})


def bits(q, d, b, nu):
    n = nu * d
    digits = ND.Uniform(-(b // 2), (b - 1) // 2)
    params = LWE.Parameters(n=n, q=q, Xs=digits, Xe=digits, m=2 * n)
    res = LWE.estimate.rough(params, quiet=True)
    return min(float(log(c["rop"], 2)) for c in res.values())


path = next((a for a in sys.argv[1:] if "=" not in a), "secure.json")
print("# q d b nu bits (least nu with at least %d bits)" % TARGET)
for q, d in pairs(path):
    for b in BASES:
        for nu in range(1, MAX_NU + 1):
            cost = bits(q, d, b, nu)
            if cost >= TARGET:
                print(q, d, b, nu, f"{cost:.1f}", flush=True)
                break
        else:
            print(q, d, b, "none", "0", flush=True)
