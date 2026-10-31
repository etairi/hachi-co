# Largest infinity-norm bound beta (as log2) for which Module-SIS over R_q stays at least `target`
# bits hard, per modulus q and rank n*d, with ample columns; and MLWE bits for Gaussian randomness.
# The SIS wrapper is copied from parameters.sage (hco_sis_cost / hco_msis), same estimator revision
# as crates/params/presets/secure.json (53da598...). Run: sage msis_thresholds.sage > thresholds.txt
# Optional arguments: moduli (digits) restrict the moduli; ranks=r1,r2,... (or ranks=none) and mlwe=n1,n2,...
# (or mlwe=none) replace the Module-SIS ranks and the MLWE secret dimensions.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "lattice-estimator"))
from estimator import *

RATIOS = [1.25, 1.35, 1.5, 1.7, 2, 2.4, 3, 3.8, 5, 6.7, 9, 12.3, 17, 24, 32, 48, 64]
_cache = {}
def sis_cost(rank, q, bound, cols):
    key = (int(rank), int(q), int(bound), int(cols))
    if key in _cache: return _cache[key]
    try:
        c = SIS.lattice(SIS.Parameters(rank, q, bound, m=cols, norm=oo), zeta=0)
        val = None if c["rop"] == oo else float(log(c["rop"], 2))
    except OverflowError:
        val = None
    except TypeError:
        c = SIS.lattice(SIS.Parameters(rank, q, bound, m=cols, norm=2))
        val = None if c["rop"] == oo else float(log(c["rop"], 2))
    _cache[key] = val
    return val

def msis_bits(rank, q, bound):
    """Cheapest attack over a geometric grid of widths (as hco_msis, without the ternary refinement)."""
    if 2 * bound + 1 >= q: return 0.0
    trivial = float(rank * log(RR(q) / (2 * bound + 1), 2))
    best = None
    for f in RATIOS:
        m_ = int(ceil(f * rank))
        bits = sis_cost(rank, q, bound, m_)
        if bits is not None and (best is None or bits < best): best = bits
    return min(trivial, best) if best is not None else trivial

qs_all = [8388637, 4294967197, 17171531776141, 34801213478789, 274744508417461, 384156492036173,
      403122206336093, 4334287545157813, 8327002901668021, 15395032342003157, 74035283097514237,
      102705810279674029, 1183762967321726413, 6062548914780259333, 6618259129420972141]
qs = [int(a) for a in sys.argv[1:] if a.isdigit()] or qs_all
opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
def int_list(key, default):
    v = opts.get(key)
    return default if v is None else ([] if v == "none" else [int(x) for x in v.split(",")])
ranks = int_list("ranks", [256, 512, 1024, 2048, 3072, 4096])
mlwe_dims = int_list("mlwe", [256, 512, 1024, 2048])
print("q log2q rank | max log2(beta) with MSIS >= 115 bits | >= 128 bits")
for q in qs:
    lq = float(log(q, 2))
    for rank in ranks:
        out = []
        for target in (115, 128):
            lo, hi = 1, int(floor(lq)) - 1          # search over integer log2(beta)
            if msis_bits(rank, q, 2**lo) < target:
                out.append("none"); continue
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if msis_bits(rank, q, 2**mid) >= target: lo = mid
                else: hi = mid - 1
            out.append(str(lo))
        print(q, "%.1f" % lq, rank, "|", out[0], "|", out[1]); sys.stdout.flush()

print()
print("MLWE with Gaussian secret and error, sigma = 8: q, secret dim nu*d, samples m = n*d -> bits")
for q in qs:
    for nd in mlwe_dims:
        for m_ in [nd, 2 * nd]:
            params = LWE.Parameters(n=nd, q=q, Xs=ND.DiscreteGaussian(8), Xe=ND.DiscreteGaussian(8), m=m_)
            try:
                res = LWE.estimate.rough(params, quiet=True)
                bits = min(float(log(c["rop"], 2)) for c in res.values() if c["rop"] != oo)
            except Exception as e:
                bits = float('nan')
            print(q, nd, m_, "%.1f" % bits); sys.stdout.flush()
