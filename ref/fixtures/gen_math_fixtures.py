"""Fixtures for the Rust crate hachi-co-math, computed with the Sage reference
ref/hachico_ref/subfield.py (the definition of the subfield maps; not modified here).

Run from the repository root:

    sage -python ref/fixtures/gen_math_fixtures.py

Output: crates/math/tests/fixtures/subfield_<name>.json, one file per modulus
(q32 = 2^32 - 99, q52 = 2^52 - 395, q61 = 2^61 - 259: the largest primes = 5 mod 8 below
2^32 (the legacy Hachi modulus), 2^52 and 2^61), each with one case per
(d, k) in {(16,1), (16,2), (16,4), (16,8), (64,4), (64,8), (1024,4)}.

Every random input comes from Sage's global generator, seeded per case with
set_random_seed(seed); the seed is stored in the case. Field elements are lists of k
coordinates in Hachi's basis (Eq. (7)), ring elements lists of d coefficients, all as
integers in [0, q). Exponents are little-endian lists of 64-bit limbs. Index conventions are
those of subfield.py (monomials ordered by nat(b) with b_1 the most significant bit).

Each case holds:
  h_exponents         sorted odd exponents of H = <sigma_{-1}, sigma_{4k+1}> mod 2d
  field.elements      e_0..e_{k-1} (unit coordinates), -1, (q-1, ..., q-1), four random elements
  field.products      products[i][j] = fmul(elements[i], elements[j])
  field.sums          fadd(elements[i], elements[i+1]) for consecutive pairs
  field.inverses      [index, finv(elements[index])] for the nonzero elements listed
  field.powers        [index, exponent limbs, fpow(elements[index], exponent)]
  field.frobenius     fpow(x, q) for every element
  embed               [x, embed(x)] for one random x
  u, trace_u          a random ring element and the coordinates of trace(u)
  sigma               [i, sigma(u, i)] for i in {2d-1, 4k+1 mod 2d, 3, q mod 2d}   (d <= 64)
  product             [w, mul(u, w)] for a random w                                (d <= 64)
  in_subfield         [in_subfield(embed(x)), in_subfield(u)]
  psi.a, psi.psi_a    a random vector of d/k field elements and psi(a)
  psi.psi_inv_u       psi_inv(u)
  theorem2            b, the coordinates of Tr_H(psi(a) sigma_{-1}(psi(b))), and <a, b>
  e_map               point (log2(d/k) field elements), monomials(point), v = psi(monomials),
                      coordinates of E(u, v) and of E(psi(a), v), <a, monomials>,
                      and E_matrix(v) (d <= 64 only)

The generator also asserts the identities it can check cheaply (Theorem 2, the E property,
psi_inv(psi(a)) = a for d <= 64, x * finv(x) = 1).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "ref"))

from hachico_ref.subfield import Subfield  # noqa: E402
from sage.all import Integer, randint, set_random_seed, version  # noqa: E402

MODULI = [("q32", 2**32 - 99), ("q52", 2**52 - 395), ("q61", 2**61 - 259)]
CASES = [(16, 1), (16, 2), (16, 4), (16, 8), (64, 4), (64, 8), (1024, 4)]
OUT = os.path.join(ROOT, "crates", "math", "tests", "fixtures")


def ints(v):
    return [int(c) for c in v]


def limbs(e):
    e = int(e)
    out = []
    while e:
        out.append(e & (2**64 - 1))
        e >>= 64
    return out


def make_case(q, d, k, seed):
    set_random_seed(seed)
    S = Subfield(q, d, k)
    F = S.F
    small = d <= 64
    case = {"q": int(q), "d": d, "k": k, "seed": seed, "h_exponents": [int(h) for h in S.H]}

    # ---- field ----
    units = [[F(1) if j == m else F(0) for j in range(k)] for m in range(k)]
    elements = units + [[F(-1)] + [F(0)] * (k - 1), [F(q - 1)] * k]
    elements += [S.frandom() for _ in range(4)]
    products = [[ints(S.fmul(x, y)) for y in elements] for x in elements]
    sums = [ints(S.fadd(elements[i], elements[i + 1])) for i in range(len(elements) - 1)]
    inverse_indices = [1 if k > 1 else 0, k + 1] + list(range(len(elements) - 4, len(elements)))
    inverses = []
    for i in inverse_indices:
        inv = S.finv(elements[i])
        assert S.fmul(elements[i], inv) == S.fone()
        inverses.append([i, ints(inv)])
    powers = []
    for i in range(len(elements) - 2, len(elements)):
        e = Integer(randint(0, 2**128 - 1))
        powers.append([i, limbs(e), ints(S.fpow(elements[i], e))])
    frobenius = [ints(S.fpow(x, q)) for x in elements]
    case["field"] = {
        "elements": [ints(x) for x in elements],
        "products": products,
        "sums": sums,
        "inverses": inverses,
        "powers": powers,
        "frobenius": frobenius,
    }

    # ---- ring, embedding, trace, automorphisms ----
    x = S.frandom()
    case["embed"] = [ints(x), ints(S.embed(x))]
    u = [F.random_element() for _ in range(d)]
    case["u"] = ints(u)
    case["trace_u"] = ints(S.extract(S.trace(u)))
    if small:
        case["sigma"] = [[i, ints(S.sigma(u, i))] for i in (2 * d - 1, (4 * k + 1) % (2 * d), 3, int(q % (2 * d)))]
        w = [F.random_element() for _ in range(d)]
        case["product"] = [ints(w), ints(S.mul(u, w))]
    case["in_subfield"] = [bool(S.in_subfield(S.embed(x))), bool(S.in_subfield(u))]

    # ---- psi ----
    a = [S.frandom() for _ in range(d // k)]
    psi_a = S.psi(a)
    if small:
        assert S.psi_inv(psi_a) == a
    case["psi"] = {"a": [ints(c) for c in a], "psi_a": ints(psi_a), "psi_inv_u": [ints(c) for c in S.psi_inv(u)]}

    # ---- Theorem 2 ----
    b = [S.frandom() for _ in range(d // k)]
    lhs = S.trace(S.mul(psi_a, S.sigma(S.psi(b), 2 * d - 1)))
    inner_ab = S.inner(a, b)
    assert lhs == S.embed([F(d // k) * c for c in inner_ab]), "Theorem 2"
    case["theorem2"] = {"b": [ints(c) for c in b], "lhs": ints(S.extract(lhs)), "inner": ints(inner_ab)}

    # ---- E ----
    s = (d // k).bit_length() - 1
    point = [S.frandom() for _ in range(s)]
    mono = S.monomials(point)
    v = S.psi(mono)
    e_psi_a = S.extract(S.E(psi_a, v))
    inner_mono = S.inner(a, mono)
    assert e_psi_a == inner_mono, "E(psi(a)) = <a, monomials>"
    case["e_map"] = {
        "point": [ints(c) for c in point],
        "monomials": [ints(c) for c in mono],
        "v": ints(v),
        "e_u": ints(S.extract(S.E(u, v))),
        "e_psi_a": ints(e_psi_a),
        "inner_a_monomials": ints(inner_mono),
    }
    if small:
        case["e_map"]["matrix"] = [ints(row) for row in S.E_matrix(v)]
    return case


def main():
    os.makedirs(OUT, exist_ok=True)
    for mi, (name, q) in enumerate(MODULI):
        q = Integer(q)
        assert q.is_prime() and q % 8 == 5
        cases = []
        for ci, (d, k) in enumerate(CASES):
            t = time.time()
            cases.append(make_case(q, d, k, seed=1000 * mi + ci))
            print("%s d=%d k=%d: %.1fs" % (name, d, k, time.time() - t), flush=True)
        doc = {
            "meta": {
                "generator": "ref/fixtures/gen_math_fixtures.py",
                "reference": "ref/hachico_ref/subfield.py",
                "sage": version(),
                "modulus_name": name,
            },
            "cases": cases,
        }
        path = os.path.join(OUT, "subfield_%s.json" % name)
        with open(path, "w") as f:
            json.dump(doc, f, separators=(",", ":"))
            f.write("\n")
        print("wrote", path, os.path.getsize(path), "bytes", flush=True)


if __name__ == "__main__":
    main()
