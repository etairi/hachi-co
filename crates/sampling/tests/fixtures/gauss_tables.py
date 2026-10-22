#!/usr/bin/env python3
"""The tables of `hachi_co_sampling::gauss`, `src/gauss_tables.rs`, computed from their definition.

A port of ntru-trapdoor's `tools/gauss_tables.py` (commit 2273fda; same author, MIT), itself a
port of Jali's `tools/kat/half_gaussian_cdf.py`. The functions `_rho`, `_suffixes` and `table`
are unchanged; the tables are new.

Every table is the reverse CDT of a half-Gaussian K on {0, 1, 2, ...} with
Pr[K = i] proportional to rho(i) = exp(-(num/den) i^2), that is of standard deviation sigma with
1/(2 sigma^2) = num/den (not the width s = sigma sqrt(2 pi) of GPV and of Hwang-Seo-Song):

    name             num/den   sigma   bits   use
    GAUSS_SIGMA_1    1/2       1       128    commitment randomness of hiding Hachi
    GAUSS_SIGMA_2    1/8       2       128    (HachiChoices::hiding, docs/hachi-hiding.md 6.4)
    GAUSS_SIGMA_4    1/32      4       128
    GAUSS_SIGMA_8    1/128     8       128
    GAUSS_SIGMA_16   1/512     16      128

Entry j of a table is round(2^128 Pr[K > j]); a table keeps the entries that do not round to 0.
A sampler draws V uniform in [0, 2^128) and sets K = #{j : V < entry j}. The entries are
non-increasing, so Pr[K > j] = entry_j / 2^128 exactly and K never exceeds the table length.

Every entry is enclosed in interval arithmetic (mpmath.iv, 1024-bit endpoints). The sums are
cut after TERMS terms and the rest is bounded by a geometric series:
(N + m)^2 >= N^2 + 2 N m gives sum_{i >= N} rho(i) <= rho(N) / (1 - exp(-2 N num/den)).
The script fails if an enclosure contains a rounding boundary, so every rounding is certain.

Run with an interpreter that has mpmath, for example SageMath's:
    sage -python crates/sampling/tests/fixtures/gauss_tables.py          print the Rust file
    sage -python crates/sampling/tests/fixtures/gauss_tables.py --check  compare it with src/gauss_tables.rs
    sage -python crates/sampling/tests/fixtures/gauss_tables.py --digest print the digests the tests pin
"""
import argparse
import hashlib
import sys
import textwrap
from pathlib import Path

import mpmath as mp

PRECISION = 1024
RUST = Path(__file__).resolve().parents[2] / "src" / "gauss_tables.rs"

# name: (num, den, sigma as the documentation writes it, bits, TERMS)
TABLES = {
    "GAUSS_SIGMA_1": (1, 2, "1", 128, 64),
    "GAUSS_SIGMA_2": (1, 8, "2", 128, 88),
    "GAUSS_SIGMA_4": (1, 32, "4", 128, 136),
    "GAUSS_SIGMA_8": (1, 128, "8", 128, 232),
    "GAUSS_SIGMA_16": (1, 512, "16", 128, 424),
}


def _rho(i, num, den):
    """An enclosure of exp(-num i^2 / den)."""
    return mp.iv.exp(-mp.iv.mpf(num * i * i) / den)


def _suffixes(num, den, terms):
    """Enclosures of sum_{i >= j} rho(i) for j = 0..terms (the tail beyond `terms` bounded)."""
    iv = mp.iv
    rho = [_rho(i, num, den) for i in range(terms)]
    # The omitted terms i >= terms lie in [0, rest].
    rest = _rho(terms, num, den) / (1 - iv.exp(-iv.mpf(2 * num * terms) / den))
    omitted = iv.mpf([0, rest.b])
    # suffix[j] encloses sum_{i >= j} rho(i); accumulated from the tail, no cancellation.
    suffix = [None] * (terms + 1)
    suffix[terms] = omitted
    for i in range(terms - 1, -1, -1):
        suffix[i] = suffix[i + 1] + rho[i]
    return suffix


def table(num, den, bits, terms, rows=None):
    """The entries of one table, as Python integers: every entry that does not round to 0, or
    the first `rows` entries (each of which must be nonzero)."""
    iv = mp.iv
    saved = iv.prec, mp.mp.prec
    try:
        iv.prec = mp.mp.prec = PRECISION
        suffix = _suffixes(num, den, terms)
        total = suffix[0]
        scale = iv.mpf(2) ** bits
        out = []
        for j in range(terms - 1):
            if rows is not None and j == rows:
                return out
            x = suffix[j + 1] / total * scale
            lo, hi = mp.mpf(x.a), mp.mpf(x.b)
            r = int(mp.nint((lo + hi) / 2))
            if not (r - mp.mpf(1) / 2 < lo and hi < r + mp.mpf(1) / 2):
                raise ArithmeticError(f"entry {j}: the enclosure [{lo}, {hi}] does not fix "
                                      "the rounding")
            if r == 0:
                if rows is not None:
                    raise ArithmeticError(f"entry {j} rounds to 0 before row {rows}")
                return out
            out.append(r)
        raise ArithmeticError("TERMS too small: no entry rounds to 0")
    finally:
        iv.prec, mp.mp.prec = saved


def tables():
    """Every table by name, in the order of TABLES."""
    return {name: table(num, den, bits, terms)
            for name, (num, den, _, bits, terms) in TABLES.items()}


HEADER = r"""//! The tables of [`crate::gauss`], written by `tests/fixtures/gauss_tables.py` from their
//! definition; do not edit by hand (`sage -python tests/fixtures/gauss_tables.py --check`
//! compares this file with the definition).
//!
//! Every table is the reverse CDT of a half-Gaussian $K \in \lbrace 0, 1, 2, \dots \rbrace$ with
//! $\Pr\[K = i\] \propto e^{-(a/b) i^2}$, that is of standard deviation $\sigma$ with
//! $1/(2\sigma^2) = a/b$ (not the width $s = \sigma\sqrt{2\pi}$ of GPV and Hwang-Seo-Song). Entry
//! $j$ is $2^{128}\Pr\[K > j\]$ rounded to the nearest integer; a table keeps the entries that do
//! not round to 0. A sampler draws $V$ uniform in $\[0, 2^{128})$ and returns
//! $K = \\#\lbrace j : V < T_j \rbrace$, so that $\Pr\[K > j\] = T_j/2^{128}$ exactly. The entries are
//! non-increasing. Every entry is enclosed in 1024-bit interval arithmetic and its rounding is
//! certain (the script fails otherwise).
//!
//! Generator: a port of ntru-trapdoor's `tools/gauss_tables.py` (same author, MIT).
"""


def rust(values):
    """The text of `src/gauss_tables.rs`."""
    parts = [HEADER]
    for name, (num, den, sigma, bits, _terms) in TABLES.items():
        entries = values[name]
        doc = (f"Standard deviation {sigma}: $\\Pr\\[K = i\\] \\propto e^{{-{num} i^2/{den}}}$, entries "
               f"$\\mathrm{{round}}(2^{{{bits}}}\\Pr\\[K > j\\])$, {len(entries)} rows (the next one "
               f"rounds to 0).")
        parts.append("\n" + "".join(f"/// {line}\n" for line in textwrap.wrap(
            doc, 96, break_long_words=False, break_on_hyphens=False)))
        assert bits == 128
        parts.append(f"pub const {name}: [u128; {len(entries)}] = [\n")
        parts.extend(f"    0x{v:032x},\n" for v in entries)
        parts.append("];\n")
    return "".join(parts)


def digest(values):
    """SHAKE256, 32 bytes in hex, per table: LE64(length), then every entry as 16 little-endian
    bytes."""
    out = {}
    for name in TABLES:
        entries = values[name]
        data = len(entries).to_bytes(8, "little")
        data += b"".join(v.to_bytes(16, "little") for v in entries)
        out[name] = hashlib.shake_256(data).hexdigest(32)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help=f"compare with {RUST.name}")
    ap.add_argument("--digest", action="store_true", help="print the pinned digests")
    a = ap.parse_args()
    values = tables()
    if a.digest:
        for name, d in digest(values).items():
            print(f"{name} {len(values[name])} {d}")
        return
    text = rust(values)
    if a.check:
        same = RUST.read_bytes() == text.encode()
        print(f"src/{RUST.name}: {'identical' if same else 'DIFFERS from the computed tables'}")
        sys.exit(0 if same else 1)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
