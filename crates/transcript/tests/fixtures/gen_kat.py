#!/usr/bin/env python3
"""Known-answer tests for hachi-co-transcript, computed with Python's hashlib.shake_128 (OpenSSL)
from the format documented in the crate: every operation sets

    state <- SHAKE128(state || f(op) || f(x_1) || ... || f(x_k))[0:32],  f(x) = LE64(len(x)) || x,

starting from the all-zero state with op "HACHI-CO-v1" and field `domain`; a challenge takes the
next state and its output from one SHAKE128 stream (bytes 0..32 are the state).

Run from anywhere:  python3 crates/transcript/tests/fixtures/gen_kat.py
It rewrites kat.json next to this script; the Rust test tests/kat.rs replays it.
"""
import hashlib
import json
import platform
import ssl
from pathlib import Path

TAG = b"HACHI-CO-v1"


def frame(x: bytes) -> bytes:
    return len(x).to_bytes(8, "little") + x


def le64(x: int) -> bytes:
    return x.to_bytes(8, "little")


class Transcript:
    def __init__(self, domain: bytes = None, state: bytes = None):
        if state is not None:
            self.state = state
        else:
            self.state = bytes(32)
            self.step(TAG, [domain])

    def stream(self, op, fields, n):
        data = self.state + frame(op) + b"".join(frame(x) for x in fields)
        return hashlib.shake_128(data).digest(n)

    def step(self, op, fields):
        self.state = self.stream(op, fields, 32)

    def message(self, label, msg):
        self.step(b"message", [label, msg])

    def u64(self, label, value):
        self.message(label, le64(value))

    def attempt(self, i):
        self.step(b"attempt", [le64(i)])

    def fork(self, label):
        child = Transcript(state=self.state)
        child.step(b"fork", [label])
        return child

    def challenge_bytes(self, label, n):
        x = self.stream(b"challenge", [label, le64(n)], 32 + n)
        self.state = x[:32]
        return x[32:]

    def challenge_u64_below(self, label, n):
        """Returns (value, words read after the state)."""
        assert n >= 1
        fields = [label, le64(n)]
        x = self.stream(b"challenge_u64_below", fields, 32)
        if n == 1:
            self.state = x
            return 0, 0
        b = (n - 1).bit_length()
        k = 0
        while True:
            k += 1
            x = self.stream(b"challenge_u64_below", fields, 32 + 8 * k)
            w = int.from_bytes(x[32 + 8 * (k - 1):32 + 8 * k], "little") & ((1 << b) - 1)
            if w < n:
                self.state = x[:32]
                return w, k


def run(t: Transcript, ops):
    """Apply ops to t and return their records, each with the digest after the op."""
    records = []
    for op in ops:
        kind = op[0]
        rec = {"op": kind}
        if kind == "message":
            _, label, msg = op
            t.message(label, msg)
            rec.update(label=label.hex(), data=msg.hex())
        elif kind == "u64":
            _, label, value = op
            t.u64(label, value)
            rec.update(label=label.hex(), value=value)
        elif kind == "attempt":
            _, i = op
            t.attempt(i)
            rec.update(value=i)
        elif kind == "challenge_bytes":
            _, label, n = op
            rec.update(label=label.hex(), len=n, out=t.challenge_bytes(label, n).hex())
        elif kind == "challenge_seed":
            _, label = op
            rec.update(label=label.hex(), out=t.challenge_bytes(label, 32).hex())
        elif kind == "challenge_u64_below":
            _, label, n = op
            value, words = t.challenge_u64_below(label, n)
            rec.update(label=label.hex(), n=n, out=value, words=words)
        elif kind == "fork":
            _, label, child_ops = op
            child = t.fork(label)
            rec.update(label=label.hex(), child_digest=child.state.hex(), child_ops=run(child, child_ops))
        else:
            raise ValueError(kind)
        rec["digest"] = t.state.hex()
        records.append(rec)
    return records


def first_rejecting_label(state, n, prefix):
    """A label whose challenge_u64_below(n) rejects at least one word, found by search."""
    for k in range(1000):
        label = prefix + str(k).encode()
        t = Transcript(state=state)
        _, words = t.challenge_u64_below(label, n)
        if words > 1:
            return label
    raise RuntimeError("no rejecting label")


def main():
    cases = []

    # A protocol-shaped script: statement, an attempt, messages, every kind of challenge.
    t0 = Transcript(b"hachi-co/eval")
    t0.message(b"pp", bytes(range(200)))
    t0.message(b"statement", b"x || y")
    reject_63 = first_rejecting_label(t0.state, 2**63 + 1, b"reject-")  # searched from this state
    ops = [
        ("message", b"pp", bytes(range(200))),
        ("message", b"statement", b"x || y"),
        ("challenge_u64_below", reject_63, 2**63 + 1),
        ("attempt", 0),
        ("message", b"v", bytes([7]) * 1000),
        ("challenge_seed", b"c"),
        ("challenge_seed", b"c"),
        ("challenge_bytes", b"alpha", 0),
        ("challenge_bytes", b"alpha", 1),
        ("challenge_bytes", b"tau", 500),
        ("u64", b"round", 3),
        ("u64", b"round", 2**64 - 1),
        ("challenge_u64_below", b"one", 1),
        ("challenge_u64_below", b"three", 3),
        ("challenge_u64_below", b"max", 2**64 - 1),
        ("challenge_u64_below", b"pow2", 2**32),
        ("attempt", 2**64 - 1),
        ("challenge_seed", b""),
        ("message", b"", b""),
        ("challenge_seed", bytes(300)),
    ]
    cases.append({"name": "protocol", "domain": b"hachi-co/eval".hex(),
                  "initial_digest": Transcript(b"hachi-co/eval").state.hex(),
                  "ops": run(Transcript(b"hachi-co/eval"), ops)})

    # Framing: the label/message split matters.
    for name, (label, msg) in [("split-ab-c", (b"ab", b"c")), ("split-a-bc", (b"a", b"bc")),
                               ("split-empty-abc", (b"", b"abc")), ("split-abc-empty", (b"abc", b""))]:
        cases.append({"name": name, "domain": b"framing".hex(), "initial_digest": Transcript(b"framing").state.hex(),
                      "ops": run(Transcript(b"framing"), [("message", label, msg), ("challenge_seed", b"c")])})

    # Forks: per-prover children, a nested fork, and the parent's own continuation.
    fork_ops = [
        ("message", b"statement", b"s"),
        ("fork", b"prover-1", [("message", b"z", b"1"), ("challenge_seed", b"c"),
                               ("fork", b"inner", [("challenge_seed", b"c")])]),
        ("fork", b"prover-2", [("message", b"z", b"1"), ("challenge_seed", b"c")]),
        ("challenge_seed", b"c"),
    ]
    cases.append({"name": "forks", "domain": b"forks".hex(), "initial_digest": Transcript(b"forks").state.hex(),
                  "ops": run(Transcript(b"forks"), fork_ops)})

    # The empty domain and a rejecting small range from the fresh state.
    t = Transcript(b"")
    reject_5 = first_rejecting_label(t.state, 5, b"r")
    cases.append({"name": "empty-domain", "domain": "", "initial_digest": Transcript(b"").state.hex(),
                  "ops": run(Transcript(b""), [("challenge_u64_below", reject_5, 5), ("challenge_seed", b"x")])})

    # Rejection must actually occur in these KATs.
    rejected = [r for c in cases for r in c["ops"] if r["op"] == "challenge_u64_below" and r["words"] > 1]
    assert len(rejected) >= 2, "the KATs must cover the rejection path"

    out = {
        "generator": {"script": "crates/transcript/tests/fixtures/gen_kat.py",
                      "python": platform.python_version(), "openssl": ssl.OPENSSL_VERSION},
        "cases": cases,
    }
    path = Path(__file__).with_name("kat.json")
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
