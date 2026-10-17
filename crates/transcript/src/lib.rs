//! A framed, domain-separated Fiat–Shamir transcript for Hachi-Co.
//!
//! The transcript is a 32-byte chaining state. Every operation hashes the state together with
//! an operation name and the operation's fields, each length-framed, and replaces the state by
//! the first 32 output bytes:
//! $$\mathit{state} \leftarrow H\left(\mathit{state} \Vert f(\mathit{op}) \Vert f(x_1) \Vert \cdots \Vert f(x_k)\right), \qquad f(x) = \mathrm{LE64}(|x|) \Vert x,$$
//! where $H$ is SHAKE128 and the number $k$ of fields is fixed by the operation:
//!
//! | Method | op | fields |
//! |---|---|---|
//! | [`Transcript::new`] (from the all-zero state) | `HACHI-CO-v1` ([`TAG`]) | `domain` |
//! | [`Transcript::append_message`] | `message` | `label`, `message` |
//! | [`Transcript::append_u64`] | `message` | `label`, $\mathrm{LE64}(\mathit{value})$ |
//! | [`Transcript::append_attempt`] | `attempt` | $\mathrm{LE64}(i)$ |
//! | [`Transcript::fork`] (state of the child) | `fork` | `label` |
//! | [`Transcript::challenge_bytes`] | `challenge` | `label`, $\mathrm{LE64}(\mathit{len})$ |
//! | [`Transcript::challenge_u64_below`] | `challenge_u64_below` | `label`, $\mathrm{LE64}(n)$ |
//!
//! Because the state has a fixed length and everything after it is framed, a hash input
//! determines the state, the operation and its fields. So two different histories reach the same
//! state only through a SHAKE128 collision on 256-bit outputs (a standard random-oracle-style
//! argument, not proven here): in particular, moving bytes between a label and a message, or
//! changing any absorbed statement byte, the domain, an attempt index or a fork label, changes
//! the state and every challenge derived after it.
//!
//! Challenges ratchet: a challenge is read from the same SHAKE128 output as the next state
//! (bytes $0..32$ are the state, the challenge follows), so the state after a challenge depends on
//! it, and two challenges with the same label in a row differ.
//!
//! The draft fixes no hash or transcript format. This is jali's transcript (`src/transcript.rs`:
//! SHAKE128 chaining with framed labels) with the Hachi-Co tag, explicit operation names, an
//! attempt counter for Fiat–Shamir with aborts, forks for the per-prover sub-proofs (plan,
//! WP5), and ratcheting challenges. `tests/fixtures/gen_kat.py` reimplements it with Python's
//! `hashlib.shake_128` for the known-answer tests.

// Ported from jali (MIT, (c) 2026 Erkan Tairi): src/transcript.rs (framed labels, challenge_seed).

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use shake::{ExtendableOutput, Shake128, Shake128Reader, Update, XofReader};

/// The protocol tag, the operation name of [`Transcript::new`].
pub const TAG: &[u8] = b"HACHI-CO-v1";

/// Errors of the transcript. (Display and `std::error::Error` are implemented by hand: this crate
/// depends on `shake` only.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    /// [`Transcript::challenge_u64_below`] was called with $n = 0$: the range $[0, n)$ is empty.
    EmptyRange,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyRange => f.write_str("empty challenge range: n must be positive"),
        }
    }
}

impl std::error::Error for Error {}

/// A Fiat–Shamir transcript: a 32-byte SHAKE128 chaining state (format in the
/// [crate documentation](crate)).
///
/// It absorbs public data only (statement, public parameters, prover messages), so the state is
/// not secret and `Debug` shows it.
///
/// Prover and verifier must perform the same operations in the same order. For Fiat–Shamir with
/// aborts, clone the transcript after the statement and, for attempt $i$, call
/// [`append_attempt`](Self::append_attempt)$(i)$ on a fresh clone before absorbing that attempt's
/// first message (plan, finding F5: the hash includes the attempt index).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Transcript {
    state: [u8; 32],
}

/// Absorb `x` as $\mathrm{LE64}(|x|) \Vert x$.
fn frame(h: &mut Shake128, x: &[u8]) {
    h.update(&(x.len() as u64).to_le_bytes());
    h.update(x);
}

impl Transcript {
    /// The SHAKE128 output stream of $\mathit{state} \Vert f(\mathit{op}) \Vert f(x_1) \Vert \cdots$.
    fn xof(&self, op: &[u8], fields: &[&[u8]]) -> Shake128Reader {
        let mut h = Shake128::default();
        h.update(&self.state);
        frame(&mut h, op);
        for x in fields {
            frame(&mut h, x);
        }
        h.finalize_xof()
    }

    /// Replace the state by the first 32 bytes of [`Self::xof`].
    fn step(&mut self, op: &[u8], fields: &[&[u8]]) {
        let mut reader = self.xof(op, fields);
        reader.read(&mut self.state);
    }

    /// A transcript for the protocol instance `domain`: the operation [`TAG`] with field `domain`
    /// applied to the all-zero state. Use `domain` to separate protocols and their versions; the
    /// statement and public parameters go in with [`append_message`](Self::append_message).
    pub fn new(domain: &[u8]) -> Self {
        let mut transcript = Self { state: [0; 32] };
        transcript.step(TAG, &[domain]);
        transcript
    }

    /// Absorb a labelled message: operation `message` with fields `label` and `message`. Both are
    /// framed, so `("ab", "c")` and `("a", "bc")` give different states.
    pub fn append_message(&mut self, label: &[u8], message: &[u8]) {
        self.step(b"message", &[label, message]);
    }

    /// Absorb an integer: [`append_message`](Self::append_message)`(label, LE64(value))`.
    pub fn append_u64(&mut self, label: &[u8], value: u64) {
        self.append_message(label, &value.to_le_bytes());
    }

    /// Absorb the attempt counter of Fiat–Shamir with aborts: operation `attempt` with field
    /// $\mathrm{LE64}(i)$. It has its own operation name, so no message can imitate it.
    pub fn append_attempt(&mut self, attempt: u64) {
        self.step(b"attempt", &[&attempt.to_le_bytes()]);
    }

    /// A child transcript for a sub-proof: its state is the result of operation `fork` with field
    /// `label`. The parent is unchanged, so forks with distinct labels can proceed independently
    /// (for example one per prover), and none of them shares a state with the parent's
    /// continuation.
    pub fn fork(&self, label: &[u8]) -> Self {
        let mut child = self.clone();
        child.step(b"fork", &[label]);
        child
    }

    /// Fill `out` with challenge bytes: with $X$ the SHAKE128 output for operation `challenge`
    /// and fields `label`, $\mathrm{LE64}(|out|)$, the new state is $X[0..32]$ and
    /// `out` $= X[32..32 + |out|]$. The length is absorbed, so challenges of different lengths
    /// are unrelated rather than prefixes of each other.
    pub fn challenge_bytes(&mut self, label: &[u8], out: &mut [u8]) {
        let mut reader = self.xof(b"challenge", &[label, &(out.len() as u64).to_le_bytes()]);
        reader.read(&mut self.state);
        reader.read(out);
    }

    /// A 32-byte challenge seed: [`challenge_bytes`](Self::challenge_bytes) with a 32-byte
    /// output. Seed a PRG with it to derive structured challenges, such as the $2^r$ elements of
    /// the $\ell_1$ ball (5_Building_Blocks.tex:122-126).
    pub fn challenge_seed(&mut self, label: &[u8]) -> [u8; 32] {
        let mut seed = [0u8; 32];
        self.challenge_bytes(label, &mut seed);
        seed
    }

    /// A challenge uniform in $[0, n)$, exactly. With $X$ the SHAKE128 output for operation
    /// `challenge_u64_below` and fields `label`, $\mathrm{LE64}(n)$, the new state is $X[0..32]$;
    /// for $n = 1$ the result is $0$, and otherwise the 8-byte little-endian words of
    /// $X[32..]$ are masked to the bit length $b$ of $n - 1$ and the first value below $n$ is
    /// returned (each word is accepted with probability $n / 2^b \gt 1/2$).
    ///
    /// # Errors
    /// [`Error::EmptyRange`] if $n = 0$; the state is then left unchanged.
    pub fn challenge_u64_below(&mut self, label: &[u8], n: u64) -> Result<u64, Error> {
        if n == 0 {
            return Err(Error::EmptyRange);
        }
        let mut reader = self.xof(b"challenge_u64_below", &[label, &n.to_le_bytes()]);
        reader.read(&mut self.state);
        if n == 1 {
            return Ok(0);
        }
        let mask = u64::MAX >> (n - 1).leading_zeros();
        loop {
            let mut word = [0u8; 8];
            reader.read(&mut word);
            let x = u64::from_le_bytes(word) & mask;
            if x < n {
                return Ok(x);
            }
        }
    }

    /// The current 32-byte state, for test vectors and for checking that parties agree.
    pub fn digest(&self) -> [u8; 32] {
        self.state
    }
}
