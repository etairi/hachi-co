//! The Fiat-Shamir transcript that makes the protocol non-interactive: every verifier challenge
//! of the paper's Fig. 7 (the sparse challenges $c_i$, the ring-switching point $\alpha$, the
//! sumcheck points $\tau_0, \tau_1$ and the per-round challenges $a_i$) is derived by hashing
//! the prover messages sent so far.
//!
//! [`FS`] is an append-only byte string. Messages are appended by [`FS::push`] through the
//! [`Serialise`] trait, and [`FS::get_seed`] hashes the *whole* transcript so far into 32
//! bytes, which seed either the sparse-challenge sampler
//! ([`crate::arithmetic::poly_chal::PChal::rand_vec`]) or a `ChaCha12Rng` from which field
//! elements are drawn with [`crate::arithmetic::utils::rand_field`]. Every challenge therefore
//! re-hashes the complete prefix; there is no incremental hash state.
//!
//! Prover ([`crate::hachi::prove`]) and verifier ([`crate::hachi::verify`]) push the same
//! sequence: the commitment $\mathbf{u}$; then $Y$ and $\mathbf{v}$ (the seed for the $c_i$);
//! then $\mathbf{u}^{\prime}$ (one seed and one RNG for $\alpha$, $\tau_0$ and $\tau_1$, in this
//! order); then, per sumcheck round, the round messages of $F_{\alpha,\tau_1}$ and
//! $F_{0,\tau_0}$ (the seed for that round's shared challenge). The transcript carries no
//! length prefixes or domain separators: it is the plain concatenation of the serialisations,
//! which is unambiguous only because every message has a length fixed by the public
//! parameters. The public parameters, the evaluation point $x$ and the claimed value $y$ are
//! not absorbed.

use sha256::digest;
use base64::{engine::general_purpose::STANDARD_NO_PAD, Engine as _};

/// A message that can be absorbed into the transcript.
pub trait Serialise {
    /// The canonical byte encoding of the message. Implemented by
    /// [`crate::arithmetic::poly_vec::PVec`] (each `u64` coefficient as 8 big-endian bytes,
    /// element after element) and by [`crate::arithmetic::sumcheck::Univariate`] (each
    /// $\mathbb F_{q^4}$ evaluation as four 8-byte little-endian residues).
    fn serialise(&self) -> Vec<u8>;
}

/// Fiat-Shamir transcript: the concatenated serialisations of all messages pushed so far.
pub struct FS {
    /// Transcript bytes, in push order.
    bytes: Vec<u8>
}

impl FS {
    /// An empty transcript.
    pub fn init() -> Self {
        Self { bytes: Vec::new() }
    }

    /// Append the serialisation of `a` to the transcript.
    pub fn push(&mut self, a: &impl Serialise) {
        self.bytes.append(&mut a.serialise());
    }

    /// Derive a 32-byte seed from the whole transcript so far: the bytes are base64-encoded
    /// (standard alphabet, no padding), the ASCII encoding is hashed with SHA-256, and the hex
    /// digest is decoded back into bytes, i.e.
    /// $$\mathrm{seed} = \mathrm{SHA256}(\mathrm{base64}(\mathrm{bytes})).$$
    /// The transcript is not modified, so two consecutive calls return the same seed; a fresh
    /// challenge requires a [`push`](FS::push) in between. The cost is linear in the transcript
    /// length. The base64 step is an injective re-encoding with no cryptographic role, but it is
    /// part of the definition of the challenges: changing it changes every derived challenge.
    pub fn get_seed(&self) -> [u8; 32] {
        // Base 64 encode then hash
        let b64 = STANDARD_NO_PAD.encode(&self.bytes);
        let digest = digest(b64);
        let mut seed = [0u8; 32];

        for i in 0..32 {
            seed[i] = u8::from_str_radix(&digest[2*i..2*(i+1)], 16).unwrap();
        }

        seed
    }
}

