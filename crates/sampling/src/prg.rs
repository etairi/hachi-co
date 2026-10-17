//! A SHAKE128 PRG with length-framed domain separation.
//!
//! Each stream is SHAKE128 over the framed fields
//! $$f(\mathtt{tag}) \Vert f(\mathit{seed}) \Vert f(\mathit{domain}) \Vert f(\mathrm{LE64}(\mathit{index})), \qquad f(x) = \mathrm{LE64}(|x|) \Vert x,$$
//! with `tag` = [`PRG_TAG`]. Every field carries its length, so distinct
//! `(seed, domain, index)` triples give distinct SHAKE128 inputs.

// Ported from jali (MIT, (c) 2026 Erkan Tairi): src/rand/prg.rs (`ShakePrg`) with the framing of
// src/rand/key.rs (`derive_key`).

use core::convert::Infallible;
use rand::{TryCryptoRng, TryRng};
use shake::{ExtendableOutput, Shake128, Shake128Reader, Update, XofReader};

/// Version tag, the first framed field of every PRG input.
pub const PRG_TAG: &[u8] = b"HACHI-CO-v1/prg";

/// A deterministic SHAKE128 byte stream, keyed by a 32-byte seed and separated by a domain
/// string and a 64-bit index.
///
/// The stream is SHAKE128 over
/// $f(\mathtt{tag}) \Vert f(\mathit{seed}) \Vert f(\mathit{domain}) \Vert f(\mathrm{LE64}(\mathit{index}))$
/// with $f(x) = \mathrm{LE64}(|x|) \Vert x$ and $\mathtt{tag}$ = [`PRG_TAG`] (the draft does not
/// fix a PRG; this is jali's `ShakePrg` with jali's length framing).
///
/// As a [`rand::Rng`] (through [`TryRng`] with error type [`Infallible`]) it reads the stream in
/// order: `next_u64` takes the next 8 bytes little-endian, `next_u32` the next 4 bytes
/// little-endian, and `fill_bytes` the next `dst.len()` bytes. It is marked [`TryCryptoRng`]:
/// with a uniform secret seed the stream is pseudorandom under the standard assumption that
/// SHAKE128 keyed this way behaves as a pseudorandom function (a heuristic, with at most 128-bit
/// security). It implements neither `Clone` nor `Copy`, so that a keystream is not duplicated by
/// accident, and its `Debug` output hides the state.
pub struct ShakePrg(Shake128Reader);

impl ShakePrg {
    /// Open the stream of `(seed, domain, index)`.
    ///
    /// For matrices, `index` is the entry's position (see [`crate::matrix::expand`]); for
    /// per-party or per-attempt randomness, it separates the streams under one seed.
    pub fn new(seed: &[u8; 32], domain: &[u8], index: u64) -> Self {
        let mut h = Shake128::default();
        let index = index.to_le_bytes();
        for field in [PRG_TAG, seed.as_slice(), domain, index.as_slice()] {
            h.update(&(field.len() as u64).to_le_bytes());
            h.update(field);
        }
        Self(h.finalize_xof())
    }
}

impl core::fmt::Debug for ShakePrg {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ShakePrg").finish_non_exhaustive()
    }
}

impl TryRng for ShakePrg {
    type Error = Infallible;

    /// The next 4 bytes of the stream, little-endian.
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        let mut bytes = [0u8; 4];
        self.0.read(&mut bytes);
        Ok(u32::from_le_bytes(bytes))
    }

    /// The next 8 bytes of the stream, little-endian.
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        let mut bytes = [0u8; 8];
        self.0.read(&mut bytes);
        Ok(u64::from_le_bytes(bytes))
    }

    /// The next `dst.len()` bytes of the stream.
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        self.0.read(dst);
        Ok(())
    }
}

impl TryCryptoRng for ShakePrg {}
