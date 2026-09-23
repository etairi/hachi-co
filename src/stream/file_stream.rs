//! The witness stream backed by a memory-mapped file of 32-bit little-endian integers.
//!
//! File format: a flat sequence of unsigned 32-bit little-endian integers, one per
//! coefficient, in coefficient order (see [`crate::stream`] for how coefficients map to ring
//! elements); each is widened to a `u64` on reading. The file is not loaded: it is mapped with
//! `memmap2` and decoded on the fly, so a witness of $2^\ell$ coefficients costs $4 \cdot 2^\ell$
//! bytes on disk and no heap memory. This is the format written by
//! `utils::gen_file::write_random_data` (feature `gen_file`). A 32-bit integer can hold any
//! residue of the default modulus $q = 2^{32} - 99$, but the reader does not reduce modulo $q$:
//! a value in $\[q, 2^{32})$ is passed through as it is.

use memmap2::Mmap;
use std::fs::File;

use crate::stream::Stream;

/// Stream of `u64` residues read from a memory-mapped file of 32-bit little-endian integers
/// (see the module documentation), viewed from a fixed byte offset onwards.
pub struct U64FileStream {
    /// Read-only memory map of the whole file.
    mmap: Mmap,
    /// Byte position of the next element to read.
    cur_byte: usize,
    /// Byte position of the first element of the view; [`Stream::reset`] returns here.
    offset: usize
}

impl U64FileStream {
    /// Map the file at `path` and position the stream `offset` bytes into it. `offset` is in
    /// bytes, not elements, and should be a multiple of 4; the crate itself always passes 0.
    /// Panics if the file cannot be opened or mapped. The map is created with
    /// `memmap2::Mmap::map`, which is `unsafe` because the file must not be modified by another
    /// process while it is mapped.
    pub fn init(path: &str, offset: usize) -> Self {
        let file = File::open(path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };

        Self {
            mmap,
            cur_byte: offset,
            offset
        }
    }
}

/// [`Stream`] over the mapped file: elements are decoded on the fly; no copy of the file is made.
impl Stream<u64> for U64FileStream {
    /// Number of 32-bit integers in the *whole file*, `file_len / 4`. The offset is not
    /// subtracted, so with a non-zero offset this exceeds the number of elements readable from
    /// the view by `offset / 4`; commit and prove compare it with $2^\ell$ to check that the
    /// file is large enough.
    fn length(&self) -> usize {
        self.mmap.len() / 4
    }

    /// Decode the next `arr.len()` integers with `u32::from_le_bytes`, widen them to `u64` and
    /// advance by `4 * arr.len()` bytes. Panics (assertion) if the file ends before `arr.len()`
    /// elements. Instrumented with `time_graph` under feature `stats`.
    #[cfg_attr(feature = "stats", time_graph::instrument)]
    fn read(&mut self, arr: &mut [u64]) {
        assert!(self.cur_byte + arr.len() * 4 <= self.mmap.len());

        for i in 0..arr.len() {
            let b0 = self.mmap[self.cur_byte];
            let b1 = self.mmap[self.cur_byte + 1];
            let b2 = self.mmap[self.cur_byte + 2];
            let b3 = self.mmap[self.cur_byte + 3];

            arr[i] = u32::from_le_bytes([b0, b1, b2, b3]) as u64;
            self.cur_byte += 4;
        }
    }

    /// Rewind to the byte offset given at construction.
    fn reset(&mut self) {
        self.cur_byte = self.offset;
    }
}
