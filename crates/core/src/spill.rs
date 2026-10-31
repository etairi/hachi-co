//! The largest part of a party's state on disk: the blocks $\mathbf f'\_i$ of a commitment to a
//! large polynomial ([`PartyState::spill`](crate::PartyState::spill)). Only the blocks of
//! $\mathbf f'$ are kept: $\mathbf s\_i = \mathbf G\_{2^m}^{-1}(\mathbf f'\_i)$ is a function of
//! them, recomputed when needed, and a share of $2^\ell$ elements of $\mathbb F\_{q^k}$ with
//! 64-bit residues and $\delta$ digits would take $2^{\ell + 3} k \delta$ bytes in memory.
//!
//! The file holds a party's secret data in the clear, in `dir`, readable by its owner only (on
//! Unix); it is removed when the last clone of the [`SpillFile`] is dropped, not wiped. This is
//! for benchmarks of large instances, not for a deployment.

use crate::{Error, Layout, PartyId, Poly, Result};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Tells apart the files of one process.
static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
struct Inner {
    path: PathBuf,
    file: Mutex<File>,
    block_bytes: usize,
    d: usize,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // Nothing can be done about a file that cannot be removed.
        let _ = fs::remove_file(&self.path);
    }
}

/// A file of equally long blocks of ring elements, as little-endian `u64` residues; see the
/// module documentation. Block slot $j$ is the $j$-th block a party holds. Clones share the file.
#[derive(Clone, Debug)]
pub struct SpillFile(Arc<Inner>);

impl PartialEq for SpillFile {
    fn eq(&self, other: &Self) -> bool {
        self.0.path == other.0.path
    }
}

impl Eq for SpillFile {}

fn io_error(path: &Path, e: &std::io::Error) -> Error {
    Error::Spill(format!("{}: {e}", path.display()))
}

impl SpillFile {
    /// Creates an empty file for party `party` in the directory `dir`, which must exist.
    ///
    /// # Errors
    /// [`Error::Spill`] if the file cannot be created.
    pub fn create(dir: &Path, party: PartyId, layout: &Layout) -> Result<Self> {
        let path = dir.join(format!(
            "hachi-co-{}-{}-party-{party}.blocks",
            std::process::id(),
            NEXT_FILE.fetch_add(1, Ordering::Relaxed)
        ));
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let file = options.open(&path).map_err(|e| io_error(&path, &e))?;
        Ok(Self(Arc::new(Inner {
            path,
            file: Mutex::new(file),
            block_bytes: layout.block_len * layout.d * 8,
            d: layout.d,
        })))
    }

    /// The path of the file.
    pub fn path(&self) -> &Path {
        &self.0.path
    }

    /// Moves to the start of block slot `slot`.
    fn seek(&self, file: &mut File, slot: usize) -> Result<()> {
        let offset = (slot as u64)
            .checked_mul(self.0.block_bytes as u64)
            .ok_or(Error::Overflow("the offset of a spilled block"))?;
        file.seek(SeekFrom::Start(offset))
            .map(|_| ())
            .map_err(|e| io_error(&self.0.path, &e))
    }

    /// Writes block slot `slot`.
    ///
    /// # Errors
    /// [`Error::Shape`] for a block of another size; [`Error::Spill`] if the write fails.
    pub fn write_block(&self, slot: usize, block: &[Poly]) -> Result<()> {
        let inner = &self.0;
        let mut bytes = Vec::with_capacity(inner.block_bytes);
        for poly in block {
            for x in &poly.0 {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
        }
        if bytes.len() != inner.block_bytes {
            return Err(Error::Shape {
                what: "a spilled block (bytes)",
                expected: inner.block_bytes,
                actual: bytes.len(),
            });
        }
        let mut file = inner.file.lock().expect("the file's lock");
        self.seek(&mut file, slot)?;
        file.write_all(&bytes)
            .map_err(|e| io_error(&inner.path, &e))
    }

    /// Reads block slot `slot`.
    ///
    /// # Errors
    /// [`Error::Spill`] if the read fails.
    pub fn read_block(&self, slot: usize) -> Result<Vec<Poly>> {
        let inner = &self.0;
        let mut bytes = vec![0u8; inner.block_bytes];
        {
            let mut file = inner.file.lock().expect("the file's lock");
            self.seek(&mut file, slot)?;
            file.read_exact(&mut bytes)
                .map_err(|e| io_error(&inner.path, &e))?;
        }
        Ok(bytes
            .chunks_exact(8 * inner.d)
            .map(|poly| {
                Poly(
                    poly.as_chunks::<8>()
                        .0
                        .iter()
                        .map(|c| u64::from_le_bytes(*c))
                        .collect(),
                )
            })
            .collect())
    }
}
