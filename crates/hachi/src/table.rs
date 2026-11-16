//! The tables of residues that an iteration's sumcheck, its reduction, the next commitment and
//! the packing of the next message read by index ([`ResidueTable`]): slices of residues, or the
//! next witness table with one byte per entry ([`DigitTable`]).

use crate::error::{Error, Result};
use core::ops::Index;
use hachi_co_math::Zq;

/// A table of residues modulo $q$ read by index (`table[i]`), with its length.
pub trait ResidueTable: Index<usize, Output = u64> + Sync {
    /// The number of residues.
    fn len(&self) -> usize;

    /// The table as a reader that keeps it holds it.
    fn table_ref(&self) -> TableRef<'_>;

    /// Whether there are no residues.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl ResidueTable for [u64] {
    fn len(&self) -> usize {
        <[u64]>::len(self)
    }

    fn table_ref(&self) -> TableRef<'_> {
        TableRef::Residues(self)
    }
}

impl ResidueTable for Vec<u64> {
    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn table_ref(&self) -> TableRef<'_> {
        TableRef::Residues(self)
    }
}

/// A table of residues as a reader that keeps it holds it (a commitment's message, the
/// batching sumcheck's first round): residues, or a [`DigitTable`].
#[derive(Clone, Copy, Debug)]
pub enum TableRef<'a> {
    /// Residues.
    Residues(&'a [u64]),
    /// Digits as bytes.
    Digits(&'a DigitTable),
}

impl Index<usize> for TableRef<'_> {
    type Output = u64;

    #[inline(always)]
    fn index(&self, i: usize) -> &u64 {
        match self {
            Self::Residues(r) => &r[i],
            Self::Digits(t) => &t[i],
        }
    }
}

impl ResidueTable for TableRef<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Residues(r) => r.len(),
            Self::Digits(t) => ResidueTable::len(*t),
        }
    }

    fn table_ref(&self) -> TableRef<'_> {
        *self
    }
}

/// A table of digits, one signed byte per entry, read as their canonical residues: the next
/// witness table of an iteration (`relation::next_table_parts`), whose entries are all digits,
/// as its range check requires, in one eighth of the memory of residues (2 GiB to 256 MiB at
/// secure-l24-n2-a8). Entry $i$ reads as the residue of its byte, so a reader gets the residue it
/// would get from the table of residues.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DigitTable {
    values: Vec<i8>,
    /// The canonical residue of each byte value $v$, at index $v$ as `u8`.
    residues: Box<[u64; 256]>,
}

impl DigitTable {
    /// `len` zeros.
    pub fn zeros(zq: &Zq, len: usize) -> Self {
        Self {
            values: vec![0; len],
            residues: Box::new(core::array::from_fn(|v| {
                zq.from_i64(i64::from(v as u8 as i8))
            })),
        }
    }

    /// Sets entries `start .. start + residues.len()` to the centered values of the canonical
    /// `residues`.
    ///
    /// # Errors
    /// [`Error::Shape`] for a range past the end, or a residue that is not the residue of a
    /// byte (its centered value of absolute value 129 or more).
    pub fn set_residues(&mut self, zq: &Zq, start: usize, residues: &[u64]) -> Result<()> {
        let out = start
            .checked_add(residues.len())
            .and_then(|end| self.values.get_mut(start..end))
            .ok_or_else(|| Error::Shape("the entries do not fit the table".into()))?;
        for (v, x) in out.iter_mut().zip(residues) {
            *v = byte_of(zq, *x)?;
        }
        Ok(())
    }

    /// The entries as rows of `width` bytes, for writing disjoint rows in parallel.
    pub(crate) fn rows_mut(&mut self, width: usize) -> core::slice::ChunksMut<'_, i8> {
        self.values.chunks_mut(width)
    }

    /// Sets entries `start .. start + values.len()` to `values`.
    ///
    /// # Errors
    /// [`Error::Shape`] for a range past the end.
    pub fn set_values(&mut self, start: usize, values: &[i8]) -> Result<()> {
        start
            .checked_add(values.len())
            .and_then(|end| self.values.get_mut(start..end))
            .ok_or_else(|| Error::Shape("the entries do not fit the table".into()))?
            .copy_from_slice(values);
        Ok(())
    }

    /// The residues of the entries.
    pub fn to_residues(&self) -> Vec<u64> {
        self.values
            .iter()
            .map(|v| self.residues[*v as u8 as usize])
            .collect()
    }
}

/// The byte whose residue is the canonical `x`.
///
/// # Errors
/// [`Error::Shape`] unless `x` is the residue of a byte.
pub(crate) fn byte_of(zq: &Zq, x: u64) -> Result<i8> {
    i8::try_from(zq.to_centered(x))
        .ok()
        .filter(|v| zq.from_i64(i64::from(*v)) == x)
        .ok_or_else(|| Error::Shape("an entry of the table is not a digit".into()))
}

impl Index<usize> for DigitTable {
    type Output = u64;

    #[inline(always)]
    fn index(&self, i: usize) -> &u64 {
        &self.residues[self.values[i] as u8 as usize]
    }
}

impl ResidueTable for DigitTable {
    fn len(&self) -> usize {
        self.values.len()
    }

    fn table_ref(&self) -> TableRef<'_> {
        TableRef::Digits(self)
    }
}

/// The same residues, entry by entry.
impl PartialEq<Vec<u64>> for DigitTable {
    fn eq(&self, other: &Vec<u64>) -> bool {
        self.values.len() == other.len() && (0..other.len()).all(|i| self[i] == other[i])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_digit_table_reads_as_its_residues() {
        let q = 15395032342003157u64; // secure-l24
        let zq = Zq::new(q).unwrap();
        let digits: Vec<i64> = vec![0, 1, -1, 2, -2, 127, -128, 5];
        let residues: Vec<u64> = digits.iter().map(|c| zq.from_i64(*c)).collect();
        let mut table = DigitTable::zeros(&zq, 10);
        table.set_residues(&zq, 1, &residues).unwrap();
        let mut want = vec![0u64; 10];
        want[1..9].copy_from_slice(&residues);
        assert_eq!(table, want);
        assert_eq!(table.to_residues(), want);
        for (i, w) in want.iter().enumerate() {
            assert_eq!(table[i], *w, "entry {i}");
        }
        assert_eq!(ResidueTable::len(&table), 10);
        let by_ref = table.table_ref();
        assert_eq!(ResidueTable::len(&by_ref), 10);
        assert!((0..10).all(|i| by_ref[i] == want[i]));
        table.set_values(8, &[-3, 4]).unwrap();
        assert_eq!(table[8], zq.from_i64(-3));
        assert_eq!(table[9], 4);
        // Residues that are not those of a byte, and ranges past the end, are refused.
        for bad in [zq.from_i64(128), zq.from_i64(-129), q / 2, q - 1 - q / 2] {
            assert!(table.set_residues(&zq, 0, &[bad]).is_err(), "{bad}");
        }
        assert!(table.set_residues(&zq, 0, &[q]).is_err(), "not canonical");
        assert!(table.set_residues(&zq, 9, &residues[..2]).is_err());
        assert!(table.set_values(9, &[0, 0]).is_err());
        assert!(table.set_values(usize::MAX, &[0]).is_err());
    }
}
