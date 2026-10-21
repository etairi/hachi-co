//! Flat encodings for the collective calls of [`hachi_co_mpc::Comm`], which carry `u64` residues
//! and `i64` vectors: every block flattens what it passes to $\mathcal F\_{\mathsf{AH}}$, the
//! exchange or the gather, and checks the exact shape of what it receives.
//!
//! The order is row-major: polynomial $i$'s coefficient $j$ at $i d + j$; element $i$'s coordinate
//! $c$ at $i K + c$. Nothing here reduces modulo $q$.

use crate::data::check_len;
use crate::{Error, Result};
use hachi_co_math::{Ext, Poly};

/// The coefficients of `polys`, one polynomial after the other.
pub fn flatten_polys(polys: &[Poly]) -> Vec<u64> {
    polys.iter().flat_map(|p| p.0.iter().copied()).collect()
}

/// `count` polynomials of `len` coefficients from `flat` (the inverse of [`flatten_polys`]).
///
/// # Errors
/// [`Error::Shape`] unless `flat` has exactly `count` $\cdot$ `len` entries;
/// [`Error::Overflow`] if that product overflows.
pub fn unflatten_polys(flat: &[u64], count: usize, len: usize) -> Result<Vec<Poly>> {
    let total = count
        .checked_mul(len)
        .ok_or(Error::Overflow("polynomial count times length"))?;
    check_len("flat polynomials", total, flat.len())?;
    if len == 0 {
        return Ok(vec![Poly(Vec::new()); count]);
    }
    Ok(flat.chunks(len).map(|c| Poly(c.to_vec())).collect())
}

/// The coordinates of `exts`, one element after the other.
pub fn flatten_exts<const K: usize>(exts: &[Ext<K>]) -> Vec<u64> {
    exts.iter().flat_map(|x| x.0).collect()
}

/// `count` elements of $\mathbb F\_{q^K}$ from `flat` (the inverse of [`flatten_exts`]).
///
/// # Errors
/// [`Error::Shape`] unless `flat` has exactly `count` $\cdot K$ entries; [`Error::Overflow`] if
/// that product overflows.
pub fn unflatten_exts<const K: usize>(flat: &[u64], count: usize) -> Result<Vec<Ext<K>>> {
    let total = count
        .checked_mul(K)
        .ok_or(Error::Overflow("element count times K"))?;
    check_len("flat field elements", total, flat.len())?;
    Ok((0..count)
        .map(|i| Ext(core::array::from_fn(|c| flat[i * K + c])))
        .collect())
}

/// Word vectors of any lengths as one flat vector, each preceded by its length:
/// $\mathrm{len}(v\_1) \Vert v\_1 \Vert \mathrm{len}(v\_2) \Vert v\_2 \Vert \cdots$ (for example a
/// party's three base-PCS commitments before [`hachi_co_mpc::Comm::all_gather`]).
pub fn join_words(parts: &[Vec<u64>]) -> Vec<u64> {
    let total: usize = parts.iter().map(Vec::len).sum();
    let mut out = Vec::with_capacity(parts.len() + total);
    for p in parts {
        out.push(p.len() as u64);
        out.extend_from_slice(p);
    }
    out
}

/// `count` word vectors from `flat` (the inverse of [`join_words`]).
///
/// # Errors
/// [`Error::Shape`] if a length runs past the end of `flat`, or if words remain after the
/// `count` vectors.
pub fn split_words(flat: &[u64], count: usize) -> Result<Vec<Vec<u64>>> {
    let mut rest = flat;
    let mut out = Vec::with_capacity(count.min(flat.len()));
    for _ in 0..count {
        let Some((len, tail)) = rest.split_first() else {
            return Err(Error::Shape {
                what: "joined word vectors (a length per vector)",
                expected: count,
                actual: out.len(),
            });
        };
        let len = usize::try_from(*len)
            .ok()
            .filter(|len| *len <= tail.len())
            .ok_or(Error::Shape {
                what: "joined word vectors (a length within the words)",
                expected: tail.len(),
                actual: usize::try_from(*len).unwrap_or(usize::MAX),
            })?;
        let (part, tail) = tail.split_at(len);
        out.push(part.to_vec());
        rest = tail;
    }
    check_len("joined word vectors (no trailing words)", 0, rest.len())?;
    Ok(out)
}

/// The entries of `rows`, one row after the other (for example the $\delta 2^m$ short ring
/// elements of a response $\mathbf z\_i$ before [`hachi_co_mpc::Comm::gather`]).
pub fn flatten_rows<T: Copy>(rows: &[Vec<T>]) -> Vec<T> {
    rows.iter().flat_map(|r| r.iter().copied()).collect()
}

/// `rows` rows of `len` entries from `flat` (the inverse of [`flatten_rows`]).
///
/// # Errors
/// [`Error::Shape`] unless `flat` has exactly `rows` $\cdot$ `len` entries; [`Error::Overflow`] if
/// that product overflows.
pub fn unflatten_rows<T: Copy>(flat: &[T], rows: usize, len: usize) -> Result<Vec<Vec<T>>> {
    let total = rows
        .checked_mul(len)
        .ok_or(Error::Overflow("row count times length"))?;
    check_len("flat rows", total, flat.len())?;
    if len == 0 {
        return Ok(vec![Vec::new(); rows]);
    }
    Ok(flat.chunks(len).map(<[T]>::to_vec).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let polys = vec![Poly(vec![1, 2, 3]), Poly(vec![4, 5, 6])];
        let flat = flatten_polys(&polys);
        assert_eq!(flat, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(unflatten_polys(&flat, 2, 3).unwrap(), polys);
        let exts = vec![Ext::<2>([7, 8]), Ext([9, 10]), Ext([11, 12])];
        let flat = flatten_exts(&exts);
        assert_eq!(flat, vec![7, 8, 9, 10, 11, 12]);
        assert_eq!(unflatten_exts::<2>(&flat, 3).unwrap(), exts);
        let rows = vec![vec![-1i64, 2], vec![3, -4], vec![5, 6]];
        let flat = flatten_rows(&rows);
        assert_eq!(flat, vec![-1, 2, 3, -4, 5, 6]);
        assert_eq!(unflatten_rows(&flat, 3, 2).unwrap(), rows);
        // Degenerate shapes.
        assert_eq!(unflatten_polys(&[], 0, 16).unwrap(), Vec::<Poly>::new());
        assert_eq!(unflatten_polys(&[], 2, 0).unwrap(), vec![Poly(vec![]); 2]);
        assert_eq!(
            unflatten_rows::<i64>(&[], 3, 0).unwrap(),
            vec![Vec::<i64>::new(); 3]
        );
        assert_eq!(unflatten_exts::<4>(&[], 0).unwrap(), Vec::<Ext<4>>::new());
        // Joined word vectors, including empty ones.
        let parts = vec![vec![7u64, 8, 9], vec![], vec![u64::MAX]];
        let flat = join_words(&parts);
        assert_eq!(flat, vec![3, 7, 8, 9, 0, 1, u64::MAX]);
        assert_eq!(split_words(&flat, 3).unwrap(), parts);
        assert_eq!(split_words(&[], 0).unwrap(), Vec::<Vec<u64>>::new());
        assert_eq!(join_words(&[]), Vec::<u64>::new());
    }

    #[test]
    fn malformed_joined_words_are_errors() {
        let flat = join_words(&[vec![1, 2], vec![3]]);
        // Too few vectors, too many, a length past the end, trailing words, a huge length.
        assert!(matches!(split_words(&flat, 1), Err(Error::Shape { .. })));
        assert!(matches!(split_words(&flat, 3), Err(Error::Shape { .. })));
        assert!(matches!(
            split_words(&[5, 1, 2], 1),
            Err(Error::Shape { .. })
        ));
        assert!(matches!(
            split_words(&[1, 1, 2], 1),
            Err(Error::Shape { .. })
        ));
        assert!(matches!(
            split_words(&[u64::MAX, 1], 1),
            Err(Error::Shape { .. })
        ));
        assert!(matches!(split_words(&[], 1), Err(Error::Shape { .. })));
    }

    #[test]
    fn wrong_shapes_are_errors() {
        assert!(matches!(
            unflatten_polys(&[1, 2, 3], 2, 2),
            Err(Error::Shape {
                expected: 4,
                actual: 3,
                ..
            })
        ));
        assert!(matches!(
            unflatten_exts::<4>(&[1, 2, 3, 4, 5], 1),
            Err(Error::Shape {
                expected: 4,
                actual: 5,
                ..
            })
        ));
        assert!(matches!(
            unflatten_rows(&[1i64, 2], 1, 3),
            Err(Error::Shape {
                expected: 3,
                actual: 2,
                ..
            })
        ));
        assert_eq!(
            unflatten_polys(&[], usize::MAX, 2),
            Err(Error::Overflow("polynomial count times length"))
        );
        assert_eq!(
            unflatten_rows::<i64>(&[], usize::MAX, 2),
            Err(Error::Overflow("row count times length"))
        );
        assert_eq!(
            unflatten_exts::<4>(&[], usize::MAX),
            Err(Error::Overflow("element count times K"))
        );
    }
}
