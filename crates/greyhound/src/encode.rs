//! Encodings of residues: $\lceil \log_2 q \rceil$ bits each, least significant bit first,
//! packed without gaps, the last byte zero-padded.
use crate::error::{Error, Result};

/// Append the residues `values`, `bits` bits each, to `out`.
pub fn pack(values: impl IntoIterator<Item = u64>, bits: u32, out: &mut Vec<u8>) {
    let (mut acc, mut held) = (0u128, 0u32);
    for v in values {
        debug_assert!(bits == 64 || v >> bits == 0);
        acc |= u128::from(v) << held;
        held += bits;
        while held >= 8 {
            out.push(acc as u8);
            acc >>= 8;
            held -= 8;
        }
    }
    if held > 0 {
        out.push(acc as u8);
    }
}

/// The length of [`pack`] of `count` residues.
pub fn packed_len(count: usize, bits: u32) -> usize {
    (count * bits as usize).div_ceil(8)
}

/// `count` residues of `bits` bits from `bytes`, which must be exactly [`packed_len`] long, each
/// below $q$, with zero padding.
///
/// # Errors
/// [`Error::Encoding`] otherwise.
pub fn unpack(bytes: &[u8], bits: u32, count: usize, q: u64) -> Result<Vec<u64>> {
    if bytes.len() != packed_len(count, bits) {
        return Err(Error::Encoding);
    }
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let mut out = Vec::with_capacity(count);
    let (mut acc, mut held) = (0u128, 0u32);
    let mut input = bytes.iter();
    for _ in 0..count {
        while held < bits {
            acc |= u128::from(*input.next().ok_or(Error::Encoding)?) << held;
            held += 8;
        }
        let v = (acc as u64) & mask;
        acc >>= bits;
        held -= bits;
        if v >= q {
            return Err(Error::Encoding);
        }
        out.push(v);
    }
    if acc != 0 || input.next().is_some() {
        return Err(Error::Encoding);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_canonical() {
        let q = 93266870910952069u64;
        let bits = 57;
        let values: Vec<u64> = (0..100u64)
            .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15) % q)
            .collect();
        let mut out = Vec::new();
        pack(values.iter().copied(), bits, &mut out);
        assert_eq!(out.len(), packed_len(100, bits));
        assert_eq!(unpack(&out, bits, 100, q).unwrap(), values);
        // A residue of q or more, a nonzero padding bit, a short or a long input.
        let mut big = Vec::new();
        pack([q], bits, &mut big);
        assert!(unpack(&big, bits, 1, q).is_err());
        let mut padded = out.clone();
        *padded.last_mut().unwrap() |= 0x80;
        assert!(unpack(&padded, bits, 100, q).is_err());
        assert!(unpack(&out[..out.len() - 1], bits, 100, q).is_err());
        let mut long = out.clone();
        long.push(0);
        assert!(unpack(&long, bits, 100, q).is_err());
        // 32-bit residues take four bytes each.
        let mut four = Vec::new();
        pack([1, 2, 4294967196], 32, &mut four);
        assert_eq!(four.len(), 12);
        assert_eq!(
            unpack(&four, 32, 3, 4294967197).unwrap(),
            vec![1, 2, 4294967196]
        );
    }
}
