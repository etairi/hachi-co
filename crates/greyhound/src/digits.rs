//! Power-of-two decompositions, as the reference computes them (`polz_decompose`).
//!
//! A centred integer $x$ is written $x = \sum_{j < p - 1} d_j 2^{jd} + d_{p-1} 2^{(p-1)d}$ with
//! the low digits $d_j \in [-2^{d-1}, 2^{d-1})$, each the low $d$ bits of what remains, sign
//! extended, and the top digit what remains after them. Adding
//! $o = 2^{d-1} \sum_{j < p-1} 2^{jd}$ shifts every low digit into $[0, 2^d)$, so digit $j$ is
//! read from $x + o$ directly ([`Digits::digit`]), which [`decompose`] checks against the
//! sequential form in the tests.

/// $x$ in `parts` digits of base $2^d$ (module documentation), sequentially.
pub fn decompose(mut x: i64, d: u32, parts: usize, out: &mut [i64]) {
    debug_assert!((1..63).contains(&d) && parts >= 1 && out.len() >= parts);
    for o in &mut out[..parts - 1] {
        let digit = (x << (64 - d)) >> (64 - d);
        *o = digit;
        x = (x - digit) >> d;
    }
    out[parts - 1] = x;
}

/// The digits of a decomposition, read one at a time from $x + o$.
#[derive(Clone, Copy, Debug)]
pub struct Digits {
    d: u32,
    parts: usize,
    offset: i64,
    mask: i64,
    half: i64,
}

impl Digits {
    /// Digits of base $2^d$, `parts` of them. The offset is below $2^{(p-1)d}$, so $x + o$ stays
    /// within `i64` for $|x| < 2^{62}$ when $(p - 1) d \le 62$.
    pub fn new(d: u32, parts: usize) -> Self {
        debug_assert!((1..63).contains(&d) && parts >= 1 && (parts - 1) * d as usize <= 62);
        let half = 1i64 << (d - 1);
        let offset = (0..parts - 1).map(|j| half << (j as u32 * d)).sum();
        Self {
            d,
            parts,
            offset,
            mask: (1i64 << d) - 1,
            half,
        }
    }

    /// Digit `j` of $x$.
    #[inline(always)]
    pub fn digit(&self, x: i64, j: usize) -> i64 {
        let y = x + self.offset;
        let shifted = y >> (j as u32 * self.d);
        if j + 1 < self.parts {
            (shifted & self.mask) - self.half
        } else {
            shifted
        }
    }

    /// The largest absolute value of a digit of any $|x| \le X$: $2^{d-1}$ for the low digits,
    /// and the top digit bounded by the recurrence $X_{j+1} = \lfloor (X_j + 2^{d-1}) / 2^d
    /// \rfloor$.
    pub fn bound(&self, x_max: u64) -> u64 {
        let mut x = u128::from(x_max);
        let half = 1u128 << (self.d - 1);
        for _ in 0..self.parts - 1 {
            x = (x + half) >> self.d;
        }
        let low = if self.parts > 1 { half as u64 } else { 0 };
        low.max(x as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reconstruct(digits: &[i64], d: u32) -> i64 {
        digits.iter().rev().fold(0i64, |acc, x| (acc << d) + x)
    }

    #[test]
    fn closed_form_is_the_sequential_decomposition() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for (d, parts, range) in [
            (8u32, 4usize, 1i64 << 31),
            (11, 3, 1 << 31),
            (7, 5, 1 << 31),
            (14, 4, 1 << 56),
            (10, 6, 1 << 56),
            (7, 2, 1 << 40),
            (5, 1, 1 << 20),
        ] {
            let digits = Digits::new(d, parts);
            let mut out = vec![0i64; parts];
            for _ in 0..20000 {
                let x = (next() % (2 * range as u64)) as i64 - range;
                decompose(x, d, parts, &mut out);
                assert_eq!(reconstruct(&out, d), x);
                for (j, v) in out.iter().enumerate() {
                    assert_eq!(digits.digit(x, j), *v, "d {d} parts {parts} x {x} j {j}");
                    if j + 1 < parts {
                        assert!((-(1 << (d - 1))..(1 << (d - 1))).contains(v));
                    }
                    assert!(v.unsigned_abs() <= digits.bound(range as u64));
                }
            }
            for x in [-range, range, 0, -1, 1] {
                decompose(x, d, parts, &mut out);
                assert_eq!(reconstruct(&out, d), x);
                assert!(
                    out.iter()
                        .all(|v| v.unsigned_abs() <= digits.bound(range as u64))
                );
            }
        }
    }

    #[test]
    fn bound_of_the_reference_decompositions() {
        // q = 2^32 - 99 in 4 digits of base 2^8: centred inputs below 2^31, every digit at most
        // 128 in absolute value.
        assert_eq!(Digits::new(8, 4).bound((4294967197 - 1) / 2), 128);
        // In 3 digits of base 2^11 the top digit is at most 512 and the low digits 1024.
        assert_eq!(Digits::new(11, 3).bound((4294967197 - 1) / 2), 1024);
    }
}
