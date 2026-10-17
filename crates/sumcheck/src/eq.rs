//! Equality polynomials and multilinear extensions over $\mathbb F\_{q^k}$
//! (`4_preliminaries.tex`:53-56; `5_Building_Blocks.tex`:261; oracle `ref/hachico_ref/fqk.py`).
//!
//! **Bit order** (`ref/README.md`, "Bit strings" and "Hypercube"): a point
//! $\mathbf x \in \lbrace 0, 1 \rbrace^s$ has index $\mathrm{nat}(\mathbf x) = \sum\_j x\_j 2^{s-j}$,
//! $x\_1$ the most significant bit, so a table of $2^s$ values lists $f(\mathbf x)$ at index
//! $\mathrm{nat}(\mathbf x)$ and the variable $X\_1$ goes with the top bit. Binding $X\_1$ to $a$
//! ([`bind_top`]) is $T'\[x\] = T\[x\] + a (T\[x + h\] - T\[x\])$ with $h$ half the table.

use crate::error::{Error, Result, check_len, pow2};
use hachi_co_math::{Ext, Fqk};

/// $\widetilde{eq}(\boldsymbol\tau, \mathbf x) = \prod\_j \bigl(\tau\_j x\_j + (1 - \tau\_j)(1 - x\_j)\bigr)$
/// for every $\mathbf x \in \lbrace 0, 1 \rbrace^s$, at index $\mathrm{nat}(\mathbf x)$ with
/// $\tau\_1$ paired with the most significant bit (`5_Building_Blocks.tex`:261; the oracle's
/// `Fqk.eq_table`). $2^s$ multiplications, in place.
///
/// # Errors
/// [`Error::Parameter`] if $2^s$ does not fit in `usize`.
pub fn eq_table<const K: usize>(field: &Fqk<K>, tau: &[Ext<K>]) -> Result<Vec<Ext<K>>> {
    let size = pow2(tau.len(), "the eq table")?;
    let mut table = vec![field.zero(); size];
    table[0] = field.one();
    for (j, t) in tau.iter().enumerate() {
        // After tau_1..tau_j the table has 2^j entries; tau_{j+1} becomes the new lowest bit.
        // Going down, entry i is read before entries 2i and 2i + 1 (both >= i) are written.
        for i in (0..1usize << j).rev() {
            let v = table[i];
            let high = field.mul(&v, t);
            table[2 * i + 1] = high;
            table[2 * i] = field.sub(&v, &high);
        }
    }
    Ok(table)
}

/// $\widetilde{eq}(\mathbf x, \mathbf y) = \prod\_j \bigl(x\_j y\_j + (1 - x\_j)(1 - y\_j)\bigr)$ for two
/// points of $\mathbb F\_{q^k}^s$ (`5_Building_Blocks.tex`:261).
///
/// # Errors
/// [`Error::Shape`] unless the points have the same length.
pub fn eq_eval<const K: usize>(field: &Fqk<K>, x: &[Ext<K>], y: &[Ext<K>]) -> Result<Ext<K>> {
    check_len("eq arguments (equal lengths)", x.len(), y.len())?;
    let one = field.one();
    Ok(x.iter().zip(y).fold(one, |acc, (a, b)| {
        let both = field.mul(a, b);
        let neither = field.mul(&field.sub(&one, a), &field.sub(&one, b));
        field.mul(&acc, &field.add(&both, &neither))
    }))
}

/// The bits of `value` in `len` bits, most significant first ($\mathsf{bit}$ of the
/// `ref/README.md` convention, e.g. the $\lceil \log n \rceil$ bits of $p - 1$ in
/// $\widetilde{eq}(\tau\_0, (U, L, \mathsf{bit}(p)))$). Bits above `len` are dropped.
pub fn msb_bits(value: usize, len: usize) -> Vec<bool> {
    (0..len)
        .map(|j| {
            let shift = len - 1 - j;
            shift < usize::BITS as usize && (value >> shift) & 1 == 1
        })
        .collect()
}

/// $\widetilde{eq}(\boldsymbol\tau, \mathbf b)$ at one bit string, as a product (the oracle's
/// `Fqk.eq_point`).
///
/// # Errors
/// [`Error::Shape`] unless `bits` has one bit per coordinate of $\boldsymbol\tau$.
pub fn eq_point<const K: usize>(field: &Fqk<K>, tau: &[Ext<K>], bits: &[bool]) -> Result<Ext<K>> {
    check_len("eq bits (one per coordinate)", tau.len(), bits.len())?;
    let one = field.one();
    Ok(tau.iter().zip(bits).fold(one, |acc, (t, bit)| {
        let factor = if *bit { *t } else { field.sub(&one, t) };
        field.mul(&acc, &factor)
    }))
}

/// Binds the most significant variable of a table of $2^s$ values to $a$, in place:
/// $T'\[x\] = T\[x\] + a (T\[x + h\] - T\[x\])$ for $x \lt h = 2^{s-1}$ (the oracle's
/// `Fqk.bind_top`; `ref/README.md`, "Hypercube"). One multiplication per entry of the result.
///
/// # Errors
/// [`Error::Shape`] unless the table has an even, nonzero length.
pub fn bind_top<const K: usize>(field: &Fqk<K>, table: &mut Vec<Ext<K>>, a: &Ext<K>) -> Result<()> {
    let len = table.len();
    if len == 0 || !len.is_multiple_of(2) {
        return Err(Error::Shape {
            what: "table to bind (even, nonzero length)",
            expected: len + len % 2 + usize::from(len == 0) * 2,
            actual: len,
        });
    }
    let h = len / 2;
    let (low, high) = table.split_at_mut(h);
    for (lo, hi) in low.iter_mut().zip(high.iter()) {
        *lo = field.add(lo, &field.mul(a, &field.sub(hi, lo)));
    }
    table.truncate(h);
    Ok(())
}

/// The multilinear extension of a table of $2^s$ values at a point of $\mathbb F\_{q^k}^s$, by
/// binding $X\_1, X\_2, \dots$ in turn (the oracle's `Fqk.mle_eval`;
/// `4_preliminaries.tex`:56).
///
/// # Errors
/// [`Error::Shape`] unless the table has $2^s$ entries for $s$ = the length of the point.
pub fn mle_eval<const K: usize>(
    field: &Fqk<K>,
    table: &[Ext<K>],
    point: &[Ext<K>],
) -> Result<Ext<K>> {
    check_len(
        "MLE table (2^s entries)",
        pow2(point.len(), "the MLE table")?,
        table.len(),
    )?;
    let mut t = table.to_vec();
    for a in point {
        bind_top(field, &mut t, a)?;
    }
    Ok(t[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    const Q: u64 = 33554501; // toy-n2

    fn ext(seed: u64) -> Ext<4> {
        Ext(core::array::from_fn(|c| {
            (seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (8 * c as u64 + 3)) % Q
        }))
    }

    #[test]
    fn eq_table_matches_the_product_formula_msb_first() {
        let f = Fqk::<4>::new(Q).unwrap();
        for s in 0..=6usize {
            let tau: Vec<Ext<4>> = (0..s as u64).map(|j| ext(j + 11)).collect();
            let table = eq_table(&f, &tau).unwrap();
            assert_eq!(table.len(), 1 << s);
            for (x, v) in table.iter().enumerate() {
                let bits = msb_bits(x, s);
                assert_eq!(*v, eq_point(&f, &tau, &bits).unwrap(), "s = {s}, x = {x}");
                let point: Vec<Ext<4>> = bits.iter().map(|b| f.from_base(u64::from(*b))).collect();
                assert_eq!(*v, eq_eval(&f, &tau, &point).unwrap());
            }
            // The MLE of the eq table at y is eq(tau, y) (eq is multilinear in each argument).
            let y: Vec<Ext<4>> = (0..s as u64).map(|j| ext(j + 101)).collect();
            assert_eq!(
                mle_eval(&f, &table, &y).unwrap(),
                eq_eval(&f, &tau, &y).unwrap()
            );
        }
    }

    #[test]
    fn mle_of_a_table_agrees_with_the_lagrange_sum() {
        let f = Fqk::<4>::new(Q).unwrap();
        let s = 5;
        let table: Vec<Ext<4>> = (0..1u64 << s).map(|i| ext(i * 3 + 1)).collect();
        let point: Vec<Ext<4>> = (0..s as u64).map(|j| ext(j + 77)).collect();
        let eq = eq_table(&f, &point).unwrap();
        let direct = table
            .iter()
            .zip(&eq)
            .fold(f.zero(), |acc, (t, e)| f.add(&acc, &f.mul(t, e)));
        assert_eq!(mle_eval(&f, &table, &point).unwrap(), direct);
        // At a hypercube point the MLE is the table entry (x_1 the most significant bit).
        let x = 0b10110usize;
        let corner: Vec<Ext<4>> = msb_bits(x, s)
            .iter()
            .map(|b| f.from_base(u64::from(*b)))
            .collect();
        assert_eq!(mle_eval(&f, &table, &corner).unwrap(), table[x]);
    }

    #[test]
    fn shapes_are_checked() {
        let f = Fqk::<4>::new(Q).unwrap();
        let mut odd = vec![f.one(); 3];
        assert!(bind_top(&f, &mut odd, &f.one()).is_err());
        let mut empty: Vec<Ext<4>> = vec![];
        assert!(bind_top(&f, &mut empty, &f.one()).is_err());
        assert!(mle_eval(&f, &[f.one(); 4], &[f.one()]).is_err());
        assert!(eq_point(&f, &[f.one()], &[true, false]).is_err());
        assert!(eq_eval(&f, &[f.one()], &[]).is_err());
        assert_eq!(msb_bits(5, 4), vec![false, true, false, true]);
        assert_eq!(msb_bits(0, 0), Vec::<bool>::new());
    }
}
