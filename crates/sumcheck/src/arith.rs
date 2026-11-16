//! The arithmetic of the provers' inner loops, [`Arith`] and its Montgomery form
//! [`Montgomery`], from `hachi_co_math::montgomery`, which Hachi's prover shares.

pub(crate) use hachi_co_math::montgomery::{Arith, Montgomery};

#[cfg(test)]
mod tests {
    use super::*;
    use hachi_co_math::{Ext, Fqk};

    /// Item 139's measurement: the product and Hachi's range polynomial at b = 4
    /// ((y^2 - 1) y (y + 2), four products), canonical (`Fqk::mul`) against Montgomery, at the
    /// modulus of secure-l24 (and of secure-l16), one thread, 2^20 values, best of 7. Run with
    /// `cargo test --release -p hachi-co-sumcheck --lib timings_products -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing"]
    fn timings_products() {
        use std::time::Instant;
        for q in [15395032342003157u64, 34801213478789] {
            let f = Fqk::<4>::new(q).unwrap();
            let mont = Montgomery::<4>::new(q).unwrap();
            let mut state = 0x1234_5678_9abc_def1u64;
            let mut next = || {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state % q
            };
            let n = 1usize << 20;
            let xs: Vec<Ext<4>> = (0..n)
                .map(|_| Ext(core::array::from_fn(|_| next())))
                .collect();
            let ys: Vec<Ext<4>> = (0..n)
                .map(|_| Ext(core::array::from_fn(|_| next())))
                .collect();
            let (one, two) = (1u64, 2u64);
            let best = |run: &dyn Fn() -> Ext<4>| {
                let mut best = f64::MAX;
                let mut keep = Ext([0; 4]);
                for _ in 0..7 {
                    let t = Instant::now();
                    keep = run();
                    best = best.min(t.elapsed().as_secs_f64());
                }
                (best * 1e9 / n as f64, keep)
            };
            let (canon_mul, a) = best(&|| {
                xs.iter()
                    .zip(&ys)
                    .fold(Ext([0; 4]), |acc, (x, y)| f.add(&acc, &f.mul(x, y)))
            });
            let (xm, ym): (Vec<Ext<4>>, Vec<Ext<4>>) = (
                xs.iter().map(|x| mont.enter(x)).collect(),
                ys.iter().map(|y| mont.enter(y)).collect(),
            );
            let (mont_mul, b) = best(&|| {
                xm.iter()
                    .zip(&ym)
                    .fold(Ext([0; 4]), |acc, (x, y)| mont.add(&acc, &mont.mul(x, y)))
            });
            assert_eq!(mont.leave(&b), a, "the same sum");
            let range_canon = |y: &Ext<4>| {
                let y2 = f.mul(y, y);
                let p = f.mul(&crate::tables::sub_base(&f, &y2, one), y);
                f.mul(&p, &f.add(y, &f.from_base(two)))
            };
            let (canon_range, c) = best(&|| {
                xs.iter()
                    .fold(Ext([0; 4]), |acc, y| f.add(&acc, &range_canon(y)))
            });
            let (one_m, two_m) = (mont.enter_base(one), mont.enter_base(two));
            let range_mont = |y: &Ext<4>| {
                let y2 = mont.mul(y, y);
                let p = mont.mul(&mont.sub_base(&y2, one_m), y);
                // y + 2 as y - (-2), both in the working representation.
                mont.mul(&p, &mont.sub_base(y, (q - two_m) % q))
            };
            let (mont_range, d) = best(&|| {
                xm.iter()
                    .fold(Ext([0; 4]), |acc, y| mont.add(&acc, &range_mont(y)))
            });
            assert_eq!(mont.leave(&d), c, "the same range sums");
            println!(
                "q = {q}: product {canon_mul:.1} ns canonical, {mont_mul:.1} ns Montgomery ({:.2}x); \
                 range polynomial {canon_range:.1} ns canonical, {mont_range:.1} ns Montgomery ({:.2}x)",
                canon_mul / mont_mul,
                canon_range / mont_range
            );
        }
    }
}
