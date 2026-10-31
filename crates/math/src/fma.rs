//! Negacyclic transforms on `f64` with fused multiply-add, for the exact products of
//! [`crate::rns`].
//!
//! **Arithmetic.** Residues modulo a prime $p \lt 2^{50}$ are integers held in doubles, which
//! represent every integer of absolute value up to $2^{53}$. For a constant $w$, $|w| \le p/2$,
//! with $\hat w = \mathrm{RN}(w/p)$ and $|a| \le 4p$ an integer, let $h = \mathrm{RN}(aw)$; a fused
//! operation gives the exact $l = aw - h$, $|l| \le 2^{-53} \cdot 2p^2 \lt 2^{48}$. Let $c$ be
//! $\mathrm{RN}(a \hat w)$ rounded to an integer: the two rounding errors of $a \hat w$ add up to
//! less than $2^{-52} (1 + 2^{-52}) |aw/p| \lt 1/2$, as $|aw/p| \le 2p$, so $|aw/p - c| \lt 1$.
//! Both $h$ and $cp$ are integers, and $h - cp$ is below $2^{51}$ in absolute value, so the fused
//! operation gives it exactly. Then (`mul_constant`) $(h - cp) + l = aw - cp$ is exact too,
//! congruent to $aw$, and below $p$ in absolute value. `reduce` takes an integer $|a| \le 8p$ to
//! $|a - cp| \le 0.51 p$ in the same way, the error of $a/p$ being about $2^{-49}$. `mul_residues`
//! multiplies residues $|a|, |b| \le 0.51 p$ in the same way, with $c$ from $h$: three relative
//! errors of $2^{-53}$ on $|ab/p| \le 0.27 p$ come to less than $0.11$, so the result is at most
//! $0.61 p$ in absolute value.
//!
//! **Transforms.** The forward transform is Cooley–Tukey with the twiddles
//! $\psi^{\mathrm{rev}(i)}$ merged into the butterflies (no twist), output in bit-reversed order,
//! and the inverse is its Gentleman–Sande mirror, so neither permutes. Entry $k$ of the output is
//! $a(\psi^{2 \mathrm{rev}(k) + 1})$. From inputs below $p$ in absolute value, values grow by at
//! most $p$ per stage forward, and at most double backward, so `reduce` every two stages keeps
//! every value below $4p$.
//!
//! **Lanes.** A forward transform takes up to [`LANES`] polynomials side by side: with $L$ of them,
//! coefficient $j$ of polynomial $l$ is at index $jL + l$. A butterfly is then a plain loop over
//! $L$ doubles or more, so every stage vectorizes the same way, without shuffles, in safe code.
//!
//! **Instructions.** [`f64::mul_add`] and [`f64::round_ties_even`] are exact on every target, so
//! nothing above depends on vectorization or on the machine: the instructions only decide the
//! speed. Where a build has no fused instructions, which is x86-64 without `target-cpu`, each one
//! is a library call; `.cargo/config.toml` builds for the CPU of the building machine, and
//! [`vector_instructions`] tells what a build has.

use crate::{
    arith::{Reducer, pow_mod},
    error::{Error, Result},
    ntt::root_of_unity,
};

/// The primes of the transforms, in descending order, each with a primitive $2^{13}$-th root of
/// unity, as [`crate::ntt::NTT_PRIMES`]: the three largest primes below $2^{50}$ with
/// $p \equiv 1 \pmod{2^{13}}$, from `crates/math/tools/ntt_primes.sage` (SageMath 10.9). Their
/// product exceeds $2^{149.99}$, so three suffice for every $q \lt 2^{63}$ and degree at most
/// 4096 ([`crate::rns`]).
pub const FMA_PRIMES: [(u64, u64); 3] = [
    (1125899906826241, 990116967873799),
    (1125899906629633, 702736788773885),
    (1125899906424833, 250104509277401),
];

/// The most polynomials of one forward transform: enough that the shortest loop of a transform
/// fills a vector of 512 bits, and the loops of narrower vectors run several times.
pub const LANES: usize = 8;

/// The vector instructions this build was compiled for, to record with measurements.
pub fn vector_instructions() -> &'static str {
    if cfg!(target_feature = "avx512f") {
        "avx512"
    } else if cfg!(all(target_feature = "avx2", target_feature = "fma")) {
        "avx2+fma"
    } else if cfg!(target_feature = "neon") {
        "neon"
    } else {
        "none"
    }
}

/// $a$ modulo $p$, of absolute value at most $0.51 p$, for an integer $|a| \le 8p$.
#[inline(always)]
fn reduce(a: f64, p: f64, p_inverse: f64) -> f64 {
    (-(a * p_inverse).round_ties_even()).mul_add(p, a)
}

/// $aw$ modulo $p$, below $p$ in absolute value, for integers $|a| \le 4p$ and $|w| \le p/2$ and
/// `ratio` $= \mathrm{RN}(w/p)$.
#[inline(always)]
fn mul_constant(a: f64, w: f64, ratio: f64, p: f64) -> f64 {
    let high = a * w;
    let low = a.mul_add(w, -high);
    (-(a * ratio).round_ties_even()).mul_add(p, high) + low
}

/// $ab$ modulo $p$, of absolute value at most $0.61 p$, for integers $|a|, |b| \le 0.51 p$.
#[inline(always)]
fn mul_residues(a: f64, b: f64, p: f64, p_inverse: f64) -> f64 {
    let high = a * b;
    let low = a.mul_add(b, -high);
    (-(high * p_inverse).round_ties_even()).mul_add(p, high) + low
}

/// The transforms of one prime $p \lt 2^{50}$ and one size $d$.
#[derive(Clone, Debug)]
pub(crate) struct FmaPlan {
    prime: u64,
    d: usize,
    /// $p$ and $1/p$.
    p: f64,
    p_inverse: f64,
    /// $\psi^{\mathrm{rev}(i)}$ at index $i$, centered, with its ratio to $p$.
    forward_twiddles: Vec<(f64, f64)>,
    /// $\psi^{-\mathrm{rev}(i)}$ at index $i$, centered, with its ratio to $p$.
    inverse_twiddles: Vec<(f64, f64)>,
    /// $d^{-1}$, centered, with its ratio to $p$.
    scale: (f64, f64),
    /// Reduction of integers modulo $p$.
    reducer: Reducer,
}

impl FmaPlan {
    /// The plan of prime `index` of [`FMA_PRIMES`] and a power of two $2 \le d \le$
    /// [`crate::ntt::MAX_DEGREE`].
    ///
    /// # Errors
    /// [`Error::Parameter`] for an index outside the table.
    pub(crate) fn new(index: usize, d: usize) -> Result<Self> {
        let (prime, root) = *FMA_PRIMES
            .get(index)
            .ok_or(Error::Parameter("FMA prime index"))?;
        let psi = root_of_unity(prime, root, d);
        let psi_inverse = pow_mod(psi, prime - 2, prime);
        let p = prime as f64;
        let centered = |w: u64| {
            let w = if w > prime / 2 {
                w as i64 - prime as i64
            } else {
                w as i64
            };
            (w as f64, w as f64 / p)
        };
        let bits = d.trailing_zeros();
        let twiddles = |base: u64| -> Vec<(f64, f64)> {
            (0..d)
                .map(|i| {
                    let reversed = (i as u32).reverse_bits() >> (32 - bits);
                    centered(pow_mod(base, u64::from(reversed), prime))
                })
                .collect()
        };
        Ok(Self {
            prime,
            d,
            p,
            p_inverse: 1.0 / p,
            forward_twiddles: twiddles(psi),
            inverse_twiddles: twiddles(psi_inverse),
            scale: centered(pow_mod(d as u64, prime - 2, prime)),
            reducer: Reducer::new(prime),
        })
    }

    /// The prime $p$.
    pub(crate) fn modulus(&self) -> u64 {
        self.prime
    }

    /// The double of absolute value below $p$ that is congruent to the integer $c$.
    #[inline(always)]
    pub(crate) fn lift(&self, c: i64) -> f64 {
        let r = self.reducer.reduce_below(u128::from(c.unsigned_abs())) as f64;
        core::hint::select_unpredictable(c < 0, -r, r)
    }

    /// `a` as residues in $[0, p)$, for doubles of absolute value below $p$.
    pub(crate) fn residues(&self, a: &[f64]) -> Vec<u64> {
        a.iter()
            .map(|x| {
                let r = *x as i64;
                let shifted = r.wrapping_add(self.prime as i64);
                core::hint::select_unpredictable(r < 0, shifted, r) as u64
            })
            .collect()
    }

    /// The forward transform of the `lanes` polynomials of `a`, side by side (module
    /// documentation), in place, for doubles of absolute value below $p$; the results have
    /// absolute value at most $0.51 p$.
    ///
    /// # Panics
    /// If `a` does not have $d$ doubles per lane.
    pub(crate) fn forward(&self, a: &mut [f64], lanes: usize) {
        assert_eq!(a.len(), self.d * lanes);
        let p = self.p;
        let (mut half, mut blocks, mut stage) = (self.d, 1, 0);
        while blocks < self.d {
            half /= 2;
            let twiddles = &self.forward_twiddles[blocks..2 * blocks];
            for (block, (w, ratio)) in a.chunks_exact_mut(2 * half * lanes).zip(twiddles) {
                let (low, high) = block.split_at_mut(half * lanes);
                for (x, y) in low.iter_mut().zip(high) {
                    let (u, v) = (*x, mul_constant(*y, *w, *ratio, p));
                    *x = u + v;
                    *y = u - v;
                }
            }
            stage += 1;
            if stage % 2 == 0 {
                self.reduce(a);
            }
            blocks *= 2;
        }
        if stage % 2 != 0 {
            self.reduce(a);
        }
    }

    /// The inverse of [`FmaPlan::forward`] of one polynomial, in place, for $d$ integers of
    /// absolute value at most $8p$; the results have absolute value at most $0.51 p$.
    ///
    /// # Panics
    /// If `a` does not have $d$ doubles.
    pub(crate) fn inverse(&self, a: &mut [f64]) {
        assert_eq!(a.len(), self.d);
        let (p, p_inverse) = (self.p, self.p_inverse);
        self.reduce(a);
        let (mut half, mut blocks, mut stage) = (1, self.d / 2, 0);
        while blocks >= 1 {
            let twiddles = &self.inverse_twiddles[blocks..2 * blocks];
            for (block, (w, ratio)) in a.chunks_exact_mut(2 * half).zip(twiddles) {
                let (low, high) = block.split_at_mut(half);
                for (x, y) in low.iter_mut().zip(high) {
                    let (u, v) = (*x, *y);
                    *x = u + v;
                    *y = mul_constant(u - v, *w, *ratio, p);
                }
            }
            stage += 1;
            if stage % 2 == 0 {
                self.reduce(a);
            }
            half *= 2;
            blocks /= 2;
        }
        let (w, ratio) = self.scale;
        for x in a.iter_mut() {
            *x = reduce(mul_constant(*x, w, ratio, p), p, p_inverse);
        }
    }

    /// `acc` $\gets$ `acc` $+ xy$ coefficientwise, for $x$ and $y$ from [`FmaPlan::forward`].
    pub(crate) fn mac(&self, acc: &mut [f64], x: &[f64], y: &[f64]) {
        let (p, p_inverse) = (self.p, self.p_inverse);
        for ((acc, x), y) in acc.iter_mut().zip(x).zip(y) {
            *acc += mul_residues(*x, *y, p, p_inverse);
        }
    }

    /// [`reduce`] of every entry of `a`, integers of absolute value at most $8p$.
    pub(crate) fn reduce(&self, a: &mut [f64]) {
        let (p, p_inverse) = (self.p, self.p_inverse);
        for x in a {
            *x = reduce(*x, p, p_inverse);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::{is_prime, mul_mod, tests::SplitMix};

    /// The negacyclic product over $\mathbb Z_p$, from its definition.
    fn schoolbook(a: &[u64], b: &[u64], p: u64) -> Vec<u64> {
        let d = a.len();
        let mut out = vec![0u64; d];
        for (i, x) in a.iter().enumerate() {
            for (j, y) in b.iter().enumerate() {
                let term = mul_mod(*x, *y, p);
                let k = (i + j) % d;
                out[k] = if i + j < d {
                    (out[k] + term) % p
                } else {
                    (out[k] + p - term) % p
                };
            }
        }
        out
    }

    /// Doubles of absolute value below $p$ congruent to `a`, centered or not.
    fn lifted(plan: &FmaPlan, a: &[u64], centered: bool) -> Vec<f64> {
        a.iter()
            .map(|x| {
                if centered && *x > plan.prime / 2 {
                    plan.lift(*x as i64 - plan.prime as i64)
                } else {
                    plan.lift(*x as i64)
                }
            })
            .collect()
    }

    /// Worst cases and random polynomials of degree `d` modulo `p`.
    fn operands(rng: &mut SplitMix, p: u64, d: usize) -> Vec<Vec<u64>> {
        let random = |rng: &mut SplitMix| (0..d).map(|_| rng.next() % p).collect::<Vec<_>>();
        vec![
            vec![p - 1; d],
            vec![p / 2; d],
            vec![p / 2 + 1; d],
            vec![1; d],
            vec![0; d],
            (0..d)
                .map(|j| if j % 2 == 0 { p - 1 } else { p / 2 })
                .collect(),
            random(rng),
            random(rng),
            random(rng),
        ]
    }

    /// One transform of `polys` side by side, back as one list per polynomial.
    fn transformed(plan: &FmaPlan, polys: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let lanes = polys.len();
        let mut buffer = vec![0f64; plan.d * lanes];
        for (lane, poly) in polys.iter().enumerate() {
            for (j, x) in poly.iter().enumerate() {
                buffer[j * lanes + lane] = *x;
            }
        }
        plan.forward(&mut buffer, lanes);
        (0..lanes)
            .map(|lane| (0..plan.d).map(|j| buffer[j * lanes + lane]).collect())
            .collect()
    }

    /// An integer in $[-\mathrm{bound}, \mathrm{bound}]$.
    fn in_range(rng: &mut SplitMix, bound: i128) -> i128 {
        i128::from(rng.next()) % (2 * bound + 1) - bound
    }

    /// The table against its definition.
    #[test]
    fn the_table_holds_primes_with_roots_of_order_two_to_the_thirteen() {
        for (p, root) in FMA_PRIMES {
            assert!(is_prime(p) && p < 1 << 50 && (p - 1).is_multiple_of(1 << 13));
            assert_eq!(pow_mod(root, 1 << 12, p), p - 1);
        }
        let product: f64 = FMA_PRIMES.iter().map(|(p, _)| (*p as f64).log2()).sum();
        assert!(product > 149.99, "{product}");
    }

    /// The three operations at both ends of their ranges and on random operands, for every prime,
    /// against exact integers: the results are integers, congruent and within the stated bounds.
    #[test]
    fn the_operations_are_exact_at_the_ends_of_their_ranges() {
        let mut rng = SplitMix(34);
        for (prime, _) in FMA_PRIMES {
            let pi = i128::from(prime);
            let p = prime as f64;
            let p_inverse = 1.0 / p;
            let integer = |r: f64| {
                assert_eq!(r.fract(), 0.0);
                r as i128
            };
            let ends = |bound: i128| -> Vec<i128> {
                let near = [0, 1, 2, pi / 2, pi / 2 + 1, pi - 1, pi, pi + 1];
                let far = [bound, bound - 1, bound - 2, bound - pi / 2, bound - pi];
                let some = near.iter().chain(&far).filter(|x| **x <= bound);
                some.flat_map(|x| [*x, -*x]).collect()
            };
            // mul_constant: |a| <= 4p and |w| <= p/2.
            let mut a_values = ends(4 * pi);
            let mut w_values = ends(pi / 2);
            a_values.extend((0..64).map(|_| in_range(&mut rng, 4 * pi)));
            w_values.extend((0..64).map(|_| in_range(&mut rng, pi / 2)));
            for a in &a_values {
                for w in &w_values {
                    let (x, y) = (*a as f64, *w as f64);
                    let r = integer(mul_constant(x, y, y / p, p));
                    assert!(r.abs() < pi, "a {a} w {w}: {r}");
                    assert_eq!((a * w - r).rem_euclid(pi), 0, "a {a} w {w}");
                }
            }
            // reduce: |a| <= 8p.
            let mut values = ends(8 * pi);
            values.extend((0..4096).map(|_| in_range(&mut rng, 8 * pi)));
            for a in &values {
                let r = integer(reduce(*a as f64, p, p_inverse));
                assert!(r.abs() as f64 <= 0.51 * p, "a {a}: {r}");
                assert_eq!((a - r).rem_euclid(pi), 0, "a {a}");
            }
            // mul_residues: |a|, |b| <= 0.51 p.
            let bound = (0.51 * p) as i128;
            let mut values = ends(bound);
            values.extend((0..128).map(|_| in_range(&mut rng, bound)));
            for a in &values {
                for b in &values {
                    let r = integer(mul_residues(*a as f64, *b as f64, p, p_inverse));
                    assert!(r.abs() as f64 <= 0.61 * p, "a {a} b {b}: {r}");
                    assert_eq!((a * b - r).rem_euclid(pi), 0, "a {a} b {b}");
                }
            }
        }
    }

    /// Transforms of one to [`LANES`] polynomials, for every prime and size up to 256, on
    /// worst-case and random polynomials in centered and non-centered lifts: the bound on the
    /// values, the round trip, and the product against the definition.
    #[test]
    fn products_equal_the_schoolbook_product_for_every_number_of_lanes() {
        let mut rng = SplitMix(31);
        for (index, (p, _)) in FMA_PRIMES.into_iter().enumerate() {
            for log_d in 1..=8 {
                let d = 1usize << log_d;
                let plan = FmaPlan::new(index, d).unwrap();
                let ops = operands(&mut rng, p, d);
                for lanes in [1, 2, 3, LANES] {
                    for offset in 0..ops.len() {
                        let chosen: Vec<&Vec<u64>> =
                            (0..lanes).map(|l| &ops[(offset + l) % ops.len()]).collect();
                        for centered in [false, true] {
                            let polys: Vec<Vec<f64>> =
                                chosen.iter().map(|x| lifted(&plan, x, centered)).collect();
                            let forms = transformed(&plan, &polys);
                            for (form, poly) in forms.iter().zip(&chosen) {
                                for value in form {
                                    assert!(value.abs() <= 0.51 * plan.p, "{value}");
                                    assert_eq!(value.fract(), 0.0);
                                }
                                let mut back = form.clone();
                                plan.inverse(&mut back);
                                assert_eq!(plan.residues(&back), **poly, "lanes {lanes}");
                            }
                            let last = lanes - 1;
                            let mut acc = vec![0f64; d];
                            plan.mac(&mut acc, &forms[0], &forms[last]);
                            plan.inverse(&mut acc);
                            assert_eq!(
                                plan.residues(&acc),
                                schoolbook(chosen[0], chosen[last], p),
                                "p {p} d {d} lanes {lanes}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The largest sizes: side by side as alone, and back.
    #[test]
    fn the_lanes_agree_at_the_largest_sizes() {
        let mut rng = SplitMix(32);
        for d in [512usize, 1024, 2048, 4096] {
            let plan = FmaPlan::new(0, d).unwrap();
            let polys = operands(&mut rng, plan.prime, d);
            let inputs: Vec<Vec<f64>> = polys.iter().map(|x| lifted(&plan, x, true)).collect();
            for chunk in inputs.chunks(LANES) {
                let together = transformed(&plan, chunk);
                for (input, form) in chunk.iter().zip(&together) {
                    let alone = transformed(&plan, core::slice::from_ref(input));
                    assert_eq!(alone[0], *form, "d {d}");
                }
            }
            for (poly, input) in polys.iter().zip(&inputs) {
                let mut back = transformed(&plan, core::slice::from_ref(input)).remove(0);
                plan.inverse(&mut back);
                assert_eq!(plan.residues(&back), *poly, "d {d}");
            }
        }
    }

    /// The sums of the plans: eight products of extreme residues, then a reduction, for as many
    /// products as a piece of a sum holds.
    #[test]
    fn lazy_sums_of_extreme_products_stay_exact() {
        let d = 64;
        let plan = FmaPlan::new(0, d).unwrap();
        let p = plan.prime;
        let x = vec![p / 2; d];
        let y = vec![p / 2 + 1; d];
        let polys = [lifted(&plan, &x, true), lifted(&plan, &y, true)];
        let forms = transformed(&plan, &polys);
        let mut acc = vec![0f64; d];
        let count = 128u64;
        for i in 1..=count {
            plan.mac(&mut acc, &forms[0], &forms[1]);
            assert!(acc.iter().all(|v| v.abs() <= 8.0 * plan.p));
            if i.is_multiple_of(8) {
                plan.reduce(&mut acc);
            }
        }
        plan.inverse(&mut acc);
        let one = schoolbook(&x, &y, p);
        let want: Vec<u64> = one.iter().map(|c| mul_mod(*c, count, p)).collect();
        assert_eq!(plan.residues(&acc), want);
    }

    /// The output order against the integer transform of [`crate::ntt`], at sizes up to the
    /// largest and at every prime: entry $k$ here is $a(\psi^{2 \mathrm{rev}(k) + 1})$, and entry
    /// $j$ there is $a(\psi^{2j + 1})$.
    #[test]
    fn the_transform_equals_the_integer_transform() {
        let mut rng = SplitMix(33);
        for (index, (p, root)) in FMA_PRIMES.iter().enumerate() {
            for d in [2usize, 16, 256, 4096] {
                let plan = FmaPlan::new(index, d).unwrap();
                let psi = root_of_unity(*p, *root, d);
                let integer = crate::ntt::NttPlan::new(*p, d, psi).unwrap();
                let bits = d.trailing_zeros();
                for poly in operands(&mut rng, *p, d) {
                    let mut want = poly.clone();
                    integer.forward(&mut want).unwrap();
                    let input = lifted(&plan, &poly, false);
                    let got = transformed(&plan, core::slice::from_ref(&input));
                    for (k, value) in plan.residues(&got[0]).iter().enumerate() {
                        let j = (k as u32).reverse_bits() >> (32 - bits);
                        assert_eq!(*value, want[j as usize], "p {p} d {d}");
                    }
                }
            }
        }
    }
}
