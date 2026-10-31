//! Exact products in $\mathbb Z_q\[X\]/(X^n + 1)$ for any modulus $2 \le q < 2^{63}$, through a
//! residue number system of NTT primes and a CRT straight to $\bmod q$.
//!
//! Ported from jali (`src/math/rns.rs` and `src/math/ring.rs`, MIT, by the same author): the
//! prime-count rule, the centered lift and the accumulation of products in the NTT domain before
//! one CRT. The transforms are those of [`crate::fma`].
//!
//! **Exactness.** Each coefficient is lifted to its centered representative, of absolute value at
//! most $(q-1)/2$, so one negacyclic product over $\mathbb Z$ has coefficients of absolute value at
//! most $n (q-1)^2/4$, and a sum of at most [`NADDS`] products at most $\mathrm{NADDS} \cdot n
//! (q-1)^2/4$. The plan takes the least number $K$ of primes from [`FMA_PRIMES`] (in table
//! order) with $P = \prod_{j<K} p_j > (q-1)^2 \cdot n \cdot \mathrm{NADDS}$ (jali's rule; twice
//! what is needed), so every such sum lies in $(-P/2, P/2)$, its residues modulo the $p_j$ are
//! computed exactly by the transforms, and [`SmallCrt`] recovers it modulo $q$ without error.
//! Longer sums are cut into pieces of at most [`NADDS`] products, and the pieces are added modulo
//! $q$.
//!
//! **Short vectors.** A coefficient of $x$ of absolute value at most $q/2$ times one of $y$ of
//! absolute value at most $\beta$ gives $\mathrm{NADDS} \cdot n (q/2) \beta$ in place of the
//! bound above, which is below $P/2$ as soon as $P > 2\beta (q-1) n \cdot \mathrm{NADDS}$. With
//! such a bound on the vector, [`Plan::prepare_short`] and [`Plan::mat_vec_short`] take the first
//! primes that satisfy this, often fewer, and check the bound on the vector.
//!
//! **Threads and lanes.** The pieces of a sum and the rows of a matrix product run on the Rayon
//! pool, and forward transforms take up to [`LANES`] polynomials at a time. The results are exact,
//! so they depend on neither.

use crate::{
    arith::{add_mod, reduce},
    crt::SmallCrt,
    error::{Error, Result},
    fma::{FMA_PRIMES, FmaPlan, LANES},
    ntt::MAX_DEGREE,
};
use core::hint::select_unpredictable;
use crypto_bigint::{U64, U576};
use rayon::prelude::*;

/// The number of products accumulated in the NTT domain before one CRT reconstruction (jali,
/// `src/params/moduli.rs`).
pub const NADDS: usize = 128;

/// An exact multiplication plan for $\mathbb Z_q\[X\]/(X^n + 1)$ (the ring $\mathcal R_q$ of draft
/// `4_preliminaries.tex`:67 for $n = d$, and the ring of degree $2d$ used for quotients): one
/// transform plan per selected NTT prime and the CRT to $\bmod q$ (jali, `src/math/ring.rs`,
/// `Ring`).
#[derive(Clone, Debug)]
pub struct Plan {
    q: u64,
    n: usize,
    /// $\lfloor q/2 \rfloor$: canonical values above it are negative when centered.
    half_q: u64,
    plans: Vec<FmaPlan>,
    /// The CRT of the first prime, of the first two primes, and so on.
    crts: Vec<SmallCrt>,
}

/// The residues of a polynomial modulo each selected prime, in the NTT form.
type Residues = Vec<Vec<f64>>;

/// The products an accumulator takes between two reductions: from at most $0.51 p$, eight
/// products of at most $0.61 p$ stay below $8p$ ([`crate::fma`]).
const LAZY_SUMS: usize = 8;

/// A sum of products in the NTT form.
struct Accumulator {
    values: Residues,
    /// The products since the last reduction.
    pending: usize,
}

/// A matrix in the NTT form of its [`Plan`], which holds one residue per selected prime for each
/// coefficient: [`Plan::mat_vec_prepared`] then transforms only the vector, for a matrix that
/// multiplies many vectors.
#[derive(Clone, Debug)]
pub struct Prepared {
    rows: Vec<Vec<Residues>>,
    /// The primes of the residues, and the bound on the vectors they suffice for.
    primes: usize,
    norm: u64,
}

impl Plan {
    /// The plan (jali, `Ring::with_modulus`) for a modulus $2 \le q < 2^{63}$ (prime or not) and a
    /// power of two $2 \le n \le 4096$, with the least number of NTT primes whose product exceeds
    /// $(q-1)^2 \cdot n \cdot \mathrm{NADDS}$ (at most three for every such $q$ and $n$).
    ///
    /// # Errors
    /// [`Error::Parameter`] for an unsupported $q$ or $n$.
    pub fn new(q: u64, n: usize) -> Result<Self> {
        if !(2..1 << 63).contains(&q) {
            return Err(Error::Parameter("RNS modulus must be in [2, 2^63)"));
        }
        if !(2..=MAX_DEGREE).contains(&n) || !n.is_power_of_two() {
            return Err(Error::Parameter(
                "RNS degree must be a power of two in [2, 4096]",
            ));
        }
        // (q-1)^2 n NADDS < 2^126 * 2^12 * 2^7 fits 576 bits, as does any product of the table.
        let qm1 = U576::from_u64(q - 1);
        let bound = qm1
            .wrapping_mul(&qm1)
            .wrapping_mul(&U576::from_u64((n * NADDS) as u64));
        let mut product = U576::ONE;
        let mut plans = Vec::new();
        for (index, (p, _)) in FMA_PRIMES.iter().enumerate() {
            plans.push(FmaPlan::new(index, n)?);
            product = product.wrapping_mul(&U64::from_u64(*p));
            if product > bound {
                break;
            }
        }
        if product <= bound {
            return Err(Error::Overflow);
        }
        let primes: Vec<u64> = plans.iter().map(FmaPlan::modulus).collect();
        let crts = (1..=primes.len())
            .map(|k| SmallCrt::new(q, &primes[..k]))
            .collect::<Result<_>>()?;
        Ok(Self {
            q,
            n,
            half_q: q / 2,
            plans,
            crts,
        })
    }

    /// The least number of primes for sums of at most [`NADDS`] products by a vector of
    /// coefficients of absolute value at most `norm` (module documentation).
    fn primes_for(&self, norm: u64) -> usize {
        // 2 norm (q-1) n NADDS < 2^64 * 2^63 * 2^19 fits 576 bits; a larger norm takes all primes.
        let bound = U576::from_u64(self.q - 1)
            .wrapping_mul(&U576::from_u64(2 * norm.min(self.half_q)))
            .wrapping_mul(&U576::from_u64((self.n * NADDS) as u64));
        let mut product = U576::ONE;
        for (k, plan) in self.plans.iter().enumerate() {
            product = product.wrapping_mul(&U64::from_u64(plan.modulus()));
            if product > bound {
                return k + 1;
            }
        }
        self.plans.len()
    }

    /// [`Error::Overflow`] if a centered coefficient of a polynomial of `v` has an absolute value
    /// above `norm`.
    fn check_short<B: AsRef<[u64]>>(&self, v: &[B], norm: u64) -> Result<()> {
        let q = self.q;
        let largest = |a: &[u64]| {
            a.iter()
                .map(|x| {
                    let v = reduce(*x, q);
                    v.min(q - v)
                })
                .fold(0, u64::max)
        };
        if v.iter().any(|x| largest(x.as_ref()) > norm) {
            return Err(Error::Overflow);
        }
        Ok(())
    }

    /// The modulus $q$ (draft `4_preliminaries.tex`:61).
    pub fn modulus(&self) -> u64 {
        self.q
    }

    /// The degree $n$ of $X^n + 1$ (draft `4_preliminaries.tex`:67 has $n = d$).
    pub fn degree(&self) -> usize {
        self.n
    }

    /// The selected NTT primes, a prefix of [`FMA_PRIMES`] (jali, `Ring::rns_primes`).
    pub fn primes(&self) -> &[u64] {
        self.crts[self.crts.len() - 1].primes()
    }

    fn check(&self, a: &[u64]) -> Result<()> {
        if a.len() != self.n {
            return Err(Error::Length {
                expected: self.n,
                actual: a.len(),
            });
        }
        Ok(())
    }

    /// The residues modulo each prime of the centered representatives of the polynomials of
    /// `batch` (read modulo $q$, at most $n$ coefficients, the others zero), at most [`LANES`],
    /// transformed into the NTT form, for the first `primes` primes.
    fn forward_batch<A: AsRef<[u64]>>(&self, batch: &[A], primes: usize) -> Vec<Residues> {
        let lanes = batch.len();
        debug_assert!((1..=LANES).contains(&lanes));
        let mut out = vec![Vec::with_capacity(primes); lanes];
        let mut centered = vec![0i64; self.n * lanes];
        for (lane, a) in batch.iter().enumerate() {
            for (slot, x) in centered[lane..].iter_mut().step_by(lanes).zip(a.as_ref()) {
                let v = reduce(*x, self.q);
                // The coefficients are random, so the selection stays free of branches.
                let negative = v as i64 - self.q as i64;
                *slot = select_unpredictable(v > self.half_q, negative, v as i64);
            }
        }
        // A centered value below a prime is its own residue, so one pass serves those primes.
        let doubles: Vec<f64> = centered.iter().map(|c| *c as f64).collect();
        let mut buffer = vec![0f64; self.n * lanes];
        for plan in &self.plans[..primes] {
            if self.half_q < plan.modulus() {
                buffer.copy_from_slice(&doubles);
            } else {
                for (slot, c) in buffer.iter_mut().zip(&centered) {
                    *slot = plan.lift(*c);
                }
            }
            plan.forward(&mut buffer, lanes);
            for (lane, residues) in out.iter_mut().enumerate() {
                residues.push(buffer[lane..].iter().step_by(lanes).copied().collect());
            }
        }
        out
    }

    /// [`Plan::forward_batch`] of any number of polynomials.
    fn forward_all<A: AsRef<[u64]> + Sync>(&self, a: &[A], primes: usize) -> Vec<Residues> {
        a.par_chunks(LANES)
            .flat_map_iter(|batch| self.forward_batch(batch, primes))
            .collect()
    }

    fn zero_accumulator(&self, primes: usize) -> Accumulator {
        Accumulator {
            values: vec![vec![0.0; self.n]; primes],
            pending: 0,
        }
    }

    /// `acc += x * y` in the NTT domain.
    fn mac(&self, acc: &mut Accumulator, x: &Residues, y: &Residues) {
        for (((values, x), y), plan) in acc.values.iter_mut().zip(x).zip(y).zip(&self.plans) {
            plan.mac(values, x, y);
        }
        acc.pending += 1;
        if acc.pending == LAZY_SUMS {
            for (values, plan) in acc.values.iter_mut().zip(&self.plans) {
                plan.reduce(values);
            }
            acc.pending = 0;
        }
    }

    /// The products of one piece of a sum of `len` products on the Rayon pool: a piece costs one
    /// inverse transform and one CRT, so a few pieces per thread, at most [`NADDS`] products.
    fn piece(len: usize) -> usize {
        (len / (4 * rayon::current_num_threads())).clamp(16, NADDS)
    }

    /// The accumulator back from the NTT domain, modulo $q$.
    fn flushed(&self, mut acc: Accumulator) -> Vec<u64> {
        let residues: Vec<Vec<u64>> = acc
            .values
            .iter_mut()
            .zip(&self.plans)
            .map(|(values, plan)| {
                plan.inverse(values);
                plan.residues(values)
            })
            .collect();
        let crt = &self.crts[residues.len() - 1];
        (0..self.n)
            .map(|j| crt.reconstruct_column(&residues, j))
            .collect()
    }

    /// $x + y$ coefficientwise, modulo $q$.
    fn sum(&self, mut x: Vec<u64>, y: &[u64]) -> Vec<u64> {
        for (x, y) in x.iter_mut().zip(y) {
            *x = add_mod(*x, *y, self.q);
        }
        x
    }

    /// The exact product $ab$ in $\mathbb Z_q\[X\]/(X^n + 1)$ (coefficients read modulo $q$; draft
    /// `4_preliminaries.tex`:67).
    ///
    /// # Errors
    /// [`Error::Length`] unless both operands have $n$ coefficients.
    pub fn mul(&self, a: &[u64], b: &[u64]) -> Result<Vec<u64>> {
        self.inner(&[a], &[b])
    }

    /// The exact inner product $\sum_t a_t b_t$ in $\mathbb Z_q\[X\]/(X^n + 1)$; up to [`NADDS`]
    /// products are accumulated in the NTT domain before each CRT (jali, `RnsProduct::mac`). The
    /// empty sum is zero.
    ///
    /// # Errors
    /// [`Error::Length`] if the two sides have different lengths or a polynomial does not have
    /// $n$ coefficients.
    pub fn inner<A, B>(&self, a: &[A], b: &[B]) -> Result<Vec<u64>>
    where
        A: AsRef<[u64]> + Sync,
        B: AsRef<[u64]> + Sync,
    {
        for x in a
            .iter()
            .map(AsRef::as_ref)
            .chain(b.iter().map(AsRef::as_ref))
        {
            self.check(x)?;
        }
        self.inner_padded(a, b)
    }

    /// [`Plan::inner`] of polynomials with at most $n$ coefficients, the others zero: with $n$ a
    /// degree of twice theirs, a product over $\mathbb Z_q\[X\]$ that does not wrap.
    ///
    /// # Errors
    /// [`Error::Length`] if the two sides have different lengths or a polynomial has more than
    /// $n$ coefficients.
    pub fn inner_padded<A, B>(&self, a: &[A], b: &[B]) -> Result<Vec<u64>>
    where
        A: AsRef<[u64]> + Sync,
        B: AsRef<[u64]> + Sync,
    {
        if a.len() != b.len() {
            return Err(Error::Length {
                expected: a.len(),
                actual: b.len(),
            });
        }
        if let Some(x) = a
            .iter()
            .map(AsRef::as_ref)
            .chain(b.iter().map(AsRef::as_ref))
            .find(|x| x.len() > self.n)
        {
            return Err(Error::Length {
                expected: self.n,
                actual: x.len(),
            });
        }
        let piece = Self::piece(a.len());
        let primes = self.plans.len();
        Ok(a.par_chunks(piece)
            .zip(b.par_chunks(piece))
            .map(|(chunk_a, chunk_b)| {
                let mut acc = self.zero_accumulator(primes);
                for (xs, ys) in chunk_a.chunks(LANES / 2).zip(chunk_b.chunks(LANES / 2)) {
                    let both: Vec<&[u64]> = xs
                        .iter()
                        .map(AsRef::as_ref)
                        .chain(ys.iter().map(AsRef::as_ref))
                        .collect();
                    let forms = self.forward_batch(&both, primes);
                    let (xs, ys) = forms.split_at(xs.len());
                    for (x, y) in xs.iter().zip(ys) {
                        self.mac(&mut acc, x, y);
                    }
                }
                self.flushed(acc)
            })
            .reduce(|| vec![0; self.n], |x, y| self.sum(x, &y)))
    }

    /// The matrix `m` in the NTT form, for [`Plan::mat_vec_prepared`].
    ///
    /// # Errors
    /// [`Error::Length`] if a polynomial does not have $n$ coefficients.
    pub fn prepare<R, A>(&self, m: &[R]) -> Result<Prepared>
    where
        R: AsRef<[A]> + Sync,
        A: AsRef<[u64]> + Sync,
    {
        self.prepare_short(m, self.half_q)
    }

    /// [`Plan::prepare`] for vectors of centered coefficients of absolute value at most `norm`,
    /// with the primes these take (module documentation).
    ///
    /// # Errors
    /// [`Error::Length`] if a polynomial does not have $n$ coefficients.
    pub fn prepare_short<R, A>(&self, m: &[R], norm: u64) -> Result<Prepared>
    where
        R: AsRef<[A]> + Sync,
        A: AsRef<[u64]> + Sync,
    {
        for x in m.iter().flat_map(AsRef::as_ref) {
            self.check(x.as_ref())?;
        }
        let primes = self.primes_for(norm);
        let rows = m
            .par_iter()
            .map(|row| self.forward_all(row.as_ref(), primes))
            .collect();
        Ok(Prepared { rows, primes, norm })
    }

    /// [`Plan::mat_vec`] of a matrix from [`Plan::prepare`].
    ///
    /// # Errors
    /// [`Error::Length`] if a row does not have one entry per entry of $v$, or a polynomial
    /// does not have $n$ coefficients; [`Error::Overflow`] for a vector above the bound of
    /// [`Plan::prepare_short`].
    pub fn mat_vec_prepared<B>(&self, m: &Prepared, v: &[B]) -> Result<Vec<Vec<u64>>>
    where
        B: AsRef<[u64]> + Sync,
    {
        for row in &m.rows {
            if row.len() != v.len() {
                return Err(Error::Length {
                    expected: v.len(),
                    actual: row.len(),
                });
            }
        }
        for x in v {
            self.check(x.as_ref())?;
        }
        if m.primes < self.plans.len() {
            self.check_short(v, m.norm)?;
        }
        let v_ntt = self.forward_all(v, m.primes);
        let piece = Self::piece(v.len());
        Ok(m.rows
            .par_iter()
            .map(|row| {
                row.par_chunks(piece)
                    .zip(v_ntt.par_chunks(piece))
                    .map(|(chunk_m, chunk_v)| {
                        let mut acc = self.zero_accumulator(m.primes);
                        for (x, y) in chunk_m.iter().zip(chunk_v) {
                            self.mac(&mut acc, x, y);
                        }
                        self.flushed(acc)
                    })
                    .reduce(|| vec![0; self.n], |x, y| self.sum(x, &y))
            })
            .collect())
    }

    /// The exact matrix-vector product $(\sum_j m_{ij} v_j)\_i$ over $\mathbb Z_q\[X\]/(X^n + 1)$,
    /// as for the commitments $\mathbf t = \mathbf A \mathbf s$ (draft
    /// `6_Collaborative_Hachi.tex`:375-381).
    /// Each entry of $v$ is transformed once; each row is an inner product as in
    /// [`Plan::inner`].
    ///
    /// # Errors
    /// [`Error::Length`] if a row does not have one entry per entry of $v$, or a polynomial
    /// does not have $n$ coefficients.
    pub fn mat_vec<R, A, B>(&self, m: &[R], v: &[B]) -> Result<Vec<Vec<u64>>>
    where
        R: AsRef<[A]> + Sync,
        A: AsRef<[u64]> + Sync,
        B: AsRef<[u64]> + Sync,
    {
        self.mat_vec_short(m, v, self.half_q)
    }

    /// [`Plan::mat_vec`] of a vector of centered coefficients of absolute value at most `norm`,
    /// with the primes these take (module documentation).
    ///
    /// # Errors
    /// As [`Plan::mat_vec`], and [`Error::Overflow`] for a vector above the bound.
    pub fn mat_vec_short<R, A, B>(&self, m: &[R], v: &[B], norm: u64) -> Result<Vec<Vec<u64>>>
    where
        R: AsRef<[A]> + Sync,
        A: AsRef<[u64]> + Sync,
        B: AsRef<[u64]> + Sync,
    {
        for row in m {
            if row.as_ref().len() != v.len() {
                return Err(Error::Length {
                    expected: v.len(),
                    actual: row.as_ref().len(),
                });
            }
            for x in row.as_ref() {
                self.check(x.as_ref())?;
            }
        }
        for x in v {
            self.check(x.as_ref())?;
        }
        let primes = self.primes_for(norm);
        if primes < self.plans.len() {
            self.check_short(v, norm)?;
        }
        let v_ntt = self.forward_all(v, primes);
        let piece = Self::piece(v.len());
        Ok(m.par_iter()
            .map(|row| {
                row.as_ref()
                    .par_chunks(piece)
                    .zip(v_ntt.par_chunks(piece))
                    .map(|(chunk_m, chunk_v)| {
                        let mut acc = self.zero_accumulator(primes);
                        for (xs, ys) in chunk_m.chunks(LANES).zip(chunk_v.chunks(LANES)) {
                            for (x, y) in self.forward_batch(xs, primes).iter().zip(ys) {
                                self.mac(&mut acc, x, y);
                            }
                        }
                        self.flushed(acc)
                    })
                    .reduce(|| vec![0; self.n], |x, y| self.sum(x, &y))
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arith::tests::SplitMix;

    /// The number of primes is the least one that satisfies the bound, checked against the bound
    /// recomputed with num-bigint, for moduli and degrees at both ends of the range.
    #[test]
    fn the_prime_count_is_the_least_that_satisfies_the_bound() {
        use num_bigint::BigUint;
        for q in [
            2u64,
            3,
            13,
            4294967197,
            4503599627370101,
            2305843009213693693,
            9223372036854775549,
            (1 << 63) - 1,
        ] {
            for log_n in 1..=12 {
                let n = 1usize << log_n;
                let plan = Plan::new(q, n).unwrap();
                let bound = BigUint::from(q - 1).pow(2) * BigUint::from(n) * BigUint::from(NADDS);
                let product = |k: usize| -> BigUint {
                    FMA_PRIMES[..k]
                        .iter()
                        .map(|(p, _)| BigUint::from(*p))
                        .product()
                };
                let k = plan.primes().len();
                assert!(product(k) > bound, "q {q} n {n}");
                assert!(
                    k == 1 || product(k - 1) <= bound,
                    "q {q} n {n}: not the least"
                );
                assert!(k <= 3);
            }
        }
        // Three primes are needed at the top of the range.
        assert_eq!(
            Plan::new(9223372036854775549, 4096).unwrap().primes().len(),
            3
        );
    }

    /// The centered lift: the residue modulo $p_j$ of the centered representative.
    #[test]
    fn the_lift_is_the_residue_of_the_centered_representative() {
        let mut rng = SplitMix(4);
        for q in [
            3u64,
            13,
            4294967197,
            2305843009213693693,
            9223372036854775549,
            (1 << 63) - 1,
        ] {
            let plan = Plan::new(q, 4).unwrap();
            let mut values = vec![0, 1, q / 2, q / 2 + 1, q - 1, q, u64::MAX];
            values.extend((0..57).map(|_| rng.next()));
            for chunk in values.chunks(4) {
                if chunk.len() < 4 {
                    continue;
                }
                let lifted = &plan.forward_batch(&[chunk], plan.plans.len())[0];
                for (j, plan_j) in plan.plans.iter().enumerate() {
                    let mut back = lifted[j].clone();
                    plan_j.inverse(&mut back);
                    let p = plan_j.modulus();
                    for (x, r) in chunk.iter().zip(plan_j.residues(&back)) {
                        let v = i128::from(x % q);
                        let centered = if v > i128::from(q / 2) {
                            v - i128::from(q)
                        } else {
                            v
                        };
                        assert_eq!(i128::from(r), centered.rem_euclid(i128::from(p)));
                    }
                }
            }
        }
    }

    /// The batches of a sum do not show in its result: inner products and matrix products of
    /// every length around the sizes of a batch and of a lazy sum, against single products.
    #[test]
    fn sums_equal_their_products_at_every_batch_boundary() {
        let mut rng = SplitMix(6);
        for q in [13u64, 4294967197, 2305843009213693693, 9223372036854775549] {
            for n in [4usize, 64] {
                let plan = Plan::new(q, n).unwrap();
                let polys: Vec<Vec<u64>> = (0..50)
                    .map(|_| (0..n).map(|_| rng.next() % q).collect())
                    .collect();
                for len in (1..=2 * LANES + 1).chain([LAZY_SUMS, LAZY_SUMS + 1, 25]) {
                    let (a, b) = (&polys[..len], &polys[25..25 + len]);
                    let mut want = vec![0; n];
                    for (x, y) in a.iter().zip(b) {
                        want = plan.sum(want, &plan.mul(x, y).unwrap());
                    }
                    assert_eq!(plan.inner(a, b).unwrap(), want, "q {q} n {n} len {len}");
                    let m = [a.to_vec(), b.to_vec()];
                    let rows = plan.mat_vec(&m, b).unwrap();
                    assert_eq!(rows[0], want, "q {q} n {n} len {len}");
                    let prepared = plan.prepare(&m).unwrap();
                    assert_eq!(plan.mat_vec_prepared(&prepared, b).unwrap(), rows);
                }
            }
        }
    }

    /// The coefficients $\pm$`norm` whose product with a constant polynomial of coefficients
    /// $t$ reaches $n t$`norm` at coefficient 0: a sum of [`NADDS`] of them meets the bound of the
    /// module documentation.
    fn extreme(q: u64, n: usize, norm: u64) -> Vec<u64> {
        let mut out = vec![q - norm; n];
        out[0] = norm;
        out
    }

    /// The short products at the largest norm that takes each number of primes, on worst-case
    /// and random operands, against the products with every prime; and the bound is checked.
    #[test]
    fn short_products_equal_the_full_products_at_the_prime_count_boundary() {
        let mut rng = SplitMix(7);
        for q in [65537u64, 4294967197, 34801213478789, 384156492036173] {
            for n in [16usize, 1024] {
                let plan = Plan::new(q, n).unwrap();
                for primes in 1..plan.plans.len() {
                    // The bisection of the rule: the largest norm below half_q that takes
                    // `primes` primes, if any.
                    let (mut low, mut high) = (0, plan.half_q);
                    if plan.primes_for(1) > primes {
                        continue;
                    }
                    while high - low > 1 {
                        let middle = low + (high - low) / 2;
                        if plan.primes_for(middle) <= primes {
                            low = middle;
                        } else {
                            high = middle;
                        }
                    }
                    let norm = low.max(1);
                    assert_eq!(plan.primes_for(norm), primes, "q {q} n {n}");
                    let top = (q - 1) / 2;
                    let row: Vec<Vec<u64>> = (0..NADDS).map(|_| vec![top; n]).collect();
                    let random: Vec<Vec<u64>> = (0..NADDS)
                        .map(|_| (0..n).map(|_| rng.next() % q).collect())
                        .collect();
                    let v: Vec<Vec<u64>> = (0..NADDS).map(|_| extreme(q, n, norm)).collect();
                    let m = [row, random];
                    let want = plan.mat_vec(&m, &v).unwrap();
                    assert_eq!(plan.mat_vec_short(&m, &v, norm).unwrap(), want);
                    let prepared = plan.prepare_short(&m, norm).unwrap();
                    assert_eq!(prepared.primes, primes);
                    assert_eq!(plan.mat_vec_prepared(&prepared, &v).unwrap(), want);
                    // One coefficient above the bound, reduced or not; an unreduced coefficient
                    // stands for its residue, so one within the bound is accepted.
                    let mut above = v.clone();
                    above[3][2] = norm + 1;
                    assert!(matches!(
                        plan.mat_vec_short(&m, &above, norm),
                        Err(Error::Overflow)
                    ));
                    assert!(matches!(
                        plan.mat_vec_prepared(&prepared, &above),
                        Err(Error::Overflow)
                    ));
                    above[3][2] = q + norm + 1;
                    assert!(matches!(
                        plan.mat_vec_short(&m, &above, norm),
                        Err(Error::Overflow)
                    ));
                    let mut unreduced = v.clone();
                    unreduced[3][2] += q;
                    assert_eq!(plan.mat_vec_short(&m, &unreduced, norm).unwrap(), want);
                }
            }
        }
    }

    #[test]
    fn invalid_plans_and_shapes_are_rejected() {
        assert!(Plan::new(1, 16).is_err());
        assert!(Plan::new(1 << 63, 16).is_err());
        assert!(Plan::new(13, 1).is_err());
        assert!(Plan::new(13, 24).is_err());
        assert!(Plan::new(13, 8192).is_err());
        let plan = Plan::new(13, 4).unwrap();
        assert!(plan.mul(&[0; 4], &[0; 3]).is_err());
        assert!(plan.inner(&[[0u64; 4]], &[[0u64; 4], [0; 4]]).is_err());
        assert!(
            plan.mat_vec(&[vec![vec![0u64; 4]]], &[vec![0u64; 4], vec![0; 4]])
                .is_err()
        );
        assert!(
            plan.mat_vec(&[vec![vec![0u64; 3]]], &[vec![0u64; 4]])
                .is_err()
        );
        assert_eq!(plan.inner::<&[u64], &[u64]>(&[], &[]).unwrap(), vec![0; 4]);
    }
}
