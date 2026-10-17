//! Exact products in $\mathbb Z_q\[X\]/(X^n + 1)$ for any modulus $2 \le q < 2^{63}$, through a
//! residue number system of NTT primes and a CRT straight to $\bmod q$.
//!
//! Ported from jali (`src/math/rns.rs` and `src/math/ring.rs`, MIT, by the same author): the
//! prime-count rule, the centered lift and the accumulation of products in the NTT domain before
//! one CRT.
//!
//! **Exactness.** Each coefficient is lifted to its centered representative, of absolute value at
//! most $(q-1)/2$, so one negacyclic product over $\mathbb Z$ has coefficients of absolute value at
//! most $n (q-1)^2/4$, and a sum of at most [`NADDS`] products at most $\mathrm{NADDS} \cdot n
//! (q-1)^2/4$. The plan takes the least number $K$ of primes from [`NTT_PRIMES`] (in table
//! order) with $P = \prod_{j<K} p_j > (q-1)^2 \cdot n \cdot \mathrm{NADDS}$ (jali's rule; twice
//! what is needed), so every such sum lies in $(-P/2, P/2)$, its residues modulo the $p_j$ are
//! computed exactly by the transforms, and [`SmallCrt`] recovers it modulo $q$ without error.
//! Longer sums are cut into chunks of [`NADDS`] products, and the chunks are added modulo $q$.

use crate::{
    arith::{add_mod, reduce, sub_mod},
    crt::SmallCrt,
    error::{Error, Result},
    ntt::{MAX_DEGREE, NTT_PRIMES, NttPlan},
};
use crypto_bigint::{U64, U576};

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
    plans: Vec<NttPlan>,
    /// $q \bmod p_j$ for each selected prime.
    q_mod_p: Vec<u64>,
    crt: SmallCrt,
}

/// The residues of a polynomial modulo each selected prime, in the Montgomery NTT form.
type Residues = Vec<Vec<u64>>;

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
        for (index, (p, _)) in NTT_PRIMES.iter().enumerate() {
            plans.push(NttPlan::from_table(index, n)?);
            product = product.wrapping_mul(&U64::from_u64(*p));
            if product > bound {
                break;
            }
        }
        if product <= bound {
            return Err(Error::Overflow);
        }
        let primes: Vec<u64> = plans.iter().map(NttPlan::modulus).collect();
        Ok(Self {
            q,
            n,
            half_q: q / 2,
            q_mod_p: primes.iter().map(|p| q % p).collect(),
            crt: SmallCrt::new(q, &primes)?,
            plans,
        })
    }

    /// The modulus $q$ (draft `4_preliminaries.tex`:61).
    pub fn modulus(&self) -> u64 {
        self.q
    }

    /// The degree $n$ of $X^n + 1$ (draft `4_preliminaries.tex`:67 has $n = d$).
    pub fn degree(&self) -> usize {
        self.n
    }

    /// The selected NTT primes, a prefix of [`NTT_PRIMES`] (jali, `Ring::rns_primes`).
    pub fn primes(&self) -> &[u64] {
        self.crt.primes()
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

    /// The residues modulo each prime of the centered representatives of `a` (read modulo $q$),
    /// transformed into the Montgomery NTT form.
    fn forward(&self, a: &[u64]) -> Residues {
        self.plans
            .iter()
            .zip(&self.q_mod_p)
            .map(|(plan, q_mod_p)| {
                let p = plan.modulus();
                let mut x: Vec<u64> = a
                    .iter()
                    .map(|x| {
                        let v = reduce(*x, self.q);
                        // v < q < 2^63 < 3p: at most two subtractions.
                        let mut r = v;
                        while r >= p {
                            r -= p;
                        }
                        if v > self.half_q {
                            sub_mod(r, *q_mod_p, p)
                        } else {
                            r
                        }
                    })
                    .collect();
                plan.forward_mont(&mut x);
                x
            })
            .collect()
    }

    fn zero_residues(&self) -> Residues {
        vec![vec![0; self.n]; self.plans.len()]
    }

    /// `acc += x * y` in the NTT domain.
    fn mac(&self, acc: &mut Residues, x: &Residues, y: &Residues) {
        for (((acc, x), y), plan) in acc.iter_mut().zip(x).zip(y).zip(&self.plans) {
            let p = plan.modulus();
            for ((acc, x), y) in acc.iter_mut().zip(x).zip(y) {
                *acc = add_mod(*acc, plan.mont_mul(*x, *y), p);
            }
        }
    }

    /// `out += reconstruct(acc) mod q`, consuming the accumulator.
    fn flush(&self, mut acc: Residues, out: &mut [u64]) {
        for (values, plan) in acc.iter_mut().zip(&self.plans) {
            plan.inverse_mont(values);
        }
        for (j, out) in out.iter_mut().enumerate() {
            *out = add_mod(*out, self.crt.reconstruct_column(&acc, j), self.q);
        }
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
    pub fn inner<A: AsRef<[u64]>, B: AsRef<[u64]>>(&self, a: &[A], b: &[B]) -> Result<Vec<u64>> {
        if a.len() != b.len() {
            return Err(Error::Length {
                expected: a.len(),
                actual: b.len(),
            });
        }
        for x in a
            .iter()
            .map(AsRef::as_ref)
            .chain(b.iter().map(AsRef::as_ref))
        {
            self.check(x)?;
        }
        let mut out = vec![0; self.n];
        for (chunk_a, chunk_b) in a.chunks(NADDS).zip(b.chunks(NADDS)) {
            let mut acc = self.zero_residues();
            for (x, y) in chunk_a.iter().zip(chunk_b) {
                let (x, y) = (self.forward(x.as_ref()), self.forward(y.as_ref()));
                self.mac(&mut acc, &x, &y);
            }
            self.flush(acc, &mut out);
        }
        Ok(out)
    }

    /// The exact matrix-vector product $(\sum_j m_{ij} v_j)\_i$ over $\mathbb Z_q\[X\]/(X^n + 1)$,
    /// as for the commitments $\mathbf t = \mathbf A \mathbf s$ (draft
    /// `6_Collaborative_Hachi.tex`:370-376).
    /// Each entry of $v$ is transformed once; each row is an inner product as in
    /// [`Plan::inner`].
    ///
    /// # Errors
    /// [`Error::Length`] if a row does not have one entry per entry of $v$, or a polynomial
    /// does not have $n$ coefficients.
    pub fn mat_vec<R, A, B>(&self, m: &[R], v: &[B]) -> Result<Vec<Vec<u64>>>
    where
        R: AsRef<[A]>,
        A: AsRef<[u64]>,
        B: AsRef<[u64]>,
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
        let v_ntt: Vec<Residues> = v.iter().map(|x| self.forward(x.as_ref())).collect();
        Ok(m.iter()
            .map(|row| {
                let mut out = vec![0; self.n];
                for (chunk_m, chunk_v) in row.as_ref().chunks(NADDS).zip(v_ntt.chunks(NADDS)) {
                    let mut acc = self.zero_residues();
                    for (x, y) in chunk_m.iter().zip(chunk_v) {
                        self.mac(&mut acc, &self.forward(x.as_ref()), y);
                    }
                    self.flush(acc, &mut out);
                }
                out
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
                    NTT_PRIMES[..k]
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
                let lifted = plan.forward(chunk);
                for (j, plan_j) in plan.plans.iter().enumerate() {
                    let mut back = lifted[j].clone();
                    plan_j.inverse_mont(&mut back);
                    let p = plan_j.modulus();
                    for (x, r) in chunk.iter().zip(&back) {
                        let v = i128::from(x % q);
                        let centered = if v > i128::from(q / 2) {
                            v - i128::from(q)
                        } else {
                            v
                        };
                        assert_eq!(i128::from(*r), centered.rem_euclid(i128::from(p)));
                    }
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
