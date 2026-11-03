//! The two sumchecks of one iteration, run with common round challenges (paper, Section 4.3,
//! Fig. 5-7): over $\mathbf x \in \lbrace 0, 1 \rbrace^L$,
//! - the range check $F_0(\mathbf x) = \widetilde{eq}(\tau_0, \mathbf x) \cdot P(\tilde w(\mathbf
//!   x))$ with $P(y) = \prod_{c \in D} (y - c)$ over the digit set $D$ of base $b$, claimed sum
//!   $0$;
//! - the linear check $F_\alpha(\mathbf x) = \tilde w(\mathbf x) \cdot \tilde g(\mathbf x)$, where
//!   $\tilde g$ is the public multilinear table $g(u \Vert \ell) = m^\ast(u) \alpha^\ell$
//!   (`crate::relation`), claimed sum $\sum_i \widetilde{eq}(\tau_1, i) y_i(\alpha)$.
//!
//! **Digit set, a deviation from `hachi-pcs.tex`.** $D$ is the set of the $b$ balanced digits
//! $\[-\lfloor b/2 \rfloor, \lceil b/2 \rceil - 1\]$ of
//! [`hachi_co_math::decompose::digit_range`], which holds every digit of an honest prover
//! ($\mathbf G^{-1}$ and $\mathbf J^{-1}$ decompose into it). The appendix's range polynomial, that
//! of Hachi's Eq. (23), runs over $\lbrace -b+1, \dots, b-1 \rbrace$ instead. With the smaller set
//! $F_0$ has degree $b + 1$ in place of $2b$, and the digits that the range check lets through lie
//! in the balanced range, which is the range the norm bounds of binding use
//! (`scripts/hachi-parameters/choose.py`). The final witness is held to the same set
//! ([`crate::pcs`], "Encodings").
//!
//! Each round the prover sends both round polynomials as their values at $t = 0, 1, \dots$,
//! $\deg$: $|D| + 2$ values for $F_0$ (degree $|D| + 1$) and 3 for $F_\alpha$ (degree 2). After
//! the last round it holds $\tilde w(\mathbf a)$ at the point $\mathbf a$ of the round
//! challenges, bound in order from the first (most significant) variable.
//!
//! **Prover's cost.** The values of a table at consecutive nodes are formed by additions, and
//! `RangeFactors` groups the digits $\pm c$, so that $P$ takes about $|D|/2$ products. In the
//! first round the witness is a table of digits, half of the whole work: its node values are
//! small integers, whose $P$ comes from a table (`SmallValues`), and the table stays in
//! $\mathbb Z_q$ until the first challenge. The pairs of points run on the Rayon pool; field sums
//! do not depend on the order, so the messages do not depend on the number of threads.

use crate::error::VerifyError;
use crate::fs::Fs;
use crate::mle::{fold_top, interpolate};
use crate::{Error, Result};
use hachi_co_math::{Ext, Fqk};
use rayon::prelude::*;

/// One round's messages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoundMessage<const K: usize> {
    /// $F_0$'s round polynomial at $t = 0, \dots, |D| + 1$.
    pub h0: Vec<Ext<K>>,
    /// $F_\alpha$'s round polynomial at $t = 0, 1, 2$.
    pub h_alpha: Vec<Ext<K>>,
}

/// $P(y) = \prod_{c \in D} (y - c)$, with the digits given as residues.
pub fn range_poly<const K: usize>(field: &Fqk<K>, digits: &[u64], y: &Ext<K>) -> Ext<K> {
    digits.iter().fold(field.one(), |acc, c| {
        field.mul(&acc, &field.sub(y, &field.from_base(*c)))
    })
}

/// $P(y) = \prod_{c \in D} (y - c)$ as $\prod_c (y^2 - c^2) \prod_e (y - e)$, over the pairs
/// $\pm c$ of the digit set and the digits $e$ without a partner: the same polynomial as
/// [`range_poly`], with one squaring in place of half the products.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RangeFactors {
    /// The residues $c^2$ of the pairs.
    squares: Vec<u64>,
    /// The residues $e$ of the digits without a partner.
    single: Vec<u64>,
}

impl RangeFactors {
    fn new<const K: usize>(field: &Fqk<K>, digits: &[u64]) -> Self {
        let zq = field.zq();
        let mut left: Vec<u64> = digits.iter().map(|c| zq.reduce(*c)).collect();
        let mut squares = Vec::new();
        let mut single = Vec::new();
        while let Some(c) = left.pop() {
            match left.iter().position(|x| *x == zq.neg(c)) {
                Some(partner) if c != 0 => {
                    left.swap_remove(partner);
                    squares.push(zq.mul(c, c));
                }
                _ => single.push(c),
            }
        }
        Self { squares, single }
    }

    /// $P(y)$.
    #[inline]
    fn eval<const K: usize>(&self, field: &Fqk<K>, y: &Ext<K>) -> Ext<K> {
        let mut acc = field.one();
        if !self.squares.is_empty() {
            let y2 = field.mul(y, y);
            for c2 in &self.squares {
                acc = field.mul(&acc, &field.sub(&y2, &field.from_base(*c2)));
            }
        }
        for e in &self.single {
            acc = field.mul(&acc, &field.sub(y, &field.from_base(*e)));
        }
        acc
    }
}

/// The range polynomial at the small integers that the first round's nodes reach: for a pair of
/// digits $u, v$ of absolute value at most `digit`, the node values $u + t (v - u)$ with $t \le$
/// `degree` lie within `reach` $= (2\,\mathrm{degree} + 1)\,\mathrm{digit}$.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SmallValues {
    /// The largest absolute value of a digit.
    digit: i64,
    /// The largest absolute value of a node value.
    reach: i64,
    /// The residue of $c$, at index $c +$ `reach`.
    value: Vec<u64>,
    /// $P(c)$ as a residue, at index $c +$ `reach`.
    range: Vec<u64>,
}

impl SmallValues {
    fn new<const K: usize>(
        field: &Fqk<K>,
        range: &RangeFactors,
        digits: &[u64],
        degree: usize,
    ) -> Self {
        let zq = field.zq();
        let digit = digits
            .iter()
            .map(|c| zq.to_centered(*c).abs())
            .max()
            .unwrap_or(0);
        let reach = (2 * degree as i64 + 1) * digit;
        let value: Vec<u64> = (-reach..=reach).map(|c| zq.from_i64(c)).collect();
        let range = value
            .iter()
            .map(|c| range.eval(field, &field.from_base(*c)).0[0])
            .collect();
        Self {
            digit,
            reach,
            value,
            range,
        }
    }

    /// The centered value of the residue `x`, if it has the size of a digit.
    #[inline(always)]
    fn small<const K: usize>(&self, field: &Fqk<K>, x: u64) -> Option<i64> {
        let c = field.zq().to_centered(x);
        (c.abs() <= self.digit).then_some(c)
    }
}

/// The values of one round's polynomials at $t = 0, \dots, \deg F_0$ and $t = 0, 1, 2$ with the
/// witness table `w` of the first round, over $\mathbb Z_q$: [`round_values`] with the node
/// values of a pair of digits read from `small`, so that each is a product of a field element
/// by a residue.
fn first_round_values<const K: usize>(
    field: &Fqk<K>,
    range: &RangeFactors,
    small: &SmallValues,
    deg0: usize,
    w: &[u64],
    eq0: &[Ext<K>],
    g: &[Ext<K>],
) -> (Vec<Ext<K>>, Vec<Ext<K>>) {
    let half = w.len() / 2;
    let zeros = || (vec![field.zero(); deg0 + 1], vec![field.zero(); 3]);
    (0..half)
        .into_par_iter()
        .fold(zeros, |(mut h0, mut h_alpha), i| {
            let (mut et, mut gt) = (eq0[i], g[i]);
            let de = field.sub(&eq0[i + half], &et);
            let dg = field.sub(&g[i + half], &gt);
            let pair = small
                .small(field, w[i])
                .zip(small.small(field, w[i + half]));
            if let Some((lo, hi)) = pair {
                let (mut wt, step) = (lo, hi - lo);
                for (t, h) in h0.iter_mut().enumerate() {
                    let index = (wt + small.reach) as usize;
                    *h = field.add(h, &field.scale(&et, small.range[index]));
                    if t < 3 {
                        let product = field.scale(&gt, small.value[index]);
                        h_alpha[t] = field.add(&h_alpha[t], &product);
                    }
                    wt += step;
                    et = field.add(&et, &de);
                    gt = field.add(&gt, &dg);
                }
            } else {
                let mut wt = field.from_base(w[i]);
                let dw = field.sub(&field.from_base(w[i + half]), &wt);
                for (t, h) in h0.iter_mut().enumerate() {
                    *h = field.add(h, &field.mul(&et, &range.eval(field, &wt)));
                    if t < 3 {
                        h_alpha[t] = field.add(&h_alpha[t], &field.mul(&wt, &gt));
                    }
                    wt = field.add(&wt, &dw);
                    et = field.add(&et, &de);
                    gt = field.add(&gt, &dg);
                }
            }
            (h0, h_alpha)
        })
        .reduce(zeros, |x, y| add_values(field, x, &y))
}

/// The sums of two parts of a round.
fn add_values<const K: usize>(
    field: &Fqk<K>,
    (mut h0, mut h_alpha): (Vec<Ext<K>>, Vec<Ext<K>>),
    (x0, x_alpha): &(Vec<Ext<K>>, Vec<Ext<K>>),
) -> (Vec<Ext<K>>, Vec<Ext<K>>) {
    for (h, x) in h0.iter_mut().zip(x0) {
        *h = field.add(h, x);
    }
    for (h, x) in h_alpha.iter_mut().zip(x_alpha) {
        *h = field.add(h, x);
    }
    (h0, h_alpha)
}

/// The values of one round's polynomials at $t = 0, \dots, \deg F_0$ and $t = 0, 1, 2$, for the
/// tables `w`, `eq0` and `g` of equal even length.
fn round_values<const K: usize>(
    field: &Fqk<K>,
    range: &RangeFactors,
    deg0: usize,
    w: &[Ext<K>],
    eq0: &[Ext<K>],
    g: &[Ext<K>],
) -> (Vec<Ext<K>>, Vec<Ext<K>>) {
    let half = w.len() / 2;
    let zeros = || (vec![field.zero(); deg0 + 1], vec![field.zero(); 3]);
    (0..half)
        .into_par_iter()
        .fold(zeros, |(mut h0, mut h_alpha), i| {
            let (mut wt, mut et, mut gt) = (w[i], eq0[i], g[i]);
            let dw = field.sub(&w[i + half], &wt);
            let de = field.sub(&eq0[i + half], &et);
            let dg = field.sub(&g[i + half], &gt);
            for (t, h) in h0.iter_mut().enumerate() {
                *h = field.add(h, &field.mul(&et, &range.eval(field, &wt)));
                if t < 3 {
                    h_alpha[t] = field.add(&h_alpha[t], &field.mul(&wt, &gt));
                }
                wt = field.add(&wt, &dw);
                et = field.add(&et, &de);
                gt = field.add(&gt, &dg);
            }
            (h0, h_alpha)
        })
        .reduce(zeros, |x, y| add_values(field, x, &y))
}

/// The round labels of the transcript.
const LABEL_H0: &[u8] = b"hachi/sumcheck/h0";
const LABEL_HA: &[u8] = b"hachi/sumcheck/h_alpha";
const LABEL_ROUND: &[u8] = b"hachi/sumcheck/round";

/// Runs the prover on the tables of $\tilde w$, over $\mathbb Z_q$, of
/// $\widetilde{eq}(\tau_0, \cdot)$ and of $\tilde g$ (each of $2^L$ values) and returns the round
/// messages, the point $\mathbf a$ and $\tilde w(\mathbf a)$.
///
/// # Errors
/// [`Error::Shape`] if the tables differ in length or are not of a power-of-two length.
pub fn prove<const K: usize>(
    field: &Fqk<K>,
    digits: &[u64],
    table: &[u64],
    mut eq0: Vec<Ext<K>>,
    mut g: Vec<Ext<K>>,
    fs: &mut Fs,
) -> Result<(Vec<RoundMessage<K>>, Vec<Ext<K>>, Ext<K>)> {
    let n = table.len();
    if !n.is_power_of_two() || eq0.len() != n || g.len() != n {
        return Err(Error::Shape(
            "the sumcheck tables must have one power-of-two length".into(),
        ));
    }
    let rounds = n.trailing_zeros() as usize;
    let deg0 = digits.len() + 1;
    let range = RangeFactors::new(field, digits);
    let small = SmallValues::new(field, &range, digits, deg0);
    let mut messages = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    // Empty until the first challenge, which takes it out of Z_q.
    let mut w: Vec<Ext<K>> = Vec::new();
    for round in 0..rounds {
        let (h0, h_alpha) = if round == 0 {
            first_round_values(field, &range, &small, deg0, table, &eq0, &g)
        } else {
            round_values(field, &range, deg0, &w, &eq0, &g)
        };
        fs.exts(LABEL_H0, &h0);
        fs.exts(LABEL_HA, &h_alpha);
        let r = fs.field_coins::<K>(LABEL_ROUND, field.modulus(), 1)?[0];
        if round == 0 {
            let (lo, hi) = table.split_at(n / 2);
            let zq = field.zq();
            w = lo
                .par_iter()
                .zip(hi.par_iter())
                .map(|(a, b)| field.add(&field.from_base(*a), &field.scale(&r, zq.sub(*b, *a))))
                .collect();
        } else {
            fold_top(field, &mut w, &r);
        }
        fold_top(field, &mut eq0, &r);
        fold_top(field, &mut g, &r);
        messages.push(RoundMessage { h0, h_alpha });
        point.push(r);
    }
    let value = w
        .first()
        .copied()
        .unwrap_or_else(|| field.from_base(table[0]));
    Ok((messages, point, value))
}

/// What the verifier is left with: the point $\mathbf a$ and the claims on $F_0(\mathbf a)$ and
/// $F_\alpha(\mathbf a)$.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reduced<const K: usize> {
    /// The point of the round challenges.
    pub point: Vec<Ext<K>>,
    /// The claimed $F_0(\mathbf a)$.
    pub claim0: Ext<K>,
    /// The claimed $F_\alpha(\mathbf a)$.
    pub claim_alpha: Ext<K>,
}

/// Verifies the rounds against the claimed sums $0$ and `claim_alpha`, absorbing the messages
/// and drawing the coins as [`prove`] does.
///
/// # Errors
/// [`VerifyError::Malformed`] for messages of the wrong number or length;
/// [`VerifyError::Check`] if a round's values do not add up to the running claim.
pub fn verify<const K: usize>(
    field: &Fqk<K>,
    digits: usize,
    rounds: &[RoundMessage<K>],
    expected_rounds: usize,
    claim_alpha: Ext<K>,
    fs: &mut Fs,
) -> core::result::Result<Reduced<K>, VerifyError> {
    if rounds.len() != expected_rounds {
        return Err(VerifyError::Malformed(format!(
            "{} sumcheck rounds, {expected_rounds} expected",
            rounds.len()
        )));
    }
    let deg0 = digits + 1;
    let mut claim0 = field.zero();
    let mut claim_a = claim_alpha;
    let mut point = Vec::with_capacity(rounds.len());
    for (i, msg) in rounds.iter().enumerate() {
        if msg.h0.len() != deg0 + 1 || msg.h_alpha.len() != 3 {
            return Err(VerifyError::Malformed(format!(
                "round {}: {} and {} values, {} and 3 expected",
                i + 1,
                msg.h0.len(),
                msg.h_alpha.len(),
                deg0 + 1
            )));
        }
        if field.add(&msg.h0[0], &msg.h0[1]) != field.reduce(&claim0) {
            return Err(VerifyError::Check(format!(
                "range sumcheck, round {}",
                i + 1
            )));
        }
        if field.add(&msg.h_alpha[0], &msg.h_alpha[1]) != field.reduce(&claim_a) {
            return Err(VerifyError::Check(format!(
                "linear sumcheck, round {}",
                i + 1
            )));
        }
        fs.exts(LABEL_H0, &msg.h0);
        fs.exts(LABEL_HA, &msg.h_alpha);
        let r = fs.field_coins::<K>(LABEL_ROUND, field.modulus(), 1)?[0];
        claim0 = interpolate(field, &msg.h0, &r);
        claim_a = interpolate(field, &msg.h_alpha, &r);
        point.push(r);
    }
    Ok(Reduced {
        point,
        claim0,
        claim_alpha: claim_a,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const K: usize = 4;
    const Q: u64 = 8_388_637;

    #[test]
    fn the_range_factors_are_the_range_polynomial() {
        // Balanced digit sets of even and odd bases, sets without pairs, and repeated residues.
        let field = Fqk::<K>::new(Q).unwrap();
        let zq = field.zq();
        let digits = |lo: i64, hi: i64| -> Vec<u64> { (lo..=hi).map(|c| zq.from_i64(c)).collect() };
        for (case, set) in [
            digits(-2, 1),
            digits(-8, 7),
            digits(-3, 3),
            digits(-1, 0),
            digits(0, 5),
            vec![],
            vec![7],
            vec![0, 0, 1, Q - 1, 1, Q - 1],
        ]
        .into_iter()
        .enumerate()
        {
            let range = RangeFactors::new(&field, &set);
            let mut y = Ext([3, 141, 59, 26]);
            for i in 0..20u64 {
                assert_eq!(
                    range.eval(&field, &y),
                    range_poly(&field, &set, &y),
                    "case {case}, point {i}"
                );
                y = field.add(&field.mul(&y, &y), &field.from_base(i));
            }
            for c in &set {
                let at_digit = field.from_base(*c);
                assert!(field.is_zero(&range.eval(&field, &at_digit)), "case {case}");
            }
        }
        // The balanced set of base 16 has 7 pairs, and 0 and -8 are left.
        let range = RangeFactors::new(&field, &digits(-8, 7));
        assert_eq!((range.squares.len(), range.single.len()), (7, 2));
    }

    #[test]
    fn the_round_values_are_the_direct_sums() {
        // The incremental nodes and the parallel sums against the definition, node by node.
        let field = Fqk::<K>::new(Q).unwrap();
        let zq = field.zq();
        let digits: Vec<u64> = (-2..=1).map(|c| zq.from_i64(c)).collect();
        let deg0 = digits.len() + 1;
        let table = |seed: u64| -> Vec<Ext<K>> {
            (0..64u64)
                .map(|i| {
                    let x = (i + 1)
                        .wrapping_mul(seed)
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15);
                    Ext(core::array::from_fn(|c| (x >> (8 * c)) % Q))
                })
                .collect()
        };
        let (w, eq0, g) = (table(1), table(2), table(3));
        let range = RangeFactors::new(&field, &digits);
        let (h0, h_alpha) = round_values(&field, &range, deg0, &w, &eq0, &g);
        // The first round's kernel: digits only, then a pair outside the digit set.
        let small = SmallValues::new(&field, &range, &digits, deg0);
        for wild in [false, true] {
            let mut base: Vec<u64> = (0..64u64)
                .map(|i| digits[(i.wrapping_mul(0x9e37_79b9) >> 7) as usize % digits.len()])
                .collect();
            if wild {
                base[5] = 1_234_567;
                base[32 + 5] = Q - 7;
            }
            let lifted: Vec<Ext<K>> = base.iter().map(|x| field.from_base(*x)).collect();
            assert_eq!(
                first_round_values(&field, &range, &small, deg0, &base, &eq0, &g),
                round_values(&field, &range, deg0, &lifted, &eq0, &g),
                "wild {wild}"
            );
        }
        let half = w.len() / 2;
        let at = |tab: &[Ext<K>], x: usize, t: u64| {
            field.add(
                &tab[x],
                &field.scale(&field.sub(&tab[x + half], &tab[x]), t),
            )
        };
        for (t, value) in h0.iter().enumerate() {
            let want = (0..half).fold(field.zero(), |acc, x| {
                let (wt, et) = (at(&w, x, t as u64), at(&eq0, x, t as u64));
                let p = range_poly(&field, &digits, &wt);
                field.add(&acc, &field.mul(&et, &p))
            });
            assert_eq!(*value, want, "F_0 at {t}");
        }
        for (t, value) in h_alpha.iter().enumerate() {
            let want = (0..half).fold(field.zero(), |acc, x| {
                let (wt, gt) = (at(&w, x, t as u64), at(&g, x, t as u64));
                field.add(&acc, &field.mul(&wt, &gt))
            });
            assert_eq!(*value, want, "F_alpha at {t}");
        }
    }
}
