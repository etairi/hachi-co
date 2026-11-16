//! The two sumchecks of one iteration, run with common round challenges (paper, Section 4.3,
//! Fig. 5-7): over $\mathbf x \in \lbrace 0, 1 \rbrace^L$,
//! - the range check $F_0(\mathbf x) = \widetilde{eq}(\tau_0, \mathbf x) \cdot P(\tilde w(\mathbf
//!   x))$ with $P(y) = \prod_{c \in D} (y - c)$ over the digit set $D$ of base $b$, claimed sum
//!   $0$. It runs over every entry of the table, the zeros that pad it to $2^L$ included, with no
//!   indicator: the reduction between levels (`hachi-pcs.tex`, `lem:hachi-transition`) relies on
//!   every entry of the next witness being range-checked;
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
//! $\mathbb Z_q$ until the first challenge. In the second round the node values of a pair of
//! digits are $x_t + r y_t$ for the first challenge $r$ and small integers $x_t, y_t$, so that
//! their $P$ comes from a table of $P(x + r y)$ formed once $r$ is drawn (`PlaneValues`): each
//! node takes one product in place of about $|D|/2$. The pairs of points run on the Rayon pool;
//! field sums do not depend on the order, so the messages do not depend on the number of threads.
//!
//! **Prover's memory.** Both weight tables are products of a row and a column factor
//! ([`Tensor`]): $\tilde g(u \Vert \ell) = m^\ast(u) \alpha^\ell$, and $\widetilde{eq}(\tau_0,
//! \cdot)$ split after the same variable. While a row variable is left, the prover sums each row
//! over its columns first, multiplies the sums by the row factors at $t$, and binds only the row
//! factors, so that it holds $2^r + 2^s$ values of each weight table in place of $2^{r+s}$. The
//! sums are exact, so the messages are those of the dense tables; once the rows are bound, the
//! $2^s$ remaining values of each table are formed and the rounds go on as before. When the second
//! round reads the table of digits (`second_round_rows`), the witness is bound after the second
//! challenge, to $n/4$ values formed from the table (`bind_two`), and the $n/2$ values bound to the
//! first challenge alone are never formed.

use crate::error::VerifyError;
use crate::fs::Fs;
use crate::mle::{fold_top, interpolate};
use crate::table::ResidueTable;
use crate::{Error, Result};
use hachi_co_math::montgomery::{Arith, Montgomery};
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

/// A table of $2^{r + s}$ values whose entry $u 2^s + j$ is `rows[u]` · `cols[j]`, the first $r$
/// variables indexing the rows: the form of both weight tables of an iteration, $\tilde g(u
/// \Vert \ell) = m^\ast(u) \alpha^\ell$ (`crate::relation`) and $\widetilde{eq}(\tau_0, u \Vert
/// j) = \widetilde{eq}(\tau_{0,U}, u)\,\widetilde{eq}(\tau_{0,L}, j)$ for $\tau_0 = (\tau_{0,U},
/// \tau_{0,L})$ split after its first $r$ coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor<const K: usize> {
    /// The $2^r$ row factors.
    pub rows: Vec<Ext<K>>,
    /// The $2^s$ column factors.
    pub cols: Vec<Ext<K>>,
}

impl<const K: usize> Tensor<K> {
    /// Whether the factors have power-of-two lengths whose product is `n`.
    fn fits(&self, n: usize) -> bool {
        self.rows.len().is_power_of_two()
            && self.cols.len().is_power_of_two()
            && self.rows.len().checked_mul(self.cols.len()) == Some(n)
    }

    /// The $2^{r + s}$ values, row by row.
    pub fn dense(&self, field: &Fqk<K>) -> Vec<Ext<K>> {
        self.rows
            .iter()
            .flat_map(|row| self.cols.iter().map(move |col| field.mul(row, col)))
            .collect()
    }
}

/// The weight tables of [`prove`]: as [`Tensor`]s while a row variable is left, then dense.
enum Weights<const K: usize> {
    /// $\widetilde{eq}(\tau_0, \cdot)$ and $\tilde g$ as row and column factors of one split.
    Split { eq0: Tensor<K>, g: Tensor<K> },
    /// The values of both tables.
    Dense { eq0: Vec<Ext<K>>, g: Vec<Ext<K>> },
}

impl<const K: usize> Weights<K> {
    /// The same tables, dense once every row variable is bound.
    fn settle(self, field: &Fqk<K>) -> Self {
        match self {
            Self::Split { eq0, g } if eq0.rows.len() == 1 => Self::Dense {
                eq0: eq0.dense(field),
                g: g.dense(field),
            },
            other => other,
        }
    }

    /// Binds the first variable to `r`.
    fn fold(&mut self, field: &Fqk<K>, r: &Ext<K>) {
        match self {
            Self::Split { eq0, g } => {
                fold_top(field, &mut eq0.rows, r);
                fold_top(field, &mut g.rows, r);
            }
            Self::Dense { eq0, g } => {
                fold_top(field, eq0, r);
                fold_top(field, g, r);
            }
        }
    }
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
    /// $\phi(c^2)$ and $\phi(e)$ in Montgomery's form, $\phi(x) = xR \bmod q$ for every $K$
    /// ([`Montgomery`]), for the nodes' products ([`NodeArith`]); `None` for the canonical
    /// products only.
    entered: Option<(Vec<u64>, Vec<u64>)>,
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
        // phi does not depend on K: the form of K = 1 enters the base-field constants.
        let entered = Montgomery::<1>::new(zq.modulus()).map(|m| {
            let enter = |v: &Vec<u64>| v.iter().map(|c| m.enter_base(*c)).collect();
            (enter(&squares), enter(&single))
        });
        Self {
            squares,
            single,
            entered,
        }
    }

    /// The same factors for the canonical products only (for the tests of both paths).
    #[cfg(test)]
    fn canonical(mut self) -> Self {
        self.entered = None;
        self
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

/// The products of a round's nodes, $e P(w)$ and $w g$ for canonical $e$ and $g$, from $w$ in a
/// working representation: Montgomery's ($\phi(w) = wR$, [`Montgomery`]) where the modulus
/// admits it for $K$, else the canonical one. A pair's nodes $w + t\,dw$ enter it through $w$
/// and $dw$ (two entries per pair: $\phi$ is additive, so the nodes follow by additions), $P$'s
/// factors are multiplied in it ($\phi(x) \phi(y) = \phi(xy)$), and the products by the canonical
/// $e$ and $g$ leave it ($\phi(x) y = xy$): the same field elements as the canonical products,
/// in about half the time of `Fqk::mul`'s divisions (item 139).
struct NodeArith<'a, const K: usize> {
    range: &'a RangeFactors,
    /// Montgomery's form with the factors of `range` entered, if both exist.
    mont: Option<(Montgomery<K>, &'a [u64], &'a [u64])>,
}

impl<'a, const K: usize> NodeArith<'a, K> {
    fn new(field: &Fqk<K>, range: &'a RangeFactors) -> Self {
        let mont = range.entered.as_ref().and_then(|(squares, single)| {
            Montgomery::<K>::new(field.modulus())
                .map(|m| (m, squares.as_slice(), single.as_slice()))
        });
        Self { range, mont }
    }

    /// $w$ in the working representation.
    #[inline(always)]
    fn enter(&self, w: &Ext<K>) -> Ext<K> {
        match &self.mont {
            Some((m, _, _)) => m.enter(w),
            None => *w,
        }
    }

    /// $e P(w)$, canonical, for $w$ in the working representation and a canonical $e$.
    #[inline(always)]
    fn range_times(&self, field: &Fqk<K>, w: &Ext<K>, e: &Ext<K>) -> Ext<K> {
        self.times(field, &self.range_value(field, w), e)
    }

    /// $P(w)$ in the working representation, for $w$ in it.
    #[inline(always)]
    fn range_value(&self, field: &Fqk<K>, w: &Ext<K>) -> Ext<K> {
        let Some((m, squares, single)) = &self.mont else {
            return self.range.eval(field, w);
        };
        // phi(P(w)): the factors phi(w^2) - phi(c^2) and phi(w) - phi(e), the first one taken
        // as it is.
        let mut acc: Option<Ext<K>> = None;
        let w2 = if squares.is_empty() { *w } else { m.mul(w, w) };
        let factors = squares
            .iter()
            .map(|c2| m.sub_base(&w2, *c2))
            .chain(single.iter().map(|c| m.sub_base(w, *c)));
        for factor in factors {
            acc = Some(match acc {
                Some(a) => m.mul(&a, &factor),
                None => factor,
            });
        }
        // P = 1 for an empty digit set.
        acc.unwrap_or_else(|| m.enter(&field.one()))
    }

    /// $w g$, canonical, for $w$ in the working representation and a canonical $g$.
    #[inline(always)]
    fn times(&self, field: &Fqk<K>, w: &Ext<K>, g: &Ext<K>) -> Ext<K> {
        match &self.mont {
            Some((m, _, _)) => m.mul(w, g),
            None => field.mul(w, g),
        }
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

    /// The number of values of $P$ in the [`PlaneValues`] of this reach.
    fn plane_entries(&self) -> usize {
        let reach = self.reach as usize;
        (2 * reach + 1).saturating_mul(4 * reach + 1)
    }
}

/// The range polynomial at the node values of the second round for pairs of digits. After the
/// first challenge $r$, entry $i$ of the bound witness is $a + r (b - a)$ for the entries $a$ and
/// $b$ at $i$ and $i + n/2$ of the table, so that at the second round's pair $(i, i + n/4)$ the
/// node values are $x_t + r y_t$ with $x_t = a + t (c - a)$ and $y_t = (b - a) + t ((e - c) - (b -
/// a))$, where $c$ and $e$ are the entries at $i + n/4$ and $i + 3n/4$. For digits of absolute
/// value at most `digit` and $t \le \deg F_0$ these are integers with $|x_t| \le$ `reach` and
/// $|y_t| \le 2\,$`reach`, the reach of [`SmallValues`].
struct PlaneValues<const K: usize> {
    /// The first challenge $r$.
    r: Ext<K>,
    /// The reach of $x$, that of [`SmallValues`]: $y$ reaches twice as far.
    reach: i64,
    /// The number of values of $y$, $4\,$`reach` $+ 1$.
    width: usize,
    /// $P(x + r y)$ at index $(x + \mathrm{reach})\,\mathrm{width} + y + 2\,\mathrm{reach}$, in
    /// the working representation of `mont` ([`NodeArith`]).
    range: Vec<Ext<K>>,
    /// Montgomery's form of `range`, or `None` for canonical values.
    mont: Option<Montgomery<K>>,
    /// The residue of $v$ for $|v| \le 2\,$`reach`, at index $v + 2\,$`reach`.
    value: Vec<u64>,
}

impl<const K: usize> PlaneValues<K> {
    fn new(field: &Fqk<K>, range: &RangeFactors, small: &SmallValues, r: &Ext<K>) -> Self {
        let zq = field.zq();
        let arith = NodeArith::new(field, range);
        let reach = small.reach;
        let value: Vec<u64> = (-2 * reach..=2 * reach).map(|v| zq.from_i64(v)).collect();
        let width = value.len();
        let ry: Vec<Ext<K>> = value.iter().map(|y| field.scale(r, *y)).collect();
        // Row x + reach holds P(x + r y) for every y; small.value holds x for |x| <= reach.
        let mut values = vec![field.zero(); small.value.len() * width];
        values
            .par_chunks_mut(width)
            .zip(small.value.par_iter())
            .for_each(|(row, x)| {
                let x = field.from_base(*x);
                for (p, ry) in row.iter_mut().zip(&ry) {
                    *p = arith.range_value(field, &arith.enter(&field.add(&x, ry)));
                }
            });
        Self {
            r: *r,
            reach,
            width,
            range: values,
            mont: arith.mont.map(|(m, _, _)| m),
            value,
        }
    }

    /// $e P(x + r y)$, canonical, for the value at `index` and a canonical $e$.
    #[inline(always)]
    fn times(&self, field: &Fqk<K>, index: usize, e: &Ext<K>) -> Ext<K> {
        match &self.mont {
            Some(m) => m.mul(&self.range[index], e),
            None => field.mul(e, &self.range[index]),
        }
    }

    /// $P(x + r y)$, canonical, at `index`.
    #[cfg(test)]
    fn value_at(&self, index: usize) -> Ext<K> {
        match &self.mont {
            Some(m) => m.leave(&self.range[index]),
            None => self.range[index],
        }
    }
}

/// The values of one round's polynomials at $t = 0, \dots, \deg F_0$ and $t = 0, 1, 2$ with the
/// witness table `w` of the first round, over $\mathbb Z_q$: [`round_values`] with the node
/// values of a pair of digits read from `small`, so that each is a product of a field element
/// by a residue.
#[allow(clippy::too_many_arguments)] // the tables, the round's nodes and the factored eq
fn first_round_values<const K: usize, T: ResidueTable + ?Sized>(
    field: &Fqk<K>,
    range: &RangeFactors,
    small: &SmallValues,
    nodes: Nodes,
    w: &T,
    eq0: &[Ext<K>],
    g: &[Ext<K>],
) -> (Vec<Ext<K>>, Vec<Ext<K>>) {
    let arith = NodeArith::new(field, range);
    let half = w.len() / 2;
    let zeros = || (vec![field.zero(); nodes.count], vec![field.zero(); 3]);
    (0..half)
        .into_par_iter()
        .fold(zeros, |(mut h0, mut h_alpha), i| {
            let (mut et, de) = eq_weight(field, nodes.factored, &eq0[i], &eq0[i + half]);
            let mut gt = g[i];
            let dg = field.sub(&g[i + half], &gt);
            let pair = small
                .small(field, w[i])
                .zip(small.small(field, w[i + half]));
            if let Some((lo, hi)) = pair {
                let (mut wt, step) = (lo, hi - lo);
                for (t, h) in h0.iter_mut().enumerate() {
                    let index = (wt + small.reach) as usize;
                    if nodes.evaluates(t) {
                        *h = field.add(h, &field.scale(&et, small.range[index]));
                    }
                    if t < 3 {
                        let product = field.scale(&gt, small.value[index]);
                        h_alpha[t] = field.add(&h_alpha[t], &product);
                    }
                    wt += step;
                    et = field.add(&et, &de);
                    gt = field.add(&gt, &dg);
                }
            } else {
                let mut wt = arith.enter(&field.from_base(w[i]));
                let dw = field.sub(&arith.enter(&field.from_base(w[i + half])), &wt);
                for (t, h) in h0.iter_mut().enumerate() {
                    if nodes.evaluates(t) {
                        *h = field.add(h, &arith.range_times(field, &wt, &et));
                    }
                    if t < 3 {
                        h_alpha[t] = field.add(&h_alpha[t], &arith.times(field, &wt, &gt));
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

/// The nodes at which a round computes its range part: `count` of them from node 0, the weight of
/// $\widetilde{eq}$ factored or not ([`eq_weight`]), and with `skip` one node left out, which the
/// prover forms from the round's claim ([`claim_node`]).
#[derive(Clone, Copy, Debug)]
struct Nodes {
    count: usize,
    factored: bool,
    skip: Option<usize>,
}

impl Nodes {
    /// `count` nodes, each weighed and evaluated.
    #[cfg(test)]
    fn plain(count: usize) -> Self {
        Self {
            count,
            factored: false,
            skip: None,
        }
    }

    /// Whether the range part is evaluated at node `t`.
    #[inline(always)]
    fn evaluates(&self, t: usize) -> bool {
        self.skip != Some(t)
    }
}

/// The weight of $\widetilde{eq}$ at a round's nodes for the entries `lo` and `hi` of a pair
/// (or the row factors of a row and its partner): its value at node 0 and its step, the affine
/// function through `lo` and `hi`, or with `factored` their sum at every node, as the rounds
/// whose $\widetilde{eq}$ factor of the round's variable is taken out ([`unfactor`]) weigh them.
#[inline(always)]
fn eq_weight<const K: usize>(
    field: &Fqk<K>,
    factored: bool,
    lo: &Ext<K>,
    hi: &Ext<K>,
) -> (Ext<K>, Ext<K>) {
    if factored {
        (field.add(lo, hi), field.zero())
    } else {
        (*lo, field.sub(hi, lo))
    }
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
#[allow(clippy::too_many_arguments)] // the tables, the round's nodes and the factored eq
fn round_values<const K: usize>(
    field: &Fqk<K>,
    range: &RangeFactors,
    nodes: Nodes,
    w: &[Ext<K>],
    eq0: &[Ext<K>],
    g: &[Ext<K>],
) -> (Vec<Ext<K>>, Vec<Ext<K>>) {
    let arith = NodeArith::new(field, range);
    let half = w.len() / 2;
    let zeros = || (vec![field.zero(); nodes.count], vec![field.zero(); 3]);
    (0..half)
        .into_par_iter()
        .fold(zeros, |(mut h0, mut h_alpha), i| {
            let (mut wt, mut gt) = (arith.enter(&w[i]), g[i]);
            let (mut et, de) = eq_weight(field, nodes.factored, &eq0[i], &eq0[i + half]);
            let dw = field.sub(&arith.enter(&w[i + half]), &wt);
            let dg = field.sub(&g[i + half], &gt);
            for (t, h) in h0.iter_mut().enumerate() {
                if nodes.evaluates(t) {
                    *h = field.add(h, &arith.range_times(field, &wt, &et));
                }
                if t < 3 {
                    h_alpha[t] = field.add(&h_alpha[t], &arith.times(field, &wt, &gt));
                }
                wt = field.add(&wt, &dw);
                et = field.add(&et, &de);
                gt = field.add(&gt, &dg);
            }
            (h0, h_alpha)
        })
        .reduce(zeros, |x, y| add_values(field, x, &y))
}

/// Adds the sums of row $u$ over its columns, times the row factors at $t$: `s0[t]`
/// $e_t(u)$ to `h0[t]` and `sa[t]` $m_t(u)$ to `h_alpha[t]`, where $e_t(u)$ and $m_t(u)$ are the
/// affine functions of $t$ through the row factors of $u$ ($t = 0$) and $u +$ `half` ($t = 1$)
/// of `eq_rows` and `g_rows`; with `factored`, $e_t(u)$ is the sum of the two at every $t$
/// ([`eq_weight`]).
#[allow(clippy::too_many_arguments)]
fn add_row<const K: usize>(
    field: &Fqk<K>,
    factored: bool,
    eq_rows: &[Ext<K>],
    g_rows: &[Ext<K>],
    u: usize,
    half: usize,
    s0: &[Ext<K>],
    sa: &[Ext<K>],
    (h0, h_alpha): &mut (Vec<Ext<K>>, Vec<Ext<K>>),
) {
    let (mut et, de) = eq_weight(field, factored, &eq_rows[u], &eq_rows[u + half]);
    for (h, s) in h0.iter_mut().zip(s0) {
        *h = field.add(h, &field.mul(&et, s));
        et = field.add(&et, &de);
    }
    let mut gt = g_rows[u];
    let dg = field.sub(&g_rows[u + half], &gt);
    for (h, s) in h_alpha.iter_mut().zip(sa) {
        *h = field.add(h, &field.mul(&gt, s));
        gt = field.add(&gt, &dg);
    }
}

/// [`first_round_values`] with the weights as [`Tensor`]s of at least two rows: for each row $u$
/// of the lower half, the sums over its columns $j$ of $\widetilde{eq}_L[j] P(w_t)$ and
/// $g_L[j] w_t$ (by residues where the pair is small), times the row factors at $t$
/// ([`add_row`]). Exact, so it returns what [`first_round_values`] returns on the dense tables.
#[allow(clippy::too_many_arguments)] // the tables, the round's nodes and the factored eq
fn first_round_rows<const K: usize, T: ResidueTable + ?Sized>(
    field: &Fqk<K>,
    range: &RangeFactors,
    small: &SmallValues,
    nodes: Nodes,
    w: &T,
    eq0: &Tensor<K>,
    g: &Tensor<K>,
) -> (Vec<Ext<K>>, Vec<Ext<K>>) {
    let arith = NodeArith::new(field, range);
    let (cols, half_rows) = (eq0.cols.len(), eq0.rows.len() / 2);
    let half = half_rows * cols;
    let zeros = || (vec![field.zero(); nodes.count], vec![field.zero(); 3]);
    (0..half_rows)
        .into_par_iter()
        .fold(
            || (zeros(), vec![field.zero(); nodes.count]),
            |(mut sums, mut s0), u| {
                s0.fill(field.zero());
                let mut sa = [field.zero(); 3];
                for j in 0..cols {
                    let i = u * cols + j;
                    let (ce, cg) = (&eq0.cols[j], &g.cols[j]);
                    let pair = small
                        .small(field, w[i])
                        .zip(small.small(field, w[i + half]));
                    if let Some((lo, hi)) = pair {
                        let (mut wt, step) = (lo, hi - lo);
                        for (t, s) in s0.iter_mut().enumerate() {
                            let index = (wt + small.reach) as usize;
                            if nodes.evaluates(t) {
                                *s = field.add(s, &field.scale(ce, small.range[index]));
                            }
                            if t < 3 {
                                sa[t] = field.add(&sa[t], &field.scale(cg, small.value[index]));
                            }
                            wt += step;
                        }
                    } else {
                        let mut wt = arith.enter(&field.from_base(w[i]));
                        let dw = field.sub(&arith.enter(&field.from_base(w[i + half])), &wt);
                        for (t, s) in s0.iter_mut().enumerate() {
                            if nodes.evaluates(t) {
                                *s = field.add(s, &arith.range_times(field, &wt, ce));
                            }
                            if t < 3 {
                                sa[t] = field.add(&sa[t], &arith.times(field, &wt, cg));
                            }
                            wt = field.add(&wt, &dw);
                        }
                    }
                }
                add_row(
                    field,
                    nodes.factored,
                    &eq0.rows,
                    &g.rows,
                    u,
                    half_rows,
                    &s0,
                    &sa,
                    &mut sums,
                );
                (sums, s0)
            },
        )
        .map(|(sums, _)| sums)
        .reduce(zeros, |x, y| add_values(field, x, &y))
}

/// [`round_values`] with the weights as [`Tensor`]s of at least two rows, as in
/// [`first_round_rows`].
#[allow(clippy::too_many_arguments)] // the tables, the round's nodes and the factored eq
fn round_rows<const K: usize>(
    field: &Fqk<K>,
    range: &RangeFactors,
    nodes: Nodes,
    w: &[Ext<K>],
    eq0: &Tensor<K>,
    g: &Tensor<K>,
) -> (Vec<Ext<K>>, Vec<Ext<K>>) {
    let arith = NodeArith::new(field, range);
    let (cols, half_rows) = (eq0.cols.len(), eq0.rows.len() / 2);
    let half = half_rows * cols;
    let zeros = || (vec![field.zero(); nodes.count], vec![field.zero(); 3]);
    (0..half_rows)
        .into_par_iter()
        .fold(
            || (zeros(), vec![field.zero(); nodes.count]),
            |(mut sums, mut s0), u| {
                s0.fill(field.zero());
                let mut sa = [field.zero(); 3];
                for j in 0..cols {
                    let i = u * cols + j;
                    let (ce, cg) = (&eq0.cols[j], &g.cols[j]);
                    let mut wt = arith.enter(&w[i]);
                    let dw = field.sub(&arith.enter(&w[i + half]), &wt);
                    for (t, s) in s0.iter_mut().enumerate() {
                        if nodes.evaluates(t) {
                            *s = field.add(s, &arith.range_times(field, &wt, ce));
                        }
                        if t < 3 {
                            sa[t] = field.add(&sa[t], &arith.times(field, &wt, cg));
                        }
                        wt = field.add(&wt, &dw);
                    }
                }
                add_row(
                    field,
                    nodes.factored,
                    &eq0.rows,
                    &g.rows,
                    u,
                    half_rows,
                    &s0,
                    &sa,
                    &mut sums,
                );
                (sums, s0)
            },
        )
        .map(|(sums, _)| sums)
        .reduce(zeros, |x, y| add_values(field, x, &y))
}

/// [`round_rows`] in the second round, from the table of residues and the first challenge in
/// place of the bound witness: where the four entries of a pair are digits, the node values come
/// from `plane` ([`PlaneValues`]), each node taking one product by the column factor of
/// $\widetilde{eq}$, and $F_\alpha$'s sums over the columns are formed from the integers $x_0$,
/// $x_1 - x_0$, $y_0$ and $y_1 - y_0$ (its node values are affine in $t$); other pairs are formed
/// in $\mathbb F_{q^K}$ as the bound witness has them. Exact, so it returns what [`round_rows`]
/// returns on the witness bound to the first challenge.
#[allow(clippy::too_many_arguments)]
fn second_round_rows<const K: usize, T: ResidueTable + ?Sized>(
    field: &Fqk<K>,
    range: &RangeFactors,
    small: &SmallValues,
    plane: &PlaneValues<K>,
    nodes: Nodes,
    table: &T,
    eq0: &Tensor<K>,
    g: &Tensor<K>,
) -> (Vec<Ext<K>>, Vec<Ext<K>>) {
    let arith = NodeArith::new(field, range);
    let zq = field.zq();
    let (cols, half_rows) = (eq0.cols.len(), eq0.rows.len() / 2);
    // The pairs of the bound witness are (i, i + n/4), and its entry i is formed from the entries
    // i and i + n/2 of the table.
    let quarter = half_rows * cols;
    let half = 2 * quarter;
    debug_assert_eq!(table.len(), 2 * half);
    let r = &plane.r;
    let (reach, width) = (plane.reach, plane.width as i64);
    let value = |v: i64| plane.value[(v + 2 * reach) as usize];
    let zeros = || (vec![field.zero(); nodes.count], vec![field.zero(); 3]);
    (0..half_rows)
        .into_par_iter()
        .fold(
            || (zeros(), vec![field.zero(); nodes.count]),
            |(mut sums, mut s0), u| {
                s0.fill(field.zero());
                // F_alpha's sums over the pairs of digits, of g_L[j] times x_0, x_1 - x_0, y_0
                // and y_1 - y_0, and over the other pairs at t = 0, 1, 2.
                let mut ints = [field.zero(); 4];
                let mut other = [field.zero(); 3];
                for j in 0..cols {
                    let i = u * cols + j;
                    let (ce, cg) = (&eq0.cols[j], &g.cols[j]);
                    let (a, b) = (table[i], table[i + half]);
                    let (c, e) = (table[i + quarter], table[i + quarter + half]);
                    let four = small
                        .small(field, a)
                        .zip(small.small(field, b))
                        .zip(small.small(field, c).zip(small.small(field, e)));
                    if let Some(((a, b), (c, e))) = four {
                        let (x0, dx) = (a, c - a);
                        let (y0, dy) = (b - a, (e - c) - (b - a));
                        let mut index = (x0 + reach) * width + y0 + 2 * reach;
                        let step = dx * width + dy;
                        for (t, s) in s0.iter_mut().enumerate() {
                            if nodes.evaluates(t) {
                                *s = field.add(s, &plane.times(field, index as usize, ce));
                            }
                            index += step;
                        }
                        for (sum, v) in ints.iter_mut().zip([x0, dx, y0, dy]) {
                            *sum = field.add(sum, &field.scale(cg, value(v)));
                        }
                    } else {
                        let lo = field.add(&field.from_base(a), &field.scale(r, zq.sub(b, a)));
                        let next = field.add(&field.from_base(c), &field.scale(r, zq.sub(e, c)));
                        let mut wt = arith.enter(&lo);
                        let dw = field.sub(&arith.enter(&next), &wt);
                        for (t, s) in s0.iter_mut().enumerate() {
                            if nodes.evaluates(t) {
                                *s = field.add(s, &arith.range_times(field, &wt, ce));
                            }
                            if t < 3 {
                                other[t] = field.add(&other[t], &arith.times(field, &wt, cg));
                            }
                            wt = field.add(&wt, &dw);
                        }
                    }
                }
                // At t the pairs of digits give (X_0 + t DX) + r (Y_0 + t DY).
                let sa: [Ext<K>; 3] = core::array::from_fn(|t| {
                    let x = field.add(&ints[0], &field.scale(&ints[1], t as u64));
                    let y = field.add(&ints[2], &field.scale(&ints[3], t as u64));
                    field.add(&other[t], &field.add(&x, &field.mul(r, &y)))
                });
                add_row(
                    field,
                    nodes.factored,
                    &eq0.rows,
                    &g.rows,
                    u,
                    half_rows,
                    &s0,
                    &sa,
                    &mut sums,
                );
                (sums, s0)
            },
        )
        .map(|(sums, _)| sums)
        .reduce(zeros, |x, y| add_values(field, x, &y))
}

/// Entry `i` of the table bound to the first two challenges `r0` and `r1` ([`bind_two`]), from
/// the table of residues whose quarter is `quarter`, with `r01` $= r_0 r_1$.
#[allow(clippy::too_many_arguments)] // the table, its quarter, the challenges and the entry
fn bound_two_at<const K: usize, T: ResidueTable + ?Sized>(
    field: &Fqk<K>,
    table: &T,
    quarter: usize,
    r0: &Ext<K>,
    r1: &Ext<K>,
    r01: &Ext<K>,
    i: usize,
) -> Ext<K> {
    let quad = (
        table[i],
        table[i + 2 * quarter],
        table[i + quarter],
        table[i + 3 * quarter],
    );
    bound_two_of(field, quad, r0, r1, r01)
}

/// The value bound to `r0` and `r1` of the residues $(a, b, c, e)$ at $i$, $i + n/2$, $i + n/4$
/// and $i + 3n/4$ ([`bound_two_at`]): $a + r_0 (b - a) + r_1 (c - a) + r_0 r_1 ((e - c) - (b - a))$.
#[inline(always)]
fn bound_two_of<const K: usize>(
    field: &Fqk<K>,
    (a, b, c, e): (u64, u64, u64, u64),
    r0: &Ext<K>,
    r1: &Ext<K>,
    r01: &Ext<K>,
) -> Ext<K> {
    let zq = field.zq();
    let ba = zq.sub(b, a);
    let ca = zq.sub(c, a);
    let cross = zq.sub(zq.sub(e, c), ba);
    let v = field.add(&field.from_base(a), &field.scale(r0, ba));
    let v = field.add(&v, &field.scale(r1, ca));
    field.add(&v, &field.scale(r01, cross))
}

/// The values bound to the first two challenges ([`bound_two_of`]) of every quadruple of entries
/// whose centered values are at most `digit` in absolute value, as the third round and
/// [`bind_three`] read them instead of forming them: the table of an iteration holds digits only.
/// Each is formed from the residues of its integers, those of the table, so it is the same value.
struct QuadValues<const K: usize> {
    digit: i64,
    width: usize,
    values: Vec<Ext<K>>,
}

impl<const K: usize> QuadValues<K> {
    fn new(field: &Fqk<K>, digit: i64, r0: &Ext<K>, r1: &Ext<K>) -> Self {
        let zq = field.zq();
        let width = (2 * digit + 1) as usize;
        let r01 = field.mul(r0, r1);
        let residue = |k: usize| zq.from_i64(k as i64 - digit);
        let values = (0..width.pow(4))
            .into_par_iter()
            .map(|index| {
                let (ab, ce) = (index / (width * width), index % (width * width));
                let quad = (
                    residue(ab / width),
                    residue(ab % width),
                    residue(ce / width),
                    residue(ce % width),
                );
                bound_two_of(field, quad, r0, r1, &r01)
            })
            .collect();
        Self {
            digit,
            width,
            values,
        }
    }

    /// Entry `i` of [`bind_two`]'s table, if its four entries are small.
    #[inline(always)]
    fn at<T: ResidueTable + ?Sized>(
        &self,
        field: &Fqk<K>,
        table: &T,
        quarter: usize,
        i: usize,
    ) -> Option<Ext<K>> {
        let zq = field.zq();
        let mut index = 0usize;
        for x in [
            table[i],
            table[i + 2 * quarter],
            table[i + quarter],
            table[i + 3 * quarter],
        ] {
            let c = zq.to_centered(x);
            if c.abs() > self.digit {
                return None;
            }
            index = index * self.width + (c + self.digit) as usize;
        }
        Some(self.values[index])
    }
}

/// Entry `i` of [`bind_two`]'s table: from `quads` where its entries are small, formed otherwise.
#[allow(clippy::too_many_arguments)] // the table, its quarter, the challenges and the entry
#[inline(always)]
fn bound_two<const K: usize, T: ResidueTable + ?Sized>(
    field: &Fqk<K>,
    quads: Option<&QuadValues<K>>,
    table: &T,
    quarter: usize,
    r0: &Ext<K>,
    r1: &Ext<K>,
    r01: &Ext<K>,
    i: usize,
) -> Ext<K> {
    quads
        .and_then(|q| q.at(field, table, quarter, i))
        .unwrap_or_else(|| bound_two_at(field, table, quarter, r0, r1, r01, i))
}

/// The table bound to the first two challenges $r_0$ and $r_1$, from the table of residues:
/// entry $i$ is $a + r_0 (b - a) + r_1 (c - a) + r_0 r_1 ((e - c) - (b - a))$ for the entries $a$,
/// $b$, $c$, $e$ at $i$, $i + n/2$, $i + n/4$ and $i + 3n/4$, the value that binding the first
/// variable to $r_0$ and then the next to $r_1$ ([`fold_top`]) gives.
fn bind_two<const K: usize, T: ResidueTable + ?Sized>(
    field: &Fqk<K>,
    table: &T,
    r0: &Ext<K>,
    r1: &Ext<K>,
) -> Vec<Ext<K>> {
    let quarter = table.len() / 4;
    let r01 = field.mul(r0, r1);
    (0..quarter)
        .into_par_iter()
        .map(|i| bound_two_at(field, table, quarter, r0, r1, &r01, i))
        .collect()
}

/// The table bound to the first three challenges, from the table of residues: entry $i$ is
/// $w_2(i) + r_2 (w_2(i + n/8) - w_2(i))$ for the entries $w_2$ of [`bind_two`], formed from the
/// residues as they are read, so that the table bound to two challenges is never held: the value
/// that binding the three variables one after the other gives.
fn bind_three<const K: usize, T: ResidueTable + ?Sized>(
    field: &Fqk<K>,
    quads: Option<&QuadValues<K>>,
    table: &T,
    r0: &Ext<K>,
    r1: &Ext<K>,
    r2: &Ext<K>,
) -> Vec<Ext<K>> {
    let quarter = table.len() / 4;
    let eighth = quarter / 2;
    let r01 = field.mul(r0, r1);
    (0..eighth)
        .into_par_iter()
        .map(|i| {
            let low = bound_two(field, quads, table, quarter, r0, r1, &r01, i);
            let high = bound_two(field, quads, table, quarter, r0, r1, &r01, i + eighth);
            field.add(&low, &field.mul(r2, &field.sub(&high, &low)))
        })
        .collect()
}

/// [`round_rows`] in the third round, from the table of residues and the first two challenges in
/// place of the witness bound to them: each entry of a pair is formed as [`bind_two`] forms it
/// when the round reads it, so that the bound table is not held. Exact, so it returns what
/// [`round_rows`] returns on the witness of [`bind_two`].
#[allow(clippy::too_many_arguments)] // the round's inputs and the two challenges
fn third_round_rows<const K: usize, T: ResidueTable + ?Sized>(
    field: &Fqk<K>,
    range: &RangeFactors,
    nodes: Nodes,
    quads: Option<&QuadValues<K>>,
    table: &T,
    r0: &Ext<K>,
    r1: &Ext<K>,
    eq0: &Tensor<K>,
    g: &Tensor<K>,
) -> (Vec<Ext<K>>, Vec<Ext<K>>) {
    let arith = NodeArith::new(field, range);
    let (cols, half_rows) = (eq0.cols.len(), eq0.rows.len() / 2);
    let half = half_rows * cols;
    let quarter = table.len() / 4;
    debug_assert_eq!(quarter, 2 * half);
    let r01 = field.mul(r0, r1);
    let zeros = || (vec![field.zero(); nodes.count], vec![field.zero(); 3]);
    (0..half_rows)
        .into_par_iter()
        .fold(
            || (zeros(), vec![field.zero(); nodes.count]),
            |(mut sums, mut s0), u| {
                s0.fill(field.zero());
                let mut sa = [field.zero(); 3];
                for j in 0..cols {
                    let i = u * cols + j;
                    let (ce, cg) = (&eq0.cols[j], &g.cols[j]);
                    let lo = bound_two(field, quads, table, quarter, r0, r1, &r01, i);
                    let next = bound_two(field, quads, table, quarter, r0, r1, &r01, i + half);
                    let mut wt = arith.enter(&lo);
                    let dw = field.sub(&arith.enter(&next), &wt);
                    for (t, s) in s0.iter_mut().enumerate() {
                        if nodes.evaluates(t) {
                            *s = field.add(s, &arith.range_times(field, &wt, ce));
                        }
                        if t < 3 {
                            sa[t] = field.add(&sa[t], &arith.times(field, &wt, cg));
                        }
                        wt = field.add(&wt, &dw);
                    }
                }
                add_row(
                    field,
                    nodes.factored,
                    &eq0.rows,
                    &g.rows,
                    u,
                    half_rows,
                    &s0,
                    &sa,
                    &mut sums,
                );
                (sums, s0)
            },
        )
        .map(|(sums, _)| sums)
        .reduce(zeros, |x, y| add_values(field, x, &y))
}

/// The second round reads the table of residues ([`second_round_rows`]) when the values of
/// [`PlaneValues`] are at most $n /$ `PLANE_SHARE` for a table of $n$ values: forming them then
/// costs a small part of the products they save, and holding them a small part of the bound copy.
const PLANE_SHARE: usize = 16;

/// The round labels of the transcript.
const LABEL_H0: &[u8] = b"hachi/sumcheck/h0";
const LABEL_HA: &[u8] = b"hachi/sumcheck/h_alpha";
const LABEL_ROUND: &[u8] = b"hachi/sumcheck/round";

/// Runs the prover on the table of $\tilde w$ ($2^L$ values over $\mathbb Z_q$) and the weight
/// tables $\widetilde{eq}(\tau_0, \cdot)$ and $\tilde g$, given as [`Tensor`]s of one split, and
/// returns the round messages, the point $\mathbf a$ and $\tilde w(\mathbf a)$.
///
/// # Errors
/// [`Error::Shape`] if the table's length is not a power of two, or the weights do not split it
/// in one way.
pub fn prove<const K: usize, T: ResidueTable + ?Sized>(
    field: &Fqk<K>,
    digits: &[u64],
    table: &T,
    eq0: Tensor<K>,
    g: Tensor<K>,
    fs: &mut Fs,
) -> Result<(Vec<RoundMessage<K>>, Vec<Ext<K>>, Ext<K>)> {
    prove_rounds(field, digits, table, eq0, g, fs, PLANE_SHARE, None)
}

/// [`prove`] for weights whose $\widetilde{eq}$ table is that of the point `tau` $=
/// (\tau_{0,U}, \tau_{0,L})$: `eq0.rows` $= \widetilde{eq}(\tau_{0,U}, \cdot)$ and `eq0.cols`
/// $= \widetilde{eq}(\tau_{0,L}, \cdot)$, as an iteration forms them. Each round takes the
/// factor $\widetilde{eq}(\tau_k, t)$ of its variable out of the range part, which it then
/// computes at one node fewer (`unfactor`), and where $\tau_k \ne 0$ forms node 1 from the
/// round's claim (`claim_node`) rather than computing it: the same messages, transcript and
/// point for a table of digits, an honest prover's, whose range part sums to the claims the
/// verifier keeps (0 before the first round). For a table with other entries the rounds still
/// match the claims, and the verifier's last check refuses the proof.
///
/// # Errors
/// As [`prove`], and [`Error::Shape`] unless `tau` has one coordinate per variable.
pub fn prove_at<const K: usize, T: ResidueTable + ?Sized>(
    field: &Fqk<K>,
    digits: &[u64],
    table: &T,
    tau: &[Ext<K>],
    eq0: Tensor<K>,
    g: Tensor<K>,
    fs: &mut Fs,
) -> Result<(Vec<RoundMessage<K>>, Vec<Ext<K>>, Ext<K>)> {
    prove_rounds_with(
        field,
        digits,
        table,
        eq0,
        g,
        fs,
        PLANE_SHARE,
        Some(tau),
        true,
    )
}

/// $Q(1)$ from the round's claim: $h_0(0) + h_0(1) = \widetilde{eq}(\tau_k, 0) Q(0) +
/// \widetilde{eq}(\tau_k, 1) Q(1)$ is the running claim, so $Q(1) = (\mathrm{claim} - (1 - \tau_k)
/// Q(0)) / \tau_k$ for $\tau_k \neq 0$.
///
/// # Errors
/// [`Error::Shape`] for $\tau_k = 0$ (the caller computes that node directly).
fn claim_node<const K: usize>(
    field: &Fqk<K>,
    claim: &Ext<K>,
    q0: &Ext<K>,
    tau: &Ext<K>,
) -> Result<Ext<K>> {
    let inv = field
        .inv(tau)
        .ok_or_else(|| Error::Shape("the claim's node needs tau_k != 0".into()))?;
    let rest = field.sub(claim, &field.mul(&field.sub(&field.one(), tau), q0));
    Ok(field.mul(&rest, &inv))
}

/// $\binom{m}{j} \bmod q$ for $j \le m$, by Pascal's rule: exact for any $m$.
fn binomial_row(zq: &hachi_co_math::Zq, m: usize) -> Vec<u64> {
    let mut row = vec![1u64];
    for i in 1..=m {
        for j in (1..i).rev() {
            row[j] = zq.add(row[j], row[j - 1]);
        }
        row.push(1);
    }
    row
}

/// The range part's values at the nodes $0, \dots, m$ from `q`, its values without the
/// $\widetilde{eq}$ factor of the round's variable at the nodes $0, \dots, m - 1$: the rows of
/// $\widetilde{eq}(\tau_0, \cdot)$ at the round are $c\,\widetilde{eq}(\tau', u)$ times
/// $1 - \tau_k$ and $\tau_k$ for the two halves, so the weight of row $u$ at node $t$ is
/// $\widetilde{eq}(\tau_k, t)$ times the sum of the two ([`eq_weight`]), and the range part is
/// $h_0(t) = \widetilde{eq}(\tau_k, t) Q(t)$ with $Q$ of degree $m - 1$. Its $m$-th difference
/// vanishes, so $Q(m) = \sum_{j < m} (-1)^{m - 1 - j} \binom{m}{j} Q(j)$ (`binomials` the row of
/// $m$). Exact, so the same values as the rounds that weigh each node.
fn unfactor<const K: usize>(
    field: &Fqk<K>,
    q: &[Ext<K>],
    tau: &Ext<K>,
    binomials: &[u64],
) -> Vec<Ext<K>> {
    let m = q.len();
    let mut last = field.zero();
    for (j, qj) in q.iter().enumerate() {
        let term = field.scale(qj, binomials[j]);
        last = if (m - 1 - j).is_multiple_of(2) {
            field.add(&last, &term)
        } else {
            field.sub(&last, &term)
        };
    }
    let one = field.one();
    let mut et = field.sub(&one, tau);
    let de = field.sub(&field.add(tau, tau), &one);
    q.iter()
        .chain(core::iter::once(&last))
        .map(|qt| {
            let v = field.mul(&et, qt);
            et = field.add(&et, &de);
            v
        })
        .collect()
}

/// [`prove`], with the second round read from the table when [`PlaneValues`] has at most $n /$
/// `plane_share` values (and the weights are still split then), and with `tau` the point of
/// $\widetilde{eq}$ ([`prove_at`]).
#[allow(clippy::too_many_arguments)] // the inputs of prove, the plane's share and the point
fn prove_rounds<const K: usize, T: ResidueTable + ?Sized>(
    field: &Fqk<K>,
    digits: &[u64],
    table: &T,
    eq0: Tensor<K>,
    g: Tensor<K>,
    fs: &mut Fs,
    plane_share: usize,
    tau: Option<&[Ext<K>]>,
) -> Result<(Vec<RoundMessage<K>>, Vec<Ext<K>>, Ext<K>)> {
    prove_rounds_with(field, digits, table, eq0, g, fs, plane_share, tau, false)
}

/// [`prove_rounds`], with `claim` the factored rounds leaving node 1 out of the range part and
/// forming it from the round's claim ([`claim_node`]), as [`prove_at`] does: the same messages
/// for a table of digits, whose range part sums to 0 over the hypercube.
#[allow(clippy::too_many_arguments)] // the inputs of prove_rounds and the claim's use
fn prove_rounds_with<const K: usize, T: ResidueTable + ?Sized>(
    field: &Fqk<K>,
    digits: &[u64],
    table: &T,
    eq0: Tensor<K>,
    g: Tensor<K>,
    fs: &mut Fs,
    plane_share: usize,
    tau: Option<&[Ext<K>]>,
    claim: bool,
) -> Result<(Vec<RoundMessage<K>>, Vec<Ext<K>>, Ext<K>)> {
    let n = table.len();
    if !n.is_power_of_two() || !eq0.fits(n) || !g.fits(n) || eq0.cols.len() != g.cols.len() {
        return Err(Error::Shape(
            "the sumcheck tables must have one power-of-two length and the weights one split"
                .into(),
        ));
    }
    let rounds = n.trailing_zeros() as usize;
    let deg0 = digits.len() + 1;
    if tau.is_some_and(|tau| tau.len() != rounds) {
        return Err(Error::Shape(
            "the point of eq0 must have one coordinate per variable".into(),
        ));
    }
    // With the point of eq0, the range part is computed at deg0 nodes and extended ([`unfactor`]);
    // the linear part needs three nodes, so deg0 >= 3 (two digits or more).
    let factor = tau.filter(|_| deg0 >= 3);
    let binomials = binomial_row(field.zq(), deg0);
    let (count, factored) = if factor.is_some() {
        (deg0, true)
    } else {
        (deg0 + 1, false)
    };
    // The range part's running claim, 0 before the first round, as the verifier keeps it.
    let mut claim0 = field.zero();
    let range = RangeFactors::new(field, digits);
    let small = SmallValues::new(field, &range, digits, deg0);
    let mut messages = Vec::with_capacity(rounds);
    let mut point = Vec::with_capacity(rounds);
    let mut weights = Weights::Split { eq0, g };
    // The second round reads the table and the first challenge, and the witness is first bound
    // after the second challenge, when the weights are still split then (four rows or more now)
    // and the plane of values of P is small next to the table.
    let from_table = rounds >= 2
        && matches!(&weights, Weights::Split { eq0, .. } if eq0.rows.len() >= 4)
        && small.plane_entries().saturating_mul(plane_share) <= n;
    let mut plane: Option<PlaneValues<K>> = None;
    // The first two challenges, while the witness bound to them is not formed: the third round
    // reads the table too, and the three variables are bound at once after it.
    let mut two: Option<(Ext<K>, Ext<K>)> = None;
    // The values bound to those two challenges of the quadruples of digits (`QuadValues`).
    let mut quads: Option<QuadValues<K>> = None;
    // Empty until the first challenge (or the second or third, with `from_table`), which takes
    // it out of Z_q.
    let mut w: Vec<Ext<K>> = Vec::new();
    for round in 0..rounds {
        // Node 1 from the claim where tau_k is invertible.
        let skip = factor
            .filter(|tau| claim && !field.is_zero(&tau[round]))
            .map(|_| 1usize);
        let nodes = Nodes {
            count,
            factored,
            skip,
        };
        weights = weights.settle(field);
        if let (Some((r0, r1)), Weights::Dense { .. }) = (&two, &weights) {
            // The weights are dense already: the third round reads the bound witness.
            w = bind_two(field, table, r0, r1);
            two = None;
        }
        let (mut h0, h_alpha) = match (&weights, round, &plane, &two) {
            (Weights::Split { eq0, g }, 0, _, _) => {
                first_round_rows(field, &range, &small, nodes, table, eq0, g)
            }
            (Weights::Split { eq0, g }, 1, Some(plane), _) => {
                second_round_rows(field, &range, &small, plane, nodes, table, eq0, g)
            }
            (Weights::Split { eq0, g }, 2, _, Some((r0, r1))) => {
                third_round_rows(field, &range, nodes, quads.as_ref(), table, r0, r1, eq0, g)
            }
            (Weights::Split { eq0, g }, _, _, _) => round_rows(field, &range, nodes, &w, eq0, g),
            (Weights::Dense { eq0, g }, 0, _, _) => {
                first_round_values(field, &range, &small, nodes, table, eq0, g)
            }
            (Weights::Dense { eq0, g }, _, _, _) => round_values(field, &range, nodes, &w, eq0, g),
        };
        if let Some(tau) = factor {
            if skip.is_some() {
                h0[1] = claim_node(field, &claim0, &h0[0], &tau[round])?;
            }
            h0 = unfactor(field, &h0, &tau[round], &binomials);
        }
        fs.exts(LABEL_H0, &h0);
        fs.exts(LABEL_HA, &h_alpha);
        let r = fs.field_coins::<K>(LABEL_ROUND, field.modulus(), 1)?[0];
        if claim {
            claim0 = interpolate(field, &h0, &r);
        }
        if round == 0 && from_table {
            plane = Some(PlaneValues::new(field, &range, &small, &r));
        } else if round == 0 {
            let half = n / 2;
            let zq = field.zq();
            w = (0..half)
                .into_par_iter()
                .map(|i| {
                    let (a, b) = (table[i], table[i + half]);
                    field.add(&field.from_base(a), &field.scale(&r, zq.sub(b, a)))
                })
                .collect();
        } else if let Some(first) = plane.take() {
            // The second round read the table: both variables are bound at once, or with the
            // third one after the third round if there is one.
            if rounds >= 3 {
                quads = Some(QuadValues::new(field, small.digit, &first.r, &r));
                two = Some((first.r, r));
            } else {
                w = bind_two(field, table, &first.r, &r);
            }
        } else if let Some((r0, r1)) = two.take() {
            // The third round read the table: the three variables are bound at once.
            w = bind_three(field, quads.take().as_ref(), table, &r0, &r1, &r);
        } else {
            fold_top(field, &mut w, &r);
        }
        weights.fold(field, &r);
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
    fn the_node_products_are_the_canonical_products() {
        // NodeArith's e P(w) and w g, with w entered and moved by additions, against the
        // canonical products of range_poly, for digit sets of bases 2 to 16 and the empty set, at
        // the test modulus, at secure-l24's and at a modulus too large for Montgomery's lazy
        // reduction at K = 4 (the canonical products then), on random and boundary values.
        use hachi_co_math::zq::is_prime;
        let mut large = (1u64 << 62) + 5;
        while !is_prime(large) {
            large += 8;
        }
        assert!(Montgomery::<K>::new(large).is_none());
        for q in [Q, 15395032342003157, large] {
            let field = Fqk::<K>::new(q).unwrap();
            let zq = field.zq();
            let special = [0, 1, q - 1, (q - 1) / 2, q / 3];
            let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ q;
            let mut next = |i: usize, c: usize| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                if i.is_multiple_of(3) {
                    special[(i / 3 + c) % special.len()]
                } else {
                    state % q
                }
            };
            for b in [0i64, 2, 3, 4, 8, 16] {
                let digits: Vec<u64> = (-(b / 2)..(b - b / 2)).map(|c| zq.from_i64(c)).collect();
                let range = RangeFactors::new(&field, &digits);
                let arith = NodeArith::new(&field, &range);
                assert_eq!(arith.mont.is_some(), q != large, "q = {q}");
                let canonical = range.clone().canonical();
                let plain = NodeArith::new(&field, &canonical);
                assert!(plain.mont.is_none());
                for i in 0..200 {
                    let mut ext = |j: usize| Ext::<K>(core::array::from_fn(|c| next(i + j, c)));
                    let (lo, hi, e, g) = (ext(0), ext(1), ext(2), ext(3));
                    let (mut wt, mut wc) = (arith.enter(&lo), plain.enter(&lo));
                    let (dw, dc) = (
                        field.sub(&arith.enter(&hi), &wt),
                        field.sub(&plain.enter(&hi), &wc),
                    );
                    let mut w = lo;
                    for t in 0..4 {
                        let want = field.mul(&e, &range_poly(&field, &digits, &w));
                        assert_eq!(
                            arith.range_times(&field, &wt, &e),
                            want,
                            "q {q}, b {b}, t {t}"
                        );
                        assert_eq!(plain.range_times(&field, &wc, &e), want, "q {q}, b {b}");
                        assert_eq!(arith.times(&field, &wt, &g), field.mul(&w, &g), "q {q}");
                        assert_eq!(plain.times(&field, &wc, &g), field.mul(&w, &g), "q {q}");
                        wt = field.add(&wt, &dw);
                        wc = field.add(&wc, &dc);
                        w = field.add(&w, &field.sub(&hi, &lo));
                    }
                }
            }
        }
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
        let (h0, h_alpha) = round_values(&field, &range, Nodes::plain(deg0 + 1), &w, &eq0, &g);
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
                first_round_values(
                    &field,
                    &range,
                    &small,
                    Nodes::plain(deg0 + 1),
                    &base,
                    &eq0,
                    &g
                ),
                round_values(&field, &range, Nodes::plain(deg0 + 1), &lifted, &eq0, &g),
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

    /// `n` pseudorandom elements of $\mathbb F_{q^K}$.
    fn exts(n: usize, seed: u64) -> Vec<Ext<K>> {
        (0..n as u64)
            .map(|i| {
                let x = (i + 1)
                    .wrapping_mul(seed)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15);
                Ext(core::array::from_fn(|c| (x >> (8 * c)) % Q))
            })
            .collect()
    }

    /// A table of `n` pseudorandom digits of `digits`, with residues outside the digit set at the
    /// indices `wild` (modulo `n`).
    fn digit_table(n: usize, digits: &[u64], wild: &[usize]) -> Vec<u64> {
        let mut table: Vec<u64> = (0..n as u64)
            .map(|i| digits[(i.wrapping_mul(0x9e37_79b9) >> 7) as usize % digits.len()])
            .collect();
        for (k, i) in wild.iter().enumerate() {
            table[i % n] = 1_234_567 + k as u64;
        }
        table
    }

    /// The balanced digits of base 8 as residues.
    fn base8(field: &Fqk<K>) -> Vec<u64> {
        (-4..=3).map(|c| field.zq().from_i64(c)).collect()
    }

    /// The prover on the dense weight tables, as it was before [`Tensor`], with the canonical
    /// products ([`RangeFactors::canonical`]): the reference of the split prover and of its
    /// products in Montgomery's form.
    fn prove_dense(
        field: &Fqk<K>,
        digits: &[u64],
        table: &[u64],
        mut eq0: Vec<Ext<K>>,
        mut g: Vec<Ext<K>>,
        fs: &mut Fs,
    ) -> (Vec<RoundMessage<K>>, Vec<Ext<K>>, Ext<K>) {
        let n = table.len();
        let deg0 = digits.len() + 1;
        let range = RangeFactors::new(field, digits).canonical();
        let small = SmallValues::new(field, &range, digits, deg0);
        let (mut messages, mut point) = (Vec::new(), Vec::new());
        let mut w: Vec<Ext<K>> = Vec::new();
        for round in 0..n.trailing_zeros() {
            let (h0, h_alpha) = if round == 0 {
                first_round_values(
                    field,
                    &range,
                    &small,
                    Nodes::plain(deg0 + 1),
                    table,
                    &eq0,
                    &g,
                )
            } else {
                round_values(field, &range, Nodes::plain(deg0 + 1), &w, &eq0, &g)
            };
            fs.exts(LABEL_H0, &h0);
            fs.exts(LABEL_HA, &h_alpha);
            let r = fs
                .field_coins::<K>(LABEL_ROUND, field.modulus(), 1)
                .unwrap()[0];
            if round == 0 {
                let (lo, hi) = table.split_at(n / 2);
                let zq = field.zq();
                w = lo
                    .iter()
                    .zip(hi)
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
        (messages, point, value)
    }

    #[test]
    fn the_split_kernels_are_the_dense_ones() {
        // Each row's sums times its row factors against the kernels on the products, for digit
        // tables with and without residues outside the digit set, and for field tables, with a
        // zero row of g as m* has after its columns.
        let field = Fqk::<K>::new(Q).unwrap();
        let digits = base8(&field);
        let deg0 = digits.len() + 1;
        let range = RangeFactors::new(&field, &digits);
        let small = SmallValues::new(&field, &range, &digits, deg0);
        for (rows, cols) in [(2, 1), (2, 8), (4, 4), (8, 2), (16, 1), (32, 8)] {
            let n = rows * cols;
            let eq0 = Tensor {
                rows: exts(rows, 2),
                cols: exts(cols, 3),
            };
            let mut g = Tensor {
                rows: exts(rows, 4),
                cols: exts(cols, 5),
            };
            g.rows[rows - 1] = field.zero();
            let (eq_dense, g_dense) = (eq0.dense(&field), g.dense(&field));
            for wild in [&[][..], &[0, n / 2 + 1, n - 1][..]] {
                let w = digit_table(n, &digits, wild);
                assert_eq!(
                    first_round_rows(&field, &range, &small, Nodes::plain(deg0 + 1), &w, &eq0, &g),
                    first_round_values(
                        &field,
                        &range,
                        &small,
                        Nodes::plain(deg0 + 1),
                        &w,
                        &eq_dense,
                        &g_dense
                    ),
                    "first round, {rows} x {cols}, wild {wild:?}"
                );
            }
            let w = exts(n, 6);
            assert_eq!(
                round_rows(&field, &range, Nodes::plain(deg0 + 1), &w, &eq0, &g),
                round_values(
                    &field,
                    &range,
                    Nodes::plain(deg0 + 1),
                    &w,
                    &eq_dense,
                    &g_dense
                ),
                "later round, {rows} x {cols}"
            );
        }
    }

    #[test]
    fn the_prover_is_the_dense_prover() {
        // Messages, point, value and transcript of the prover on split weights against the prover
        // on their products: splits without rows (dense from the start), without columns, and
        // with both, a table of one value (no round), and tables with residues outside the digit
        // set.
        let field = Fqk::<K>::new(Q).unwrap();
        let digits = base8(&field);
        for (rows, cols) in [
            (1, 1),
            (1, 16),
            (2, 8),
            (4, 4),
            (16, 2),
            (32, 1),
            (8, 8),
            (64, 16),
        ] {
            let n = rows * cols;
            for wild in [&[][..], &[1, 3, n / 2][..]] {
                let w = digit_table(n, &digits, wild);
                let eq0 = Tensor {
                    rows: exts(rows, 7),
                    cols: exts(cols, 8),
                };
                let g = Tensor {
                    rows: exts(rows, 9),
                    cols: exts(cols, 10),
                };
                let mut fs_dense = Fs::new(&[3; 32]);
                let mut fs_split = fs_dense.clone();
                let (eq_dense, g_dense) = (eq0.dense(&field), g.dense(&field));
                let dense = prove_dense(&field, &digits, &w, eq_dense, g_dense, &mut fs_dense);
                let split = prove(&field, &digits, &w, eq0, g, &mut fs_split).unwrap();
                assert_eq!(split, dense, "{rows} x {cols}, wild {wild:?}");
                assert_eq!(fs_split.digest(), fs_dense.digest(), "{rows} x {cols}");
            }
        }
    }

    /// The balanced digits of base `b` as residues.
    fn balanced(field: &Fqk<K>, b: i64) -> Vec<u64> {
        (-(b / 2)..b - b / 2)
            .map(|c| field.zq().from_i64(c))
            .collect()
    }

    /// A table of `n` pseudorandom digits with, if `wild`, residues outside the digit set in each
    /// quarter of the table and a residue of the size of a digit that is not one (`outside`).
    fn second_round_table(n: usize, digits: &[u64], wild: bool, outside: u64) -> Vec<u64> {
        let mut table = if wild {
            digit_table(n, digits, &[0, n / 4 + 1, n / 2 + 2, 3 * n / 4 + 3, n - 1])
        } else {
            digit_table(n, digits, &[])
        };
        if wild && n >= 8 {
            table[n / 2 + n / 8] = outside;
        }
        table
    }

    #[test]
    fn the_plane_holds_the_range_polynomial() {
        let field = Fqk::<K>::new(Q).unwrap();
        let zq = field.zq();
        for b in [2, 4, 8] {
            let digits = balanced(&field, b);
            let deg0 = digits.len() + 1;
            let range = RangeFactors::new(&field, &digits);
            let small = SmallValues::new(&field, &range, &digits, deg0);
            let r = exts(1, 21)[0];
            let plane = PlaneValues::new(&field, &range, &small, &r);
            assert_eq!(plane.range.len(), small.plane_entries());
            let reach = small.reach;
            for (x, y) in [
                (0, 0),
                (-reach, -2 * reach),
                (reach, 2 * reach),
                (1, -3),
                (-reach, 7),
            ] {
                let node = field.add(
                    &field.from_base(zq.from_i64(x)),
                    &field.scale(&r, zq.from_i64(y)),
                );
                let index = ((x + reach) * plane.width as i64 + y + 2 * reach) as usize;
                assert_eq!(
                    plane.value_at(index),
                    range_poly(&field, &digits, &node),
                    "b {b}, ({x}, {y})"
                );
                // The canonical plane holds the same values.
                let plain = PlaneValues::new(&field, &range.clone().canonical(), &small, &r);
                assert!(plain.mont.is_none() && plane.mont.is_some());
                assert_eq!(plain.value_at(index), plane.value_at(index), "b {b}");
            }
        }
    }

    #[test]
    fn the_second_round_from_the_table_is_the_round_on_the_bound_witness() {
        // Against `round_rows` on the witness bound to the first challenge and the weights bound
        // to it, for digit sets of bases 2, 4 and 8, tables with and without residues outside the
        // digit set (and one of the size of a digit that is not one), and a zero row of g.
        let field = Fqk::<K>::new(Q).unwrap();
        let zq = field.zq();
        for b in [2, 4, 8] {
            let digits = balanced(&field, b);
            let deg0 = digits.len() + 1;
            let range = RangeFactors::new(&field, &digits);
            let small = SmallValues::new(&field, &range, &digits, deg0);
            let outside = zq.from_i64(b / 2);
            for (rows, cols) in [(4, 1), (4, 8), (8, 2), (16, 4), (32, 8), (64, 1)] {
                let n = rows * cols;
                let r = exts(1, 22 + rows as u64)[0];
                let plane = PlaneValues::new(&field, &range, &small, &r);
                let mut eq0 = Tensor {
                    rows: exts(rows, 23),
                    cols: exts(cols, 24),
                };
                let mut g = Tensor {
                    rows: exts(rows, 25),
                    cols: exts(cols, 26),
                };
                g.rows[rows - 1] = field.zero();
                fold_top(&field, &mut eq0.rows, &r);
                fold_top(&field, &mut g.rows, &r);
                for wild in [false, true] {
                    let table = second_round_table(n, &digits, wild, outside);
                    let (lo, hi) = table.split_at(n / 2);
                    let bound: Vec<Ext<K>> = lo
                        .iter()
                        .zip(hi)
                        .map(|(a, b)| {
                            field.add(&field.from_base(*a), &field.scale(&r, zq.sub(*b, *a)))
                        })
                        .collect();
                    assert_eq!(
                        second_round_rows(
                            &field,
                            &range,
                            &small,
                            &plane,
                            Nodes::plain(deg0 + 1),
                            &table,
                            &eq0,
                            &g
                        ),
                        round_rows(&field, &range, Nodes::plain(deg0 + 1), &bound, &eq0, &g),
                        "b {b}, {rows} x {cols}, wild {wild}"
                    );
                }
            }
        }
    }

    #[test]
    fn binding_two_variables_is_binding_one_after_the_other() {
        let field = Fqk::<K>::new(Q).unwrap();
        for n in [4usize, 8, 64, 1024] {
            // Residues of every size, not only digits.
            let table: Vec<u64> = exts(n, 27).iter().map(|x| x.0[1]).collect();
            let (r0, r1) = (exts(2, 28)[0], exts(2, 28)[1]);
            let mut lifted: Vec<Ext<K>> = table.iter().map(|x| field.from_base(*x)).collect();
            fold_top(&field, &mut lifted, &r0);
            fold_top(&field, &mut lifted, &r1);
            assert_eq!(bind_two(&field, &table, &r0, &r1), lifted, "n {n}");
        }
    }

    #[test]
    fn the_rounds_with_the_eq_factor_taken_out_are_the_rounds() {
        // `prove_rounds` with the point of eq0 (the range part at one node fewer, `unfactor`)
        // against the same prover without it: the same messages, point, value and transcript,
        // for bases 2, 4 and 8, split weights of several shapes and dense weights from the start,
        // the second round from the table or not, and tables of digits with and without entries
        // outside the digit set.
        let field = Fqk::<K>::new(Q).unwrap();
        for b in [2, 4, 8] {
            let digits = balanced(&field, b);
            let outside = field.zq().from_i64(b + 3);
            for (rows, cols) in [
                (4usize, 1usize),
                (4, 4),
                (16, 2),
                (8, 8),
                (8, 1),
                (64, 16),
                (1, 8),
            ] {
                let n = rows * cols;
                let (vu, vl) = (
                    rows.trailing_zeros() as usize,
                    cols.trailing_zeros() as usize,
                );
                for wild in [false, true] {
                    let w = second_round_table(n, &digits, wild, outside);
                    let tau = exts(vu + vl, 41 + n as u64);
                    let eq0 = Tensor {
                        rows: crate::mle::eq_table(&field, &tau[..vu]),
                        cols: crate::mle::eq_table(&field, &tau[vu..]),
                    };
                    let g = Tensor {
                        rows: exts(rows, 42),
                        cols: exts(cols, 43),
                    };
                    for share in [0, usize::MAX] {
                        let mut fs_plain = Fs::new(&[7; 32]);
                        let mut fs_factored = fs_plain.clone();
                        let plain = prove_rounds(
                            &field,
                            &digits,
                            &w,
                            eq0.clone(),
                            g.clone(),
                            &mut fs_plain,
                            share,
                            None,
                        )
                        .unwrap();
                        let factored = prove_rounds(
                            &field,
                            &digits,
                            &w,
                            eq0.clone(),
                            g.clone(),
                            &mut fs_factored,
                            share,
                            Some(&tau),
                        )
                        .unwrap();
                        assert_eq!(factored, plain, "b {b}, {rows} x {cols}, wild {wild}");
                        assert_eq!(fs_factored.digest(), fs_plain.digest(), "b {b}");
                    }
                    // A point of the wrong length is refused.
                    let short = prove_rounds(
                        &field,
                        &digits,
                        &w,
                        eq0.clone(),
                        g.clone(),
                        &mut Fs::new(&[7; 32]),
                        0,
                        Some(&tau[1..]),
                    );
                    assert!(matches!(short, Err(Error::Shape(_))), "b {b}");
                }
            }
        }
    }

    #[test]
    fn the_rounds_with_the_claim_s_node_are_the_rounds() {
        // Item 142: the factored rounds with node 1 formed from the round's claim against those
        // without, on tables of digits (the range part sums to 0): the same messages, point,
        // value and transcript.
        let field = Fqk::<K>::new(Q).unwrap();
        for b in [2, 4, 8] {
            let digits = balanced(&field, b);
            let outside = field.zq().from_i64(b + 3);
            for (rows, cols) in [(4usize, 1usize), (16, 2), (8, 8), (64, 16), (1, 8)] {
                let n = rows * cols;
                let (vu, vl) = (
                    rows.trailing_zeros() as usize,
                    cols.trailing_zeros() as usize,
                );
                let w = second_round_table(n, &digits, false, outside);
                let tau = exts(vu + vl, 90 + n as u64);
                let eq0 = Tensor {
                    rows: crate::mle::eq_table(&field, &tau[..vu]),
                    cols: crate::mle::eq_table(&field, &tau[vu..]),
                };
                let g = Tensor {
                    rows: exts(rows, 91),
                    cols: exts(cols, 92),
                };
                for share in [0, usize::MAX] {
                    let mut fs_plain = Fs::new(&[9; 32]);
                    let mut fs_claim = fs_plain.clone();
                    let plain = prove_rounds_with(
                        &field,
                        &digits,
                        &w,
                        eq0.clone(),
                        g.clone(),
                        &mut fs_plain,
                        share,
                        Some(&tau),
                        false,
                    )
                    .unwrap();
                    let claim = prove_rounds_with(
                        &field,
                        &digits,
                        &w,
                        eq0.clone(),
                        g.clone(),
                        &mut fs_claim,
                        share,
                        Some(&tau),
                        true,
                    )
                    .unwrap();
                    assert_eq!(claim, plain, "b {b}, {rows} x {cols}, share {share}");
                    assert_eq!(fs_claim.digest(), fs_plain.digest(), "b {b}");
                }
                // prove_at, which forms the claim's node, is prove.
                let mut fs_at = Fs::new(&[9; 32]);
                let mut fs_prove = fs_at.clone();
                let at = prove_at(
                    &field,
                    &digits,
                    &w,
                    &tau,
                    eq0.clone(),
                    g.clone(),
                    &mut fs_at,
                );
                let plain = prove(&field, &digits, &w, eq0, g, &mut fs_prove);
                assert_eq!(at.unwrap(), plain.unwrap(), "b {b}, {rows} x {cols}");
                assert_eq!(fs_at.digest(), fs_prove.digest(), "b {b}");
            }
        }
    }

    /// Item 142's measurement: the iteration sumcheck at b = 4 on a table of 2^22 digits, split
    /// after 2^14 rows, with and without the claim's node, best of 3. Run with
    /// `RAYON_NUM_THREADS=5 cargo test --release -p hachi --lib timings_claim -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing"]
    fn timings_claim_node() {
        use std::time::Instant;
        let field = Fqk::<K>::new(15395032342003157).unwrap();
        let digits = balanced(&field, 4);
        let (vu, vl) = (14usize, 8usize);
        let n = 1usize << (vu + vl);
        let outside = field.zq().from_i64(7);
        let w = second_round_table(n, &digits, false, outside);
        let tau = exts(vu + vl, 93);
        let eq0 = Tensor {
            rows: crate::mle::eq_table(&field, &tau[..vu]),
            cols: crate::mle::eq_table(&field, &tau[vu..]),
        };
        let g = Tensor {
            rows: exts(1 << vu, 94),
            cols: exts(1 << vl, 95),
        };
        let mut results = Vec::new();
        for claim in [false, true, false, true, false, true] {
            let start = Instant::now();
            let out = prove_rounds_with(
                &field,
                &digits,
                &w,
                eq0.clone(),
                g.clone(),
                &mut Fs::new(&[1; 32]),
                PLANE_SHARE,
                Some(&tau),
                claim,
            )
            .unwrap();
            println!("claim {claim}: {:.3} s", start.elapsed().as_secs_f64());
            results.push(out);
        }
        assert!(
            results.windows(2).all(|x| x[0] == x[1]),
            "the same messages"
        );
    }

    /// Item 139's measurement: the kernels of rounds 3 and later (`third_round_rows` from the
    /// table, `round_rows` on the witness bound to one challenge, `round_values` on dense
    /// weights) with the node products in Montgomery's form against the canonical ones, at b = 4
    /// on a table of 2^22 digits at secure-l24's modulus, split after 2^14 rows, alternating,
    /// best of 3. Run with
    /// `RAYON_NUM_THREADS=5 cargo test --release -p hachi --lib timings_node -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing"]
    fn timings_node_products() {
        use std::time::Instant;
        let field = Fqk::<K>::new(15395032342003157).unwrap();
        let digits = balanced(&field, 4);
        let deg0 = digits.len() + 1;
        let (vu, vl) = (14usize, 8usize);
        let n = 1usize << (vu + vl);
        let outside = field.zq().from_i64(7);
        let table = second_round_table(n, &digits, false, outside);
        let tau = exts(vu + vl, 96);
        let eq0 = Tensor {
            rows: crate::mle::eq_table(&field, &tau[..vu]),
            cols: crate::mle::eq_table(&field, &tau[vu..]),
        };
        let g = Tensor {
            rows: exts(1 << vu, 97),
            cols: exts(1 << vl, 98),
        };
        let (r0, r1) = (exts(1, 99)[0], exts(1, 100)[0]);
        // The weights bound to r0, and to r0 and r1 (their row factors folded).
        let fold_rows = |t: &Tensor<K>, r: &Ext<K>| {
            let mut t = t.clone();
            fold_top(&field, &mut t.rows, r);
            t
        };
        let (bound, gb) = (fold_rows(&eq0, &r0), fold_rows(&g, &r0));
        let (bound2, gb2) = (fold_rows(&bound, &r1), fold_rows(&gb, &r1));
        let half = n / 2;
        let zq = field.zq();
        let w: Vec<Ext<K>> = (0..half)
            .map(|i| {
                let (a, b) = (table[i], table[i + half]);
                field.add(&field.from_base(a), &field.scale(&r0, zq.sub(b, a)))
            })
            .collect();
        let (dense_eq, dense_g) = (bound.dense(&field), gb.dense(&field));
        let quads = QuadValues::new(&field, 2, &r0, &r1);
        let nodes = Nodes {
            count: deg0,
            factored: true,
            skip: Some(1),
        };
        let range = RangeFactors::new(&field, &digits);
        let canonical = range.clone().canonical();
        let best = |run: &dyn Fn() -> (Vec<Ext<K>>, Vec<Ext<K>>)| {
            let mut out = None;
            let mut best = f64::MAX;
            for _ in 0..3 {
                let start = Instant::now();
                out = Some(run());
                best = best.min(start.elapsed().as_secs_f64());
            }
            (best, out.unwrap())
        };
        for (name, montgomery, plain) in [
            (
                "third_round_rows",
                best(&|| {
                    third_round_rows(
                        &field,
                        &range,
                        nodes,
                        Some(&quads),
                        &table,
                        &r0,
                        &r1,
                        &bound2,
                        &gb2,
                    )
                }),
                best(&|| {
                    third_round_rows(
                        &field,
                        &canonical,
                        nodes,
                        Some(&quads),
                        &table,
                        &r0,
                        &r1,
                        &bound2,
                        &gb2,
                    )
                }),
            ),
            (
                "round_rows",
                best(&|| round_rows(&field, &range, nodes, &w, &bound, &gb)),
                best(&|| round_rows(&field, &canonical, nodes, &w, &bound, &gb)),
            ),
            (
                "round_values",
                best(&|| round_values(&field, &range, nodes, &w, &dense_eq, &dense_g)),
                best(&|| round_values(&field, &canonical, nodes, &w, &dense_eq, &dense_g)),
            ),
        ] {
            assert_eq!(montgomery.1, plain.1, "{name}: the same values");
            println!(
                "{name}: {:.3} s Montgomery, {:.3} s canonical ({:.2}x)",
                montgomery.0,
                plain.0,
                plain.0 / montgomery.0
            );
        }
    }

    #[test]
    fn the_extension_by_differences_is_the_polynomial() {
        // unfactor's Q(m) against a polynomial of degree m - 1 evaluated at m, for m = 3 to 9,
        // and its values against eq(tau, t) Q(t).
        let field = Fqk::<K>::new(Q).unwrap();
        for m in 3..=9usize {
            let coeffs = exts(m, 50 + m as u64);
            let at = |t: u64| {
                let x = field.from_base(t);
                coeffs
                    .iter()
                    .rev()
                    .fold(field.zero(), |acc, c| field.add(&field.mul(&acc, &x), c))
            };
            let q: Vec<Ext<K>> = (0..m as u64).map(at).collect();
            let tau = exts(1, 60 + m as u64)[0];
            let got = unfactor(&field, &q, &tau, &binomial_row(field.zq(), m));
            for (t, v) in got.iter().enumerate() {
                let x = field.from_base(t as u64);
                let eq = field.add(
                    &field.mul(&field.sub(&field.one(), &tau), &field.sub(&field.one(), &x)),
                    &field.mul(&tau, &x),
                );
                assert_eq!(*v, field.mul(&eq, &at(t as u64)), "m {m}, t {t}");
            }
        }
    }

    #[test]
    fn the_two_coin_values_from_the_quadruples_are_the_values_formed() {
        // `bound_two` with the values of the quadruples of digits (`QuadValues`) against
        // `bound_two_at`, and `bind_three` and the third round with them against those without,
        // for bases 2, 4 and 8 and tables of digits with and without entries outside the digit
        // set (those take the formed values).
        let field = Fqk::<K>::new(Q).unwrap();
        for b in [2, 4, 8] {
            let digits = balanced(&field, b);
            let range = RangeFactors::new(&field, &digits);
            let deg0 = digits.len() + 1;
            let small = SmallValues::new(&field, &range, &digits, deg0);
            let outside = field.zq().from_i64(b + 3);
            for (rows, cols) in [(8usize, 1usize), (8, 4), (64, 8)] {
                let n = rows * cols;
                for wild in [false, true] {
                    let table = second_round_table(n, &digits, wild, outside);
                    let r = exts(3, 70 + n as u64);
                    let quads = QuadValues::new(&field, small.digit, &r[0], &r[1]);
                    let (quarter, r01) = (n / 4, field.mul(&r[0], &r[1]));
                    for i in 0..quarter {
                        assert_eq!(
                            bound_two(&field, Some(&quads), &table, quarter, &r[0], &r[1], &r01, i),
                            bound_two_at(&field, &table, quarter, &r[0], &r[1], &r01, i),
                            "b {b}, entry {i}"
                        );
                    }
                    assert_eq!(
                        bind_three(&field, Some(&quads), &table, &r[0], &r[1], &r[2]),
                        bind_three(&field, None, &table, &r[0], &r[1], &r[2]),
                        "b {b}, n {n}, wild {wild}"
                    );
                    let tensor = |seed: u64| Tensor {
                        rows: exts(rows / 4, seed),
                        cols: exts(cols, seed + 1),
                    };
                    let (eq0, g) = (tensor(80), tensor(82));
                    if rows >= 8 {
                        assert_eq!(
                            third_round_rows(
                                &field,
                                &range,
                                Nodes::plain(deg0 + 1),
                                Some(&quads),
                                &table,
                                &r[0],
                                &r[1],
                                &eq0,
                                &g
                            ),
                            third_round_rows(
                                &field,
                                &range,
                                Nodes::plain(deg0 + 1),
                                None,
                                &table,
                                &r[0],
                                &r[1],
                                &eq0,
                                &g
                            ),
                            "b {b}, n {n}, wild {wild}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn binding_three_variables_is_binding_one_after_the_other() {
        let field = Fqk::<K>::new(Q).unwrap();
        for n in [8usize, 16, 64, 1024] {
            let table: Vec<u64> = exts(n, 33).iter().map(|x| x.0[2]).collect();
            let r = exts(3, 34);
            let mut lifted: Vec<Ext<K>> = table.iter().map(|x| field.from_base(*x)).collect();
            for ri in &r {
                fold_top(&field, &mut lifted, ri);
            }
            assert_eq!(
                bind_three(&field, None, &table, &r[0], &r[1], &r[2]),
                lifted,
                "n {n}"
            );
        }
    }

    #[test]
    fn the_third_round_from_the_table_is_the_round_on_the_bound_witness() {
        // Against `round_rows` on the witness of `bind_two` and the weights bound to the first two
        // challenges, for digit sets of bases 2, 4 and 8 and residues of every size.
        let field = Fqk::<K>::new(Q).unwrap();
        for b in [2, 4, 8] {
            let digits = balanced(&field, b);
            let deg0 = digits.len() + 1;
            let range = RangeFactors::new(&field, &digits);
            for (rows, cols) in [(8, 1), (8, 4), (16, 2), (64, 8)] {
                let n = rows * cols;
                let table: Vec<u64> = exts(n, 35 + rows as u64).iter().map(|x| x.0[3]).collect();
                let r = exts(2, 36 + cols as u64);
                let mut eq0 = Tensor {
                    rows: exts(rows, 37),
                    cols: exts(cols, 38),
                };
                let mut g = Tensor {
                    rows: exts(rows, 39),
                    cols: exts(cols, 40),
                };
                for ri in &r {
                    fold_top(&field, &mut eq0.rows, ri);
                    fold_top(&field, &mut g.rows, ri);
                }
                let bound = bind_two(&field, &table, &r[0], &r[1]);
                assert_eq!(
                    third_round_rows(
                        &field,
                        &range,
                        Nodes::plain(deg0 + 1),
                        None,
                        &table,
                        &r[0],
                        &r[1],
                        &eq0,
                        &g
                    ),
                    round_rows(&field, &range, Nodes::plain(deg0 + 1), &bound, &eq0, &g),
                    "b {b}, {rows} x {cols}"
                );
            }
        }
    }

    #[test]
    fn the_prover_reading_the_table_is_the_dense_prover() {
        // `prove_rounds` with every plane allowed against the prover on dense weights: splits
        // with four rows or more (the second round reads the table), the smallest such table (two
        // rounds), digit sets of bases 2, 4 and 8, and tables with residues outside the digit set.
        let field = Fqk::<K>::new(Q).unwrap();
        for b in [2, 4, 8] {
            let digits = balanced(&field, b);
            let outside = field.zq().from_i64(b / 2);
            // Four rows: the weights are dense at the third round; eight or more: it reads the
            // table too.
            for (rows, cols) in [(4, 1), (4, 4), (16, 2), (8, 8), (8, 1), (64, 16), (256, 1)] {
                let n = rows * cols;
                for wild in [false, true] {
                    let w = second_round_table(n, &digits, wild, outside);
                    let eq0 = Tensor {
                        rows: exts(rows, 29),
                        cols: exts(cols, 30),
                    };
                    let g = Tensor {
                        rows: exts(rows, 31),
                        cols: exts(cols, 32),
                    };
                    let mut fs_dense = Fs::new(&[4; 32]);
                    let mut fs_table = fs_dense.clone();
                    let (eq_dense, g_dense) = (eq0.dense(&field), g.dense(&field));
                    let dense = prove_dense(&field, &digits, &w, eq_dense, g_dense, &mut fs_dense);
                    let mut fs_bytes = fs_table.clone();
                    let (eq0_bytes, g_bytes) = (eq0.clone(), g.clone());
                    let table =
                        prove_rounds(&field, &digits, &w, eq0, g, &mut fs_table, 0, None).unwrap();
                    assert_eq!(table, dense, "b {b}, {rows} x {cols}, wild {wild}");
                    assert_eq!(
                        fs_table.digest(),
                        fs_dense.digest(),
                        "b {b}, {rows} x {cols}"
                    );
                    // The same table as bytes, where its entries are digits: the same prover.
                    let mut bytes = crate::table::DigitTable::zeros(field.zq(), n);
                    if bytes.set_residues(field.zq(), 0, &w).is_ok() {
                        let from_bytes = prove_rounds(
                            &field,
                            &digits,
                            &bytes,
                            eq0_bytes,
                            g_bytes,
                            &mut fs_bytes,
                            0,
                            None,
                        )
                        .unwrap();
                        assert_eq!(from_bytes, table, "b {b}, {rows} x {cols}, as bytes");
                        assert_eq!(fs_bytes.digest(), fs_table.digest(), "as bytes");
                    } else {
                        assert!(
                            wild,
                            "b {b}: only a wild table has entries that are not bytes"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_prover_rejects_weights_of_another_shape() {
        let field = Fqk::<K>::new(Q).unwrap();
        let digits = base8(&field);
        let w = vec![0u64; 16];
        let tensor = |rows, cols| Tensor {
            rows: exts(rows, 1),
            cols: exts(cols, 2),
        };
        // Two splits, a product other than the table's length, a factor of another length.
        for (case, (eq0, g)) in [
            (tensor(4, 4), tensor(2, 8)),
            (tensor(4, 2), tensor(4, 2)),
            (tensor(3, 4), tensor(4, 4)),
        ]
        .into_iter()
        .enumerate()
        {
            let result = prove(&field, &digits, &w, eq0, g, &mut Fs::new(&[0; 32]));
            assert!(matches!(result, Err(Error::Shape(_))), "case {case}");
        }
        // One column is a split too.
        let one_column = prove(
            &field,
            &digits,
            &w,
            tensor(16, 1),
            tensor(16, 1),
            &mut Fs::new(&[0; 32]),
        );
        assert!(one_column.is_ok());
    }

    #[test]
    #[ignore = "timing: cargo test --release -p hachi --lib timings_second -- --ignored --nocapture"]
    fn timings_second_round_from_the_table() {
        // The prover with the second round on the bound witness and read from the table, on
        // HACHI_SUMCHECK_VARS (default 24) variables, 2^6 columns and the digits of bases 4, 8 and
        // 16; one run each, on the Rayon pool.
        use crate::mle::{eq_table, powers};
        use std::time::Instant;
        let vars: usize = std::env::var("HACHI_SUMCHECK_VARS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(24);
        let field = Fqk::<K>::new(Q).unwrap();
        let (n, s) = (1usize << vars, 6);
        let tau0 = exts(vars, 33);
        let (tau_u, tau_l) = tau0.split_at(vars - s);
        let m = exts(n >> s, 34);
        let alpha_pows = powers(&field, &exts(1, 35)[0], 1 << s);
        let weights = || {
            (
                Tensor {
                    rows: eq_table(&field, tau_u),
                    cols: eq_table(&field, tau_l),
                },
                Tensor {
                    rows: m.clone(),
                    cols: alpha_pows.clone(),
                },
            )
        };
        for b in [4, 8, 16] {
            let digits = balanced(&field, b);
            let w = digit_table(n, &digits, &[]);
            let mut times = Vec::new();
            let mut results = Vec::new();
            for share in [usize::MAX, PLANE_SHARE] {
                let (eq0, g) = weights();
                let start = Instant::now();
                let result = prove_rounds(
                    &field,
                    &digits,
                    &w,
                    eq0,
                    g,
                    &mut Fs::new(&[6; 32]),
                    share,
                    None,
                )
                .unwrap();
                times.push(start.elapsed().as_secs_f64());
                results.push(result);
            }
            assert_eq!(results[0], results[1]);
            let range = RangeFactors::new(&field, &digits);
            let small = SmallValues::new(&field, &range, &digits, digits.len() + 1);
            println!(
                "{vars} variables, base {b}, {} threads, plane of {} values ({}): bound copy {:.2} s, from the table {:.2} s",
                rayon::current_num_threads(),
                small.plane_entries(),
                if small.plane_entries().saturating_mul(PLANE_SHARE) <= n {
                    "used"
                } else {
                    "not used"
                },
                times[0],
                times[1]
            );
        }
    }

    #[test]
    #[ignore = "timing: cargo test --release -p hachi --lib timings_split -- --ignored --nocapture"]
    fn timings_split_against_dense() {
        // The prover of an iteration with its weights built as before (the full eq table and g
        // as the products m* ⊗ powers of alpha) and as factors, on HACHI_SUMCHECK_VARS (default
        // 24) variables, 2^8 columns and digits of base 8; one run each, on the Rayon pool.
        use crate::mle::{eq_table, powers};
        use std::time::Instant;
        let vars: usize = std::env::var("HACHI_SUMCHECK_VARS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(24);
        let field = Fqk::<K>::new(Q).unwrap();
        let digits = base8(&field);
        let (n, s) = (1usize << vars, 8);
        let w = digit_table(n, &digits, &[]);
        let tau0 = exts(vars, 11);
        let m = exts(n >> s, 12);
        let alpha_pows = powers(&field, &exts(1, 13)[0], 1 << s);
        let start = Instant::now();
        let eq_dense = eq_table(&field, &tau0);
        let f = &field;
        let g_dense: Vec<Ext<K>> = m
            .iter()
            .flat_map(|mu| alpha_pows.iter().map(move |ap| f.mul(mu, ap)))
            .collect();
        let mut fs_dense = Fs::new(&[5; 32]);
        let dense = prove_dense(&field, &digits, &w, eq_dense, g_dense, &mut fs_dense);
        let t_dense = start.elapsed();
        let start = Instant::now();
        let (tau_u, tau_l) = tau0.split_at(vars - s);
        let eq0 = Tensor {
            rows: eq_table(&field, tau_u),
            cols: eq_table(&field, tau_l),
        };
        let g = Tensor {
            rows: m,
            cols: alpha_pows,
        };
        let mut fs_split = Fs::new(&[5; 32]);
        let split = prove(&field, &digits, &w, eq0, g, &mut fs_split).unwrap();
        let t_split = start.elapsed();
        assert_eq!(split, dense);
        println!(
            "{vars} variables, {} threads: dense {:.2} s, split {:.2} s",
            rayon::current_num_threads(),
            t_dense.as_secs_f64(),
            t_split.as_secs_f64()
        );
    }
}
