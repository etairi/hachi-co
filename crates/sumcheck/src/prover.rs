//! Party $p$'s side of the collaborative sumcheck (`5_Building_Blocks.tex`:238-352; oracle
//! `sumcheck.prove`), as a state machine that the driver [`prove`](crate::prove) runs against a
//! [`Comm`](hachi_co_mpc::Comm) and that tests can run for several emulated parties.

use crate::arith::{Arith, Montgomery};
use crate::blinder::{BlinderRounds, evaluate, hypercube_sum};
use crate::eq::{eq_point, eq_table, msb_bits};
use crate::error::{Error, Result, check_len};
use crate::interp::Extrapolation;
use crate::tables::{
    MStarParts, RangeProduct, ind_rows, lift_table, pad_index, row_indicator, table_size,
};
use crate::transcript::RoundValues;
use hachi_co_core::{Blinder, Context, SumcheckCoins};
use hachi_co_math::{Ext, Fqk, Zq};
use hachi_co_mpc::PartyId;
use rayon::prelude::*;

/// The tables that change with the coins (built by [`PartyProver::start`]).
#[derive(Clone, Debug)]
struct Running<const K: usize> {
    /// $\rho$.
    rho: Ext<K>,
    /// $\widetilde{eq}(\tau\_0\[\ell''..\], \mathsf{bit}(p))$, the factor of $F\_{0,p}$ for the bits of
    /// the party (`ref/README.md`, "eq").
    eq_party: Ext<K>,
    /// $\langle \widetilde w \rangle\_p$ after the first bind (empty in round 1, which reads the
    /// base-field table, and in round 2 when it reads that table too, `PlaneValues`).
    w: Vec<Ext<K>>,
    /// $M^\ast$.
    mstar: MStar<K>,
    /// $\widetilde{eq}(\tau\_0\[0..\ell\_U\], U)$ over the $U$ variables not yet bound.
    eq_u: Vec<Ext<K>>,
    /// $\widetilde{eq}(\tau\_0\[\ell\_U..\ell''\], L)$ over the $L$ variables not yet bound.
    eq_l: Vec<Ext<K>>,
    /// $e\_p$ over $U$.
    e: Vec<Ext<K>>,
    /// $\mathrm{ind}$ over $U$.
    ind: Vec<Ext<K>>,
    /// The round polynomials of $g\_{0,p}$ and $g\_{1,p}$.
    g0: BlinderRounds<K>,
    g1: BlinderRounds<K>,
    /// The coins $a\_1, \dots, a\_{r-1}$ so far.
    point: Vec<Ext<K>>,
    /// $\tau\_0\[0..\ell''\]$, whose coordinate $r - 1$ is that of round $r$'s variable in
    /// $\widetilde{eq}\_U$ and $\widetilde{eq}\_L$ ([`unfactor`]).
    tau: Vec<Ext<K>>,
}

/// $M^\ast$ as the rounds read it: formed from its parts where rounds 1 to 3 read it
/// ([`MStarParts::at`], then bound to $a\_1$ and to $a\_1, a\_2$ as the first two binds bind a
/// table), so that neither its table of $2^{\ell''}$ entries nor the tables bound to one and two
/// coins are held, then the table bound by the coins so far (formed bound to $a\_1, a\_2, a\_3$ by
/// the third bind, or to $a\_1, a\_2$ by the second for a hypercube of two variables).
#[derive(Clone, Debug)]
enum MStar<const K: usize> {
    /// The table, canonical.
    Table(Vec<Ext<K>>),
    /// The parts, before the first bind.
    Parts(MStarParts<K>),
    /// The parts and $a\_1$, between the first and the second bind.
    Bound(MStarParts<K>, Coin<K>),
    /// The parts, $a\_1$ and $a\_2$, between the second and the third bind.
    Bound2(MStarParts<K>, Coin<K>, Coin<K>),
}

impl<const K: usize> MStar<K> {
    /// Entry $x$, canonical: for [`MStar::Bound`] the entry $T\[x\] + a\_1 (T\[x + h\] - T\[x\])$ of
    /// the table $T$ of the parts, and for [`MStar::Bound2`] the same with $a\_2$ of that table
    /// bound to $a\_1$: the values that [`fold`] gives them (the product of a coin by a canonical
    /// difference is the same field element in either arithmetic, and canonical).
    #[inline(always)]
    fn at(&self, field: &Fqk<K>, x: usize) -> Ext<K> {
        match self {
            Self::Table(table) => table[x],
            Self::Parts(parts) => parts.at(field, x),
            Self::Bound(parts, a1) => {
                let h = parts.len() / 2;
                let (lo, hi) = (parts.at(field, x), parts.at(field, x + h));
                field.add(&lo, &a1.times(field, &field.sub(&hi, &lo)))
            }
            Self::Bound2(parts, a1, a2) => {
                let (h, q) = (parts.len() / 2, parts.len() / 4);
                let bound = |y: usize| {
                    let (lo, hi) = (parts.at(field, y), parts.at(field, y + h));
                    field.add(&lo, &a1.times(field, &field.sub(&hi, &lo)))
                };
                let (lo, hi) = (bound(x), bound(x + q));
                field.add(&lo, &a2.times(field, &field.sub(&hi, &lo)))
            }
        }
    }

    /// The number of entries.
    fn len(&self) -> usize {
        match self {
            Self::Table(table) => table.len(),
            Self::Parts(parts) => parts.len(),
            Self::Bound(parts, _) => parts.len() / 2,
            Self::Bound2(parts, ..) => parts.len() / 4,
        }
    }
}

/// A coin of a bind, canonical, with its Montgomery form where the modulus admits it
/// ([`Montgomery`]): [`Coin::times`] gives the canonical product $a d$ either way.
#[derive(Clone, Copy, Debug)]
struct Coin<const K: usize> {
    canonical: Ext<K>,
    mont: Option<(Montgomery<K>, Ext<K>)>,
}

impl<const K: usize> Coin<K> {
    fn new(field: &Fqk<K>, a: &Ext<K>) -> Self {
        let canonical = field.reduce(a);
        let mont = Montgomery::<K>::new(field.modulus()).map(|m| (m, m.enter(&canonical)));
        Self { canonical, mont }
    }

    /// $a d$, canonical, for a canonical $d$ ($\phi(a) d = ad$).
    #[inline(always)]
    fn times(&self, field: &Fqk<K>, d: &Ext<K>) -> Ext<K> {
        match &self.mont {
            Some((m, am)) => m.mul(am, d),
            None => field.mul(&self.canonical, d),
        }
    }
}

/// The values at the small integers that round 1's nodes reach, in the working representation:
/// for a pair of integers of absolute value at most `digit` and a node $t$ among the first
/// `nodes`, the value $v + t (v - u)$ has absolute value at most `reach`.
struct SmallValues {
    /// The largest absolute value of an entry that takes this path.
    digit: i64,
    /// The largest absolute value of a node value.
    reach: i64,
    /// $\phi(c)$ at index $c +$ `reach`.
    value: Vec<u64>,
    /// $\phi\bigl(\prod\_{j=-b+1}^{b-1} (c - j)\bigr)$ at index $c +$ `reach`.
    range: Vec<u64>,
}

impl SmallValues {
    fn new<const K: usize, A: Arith<K>>(
        arith: &A,
        field: &Fqk<K>,
        range: &RangeProduct,
        b: u64,
        nodes: usize,
    ) -> Self {
        let zq = field.zq();
        let digit = b.saturating_sub(1) as i64;
        let reach = (2 * nodes as i64 + 1) * digit;
        let residues: Vec<u64> = (-reach..=reach).map(|c| zq.from_i64(c)).collect();
        Self {
            digit,
            reach,
            value: residues.iter().map(|c| arith.enter_base(*c)).collect(),
            range: residues
                .iter()
                .map(|c| arith.enter_base(range.eval(field, &field.from_base(*c)).0[0]))
                .collect(),
        }
    }

    /// The centered value of the residue `x`, if it is an integer of this path.
    #[inline(always)]
    fn small<const K: usize>(&self, field: &Fqk<K>, x: u64) -> Option<i64> {
        let c = field.zq().to_centered(x);
        (c.abs() <= self.digit).then_some(c)
    }
}

/// The range product at the node values of round 2 for points whose entries are small
/// integers, in the working representation. After the first coin $a\_1$ the folded table is
/// $W\[y\] = w\[y\] + a\_1 (w\[y + n/2\] - w\[y\])$, so that round 2's pair $(x, x + n/4)$ has at
/// node $t$ (point $t + 1$) the value $X + a\_1 Y$ with $X = C + t (C - A)$ and
/// $Y = (E - C) + t ((E - C) - (B - A))$ for the entries $A, B, C, E$ of the base-field table at
/// $x$, $x + n/2$, $x + n/4$ and $x + 3n/4$. For entries of absolute value at most `digit` of
/// [`SmallValues`] and its nodes, $\lvert X \rvert \le$ `reach` and $\lvert Y \rvert \le 2\,$`reach`.
struct PlaneValues<const K: usize> {
    /// $\phi(a\_1)$.
    coin: Ext<K>,
    /// The largest absolute value of an entry that takes this path.
    digit: i64,
    /// The reach of $X$, that of [`SmallValues`]: $Y$ reaches twice as far.
    reach: i64,
    /// The number of values of $Y$, $4\,$`reach` $+ 1$.
    width: usize,
    /// $\phi(v)$ for $\lvert v \rvert \le 2\,$`reach`, at index $v + 2\,$`reach`.
    value: Vec<u64>,
    /// $\phi\bigl(\prod\_{j=-b+1}^{b-1} (X + a\_1 Y - j)\bigr)$ at index
    /// $(X + \mathrm{reach})\,\mathrm{width} + Y + 2\,\mathrm{reach}$.
    range: Vec<Ext<K>>,
}

impl<const K: usize> PlaneValues<K> {
    /// The number of values of $\prod\_j (X + a\_1 Y - j)$ for entries of absolute value at most
    /// `digit` and `nodes` nodes.
    fn entries(digit: i64, nodes: usize) -> usize {
        let reach = ((2 * nodes as i64 + 1) * digit).max(0) as usize;
        (2 * reach + 1).saturating_mul(4 * reach + 1)
    }

    fn new<A: Arith<K> + Sync>(
        arith: &A,
        field: &Fqk<K>,
        range: &RangeProduct,
        small: &SmallValues,
        a1: &Ext<K>,
    ) -> Self {
        let zq = field.zq();
        let reach = small.reach;
        let residues: Vec<u64> = (-2 * reach..=2 * reach).map(|v| zq.from_i64(v)).collect();
        let width = residues.len();
        let a1y: Vec<Ext<K>> = residues.iter().map(|y| field.scale(a1, *y)).collect();
        let mut values = vec![field.zero(); (2 * reach as usize + 1) * width];
        // The range product in the working representation, as the rounds form it
        // ([`RangePairs::eval`]): the same element as the canonical product entered.
        let factors = RangePairs::new(arith, zq, range.squares());
        // Row X + reach: the X of residue index X + 2 reach.
        values
            .par_chunks_mut(width)
            .zip(residues[reach as usize..=3 * reach as usize].par_iter())
            .for_each(|(row, x)| {
                let x = field.from_base(*x);
                for (out, a1y) in row.iter_mut().zip(&a1y) {
                    let w = arith.enter(&field.add(&x, a1y));
                    *out = factors.eval(arith, &arith.mul(&w, &w), &w);
                }
            });
        Self {
            coin: arith.enter(a1),
            digit: small.digit,
            reach,
            width,
            value: residues.iter().map(|v| arith.enter_base(*v)).collect(),
            range: values,
        }
    }

    /// $\phi(v)$ as a residue, for $\lvert v \rvert \le 2\,$`reach`.
    #[inline(always)]
    fn value(&self, v: i64) -> u64 {
        self.value[(v + 2 * self.reach) as usize]
    }

    /// The centered value of the residue `x`, if it is an integer of this path.
    #[inline(always)]
    fn small(&self, field: &Fqk<K>, x: u64) -> Option<i64> {
        let c = field.zq().to_centered(x);
        (c.abs() <= self.digit).then_some(c)
    }
}

/// Round 2's sums of the terms $\widetilde{eq}\_L W$ over the points of a row that take
/// [`PlaneValues`], by the integers of their node values: $\sum \phi(e X\_1)$, $\sum \phi(e\,\Delta X)$,
/// $\sum \phi(e Y\_1)$ and $\sum \phi(e\,\Delta Y)$, so that the sum at node $t$ is
/// $(X\_1 + t \Delta X) + a\_1 (Y\_1 + t \Delta Y)$ summed with the weights.
type PlaneSums<const K: usize> = [Ext<K>; 4];

/// One point's contribution to the sums of a round, at the nodes computed directly.
///
/// The products of [`PartyProver::accumulate_pair`] go into the `_wide` sums unreduced
/// ([`Arith::mul_wide`]) and are reduced once per [`Arith::wide_capacity`] points or before the
/// sums are read ([`Accumulators::flush`]): the same field elements as the reduced products added
/// one by one (item 153).
struct Accumulators<const K: usize> {
    /// $\sum \widetilde{eq}\_L \cdot \prod\_c (W - c)$ per node.
    rp: Vec<Ext<K>>,
    /// $\sum \widetilde{eq}\_L \cdot W$ per node.
    w: Vec<Ext<K>>,
    /// $\sum W \cdot M^\ast$ per node.
    m: Vec<Ext<K>>,
    /// The unreduced sums of the same, since the last [`Accumulators::flush`].
    rp_wide: Vec<[u128; K]>,
    w_wide: Vec<[u128; K]>,
    m_wide: Vec<[u128; K]>,
    /// The points added to the `_wide` sums since, and how many they may hold.
    pending: usize,
    capacity: usize,
}

impl<const K: usize> Accumulators<K> {
    fn new(field: &Fqk<K>, n0: usize, n1: usize, capacity: usize) -> Self {
        Self {
            rp: vec![field.zero(); n0],
            w: vec![field.zero(); n0],
            m: vec![field.zero(); n1],
            rp_wide: vec![[0; K]; n0],
            w_wide: vec![[0; K]; n0],
            m_wide: vec![[0; K]; n1],
            pending: 0,
            capacity,
        }
    }

    fn clear(&mut self, field: &Fqk<K>) {
        for v in self.rp.iter_mut().chain(&mut self.w).chain(&mut self.m) {
            *v = field.zero();
        }
        for v in self
            .rp_wide
            .iter_mut()
            .chain(&mut self.w_wide)
            .chain(&mut self.m_wide)
        {
            *v = [0; K];
        }
        self.pending = 0;
    }

    /// Reduces the `_wide` sums into the sums ([`Arith::reduce_wide`]).
    fn flush<A: Arith<K>>(&mut self, arith: &A) {
        if self.pending == 0 {
            return;
        }
        for (sums, wides) in [
            (&mut self.rp, &mut self.rp_wide),
            (&mut self.w, &mut self.w_wide),
            (&mut self.m, &mut self.m_wide),
        ] {
            for (x, wide) in sums.iter_mut().zip(wides.iter_mut()) {
                *x = arith.add(x, &arith.reduce_wide(wide));
                *wide = [0; K];
            }
        }
        self.pending = 0;
    }

    /// Counts a point added to the `_wide` sums, and reduces them when they are full.
    #[inline(always)]
    fn added<A: Arith<K>>(&mut self, arith: &A) {
        self.pending += 1;
        if self.pending >= self.capacity {
            self.flush(arith);
        }
    }

    /// Adds `other`, in the working representation of `arith`.
    fn merge<A: Arith<K>>(&mut self, arith: &A, mut other: Self) {
        self.flush(arith);
        other.flush(arith);
        for (x, y) in self.rp.iter_mut().zip(&other.rp) {
            *x = arith.add(x, y);
        }
        for (x, y) in self.w.iter_mut().zip(&other.w) {
            *x = arith.add(x, y);
        }
        for (x, y) in self.m.iter_mut().zip(&other.m) {
            *x = arith.add(x, y);
        }
    }
}

/// Party $p$'s prover state: its witness table $\langle \widetilde w \rangle\_p$, its blinders and,
/// once the coins are known, the tables it folds round by round.
///
/// **Use.** [`PartyProver::new`] (the hypercube sums $G\_{0,p}, G\_{1,p}$ are then available from
/// [`PartyProver::hypercube_sums`]); [`PartyProver::start`] with the coins
/// $\alpha, \rho, \tau\_0, \tau\_1$ and the table of $M^\ast$; then for $r = 1, \dots, \ell''$:
/// [`PartyProver::round_values`] (the input to $\mathcal F\_{\mathsf{AH}}(\mathbb I)$) and
/// [`PartyProver::bind`] with the coin $a\_r$; finally [`PartyProver::final_values`]. Calls out
/// of order are [`Error::State`].
///
/// **Round values** (`5_Building_Blocks.tex`:299-307, 315-332; fixture conventions `F0` and
/// `rounds`). With $\widetilde{eq}\_0 = \widetilde{eq}(\tau\_0, (U, L, \mathsf{bit}(p)))$,
/// $$F\_{0,p} = \widetilde{eq}\_0 \Bigl(e\_p \prod\_{c=-b+1}^{b-1} (\langle \widetilde w \rangle\_p -
/// c) + (1 - e\_p - \mathrm{ind}) \langle \widetilde w \rangle\_p\Bigr), \qquad
/// F\_{1,p} = \langle \widetilde w \rangle\_p \cdot \widetilde M^\ast,$$
/// and $H\_{i,p} = F\_{i,p} + \rho\, g\_{i,p}$. Round $r$ binds $V\_r$, the most significant bit of
/// the remaining table index ([`bind_top`](crate::bind_top)); the value at node $t$ is the sum over
/// the lower half $x$ of the tables of $F$ at $T\[x\] + t (T\[x + h\] - T\[x\])$.
///
/// **Cost.** Linear in the table size per round, so $O(2^{\ell''})$ over all rounds. The factors
/// $e\_p$ and $\mathrm{ind}$ depend on $U$ only and $\widetilde{eq}(\tau\_0\[0..\ell''\], (U, L))
/// = \widetilde{eq}\_U(U) \cdot \widetilde{eq}\_L(L)$, so they are kept as tables over $U$
/// ($2^{\ell\_U}$ entries) and $L$ ($2d$ entries) and enter once per row of $2d$ points; the values
/// at consecutive nodes are formed by additions. The round polynomial of $F\_{0,p}$ has degree at
/// most $2b + 1$ in $V\_r$ ($1$ from $\widetilde{eq}$, $1$ from $e\_p$, $2b - 1$ from the range
/// product; `ref/README.md`, "Observations about the draft") and that of $F\_{1,p}$ at most $2$,
/// so they are computed at the nodes $1, \dots, 2b + 2$ resp. $1, 2, 3$ and extended exactly to
/// $d\_0 + 1$ resp. $d\_1 + 1$ nodes by Lagrange interpolation. The blinders' round values are the
/// closed form of [`round_poly`](crate::round_poly). The sums are formed in Montgomery form
/// when $\max(3k - 3, 1)\, q \lt 2^{64}$ (every $q \lt 2^{59}$), which avoids the 128-bit
/// divisions of the canonical product, and in the canonical arithmetic otherwise; the tables
/// stay canonical and both give the same values. The range product is skipped on the rows where
/// $e\_p$ vanishes at both ends. In round 1, half of the work, a pair of digits has small integers
/// as node values, and their range products come from a table (`SmallValues`), so that each
/// term is a product by a residue. The tests check every value against the oracle's direct
/// evaluation at all nodes, in both arithmetics.
#[derive(Clone, Debug)]
pub struct PartyProver<const K: usize> {
    field: Fqk<K>,
    party: PartyId,
    n: usize,
    log_n: usize,
    ell_u: usize,
    ell_pp: usize,
    two_d: usize,
    d0: usize,
    d1: usize,
    /// The nodes $1, \dots, \mathrm{direct}\_0$ at which $F\_{0,p}$ is computed directly.
    direct0: usize,
    direct1: usize,
    ext0: Extrapolation,
    ext1: Extrapolation,
    /// $\binom{m}{i} \bmod q$ for $m = \mathrm{direct}\_0 - 1$ ([`unfactor`]).
    binomials: Vec<u64>,
    range: RangeProduct,
    /// The base $b$.
    b: u64,
    /// The Montgomery form of the inner loops, when $q$ admits it (otherwise the canonical
    /// field is used; both give the same values).
    mont: Option<Montgomery<K>>,
    w_base: Vec<u64>,
    /// The table index and the value of a pad in $\mathbb F\_{q^K}$, the one entry of the table
    /// outside $\mathbb Z\_q$; `w_base` has 0 there.
    pad: Option<(usize, Ext<K>)>,
    g0: Blinder<K>,
    g1: Blinder<K>,
    sums: (Ext<K>, Ext<K>),
    e_rows: Vec<u64>,
    ind_rows: Vec<u64>,
    run: Option<Running<K>>,
    /// Round 2 reads the base-field table ([`PlaneValues`]) when it is a round over $U$ and the
    /// table of the range product has at most $2^{\ell''} /$ `plane_share` values
    /// ([`PLANE_SHARE`]).
    plane_share: usize,
}

/// The default `plane_share` of [`PartyProver`]: at $b = 16$ the table of [`PlaneValues`] has
/// $8\,576\,011$ values (274 MB), against the $2^{\ell''}/2$ values of the folded table that round 2
/// would otherwise read ($2^{26}$ at secure-l24-n2-a8).
const PLANE_SHARE: usize = 1;

impl<const K: usize> PartyProver<K> {
    /// The prover of party `party` with the hypercube table `w_table` of
    /// $\langle \widetilde w \rangle\_p$ ([`w_table`](crate::w_table)), the pad $\rho^w\_p$ if it
    /// is a field element, and the blinders `g0`, `g1`
    /// (coefficients from [`PartyRandomness::blinder`](hachi_co_core::PartyRandomness::blinder)).
    /// Computes $G\_{0,p}$ and $G\_{1,p}$ ([`hypercube_sum`]).
    ///
    /// # Errors
    /// [`Error::InvalidParty`] for $p \notin \lbrace 1, \dots, n \rbrace$; [`Error::Shape`] unless
    /// the table has $2^{\ell''}$ entries; [`Error::Core`] unless the blinders are $\ell'' \times
    /// d\_0$ and $\ell'' \times d\_1$; [`Error::Parameter`] if $d\_0 \lt 2b + 1$ or $d\_1 \lt 2$
    /// (the round polynomials would not fit in the messages) or $q \lt 2b + 2$.
    pub fn new(
        ctx: &Context<K>,
        party: PartyId,
        w_table: Vec<u64>,
        pad: Option<Ext<K>>,
        g0: Blinder<K>,
        g1: Blinder<K>,
    ) -> Result<Self> {
        let l = &ctx.layout;
        if party == 0 || party > l.n {
            return Err(Error::InvalidParty { party, n: l.n });
        }
        check_len(
            "<w~>_p table (2^l'' entries)",
            table_size(l)?,
            w_table.len(),
        )?;
        g0.validate(l.ell_pp, l.d0)?;
        g1.validate(l.ell_pp, l.d1)?;
        let field = ctx.field().clone();
        let b = ctx.b();
        // deg F_0 <= 2b + 1 and deg F_1 <= 2 in each variable.
        let deg0 = usize::try_from(b)
            .ok()
            .and_then(|b| b.checked_mul(2))
            .and_then(|x| x.checked_add(1))
            .ok_or_else(|| Error::Parameter(format!("the base b = {b} is too large")))?;
        if l.d0 < deg0 || l.d1 < 2 {
            return Err(Error::Parameter(format!(
                "d0 = {} and d1 = {} do not bound the degrees 2b + 1 = {deg0} and 2 of F_0 and F_1",
                l.d0, l.d1
            )));
        }
        let (direct0, direct1) = (deg0 + 1, 3);
        let zq = field.zq();
        let ext0 = Extrapolation::new(zq, direct0, l.d0 + 1)?;
        let ext1 = Extrapolation::new(zq, direct1, l.d1 + 1)?;
        let sums = (hypercube_sum(&field, &g0), hypercube_sum(&field, &g1));
        let binomials = binomial_row(zq, direct0 - 1);
        Ok(Self {
            range: RangeProduct::new(zq, b),
            b,
            mont: Montgomery::new(zq.modulus()),
            e_rows: row_indicator(l, &l.s_ranges(party))?,
            ind_rows: row_indicator(l, &[ind_rows(l)])?,
            pad: pad.map(|p| (pad_index(l), field.reduce(&p))),
            field,
            party,
            n: l.n,
            log_n: l.log_n,
            ell_u: l.ell_u,
            ell_pp: l.ell_pp,
            two_d: 2 * l.d,
            d0: l.d0,
            d1: l.d1,
            direct0,
            direct1,
            ext0,
            ext1,
            binomials,
            w_base: w_table,
            g0,
            g1,
            sums,
            run: None,
            plane_share: PLANE_SHARE,
        })
    }

    /// Whether round 2 reads the base-field table and the first coin ([`PlaneValues`]) instead of
    /// the table folded by the first coin, which is then never formed: when round 2 is a round
    /// over $U$ and the table of the range product is small next to the hypercube.
    fn second_from_table(&self) -> bool {
        let nodes = self.direct0.max(self.direct1);
        let digit = self.b.saturating_sub(1) as i64;
        self.ell_pp >= 2
            && self.ell_u >= 2
            && PlaneValues::<K>::entries(digit, nodes).saturating_mul(self.plane_share)
                <= self.w_base.len()
    }

    /// Whether round 3 reads the base-field table and the first two coins too
    /// ([`bound_two_entry`]), so that the witness table bound to two coins is never formed and
    /// the third bind forms it bound to three: when round 2 reads the table and round 3 is a
    /// round over $U$.
    fn third_from_table(&self) -> bool {
        self.second_from_table() && self.ell_pp >= 3 && self.ell_u >= 3
    }

    /// The party $p$.
    pub fn party(&self) -> PartyId {
        self.party
    }

    /// $(G\_{0,p}, G\_{1,p})$, the hypercube sums of the blinders (`5_Building_Blocks.tex`:247-248).
    pub fn hypercube_sums(&self) -> (Ext<K>, Ext<K>) {
        self.sums
    }

    /// The hypercube table of $\langle \widetilde w \rangle\_p$ without a field pad (base field).
    pub fn w_table(&self) -> &[u64] {
        &self.w_base
    }

    /// Lends the table of [`PartyProver::w_table`], which is then empty, to a caller that gives
    /// it back unchanged with [`PartyProver::restore_w_table`] before the prover is used again:
    /// step 3 of the driver transforms it in place into the commitment's coefficients and back,
    /// so that it is not copied.
    pub(crate) fn take_w_table(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.w_base)
    }

    /// Takes back the table of [`PartyProver::take_w_table`].
    ///
    /// # Errors
    /// [`Error::Shape`] unless it has the table's $2^{\ell''}$ entries.
    pub(crate) fn restore_w_table(&mut self, table: Vec<u64>) -> Result<()> {
        check_len(
            "<w~>_p table (2^l'' entries)",
            1 << self.ell_pp,
            table.len(),
        )?;
        self.w_base = table;
        Ok(())
    }

    /// The pad $\rho^w\_p \in \mathbb F\_{q^K}$, if the pad is a field element.
    pub fn pad(&self) -> Option<Ext<K>> {
        self.pad.map(|(_, value)| value)
    }

    /// The pad's index in the table and its value, if the pad is a field element.
    pub fn pad_entry(&self) -> Option<(usize, Ext<K>)> {
        self.pad
    }

    /// The hypercube table of $\langle \widetilde w \rangle\_p$ over $\mathbb F\_{q^K}$, the pad
    /// included.
    pub fn w_field_table(&self) -> Vec<Ext<K>> {
        lift_table(&self.field, &self.w_base, self.pad)
    }

    /// The blinders $g\_{0,p}$ and $g\_{1,p}$.
    pub fn blinders(&self) -> (&Blinder<K>, &Blinder<K>) {
        (&self.g0, &self.g1)
    }

    /// The coins $a\_1, \dots$ bound so far (empty before [`PartyProver::start`]).
    pub fn point(&self) -> &[Ext<K>] {
        self.run.as_ref().map_or(&[], |run| &run.point)
    }

    /// The table, the pad, the blinders and the point, consuming the prover.
    #[allow(clippy::type_complexity)] // the prover's parts, in order
    pub fn into_parts(
        self,
    ) -> (
        Vec<u64>,
        Option<(usize, Ext<K>)>,
        Blinder<K>,
        Blinder<K>,
        Vec<Ext<K>>,
    ) {
        let point = self.run.map(|run| run.point).unwrap_or_default();
        (self.w_base, self.pad, self.g0, self.g1, point)
    }

    /// Starts the rounds with the verifier's coins and the table of $M^\ast$
    /// ([`mstar_table`](crate::mstar_table)), which the prover takes over and folds: builds
    /// $\widetilde{eq}(\tau\_0\[0..\ell\_U\], \cdot)$, $\widetilde{eq}(\tau\_0\[\ell\_U..\ell''\], \cdot)$ and
    /// $\widetilde{eq}(\tau\_0\[\ell''..\], \mathsf{bit}(p))$.
    ///
    /// # Errors
    /// [`Error::State`] if already started; [`Error::Shape`] unless $\tau\_0$ has
    /// $\ell'' + \lceil \log n \rceil$ entries and the table $2^{\ell''}$.
    pub fn start(&mut self, coins: &SumcheckCoins<K>, mstar: Vec<Ext<K>>) -> Result<()> {
        self.start_with(coins, MStar::Table(mstar))
    }

    /// [`PartyProver::start`] with $M^\ast$ given by its parts ([`MStarParts`]): round 1 forms
    /// its entries where it reads them and the first bind forms the table bound to $a\_1$, so
    /// that the whole table is never held. The same values as with the table of
    /// [`mstar_table_rows`](crate::mstar_table_rows).
    ///
    /// # Errors
    /// As [`PartyProver::start`].
    pub fn start_parts(&mut self, coins: &SumcheckCoins<K>, mstar: MStarParts<K>) -> Result<()> {
        self.start_with(coins, MStar::Parts(mstar))
    }

    fn start_with(&mut self, coins: &SumcheckCoins<K>, mstar: MStar<K>) -> Result<()> {
        if self.run.is_some() {
            return Err(Error::State("start called twice".into()));
        }
        let f = &self.field;
        check_len(
            "tau_0 (l'' + ceil(log n))",
            self.ell_pp + self.log_n,
            coins.tau0.len(),
        )?;
        check_len("M* table (2^l'' entries)", self.w_base.len(), mstar.len())?;
        let tau0: Vec<Ext<K>> = coins.tau0.iter().map(|t| f.reduce(t)).collect();
        let lift = |rows: &[u64]| -> Vec<Ext<K>> { rows.iter().map(|x| f.from_base(*x)).collect() };
        self.run = Some(Running {
            rho: f.reduce(&coins.rho),
            eq_party: eq_point(
                f,
                &tau0[self.ell_pp..],
                &msb_bits(self.party - 1, self.log_n),
            )?,
            w: Vec::new(),
            mstar,
            eq_u: eq_table(f, &tau0[..self.ell_u])?,
            eq_l: eq_table(f, &tau0[self.ell_u..self.ell_pp])?,
            e: lift(&self.e_rows),
            ind: lift(&self.ind_rows),
            g0: BlinderRounds::new(f, &self.g0),
            g1: BlinderRounds::new(f, &self.g1),
            point: Vec::with_capacity(self.ell_pp),
            tau: tau0[..self.ell_pp].to_vec(),
        });
        Ok(())
    }

    /// The current round $r$ (1-based; $\ell'' + 1$ after the last bind), or `None` before
    /// [`PartyProver::start`].
    pub fn round(&self) -> Option<usize> {
        self.run.as_ref().map(|run| run.point.len() + 1)
    }

    fn running(&self, what: &str) -> Result<(&Running<K>, usize)> {
        let run = self
            .run
            .as_ref()
            .ok_or_else(|| Error::State(format!("{what} before start")))?;
        let r = run.point.len() + 1;
        if r > self.ell_pp {
            return Err(Error::State(format!("{what} after the last round")));
        }
        Ok((run, r))
    }

    /// Party $p$'s values of the current round $r$: its input
    /// $\bigl(H\_{0,p,r}(t)\bigr)\_{t \le d\_0 + 1} \Vert \bigl(H\_{1,p,r}(t)\bigr)\_{t \le d\_1 + 1}$ to
    /// $\mathcal F\_{\mathsf{AH}}(\mathbb I)$ (`5_Building_Blocks.tex`:326) and the unblinded
    /// $F\_{0,p,r}(t)$, $F\_{1,p,r}(t)$.
    ///
    /// # Errors
    /// [`Error::State`] before [`PartyProver::start`] or after the last round.
    pub fn round_values(&self) -> Result<RoundValues<K>> {
        let (run, r) = self.running("round_values")?;
        let f = &self.field;
        let (mut f0, mut f1) = match &self.mont {
            Some(mont) => self.dispatch(mont, run, r),
            None => self.dispatch(f, run, r),
        };
        // The kernel takes the eq factor of the round's variable out of F_0 and leaves it at one
        // node fewer: F_0 at the nodes 1, ..., direct0 from it.
        f0 = unfactor(f, &f0, &run.tau[r - 1], &self.binomials);
        self.ext0.extend(f, &mut f0)?;
        self.ext1.extend(f, &mut f1)?;
        let blind = |values: &[Ext<K>], blinder: Vec<Ext<K>>| -> Vec<Ext<K>> {
            values
                .iter()
                .zip(&blinder)
                .map(|(v, g)| f.add(v, &f.mul(&run.rho, g)))
                .collect()
        };
        let h0 = blind(&f0, run.g0.values(f, &self.g0, r, self.d0 + 1));
        let h1 = blind(&f1, run.g1.values(f, &self.g1, r, self.d1 + 1));
        Ok(RoundValues { h0, h1, f0, f1 })
    }

    /// The kernel on the base-field table in round 1 and on the folded table afterwards.
    fn dispatch<A: Arith<K> + Sync>(
        &self,
        arith: &A,
        run: &Running<K>,
        r: usize,
    ) -> (Vec<Ext<K>>, Vec<Ext<K>>) {
        if r == 1 {
            let (f, wb) = (&self.field, &self.w_base);
            let (at, value) = self.pad.unwrap_or((usize::MAX, f.zero()));
            let nodes = self.direct0.max(self.direct1);
            let small = SmallValues::new(arith, f, &self.range, self.b, nodes);
            self.kernel(arith, run, r, wb.len(), Some((&small, wb, at)), None, |x| {
                if x == at { value } else { f.from_base(wb[x]) }
            })
        } else if r == 2 && run.w.is_empty() {
            // The table folded by a_1 was not formed (`second_from_table`): points with small
            // entries read `PlaneValues`, the others the folded entries formed on the fly as
            // `fold_tables` forms them.
            let (f, wb) = (&self.field, &self.w_base);
            let (at, value) = self.pad.unwrap_or((usize::MAX, f.zero()));
            let a1 = run.point[0];
            let half = wb.len() / 2;
            let base = |i: usize| if i == at { value } else { f.from_base(wb[i]) };
            let nodes = self.direct0.max(self.direct1);
            let small = SmallValues::new(arith, f, &self.range, self.b, nodes);
            let plane = PlaneValues::new(arith, f, &self.range, &small, &a1);
            self.kernel(arith, run, r, half, None, Some((&plane, wb, at)), |y| {
                f.add(&base(y), &f.mul(&a1, &f.sub(&base(y + half), &base(y))))
            })
        } else if r == 3 && run.w.is_empty() {
            // The table bound to a_1 and a_2 was not formed (`third_from_table`): its entries are
            // formed where they are read, as `fold_tables` formed them.
            let (f, wb) = (&self.field, &self.w_base);
            let (a1, a2) = (run.point[0], run.point[1]);
            let a12 = f.mul(&a1, &a2);
            let pad = self.pad.as_ref();
            self.kernel(arith, run, r, wb.len() / 4, None, None, |x| {
                bound_two_entry(f, wb, pad, &a1, &a2, &a12, x)
            })
        } else {
            let w = &run.w;
            self.kernel(arith, run, r, w.len(), None, None, |x| w[x])
        }
    }

    /// The sums of round $r$ at the nodes $1, \dots, \mathrm{direct}\_0$ (for $F\_{0,p}$) and
    /// $1, \dots, \mathrm{direct}\_1$ (for $F\_{1,p}$), for the current table of
    /// $\langle \widetilde w \rangle\_p$ of `len` entries read through `w_at`. The tables are
    /// canonical; the values of a pair of points and the sums are formed in the working
    /// representation of `arith` and the results returned canonical. The rows, or the pairs of
    /// points, run on the Rayon pool; field sums do not depend on the order. `round1` gives the
    /// small values, the table over $\mathbb Z\_q$ and the index of a field pad, if any.
    #[allow(clippy::too_many_arguments)] // the tables and the paths of rounds 1 and 2
    fn kernel<A: Arith<K> + Sync, W: Fn(usize) -> Ext<K> + Sync>(
        &self,
        arith: &A,
        run: &Running<K>,
        r: usize,
        len: usize,
        round1: Option<(&SmallValues, &[u64], usize)>,
        round2: Option<(&PlaneValues<K>, &[u64], usize)>,
        w_at: W,
    ) -> (Vec<Ext<K>>, Vec<Ext<K>>) {
        let f = &self.field;
        let zero = f.zero(); // phi(0) = 0 in every representation
        let one = arith.enter(&f.one());
        let factors = RangePairs::new(arith, f.zq(), self.range.squares());
        let h = len / 2;
        // F_0 without the eq factor of the round's variable, of one degree less: one node fewer.
        let (n0, n1) = (self.direct0 - 1, self.direct1);
        let (f0, f1) = if r <= self.ell_u {
            // U round: the lower half is rows u < h / 2d of 2d points; e_p, ind and eq_U are
            // row values, eq_L does not change.
            let rows = h / self.two_d;
            let eq_l: Vec<Ext<K>> = run.eq_l.iter().map(|e| arith.enter(e)).collect();
            let capacity = arith.wide_capacity();
            let start = || {
                (
                    Accumulators::new(f, n0, n1, capacity),
                    vec![zero; n0],
                    vec![zero; n1],
                )
            };
            let (_, f0, f1) = (0..rows)
                .into_par_iter()
                .fold(start, |(mut acc, mut f0, mut f1), u| {
                    acc.clear(f);
                    let base = u * self.two_d;
                    // Without e_p on this row, the range product does not count.
                    let range = !(f.is_zero(&run.e[u]) && f.is_zero(&run.e[u + rows]));
                    let mut sums: PlaneSums<K> = [zero; 4];
                    // The terms eq_L W of the other points of the row, by W at node 1 and its step
                    // (eq_L is the same at every node of a U round).
                    let mut pair_sums = [zero; 2];
                    for (l, e) in eq_l.iter().enumerate() {
                        let x = base + l;
                        if let Some((lo, hi)) = small_pair(f, round1, x, h) {
                            let (small, ..) = round1.expect("a small pair comes from round 1");
                            let m = (run.mstar.at(f, x), run.mstar.at(f, x + h));
                            self.accumulate_small(
                                arith, small, range, &mut acc, m, lo, hi, e, &zero,
                            );
                        } else if let Some(quad) = small_quad(f, round2, x, h) {
                            let (plane, ..) = round2.expect("a small quadruple comes from round 2");
                            let m = (run.mstar.at(f, x), run.mstar.at(f, x + h));
                            self.accumulate_plane(
                                arith, plane, range, &mut acc, &mut sums, m, quad, e,
                            );
                        } else {
                            let m = (run.mstar.at(f, x), run.mstar.at(f, x + h));
                            self.accumulate_pair(
                                arith,
                                &factors,
                                range,
                                &mut acc,
                                Some(&mut pair_sums),
                                &w_at,
                                m,
                                x,
                                h,
                                e,
                                &zero,
                            );
                        }
                    }
                    acc.flush(arith);
                    let [mut x, dx] = pair_sums;
                    for w in acc.w.iter_mut() {
                        *w = arith.add(w, &x);
                        x = arith.add(&x, &dx);
                    }
                    if let Some((plane, ..)) = round2 {
                        // The terms eq_L W of the points that read the plane, from their sums.
                        let [mut x, dx, mut y, dy] = sums;
                        for w in acc.w.iter_mut() {
                            let sum = arith.add(&x, &arith.mul(&plane.coin, &y));
                            *w = arith.add(w, &sum);
                            x = arith.add(&x, &dx);
                            y = arith.add(&y, &dy);
                        }
                    }
                    // The row factors at t = 1 and their increments.
                    let affine = |t: &[Ext<K>]| {
                        (
                            arith.enter(&t[u + rows]),
                            arith.enter(&f.sub(&t[u + rows], &t[u])),
                        )
                    };
                    // eq_U of the round's variable taken out (`unfactor`): the row's weight is the
                    // sum of its two row factors at every node.
                    let (mut eu, deu) =
                        (arith.enter(&f.add(&run.eq_u[u], &run.eq_u[u + rows])), zero);
                    let (mut ee, dee) = affine(&run.e);
                    let (mut ii, dii) = affine(&run.ind);
                    for ((out, rp), w) in f0.iter_mut().zip(&acc.rp).zip(&acc.w) {
                        let bracket = arith.sub(&arith.sub(&one, &ee), &ii);
                        let v = arith.add(&arith.mul(&ee, rp), &arith.mul(&bracket, w));
                        *out = arith.add(out, &arith.mul(&eu, &v));
                        eu = arith.add(&eu, &deu);
                        ee = arith.add(&ee, &dee);
                        ii = arith.add(&ii, &dii);
                    }
                    for (out, m) in f1.iter_mut().zip(&acc.m) {
                        *out = arith.add(out, m);
                    }
                    (acc, f0, f1)
                })
                .reduce(start, |(acc, mut f0, mut f1), (_, g0, g1)| {
                    for (x, y) in f0.iter_mut().zip(&g0) {
                        *x = arith.add(x, y);
                    }
                    for (x, y) in f1.iter_mut().zip(&g1) {
                        *x = arith.add(x, y);
                    }
                    (acc, f0, f1)
                });
            (f0, f1)
        } else {
            // L round: U is bound, so e_p, ind and eq_U are scalars; eq_L is folded with W.
            // Round 2 reads the base-field table only when it is a U round (`second_from_table`).
            debug_assert!(round2.is_none());
            let capacity = arith.wide_capacity();
            let start = || Accumulators::new(f, n0, n1, capacity);
            let range = !f.is_zero(&run.e[0]);
            let mut acc = (0..h)
                .into_par_iter()
                .fold(start, |mut acc, x| {
                    let (lo, hi) = (&run.eq_l[x], &run.eq_l[x + h]);
                    // eq_L of the round's variable taken out (`unfactor`): the pair's weight is
                    // the sum of its two entries at every node.
                    let (e_hi, de) = (arith.enter(&f.add(lo, hi)), zero);
                    let m = (run.mstar.at(f, x), run.mstar.at(f, x + h));
                    if let Some((a, b)) = small_pair(f, round1, x, h) {
                        let (small, ..) = round1.expect("a small pair comes from round 1");
                        self.accumulate_small(arith, small, range, &mut acc, m, a, b, &e_hi, &de);
                    } else {
                        self.accumulate_pair(
                            arith, &factors, range, &mut acc, None, &w_at, m, x, h, &e_hi, &de,
                        );
                    }
                    acc
                })
                .reduce(start, |mut acc, other| {
                    acc.merge(arith, other);
                    acc
                });
            acc.flush(arith);
            let (eu, ee, ii) = (
                arith.enter(&run.eq_u[0]),
                arith.enter(&run.e[0]),
                arith.enter(&run.ind[0]),
            );
            let bracket = arith.sub(&arith.sub(&one, &ee), &ii);
            let f0 = acc
                .rp
                .iter()
                .zip(&acc.w)
                .map(|(rp, w)| {
                    let v = arith.add(&arith.mul(&ee, rp), &arith.mul(&bracket, w));
                    arith.mul(&eu, &v)
                })
                .collect();
            (f0, acc.m)
        };
        // phi(v) times the canonical eq factor of the party is canonical.
        let f0: Vec<Ext<K>> = f0.iter().map(|v| arith.mul(v, &run.eq_party)).collect();
        let f1 = f1.iter().map(|v| arith.leave(v)).collect();
        (f0, f1)
    }

    /// Adds the contribution of the point pair $(x, x + h)$ at the nodes $1, 2, \dots$: the values
    /// at node $t + 1$ are those at node $t$ plus the difference of the pair. The factor
    /// $\widetilde{eq}\_L$ is given in the working representation, at node 1 (`e_hi`) with its
    /// increment `de`; with `range` false the range product is left out.
    ///
    /// The range product takes $u = W^2$ along the nodes by its increments,
    /// $(W + \Delta W)^2 = u + 2 W \Delta W + \Delta W^2$, and its factors two at a time
    /// ([`RangePairs`]). With `sums` (a round over $U$, where $\widetilde{eq}\_L$ is the same at
    /// every node) the terms $\widetilde{eq}\_L W$ go into the row's sums of $\widetilde{eq}\_L W$ at
    /// node 1 and of $\widetilde{eq}\_L \Delta W$, which the caller spreads over the nodes; otherwise
    /// into the accumulators node by node. Exact: the same sums as node by node.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn accumulate_pair<A: Arith<K>, W: Fn(usize) -> Ext<K>>(
        &self,
        arith: &A,
        factors: &RangePairs,
        range: bool,
        acc: &mut Accumulators<K>,
        sums: Option<&mut [Ext<K>; 2]>,
        w_at: &W,
        (m_lo, m_hi): (Ext<K>, Ext<K>),
        x: usize,
        h: usize,
        e_hi: &Ext<K>,
        de: &Ext<K>,
    ) {
        let f = &self.field;
        let (w_lo, w_hi) = (w_at(x), w_at(x + h));
        let (dw, dm) = (
            arith.enter(&f.sub(&w_hi, &w_lo)),
            arith.enter(&f.sub(&m_hi, &m_lo)),
        );
        let w = arith.enter(&w_hi);
        let (n0, n1) = (acc.rp.len(), acc.m.len());
        if range {
            let dw2 = arith.mul(&dw, &dw);
            let wdw = arith.mul(&w, &dw);
            let mut u = arith.mul(&w, &w);
            let mut du = arith.add(&arith.add(&wdw, &wdw), &dw2);
            let ddu = arith.add(&dw2, &dw2);
            let (mut wt, mut et) = (w, *e_hi);
            for wide in acc.rp_wide.iter_mut().take(n0) {
                arith.mul_wide(wide, &et, &factors.eval(arith, &u, &wt));
                u = arith.add(&u, &du);
                du = arith.add(&du, &ddu);
                wt = arith.add(&wt, &dw);
                et = arith.add(&et, de);
            }
        }
        match sums {
            Some([at_one, step]) => {
                *at_one = arith.add(at_one, &arith.mul(e_hi, &w));
                *step = arith.add(step, &arith.mul(e_hi, &dw));
            }
            None => {
                let (mut wt, mut et) = (w, *e_hi);
                for wide in acc.w_wide.iter_mut().take(n0) {
                    arith.mul_wide(wide, &et, &wt);
                    wt = arith.add(&wt, &dw);
                    et = arith.add(&et, de);
                }
            }
        }
        let (mut wt, mut m) = (w, arith.enter(&m_hi));
        for wide in acc.m_wide.iter_mut().take(n1) {
            arith.mul_wide(wide, &wt, &m);
            wt = arith.add(&wt, &dw);
            m = arith.add(&m, &dm);
        }
        acc.added(arith);
    }

    /// [`PartyProver::accumulate_pair`] for a pair of small integers `lo` and `hi`: their node
    /// values and range products come from `small`, in the working representation, so that
    /// each term is a product by a residue.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn accumulate_small<A: Arith<K>>(
        &self,
        arith: &A,
        small: &SmallValues,
        range: bool,
        acc: &mut Accumulators<K>,
        (m_lo, m_hi): (Ext<K>, Ext<K>),
        lo: i64,
        hi: i64,
        e_hi: &Ext<K>,
        de: &Ext<K>,
    ) {
        let f = &self.field;
        let dm = arith.enter(&f.sub(&m_hi, &m_lo));
        let (mut w, mut m, mut e) = (hi, arith.enter(&m_hi), *e_hi);
        let (n0, n1) = (acc.rp.len(), acc.m.len());
        for t in 0..n0.max(n1) {
            let index = (w + small.reach) as usize;
            let value = small.value[index];
            if t < n0 {
                if range {
                    let rp = arith.scale_out(&e, small.range[index]);
                    acc.rp[t] = arith.add(&acc.rp[t], &rp);
                }
                acc.w[t] = arith.add(&acc.w[t], &arith.scale_out(&e, value));
            }
            if t < n1 {
                acc.m[t] = arith.add(&acc.m[t], &arith.scale_out(&m, value));
            }
            w += hi - lo;
            m = arith.add(&m, &dm);
            e = arith.add(&e, de);
        }
    }

    /// [`PartyProver::accumulate_pair`] in round 2 for a point $x$ whose entries
    /// $A, B, C, E$ of the base-field table are small integers ([`PlaneValues`]): the range
    /// products come from the plane, one product by $\widetilde{eq}\_L$ per node; the terms
    /// $\widetilde{eq}\_L W$ go into `sums` by the integers $X\_1, \Delta X, Y\_1, \Delta Y$ (a round over
    /// $U$, where $\widetilde{eq}\_L$ is the same at every node); the terms $W M^\ast$ are formed from
    /// $\phi(W)$ at node 1 and its increment.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn accumulate_plane<A: Arith<K>>(
        &self,
        arith: &A,
        plane: &PlaneValues<K>,
        range: bool,
        acc: &mut Accumulators<K>,
        sums: &mut PlaneSums<K>,
        (m_lo, m_hi): (Ext<K>, Ext<K>),
        (a, b, c, e_entry): (i64, i64, i64, i64),
        e: &Ext<K>,
    ) {
        let f = &self.field;
        // Node t is the point t + 1: X and Y at point 1 and their increments.
        let (x1, dx) = (c, c - a);
        let (y1, dy) = (e_entry - c, (e_entry - c) - (b - a));
        let (n0, n1) = (acc.rp.len(), acc.m.len());
        if range {
            let width = plane.width as i64;
            let mut index = (x1 + plane.reach) * width + y1 + 2 * plane.reach;
            let step = dx * width + dy;
            for rp in acc.rp.iter_mut().take(n0) {
                *rp = arith.add(rp, &arith.mul(e, &plane.range[index as usize]));
                index += step;
            }
        }
        for (sum, v) in sums.iter_mut().zip([x1, dx, y1, dy]) {
            *sum = arith.add(sum, &arith.scale_out(e, plane.value(v)));
        }
        let point = |x: i64, y: i64| {
            let mut w = arith.scale_out(&plane.coin, plane.value(y));
            w = arith.add(&w, &base_ext(plane.value(x)));
            w
        };
        let (mut w, dw) = (point(x1, y1), point(dx, dy));
        let dm = arith.enter(&f.sub(&m_hi, &m_lo));
        let mut m = arith.enter(&m_hi);
        for out in acc.m.iter_mut().take(n1) {
            *out = arith.add(out, &arith.mul(&w, &m));
            w = arith.add(&w, &dw);
            m = arith.add(&m, &dm);
        }
    }

    /// Binds the variable of the current round to the coin $a\_r$ in every table
    /// ($T'\[x\] = T\[x\] + a\_r (T\[x + h\] - T\[x\])$, as [`bind_top`](crate::bind_top)) and in the
    /// blinders' round polynomials.
    ///
    /// # Errors
    /// [`Error::State`] before [`PartyProver::start`] or after the last round.
    pub fn bind(&mut self, a: &Ext<K>) -> Result<()> {
        let r = self.running("bind")?.1;
        let a = self.field.reduce(a);
        let defer = self.second_from_table();
        let defer3 = self.third_from_table();
        let Some(run) = self.run.as_mut() else {
            return Err(Error::State("bind before start".into()));
        };
        match &self.mont {
            Some(mont) => fold_tables(
                mont,
                &self.field,
                &self.w_base,
                self.pad.as_ref(),
                run,
                r,
                self.ell_u,
                &a,
                (defer, defer3),
            ),
            None => fold_tables(
                &self.field,
                &self.field,
                &self.w_base,
                self.pad.as_ref(),
                run,
                r,
                self.ell_u,
                &a,
                (defer, defer3),
            ),
        }
        let f = &self.field;
        run.g0.bind(f, &self.g0, r, &a);
        run.g1.bind(f, &self.g1, r, &a);
        run.point.push(a);
        Ok(())
    }

    /// The openings of `5_Building_Blocks.tex`:348-350 after the last round:
    /// $(\langle \widetilde w \rangle\_p(\mathbf a), g\_{0,p}(\mathbf a), g\_{1,p}(\mathbf a))$. The first
    /// is the fully folded table, the others [`evaluate`].
    ///
    /// # Errors
    /// [`Error::State`] unless all $\ell''$ rounds are bound.
    pub fn final_values(&self) -> Result<(Ext<K>, Ext<K>, Ext<K>)> {
        let run = self
            .run
            .as_ref()
            .ok_or_else(|| Error::State("final_values before start".into()))?;
        if run.point.len() != self.ell_pp || run.w.len() != 1 {
            return Err(Error::State(format!(
                "final_values after {} of {} rounds",
                run.point.len(),
                self.ell_pp
            )));
        }
        let f = &self.field;
        Ok((
            run.w[0],
            evaluate(f, &self.g0, &run.point)?,
            evaluate(f, &self.g1, &run.point)?,
        ))
    }

    /// The number of parties $n$.
    pub fn parties(&self) -> usize {
        self.n
    }
}

/// The centered entries of the point pair $(x, x + h)$ in round 1, if both are small and neither
/// is a field pad.
#[inline(always)]
fn small_pair<const K: usize>(
    f: &Fqk<K>,
    round1: Option<(&SmallValues, &[u64], usize)>,
    x: usize,
    h: usize,
) -> Option<(i64, i64)> {
    let (small, table, pad) = round1?;
    if x == pad || x + h == pad {
        return None;
    }
    Some((small.small(f, table[x])?, small.small(f, table[x + h])?))
}

/// The centered entries $A, B, C, E$ of the base-field table at $x$, $x + 2h$, $x + h$ and
/// $x + 3h$ for round 2's pair $(x, x + h)$ (with $h = 2^{\ell''}/4$), if all four are small and
/// none is a field pad.
#[inline(always)]
fn small_quad<const K: usize>(
    f: &Fqk<K>,
    round2: Option<(&PlaneValues<K>, &[u64], usize)>,
    x: usize,
    h: usize,
) -> Option<(i64, i64, i64, i64)> {
    let (plane, table, pad) = round2?;
    let at = [x, x + 2 * h, x + h, x + 3 * h];
    if at.contains(&pad) {
        return None;
    }
    Some((
        plane.small(f, table[at[0]])?,
        plane.small(f, table[at[1]])?,
        plane.small(f, table[at[2]])?,
        plane.small(f, table[at[3]])?,
    ))
}

/// The element with first coordinate `c` and the others zero ($\phi(c)$ for $\phi(c)$ given).
#[inline(always)]
fn base_ext<const K: usize>(c: u64) -> Ext<K> {
    let mut out = [0u64; K];
    out[0] = c;
    Ext(out)
}

/// The factors $u - c^2$ ($u = w^2$, $c = 1, \dots, b - 1$) of the range product, two at a time, in
/// the working representation: $(u - a)(u - b) = u^2 - (a + b) u + ab$ with the constants $a + b$
/// and $-ab$, and a last factor $u - c^2$ when their number is odd. [`RangePairs::eval`] is
/// [`range_eval`] with one product by a constant in place of one product of two in every pair.
struct RangePairs {
    /// $\phi(a + b)$ and $\phi(-ab)$ of each pair.
    pairs: Vec<(u64, u64)>,
    /// $\phi(c^2)$ of the last factor, if their number is odd.
    single: Option<u64>,
}

impl RangePairs {
    /// The pairs of the canonical `squares` $c^2$, in order.
    fn new<const K: usize, A: Arith<K>>(arith: &A, zq: &Zq, squares: &[u64]) -> Self {
        let (chunks, rest) = squares.as_chunks::<2>();
        let pairs = chunks
            .iter()
            .map(|[a, b]| {
                (
                    arith.enter_base(zq.add(*a, *b)),
                    arith.enter_base(zq.neg(zq.mul(*a, *b))),
                )
            })
            .collect();
        let single = rest.first().map(|c| arith.enter_base(*c));
        Self { pairs, single }
    }

    /// $w \prod\_c (u - c^2)$ for $u = w^2$, both in the working representation: [`range_eval`] of
    /// $w$.
    #[inline(always)]
    fn eval<const K: usize, A: Arith<K>>(&self, arith: &A, u: &Ext<K>, w: &Ext<K>) -> Ext<K> {
        let mut product: Option<Ext<K>> = None;
        if !self.pairs.is_empty() {
            let u2 = arith.mul(u, u);
            for (sum, neg_product) in &self.pairs {
                let v = arith.sub_base(&arith.sub(&u2, &arith.scale_out(u, *sum)), *neg_product);
                product = Some(product.map_or(v, |x| arith.mul(&x, &v)));
            }
        }
        if let Some(c) = self.single {
            let v = arith.sub_base(u, c);
            product = Some(product.map_or(v, |x| arith.mul(&x, &v)));
        }
        product.map_or(*w, |x| arith.mul(&x, w))
    }
}

/// $\prod\_{c=-b+1}^{b-1} (w - c) = w \prod\_{c=1}^{b-1} (w^2 - c^2)$ in the working representation,
/// with the constants $c^2$ in that representation ([`RangeProduct`] for the canonical one): the
/// reference of [`RangePairs::eval`] in the tests.
#[cfg(test)]
fn range_eval<const K: usize, A: Arith<K>>(arith: &A, w: &Ext<K>, squares: &[u64]) -> Ext<K> {
    match squares.split_first() {
        None => *w,
        Some((first, rest)) => {
            let w2 = arith.mul(w, w);
            let mut prod = arith.sub_base(&w2, *first);
            for c2 in rest {
                prod = arith.mul(&prod, &arith.sub_base(&w2, *c2));
            }
            arith.mul(&prod, w)
        }
    }
}

/// $\binom{m}{i} \bmod q$ for $i \le m$, by Pascal's rule: exact for any $m$.
fn binomial_row(zq: &Zq, m: usize) -> Vec<u64> {
    let mut row = vec![1u64];
    for i in 1..=m {
        for j in (1..i).rev() {
            row[j] = zq.add(row[j], row[j - 1]);
        }
        row.push(1);
    }
    row
}

/// $F\_{0,p}$ at the nodes $1, \dots, m + 1$ from `q`, its values at the nodes $1, \dots, m$
/// without the factor of $\widetilde{eq}$ for the round's variable: $\widetilde{eq}\_U$ (a round
/// over $U$) or $\widetilde{eq}\_L$ (over $L$) of the two halves are $c\,\widetilde{eq}(\tau', \cdot)$
/// times $1 - \tau\_r$ and $\tau\_r$, so at node $t$ they weigh $\widetilde{eq}(\tau\_r, t)$ times
/// their sum, which the kernel uses ([`PartyProver::kernel`]). Then $F\_{0,p}(t) =
/// \widetilde{eq}(\tau\_r, t) Q(t)$ with $Q$ of degree $m - 1$, whose $m$-th difference vanishes:
/// $Q(m + 1) = \sum\_{i \lt m} (-1)^{m - 1 - i} \binom{m}{i} Q(1 + i)$ (`binomials` the row of
/// $m$). Exact, so the values that the nodes give one by one.
fn unfactor<const K: usize>(
    f: &Fqk<K>,
    q: &[Ext<K>],
    tau: &Ext<K>,
    binomials: &[u64],
) -> Vec<Ext<K>> {
    let m = q.len();
    let mut last = f.zero();
    for (i, qi) in q.iter().enumerate() {
        let term = f.scale(qi, binomials[i]);
        last = if (m - 1 - i).is_multiple_of(2) {
            f.add(&last, &term)
        } else {
            f.sub(&last, &term)
        };
    }
    // eq(tau, t) = (1 - tau)(1 - t) + tau t at t = 1, 2, ...: tau, then steps of 2 tau - 1.
    let mut et = *tau;
    let de = f.sub(&f.add(tau, tau), &f.one());
    q.iter()
        .chain(core::iter::once(&last))
        .map(|qt| {
            let v = f.mul(&et, qt);
            et = f.add(&et, &de);
            v
        })
        .collect()
}

/// $T'\[x\] = T\[x\] + a (T\[x + h\] - T\[x\])$ on a canonical table, with the coin in the working
/// representation (so that the product with a canonical difference is canonical).
fn fold<const K: usize, A: Arith<K> + Sync>(
    arith: &A,
    f: &Fqk<K>,
    table: &mut Vec<Ext<K>>,
    coin: &Ext<K>,
) {
    let h = table.len() / 2;
    let (low, high) = table.split_at_mut(h);
    low.par_iter_mut()
        .zip(high.par_iter())
        .for_each(|(lo, hi)| {
            *lo = f.add(lo, &arith.mul(coin, &f.sub(hi, lo)));
        });
    table.truncate(h);
    table.shrink_to_fit();
}

/// Entry $x$ of the witness table bound to the first two coins $a\_1, a\_2$, from the base-field
/// `w_base` ($n$ entries): $W\[x\] + a\_2 (W\[x + n/4\] - W\[x\])$ for the table $W$ folded by
/// $a\_1$, which is $A + a\_1 (B - A) + a\_2 (C - A) + a\_1 a\_2 ((E - C) - (B - A))$ for the entries
/// $A, B, C, E$ at $x$, $x + n/2$, $x + n/4$ and $x + 3n/4$; the pad's quadruple in the field, as
/// the two folds form it. `a12` is $a\_1 a\_2$.
#[inline(always)]
fn bound_two_entry<const K: usize>(
    f: &Fqk<K>,
    w_base: &[u64],
    pad: Option<&(usize, Ext<K>)>,
    a1: &Ext<K>,
    a2: &Ext<K>,
    a12: &Ext<K>,
    x: usize,
) -> Ext<K> {
    let quarter = w_base.len() / 4;
    let (ia, ib, ic, ie) = (x, x + 2 * quarter, x + quarter, x + 3 * quarter);
    if let Some((at, value)) = pad.filter(|(at, _)| [ia, ib, ic, ie].contains(at)) {
        let base = |i: usize| {
            if i == *at {
                *value
            } else {
                f.from_base(w_base[i])
            }
        };
        let lo = f.add(&base(ia), &f.mul(a1, &f.sub(&base(ib), &base(ia))));
        let hi = f.add(&base(ic), &f.mul(a1, &f.sub(&base(ie), &base(ic))));
        return f.add(&lo, &f.mul(a2, &f.sub(&hi, &lo)));
    }
    let zq = f.zq();
    let (va, vb, vc, ve) = (w_base[ia], w_base[ib], w_base[ic], w_base[ie]);
    let ba = zq.sub(vb, va);
    let cross = zq.sub(zq.sub(ve, vc), ba);
    let v = f.add(&f.from_base(va), &f.scale(a1, ba));
    let v = f.add(&v, &f.scale(a2, zq.sub(vc, va)));
    f.add(&v, &f.scale(a12, cross))
}

/// Binds round $r$'s variable to $a$ in every table of `run` (the witness table from the
/// base-field `w_base` and the pad in round 1). With `defer.0` (round 2 reads the base-field
/// table), the witness table is not folded in round 1, and in round 2 it is bound to both coins
/// at once from `w_base` ([`bound_two_entry`]); with `defer.1` as well (round 3 reads it too) it is
/// not formed in round 2 either, and in round 3 it is formed bound to the three coins from the
/// entries bound to two, each as [`fold`] binds a table.
#[allow(clippy::too_many_arguments)] // the tables and the coin
fn fold_tables<const K: usize, A: Arith<K> + Sync>(
    arith: &A,
    f: &Fqk<K>,
    w_base: &[u64],
    pad: Option<&(usize, Ext<K>)>,
    run: &mut Running<K>,
    r: usize,
    ell_u: usize,
    a: &Ext<K>,
    (defer, defer3): (bool, bool),
) {
    let coin = arith.enter(a);
    // M* first, so that its table is not held at its size before the bind next to the witness
    // table formed below. The first bind keeps the parts and a_1, the second the parts and both
    // coins where a third round reads them (rounds 2 and 3 form the entries bound to the coins),
    // and the next forms the table bound to all of them from the parts, each entry as `fold`
    // forms it from the table.
    run.mstar = match core::mem::replace(&mut run.mstar, MStar::Table(Vec::new())) {
        MStar::Table(mut table) => {
            fold(arith, f, &mut table, &coin);
            MStar::Table(table)
        }
        MStar::Parts(parts) => MStar::Bound(parts, Coin::new(f, a)),
        MStar::Bound(parts, a1) if run.tau.len() >= 3 => MStar::Bound2(parts, a1, Coin::new(f, a)),
        bound @ (MStar::Bound(..) | MStar::Bound2(..)) => {
            let h = bound.len() / 2;
            MStar::Table(
                (0..h)
                    .into_par_iter()
                    .map(|x| {
                        let (lo, hi) = (bound.at(f, x), bound.at(f, x + h));
                        f.add(&lo, &arith.mul(&coin, &f.sub(&hi, &lo)))
                    })
                    .collect(),
            )
        }
    };
    if r == 1 && defer {
        // Round 2 reads w_base and a_1 (`PlaneValues`).
    } else if r == 2 && defer3 {
        // Round 3 reads w_base, a_1 and a_2 (`bound_two_entry`).
    } else if r == 2 && defer {
        let (a1, a12) = (run.point[0], f.mul(&run.point[0], a));
        run.w = (0..w_base.len() / 4)
            .into_par_iter()
            .map(|x| bound_two_entry(f, w_base, pad, &a1, a, &a12, x))
            .collect();
    } else if r == 3 && defer3 {
        let (a1, a2) = (run.point[0], run.point[1]);
        let a12 = f.mul(&a1, &a2);
        let eighth = w_base.len() / 8;
        run.w = (0..eighth)
            .into_par_iter()
            .map(|x| {
                let lo = bound_two_entry(f, w_base, pad, &a1, &a2, &a12, x);
                let hi = bound_two_entry(f, w_base, pad, &a1, &a2, &a12, x + eighth);
                f.add(&lo, &arith.mul(&coin, &f.sub(&hi, &lo)))
            })
            .collect();
    } else if r == 1 {
        let zq = f.zq();
        let (low, high) = w_base.split_at(w_base.len() / 2);
        run.w = low
            .par_iter()
            .zip(high.par_iter())
            .map(|(lo, hi)| f.add(&f.from_base(*lo), &arith.scale_out(&coin, zq.sub(*hi, *lo))))
            .collect();
        // The pad's pair has one entry outside the base field.
        if let Some((at, value)) = pad {
            let h = w_base.len() / 2;
            let x = at % h;
            let (lo, hi) = if *at < h {
                (*value, f.from_base(w_base[x + h]))
            } else {
                (f.from_base(w_base[x]), *value)
            };
            run.w[x] = f.add(&lo, &arith.mul(&coin, &f.sub(&hi, &lo)));
        }
    } else {
        fold(arith, f, &mut run.w, &coin);
    }
    if r <= ell_u {
        fold(arith, f, &mut run.eq_u, &coin);
        fold(arith, f, &mut run.e, &coin);
        fold(arith, f, &mut run.ind, &coin);
    } else {
        fold(arith, f, &mut run.eq_l, &coin);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blinder::round_poly;
    use crate::eq::bind_top;
    use hachi_co_params::{derive, presets::toy_choices};

    /// splitmix64 (seed in each test).
    struct SplitMix(u64);
    impl SplitMix {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn ext(&mut self, q: u64) -> Ext<4> {
            Ext(core::array::from_fn(|_| self.next() % q))
        }
    }

    /// The oracle's direct computation of one round (`sumcheck.prove`) over full tables of the
    /// current size: every node t = 1..=D+1, every table at T[x] + t (T[x + h] - T[x]), and the
    /// literal range product.
    #[allow(clippy::too_many_arguments)]
    fn direct_round(
        f: &Fqk<4>,
        b: u64,
        eq_party: &Ext<4>,
        eq: &[Ext<4>],
        e: &[Ext<4>],
        ind: &[Ext<4>],
        w: &[Ext<4>],
        ms: &[Ext<4>],
        d0: usize,
        d1: usize,
    ) -> (Vec<Ext<4>>, Vec<Ext<4>>) {
        let h = w.len() / 2;
        let at = |tab: &[Ext<4>], x: usize, t: u64| {
            f.add(&tab[x], &f.scale(&f.sub(&tab[x + h], &tab[x]), t))
        };
        let literal = |w: &Ext<4>| {
            let mut acc = f.one();
            for c in -(b as i64) + 1..=(b as i64) - 1 {
                acc = f.mul(&acc, &f.sub(w, &f.from_base(f.zq().from_i64(c))));
            }
            acc
        };
        let f0 = (1..=d0 as u64 + 1)
            .map(|t| {
                let mut acc = f.zero();
                for x in 0..h {
                    let (wt, et, it) = (at(w, x, t), at(e, x, t), at(ind, x, t));
                    let bracket = f.sub(&f.sub(&f.one(), &et), &it);
                    let term = f.add(&f.mul(&et, &literal(&wt)), &f.mul(&bracket, &wt));
                    acc = f.add(&acc, &f.mul(&at(eq, x, t), &term));
                }
                f.mul(eq_party, &acc)
            })
            .collect();
        let f1 = (1..=d1 as u64 + 1)
            .map(|t| {
                (0..h).fold(f.zero(), |acc, x| {
                    f.add(&acc, &f.mul(&at(w, x, t), &at(ms, x, t)))
                })
            })
            .collect();
        (f0, f1)
    }

    /// The kernel (factored tables, incremental node values, extrapolation from 2b + 2 resp. 3
    /// nodes) in both arithmetics against the direct evaluation at all nodes, on random tables
    /// that satisfy no relation (so that the degree bound is exercised on arbitrary inputs), for
    /// every party of the toy parameters with n = 1, 2, 3 (contexts from `Context::setup`, seeds
    /// [n; 32]; tables from splitmix64 with seed 1000 n + p).
    #[test]
    fn kernel_matches_the_direct_evaluation_on_random_tables() {
        for n in 1..=3u64 {
            let (params, _) = derive(&toy_choices(n)).unwrap();
            let ctx = Context::<4>::setup(params, &[n as u8; 32]).unwrap();
            let l = &ctx.layout;
            let f = ctx.field();
            let q = ctx.q();
            let size = 1usize << l.ell_pp;
            let two_d = 2 * l.d;
            for p in 1..=l.n {
                let mut rng = SplitMix(1000 * n + p as u64);
                // A table with small entries in places, so that the range product vanishes
                // at some points and not at others.
                let w: Vec<u64> = (0..size)
                    .map(|x| {
                        if x % 3 == 0 {
                            rng.next() % 5
                        } else {
                            rng.next() % q
                        }
                    })
                    .collect();
                let mut blinder = |degree: usize| Blinder {
                    s0: rng.ext(q),
                    s: (0..l.ell_pp)
                        .map(|_| (0..degree).map(|_| rng.ext(q)).collect())
                        .collect(),
                };
                let (g0, g1) = (blinder(l.d0), blinder(l.d1));
                let coins = SumcheckCoins {
                    alpha: rng.ext(q),
                    rho: rng.ext(q),
                    tau0: (0..l.ell_pp + l.log_n).map(|_| rng.ext(q)).collect(),
                    tau1: (0..l.log_nu).map(|_| rng.ext(q)).collect(),
                };
                let mstar: Vec<Ext<4>> = (0..size).map(|_| rng.ext(q)).collect();
                let point: Vec<Ext<4>> = (0..l.ell_pp).map(|_| rng.ext(q)).collect();
                let mut fast =
                    PartyProver::new(&ctx, p, w.clone(), None, g0.clone(), g1.clone()).unwrap();
                assert!(fast.mont.is_some());
                let mut slow = fast.clone();
                slow.mont = None;
                fast.start(&coins, mstar.clone()).unwrap();
                slow.start(&coins, mstar.clone()).unwrap();
                // The direct computation's full tables over (U, L).
                let expand = |rows: &[u64]| -> Vec<Ext<4>> {
                    (0..size).map(|x| f.from_base(rows[x / two_d])).collect()
                };
                let mut eq = eq_table(f, &coins.tau0[..l.ell_pp]).unwrap();
                let mut e = expand(&row_indicator(l, &l.s_ranges(p)).unwrap());
                let mut ind = expand(&row_indicator(l, &[ind_rows(l)]).unwrap());
                let mut wt: Vec<Ext<4>> = w.iter().map(|x| f.from_base(*x)).collect();
                let mut ms = mstar;
                let eq_party =
                    eq_point(f, &coins.tau0[l.ell_pp..], &msb_bits(p - 1, l.log_n)).unwrap();
                for r in 1..=l.ell_pp {
                    let (f0, f1) =
                        direct_round(f, ctx.b(), &eq_party, &eq, &e, &ind, &wt, &ms, l.d0, l.d1);
                    let values = fast.round_values().unwrap();
                    assert_eq!(values.f0, f0, "n = {n}, p = {p}, round {r}: F_0");
                    assert_eq!(values.f1, f1, "n = {n}, p = {p}, round {r}: F_1");
                    assert_eq!(slow.round_values().unwrap(), values, "canonical arithmetic");
                    for (t, (h0, f0)) in values.h0.iter().zip(&f0).enumerate() {
                        let x = f.from_base(t as u64 + 1);
                        let g = round_poly(f, &g0, r, &point[..r - 1], &x).unwrap();
                        assert_eq!(*h0, f.add(f0, &f.mul(&coins.rho, &g)));
                    }
                    for (t, (h1, f1)) in values.h1.iter().zip(&f1).enumerate() {
                        let x = f.from_base(t as u64 + 1);
                        let g = round_poly(f, &g1, r, &point[..r - 1], &x).unwrap();
                        assert_eq!(*h1, f.add(f1, &f.mul(&coins.rho, &g)));
                    }
                    let a = point[r - 1];
                    fast.bind(&a).unwrap();
                    slow.bind(&a).unwrap();
                    for t in [&mut eq, &mut e, &mut ind, &mut wt, &mut ms] {
                        bind_top(f, t, &a).unwrap();
                    }
                }
                let fin = fast.final_values().unwrap();
                assert_eq!(slow.final_values().unwrap(), fin);
                assert_eq!(fin.0, wt[0]);
                assert_eq!(fin.1, evaluate(f, &g0, &point).unwrap());
                assert_eq!(fin.2, evaluate(f, &g1, &point).unwrap());
                assert_eq!(fast.point(), &point[..]);
            }
        }
    }

    /// Round 2 from the base-field table and the first coin (`PlaneValues`, forced by a plane share
    /// of 0) against the direct evaluation at all nodes and against the prover that folds after
    /// round 1, in both arithmetics: toy parameters with n = 1, 2, 3, tables of digits with
    /// residues outside the digit range in places, and a field pad for party 1 (its quadruple
    /// takes the general path).
    #[test]
    fn round_two_from_the_base_table_matches_the_direct_evaluation() {
        // Round 3 from the base table too (`third_from_table`) where round 3 is a U round.
        let mut thirds = 0;
        for n in 1..=3u64 {
            let (params, _) = derive(&toy_choices(n)).unwrap();
            let ctx = Context::<4>::setup(params, &[n as u8 + 10; 32]).unwrap();
            let l = &ctx.layout;
            assert!(l.ell_u >= 2 && l.ell_pp >= 2, "round 2 is a U round");
            let f = ctx.field();
            let q = ctx.q();
            let b = ctx.b() as i64;
            let size = 1usize << l.ell_pp;
            let two_d = 2 * l.d;
            for p in 1..=l.n {
                let mut rng = SplitMix(2000 * n + p as u64);
                let span = (2 * b - 1) as u64;
                let w: Vec<u64> = (0..size)
                    .map(|x| {
                        if x % 97 == 5 {
                            rng.next() % q
                        } else {
                            f.zq().from_i64((rng.next() % span) as i64 - (b - 1))
                        }
                    })
                    .collect();
                let pad = (p == 1).then(|| rng.ext(q));
                let mut blinder = |degree: usize| Blinder {
                    s0: rng.ext(q),
                    s: (0..l.ell_pp)
                        .map(|_| (0..degree).map(|_| rng.ext(q)).collect())
                        .collect(),
                };
                let (g0, g1) = (blinder(l.d0), blinder(l.d1));
                let coins = SumcheckCoins {
                    alpha: rng.ext(q),
                    rho: rng.ext(q),
                    tau0: (0..l.ell_pp + l.log_n).map(|_| rng.ext(q)).collect(),
                    tau1: (0..l.log_nu).map(|_| rng.ext(q)).collect(),
                };
                let mstar: Vec<Ext<4>> = (0..size).map(|_| rng.ext(q)).collect();
                let point: Vec<Ext<4>> = (0..l.ell_pp).map(|_| rng.ext(q)).collect();
                let mut folding =
                    PartyProver::new(&ctx, p, w.clone(), pad, g0.clone(), g1.clone()).unwrap();
                let mut fast = folding.clone();
                fast.plane_share = 0;
                folding.plane_share = usize::MAX;
                assert!(!folding.second_from_table());
                assert!(fast.second_from_table() && fast.mont.is_some());
                let third = fast.third_from_table();
                assert_eq!(third, l.ell_pp >= 3 && l.ell_u >= 3);
                thirds += usize::from(third);
                let mut slow = fast.clone();
                slow.mont = None;
                for prover in [&mut fast, &mut slow, &mut folding] {
                    prover.start(&coins, mstar.clone()).unwrap();
                }
                let expand = |rows: &[u64]| -> Vec<Ext<4>> {
                    (0..size).map(|x| f.from_base(rows[x / two_d])).collect()
                };
                let mut eq = eq_table(f, &coins.tau0[..l.ell_pp]).unwrap();
                let mut e = expand(&row_indicator(l, &l.s_ranges(p)).unwrap());
                let mut ind = expand(&row_indicator(l, &[ind_rows(l)]).unwrap());
                let mut wt: Vec<Ext<4>> = w.iter().map(|x| f.from_base(*x)).collect();
                if let Some(value) = pad {
                    wt[pad_index(l)] = f.reduce(&value);
                }
                let mut ms = mstar;
                let eq_party =
                    eq_point(f, &coins.tau0[l.ell_pp..], &msb_bits(p - 1, l.log_n)).unwrap();
                for r in 1..=l.ell_pp {
                    let (f0, f1) =
                        direct_round(f, ctx.b(), &eq_party, &eq, &e, &ind, &wt, &ms, l.d0, l.d1);
                    let values = fast.round_values().unwrap();
                    assert_eq!(values.f0, f0, "n = {n}, p = {p}, round {r}: F_0");
                    assert_eq!(values.f1, f1, "n = {n}, p = {p}, round {r}: F_1");
                    assert_eq!(slow.round_values().unwrap(), values, "canonical arithmetic");
                    assert_eq!(
                        folding.round_values().unwrap(),
                        values,
                        "the folding prover"
                    );
                    let a = point[r - 1];
                    for prover in [&mut fast, &mut slow, &mut folding] {
                        prover.bind(&a).unwrap();
                    }
                    if r == 1 {
                        assert!(fast.run.as_ref().unwrap().w.is_empty(), "not folded");
                    }
                    for t in [&mut eq, &mut e, &mut ind, &mut wt, &mut ms] {
                        bind_top(f, t, &a).unwrap();
                    }
                    // With round 3 from the base table, the table bound to two coins is never
                    // formed and the third bind forms it bound to three.
                    let formed = if third { 3 } else { 2 };
                    if r == 2 && third {
                        assert!(fast.run.as_ref().unwrap().w.is_empty(), "not formed");
                    }
                    if r == formed {
                        assert_eq!(fast.run.as_ref().unwrap().w, wt, "bound to {r} coins");
                    }
                }
                let fin = fast.final_values().unwrap();
                assert_eq!(slow.final_values().unwrap(), fin);
                assert_eq!(folding.final_values().unwrap(), fin);
                assert_eq!(fin.0, wt[0]);
            }
        }
        assert!(
            thirds > 0,
            "round 3 read the base table at some toy parameters"
        );
    }

    /// The prover started with $M^\ast$ by its parts ([`PartyProver::start_parts`]) against the
    /// prover started with its table: the same round values in every round and the same final
    /// values, in both arithmetics and with round 2 from the base table or not, for toy parameters
    /// with n = 1, 2, 3 and $M^\ast$ from a random $\mathbf M$ (built by `MRows`), $E$ and coins.
    #[test]
    fn the_prover_with_the_parts_of_m_star_is_the_prover_with_its_table() {
        use crate::tables::EqSplit;
        use hachi_co_core::{MRows, Poly};
        use std::borrow::Cow;
        for n in 1..=3u64 {
            let (params, _) = derive(&toy_choices(n)).unwrap();
            let ctx = Context::<4>::setup(params, &[n as u8 + 20; 32]).unwrap();
            let l = &ctx.layout;
            let f = ctx.field();
            let (q, d) = (ctx.q(), l.d);
            let size = 1usize << l.ell_pp;
            let mut rng = SplitMix(3000 * n);
            let b_ring: Vec<Poly> = (0..l.n_blocks)
                .map(|_| ctx.sub.embed(&rng.ext(q)))
                .collect();
            let a_ring: Vec<Poly> = (0..l.block_len)
                .map(|_| ctx.sub.embed(&rng.ext(q)))
                .collect();
            let c: Vec<Vec<i64>> = (0..l.n_blocks)
                .map(|_| (0..d).map(|_| (rng.next() % 3) as i64 - 1).collect())
                .collect();
            let e_matrix: Vec<Vec<u64>> = (0..d)
                .map(|_| (0..d).map(|_| rng.next() % q).collect())
                .collect();
            let rows = MRows::new(&ctx, &b_ring, &a_ring, &c, 0).unwrap();
            for p in 1..=l.n {
                let b = ctx.b() as i64;
                let span = (2 * b - 1) as u64;
                let w: Vec<u64> = (0..size)
                    .map(|x| {
                        if x % 7 == 3 {
                            rng.next() % q
                        } else {
                            f.zq().from_i64((rng.next() % span) as i64 - (b - 1))
                        }
                    })
                    .collect();
                let mut blinder = |degree: usize| Blinder {
                    s0: rng.ext(q),
                    s: (0..l.ell_pp)
                        .map(|_| (0..degree).map(|_| rng.ext(q)).collect())
                        .collect(),
                };
                let (g0, g1) = (blinder(l.d0), blinder(l.d1));
                let coins = SumcheckCoins {
                    alpha: rng.ext(q),
                    rho: rng.ext(q),
                    tau0: (0..l.ell_pp + l.log_n).map(|_| rng.ext(q)).collect(),
                    tau1: (0..l.log_nu).map(|_| rng.ext(q)).collect(),
                };
                let built = |i: usize| Ok(Cow::Owned(rows.row(i)?));
                let eq1 = EqSplit::new(f, &coins.tau1).unwrap();
                let parts =
                    MStarParts::from_rows(&ctx, built, &e_matrix, eq1, &coins.alpha).unwrap();
                let table = parts.table(f);
                let point: Vec<Ext<4>> = (0..l.ell_pp).map(|_| rng.ext(q)).collect();
                let base =
                    PartyProver::new(&ctx, p, w.clone(), None, g0.clone(), g1.clone()).unwrap();
                for (mont, plane_share) in [(true, PLANE_SHARE), (false, PLANE_SHARE), (true, 0)] {
                    let mut with_table = base.clone();
                    if !mont {
                        with_table.mont = None;
                    }
                    with_table.plane_share = plane_share;
                    let mut with_parts = with_table.clone();
                    with_table.start(&coins, table.clone()).unwrap();
                    with_parts.start_parts(&coins, parts.clone()).unwrap();
                    for (r, a) in point.iter().enumerate() {
                        assert_eq!(
                            with_parts.round_values().unwrap(),
                            with_table.round_values().unwrap(),
                            "n = {n}, p = {p}, mont {mont}, plane share {plane_share}, round {}",
                            r + 1
                        );
                        with_table.bind(a).unwrap();
                        with_parts.bind(a).unwrap();
                    }
                    assert_eq!(
                        with_parts.final_values().unwrap(),
                        with_table.final_values().unwrap(),
                        "n = {n}, p = {p}"
                    );
                }
                let mut twice = base.clone();
                twice.start_parts(&coins, parts.clone()).unwrap();
                assert!(
                    twice.start_parts(&coins, parts.clone()).is_err(),
                    "started twice"
                );
            }
        }
    }

    /// The pairs of factors against the product one factor at a time, for every base up to 17
    /// (an even and an odd number of factors, none, one), in both arithmetics, at random points and
    /// at the roots.
    #[test]
    fn the_pairs_of_factors_give_the_range_product() {
        let q = 34801213478789u64; // secure-l16
        let f = Fqk::<4>::new(q).unwrap();
        let mont = Montgomery::<4>::new(q).unwrap();
        let mut rng = SplitMix(78);
        for b in 0..=17u64 {
            let range = RangeProduct::new(f.zq(), b);
            let mut points: Vec<Ext<4>> = (0..20).map(|_| rng.ext(q)).collect();
            points.extend((0..b).map(|c| f.from_base(c)));
            points.extend((0..b).map(|c| f.from_base(f.zq().neg(c))));
            for w in &points {
                let canonical = RangePairs::new(&f, f.zq(), range.squares());
                let u = f.mul(w, w);
                let squares: Vec<u64> = range.squares().to_vec();
                assert_eq!(
                    canonical.eval(&f, &u, w),
                    range_eval(&f, w, &squares),
                    "b = {b}"
                );
                let pairs = RangePairs::new(&mont, f.zq(), range.squares());
                let (wm, entered) = (mont.enter(w), squares.iter().map(|c| mont.enter_base(*c)));
                let entered: Vec<u64> = entered.collect();
                let um = mont.mul(&wm, &wm);
                assert_eq!(
                    mont.leave(&pairs.eval(&mont, &um, &wm)),
                    mont.leave(&range_eval(&mont, &wm, &entered)),
                    "b = {b}, Montgomery"
                );
                if b > 0 {
                    assert_eq!(canonical.eval(&f, &u, w), range.eval(&f, w), "b = {b}");
                }
            }
        }
    }

    /// The time of the rounds after the second at secure-l16-n2-a8 (b = 16, 2^24 points), on a
    /// table of random digits, random M* and coins, in the arithmetic the prover picks; rounds 1
    /// and 2 are run first and not timed. Prints the time and a digest of the round values.
    #[test]
    #[ignore = "timing: cargo test --release -p hachi-co-sumcheck --lib timings_later_rounds -- --ignored --nocapture"]
    fn timings_later_rounds() {
        use std::time::Instant;
        let preset = hachi_co_params::presets::preset("secure-l16-n2-a8").unwrap();
        let (params, _) = preset.derive().unwrap();
        let ctx = Context::<4>::setup(params, &[1; 32]).unwrap();
        let l = &ctx.layout;
        let (q, b) = (ctx.q(), ctx.b() as i64);
        let size = 1usize << l.ell_pp;
        let mut rng = SplitMix(77);
        let span = (2 * b - 1) as u64;
        let w: Vec<u64> = (0..size)
            .map(|_| {
                ctx.field()
                    .zq()
                    .from_i64((rng.next() % span) as i64 - (b - 1))
            })
            .collect();
        let mut blinder = |degree: usize| Blinder {
            s0: rng.ext(q),
            s: (0..l.ell_pp)
                .map(|_| (0..degree).map(|_| rng.ext(q)).collect())
                .collect(),
        };
        let (g0, g1) = (blinder(l.d0), blinder(l.d1));
        let coins = SumcheckCoins {
            alpha: rng.ext(q),
            rho: rng.ext(q),
            tau0: (0..l.ell_pp + l.log_n).map(|_| rng.ext(q)).collect(),
            tau1: (0..l.log_nu).map(|_| rng.ext(q)).collect(),
        };
        let mstar: Vec<Ext<4>> = (0..size).map(|_| rng.ext(q)).collect();
        let mut prover = PartyProver::new(&ctx, 1, w, None, g0, g1).unwrap();
        prover.start(&coins, mstar).unwrap();
        let mut digest = 0u64;
        let mut timed = 0.0;
        for r in 1..=6 {
            let start = Instant::now();
            let values = prover.round_values().unwrap();
            if r >= 3 {
                timed += start.elapsed().as_secs_f64();
            }
            for v in values.f0.iter().chain(&values.f1) {
                digest = digest.rotate_left(7) ^ v.0[0] ^ v.0[3].rotate_left(29);
            }
            prover.bind(&rng.ext(q)).unwrap();
        }
        println!(
            "secure-l16-n2-a8, 2^{} points, {} threads: rounds 3 to 6 {timed:.2} s, digest {digest:016x}",
            l.ell_pp,
            rayon::current_num_threads()
        );
    }

    /// Calls out of order are errors, not panics; shapes and parameters are checked.
    #[test]
    fn misuse_is_an_error() {
        let (params, _) = derive(&toy_choices(2)).unwrap();
        let ctx = Context::<4>::setup(params.clone(), &[9; 32]).unwrap();
        let l = &ctx.layout;
        let f = ctx.field();
        let size = 1usize << l.ell_pp;
        let blinder = |degree: usize| Blinder {
            s0: f.one(),
            s: vec![vec![f.one(); degree]; l.ell_pp],
        };
        let coins = SumcheckCoins {
            alpha: f.one(),
            rho: f.one(),
            tau0: vec![f.one(); l.ell_pp + l.log_n],
            tau1: vec![f.one(); l.log_nu],
        };
        let new = |p, w: Vec<u64>, g0, g1| PartyProver::new(&ctx, p, w, None, g0, g1);
        assert!(matches!(
            new(0, vec![0; size], blinder(l.d0), blinder(l.d1)),
            Err(Error::InvalidParty { .. })
        ));
        assert!(matches!(
            new(3, vec![0; size], blinder(l.d0), blinder(l.d1)),
            Err(Error::InvalidParty { .. })
        ));
        assert!(matches!(
            new(1, vec![0; size - 1], blinder(l.d0), blinder(l.d1)),
            Err(Error::Shape { .. })
        ));
        assert!(matches!(
            new(1, vec![0; size], blinder(l.d1), blinder(l.d1)),
            Err(Error::Core(_))
        ));
        let mut prover = new(1, vec![0; size], blinder(l.d0), blinder(l.d1)).unwrap();
        assert!(matches!(prover.round_values(), Err(Error::State(_))));
        assert!(matches!(prover.bind(&f.one()), Err(Error::State(_))));
        assert!(matches!(prover.final_values(), Err(Error::State(_))));
        assert_eq!(prover.round(), None);
        let mut short = coins.clone();
        short.tau0.pop();
        assert!(matches!(
            prover.start(&short, vec![f.zero(); size]),
            Err(Error::Shape { .. })
        ));
        assert!(matches!(
            prover.start(&coins, vec![f.zero(); size / 2]),
            Err(Error::Shape { .. })
        ));
        prover.start(&coins, vec![f.zero(); size]).unwrap();
        assert!(matches!(
            prover.start(&coins, vec![f.zero(); size]),
            Err(Error::State(_))
        ));
        assert!(matches!(prover.final_values(), Err(Error::State(_))));
        for r in 1..=l.ell_pp {
            assert_eq!(prover.round(), Some(r));
            prover.round_values().unwrap();
            prover.bind(&f.one()).unwrap();
        }
        assert!(matches!(prover.round_values(), Err(Error::State(_))));
        assert!(matches!(prover.bind(&f.one()), Err(Error::State(_))));
        prover.final_values().unwrap();
        // d0 below 2b + 1: the round polynomials of F_0 would not fit in the messages.
        let mut small = params;
        small.d0 = 2 * small.choices.b;
        let ctx = Context::<4>::new(small, (*ctx.pp()).clone()).unwrap();
        let l = &ctx.layout;
        let r = PartyProver::new(&ctx, 1, vec![0; size], None, blinder(l.d0), blinder(l.d1));
        assert!(matches!(r, Err(Error::Parameter(_))));
    }
}
