//! Party $p$'s side of the collaborative sumcheck (`5_Building_Blocks.tex`:209-320; oracle
//! `sumcheck.prove`), as a state machine that the driver [`prove`](crate::prove) runs against a
//! [`Comm`](hachi_co_mpc::Comm) and that tests can run for several emulated parties.

use crate::arith::{Arith, Montgomery};
use crate::blinder::{BlinderRounds, evaluate, hypercube_sum};
use crate::eq::{eq_point, eq_table, msb_bits};
use crate::error::{Error, Result, check_len};
use crate::interp::Extrapolation;
use crate::tables::{RangeProduct, ind_rows, row_indicator, table_size};
use crate::transcript::RoundValues;
use hachi_co_core::{Blinder, Context, SumcheckCoins};
use hachi_co_math::{Ext, Fqk};
use hachi_co_mpc::PartyId;

/// The tables that change with the coins (built by [`PartyProver::start`]).
#[derive(Clone, Debug)]
struct Running<const K: usize> {
    /// $\rho$.
    rho: Ext<K>,
    /// $\widetilde{eq}(\tau\_0\[\ell''..\], \mathsf{bit}(p))$, the factor of $F\_{0,p}$ for the bits of
    /// the party (`ref/README.md`, "eq").
    eq_party: Ext<K>,
    /// $\langle \widetilde w \rangle\_p$ after the first bind (empty in round 1, which reads the
    /// base-field table).
    w: Vec<Ext<K>>,
    /// $M^\ast$.
    mstar: Vec<Ext<K>>,
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
}

/// One point's contribution to the sums of a round, at the nodes computed directly.
struct Accumulators<const K: usize> {
    /// $\sum \widetilde{eq}\_L \cdot \prod\_c (W - c)$ per node.
    rp: Vec<Ext<K>>,
    /// $\sum \widetilde{eq}\_L \cdot W$ per node.
    w: Vec<Ext<K>>,
    /// $\sum W \cdot M^\ast$ per node.
    m: Vec<Ext<K>>,
}

impl<const K: usize> Accumulators<K> {
    fn new(field: &Fqk<K>, n0: usize, n1: usize) -> Self {
        Self {
            rp: vec![field.zero(); n0],
            w: vec![field.zero(); n0],
            m: vec![field.zero(); n1],
        }
    }

    fn clear(&mut self, field: &Fqk<K>) {
        for v in self.rp.iter_mut().chain(&mut self.w).chain(&mut self.m) {
            *v = field.zero();
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
/// **Round values** (`5_Building_Blocks.tex`:267-275, 283-300; fixture conventions `F0` and
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
/// stay canonical and both give the same values. The tests check every value against the
/// oracle's direct evaluation at all nodes, in both arithmetics.
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
    range: RangeProduct,
    /// The Montgomery form of the inner loops, when $q$ admits it (otherwise the canonical
    /// field is used; both give the same values).
    mont: Option<Montgomery<K>>,
    w_base: Vec<u64>,
    g0: Blinder<K>,
    g1: Blinder<K>,
    sums: (Ext<K>, Ext<K>),
    e_rows: Vec<u64>,
    ind_rows: Vec<u64>,
    run: Option<Running<K>>,
}

impl<const K: usize> PartyProver<K> {
    /// The prover of party `party` with the hypercube table `w_table` of
    /// $\langle \widetilde w \rangle\_p$ ([`w_table`](crate::w_table)) and the blinders `g0`, `g1`
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
        Ok(Self {
            range: RangeProduct::new(zq, b),
            mont: Montgomery::new(zq.modulus()),
            e_rows: row_indicator(l, &l.s_ranges(party))?,
            ind_rows: row_indicator(l, &[ind_rows(l)])?,
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
            w_base: w_table,
            g0,
            g1,
            sums,
            run: None,
        })
    }

    /// The party $p$.
    pub fn party(&self) -> PartyId {
        self.party
    }

    /// $(G\_{0,p}, G\_{1,p})$, the hypercube sums of the blinders (`5_Building_Blocks.tex`:218-219).
    pub fn hypercube_sums(&self) -> (Ext<K>, Ext<K>) {
        self.sums
    }

    /// The hypercube table of $\langle \widetilde w \rangle\_p$ (base field).
    pub fn w_table(&self) -> &[u64] {
        &self.w_base
    }

    /// The blinders $g\_{0,p}$ and $g\_{1,p}$.
    pub fn blinders(&self) -> (&Blinder<K>, &Blinder<K>) {
        (&self.g0, &self.g1)
    }

    /// The coins $a\_1, \dots$ bound so far (empty before [`PartyProver::start`]).
    pub fn point(&self) -> &[Ext<K>] {
        self.run.as_ref().map_or(&[], |run| &run.point)
    }

    /// The table, the blinders and the point, consuming the prover.
    pub fn into_parts(self) -> (Vec<u64>, Blinder<K>, Blinder<K>, Vec<Ext<K>>) {
        let point = self.run.map(|run| run.point).unwrap_or_default();
        (self.w_base, self.g0, self.g1, point)
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
    /// $\mathcal F\_{\mathsf{AH}}(\mathbb I)$ (`5_Building_Blocks.tex`:294) and the unblinded
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
    fn dispatch<A: Arith<K>>(
        &self,
        arith: &A,
        run: &Running<K>,
        r: usize,
    ) -> (Vec<Ext<K>>, Vec<Ext<K>>) {
        if r == 1 {
            let (f, wb) = (&self.field, &self.w_base);
            self.kernel(arith, run, r, wb.len(), |x| f.from_base(wb[x]))
        } else {
            let w = &run.w;
            self.kernel(arith, run, r, w.len(), |x| w[x])
        }
    }

    /// The sums of round $r$ at the nodes $1, \dots, \mathrm{direct}\_0$ (for $F\_{0,p}$) and
    /// $1, \dots, \mathrm{direct}\_1$ (for $F\_{1,p}$), for the current table of
    /// $\langle \widetilde w \rangle\_p$ of `len` entries read through `w_at`. The tables are
    /// canonical; the values of a pair of points and the sums are formed in the working
    /// representation of `arith` and the results returned canonical.
    fn kernel<A: Arith<K>, W: Fn(usize) -> Ext<K>>(
        &self,
        arith: &A,
        run: &Running<K>,
        r: usize,
        len: usize,
        w_at: W,
    ) -> (Vec<Ext<K>>, Vec<Ext<K>>) {
        let f = &self.field;
        let zero = f.zero(); // phi(0) = 0 in every representation
        let one = arith.enter(&f.one());
        let squares: Vec<u64> = self
            .range
            .squares()
            .iter()
            .map(|c| arith.enter_base(*c))
            .collect();
        let h = len / 2;
        let (n0, n1) = (self.direct0, self.direct1);
        let mut acc = Accumulators::new(f, n0, n1);
        let mut f0 = vec![zero; n0];
        let mut f1 = vec![zero; n1];
        if r <= self.ell_u {
            // U round: the lower half is rows u < h / 2d of 2d points; e_p, ind and eq_U are
            // row values, eq_L does not change.
            let rows = h / self.two_d;
            let eq_l: Vec<Ext<K>> = run.eq_l.iter().map(|e| arith.enter(e)).collect();
            for u in 0..rows {
                acc.clear(f);
                let base = u * self.two_d;
                for (l, e) in eq_l.iter().enumerate() {
                    self.accumulate_pair(
                        arith,
                        &squares,
                        &mut acc,
                        &w_at,
                        &run.mstar,
                        base + l,
                        h,
                        e,
                        &zero,
                    );
                }
                // The row factors at t = 1 and their increments.
                let affine = |t: &[Ext<K>]| {
                    (
                        arith.enter(&t[u + rows]),
                        arith.enter(&f.sub(&t[u + rows], &t[u])),
                    )
                };
                let (mut eu, deu) = affine(&run.eq_u);
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
            }
        } else {
            // L round: U is bound, so e_p, ind and eq_U are scalars; eq_L is folded with W.
            for x in 0..h {
                let (lo, hi) = (&run.eq_l[x], &run.eq_l[x + h]);
                let (e_hi, de) = (arith.enter(hi), arith.enter(&f.sub(hi, lo)));
                self.accumulate_pair(
                    arith, &squares, &mut acc, &w_at, &run.mstar, x, h, &e_hi, &de,
                );
            }
            let (eu, ee, ii) = (
                arith.enter(&run.eq_u[0]),
                arith.enter(&run.e[0]),
                arith.enter(&run.ind[0]),
            );
            let bracket = arith.sub(&arith.sub(&one, &ee), &ii);
            for ((out, rp), w) in f0.iter_mut().zip(&acc.rp).zip(&acc.w) {
                let v = arith.add(&arith.mul(&ee, rp), &arith.mul(&bracket, w));
                *out = arith.mul(&eu, &v);
            }
            f1.clone_from(&acc.m);
        }
        // phi(v) times the canonical eq factor of the party is canonical.
        let f0 = f0.iter().map(|v| arith.mul(v, &run.eq_party)).collect();
        let f1 = f1.iter().map(|v| arith.leave(v)).collect();
        (f0, f1)
    }

    /// Adds the contribution of the point pair $(x, x + h)$ at the nodes $1, 2, \dots$: the values
    /// at node $t + 1$ are those at node $t$ plus the difference of the pair. The factor
    /// $\widetilde{eq}\_L$ is given in the working representation, at node 1 (`e_hi`) with its
    /// increment `de`.
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn accumulate_pair<A: Arith<K>, W: Fn(usize) -> Ext<K>>(
        &self,
        arith: &A,
        squares: &[u64],
        acc: &mut Accumulators<K>,
        w_at: &W,
        mstar: &[Ext<K>],
        x: usize,
        h: usize,
        e_hi: &Ext<K>,
        de: &Ext<K>,
    ) {
        let f = &self.field;
        let (w_lo, w_hi) = (w_at(x), w_at(x + h));
        let (m_lo, m_hi) = (&mstar[x], &mstar[x + h]);
        let (dw, dm) = (
            arith.enter(&f.sub(&w_hi, &w_lo)),
            arith.enter(&f.sub(m_hi, m_lo)),
        );
        let (mut w, mut m, mut e) = (arith.enter(&w_hi), arith.enter(m_hi), *e_hi);
        let (n0, n1) = (self.direct0, self.direct1);
        for t in 0..n0.max(n1) {
            if t < n0 {
                let rp = range_eval(arith, &w, squares);
                acc.rp[t] = arith.add(&acc.rp[t], &arith.mul(&e, &rp));
                acc.w[t] = arith.add(&acc.w[t], &arith.mul(&e, &w));
            }
            if t < n1 {
                acc.m[t] = arith.add(&acc.m[t], &arith.mul(&w, &m));
            }
            w = arith.add(&w, &dw);
            m = arith.add(&m, &dm);
            e = arith.add(&e, de);
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
        let Some(run) = self.run.as_mut() else {
            return Err(Error::State("bind before start".into()));
        };
        match &self.mont {
            Some(mont) => fold_tables(mont, &self.field, &self.w_base, run, r, self.ell_u, &a),
            None => fold_tables(
                &self.field,
                &self.field,
                &self.w_base,
                run,
                r,
                self.ell_u,
                &a,
            ),
        }
        let f = &self.field;
        run.g0.bind(f, &self.g0, r, &a);
        run.g1.bind(f, &self.g1, r, &a);
        run.point.push(a);
        Ok(())
    }

    /// The openings of `5_Building_Blocks.tex`:316-318 after the last round:
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

/// $\prod\_{c=-b+1}^{b-1} (w - c) = w \prod\_{c=1}^{b-1} (w^2 - c^2)$ in the working representation,
/// with the constants $c^2$ in that representation ([`RangeProduct`] for the canonical one).
#[inline(always)]
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

/// $T'\[x\] = T\[x\] + a (T\[x + h\] - T\[x\])$ on a canonical table, with the coin in the working
/// representation (so that the product with a canonical difference is canonical).
fn fold<const K: usize, A: Arith<K>>(
    arith: &A,
    f: &Fqk<K>,
    table: &mut Vec<Ext<K>>,
    coin: &Ext<K>,
) {
    let h = table.len() / 2;
    let (low, high) = table.split_at_mut(h);
    for (lo, hi) in low.iter_mut().zip(high.iter()) {
        *lo = f.add(lo, &arith.mul(coin, &f.sub(hi, lo)));
    }
    table.truncate(h);
}

/// Binds round $r$'s variable to $a$ in every table of `run` (the witness table from the
/// base-field `w_base` in round 1).
fn fold_tables<const K: usize, A: Arith<K>>(
    arith: &A,
    f: &Fqk<K>,
    w_base: &[u64],
    run: &mut Running<K>,
    r: usize,
    ell_u: usize,
    a: &Ext<K>,
) {
    let coin = arith.enter(a);
    if r == 1 {
        let zq = f.zq();
        let (low, high) = w_base.split_at(w_base.len() / 2);
        run.w = low
            .iter()
            .zip(high)
            .map(|(lo, hi)| f.add(&f.from_base(*lo), &arith.scale_out(&coin, zq.sub(*hi, *lo))))
            .collect();
    } else {
        fold(arith, f, &mut run.w, &coin);
    }
    fold(arith, f, &mut run.mstar, &coin);
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
                    PartyProver::new(&ctx, p, w.clone(), g0.clone(), g1.clone()).unwrap();
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
        let new = |p, w: Vec<u64>, g0, g1| PartyProver::new(&ctx, p, w, g0, g1);
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
        let ctx = Context::<4>::new(small, ctx.pp.clone()).unwrap();
        let l = &ctx.layout;
        let r = PartyProver::new(&ctx, 1, vec![0; size], blinder(l.d0), blinder(l.d1));
        assert!(matches!(r, Err(Error::Parameter(_))));
    }
}
