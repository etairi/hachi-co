// Doc comments in this file use `\_` for underscores and `\[` `\]` for brackets inside the
// $...$ KaTeX formulas: rustdoc's Markdown would otherwise pair underscores into emphasis (and
// brackets into links) and split the formula; the escapes reach KaTeX as plain `_`, `[`, `]`.

//! The two sumcheck polynomials of the ring-switched relation and the prover side of the
//! sumcheck protocol (paper, Section 4.3, Fig. 5, 6 and 7), over $\mathbb{F}\_{q^4}$
//! ([`ExtField`]).
//!
//! The next witness $\widetilde{w}$ (built by
//! [`crate::hachi::prover_utils::zq_zq::form_next_witness`]) is a table of $2^{v}$ short digits,
//! indexed by $(x, y)$ with $x$ the entry index (a ring element of
//! $(\mathbf z^{\prime}, \mathbf{r})$) and $y \in \[0, d)$ the coefficient index: table index
//! $= xd + y$, so the $\log d$ bits of $y$ are the low bits. Its multilinear extension
//! $\widetilde{w}(x, y)$ is the paper's $\widetilde{w}(u, \ell)$ of Eq. (21). The two
//! polynomials are
//! $$F\_{0,\tau\_0}(x, y) = \mathrm{eq}(\tau\_0, (x, y)) \cdot P\_b(\widetilde{w}(x, y)), \qquad F\_{\alpha,\tau\_1}(x, y) = \widetilde{w}(x, y) \cdot \widetilde{\alpha}(y) \cdot \bar{m}(x),$$
//! where $P\_b$ vanishes exactly on the digit set (see [`F0`]), $\widetilde{\alpha}(y) = \alpha^y$
//! and $\bar{m}(x) = \sum\_i \mathrm{eq}(\tau\_1, i) \widetilde{M}\_\alpha(i, x)$ (see
//! [`FAlpha`]). The prover shows $\sum\_{x,y} F\_{0,\tau\_0}(x,y) = 0$ (every entry is a digit)
//! and $\sum\_{x,y} F\_{\alpha,\tau\_1}(x,y) = \sum\_i \mathrm{eq}(\tau\_1, i) y\_i(\alpha)$ (the
//! linear relation $\mathbf{M}\mathbf z^{\prime} = \mathbf{y} + (X^d+1)\mathbf{r}$ evaluated at
//! $X = \alpha$ and batched over its rows with $\tau\_1$).
//!
//! Folding convention: both polynomials implement [`SumCheckPoly`] over the same $v$ variables
//! and fold least-significant bit first, i.e. round $k$ binds the $k$-th lowest bit of the table
//! index: the $\log d$ coefficient variables $y$ first, then the entry variables $x$. A table
//! $T$ of evaluations on $\lbrace 0,1 \rbrace^{v}$ is halved by [`fix_first_variable`]: the
//! entries at indices $2s$ and $2s+1$ become $(1-r) T\_{2s} + r T\_{2s+1}$ at index $s$.
//! Because the two polynomials are over the same variables in the same order,
//! [`sumcheck_proof`] runs both in lock-step with one shared challenge per round, and the final
//! reduced value $y^{\prime} = \widetilde{w}(r\_1, \dots, r\_v)$ is common to both (the
//! evaluation claim on the next witness that the next round would prove).
//!
//! Round messages are [`Univariate`]s stored as their values on $0, 1, \dots, \deg$; the
//! verifier ([`crate::hachi::verify`]) checks $g\_k(0) + g\_k(1)$ against the running claim and
//! evaluates $g\_k(r\_k)$ by Lagrange interpolation.

use ark_ff::{AdditiveGroup, Field};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

#[cfg(feature = "verbose")]
use crate::utils::verbose::{progress_bar};

use crate::arithmetic::utils::{Logarithm, eq, eq_bin, lift_int, mul_int_field, rand_field};
use crate::arithmetic::sumcheck::fix_first_variable;
use crate::arithmetic::sumcheck::Univariate;
use crate::arithmetic::sumcheck::SumCheckPoly;
use crate::arithmetic::fs::FS;
use crate::arithmetic::ExtField;

use crate::hachi::setup::Q;

/// The range-check polynomial $F\_{0,\tau\_0}$ (paper, Section 4.3, Eq. (23)) in the form the
/// code uses:
/// $$F\_0(x) = \mathrm{eq}(\tau\_0, x) \cdot P\_b(\widetilde{w}(x)), \qquad P\_b(w) = w \cdot (w + b/2) \cdot \prod\_{k=1}^{b/2 - 1} (w - k)(w + k).$$
/// $P\_b$ has the $b$ roots $0, -b/2, \pm 1, \dots, \pm(b/2-1)$, i.e. it vanishes exactly on
/// the balanced digit set $\lbrace -b/2, \dots, b/2 - 1 \rbrace$, the image of
/// $\mathbf G^{-1}$, and has degree $b$; so $F\_0$ has degree $b + 1$ in each variable
/// ([`SumCheckPoly::degree`] returns `base + 1`). The sum $\sum\_x F\_0(x)$ is the multilinear
/// extension of $x \mapsto P\_b(\widetilde{w}(x))$ evaluated at $\tau\_0$: it is identically
/// zero in $\tau\_0$ if and only if every entry of the witness table is a digit, and for a
/// uniform $\tau\_0$ a table with a non-digit entry sums to zero with probability at most
/// $v/q^4$ (Schwartz-Zippel).
///
/// This differs from the paper's Eq. (23), which uses the symmetric root set
/// $0, \pm 1, \dots, \pm(b-1)$ (degree $2b$ per variable) for the relation
/// $\lVert \mathbf{z} \rVert\_\infty \le b - 1$; the code checks the tighter set that the
/// decomposition actually produces. The paper's indicator $1\_{\le\mu}$ is not used either:
/// every entry of the table, including the decomposed quotients and the zero padding, must be a
/// digit.
///
/// The $\mathrm{eq}$ factor is maintained incrementally: `eq_scalar` holds
/// $\prod\_k \mathrm{eq}(\tau\_{0,k}, r\_k)$ over the variables already fixed, and `tau_0` is
/// truncated to the coordinates of the remaining variables.
pub struct F0 {
    /// The witness table as wrapping-signed digits, kept for the integer fast path of the first round.
    w: Vec<u64>,
    /// The current table of $\widetilde{w}$ on the remaining variables, over $\mathbb{F}\_{q^4}$; halved by every fold.
    w_eval_table: Vec<ExtField>,
    /// The decomposition base $b$ (`params.b`); the digit set is $\[-b/2, b/2 - 1\]$.
    base: u64,
    /// The coordinates of $\tau\_0$ for the variables not yet fixed; the first entry belongs to the next variable.
    tau_0: Vec<ExtField>,
    /// $\prod\_k \mathrm{eq}(\tau\_{0,k}, r\_k)$ over the variables already fixed.
    eq_scalar: ExtField,
    /// The modulus $q$, for the first-round integer arithmetic.
    q: u64,
    /// Length of the unpadded part of `w`; entries at or beyond it are zero and are skipped in the first round.
    len: usize
}

impl F0 {
    /// Build $F\_0$ from the next witness `z_r` (wrapping-signed digits, zero-padded to $2^v$
    /// entries), the base $b$, the verifier's point $\tau\_0 \in \mathbb{F}\_{q^4}^{v}$, the
    /// modulus $q$ and the length `len` of the unpadded part. Every entry is lifted to
    /// $\mathbb{F}\_{q^4}$ (a negative wrapping value maps to the corresponding negative field
    /// element) and a copy of the integer table is kept for the first round.
    pub fn init(z_r: &Vec<u64>, base: u64, tau_0: Vec<ExtField>, q: u64, len: usize) -> Self {
        // lift evaluations to field
        // let w: Vec<BaseField> = z_r.iter().map(|x| BaseField::from(*x)).collect();
        let w_eval_table: Vec<ExtField> = z_r.iter().map(|x| lift_int(*x)).collect();

        Self { w: z_r.clone(), w_eval_table, base, tau_0, eq_scalar: ExtField::ONE, q, len }
    }
}

impl SumCheckPoly<ExtField> for F0 {
    /// Per-variable degree $b + 1$: $P\_b$ has degree $b$ and $\mathrm{eq}$ is linear in each variable.
    fn degree(&self) -> usize {
        (self.base + 1) as usize
    }

    /// Number of variables still to be folded, $\log\_2$ of the current table length.
    fn num_vars(&self) -> usize {
        self.w_eval_table.len().log()
    }

    /// The round message $g(X) = \sum\_{s \in \lbrace 0,1 \rbrace^{v-1}} F\_0(X, s)$, returned
    /// as its values at $X = 0, 1, \dots, b + 1$.
    ///
    /// For each evaluation point $X = t$ and each suffix $s$, $\widetilde{w}(t, s)$ is the
    /// linear interpolation $(1 - t) T\_{2s} + t T\_{2s+1}$ of the two table entries whose low
    /// bit differs, $P\_b$ is applied to it (using $(w - k)(w + k) = w^2 - k^2$ to halve the
    /// multiplications), and the result is weighted by the product of `eq_scalar`,
    /// $\mathrm{eq}(\tau\_{0,0}, t)$ and $\prod\_{k \ge 1} \mathrm{eq}(\tau\_{0,k}, s\_{k-1})$;
    /// the suffix products are precomputed once per round.
    ///
    /// First-round fast path: before the first fold (`w_eval_table.len() == w.len()`) the
    /// table entries are the original small integers, so $\widetilde{w}(t, s)$ and $P\_b$ are
    /// evaluated in `i64` arithmetic modulo $q$ and only the final value is lifted to
    /// $\mathbb{F}\_{q^4}$. This is exact because the entries are digits
    /// ($\lvert w \rvert \le b/2$) and $t \le b + 1$, so no intermediate product overflows
    /// `i64` before its reduction. Suffixes with $2s \ge$ `len` are skipped: they index the
    /// zero padding, where $P\_b(0) = 0$. In later rounds the entries are field elements and
    /// the padding has been folded in, so the whole table is processed.
    ///
    /// Cost per round: $(b + 2) \cdot 2^{v-1}$ interpolations, each followed by about $b/2$
    /// multiplications, i.e. $O(2^v b^2)$ field operations, plus $2^{v-1}(v-1)$ multiplications
    /// for the suffix products.
    fn get_univariate(&self) -> Univariate<ExtField> {
        let num_vars = self.num_vars();
        assert_eq!(num_vars, self.tau_0.len());

        // pre-compute eq(tau, suffix) for each possible suffix
        let mut eq_suffix = vec![ExtField::ONE; 1 << (num_vars - 1)];

        for suffix in 0..1 << (num_vars - 1) {
            for i in 0..num_vars - 1 {
                eq_suffix[suffix] *= eq_bin(self.tau_0[i + 1], (suffix >> i) & 1);
            }
        }

        // evaluate the univariate polynomial at degree + 1 different points
        let deg = self.degree();
        let mut ys = Vec::<ExtField>::with_capacity(deg + 1);

        for x_i in 0..=deg {
            let x_i = x_i as u64;

            // build sum_{0,1}^n-1 F0(x_i, b_2, ..., b_n)
            let mut y_i = ExtField::ZERO;

            for suffix in 0..1 << (num_vars - 1) {
                let v_x =
                // round 1 - perform operations on w over integers for improved efficiency
                if self.w_eval_table.len() == self.w.len() {
                    // short circuit if in padded section
                    if (suffix << 1) >= self.len {
                        break;
                    }

                    // get w(0, suffix) and w(1, suffix)
                    let w_0 = self.w[suffix << 1];
                    let w_1 = self.w[(suffix << 1) | 1];

                    // evaluation of w at x is linear interpolation
                    let w_x = (w_0 as i64 + (w_1 as i64 - w_0 as i64) * x_i as i64) % self.q as i64;

                    // perform the multiplication v=w.(w+b/2).(w-1)(w+1). ... .(w-b/2-1)(w+b/2-1)
                    let mut v_x = (w_x * (w_x + self.base as i64 / 2)) % self.q as i64;
                    
                    // multiply with difference of two squares to improve efficiency
                    let w_x_squared = (w_x * w_x) % self.q as i64;

                    for r in 1..self.base as usize / 2 { 
                        v_x = (v_x * (w_x_squared - r as i64 * r as i64)) % self.q as i64;
                    }

                    lift_int(v_x as u64)
                }
                // subsequent rounds
                else {
                    // get w(0, suffix) and w(1, suffix)
                    let w_0 = self.w_eval_table[suffix << 1];
                    let w_1 = self.w_eval_table[(suffix << 1) | 1];

                    // evaluation of w at x is linear interpolation
                    let w_x = w_0 + mul_int_field(x_i, w_1 - w_0);

                    // perform the multiplication v=w.(w+b/2).(w-1)(w+1). ... .(w-b/2-1)(w+b/2-1)
                    let mut v_x = w_x * (w_x + lift_int(self.base / 2));
                    
                    // multiply with difference of two squares to improve efficiency
                    let w_x_squared = w_x * w_x;

                    for r in 1..self.base as usize / 2 { 
                        v_x *= w_x_squared - lift_int((r * r) as u64);
                    }

                    v_x
                };

                // calculate equality with tau_0
                let eq_x = self.eq_scalar * eq(self.tau_0[0], lift_int(x_i)) * eq_suffix[suffix];

                // multiply by equality
                let y = eq_x * v_x;

                y_i += y;
            }

            ys.push(y_i);
        }

        Univariate::init(ys)
    }

    /// Bind the next variable to `r`: halve `w_eval_table` with
    /// [`crate::arithmetic::sumcheck::fix_first_variable`], multiply `eq_scalar` by
    /// $\mathrm{eq}(\tau\_{0,0}, r)$ and drop $\tau\_{0,0}$.
    fn fix_first_variable(&mut self, r: ExtField) {
        let half = self.num_vars() - 1;

        // update the tables of evaluations on boolean inputs
        self.w_eval_table = fix_first_variable(&self.w_eval_table, r);

        // track eq(tau, r)
        self.eq_scalar *= eq(self.tau_0[0], r);

        // update tau_0
        let mut tmp = vec![ExtField::ZERO; half];
        tmp.copy_from_slice(&self.tau_0[1..=half]);
        self.tau_0 = tmp;
    }
}

/// The linear-relation polynomial $F\_{\alpha,\tau\_1}$ (paper, Section 4.3, Fig. 5 to 7):
/// $$F\_\alpha(x, y) = \widetilde{w}(x, y) \cdot \widetilde{\alpha}(y) \cdot \bar{m}(x), \qquad \bar{m}(x) = \sum\_{i} \mathrm{eq}(\tau\_1, i) \widetilde{M}\_\alpha(i, x),$$
/// where $\widetilde{w}$ is the multilinear extension of the next witness table (entry index
/// $x$, coefficient index $y$), $\widetilde{\alpha}(y) = \alpha^y$ and
/// $\widetilde{M}\_\alpha(i, x)$ is the entry of the matrix
/// $\[\mathbf{M} \mid -(X^d+1)\mathbf{G}\]$ evaluated at $X = \alpha$ (built by
/// [`crate::hachi::common::form_m_alpha`]). The row index $i$ is folded into the public factor
/// $\bar{m}$ up front with the verifier's $\tau\_1$, so the sumcheck runs over $(x, y)$ only.
/// Its sum over the hypercube is
/// $$\sum\_{x,y} F\_\alpha(x,y) = \sum\_i \mathrm{eq}(\tau\_1, i) \sum\_x \widetilde{M}\_\alpha(i,x) \widetilde{w}\_x(\alpha) = \sum\_i \mathrm{eq}(\tau\_1, i) y\_i(\alpha),$$
/// with $\widetilde{w}\_x(\alpha) = \sum\_y \widetilde{w}(x,y)\alpha^y$ the evaluation at
/// $\alpha$ of ring element $x$ of the witness; the verifier computes the right-hand side from
/// $\mathbf{y} = (\mathbf{v}, \mathbf{u}, Y, 0, \mathbf{0})$.
///
/// Degree $2$ in each variable: a $y$ variable appears in $\widetilde{w}$ and
/// $\widetilde{\alpha}$, an $x$ variable in $\widetilde{w}$ and $\bar{m}$, each linearly.
///
/// The variables are folded least-significant bit first in the table index order (the $\log d$
/// coefficient variables $y$ first, then the entry variables $x$), exactly as [`F0`] folds the
/// same table, so the two polynomials share every round's challenge (see [`sumcheck_proof`]).
/// While $y$ variables remain, `alpha_pows_eval_table` is folded alongside the witness table;
/// once it is the single scalar $\widetilde{\alpha}(r\_y)$, `m_bar_eval_table` is folded
/// instead.
pub struct FAlpha {
    /// Table of $\widetilde{w}$ on the remaining variables (entry index in the high bits, coefficient index in the low bits).
    w_eval_table: Vec<ExtField>,
    /// Table of $\widetilde{\alpha}$ on the remaining $y$ variables: initially $(\alpha^y)\_{y \lt d}$, a single scalar once $y$ is fully folded.
    alpha_pows_eval_table: Vec<ExtField>,
    /// Table of $\bar{m}$ on the $x$ variables (one entry per ring element of the padded witness), folded only after $y$.
    m_bar_eval_table: Vec<ExtField>
}

impl FAlpha {
    /// Build $F\_\alpha$ from the next witness `z_r` (wrapping-signed digits), the powers
    /// `alpha_pows_eval_table` $= (1, \alpha, \dots, \alpha^{d-1})$, the verifier's
    /// $\tau\_1 \in \mathbb{F}\_{q^4}^{\log n}$ and the row-major table `m_alpha_eval_table` of
    /// $\widetilde{M}\_\alpha$ (index $i \cdot \mathrm{width} + x$, with $2^{\log n}$ rows and
    /// $\mathrm{width} = 2^{v}/d$ columns). Precomputes
    /// $\bar{m}(x) = \sum\_i \mathrm{eq}(\tau\_1, i) \widetilde{M}\_\alpha(i, x)$ in
    /// $O(2^{\log n} \cdot \mathrm{width})$ field operations, pairing bit $k$ of $i$ with
    /// $\tau\_{1,k}$. Requires `m_alpha_eval_table.len() == height * width` and
    /// `width * alpha_pows_eval_table.len() == z_r.len()`.
    pub fn init(
        z_r: &Vec<u64>,
        alpha_pows_eval_table: Vec<ExtField>,
        tau_1: Vec<ExtField>,
        m_alpha_eval_table: Vec<ExtField>
    ) -> Self {
        // lift evaluations to field
        let w_eval_table: Vec<ExtField> = z_r.iter().map(|x| lift_int(*x)).collect();

        // M_alpha is stored row-major: index = row * width + column
        let log_n = tau_1.len();
        let height = 1 << log_n;
        let width = m_alpha_eval_table.len() / height;
        assert_eq!(height * width, m_alpha_eval_table.len());
        assert_eq!(width * alpha_pows_eval_table.len(), w_eval_table.len());

        // mbar(x) = sum_i eq(tau_1, i) . M_alpha(i, x)
        let mut m_bar_eval_table = vec![ExtField::ZERO; width];

        for i in 0..height {
            let mut eq_i = ExtField::ONE;

            for b in 0..log_n {
                eq_i *= eq_bin(tau_1[b], (i >> b) & 1);
            }

            for x in 0..width {
                m_bar_eval_table[x] += eq_i * m_alpha_eval_table[i * width + x];
            }
        }

        Self { w_eval_table, alpha_pows_eval_table, m_bar_eval_table }
    }
}

impl SumCheckPoly<ExtField> for FAlpha {
    /// Per-variable degree $2$.
    fn degree(&self) -> usize {
        2
    }

    /// Number of variables still to be folded, $\log\_2$ of the current witness table length.
    fn num_vars(&self) -> usize {
        self.w_eval_table.len().log()
    }

    /// The round message $g(X) = \sum\_{s} F\_\alpha(X, s)$ as its values at $X = 0, 1, 2$.
    ///
    /// While $y$ variables remain (`alpha_pows_eval_table.len() > 1`) the variable being folded
    /// is the lowest bit of the coefficient index: for every entry $x$ and every $y$-suffix,
    /// both $\widetilde{w}$ and $\widetilde{\alpha}$ are interpolated between their two table
    /// entries and the products are summed, weighted by $\bar{m}(x)$. Once $y$ is fully folded,
    /// $\widetilde{\alpha}$ is the scalar $\widetilde{\alpha}(r\_y)$ and the variable is the
    /// lowest remaining bit of the entry index: $\widetilde{w}$ and $\bar{m}$ are interpolated
    /// between their two table entries. Cost: $O(2^{v})$ field multiplications per round.
    fn get_univariate(&self) -> Univariate<ExtField> {
        // evaluate the univariate polynomial at degree + 1 different points
        let deg = self.degree();
        let mut ys = Vec::<ExtField>::with_capacity(deg + 1);

        // get lengths of the remaining variables
        let len_y = self.alpha_pows_eval_table.len();
        let len_x = self.m_bar_eval_table.len();
        assert_eq!(len_x * len_y, self.w_eval_table.len());

        for x_i in 0..=deg {
            let x_i = x_i as u64;

            // build sum_{b in {0,1}^{n-1}} f_alpha(x_i, b)
            let mut y_i = ExtField::ZERO;

            // still folding in y: the first variable is the lowest bit of the coefficient index
            if len_y > 1 {
                let half_y = len_y / 2;

                for x in 0..len_x {
                    let mut acc = ExtField::ZERO;

                    for y_suffix in 0..half_y {
                        let idx = x * half_y + y_suffix;

                        // get w(x, 0, y_suffix) and w(x, 1, y_suffix)
                        let w_0 = self.w_eval_table[idx << 1];
                        let w_1 = self.w_eval_table[(idx << 1) | 1];

                        // evaluation at x_i is linear interpolation
                        let w_x = w_0 + mul_int_field(x_i, w_1 - w_0);

                        // get alpha(0, y_suffix) and alpha(1, y_suffix)
                        let alpha_0 = self.alpha_pows_eval_table[y_suffix << 1];
                        let alpha_1 = self.alpha_pows_eval_table[(y_suffix << 1) | 1];
                        let alpha_x = alpha_0 + mul_int_field(x_i, alpha_1 - alpha_0);

                        acc += w_x * alpha_x;
                    }

                    y_i += self.m_bar_eval_table[x] * acc;
                }
            }

            // y fully folded (alpha is a scalar): folding in x
            else {
                let alpha = self.alpha_pows_eval_table[0];

                for x_suffix in 0..len_x / 2 {
                    // get w(0, x_suffix) and w(1, x_suffix)
                    let w_0 = self.w_eval_table[x_suffix << 1];
                    let w_1 = self.w_eval_table[(x_suffix << 1) | 1];
                    let w_x = w_0 + mul_int_field(x_i, w_1 - w_0);

                    // get mbar(0, x_suffix) and mbar(1, x_suffix)
                    let m_0 = self.m_bar_eval_table[x_suffix << 1];
                    let m_1 = self.m_bar_eval_table[(x_suffix << 1) | 1];
                    let m_x = m_0 + mul_int_field(x_i, m_1 - m_0);

                    y_i += w_x * m_x * alpha;
                }
            }

            ys.push(y_i);
        }

        Univariate::init(ys)
    }

    /// Bind the next variable to `r`: halve the witness table, and halve
    /// `alpha_pows_eval_table` while it has more than one entry, otherwise `m_bar_eval_table`.
    fn fix_first_variable(&mut self, r: ExtField) {
        // fold in w(x, y)
        self.w_eval_table = fix_first_variable(&self.w_eval_table, r);

        // fold in alpha(y) while y variables remain, then mbar(x)
        if self.alpha_pows_eval_table.len() > 1 {
            self.alpha_pows_eval_table = fix_first_variable(&self.alpha_pows_eval_table, r);
        }
        else {
            self.m_bar_eval_table = fix_first_variable(&self.m_bar_eval_table, r);
        }
    }
}

#[time_graph::instrument]
/// Run the prover side of the two sumchecks of the paper's Fig. 7 in lock-step, with
/// Fiat-Shamir challenges drawn from the transcript `fs`.
///
/// Both polynomials must have the same number of variables $v$ (asserted); they do, since both
/// are defined over the next witness table and fold it in the same order. In round
/// $k = 1, \dots, v$ the prover computes the round messages $g^{\alpha}\_k$ of `f_alpha` and
/// $g^{0}\_k$ of `f_0`, pushes them into `fs` in that order, derives the shared challenge
/// $r\_k \in \mathbb{F}\_{q^4}$ from a ChaCha12 generator seeded with the transcript hash, and
/// binds the next variable of both polynomials to $r\_k$. Each challenge is therefore committed
/// to everything that precedes it in the transcript (the commitments $\mathbf{u}$, $Y$,
/// $\mathbf{v}$, $\mathbf u^{\prime}$ pushed by [`crate::hachi::prove`] and all earlier round
/// messages); [`crate::hachi::verify`] replays the same pushes to recover the same challenges.
///
/// Returns `(univariates_f_alpha, univariates_f_0, y_dash)`: the two lists of round messages
/// in round order, and $y^{\prime} = \widetilde{w}(r\_1, \dots, r\_v)$, the value of the fully
/// folded witness table (read from `f_0` and asserted equal to the value held by `f_alpha`).
/// In this first-round prototype $y^{\prime}$ is sent to the verifier as the claimed evaluation
/// of the next witness and is not proven (see the crate-level documentation).
///
/// The challenge is sampled with the crate constant [`Q`] as modulus, which equals `params.q`.
pub fn sumcheck_proof(f_0: &mut F0, f_alpha: &mut FAlpha, fs: &mut FS) -> (Vec<Univariate<ExtField>>, Vec<Univariate<ExtField>>, ExtField) {
    // both polynomials are over the same witness variables (x, y), folded in the same order,
    // so they have the same number of rounds and share every round's challenge
    let rounds_f_0 = f_0.num_vars();
    let rounds_f_alpha = f_alpha.num_vars();
    assert_eq!(rounds_f_0, rounds_f_alpha);
    let mut cur = rounds_f_alpha;

    // store univariate polynomials of F_0 and F_alpha
    let mut univariates_f_0 = Vec::<Univariate<ExtField>>::with_capacity(rounds_f_0);
    let mut univariates_f_alpha = Vec::<Univariate<ExtField>>::with_capacity(rounds_f_alpha);

    while cur > 0 {
        #[cfg(feature = "verbose")]
        progress_bar("Sum Check", rounds_f_alpha - cur, rounds_f_alpha);

        let univariate_f_alpha = f_alpha.get_univariate();
        fs.push(&univariate_f_alpha);
        univariates_f_alpha.push(univariate_f_alpha);

        let univariate_f_0 = f_0.get_univariate();
        fs.push(&univariate_f_0);
        univariates_f_0.push(univariate_f_0);

        let mut rng = ChaCha12Rng::from_seed(fs.get_seed());
        let r = rand_field(Q, &mut rng);

        f_alpha.fix_first_variable(r);
        f_0.fix_first_variable(r);

        cur -= 1;
    }

    // get the evaluation of the new witness
    let y_dash = f_0.w_eval_table[0];

    // sense check
    assert_eq!(y_dash, f_alpha.w_eval_table[0]);

    (univariates_f_alpha, univariates_f_0, y_dash)
}
#[cfg(test)]
mod test_sumcheck {
    use super::*;
    use ark_poly::{DenseMultilinearExtension, Polynomial};
    use crate::arithmetic::utils::rand_int;

    /// Round-by-round consistency of the F_alpha prover against an independent computation of the
    /// claimed sum, and of its final value against the verifier's evaluation formula. Regression
    /// test for the folding-order defect: the prover must fold the coefficient variables first,
    /// then the entry variables, and never the row index (which is folded into mbar up front).
    #[test]
    fn test_f_alpha_consistency() {
        let q = Q;
        let log_n = 2; let height = 1 << log_n;
        let log_x = 3; let width = 1 << log_x;
        let log_d = 2; let d = 1 << log_d;
        let mut rng = ChaCha12Rng::from_seed([3u8; 32]);

        // witness digits in [-8, 7] as wrapping u64, random public data
        let z_r: Vec<u64> = (0..width * d).map(|_| (rand_int(16, 4, &mut rng) as i64 - 8) as u64).collect();
        let m_alpha: Vec<ExtField> = (0..height * width).map(|_| rand_field(q, &mut rng)).collect();
        let alpha_pows: Vec<ExtField> = (0..d).map(|_| rand_field(q, &mut rng)).collect();
        let tau_1: Vec<ExtField> = (0..log_n).map(|_| rand_field(q, &mut rng)).collect();

        // claimed sum: sum_i eq(tau_1, i) sum_x M(i, x) sum_y w(x, y) alpha^y
        let mut claim = ExtField::ZERO;

        for i in 0..height {
            let mut eq_i = ExtField::ONE;

            for b in 0..log_n {
                eq_i *= eq_bin(tau_1[b], (i >> b) & 1);
            }

            for x in 0..width {
                let mut wy = ExtField::ZERO;

                for y in 0..d {
                    wy += lift_int(z_r[x * d + y]) * alpha_pows[y];
                }

                claim += eq_i * m_alpha[i * width + x] * wy;
            }
        }

        // run the prover round by round with the verifier's consistency check
        let mut f_alpha = FAlpha::init(&z_r, alpha_pows.clone(), tau_1.clone(), m_alpha.clone());
        assert_eq!(log_x + log_d, f_alpha.num_vars());

        let mut chals = Vec::new();
        let mut cur = claim;

        for _ in 0..f_alpha.num_vars() {
            let g = f_alpha.get_univariate();
            assert_eq!(cur, g.binary_sum());
            let r = rand_field(q, &mut rng);
            cur = g.eval(r);
            f_alpha.fix_first_variable(r);
            chals.push(r);
        }

        // final check exactly as the verifier computes it
        let w_table: Vec<ExtField> = z_r.iter().map(|x| lift_int(*x)).collect();
        let w_r = DenseMultilinearExtension::from_evaluations_vec(log_x + log_d, w_table).evaluate(&chals);
        let alpha_r = DenseMultilinearExtension::from_evaluations_vec(log_d, alpha_pows).evaluate(&chals[0..log_d].to_vec());
        let mut m_point = chals[log_d..].to_vec();
        m_point.extend_from_slice(&tau_1);
        let m_r = DenseMultilinearExtension::from_evaluations_vec(log_n + log_x, m_alpha).evaluate(&m_point);

        assert_eq!(w_r, f_alpha.w_eval_table[0]);
        assert_eq!(cur, w_r * alpha_r * m_r);
    }
}
