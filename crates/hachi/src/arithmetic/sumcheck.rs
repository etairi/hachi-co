//! Generic sumcheck helpers: the [`SumCheckPoly`] interface implemented by the prover's
//! polynomials, the [`Univariate`] round message, and the least-significant-bit-first fold of an
//! evaluation table.
//!
//! This is the field-side machinery of the sumcheck protocol of the paper, Section 4.3
//! (Fig. 6 and Fig. 7). The prover holds a polynomial $H$ in $\nu$ variables over
//! $\mathbb F_{q^4}$ as tables of evaluations on the boolean hypercube $\lbrace 0, 1 \rbrace^\nu$
//! and, in round $i$, sends the univariate polynomial
//! $$g_i(X_i) = \sum_{b_{i+1}, \dots, b_\nu \in \lbrace 0, 1 \rbrace} H(a_1, \dots, a_{i-1}, X_i, b_{i+1}, \dots, b_\nu),$$
//! after which the verifier replies with a challenge $a_i \in \mathbb F_{q^4}$ and the prover
//! substitutes $X_i = a_i$ into its tables. The concrete polynomials $F_{0,\tau_0}$ and
//! $F_{\alpha,\tau_1}$ of the paper are built in the prover module `hachi::prover_utils::sumcheck`; the
//! verifier's side of the protocol is in [`crate::hachi::verify`].
//!
//! Conventions: variables are numbered from the least significant bit of the table index, so
//! "the first variable" of a table `t` is the bit selecting between `t[2s]` and `t[2s + 1]`, and
//! every fold halves the table (see [`fix_first_variable`]). A [`Univariate`] of degree $D$ is
//! stored as its $D + 1$ evaluations at $0, 1, \dots, D$, not as coefficients.

use ark_ff::{AdditiveGroup, BigInteger, Field, PrimeField};

use crate::arithmetic::{ExtField, fs::Serialise, utils::lift_int};

/// Prover-side interface of a polynomial $H(X_1, \dots, X_\nu)$ over $F$ to which the sumcheck
/// protocol is applied (paper, Section 4.3, Fig. 6, where $H = P \cdot Q(\tilde{w})$ with $P$,
/// $Q$ public and $\tilde{w}$ the witness).
///
/// An implementor keeps its state as evaluation tables on the boolean hypercube and supports the
/// two operations of a sumcheck round: producing the round message
/// ([`get_univariate`](SumCheckPoly::get_univariate)) and substituting the verifier's challenge
/// into the first remaining variable ([`fix_first_variable`](SumCheckPoly::fix_first_variable)).
/// Variables are consumed in table-index order, least significant bit first, and
/// [`num_vars`](SumCheckPoly::num_vars) decreases by one per round. Table indices are `usize`,
/// so a polynomial has at most 64 variables.
pub trait SumCheckPoly<F: Field> {
    /// Number of variables $\nu$ not yet fixed, i.e. the number of sumcheck rounds remaining.
    fn num_vars(&self) -> usize;

    /// Upper bound $D$ on the degree of $H$ in each single variable; a round message is
    /// represented by $D + 1$ evaluations.
    fn degree(&self) -> usize;

    /// The round message for the first remaining variable,
    /// $$g(X_1) = \sum_{b \in \lbrace 0, 1 \rbrace^{\nu - 1}} H(X_1, b),$$
    /// represented by its $D + 1$ evaluations at $X_1 = 0, 1, \dots, D$ with
    /// $D$ = [`degree`](SumCheckPoly::degree).
    fn get_univariate(&self) -> Univariate<F>;

    /// Substitute the verifier's challenge $r$ for the first remaining variable,
    /// $H \leftarrow H(r, X_2, \dots, X_\nu)$, which reduces [`num_vars`](SumCheckPoly::num_vars)
    /// by one.
    fn fix_first_variable(&mut self, r: F);
}

#[derive(Debug, Clone)]
/// A univariate polynomial $g \in F\[X\]$ of degree at most $D$, stored as its $D + 1$
/// evaluations $g(0), g(1), \dots, g(D)$: the sumcheck round message $g_i(X_i)$ of the paper's
/// Fig. 6. The verifier only ever needs $g(0) + g(1)$ and $g(r)$ at its challenge $r$, which
/// [`binary_sum`](Univariate::binary_sum) and [`eval`](Univariate::eval) compute directly from
/// the evaluations, so the coefficient form is never materialised.
pub struct Univariate<F : Field> {
    /// `evals[i]` $= g(i)$ for $i = 0, \dots, D$.
    evals: Vec<F>
}

/// Operations on round messages over the sumcheck field $\mathbb F_{q^4}$ ([`ExtField`]).
impl Univariate<ExtField> {
    /// Wrap the evaluations $g(0), \dots, g(D)$, in this order, of a polynomial of degree at
    /// most $D$. At least two evaluations are needed for [`binary_sum`](Univariate::binary_sum).
    pub fn init(evals: Vec<ExtField>) -> Self {
        Self { evals }
    }

    /// $g(0) + g(1)$: the value the verifier compares with the running claim $z_{i-1}$ of the
    /// paper's Fig. 6.
    pub fn binary_sum(&self) -> ExtField {
        self.evals[0] + self.evals[1]
    }

    /// $g(x)$ by Lagrange interpolation through the nodes $0, 1, \dots, D$:
    /// $$g(x) = \sum_{i=0}^{D} g(i) \prod_{j \ne i} \frac{x - j}{i - j},$$
    /// with the integer nodes lifted into $\mathbb F_{q^4}$ by [`lift_int`]. Costs
    /// $O(D^2)$ field multiplications and $(D + 1) D$ field inversions, which is negligible for
    /// the degrees used here ($D \le b + 1$).
    pub fn eval(&self, x: ExtField) -> ExtField {
        let n = self.evals.len();
        let mut y = ExtField::ZERO;

        for i in 0..n {
            let mut l_i = ExtField::ONE;

            for j in 0..n {
                if i != j {
                    let x_i = lift_int(i as u64);
                    let x_j = lift_int(j as u64);
                    l_i *= (x - x_j) * (x_i - x_j).inverse().unwrap();
                }
            }

            y += self.evals[i] * l_i;
        }

        y
    }
}

/// Transcript encoding of a round message (see [`crate::arithmetic::fs`]): each evaluation
/// $g(i)$, in order $i = 0, \dots, D$, as its four $\mathbb Z_q$ coordinates
/// $(c_{0,0}, c_{0,1}, c_{1,0}, c_{1,1})$ in the tower
/// $\mathbb F_{q^2} = \mathbb Z_q\[u\]/(u^2 - \nu)$ with $\nu = 5 \cdot 2^{-64} \bmod q$ (see [`crate::arithmetic::field`]), $\mathbb F_{q^4} = \mathbb F_{q^2}\[v\]/(v^2 - u)$
/// of [`crate::arithmetic::field`] (an element is $c_0 + c_1 v$ with $c_i = c_{i,0} + c_{i,1} u$),
/// each coordinate as its canonical (non-Montgomery) residue in 8 little-endian bytes;
/// $32 (D + 1)$ bytes in total.
impl Serialise for Univariate<ExtField> {
    fn serialise(&self) -> Vec<u8> {
        let mut bytes = Vec::<u8>::new();

        for y in &self.evals {
            bytes.extend_from_slice(&y.c0.c0.into_bigint().to_bytes_le());
            bytes.extend_from_slice(&y.c0.c1.into_bigint().to_bytes_le());
            bytes.extend_from_slice(&y.c1.c0.into_bigint().to_bytes_le());
            bytes.extend_from_slice(&y.c1.c1.into_bigint().to_bytes_le());
        }

        bytes
    }
}

/// Fix the first variable of a multilinear evaluation table to $r$, least significant bit first.
///
/// `eval_table` holds the $2^\nu$ values $p(b_1, \dots, b_\nu)$ of a multilinear polynomial on
/// the hypercube at index $b_1 + 2 b_2 + \dots + 2^{\nu - 1} b_\nu$, so that
/// `eval_table[2s]` $= p(0, s)$ and `eval_table[2s + 1]` $= p(1, s)$ for every suffix $s$. The
/// result holds the $2^{\nu - 1}$ values of $p(r, b_2, \dots, b_\nu)$ at index
/// $b_2 + 2 b_3 + \dots$, obtained by linear interpolation
/// $$p(r, s) = p(0, s) + r \cdot (p(1, s) - p(0, s)).$$
/// The table length must be even (in use, a power of two of at least 2); the cost is linear in
/// the table length. This is the table update behind every
/// [`SumCheckPoly::fix_first_variable`], and it fixes the order in which the sumcheck
/// challenges are bound to variables: the challenge of round $i$ goes to bit $i - 1$ of the
/// original table index.
pub fn fix_first_variable(eval_table: &Vec<ExtField>, r: ExtField) -> Vec<ExtField> {
    let half = eval_table.len() / 2;
    let mut new_table = vec![ExtField::ZERO; half];

    for suffix in 0..half {
        // p(0, suffix)
        let p_0 = eval_table[suffix << 1];

        // p(1, suffix)
        let p_1 = eval_table[(suffix << 1) | 1];

        // interpolate
        new_table[suffix] = p_0 + (p_1 - p_0) * r;
    }

    new_table
}
