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

/// Representation of the polynomial f_0 = eq(tau_0, x) w(x)(w(x)-1)(w(x)+1)...
pub struct F0 {
    w: Vec<u64>,                            // witness w
    w_eval_table: Vec<ExtField>,            // table of evaluations for w
    base: u64,                              // decomposition base
    tau_0: Vec<ExtField>,                   // random vector tau_0
    eq_scalar: ExtField,                    // scalar carried through for eq multiplication
    q: u64,                                 // modulus,
    len: usize                              // length of non-padded part of w
}

impl F0 {
    /// Construct F0.
    pub fn init(z_r: &Vec<u64>, base: u64, tau_0: Vec<ExtField>, q: u64, len: usize) -> Self {
        // lift evaluations to field
        // let w: Vec<BaseField> = z_r.iter().map(|x| BaseField::from(*x)).collect();
        let w_eval_table: Vec<ExtField> = z_r.iter().map(|x| lift_int(*x)).collect();

        Self { w: z_r.clone(), w_eval_table, base, tau_0, eq_scalar: ExtField::ONE, q, len }
    }
}

impl SumCheckPoly<ExtField> for F0 {
    fn degree(&self) -> usize {
        (self.base + 1) as usize
    }

    fn num_vars(&self) -> usize {
        self.w_eval_table.len().log()
    }

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

/// Representation of the polynomial f_alpha(x,y) = w(x,y) . alpha(y) . mbar(x), where
/// mbar(x) = sum_i eq(tau_1, i) . M_alpha(i, x) folds the row index i of the linear relation
/// into the public factor (this is F_{alpha,tau_1} of the Hachi paper, Section 4.3). Its sum over
/// the boolean hypercube is sum_i eq(tau_1, i) . (M_alpha w)_i = sum_i eq(tau_1, i) . y_i(alpha).
///
/// The variables are folded LSB-first in the witness index order (the log d ring-coefficient
/// variables y first, then the witness-entry variables x), exactly as F0 folds the same table,
/// so both polynomials share every round's challenge.
pub struct FAlpha {
    w_eval_table: Vec<ExtField>,             // table of evaluations for w (entry index in the high bits, coefficient index in the low bits)
    alpha_pows_eval_table: Vec<ExtField>,    // table of evaluations for the powers of alpha (over y)
    m_bar_eval_table: Vec<ExtField>          // table of evaluations for mbar (over x)
}

impl FAlpha {
    /// Construct FAlpha from the witness, the powers of alpha, tau_1 and the (row-major) table of M_alpha.
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
    fn degree(&self) -> usize {
        2
    }

    fn num_vars(&self) -> usize {
        self.w_eval_table.len().log()
    }

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
/// Sum check proof.
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
