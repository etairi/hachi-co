//! # Verifier of one round (paper, Sections 3.2, 4.2 and 4.3; Fig. 3 and Fig. 7)
//!
//! Checks a [`ProofRound`] against the public commitment $\mathbf u$, the evaluation point
//! $\mathbf x \in \mathbb Z_q^\ell$ (0-based coordinates, residues in $\[0, q)$) and the claimed
//! value $y \in \mathbb Z_q$. Every challenge is re-derived from the Fiat-Shamir transcript
//! ([`FS`]) exactly as the prover derives it, so both sides push the same objects in the same
//! order. Checks are assertions: a failing check panics, success returns normally. In order:
//!
//! 1. **Reduction to $\mathbf R_q$**: $y = \langle \mathrm{cf}(Y), \mathbf v_x \rangle \bmod q$
//!    with $v_{x,i} = \prod_{t \lt \alpha} x_{\ell-\alpha+t}^{i_t}$ for $i \in \[0, d)$, i.e. the
//!    last $\alpha$ coordinates of $\mathbf x$ act on the coefficient index of $Y$ (the $k = 1$
//!    form of the check of Section 3).
//! 2. **Challenges**: $c_1, \dots, c_{2^r}$ from the hash of $(\mathbf u, Y, \mathbf v)$
//!    ([`PChal::rand_vec`]), then $\alpha$, $\tau_0 \in \mathbb F_{q^4}^{\nu}$ and
//!    $\tau_1 \in \mathbb F_{q^4}^{\lceil \log_2(3n+2) \rceil}$, in this order, from a ChaCha12
//!    generator seeded by the hash of $(\mathbf u, Y, \mathbf v, \mathbf u^{\prime})$; $\nu$ is recomputed
//!    from the parameters as in the prover's `form_next_witness`.
//! 3. **Public data of the sumchecks**: the table of
//!    $\mathbf M_\alpha = \[\mathbf M^{\prime}(\alpha) \mid -(\alpha^d+1)\mathbf G_{3n+2}\]$ ([`form_m_alpha`],
//!    regenerating $\mathbf A$, $\mathbf B$, $\mathbf D$ from their seeds) and the claimed sum
//!    $a = \sum_i \mathrm{eq}(\tau_1, i) y_i(\alpha)$ of $F_{\alpha,\tau_1}$
//!    ([`compute_expected_sum_f_alpha`]).
//! 4. **Sumcheck rounds** ([`sumcheck_verify`]): for $j = 1, \dots, \nu$ and for both polynomials,
//!    $g_j(0) + g_j(1)$ must equal the running claim ($a$ for $F_\alpha$, $0$ for $F_0$); then the
//!    shared challenge $r_j$ is derived from the transcript extended by $g_{\alpha,j}$ and
//!    $g_{0,j}$, and the claim becomes $g_j(r_j)$.
//! 5. **Final evaluations**, with the prover's $y^{\prime} = \tilde w(\mathbf r)$ in place of an opening of
//!    $\mathbf u^{\prime}$: the last claim of $F_0$ must equal
//!    $\mathrm{eq}(\tau_0, \mathbf r) \cdot y^{\prime}(y^{\prime} + b/2)\prod_{j=1}^{b/2-1}(y^{\prime}^2 - j^2)$, and the
//!    last claim of $F_\alpha$ must equal
//!    $y^{\prime} \cdot \tilde\alpha(r_1, \dots, r_{\log d}) \cdot \bar m(r_{\log d + 1}, \dots, r_\nu)$,
//!    where $\tilde\alpha$ is the multilinear extension of $(1, \alpha, \dots, \alpha^{d-1})$
//!    (the coefficient variables are the first $\log d$ challenges, folded first) and
//!    $\bar m(\mathbf r_x) = \sum_i \mathrm{eq}(\tau_1, i)\tilde M_\alpha(i, \mathbf r_x)$ is the
//!    multilinear extension of the row-major table of $\mathbf M_\alpha$ (column bits low) at the
//!    point $(\mathbf r_x, \tau_1)$.
//!
//! What is not checked (first-round prototype): $y^{\prime}$ is trusted, $\mathbf u^{\prime}$ is never opened, and
//! no norm bound is verified directly (shortness of the next witness is what the $F_0$ sumcheck
//! asserts). The dominant cost is building the $\mathbf M_\alpha$ table and evaluating its
//! multilinear extension, both linear in the table size (paper, Section 4.4, verifier time).

use ark_ff::{AdditiveGroup, Field};
use ark_poly::{DenseMultilinearExtension, Polynomial};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::arithmetic::ExtField;
use crate::arithmetic::fs::FS;
use crate::arithmetic::poly::Poly;
use crate::arithmetic::poly_chal::PChal;
use crate::arithmetic::poly_vec::PVec;
use crate::arithmetic::sumcheck::Univariate;
use crate::arithmetic::utils::{Logarithm, eq, eq_bin, lift_int, multi_lin_coeff_int, powers, rand_field};

use crate::hachi::Hachi;
use crate::hachi::common::form_m_alpha;
use crate::hachi::prove::ProofRound;
use crate::hachi::setup::Parameters;

#[cfg(feature = "verbose")]
use crate::utils::verbose::tick_item;

/// Verification of an evaluation proof for a claimed value of type `F`.
pub trait Verify<F> {
    /// Verify that `proof` shows the polynomial committed to by `com` $= \mathbf u$ to evaluate to
    /// `y` at `x` (`x.len() == params.l`), under `params`. Panics if any check fails.
    fn verify(
        params: &Parameters,
        x: &[u64],
        y: F,
        com: &PVec,
        proof: &ProofRound
    );
}

/// The verifier for a claimed value $y \in \mathbb Z_q$ (as a residue in $\[0, q)$); the checks
/// are listed in the module documentation. `params` must be the prover's parameters
/// (`Hachi::setup(l, true)`).
impl Verify<u64> for Hachi{
    #[time_graph::instrument]
    fn verify(
        params: &Parameters,
        x: &[u64],
        y: u64,
        com : &PVec,
        proof: &ProofRound
        ) {
            #[cfg(feature = "verbose")]
            println!("\n==== Verify ====");

            // initialise a FS stream and push the commitment
            let mut fs = FS::init();
            fs.push(com);

            // verify the reduction to Rq
            let log_d = params.d.log();
            let x_v = &x[params.l - log_d..params.l];
            let mut v = vec![0u64; params.d];

            for i in 0..params.d {
                v[i] = multi_lin_coeff_int(x_v, i, log_d, params.q);
            }

            let ring_y = proof.y.slice();
            assert_eq!(v.len(), ring_y.len());

            let mut y_actual = 0;

            for i in 0..params.d {
                y_actual = (y_actual + ring_y[i] * v[i]) % params.q;
            }

            assert_eq!(y, y_actual);

            #[cfg(feature = "verbose")]
            tick_item("Verify reduction to Rq");

            // sample challenges
            fs.push(&proof.y);
            fs.push(&proof.v);
            let seed = fs.get_seed();
            let challenges = PChal::rand_vec(1 << params.r, params.d, params.k, seed);

            // sample alpha
            fs.push(&proof.u_dash);
            let mut rng = ChaCha12Rng::from_seed(fs.get_seed());
            let alpha = rand_field(params.q, &mut rng);

            // Get the powers of alpha
            let alpha_pows = powers(alpha, params.d);

            // Get [M' | -(X^d+1).G_{3n+2}] evaluated at alpha (the quotient r is gadget-decomposed)
            let m_alpha = form_m_alpha(params, x, &challenges, &alpha_pows);

            // Get lengths for tau_0 and tau_1
            let mu = 
            (1 << params.r) * params.delta                          // w_hat
            + params.n * (1 << params.r) * params.delta             // t_hat
            + (1 << params.m) * params.delta * params.delta_z;      // z_hat    
            
            // number of rows of M (the r part of the new witness holds n * delta decomposed quotients)
            let n = params.n * 3 + 2;

            // new witness must have power of 2 length, so we pad (mu + n * delta) to the next power of 2
            // (the same formula as form_next_witness on the prover side)
            let num_vars = ((mu + n * params.delta) * params.d).log();

            // Sample the random field elements tau_0 and tau_1
            let mut tau_0 = vec![ExtField::ZERO; num_vars];
            
            for i in 0..num_vars {
                tau_0[i] = rand_field(params.q, &mut rng);
            }

            let log_n = (n as u64).log();
            let mut tau_1 = vec![ExtField::ZERO; log_n];
            
            for i in 0..log_n {
                tau_1[i] = rand_field(params.q, &mut rng);
            }

            // Get expected sum for F_alpha
            let sum_f_alpha = compute_expected_sum_f_alpha(params, com, proof, &alpha_pows, &tau_1);

            // Verify sum check
            let (f_alpha_expected, f_0_expected, chals_f_alpha, chals_f_0) = sumcheck_verify(
                &proof.univariates_f_alpha, 
                &proof.univariates_f_0, 
                sum_f_alpha, 
                &mut fs,
                params.q
            );

            // Check evaluation of f_0
            // evaluate eq
            let mut eq_r = ExtField::ONE;
            assert_eq!(tau_0.len(), chals_f_0.len());

            for i in 0..tau_0.len() {
                eq_r *= eq(tau_0[i], chals_f_0[i]);
            }

            // perform the multiplication v_r=w.(w+b/2).(w-1)(w+1). ... .(w-b/2-1)(w+b/2-1)
            let mut v_r = proof.y_dash * (proof.y_dash + lift_int(params.b / 2));
            
            // multiply with difference of two squares to improve efficiency
            let w_r_squared = proof.y_dash * proof.y_dash;

            for r in 1..params.b as usize / 2 { 
                v_r *= w_r_squared - lift_int((r * r) as u64);
            }

            let f_0_actual = eq_r * v_r;
            assert_eq!(f_0_expected, f_0_actual);

            #[cfg(feature = "verbose")]
            tick_item("Verify sum check on F_0");
            
            // Check evaluation of f_alpha = w(x,y) . alpha(y) . mbar(x) at the sumcheck point.
            // The challenges bind the witness index LSB-first: the log d coefficient variables y
            // first, then the entry variables x (the same order as F0, with which every round is shared).
            assert_eq!(1 << log_d, alpha_pows.len());
            let alpha_mle = DenseMultilinearExtension::from_evaluations_vec(log_d, alpha_pows);
            let alpha_r = alpha_mle.evaluate(&chals_f_alpha[0..log_d].to_vec());

            // mbar(r_x) = sum_i eq(tau_1, i) M_alpha(i, r_x) is the multilinear extension of the
            // row-major M_alpha table evaluated at (r_x, tau_1): the column bits are its low bits.
            let m_vars = (m_alpha.len() as u64).log();
            assert_eq!(chals_f_alpha.len() - log_d + tau_1.len(), m_vars);
            let mut m_point = chals_f_alpha[log_d..].to_vec();
            m_point.extend_from_slice(&tau_1);
            let m_mle = DenseMultilinearExtension::from_evaluations_vec(m_vars, m_alpha);
            let m_r = m_mle.evaluate(&m_point);

            let f_alpha_actual = proof.y_dash * alpha_r * m_r;
            assert_eq!(f_alpha_expected, f_alpha_actual);

            #[cfg(feature = "verbose")]
            tick_item("Verify sum check on F_alpha");

            #[cfg(feature = "verbose")]
            println!("==== Complete ====\n");
    }
}

/// The claimed sum $a = \sum_i \mathrm{eq}(\tau_1, i) y_i(\alpha)$ of the $F_{\alpha,\tau_1}$
/// sumcheck (paper, Section 4.3, below Eq. (23)), where
/// $\mathbf y = (\mathbf v, \mathbf u, Y, 0, \mathbf 0_n)$ is the right-hand side of Eq. (20) in
/// the row order of $\mathbf M_\alpha$ ($3n+2$ entries; the rows padding the table up to a power
/// of two are zero and contribute nothing) and $y_i(\alpha)$ evaluates the ring element $y_i$ at
/// $\alpha$ using `alpha_pows` $= (1, \alpha, \dots, \alpha^{d-1})$. The bits of $i$ are matched
/// with $\tau_1$ least significant first.
fn compute_expected_sum_f_alpha(
    params: &Parameters, 
    com: &PVec, 
    proof: &ProofRound, 
    alpha_pows: &[ExtField], tau_1: &[ExtField]
) -> ExtField {
    // evaluate Y=[v u y 0 0] at alpha
    let mut y_alpha = Vec::<ExtField>::new();

    for i in 0..proof.v.length() {
        y_alpha.push(proof.v.element(i).eval(alpha_pows));
    }

    for i in 0..com.length() {
        y_alpha.push(com.element(i).eval(alpha_pows));
    }

    y_alpha.push(proof.y.element(0).eval(alpha_pows));

    y_alpha.push(ExtField::ZERO);

    for _ in 0..params.n {
        y_alpha.push(ExtField::ZERO);
    }

    // sense check
    assert_eq!(y_alpha.len().log(), tau_1.len());

    // compute a=sum_i eq(tau_1, i) . y_alpha[i]
    let mut a = ExtField::ZERO;

    for i in 0..y_alpha.len() {
        let mut eq = ExtField::ONE;

        for j in 0..tau_1.len() {
            eq *= eq_bin(tau_1[j], (i >> j) & 1);
        }

        a += eq * y_alpha[i];
    }

    a
}

/// The round-by-round check of both sumchecks (paper, Fig. 6), which run over the same variables
/// and share every challenge. Starting from the claims `sum_f_alpha` (for $F_\alpha$) and $0$ (for
/// $F_0$), round $j$ asserts $g_j(0) + g_j(1) =$ claim for each polynomial, appends
/// $g_{\alpha,j}$ then $g_{0,j}$ to the transcript `fs`, derives $r_j \in \mathbb F_{q^4}$ from its
/// hash and replaces the claim by $g_j(r_j)$ (Lagrange interpolation, [`Univariate::eval`]).
///
/// Returns the final claims of $F_\alpha$ and $F_0$ (the values the verifier must reproduce from
/// $y^{\prime}$) and the challenge vector, once per polynomial (they are identical). Panics if the two
/// proofs have different numbers of rounds.
fn sumcheck_verify(
    univariates_f_alpha: &Vec<Univariate<ExtField>>, 
    univariates_f_0: &Vec<Univariate<ExtField>>,
    sum_f_alpha: ExtField,
    fs: &mut FS,
    q: u64
) -> (ExtField, ExtField, Vec<ExtField>, Vec<ExtField>) {
    // both sumchecks run over the same variables and share every round's challenge
    let rounds_f_0 = univariates_f_0.len();
    let rounds_f_alpha = univariates_f_alpha.len();
    assert_eq!(rounds_f_0, rounds_f_alpha);
    let mut cur = rounds_f_alpha;

    let mut cur_check_f_alpha = sum_f_alpha;
    let mut cur_check_f_0 = ExtField::ZERO;

    let mut challenges_f_alpha = Vec::<ExtField>::new();
    let mut challenges_f_0 = Vec::<ExtField>::new();

    while cur > 0 {
        // get the univariate polynomial g(x) for this round
        let univariate_f_alpha = univariates_f_alpha[rounds_f_alpha - cur].clone();
        let univariate_f_0 = univariates_f_0[rounds_f_0 - cur].clone();

        // check g(0) + g(1)
        assert_eq!(cur_check_f_alpha, univariate_f_alpha.binary_sum());
        assert_eq!(cur_check_f_0, univariate_f_0.binary_sum());

        // sample the challenge
        fs.push(&univariate_f_alpha);
        fs.push(&univariate_f_0);
        let mut rng = ChaCha12Rng::from_seed(fs.get_seed());
        let r = rand_field(q, &mut rng);
        challenges_f_alpha.push(r);
        challenges_f_0.push(r);

        // update the expected binary sum
        cur_check_f_alpha = univariate_f_alpha.eval(r);
        cur_check_f_0 = univariate_f_0.eval(r);

        cur -= 1;
    }

    (cur_check_f_alpha, cur_check_f_0, challenges_f_alpha, challenges_f_0)
}
