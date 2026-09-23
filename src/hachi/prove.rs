//! # Evaluation proof, one round (paper, Sections 3.2, 4.2 and 4.3; Fig. 3 and Fig. 7)
//!
//! Proves that the committed polynomial $f \in \mathbb Z_q^{\le 1}\[X_1, \dots, X_\ell\]$ evaluates
//! to $y$ at the public point $\mathbf x \in \mathbb Z_q^\ell$ (the base-field case, $k = 1$: the
//! coefficients and the point are in $\mathbb Z_q$, the sumchecks run over
//! $\mathbb F_{q^4}$ = [`ExtField`]). Coordinates of $\mathbf x$ are 0-based, `x[t]` $= x_t$,
//! residues in $\[0, q)$. The prover is [`Prove::prove`]; its steps, in order, are:
//!
//! 1. **Reduction to $\mathbf R_q$ and first message** ([`compute_y_and_w`], one pass over the
//!    witness). With $b_i = \prod_{t \lt r} x_t^{i_t}$ for $i \in \lbrace 0,1 \rbrace^r$ and
//!    $a_j = \prod_{t \lt m} x_{r+t}^{j_t}$ for $j \in \lbrace 0,1 \rbrace^m$ (Eq. (12), (15)), the
//!    prover computes $w_i = \mathbf a^{\mathsf T} \mathbf f_i \in \mathbf R_q$ and
//!    $Y = \sum_i b_i w_i = \mathbf b^{\mathsf T} \mathbf w \in \mathbf R_q$ (Eq. (17)). The last
//!    $\alpha$ variables are absorbed into the ring coefficient index, so the verifier checks
//!    $y = \langle \mathrm{cf}(Y), \mathbf v_x \rangle$ with $v_{x,i} = \prod_{t \lt \alpha} x_{r+m+t}^{i_t}$:
//!    this is the $k = 1$ form of the reduction of Section 3, where the trace-map check
//!    degenerates to an inner product over $\mathbb Z_q$.
//! 2. **Commitment to $\mathbf w$** ([`commit_w_and_lift`]): $\hat{\mathbf w} = \mathbf G^{-1}(\mathbf w) \in \mathbf R_q^{\delta 2^r}$
//!    and $\mathbf v = \mathbf D \hat{\mathbf w} \in \mathbf R_q^{n}$ (Eq. (16)), computed over
//!    $\mathbb Z_q\[X\]$ and divided by $X^d + 1$ so that the quotient $\mathbf v_{quo}$ with
//!    $\mathbf D \hat{\mathbf w} = \mathbf v + (X^d+1)\mathbf v_{quo}$ is available for step 5.
//! 3. **Challenges.** The transcript $(\mathbf u, Y, \mathbf v)$ is hashed ([`FS`]) into the seed
//!    of $c_1, \dots, c_{2^r} \in \mathcal C$, each with $\omega$ nonzero $\pm 1$ coefficients
//!    ([`PChal::rand_vec`]).
//! 4. **Response** ([`compute_z`], second pass over the witness):
//!    $\mathbf z = \sum_i c_i \mathbf s_i \in \mathbf R_q^{\delta 2^m}$ as wrapping-signed
//!    integers, with $\lVert \mathbf z \rVert_\infty \le \beta$ asserted (the abort of Fig. 3).
//! 5. **Ring switching** (Section 4.3): every row of the verification equation Eq. (20),
//!    $$\begin{bmatrix} \mathbf D & 0 & 0 \cr 0 & \mathbf B & 0 \cr \mathbf b^{\mathsf T}\mathbf G_{2^r} & 0 & 0 \cr
//!    \mathbf c^{\mathsf T} \otimes \mathbf G_1 & 0 & -\mathbf a^{\mathsf T}\mathbf G_{2^m}\mathbf J_{2^m} \cr
//!    0 & \mathbf c^{\mathsf T} \otimes \mathbf G_n & -\mathbf A \mathbf J_{2^m} \end{bmatrix}
//!    \begin{bmatrix} \hat{\mathbf w} \cr \hat{\mathbf t} \cr \hat{\mathbf z} \end{bmatrix} =
//!    \begin{bmatrix} \mathbf v \cr \mathbf u \cr Y \cr 0 \cr 0 \end{bmatrix} \quad \text{over } \mathbf R_q,$$
//!    is recomputed over $\mathbb Z_q\[X\]$ (products of degree $\lt 2d$, [`Ring`] with
//!    `cyclotomic = false`) and divided by $X^d + 1$ ([`PVec::cyclotomic_div`]) to obtain the
//!    quotient $\mathbf r$ with $\mathbf M^{\prime} \mathbf z^{\prime} = \mathbf y + (X^d+1)\mathbf r$. Here
//!    $\mathbf J_{2^m}$ is the $\tau$-digit gadget and $\hat{\mathbf z} = \mathbf J^{-1}(\mathbf z)$.
//!    Row by row: (1) $\mathbf v$, quotient from step 2; (2) $\mathbf u = \mathbf B \hat{\mathbf t}$
//!    recomputed, its remainder asserted equal to $\mathbf u$, quotient $\mathbf u_{quo}$;
//!    (3) $\mathbf b^{\mathsf T} \mathbf w = Y$ has integer scalars $b_i$, so the degree stays
//!    $\lt d$ and the quotient is $0$; (4) $\sum_i c_i w_i - \mathbf a^{\mathsf T}\mathbf z = 0$:
//!    only $\sum_i c_i w_i$ (computed from $\mathbf w$, which equals
//!    $(\mathbf c^{\mathsf T} \otimes \mathbf G_1)\hat{\mathbf w}$) has a quotient, since the $a_j$
//!    are integers; (5) $\sum_i c_i \mathbf t_i - \mathbf A \mathbf z = 0$: both terms have
//!    quotients ([`lift_c_i_t_i`] and $\mathbf A \mathbf z$), their remainders are asserted equal
//!    and the row's quotient is their difference.
//! 6. **Next witness** ([`form_next_witness`]): $\mathbf z^{\prime} = (\hat{\mathbf w}, \hat{\mathbf t}, \hat{\mathbf z})$,
//!    $\mu = \delta 2^r + n\delta 2^r + \delta\tau 2^m$ ring elements, followed by
//!    $\mathbf r = \mathbf G^{-1}$ of the five quotients, $(3n+2)\delta$ elements in row order:
//!    digits of $\mathbf v_{quo}$ ($n\delta$), of $\mathbf u_{quo}$ ($n\delta$), zeros for row 3
//!    ($\delta$), digits of the row-4 quotient ($\delta$) and of the row-5 quotient ($n\delta$).
//!    The elements are flattened coefficient-wise (witness index $= $ entry $\cdot d +$ coefficient)
//!    and zero-padded to $2^\nu$ entries, $\nu = \lceil \log_2((\mu + (3n+2)\delta) d) \rceil$
//!    (`num_vars`). This is the table of $\tilde w$ of Eq. (21), with the $\log d$ coefficient
//!    bits as the low variables.
//! 7. **Commitment to the next witness**: `Hachi::setup(num_vars, false)` and
//!    [`Commit::commit`] without decomposition (Section 4.5), giving $\mathbf u^{\prime} \in \mathbf R_q^{n^{\prime}}$.
//! 8. **Field challenges.** The transcript is extended by $\mathbf u^{\prime}$ and its hash seeds a
//!    ChaCha12 generator that yields, in this order, $\alpha \in \mathbb F_{q^4}$,
//!    $\tau_0 \in \mathbb F_{q^4}^{\nu}$ and $\tau_1 \in \mathbb F_{q^4}^{\lceil \log_2(3n+2) \rceil}$.
//! 9. **Verification matrix** ([`crate::hachi::common`]): the table of
//!    $\mathbf M_\alpha = \[\mathbf M^{\prime}(\alpha) \mid -(\alpha^d+1)\mathbf G_{3n+2}\]$.
//! 10. **Sumchecks** ([`F0`], [`FAlpha`], [`sumcheck_proof`]) over the $\nu$ variables of
//!     $\tilde w$, both folded least-significant bit first (the $\log d$ coefficient bits, then the
//!     entry bits) and sharing every round's challenge $r_j$, which is derived from the transcript
//!     after the round polynomial of $F_\alpha$ and then that of $F_0$ have been appended:
//!     $$F_{0,\tau_0}(\mathbf x) = \mathrm{eq}(\tau_0, \mathbf x) \cdot \tilde w(\mathbf x) (\tilde w(\mathbf x) + b/2) \prod_{j=1}^{b/2-1} (\tilde w(\mathbf x)^2 - j^2),$$
//!     claimed sum $0$, degree $b+1$ per variable, vanishing on the hypercube iff every entry is
//!     a balanced digit in $\[-b/2, b/2-1\]$ (the paper's Eq. (23) instead uses the roots
//!     $0, \pm 1, \dots, \pm(b-1)$; the code checks the exact digit set of $\mathbf G^{-1}$, which is
//!     stricter), and
//!     $$F_{\alpha,\tau_1}(\mathbf x, \mathbf y) = \tilde w(\mathbf x, \mathbf y) \cdot \tilde\alpha(\mathbf y) \cdot \sum_i \mathrm{eq}(\tau_1, i) \tilde M_\alpha(i, \mathbf x),$$
//!     claimed sum $a = \sum_i \mathrm{eq}(\tau_1, i) y_i(\alpha)$ with $\mathbf y = (\mathbf v, \mathbf u, Y, 0, \mathbf 0_n)$,
//!     degree $2$ per variable (Section 4.3, Fig. 5 and Fig. 6). Here $\mathbf y$ are the $\log d$
//!     coefficient variables, $\mathbf x$ the entry variables and $\tilde\alpha(\mathbf y) = \alpha^{\mathbf y}$.
//! 11. **Output**: [`ProofRound`] with $y^{\prime} = \tilde w(r_1, \dots, r_\nu)$, the evaluation of the
//!     next witness at the sumcheck challenges.
//!
//! Not implemented (first-round prototype): the opening of $\mathbf u^{\prime}$ at $\mathbf r$ that would
//! justify $y^{\prime}$, any hiding or zero-knowledge, and the recursion to the next round. The witness is
//! read twice (steps 1 and 4); the table of $\mathbf M_\alpha$ and the sumcheck tables are held in
//! memory.

use ark_ff::AdditiveGroup;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::arithmetic::ExtField;
use crate::arithmetic::poly_chal::PChal;
use crate::arithmetic::poly_vec::PVec;
use crate::arithmetic::ring::Ring;
use crate::arithmetic::fs::FS;
use crate::arithmetic::sumcheck::Univariate;
use crate::arithmetic::utils::{Logarithm, powers, rand_field};

use crate::hachi::prover_utils::sumcheck::{F0, FAlpha, sumcheck_proof};
use crate::stream::Stream;
use crate::stream::file_stream::U64FileStream;

#[cfg(feature = "verbose")]
use crate::utils::verbose::{tick_item};

use crate::hachi::Hachi;
use crate::hachi::setup::{Setup, Parameters};
use crate::hachi::commit::{Commit, CommitmentWithState};
use crate::hachi::prover_utils::rq::{commit_w_and_lift, compute_z, lift_c_i_t_i, sample_matrices};
use crate::hachi::prover_utils::zq_zq::{compute_y_and_w, form_next_witness};
use crate::hachi::common::{form_m_alpha_different_matrices, form_m_alpha_same_matrix};

/// One round of the evaluation proof: the prover's messages of Fig. 3 and Fig. 7, made
/// non-interactive by Fiat-Shamir (every challenge is re-derived by the verifier from
/// $\mathbf u$ and these fields).
pub struct ProofRound {
    /// $Y \in \mathbf R_q$ (a [`PVec`] of one element): the reduction of the evaluation claim to
    /// $\mathbf R_q$, $Y = \mathbf b^{\mathsf T}\mathbf w$; the verifier checks
    /// $y = \langle \mathrm{cf}(Y), \mathbf v_x \rangle$.
    pub y: PVec,

    /// $\mathbf v = \mathbf D \hat{\mathbf w} \in \mathbf R_q^{n}$: commitment to the decomposed
    /// first message (Eq. (16)).
    pub v: PVec,

    /// $\mathbf u^{\prime} \in \mathbf R_q^{n^{\prime}}$: outer commitment to the next witness
    /// $(\mathbf z^{\prime}, \mathbf r)$ under the parameters `Hachi::setup(num_vars, false)`.
    pub u_dash: PVec,

    /// Round polynomials of the sumcheck for $F_{\alpha,\tau_1}$, one per variable of $\tilde w$,
    /// each given by its $3$ evaluations at $0, 1, 2$.
    pub univariates_f_alpha: Vec<Univariate<ExtField>>,

    /// Round polynomials of the sumcheck for $F_{0,\tau_0}$, one per variable, each given by its
    /// $b+2$ evaluations at $0, \dots, b+1$.
    pub univariates_f_0: Vec<Univariate<ExtField>>,

    /// $y^{\prime} = \tilde w(r_1, \dots, r_\nu)$: the claimed evaluation of the next witness at the
    /// sumcheck challenges. Supplied by the prover and trusted by the verifier in this prototype
    /// (no opening of $\mathbf u^{\prime}$).
    pub y_dash: ExtField
}

/// Evaluation proof for a witness of type `T` at a point with coordinates of type `F`.
pub trait Prove<T, F> {
    /// One round of the evaluation proof for the multilinear polynomial `witness`, committed as
    /// `com` under `params`, at the point `x` (`x.len() == params.l`). The claimed value $y$ is not
    /// an input: it is determined by the witness and checked by the verifier against $Y$.
    fn prove(
        witness: T,                     // witness polynomial
        params: &Parameters,            // parameters
        x: &[F],                        // evaluation point
        com: &CommitmentWithState       // commitment with internal state
    ) -> ProofRound;
}

/// The prover for a witness streamed from a file and an evaluation point over $\mathbb Z_q$
/// (coordinates as residues in $\[0, q)$). The steps are listed in the module documentation.
///
/// Preconditions: `params` from `Hachi::setup(l, true)` (the response is folded from the
/// decomposed chunks, so `decomp_witness` must be set), `witness.length() >= 1 << params.l`, and
/// `com` produced by [`Commit::commit`] from the same witness and parameters. Panics, via
/// assertions, if $\lVert \mathbf z \rVert_\infty \gt \beta$ or if a lifted equation disagrees with
/// its $\mathbf R_q$ counterpart. The commitment matrices are regenerated from the seeds.
impl Prove<&mut U64FileStream, u64> for Hachi {
    #[time_graph::instrument]
    fn prove(
            witness: &mut U64FileStream,
            params: &Parameters,
            x: &[u64],
            CommitmentWithState { t, t_hat, u }: &CommitmentWithState
        ) -> ProofRound {
        #[cfg(feature = "verbose")]
        println!("\n==== Evaluate ====");
        
        // ensure stream has sufficient length
        assert!(witness.length() >= 1 << params.l);

        // initialise a FS stream and push the commitment
        let mut fs = FS::init();
        fs.push(u);

        // create the rings Zq[X] and Zq[X]/(X^d+1)
        let ring_full = Ring::init(params.q, params.d, false);
        let ring_cyclotomic = Ring::init(params.q, params.d, true);

        // sample commitment matrices
        let 
        (
            mat_a, mat_a_ntt,
            mat_b, mat_b_ntt, 
            mat_d, mat_d_ntt
        ) = sample_matrices(params, &ring_full);

        // ---- First prover message ----
        // reduce to an evaluation over Rq and calculate w
        let mut y = PVec::zero(1, params.d);
        let mut w = PVec::zero(1 << params.r, params.d);
        compute_y_and_w(witness, params, x, &ring_cyclotomic, &mut y, &mut w);

        // commit to w and lift to Zq[X]
        let mut w_hat = PVec::zero(w.length() * params.delta, params.d);
        let mut v_quo = PVec::zero(params.n, params.d);
        let mut v = PVec::zero(params.n, params.d);
        commit_w_and_lift(params, &ring_full, &mat_d_ntt, &w, &mut w_hat, &mut v, &mut v_quo);

        // ---- Sample challenges ----
        fs.push(&y);
        fs.push(&v);
        let seed = fs.get_seed();
        let challenges = PChal::rand_vec(1 << params.r, params.d, params.k, seed);

        #[cfg(feature = "verbose")]
        tick_item("Sample challenges");

        // ---- Prover response z ----
        let mut z = PVec::zero((1 << params.m) * params.delta, params.d);
        compute_z(witness, params, &ring_cyclotomic, &challenges, &mut z);

        // ---- Lift the verification equations to Zq[X]

        // - already calculated commitment to w over Zq[X]

        // - lift the outer commitment (re-calculate commitment to inner commitment t over Zq[X]).
        let mut u_full = PVec::zero(params.n, 2 * params.d);
        ring_full.mat_mul_vec(&mat_b_ntt, &t_hat, &mut u_full, 0);

        let mut u_quo = PVec::zero(params.n, params.d);
        let mut u_rem = PVec::zero(params.n, params.d);
        u_full.cyclotomic_div(params.q, &mut u_quo, &mut u_rem);

        // sense check
        assert_eq!(u.slice(), u_rem.slice());

        // - the product [[b_1 ... b_2^r]^T [w_1 ... w^2^r]] does not need to be lifted
        // since b_i are integers so the quotient is zero.

        // - calculate the inner product [c_1 ... c_2^r]^T [w_1 ... w^2^r] over Zq[X]
        let mut c_i_w_i_full = PVec::zero(1, 2 * params.d);
        ring_full.chal_vec_mul_poly_vec(&challenges, &w, &mut c_i_w_i_full, 0);

        let mut c_i_w_i_quo = PVec::zero(1, params.d);
        let mut c_i_w_i_rem = PVec::zero(1, params.d);
        c_i_w_i_full.cyclotomic_div(params.q, &mut c_i_w_i_quo, &mut c_i_w_i_rem);

        // - the product [[a_1 ... a_2^m]^T [z'_1 ... z'^2^m]] does not need to be lifted
        // since a_i are integers so the quotient is zero.

        // - calculate the inner products ([c_1 ... c_2^r]^T o I_n] [t_1,1..t_1,n ... t_2^r,1..t_2^r,n] over Zq[X]       
        let mut c_i_t_i_quo = PVec::zero(params.n, params.d);
        let mut c_i_t_i_rem = PVec::zero(params.n, params.d);
        lift_c_i_t_i(params, &ring_full, &challenges, t, &mut c_i_t_i_quo, &mut c_i_t_i_rem);

        // - calculate A.z over Zq[X]
        let mut mat_a_z_full = PVec::zero(params.n, 2 * params.d);
        ring_full.mat_mul_vec(&mat_a_ntt, &z, &mut mat_a_z_full, 0);

        let mut mat_a_z_quo = PVec::zero(params.n, params.d);
        let mut mat_a_z_rem = PVec::zero(params.n, params.d);
        mat_a_z_full.cyclotomic_div(params.q, &mut mat_a_z_quo, &mut mat_a_z_rem);

        // sense check
        assert_eq!(c_i_t_i_rem.slice(), mat_a_z_rem.slice());

        #[cfg(feature = "verbose")]
        tick_item("Lift verification equations to Zq[X]");

        // Produce the new witness (z,r)
        let (num_vars, mu, n, z_r) = form_next_witness(
            params, &w_hat, &t_hat, &z, &v_quo, &u_quo, &c_i_w_i_quo, &c_i_t_i_quo, &mat_a_z_quo
        );

        // Commit to the next witness
        let next_params = Hachi::setup(num_vars, false);
        let com_dash = Hachi::commit(z_r.as_slice(), &next_params);

        // Sample a random field element
        fs.push(&com_dash.u);
        let mut rng = ChaCha12Rng::from_seed(fs.get_seed());
        let alpha = rand_field(params.q, &mut rng);

        // Get the powers of alpha
        let alpha_pows = powers(alpha, params.d);

        // Get [M' | -(X^d+1).G_{3n+2}] evaluated at alpha (the quotient r is gadget-decomposed)
        let m_alpha = if params.reuse_mats {
            form_m_alpha_same_matrix(params, x, &challenges, &alpha_pows, &mat_d)
        } else {
            form_m_alpha_different_matrices(params, x, &challenges, &alpha_pows, &mat_a, &mat_b, &mat_d)
        };

        // Sample the random field elements tau_0 and tau_1
        let mut tau_0 = vec![ExtField::ZERO; num_vars];
        
        for i in 0..num_vars {
            tau_0[i] = rand_field(params.q, &mut rng);
        }

        let log_n = n.log();
        let mut tau_1 = vec![ExtField::ZERO; log_n];
        
        for i in 0..log_n {
            tau_1[i] = rand_field(params.q, &mut rng);
        }

        // Form F_0,tau_0 and F_alpha_tau_1 for sum check
        let mut f_0 = F0::init(&z_r, params.b, tau_0, params.q, (mu + n * params.delta) * params.d);
        let mut f_alpha = FAlpha::init(&z_r, alpha_pows, tau_1, m_alpha);

        let (univariates_f_alpha, univariates_f_0, y_dash) = sumcheck_proof(&mut f_0, &mut f_alpha, &mut fs);

        #[cfg(feature = "verbose")]
        println!("==== Complete ====\n");

        ProofRound { y, v, u_dash: com_dash.u, univariates_f_alpha, univariates_f_0, y_dash }
    }
}
