//! Parameters: the user's [`HachiChoices`], the [`Layout`] of a committed witness, and the
//! [`Schedule`] of iterations that the prover and the verifier both derive from them.
//!
//! An iteration (paper, Section 4.2-4.3) works on a witness of $2^r$ chunks of $2^m$ message
//! ring elements $\mathbf f_i$. The committed vectors are $\mathbf s_i = \mathbf G^{-1}(\mathbf f_i)$
//! when the message is not short (the input polynomial or vector), and $\mathbf s_i = \mathbf f_i$
//! when it is (every later witness, which is already made of short digits; Section 4.5,
//! "Avoiding re-decomposition"). The iteration folds the chunks into $\mathbf z = \sum_i c_i
//! \mathbf s_i$ and reduces the claim to one about its next witness $\tilde w$, a table of
//! $2^L$ short residues, which the next iteration commits packed into $2^{L-\alpha}$ ring elements.
//! Iterations repeat (Section 4.4), and the last witness is sent in the clear.
//!
//! **How many iterations.** Only the first iteration pays $\delta$ digits per coefficient of the
//! input; later witnesses are already short, are committed as they are, and shrink much faster.
//! So the number of iterations that gives the smallest proof is not found by stopping at the
//! first iteration that does not shrink the witness: for a polynomial of $2^{17}$ coefficients
//! at $d = 16$, $b = 4$ and $q \approx 2^{29}$, one iteration doubles the witness and three make
//! it about four times smaller. [`Schedule::candidates`] lists the schedules of $0, 1, \dots$
//! [`HachiChoices::max_iterations`] iterations, and the scheme picks the one with the smallest
//! proof (`HachiPcs`, which knows the encoding); [`HachiChoices::fixed_iterations`] overrides it.
//!
//! **Hiding** ([`HachiChoices::hiding`], `hachi-pcs.tex`, the blue parts of Appendix A.2). With
//! [`Hiding`] the first iteration of an opening over the field is masked and the later ones are
//! not, since their messages follow from the next witness of the first one, the public data and
//! the coins. Level 0 commits one more ring element per chunk (the masking row), so that
//! $W^{(0)} = \delta (2^m + 1)$, and its fold is masked by discrete Gaussians of parameter
//! $\mathfrak s \ge \phi \sqrt N \, 2^r \omega b$, $N = (W^{(0)} + \nu + n_A) d$. The norm bound
//! $\beta^{(0)}$ and with it $\tau^{(0)}$ rise, and the next witness gains the digits of $\hat w_0$,
//! $\hat{\mathbf t}_0$ and the folded randomness ([`IterParams`]). A masked witness needs at least
//! one iteration, as the last witness is sent in the clear.

use crate::{Error, Result};
use hachi_co_math::decompose::{digit_range, num_digits, representable_range, tight_tau};
use hachi_co_sampling::gauss::{MAX_WIDE_SIGMA, WIDE_BASE};

/// What the user chooses; everything else is derived. These are the parameters of Hachi as the
/// base PCS; they need not equal those of Hachi-Co (authors, 2026-10-21), except $q$ and $k$,
/// which the polynomials and points fix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HachiChoices {
    /// The ring degree $d = 2^\alpha$; $k$ must divide $d/2$ (paper, Lemma 5).
    pub d: usize,
    /// The decomposition base $b \ge 2$ of every gadget ($\mathbf G^{-1}$, $\mathbf J^{-1}$).
    pub b: u64,
    /// The $\ell_1$ radius $\omega$ of the challenge set $\mathcal C = \lbrace c : \lVert c
    /// \rVert_1 \le \omega \rbrace$ (paper, Section 4.2).
    pub omega: u64,
    /// The heights $n_A, n_B, n_D$ of the commitment matrices.
    pub n_a: usize,
    /// See [`HachiChoices::n_a`].
    pub n_b: usize,
    /// See [`HachiChoices::n_a`].
    pub n_d: usize,
    /// The most iterations the automatic rule considers (module documentation, "How many
    /// iterations"), and a bound on [`HachiChoices::fixed_iterations`]; at most
    /// [`MAX_ITERATIONS`].
    pub max_iterations: usize,
    /// Run exactly this many iterations, at most [`HachiChoices::max_iterations`], instead of
    /// the automatic rule (for tests and measurements).
    pub fixed_iterations: Option<usize>,
    /// The seed of the public matrices.
    pub seed: [u8; 32],
    /// Masking of the first iteration ([`Hiding`]), or `None` for the published scheme.
    pub hiding: Option<Hiding>,
}

impl HachiChoices {
    /// Small, **insecure** choices for tests at a ring degree $d$: $b = 4$, $\omega = 2$,
    /// $n_A = n_B = n_D = 1$, at most 16 iterations. Security needs Module-SIS-sized matrices
    /// and a larger challenge set; that is left to the final parameters.
    pub fn toy(d: usize) -> Self {
        Self {
            d,
            b: 4,
            omega: 2,
            n_a: 1,
            n_b: 1,
            n_d: 1,
            max_iterations: 16,
            fixed_iterations: None,
            seed: [7u8; 32],
            hiding: None,
        }
    }

    /// [`HachiChoices::toy`] with masks: $\nu = 1$, the same height at level 0, $\lambda = 40$ and
    /// $\phi = 13$. Equally **insecure**: $\nu$ is far below an MLWE rank of any security, and
    /// $\lambda$ is low so that the masks fit the 23-bit modulus of Hachi-Co's `toy-n1` preset.
    pub fn toy_hiding(d: usize) -> Self {
        Self {
            hiding: Some(Hiding {
                nu: 1,
                n_a: 1,
                lambda: 40,
                phi: 13,
            }),
            ..Self::toy(d)
        }
    }
}

/// Masking the first iteration (`hachi-pcs.tex`, Appendix A.2, after Jindo Section 2.4). At level
/// 0, chunk $i$ of the witness carries a uniform masking row and is committed as $\mathbf t_i =
/// \mathbf A \mathbf s_i + \mathbf A' \boldsymbol\varrho_i$, where $\mathbf A' = [\mathbf A_1'
/// \mid \mathbf I_{n_A}]$ with $\mathbf A_1'$ uniform of width $\nu$ and the coefficients of
/// $\boldsymbol\varrho_i \in \mathcal R^{\nu + n_A}$ are uniform digits. Later levels are as
/// published.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hiding {
    /// The MLWE rank $\nu = n_{A'}$: the width of $\mathbf A_1'$.
    pub nu: usize,
    /// The height $n_A$ of $\mathbf A$ and $\mathbf A'$ at level 0. The masks raise the norm at
    /// level 0 only, so the later levels keep [`HachiChoices::n_a`].
    pub n_a: usize,
    /// $\lambda$: the statistical errors of the rejection sampling and of the norm bound are at
    /// most $2^{-\lambda}$.
    pub lambda: u32,
    /// $\phi$, as a positive integer, so that the rejection constant has an exact logarithm.
    pub phi: u32,
}

impl Hiding {
    /// $\mathfrak t = \max\lbrace 14, \lceil \sqrt{2 \lambda \ln 2} \rceil \rbrace$ (Setup, item
    /// 1(f), with the square root rounded up).
    pub fn t(&self) -> u64 {
        let root = (2.0 * f64::from(self.lambda) * core::f64::consts::LN_2).sqrt();
        14u64.max(root.ceil() as u64)
    }

    /// $\ln M = \mathfrak t / \phi + 1 / (2 \phi^2)$ as the fraction `(num, den)`.
    pub fn log_rejection(&self) -> (u128, u128) {
        let phi = u128::from(self.phi);
        (2 * phi * u128::from(self.t()) + 1, 2 * phi * phi)
    }

    /// The attempts of a fold before the prover gives up: $\lceil \lambda M \rceil + 1$, the strict
    /// variant of `Eval` in Appendix A.3, a little above $\lceil \lambda \bar M \rceil$.
    pub fn attempts(&self) -> u32 {
        let (num, den) = self.log_rejection();
        let m = (num as f64 / den as f64).exp();
        (f64::from(self.lambda) * m).ceil() as u32 + 1
    }

    /// The words that name these choices in statements and digests.
    pub fn words(&self) -> [u64; 4] {
        [
            self.nu as u64,
            self.n_a as u64,
            u64::from(self.lambda),
            u64::from(self.phi),
        ]
    }
}

/// The shape of a committed witness: $2^r$ chunks of $2^m$ message ring elements each, plus a
/// masking row if [`Layout::masked`], committed as digits ([`Layout::decomposed`]) or as they are,
/// with coefficients of absolute value at most [`Layout::beta_s`] in the committed vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    /// $\log_2$ of the number of chunks.
    pub r: usize,
    /// $\log_2$ of the number of message ring elements per chunk.
    pub m: usize,
    /// Whether the committed vectors are $\mathbf G^{-1}$ of the message ($\delta$ digits per
    /// message element) rather than the message itself.
    pub decomposed: bool,
    /// The bound $\beta_s$ on the absolute value (centred) of every coefficient of a committed
    /// vector $\mathbf s_i$.
    pub beta_s: u64,
    /// Whether every chunk ends with a masking row, whose fold is masked ([`Hiding`]).
    pub masked: bool,
}

impl Layout {
    /// The number of chunks $2^r$.
    pub fn chunks(&self) -> usize {
        1 << self.r
    }

    /// The number $2^m$ of message ring elements in a chunk, without the masking row.
    pub fn message_len(&self) -> usize {
        1 << self.m
    }

    /// The number of ring elements in a chunk: [`Layout::message_len`], plus the masking row of a
    /// masked layout.
    pub fn chunk_len(&self) -> usize {
        self.message_len() + usize::from(self.masked)
    }

    /// The number of committed ring elements in a chunk: [`Layout::chunk_len`] times $\delta$ if
    /// decomposed.
    pub fn width(&self, delta: usize) -> usize {
        if self.decomposed {
            self.chunk_len() * delta
        } else {
            self.chunk_len()
        }
    }

    /// The number of message residues, $2^{r+m} d$: what the final step sends.
    pub fn words(&self, d: usize) -> usize {
        (1usize << (self.r + self.m)) * d
    }
}

/// One iteration's derived quantities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IterParams {
    /// The layout of the witness this iteration proves.
    pub layout: Layout,
    /// The digits $\delta = \lceil \log_b q \rceil$ (window rule of `hachi_co_math::decompose`)
    /// of $\mathbf G^{-1}$.
    pub delta: usize,
    /// The height $n_A$ of this iteration's matrices.
    pub n_a: usize,
    /// The bound on $\lVert \mathbf z \rVert_\infty$: $2^r \omega \beta_s$ (paper, Section 4.4,
    /// with Lemma 2), and at a masked level $\beta^{(0)}$, a tail bound of the masks that also
    /// covers $\mathbf z_\varrho$.
    pub beta: u64,
    /// The digits $\tau$ of $\mathbf J^{-1}(\mathbf z)$: the least $\tau$ whose balanced digits
    /// cover $[-\beta, \beta]$. An iteration is refused if the values that $\tau$ digits represent
    /// reach beyond the centred residues mod $q$: the range check lets them all through, and the
    /// norm arguments need each to be its own centred residue.
    pub tau: usize,
    /// The Gaussian parameter $\mathfrak s$ of the masks, a multiple of 16, or 0 at a level without
    /// masks.
    pub sigma: u64,
    /// The ring elements of $\mathbf z' = (\hat{\mathbf w}, \hat{\mathbf t}, \hat{\mathbf z})$:
    /// $2^r \delta + 2^r n_A \delta + W \tau$ with $W$ the chunk width. At a masked level
    /// $\mathbf z' = (\hat{\mathbf w}, \hat w_0, \hat{\mathbf t}_0, \hat{\mathbf t}, \hat{\mathbf
    /// z}, \hat{\mathbf z}_\varrho)$: $2^r \delta + \delta + n_A \delta + 2^r n_A \delta + W \tau +
    /// (\nu + n_A) \tau$.
    pub mu: usize,
    /// The rows $n' = n_D + n_B + 2 + n_A$ of the linear relation (paper, Eq. (20)).
    pub rows: usize,
    /// $L$: the next witness $\tilde w$ is a table of $2^L$ residues, $(\mu + n'\delta) d$ padded
    /// to a power of two (the quotient rows are decomposed, Section 4.3).
    pub table_vars: usize,
    /// The layout in which the next iteration (or the final step) commits $\tilde w$.
    pub next: Layout,
}

impl IterParams {
    /// The width of $\mathbf D$: $2^r \delta$, and $\delta (2^r + 1 + n_A)$ at a masked level, where
    /// it also commits $\hat w_0$ and $\hat{\mathbf t}_0$.
    pub fn d_cols(&self) -> usize {
        let chunks = self.layout.chunks();
        if self.layout.masked {
            self.delta * (chunks + 1 + self.n_a)
        } else {
            self.delta * chunks
        }
    }
}

/// The iterations of one evaluation proof and the layout of the witness the final step sends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schedule {
    /// The iterations, in order.
    pub iterations: Vec<IterParams>,
    /// The layout of the witness sent in the clear.
    pub final_layout: Layout,
}

/// The largest [`HachiChoices::max_iterations`] that [`check`] accepts: a bound on the work of
/// deriving and running a schedule, far above what the schedules use (2 to 4 iterations at
/// Hachi-Co's toy and development presets).
pub const MAX_ITERATIONS: usize = 64;

/// The largest MLWE rank $\nu$ that [`check`] accepts ([`Hiding::nu`]): a bound on the size of
/// $\mathbf A_1'$, far above any useful rank.
pub const MAX_NU: usize = 4096;

/// The largest absolute value of a digit of base $b$: $\lfloor b/2 \rfloor$.
pub fn max_digit(b: u64) -> u64 {
    let (lo, hi) = digit_range(b);
    lo.unsigned_abs().max(hi.unsigned_abs()) as u64
}

/// $2^r \omega \beta$ for the fold of $2^r$ chunks of coefficients at most $\beta$.
fn fold_bound(choices: &HachiChoices, layout: &Layout, beta: u64) -> u128 {
    (1u128 << layout.r) * u128::from(choices.omega) * u128::from(beta)
}

/// The height $n_A$ of a level with this layout.
pub fn height(choices: &HachiChoices, layout: &Layout) -> Result<usize> {
    if !layout.masked {
        return Ok(choices.n_a);
    }
    choices
        .hiding
        .map(|h| h.n_a)
        .ok_or_else(|| Error::Parameter("a masked layout without hiding choices".into()))
}

/// $\lceil \sqrt x \rceil$.
fn ceil_sqrt(x: u128) -> u128 {
    let mut root = (x as f64).sqrt() as u128;
    while root * root > x {
        root -= 1;
    }
    while root * root < x {
        root += 1;
    }
    root
}

/// The Gaussian parameter and the norm bound $(\mathfrak s, \beta^{(0)})$ of a masked level with
/// $N$ = `coeffs` coefficients in the masks (Setup, item 2(e)):
/// - $\mathfrak s$ is $\phi \lceil \sqrt N \rceil 2^r \omega b$ rounded up to a multiple of 16,
///   the widths of [`hachi_co_sampling::gauss::WideGaussian`].
/// - $\beta^{(0)} = \lceil \mathfrak s \sqrt{2 \ln 2 (\lceil \log_2 2N \rceil + \lambda)} \rceil$,
///   at least the paper's $\mathfrak s \sqrt{2 \ln(2N) + 2 \lambda \ln 2}$.
///
/// A larger $\mathfrak s$ only helps Lemma 21 (the fold at level 0), and a larger $\beta^{(0)}$ only
/// makes its error $\varepsilon_\beta$ smaller. The computation uses only exact integer operations
/// and the correctly rounded operations of IEEE 754, so prover and verifier agree on any platform.
fn mask_widths(
    choices: &HachiChoices,
    layout: &Layout,
    hiding: &Hiding,
    coeffs: usize,
) -> Result<(u64, u64)> {
    let too_large = || Error::Parameter("the masks do not fit 64 bits".into());
    let root = ceil_sqrt(coeffs as u128);
    let target = u128::from(hiding.phi)
        .checked_mul(root)
        .and_then(|x| x.checked_mul(1u128 << layout.r))
        .and_then(|x| x.checked_mul(u128::from(choices.omega)))
        .and_then(|x| x.checked_mul(u128::from(choices.b)))
        .ok_or_else(too_large)?;
    let sigma = target
        .div_ceil(u128::from(WIDE_BASE))
        .checked_mul(u128::from(WIDE_BASE))
        .filter(|s| *s <= u128::from(MAX_WIDE_SIGMA))
        .ok_or_else(too_large)?;
    let log = usize::BITS - (2 * coeffs - 1).leading_zeros();
    let tail = (2.0 * core::f64::consts::LN_2 * f64::from(log + hiding.lambda)).sqrt();
    let beta = (sigma as f64 * tail).ceil();
    if beta >= 2f64.powi(62) {
        return Err(too_large());
    }
    Ok((sigma as u64, beta as u64))
}

/// The bound $\beta$, the digits $\tau$, the mask parameter $\mathfrak s$ and $\mu$ of an iteration
/// on `layout` ([`IterParams`]).
fn witness_size(
    choices: &HachiChoices,
    delta: usize,
    layout: &Layout,
) -> Result<(u64, usize, u64, usize)> {
    let chunks = layout.chunks();
    let width = layout.width(delta);
    if layout.masked {
        let hiding = choices
            .hiding
            .ok_or_else(|| Error::Parameter("a masked layout without hiding choices".into()))?;
        let rand = hiding.nu + hiding.n_a;
        let (sigma, beta) = mask_widths(choices, layout, &hiding, (width + rand) * choices.d)?;
        let tau = tight_tau(u128::from(beta), choices.b)?;
        let mu =
            chunks * delta * (1 + hiding.n_a) + delta * (1 + hiding.n_a) + (width + rand) * tau;
        return Ok((beta, tau, sigma, mu));
    }
    let bound = fold_bound(choices, layout, layout.beta_s);
    let beta = u64::try_from(bound)
        .map_err(|_| Error::Parameter("the fold bound beta exceeds 64 bits".into()))?;
    let tau = tight_tau(bound.max(1), choices.b)?;
    let mu = chunks * delta * (1 + choices.n_a) + width * tau;
    Ok((beta, tau, 0, mu))
}

/// Whether the values that `tau` balanced digits of base `b` represent are centred residues mod
/// $q$. The range checks let through every such value, and the norm arguments read them as
/// integers, so each must be its own centred residue: otherwise a relaxed opening has norm $q/2$
/// or more, and binding is void.
fn digits_fit(b: u64, tau: usize, q: u64) -> bool {
    let half = i128::from((q - 1) / 2);
    let (lo, hi) = representable_range(b, tau);
    -lo <= half && hi <= half
}

/// The quantities an iteration derives from its layout.
fn iteration(choices: &HachiChoices, q: u64, layout: Layout) -> Result<IterParams> {
    let d = choices.d;
    let delta = num_digits(q, choices.b)?;
    let (beta, tau, sigma, mu) = witness_size(choices, delta, &layout)?;
    if !digits_fit(choices.b, tau, q) {
        let (lo, hi) = representable_range(choices.b, tau);
        return Err(Error::Parameter(format!(
            "the {tau} digits of z represent [{lo}, {hi}], beyond the centred residues mod q = {q}"
        )));
    }
    let n_a = height(choices, &layout)?;
    let rows = choices.n_d + choices.n_b + 2 + n_a;
    let entries = (mu + rows * delta) * d;
    let table_vars = entries.next_power_of_two().trailing_zeros() as usize;
    let alpha = d.trailing_zeros() as usize;
    let packed = table_vars.checked_sub(alpha).ok_or_else(|| {
        Error::Parameter("the next witness is smaller than a ring element".into())
    })?;
    let next_beta = 2 * max_digit(choices.b);
    let next = best_layout(choices, q, packed, false, next_beta, false)?;
    Ok(IterParams {
        layout,
        n_a,
        delta,
        beta,
        tau,
        sigma,
        mu,
        rows,
        table_vars,
        next,
    })
}

/// The split $r + m = $ `vars` that minimises the next witness of an iteration on this witness
/// (ties: the smaller $r$).
pub fn best_layout(
    choices: &HachiChoices,
    q: u64,
    vars: usize,
    decomposed: bool,
    beta_s: u64,
    masked: bool,
) -> Result<Layout> {
    let delta = num_digits(q, choices.b)?;
    let mut best: Option<(usize, Layout)> = None;
    for r in 0..=vars {
        let layout = Layout {
            r,
            m: vars - r,
            decomposed,
            beta_s,
            masked,
        };
        let Ok((_, tau, _, mu)) = witness_size(choices, delta, &layout) else {
            continue;
        };
        // The masks raise tau, so at a masked level a split can pass the sizes and fail
        // [`digits_fit`]. Without masks the schedule reports that case instead.
        if masked && !digits_fit(choices.b, tau, q) {
            continue;
        }
        if best.is_none_or(|(size, _)| mu < size) {
            best = Some((mu, layout));
        }
    }
    best.map(|(_, l)| l)
        .ok_or_else(|| Error::Parameter("no admissible split of the witness".into()))
}

impl Schedule {
    /// Exactly `n` iterations on a first witness of layout `first`.
    ///
    /// # Errors
    /// [`Error::Parameter`] if `n` exceeds [`HachiChoices::max_iterations`], if one of the
    /// iterations cannot be parametrised, or if a masked witness has no iteration, which would
    /// send it in the clear.
    pub fn fixed(choices: &HachiChoices, q: u64, first: Layout, n: usize) -> Result<Self> {
        if n == 0 && first.masked {
            return Err(Error::Parameter(
                "a masked witness needs at least one iteration".into(),
            ));
        }
        if n > choices.max_iterations {
            return Err(Error::Parameter(format!(
                "{n} iterations exceed max_iterations = {}",
                choices.max_iterations
            )));
        }
        let mut iterations = Vec::new();
        let mut current = first;
        for _ in 0..n {
            let it = iteration(choices, q, current)?;
            current = it.next;
            iterations.push(it);
        }
        Ok(Self {
            iterations,
            final_layout: current,
        })
    }

    /// The schedules the automatic rule chooses from: $0, 1, \dots$
    /// [`HachiChoices::max_iterations`] iterations on a first witness of layout `first`, each a
    /// prefix of the next. It contains the schedule without iterations unless `first` is masked.
    /// The list ends early at the first iteration that cannot be parametrised (for example a fold
    /// bound beyond 64 bits), or whose next witness has the layout of an earlier final witness:
    /// the next layout depends only on the current one, so from there on every schedule has the
    /// final witness of an earlier one and strictly more words, and none can be the smallest.
    pub fn candidates(choices: &HachiChoices, q: u64, first: Layout) -> Vec<Self> {
        let mut out = Vec::new();
        if !first.masked {
            out.push(Self {
                iterations: Vec::new(),
                final_layout: first,
            });
        }
        let mut current = first;
        let mut iterations = Vec::new();
        while iterations.len() < choices.max_iterations {
            let Ok(it) = iteration(choices, q, current) else {
                break;
            };
            if out.iter().any(|s| s.final_layout == it.next) {
                break;
            }
            current = it.next;
            iterations.push(it);
            out.push(Self {
                iterations: iterations.clone(),
                final_layout: current,
            });
        }
        out
    }
}

/// Checks the choices against $q$ and $k$.
///
/// # Errors
/// [`Error::Parameter`] unless $d$ is a power of two with $2k \mid d$, $2 \le b \le q$ (so that
/// the digits are distinct residues whose centred representatives are the digits themselves,
/// which the encoding of the final witness relies on), $\omega \ge 1$, the heights are
/// positive, `max_iterations` $\le$ [`MAX_ITERATIONS`], `fixed_iterations` $\le$
/// `max_iterations`, $q \equiv 5 \pmod 8$, and, with [`Hiding`], $1 \le \nu \le$ [`MAX_NU`],
/// $n_A \ge 1$, $\lambda, \phi \ge 1$, and $b \le \sqrt{q/2}$ and $\omega < \sqrt q / (2 \sqrt 2)$
/// (Setup, item 1(b)).
pub fn check(choices: &HachiChoices, q: u64, k: usize) -> Result<()> {
    let d = choices.d;
    if !d.is_power_of_two() || k == 0 || !d.is_multiple_of(2 * k) {
        return Err(Error::Parameter(format!(
            "the ring degree {d} must be a power of two divisible by 2k = {}",
            2 * k
        )));
    }
    if choices.b < 2 || choices.b > q || choices.omega == 0 {
        return Err(Error::Parameter(
            "2 <= b <= q and omega >= 1 are needed".into(),
        ));
    }
    if choices.max_iterations > MAX_ITERATIONS
        || choices
            .fixed_iterations
            .is_some_and(|n| n > choices.max_iterations)
    {
        return Err(Error::Parameter(format!(
            "fixed_iterations <= max_iterations <= {MAX_ITERATIONS} are needed"
        )));
    }
    if choices.n_a == 0 || choices.n_b == 0 || choices.n_d == 0 {
        return Err(Error::Parameter(
            "the matrix heights must be positive".into(),
        ));
    }
    if q % 8 != 5 {
        return Err(Error::Parameter(format!(
            "q = {q} must be 5 mod 8 (paper, Lemma 5)"
        )));
    }
    if let Some(h) = choices.hiding {
        if h.nu == 0 || h.nu > MAX_NU || h.n_a == 0 || h.lambda == 0 || h.phi == 0 {
            return Err(Error::Parameter(format!(
                "hiding needs 1 <= nu <= {MAX_NU} and positive n_A, lambda and phi"
            )));
        }
        let (q, b, omega) = (
            u128::from(q),
            u128::from(choices.b),
            u128::from(choices.omega),
        );
        if 2 * b * b > q || 8 * omega * omega >= q {
            return Err(Error::Parameter(
                "hiding needs b <= sqrt(q / 2) and omega < sqrt(q) / (2 sqrt(2))".into(),
            ));
        }
    }
    Ok(())
}
