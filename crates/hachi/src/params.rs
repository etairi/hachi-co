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
//! **Randomized commitments** ([`HachiChoices::hiding`], `docs/hachi-hiding.md` §6). With [`Hiding`],
//! every inner commitment carries $\mathbf A' \mathbf r_i$ and every first message
//! $\mathbf D' \boldsymbol\rho$. The iteration folds the $\mathbf r_i$ with the witness, and the
//! next witness gains the digits of $\boldsymbol\rho$ and of the folded $\mathbf r_z$:
//! $(\nu + n_D)\tau_\rho + (\nu + n_A)\tau_r$ ring elements ([`IterParams`]).

use crate::{Error, Result};
use hachi_co_math::decompose::{digit_range, num_digits, tight_tau};
use hachi_co_sampling::gauss;

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
    /// Randomized commitments ([`Hiding`]), or `None` for the deterministic commitments of the
    /// paper.
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

    /// [`HachiChoices::toy`] with randomized commitments: $\nu = 1$ and the discrete Gaussian of
    /// standard deviation 8, which meets Hwang-Seo-Song's width rule (`docs/hachi-hiding.md`, 6.4).
    /// Equally **insecure**: $\nu$ is far below an MLWE rank of any security.
    pub fn toy_hiding(d: usize) -> Self {
        Self {
            hiding: Some(Hiding {
                nu: 1,
                noise: Noise::Gaussian { log_sigma: 3 },
            }),
            ..Self::toy(d)
        }
    }
}

/// Randomized commitments (`docs/hachi-hiding.md` §6; the draft's item (i), 4_preliminaries.tex:334,
/// after Hwang-Seo-Song §3.3). Chunk $i$ of every committed witness is committed as
/// $\mathbf t_i = \mathbf A \mathbf s_i + \mathbf A' \mathbf r_i$ and every iteration sends
/// $\mathbf v = \mathbf D \hat{\mathbf w} + \mathbf D' \boldsymbol\rho$, where
/// $\mathbf A' = [\mathbf A_1' \mid \mathbf I_{n_A}]$ and
/// $\mathbf D' = [\mathbf D_1' \mid \mathbf I_{n_D}]$ with $\mathbf A_1'$, $\mathbf D_1'$ uniform
/// of width $\nu$, and the coefficients of $\mathbf r_i \in \mathcal R^{\nu + n_A}$ and
/// $\boldsymbol\rho \in \mathcal R^{\nu + n_D}$ follow [`Noise`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hiding {
    /// The MLWE rank $\nu \ge 1$: the width of $\mathbf A_1'$ and $\mathbf D_1'$.
    pub nu: usize,
    /// The distribution $\chi$ of the coefficients of $\mathbf r_i$ and $\boldsymbol\rho$.
    pub noise: Noise,
}

/// The distribution $\chi$ of the commitment randomness (`docs/hachi-hiding.md`, 6.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Noise {
    /// Uniform over the digit set $D$ of base $b$, like Hachi-Co's $\boldsymbol\rho_i$
    /// (5_Building_Blocks.tex:110).
    Digits,
    /// $D_{\mathbb Z,\sigma}$ with standard deviation $\sigma = 2^j$
    /// ([`hachi_co_sampling::gauss`]), supported on $[-B, B]$, $B$ the length of its table.
    Gaussian {
        /// $j = \log_2 \sigma$, at most [`gauss::MAX_LOG_SIGMA`].
        log_sigma: u32,
    },
}

impl Noise {
    /// The support $[\mathrm{lo}, \mathrm{hi}]$ of $\chi$ at base $b$.
    ///
    /// # Errors
    /// [`Error::Parameter`] for a Gaussian without a table.
    pub fn range(&self, b: u64) -> Result<(i64, i64)> {
        match *self {
            Self::Digits => {
                let (lo, hi) = digit_range(b);
                Ok((lo as i64, hi as i64))
            }
            Self::Gaussian { log_sigma } => {
                let bound = gauss::bound(log_sigma).map_err(|_| {
                    Error::Parameter(format!("no Gaussian table for sigma = 2^{log_sigma}"))
                })?;
                Ok((-(bound as i64), bound as i64))
            }
        }
    }

    /// $B_\chi$, the largest absolute value in the support.
    ///
    /// # Errors
    /// As [`Noise::range`].
    pub fn bound(&self, b: u64) -> Result<u64> {
        let (lo, hi) = self.range(b)?;
        Ok(lo.unsigned_abs().max(hi.unsigned_abs()))
    }

    /// The two words that name $\chi$ in statements and digests: $(1, 0)$ for
    /// [`Noise::Digits`], $(2, j)$ for [`Noise::Gaussian`].
    pub fn words(&self) -> [u64; 2] {
        match *self {
            Self::Digits => [1, 0],
            Self::Gaussian { log_sigma } => [2, u64::from(log_sigma)],
        }
    }
}

/// The shape of a committed witness: $2^r$ chunks of $2^m$ message ring elements each, committed
/// as digits ([`Layout::decomposed`]) or as they are, with coefficients of absolute value at most
/// [`Layout::beta_s`] in the committed vectors.
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
}

impl Layout {
    /// The number of chunks $2^r$.
    pub fn chunks(&self) -> usize {
        1 << self.r
    }

    /// The number $2^m$ of message ring elements in a chunk.
    pub fn chunk_len(&self) -> usize {
        1 << self.m
    }

    /// The number of committed ring elements in a chunk: $2^m \delta$ if decomposed, else $2^m$.
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
    /// The bound $\beta = 2^r \omega \beta_s$ on $\lVert \mathbf z \rVert_\infty$ (paper, Section
    /// 4.4, with Lemma 2).
    pub beta: u64,
    /// The digits $\tau$ of $\mathbf J^{-1}(\mathbf z)$: the least $\tau$ whose balanced digits
    /// cover $[-\beta, \beta]$.
    pub tau: usize,
    /// The bound $\beta_r = 2^r \omega B_\chi$ on $\lVert \mathbf r_z \rVert_\infty$, the folded
    /// commitment randomness (0 without [`Hiding`]).
    pub beta_r: u64,
    /// The digits $\tau_r$ of $\mathbf J^{-1}(\mathbf r_z)$: the least covering
    /// $[-\beta_r, \beta_r]$ (0 without [`Hiding`]).
    pub tau_r: usize,
    /// The digits $\tau_\rho$ of $\mathbf J^{-1}(\boldsymbol\rho)$: 1 for [`Noise::Digits`],
    /// whose entries are digits, else the least covering $[-B_\chi, B_\chi]$ (0 without
    /// [`Hiding`]).
    pub tau_rho: usize,
    /// The ring elements of $\mathbf z' = (\hat{\mathbf w}, \hat{\mathbf t}, \hat{\mathbf z},
    /// \hat{\boldsymbol\rho}, \hat{\mathbf r}_z)$: $2^r \delta + 2^r n_A \delta + W \tau +
    /// (\nu + n_D)\tau_\rho + (\nu + n_A)\tau_r$ with $W$ the chunk width (the last two terms
    /// only with [`Hiding`]).
    pub mu: usize,
    /// The rows $n' = n_D + n_B + 2 + n_A$ of the linear relation (paper, Eq. (20)).
    pub rows: usize,
    /// $L$: the next witness $\tilde w$ is a table of $2^L$ residues, $(\mu + n'\delta) d$ padded
    /// to a power of two (the quotient rows are decomposed, Section 4.3).
    pub table_vars: usize,
    /// The layout in which the next iteration (or the final step) commits $\tilde w$.
    pub next: Layout,
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
/// $\mathbf A_1'$ and $\mathbf D_1'$, far above any useful rank (Hwang-Seo-Song use $\nu = 2$ at
/// $d = 2048$).
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

/// The digit counts $(\tau, \tau_\rho, \tau_r)$ and $\mu$ of an iteration on `layout`
/// ([`IterParams`]).
fn witness_size(
    choices: &HachiChoices,
    delta: usize,
    layout: &Layout,
) -> Result<(usize, usize, usize, usize)> {
    let tau = tight_tau(fold_bound(choices, layout, layout.beta_s).max(1), choices.b)?;
    let chunks = layout.chunks();
    let mut mu = chunks * delta * (1 + choices.n_a) + layout.width(delta) * tau;
    let (mut tau_rho, mut tau_r) = (0, 0);
    if let Some(h) = choices.hiding {
        let bound = h.noise.bound(choices.b)?;
        tau_r = tight_tau(fold_bound(choices, layout, bound).max(1), choices.b)?;
        tau_rho = match h.noise {
            Noise::Digits => 1,
            Noise::Gaussian { .. } => tight_tau(u128::from(bound).max(1), choices.b)?,
        };
        mu += (h.nu + choices.n_d) * tau_rho + (h.nu + choices.n_a) * tau_r;
    }
    Ok((tau, tau_rho, tau_r, mu))
}

/// The quantities an iteration derives from its layout.
fn iteration(choices: &HachiChoices, q: u64, layout: Layout) -> Result<IterParams> {
    let d = choices.d;
    let delta = num_digits(q, choices.b)?;
    let beta = u64::try_from(fold_bound(choices, &layout, layout.beta_s))
        .map_err(|_| Error::Parameter("the fold bound beta exceeds 64 bits".into()))?;
    let beta_r = match choices.hiding {
        Some(h) => u64::try_from(fold_bound(choices, &layout, h.noise.bound(choices.b)?)).map_err(
            |_| Error::Parameter("the bound of the folded randomness exceeds 64 bits".into()),
        )?,
        None => 0,
    };
    let (tau, tau_rho, tau_r, mu) = witness_size(choices, delta, &layout)?;
    let rows = choices.n_d + choices.n_b + 2 + choices.n_a;
    let entries = (mu + rows * delta) * d;
    let table_vars = entries.next_power_of_two().trailing_zeros() as usize;
    let alpha = d.trailing_zeros() as usize;
    let packed = table_vars.checked_sub(alpha).ok_or_else(|| {
        Error::Parameter("the next witness is smaller than a ring element".into())
    })?;
    let next_beta = 2 * max_digit(choices.b);
    let next = best_layout(choices, q, packed, false, next_beta)?;
    Ok(IterParams {
        layout,
        delta,
        beta,
        tau,
        beta_r,
        tau_r,
        tau_rho,
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
) -> Result<Layout> {
    let delta = num_digits(q, choices.b)?;
    let mut best: Option<(usize, Layout)> = None;
    for r in 0..=vars {
        let layout = Layout {
            r,
            m: vars - r,
            decomposed,
            beta_s,
        };
        let Ok((_, _, _, mu)) = witness_size(choices, delta, &layout) else {
            continue;
        };
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
    /// [`Error::Parameter`] if `n` exceeds [`HachiChoices::max_iterations`] or one of the
    /// iterations cannot be parametrised.
    pub fn fixed(choices: &HachiChoices, q: u64, first: Layout, n: usize) -> Result<Self> {
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
    /// prefix of the next; it always contains the schedule without iterations. The list ends
    /// early at the first iteration that cannot be parametrised (for example a fold bound beyond
    /// 64 bits), or whose next witness has the layout of an earlier final witness: the next
    /// layout depends only on the current one, so from there on every schedule has the final
    /// witness of an earlier one and strictly more words, and none can be the smallest.
    pub fn candidates(choices: &HachiChoices, q: u64, first: Layout) -> Vec<Self> {
        let mut out = vec![Self {
            iterations: Vec::new(),
            final_layout: first,
        }];
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
/// `max_iterations`, $q \equiv 5 \pmod 8$, and, with [`Hiding`], $1 \le \nu \le$ [`MAX_NU`] and
/// a support $\[\mathrm{lo}, \mathrm{hi}\]$ of $\chi$ inside the centred residues,
/// $\mathrm{hi} - \mathrm{lo} \lt q$ with $\lvert \mathrm{lo} \rvert, \mathrm{hi} \le (q - 1)/2$
/// (a Gaussian needs $2B + 1 \le q$; digits are covered by $b \le q$). Then every value of the
/// support is a distinct residue whose centred representative is the value itself, which the
/// verifier's support check and the encoding of the final randomness rely on.
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
        if h.nu == 0 || h.nu > MAX_NU {
            return Err(Error::Parameter(format!(
                "hiding needs 1 <= nu <= {MAX_NU}"
            )));
        }
        let (lo, hi) = h.noise.range(choices.b)?;
        let half = (q - 1) / 2;
        if lo.unsigned_abs() > half || hi.unsigned_abs() > half {
            return Err(Error::Parameter(format!(
                "the support [{lo}, {hi}] of the randomness exceeds the centred residues mod q = {q}"
            )));
        }
    }
    Ok(())
}
