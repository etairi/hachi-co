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
//! Iterations repeat (Section 4.4) until the witness has at most
//! [`HachiChoices::final_words`] residues, or would not shrink; the last one is sent in the clear.

use crate::{Error, Result};
use hachi_co_math::decompose::{digit_range, num_digits, tight_tau};

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
    /// Stop iterating once the witness has at most this many residues; it is then sent in the
    /// clear.
    pub final_words: usize,
    /// An upper bound on the number of iterations (a safety net; the schedule normally stops
    /// much earlier).
    pub max_iterations: usize,
    /// Run exactly this many iterations instead of the automatic rule (for tests and
    /// measurements: an iteration shrinks the witness only if the witness is large enough, so
    /// on a small input the automatic rule sends the input in the clear).
    pub fixed_iterations: Option<usize>,
    /// The seed of the public matrices.
    pub seed: [u8; 32],
}

impl HachiChoices {
    /// Small, **insecure** choices for tests at a ring degree $d$: $b = 4$, $\omega = 2$,
    /// $n_A = n_B = n_D = 1$, a final witness of at most $2^{11}$ residues, at most 16
    /// iterations. Security needs Module-SIS-sized matrices and a larger challenge set; that is
    /// left to the final parameters.
    pub fn toy(d: usize) -> Self {
        Self {
            d,
            b: 4,
            omega: 2,
            n_a: 1,
            n_b: 1,
            n_d: 1,
            final_words: 1 << 11,
            max_iterations: 16,
            fixed_iterations: None,
            seed: [7u8; 32],
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
    /// The ring elements of $\mathbf z' = (\hat{\mathbf w}, \hat{\mathbf t}, \hat{\mathbf z})$:
    /// $2^r \delta + 2^r n_A \delta + W \tau$ with $W$ the chunk width.
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

/// The largest absolute value of a digit of base $b$: $\lfloor b/2 \rfloor$.
pub fn max_digit(b: u64) -> u64 {
    let (lo, hi) = digit_range(b);
    lo.unsigned_abs().max(hi.unsigned_abs()) as u64
}

/// The quantities an iteration derives from its layout.
fn iteration(choices: &HachiChoices, q: u64, layout: Layout) -> Result<IterParams> {
    let d = choices.d;
    let delta = num_digits(q, choices.b)?;
    let beta_wide = (1u128 << layout.r) * u128::from(choices.omega) * u128::from(layout.beta_s);
    let beta = u64::try_from(beta_wide)
        .map_err(|_| Error::Parameter("the fold bound beta exceeds 64 bits".into()))?;
    let tau = tight_tau(beta_wide.max(1), choices.b)?;
    let chunks = layout.chunks();
    let mu = chunks * delta + chunks * choices.n_a * delta + layout.width(delta) * tau;
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
        let beta = (1u128 << r) * u128::from(choices.omega) * u128::from(beta_s);
        let Ok(tau) = tight_tau(beta.max(1), choices.b) else {
            continue;
        };
        let chunks = layout.chunks();
        let mu = chunks * delta * (1 + choices.n_a) + layout.width(delta) * tau;
        if best.is_none_or(|(size, _)| mu < size) {
            best = Some((mu, layout));
        }
    }
    best.map(|(_, l)| l)
        .ok_or_else(|| Error::Parameter("no admissible split of the witness".into()))
}

impl Schedule {
    /// The schedule for a first witness of layout `first`: exactly
    /// [`HachiChoices::fixed_iterations`] iterations if set; otherwise iterate while the witness
    /// has more than [`HachiChoices::final_words`] residues, the iteration would shrink it (in
    /// residues), and fewer than [`HachiChoices::max_iterations`] iterations have run.
    ///
    /// # Errors
    /// [`Error::Parameter`] if an iteration cannot be parametrised.
    pub fn derive(choices: &HachiChoices, q: u64, first: Layout) -> Result<Self> {
        let d = choices.d;
        let mut iterations = Vec::new();
        let mut current = first;
        if let Some(n) = choices.fixed_iterations {
            for _ in 0..n {
                let it = iteration(choices, q, current)?;
                current = it.next;
                iterations.push(it);
            }
            return Ok(Self {
                iterations,
                final_layout: current,
            });
        }
        while current.words(d) > choices.final_words && iterations.len() < choices.max_iterations {
            let it = iteration(choices, q, current)?;
            if it.next.words(d) >= current.words(d) {
                break;
            }
            current = it.next;
            iterations.push(it);
        }
        Ok(Self {
            iterations,
            final_layout: current,
        })
    }
}

/// Checks the choices against $q$ and $k$.
///
/// # Errors
/// [`Error::Parameter`] unless $d$ is a power of two with $2k \mid d$, $b \ge 2$,
/// $\omega \ge 1$, the heights are positive, and $q \equiv 5 \pmod 8$.
pub fn check(choices: &HachiChoices, q: u64, k: usize) -> Result<()> {
    let d = choices.d;
    if !d.is_power_of_two() || k == 0 || !d.is_multiple_of(2 * k) {
        return Err(Error::Parameter(format!(
            "the ring degree {d} must be a power of two divisible by 2k = {}",
            2 * k
        )));
    }
    if choices.b < 2 || choices.omega == 0 {
        return Err(Error::Parameter("b >= 2 and omega >= 1 are needed".into()));
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
    Ok(())
}
